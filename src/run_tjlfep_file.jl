using Distributed
using Serialization

"""
    _with_mps_team(f, n_team::Int)

Run `f(team)` where `team` is a vector of `n_team` worker pids that all share this
process's pinned GPU as NVIDIA MPS clients (so the within-radius kw-scan eigensolves
overlap on the device via Hyper-Q). Requires the MPS control daemon to be running on
the node and `CUDA_MPS_PIPE_DIRECTORY` in the env (the run_tjlfep / gacode batch scripts
set these up). The team is torn down (`rmprocs`) on exit.

`n_team <= 0` runs `f(nothing)` (single-process threaded inner). If team spawn fails
for any reason (notably: `addprocs` is only valid from the cluster master, so a nested
call from a Distributed worker is not supported), this falls back to `f(nothing)` with a
warning so the run never breaks. Shared by the gacode-file MPS task and the FUSE-dd
`runTHD(::IMAS.dd)` path.
"""
function _with_mps_team(f, n_team::Int)
    if n_team <= 0
        return f(nothing)
    end
    team = Int[]
    try
        base_env = Dict{String,String}()
        for k in ("JULIA_DEPOT_PATH", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
                  "JULIA_CUDA_USE_COMPAT", "JULIA_CUDA_MEMORY_POOL")
            haskey(ENV, k) && (base_env[k] = ENV[k])
        end
        base_env["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
        vis = get(ENV, "CUDA_VISIBLE_DEVICES", "0")
        threads_per = parse(Int, get(ENV, "JULIA_WORKER_THREADS", "2"))
        proj = pkgdir(TJLFEP)
        sysimg = get(ENV, "TGLFEP_JULIA_GPU_SYSIMAGE", "")
        sysflag = (!isempty(sysimg) && isfile(sysimg)) ? `--sysimage=$(sysimg)` : ``
        for _ in 1:n_team
            env = copy(base_env)
            env["CUDA_VISIBLE_DEVICES"] = vis
            pids = addprocs(1; exeflags=`--project=$(proj) -t $(threads_per) --startup-file=no $(sysflag)`, env=env)
            append!(team, pids)
        end
        # Load the GPU eigensolve stack + pin the shared GPU on the team workers. Use
        # remotecall_eval (what @everywhere lowers to) so the `using` statements run at
        # top level on the workers -- @everywhere itself cannot be expanded inside a function.
        Distributed.remotecall_eval(Main, team, quote
            using CUDA
            using TJLFEP
            using TJLF
            using LinearAlgebra
            LinearAlgebra.BLAS.set_num_threads(1)
            CUDA.functional() && CUDA.device!(first(CUDA.devices()))
        end)
        return f(team)
    catch err
        @warn "_with_mps_team: team setup failed; falling back to single-process threaded inner" exception = (err, catch_backtrace())
        return f(nothing)
    finally
        if !isempty(team)
            try
                rmprocs(team)
            catch
            end
        end
    end
end

"""Unpack `mainsub` return for PROCESS_IN=5: `((growth, ep, mt, marginal_ql), buffers)`."""
function _unpack_mainsub!(ret)
    (growth, ep, mt, marginal_ql), _buffers = ret
    return growth, ep, mt, marginal_ql
end

"""
Write one radius's `mainsub` output buffers `(sf_buf, wf_buf_all)` to disk under `out_dir`:
the scalefactor buffer to `out.scalefactor<suffix>` and each `(filename, lines)` entry of
`wf_buf_all` (wavefunction files for PROCESS_IN=5; eigenvalue/width-scan spectra for
PROCESS_IN=3) to its own file. No-ops on `nothing`/empty buffers.
"""
function _write_radius_buffers(buffers, suffix::AbstractString; out_dir::AbstractString=".")
    buffers === nothing && return nothing
    sf_buf, wf_buf_all = buffers
    if sf_buf !== nothing && !isempty(sf_buf)
        open(joinpath(out_dir, "out.scalefactor" * suffix), "w") do io
            for line in sf_buf
                println(io, line)
            end
        end
    end
    if wf_buf_all !== nothing
        for (fname, fbuf) in wf_buf_all
            if fbuf !== nothing && !isempty(fbuf)
                open(joinpath(out_dir, fname), "w") do io
                    for line in fbuf
                        println(io, line)
                    end
                end
            end
        end
    end
    return nothing
end

"""Run mainsub for one scan index (radius). Used by threads and distributed loops."""
function _runTHD_radius!(
    i::Int,
    arrTGLFEP,
    arrMTGLF,
    arrgrowth,
    printout::Bool,
    use_gpu::Bool;
    stdout_lock::Union{ReentrantLock,Nothing} = nothing,
    inner::Symbol = :threads,
    team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    ql_flux_scan::Bool = false,
    solver::Symbol = :grid,
    refine_rounds::Int = 1,
    k_max::Int = _k_max_env(),
    arrbuffers::Union{Nothing,AbstractVector} = nothing,
)
    arrTGLFEP[i].IR = arrTGLFEP[i].IR_EXP[i]
    ir = arrTGLFEP[i].IR
    arrTGLFEP[i].SUFFIX = "_r" * lpad(string(ir), 3, '0')
    arrTGLFEP[i].FACTOR_IN = arrTGLFEP[i].FACTOR[i]
    if stdout_lock !== nothing
        lock(stdout_lock) do
            println("=============================================================")
            println("pre mainsub")
            println("i is ", i, " ir is ", ir)
            println("=============================================================")
        end
    end
    ret = TJLFEP.mainsub(arrTGLFEP[i], arrMTGLF[i], printout; use_gpu=use_gpu, inner=inner, team=team,
                         ql_flux_scan=ql_flux_scan, solver=solver, refine_rounds=refine_rounds, k_max=k_max)
    growth, ep, mt, _ = _unpack_mainsub!(ret)
    arrgrowth[i] = growth
    arrTGLFEP[i] = ep
    arrMTGLF[i] = mt
    if arrbuffers !== nothing
        arrbuffers[i] = ret[2]
    end
    return nothing
end

function _resolve_runTHD_parallel(parallel::Symbol)
    parallel == :auto && return nworkers() > 1 ? :distributed : :threads
    return parallel
end

"""Apply FACTOR scaling, F_REAL, and profile γ fields before the radial scan."""
function _apply_runthd_expro_setup!(Options::Options, profile::profile, expro)
    profile.gammaE = expro.gammaE
    profile.gammap = expro.gammap
    profile.omegaGAM = expro.omegaGAM

    if isempty(Options.IR_EXP) || all(iszero, Options.IR_EXP)
        Options.IR_EXP = ir_exp_from_scan(profile.NR, profile.IRS, Options.SCAN_N)
        println("IR_EXP not set, using linear spacing: ", Options.IR_EXP)
    else
        println("options ir_exp is ", Options.IR_EXP)
    end

    ni, Ti, dlnnidr, dlntidr, cs, rmin_ex = expro.ni, expro.Ti, expro.dlnnidr, expro.dlntidr, expro.cs, expro.rmin_ex
    nr = profile.NR
    if ismissing(Options.F_REAL) || length(Options.F_REAL) != nr
        Options.F_REAL = ones(nr)
    else
        Options.F_REAL .= 1.0
    end

    if Options.INPUT_PROFILE_METHOD == 2
        dpdr_EP = similar(ni)
        for i in eachindex(dpdr_EP)
            dpdr_EP[i] = ni[i] * Ti[i] * (dlnnidr[i] + dlntidr[i])
        end
        dpdr_EP_abs = abs.(dpdr_EP)
        dpdr_EP_max = maximum(dpdr_EP_abs)
        if Options.PROCESS_IN != 5
            for ir in 1:Options.SCAN_N
                Options.FACTOR[ir] = Options.FACTOR[ir] * dpdr_EP_max / dpdr_EP_abs[Options.IR_EXP[ir]]
            end
        end
        Options.FACTOR_MAX_PROFILE .= Options.FACTOR
    end

    if Options.REAL_FREQ == 1
        # EXPRO `cs`/`rmin_ex` from readEXPRO are length 201; profile.NR is the gacode grid (e.g. 101).
        # Match Fortran: F_REAL(i) = cs(i) / rmin_ex(NR) / (2π×1e3) for i = 1:NR.
        rmin_ref = rmin_ex[min(nr, length(rmin_ex))]
        Options.F_REAL .= (cs[1:nr] ./ rmin_ref) ./ (2 * pi * 1.0e3)
    end
    return nothing
end

"""Radial scan + α post-processing given in-memory `Options` and `profile`."""
function _runTHD_core!(
    Options::Options,
    profile::profile,
    expro;
    printout::Bool=false,
    use_gpu::Bool=false,
    parallel::Symbol=:auto,
    inner::Symbol=:threads,
    team::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    ql_flux_scan::Bool=false,
    solver::Symbol=:grid,
    refine_rounds::Int=1,
    k_max::Int=_k_max_env(),
)
    ni, Ti, dlnnidr, dlntidr = expro.ni, expro.Ti, expro.dlnnidr, expro.dlntidr

    n_ir = Options.SCAN_N
    Ts = fill(Options, n_ir)
    Ts[1] = deepcopy(Options)
    for i in 2:n_ir
        Ts[i] = deepcopy(Ts[i - 1])
    end
    arrTGLFEP = Ts
    arrMTGLF = Vector{typeof(profile)}(undef, n_ir)
    arrMTGLF[1] = deepcopy(profile)
    for i in 2:n_ir
        arrMTGLF[i] = deepcopy(arrMTGLF[i - 1])
    end
    arrgrowth = fill(fill(NaN, (5, 10, 10, Options.NMODES)), n_ir)

    par = _resolve_runTHD_parallel(parallel)
    if par === :threads
        stdout_lock = ReentrantLock()
        arrbuffers = printout ? Vector{Any}(undef, n_ir) : nothing
        # set BLAS=1 before spawning radius threads so the nested per-radius
        # kwscale_scan scope short-circuits (no concurrent set_num_threads)
        TJLF.with_blas_threads(1) do
        Threads.@threads for i in 1:n_ir
            _runTHD_radius!(i, arrTGLFEP, arrMTGLF, arrgrowth, printout, use_gpu;
                            stdout_lock=stdout_lock, inner=inner, team=team, ql_flux_scan=ql_flux_scan,
                            solver=solver, refine_rounds=refine_rounds, k_max=k_max, arrbuffers=arrbuffers)
        end
        end
        if printout
            for i in 1:n_ir
                _write_radius_buffers(arrbuffers[i], coalesce(arrTGLFEP[i].SUFFIX, ""))
            end
        end
    elseif par === :distributed
        pmap_outputs = pmap(i -> begin
            ep = deepcopy(arrTGLFEP[i])
            mt = deepcopy(arrMTGLF[i])
            ep.IR = ep.IR_EXP[i]
            ir = ep.IR
            ep.SUFFIX = "_r" * lpad(string(ir), 3, '0')
            ep.FACTOR_IN = ep.FACTOR[i]
            println("worker $(myid()) on $(gethostname()): i=$i ir=$ir start")
            flush(stdout)
            ret = TJLFEP.mainsub(ep, mt, printout; use_gpu=use_gpu, ql_flux_scan=ql_flux_scan,
                                 solver=solver, refine_rounds=refine_rounds, k_max=k_max)
            println("worker $(myid()) on $(gethostname()): i=$i ir=$ir done")
            flush(stdout)
            return ret
        end, 1:n_ir)
        results = [_unpack_mainsub!(p) for p in pmap_outputs]
        all_buffers = [p[2] for p in pmap_outputs]
        for (i, (growth, ep, mt, _)) in enumerate(results)
            arrgrowth[i] = growth
            arrTGLFEP[i] = ep
            arrMTGLF[i] = mt
        end
        if printout
            for i in 1:n_ir
                _write_radius_buffers(all_buffers[i], coalesce(arrTGLFEP[i].SUFFIX, ""))
            end
        end
    else
        error("parallel must be :auto, :threads, or :distributed (got $parallel)")
    end

    profile.IS = arrMTGLF[1].IS
    Options = arrTGLFEP[1]

    kymark_out = fill(NaN, Options.SCAN_N)
    width = fill(NaN, Options.SCAN_N)

    if !Options.WIDTH_IN_FLAG
        for i in 1:n_ir
            width[i] = arrTGLFEP[i].WIDTH_IN
            kymark_out[i] = arrTGLFEP[i].KYMARK
        end
    end

    outTGLFEP_buffer = String[]
    if printout
        push!(outTGLFEP_buffer, "process_in = $(Options.PROCESS_IN)")
        if Options.PROCESS_IN <= 1
            push!(outTGLFEP_buffer, "mode_in = $(Options.MODE_IN)")
        end
        if (Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5)
            push!(outTGLFEP_buffer, "threshold_flag = $(Options.THRESHOLD_FLAG)")
        end
        push!(outTGLFEP_buffer, "ky_mode = $(Options.KY_MODEL)")
        push!(outTGLFEP_buffer, "--------------------------------------------------------------")
        push!(outTGLFEP_buffer, "scan_n = $(Options.SCAN_N)")
        push!(outTGLFEP_buffer, "irs = $(Options.IRS)")
        push!(outTGLFEP_buffer, "n_basis = $(Options.N_BASIS)")
        push!(outTGLFEP_buffer, "scan_method = $(Options.SCAN_METHOD)")
        if Options.WIDTH_IN_FLAG
            push!(outTGLFEP_buffer, "ir,  width")
            for i in 1:Options.SCAN_N
                push!(outTGLFEP_buffer, "$(Options.IRS + i - 1) $(width[i])")
            end
        else
            push!(outTGLFEP_buffer, "ir,  width,  kymark")
            for i in 1:Options.SCAN_N
                push!(outTGLFEP_buffer, "$(Options.IRS + i - 1) $(width[i]) $(kymark_out[i])")
            end
        end
        push!(outTGLFEP_buffer, "--------------------------------------------------------------")
        push!(outTGLFEP_buffer, "factor_in_profile = $(Options.FACTOR_IN_PROFILE)")
        if Options.FACTOR_IN_PROFILE
            for i in 1:Options.SCAN_N
                push!(outTGLFEP_buffer, string(Options.FACTOR[i]))
            end
        else
            push!(outTGLFEP_buffer, string(Options.FACTOR[1]))
        end
        push!(outTGLFEP_buffer, "width_in_flag = $(Options.WIDTH_IN_FLAG)")
        if !Options.WIDTH_IN_FLAG
            push!(outTGLFEP_buffer, "width_min = $(Options.WIDTH_MIN) width_max = $(Options.WIDTH_MAX)")
        end
    end

    SFmin = fill(0.0, Options.SCAN_N)
    dpdr_crit_out = fill(NaN, profile.NR)
    dndr_crit_out = fill(NaN, profile.NR)

    if (Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5)
        if printout
            push!(outTGLFEP_buffer, "**************************************************************")
            push!(outTGLFEP_buffer, "************** The critical EP density gradient **************")
            push!(outTGLFEP_buffer, "**************************************************************")
        end

        SFmin_out = fill(0.0, profile.NR)
        dndr_crit = fill(NaN, Options.SCAN_N)
        dpdr_crit = fill(NaN, Options.SCAN_N)

        if Options.THRESHOLD_FLAG == 0
            for i in 1:n_ir
                SFmin[i] = arrTGLFEP[i].FACTOR_IN
            end
            if printout
                push!(outTGLFEP_buffer, "--------------------------------------------------------------")
                push!(outTGLFEP_buffer, "SFmin")
            end

            SFmin, SFmin_out, ir_min, ir_max, l_accept_profile = tjlfep_complete_output(SFmin, Options, profile)

            if printout
                push!(outTGLFEP_buffer, string(SFmin, " SFmin after complete_output"))
            end

            if (ir_min - Options.IRS + 1) > 1
                if ir_min - Options.IRS > Options.SCAN_N
                    SFmin[1:Options.SCAN_N] = Options.FACTOR_MAX_PROFILE[1:Options.SCAN_N]
                else
                    SFmin[1:(ir_min - Options.IRS)] = Options.FACTOR_MAX_PROFILE[1:(ir_min - Options.IRS)]
                end
                if Options.IRS > 1
                    SFmin_out[1:(Options.IRS - 1)] .= Options.FACTOR_MAX_PROFILE[1]
                end
                if ir_min - Options.IRS > Options.SCAN_N
                    SFmin_out[Options.IRS:(Options.SCAN_N + 1)] = Options.FACTOR_MAX_PROFILE[1:Options.SCAN_N]
                else
                    SFmin_out[Options.IRS:(ir_min - 1)] = Options.FACTOR_MAX_PROFILE[1:(ir_min - Options.IRS)]
                end
            end

            if (ir_max - Options.IRS + 1) < Options.SCAN_N
                SFmin[(ir_max - Options.IRS + 2):Options.SCAN_N] =
                    Options.FACTOR_MAX_PROFILE[(ir_max - Options.IRS + 2):Options.SCAN_N]
                if Options.IRS + Options.SCAN_N - 1 < profile.NR
                    SFmin_out[(Options.IRS + Options.SCAN_N):profile.NR] .= Options.FACTOR_MAX_PROFILE[Options.SCAN_N]
                end
                SFmin_out[(ir_max + 1):(Options.IRS + Options.SCAN_N - 1)] =
                    Options.FACTOR_MAX_PROFILE[(ir_max - Options.IRS + 2):Options.SCAN_N]
            end

            if printout
                push!(outTGLFEP_buffer, string(SFmin, " SFmin after Max assign"))
            end

            if printout
                for i in 1:Options.SCAN_N
                    if l_accept_profile[i]
                        push!(outTGLFEP_buffer, string(SFmin[i]))
                    else
                        push!(outTGLFEP_buffer, string(SFmin_out[i + Options.IRS - 1], "   (*)"))
                    end
                    push!(outTGLFEP_buffer, "--------------------------------------------------------------")
                    push!(outTGLFEP_buffer, string(l_accept_profile))
                    push!(outTGLFEP_buffer, string(ir_min, " : ", ir_max))
                end
            end

            if Options.INPUT_PROFILE_METHOD == 2
                dndr_crit .= 10000.0
                for i in 1:Options.SCAN_N
                    if SFmin[i] < 9000.0
                        dndr_crit[i] = SFmin[i] * ni[Int(Options.IR_EXP[i])] * dlnnidr[Int(Options.IR_EXP[i])]
                    elseif (i < ir_min - Options.IRS + 1) || (i > ir_max - Options.IRS + 1)
                        dndr_crit[i] = Options.FACTOR_MAX_PROFILE[i] * ni[Int(Options.IR_EXP[i])] * dlnnidr[Int(Options.IR_EXP[i])]
                    end
                end
                dndr_crit, dndr_crit_out, _, _, _ = tjlfep_complete_output(dndr_crit, Options, profile)
                if printout
                    write_crit_grad("alpha_dndr_crit.input", "Density critical gradient (10^19/m^4)", dndr_crit_out)
                end
            end

            if Options.INPUT_PROFILE_METHOD == 2
                dpdr_crit .= 10000.0
                nr = profile.NR
                dpdr_EP = Vector{Float64}(undef, nr)
                for i in 1:nr
                    dpdr_EP[i] = ni[i] * Ti[i] * (dlnnidr[i] + dlntidr[i]) * 0.16022
                end
                for i in 1:Options.SCAN_N
                    if SFmin[i] < 9000.0
                        if (Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5)
                            case = Options.SCAN_METHOD
                            if case == 1
                                dpdr_scale = SFmin[i]
                            elseif case == 2
                                dpdr_scale = (SFmin[i] * dlnnidr[Options.IR_EXP[i]] + dlntidr[Options.IR_EXP[i]]) /
                                             (dlnnidr[Options.IR_EXP[i]] + dlntidr[Options.IR_EXP[i]])
                            end
                            dpdr_crit[i] = dpdr_scale * dpdr_EP[Options.IR_EXP[i]]
                        end
                    elseif (i < ir_min - Options.IRS + 1) || (i > ir_max - Options.IRS + 1)
                        dpdr_crit[i] = Options.FACTOR_MAX_PROFILE[i] * dpdr_EP[Options.IR_EXP[i]]
                    end
                end
                dpdr_crit, dpdr_crit_out, _, _, _ = tjlfep_complete_output(dpdr_crit, Options, profile)
                if printout
                    write_crit_grad("alpha_dpdr_crit.input", "Pressure critical gradient (10 kPa/m)", dpdr_crit_out)
                end
            end

            if printout
                push!(outTGLFEP_buffer, "--------------------------------------------------------------")
                push!(outTGLFEP_buffer, "The EP density threshold n_EP/n_e (%) for gamma_AE = 0")
                for i in 1:Options.SCAN_N
                    push!(outTGLFEP_buffer, string(SFmin[i] * profile.AS[Options.IRS + i - 1, profile.IS] * 100.0))
                end
            end

            if printout
                push!(outTGLFEP_buffer, "--------------------------------------------------------------")
                push!(outTGLFEP_buffer, "The EP beta crit (%) = beta_e*(n_EP_th/n_e)*(T_EP/T_e)")
                for i in 1:Options.SCAN_N
                    if coalesce(profile.GEOMETRY_FLAG, 1) == 0
                        push!(outTGLFEP_buffer, string(SFmin[i] * profile.BETAE[Options.IRS + i - 1] * 100.0 *
                            profile.AS[Options.IRS + i - 1, profile.IS] * profile.TAUS[Options.IRS + i - 1, profile.IS]))
                    else
                        push!(outTGLFEP_buffer, string(SFmin[i] * profile.BETAE[Options.IRS + i - 1] * 100.0 *
                            profile.AS[Options.IRS + i - 1, profile.IS] * profile.TAUS[Options.IRS + i - 1, profile.IS] *
                            profile.KAPPA[Options.IRS + i - 1]^2))
                    end
                end
            end
        end
    end

    if printout
        open("out.TGLFEP", "w") do io
            for line in outTGLFEP_buffer
                println(io, line)
            end
        end
    end

    return width, kymark_out, SFmin, dpdr_crit_out, dndr_crit_out
end

"""
    runTHD(tglfepfilepath, mtglffilepath, exprofilepath; kwargs...)

Run a TJLFEP critical-gradient scan from the file-based inputs: an
`input.TGLFEP` (scan control), an `input.MTGLF` (profiles), and an `input.EXPRO`
(experimental profiles for the α/EP quantities).

Reads the three files into [`Options`](@ref)/[`profile`](@ref), applies the
EXPRO-derived setup, and runs the scan.

# Keywords
- `printout::Bool=false`   - also write an `out.TGLFEP` summary file
- `use_gpu=:auto`          - eigensolve device: `:auto` (GPU when CUDA is functional, via
                             `TJLF.pick_device`), `:gpu` (error if none), `:cpu`, or a `Bool`
- `parallel::Symbol=:auto` - radial parallel layout
- `inner::Symbol=:threads` - per-radius backend (`:threads` or `:mps_team`)
- `team`                   - GPU ids for the MPS team layout
- `ql_flux_scan::Bool`     - also extract QL fluxes at the marginal point
- `solver::Symbol=:grid`   - critical-factor solver (`:grid`, `:ad`, `:truth`, ...)
- `refine_rounds::Int=1`   - AD (ky,width) refinement passes
- `nmodes::Int`            - number of eigenmodes kept per solve (default from
  `TGLFEP_JULIA_NMODES`, else 4). Applies to all solvers; raise it to catch a higher
  (e.g. 5th) AE mode.
- `k_max::Int`             - number of `:grid` scale-factor bracketing rounds
  (default from `TGLFEP_JULIA_K_MAX`, else 4). `:grid` solver only (inert for the AD solvers).

Returns `(width, kymark_out, SFmin, dpdr_crit_out, dndr_crit_out)`.

An IMAS data-dictionary method `runTHD(dd, rho, OptionsDict; ...)` is also
available when the `TJLFEPIMASExt` extension is loaded (e.g. under FUSE).
"""
function runTHD(tglfepfilepath::String, mtglffilepath::String, exprofilepath::String;
                printout::Bool=false, use_gpu::Union{Bool,Symbol}=:auto, parallel::Symbol=:auto,
                inner::Symbol=:threads, team::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                ql_flux_scan::Bool=_ql_flux_scan_env(), solver::Symbol=:grid, refine_rounds::Int=1,
                nmodes::Int=_nmodes_env(), k_max::Int=_k_max_env())

    use_gpu = _resolve_use_gpu(use_gpu)
    # Thread safety: Threads.@threads runs each iteration in a separate Julia task.
    # CUDA.jl v5 assigns per-task streams, so concurrent GPU calls are stream-isolated.

    @assert isfile(tglfepfilepath) "Requested TGLFEP input file path does not exist"
    @assert isfile(mtglffilepath) "Requested MTGLF input file path does not exist"
    @assert isfile(exprofilepath) "Requested EXPRO input file path does not exist"

    prof = TJLFEP.readMTGLF(mtglffilepath)
    profile = prof[1]
    ir_exp = prof[2]
    Options = TJLFEP.readTGLFEP(tglfepfilepath, ir_exp; nmodes=nmodes)

    gacode_dump = get(ENV, "GACODE_DUMP", nothing)
    ni, Ti, dlnnidr, dlntidr, cs, rmin_ex, gammaE, gammap, omegaGAM = TJLFEP.read_expro_for_alpha(
        exprofilepath, profile, Options.IS_EP; gacode_file=gacode_dump,
    )
    expro = (
        ni=ni, Ti=Ti, dlnnidr=dlnnidr, dlntidr=dlntidr, cs=cs, rmin_ex=rmin_ex,
        gammaE=gammaE, gammap=gammap, omegaGAM=omegaGAM,
    )
    _apply_runthd_expro_setup!(Options, profile, expro)
    return _runTHD_core!(Options, profile, expro; printout=printout, use_gpu=use_gpu, parallel=parallel,
                         inner=inner, team=team, ql_flux_scan=ql_flux_scan, solver=solver,
                         refine_rounds=refine_rounds, k_max=k_max)
end

"""
    runTHD_from_gacode(gacode_file, tglfep_file; kwargs...)

Run TJLFEP from `input.gacode` + `input.TGLFEP` only. No `dump.gacode`, `input.MTGLF`,
or `input.EXPRO` required. Keywords as for [`runTHD`](@ref); `use_gpu` defaults to `:auto`
(GPU when CUDA is functional).

# Example
```julia
using TJLFEP
width, kymark, SFmin, dpdr, dndr = runTHD_from_gacode(
    "case/input.gacode",
    "case/input.TGLFEP";
    printout=true,
)
```
"""
function runTHD_from_gacode(
    gacode_file::AbstractString,
    tglfep_file::AbstractString;
    printout::Bool=false,
    use_gpu::Union{Bool,Symbol}=:auto,
    parallel::Symbol=:auto,
    inner::Symbol=:threads,
    team::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    ql_flux_scan::Bool=_ql_flux_scan_env(),
    solver::Symbol=:grid,
    refine_rounds::Int=1,
    nmodes::Int=_nmodes_env(),
    k_max::Int=_k_max_env(),
)
    use_gpu = _resolve_use_gpu(use_gpu)
    Options, profile, expro = preprocess_gacode_inputs(gacode_file, tglfep_file; nmodes=nmodes)
    _apply_runthd_expro_setup!(Options, profile, expro)
    return _runTHD_core!(Options, profile, expro; printout=printout, use_gpu=use_gpu, parallel=parallel,
                         inner=inner, team=team, ql_flux_scan=ql_flux_scan, solver=solver,
                         refine_rounds=refine_rounds, k_max=k_max)
end  # runTHD_from_gacode

"""Apply `tjlfep_complete_output` padding on `SFmin`, then write α profiles (matches `_runTHD_core`)."""
function _gacode_alpha_postprocess!(
    SFmin::Vector{Float64},
    Options::Options,
    profile::profile,
    expro;
    out_dir::AbstractString=".",
    printout::Bool=true,
)
    ni, Ti, dlnnidr, dlntidr = expro.ni, expro.Ti, expro.dlnnidr, expro.dlntidr
    scan_n = Options.SCAN_N
    dndr_crit_out = fill(NaN, profile.NR)
    dpdr_crit_out = fill(NaN, profile.NR)

    # PROCESS_IN 6 (MODE_IN=4 variant) shares the mode-5 threshold post-processing. Unlike the
    # in-process paths, finalize_gacode_scan re-reads the input file and never runs mainsub, so
    # PROCESS_IN is still the pre-fold 6 here (it is not collapsed to 5) and must be handled.
    if Options.THRESHOLD_FLAG != 0 ||
       !((Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5) || (Options.PROCESS_IN == 6))
        return SFmin, dndr_crit_out, dpdr_crit_out
    end

    SFmin_out = fill(0.0, profile.NR)
    SFmin, SFmin_out, ir_min, ir_max, l_accept_profile = tjlfep_complete_output(SFmin, Options, profile)

    if (ir_min - Options.IRS + 1) > 1
        if ir_min - Options.IRS > scan_n
            SFmin[1:scan_n] = Options.FACTOR_MAX_PROFILE[1:scan_n]
        else
            SFmin[1:(ir_min - Options.IRS)] = Options.FACTOR_MAX_PROFILE[1:(ir_min - Options.IRS)]
        end
        if Options.IRS > 1
            SFmin_out[1:(Options.IRS - 1)] .= Options.FACTOR_MAX_PROFILE[1]
        end
        if ir_min - Options.IRS > scan_n
            SFmin_out[Options.IRS:(scan_n + 1)] = Options.FACTOR_MAX_PROFILE[1:scan_n]
        else
            SFmin_out[Options.IRS:(ir_min - 1)] = Options.FACTOR_MAX_PROFILE[1:(ir_min - Options.IRS)]
        end
    end

    if (ir_max - Options.IRS + 1) < scan_n
        SFmin[(ir_max - Options.IRS + 2):scan_n] =
            Options.FACTOR_MAX_PROFILE[(ir_max - Options.IRS + 2):scan_n]
        if Options.IRS + scan_n - 1 < profile.NR
            SFmin_out[(Options.IRS + scan_n):profile.NR] .= Options.FACTOR_MAX_PROFILE[scan_n]
        end
        SFmin_out[(ir_max + 1):(Options.IRS + scan_n - 1)] =
            Options.FACTOR_MAX_PROFILE[(ir_max - Options.IRS + 2):scan_n]
    end

    if Options.INPUT_PROFILE_METHOD == 2
        dndr_crit = fill(10000.0, scan_n)
        for i in 1:scan_n
            if SFmin[i] < 9000.0
                dndr_crit[i] = SFmin[i] * ni[Int(Options.IR_EXP[i])] * dlnnidr[Int(Options.IR_EXP[i])]
            elseif (i < ir_min - Options.IRS + 1) || (i > ir_max - Options.IRS + 1)
                dndr_crit[i] = Options.FACTOR_MAX_PROFILE[i] * ni[Int(Options.IR_EXP[i])] * dlnnidr[Int(Options.IR_EXP[i])]
            end
        end
        dndr_crit, dndr_crit_out, _, _, _ = tjlfep_complete_output(dndr_crit, Options, profile)
        if printout
            write_crit_grad(joinpath(out_dir, "alpha_dndr_crit.input"), "Density critical gradient (10^19/m^4)", dndr_crit_out)
        end

        dpdr_crit = fill(10000.0, scan_n)
        nr = profile.NR
        dpdr_EP = Vector{Float64}(undef, nr)
        for i in 1:nr
            dpdr_EP[i] = ni[i] * Ti[i] * (dlnnidr[i] + dlntidr[i]) * 0.16022
        end
        for i in 1:scan_n
            if SFmin[i] < 9000.0
                if (Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5) || (Options.PROCESS_IN == 6)
                    dpdr_scale = if Options.SCAN_METHOD == 1
                        SFmin[i]
                    elseif Options.SCAN_METHOD == 2
                        denom = dlnnidr[Options.IR_EXP[i]] + dlntidr[Options.IR_EXP[i]]
                        denom == 0 ? SFmin[i] :
                            (SFmin[i] * dlnnidr[Options.IR_EXP[i]] + dlntidr[Options.IR_EXP[i]]) / denom
                    else
                        SFmin[i]
                    end
                    dpdr_crit[i] = dpdr_scale * dpdr_EP[Options.IR_EXP[i]]
                end
            elseif (i < ir_min - Options.IRS + 1) || (i > ir_max - Options.IRS + 1)
                dpdr_crit[i] = Options.FACTOR_MAX_PROFILE[i] * dpdr_EP[Options.IR_EXP[i]]
            end
        end
        dpdr_crit, dpdr_crit_out, _, _, _ = tjlfep_complete_output(dpdr_crit, Options, profile)
        if printout
            write_crit_grad(joinpath(out_dir, "alpha_dpdr_crit.input"), "Pressure critical gradient (10 kPa/m)", dpdr_crit_out)
        end
    end

    return SFmin, dndr_crit_out, dpdr_crit_out
end

"""
    _resolve_use_gpu(use_gpu) -> Bool

Device selection shared by the public entry points: a `Bool` passes through; `:auto` picks the
GPU when CUDA is functional (`TJLF.pick_device(:auto)`), `:gpu` errors if none is available,
`:cpu` forces the CPU path. Resolved once at the entry point so the internal per-radius code
keeps a plain `Bool`.
"""
_resolve_use_gpu(use_gpu::Bool) = use_gpu
_resolve_use_gpu(use_gpu::Symbol) = TJLF.pick_device(use_gpu) === :gpu

"""
Slurm task id (0-based): `SLURM_ARRAY_TASK_ID` for `--array` jobs, else `SLURM_PROCID`
for multi-task jobs (`srun -n SCAN_N`), else `0`.
"""
function slurm_array_task_id()
    for key in ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASKID")
        haskey(ENV, key) || continue
        return parse(Int, ENV[key])
    end
    if haskey(ENV, "SLURM_PROCID")
        return parse(Int, ENV["SLURM_PROCID"])
    end
    return 0
end

"""
    run_gacode_scan_task(gacode_file, tglfep_file, scan_index; out_dir, use_gpu, printout)

Run **one** radius of a `SCAN_N` job (for Slurm `--array=0-(SCAN_N-1)`).
Writes `task_<scan_index>.jls` under `out_dir` with `sfmin`, `width`, `kymark`, and optional
`out.scalefactor_r###` when `printout=true`. `use_gpu` defaults to `:auto` (GPU when CUDA is
functional); pass `true`/`false`/`:gpu`/`:cpu` to force a device.

Use `finalize_gacode_scan` after all array tasks finish to build α profiles.
"""
function run_gacode_scan_task(
    gacode_file::AbstractString,
    tglfep_file::AbstractString,
    scan_index::Integer;
    out_dir::AbstractString=".",
    use_gpu::Union{Bool,Symbol}=:auto,
    printout::Bool=false,
    inner::Symbol=:threads,
    team::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    ql_flux_scan::Bool=_ql_flux_scan_env(),
    solver::Symbol=:grid,
    refine_rounds::Int=1,
    nmodes::Int=_nmodes_env(),
    k_max::Int=_k_max_env(),
)
    use_gpu = _resolve_use_gpu(use_gpu)
    mkpath(out_dir)
    Options, profile, expro = preprocess_gacode_inputs(gacode_file, tglfep_file; nmodes=nmodes)
    _apply_runthd_expro_setup!(Options, profile, expro)
    1 <= scan_index <= Options.SCAN_N ||
        error("scan_index=$scan_index out of range 1:$(Options.SCAN_N)")

    ep = deepcopy(Options)
    mt = deepcopy(profile)
    ep.IR = ep.IR_EXP[scan_index]
    ir = ep.IR
    ep.SUFFIX = "_r" * lpad(string(ir), 3, '0')
    ep.FACTOR_IN = ep.FACTOR[scan_index]

    logmsg = printout ? println : (_, args...) -> nothing
    logmsg("scan_index=$scan_index ir=$ir use_gpu=$use_gpu host=$(gethostname()) inner=$inner team=$(team === nothing ? "—" : length(team))")

    ret = TJLFEP.mainsub(ep, mt, printout; use_gpu=use_gpu, inner=inner, team=team, ql_flux_scan=ql_flux_scan,
                         solver=solver, refine_rounds=refine_rounds, k_max=k_max)
    growth, ep_out, mt_out, fourth = _unpack_mainsub!(ret)
    sfmin = ep_out.FACTOR_IN
    width = coalesce(ep_out.WIDTH_IN, ep_out.WIDTH, NaN)
    kymark = coalesce(ep_out.KYMARK, NaN)
    # PROCESS_IN=3 returns the in-memory γ/ω spectra (Dict mode_in => (ky,gamma,freq)) in the
    # 4th slot; for the threshold paths the 4th slot is marginal-QL data we don't serialize.
    spectra = ep_out.PROCESS_IN == 3 ? fourth : nothing

    if printout
        _write_radius_buffers(ret[2], coalesce(ep_out.SUFFIX, ""); out_dir=out_dir)
    end

    result = (
        scan_index=scan_index,
        ir=ir,
        sfmin=Float64(sfmin),
        width=Float64(width),
        kymark=Float64(kymark),
        spectra=spectra,
        use_gpu=use_gpu,
        hostname=gethostname(),
    )
    task_file = joinpath(out_dir, "task_$(scan_index).jls")
    open(task_file, "w") do io
        serialize(io, result)
    end
    logmsg("wrote $task_file sfmin=$sfmin")
    return result
end

"""
    finalize_gacode_scan(gacode_file, tglfep_file, tasks_dir; printout)

Merge per-radius `task_*.jls` files and write `alpha_*_crit.input` (and optional `out.TGLFEP` header).
"""
function finalize_gacode_scan(
    gacode_file::AbstractString,
    tglfep_file::AbstractString,
    tasks_dir::AbstractString;
    printout::Bool=true,
)
    Options, profile, expro = preprocess_gacode_inputs(gacode_file, tglfep_file)
    _apply_runthd_expro_setup!(Options, profile, expro)
    scan_n = Options.SCAN_N

    SFmin = Vector{Float64}(undef, scan_n)
    width = Vector{Float64}(undef, scan_n)
    kymark_out = Vector{Float64}(undef, scan_n)
    for i in 1:scan_n
        task_file = joinpath(tasks_dir, "task_$(i).jls")
        isfile(task_file) || error("missing array task output: $task_file")
        result = open(task_file) do io
            deserialize(io)
        end
        SFmin[i] = result.sfmin
        width[i] = result.width
        kymark_out[i] = result.kymark
    end

    SFmin, dndr_out, dpdr_out = _gacode_alpha_postprocess!(
        SFmin, Options, profile, expro; out_dir=tasks_dir, printout=printout)

    open(joinpath(tasks_dir, "sfmin_scan.txt"), "w") do io
        for (i, s) in enumerate(SFmin)
            println(io, i, " ", Options.IR_EXP[i], " ", s)
        end
    end

    println("finalize_gacode_scan: merged $scan_n tasks -> $tasks_dir")
    return width, kymark_out, SFmin, dpdr_out, dndr_out
end

# NOTE: the former TJLFEP-local `checkInput(::InputTJLF)` / `checkInput(::Vector{InputTJLF})`
# operated on the now-removed duplicate `TJLFEP.InputTJLF` type and were dead (the live path
# calls `TJLF.checkInput` directly, e.g. in tjlfep_ad_extensions.jl). Removed during the
# consolidation onto `TJLF.InputTJLF`.