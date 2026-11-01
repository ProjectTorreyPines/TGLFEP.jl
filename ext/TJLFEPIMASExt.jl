module TJLFEPIMASExt

# IMAS/FUSE entry points for TJLFEP. This package extension is loaded automatically
# by Julia only when IMAS, GACODE, and TurbulentTransport are all available in the
# environment (e.g. under FUSE). When TJLFEP is loaded standalone for the
# file-based path (`runTHD(::String,...)` / `runTHD_from_gacode`), this extension is
# NOT loaded, so `using TJLFEP` stays light (no IMAS/HDF5/FUSE pulled in).
#
# Previously this code lived in src/run_tjlfep_imas.jl behind a `TJLFEP_FILE_ONLY`
# ENV flag and a precompile-time `_FILE_ONLY` const. That was fragile because ENV is
# not part of Julia's precompile cache key. The weakdep/extension model makes the
# split automatic and cache-correct.

using TJLFEP
using IMAS
import GACODE
using TurbulentTransport
using Distributed
using Serialization

"""
    remap_extraEP_for_fortran_save!(extraEP, ep_slot)

Remap EP species data from `ep_slot` to `ep_slot-1` in `extraEP` so saved EXPRO
matches the Fortran 3-species convention.
"""
function TJLFEP.remap_extraEP_for_fortran_save!(extraEP::Dict, ep_slot::Int)
    for prefix in ("DENS", "TEMP", "DLNNDR", "DLNTDR")
        extraEP["$(prefix)_$(ep_slot - 1)"] = extraEP["$(prefix)_$ep_slot"]
        delete!(extraEP, "$(prefix)_$ep_slot")
    end
    return extraEP
end

"""
    save_imas_preprocessed_inputs(Options, profile, extraEP, dir)

Write input.TGLFEP / input.MTGLF / input.EXPRO after Fortran-style EP remapping.
"""
function TJLFEP.save_imas_preprocessed_inputs(Options, profile, extraEP::Dict, dir::AbstractString)
    ep_slot = extraEP["EP_SLOT"]
    extraEP_save = deepcopy(extraEP)
    remap_extraEP_for_fortran_save!(extraEP_save, ep_slot)
    save_all(Options, profile, extraEP_save, dir)
    return nothing
end

"""
    preprocess_imas_inputs(dd, rho, OptionsDict; verbose=false)

Build `Options`, `profile`, and `extraEP` from IMAS without running TJLF.
Also returns EP-slot EXPRO vectors used downstream by `runTHD`.
"""
function TJLFEP.preprocess_imas_inputs(dd::IMAS.dd, rho::AbstractVector{Float64}, OptionsDict::Dict{String, Any};
        verbose::Bool=false)
    input_tglfep, extraEP = TurbulentTransport.InputTGLFEP(dd, rho; is_ep=OptionsDict["IS_EP"])

    if verbose
        println("printing species masses")
        for is = 1:extraEP["NS"]
            println("mass[", is, "] = ", extraEP["MASS"][is])
        end
    end
    ep_slot = extraEP["EP_SLOT"]
    if verbose
        println("EP mass = ", getfield(input_tglfep[1], Symbol("MASS_$ep_slot")))
    end

    prof = TJLFEP.profile{Float64}(extraEP["NR"], extraEP["NS"])
    profile = TJLFEP.populate_tjlfep_profile!(prof, extraEP, extraEP["NR"], extraEP["NS"])

    nmodes = get(OptionsDict, "nmodes", TJLFEP._nmodes_env())  # keep/reject count; env TJLFEP_NMODES default 4
    Options = TJLFEP.Options{Float64}(OptionsDict["SCAN_N"], OptionsDict["WIDTH_IN_FLAG"], OptionsDict["nn"], extraEP["NR"], OptionsDict["jtscale_max"], nmodes)

    if OptionsDict["KY_MODEL"] == 0
        Options.NTOROIDAL = 4
    else
        Options.NTOROIDAL = 3
    end

    if OptionsDict["PROCESS_IN"] in (4, 5, 6)
        Options.NN = OptionsDict["nn"]
    end

    if !OptionsDict["FACTOR_IN_PROFILE"]
        Options.FACTOR = fill(OptionsDict["FACTOR_IN"], OptionsDict["SCAN_N"])
    end
    Options.FACTOR_MAX_PROFILE = Options.FACTOR

    for key in keys(OptionsDict)
        if hasfield(typeof(Options), Symbol(key))
            setfield!(Options, Symbol(key), OptionsDict[key])
        end
    end
    TJLFEP.enforce_width_floor!(Options)   # ballooning-width physics floor (WIDTH_FLOOR = 0.5)

    Options.IR_EXP = fill(0, Options.SCAN_N)
    Options.NMODES = nmodes

    Options.IS_EP = ep_slot - 1
    ni = extraEP["DENS_$ep_slot"]
    Ti = extraEP["TEMP_$ep_slot"]
    dlnnidr = extraEP["DLNNDR_$ep_slot"]
    dlntidr = extraEP["DLNTDR_$ep_slot"]
    cs = extraEP["CS"]
    rmin_ex = extraEP["RMIN"]
    gammaE = extraEP["gammaE"]
    gammap = extraEP["gammap"]
    omegaGAM = extraEP["omegaGAM"]

    if Options.INPUT_PROFILE_METHOD == 2
        Options.IR_EXP = fill(0, Options.SCAN_N)
        for i = 1:Options.SCAN_N
            if Options.SCAN_N != 1
                jr_exp = profile.IRS + floor(Int, (i - 1) * (profile.NR - profile.IRS) / (Options.SCAN_N - 1))
            else
                jr_exp = profile.IRS
            end
            Options.IR_EXP[i] = jr_exp
        end

        dpdr_EP = similar(ni, Float64)
        for i in eachindex(dpdr_EP)
            dpdr_EP[i] = ni[i] * Ti[i] * (dlnnidr[i] + dlntidr[i])
        end
        dpdr_EP_abs = abs.(dpdr_EP)
        dpdr_EP_max = maximum(dpdr_EP_abs)
        if Options.PROCESS_IN != 5
            for ir = 1:Options.SCAN_N
                Options.FACTOR[ir] = Options.FACTOR[ir] * dpdr_EP_max / dpdr_EP_abs[Options.IR_EXP[ir]]
            end
        end
        Options.FACTOR_MAX_PROFILE .= Options.FACTOR
    end

    Options.F_REAL .= 1.0
    if Options.REAL_FREQ == 1
        Options.F_REAL .= (cs[:] / (rmin_ex[profile.NR] * 100.0)) / (2 * pi * 1.0e3)
    end

    expro_state = (; ni, Ti, dlnnidr, dlntidr, cs, rmin_ex, gammaE, gammap, omegaGAM, ep_slot)
    return Options, profile, extraEP, expro_state
end

"""
    _dd_radius_output(Options, profile, i; use_gpu, inner, team, ql_flux_scan)

Run the kw-scan for ONE radius `i` and return the `pmap`-element
`((growth, tglfep_i, mtglf_i, marginal_ql), (scalefactor_buffer, wavebuffer_all))`.
Shared by `runTHD(::IMAS.dd)`'s in-process `pmap` and the SPMD per-radius task
`runTHD_dd_radius` so both produce byte-identical per-radius results.
"""
function _dd_radius_output(Options, profile, i::Int;
        use_gpu::Bool=false, inner::Symbol=:threads,
        team::Union{Nothing,AbstractVector{<:Integer}}=nothing,
        ql_flux_scan::Bool=false, printout::Bool=false, solver::Symbol=:grid,
        refine_rounds::Int=1, k_max::Int=TJLFEP._k_max_env(), extend_mode::Union{Nothing,Symbol}=nothing,
        wide_kdesc::Union{Nothing,Int}=nothing, faithful_confirm::Union{Nothing,Bool}=nothing)
    arrTGLFEP_i = deepcopy(Options)
    arrMTGLF_i = deepcopy(profile)

    arrTGLFEP_i.IR = arrTGLFEP_i.IR_EXP[i]
    ir = arrTGLFEP_i.IR
    arrTGLFEP_i.SUFFIX = "_r" * lpad(string(ir), 3, '0')
    arrTGLFEP_i.FACTOR_IN = arrTGLFEP_i.FACTOR[i]

    println("=============================================================")
    println("pre mainsub: i=", i, " ir=", ir, " inner=", inner,
        " team=", team === nothing ? 0 : length(team))
    println("=============================================================")

    return TJLFEP.mainsub(arrTGLFEP_i, arrMTGLF_i, printout; use_gpu=use_gpu, inner=inner,
        team=team, ql_flux_scan=ql_flux_scan, solver=solver, refine_rounds=refine_rounds, k_max=k_max,
        extend_mode=extend_mode, wide_kdesc=wide_kdesc, faithful_confirm=faithful_confirm)
end

"""
    runTHD(dd, rho, OptionsDict; ..., precomputed_dir="")

IMAS/FUSE entry point. Available when IMAS + GACODE + TurbulentTransport are loaded.

`precomputed_dir` (defaults to ENV `TJLFEP_PRECOMPUTED_DIR`): when set, the per-radius
kw-scans are NOT computed in-process; instead the `task_<i>.jls` files written by the
SPMD per-radius tasks (`runTHD_dd_radius`) are loaded. This is the merge phase of the
MPS-team SPMD layout (each radius ran on its own GPU with its own MPS team); the
cross-radius post-processing below is identical to the in-process path.
"""
function TJLFEP.runTHD(dd::IMAS.dd, rho::AbstractVector{Float64}, OptionsDict::Dict{String, Any};
                printout::Bool=false, saveFiles::Bool=false, dir::String="ddFiles", use_gpu::Union{Bool,Symbol}=:auto,
                ql_flux_scan::Bool=false, inner::Symbol=:threads, mps_team::Int=0, solver::Symbol=:grid,
                refine_rounds::Int=1, k_max::Int=TJLFEP._k_max_env(), extend_mode::Union{Nothing,Symbol}=nothing,
                wide_kdesc::Union{Nothing,Int}=nothing, faithful_confirm::Union{Nothing,Bool}=nothing,
                precomputed_dir::AbstractString=get(ENV, "TJLFEP_PRECOMPUTED_DIR", ""))

    use_gpu = TJLFEP._resolve_use_gpu(use_gpu)   # :auto -> GPU when CUDA is functional
    Options, profile, extraEP, expro_state = preprocess_imas_inputs(dd, rho, OptionsDict; verbose=printout)
    ni = expro_state.ni
    Ti = expro_state.Ti
    dlnnidr = expro_state.dlnnidr
    dlntidr = expro_state.dlntidr

    if saveFiles
        save_imas_preprocessed_inputs(Options, profile, extraEP, dir)
    end

    n_ir = Options.SCAN_N
    arrTGLFEP = [deepcopy(Options) for _ in 1:n_ir]
    arrMTGLF = [deepcopy(profile) for _ in 1:n_ir]
    arrgrowth = fill(fill(NaN,(5, 10, 10, Options.NMODES)), n_ir)

    if isempty(precomputed_dir)
        # In-process scan. NOTE: this runs each radius's inner kw-scan on the calling
        # process (threaded). MPS-team concurrency is NOT used here because addprocs is
        # only valid from a cluster master, not from a pmap worker -- the MPS path is the
        # SPMD layout (run_tjlfep inner=:mps_team -> runTHD_dd_radius per task + this merge).
        pmap_outputs = pmap(i -> _dd_radius_output(Options, profile, i;
            use_gpu=use_gpu, inner=:threads, team=nothing, ql_flux_scan=ql_flux_scan,
            printout=printout, solver=solver, refine_rounds=refine_rounds, k_max=k_max,
            extend_mode=extend_mode, wide_kdesc=wide_kdesc, faithful_confirm=faithful_confirm), 1:n_ir)
    else
        @info "runTHD(dd): loading precomputed per-radius results (SPMD merge)" precomputed_dir n_ir
        pmap_outputs = map(1:n_ir) do i
            tf = joinpath(precomputed_dir, "task_$(i).jls")
            isfile(tf) || error("runTHD(dd): missing SPMD task output $tf")
            open(Serialization.deserialize, tf)
        end
    end

    # pmap_outputs is a Vector of 2-tuples: ((growth, tglfep_i, mtglf_i), (scalefactor_buffer, wavebuffer_all))
    results   = [p[1] for p in pmap_outputs]   # Vector of (growth, tglfep_i, mtglf_i)
    all_buffers = [p[2] for p in pmap_outputs] # Vector of (scalefactor_buffer, wavebuffer_all)

    for (i, (growth, tglfep_i, mtglf_i)) in enumerate(results)
        arrgrowth[i] = growth
        arrTGLFEP[i] = tglfep_i
        arrMTGLF[i] = mtglf_i
    end

    # IS was set on each arrMTGLF[i] deepcopy by TJLF_map; copy it back to the
    # original profile which is used below for indexing (e.g. profile.AS[..., profile.IS])
    profile.IS = arrMTGLF[1].IS

    Options = arrTGLFEP[1]

    if printout
        # Write one scalefactor file and wavefunction file(s) per radial point.
        # Each entry in all_buffers[i] is (scalefactor_buffer, wavebuffer_all) for radius i.
        # wavebuffer_all is a Vector of (filename, buffer) pairs from kwscale_scan.
        for i in 1:n_ir
            sf_buf, wf_buf_all = all_buffers[i]
            suffix_i = coalesce(arrTGLFEP[i].SUFFIX, "")

            # Write scalefactor file for this radius
            if sf_buf !== nothing && !isempty(sf_buf)
                open("out.scalefactor" * suffix_i, "w") do io
                    for line in sf_buf
                        println(io, line)
                    end
                end
            end

            # Write wavefunction file(s) for this radius.
            # Each element is a (str_wf_file, buffer) pair produced by kwscale_scan.
            if wf_buf_all !== nothing && !isempty(wf_buf_all)
                for (str_wf_file, wfbuf) in wf_buf_all
                    if wfbuf !== nothing && !isempty(wfbuf)
                        open(str_wf_file, "w") do io
                            for line in wfbuf
                                println(io, line)
                            end
                        end
                    end
                end
            end
        end
    end
    
    kymark_out::Vector{Float64} = fill(NaN, Options.SCAN_N)
    width::Vector{Float64} = fill(NaN, Options.SCAN_N)

    if (!Options.WIDTH_IN_FLAG)
        for i = 1:n_ir
            width[i] = arrTGLFEP[i].WIDTH_IN
            kymark_out[i] = arrTGLFEP[i].KYMARK
        end
    end

    if (printout)
        io2 = open("out.TGLFEP", "w")
        println(io2, "process_in = ", Options.PROCESS_IN)

        if (Options.PROCESS_IN <= 1) println(io2, "mode_in = ", Options.MODE_IN) end
        if ((Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5)) println(io2, "threshold_flag = ", Options.THRESHOLD_FLAG) end

        println(io2, "ky_mode = ", Options.KY_MODEL)
        println(io2, "--------------------------------------------------------------")
        println(io2, "scan_n = ", Options.SCAN_N)
        println(io2, "irs = ", Options.IRS)
        println(io2, "n_basis = ", Options.N_BASIS)
        println(io2, "scan_method = ", Options.SCAN_METHOD)

        if (Options.WIDTH_IN_FLAG)
            println(io2, "ir,  width")
            for i = 1:Options.SCAN_N
                println(io2, Options.IRS+i-1, " ", width[i])
            end
        else
            println(io2, "ir,  width,  kymark")
            for i = 1:Options.SCAN_N
                println(io2, Options.IRS+i-1, " ", width[i], " ", kymark_out[i])
            end
        end

        println(io2, "--------------------------------------------------------------")
        println(io2, "factor_in_profile = ", Options.FACTOR_IN_PROFILE)
        if (Options.FACTOR_IN_PROFILE)
            for i = 1:Options.SCAN_N
                println(io2, Options.FACTOR[i])
            end
        else
            println(io2, Options.FACTOR[1])
        end

        println(io2, "width_in_flag = ", Options.WIDTH_IN_FLAG)
        if (!Options.WIDTH_IN_FLAG) println(io2, "width_min = ", Options.WIDTH_MIN, " width_max = ", Options.WIDTH_MAX) end
        close(io2)
    end
    
    # Initialize output arrays
    # kymark_out and width already defined above
    
    # Now continue on to radii-dependent part:
    if ((Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5))
        
        if (printout)
            io3 = open("out.TGLFEP", "a")
            println(io3, "**************************************************************")
            println(io3, "************** The critical EP density gradient **************")
            println(io3, "**************************************************************")
        end

        SFmin = fill(0.0, Options.SCAN_N)
        SFmin_out = fill(0.0, profile.NR)
        dndr_crit = fill(NaN, Options.SCAN_N)
        dndr_crit_out = fill(NaN, profile.NR)
        dpdr_crit = fill(NaN, Options.SCAN_N)
        dpdr_crit_out = fill(NaN, profile.NR)

        if (Options.THRESHOLD_FLAG == 0)
            for i = 1:n_ir
                SFmin[i] = arrTGLFEP[i].FACTOR_IN
            end
            if (printout)
                println(io3, "--------------------------------------------------------------")
                println(io3, "SFmin")
            end
        
        # Next is TGLFEP_complete_output(SFmin, SFmin_out, ir_min, ir_max, l_accept_profile)
        # This function's goal is to determine whether 

            SFmin, SFmin_out, ir_min, ir_max, l_accept_profile = tjlfep_complete_output(SFmin, Options, profile)
        
            if (printout)
                println(io3, SFmin, " SFmin after complete_output")
            end

            # We've received the altered profile (interpolated and accepted or not).
            # If the minimum radius is not the first one...
            if ((ir_min-Options.IRS+1) > 1)
                # Originally, this had no concern for accessing out-of-range values;
                # it does now:

                # Set any scans before this point's factor values to the factor_max_profile values respectively (?? Why so physically)
                # If you're running a normal amount of scans, this will pretty much never be done, right?
                if (ir_min-Options.IRS > Options.SCAN_N)
                    SFmin[1:Options.SCAN_N] = Options.FACTOR_MAX_PROFILE[1:Options.SCAN_N]
                else # original alone:
                    SFmin[1:ir_min-Options.IRS] = Options.FACTOR_MAX_PROFILE[1:ir_min-Options.IRS]
                end

                # If the starting radius is greater than 1, set the values before it in the interpolated profile to
                # the first value of the max_profile
                if (Options.IRS > 1) SFmin_out[1:Options.IRS-1] .= Options.FACTOR_MAX_PROFILE[1] end

                # This one doesn't look right...
                # This says that in the interpolated profile, you should set any values from the initial to the first point (minus 1)
                # to the same factor_max_profile. But this factor_max_profile is not an interpolation...
                # Is this being done as a default value? FACTOR_MAX_PROFILE if just scan_n of the same value for the case I've been testing
                # hence it's return of 1.0 (*) when rejected. The problem is that this doesn't make much sense for SFmin_out...

                # The problem is that this ignores defaults again. If factor_max_profile is accessed outside of scan_n, they should be set to 0...
                if (ir_min-Options.IRS > Options.SCAN_N)
                    # The +1 on SFmin_out exists because the original is keeping a spacing of 1 between the two...
                    SFmin_out[Options.IRS:Options.SCAN_N+1] = Options.FACTOR_MAX_PROFILE[1:Options.SCAN_N]

                    # This is a test input:
                    # SFmin_out[Options.SCAN_N+1:ir_min-Options.IRS] = 0.0
                else
                    SFmin_out[Options.IRS:ir_min-1] = Options.FACTOR_MAX_PROFILE[1:ir_min-Options.IRS]
                end
            end

            # Perform a similar maneuver for above the maximum:
            if ((ir_max-Options.IRS+1) < Options.SCAN_N)
                SFmin[ir_max-Options.IRS+2:Options.SCAN_N] = Options.FACTOR_MAX_PROFILE[ir_max-Options.IRS+2:Options.SCAN_N]
                if (Options.IRS+Options.SCAN_N-1 < profile.NR) SFmin_out[Options.IRS+Options.SCAN_N:profile.NR] .= Options.FACTOR_MAX_PROFILE[Options.SCAN_N] end
                SFmin_out[ir_max+1:Options.IRS+Options.SCAN_N-1] = Options.FACTOR_MAX_PROFILE[ir_max-Options.IRS+2:Options.SCAN_N]
            end

            if (printout)
                println(io3, SFmin, " SFmin after Max assign")
            end

            if (printout)
                for i = 1:Options.SCAN_N
                    if (l_accept_profile[i])
                        println(io3, SFmin[i])
                    else
                        println(io3, SFmin_out[i+Options.IRS-1], "   (*)")
                    end
                    println(io3, "--------------------------------------------------------------")
                    println(io3, l_accept_profile)
                    println(io3, ir_min, " : ", ir_max)
                end
            end

            # Calculate the density critical gradient at each of the scanned radii.
            if (Options.INPUT_PROFILE_METHOD == 2)
                dndr_crit .= 10000.0
                for i = 1:Options.SCAN_N
                    # If SFmin[i] is not the default non-rejected value, multiply the scalefactor by the density and the density gradient at that point 
                    # for the energetic ion. 
                    # If SFmin[i] is the default or >= 9k, check if it is one of the factor_max_profile ones, and if so, calculate it with that.
                    # otherwise, leave it at 10k.
                    if (SFmin[i] < 9000.0)
                        dndr_crit[i] = SFmin[i]*ni[Int(Options.IR_EXP[i])]*dlnnidr[Int(Options.IR_EXP[i])]
                    elseif ((i < ir_min-Options.IRS+1) || (i > ir_max-Options.IRS+1))
                        dndr_crit[i] = Options.FACTOR_MAX_PROFILE[i]*ni[Int(Options.IR_EXP[i])]*dlnnidr[Int(Options.IR_EXP[i])]
                    end
                end
                # Interpolate and accept are reject needed values of this profile:
                dndr_crit, dndr_crit_out, ir_dum_1, ir_dum_2, l_accept_profile = tjlfep_complete_output(dndr_crit, Options, profile)
                
                if (printout)
                    write_crit_grad("alpha_dndr_crit.input", "Density critical gradient (10^19/m^4)", dndr_crit_out)
                end
            end
            
            if (Options.INPUT_PROFILE_METHOD == 2)
                dpdr_crit .= 10000.0
                dpdr_EP = ni .* Ti .* (dlnnidr .+ dlntidr) .* 0.16022
                for i = 1:Options.SCAN_N
                    if (SFmin[i] < 9000.0)
                        if ((Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5))
                            case = Options.SCAN_METHOD
                            if (case == 1)
                                dpdr_scale = SFmin[i]
                            elseif (case == 2)
                                dpdr_scale = ((SFmin[i]*dlnnidr[Options.IR_EXP[i]]+dlntidr[Options.IR_EXP[i]]) /
                                (dlnnidr[Options.IR_EXP[i]]+dlntidr[Options.IR_EXP[i]]))
                            end
                            dpdr_crit[i] = dpdr_scale*dpdr_EP[Options.IR_EXP[i]]
                        end # 4 || 5
                    end # < 9000
                end # over scan_n
                dpdr_crit, dpdr_crit_out, ir_dum_1, ir_dum_2, l_accept_profile = tjlfep_complete_output(dpdr_crit, Options, profile)
                
                if (printout)
                    write_crit_grad("alpha_dpdr_crit.input", "Pressure critical gradient (10 kPa/m)", dpdr_crit_out)
                end
            end # end prof. method 2
            if (printout)
                println(io3, "--------------------------------------------------------------")
                println(io3, "The EP density threshold n_EP/n_e (%) for gamma_AE = 0")
                for i = 1:Options.SCAN_N
                    println(io3, SFmin[i]*profile.AS[Options.IRS+i-1, profile.IS]*100.0) #percent
                end
            end
            if (printout)
                println(io3, "--------------------------------------------------------------")
                println(io3, "The EP beta crit (%) = beta_e*(n_EP_th/n_e)*(T_EP/T_e)")
                for i = 1:Options.SCAN_N
                    if coalesce(profile.GEOMETRY_FLAG, 1) == 0
                        println(io3, SFmin[i]*profile.BETAE[Options.IRS+i-1]*100.0*profile.AS[Options.IRS+i-1, profile.IS]*profile.TAUS[Options.IRS+i-1, profile.IS]) #percent
                    else
                        println(io3, SFmin[i]*profile.BETAE[Options.IRS+i-1]*100.0*profile.AS[Options.IRS+i-1, profile.IS]*profile.TAUS[Options.IRS+i-1, profile.IS]*profile.KAPPA[Options.IRS+i-1]^2) #percent
                    end
                end
            end
            # there is a process_in == 4 addition I won't be doing quite yet.
        else # ThreshFlag != 0
            # Skipping for now as I want to test just threshold flag == 0 first
        end # ThreshFlag
        if (printout)
            close(io3)
        end
    end # process 4 || 5
    marginal_ql = [r[4] for r in results]
    return width, kymark_out, SFmin, dpdr_crit_out, dndr_crit_out, marginal_ql
end  # End of struct-based runTHD

"""
    runTHD_dd_radius(dd, rho, OptionsDict, scan_index; out_dir, use_gpu, inner, team, ql_flux_scan,
                     solver, refine_rounds, extend_mode, wide_kdesc, faithful_confirm)

SPMD per-radius entry point (DIII-D-style layout). Runs the kw-scan for the single
radius `scan_index` and serializes the `pmap`-element result to `out_dir/task_<scan_index>.jls`
so the merge phase (`runTHD(dd; precomputed_dir=out_dir)`) can assemble the cross-radius
critical gradients exactly as the in-process path would.

Invoked once per Slurm task (`srun -n SCAN_N`). The MPS `team` (worker pids sharing this
task's pinned GPU) is spawned by the task driver at top level and passed in explicitly --
matching the verified `run_gacode_scan_task` path (where `addprocs`/`@everywhere` run from
the script's top level, not from inside a function or a `pmap` worker). `team=nothing` runs
the single-process threaded baseline.
"""
function TJLFEP.runTHD_dd_radius(dd::IMAS.dd, rho::AbstractVector{Float64},
        OptionsDict::Dict{String, Any}, scan_index::Int;
        out_dir::AbstractString=".", use_gpu::Union{Bool,Symbol}=:auto,
        inner::Symbol=:mps_team,
        team::Union{Nothing,AbstractVector{<:Integer}}=nothing,
        ql_flux_scan::Bool=false, printout::Bool=false, solver::Symbol=:grid,
        refine_rounds::Int=1, k_max::Int=TJLFEP._k_max_env(), extend_mode::Union{Nothing,Symbol}=nothing,
        wide_kdesc::Union{Nothing,Int}=nothing, faithful_confirm::Union{Nothing,Bool}=nothing)
    use_gpu = TJLFEP._resolve_use_gpu(use_gpu)   # :auto -> GPU when CUDA is functional
    Options, profile, _, _ = preprocess_imas_inputs(dd, rho, OptionsDict; verbose=printout)
    1 <= scan_index <= Options.SCAN_N ||
        error("runTHD_dd_radius: scan_index=$scan_index out of range 1:$(Options.SCAN_N)")
    mkpath(out_dir)

    @info "runTHD_dd_radius" scan_index ir=Options.IR_EXP[scan_index] inner team=(team === nothing ? 0 : length(team)) host=gethostname()

    output = _dd_radius_output(Options, profile, scan_index;
        use_gpu=use_gpu, inner=inner, team=team, ql_flux_scan=ql_flux_scan, printout=printout,
        solver=solver, refine_rounds=refine_rounds, k_max=k_max, extend_mode=extend_mode,
        wide_kdesc=wide_kdesc, faithful_confirm=faithful_confirm)

    task_file = joinpath(out_dir, "task_$(scan_index).jls")
    open(task_file, "w") do io
        Serialization.serialize(io, output)
    end
    @info "runTHD_dd_radius wrote $task_file"
    return task_file
end

end # module TJLFEPIMASExt
