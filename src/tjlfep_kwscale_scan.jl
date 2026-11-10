# Match Fortran out.scalefactor layout (format 20 / 60 in TGLFEP_kwscale_scan.f90).
function _scalefactor_section_header(kyhat::T, ky_write::T, efwid::T, fmark::T) where {T<:Real}
    return @sprintf(
        "--------------- ky*rho_EP= %7.4f  (ky*rho_s=%7.4f)   width=%7.4f   scalefactor=%10.4f -------------------",
        kyhat, ky_write, efwid, fmark,
    )
end

# `out.wavefunction..._sf` filename tag: FACTOR_IN as the 6-digit "NNN.NNN" string the Fortran
# TGLFEP_kwscale_scan wrote. Shared by the grid inner loop and the AD-winner replay so both engines
# emit identically-named wavefunction files.
function _sf_factor_tag(factor::Real)
    return string(Char(mod(floor(Int, factor/100.0), 10) + UInt32('0'))) *
           string(Char(mod(floor(Int, factor/10.0), 10) + UInt32('0')))  *
           string(Char(mod(floor(Int, factor), 10) + UInt32('0'))) *
           "." *
           string(Char(mod(floor(Int, 10*factor), 10) + UInt32('0'))) *
           string(Char(mod(floor(Int, 100*factor), 10) + UInt32('0'))) *
           string(Char(mod(floor(Int, 1000*factor), 10) + UInt32('0')))
end

function _scalefactor_factor_line(fac::T, g::AbstractVector{T}, f::AbstractVector{T}, keep_label::AbstractVector{String}) where {T<:Real}
    parts = [@sprintf("%11.4f", fac)]
    for n in eachindex(keep_label)
        push!(parts, @sprintf(" %14.5E", g[n]), @sprintf(" %14.5E", f[n]), " ", keep_label[n])
    end
    return join(parts)
end

# Process-wide stock of TJLF.LSWorkspace buffers for the threaded CPU combo scan. A worker
# task borrows one for a round of combos and hands it back, so the number alive tracks the
# number of combos actually running at once (about nthreads()), also when several radii are
# scanned concurrently in one process. The stock is dropped when the last active scan ends,
# so a long-lived session does not keep the memory.
const _WS_LOCK = ReentrantLock()
const _WS_FREE = TJLF.LSWorkspace{Float64}[]
const _WS_SCANS = Ref(0)

# false = do not reuse (GPU solves, non-Float64 element types, or TGLFEP_JULIA_WORKSPACE=0)
_use_combo_workspaces(::Type{T}, use_gpu::Bool) where {T<:Real} =
    !use_gpu && T === Float64 && get(ENV, "TGLFEP_JULIA_WORKSPACE", "1") != "0"

_ws_scan_begin!() = lock(() -> (_WS_SCANS[] += 1; nothing), _WS_LOCK)
function _ws_scan_end!()
    lock(_WS_LOCK) do
        _WS_SCANS[] -= 1
        _WS_SCANS[] <= 0 && (_WS_SCANS[] = 0; empty!(_WS_FREE))
        nothing
    end
end
_ws_borrow!() = lock(() -> isempty(_WS_FREE) ? TJLF.LSWorkspace{Float64}() : pop!(_WS_FREE), _WS_LOCK)
_ws_return!(ws::TJLF.LSWorkspace{Float64}) = lock(() -> (push!(_WS_FREE, ws); nothing), _WS_LOCK)

# One ky/width/factor combo of the inner scan. Pure w.r.t. shared state: it deepcopies
# inputsEP, runs TJLFEP_ky, and returns everything the caller needs to scatter back into
# the shared result arrays. Kept allocation-light (only NMODES-length vectors + a few
# scalars) so it is cheap to serialize when distributed across MPS team workers.
function _kw_combo(i::Int, k::Int, k_max::Int, nfactor::Int, nefwid::Int,
                   factor::AbstractVector{T}, efwid::AbstractVector{T}, kyhat::AbstractVector{T},
                   inputsEP::Options{T}, inputsPR::profile{T},
                   ikyhat_write::Int, iefwid_write::Int, ifactor_write::Int,
                   printout::Bool; use_gpu::Bool = false, mode_in::Int = 2,
                   workspace::Union{Nothing,TJLF.LSWorkspace} = nothing) where {T<:Real}
    local_inputsEP = deepcopy(inputsEP)  # thread/process-local; avoids races on FACTOR_IN/KYHAT_IN/WIDTH_IN/LKEEP/etc.

    l_wavefunction_out = 0

    # The following 3 statements define each combination of ikyhat, iefwid, and ifactor.
    ikyhat = Int(floor((i-1)/(nefwid*nfactor))+1)
    iefwid = Int(floor(1.0*mod(i-1, nefwid*nfactor)/nfactor)+1)
    ifactor = mod(i-1, nfactor)+1

    local_inputsEP.FACTOR_IN = factor[ifactor]
    local_inputsEP.KYHAT_IN = kyhat[ikyhat]
    local_inputsEP.WIDTH_IN = efwid[iefwid]
    debug_dump_kw_combo(local_inputsEP, i)

    str_sf = _sf_factor_tag(local_inputsEP.FACTOR_IN)

    str_wf_file = "out.wavefunction"*coalesce(local_inputsEP.SUFFIX, "")*"_sf"*str_sf

    if ((local_inputsEP.WRITE_WAVEFUNCTION == 1) &&
        (ikyhat == ikyhat_write) &&
        (iefwid == iefwid_write) &&
        (ifactor == ifactor_write) &&
        (k == k_max) && printout)
        l_wavefunction_out = 1
    end

    # eigen_cache=nothing: the sequential-seeding optimization is incompatible with the
    # parallel (threaded or distributed) execution here, so each combo starts cold.
    # mode_in threads the drive selector to TJLF_map (2 = EP drive only for PROCESS_IN=5;
    # 4 = thermal+EP gradients on + FILTER=2 for the PROCESS_IN=6 ITG/TEM variant).
    gamma_out, freq_out, inputTJLF, _, wavefunction_buffer, dep_out, ep_flux_out =
        TJLFEP_ky(local_inputsEP, inputsPR, str_wf_file, l_wavefunction_out; eigen_cache=nothing,
                  use_gpu=use_gpu, mode_in_override=mode_in, workspace=workspace)

    NM = local_inputsEP.NMODES
    return (
        ikyhat = ikyhat, iefwid = iefwid, ifactor = ifactor,
        gamma_out = gamma_out[1:NM], freq_out = freq_out[1:NM],
        dep_out = dep_out[1:NM], ep_flux_out = ep_flux_out[1:NM],
        LKEEP      = local_inputsEP.LKEEP[1:NM],
        LTEARING   = local_inputsEP.LTEARING[1:NM],
        L_TH_PINCH = local_inputsEP.L_TH_PINCH[1:NM],
        L_I_PINCH  = local_inputsEP.L_I_PINCH[1:NM],
        L_E_PINCH  = local_inputsEP.L_E_PINCH[1:NM],
        L_EP_PINCH = local_inputsEP.L_EP_PINCH[1:NM],
        L_THETA_SQ = local_inputsEP.L_THETA_SQ[1:NM],
        L_QL_RATIO = local_inputsEP.L_QL_RATIO[1:NM],
        str_wf_file = str_wf_file,
        wavefunction_buffer = wavefunction_buffer,
        # Species params (constant across combos) needed once for the out.scalefactor header.
        zs_ep   = inputTJLF.ZS[inputsEP.IS_EP+1],
        mass_ep = inputTJLF.MASS[inputsEP.IS_EP+1],
        taus_ep = inputTJLF.TAUS[inputsEP.IS_EP+1],
    )
end

# ── Opt-in batched shift-invert eigensolver for the grid (inner=:batched_si) ──────────────────
# Motivation: the grid's cost at large N_BASIS is dominated by the dense per-combo eigensolve
# (serial Xgeev on GPU / geev on CPU). TJLF.si_eigvals_batch solves MANY pencils in one batched
# GPU sweep (~7x faster per pencil than Xgeev at n=720; the margin grows with n). We collect every
# combo's assembled (A,B) pencil, batch-solve for the dominant unstable modes, then replay the
# combos with those eigenvalues. Deadlock-free (no cross-thread barrier): phase 1 runs each combo
# to completion with the eigensolve returning a benign stable placeholder while recording the
# pencil; phase 2 batch-solves; phase 3 re-runs each combo with the batched eigenvalues.
#
# The SI eigenvalues are approximate (dominant window only), so this is intended for the onset
# scan (IFLUX=false). Optionally the winning configuration is dense-confirmed for an exact sfmin.

const _SI_PLACEHOLDER = ComplexF64[-1.0 + 0.0im]  # phase-1: harmless "stable" spectrum

# TJLF eigensolve hook (installed only during the batched-si round). Routes by task-local state
# set per combo in the threaded loop body, so concurrent combos stay isolated.
function _si_eigsolve_hook(A::Matrix{ComplexF64}, B::Matrix{ComplexF64})
    st = get(task_local_storage(), :tjlfep_si, nothing)
    st === nothing && error("batched-si hook fired without task-local state")
    if st.mode === :collect
        push!(st.store[st.id], (copy(A), copy(B)))
        return _SI_PLACEHOLDER
    else # :replay
        c = st.counter[] += 1
        evs = st.evals[st.id]
        return c <= length(evs) ? evs[c] : _SI_PLACEHOLDER
    end
end

# Run one k-round of combos through the batched shift-invert eigensolver. Returns results::Vector
# in the same shape as the threaded path (one _kw_combo NamedTuple per combo).
function _run_combos_batched_si(nkwf::Int, k::Int, k_max::Int, nfactor::Int, nefwid::Int,
                                factor, efwid, kyhat, inputsEP, inputsPR,
                                ikyhat_write::Int, iefwid_write::Int, ifactor_write::Int,
                                printout::Bool; use_gpu::Bool, mode_in::Int,
                                si_method::Symbol = :si, si_cfg::TJLF.SIConfig = TJLF.SIConfig(),
                                contour_cfg::TJLF.ContourConfig = TJLF.ContourConfig(),
                                csi_cfg::TJLF.CertifiedSIConfig = TJLF.CertifiedSIConfig())
    store = [Vector{Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}() for _ in 1:nkwf]
    results = Vector{Any}(undef, nkwf)

    # Phase 1 — assemble: collect each combo's pencil(s); eigensolve returns the stable placeholder.
    TJLF._EIGSOLVE_HOOK[] = _si_eigsolve_hook
    try
        TJLF.with_blas_threads(1) do
            Threads.@threads for i = 1:nkwf
                task_local_storage(:tjlfep_si, (mode = :collect, id = i, store = store))
                _kw_combo(i, k, k_max, nfactor, nefwid, factor, efwid, kyhat,
                          inputsEP, inputsPR, ikyhat_write, iefwid_write, ifactor_write,
                          printout; use_gpu = false, mode_in = mode_in)
            end
        end
    finally
        TJLF._EIGSOLVE_HOOK[] = nothing
    end

    # Phase 2 — batched solve: flatten all pencils, one GPU sweep, scatter eigenvalues back.
    flatA = Matrix{ComplexF64}[]; flatB = Matrix{ComplexF64}[]; owner = Tuple{Int,Int}[]
    for i in 1:nkwf, (j, (A, B)) in enumerate(store[i])
        push!(flatA, A); push!(flatB, B); push!(owner, (i, j))
    end
    evals = [Vector{Vector{ComplexF64}}(undef, length(store[i])) for i in 1:nkwf]
    if !isempty(flatA)
        # Batched solvers require a uniform matrix size; group by n and solve each group.
        for n in unique(size.(flatA, 1))
            idx = findall(p -> size(flatA[p], 1) == n, eachindex(flatA))
            got = if si_method === :certified
                # Coverage-certified adaptive SI: residual-certified eigenvalues with a
                # per-pencil geometric coverage certificate; uncovered pencils fall back to
                # the exact dense solve (per pencil, not per radius).
                first(TJLF.certified_si_eigvals_batch(flatA[idx], flatB[idx]; cfg = csi_cfg,
                                                      use_gpu = use_gpu, dense_fallback = :cpu))
            elseif si_method === :contour
                # Contour (SS-RR) solver: kept for reference; on real TJLF spectra the axis-
                # hugging mode crowd saturates it and ~everything falls back dense.
                first(TJLF.contour_eigvals_batch(flatA[idx], flatB[idx]; cfg = contour_cfg,
                                                 use_gpu = use_gpu, dense_fallback = :cpu))
            else
                TJLF.si_eigvals_batch(flatA[idx], flatB[idx]; cfg = si_cfg, use_gpu = use_gpu)
            end
            for (g, p) in zip(got, idx)
                (i, j) = owner[p]; evals[i][j] = g
            end
        end
    end

    # Phase 3 — replay: re-run each combo with the batched eigenvalues.
    TJLF._EIGSOLVE_HOOK[] = _si_eigsolve_hook
    try
        TJLF.with_blas_threads(1) do
            Threads.@threads for i = 1:nkwf
                task_local_storage(:tjlfep_si,
                    (mode = :replay, id = i, evals = evals, counter = Ref(0)))
                results[i] = _kw_combo(i, k, k_max, nfactor, nefwid, factor, efwid, kyhat,
                                       inputsEP, inputsPR, ikyhat_write, iefwid_write, ifactor_write,
                                       printout; use_gpu = false, mode_in = mode_in)
            end
        end
    finally
        TJLF._EIGSOLVE_HOOK[] = nothing
    end
    return results
end

# Scatter one _kw_combo result into the shared per-combo arrays. Identical for the
# threaded and distributed paths, so the reduction below the loop sees the same data.
function _scatter_combo!(r, growthrate, frequency, dep_scan, lkeep_i, ltearing_i, l_th_pinch_i,
                         l_i_pinch_i, l_e_pinch_i, l_EP_pinch_i, l_theta_sq_i, l_QL_ratio_i)
    ik = r.ikyhat; iw = r.iefwid; ifa = r.ifactor
    for n in eachindex(r.gamma_out)
        growthrate[ik, iw, ifa, n]   = r.gamma_out[n]
        frequency[ik, iw, ifa, n]    = r.freq_out[n]
        dep_scan[ik, iw, ifa, n]     = r.dep_out[n]
        lkeep_i[ik, iw, ifa, n]      = r.LKEEP[n]
        ltearing_i[ik, iw, ifa, n]   = r.LTEARING[n]
        l_th_pinch_i[ik, iw, ifa, n] = r.L_TH_PINCH[n]
        l_i_pinch_i[ik, iw, ifa, n]  = r.L_I_PINCH[n]
        l_e_pinch_i[ik, iw, ifa, n]  = r.L_E_PINCH[n]
        l_EP_pinch_i[ik, iw, ifa, n] = r.L_EP_PINCH[n]
        l_theta_sq_i[ik, iw, ifa, n] = r.L_THETA_SQ[n]
        l_QL_ratio_i[ik, iw, ifa, n] = r.L_QL_RATIO[n]
    end
    return nothing
end

# Distribute f(i) for i in 1:n across the MPS "team" worker processes (their pids),
# returning results in index order. Each worker is a separate process with its own CUDA
# context, so the GPU eigensolves overlap on a shared device via MPS Hyper-Q (safe — the
# Xgeev corruption only affects concurrency *within* one context).
#
# Dispatch is CHUNKED, not per-combo: the n combos are split into one contiguous block per
# worker, and each block is sent in a SINGLE remotecall that runs a tight local `map(f, range)`
# loop on the worker. This is the key to realizing the GPU concurrency — per-combo pmap pays a
# coordinator round-trip for every one of the ~1024 combos, which starves the workers' GPU
# contexts (only ~1.35x). One round-trip per worker per round keeps all contexts saturated
# (~3x, matching the standalone Xgeev benchmark). The (inputsEP/inputsPR) closure is serialized
# once per worker per round; factor/efwid/kyhat change each round, so caching across rounds
# would not help anyway.
# Run f over the index range on THIS worker, multi-threaded. CPU matrix assembly (the bulk of
# each combo, ~0.75s vs ~0.32s GPU) parallelizes across the worker's threads; the per-device
# lock in TJLFCUDAExt serializes only this worker's GPU solves (which overlap ACROSS workers
# via MPS). BLAS pinned to 1 thread so the dense LAPACK inside each combo doesn't oversubscribe.
function _team_chunk_map(f, rng)
    res = Vector{Any}(undef, length(rng))
    TJLF.with_blas_threads(1) do
        Threads.@threads for j in eachindex(rng)
            res[j] = f(rng[j])
        end
    end
    return res
end

function _inner_team_map(f, team::AbstractVector{<:Integer}, n::Int)
    ws = collect(team)
    nw = length(ws)
    # contiguous, near-equal blocks of 1:n (block j is empty if n < nw)
    chunks = [(div((j - 1) * n, nw) + 1):(div(j * n, nw)) for j in 1:nw]
    futs = Vector{Distributed.Future}(undef, nw)
    for j in 1:nw
        rng = chunks[j]
        futs[j] = Distributed.remotecall(_team_chunk_map, ws[j], f, rng)
    end
    results = Vector{Any}(undef, n)
    for j in 1:nw
        block = fetch(futs[j])
        if block isa Exception
            throw(block)
        end
        for (m, idx) in enumerate(chunks[j])
            results[idx] = block[m]
        end
    end
    return results
end

function kwscale_scan(inputsEP::Options{T}, inputsPR::profile{T}, printout::Bool = true;
                      use_gpu::Bool = false, inner::Symbol = :threads,
                      team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                      ql_flux_scan::Bool = false, mode_in::Int = 2,
                      nfactor::Int = 8, nefwid::Int = 8, nkyhat::Int = 4, k_max::Int = 4,
                      si_dense_fallback::Bool = true, si_fallback_ratio::Real = 1.5,
                      si_method::Symbol = :si,
                      contour_cfg::TJLF.ContourConfig = TJLF.ContourConfig(),
                      csi_cfg::TJLF.CertifiedSIConfig = TJLF.CertifiedSIConfig()) where {T<:Real}
    # mode_in is the TGLF-EP drive selector passed to TJLF_map: 2 (EP drive only, the
    # PROCESS_IN=5 default) or 4 (thermal+EP gradients on + ITG/TEM FILTER=2, PROCESS_IN=6).
    # These are for testing purposes:
    #baseDirectory = "/Users/benagnew/TJLF.jl/outputs/tglfep_tests/input.MTGLF"
    #inputsPR = readMTGLF(baseDirectory)

    # Robust hybrid guard. The batched shift-invert eigensolver reproduces the dense grid EXACTLY
    # where the AE drive is strong (marginal factor <~ nominal), but at WEAKLY-DRIVEN radii the
    # near-marginal mode's growth rate is tiny and SI's worst-case absolute error swamps it — the
    # (ky,width) mark stays correct but the factor-axis crossing drifts. Detect that regime (the
    # hybrid returns a marginal well above the nominal drive) and redo the radius with the dense
    # eigensolver so the marginal stays exact. Clearly-unstable radii (the common, expensive case)
    # keep the SI speedup; only the stable-at-operating-point radii pay for a dense redo.
    if inner === :batched_si && si_dense_fallback
        nominal = inputsEP.FACTOR_IN
        r = kwscale_scan(deepcopy(inputsEP), inputsPR, printout;
                         use_gpu=use_gpu, inner=:batched_si, team=team, ql_flux_scan=ql_flux_scan,
                         mode_in=mode_in, nfactor=nfactor, nefwid=nefwid, nkyhat=nkyhat, k_max=k_max,
                         si_dense_fallback=false, si_method=si_method, contour_cfg=contour_cfg,
                         csi_cfg=csi_cfg)
        (isfinite(r[2].FACTOR_IN) && r[2].FACTOR_IN <= si_fallback_ratio * nominal) && return r
        return kwscale_scan(inputsEP, inputsPR, printout;
                            use_gpu=use_gpu, inner=:threads, team=team, ql_flux_scan=ql_flux_scan,
                            mode_in=mode_in, nfactor=nfactor, nefwid=nefwid, nkyhat=nkyhat, k_max=k_max)
    end

    # Grid resolution / refinement depth (defaults reproduce the canonical scan).
    # Exposed as kwargs so the factor-grid convergence of the marginal sfmin can be
    # studied (e.g. nfactor ∈ {8,16,32,64}, k_max ∈ {4,6,8}).
    nkwf = nfactor*nefwid*nkyhat
    l_write_out = true

    kyhat_min = 0.0
    kyhat_max = 1.0

    # ikymin = 0
    # ikymax =0
    # iefwmin =0
    # iefwmax =0

    growthrate = zeros(T, nkyhat, nefwid, nfactor, inputsEP.NMODES)
    growthrate_out = zeros(T, nkyhat, nefwid, nfactor, inputsEP.NMODES)
    dep_scan = zeros(T, nkyhat, nefwid, nfactor, inputsEP.NMODES)

    frequency = zeros(T, nkyhat, nefwid, nfactor, inputsEP.NMODES)
    frequency_out = zeros(T, nkyhat, nefwid, nfactor, inputsEP.NMODES)
    
    lkeep_i = fill(true, (nkyhat, nefwid, nfactor, inputsEP.NMODES))
    ltearing_i = fill(false, (nkyhat, nefwid, nfactor, inputsEP.NMODES))
    l_th_pinch_i = fill(false, (nkyhat, nefwid, nfactor, inputsEP.NMODES))
    l_i_pinch_i = fill(false, (nkyhat, nefwid, nfactor, inputsEP.NMODES))
    l_e_pinch_i = fill(false, (nkyhat, nefwid, nfactor, inputsEP.NMODES))
    l_EP_pinch_i = fill(false, (nkyhat, nefwid, nfactor, inputsEP.NMODES))
    #l_max_outer_panel_i = fill(false, (nkyhat, nefwid, nfactor, inputsEP.NMODES))
    l_QL_ratio_i = fill(false, (nkyhat, nefwid, nfactor, inputsEP.NMODES))
    l_theta_sq_i = fill(false, (nkyhat, nefwid, nfactor, inputsEP.NMODES))
    l_theta_sq_i_out = fill(false, (nkyhat, nefwid, nfactor, inputsEP.NMODES))
    #this variable was not in the code, but used later per the commit
    l_QL_ratio_i_out = fill(false, (nkyhat, nefwid, nfactor, inputsEP.NMODES))
    l_QL_ratio_i_ = fill(false, (nkyhat, nefwid, nfactor, inputsEP.NMODES))

    f0 = zero(T)
    f1 = inputsEP.FACTOR_IN # 1 for first round
    w0 = inputsEP.WIDTH_MIN
    w1 = inputsEP.WIDTH_MAX
    kyhat_min = zero(T)
    kyhat_max = one(T)
    kyhat0 = kyhat_min
    kyhat1 = kyhat_max
    factor = fill(T(NaN), nfactor)
    efwid = fill(T(NaN), nefwid)
    kyhat = fill(T(NaN), nkyhat)

    ikyhat_write = floor(Int, nkyhat/2) # 2
    iefwid_write = floor(Int, nefwid/2) # 5
    ifactor_write = nfactor # 10
    f_guess_mark = T(1.0E20)
    lkeep_ref = fill(false, (nkyhat, nefwid))
    # Species params for the out.scalefactor header (set from any combo each k; constant
    # across combos). Replaces the old per-scan placeholder InputTJLF used only for these.
    zs_ep = T(NaN); mass_ep = T(NaN); taus_ep = T(NaN)
    imark_min = 0
    imark = fill(nfactor + 1, (nkyhat, nefwid))  # declared here so its final-k value survives the loop (used by extract_marginal_ql below)
    f_guess = fill(T(NaN), (nkyhat, nefwid))
    ikyhat_mark::Int64 = 0
    iefwid_mark::Int64 = 0
    fmark = T(1.0E20)
    scalefactor_buffer = l_write_out && printout ? String[] : nothing
    wavebuffer_all = []
    # Opt-in probe: wall time of the parallel combo map vs the serial reduction, per k-round.
    _probe = _probe_on()
    _probe && _probe_reset!()
    _map_wall = 0.0
    _ser_wall = 0.0
    _t_scan = _probe ? time_ns() : UInt64(0)
    # Reuse TJLF solver workspaces in the threaded CPU scan (see _WS_FREE). GPU and
    # team/batched inners keep allocating per call.
    use_ws = _use_combo_workspaces(T, use_gpu)
    use_ws && _ws_scan_begin!()
    for k = 1:k_max
        # k doesn't use Threads!
        fill!(factor, T(NaN))
        fill!(efwid, T(NaN))
        fill!(kyhat, T(NaN))
        for i = 1:nfactor
            factor[i] = ((f1-f0)/nfactor)*i+f0
            # k = 1: FACTOR_IN = 1.0
            # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        end
        for i = 1:nefwid
            efwid[i] = ((w1-w0)/(nefwid-1))*(i-1)+w0
        end
        
        for i = 1:nkyhat
            if kyhat0 == 0.0
                kyhat[i]= ((kyhat1-kyhat0)/nkyhat)*i + kyhat0
            else 
                kyhat[i]= ((kyhat1-kyhat0)/(nkyhat-1))*(i-1)+kyhat0
            end
            #kyhat[i] = ((kyhat1-kyhat0)/nkyhat)*i+kyhat0
        end
        wavebuffer_all = []

        # Compute all nkwf combos for this k round. Each combo is independent (cold
        # eigen_cache), so the scan is order-free; only the dispatch differs:
        #   :mps_team -> distribute combos across the team's worker processes (each its
        #                own CUDA context) so their GPU eigensolves overlap on the shared
        #                device via MPS. Falls back to threads if no team was provided.
        #   :threads  -> the original in-process Threads.@threads scan. BLAS=1 because
        #                each task does small dense LAPACK solves that would otherwise
        #                oversubscribe cores (scoped; short-circuits if already set).
        local results::Vector{Any}
        _t_map = _probe ? time_ns() : UInt64(0)
        # Hybrid batched-SI: the approximate SI growth rate can carry occasional large ABSOLUTE
        # errors (max |dgamma| ~0.04) that perturb the gamma=gamma* crossing at weakly-driven radii,
        # where the driving mode's gamma is itself ~0.05. Both the WIDE localization (first round)
        # and the final PIN (last round) are done with the dense eigensolver so the marginal box is
        # correctly bracketed and exactly resolved even for far-out marginals (e.g. edge radii whose
        # marginal factor is several times the nominal drive). The fast batched SI accelerates only
        # the INTERMEDIATE refinement rounds, where the box is already localized. For k_max <= 2
        # this reduces to the all-dense grid (no intermediate round to accelerate).
        if inner === :batched_si && 1 < k < k_max
            results = _run_combos_batched_si(nkwf, k, k_max, nfactor, nefwid, factor, efwid, kyhat,
                                             inputsEP, inputsPR, ikyhat_write, iefwid_write,
                                             ifactor_write, printout;
                                             use_gpu=use_gpu, mode_in=mode_in, si_method=si_method,
                                             si_cfg=TJLF.SIConfig(), contour_cfg=contour_cfg,
                                             csi_cfg=csi_cfg)
        elseif inner === :mps_team && team !== nothing && !isempty(team)
            results = _inner_team_map(team, nkwf) do i
                _kw_combo(i, k, k_max, nfactor, nefwid, factor, efwid, kyhat,
                          inputsEP, inputsPR, ikyhat_write, iefwid_write, ifactor_write,
                          printout; use_gpu=use_gpu, mode_in=mode_in)
            end
        else
            results = Vector{Any}(undef, nkwf)
            TJLF.with_blas_threads(1) do
                # nthreads() worker tasks pull combo indices from a shared counter (same
                # dynamic load balancing as a plain threaded loop). Each worker keeps ONE
                # solver workspace for the whole round, so the big eigenmatrix buffers are
                # reused across combos instead of being reallocated for each one.
                next = Threads.Atomic{Int}(1)
                Threads.@threads for c = 1:min(Threads.nthreads(), nkwf)
                    ws = use_ws ? _ws_borrow!() : nothing
                    try
                        while true
                            i = Threads.atomic_add!(next, 1)
                            i > nkwf && break
                            results[i] = _kw_combo(i, k, k_max, nfactor, nefwid, factor, efwid, kyhat,
                                                   inputsEP, inputsPR, ikyhat_write, iefwid_write, ifactor_write,
                                                   printout; use_gpu=use_gpu, mode_in=mode_in, workspace=ws)
                        end
                    finally
                        ws === nothing || _ws_return!(ws)
                    end
                end
            end
        end
        _probe && (_map_wall += (time_ns() - _t_map) / 1e9)

        # Serial scatter + reduction (deterministic, order-independent across combos).
        _t_ser = _probe ? time_ns() : UInt64(0)
        for i = 1:nkwf
            r = results[i]
            _scatter_combo!(r, growthrate, frequency, dep_scan, lkeep_i, ltearing_i, l_th_pinch_i,
                            l_i_pinch_i, l_e_pinch_i, l_EP_pinch_i, l_theta_sq_i, l_QL_ratio_i)
            if r.wavefunction_buffer !== nothing
                push!(wavebuffer_all, (r.str_wf_file, r.wavefunction_buffer))
            end
            zs_ep = r.zs_ep; mass_ep = r.mass_ep; taus_ep = r.taus_ep
        end

        # This loop creates a 5x10 matrix full of 11. It then runs
        # through all dimensions of lkeep_i, which is a reference matrix
        # telling you where each 
        imark = fill(nfactor+1, (nkyhat, nefwid))

        # The following loop sets the matrix imark finds the first mode for a specific ikyhat, iefwid, ifactor combination
        # that is set to be kept in lkeep_i. This mode is thus representing that specific ifactor value for the run
        # and imark is set to ifactor for that ikyhat and iefwid. The same is also done for the factor values as once the first non-zeros
        # growthrate is found along the factor spectrum, we return back to the next width run. Essentially, we are trying to find
        # all of the first non-zero growthrates at each kyhat, width combo; these are then allotted to the imark matrix which
        # describes whether a kyhat, width combo even has a non-zero growthrate, and if it does, where in the scan did it find that.

        for ikyhat = 1:nkyhat
            for iefwid = 1:nefwid
                for ifactor = 1:nfactor
                    for n = 1:inputsEP.NMODES # For a specified mode, 
                        if (lkeep_i[ikyhat, iefwid, ifactor, n])
                            imark[ikyhat, iefwid] = ifactor
                            # these could range from 1 to 10.
                            break
                        end
                    end # n
                    if (imark[ikyhat,iefwid] <= nfactor) break end
                end # ifactor
            end # iefwid
        end #ikyhat
        
        imark_min = nfactor + 1
        for ikyhat = 1:nkyhat
            for iefwid = 1:nefwid
                imark_min = min(imark[ikyhat, iefwid], imark_min)
            end
        end

        
        fmark = T(1.0E20)
        gmark = zero(T)
        f_guess_mark = T(1.0E20)
        gamma_mark_i_1 = fill(T(NaN), (nkyhat, nefwid))
        gamma_mark_i_2 = fill(T(NaN), (nkyhat, nefwid))
        f_mark_i = fill(T(NaN), (nkyhat, nefwid))
        
        # if there are any unstable modes:
        if (imark_min <= nfactor)

            # default the marked values as the last ones
            ikyhat_mark = nkyhat
            iefwid_mark = nefwid
            # Reset write indices to midpoint defaults each k (matches Fortran)
            ikyhat_write = floor(Int, nkyhat/2)
            iefwid_write = floor(Int, nefwid/2)

            # loop all combos of kyhat, width in imark:
            for ikyhat = 1:nkyhat
                for iefwid = 1:nefwid
                    lkeep_ref[ikyhat, iefwid] = false  # reset each k iteration (matches Fortran)
                    imark_ref = nfactor-1 # ? What is the point ?

                    # if imark is not the final one of the factor spectrum, set the reference to imark
                    # and set it as kept; otherwise, if it's the last one, set the reference to imark-1
                    # (9) and set as kept, and if it's a default value of 11, set reference to 10.
                    if (imark[ikyhat, iefwid] < nfactor) # < 10
                        imark_ref = imark[ikyhat, iefwid]
                        lkeep_ref[ikyhat, iefwid] = true
                    else # 10 or 11
                        if (imark[ikyhat, iefwid] == nfactor) # == 10
                            lkeep_ref[ikyhat,iefwid] = true
                        end
                    end
                    # Now we set the f_g1 and f_g2 values, which allows us to guess the
                    # value of f_guess based on this equation. This equation is not clear yet to me.
                    f_g1 = factor[imark_ref]
                    f_g2 = factor[imark_ref+1]

                    # This is the portion that is NOT consistent with the Fortran and is
                    # causing problems due to default values:
                    gamma_g1 = zero(T) #-2.0
                    gamma_g2 = T(100.0) #-1.0
                    
                    # For all modes, set gamma_g1 to the maximum growthrate of kept modes at this point of ikyhat, iefwid, and imark_ref
                    # This also sets gamma_g2 to the maximum neighboring factor value of growthrate (imark_ref+1).
                    for n = 1:inputsEP.NMODES
                        if (lkeep_i[ikyhat, iefwid, imark_ref, n])
                            gamma_g1 = max((growthrate[ikyhat, iefwid, imark_ref, n]), gamma_g1)
                        end
                        # if (imark_ref+1 < nfactor+1)
                        if (lkeep_i[ikyhat, iefwid, imark_ref+1, n])
                            gamma_g2 = max((growthrate[ikyhat, iefwid, imark_ref+1, n]), gamma_g2)
                        end
                        # end
                    end
                    f_guess[ikyhat, iefwid] = f_g1 + (inputsEP.GAMMA_THRESH-gamma_g1)*(f_g2-f_g1)/(gamma_g2-gamma_g1)
                    gamma_mark_i_1[ikyhat, iefwid] = gamma_g1
                    gamma_mark_i_2[ikyhat, iefwid] = gamma_g2
                    if imark[ikyhat, iefwid] < nfactor 
                        f_mark_i[ikyhat, iefwid] = f_g1
                    else
                        f_mark_i[ikyhat,iefwid] = f_g2
                    end
                end # iefwid
            end # ikyhat

            # Now we scan across imark again, looking for modes that are:
            # kept; have a max growthrate along the modes that is less than 95% of the neighboring maximum; have a factor value that is 
            # less than or equal to the current maximum... (starts at 1e20)
            # then if these are satisfied, find modes that:
            # have a growthrate that is larger than the current maximum (starts at 0) or
            # have a factor that is smaller than the max
            ikymin =nkyhat
            ikymax = 1
            iefwmin = nefwid
            iefwmax = 1
            for ikyhat = 1:nkyhat
                for iefwid = 1:nefwid
                    if (lkeep_ref[ikyhat, iefwid] && f_mark_i[ikyhat, iefwid] <= fmark)
                        if f_mark_i[ikyhat, iefwid] < fmark
                            # If all of these are satisfied, set new marks away from default:
                            ikyhat_write = ikyhat
                            iefwid_write = iefwid
                            fmark = f_mark_i[ikyhat,iefwid]
                            ikymin = nkyhat
                            ikymax = 1
                            iefwmin = nefwid
                            iefwmax = 1
                            f_guess_mark = T(1.0E20)
                            ikyhat_mark = ikyhat
                            iefwid_mark = iefwid
                        end
                        ikymin = min(ikyhat,ikymin)
                        ikymax = max(ikyhat,ikymax)
                        iefwmin = min(iefwid,iefwmin)
                        iefwmax = max(iefwid,iefwmax)
                        f_guess_mark = min(f_guess[ikyhat,iefwid],f_guess_mark)

                            # then set new maximums (fmark and gmark)
                            # and set f_guess_mark which corresponds to a maximized growthrate and
                            # minimized factor.
 
                            

                            # This loops over all combos of kyhat and width to find a single guess for "f"
                        

                    end
                end # iefwid
            end # ikyhat
        end # if ending (mode found)

        # Next is the writing of the out.scalefactor files.

        g = fill(T(NaN), inputsEP.NMODES)
        f = fill(T(NaN), inputsEP.NMODES)
        keep_label = fill("", inputsEP.NMODES)
        # Set FREQ_AE_UPPER and GAMMA_THRESH on the main inputsEP so that display labels
        # (BT/F/?) and kHz header lines are computed correctly.  The threaded scan used
        # deepcopy'd local structs and never wrote these back to the shared struct.
        inputsEP.FREQ_AE_UPPER = -abs(inputsPR.omegaGAM[inputsEP.IR])
        if inputsEP.ROTATIONAL_SUPPRESSION_FLAG == 1
            r_over_a = inputsPR.RMIN[inputsEP.IR] / inputsPR.RMIN[end]
            inputsEP.GAMMA_THRESH_MAX = abs(inputsPR.gammap[inputsEP.IR]) * 2.0 * (min(1.0 - r_over_a, r_over_a) / inputsPR.RMAJ[inputsEP.IR])
            inputsEP.GAMMA_THRESH = 0.15 * abs(inputsPR.gammaE[inputsEP.IR] / inputsPR.SHEAR[inputsEP.IR])
            inputsEP.GAMMA_THRESH = min(inputsEP.GAMMA_THRESH, inputsEP.GAMMA_THRESH_MAX)
        else
            inputsEP.GAMMA_THRESH = 1.0e-7
            inputsEP.GAMMA_THRESH_MAX = 1.0e-7
        end
        if (l_write_out && printout)
            filename = "out.scalefactor"*coalesce(inputsEP.SUFFIX, "")
            # Header written only once (first k), matching Fortran's file-create-once behavior
            if isempty(scalefactor_buffer)
                push!(scalefactor_buffer, "factor,(gamma(n),freq(n),flag,n=1,nmodes_in)")
                push!(scalefactor_buffer, "flag key:  'K'  mode is kept")
                if (inputsEP.REJECT_TEARING_FLAG == 1) push!(scalefactor_buffer, "           'T' rejected for tearing parity") end
                if (inputsEP.REJECT_I_PINCH_FLAG == 1) push!(scalefactor_buffer, "           'Pi' rejected for ion pinch") end
                if (inputsEP.REJECT_E_PINCH_FLAG == 1) push!(scalefactor_buffer, "           'Pe' rejected for electron pinch") end
                if (inputsEP.REJECT_TH_PINCH_FLAG == 1) push!(scalefactor_buffer, "           'Pth' rejected for thermal pinch") end
                if (inputsEP.REJECT_EP_PINCH_FLAG == 1) push!(scalefactor_buffer, "           'PEP' rejected for EP pinch") end
                push!(scalefactor_buffer, "           'QLR' rejected for QL ratio EP/|chi_i| < $(inputsEP.QL_RATIO_THRESH)")
                push!(scalefactor_buffer, "           'TH2' rejected for <theta^2>  > $(inputsEP.THETA_SQ_THRESH)")
                push!(scalefactor_buffer, "           'F'   rejected for non-AE frequency > $(inputsEP.F_REAL[inputsEP.IR]*inputsEP.FREQ_AE_UPPER) kHz")
                push!(scalefactor_buffer, "           'BT'  mode growth rate is below threshold gamma_thresh = $(inputsEP.F_REAL[inputsEP.IR] * inputsEP.GAMMA_THRESH) kHz")
                push!(scalefactor_buffer, "omega_TAE = $(inputsEP.F_REAL[inputsEP.IR] * inputsPR.OMEGA_TAE[inputsEP.IR]) ;  omega_GAM = $(-inputsEP.F_REAL[inputsEP.IR] * inputsPR.omegaGAM[inputsEP.IR])")
                if inputsEP.REAL_FREQ != 0
                    push!(scalefactor_buffer, "Frequencies in real units, plasma frame [kHz] ; (c_s/a)/(2*pi) = $(inputsEP.F_REAL[inputsEP.IR]) kHz")
                end
            end

            ky_write = kyhat[ikyhat_write]*zs_ep/sqrt(mass_ep*taus_ep)

            push!(scalefactor_buffer, _scalefactor_section_header(kyhat[ikyhat_write], ky_write, efwid[iefwid_write], fmark))

            for ifactor = 1:nfactor
                ikyhat = ikyhat_write
                iefwid = iefwid_write
                for n = 1:inputsEP.NMODES
                    g[n] = inputsEP.F_REAL[inputsEP.IR]*growthrate[ikyhat, iefwid, ifactor, n]
                    f[n] = inputsEP.F_REAL[inputsEP.IR]*frequency[ikyhat, iefwid, ifactor, n]
                    if (lkeep_i[ikyhat,iefwid,ifactor,n])
                        keep_label[n] = " K  "
                    elseif growthrate[ikyhat, iefwid, ifactor, n] < inputsEP.GAMMA_THRESH 
                        keep_label[n] = " BT  "
                    elseif ((ltearing_i[ikyhat,iefwid,ifactor,n]) && (inputsEP.REJECT_TEARING_FLAG == 1))
                        keep_label[n] = " T  "
                    elseif ((l_i_pinch_i[ikyhat,iefwid,ifactor,n]) && (inputsEP.REJECT_I_PINCH_FLAG == 1))
                        keep_label[n] = " Pi "
                    elseif ((l_e_pinch_i[ikyhat,iefwid,ifactor,n]) && (inputsEP.REJECT_E_PINCH_FLAG == 1))
                        keep_label[n] = " Pe "
                    elseif ((l_th_pinch_i[ikyhat,iefwid,ifactor,n]) && (inputsEP.REJECT_TH_PINCH_FLAG == 1))
                        keep_label[n] = " Pth"
                    elseif ((l_EP_pinch_i[ikyhat,iefwid,ifactor,n]) && (inputsEP.REJECT_EP_PINCH_FLAG == 1))
                        keep_label[n] = " PEP"
                    elseif (l_QL_ratio_i[ikyhat,iefwid,ifactor,n])
                        keep_label[n] = " QLR"
                    elseif l_theta_sq_i[ikyhat, iefwid, ifactor, n]
                        keep_label[n] = " TH2"
                    elseif (frequency[ikyhat,iefwid,ifactor,n] > inputsEP.FREQ_AE_UPPER)
                        keep_label[n] = " F  "
                    else
                        keep_label[n] = " ?  "
                    end
                end
                push!(scalefactor_buffer, _scalefactor_factor_line(factor[ifactor], g, f, keep_label))
            end # ifactor
        end 

        # After printing the info just calculated, if fmark is already significantly small, adjust 
        # ranges within the range, otherwise, move to a new order of 10 in the factor.
        # if (fmark < 1.0e10)
        #     # If in the first scan round, set the new max factor to the minimized factor, otherwise, do
        #     # some more specific things to hone in each round (including in kyhat and width)
        if fmark < 1.0E10  # accepted mode with all constraints
            f1 = 2.0 * fmark
            f0 = zero(T)
            delw = (w1 - w0) / (nefwid - 1)
        
            w1 = efwid[iefwmax] + delw
            w0 = efwid[iefwmin] - delw
            if w1 > inputsEP.WIDTH_MAX
                w1 =inputsEP.WIDTH_MAX
            end
            if w0 < inputsEP.WIDTH_MIN
                w0 = inputsEP.WIDTH_MIN
            end
        
            delky = (kyhat1 - kyhat0) / nkyhat
            kyhat1 = kyhat[ikymax] + delky
            kyhat0 = kyhat[ikymin] - delky
            if kyhat1 > kyhat_max
                kyhat1 = kyhat_max
            end
            if kyhat0 < kyhat_min
                kyhat0 = kyhat_min
            end
        
        else
            f0 = f1
            f1 = 10.0*f1
        end

        _probe && (_ser_wall += (time_ns() - _t_ser) / 1e9)
    end # end of k
    use_ws && _ws_scan_end!()

    if _probe
        _tot = (time_ns() - _t_scan) / 1e9
        _ncombo = _PROBE_N[]
        @info string("[TGLFEP_JULIA_PROBE] kwscale_scan radius IR=", inputsEP.IR,
            " inner=", inner, " team=", team === nothing ? 0 : length(team),
            " : total=", round(_tot; digits=2), "s",
            " | combo_map=", round(_map_wall; digits=2), "s",
            " serial_reduce=", round(_ser_wall; digits=3), "s",
            " | combos=", _ncombo,
            " sum(TJLFEP_ky)=", round(_PROBE_KY[]; digits=1), "s",
            " sum(TJLF.run/eigensolve)=", round(_PROBE_RUN[]; digits=1), "s",
            " eigensolve_frac=", _PROBE_KY[] > 0 ? round(100 * _PROBE_RUN[] / _PROBE_KY[]; digits=1) : 0.0, "%",
            " (per-process sums; for :threads this is the whole radius)")
    end

    # After the fourth round, if no unstable modes have been found, default the scalefactor which will be used In
    # the calculation in the driver to 10k, the width to its default (I think around 100) and same for kyhat.
    # Otherwise, set the factor to the best guess at ikyhat_mark, iefwid_mark and their corresponding width and kyhat
    # to those marks.
    if (imark_min > nfactor)
        # If, over the scan of k, there's no unstable modes, set each to lowest.
        inputsEP.FACTOR_IN = 10000
        inputsEP.WIDTH_IN = efwid[1]
        inputsEP.KYMARK = kyhat[1]
    
    else
        # If, over the scan of k, there's an unstable mode, set each to each marked point.
        inputsEP.FACTOR_IN = f_guess_mark
        inputsEP.WIDTH_IN = efwid[iefwid_mark]
        inputsEP.KYMARK = kyhat[ikyhat_mark]
    end

    marginal_ql = extract_marginal_ql(
        inputsEP, inputsPR, growthrate, dep_scan, factor;
        imark=imark, ikyhat_mark=ikyhat_mark, iefwid_mark=iefwid_mark,
        imark_min=imark_min, nkyhat=nkyhat, nefwid=nefwid, nfactor=nfactor,
        use_gpu=use_gpu, use_flux_scan=ql_flux_scan,
    )

    # Return to the driver these values and the final growthrate values of the last scan.
    return growthrate, inputsEP, inputsPR, marginal_ql, scalefactor_buffer, wavebuffer_all

    # This function will be done for however many radii you are testing. These values do not interact in the driver. 
end
