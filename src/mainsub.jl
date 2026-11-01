"""
    mainsub(inputsEP, inputsPR, printout=true; solver=:grid, kwargs...)

Single-radius TGLF-EP driver. Dispatches on `inputsEP.PROCESS_IN`: `3` runs the linear
γ/ω spectrum diagnostic ([`_mainsub_spectrum`](@ref)); `5` and `6` run the
critical-EP-density-gradient threshold scan, differing only in the drive selector —
`5` uses `MODE_IN=2` (EP drive only) while `6` uses `MODE_IN=4` (thermal+EP gradients on,
ITG/TEM basis via `FILTER=2`). Both fold to `PROCESS_IN=5` before the scan (matching Fortran
`TGLFEP_mainsub` `case(5:6)`), so downstream output is identical. Mode 6 is only supported
with `solver=:grid` (the AD engines model the EP-drive-only onset). `solver` selects the
critical-factor engine (PROCESS_IN=5 only):

  - `:grid` (default) — the Fortran-equivalent `kwscale_scan` `(kyhat × width ×
    factor)` grid sweep (full `IFLUX=true` per combo). The trusted reference path.
  - `:ad` — the fast autodiff route: [`critical_factor_optimize`](@ref) finds the
    critical EP scale factor by AD-Newton onset solves (cheap `IFLUX=false`
    eigenvalue passes) with implicit-function-theorem `(kyhat,width)` descent,
    then confirms the all-filter onset with [`marginal_factor_faithful`](@ref).
    Cheapest, but the descent can land in a shallower local basin on the bumpy
    `f★(ky,w)` surface (so it can disagree with the grid at some radii).
  - `:robust_ad` — the robust autodiff route: [`critical_factor_robust`](@ref)
    takes the global minimum of the all-filter onset over a `(kyhat,width)` grid
    (plus `refine_rounds` of `(ky,w)` window narrowing that mirror the Fortran
    refinement), so it reproduces the grid `sfmin` to within the continuous-vs-
    discrete-factor difference. `refine_rounds` is the accuracy/speed knob: `0` is
    the cheapest coarse-grid min, larger values better resolve radii whose binding
    `(ky,w)` lies between coarse nodes (e.g. the plasma edge) at proportionally
    higher cost.
  - `:truth` — the physical-truth route: [`critical_factor_truth`](@ref) extends the
    `width` search **below** `WIDTH_MIN` (log-spaced down to `w_lo`, default the ballooning-width
    physics floor `WIDTH_FLOOR` = 0.5) to capture the
    genuine narrow-width EP-driven AEs that the Fortran-faithful `w≥1` box excludes,
    then converges the value in `nbasis` at the located optimum. NOT Fortran-faithful:
    at near-marginal radii it can return a `sfmin` up to ~25× lower (more unstable)
    than `:grid`/`:robust_ad`. Use it for the most-unstable physical threshold rather
    than Fortran equivalence.
  - `:multistart` / `:nlopt` — the derivative-free routes (no AD in the
    `(kyhat,width)` search): [`critical_factor_multistart`](@ref) (cheap AE-onset
    multistart + local BOBYQA/SBPLX descents) and [`critical_factor_nlopt`](@ref)
    (DIRECT-L global + BOBYQA polish).
    Both are narrow-width extended like `:truth` and faithful-confirm their optima.

All AD solvers set `FACTOR_IN`/`KYMARK`/`WIDTH_IN` like the grid path and return
the same `((growthrate, inputsEP, inputsPR, marginal_ql), (scalefactor_buffer,
wavebuffer_all))` shape, with placeholder `growthrate`/`marginal_ql` (the AD paths
do not assemble the full QL buffers).
"""
function mainsub(inputsEP::Options, inputsPR::profile, printout::Bool = true; use_gpu::Bool = false,
                 inner::Symbol = :threads, team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                 ql_flux_scan::Bool = false, solver::Symbol = :grid, refine_rounds::Int = 1,
                 k_max::Int = _k_max_env(),
                 extend_mode::Union{Nothing,Symbol} = nothing, wide_kdesc::Union{Nothing,Int} = nothing,
                 faithful_confirm::Union{Nothing,Bool} = nothing)
    solver in (:grid, :ad, :robust_ad, :truth, :multistart, :nlopt) ||
        error("mainsub: solver must be :grid, :ad, :robust_ad, :truth, :multistart, or :nlopt (got $solver)")
    x = inputsEP.PROCESS_IN
    if (x == 3)
        return _mainsub_spectrum(inputsEP, inputsPR, printout; use_gpu=use_gpu, inner=inner, team=team)
    elseif (x == 5 || x == 6)
        # Fortran TGLFEP_mainsub case(5:6): both run the (kyhat,width,factor) kwscale_scan,
        # differing only in the drive selector — PROCESS_IN=5 uses MODE_IN=2 (EP drive only),
        # PROCESS_IN=6 uses MODE_IN=4 (thermal+EP gradients on, ITG/TEM basis via FILTER=2).
        # Both then collapse to PROCESS_IN=5 so all downstream threshold output is identical
        # (mirrors the Fortran `process_in = 5` right before the scan). The upstream FACTOR
        # rescale in preprocessing keys on the *pre-fold* PROCESS_IN (5 skips it; 6 rescales
        # like 4), so this fold must happen here and not in the input preprocessing.
        mode_in = x == 6 ? 4 : 2
        inputsEP.WIDTH_IN_FLAG = false
        inputsEP.MODE_IN = mode_in
        inputsEP.KY_MODEL = 3
        inputsEP.PROCESS_IN = 5
        dbgmsg("mainsub ir=", inputsEP.IR, " suffix=", inputsEP.SUFFIX,
            " SCAN_N=", inputsEP.SCAN_N, " N_BASIS=", inputsEP.N_BASIS, " MODE_IN=", mode_in)

        # The AD building blocks now *honor* MODE_IN (they thread it into TJLF_map/TJLFEP_ky),
        # but the AD engines' onset/IFT math is still built around the EP-drive-only (MODE_IN=2)
        # AE picture: with thermal gradients on (MODE_IN=4) the critical EP factor is no longer a
        # clean γ(factor) threshold crossing, so the descent/marginal logic is not yet validated
        # for mode 6. Keep it grid-only until that onset behavior is checked; drop this guard once
        # the AD path is validated for the thermal+EP case.
        if mode_in == 4 && solver != :grid
            error("mainsub: PROCESS_IN=6 (MODE_IN=4 thermal+EP / ITG-TEM threshold) is only " *
                  "supported with solver=:grid; got solver=$(solver). Use PROCESS_IN=5 for the " *
                  "AD solvers, or set solver=:grid.")
        end

        if solver == :ad
            return _mainsub_ad(inputsEP, inputsPR, printout; use_gpu=use_gpu, inner=inner, team=team, k_max=k_max,
                               extend_mode=extend_mode, wide_kdesc=wide_kdesc, faithful_confirm=faithful_confirm)
        elseif solver == :robust_ad
            return _mainsub_robust_ad(inputsEP, inputsPR, printout; use_gpu=use_gpu,
                                      inner=inner, team=team, refine_rounds=refine_rounds, k_max=k_max)
        elseif solver == :truth
            return _mainsub_truth(inputsEP, inputsPR, printout; use_gpu=use_gpu, inner=inner, team=team, k_max=k_max)
        elseif solver == :multistart
            return _mainsub_multistart(inputsEP, inputsPR, printout; use_gpu=use_gpu, inner=inner, team=team, k_max=k_max)
        elseif solver == :nlopt
            return _mainsub_nlopt(inputsEP, inputsPR, printout; use_gpu=use_gpu, inner=inner, team=team, k_max=k_max)
        end

        growthrate, inputsEP, inputsPR, marginal_ql, scalefactor_buffer, wavebuffer_all =
            kwscale_scan(inputsEP, inputsPR, printout; use_gpu=use_gpu, inner=inner, team=team,
                         ql_flux_scan=ql_flux_scan, mode_in=mode_in, k_max=k_max)
        return (growthrate, inputsEP, inputsPR, marginal_ql), (scalefactor_buffer, wavebuffer_all)
    else
        _process_in_unsupported(x)
    end
end

# Fortran TGLFEP defines PROCESS_IN modes 0–7, but TJLFEP ports only the linear spectrum
# diagnostic (3) and the critical-EP-density-gradient threshold scan (5, plus its MODE_IN=4
# thermal+EP / ITG-TEM variant 6, which folds onto the same kwscale_scan). Callers
# destructure a `(growth, ep, pr, ...)` tuple, so silently returning a sentinel would crash
# downstream with a cryptic MethodError; throw an actionable error naming the supported modes.
function _process_in_unsupported(x)
    error("mainsub: PROCESS_IN=$(x) is not implemented in TJLFEP. Supported modes are " *
          "3 (linear γ/ω spectrum diagnostic), 5 (critical-EP-density-gradient threshold " *
          "scan, EP drive only), and 6 (the same scan with thermal+EP gradients / ITG-TEM " *
          "basis). Fortran TGLFEP modes 0, 1, 2, 4, and 7 have not been ported; use " *
          "PROCESS_IN=5 or 6 for critical-gradient scans or PROCESS_IN=3 for the spectrum.")
end

"""
    _mainsub_spectrum(inputsEP, inputsPR, printout; use_gpu=false, inner=:threads, team=nothing)
        -> ((growthrate, ep, pr, spectra), (nothing, file_buffers))

Spectrum branch of [`mainsub`](@ref) (PROCESS_IN==3). Mirrors the Fortran
`TGLFEP_mainsub.f90` `case(3)`:

  1. When `WIDTH_IN_FLAG` is false, set `MODE_IN=2` and run
     [`TJLFEP_ky_widthscan`](@ref) to locate the width at maximum `gamma_AE`; store it on
     `inputsEP.WIDTH_IN` and the scan ky on `inputsEP.KYMARK`.
  2. For `mode_in ∈ {1,2,4}` run [`TJLFEP_TM`](@ref) (full transport-model `ky` spectrum)
     and collect the `out.eigenvalue_m<mode>` buffers.

Returns the standard `mainsub` tuple shape. `spectra` (the 4th element) is a `Dict`
`mode_in => (ky, gamma, freq)` of in-memory spectra; `file_buffers` (the 2nd tuple's
`wf`-slot) is the `(filename, lines)` list the radial driver writes to disk.
"""
function _mainsub_spectrum(inputsEP::Options, inputsPR::profile, printout::Bool;
                           use_gpu::Bool = false, inner::Symbol = :threads,
                           team::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    file_buffers = Tuple{String,Vector{String}}[]

    # Auto width: scan width at EP-only drive (mode_in=2) and mark the max-gamma width.
    if !inputsEP.WIDTH_IN_FLAG
        inputsEP.MODE_IN = 2
        width_in, _gmark, _fmark, ky_in, ws_buf =
            TJLFEP_ky_widthscan(inputsEP, inputsPR; use_gpu=use_gpu, inner=inner, team=team)
        inputsEP.WIDTH_IN = width_in
        inputsEP.KYMARK = ky_in
        printout && push!(file_buffers, ws_buf)
    end

    # gamma/omega spectra at the three Fortran spectrum modes.
    spectra = Dict{Int,NamedTuple{(:ky, :gamma, :freq)}}()
    for mode in (1, 2, 4)
        inputsEP.MODE_IN = mode
        ky, gamma, freq, ev_buf = TJLFEP_TM(inputsEP, inputsPR; mode_in=mode, use_gpu=use_gpu)
        spectra[mode] = (ky=ky, gamma=gamma, freq=freq)
        printout && push!(file_buffers, ev_buf)
    end

    growthrate = fill(NaN, (5, 10, 10, inputsEP.NMODES))
    return (growthrate, inputsEP, inputsPR, spectra), (nothing, file_buffers)
end

# Replay a single IFLUX=true TJLFEP_ky at an AD solver's winning (factor, ky, width) to emit the same
# out.wavefunction file the grid path writes for that combo. The AD engines locate the critical point
# via marginal-onset root-finding and never assemble the eigenvector/QL post-processing themselves, so
# without this replay `WRITE_WAVEFUNCTION=1` yields no wavefunction file under ANY AD solver (the file
# writers no-op on a `nothing` buffer). Returns the `[(filename, buffer)]` wf_buf_all the radius
# writers expect, or `nothing` when not requested or the winner is not a finite located point.
# `mode_in_override=2` matches the grid's PROCESS_IN=5 EP-drive selector; `_sf_factor_tag` reproduces
# the grid's `_sfNNN.NNN` filename so both engines emit identically-named files.
function _ad_wavefunction_buffer(inputsEP::Options, inputsPR::profile, sfmin, kymark, wmark,
                                 printout::Bool; use_gpu::Bool = false)
    (printout && coalesce(inputsEP.WRITE_WAVEFUNCTION, 0) == 1 &&
     isfinite(sfmin) && isfinite(kymark) && isfinite(wmark)) || return nothing
    local_ep = deepcopy(inputsEP)
    local_ep.FACTOR_IN = sfmin
    local_ep.KYHAT_IN  = kymark
    local_ep.WIDTH_IN  = wmark
    str_wf_file = "out.wavefunction" * coalesce(local_ep.SUFFIX, "") * "_sf" * _sf_factor_tag(sfmin)
    _, _, _, _, wavefunction_buffer, _, _ =
        TJLFEP_ky(local_ep, inputsPR, str_wf_file, 1; eigen_cache=nothing,
                  use_gpu=use_gpu, mode_in_override=2)
    return wavefunction_buffer === nothing ? nothing : [(str_wf_file, wavefunction_buffer)]
end

"""
    _mainsub_ad(inputsEP, inputsPR, printout; use_gpu=false, inner=:threads, team=nothing) -> ((growthrate, ep, pr, marginal_ql), (sf_buf, wf_buf))

AD-solver branch of [`mainsub`](@ref) (PROCESS_IN==5). Runs
[`critical_factor_optimize`](@ref) to obtain the critical EP scale factor and its
marking `(kymark, width)`. The `extend_mode` knob selects the cost/accuracy tier:
`:locate` (default, faithful-confirmed + **width-extended** `extend_width=true`,
tracks `:robust_ad`), `:wide` (cheap single-pass width extension), or `:only`
(fast-turnaround "bare" AD — canonical `w≥1` box, `extend_width=false` and no
faithful confirm; fastest, but misses the narrow-width edge modes). Writes the
result onto `inputsEP`
(`FACTOR_IN`, `KYMARK`, `WIDTH_IN`) exactly like the grid path, and returns the
`mainsub` tuple shape so the radial drivers/output writers are unchanged. The AD
path does not assemble the full QL/growthrate buffers, so `growthrate` is a NaN
placeholder and `marginal_ql` is `nothing`; the scalefactor buffer carries a short
AD summary when `printout`.

`inner`/`team` select the within-radius parallelism of the AD path's independent-
eval regions (seed grid, eigenvalue hull scan, faithful sweep): `:mps_team`
distributes them across the MPS team (the production GPU route, separate CUDA
contexts overlapping via Hyper-Q), else they run in-process threaded.
"""
function _mainsub_ad(inputsEP::Options, inputsPR::profile, printout::Bool; use_gpu::Bool = false,
                     inner::Symbol = :threads,
                     team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                     k_max::Int = _k_max_env(),
                     extend_mode::Union{Nothing,Symbol} = nothing,
                     wide_kdesc::Union{Nothing,Int} = nothing,
                     faithful_confirm::Union{Nothing,Bool} = nothing)
    # Use the SAME scan floor as :robust_ad (the grid-equivalent _ad_factor_floor, not the optimizer's
    # 1e-3 default) so the width-extended :ad reports the same floor-pinned sfmin as the production
    # solver at near-marginal radii (otherwise it would clip ~10× lower at the 1e-3 floor and spuriously
    # beat :robust_ad). The floor tracks k_max: FACTOR_IN/(nfactor·4^(k_max-1)) (= FACTOR_IN/512 at
    # the default k_max=4), so a larger k_max deepens the AD factor search like it does the grid.
    #
    # The three AD-extension knobs resolve as: explicit kwarg (e.g. from the FUSE actor) when given,
    # else the matching environment variable, else the production default. This keeps env-driven runs
    # (timing harnesses, NN-DB scripts) working while letting callers pass the knobs directly.
    #
    # `faithful_confirm`/`AD_FAITHFUL_CONFIRM=0` → cheap "pure AD" path (no IFLUX=true confirm anywhere
    #   → reports the AE-band onset, which does NOT match :robust_ad bitwise). Default keeps confirm on.
    # `extend_mode`/`AD_EXTEND_MODE`:
    #   `:locate` (default) → dense narrow-width locate, tracks :robust_ad essentially bit-for-bit.
    #   `:wide`             → cheap single-pass width extension (widened-box log-seeded multistart,
    #                         conservative but not bitwise-:robust_ad).
    #   `:only`             → fast-turnaround "bare" AD: the canonical `w≥1` box only, no width
    #                         extension and no faithful confirm (AE-band onset). Fastest and least
    #                         accurate — for quick iteration, NOT production / NN-DB generation, since
    #                         it misses the narrow-width edge modes.
    # `wide_kdesc`/`AD_WIDE_KDESC` → :wide multistart breadth (mode=:wide only).
    ad_confirm = faithful_confirm === nothing ? (get(ENV, "AD_FAITHFUL_CONFIRM", "1") != "0") : faithful_confirm
    ad_mode    = extend_mode === nothing ? Symbol(get(ENV, "AD_EXTEND_MODE", "locate")) : extend_mode
    ad_kdesc   = wide_kdesc === nothing ? parse(Int, get(ENV, "AD_WIDE_KDESC", "2")) : wide_kdesc
    ad_wlo     = _ad_width_floor(inputsEP, inputsPR)   # narrow-width search floor, never below WIDTH_FLOOR (0.5)
    # :only forces the bare config (no width extension, no faithful confirm).
    ad_extend  = ad_mode !== :only
    ad_confirm = ad_extend ? ad_confirm : false
    res = critical_factor_optimize(inputsEP, inputsPR; use_gpu=use_gpu, faithful_confirm=ad_confirm,
                                   extend_width=ad_extend,
                                   w_lo=ad_wlo,
                                   extend_mode=(ad_extend ? ad_mode : :locate), wide_kdesc=ad_kdesc,
                                   scan_lo=_ad_factor_floor(Float64(inputsEP.FACTOR_IN), k_max),
                                   inner=inner, team=team)

    # Prefer the all-filter (faithful) onset when a mode actually binds; otherwise
    # fall back to the AE-band optimum from the descent.
    sfmin = res.sfmin
    if res.faithful !== nothing && res.faithful.binding != :none &&
       isfinite(res.faithful.factor_faithful)
        sfmin = res.faithful.factor_faithful
    end
    kymark = res.kyhat
    wmark  = res.width

    inputsEP.FACTOR_IN = isfinite(sfmin) ? sfmin : inputsEP.FACTOR_IN
    inputsEP.KYMARK    = kymark
    if isfinite(wmark)
        inputsEP.WIDTH_IN = wmark
    end

    growthrate  = fill(NaN, (5, 10, 10, inputsEP.NMODES))
    marginal_ql = nothing

    sf_buf = String[]
    if printout
        bind = res.faithful === nothing ? :none : res.faithful.binding
        push!(sf_buf, "# TJLFEP AD solver (critical_factor_optimize + marginal_factor_faithful)")
        push!(sf_buf, "ir = $(inputsEP.IR)")
        push!(sf_buf, "sfmin = $(sfmin)")
        push!(sf_buf, "kymark = $(kymark)")
        push!(sf_buf, "width = $(wmark)")
        push!(sf_buf, "binding = $(bind)")
        push!(sf_buf, "w_lo = $(ad_wlo)   # narrow-width search floor (>= WIDTH_FLOOR = $(WIDTH_FLOOR); AD_W_LO)")
        push!(sf_buf, "iters = $(res.iters)  evals = $(res.evals)  converged = $(res.converged)")
        if get(res, :extended, false)
            push!(sf_buf, "extended = true  n_ext_confirm = $(get(res, :n_ext_confirm, 0))")
        end
    end

    # Wavefunction output at the AD winner (see `_ad_wavefunction_buffer`).
    wf_buf_all = _ad_wavefunction_buffer(inputsEP, inputsPR, sfmin, kymark, wmark, printout; use_gpu=use_gpu)

    return (growthrate, inputsEP, inputsPR, marginal_ql), (sf_buf, wf_buf_all)
end

"""
    _mainsub_robust_ad(inputsEP, inputsPR, printout; use_gpu=false, inner=:threads,
                       team=nothing, refine_rounds=1) -> ((growthrate, ep, pr, marginal_ql), (sf_buf, wf_buf))

Robust AD-solver branch of [`mainsub`](@ref) (PROCESS_IN==5). Runs
[`critical_factor_robust`](@ref) — the global minimum of the all-filter
marginal factor over a `(kyhat,width)` grid with `refine_rounds` of window
narrowing, **width-extended** (`extend_width=true`) below `WIDTH_MIN` to capture the
narrow EP-driven AE modes the canonical `w≥1` box misses — to obtain the critical EP
scale factor and its marking `(kymark, width)`, writes them onto `inputsEP`
(`FACTOR_IN`, `KYMARK`, `WIDTH_IN`) exactly like the grid path, and returns the
`mainsub` tuple shape. This is the middle (`width-correct`, `nb=N_BASIS`) rung of the
`grid → robust_ad → truth` ladder; the `nbasis`-converged tier is `:truth`.
`refine_rounds` is the user-facing accuracy/speed knob (see [`critical_factor_robust`](@ref)).

If the reduction reports no genuine interior onset (`status ∈ (:no_onset, :cap)`),
`FACTOR_IN` is left at its incoming value (the grid path remains the fallback for
that radius) and the status is recorded in the scalefactor buffer.

`inner`/`team` select the within-radius parallelism (the `(ky,w)` grid points are
distributed across the MPS team or in-process threads); each point's inner
faithful factor sweep runs serially to avoid nested oversubscription.
"""
function _mainsub_robust_ad(inputsEP::Options, inputsPR::profile, printout::Bool;
                            use_gpu::Bool = false, inner::Symbol = :threads,
                            team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                            refine_rounds::Int = 1, k_max::Int = _k_max_env())
    res = critical_factor_robust(inputsEP, inputsPR; use_gpu=use_gpu, k_max=k_max,
                                 inner=inner, team=team, refine_rounds=refine_rounds,
                                 w_lo=_ad_width_floor(inputsEP, inputsPR))

    sfmin  = res.sfmin
    kymark = res.kyhat
    wmark  = res.width
    genuine = res.status === :ok && isfinite(sfmin)

    # Only adopt the AD onset when a genuine interior onset was found; otherwise leave
    # FACTOR_IN untouched so the radius can fall back to the grid path / its prior value.
    if genuine
        inputsEP.FACTOR_IN = sfmin
        inputsEP.KYMARK    = kymark
        if isfinite(wmark)
            inputsEP.WIDTH_IN = wmark
        end
    end

    growthrate  = fill(NaN, (5, 10, 10, inputsEP.NMODES))
    marginal_ql = nothing

    sf_buf = String[]
    if printout
        push!(sf_buf, "# TJLFEP AD solver (critical_factor_robust width-extended, refine_rounds=$(refine_rounds))")
        push!(sf_buf, "ir = $(inputsEP.IR)")
        push!(sf_buf, "sfmin = $(sfmin)")
        push!(sf_buf, "kymark = $(kymark)")
        push!(sf_buf, "width = $(wmark)")
        push!(sf_buf, "binding = $(res.binding)")
        push!(sf_buf, "status = $(res.status)")
        push!(sf_buf, "refine_done = $(res.refine_done)  (budget=$(refine_rounds), adaptive)")
        push!(sf_buf, "extended = $(res.extended)  n_ext_confirm = $(res.n_ext_confirm)")
        push!(sf_buf, "n_feasible_coarse = $(res.n_feasible_coarse) / $(res.npts_coarse)")
        push!(sf_buf, "npts = $(res.npts)  evals_full = $(res.total_evals_full)  evals_eig = $(res.total_evals_eig)")
    end

    # Wavefunction at the adopted winner (only for a genuine interior onset; see `_ad_wavefunction_buffer`).
    wf_buf_all = genuine ?
        _ad_wavefunction_buffer(inputsEP, inputsPR, sfmin, kymark, wmark, printout; use_gpu=use_gpu) : nothing

    return (growthrate, inputsEP, inputsPR, marginal_ql), (sf_buf, wf_buf_all)
end

"""
    _mainsub_truth(inputsEP, inputsPR, printout; use_gpu=false, inner=:threads,
                   team=nothing) -> ((growthrate, ep, pr, marginal_ql), (sf_buf, wf_buf))

Physical-truth AD-solver branch of [`mainsub`](@ref) (PROCESS_IN==5). Runs
[`critical_factor_truth`](@ref) — locates the global `(kyhat,width)` minimum of the
all-filter marginal factor over a width box **extended below `WIDTH_MIN`** (log-spaced
down to `w_lo`, default the physics floor `WIDTH_FLOOR` = 0.5) and converges the value in `nbasis` at that optimum — then writes the
critical EP scale factor and its marking `(kymark, width)` onto `inputsEP`
(`FACTOR_IN`, `KYMARK`, `WIDTH_IN`) like the grid path. The reported `sfmin` is the
nbasis-converged estimate; `sfmin_work` (the value at the working basis) is recorded too.

NOT Fortran-faithful: the narrow-width modes it captures are excluded by the canonical
`w≥1` box, so at near-marginal radii `sfmin` can be much lower (more unstable). If no
genuine onset is found (`status ∈ (:no_onset,:cap)`), `FACTOR_IN` is left untouched.
`inner`/`team` select within-radius parallelism over the seed grid / nbasis evals.
"""
function _mainsub_truth(inputsEP::Options, inputsPR::profile, printout::Bool;
                        use_gpu::Bool = false, inner::Symbol = :threads,
                        team::Union{Nothing,AbstractVector{<:Integer}} = nothing, k_max::Int = _k_max_env())
    # Honor the requested working basis (N_BASIS); the nbasis-convergence sweep climbs from it in
    # +8 steps up to the max stable basis (nb=56; nb>=64 Hermite overlap is singular). At the
    # production nb=32 this is nb_steps=[32,40,48,56] -- the 4th point (56) is the best convergence
    # evidence available for the still-climbing outer radii and strengthens the geometric test.
    nbw = Int(inputsEP.N_BASIS)
    nb_steps = sort(unique(filter(<=(56), [nbw, nbw + 8, nbw + 16, nbw + 24])))
    res = critical_factor_truth(inputsEP, inputsPR; use_gpu=use_gpu, inner=inner, team=team, k_max=k_max,
                                nb_work=nbw, nb_steps=nb_steps,
                                w_lo=_ad_width_floor(inputsEP, inputsPR))

    sfmin  = res.sfmin
    kymark = res.kyhat
    wmark  = res.width
    genuine = res.status === :ok && isfinite(sfmin)

    if genuine
        inputsEP.FACTOR_IN = sfmin
        inputsEP.KYMARK    = kymark
        if isfinite(wmark)
            inputsEP.WIDTH_IN = wmark
        end
    end

    growthrate  = fill(NaN, (5, 10, 10, inputsEP.NMODES))
    marginal_ql = nothing

    sf_buf = String[]
    if printout
        nbs, vals = res.nb_table
        push!(sf_buf, "# TJLFEP AD solver (critical_factor_truth = robust_ad width-extended + nbasis-converged)")
        push!(sf_buf, "ir = $(inputsEP.IR)")
        push!(sf_buf, "sfmin = $(sfmin)   # nbasis-converged (production accuracy tier)")
        push!(sf_buf, "sfmin_work = $(res.sfmin_work)   # robust_ad value at working nbasis ($(Int(inputsEP.N_BASIS)))")
        push!(sf_buf, "kymark = $(kymark)")
        push!(sf_buf, "width = $(wmark)")
        push!(sf_buf, "binding = $(res.binding)")
        push!(sf_buf, "status = $(res.status)")
        push!(sf_buf, "nbasis = $(collect(nbs)) -> $(collect(vals))  (converged=$(res.nb_converged), limit=$(res.nb_limit))")
        push!(sf_buf, "sfmin_conv = $(res.sfmin_conv)   # nbasis-converged narrow (pre-floor)")
        push!(sf_buf, "sfmin_w1 = $(res.sfmin_w1)   # robust_ad w>=1 floor (floored=$(res.floored))")
        push!(sf_buf, "feasible_frac = $(res.feasible_frac)")
        push!(sf_buf, "n_confirm = $(res.n_confirm)  evals_full = $(res.total_evals_full)  evals_eig = $(res.total_evals_eig)")
    end

    # Wavefunction at the adopted winner (only for a genuine interior onset; see `_ad_wavefunction_buffer`).
    wf_buf_all = genuine ?
        _ad_wavefunction_buffer(inputsEP, inputsPR, sfmin, kymark, wmark, printout; use_gpu=use_gpu) : nothing

    return (growthrate, inputsEP, inputsPR, marginal_ql), (sf_buf, wf_buf_all)
end

# ── shared adopt+report tail for the derivative-free (ky,w) solvers ──
# `res` is the `critical_factor_multistart`/`critical_factor_nlopt` contract. Like :robust_ad/:truth,
# adopt the onset onto inputsEP only for a genuine interior onset (status===:ok); otherwise leave
# FACTOR_IN untouched so the radius falls back to its prior value / the grid path.
function _finalize_nls_result!(inputsEP::Options, inputsPR::profile, printout::Bool,
                               res, header::String; use_gpu::Bool = false)
    sfmin  = res.sfmin
    kymark = res.kyhat
    wmark  = res.width
    genuine = res.status === :ok && isfinite(sfmin)
    if genuine
        inputsEP.FACTOR_IN = sfmin
        inputsEP.KYMARK    = kymark
        isfinite(wmark) && (inputsEP.WIDTH_IN = wmark)
    end
    growthrate  = fill(NaN, (5, 10, 10, inputsEP.NMODES))
    marginal_ql = nothing
    sf_buf = String[]
    if printout
        push!(sf_buf, header)
        push!(sf_buf, "ir = $(inputsEP.IR)")
        push!(sf_buf, "sfmin = $(sfmin)")
        push!(sf_buf, "kymark = $(kymark)")
        push!(sf_buf, "width = $(wmark)")
        push!(sf_buf, "binding = $(res.binding)")
        push!(sf_buf, "status = $(res.status)")
        push!(sf_buf, "n_samples = $(res.n_samples)  n_confirm = $(res.n_confirm)")
        push!(sf_buf, "evals_full = $(res.total_evals_full)  evals_eig = $(res.total_evals_eig)")
    end
    # Wavefunction at the adopted winner (only for a genuine interior onset; see `_ad_wavefunction_buffer`).
    wf_buf_all = genuine ?
        _ad_wavefunction_buffer(inputsEP, inputsPR, sfmin, kymark, wmark, printout; use_gpu=use_gpu) : nothing
    return (growthrate, inputsEP, inputsPR, marginal_ql), (sf_buf, wf_buf_all)
end

"""
    _mainsub_multistart(inputsEP, inputsPR, printout; use_gpu=false, inner=:threads, team=nothing)

Derivative-free (`:multistart`) branch of [`mainsub`](@ref) (PROCESS_IN==5). Runs
[`critical_factor_multistart`](@ref): a cheap AE-onset multistart over a narrow-width-extended
`(kyhat,width)` grid, local derivative-free descents on the cheap onset surface (BOBYQA by
default), and a faithful early-stop confirm. Env knobs (kwargs override): `NLS_LOCAL_ALGO`
(`LN_BOBYQA`|`LN_SBPLX`|`LN_NELDERMEAD`, default `LN_BOBYQA`), `NLS_KDESCEND` (basins
descended, default 3), `NLS_LOCAL_EVALS` (cheap evals per descent, default 15), `NLS_NSEED_KY` /
`NLS_NSEED_W` (seed grid, defaults 6/10).
"""
function _mainsub_multistart(inputsEP::Options, inputsPR::profile, printout::Bool;
                             use_gpu::Bool = false, inner::Symbol = :threads,
                             team::Union{Nothing,AbstractVector{<:Integer}} = nothing, k_max::Int = _k_max_env())
    lalgo = Symbol(get(ENV, "NLS_LOCAL_ALGO", "LN_BOBYQA"))
    kdesc = parse(Int, get(ENV, "NLS_KDESCEND", "3"))
    lev   = parse(Int, get(ENV, "NLS_LOCAL_EVALS", "15"))
    nky   = parse(Int, get(ENV, "NLS_NSEED_KY", "6"))
    nw    = parse(Int, get(ENV, "NLS_NSEED_W", "10"))
    res = critical_factor_multistart(inputsEP, inputsPR; use_gpu = use_gpu, inner = inner, team = team, k_max = k_max,
                                     local_algo = lalgo, k_descend = kdesc, local_evals = lev,
                                     nseed_ky = nky, nseed_w = nw,
                                     scan_lo = _ad_factor_floor(Float64(inputsEP.FACTOR_IN), k_max))
    return _finalize_nls_result!(inputsEP, inputsPR, printout, res,
                                 "# TJLFEP derivative-free solver (critical_factor_multistart: $(lalgo) descents on cheap AE onset + faithful confirm)";
                                 use_gpu = use_gpu)
end

"""
    _mainsub_nlopt(inputsEP, inputsPR, printout; use_gpu=false, inner=:threads, team=nothing)

Derivative-free (`:nlopt`) branch of [`mainsub`](@ref) (PROCESS_IN==5). Runs
[`critical_factor_nlopt`](@ref): NLopt global derivative-free search on the cheap AE-onset surface
(narrow-width extended), local BOBYQA polish, then a faithful early-stop confirm. Env knobs
(kwargs override): `NLOPT_ALGO` (default `GN_DIRECT_L`), `NLOPT_LOCAL` (local polish algo, default
`LN_BOBYQA`, `none` disables), `NLOPT_MAXEVAL` (global evals, default 40).
"""
function _mainsub_nlopt(inputsEP::Options, inputsPR::profile, printout::Bool;
                        use_gpu::Bool = false, inner::Symbol = :threads,
                        team::Union{Nothing,AbstractVector{<:Integer}} = nothing, k_max::Int = _k_max_env())
    algo  = Symbol(get(ENV, "NLOPT_ALGO", "GN_DIRECT_L"))
    lstr  = get(ENV, "NLOPT_LOCAL", "LN_BOBYQA")
    lalgo = (isempty(lstr) || lstr == "none") ? nothing : Symbol(lstr)
    mev   = parse(Int, get(ENV, "NLOPT_MAXEVAL", "40"))
    res = critical_factor_nlopt(inputsEP, inputsPR; use_gpu = use_gpu, inner = inner, team = team, k_max = k_max,
                                algo = algo, local_algo = lalgo, max_evals = mev,
                                scan_lo = _ad_factor_floor(Float64(inputsEP.FACTOR_IN), k_max))
    return _finalize_nls_result!(inputsEP, inputsPR, printout, res,
                                 "# TJLFEP derivative-free solver (critical_factor_nlopt: NLopt $(algo) + faithful confirm)";
                                 use_gpu = use_gpu)
end
# ── Narrow-width search floor for the extended-width AD solvers (:ad :locate/:wide, :robust_ad, :truth) ──
# Default: the ballooning-width physics floor `width_floor()` (= WIDTH_FLOOR = 0.5; see tjlfep_modules.jl).
# Optional environment hook AD_W_LO=<float> (a constant floor), clamped so the search never goes below the
# physics floor.
function _ad_width_floor(inputsEP::Options, inputsPR::profile)
    return floor_w_lo(parse(Float64, get(ENV, "AD_W_LO", string(width_floor()))); what = "AD_W_LO")
end
