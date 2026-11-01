# Derivative-free critical-factor solvers borrowed from the FluxMatcher's nonlinear-optimizer
# playbook. These replace the AD gradients used by `critical_factor_optimize` at BOTH layers:
#
#   inner  (factor root)  : `marginal_factor_df` — bracketing ITP/Brent instead of AD-Newton
#                           (defined in tjlfep_ad_extensions.jl).
#   outer  ((ky,w) search): `critical_factor_multistart` — cheap AE-onset multistart + local
#                           derivative-free descents (BOBYQA / SBPLX / Nelder-Mead)
#                           on the CHEAP onset surface in (ky, log w) coordinates;
#                           `critical_factor_nlopt`  — NLopt derivative-free global (GN_DIRECT_L /
#                           GN_CRS2_LM) + local BOBYQA polish.
#
# Design note (why no DFSane route): DFSane is a spectral-residual solver for SQUARE nonlinear
# systems — the FluxMatcher's flux-matching problem genuinely is one. The outer TJLFEP problem is
# instead a 2D bound-constrained MINIMIZATION of a bumpy, expensive surface; recasting it as the
# stationarity system ∇f★=0 costs 3 expensive `marginal_factor_df` solves per residual, is
# attracted to saddles/maxima, and DFSane has no bound handling. Model-based (BOBYQA) or
# noise-robust simplex (SBPLX) minimizers on the ~10-20× cheaper batched AE-onset surface — the
# same surface `:grid`'s kwscale_scan ranks on — are the right tool, so the DFSane route was
# removed; use `:multistart` (local BOBYQA/SBPLX/Nelder-Mead) or `:nlopt` instead.
#
# All solvers search the width box EXTENDED below WIDTH_MIN (down to `w_lo`, default the ballooning-width
# physics floor `width_floor()` = 0.5 — never below it — like :ad :locate /
# :robust_ad) to capture the narrow EP-driven AE modes the `w≥1` box misses, then faithful-confirm
# (IFLUX=true) the candidate optima with an early-stop bound. They return the SAME contract as
# `critical_factor_robust`/`critical_factor_direct` so the `_mainsub_*` branches consume them
# uniformly: `(; sfmin, kyhat, width, binding, status, converged, faithful, n_confirm, n_samples,
# total_evals_full, total_evals_eig)` with `status ∈ (:ok, :no_onset, :cap)`.

# ── shared: cheap AE-onset (ky,w) ranking (IFLUX=false), narrow-width extended ──
# Log-spaced widths (down to w_lo) so the multistart sees the narrow band, linear kyhat.
# Returns feasible seeds `(ky, w, f1, pinned)` sorted ascending by cheap onset f1, plus eig count.
function _cheap_onset_seeds(ep0::Options{Float64}, prof::profile{Float64}, gth::Float64;
                            slo::Float64, shi::Float64, ky_lo::Float64, w_lo::Float64, w_hi::Float64,
                            nseed_ky::Int, nseed_w::Int, n_eig::Int,
                            inner::Symbol, team, use_gpu::Bool)
    kyhats = nseed_ky <= 1 ? [0.5 * (ky_lo + 1.0)] :
             [ky_lo + (1.0 - ky_lo) / (nseed_ky - 1) * (i - 1) for i in 1:nseed_ky]
    # Widths span two regimes: LOG-spaced from w_lo (resolves the narrow EP-driven AE band that
    # the `w≥1` box misses) AND LINEAR across [WIDTH_MIN, WIDTH_MAX] (the canonical box where
    # kwscale_scan finds the dense core modes — log spacing alone under-samples it). Union both so
    # the multistart brackets both the narrow-edge and wide-core basins.
    wmn = Float64(ep0.WIDTH_MIN); wmx = Float64(ep0.WIDTH_MAX)
    logw = nseed_w <= 1 ? [sqrt(w_lo * w_hi)] : collect(exp.(range(log(w_lo), log(w_hi), length = nseed_w)))
    nlin = max(2, cld(nseed_w, 2))
    linw = [wmn + (wmx - wmn) * (i - 1) / (nlin - 1) for i in 1:nlin]
    widths = sort(unique(vcat(logw, linw, wmn, wmx)))
    pts = [(ky, w) for ky in kyhats for w in widths]
    cheap = _ad_pmap(idx -> begin
            ky, w = pts[idx]; ep = deepcopy(ep0); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
            win = _ae_unstable_window(ep, prof, gth; scan_lo = slo, scan_hi = shi,
                                      n_eig = n_eig, threaded = false, use_gpu = use_gpu)
            (; ky = ky, w = w, f = (win.unstable ? win.f1 : Inf), pinned = win.pinned_lo, evals = win.evals)
        end, length(pts); inner = inner, team = team)
    eig = 0
    seeds = NamedTuple[]
    for c in cheap
        eig += c.evals
        isfinite(c.f) && push!(seeds, (; ky = c.ky, w = c.w, f = c.f, pinned = c.pinned))
    end
    sort!(seeds, by = s -> s.f)
    return (; seeds = seeds, npts = length(pts), eig = eig)
end

# ── shared: faithful (IFLUX=true) early-stop confirm of candidate (ky,w) optima ──
# `cands` :: Vector of (ky, w, cheap_f), sorted ascending by cheap_f. The cheap AE-band onset
# lower-bounds the faithful keep onset, so once a candidate's cheap_f ≥ the running faithful best
# it (and all later) cannot win — an exact prune that bounds the confirm count. Candidates within
# (`dedup_ky`, `dedup_logw`) of an already-confirmed point are skipped (local-descent samples
# cluster tightly in the winning basin; re-confirming them buys nothing), and `max_confirm` hard-
# caps the confirm count. Returns the winning faithful onset, marking, binding, the
# `marginal_factor_faithful` result, and eval tallies.
function _confirm_candidates(ep0::Options{Float64}, prof::profile{Float64}, gth::Float64,
                             cands::Vector{Tuple{Float64,Float64,Float64}};
                             slo::Float64, shi::Float64, inner::Symbol, team, use_gpu::Bool,
                             max_confirm::Int = typemax(Int),
                             dedup_ky::Float64 = 0.05, dedup_logw::Float64 = 0.1)
    best_f = Inf; best_ky = NaN; best_w = NaN; best_bind = :none; best_faithful = nothing
    n_confirm = 0; total_full = 0; total_eig = 0
    done = Tuple{Float64,Float64}[]
    for c in cands
        isfinite(c[3]) || continue
        stop_bound = isfinite(best_f) ? best_f : shi
        c[3] >= stop_bound && break
        n_confirm >= max_confirm && break
        any(d -> abs(c[1] - d[1]) < dedup_ky && abs(log(c[2]) - log(d[2])) < dedup_logw, done) && continue
        push!(done, (c[1], c[2]))
        r = marginal_factor_faithful(ep0, prof; kyhat = c[1], width = c[2], gamma_thresh = gth,
                                     scan_lo = slo, scan_hi = shi, threaded = true,
                                     inner = inner, team = team, use_gpu = use_gpu)
        n_confirm += 1; total_full += r.evals_full; total_eig += r.evals_eig
        if r.binding != :none && isfinite(r.factor_faithful) && r.factor_faithful < best_f
            best_f = r.factor_faithful; best_ky = c[1]; best_w = c[2]
            best_bind = r.binding; best_faithful = r
        end
    end
    return (; sfmin = best_f, kyhat = best_ky, width = best_w, binding = best_bind,
            faithful = best_faithful, n_confirm = n_confirm,
            total_evals_full = total_full, total_evals_eig = total_eig)
end

_nls_status(best_f, best_bind, shi) =
    (best_bind === :none || !isfinite(best_f)) ? :no_onset : (best_f >= 0.999 * shi ? :cap : :ok)

# ── shared: cheap AE-onset objective for the outer DF minimizers ──
# Returns `(; obj, samples, eig)`: `obj(ky,w)` runs one batched `_ae_unstable_window` (IFLUX=false)
# and returns the onset f1, or the `PENALTY=shi` plateau where the AE band is stable. Every
# evaluation is recorded in `samples` so the faithful confirm can consider ALL points the optimizer
# visited, not just its final iterate; `eig[]` tallies eigensolves.
function _make_cheap_objective(ep0::Options{Float64}, prof::profile{Float64}, gth::Float64;
                               slo::Float64, shi::Float64, n_eig::Int,
                               inner::Symbol, team, use_gpu::Bool)
    samples = NamedTuple[]
    eig = Ref(0)
    obj = function (ky::Float64, w::Float64)
        ep = deepcopy(ep0); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
        win = _ae_unstable_window(ep, prof, gth; scan_lo = slo, scan_hi = shi,
                                  n_eig = n_eig, threaded = true, use_gpu = use_gpu,
                                  inner = inner, team = team)
        eig[] += win.evals
        f = win.unstable ? win.f1 : shi
        push!(samples, (; ky = ky, w = w, f = f, unstable = win.unstable, pinned = win.pinned_lo))
        return f
    end
    return (; obj = obj, samples = samples, eig = eig)
end

"""
    critical_factor_multistart(inputsEP, inputsPR; local_algo=:LN_BOBYQA, kwargs...) -> NamedTuple

Derivative-free `(kyhat, width)` critical-factor search: cheap AE-onset multistart + local
derivative-free descents on the **cheap onset surface** + faithful early-stop confirm. The bumpy
onset surface `f1(ky,w)` is multimodal, so global-ness comes from the multistart
(`_cheap_onset_seeds`); the local driver only refines within a basin:

  1. Cheap eigenvalue-only (`IFLUX=false`) AE-onset rank over a (log+linear width, down to `w_lo`)
     × kyhat grid → feasible seeds sorted by onset.
  2. For the top `k_descend` well-separated non-pinned seeds, run a local derivative-free
     minimization of the cheap onset in `(ky, log w)` coordinates (log-width matches the band's
     natural scaling). `local_algo` selects the driver:
       `:LN_BOBYQA` (default) — Powell quadratic trust-region model; most eval-efficient.
       `:LN_SBPLX`            — Rowan subplex; noise-robust on the bumpiest surfaces.
       `:LN_NELDERMEAD`       — plain simplex.
     Every point the optimizer visits is recorded, and each cheap eval is ONE batched-parallel
     `_ae_unstable_window` (~10-20× cheaper than a `marginal_factor_df` f★ solve).
  3. Faithful-confirm (`IFLUX=true`) the pooled candidates — ALL feasible cheap seeds (floor
     guards) plus all unstable descent samples — in ascending cheap order with an early-stop bound
     (`_confirm_candidates`) → the critical factor.

Keywords: `local_algo=:LN_BOBYQA`, `gamma_thresh=nothing`, `scan_lo=nothing`(→`shi/(nfactor·4^(k_max-1))`; `shi/512` at k_max=4),
`scan_hi=nothing`(→`FACTOR_IN`), `ky_lo=0.05`, `w_lo=width_floor()` (0.5), `nseed_ky=6`, `nseed_w=10`,
`n_eig_seed=12`, `n_eig=16`, `k_descend=3`, `local_evals=15`, `max_confirm=8`,
`inner=:threads`, `team=nothing`, `use_gpu=false`, `verbose=false`.
The faithful confirm is the wall-time driver: `max_confirm` hard-caps it (the early-stop prune
plus basin dedup usually stop well before the cap).

Returns `(; sfmin, kyhat, width, binding, status, converged, faithful, n_confirm, n_samples,
total_evals_full, total_evals_eig)`, `status ∈ (:ok,:no_onset,:cap)`.
"""
function critical_factor_multistart(ep0::Options{Float64}, prof::profile{Float64};
                                    local_algo::Symbol = :LN_BOBYQA,
                                    gamma_thresh::Union{Nothing,Float64} = nothing,
                                    scan_lo::Union{Nothing,Float64} = nothing,
                                    scan_hi::Union{Nothing,Float64} = nothing,
                                    k_max::Int = _k_max_env(),
                                    ky_lo::Float64 = 0.05, w_lo::Float64 = width_floor(),
                                    nseed_ky::Int = 6, nseed_w::Int = 10, n_eig_seed::Int = 12,
                                    n_eig::Int = 16, k_descend::Int = 3, local_evals::Int = 15,
                                    max_confirm::Int = 8,
                                    inner::Symbol = :threads, team = nothing,
                                    use_gpu::Bool = false, verbose::Bool = false)
    gth = gamma_thresh === nothing ? _gamma_thresh_for(ep0, prof) : gamma_thresh
    shi = scan_hi === nothing ? Float64(ep0.FACTOR_IN) : scan_hi
    slo = scan_lo === nothing ? _ad_factor_floor(shi, k_max) : scan_lo
    w_lo = floor_w_lo(w_lo)   # ballooning-width physics floor (WIDTH_FLOOR = 0.5)
    kylo, kyhi = ky_lo, 1.0
    wlo, whi = w_lo, Float64(ep0.WIDTH_MAX)

    # (1) cheap AE-onset multistart seeds (narrow-width extended)
    cg = _cheap_onset_seeds(ep0, prof, gth; slo = slo, shi = shi, ky_lo = ky_lo, w_lo = w_lo,
                            w_hi = whi, nseed_ky = nseed_ky, nseed_w = nseed_w, n_eig = n_eig_seed,
                            inner = inner, team = team, use_gpu = use_gpu)
    total_eig = cg.eig
    if isempty(cg.seeds)
        return (; sfmin = Inf, kyhat = NaN, width = NaN, binding = :none, status = :no_onset,
                converged = false, faithful = nothing, n_confirm = 0, n_samples = cg.npts,
                total_evals_full = 0, total_evals_eig = total_eig)
    end

    # (2) local DF descents on the cheap onset, in u = (ky, log w) coordinates
    co = _make_cheap_objective(ep0, prof, gth; slo = slo, shi = shi, n_eig = n_eig,
                               inner = inner, team = team, use_gpu = use_gpu)
    obj_u = u -> co.obj(_clamp_to(u[1], kylo, kyhi), _clamp_to(exp(u[2]), wlo, whi))
    llo = [kylo, log(wlo)]; lhi = [kyhi, log(whi)]

    # greedily pick well-separated non-pinned seeds (distinct basins)
    picks = NamedTuple[]
    for s in cg.seeds
        s.pinned && continue
        if all(p -> abs(log(s.w) - log(p.w)) > 0.5 || abs(s.ky - p.ky) > 0.2, picks)
            push!(picks, s)
        end
        length(picks) >= k_descend && break
    end
    isempty(picks) && (picks = [cg.seeds[1]])

    for s in picks
        u0 = [s.ky, log(s.w)]
        try
            lopt = NLopt.Opt(local_algo, 2)
            NLopt.lower_bounds!(lopt, llo)
            NLopt.upper_bounds!(lopt, lhi)
            NLopt.maxeval!(lopt, local_evals)
            NLopt.initial_step!(lopt, [0.1 * (kyhi - kylo), 0.3])
            NLopt.min_objective!(lopt, (x, grad) -> obj_u(x))
            NLopt.optimize(lopt, u0)
        catch e
            e isa InterruptException && rethrow(e)
            verbose && @warn "critical_factor_multistart: $local_algo descent failed at seed ($(s.ky),$(s.w)); seed and visited samples still confirmed" exception=e
        end
    end
    total_eig += co.eig[]

    # (3) candidate pool = EVERY feasible cheap seed (floor guards, like `_locate_extended`) plus
    # every unstable point the descents visited — not just their final iterates — so the faithful
    # early-stop confirm considers all cheap basins. Without the seeds the winning basin can be
    # pruned before it is confirmed.
    cands = Tuple{Float64,Float64,Float64}[(s.ky, s.w, s.f) for s in cg.seeds]
    for p in co.samples
        p.unstable && push!(cands, (p.ky, p.w, Float64(p.f)))
    end
    sort!(cands, by = c -> c[3])

    conf = _confirm_candidates(ep0, prof, gth, cands; slo = slo, shi = shi,
                               inner = inner, team = team, use_gpu = use_gpu,
                               max_confirm = max_confirm)
    total_eig += conf.total_evals_eig
    status = _nls_status(conf.sfmin, conf.binding, shi)
    verbose && @info "critical_factor_multistart" local_algo=local_algo sfmin=conf.sfmin ky=conf.kyhat w=conf.width binding=conf.binding status=status n_confirm=conf.n_confirm
    return (; sfmin = conf.sfmin, kyhat = conf.kyhat, width = conf.width, binding = conf.binding,
            status = status, converged = (status === :ok), faithful = conf.faithful,
            n_confirm = conf.n_confirm, n_samples = cg.npts + length(co.samples),
            total_evals_full = conf.total_evals_full, total_evals_eig = total_eig)
end

"""
    critical_factor_nlopt(inputsEP, inputsPR; kwargs...) -> NamedTuple

Derivative-free `(kyhat, width)` critical-factor search using **NLopt** global derivative-free
optimization on the cheap AE-band onset surface, then a faithful early-stop confirm. This is the
core promotion of the offline `critical_factor_direct` prototype, generalized to a selectable
algorithm and an optional local polish:

  1. `NLopt` global search (`algo`, default `:GN_DIRECT_L`; also `:GN_CRS2_LM`, `:GN_AGS`, ...) over
     `(ky,w)` minimizing the cheap AE onset `f1` (`_ae_unstable_window`, `IFLUX=false`; stable points
     penalized to `FACTOR_IN`), across the narrow-width-extended box `ky∈[ky_lo,1]`, `w∈[w_lo,WIDTH_MAX]`.
  2. Local polish (`local_algo`, default `:LN_BOBYQA`; also `:LN_SBPLX`/`:LN_COBYLA`, `nothing`
     disables) seeded at the global best — DIRECT-L brackets the winning basin but converges
     slowly inside it; BOBYQA's quadratic model closes the last stretch in a handful of evals.
  3. Faithful-confirm (`IFLUX=true`) the sampled points in ascending cheap order with early stop.

Keywords: `algo=:GN_DIRECT_L`, `local_algo=:LN_BOBYQA`, `max_evals=40`, `local_evals=20`,
`gamma_thresh=nothing`, `scan_lo=nothing`, `scan_hi=nothing`, `ky_lo=0.05`, `w_lo=width_floor()` (0.5),
`n_eig=16`, `inner=:threads`, `team=nothing`, `use_gpu=false`, `verbose=false`.

Returns the same contract as [`critical_factor_multistart`](@ref).
"""
function critical_factor_nlopt(ep0::Options{Float64}, prof::profile{Float64};
                               algo::Symbol = :GN_DIRECT_L, local_algo::Union{Nothing,Symbol} = :LN_BOBYQA,
                               max_evals::Int = 40, local_evals::Int = 20,
                               gamma_thresh::Union{Nothing,Float64} = nothing,
                               scan_lo::Union{Nothing,Float64} = nothing,
                               scan_hi::Union{Nothing,Float64} = nothing,
                               k_max::Int = _k_max_env(),
                               ky_lo::Float64 = 0.05, w_lo::Float64 = width_floor(), n_eig::Int = 16,
                               inner::Symbol = :threads, team = nothing,
                               use_gpu::Bool = false, verbose::Bool = false)
    gth = gamma_thresh === nothing ? _gamma_thresh_for(ep0, prof) : gamma_thresh
    shi = scan_hi === nothing ? Float64(ep0.FACTOR_IN) : scan_hi
    slo = scan_lo === nothing ? _ad_factor_floor(shi, k_max) : scan_lo
    w_lo = floor_w_lo(w_lo)   # ballooning-width physics floor (WIDTH_FLOOR = 0.5)
    kylo, kyhi = ky_lo, 1.0
    wlo, whi = w_lo, Float64(ep0.WIDTH_MAX)
    co = _make_cheap_objective(ep0, prof, gth; slo = slo, shi = shi, n_eig = n_eig,
                               inner = inner, team = team, use_gpu = use_gpu)
    cheap_onset = co.obj
    samples = co.samples
    eig = co.eig

    # (1) global derivative-free search
    opt = NLopt.Opt(algo, 2)
    NLopt.lower_bounds!(opt, [kylo, wlo])
    NLopt.upper_bounds!(opt, [kyhi, whi])
    NLopt.maxeval!(opt, max_evals)
    NLopt.min_objective!(opt, (x, grad) -> cheap_onset(x[1], x[2]))
    (_minf, minx, ret) = NLopt.optimize(opt, [0.5 * (kylo + kyhi), sqrt(wlo * whi)])

    # (2) optional local polish seeded at the global best
    if local_algo !== nothing
        lopt = NLopt.Opt(local_algo, 2)
        NLopt.lower_bounds!(lopt, [kylo, wlo])
        NLopt.upper_bounds!(lopt, [kyhi, whi])
        NLopt.maxeval!(lopt, local_evals)
        # trust region sized to the basin, not the box (w can sit at ~0.05 near the narrow edge)
        NLopt.initial_step!(lopt, [0.05 * (kyhi - kylo), max(0.05, 0.15 * minx[2])])
        NLopt.min_objective!(lopt, (x, grad) -> cheap_onset(x[1], x[2]))
        try
            NLopt.optimize(lopt, [minx[1], minx[2]])
        catch e
            e isa InterruptException && rethrow(e)
            verbose && @warn "critical_factor_nlopt: local polish ($local_algo) failed" exception=e
        end
    end

    # (3) faithful confirm the sampled unstable points, ascending cheap order, early stop
    unstable = [(s.ky, s.w, Float64(s.f)) for s in samples if s.unstable]
    sort!(unstable, by = c -> c[3])
    conf = _confirm_candidates(ep0, prof, gth, unstable; slo = slo, shi = shi,
                               inner = inner, team = team, use_gpu = use_gpu)
    total_eig = eig[] + conf.total_evals_eig
    status = _nls_status(conf.sfmin, conf.binding, shi)
    verbose && @info "critical_factor_nlopt" algo=algo nlopt_ret=ret sfmin=conf.sfmin ky=conf.kyhat w=conf.width binding=conf.binding status=status
    return (; sfmin = conf.sfmin, kyhat = conf.kyhat, width = conf.width, binding = conf.binding,
            status = status, converged = (status === :ok), faithful = conf.faithful,
            n_confirm = conf.n_confirm, n_samples = length(samples),
            total_evals_full = conf.total_evals_full, total_evals_eig = total_eig)
end
