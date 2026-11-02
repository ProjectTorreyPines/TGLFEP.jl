# Phase 5 benchmark on DIII-D (mirror of benchmark_ad_vs_production.jl for ITER).
# DIII-D is the "pinned_lo" grid-artifact case: production reports a small finite
# sfmin from its coarse factor grid, while the AD hull scan reveals the AE-band
# instability/keep onset sits at/below the scan floor. We still report the
# full-eval-count and wall-time economy of the AD path.
#
#   module load julia/1.11.7 && export JULIA_DEPOT_PATH=$PSCRATCH/.julia
#   julia -t 8 --project=. dev/ad/benchmark_ad_vs_production_diiid.jl

ENV["TGLFEP_JULIA_PROBE"] = "1"

using TJLFEP
using Printf

const CASE = joinpath(@__DIR__, "..", "..", "examples", "DIIID_202017C42_500ms_v3.1")
const IR   = 38

function main()
    @printf("threads = %d\n", Threads.nthreads())
    ep0, prof, _ = preprocess_gacode_inputs(joinpath(CASE, "input.gacode"), joinpath(CASE, "input.TGLFEP"))
    ep0.IR = IR
    @printf("DIII-D IR=%d  N_BASIS=%d  FACTOR_IN(f1)=%.3f\n", IR, ep0.N_BASIS, ep0.FACTOR_IN)

    # warm up compile paths
    TJLFEP.keep_at(ep0, prof, 0.02; kyhat = 0.25, width = 1.5)
    gamma_dgamma_dfactor(let e = deepcopy(ep0); e.KYHAT_IN = 0.25; e.WIDTH_IN = 1.5; e.FACTOR_IN = 0.02; e end, prof)

    # ── 1. production kwscale_scan ──
    TJLFEP._probe_reset!()
    epp = deepcopy(ep0)
    t0 = time()
    _, epp, _, _, _, _ = TJLFEP.kwscale_scan(epp, prof, false; inner = :threads)
    t_prod = time() - t0
    n_prod = TJLFEP._PROBE_N[]
    sf_prod = epp.FACTOR_IN
    kymark = epp.KYMARK
    wmark  = epp.WIDTH_IN
    @printf("\n[PRODUCTION kwscale_scan]\n")
    @printf("  sfmin (f_guess_mark) = %.5e   at (kymark=%.4f, width=%.4f)\n", sf_prod, kymark, wmark)
    @printf("  full IFLUX=true evals = %d   wall = %.1f s\n", n_prod, t_prod)
    flush(stdout)

    # ── 2. AD faithful onset at the production-marked (ky,w) ──
    shi = max(0.3, 4.0 * sf_prod)
    t1 = time()
    r = marginal_factor_faithful(ep0, prof; kyhat = kymark, width = wmark, scan_lo = 1e-4, scan_hi = shi)
    t_ad = time() - t1
    @printf("\n[AD marginal_factor_faithful @ (%.4f, %.4f), scan_hi=%.3f]\n", kymark, wmark, shi)
    @printf("  faithful onset = %.5e   binding=%s   pinned_lo=%s   AE hull=(%.4e,%.4e)\n",
            r.factor_faithful, r.binding, r.pinned_lo, r.window[1], r.window[2])
    @printf("  cheap eigensolves(IFLUX=false) = %d ; full evals(IFLUX=true) = %d ; wall = %.1f s\n",
            r.evals_eig, r.evals_full, t_ad)
    flush(stdout)

    # ── comparison ──
    @printf("\n[COMPARISON  DIII-D IR=%d]\n", IR)
    @printf("  onset:   production = %.5e   AD = %.5e   pinned_lo=%s\n", sf_prod, r.factor_faithful, r.pinned_lo)
    @printf("  full IFLUX=true evals:  production = %d   AD = %d   reduction = %.1fx\n",
            n_prod, r.evals_full, n_prod / max(r.evals_full, 1))
    @printf("  wall:  production = %.1f s   AD = %.1f s   speedup = %.1fx\n",
            t_prod, t_ad, t_prod / max(t_ad, 1e-6))
    flush(stdout)
end

main()
