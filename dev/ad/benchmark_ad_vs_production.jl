# Phase 5 benchmark: AD eigenvalue-Newton onset finder vs the production
# kwscale_scan (Fortran-equivalent (kyhat x width x factor) grid, IFLUX=true).
#
# On ITER IR=83 we:
#   1. run kwscale_scan (TGLFEP_JULIA_PROBE=1, timed) → production sfmin (f_guess_mark),
#      its marked (kymark, width), the number of full IFLUX=true TJLFEP_ky evals,
#      and wall time.
#   2. run marginal_factor_faithful at that SAME marked (ky,w) → the AD-localized
#      faithful keep onset, its cheap-eigensolve and full-eval counts, wall time.
# Then compare accuracy (onset values) and cost (full-eval count, wall time).
#
#   module load julia/1.11.7 && export JULIA_DEPOT_PATH=$PSCRATCH/.julia
#   julia -t 8 --project=. dev/ad/benchmark_ad_vs_production.jl

ENV["TGLFEP_JULIA_PROBE"] = "1"

using TJLFEP
using Printf

const ITER = joinpath(@__DIR__, "..", "..", "examples", "ITER")
const IR   = 83

function iter_inputs()
    prof = TJLFEP.readMTGLF(joinpath(ITER, "input.MTGLF"))
    profile = prof[1]; ir_exp = prof[2]
    opts = TJLFEP.readTGLFEP(joinpath(ITER, "input.TGLFEP"), ir_exp)
    ni, Ti, dlnnidr, dlntidr, cs, rmin_ex, gammaE, gammap, omegaGAM =
        TJLFEP.read_expro_for_alpha(joinpath(ITER, "input.EXPRO"), profile, opts.IS_EP; gacode_file = nothing)
    expro = (ni=ni, Ti=Ti, dlnnidr=dlnnidr, dlntidr=dlntidr, cs=cs, rmin_ex=rmin_ex,
             gammaE=gammaE, gammap=gammap, omegaGAM=omegaGAM)
    TJLFEP._apply_runthd_expro_setup!(opts, profile, expro)
    opts.IR = IR
    return opts, profile
end

function main()
    @printf("threads = %d\n", Threads.nthreads())
    ep0, prof = iter_inputs()
    @printf("ITER IR=%d  N_BASIS=%d  FACTOR_IN(f1)=%.3f\n", IR, ep0.N_BASIS, ep0.FACTOR_IN)

    # warm up compile paths cheaply (one full eval + one AD eval) so the timings below
    # measure compute, not first-call JIT.
    TJLFEP.keep_at(ep0, prof, 1.0; kyhat = 0.15, width = 1.5)
    gamma_dgamma_dfactor(let e = deepcopy(ep0); e.KYHAT_IN = 0.15; e.WIDTH_IN = 1.5; e.FACTOR_IN = 1.0; e end, prof)

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
    t1 = time()
    r = marginal_factor_faithful(ep0, prof; kyhat = kymark, width = wmark,
                                 scan_lo = 1e-2, scan_hi = 2.0 * max(sf_prod, 1.0))
    t_ad = time() - t1
    ad_full = r.evals_full; ad_eig = r.evals_eig
    @printf("\n[AD marginal_factor_faithful @ (%.4f, %.4f)]\n", kymark, wmark)
    @printf("  faithful onset = %.5e   binding=%s   AE hull=(%.4e,%.4e)\n",
            r.factor_faithful, r.binding, r.window[1], r.window[2])
    @printf("  cheap eigensolves(IFLUX=false) = %d ; full evals(IFLUX=true) = %d ; wall = %.1f s\n",
            ad_eig, ad_full, t_ad)
    flush(stdout)

    # ── comparison ──
    @printf("\n[COMPARISON  ITER IR=%d]\n", IR)
    @printf("  onset:   production = %.5e   AD = %.5e   |Δ| = %.3e   rel = %.3e\n",
            sf_prod, r.factor_faithful, abs(sf_prod - r.factor_faithful),
            abs(sf_prod - r.factor_faithful) / max(abs(sf_prod), 1e-12))
    @printf("  full IFLUX=true evals:  production = %d   AD = %d   reduction = %.1fx\n",
            n_prod, ad_full, n_prod / max(ad_full, 1))
    @printf("  wall:  production = %.1f s   AD = %.1f s   speedup = %.1fx\n",
            t_prod, t_ad, t_prod / max(t_ad, 1e-6))
    flush(stdout)
end

main()
