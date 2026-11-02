# End-to-end validation of the solver=:grid vs solver=:ad toggle through the
# TJLFEP driver on DIII-D. This is the exact code path the FUSE ActorTJLFEP
# reaches (ActorTJLFEP -> TJLFEP.runTHD(dd) -> _dd_radius_output -> mainsub),
# minus IMAS preprocessing/ALPHA integration (unchanged); the runTHD(dd) and the
# file-path runTHD share the same mainsub(:grid|:ad) branch + driver plumbing.
#
# Two checks:
#   1. Single-radius IR=38: mainsub(:grid) vs mainsub(:ad) agreement (sfmin, ky, w).
#   2. Full multi-radius AD driver: runTHD_from_gacode(...; solver=:ad) end-to-end,
#      exercising _runTHD_core! -> _runTHD_radius! -> mainsub(:ad) over all radii.
#
#   julia -t 64 --project=. dev/ad/validate_solver_toggle.jl

ENV["TGLFEP_JULIA_PROBE"] = "1"

using TJLFEP
using Printf

const CASE   = joinpath(@__DIR__, "..", "..", "examples", "DIIID_202017C42_500ms_v3.1")
const GACODE = joinpath(CASE, "input.gacode")
const TGLFEP = joinpath(CASE, "input.TGLFEP")
const IR     = 38

function main()
    @printf("threads = %d\n", Threads.nthreads())
    ep0, prof, _ = preprocess_gacode_inputs(GACODE, TGLFEP)
    @printf("DIII-D  N_BASIS=%d  SCAN_N=%d  IR_EXP=%s\n",
            ep0.N_BASIS, ep0.SCAN_N, string(Int.(ep0.IR_EXP[1:ep0.SCAN_N])))
    flush(stdout)

    # warm compile paths
    TJLFEP.keep_at(ep0, prof, 0.02; kyhat = 0.25, width = 1.5)
    gamma_dgamma_dfactor(let e = deepcopy(ep0); e.KYHAT_IN = 0.25; e.WIDTH_IN = 1.5; e.FACTOR_IN = 0.02; e end, prof)

    # ── 1. single-radius grid vs ad agreement at IR=38 ──
    eg = deepcopy(ep0); eg.IR = IR; eg.PROCESS_IN = 5
    t = time(); (_, epg, _, _), _ = TJLFEP.mainsub(eg, prof, false; solver = :grid); tg = time() - t

    ea = deepcopy(ep0); ea.IR = IR; ea.PROCESS_IN = 5
    t = time(); (_, epa, _, _), (sfb, _) = TJLFEP.mainsub(ea, prof, true; solver = :ad); ta = time() - t

    @printf("\n[mainsub IR=%d]\n", IR)
    @printf("  grid: sfmin=%.5e  ky=%.4f  w=%.4f   wall=%.1fs\n", epg.FACTOR_IN, epg.KYMARK, epg.WIDTH_IN, tg)
    @printf("  ad  : sfmin=%.5e  ky=%.4f  w=%.4f   wall=%.1fs\n", epa.FACTOR_IN, epa.KYMARK, epa.WIDTH_IN, ta)
    @printf("  |Δsfmin|/grid = %.3e   wall speedup = %.1fx\n",
            abs(epg.FACTOR_IN - epa.FACTOR_IN) / max(abs(epg.FACTOR_IN), 1e-12), tg / max(ta, 1e-6))
    println("  AD sf_buf:"); foreach(l -> println("    ", l), sfb)
    flush(stdout)

    # ── 2. full multi-radius AD driver end-to-end (the actor's call path) ──
    t = time()
    width, kymark, SFmin, dpdr, dndr =
        runTHD_from_gacode(GACODE, TGLFEP; solver = :ad, parallel = :threads, printout = false)
    tdrv = time() - t
    @printf("\n[runTHD_from_gacode solver=:ad]  wall=%.1fs\n", tdrv)
    @printf("  %-3s  %-12s  %-9s  %-9s  %-12s  %-12s\n", "i", "SFmin", "width", "kymark", "dpdr_crit", "dndr_crit")
    for i in eachindex(SFmin)
        @printf("  %-3d  %-12.5e  %-9.4f  %-9.4f  %-12.4e  %-12.4e\n",
                i, SFmin[i], width[i], kymark[i], dpdr[i], dndr[i])
    end
    @printf("\nfinite SFmin: %d/%d\n", count(isfinite, SFmin), length(SFmin))
    flush(stdout)
end

main()
