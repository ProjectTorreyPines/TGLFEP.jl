# Quick end-to-end smoke test of the full FUSE chain for the AD width-aware solver:
#   FUSE.ActorTJLFEP(solver=:ad, extend_mode=:wide) -> TJLFEP.runTHD(dd) -> mainsub ->
#   critical_factor_optimize(extend_width=true, extend_mode=:wide).
#
# Deliberately tiny (2 radii, N_BASIS=2) -- this only confirms the chain compiles, the new
# extend_mode/wide_kdesc/faithful_confirm kwargs flow through, and SFmin comes back finite.
# NOT an accuracy/timing run.

using Pkg
const TGLFEP_JULIA_ROOT = get(ENV, "TGLFEP_JULIA_ROOT", normpath(@__DIR__, "..", ".."))
const FUSE_ROOT = get(ENV, "FUSE_ROOT", normpath(TGLFEP_JULIA_ROOT, "..", "FUSE"))
Pkg.activate(FUSE_ROOT)
push!(LOAD_PATH, TGLFEP_JULIA_ROOT)

logmsg(args...) = (println(args...); flush(stdout); flush(stderr))

const USE_GPU = get(ENV, "USE_GPU", "1") == "1"
const SOLVER = Symbol(get(ENV, "SOLVER", "ad"))
const EXTEND_MODE = Symbol(get(ENV, "AD_EXTEND_MODE", "wide"))
const WIDE_KDESC = parse(Int, get(ENV, "AD_WIDE_KDESC", "2"))

import FUSE
import IMAS

# This driver runs the in-process (master) pmap path (inner=:threads), so the eigensolve happens
# on THIS process -- we must `using CUDA` here to trigger TJLF's GPU extension (which populates the
# _CUDA_SOLVE refs). The distributed production scripts instead load CUDA on the GPU workers.
if USE_GPU
    using CUDA
    CUDA.functional() && CUDA.device!(0)
    logmsg("CUDA.functional() = ", CUDA.functional())
end

logmsg("=== smoke: ActorTJLFEP solver=$SOLVER extend_mode=$EXTEND_MODE wide_kdesc=$WIDE_KDESC use_gpu=$USE_GPU ===")

t0 = time()
ini, act = FUSE.case_parameters(:ITER; init_from=:ods)
ini.core_profiles.ngrid = 51   # coarse grid maps rho=0.7 onto the fragile edge index (ir=51) -> exercises the singular-matrix guard
act.ActorTJLFEP.rho_scan = [0.5, 0.7]
act.ActorTJLFEP.n_basis = 8   # nb=2 makes TJLF's h_ratios matrix singular on some :wide seeds; 8 is the smallest validated value
act.ActorTJLFEP.use_gpu = USE_GPU
act.ActorTJLFEP.solver = SOLVER
act.ActorTJLFEP.extend_mode = EXTEND_MODE
act.ActorTJLFEP.wide_kdesc = WIDE_KDESC

dd = IMAS.dd()
FUSE.init(dd, ini, act)
logmsg("dd init OK in $(round(time() - t0; digits=1)) s; rho_scan = ", act.ActorTJLFEP.rho_scan)

ta = time()
actor = FUSE.ActorTJLFEP(dd, act)
logmsg("actor ran in $(round(time() - ta; digits=1)) s")
logmsg("SFmin = ", actor.SFmin)
logmsg("width = ", actor.width)
logmsg("kymark = ", actor.kymark)

@assert length(actor.SFmin) == 2 "expected 2 radii, got $(length(actor.SFmin))"
@assert all(x -> isfinite(x) || x >= 9000, actor.SFmin) "SFmin has non-finite, non-sentinel entries: $(actor.SFmin)"
logmsg("SMOKE_OK solver=$SOLVER extend_mode=$EXTEND_MODE SFmin=$(actor.SFmin)")
