# Phase 3 of the FUSE-dd MPS-team SPMD layout: assemble the per-radius results (phase 2) into the
# critical gradients + EP profiles, by running the NORMAL FUSE.ActorTJLFEP. Setting
# TGLFEP_JULIA_PRECOMPUTED_DIR makes TJLFEP.runTHD(dd) load tasks/task_<i>.jls instead of re-running the
# kw-scans, so ALL of the cross-radius post-processing + ALPHA + dd update is the production actor
# path (no duplication). This step is CPU-only and cheap.
#
# Writes into TGLFEP_JULIA_OUT_DIR (consumed by run_tjlfep / load_tjlfep_results):
#   tglfep_julia_results.jls   dd_out.json

include(joinpath(@__DIR__, "fuse_dd_common.jl"))

using Serialization
import IMAS

@assert isdir(TASKS_DIR) "missing $TASKS_DIR (run per-radius tasks first)"
for i in 1:SCAN_N
    tf = joinpath(TASKS_DIR, "task_$(i).jls")
    @assert isfile(tf) "missing per-radius output $tf"
end

# Point runTHD(dd) at the precomputed per-radius results (SPMD merge mode).
ENV["TGLFEP_JULIA_PRECOMPUTED_DIR"] = TASKS_DIR

job_t0 = time()
logmsg("=== fuse-dd merge (SPMD) === CASE=$CASE SCAN_N=$SCAN_N OUT_DIR=$OUT_DIR")
logmsg("TGLFEP_JULIA_PRECOMPUTED_DIR=$TASKS_DIR")

ini, act = build_act()
dd = IMAS.json2imas(DD_IN_JSON)

ta = time()
actor = FUSE.ActorTJLFEP(dd, act)
logmsg("TIMING_RESULT path=fuse-dd-spmd phase=merge_actor seconds=$(round(time() - ta; digits=3)) SCAN_N=$SCAN_N")

logmsg("SFmin  = ", round.(actor.SFmin; digits=4))
logmsg("width  = ", round.(actor.width; digits=4))
logmsg("kymark = ", round.(actor.kymark; digits=4))
if actor.alpha !== nothing
    res = actor.alpha
    logmsg("ALPHA: n_EP[1]=", round(res.n_EP[1]; digits=4), " 10^19 m^-3   p_EP[1]=",
        round(res.p_EP[1]; digits=4), " 10 kPa   rho_grid=", length(actor.rho_grid))
end

n_EP = (actor.alpha !== nothing) ? actor.alpha.n_EP : Float64[]
p_EP = (actor.alpha !== nothing) ? actor.alpha.p_EP : Float64[]
Serialization.serialize(joinpath(OUT_DIR, "tglfep_julia_results.jls"),
    (; rho_scan=act.ActorTJLFEP.rho_scan, SFmin=actor.SFmin, width=actor.width,
        kymark=actor.kymark, n_EP=n_EP, p_EP=p_EP))
try
    IMAS.imas2json(dd, joinpath(OUT_DIR, "dd_out.json"))
catch err
    logmsg("note: imas2json(dd) failed ($(typeof(err))); skipping dd_out.json")
end
logmsg("persisted results to $OUT_DIR (tglfep_julia_results.jls + dd_out.json)")
logmsg("TIMING_RESULT path=fuse-dd-spmd phase=merge_total seconds=$(round(time() - job_t0; digits=3))")
logmsg("=== merge done ===")
