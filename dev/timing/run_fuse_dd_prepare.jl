# Phase 1 of the FUSE-dd MPS-team SPMD layout: build the `dd` ONCE on the master and serialize
# everything the per-radius tasks (phase 2) and the merge (phase 3) need. Doing this once avoids
# 20x FUSE.init (~218 s each).
#
# Writes into TGLFEP_JULIA_OUT_DIR:
#   dd_in.json      -- the initialized ITER dd (IMAS json)
#   optionsdict.jls -- the TGLF-EP OptionsDict (identical to what the phase-3 actor rebuilds)
#   rho_scan.jls    -- the SCAN_N rho_tor_norm scan grid
#
# Env knobs: see fuse_dd_common.jl (CASE / SCAN_N / N_BASIS / NGRID / ALPHA_SOLVER / INNER / MPS_TEAM).

include(joinpath(@__DIR__, "fuse_dd_common.jl"))

using Serialization
import IMAS

job_t0 = time()
logmsg("=== fuse-dd prepare (SPMD) === CASE=$CASE SCAN_N=$SCAN_N N_BASIS=$N_BASIS NGRID=$NGRID ",
    "INNER=$INNER MPS_TEAM=$MPS_TEAM OUT_DIR=$OUT_DIR")
mkpath(OUT_DIR)
mkpath(TASKS_DIR)

td = time()
ini, act = build_act()
dd = IMAS.dd()
FUSE.init(dd, ini, act)
logmsg("TIMING_RESULT path=fuse-dd-spmd phase=dd_build seconds=$(round(time() - td; digits=3)) NGRID=$NGRID")

OptionsDict, rho_scan = optionsdict_and_rho(act)
logmsg("rho_scan = ", rho_scan)

IMAS.imas2json(dd, DD_IN_JSON)
Serialization.serialize(OPTIONSDICT_JLS, OptionsDict)
Serialization.serialize(RHOSCAN_JLS, rho_scan)
logmsg("wrote $DD_IN_JSON")
logmsg("wrote $OPTIONSDICT_JLS  (SCAN_N=$(OptionsDict["SCAN_N"]))")
logmsg("wrote $RHOSCAN_JLS")
logmsg("TIMING_RESULT path=fuse-dd-spmd phase=prepare_total seconds=$(round(time() - job_t0; digits=3))")
logmsg("=== prepare done ===")
