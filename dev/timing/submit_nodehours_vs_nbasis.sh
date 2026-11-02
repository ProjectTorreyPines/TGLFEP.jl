#!/bin/bash -l
# Submit the SINGLE-NODE BACKFILL (batch_nndb_scan20_1node.sh, BACKFILL_MODE=1, nodes=1) timing sweep
# over nbasis for one SOLVER. This is the node-hours-minimal layout (4 GPU-workers drain the 20-radius
# claim queue on ONE node, reusing their MPS team), so the resulting TIMING_RESULT lines carry
# nodes=1 and feed the node-hours-vs-nbasis plot directly.
#
# Output logs are named to match the collector's per-solver series globs so they flow into the same
# columns as the 5-node wave runs (the newest job id wins, so a backfill run supersedes the 5-node
# number for that solver/nbasis):
#   grid       -> time_scan20_nb${nb}_julia_gpu_${jobid}.out
#   robust_ad  -> time_scan20_nb${nb}_julia_gpu_robust_ad_${jobid}.out
#   truth      -> time_scan20_nb${nb}_julia_gpu_truth_${jobid}.out
# Job ids -> timing_runs/last_nodehours_vs_nbasis_${SOLVER}.txt
#
# Usage:
#   SOLVER=robust_ad ./timing/submit_nodehours_vs_nbasis.sh
#   SOLVER=truth NB_LIST="6 32" ./timing/submit_nodehours_vs_nbasis.sh
#   SOLVER=robust_ad QOS=regular ./timing/submit_nodehours_vs_nbasis.sh
#   GPU_SYSIMAGE_BUILD_JOB=<id> SOLVER=robust_ad ./timing/submit_nodehours_vs_nbasis.sh

set -uo pipefail
cd "$(dirname "$0")/.."

SOLVER="${SOLVER:-robust_ad}"
NB_LIST="${NB_LIST:-6 8 16 32}"
TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
GPU_SYSIMG="${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_gpu_generic_sysimage.so"
REC="${TGLFEP_JULIA_ROOT}/dev/timing_runs/last_nodehours_vs_nbasis_${SOLVER}.txt"
mkdir -p "$(dirname "${REC}")"
: > "${REC}"

# Map solver -> collector filename tag (must match collect_scan20_timing.jl globs).
case "${SOLVER}" in
    grid)      TAG="julia_gpu" ;;
    robust_ad) TAG="julia_gpu_robust_ad" ;;
    truth)     TAG="julia_gpu_truth" ;;
    ad)        TAG="julia_gpu_ad" ;;
    *) echo "ERROR: unknown SOLVER='${SOLVER}' (want grid|robust_ad|truth|ad)"; exit 1 ;;
esac

DEP_ARG=()
if [[ -n "${GPU_SYSIMAGE_BUILD_JOB:-}" ]]; then
    DEP_ARG=(--dependency=afterok:${GPU_SYSIMAGE_BUILD_JOB})
    echo "${SOLVER} 1-node jobs depend on afterok:${GPU_SYSIMAGE_BUILD_JOB} (sysimage build)"
elif [[ ! -f "${GPU_SYSIMG}" ]]; then
    echo "WARNING: ${GPU_SYSIMG} missing and no GPU_SYSIMAGE_BUILD_JOB set -> jobs would JIT (much slower)."
fi

# These single-node jobs are not urgent; default to regular QoS to avoid the premium 5-job cap.
QOS_ARG=()
QOS="${QOS:-regular}"
[[ -n "${QOS}" ]] && QOS_ARG=(-q "${QOS}")
echo "${SOLVER} node-hours timing jobs -> QoS=${QOS:-<batch default>}  tag=${TAG}"

for nb in ${NB_LIST}; do
    g=$(sbatch --parsable "${DEP_ARG[@]}" "${QOS_ARG[@]}" \
        --output="time_scan20_nb${nb}_${TAG}_%j.out" \
        --error="time_scan20_nb${nb}_${TAG}_%j.err" \
        --export=ALL,NB=${nb},SOLVER=${SOLVER} timing/batch_nndb_scan20_1node.sh)
    echo "nb=${nb} 1NODE-${SOLVER}  job=${g}"
    echo "nb=${nb} ${TAG} ${g}" >> "${REC}"
done

echo "=== recorded job ids -> ${REC} ==="
cat "${REC}"
echo "=== after runs complete (from dev/): julia --project=${TGLFEP_JULIA_ROOT} timing/collect_scan20_timing.jl && ./timing/plot_scan20_timing.sh ==="
