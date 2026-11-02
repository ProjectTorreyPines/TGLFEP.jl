#!/bin/bash -l
# Submit the ROBUST_AD (solver=:robust_ad, extend_width=true) timing-vs-nbasis sweep on 5 GPU nodes
# (batch_time_scan20_julia_gpu_robust_ad.sh, inner=:mps_team, MPS_TEAM=8). This is the WIDTH tier of
# the grid -> robust_ad -> truth ladder: per radius it runs the canonical w>=1 grid-zoom + the
# extended narrow-width locate at a single working basis (NB), WITHOUT the truth nbasis ladder. So
# it measures the cost of the width extension alone vs the full :truth tier.
#
# Output logs are named time_scan20_nb${nb}_julia_gpu_robust_ad_<jobid>.out so the (extended)
# collect_scan20_timing.jl picks them up per nbasis (the julia_gpu_robust_ad_s column / "Julia
# robust_ad MPS" plot series). Job ids -> timing_runs/last_timing_vs_nbasis_robust_ad.txt.
#
# The width-extended robust_ad path REQUIRES the freshly-rebuilt GPU sysimage. If it is still
# building, pass its job id so the jobs wait for it:
#   GPU_SYSIMAGE_BUILD_JOB=54629366 ./timing/submit_timing_vs_nbasis_robust_ad.sh
#
# Usage:
#   ./timing/submit_timing_vs_nbasis_robust_ad.sh
#   NB_LIST="6 32" ./timing/submit_timing_vs_nbasis_robust_ad.sh
#   GPU_SYSIMAGE_BUILD_JOB=<id> ./timing/submit_timing_vs_nbasis_robust_ad.sh

set -uo pipefail
cd "$(dirname "$0")/.."

NB_LIST="${NB_LIST:-6 8 16 32}"
TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
GPU_SYSIMG="${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_gpu_generic_sysimage.so"
REC="${TGLFEP_JULIA_ROOT}/dev/timing_runs/last_timing_vs_nbasis_robust_ad.txt"
mkdir -p "$(dirname "${REC}")"
: > "${REC}"

DEP_ARG=()
if [[ -n "${GPU_SYSIMAGE_BUILD_JOB:-}" ]]; then
    DEP_ARG=(--dependency=afterok:${GPU_SYSIMAGE_BUILD_JOB})
    echo "robust_ad jobs depend on afterok:${GPU_SYSIMAGE_BUILD_JOB} (sysimage build)"
elif [[ ! -f "${GPU_SYSIMG}" ]]; then
    echo "WARNING: ${GPU_SYSIMG} missing and no GPU_SYSIMAGE_BUILD_JOB set -> jobs would JIT (much slower)."
fi

# The premium GPU QoS caps submitted jobs at 5/user. These timing runs are not urgent, so route
# them to `regular` by default (QOS=regular) to avoid contending with premium scan/build jobs.
QOS_ARG=()
QOS="${QOS:-regular}"
[[ -n "${QOS}" ]] && QOS_ARG=(-q "${QOS}")
echo "robust_ad timing jobs -> QoS=${QOS:-<batch default>}"

for nb in ${NB_LIST}; do
    g=$(sbatch --parsable "${DEP_ARG[@]}" "${QOS_ARG[@]}" \
        --output="time_scan20_nb${nb}_julia_gpu_robust_ad_%j.out" \
        --error="time_scan20_nb${nb}_julia_gpu_robust_ad_%j.err" \
        --export=ALL,NB=${nb} timing/batch_time_scan20_julia_gpu_robust_ad.sh)
    echo "nb=${nb} GPU-ROBUST_AD  job=${g}"
    echo "nb=${nb} gpu_robust_ad ${g}" >> "${REC}"
done

echo "=== recorded job ids -> ${REC} ==="
cat "${REC}"
echo "=== after runs complete (from dev/): julia --project=${TGLFEP_JULIA_ROOT} timing/collect_scan20_timing.jl && ./timing/plot_scan20_timing.sh ==="
