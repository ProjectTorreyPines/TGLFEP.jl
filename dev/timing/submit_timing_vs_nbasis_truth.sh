#!/bin/bash -l
# Submit the PHYSICAL-TRUTH (solver=:truth) timing-vs-nbasis sweep on 5 GPU nodes
# (batch_time_scan20_julia_gpu_truth.sh, inner=:mps_team, MPS_TEAM=8). This is the production
# min(robust_ad, truth) profile: per radius it runs the extended-width (ky,w) locate + separable
# nbasis convergence + a refined-faithful w>=1 grid-zoom FLOOR, so it dominates both the grid and
# the robust_ad profiles everywhere. NB sets the WORKING basis; nb_steps climb {NB,NB+8,NB+16}.
#
# Output logs are named time_scan20_nb${nb}_julia_gpu_truth_<jobid>.out so the (extended)
# collect_scan20_timing.jl picks them up per nbasis (the julia_gpu_truth_s column / "Julia truth
# MPS" plot series). Job ids -> timing_runs/last_timing_vs_nbasis_truth.txt.
#
# The truth path REQUIRES the freshly-rebuilt GPU sysimage (it bakes solver=:truth +
# critical_factor_robust). If the sysimage is still building, pass its job id so the truth jobs
# wait for it:
#   GPU_SYSIMAGE_BUILD_JOB=54584234 ./timing/submit_timing_vs_nbasis_truth.sh
#
# Usage:
#   ./timing/submit_timing_vs_nbasis_truth.sh
#   NB_LIST="6 32" ./timing/submit_timing_vs_nbasis_truth.sh
#   GPU_SYSIMAGE_BUILD_JOB=<id> ./timing/submit_timing_vs_nbasis_truth.sh

set -uo pipefail
cd "$(dirname "$0")/.."

NB_LIST="${NB_LIST:-6 8 16 32}"
TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
GPU_SYSIMG="${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_gpu_generic_sysimage.so"
REC="${TGLFEP_JULIA_ROOT}/dev/timing_runs/last_timing_vs_nbasis_truth.txt"
mkdir -p "$(dirname "${REC}")"
: > "${REC}"

DEP_ARG=()
if [[ -n "${GPU_SYSIMAGE_BUILD_JOB:-}" ]]; then
    DEP_ARG=(--dependency=afterok:${GPU_SYSIMAGE_BUILD_JOB})
    echo "truth jobs depend on afterok:${GPU_SYSIMAGE_BUILD_JOB} (sysimage build)"
elif [[ ! -f "${GPU_SYSIMG}" ]]; then
    echo "WARNING: ${GPU_SYSIMG} missing and no GPU_SYSIMAGE_BUILD_JOB set -> truth jobs would JIT (much slower)."
fi

for nb in ${NB_LIST}; do
    g=$(sbatch --parsable "${DEP_ARG[@]}" \
        --output="time_scan20_nb${nb}_julia_gpu_truth_%j.out" \
        --error="time_scan20_nb${nb}_julia_gpu_truth_%j.err" \
        --export=ALL,NB=${nb} timing/batch_time_scan20_julia_gpu_truth.sh)
    echo "nb=${nb} GPU-TRUTH  job=${g}"
    echo "nb=${nb} gpu_truth ${g}" >> "${REC}"
done

echo "=== recorded job ids -> ${REC} ==="
cat "${REC}"
echo "=== after runs complete (from dev/): julia --project=${TGLFEP_JULIA_ROOT} timing/collect_scan20_timing.jl && ./timing/plot_scan20_timing.sh ==="
