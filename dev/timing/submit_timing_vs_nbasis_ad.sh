#!/bin/bash -l
# Submit the AD (solver=:ad) timing-vs-nbasis sweep, mirroring submit_timing_vs_nbasis.sh:
#   * Julia GPU AD : 5 nodes, inner=:threads baseline + GPU sysimage (batch_time_scan20_julia_gpu_ad.sh)
#   * Julia CPU AD : 10 nodes, SlurmClusterManager + full CPU sysimage (batch_time_scan20_julia_cpu_ad.sh)
#
# GPU+threads is the FASTEST AD layout (~43 s at N_BASIS=32). The MPS-team AD variant
# (batch_time_scan20_julia_gpu_ad_mps.sh) is measured to be ~1.6-6x SLOWER for AD (the
# per-radius AD parallel regions are small and the Newton descent is sequential), so it is
# NOT submitted here; submit it manually only to refresh the plot's MPS-AD series, e.g.
#   for nb in 6 8 16 32; do NB=$nb sbatch timing/batch_time_scan20_julia_gpu_ad_mps.sh; done
#
# Output logs are named time_scan20_nb${nb}_julia_{cpu,gpu}_ad_<jobid>.out so the (extended)
# collect_scan20_timing.jl picks them up per nbasis. Job ids -> timing_runs/last_timing_vs_nbasis_ad.txt.
#
# Usage:
#   ./timing/submit_timing_vs_nbasis_ad.sh
#   NB_LIST="6 32" ./timing/submit_timing_vs_nbasis_ad.sh
#   GPU_ONLY=1 ./timing/submit_timing_vs_nbasis_ad.sh

set -uo pipefail
cd "$(dirname "$0")/.."

NB_LIST="${NB_LIST:-6 8 16 32}"
TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
CPU_SYSIMG="${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_cpu_sysimage.so"
GPU_SYSIMG="${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_gpu_generic_sysimage.so"
REC="${TGLFEP_JULIA_ROOT}/dev/timing_runs/last_timing_vs_nbasis_ad.txt"
mkdir -p "$(dirname "${REC}")"
: > "${REC}"

[[ -f "${GPU_SYSIMG}" ]] || echo "WARNING: ${GPU_SYSIMG} missing -> GPU jobs would JIT (slower)."

CPU_DEP_ARG=()
SUBMIT_CPU=1
if [[ "${GPU_ONLY:-0}" == "1" ]]; then
    SUBMIT_CPU=0
    echo "GPU_ONLY=1 -> skipping Julia CPU AD submits"
elif [[ ! -f "${CPU_SYSIMG}" ]]; then
    if [[ -n "${CPU_SYSIMAGE_BUILD_JOB:-}" ]]; then
        CPU_DEP_ARG=(--dependency=afterok:${CPU_SYSIMAGE_BUILD_JOB})
        echo "CPU sysimage not built yet -> CPU AD jobs depend on afterok:${CPU_SYSIMAGE_BUILD_JOB}"
    else
        SUBMIT_CPU=0
        echo "WARNING: ${CPU_SYSIMG} missing and no CPU_SYSIMAGE_BUILD_JOB set -> skipping CPU AD submits (would JIT)."
    fi
fi

for nb in ${NB_LIST}; do
    g=$(sbatch --parsable \
        --output="time_scan20_nb${nb}_julia_gpu_ad_%j.out" \
        --error="time_scan20_nb${nb}_julia_gpu_ad_%j.err" \
        --export=ALL,NB=${nb} timing/batch_time_scan20_julia_gpu_ad.sh)
    echo "nb=${nb} GPU-AD  job=${g}"
    echo "nb=${nb} gpu_ad ${g}" >> "${REC}"

    if [[ "${SUBMIT_CPU}" == "1" ]]; then
        c=$(sbatch --parsable "${CPU_DEP_ARG[@]}" \
            --output="time_scan20_nb${nb}_julia_cpu_ad_%j.out" \
            --error="time_scan20_nb${nb}_julia_cpu_ad_%j.err" \
            --export=ALL,NB=${nb} timing/batch_time_scan20_julia_cpu_ad.sh)
        echo "nb=${nb} CPU-AD  job=${c}"
        echo "nb=${nb} cpu_ad ${c}" >> "${REC}"
    fi
done

echo "=== recorded job ids -> ${REC} ==="
cat "${REC}"
echo "=== after runs complete (from dev/): julia --project=${TGLFEP_JULIA_ROOT} timing/collect_scan20_timing.jl && ./timing/plot_scan20_timing.sh ==="
