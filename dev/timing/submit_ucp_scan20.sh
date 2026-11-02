#!/bin/bash -l
# Orchestrate the full UCP_complete SCAN_N=20 timing sweep (reactor-relevant, N_ION=4).
# Series (per N_BASIS in NB_LIST):
#   Fortran CPU -n 1280            (premium CPU)   -> batch_time_scan20_fortran_ucp.sh
#   Julia GPU :grid       (5N MPS) (premium GPU)   -> batch_time_scan20_julia_gpu_ucp.sh
#   Julia GPU :ad :only   (5N thr) (regular GPU)   -> batch_time_scan20_julia_gpu_ad_only_ucp.sh
#   Julia GPU :ad :locate (1N bf)  (regular GPU)   -> batch_nndb_scan20_1node_ucp.sh (AD_EXTEND_MODE=locate)
#   Julia GPU :ad :wide   (1N bf)  (regular GPU)   -> batch_nndb_scan20_1node_ucp.sh (AD_EXTEND_MODE=wide)
#   Julia CPU :grid       (10N)    (regular CPU)   -> batch_time_scan20_julia_cpu_ucp.sh
#   Julia CPU :ad         (10N)    (regular CPU)   -> batch_time_scan20_julia_cpu_ad_ucp.sh
#
# QoS routing respects the premium 5-job submit cap (separate for CPU/GPU): the headline
# reference (Fortran) and headline GPU (:grid) go premium (<=5 each), everything else goes
# regular (no cap; measured node-hours are compute-based, unaffected by QoS). GPU sysimage:
# file-only for nb 8/16/32, generic for nb6 (cold-load I/O), matching the DIII-D methodology.
#
#   cd dev && ./timing/submit_ucp_scan20.sh
#   NB_LIST="6 32" ./timing/submit_ucp_scan20.sh
#   SERIES="fortran gpu_grid" ./timing/submit_ucp_scan20.sh   # subset

set -uo pipefail
cd "$(dirname "$0")/.."   # dev/

NB_LIST="${NB_LIST:-6 8 16 32}"
SERIES="${SERIES:-fortran gpu_grid gpu_ad_only gpu_ad_locate gpu_ad_wide cpu_grid}"
TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
GEN_IMG="${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_gpu_generic_sysimage.so"
FILE_IMG="${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_gpu_sysimage.so"
REC="${TGLFEP_JULIA_ROOT}/dev/timing_runs/last_ucp_scan20.txt"
mkdir -p "$(dirname "${REC}")"
: > "${REC}"

gpu_img() { [[ "$1" == "6" ]] && echo "${GEN_IMG}" || echo "${FILE_IMG}"; }
has() { [[ " ${SERIES} " == *" $1 "* ]]; }
submit() { # label qos batch_script extra_export...
    local label="$1" qos="$2" script="$3"; shift 3
    local out="time_scan20_ucp_nb${nb}_${label}_%j.out"
    local err="time_scan20_ucp_nb${nb}_${label}_%j.err"
    local exp="ALL,NB=${nb}"
    for kv in "$@"; do exp="${exp},${kv}"; done
    local j
    j=$(sbatch --parsable -q "${qos}" --output="${out}" --error="${err}" \
        --export="${exp}" "timing/${script}")
    echo "nb=${nb} ${label} (${qos}) job=${j}"
    echo "nb=${nb} ${label} ${qos} ${j}" >> "${REC}"
}

for nb in ${NB_LIST}; do
    IMG="$(gpu_img "${nb}")"
    has fortran       && submit fortran        premium batch_time_scan20_fortran_ucp.sh
    has gpu_grid      && submit julia_gpu       premium batch_time_scan20_julia_gpu_ucp.sh          "TGLFEP_JULIA_GPU_SYSIMAGE=${IMG}"
    has gpu_ad_only   && submit julia_gpu_ad_only  regular batch_time_scan20_julia_gpu_ad_only_ucp.sh "TGLFEP_JULIA_GPU_SYSIMAGE=${IMG}"
    has gpu_ad_locate && submit julia_gpu_ad_locate regular batch_nndb_scan20_1node_ucp.sh "TGLFEP_JULIA_GPU_SYSIMAGE=${IMG}" "SOLVER=ad" "AD_EXTEND_MODE=locate"
    has gpu_ad_wide   && submit julia_gpu_ad_wide   regular batch_nndb_scan20_1node_ucp.sh "TGLFEP_JULIA_GPU_SYSIMAGE=${IMG}" "SOLVER=ad" "AD_EXTEND_MODE=wide"
    has cpu_grid      && submit julia_cpu       regular batch_time_scan20_julia_cpu_ucp.sh
done

echo "=== recorded job ids -> ${REC} ==="
cat "${REC}"
echo "=== after runs: julia --project=${TGLFEP_JULIA_ROOT} timing/collect_ucp_timing.jl && julia --project=${TGLFEP_JULIA_ROOT} timing/plot_ucp_timing.jl && julia --project=${TGLFEP_JULIA_ROOT} ad/plot_ucp_accuracy_nb32.jl ==="
