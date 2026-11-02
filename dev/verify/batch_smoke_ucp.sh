#!/bin/bash -l
# One-radius GPU smoke test for the reactor-relevant UCP_complete case (N_ION=4, IS_EP=4).
# Confirms the file-based gacode path (preprocess_gacode_inputs + run_gacode_scan_task) works
# for the 4-ion case before launching the full timing sweep. Runs a single mid-profile radius
# with SOLVER=grid, INNER=threads (no MPS team needed for a smoke), N_BASIS=6.
#   cd dev && sbatch verify/batch_smoke_ucp.sh
#SBATCH -A m3739_g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH -J ucp_smoke
#SBATCH -o ucp_smoke_%j.out
#SBATCH -e ucp_smoke_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

export TGLFEP_JULIA_FILE_ONLY=1 USE_GPU=1 TGLFEP_JULIA_DEBUG=0 TGLFEP_JULIA_PRINTOUT=1
export INNER=threads
export SOLVER="${SOLVER:-grid}"
export JULIA_CUDA_USE_COMPAT=false

NB="${NB:-6}"
SCAN_INDEX="${SCAN_INDEX:-10}"
export SCAN_INDEX

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TGLFEP_JULIA_ROOT}/examples/UCP_complete}"
export GACODE_FILE="${GACODE_FILE:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_scan20_nb${NB}.TGLFEP}"
export OUT_DIR="${TGLFEP_JULIA_ROOT}/dev/ucp_smoke_nb${NB}_${SLURM_JOB_ID}_tasks"

# Leaner file-only GPU sysimage (matches the file-based runTHD/run_gacode_scan_task path).
GPU_SYSIMG="${TGLFEP_JULIA_GPU_SYSIMAGE:-${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_gpu_sysimage.so}"
if [[ -f "${GPU_SYSIMG}" ]]; then
    export TGLFEP_JULIA_GPU_SYSIMAGE="${GPU_SYSIMG}"
    SYSIMG_ARGS=(--sysimage="${GPU_SYSIMG}")
    echo "GPU sysimage: ${GPU_SYSIMG}"
else
    SYSIMG_ARGS=()
    echo "GPU sysimage: none found at '${GPU_SYSIMG}' -> running with JIT"
fi

cd "${TGLFEP_JULIA_ROOT}/dev"
echo "=== UCP smoke: nb=${NB} scan_index=${SCAN_INDEX} solver=${SOLVER} case=${CASE_DIR} ==="
nvidia-smi -L 2>/dev/null | head -2 || true

stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    "${SYSIMG_ARGS[@]}" -t 8 ${TGLFEP_JULIA_ROOT}/slurm/common/run_gacode_scan20_mps_task.jl

echo "=== UCP smoke finished; outputs in ${OUT_DIR} ==="
