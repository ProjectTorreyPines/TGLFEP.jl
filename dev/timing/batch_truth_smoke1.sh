#!/bin/bash -l
# 1-GPU plumbing smoke for SOLVER=truth: validates run_gacode_scan_task -> mainsub -> _mainsub_truth
# -> critical_factor_truth end-to-end under TGLFEP_JULIA_FILE_ONLY + mps_team on a real GPU, before the
# 20-GPU production profile. One radius (SCAN_INDEX), nb6 input (truth still sweeps nb=32/40/48).
#   cd dev && SCAN_INDEX=10 sbatch timing/batch_truth_smoke1.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH -J truth_smoke1
#SBATCH -o truth_smoke1_%j.out
#SBATCH -e truth_smoke1_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

export TGLFEP_JULIA_FILE_ONLY=1 USE_GPU=1 TGLFEP_JULIA_DEBUG=0 TGLFEP_JULIA_PRINTOUT=1
export INNER="${INNER:-mps_team}"
export MPS_TEAM="${MPS_TEAM:-4}"
export SOLVER=truth
export GPUS_PER_RADIUS=1
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"
export JULIA_CUDA_USE_COMPAT=false
export SCAN_INDEX="${SCAN_INDEX:-10}"

NB="${NB:-6}"
TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TGLFEP_JULIA_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"
export GACODE_FILE="${GACODE_FILE:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_scan20_nb32.TGLFEP}"
export OUT_DIR="${TGLFEP_JULIA_ROOT}/dev/truth_smoke1_${SLURM_JOB_ID}_tasks"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.$SLURM_JOB_ID"
export TGLFEP_JULIA_GPU_SYSIMAGE="/nonexistent/force-jit"

cd "${TGLFEP_JULIA_ROOT}/dev"
echo "=== truth plumbing smoke: SCAN_INDEX=${SCAN_INDEX} SOLVER=truth INNER=${INNER} ==="
nvidia-smi -L 2>/dev/null | head -1 || true

srun --export=ALL --label -n 1 --ntasks-per-node=1 --cpu-bind=cores \
    ${TGLFEP_JULIA_ROOT}/slurm/common/mps-scan-wrapper.sh \
    stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    -t "${JULIA_WORKER_THREADS}" ${TGLFEP_JULIA_ROOT}/slurm/common/run_gacode_scan20_mps_task.jl

srun --export=ALL -n 1 --ntasks-per-node=1 \
    bash -c 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' || true
echo "=== truth smoke done; check out.scalefactor in ${OUT_DIR} ==="
