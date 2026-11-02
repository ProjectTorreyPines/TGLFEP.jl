#!/bin/bash -l
# UCP variant of batch_time_scan20_julia_gpu.sh: Julia GPU :grid SCAN_N=20, 5 nodes
# (4 radii/node, 1 A100/radius), MPS team + GPU sysimage, with STAGE=1 sbcast staging of the
# sysimage to node-local /tmp. Reactor-relevant UCP_complete case (N_ION=4, IS_EP=4).
#   cd dev && NB=32 sbatch timing/batch_time_scan20_julia_gpu_ucp.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 5
#SBATCH -n 20
#SBATCH -t 01:30:00
#SBATCH -C gpu
#SBATCH -J ucp_s20_jgpu
#SBATCH -o time_scan20_ucp_julia_gpu_%j.out
#SBATCH -e time_scan20_ucp_julia_gpu_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}:${PSCRATCH}/.julia:${TGLFEP_JULIA_DEPOT:-/global/cfs/cdirs/m3739/TJLFEP/depot}"

export TGLFEP_JULIA_FILE_ONLY=1 USE_GPU=1 TGLFEP_JULIA_DEBUG=0 TGLFEP_JULIA_PRINTOUT=0
export INNER=mps_team
export SOLVER=grid
export GPUS_PER_RADIUS=1
export MPS_TEAM="${MPS_TEAM:-8}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"
export JULIA_CUDA_USE_COMPAT=false
export TGLFEP_JULIA_PROBE="${TGLFEP_JULIA_PROBE:-0}"

NB="${NB:-32}"

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TGLFEP_JULIA_ROOT}/examples/UCP_complete}"
export GACODE_FILE="${GACODE_FILE:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_scan20_nb${NB}.TGLFEP}"
export OUT_DIR="${TGLFEP_JULIA_ROOT}/dev/ucp_nb${NB}_scan20_jgpu_${SLURM_JOB_ID}_tasks"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.$SLURM_JOB_ID"

# GPU sysimage: default file-only (leaner, faster worker load). Override TGLFEP_JULIA_GPU_SYSIMAGE per nb.
GPU_SYSIMG="${TGLFEP_JULIA_GPU_SYSIMAGE:-${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_gpu_sysimage.so}"
# STAGE=1: sbcast the sysimage to node-local /tmp once per node to cut the concurrent-Lustre-read tail.
STAGE="${STAGE:-1}"
if [[ "${STAGE}" == "1" && -n "${GPU_SYSIMG}" && -f "${GPU_SYSIMG}" ]]; then
    STAGED_SO="/tmp/tglfep_julia_gpusys_${SLURM_JOB_ID}.so"
    echo "STAGE=1: sbcast ${GPU_SYSIMG} -> ${STAGED_SO} (all nodes)"
    t_bcast=$(date +%s)
    if sbcast -f "${GPU_SYSIMG}" "${STAGED_SO}"; then
        GPU_SYSIMG="${STAGED_SO}"
        echo "sbcast done in $(( $(date +%s) - t_bcast )) s"
    else
        echo "sbcast failed; falling back to shared path ${GPU_SYSIMG}"
    fi
fi
if [[ -n "${GPU_SYSIMG}" && -f "${GPU_SYSIMG}" ]]; then
    export TGLFEP_JULIA_GPU_SYSIMAGE="${GPU_SYSIMG}"
    MASTER_SYSIMG_ARGS=(--sysimage="${GPU_SYSIMG}")
    echo "GPU sysimage (master+workers): ${GPU_SYSIMG}"
else
    MASTER_SYSIMG_ARGS=()
    echo "GPU sysimage: none found -> running with JIT"
fi

cd "${TGLFEP_JULIA_ROOT}/dev"
JOB_T0=$(date +%s.%N)
echo "TIMING_START backend=julia device=gpu path=gacode-mps nodes=${SLURM_NNODES:-5} tasks=${SLURM_NTASKS:-20} SCAN_N=20 N_BASIS=${NB} MPS_TEAM=${MPS_TEAM} case=ucp"
echo "OUT_DIR=${OUT_DIR}"
nvidia-smi -L 2>/dev/null | head -4 || true

T0=$(date +%s.%N)
srun --export=ALL --label -n "${SLURM_NTASKS:-20}" --ntasks-per-node=4 --cpu-bind=cores \
    ${TGLFEP_JULIA_ROOT}/slurm/common/mps-scan-wrapper.sh \
    stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" \
    -t "${JULIA_WORKER_THREADS}" ${TGLFEP_JULIA_ROOT}/slurm/common/run_gacode_scan20_mps_task.jl
T1=$(date +%s.%N)
SCAN_S=$(python3 -c "print(f'{float(\"${T1}\") - float(\"${T0}\"): .3f}')")
echo "TIMING_RESULT backend=julia device=gpu phase=scan seconds=${SCAN_S} SCAN_N=20 N_BASIS=${NB} nodes=${SLURM_NNODES:-5} tasks=${SLURM_NTASKS:-20} case=ucp"

srun --export=ALL -n "${SLURM_NNODES:-5}" --ntasks-per-node=1 \
    bash -c 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' || true

export USE_GPU=0
T0=$(date +%s.%N)
stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" -t 8 ${TGLFEP_JULIA_ROOT}/slurm/common/merge_gacode_scan20_array.jl
T1=$(date +%s.%N)
MERGE_S=$(python3 -c "print(f'{float(\"${T1}\") - float(\"${T0}\"): .3f}')")
echo "TIMING_RESULT backend=julia device=gpu phase=merge seconds=${MERGE_S} SCAN_N=20 N_BASIS=${NB} case=ucp"

JOB_T1=$(date +%s.%N)
TOTAL_S=$(python3 -c "print(f'{float(\"${JOB_T1}\") - float(\"${JOB_T0}\"): .3f}')")
echo "TIMING_RESULT backend=julia device=gpu phase=total_job seconds=${TOTAL_S} SCAN_N=20 N_BASIS=${NB} nodes=${SLURM_NNODES:-5} tasks=${SLURM_NTASKS:-20} case=ucp"
echo "=== done; outputs in ${OUT_DIR} ==="
