#!/bin/bash -l
# SINGLE-NODE BACKFILL layout (the NN-database-generation strategy): 1 node, 4 A100s, 4 GPU-worker
# tasks (G=1 -> one GPU each) that share a directory-based ATOMIC claim queue (BACKFILL_MODE=1) over
# all SCAN_N radii. Each worker spawns its MPS team ONCE and then drains the queue, reusing the team
# across every radius it claims, so the team spawn/JIT is paid once per GPU and there is no per-node
# tail until the very last radius. This trades ~SCAN_N/4 waves of wallclock for using only 1 node,
# which is the throughput-optimal (node-hours-minimal) layout for bulk DB generation and the basis
# for the node-hours-vs-nbasis timing plot (nodes=1).
#
# Sweep basis via NB; pick engine via SOLVER (grid|robust_ad|truth); fill each GPU via MPS_TEAM:
#   cd dev && NB=32 SOLVER=robust_ad MPS_TEAM=8 sbatch timing/batch_nndb_scan20_1node.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -J nndb_s20_1node
#SBATCH -o nndb_scan20_1node_%j.out
#SBATCH -e nndb_scan20_1node_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

export TGLFEP_JULIA_FILE_ONLY=1 USE_GPU=1 TGLFEP_JULIA_DEBUG=0 TGLFEP_JULIA_PRINTOUT="${TGLFEP_JULIA_PRINTOUT:-0}"
# Single-node backfill: each of the 4 GPU-worker tasks drains the shared claim queue, reusing its
# MPS team across radii. mps_team fills the A100 within each radius (matches the :grid/:truth paths).
export INNER="${INNER:-mps_team}"
export MPS_TEAM="${MPS_TEAM:-8}"
export SOLVER="${SOLVER:-robust_ad}"
export REFINE_ROUNDS="${REFINE_ROUNDS:-1}"
export GPUS_PER_RADIUS=1
export BACKFILL_MODE=1
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"
export JULIA_CUDA_USE_COMPAT=false
export TGLFEP_JULIA_PROBE="${TGLFEP_JULIA_PROBE:-0}"

NB="${NB:-32}"

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TGLFEP_JULIA_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"
export GACODE_FILE="${GACODE_FILE:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_scan20_nb${NB}.TGLFEP}"
# All 4 tasks MUST share one OUT_DIR so the .claims/ queue is shared (no SLURM_PROCID in the path).
export OUT_DIR="${TGLFEP_JULIA_ROOT}/dev/gacode_nb${NB}_scan20_1node_${SOLVER}_${SLURM_JOB_ID}_tasks"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.$SLURM_JOB_ID"

GPU_SYSIMG="${TGLFEP_JULIA_GPU_SYSIMAGE:-${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_gpu_generic_sysimage.so}"
if [[ "${FORCE_JIT:-0}" == "1" ]]; then
    export TGLFEP_JULIA_GPU_SYSIMAGE="/nonexistent/force-jit"
    MASTER_SYSIMG_ARGS=()
    echo "FORCE_JIT=1 -> running with JIT (no sysimage)"
elif [[ -f "${GPU_SYSIMG}" ]]; then
    export TGLFEP_JULIA_GPU_SYSIMAGE="${GPU_SYSIMG}"
    MASTER_SYSIMG_ARGS=(--sysimage="${GPU_SYSIMG}")
    echo "GPU sysimage (master+workers): ${GPU_SYSIMG}"
else
    export TGLFEP_JULIA_GPU_SYSIMAGE="/nonexistent/force-jit"
    MASTER_SYSIMG_ARGS=()
    echo "GPU sysimage not found at '${GPU_SYSIMG}' -> running with JIT"
fi

cd "${TGLFEP_JULIA_ROOT}/dev"
JOB_T0=$(date +%s.%N)
echo "TIMING_START backend=julia device=gpu solver=${SOLVER} path=gacode-1node-backfill nodes=1 tasks=4 SCAN_N=20 N_BASIS=${NB} MPS_TEAM=${MPS_TEAM}"
echo "OUT_DIR=${OUT_DIR}"
nvidia-smi -L 2>/dev/null | head -4 || true

T0=$(date +%s.%N)
srun --export=ALL --label -N 1 -n 4 --ntasks-per-node=4 --cpu-bind=cores \
    ${TGLFEP_JULIA_ROOT}/slurm/common/mps-scan-wrapper.sh \
    stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" \
    -t "${JULIA_WORKER_THREADS}" ${TGLFEP_JULIA_ROOT}/slurm/common/run_gacode_scan20_mps_task.jl
T1=$(date +%s.%N)
SCAN_S=$(python3 -c "print(f'{float(\"${T1}\") - float(\"${T0}\"): .3f}')")
echo "TIMING_RESULT backend=julia device=gpu solver=${SOLVER} phase=scan seconds=${SCAN_S} SCAN_N=20 N_BASIS=${NB} nodes=1 tasks=4"

# stop MPS daemon
srun --export=ALL -N 1 -n 1 --ntasks-per-node=1 \
    bash -c 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' || true

export USE_GPU=0
stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" -t 8 ${TGLFEP_JULIA_ROOT}/slurm/common/merge_gacode_scan20_array.jl
echo "merged -> ${OUT_DIR}/sfmin_scan.txt"

JOB_T1=$(date +%s.%N)
TOTAL_S=$(python3 -c "print(f'{float(\"${JOB_T1}\") - float(\"${JOB_T0}\"): .3f}')")
echo "TIMING_RESULT backend=julia device=gpu solver=${SOLVER} phase=total_job seconds=${TOTAL_S} SCAN_N=20 N_BASIS=${NB} nodes=1 tasks=4"
echo "=== done; outputs in ${OUT_DIR} ==="
