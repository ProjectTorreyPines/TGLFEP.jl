#!/bin/bash -l
# Timing: Julia GPU SCAN_N=20 ROBUST_AD (solver=:robust_ad, inner=:mps_team), 5 nodes
# (4 radii/node, 1 A100/radius, MPS_TEAM clients/radius). Each radius runs critical_factor_robust
# with extend_width=true: the canonical w>=1 (ky,w) faithful grid-zoom PLUS the extended narrow-width
# locate (cheap AE-onset rank -> :ad descent -> faithful confirm). This is the WIDTH tier of the
# grid -> robust_ad -> truth ladder (the 2-11x sfmin reduction at nb=N_BASIS), WITHOUT the truth
# nbasis ladder -- so it isolates the cost of the width extension vs the full truth tier.
# The within-radius work (~66-pt extended seed grid + descents + confirms + the w>=1 grid-zoom) is
# embarrassingly parallel, so use MPS_TEAM=8 to fill the A100 (matches the :grid / :truth paths).
# N_BASIS=NB sets the working basis (single nb; no ladder). Sweep nbasis via NB:
#   NB=32 sbatch timing/batch_time_scan20_julia_gpu_robust_ad.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 5
#SBATCH -n 20
#SBATCH -t 00:45:00
#SBATCH -C gpu
#SBATCH -J time_s20_jgpu_robust_ad
#SBATCH -o time_scan20_julia_gpu_robust_ad_%j.out
#SBATCH -e time_scan20_julia_gpu_robust_ad_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

export TJLFEP_FILE_ONLY=1 USE_GPU=1 TJLFEP_DEBUG=0 TJLFEP_PRINTOUT=0
export INNER=mps_team
export SOLVER=robust_ad
export REFINE_ROUNDS="${REFINE_ROUNDS:-1}"
export GPUS_PER_RADIUS=1
export MPS_TEAM="${MPS_TEAM:-8}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"
export JULIA_CUDA_USE_COMPAT=false
export TJLFEP_PROBE="${TJLFEP_PROBE:-0}"

NB="${NB:-32}"

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TJLFEP_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"
export GACODE_FILE="${GACODE_FILE:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_scan20_nb${NB}.TGLFEP}"
export OUT_DIR="${TJLFEP_ROOT}/dev/gacode_nb${NB}_scan20_jgpu_robust_ad_${SLURM_JOB_ID}_tasks"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.$SLURM_JOB_ID"

GPU_SYSIMG="${TJLFEP_GPU_SYSIMAGE:-${TJLFEP_ROOT}/dev/TJLFEP_gpu_generic_sysimage.so}"
if [[ -f "${GPU_SYSIMG}" ]]; then
    export TJLFEP_GPU_SYSIMAGE="${GPU_SYSIMG}"
    MASTER_SYSIMG_ARGS=(--sysimage="${GPU_SYSIMG}")
    echo "GPU sysimage (master+workers): ${GPU_SYSIMG}"
else
    MASTER_SYSIMG_ARGS=()
    echo "GPU sysimage: none found at '${GPU_SYSIMG}' -> running with JIT"
fi

cd "${TJLFEP_ROOT}/dev"
JOB_T0=$(date +%s.%N)
echo "TIMING_START backend=julia device=gpu solver=robust_ad path=gacode-mps nodes=${SLURM_NNODES:-5} tasks=${SLURM_NTASKS:-20} SCAN_N=20 N_BASIS=${NB} MPS_TEAM=${MPS_TEAM}"
echo "OUT_DIR=${OUT_DIR}"
nvidia-smi -L 2>/dev/null | head -4 || true

T0=$(date +%s.%N)
srun --export=ALL --label -n "${SLURM_NTASKS:-20}" --ntasks-per-node=4 --cpu-bind=cores \
    ${TJLFEP_ROOT}/slurm/common/mps-scan-wrapper.sh \
    stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" \
    -t "${JULIA_WORKER_THREADS}" ${TJLFEP_ROOT}/slurm/common/run_gacode_scan20_mps_task.jl
T1=$(date +%s.%N)
SCAN_S=$(python3 -c "print(f'{float(\"${T1}\") - float(\"${T0}\"): .3f}')")
echo "TIMING_RESULT backend=julia device=gpu solver=robust_ad phase=scan seconds=${SCAN_S} SCAN_N=20 N_BASIS=${NB} nodes=${SLURM_NNODES:-5} tasks=${SLURM_NTASKS:-20}"

# stop MPS daemons (one per node)
srun --export=ALL -n "${SLURM_NNODES:-5}" --ntasks-per-node=1 \
    bash -c 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' || true

export USE_GPU=0
T0=$(date +%s.%N)
stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" -t 8 ${TJLFEP_ROOT}/slurm/common/merge_gacode_scan20_array.jl
T1=$(date +%s.%N)
MERGE_S=$(python3 -c "print(f'{float(\"${T1}\") - float(\"${T0}\"): .3f}')")
echo "TIMING_RESULT backend=julia device=gpu solver=robust_ad phase=merge seconds=${MERGE_S} SCAN_N=20 N_BASIS=${NB}"

JOB_T1=$(date +%s.%N)
TOTAL_S=$(python3 -c "print(f'{float(\"${JOB_T1}\") - float(\"${JOB_T0}\"): .3f}')")
echo "TIMING_RESULT backend=julia device=gpu solver=robust_ad phase=total_job seconds=${TOTAL_S} SCAN_N=20 N_BASIS=${NB} nodes=${SLURM_NNODES:-5} tasks=${SLURM_NTASKS:-20}"
echo "=== done; outputs in ${OUT_DIR} ==="
