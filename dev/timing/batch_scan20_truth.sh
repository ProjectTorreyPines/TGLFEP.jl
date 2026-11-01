#!/bin/bash -l
# Production-layout PHYSICAL-TRUTH sfmin(radius) profile: DIII-D SCAN_N=20, 5 nodes × 4 GPUs =
# 20 GPUs, ONE radius per GPU. Each task runs the physical-truth engine (solver=truth) — the
# extended-width (w down to ~0.05) (ky,w) locate + separable nbasis convergence (critical_factor_truth)
# — on its pinned A100, then a merge step assembles sfmin_scan.txt. This is NOT Fortran-faithful: it
# captures the narrow-width EP-driven AEs excluded by the canonical w≥1 box (sfmin up to ~25× lower
# at near-marginal radii). Runs current source via JIT (FORCE_JIT=1 default) since the truth path is
# new and not yet in the shared sysimage.
#   cd dev && NB=32 sbatch timing/batch_scan20_truth.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 5
#SBATCH -n 20
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH -J s20_truth
#SBATCH -o scan20_truth_%j.out
#SBATCH -e scan20_truth_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

# TJLFEP_PRINTOUT=1 → each task writes out.scalefactor_r### with the truth summary
# (sfmin nbasis-converged, sfmin_work, binding, status, nbasis table) so this run doubles as the
# per-radius diagnostic.
export TJLFEP_FILE_ONLY=1 USE_GPU=1 TJLFEP_DEBUG=0 TJLFEP_PRINTOUT=1
# INNER selects within-radius parallelism over the extended (ky,w) seed grid + nbasis evals.
# mps_team: MPS_TEAM worker procs share the radius's GPU; the independent eigensolve chains overlap
# via Hyper-Q (validated ~2.6× over threads). At nb≤48 a single eigensolve underfills the A100, and the
# ~66-point extended (ky,w) seed grid is embarrassingly parallel, so use team=8 to match the :grid path
# (8 workers/GPU; 8×2 threads = cpus-per-task=16).
export INNER="${INNER:-mps_team}"
export MPS_TEAM="${MPS_TEAM:-8}"
export SOLVER=truth
export GPUS_PER_RADIUS=1
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"
export JULIA_CUDA_USE_COMPAT=false
export TJLFEP_PROBE="${TJLFEP_PROBE:-0}"

NB="${NB:-32}"

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TJLFEP_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"
export GACODE_FILE="${GACODE_FILE:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_scan20_nb${NB}.TGLFEP}"
export OUT_DIR="${TJLFEP_ROOT}/dev/gacode_nb${NB}_scan20_truth_${INNER}_${SLURM_JOB_ID}_tasks"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.$SLURM_JOB_ID"

# Sysimage policy: the truth path is new and NOT in the shared sysimage, so JIT current source by
# default. Set FORCE_JIT=0 + TJLFEP_GPU_SYSIMAGE=... once a truth-baked sysimage exists.
if [[ "${FORCE_JIT:-1}" == "1" ]]; then
    export TJLFEP_GPU_SYSIMAGE="/nonexistent/force-jit"
    MASTER_SYSIMG_ARGS=()
    echo "FORCE_JIT=1 -> running truth path with JIT (no sysimage)"
else
    GPU_SYSIMG="${TJLFEP_GPU_SYSIMAGE:-${TJLFEP_ROOT}/dev/TJLFEP_gpu_generic_sysimage.so}"
    if [[ -f "${GPU_SYSIMG}" ]]; then
        export TJLFEP_GPU_SYSIMAGE="${GPU_SYSIMG}"
        MASTER_SYSIMG_ARGS=(--sysimage="${GPU_SYSIMG}")
        echo "GPU sysimage: ${GPU_SYSIMG}"
    else
        export TJLFEP_GPU_SYSIMAGE="/nonexistent/force-jit"
        MASTER_SYSIMG_ARGS=()
        echo "GPU sysimage not found at '${GPU_SYSIMG}' -> running with JIT"
    fi
fi

cd "${TJLFEP_ROOT}/dev"
JOB_T0=$(date +%s.%N)
echo "START solver=truth inner=${INNER} mps_team=${MPS_TEAM} nodes=${SLURM_NNODES:-5} tasks=${SLURM_NTASKS:-20} SCAN_N=20 N_BASIS=${NB}"
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
echo "TIMING_RESULT solver=truth inner=${INNER} mps_team=${MPS_TEAM} phase=scan seconds=${SCAN_S} SCAN_N=20 N_BASIS=${NB} nodes=${SLURM_NNODES:-5} tasks=${SLURM_NTASKS:-20}"

# stop MPS daemons (one per node)
srun --export=ALL -n "${SLURM_NNODES:-5}" --ntasks-per-node=1 \
    bash -c 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' || true

export USE_GPU=0
stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" -t 8 ${TJLFEP_ROOT}/slurm/common/merge_gacode_scan20_array.jl
echo "merged -> ${OUT_DIR}/sfmin_scan.txt"

JOB_T1=$(date +%s.%N)
TOTAL_S=$(python3 -c "print(f'{float(\"${JOB_T1}\") - float(\"${JOB_T0}\"): .3f}')")
echo "TIMING_RESULT solver=truth inner=${INNER} mps_team=${MPS_TEAM} phase=total_job seconds=${TOTAL_S} SCAN_N=20 N_BASIS=${NB}"
echo "=== done; sfmin_scan.txt in ${OUT_DIR} ==="
