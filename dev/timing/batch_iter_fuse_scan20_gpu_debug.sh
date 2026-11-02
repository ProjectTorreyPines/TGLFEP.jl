#!/bin/bash -l
# FUSE ITER case, full N_BASIS=32 / SCAN_N=20 / GPU, on the 5-node / 20-GPU layout
# (4 radii/node, 1 A100/radius) in the gpu *debug* queue. The master (this batch step)
# builds the ITER dd via FUSE and addprocs(SlurmManager()) launches 20 GPU-pinned workers;
# FUSE.ActorTJLFEP -> TJLFEP.runTHD pmap's the 20 radii 1:1 onto the 20 A100s. No MPS
# (1 radius/GPU => no oversubscription). Emits TIMING_RESULT markers.
#
#   sbatch dev/timing/batch_iter_fuse_scan20_gpu_debug.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 5
#SBATCH -n 20
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH -J iter_fuse_s20_jgpu
#SBATCH -o iter_fuse_scan20_gpu_%j.out
#SBATCH -e iter_fuse_scan20_gpu_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

export TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export FUSE_ROOT="${FUSE_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/FUSE}"

export SCAN_N="${SCAN_N:-20}"
export N_BASIS="${N_BASIS:-32}"
export NGRID="${NGRID:-201}"
export ALPHA_SOLVER="${ALPHA_SOLVER:-stiff}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-8}"
export JULIA_CUDA_USE_COMPAT=false

# Workers JIT TJLFEP from the dev source (NOT the baked sysimage): the master serializes the
# runTHD pmap closure to the workers, and a stale-source sysimage has mismatched closure types
# (UndefVarError #NNN#NNN in TJLFEP). Leave TGLFEP_JULIA_GPU_SYSIMAGE empty unless the sysimage is
# rebuilt from the current TJLFEP source. Costs ~110 s of one-time GPU-eigensolve JIT per worker.
export TGLFEP_JULIA_GPU_SYSIMAGE="${TGLFEP_JULIA_GPU_SYSIMAGE:-}"

cd "${TGLFEP_JULIA_ROOT}/dev"
echo "=== iter_fuse_scan20_gpu_debug  job=${SLURM_JOB_ID}  nodes=${SLURM_NNODES} tasks=${SLURM_NTASKS} ==="
echo "N_BASIS=${N_BASIS} SCAN_N=${SCAN_N} NGRID=${NGRID} ALPHA_SOLVER=${ALPHA_SOLVER}"
nvidia-smi -L 2>/dev/null | head -4 || true

# Master runs directly (NOT under srun) so addprocs(SlurmManager()) can claim all 20 task
# slots for the GPU workers. Master uses the FUSE project (FUSE is not in the GPU sysimage).
stdbuf -oL -eL julia --startup-file=no --project="${FUSE_ROOT}" \
    timing/run_iter_fuse_scan20_gpu.jl

echo "=== done; see TIMING_RESULT markers above ==="
