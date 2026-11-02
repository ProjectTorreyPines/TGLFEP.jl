#!/bin/bash -l
# Correctness check for the L1 Complex{Dual} GPU eigensolve+IFT kernel: GPU Dual derivatives
# vs CPU Dual + finite differences at N_BASIS=32 on DIII-D. Runs JIT (NO sysimage) so it
# picks up the freshly added TJLF._gpu_solve_eig_grad! source.
#   cd dev && sbatch ad/batch_validate_gpu_dual_grad.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH -J ad_gpu_dual_validate
#SBATCH -o ad_gpu_dual_validate_%j.out
#SBATCH -e ad_gpu_dual_validate_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"
export JULIA_CUDA_USE_COMPAT=false
export TGLFEP_JULIA_FILE_ONLY=1

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TGLFEP_JULIA_ROOT}"
nvidia-smi -L 2>/dev/null | head -1 || true

stdbuf -oL -eL julia --startup-file=no --project=. -t 8 dev/ad/validate_gpu_dual_grad.jl
echo "=== validate_gpu_dual_grad done (exit $?) ==="
