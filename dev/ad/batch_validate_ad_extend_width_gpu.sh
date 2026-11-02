#!/bin/bash -l
# Sanity check for the width-extended :ad path (single GPU, premium, JIT).
#   cd dev && RADII=22,95 NB=8 sbatch ad/batch_validate_ad_extend_width_gpu.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -C gpu
#SBATCH -J tglfep_julia_ad_extw_gpu
#SBATCH -o ad_extend_width_gpu_%j.out
#SBATCH -e ad_extend_width_gpu_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"
export JULIA_CUDA_USE_COMPAT=false

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TGLFEP_JULIA_ROOT}/dev"

export USE_GPU=1
export RADII="${RADII:-22,95}"
export NB="${NB:-8}"

# JIT only: the prebuilt GPU sysimage predates the extend_width code and would mask it.
echo "=== width-extended :ad sanity (GPU, JIT, no sysimage) ==="
echo "host: $(hostname)  date: $(date)  RADII=${RADII} NB=${NB}"
nvidia-smi -L 2>/dev/null | head -1 || true

stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    -t 16 ad/validate_ad_extend_width.jl

echo "=== width-extended :ad sanity (GPU) done ==="
