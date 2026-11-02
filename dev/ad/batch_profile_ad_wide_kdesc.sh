#!/bin/bash -l
# Formalized :wide kdesc sweep over the 20-radius scan (single GPU, JIT).
#   cd dev && NB=8 sbatch ad/batch_profile_ad_wide_kdesc.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -J tglfep_julia_ad_wide_kdesc
#SBATCH -o ad_wide_kdesc_%j.out
#SBATCH -e ad_wide_kdesc_%j.err
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
export NB="${NB:-8}"
export KDESC="${KDESC:-1,2,3}"

# JIT only: validates the refactored extend_mode=:wide code (sysimage predates it).
echo "=== formalized :wide kdesc sweep (GPU, JIT, no sysimage) ==="
echo "host: $(hostname)  date: $(date)  NB=${NB} KDESC=${KDESC}"
nvidia-smi -L 2>/dev/null | head -1 || true

stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    -t 16 ad/profile_ad_wide_kdesc.jl

echo "=== formalized :wide kdesc sweep done ==="
