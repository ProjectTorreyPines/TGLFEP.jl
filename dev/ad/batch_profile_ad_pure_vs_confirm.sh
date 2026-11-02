#!/bin/bash -l
# Pure-AD vs confirm-AD width-extended :ad comparison over the 20-radius scan (single GPU, JIT).
#   cd dev && NB=8 sbatch ad/batch_profile_ad_pure_vs_confirm.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -J tglfep_julia_ad_pure_vs_conf
#SBATCH -o ad_pure_vs_confirm_%j.out
#SBATCH -e ad_pure_vs_confirm_%j.err
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
export MODES="${MODES:-pure,confirm,robust}"

# JIT only: the prebuilt GPU sysimage predates the faithful_confirm-relaxed extend_width code.
echo "=== pure vs confirm width-extended :ad (GPU, JIT, no sysimage) ==="
echo "host: $(hostname)  date: $(date)  NB=${NB} MODES=${MODES} RADII=${RADII:-<all 20>}"
nvidia-smi -L 2>/dev/null | head -1 || true

stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    -t 16 ad/profile_ad_pure_vs_confirm.jl

echo "=== pure vs confirm width-extended :ad done ==="
