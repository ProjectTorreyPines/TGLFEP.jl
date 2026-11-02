#!/bin/bash -l
# GPU A/B exactness for confirm_grid at SPECIFIC scan radii (single GPU, uses the fresh sysimage).
#   cd dev && RADII=2,17 NB_LIST=6 sbatch ad/batch_validate_confirm_grid_gpu.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -C gpu
#SBATCH -J tglfep_julia_confirm_ab_gpu
#SBATCH -o confirm_grid_ab_gpu_%j.out
#SBATCH -e confirm_grid_ab_gpu_%j.err
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
export RADII="${RADII:-2,17}"
export NB_LIST="${NB_LIST:-6}"

GPU_SYSIMG="${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_gpu_generic_sysimage.so"
if [[ -f "${GPU_SYSIMG}" ]]; then
    SYSIMG_ARGS=(--sysimage="${GPU_SYSIMG}")
    echo "GPU sysimage: ${GPU_SYSIMG}"
else
    SYSIMG_ARGS=()
    echo "GPU sysimage missing -> JIT"
fi

echo "=== confirm_grid GPU A/B  RADII=${RADII} NB_LIST=${NB_LIST} ==="
echo "host: $(hostname)  date: $(date)"
nvidia-smi -L 2>/dev/null | head -1 || true

stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    "${SYSIMG_ARGS[@]}" -t 16 ad/validate_confirm_grid.jl

echo "=== confirm_grid GPU A/B done ==="
