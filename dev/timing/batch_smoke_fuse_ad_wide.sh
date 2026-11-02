#!/bin/bash -l
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:50:00
#SBATCH -C gpu
#SBATCH -J smoke_ad_wide
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

export TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-${PSCRATCH}/.julia/dev/TJLFEP}"
export FUSE_ROOT="${FUSE_ROOT:-${PSCRATCH}/.julia/dev/FUSE}"
export USE_GPU=1
export SOLVER=ad
export AD_EXTEND_MODE=wide
export AD_WIDE_KDESC=2
export JULIA_CUDA_USE_COMPAT=false

# JIT from dev source on purpose: the baked sysimage freezes FUSE/TJLFEP, so it CANNOT see the
# new extend_mode/wide_kdesc/faithful_confirm code paths. This smoke test must compile dev source.
SYSFLAG=""
echo "sysimage = <none, JIT from dev source (testing local edits)>"

cd "${TGLFEP_JULIA_ROOT}/dev/timing"
nvidia-smi -L 2>/dev/null | head -2 || true

stdbuf -oL -eL julia --startup-file=no ${SYSFLAG} --project="${FUSE_ROOT}" \
    "${TGLFEP_JULIA_ROOT}/dev/timing/smoke_fuse_ad_wide.jl"

echo "=== smoke done (look for SMOKE_OK) ==="
