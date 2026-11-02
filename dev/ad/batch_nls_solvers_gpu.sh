#!/bin/bash -l
# Head-to-head of the derivative-free (ky,w) solvers (:multistart, :nlopt) vs the
# reference tiers (:grid wide box, :ad :locate narrow) on the 20-radius DIII-D 202017C42_500ms
# scan, using the GPU eigensolve path (the CPU path is far too slow for a 20-radius sweep). JIT
# only: the prebuilt GPU sysimage predates the NLS-solver code and would mask it.
#
#   cd TJLFEP && sbatch dev/ad/batch_nls_solvers_gpu.sh
# Env overrides: NB (default 32), RADII (default all 20), SOLVERS (default grid,ad,multistart,nlopt),
#                NLS_LOCAL_ALGO (default LN_BOBYQA), NLOPT_ALGO/NLOPT_MAXEVAL.
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -J tglfep_julia_nls_gpu
#SBATCH -o dev/ad/nls_solvers_gpu_%j.out
#SBATCH -e dev/ad/nls_solvers_gpu_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"
export JULIA_CUDA_USE_COMPAT=false

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TGLFEP_JULIA_ROOT}"

export USE_GPU=1
export GKSwstype=nul                       # headless GR (Plots writes the PNG, no display)
export NB="${NB:-16}"                       # nb=16 keeps the 20-radius x 4-solver sweep tractable
export RADII="${RADII:-}"                   # empty = all SCAN_N radii
export SOLVERS="${SOLVERS:-grid,ad,multistart,nlopt}"

echo "=== NLS derivative-free solvers head-to-head (GPU, JIT) ==="
echo "host: $(hostname)  date: $(date)  job=${SLURM_JOB_ID:-?}"
echo "NB=${NB}  RADII=${RADII:-all}  SOLVERS=${SOLVERS}  NLS_LOCAL_ALGO=${NLS_LOCAL_ALGO:-LN_BOBYQA}"
nvidia-smi -L 2>/dev/null | head -1 || true

stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    -t 16 dev/ad/benchmark_nls_solvers.jl

echo "=== NLS solvers head-to-head (GPU) done $(date) ==="
