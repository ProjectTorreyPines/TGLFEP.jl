#!/bin/bash -l
# Regenerate the production AD sfmin(radius) profile for DIII-D N_BASIS=32, SCAN_N=20 (serial
# over the 20 radii on one GPU). Env: SOLVER (robust_ad), REFINE_ROUNDS (1). Runs JIT (NO
# sysimage) to pick up current source.
#   cd dev && SOLVER=robust_ad REFINE_ROUNDS=1 sbatch ad/batch_ad_threads_sfmin.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 02:30:00
#SBATCH -C gpu
#SBATCH -J ad_threads_sfmin
#SBATCH -o ad/ad_threads_sfmin_%j.out
#SBATCH -e ad/ad_threads_sfmin_%j.err
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

export SOLVER="${SOLVER:-robust_ad}"
export REFINE_ROUNDS="${REFINE_ROUNDS:-1}"
export AD_EXTEND_MODE="${AD_EXTEND_MODE:-locate}"   # :ad extend strategy (locate|wide|only)
echo "SOLVER=${SOLVER} REFINE_ROUNDS=${REFINE_ROUNDS} AD_EXTEND_MODE=${AD_EXTEND_MODE}"

stdbuf -oL -eL julia --startup-file=no --project=. -t 8 dev/ad/ad_threads_sfmin_profile.jl
echo "=== ad_threads_sfmin done (exit $?) ==="
