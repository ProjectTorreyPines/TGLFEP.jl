#!/bin/bash -l
# §7.1 A/B exactness for the confirm_grid cheap-rank→few-confirm path (CPU, single node).
#   cd dev && sbatch ad/batch_validate_confirm_grid.sh
#SBATCH -A m3739
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -C cpu
#SBATCH -J tglfep_julia_confirm_ab
#SBATCH -o confirm_grid_ab_%j.out
#SBATCH -e confirm_grid_ab_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

set -uo pipefail

module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TGLFEP_JULIA_ROOT}/dev"

export RADII="${RADII:-22,95}"
export NB_LIST="${NB_LIST:-8,16}"

# NOTE: deliberately NOT using a prebuilt sysimage — it bakes in a stale TJLFEP and would
# mask the new confirm_grid code path. Run JIT so the edited source is recompiled+used.
echo "=== confirm_grid A/B exactness (JIT, no sysimage) ==="
echo "host: $(hostname)  date: $(date)  RADII=${RADII} NB_LIST=${NB_LIST}"

stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    -t 32 ad/validate_confirm_grid.jl

echo "=== confirm_grid A/B done ==="
