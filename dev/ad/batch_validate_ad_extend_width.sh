#!/bin/bash -l
# Sanity check for the width-extended :ad path (CPU, single node, JIT).
#   cd dev && sbatch ad/batch_validate_ad_extend_width.sh
#SBATCH -A m3739
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C cpu
#SBATCH -J tglfep_julia_ad_extw
#SBATCH -o ad_extend_width_%j.out
#SBATCH -e ad_extend_width_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

set -uo pipefail

module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TGLFEP_JULIA_ROOT}/dev"

export RADII="${RADII:-22,95}"
export NB="${NB:-8}"

# JIT (no sysimage) so the edited extend_width source is recompiled and used.
echo "=== width-extended :ad sanity (JIT, no sysimage) ==="
echo "host: $(hostname)  date: $(date)  RADII=${RADII} NB=${NB}"

stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    -t 32 ad/validate_ad_extend_width.jl

echo "=== width-extended :ad sanity done ==="
