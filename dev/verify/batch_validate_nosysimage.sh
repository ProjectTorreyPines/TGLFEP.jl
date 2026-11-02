#!/bin/bash -l
# Full DIII-D validation (20 radii) without sysimage — longer job.
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -J TJLFEP_val_ns
#SBATCH -o validate_nosysimage_%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7

export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"
export SCAN_N=20
export N_BASIS=32

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TGLFEP_JULIA_ROOT}/dev"

echo "=== TJLFEP validate (no sysimage) SCAN_N=$SCAN_N ==="
julia --version

julia --project="${TGLFEP_JULIA_ROOT}" -e 'using Pkg; Pkg.instantiate()'

julia --project="${TGLFEP_JULIA_ROOT}" -t "${SLURM_CPUS_PER_TASK:-32}" verify/validate_nosysimage.jl

echo "=== validate finished ==="
