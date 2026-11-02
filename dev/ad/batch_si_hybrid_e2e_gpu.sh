#!/bin/bash -l
# Validate the HYBRID batched-SI grid: coarse k-rounds use the fast batched shift-invert
# eigensolver to localize the marginal box, the FINAL k-round uses the dense eigensolver to pin
# an exact marginal. Compares inner=:batched_si (hybrid, GPU) vs inner=:threads (dense golden)
# per radius, including the two radii that diverged under pure-SI (IR 74, 90 = indices 15, 18).
#
#   cd TJLFEP && sbatch dev/ad/batch_si_hybrid_e2e_gpu.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -J tglfep_julia_si_hybrid
#SBATCH -o dev/ad/si_hybrid_e2e_%j.out
#SBATCH -e dev/ad/si_hybrid_e2e_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=1

set -uo pipefail
module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${ROOT}"
JL=(julia --startup-file=no --project="${ROOT}" -t 32)

echo "=== si hybrid e2e  job=${SLURM_JOB_ID:-?}  $(date) ==="
nvidia-smi -L 2>/dev/null | head -1 || true

# Full spread of radii (same grid for both paths, so agreement proves exactness).
NB=16 USE_GPU=1 NFACTOR=6 NEFWID=4 NKYHAT=2 KMAX=3 RADII=3,6,9,12,15,18 \
    stdbuf -oL "${JL[@]}" dev/ad/compare_batched_si_e2e.jl 2>&1 | tail -20

echo ""; echo "=== done $(date) ==="
