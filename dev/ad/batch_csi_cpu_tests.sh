#!/bin/bash -l
# CPU-side validation of the coverage-certified adaptive SI solver (no GPU needed):
#   1) TJLF unit tests (fixed SI + contour + certified SI, planted spectra)
#   2) certified SI on REAL harvested pencils (pencils_nb16) vs dense geev reference:
#      every unstable in-window mode recovered or the pencil flagged; branch leaders exact.
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -C cpu
#SBATCH -J tglfep_julia_csi_cpu
#SBATCH -o dev/ad/csi_cpu_%j.out
#SBATCH -e dev/ad/csi_cpu_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

set -uo pipefail
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${ROOT}"

echo "=== 1) TJLF unit tests  $(date) ==="
julia --startup-file=no --project="${PSCRATCH}/.julia/dev/TJLF" -t 32 \
  -e 'using Test; include(joinpath(ENV["PSCRATCH"], ".julia/dev/TJLF/test/runtests_batched_si.jl"))'

echo ""; echo "=== 2) certified SI on real pencils (NP=${NP:-24})  $(date) ==="
NP="${NP:-24}" PENCILS="${PENCILS:-${ROOT}/dev/ad/pencils_nb16}" \
  julia --startup-file=no --project="${ROOT}" -t 64 dev/ad/_test_csi_cpu_real.jl

echo ""; echo "=== done $(date) ==="
