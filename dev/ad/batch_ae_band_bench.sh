#!/bin/bash -l
# Validate ae_band=true: AD-Newton critical factor (AE-band-filtered γ) vs the
# traditional kwscale_scan, checking sfmin agreement + eigensolve count.
#
#   sbatch dev/ad/batch_ae_band_bench.sh
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -C cpu
#SBATCH -J ae_band_bench
#SBATCH -o ae_band_bench_%j.out
#SBATCH -e ae_band_bench_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

TGLFEP_JULIA_ROOT="/pscratch/sd/t/tneiser/.julia/dev/TJLFEP"
cd "${TGLFEP_JULIA_ROOT}"

NTHREADS="${NTHREADS:-8}"
JL=(julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" -t "${NTHREADS}")

echo "=== node=$(hostname) threads=${NTHREADS} job=${SLURM_JOB_ID:-?} ==="
stdbuf -oL -eL "${JL[@]}" dev/ad/benchmark_critical_factor.jl 2>&1 | tee dev/ad/benchmark_ae_band_out.txt
echo "=== ae_band benchmark finished ==="
