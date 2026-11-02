#!/bin/bash -l
# AD eigen-rule validation + speed-up benchmark on a single premium CPU node.
# Runs three scripts in sequence:
#   1. validate_gamma_dfactor.jl  - AD dγ/dfactor vs finite differences (correctness)
#   2. micro_benchmark_solve.jl   - per-solve Float64-vs-Dual cost at nb=6 and nb=32
#   3. benchmark_critical_factor.jl - AD-Newton grid vs traditional kwscale_scan (nb=6)
#
#   sbatch dev/ad/batch_ad_validate_benchmark.sh
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:45:00
#SBATCH -C cpu
#SBATCH -J ad_valid_bench
#SBATCH -o ad_valid_bench_%j.out
#SBATCH -e ad_valid_bench_%j.err
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

echo "### [1/3] validate_gamma_dfactor.jl (AD vs finite difference) ###"
stdbuf -oL -eL "${JL[@]}" dev/ad/validate_gamma_dfactor.jl 2>&1 | tee dev/ad/validate_out_job.txt

echo "### [2/3] micro_benchmark_solve.jl (per-solve Float64 vs Dual) ###"
stdbuf -oL -eL "${JL[@]}" dev/ad/micro_benchmark_solve.jl 2>&1 | tee dev/ad/micro_out_job.txt

echo "### [3/3] benchmark_critical_factor.jl (AD-Newton vs kwscale_scan) ###"
stdbuf -oL -eL "${JL[@]}" dev/ad/benchmark_critical_factor.jl 2>&1 | tee dev/ad/benchmark_out_job.txt

echo "=== AD validation + benchmark finished ==="
