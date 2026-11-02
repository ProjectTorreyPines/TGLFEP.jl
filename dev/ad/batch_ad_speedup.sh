#!/bin/bash -l
# Measure the improved AD-Newton speed-up after the endpoint-first bracketing.
#   1. demo_marginal_newton.jl    - marginal_factor still finds the correct onset
#   2. benchmark_critical_factor.jl - AD-Newton grid vs traditional kwscale_scan
#
#   sbatch dev/ad/batch_ad_speedup.sh
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -C cpu
#SBATCH -J ad_speedup
#SBATCH -o ad_speedup_%j.out
#SBATCH -e ad_speedup_%j.err
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

echo "### [1/2] demo_marginal_newton.jl (onset correctness after endpoint-first bracketing) ###"
stdbuf -oL -eL "${JL[@]}" dev/ad/demo_marginal_newton.jl 2>&1 | tee dev/ad/demo_marginal_out2.txt

echo "### [2/2] benchmark_critical_factor.jl (AD-Newton vs kwscale_scan) ###"
stdbuf -oL -eL "${JL[@]}" dev/ad/benchmark_critical_factor.jl 2>&1 | tee dev/ad/benchmark_out_job2.txt

echo "=== AD speed-up benchmark finished ==="
