#!/bin/bash -l
# AD-vs-production speed-up benchmark on a single dedicated Perlmutter GPU-queue
# node (clean, reproducible wall times vs the contended login node). Both the
# production kwscale_scan and the AD marginal_factor_faithful run CPU-threaded
# (use_gpu=false) over the full node so the production/AD comparison is
# apples-to-apples; the GPU queue is used only for the dedicated allocation.
#
#   sbatch dev/ad/batch_ad_benchmark_gpu.sh
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -J ad_bench
#SBATCH -o dev/ad/ad_bench_%j.out
#SBATCH -e dev/ad/ad_bench_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

TGLFEP_JULIA_ROOT="/pscratch/sd/t/tneiser/.julia/dev/TJLFEP"
cd "${TGLFEP_JULIA_ROOT}"

NTHREADS="${NTHREADS:-64}"
JL=(julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" -t "${NTHREADS}")

echo "=== node=$(hostname) threads=${NTHREADS} job=${SLURM_JOB_ID:-?} $(date) ==="

echo "### [1/2] ITER  (N_BASIS=2)  AD vs production ###"
stdbuf -oL -eL "${JL[@]}" dev/ad/benchmark_ad_vs_production.jl 2>&1 | tee dev/ad/bench_iter_job.txt

echo "### [2/2] DIII-D (N_BASIS=32) AD vs production ###"
stdbuf -oL -eL "${JL[@]}" dev/ad/benchmark_ad_vs_production_diiid.jl 2>&1 | tee dev/ad/bench_diiid_job.txt

echo "=== AD benchmark finished $(date) ==="
