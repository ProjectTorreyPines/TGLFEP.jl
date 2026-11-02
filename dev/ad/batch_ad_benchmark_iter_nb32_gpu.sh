#!/bin/bash -l
# AD-vs-production speed-up benchmark for ITER IR=83 at the heavy N_BASIS=32
# resolution, on a dedicated Perlmutter GPU-queue node. Both production
# kwscale_scan and AD marginal_factor_faithful run CPU-threaded over the full
# node (use_gpu=false); the GPU queue is used only for the dedicated allocation.
#
#   sbatch dev/ad/batch_ad_benchmark_iter_nb32_gpu.sh
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -J ad_bench_iter32
#SBATCH -o dev/ad/ad_bench_iter32_%j.out
#SBATCH -e dev/ad/ad_bench_iter32_%j.err
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

echo "### ITER IR=83  N_BASIS=32  AD vs production ###"
stdbuf -oL -eL "${JL[@]}" dev/ad/benchmark_ad_vs_production_iter_nb32.jl 2>&1 | tee dev/ad/bench_iter_nb32_job.txt

echo "=== ITER N_BASIS=32 benchmark finished $(date) ==="
