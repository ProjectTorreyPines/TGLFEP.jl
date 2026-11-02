#!/bin/bash -l
# End-to-end validation of the TJLFEP solver=:grid|:ad toggle on a dedicated
# Perlmutter GPU-queue node (clean timing vs the contended login node). Runs the
# single-radius grid-vs-ad agreement check and the full multi-radius AD driver
# (runTHD_from_gacode; solver=:ad) — the exact mainsub(:ad) + driver path the
# FUSE ActorTJLFEP reaches via runTHD(dd).
#
#   sbatch dev/ad/batch_validate_solver_toggle_gpu.sh
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -J ad_toggle
#SBATCH -o dev/ad/ad_toggle_%j.out
#SBATCH -e dev/ad/ad_toggle_%j.err
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

stdbuf -oL -eL "${JL[@]}" dev/ad/validate_solver_toggle.jl 2>&1 | tee dev/ad/validate_solver_toggle_job.txt

echo "=== solver-toggle validation finished $(date) ==="
