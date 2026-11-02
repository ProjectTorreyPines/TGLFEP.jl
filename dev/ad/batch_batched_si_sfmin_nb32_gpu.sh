#!/bin/bash -l
# Run the wired inner=:batched_si (fixed-shift hybrid) solver over the full 20-radius scan at
# N_BASIS=32, writing sfmin(IR) for overlay on the stored nb32 grid/Fortran reference.
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 02:30:00
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -J tglfep_julia_bsi_sfmin32
#SBATCH -o dev/ad/bsi_sfmin_nb32_%j.out
#SBATCH -e dev/ad/bsi_sfmin_nb32_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=1

set -uo pipefail
module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"
export JULIA_CUDA_USE_COMPAT=false
ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${ROOT}"
SO="${ROOT}/build/TGLFEP_JULIA_gpu_sysimage.so"
SYSARG=(); [[ -f "${SO}" ]] && SYSARG=(--sysimage="${SO}")

echo "=== batched_si sfmin nb32  $(date) ==="
NB=32 USE_GPU=1 NFACTOR=8 NEFWID=8 NKYHAT=4 KMAX=4 \
  OUT="${ROOT}/dev/ad/batched_si_sfmin_nb32.txt" \
  julia --startup-file=no "${SYSARG[@]}" --project="${ROOT}" -t 32 dev/ad/run_batched_si_sfmin.jl 2>&1 | tail -30
echo "=== done $(date) ==="
