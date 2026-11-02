#!/bin/bash -l
# Full 20-radii end-to-end run: inner=:batched_si (hybrid batched shift-invert + dense endpoints +
# per-radius dense fallback) vs inner=:threads (dense-geev golden), on the REAL grid resolution
# (nfactor=8, nefwid=8, nkyhat=4, k_max=4). Reports per-radius sfmin/ky/width agreement and the
# aggregate wall-time speedup. Uses the freshly-built file-only GPU sysimage for fast startup.
#
# Submit chained after the sysimage build:
#   cd TJLFEP && sbatch --dependency=afterok:<BUILD_JOBID> dev/ad/batch_si_full20_gpu.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -J tglfep_julia_si_full20
#SBATCH -o dev/ad/si_full20_%j.out
#SBATCH -e dev/ad/si_full20_%j.err
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
SYSARG=()
if [[ -f "${SO}" ]]; then SYSARG=(--sysimage="${SO}"); echo "using sysimage ${SO} ($(du -h "${SO}"|cut -f1))"
else echo "WARNING: sysimage ${SO} not found; JIT fallback"; fi
JL=(julia --startup-file=no "${SYSARG[@]}" --project="${ROOT}" -t 32)

echo "=== si full-20 e2e  job=${SLURM_JOB_ID:-?}  $(date) ==="
nvidia-smi -L 2>/dev/null | head -1 || true

# nb16 full grid, all 20 radii (dense CPU golden is tractable at n=720).
echo ""; echo "######## nb16  full grid (8,8,4,4)  all radii ########"
NB=16 USE_GPU=1 NFACTOR=8 NEFWID=8 NKYHAT=4 KMAX=4 RADII="" \
    stdbuf -oL "${JL[@]}" dev/ad/compare_batched_si_e2e.jl 2>&1 | tail -40

# nb32 full grid, all 20 radii (n=1440: where the batched eigensolver's advantage is largest).
echo ""; echo "######## nb32  full grid (8,8,4,4)  all radii ########"
NB=32 USE_GPU=1 NFACTOR=8 NEFWID=8 NKYHAT=4 KMAX=4 RADII="" \
    stdbuf -oL "${JL[@]}" dev/ad/compare_batched_si_e2e.jl 2>&1 | tail -40

echo ""; echo "=== done $(date) ==="
