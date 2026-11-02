#!/bin/bash -l
# Follow-ups after (a)/(b)/(c):
#   1. Re-check batched-SI leader recovery at nb16 & nb32 with the RETUNED shift set (denser near
#      the real axis where the electron branch lives) — expect the nb32 electron misses to drop.
#   2. End-to-end: inner=:batched_si (GPU batched solver) vs inner=:threads (dense-geev golden) on
#      the DIII-D scan at a moderate grid, comparing marginal sfmin / ky / width per radius.
#
#   cd TJLFEP && sbatch dev/ad/batch_si_tune_e2e_gpu.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -J tglfep_julia_si_tune
#SBATCH -o dev/ad/si_tune_e2e_%j.out
#SBATCH -e dev/ad/si_tune_e2e_%j.err
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

echo "=== si tune + e2e  job=${SLURM_JOB_ID:-?}  $(date) ==="
nvidia-smi -L 2>/dev/null | head -1 || true

echo ""; echo "######## 1) retuned-shift accuracy re-check ########"
for cfg in "16 dev/ad/pencils_nb16" "32 dev/ad/pencils_nb32"; do
    set -- $cfg
    echo ""; echo "==== nb=$1  (retuned 13-shift set) ===="
    PENCILS="${ROOT}/$2" METHOD=trsm ORTH=cholqr M=16 Q=12 \
        stdbuf -oL "${JL[@]}" dev/ad/benchmark_batched_si_gpu.jl 2>&1 | tail -14
done

echo ""; echo "######## 2) end-to-end  batched_si vs threads-golden ########"
# nb16 keeps the dense CPU golden tractable; a spread of radii (stable + unstable).
NB=16 USE_GPU=1 NFACTOR=6 NEFWID=4 NKYHAT=2 KMAX=2 RADII=3,6,9,12,15,18 \
    stdbuf -oL "${JL[@]}" dev/ad/compare_batched_si_e2e.jl 2>&1 | tail -20

echo ""; echo "=== done $(date) ==="
