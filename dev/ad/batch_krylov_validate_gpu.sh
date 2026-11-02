#!/bin/bash -l
# Phases (a)+(b): validate the batched shift-invert eigensolver on a DEDICATED A100 (the login GPU
# is shared/contended, so interactive timings are unreliable). Steps:
#   1. Harvest 48 real pencils at nb=16 (n=720) and nb=32 (n=1440) from one DIII-D grid radius each
#      (CPU-threaded grid, use_gpu=false, via the TJLF_DUMP_PENCILS hook) — skipped if present.
#   2. Run benchmark_batched_si_gpu.jl at both sizes, method=trsm + orth=cholqr, and (nb16 only)
#      the inv baseline, printing ms/pencil, speedup vs full geev, and leader-recovery accuracy.
# The point of (b): at n=1440 the serial Xgeev cost grows ~8x (O(n^3)) while the batched solver
# amortizes the fixed launch/transfer overhead over bigger GEMMs, so the speedup should widen.
#
#   cd TJLFEP && sbatch dev/ad/batch_krylov_validate_gpu.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -J tglfep_julia_krylov_val
#SBATCH -o dev/ad/krylov_validate_%j.out
#SBATCH -e dev/ad/krylov_validate_%j.err
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
SHIFTS="0.02,0.02+0.1im,0.02-0.1im,0.02+0.25im,0.02-0.25im,0.05+0.6im,0.05-0.6im,0.05+1.1im,0.05-1.1im"

echo "=== krylov batched-SI validation  job=${SLURM_JOB_ID:-?}  $(date) ==="
nvidia-smi -L 2>/dev/null | head -1 || true

harvest() {  # $1=NB  $2=dir
    local nb="$1" dir="$2"
    if [ -d "$dir" ] && [ "$(ls "$dir"/*.jls 2>/dev/null | wc -l)" -ge 48 ]; then
        echo "### nb=${nb}: pencils present in ${dir}, skipping harvest ###"; return
    fi
    echo "### harvesting nb=${nb} pencils into ${dir} (CPU grid, 1 radius) ###"
    rm -rf "$dir"
    TJLF_DUMP_PENCILS="${ROOT}/${dir}" TJLF_DUMP_PENCILS_MAX=48 NB="${nb}" RADII=5 SOLVERS=grid \
        stdbuf -oL "${JL[@]}" dev/ad/benchmark_nls_solvers.jl 2>&1 | tail -3
    echo "  -> $(ls "$dir"/*.jls 2>/dev/null | wc -l) pencils"
}

harvest 16 dev/ad/pencils_nb16
harvest 32 dev/ad/pencils_nb32

for cfg in "16 dev/ad/pencils_nb16 trsm cholqr" \
           "16 dev/ad/pencils_nb16 inv  cholqr" \
           "32 dev/ad/pencils_nb32 trsm cholqr" \
           "32 dev/ad/pencils_nb32 inv  cholqr"; do
    set -- $cfg
    echo ""; echo "======== nb=$1  method=$3  orth=$4 ========"
    PENCILS="${ROOT}/$2" METHOD="$3" ORTH="$4" M=16 Q=12 SHIFTS="${SHIFTS}" \
        stdbuf -oL "${JL[@]}" dev/ad/benchmark_batched_si_gpu.jl 2>&1 | tail -16
done

echo ""; echo "=== done $(date) ==="
