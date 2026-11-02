#!/bin/bash -l
# Diagnose the batched_si e2e mismatch: harvest the ACTUAL grid pencils for the worst-failing radius
# (IR101, nb16, full grid) with the dense solver, then run geev-vs-batched-SI on them using the
# SAME config the e2e used (M=16, Q=12, 13-shift set, trsm+cholqr). Reports leader misses + gamma
# errors. Many misses => fundamental SI inaccuracy; near-zero misses => kwscale replay/wiring bug.
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -J tglfep_julia_diag_ir101
#SBATCH -o dev/ad/diag_ir101_%j.out
#SBATCH -e dev/ad/diag_ir101_%j.err
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
JL=(julia --startup-file=no "${SYSARG[@]}" --project="${ROOT}" -t 64)

IR="${IR:-101}"; NB="${NB:-16}"
PDIR="${ROOT}/dev/ad/pencils_ir${IR}_nb${NB}"
rm -rf "${PDIR}"; mkdir -p "${PDIR}"

echo "=== harvest IR=${IR} nb=${NB} full grid  $(date) ==="
IR="${IR}" NB="${NB}" NFACTOR=8 NEFWID=8 NKYHAT=4 KMAX=4 \
  TJLF_DUMP_PENCILS="${PDIR}" TJLF_DUMP_PENCILS_MAX=100000 TJLF_DUMP_PENCILS_MINSIZE=0 \
  "${JL[@]}" dev/ad/harvest_radius_pencils.jl 2>&1 | tail -20
echo "harvested $(ls "${PDIR}"/*.jls 2>/dev/null | wc -l) pencils"

echo ""; echo "=== geev vs batched-SI on IR${IR} pencils (e2e config: M=16 Q=12 trsm cholqr) ==="
PENCILS="${PDIR}" M=16 Q=12 METHOD=trsm ORTH=cholqr NMODES=4 \
  "${JL[@]}" dev/ad/benchmark_batched_si_gpu.jl 2>&1 | tail -40
echo ""; echo "=== done $(date) ==="
