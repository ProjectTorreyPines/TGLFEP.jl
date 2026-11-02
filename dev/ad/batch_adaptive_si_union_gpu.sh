#!/bin/bash -l
# Iteration 2 of the adaptive-shift salvage: UNION_FIXED — keep the full fixed near-axis band
# (guarantees electron coverage) and ADD adaptive ion-branch shifts from a geev-calibrated subset,
# rather than redistributing a fixed budget (which fixed ion 336->78 but broke electron 1->112).
# Tests two shift budgets on IR101 (worst case), 4 A100s.
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:50:00
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -J tglfep_julia_adasi_union
#SBATCH -o dev/ad/adaptive_si_union_%j.out
#SBATCH -e dev/ad/adaptive_si_union_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4

set -uo pipefail
module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"
export JULIA_CUDA_USE_COMPAT=false

ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${ROOT}"
SO="${ROOT}/build/TGLFEP_JULIA_gpu_sysimage.so"
SYSARG=(); [[ -f "${SO}" ]] && SYSARG=(--sysimage="${SO}")
JL=(julia --startup-file=no "${SYSARG[@]}" --project="${ROOT}" -t 32)

IR="${IR:-101}"; NB="${NB:-16}"
PDIR="${ROOT}/dev/ad/pencils_ir${IR}_nb${NB}"
rm -rf "${PDIR}"; mkdir -p "${PDIR}"
trap 'rm -rf "${PDIR}"' EXIT

echo "=== harvest IR=${IR} nb=${NB} full grid  $(date) ==="
IR="${IR}" NB="${NB}" NFACTOR=8 NEFWID=8 NKYHAT=4 KMAX=4 \
  TJLF_DUMP_PENCILS="${PDIR}" TJLF_DUMP_PENCILS_MAX=100000 TJLF_DUMP_PENCILS_MINSIZE=0 \
  "${JL[@]}" dev/ad/harvest_radius_pencils.jl 2>&1 | tail -8
echo "harvested $(ls "${PDIR}"/*.jls 2>/dev/null | wc -l) pencils"

echo ""; echo "######## UNION_FIXED  MAXSHIFTS=32  CALIB_FRAC=0.10  NGPU=4 ########"
PENCILS="${PDIR}" NGPU=4 CALIB_FRAC=0.10 UNION_FIXED=1 MAXSHIFTS=32 M=16 Q=12 \
  "${JL[@]}" dev/ad/benchmark_adaptive_si_gpu.jl 2>&1 | tail -18

echo ""; echo "######## UNION_FIXED  MAXSHIFTS=48  CALIB_FRAC=0.10  NGPU=4 ########"
PENCILS="${PDIR}" NGPU=4 CALIB_FRAC=0.10 UNION_FIXED=1 MAXSHIFTS=48 M=16 Q=12 \
  "${JL[@]}" dev/ad/benchmark_adaptive_si_gpu.jl 2>&1 | tail -18

echo ""; echo "=== done $(date) ==="
