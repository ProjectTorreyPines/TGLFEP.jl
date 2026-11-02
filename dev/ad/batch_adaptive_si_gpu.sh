#!/bin/bash -l
# Adaptive-shift salvage of batched SI, validated on the worst-failing radius (IR101, nb16, where
# the fixed shift set missed 333/503 ion leaders). Harvests IR101's real grid pencils, then:
#   1) accuracy gate (NGPU=1): fixed vs adaptive (geev-calibrated) leader-miss counts
#   2) throughput (NGPU=4): same accuracy, 4 A100s sharing the batch
#   3) calibration-fraction sensitivity (5% vs 10% vs 20%)
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:50:00
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -J tglfep_julia_adaptive_si
#SBATCH -o dev/ad/adaptive_si_%j.out
#SBATCH -e dev/ad/adaptive_si_%j.err
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
nvidia-smi -L 2>/dev/null || true
IR="${IR}" NB="${NB}" NFACTOR=8 NEFWID=8 NKYHAT=4 KMAX=4 \
  TJLF_DUMP_PENCILS="${PDIR}" TJLF_DUMP_PENCILS_MAX=100000 TJLF_DUMP_PENCILS_MINSIZE=0 \
  "${JL[@]}" dev/ad/harvest_radius_pencils.jl 2>&1 | tail -12
echo "harvested $(ls "${PDIR}"/*.jls 2>/dev/null | wc -l) pencils"

echo ""; echo "######## 1) ACCURACY GATE  NGPU=1  CALIB_FRAC=0.10 ########"
PENCILS="${PDIR}" NGPU=1 CALIB_FRAC=0.10 M=16 Q=12 "${JL[@]}" dev/ad/benchmark_adaptive_si_gpu.jl 2>&1 | tail -30

echo ""; echo "######## 2) THROUGHPUT     NGPU=4  CALIB_FRAC=0.10 ########"
PENCILS="${PDIR}" NGPU=4 CALIB_FRAC=0.10 M=16 Q=12 "${JL[@]}" dev/ad/benchmark_adaptive_si_gpu.jl 2>&1 | tail -30

echo ""; echo "######## 3) CALIB SENSITIVITY  NGPU=4  CALIB_FRAC=0.05 ########"
PENCILS="${PDIR}" NGPU=4 CALIB_FRAC=0.05 M=16 Q=12 "${JL[@]}" dev/ad/benchmark_adaptive_si_gpu.jl 2>&1 | tail -16

echo ""; echo "=== done $(date) ==="
