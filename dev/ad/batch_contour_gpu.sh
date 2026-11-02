#!/bin/bash -l
# Contour-integral (Beyn) batched eigensolver, validated on the worst-failing radius (IR101,
# nb16, where fixed-shift SI missed 333/503 ion leaders and adaptive-union still missed ~8).
#   0) GPU smoke test: batched moment path vs CPU reference on planted spectra
#   1) accuracy gate (NGPU=1): leader misses on IR101's real grid pencils, default window
#   2) throughput (NGPU=4): same accuracy, 4 A100s sharing the batch
#   3) quadrature sensitivity: NQUAD=16 vs 32
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:50:00
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -J tglfep_julia_contour
#SBATCH -o dev/ad/contour_%j.out
#SBATCH -e dev/ad/contour_%j.err
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
JL=(julia --startup-file=no --project="${ROOT}" -t 32)

IR="${IR:-101}"; NB="${NB:-16}"
PDIR="${ROOT}/dev/ad/pencils_ir${IR}_nb${NB}"

echo "=== 0) GPU contour smoke test  $(date) ==="
nvidia-smi -L 2>/dev/null || true
"${JL[@]}" dev/ad/_test_contour_gpu.jl

if [[ ! -d "${PDIR}" || -z "$(ls "${PDIR}"/*.jls 2>/dev/null)" ]]; then
  rm -rf "${PDIR}"; mkdir -p "${PDIR}"
  echo ""; echo "=== harvest IR=${IR} nb=${NB} full grid  $(date) ==="
  IR="${IR}" NB="${NB}" NFACTOR=8 NEFWID=8 NKYHAT=4 KMAX=4 \
    TJLF_DUMP_PENCILS="${PDIR}" TJLF_DUMP_PENCILS_MAX=100000 TJLF_DUMP_PENCILS_MINSIZE=0 \
    "${JL[@]}" dev/ad/harvest_radius_pencils.jl 2>&1 | tail -12
fi
echo "pencils available: $(ls "${PDIR}"/*.jls 2>/dev/null | wc -l)"
trap 'rm -rf "${PDIR}"' EXIT

echo ""; echo "######## 1) ACCURACY GATE  NGPU=1  (default window, 48 nodes, L=64 K=3) ########"
PENCILS="${PDIR}" NGPU=1 "${JL[@]}" dev/ad/benchmark_contour_gpu.jl

echo ""; echo "######## 2) THROUGHPUT     NGPU=4 ########"
PENCILS="${PDIR}" NGPU=4 RUN_SI=0 "${JL[@]}" dev/ad/benchmark_contour_gpu.jl

echo ""; echo "######## 3) QUADRATURE SENSITIVITY  NGPU=4  32 nodes (N_LONG=12 N_SHORT=4) ########"
PENCILS="${PDIR}" NGPU=4 RUN_SI=0 N_LONG=12 N_SHORT=4 "${JL[@]}" dev/ad/benchmark_contour_gpu.jl

echo ""; echo "######## 4) QUADRATURE SENSITIVITY  NGPU=4  64 nodes (N_LONG=24 N_SHORT=8) ########"
PENCILS="${PDIR}" NGPU=4 RUN_SI=0 N_LONG=24 N_SHORT=8 "${JL[@]}" dev/ad/benchmark_contour_gpu.jl

echo ""; echo "=== done $(date) ==="
