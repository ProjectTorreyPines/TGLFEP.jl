#!/bin/bash -l
# Coverage-certified adaptive SI, validated on the worst-failing radius (IR101, nb16, where
# fixed-shift SI missed 333/503 ion leaders, adaptive-union missed ~8, and the contour solver
# saturated on the axis-hugging crowd).
#   0) GPU smoke test: batched per-pencil-shift sweep + full solver vs CPU on planted spectra
#   1) accuracy gate (NGPU=1): silent leader misses on IR101's real grid pencils must be 0
#   2) throughput (NGPU=4): same accuracy, 4 A100s sharing the batch
#   3) budget sensitivity: Q=14 / MAX_ROUNDS=30 (cheaper budget vs flag rate)
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -J tglfep_julia_csi
#SBATCH -o dev/ad/csi_%j.out
#SBATCH -e dev/ad/csi_%j.err
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
# -t 64 = one Julia thread per physical core (EPYC 7763); BLAS is pinned to 1 thread inside the
# benchmark so the pencil-parallel geev/finalize loops get full core parallelism without an
# OpenBLAS pool oversubscribing on top of them.
JL=(julia --startup-file=no --project="${ROOT}" -t 64)

IR="${IR:-101}"; NB="${NB:-16}"
PDIR="${ROOT}/dev/ad/pencils_ir${IR}_nb${NB}"

echo "=== 0) GPU certified-SI smoke test  $(date) ==="
nvidia-smi -L 2>/dev/null || true
"${JL[@]}" dev/ad/_test_csi_gpu.jl

if [[ ! -d "${PDIR}" || -z "$(ls "${PDIR}"/*.jls 2>/dev/null)" ]]; then
  rm -rf "${PDIR}"; mkdir -p "${PDIR}"
  echo ""; echo "=== harvest IR=${IR} nb=${NB} full grid  $(date) ==="
  IR="${IR}" NB="${NB}" NFACTOR=8 NEFWID=8 NKYHAT=4 KMAX=4 \
    TJLF_DUMP_PENCILS="${PDIR}" TJLF_DUMP_PENCILS_MAX=100000 TJLF_DUMP_PENCILS_MINSIZE=0 \
    "${JL[@]}" dev/ad/harvest_radius_pencils.jl 2>&1 | tail -12
fi
echo "pencils available: $(ls "${PDIR}"/*.jls 2>/dev/null | wc -l)"
trap 'rm -rf "${PDIR}"' EXIT

# IR101 at factor-10 drive has consumed modes out to |freq| ~ 6.5 (window audit): the coverage
# window must contain them, so widen IM_MAX beyond the nb16 default of 2.6.
IM="${IM_MAX:-7.0}"

echo ""; echo "######## 1) ACCURACY GATE  NGPU=1  (struct defaults, IM_MAX=${IM}) ########"
PENCILS="${PDIR}" NGPU=1 IM_MAX="${IM}" "${JL[@]}" dev/ad/benchmark_certified_si_gpu.jl

echo ""; echo "######## 2) THROUGHPUT     NGPU=4 ########"
PENCILS="${PDIR}" NGPU=4 RUN_SI=0 IM_MAX="${IM}" "${JL[@]}" dev/ad/benchmark_certified_si_gpu.jl

echo ""; echo "######## 3) BUDGET SENSITIVITY  NGPU=4  Q=14 MAX_ROUNDS=30 ########"
PENCILS="${PDIR}" NGPU=4 RUN_SI=0 IM_MAX="${IM}" Q=14 MAX_ROUNDS=30 \
  "${JL[@]}" dev/ad/benchmark_certified_si_gpu.jl

echo ""; echo "=== done $(date) ==="
