#!/bin/bash -l
# Adaptive-union batched SI timing vs N_BASIS (matching the README's timing-vs-nbasis framing):
# harvest a small pencil batch at nb6/nb8/nb16/nb32, then time the adaptive-union solve at each on
# 4 A100s, writing a CSV (nb,n,npencils,ngpu,ms_fixed,ms_adaptive,net_speedup,misses).
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:45:00
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -J tglfep_julia_adasi_nbasis
#SBATCH -o dev/ad/adaptive_si_nbasis_%j.out
#SBATCH -e dev/ad/adaptive_si_nbasis_%j.err
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

CSV="${ROOT}/dev/ad/adaptive_si_vs_nbasis.csv"
echo "nb,n,npencils,ngpu,ms_fixed,ms_adaptive,net_speedup,misses" > "${CSV}"
echo "=== adaptive-union timing vs nbasis  $(date) ==="
nvidia-smi -L 2>/dev/null | head -1 || true

# Harvest a modest, uniform batch per nbasis (IR=54, a mid-scan radius that IS in IR_EXP).
# nb16/nb32 fixtures exist but re-harvest all four here so the batch size is consistent.
for NB in 6 8 16 32; do
  PDIR="${ROOT}/dev/ad/pencils_nbasis_scan_nb${NB}"
  rm -rf "${PDIR}"; mkdir -p "${PDIR}"
  # full-nbasis pencils only (n = 45*nbasis): uniform size, capped for disk + good GPU utilization.
  MINSZ=$(( 45 * NB ))
  IR=54 NB="${NB}" NFACTOR=8 NEFWID=8 NKYHAT=4 KMAX=4 \
    TJLF_DUMP_PENCILS="${PDIR}" TJLF_DUMP_PENCILS_MAX=400 TJLF_DUMP_PENCILS_MINSIZE="${MINSZ}" \
    "${JL[@]}" dev/ad/harvest_radius_pencils.jl 2>&1 | tail -3
  np=$(ls "${PDIR}"/*.jls 2>/dev/null | wc -l)
  echo ""; echo "######## nb=${NB}  (${np} pencils)  UNION_FIXED  NGPU=4 ########"
  PENCILS="${PDIR}" NB="${NB}" NGPU=4 CALIB_FRAC=0.10 UNION_FIXED=1 MAXSHIFTS=32 M=16 Q=12 \
    CSV_OUT="${CSV}" "${JL[@]}" dev/ad/benchmark_adaptive_si_gpu.jl 2>&1 \
    | grep -E "pencils=|calibration:|ADAPTIVE|total leader|batched-SI wall|effective speedup"
  rm -rf "${PDIR}"
done

echo ""; echo "=== CSV ==="; cat "${CSV}"
echo "=== done $(date) ==="
