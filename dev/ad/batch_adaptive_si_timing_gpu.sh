#!/bin/bash -l
# Quick timing scan of the ADAPTIVE-UNION batched SI, using the GPU sysimage for fast startup.
# Scans problem size (nb16 n=720, nb32 n=1440) x GPU count (1 vs 4 A100s) on the harvested pencil
# fixtures. Reports ms/pencil and net speedup over full geev (incl. calibration).
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:25:00
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -J tglfep_julia_adasi_timing
#SBATCH -o dev/ad/adaptive_si_timing_%j.out
#SBATCH -e dev/ad/adaptive_si_timing_%j.err
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
SYSARG=(); [[ -f "${SO}" ]] && { SYSARG=(--sysimage="${SO}"); echo "sysimage: ${SO}"; } || echo "WARN no sysimage"
JL=(julia --startup-file=no "${SYSARG[@]}" --project="${ROOT}" -t 32)

echo "=== adaptive-union timing scan  $(date) ==="
nvidia-smi -L 2>/dev/null | head -4 || true

for NBSET in nb16 nb32; do
  PDIR="${ROOT}/dev/ad/pencils_${NBSET}"
  [[ -d "${PDIR}" ]] || { echo "skip ${NBSET}: ${PDIR} missing"; continue; }
  np=$(ls "${PDIR}"/*.jls 2>/dev/null | wc -l)
  for NG in 1 4; do
    echo ""; echo "######## ${NBSET}  (${np} pencils)  UNION_FIXED  NGPU=${NG} ########"
    PENCILS="${PDIR}" NGPU=${NG} CALIB_FRAC=0.10 UNION_FIXED=1 MAXSHIFTS=32 M=16 Q=12 \
      "${JL[@]}" dev/ad/benchmark_adaptive_si_gpu.jl 2>&1 | grep -E "pencils=|calibration:|ADAPTIVE|ion leader|ele leader|verdict|total leader|batched-SI wall|effective speedup" 
  done
done
echo ""; echo "=== done $(date) ==="
