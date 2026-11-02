#!/bin/bash -l
# Single-radius ITER probe: why is ITER's per-radius kw-scan slower than DIII-D's?
# Runs ONE interior radius with inner=threads + TGLFEP_JULIA_PROBE=1 so kwscale_scan prints
# combos / sum(eigensolve) / eigensolve_frac on the master. Compare to the DIII-D nb32
# reference (IR=7: combos=1024, ~4.3 s/combo). Reuses the already-prepared ITER dd
# artifacts (dd_in.json/optionsdict.jls/rho_scan.jls) in dev/timing/probe_iter.
#SBATCH -A m3739_g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH -J probe_iter_radius
#SBATCH -o probe_iter_radius_%j.out
#SBATCH -e probe_iter_radius_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

export TGLFEP_JULIA_ROOT="/pscratch/sd/t/tneiser/.julia/dev/TJLFEP"
export FUSE_ROOT="/pscratch/sd/t/tneiser/.julia/dev/FUSE"
export TGLFEP_JULIA_OUT_DIR="${TGLFEP_JULIA_ROOT}/dev/timing/probe_iter"
export TGLFEP_JULIA_GPU_SYSIMAGE="/global/cfs/cdirs/m3739/TJLFEP/TGLFEP_JULIA_gpu_generic_sysimage.so"

export USE_GPU=1
export INNER="${INNER:-threads}"      # threads => master prints the probe line directly
export MPS_TEAM="${MPS_TEAM:-8}"
export SCAN_INDEX="${SCAN_INDEX:-8}"  # an interior AE-active radius (rho~0.36)
export TGLFEP_JULIA_PROBE=1
export JULIA_CUDA_USE_COMPAT=false
export CUDA_VISIBLE_DEVICES=0

SYSFLAG=""
[ -f "${TGLFEP_JULIA_GPU_SYSIMAGE}" ] && SYSFLAG="--sysimage=${TGLFEP_JULIA_GPU_SYSIMAGE}"
echo "=== ITER per-radius probe: SCAN_INDEX=${SCAN_INDEX} INNER=${INNER} N_BASIS=32 ==="
echo "sysimage=${TGLFEP_JULIA_GPU_SYSIMAGE}"
nvidia-smi -L 2>/dev/null | head -2 || true

# For mps_team we'd need the per-node MPS daemon; threads (default) needs none.
if [ "${INNER}" = "mps_team" ]; then
  export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.${SLURM_JOB_ID}"
  export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.${SLURM_JOB_ID}"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  nvidia-cuda-mps-control -d || true
fi

stdbuf -oL -eL julia --startup-file=no ${SYSFLAG} --project="${FUSE_ROOT}" \
    "${TGLFEP_JULIA_ROOT}/dev/timing/run_fuse_dd_mps_task.jl"

if [ "${INNER}" = "mps_team" ]; then
  echo quit | nvidia-cuda-mps-control || true
fi
echo "=== probe done; grep '[TGLFEP_JULIA_PROBE]' above ==="
