#!/bin/bash
#SBATCH --job-name=tglfep_julia-container-mps
#SBATCH --account=m5377_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=tglfep_julia_container_mps_%j.out
#
# CUDA MPS team run of the TJLFEP container on a Perlmutter A100: one radius
# whose inner kw-scan is distributed over MPS_TEAM in-container worker
# processes that all share this task's single GPU (Hyper-Q overlap of the
# Xgeev eigensolves). This is the containerized analogue of
# slurm/common/run_gacode_scan20_mps_task.jl, self-contained on the packaged
# DIII-D example — no repo checkout needed on the compute node.
#
#   TGLFEP_JULIA_ENVIRONMENT=v2.0.14-gpu sbatch --export=ALL deploy/container/test_slurm_container_gpu_mps.sbatch
#
# Optional: MPS_TEAM (default 4), JULIA_WORKER_THREADS (default 2), SQUASH_DIR.
#
# The MPS caveats this script demonstrates (see also mps-scan-wrapper.sh for
# the multi-task/multi-node version of the launch-order barrier):
#   1. The MPS control daemon starts on the HOST, BEFORE the container runs:
#      a client that calls cuInit with no daemon up silently falls back to a
#      private context and time-slices instead of overlapping.
#   2. The pipe directory is bind-mounted and its env var passed through —
#      podman-hpc shares nothing by default. (Apptainer sites get both for
#      free: /tmp and host env are forwarded unless --cleanenv.)
#   3. JULIA_CUDA_USE_COMPAT=false (baked into the image ENV) is REQUIRED:
#      CUDA.jl's forward-compat shim hangs in cuInit as an MPS client.
#
# Pass: team spawns, every worker sees the GPU through MPS, the solve
# completes, ends with CONTAINER MPS TEAM OK.

set -euo pipefail

version="${TGLFEP_JULIA_ENVIRONMENT:?set TGLFEP_JULIA_ENVIRONMENT (e.g. v2.0.14 or v2.0.14-gpu)}"
export MPS_TEAM="${MPS_TEAM:-4}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"

podman=(podman-hpc)
[[ -n "${SQUASH_DIR:-}" ]] && podman=(podman-hpc --squash-dir "$SQUASH_DIR")

echo "node: $(hostname)"
nvidia-smi -L

# --- [host] start the MPS daemon and BLOCK until its control pipe exists ----
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
for _ in $(seq 1 60); do
    [[ -e "$CUDA_MPS_PIPE_DIRECTORY/control" ]] && break
    sleep 0.5
done
[[ -e "$CUDA_MPS_PIPE_DIRECTORY/control" ]] || { echo "ERROR: MPS control pipe never appeared" >&2; exit 1; }
trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' EXIT
echo "MPS daemon up: $CUDA_MPS_PIPE_DIRECTORY"

# --- team driver (shared with the Defiant variant), run by the tglfep_julia launcher
driver="${DRIVER:-$SLURM_SUBMIT_DIR/deploy/container/mps_team_driver.jl}"
[[ -f "$driver" ]] || { echo "ERROR: driver not at $driver — submit from the repo root or set DRIVER=" >&2; exit 1; }

# --- run: bind the pipe dir + driver, pass the MPS/team env explicitly -------
"${podman[@]}" run --rm --gpu --network none \
    -v "$CUDA_MPS_PIPE_DIRECTORY":"$CUDA_MPS_PIPE_DIRECTORY" \
    -v "$(dirname "$driver")":/driver:ro \
    -e CUDA_MPS_PIPE_DIRECTORY -e CUDA_MPS_LOG_DIRECTORY \
    -e MPS_TEAM -e JULIA_WORKER_THREADS \
    -e JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK" \
    "localhost/tglfep_julia:$version" tglfep_julia "/driver/$(basename "$driver")"
