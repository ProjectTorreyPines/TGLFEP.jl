#!/bin/bash
#SBATCH --job-name=tglfep_julia-container-mps
#SBATCH --account=fus176
#SBATCH --partition=batch-gpu
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=tglfep_julia_container_mps_%j.out
#
# CUDA MPS team run of the TJLFEP container on a Defiant H200 (Apptainer):
# one radius whose inner kw-scan is distributed over MPS_TEAM in-container
# worker processes sharing this task's GPU. Same driver as the Perlmutter
# variant (mps_team_driver.jl); only the host-side wrapper differs.
#
#   SIF=$WORK/tglfep_julia_v2.0.14-gpu.sif sbatch --export=ALL \
#       deploy/container/test_slurm_container_gpu_mps_defiant.sbatch
#
# Optional: MPS_TEAM (default 4), JULIA_WORKER_THREADS (default 2), DRIVER.
#
# MPS caveats, Apptainer edition:
#   1. The MPS control daemon starts on the HOST, BEFORE `apptainer exec`: a
#      client that calls cuInit with no daemon up silently falls back to a
#      private context and time-slices instead of overlapping.
#   2. NO --cleanenv here (unlike the single-process smoke jobs): Apptainer's
#      default host-env forwarding + /tmp bind are exactly what make the
#      CUDA_MPS_PIPE_DIRECTORY env var and the pipes themselves visible
#      in-container. With --cleanenv you would have to --env/--bind them all
#      back by hand.
#   3. JULIA_CUDA_USE_COMPAT=false (baked into the image ENV) is REQUIRED:
#      CUDA.jl's forward-compat shim hangs in cuInit as an MPS client.
#
# Pass: team spawns, every worker sees the GPU through MPS, the solve
# completes, ends with CONTAINER MPS TEAM OK.

set -euo pipefail

sif="${SIF:?set SIF (path to tglfep_julia_<v>.sif, e.g. \$WORK/tglfep_julia_v2.0.14-gpu.sif)}"
export MPS_TEAM="${MPS_TEAM:-4}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"

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

# --- team driver (shared with the Perlmutter variant) -------------------------
driver="${DRIVER:-$SLURM_SUBMIT_DIR/deploy/container/mps_team_driver.jl}"
[[ -f "$driver" ]] || { echo "ERROR: driver not at $driver — submit from the repo root or set DRIVER=" >&2; exit 1; }

# --- run: host env (MPS vars, CUDA_VISIBLE_DEVICES, team knobs) and /tmp are
# forwarded by default; only the driver's directory may need an explicit bind
# (e.g. when the repo lives on a filesystem Apptainer does not auto-bind).
apptainer exec --nv \
    --bind "$(dirname "$driver")" \
    "$sif" tglfep_julia "$driver"
