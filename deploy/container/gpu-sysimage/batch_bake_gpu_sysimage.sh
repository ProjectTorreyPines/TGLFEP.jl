#!/bin/bash
#SBATCH --job-name=tglfep_julia-gpu-sysimage-bake
#SBATCH --account=m5377_g
#SBATCH --qos=premium
#SBATCH --constraint=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=tglfep_julia_gpu_sysimage_bake_%j.out
#
# Bake the GPU-traced sysimage inside the TJLFEP container on a Perlmutter
# A100. The trace executes real GPU solves (all 5 solver paths), so this
# CANNOT run in CI. The image must already be pulled + migrated on this
# system (bake_and_publish.sh does that first). Submit from this directory
# (or let bake_and_publish.sh do it):
#
#   TGLFEP_JULIA_ENVIRONMENT=v2.0.14 sbatch --export=ALL batch_bake_gpu_sysimage.sh
#
# Output: $SYSIMAGE_OUT/sys_tglfep_julia_gpu.so (default $PSCRATCH/tglfep_julia_sysimage_out),
# which is the build context for Containerfile.gpu.

set -euo pipefail

version="${TGLFEP_JULIA_ENVIRONMENT:?set TGLFEP_JULIA_ENVIRONMENT (e.g. v2.0.14)}"
image="localhost/tglfep_julia:$version"
out="${SYSIMAGE_OUT:-$PSCRATCH/tglfep_julia_sysimage_out}"
# Under sbatch this script runs from slurmd's spool dir, so BASH_SOURCE cannot
# locate gpu-sysimage/. bake_and_publish.sh exports GPU_SYSIMAGE_DIR; a direct
# sbatch from this directory falls back to SLURM_SUBMIT_DIR.
scriptdir="${GPU_SYSIMAGE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
[[ -f "$scriptdir/build_sysimage_container.jl" ]] || {
    echo "ERROR: $scriptdir does not contain build_sysimage_container.jl" \
         "(set GPU_SYSIMAGE_DIR or submit from deploy/container/gpu-sysimage)" >&2
    exit 1
}

mkdir -p "$out"

echo "=== TJLFEP container GPU sysimage bake ==="
echo "host: $(hostname)  date: $(date)"
nvidia-smi -L

podman=(podman-hpc)
[[ -n "${SQUASH_DIR:-}" ]] && podman=(podman-hpc --squash-dir "$SQUASH_DIR")

# Plain julia, NOT the tglfep_julia launcher: the launcher unsets JULIA_CPU_TARGET
# (correct for JIT use), but the bake needs the image's multi-target value.
# The writable overlay absorbs PackageCompiler's temp objects; only /out
# (bind mount) persists.
"${podman[@]}" run --rm --gpu \
    -v "$out":/out \
    -v "$scriptdir":/bake:ro \
    -e SYSIMAGE_PATH=/out/sys_tglfep_julia_gpu.so \
    -e JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" \
    "$image" julia --project=/opt/tglfep_julia /bake/build_sysimage_container.jl

ls -lh "$out/sys_tglfep_julia_gpu.so"
echo "=== BAKE OK — layer it with Containerfile.gpu (see bake_and_publish.sh)"
