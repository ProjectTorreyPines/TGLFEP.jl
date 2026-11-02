#!/bin/bash
#SBATCH --job-name=tglfep_julia-container-gpu-trace
#SBATCH --account=fus176
#SBATCH --partition=batch-gpu
#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=tglfep_julia_gpu_trace_%j.out
#
# Defiant (OLCF, Apptainer) stage 1 of 2 of the GPU sysimage bake: run the
# real-GPU precompile workload under --trace-compile on an H200 and persist
# the statements file. Kept deliberately SHORT (the trace is ~20-30 min) so it
# can backfill into gaps on the two batch-gpu nodes; the heavy sysimage
# compile then runs WITHOUT a GPU on batch-cpu (defiant_compile_sysimage.sbatch).
#
#   SIF=$WORK/tglfep_julia_v2.0.14.sif OUT=$WORK/sysimage_out \
#     BAKE_DIR=$REPO/deploy/container/gpu-sysimage \
#     sbatch --export=ALL defiant_trace_gpu.sbatch
#
# Output: $OUT/precompile_statements.jl
#
# Notes vs the Perlmutter podman flow (batch_bake_gpu_sysimage.sh):
#  - apptainer SIFs are read-only with no writable overlay, so a node-local
#    writable depot is PREPENDED for scratchspaces (GPUCompiler cache etc.);
#    package sources still resolve from the baked /opt/tglfep_julia/.julia.
#  - --cleanenv keeps host Julia/module env out; the image ENV (JULIA_CPU_TARGET,
#    JULIA_CUDA_USE_COMPAT=false, ...) still applies. CUDA_VISIBLE_DEVICES must
#    be forwarded explicitly for the Slurm-allocated GPU.

set -euo pipefail

sif="${SIF:?set SIF (path to tglfep_julia_<v>.sif)}"
out="${OUT:?set OUT (dir for precompile_statements.jl)}"
bake="${BAKE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

mkdir -p "$out"

# The job may be submitted while `apptainer pull` is still converting the SIF
# (to catch a backfill slot); wait until the SIF is present and readable.
for i in $(seq 1 120); do
    apptainer sif list "$sif" >/dev/null 2>&1 && break
    [[ $i == 120 ]] && { echo "ERROR: $sif not usable after 20 min" >&2; exit 1; }
    sleep 10
done

echo "=== TJLFEP GPU trace (Defiant) ==="
echo "host: $(hostname)  date: $(date)"
echo "sif: $sif"
nvidia-smi -L
grep -m1 'model name' /proc/cpuinfo

run() {
    apptainer exec --nv --cleanenv \
        --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        --env JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" \
        --env JULIA_DEPOT_PATH="/tmp/tglfep_julia_trace_depot_$SLURM_JOB_ID:/opt/tglfep_julia/.julia" \
        --bind "$out":/out \
        --bind "$bake":/bake:ro \
        "$sif" "$@"
}

echo "=== CUDA versioninfo (driver-compat check for the 12.9 runtime artifact)"
run julia --project=/opt/tglfep_julia -e 'import CUDA; CUDA.versioninfo()'

echo "=== Tracing the GPU precompile workload (all 5 solver paths)"
run julia --project=/opt/tglfep_julia \
    --trace-compile=/out/precompile_statements.jl \
    /bake/precompile_gpu_workload_container.jl

lines=$(wc -l < "$out/precompile_statements.jl")
echo "=== TRACE OK — $lines precompile statements -> $out/precompile_statements.jl"
echo "=== next: sbatch defiant_compile_sysimage.sbatch (no GPU needed)"
