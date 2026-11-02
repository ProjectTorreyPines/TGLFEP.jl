#!/bin/bash
#SBATCH --job-name=tglfep_julia-container-sysimage-compile
#SBATCH --account=fus176
#SBATCH --partition=batch-cpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=tglfep_julia_sysimage_compile_%j.out
#
# Defiant (OLCF, Apptainer) stage 2 of 2 of the GPU sysimage bake: compile the
# sysimage from the statements file traced on the H200 by defiant_trace_gpu.sbatch.
# Replaying precompile statements is type-level only, so NO GPU is needed and
# this runs on the (plentiful) batch-cpu partition.
#
#   SIF=$WORK/tglfep_julia_v2.0.14.sif OUT=$WORK/sysimage_out \
#     BAKE_DIR=$REPO/deploy/container/gpu-sysimage \
#     sbatch --export=ALL defiant_compile_sysimage.sbatch
#
# Requires $OUT/precompile_statements.jl. Output: $OUT/sys_tglfep_julia_gpu.so —
# the build context for Containerfile.gpu / the local -gpu SIF.
#
# JULIA_CPU_TARGET comes from the image ENV (multi-target: znver3/znver4/
# sapphirerapids/generic), NOT from this host, so compiling on a batch-cpu
# node yields the identical portable sysimage a GPU-node compile would.

set -euo pipefail

sif="${SIF:?set SIF (path to tglfep_julia_<v>.sif)}"
out="${OUT:?set OUT (dir holding precompile_statements.jl)}"
bake="${BAKE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

[[ -s "$out/precompile_statements.jl" ]] || {
    echo "ERROR: $out/precompile_statements.jl missing — run defiant_trace_gpu.sbatch first" >&2
    exit 1
}

echo "=== TJLFEP sysimage compile (Defiant, CPU-only) ==="
echo "host: $(hostname)  date: $(date)"
grep -m1 'model name' /proc/cpuinfo
echo "statements: $(wc -l < "$out/precompile_statements.jl") lines"

apptainer exec --cleanenv \
    --env JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" \
    --env JULIA_DEPOT_PATH="/tmp/tglfep_julia_compile_depot_$SLURM_JOB_ID:/opt/tglfep_julia/.julia" \
    --env SYSIMAGE_PATH=/out/sys_tglfep_julia_gpu.so \
    --env PRECOMPILE_STATEMENTS_FILE=/out/precompile_statements.jl \
    --bind "$out":/out \
    --bind "$bake":/bake:ro \
    "$sif" julia --project=/opt/tglfep_julia /bake/build_sysimage_container.jl

ls -lh "$out/sys_tglfep_julia_gpu.so"
echo "=== COMPILE OK — layer it with Containerfile.gpu or bind-mount it at"
echo "    /opt/tglfep_julia/sys_tglfep_julia_gpu.so for the tglfep_julia launcher to pick up"
