#!/bin/bash
#SBATCH --job-name=tglfep_julia-container-cpu
#SBATCH --account=m3739
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=0:25:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=tglfep_julia_container_cpu_%j.out
#
# Smoke-test the migrated TJLFEP podman-hpc image inside a Slurm job on a
# Perlmutter CPU node (AMD Milan, znver3 — one of the pkgimage clone targets).
# Pull/build + migrate first (see README), then:
#
#   TGLFEP_JULIA_ENVIRONMENT=v2.0.14 sbatch --export=ALL deploy/container/test_slurm_cpu.sbatch
#
# TGLFEP_JULIA_ENVIRONMENT expects the localhost/tglfep_julia:<tag> name that build.sh and the
# NERSC installer produce. A plain `podman-hpc pull ghcr.io/...` keeps the ghcr name
# instead — pass it in full via TGLFEP_JULIA_IMAGE:
#
#   TGLFEP_JULIA_IMAGE=ghcr.io/projecttorreypines/tglfep_julia:v2.0.14 sbatch --export=ALL ...
#
# Either way the image must be MIGRATED (podman-hpc migrate <image>) under exactly that
# name, or compute nodes won't see it and will try to pull from a bogus registry.
# Optional: SQUASH_DIR=... to test a shared image store instead of the
# per-user one.
#
# Pass: Sys.CPU_NAME reports znver3, the single-radius DIII-D critical-gradient
# solve completes offline on CPU, ends with SLURM CPU SMOKE OK.

set -euo pipefail

image="${TGLFEP_JULIA_IMAGE:-localhost/tglfep_julia:${TGLFEP_JULIA_ENVIRONMENT:?set TGLFEP_JULIA_ENVIRONMENT (e.g. v2.0.14) or a full TGLFEP_JULIA_IMAGE}}"

podman=(podman-hpc)
[[ -n "${SQUASH_DIR:-}" ]] && podman=(podman-hpc --squash-dir "$SQUASH_DIR")

echo "node: $(hostname)"
lscpu | grep 'Model name'

"${podman[@]}" run --rm --network none \
    -e JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK" \
    "$image" tglfep_julia -e '
        println("cpu: ", Sys.CPU_NAME, "  threads: ", Threads.nthreads())
        @time import TJLFEP
        import CUDA
        println("CUDA.functional() = ", CUDA.functional(), " (CPU node — expect false)")
        ex = joinpath(pkgdir(TJLFEP), "examples", "DIIID_202017C42_500ms_v3.1")
        TJLFEP.run_gacode_scan_task(
            joinpath(ex, "input.gacode"),
            joinpath(ex, "input_singleradius_nb6.TGLFEP"),
            1; out_dir=mktempdir(), use_gpu=false, printout=true)
        println("SLURM CPU SMOKE OK  TJLFEP ", pkgversion(TJLFEP))'
