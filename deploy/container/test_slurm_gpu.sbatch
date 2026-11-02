#!/bin/bash
#SBATCH --job-name=tglfep_julia-container-gpu
#SBATCH --account=m5377_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=tglfep_julia_container_gpu_%j.out
#
# GPU smoke-test of the migrated TJLFEP podman-hpc image on a Perlmutter A100.
# `podman-hpc run --gpu` injects the host NVIDIA driver; the image carries its
# own CUDA 12.9 runtime artifact (JULIA_CUDA_USE_COMPAT=false is baked in so
# the artifact's forward-compat libcuda never shadows the injected driver).
# Pull/build + migrate first (see README), then:
#
#   TGLFEP_JULIA_ENVIRONMENT=v2.0.14 sbatch --export=ALL deploy/container/test_slurm_gpu.sbatch
#
# TGLFEP_JULIA_ENVIRONMENT expects the localhost/tglfep_julia:<tag> name that build.sh and the
# NERSC installer produce. A plain `podman-hpc pull ghcr.io/...` keeps the ghcr name
# instead — pass it in full via TGLFEP_JULIA_IMAGE:
#
#   TGLFEP_JULIA_IMAGE=ghcr.io/projecttorreypines/tglfep_julia:v2.0.14-gpu sbatch --export=ALL ...
#
# Either way the image must be MIGRATED (podman-hpc migrate <image>) under exactly that
# name, or compute nodes won't see it and will try to pull from a bogus registry.
# Works for both the plain (JIT) and -gpu (sysimage) tags. Optional:
# SQUASH_DIR=... to test a shared image store.
#
# Multi-process GPU sharing (CUDA MPS): TJLFEP's SPMD team runs share one A100
# between worker processes via MPS. Start the daemon on the host BEFORE the
# container runs, e.g.
#     export CUDA_MPS_PIPE_DIRECTORY=$SCRATCH/mps_pipe
#     export CUDA_MPS_LOG_DIRECTORY=$SCRATCH/mps_log
#     nvidia-cuda-mps-control -d
# and pass the pipe dir through with -v/-e. Single-process smoke (this job)
# needs none of that.
#
# Pass: CUDA.functional() is true, device is an A100, the single-radius DIII-D
# solve completes on GPU offline, ends with SLURM GPU SMOKE OK.

set -euo pipefail

image="${TGLFEP_JULIA_IMAGE:-localhost/tglfep_julia:${TGLFEP_JULIA_ENVIRONMENT:?set TGLFEP_JULIA_ENVIRONMENT (e.g. v2.0.14 or v2.0.14-gpu) or a full TGLFEP_JULIA_IMAGE}}"

podman=(podman-hpc)
[[ -n "${SQUASH_DIR:-}" ]] && podman=(podman-hpc --squash-dir "$SQUASH_DIR")

echo "node: $(hostname)"
nvidia-smi -L

"${podman[@]}" run --rm --gpu --network none \
    -e JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK" \
    "$image" tglfep_julia -e '
        println("sysimage: ", unsafe_string(Base.JLOptions().image_file))
        @time import TJLFEP
        import CUDA
        CUDA.functional() || error("CUDA.functional() is false — driver injection failed")
        println("device: ", CUDA.name(CUDA.device()), "  runtime: ", CUDA.runtime_version(), "  driver: ", CUDA.driver_version())
        ex = joinpath(pkgdir(TJLFEP), "examples", "DIIID_202017C42_500ms_v3.1")
        @time TJLFEP.run_gacode_scan_task(
            joinpath(ex, "input.gacode"),
            joinpath(ex, "input_singleradius_nb6.TGLFEP"),
            1; out_dir=mktempdir(), use_gpu=true, printout=true)
        println("SLURM GPU SMOKE OK  TJLFEP ", pkgversion(TJLFEP))'
