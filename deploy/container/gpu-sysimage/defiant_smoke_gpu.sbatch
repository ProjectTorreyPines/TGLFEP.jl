#!/bin/bash
#SBATCH --job-name=tglfep_julia-container-gpu-smoke
#SBATCH --account=fus176
#SBATCH --partition=batch-gpu
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=tglfep_julia_gpu_smoke_%j.out
#
# Defiant GPU acceptance of the -gpu sysimage artifact: bind the baked
# sys_tglfep_julia_gpu.so into the plain SIF at the exact path the layered
# :<v>-gpu image carries it, run through the tglfep_julia launcher (which must
# pick the sysimage up), and do a real single-radius GPU solve offline.
# Kept short (20 min) so it can backfill into batch-gpu gaps.
#
#   SIF=$WORK/tglfep_julia_v2.0.14.sif OUT=$WORK/sysimage_out \
#     sbatch --export=ALL defiant_smoke_gpu.sbatch
#
# Pass: sysimage path asserted, CUDA.functional() true on an H200, solve
# completes, ends with DEFIANT GPU SMOKE OK.

set -euo pipefail

sif="${SIF:?set SIF (path to tglfep_julia_<v>.sif)}"
out="${OUT:?set OUT (dir holding sys_tglfep_julia_gpu.so)}"

echo "node: $(hostname)  date: $(date)"
nvidia-smi -L

apptainer exec --nv --cleanenv \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    --env JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" \
    --bind "$out/sys_tglfep_julia_gpu.so":/opt/tglfep_julia/sys_tglfep_julia_gpu.so:ro \
    "$sif" tglfep_julia -e '
        img = unsafe_string(Base.JLOptions().image_file)
        println("sysimage: ", img)
        img == "/opt/tglfep_julia/sys_tglfep_julia_gpu.so" || error("launcher did not pick up the sysimage")
        @time import TJLFEP
        import CUDA
        CUDA.functional() || error("CUDA.functional() is false — driver injection failed")
        println("device: ", CUDA.name(CUDA.device()), "  runtime: ", CUDA.runtime_version(), "  driver: ", CUDA.driver_version())
        ex = joinpath(pkgdir(TJLFEP), "examples", "DIIID_202017C42_500ms_v3.1")
        @time TJLFEP.run_gacode_scan_task(
            joinpath(ex, "input.gacode"),
            joinpath(ex, "input_singleradius_nb6.TGLFEP"),
            1; out_dir=mktempdir(), use_gpu=true, printout=true)
        println("DEFIANT GPU SMOKE OK  TJLFEP ", pkgversion(TJLFEP))'
