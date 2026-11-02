#!/bin/bash -l
# Measure per-process Julia startup/load time for the file-only vs generic GPU sysimages
# on a GPU compute node (representative of what each MPS/threads worker pays before it can
# start solving). Two variants per image:
#   * startup    : `julia --sysimage=IMG -e exit()` -> Julia start + sysimage mmap + baked
#                  package __init__ (CUDA, TJLF/TJLFEP, and FUSE/IMAS in the generic image).
#   * using+cuda : also `using TJLFEP, CUDA; CUDA.functional()` -> forces the GPU device
#                  probe, i.e. the full "worker ready on GPU" cost.
# The leaner 1.1 GB file-only image should load faster than the 3.0 GB generic one because
# the IMAS/FUSE stack is not baked.
#   cd build && sbatch sysimage/batch_measure_sysimage_load.sh
#SBATCH -A m3739_g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -C gpu
#SBATCH -J sysimg_loadtime
#SBATCH -o sysimg_loadtime_%j.out
#SBATCH -e sysimg_loadtime_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32

set -uo pipefail

module load cudatoolkit/12.9 julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"
export JULIA_CUDA_USE_COMPAT=false

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TGLFEP_JULIA_ROOT}"
echo "host: $(hostname)  date: $(date)"
nvidia-smi -L 2>/dev/null | head -1 || true

REPS="${REPS:-4}"
timeit () {  # $1 = label, $2 = image, $3 = julia -e expr
    local t0 t1
    t0=$(date +%s.%N)
    julia --startup-file=no --sysimage="$2" --project=. -e "$3" >/dev/null 2>&1
    t1=$(date +%s.%N)
    python3 -c "print(f'  LOADTIME variant=$1 rep=$4 sec={float(\"$t1\")-float(\"$t0\"):.2f}')"
}

for img in build/TGLFEP_JULIA_gpu_sysimage.so build/TGLFEP_JULIA_gpu_generic_sysimage.so; do
    [[ -f "$img" ]] || { echo "missing $img"; continue; }
    echo "=== $img ($(ls -lh "$img" | awk '{print $5}')) ==="
    julia --startup-file=no --sysimage="$img" --project=. -e 'exit()' >/dev/null 2>&1   # warm FS cache
    for i in $(seq 1 "$REPS"); do timeit startup    "$img" 'exit()' "$i"; done
    for i in $(seq 1 "$REPS"); do timeit using+cuda "$img" 'using TJLFEP, CUDA; CUDA.functional()' "$i"; done
done
echo "SYSIMG_LOADTIME_DONE"
