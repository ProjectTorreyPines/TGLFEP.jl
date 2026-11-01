#!/bin/bash -l
# Head-to-head: lean DIRECT (20 evals) vs f1-seed guarded :ad, scored against the DIRECT-40 accuracy
# ceiling. Which reaches full accuracy (esp. off-node spikes IR=48/95) at the lowest cost? Runs from
# LIVE SOURCE on ONE GPU under MPS team=4 (matches how DIRECT times best). JITs once then sweeps radii.
#   cd dev && sbatch ad/batch_headtohead_experiment.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -C gpu
#SBATCH -J tjlfep_h2h_exp
#SBATCH -o headtohead_experiment_%j.out
#SBATCH -e headtohead_experiment_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"
export JULIA_CUDA_USE_COMPAT=false

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TJLFEP_ROOT}/dev"

export USE_GPU="${USE_GPU:-1}"
export NB="${NB:-32}"
export RADII="${RADII:-22,38,48,95}"
export DIRECT20_EVALS="${DIRECT20_EVALS:-20}"
export INNER="${INNER:-mps_team}"
export MPS_TEAM="${MPS_TEAM:-4}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"

echo "=== head-to-head: lean DIRECT vs f1-seed guarded :ad ==="
echo "host: $(hostname)  date: $(date)  USE_GPU=${USE_GPU} NB=${NB} RADII=${RADII} D20=${DIRECT20_EVALS} INNER=${INNER} MPS_TEAM=${MPS_TEAM}"
nvidia-smi -L 2>/dev/null | head -1 || true

if [[ "${INNER}" == "mps_team" ]]; then
    export GPUS_PER_RADIUS=1
    srun -n 1 --ntasks-per-node=1 --gpus-per-node=1 \
        ${TJLFEP_ROOT}/slurm/common/mps-scan-wrapper.sh \
        julia --startup-file=no --project="${TJLFEP_ROOT}" \
        -t 4 ad/headtohead_experiment.jl
else
    stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
        -t "${SLURM_CPUS_PER_TASK:-32}" ad/headtohead_experiment.jl
fi

echo "=== headtohead experiment job done ==="
