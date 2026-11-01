#!/bin/bash -l
# Validation of the extended-box + separable-nbasis "truth" protocol + triggered wrapper.
#   cd dev && sbatch ad/batch_truth.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -C gpu
#SBATCH -J tjlfep_truth
#SBATCH -o truth_%j.out
#SBATCH -e truth_%j.err
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
export INNER="${INNER:-mps_team}"
export MPS_TEAM="${MPS_TEAM:-4}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"

echo "=== truth-protocol validation ==="
echo "host: $(hostname)  date: $(date)  NB=${NB} RADII=${RADII} INNER=${INNER} MPS_TEAM=${MPS_TEAM}"
nvidia-smi -L 2>/dev/null | head -1 || true

if [[ "${INNER}" == "mps_team" ]]; then
    export GPUS_PER_RADIUS=1
    srun -n 1 --ntasks-per-node=1 --gpus-per-node=1 \
        ${TJLFEP_ROOT}/slurm/common/mps-scan-wrapper.sh \
        julia --startup-file=no --project="${TJLFEP_ROOT}" \
        -t 4 ad/truth_experiment.jl
else
    stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
        -t "${SLURM_CPUS_PER_TASK:-32}" ad/truth_experiment.jl
fi

echo "=== truth job done ==="
