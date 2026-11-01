#!/bin/bash -l
# Validate adf1->escalation under canonical ky>=0.25: adf1 + DIRECT-40(ky>=0.25, UNTESTED) + grid-zoom
# robust, with the escalation decision synthesized for :direct vs :grid targets.
#   cd dev && sbatch ad/batch_escalation_experiment.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -C gpu
#SBATCH -J tjlfep_escalate
#SBATCH -o escalation_experiment_%j.out
#SBATCH -e escalation_experiment_%j.err
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
export DIRECT_EVALS="${DIRECT_EVALS:-40}"
export KY_LO="${KY_LO:-0.25}"
export INNER="${INNER:-mps_team}"
export MPS_TEAM="${MPS_TEAM:-4}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"

echo "=== escalation validation: adf1 + DIRECT-40(ky>=0.25) + grid-zoom robust ==="
echo "host: $(hostname)  date: $(date)  NB=${NB} RADII=${RADII} DIRECT_EVALS=${DIRECT_EVALS} KY_LO=${KY_LO} INNER=${INNER} MPS_TEAM=${MPS_TEAM}"
nvidia-smi -L 2>/dev/null | head -1 || true

if [[ "${INNER}" == "mps_team" ]]; then
    export GPUS_PER_RADIUS=1
    srun -n 1 --ntasks-per-node=1 --gpus-per-node=1 \
        ${TJLFEP_ROOT}/slurm/common/mps-scan-wrapper.sh \
        julia --startup-file=no --project="${TJLFEP_ROOT}" \
        -t 4 ad/escalation_experiment.jl
else
    stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
        -t "${SLURM_CPUS_PER_TASK:-32}" ad/escalation_experiment.jl
fi

echo "=== escalation experiment job done ==="
