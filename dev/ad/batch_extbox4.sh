#!/bin/bash -l
# Ext-box round 4: high-nbasis {32,48,64,96} convergence at narrow-width optima + standard-width control.
#   cd dev && sbatch ad/batch_extbox4.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH -J tglfep_julia_extbox4
#SBATCH -o extbox4_%j.out
#SBATCH -e extbox4_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"
export JULIA_CUDA_USE_COMPAT=false

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TGLFEP_JULIA_ROOT}/dev"

export USE_GPU="${USE_GPU:-1}"
export NB="${NB:-32}"
export INNER="${INNER:-mps_team}"
export MPS_TEAM="${MPS_TEAM:-4}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"

echo "=== ext-box round 4: high-nbasis convergence ==="
echo "host: $(hostname)  date: $(date)  NB=${NB} INNER=${INNER} MPS_TEAM=${MPS_TEAM}"
nvidia-smi -L 2>/dev/null | head -1 || true

if [[ "${INNER}" == "mps_team" ]]; then
    export GPUS_PER_RADIUS=1
    srun -n 1 --ntasks-per-node=1 --gpus-per-node=1 \
        ${TGLFEP_JULIA_ROOT}/slurm/common/mps-scan-wrapper.sh \
        julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
        -t 4 ad/extbox4_experiment.jl
else
    stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
        -t "${SLURM_CPUS_PER_TASK:-32}" ad/extbox4_experiment.jl
fi

echo "=== extbox4 job done ==="
