#!/bin/bash -l
# Julia file-based: N_BASIS=6, SCAN_N=20, 10 nodes, 20 workers (SlurmClusterManager + pmap).
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 10
#SBATCH -t 02:00:00
#SBATCH -C cpu
#SBATCH -J TJLFEP_nb6_s20_10n
#SBATCH -o debug_nb6_julia20_10n_%j.out
#SBATCH -e debug_nb6_julia20_10n_%j.err
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=64

set -euo pipefail

module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"
export TGLFEP_JULIA_FILE_ONLY=1
export SCAN_N=20
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"
export TGLFEP_JULIA_DEBUG=0

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TGLFEP_JULIA_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"
export GACODE_DUMP="${GACODE_DUMP:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_scan20_nb6.TGLFEP}"
export FILE_DIR="${FILE_DIR:-${TGLFEP_JULIA_ROOT}/dev/fileInput_nb6_scan20_10n_${SLURM_JOB_ID:-local}}"

cd "${TGLFEP_JULIA_ROOT}/dev"
# shellcheck source=../${TGLFEP_JULIA_ROOT}/slurm/common/tglfep_julia_env.inc.sh
source "${TGLFEP_JULIA_ROOT}/slurm/common/tglfep_julia_env.inc.sh"
run_julia() { stdbuf -oL -eL julia "$@"; }

echo "=== Julia nb6 SCAN_N=${SCAN_N} on ${SLURM_NNODES:-?} nodes ==="
echo "SLURM_NTASKS=${SLURM_NTASKS:-?} JULIA_WORKER_THREADS=${JULIA_WORKER_THREADS}"
echo "TGLFEP_FILE=${TGLFEP_FILE} FILE_DIR=${FILE_DIR} GACODE_DUMP=${GACODE_DUMP}"

run_julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    "${JULIA_SYSIMAGE_ARGS[@]}" \
    verify/debug_compare_nb6_scan20_distributed.jl

echo "=== Julia nb6 scan20 distributed finished ==="
