#!/bin/bash -l
# Julia file-based debug: N_BASIS=6, SCAN_N=1, single node.
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -C cpu
#SBATCH -J TJLFEP_nb6
#SBATCH -o debug_nb6_julia_%j.out
#SBATCH -e debug_nb6_julia_%j.err
#SBATCH --cpus-per-task=8

set -euo pipefail

module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"
export TGLFEP_JULIA_DEBUG=1
# Skip IMAS/FUSE/TurbulentTransport (not used on file-based runTHD path).
export TGLFEP_JULIA_FILE_ONLY=1

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TGLFEP_JULIA_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"
export GACODE_DUMP="${GACODE_DUMP:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_singleradius_nb6.TGLFEP}"
export FILE_DIR="${FILE_DIR:-${TGLFEP_JULIA_ROOT}/dev/fileInput_nb6_${SLURM_JOB_ID:-local}}"

cd "${TGLFEP_JULIA_ROOT}/dev"

run_julia() { stdbuf -oL -eL julia "$@"; }

echo "=== TJLFEP debug nb6 (file-only) ==="
run_julia --project="${TGLFEP_JULIA_ROOT}" -t "${SLURM_CPUS_PER_TASK:-8}" verify/debug_compare_nb6.jl
echo "=== done ==="
