#!/bin/bash -l
# Fortran: N_BASIS=6, SCAN_N=20, 10 nodes, 20 MPI ranks (one radius per rank).
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 10
#SBATCH -t 02:00:00
#SBATCH -C cpu
#SBATCH -J TGLFEP_nb6_s20_10n
#SBATCH -o debug_nb6_fortran20_10n_%j.out
#SBATCH -e debug_nb6_fortran20_10n_%j.err
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=8

set -euo pipefail

# Canonical Fortran TGLF-EP build (shared m3739 reference; override TGLFEP_DIR for a local build).
export TGLFEP_DIR="${TGLFEP_DIR:-/global/cfs/cdirs/m3739/gacode_add_d3d/TGLF-EP}"
DRIVER="${TGLFEP_DRIVER:-${TGLFEP_DIR}/TGLFEP_driver}"
# GACODE source that provides shared/bin/gacode_setup + platform env (override for your install).
export GACODE_ROOT="${GACODE_ROOT:-/pscratch/sd/t/tneiser/gacode_cpu/gacode}"
export GACODE_PLATFORM="${GACODE_PLATFORM:-PERLMUTTER_CPU}"
export TGLFEP_DEBUG=0

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
CASE_DIR="${CASE_DIR:-${TGLFEP_JULIA_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"
TGLFEP_INPUT="${TGLFEP_INPUT:-${CASE_DIR}/input_scan20_nb6.TGLFEP}"
RUN_DIR="${TGLFEP_JULIA_ROOT}/dev/fortran_runs/debug_nb6_scan20_10n_${SLURM_JOB_ID:-local}"

set +u
# shellcheck source=/dev/null
. "${GACODE_ROOT}/shared/bin/gacode_setup"
# shellcheck source=/dev/null
. "${GACODE_ROOT}/platform/env/env.${GACODE_PLATFORM}"
set -u

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
ln -sf "${TGLFEP_INPUT}" input.TGLFEP
ln -sf "${CASE_DIR}/input.gacode" input.gacode

echo "=== Fortran nb6 SCAN_N=20 on ${SLURM_NNODES:-?} nodes, ntasks=${SLURM_NTASKS:-20} ==="
srun -n "${SLURM_NTASKS:-20}" "${DRIVER}"
echo "=== done -> ${RUN_DIR} ==="
