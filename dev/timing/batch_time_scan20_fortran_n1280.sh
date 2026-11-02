#!/bin/bash -l
# DIII-D variant of batch_time_scan20_fortran.sh but at -n 1280 (10 CPU nodes x 128 ranks/node =
# the fast production-style MPI decomposition, ~64 factor ranks/radius for SCAN_N=20), matching the
# UCP_complete Fortran timing methodology so the DIII-D Fortran<->GPU comparison is apples-to-apples.
# (The original batch_time_scan20_fortran.sh used -n 20 = 1 rank/radius.) Writes out.TGLFEP into
# its run dir.
#   cd dev && NB=32 sbatch timing/batch_time_scan20_fortran_n1280.sh
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 10
#SBATCH -t 02:00:00
#SBATCH -C cpu
#SBATCH -J d3d_s20_frt1280
#SBATCH -o time_scan20_fortran_n1280_%j.out
#SBATCH -e time_scan20_fortran_n1280_%j.err
#SBATCH --ntasks=1280
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

set -uo pipefail

# Canonical Fortran TGLF-EP build (shared m3739 reference; override TGLFEP_DIR for a local build).
export TGLFEP_DIR="${TGLFEP_DIR:-/global/cfs/cdirs/m3739/gacode_add_d3d/TGLF-EP}"
DRIVER="${TGLFEP_DRIVER:-${TGLFEP_DIR}/TGLFEP_driver}"
# GACODE source that provides shared/bin/gacode_setup + platform env (override for your install).
export GACODE_ROOT="${GACODE_ROOT:-/pscratch/sd/t/tneiser/gacode_cpu/gacode}"
export GACODE_PLATFORM="${GACODE_PLATFORM:-PERLMUTTER_CPU}"
export TGLFEP_DEBUG=0

NB="${NB:-6}"
NTASKS="${NTASKS:-1280}"
TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
CASE_DIR="${CASE_DIR:-${TGLFEP_JULIA_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"
TGLFEP_INPUT="${TGLFEP_INPUT:-${CASE_DIR}/input_scan20_nb${NB}.TGLFEP}"
RUN_DIR="${TGLFEP_JULIA_ROOT}/dev/timing_runs/fortran_scan20_n1280_nb${NB}_${SLURM_JOB_ID}"

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

JOB_T0=$(date +%s.%N)
echo "TIMING_START backend=fortran device=cpu path=gacode nodes=${SLURM_NNODES:-10} tasks=${NTASKS} SCAN_N=20 N_BASIS=${NB} case=d3d"

T0=$(date +%s.%N)
set +e
srun -n "${NTASKS}" "${DRIVER}"
SRUN_RC=$?
set -e
T1=$(date +%s.%N)
SCAN_S=$(python3 -c "print(f'{float(\"${T1}\") - float(\"${T0}\"): .3f}')")
JOB_T1=$(date +%s.%N)
TOTAL_S=$(python3 -c "print(f'{float(\"${JOB_T1}\") - float(\"${JOB_T0}\"): .3f}')")

echo "TIMING_RESULT backend=fortran device=cpu phase=scan seconds=${SCAN_S} SCAN_N=20 N_BASIS=${NB} nodes=${SLURM_NNODES:-10} tasks=${NTASKS} srun_exit=${SRUN_RC} case=d3d"
echo "TIMING_RESULT backend=fortran device=cpu phase=total_job seconds=${TOTAL_S} SCAN_N=20 N_BASIS=${NB} nodes=${SLURM_NNODES:-10} tasks=${NTASKS} case=d3d"
[ "${SRUN_RC}" -ne 0 ] && echo "NOTE: srun exited ${SRUN_RC} (often MPI teardown); check outputs in ${RUN_DIR}"
echo "srun_exit_code=${SRUN_RC} (MPI teardown may segfault after successful scan)"
echo "=== done; out.TGLFEP + out.scalefactor_* in ${RUN_DIR} ==="
