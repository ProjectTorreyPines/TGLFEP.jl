#!/bin/bash -l
# Timing: Julia CPU SCAN_N=20, 10 nodes, 20 workers (2/node), SlurmClusterManager + pmap,
# with the FULL CPU sysimage (TJLF + TJLFEP baked) on master AND workers for a fair
# comparison against the GPU run (which bakes the same compute path).
# Sweep nbasis via NB env (default 6, from build/):  NB=32 sbatch timing/batch_time_scan20_julia_cpu.sh
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 10
#SBATCH -t 00:45:00
#SBATCH -C cpu
#SBATCH -J time_s20_jcpu
#SBATCH -o time_scan20_julia_cpu_%j.out
#SBATCH -e time_scan20_julia_cpu_%j.err
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64

set -uo pipefail

module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"
export TGLFEP_JULIA_FILE_ONLY=1
export TGLFEP_JULIA_DEBUG=0
export SCAN_N=20
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"

NB="${NB:-6}"
export N_BASIS="${NB}"

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TGLFEP_JULIA_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"
export GACODE_DUMP="${GACODE_DUMP:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_scan20_nb${NB}.TGLFEP}"
export FILE_DIR="${FILE_DIR:-${TGLFEP_JULIA_ROOT}/dev/fileInput_nb${NB}_scan20_10n_${SLURM_JOB_ID}}"

# Full CPU sysimage (TJLF + TJLFEP baked) for BOTH the master and the SlurmClusterManager
# workers. Workers pick it up via TGLFEP_JULIA_SYSIMAGE in time_scan20_julia_cpu.jl; the master
# gets the flag below. Missing -> JIT (slower; not a fair comparison vs the GPU sysimage run).
CPU_SYSIMG="${TGLFEP_JULIA_SYSIMAGE:-${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_cpu_sysimage.so}"
if [[ -f "${CPU_SYSIMG}" ]]; then
    export TGLFEP_JULIA_SYSIMAGE="${CPU_SYSIMG}"
    MASTER_SYSIMG_ARGS=(--sysimage="${CPU_SYSIMG}")
    echo "CPU sysimage (master+workers): ${CPU_SYSIMG}"
else
    MASTER_SYSIMG_ARGS=()
    echo "CPU sysimage: none found at '${CPU_SYSIMG}' -> running with JIT"
fi

cd "${TGLFEP_JULIA_ROOT}/dev"
echo "=== Julia CPU timing: SCAN_N=${SCAN_N} N_BASIS=${NB} nodes=${SLURM_NNODES:-?} ntasks=${SLURM_NTASKS:-?} (SlurmClusterManager) ==="
echo "FILE_DIR=${FILE_DIR}"

stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" \
    -t 8 timing/time_scan20_julia_cpu.jl

echo "=== Julia CPU distributed timing finished ==="
