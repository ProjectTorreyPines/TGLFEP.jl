#!/bin/bash -l
# UCP variant of batch_time_scan20_julia_cpu.sh: Julia CPU :grid SCAN_N=20, 10 nodes, 20 workers
# (2/node), SlurmClusterManager + full CPU sysimage, STAGE=1 sbcast staging. Reactor-relevant
# UCP_complete case. NB via env.  cd dev && NB=32 sbatch timing/batch_time_scan20_julia_cpu_ucp.sh
#SBATCH -A m3739
#SBATCH -q premium
#SBATCH -N 10
#SBATCH -t 12:00:00
#SBATCH -C cpu
#SBATCH -J ucp_s20_jcpu
#SBATCH -o time_scan20_ucp_julia_cpu_%j.out
#SBATCH -e time_scan20_ucp_julia_cpu_%j.err
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64

set -uo pipefail

module load julia/1.11.7
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}:${PSCRATCH}/.julia:${TGLFEP_JULIA_DEPOT:-/global/cfs/cdirs/m3739/TJLFEP/depot}"
export TGLFEP_JULIA_FILE_ONLY=1
export TGLFEP_JULIA_DEBUG=0
export SCAN_N=20
export SOLVER="${SOLVER:-grid}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"

NB="${NB:-6}"
export N_BASIS="${NB}"

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TGLFEP_JULIA_ROOT}/examples/UCP_complete}"
export GACODE_DUMP="${GACODE_DUMP:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_scan20_nb${NB}.TGLFEP}"
export FILE_DIR="${FILE_DIR:-${TGLFEP_JULIA_ROOT}/dev/ucp_fileInput_nb${NB}_scan20_10n_${SLURM_JOB_ID}}"

# Full CPU sysimage for master + SlurmClusterManager workers, with STAGE=1 sbcast to node-local /tmp.
CPU_SYSIMG="${TGLFEP_JULIA_SYSIMAGE:-${TGLFEP_JULIA_ROOT}/dev/TGLFEP_JULIA_cpu_sysimage.so}"
STAGE="${STAGE:-1}"
if [[ "${STAGE}" == "1" && -f "${CPU_SYSIMG}" ]]; then
    STAGED_SO="/tmp/tglfep_julia_cpusys_${SLURM_JOB_ID}.so"
    echo "STAGE=1: sbcast ${CPU_SYSIMG} -> ${STAGED_SO} (all nodes)"
    t_bcast=$(date +%s)
    if sbcast -f "${CPU_SYSIMG}" "${STAGED_SO}"; then
        CPU_SYSIMG="${STAGED_SO}"
        echo "sbcast done in $(( $(date +%s) - t_bcast )) s"
    else
        echo "sbcast failed; falling back to shared path ${CPU_SYSIMG}"
    fi
fi
if [[ -f "${CPU_SYSIMG}" ]]; then
    export TGLFEP_JULIA_SYSIMAGE="${CPU_SYSIMG}"
    MASTER_SYSIMG_ARGS=(--sysimage="${CPU_SYSIMG}")
    echo "CPU sysimage (master+workers): ${CPU_SYSIMG}"
else
    MASTER_SYSIMG_ARGS=()
    echo "CPU sysimage: none found -> running with JIT"
fi

cd "${TGLFEP_JULIA_ROOT}/dev"
echo "=== UCP Julia CPU timing: SCAN_N=${SCAN_N} N_BASIS=${NB} solver=${SOLVER} nodes=${SLURM_NNODES:-?} ntasks=${SLURM_NTASKS:-?} (SlurmClusterManager) ==="
echo "FILE_DIR=${FILE_DIR}"

# Pre-flight: prove an in-allocation srun step can start on all tasks before the julia
# head process attempts its worker launch (SlurmClusterManager srun has failed silently
# with no stderr on some runs; this localizes slurm-step vs julia-worker failures).
echo "preflight: srun -n ${SLURM_NTASKS:-20} hostname"
if timeout 120 srun -n "${SLURM_NTASKS:-20}" hostname | sort | uniq -c | sort -rn | head -3; then
    echo "preflight srun OK"
else
    echo "preflight srun FAILED/HUNG (rc=$?)"
fi
echo "preflight: worker julia startup (1 task, sysimage load + --project)"
timeout 300 srun -n 1 julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" -e 'println("worker-preflight OK: ", VERSION)' \
    || echo "preflight worker julia FAILED (rc=$?)"

stdbuf -oL -eL julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" \
    -t 8 timing/time_scan20_julia_cpu.jl

echo "=== UCP Julia CPU distributed timing finished ==="
