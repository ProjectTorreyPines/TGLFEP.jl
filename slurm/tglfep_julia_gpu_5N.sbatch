#!/bin/bash -l
# =============================================================================
# TJLFEP on GPU, HIGH THROUGHPUT: SCAN_N=20 radii on 5 GPU nodes in ONE wave.
#
#   20 Slurm tasks = 20 radii, 4 radii per node, one A100 per radius, an NVIDIA MPS team of
#   MPS_TEAM worker processes sharing each GPU. Best for solvers whose per-radius cost is
#   uniform: SOLVER=grid (Fortran-equivalent) and SOLVER=ad AD_EXTEND_MODE=only.
#   For the extended-width solvers (ad locate/wide, robust_ad, truth) use
#   slurm/tglfep_julia_gpu_backfill.sbatch instead: their edge radii straggle and would leave
#   4 of 5 nodes idle here.
#
# Usage (from the repo root, so the job can find this checkout):
#   cd TJLFEP
#   edit the CONFIG block below (at least: #SBATCH -A, TGLFEP_FILE)
#   sbatch slurm/tglfep_julia_gpu_5N.sbatch
# Any CONFIG value can also be overridden at submit time:
#   SOLVER=ad AD_EXTEND_MODE=only sbatch slurm/tglfep_julia_gpu_5N.sbatch
#
# SCAN_N in the input.TGLFEP MUST equal -n below (20). The job checks and refuses otherwise.
# See slurm/README.md for the solver decision table, outputs, and troubleshooting.
# =============================================================================
#SBATCH -A m3739_g                 # EDIT: your NERSC GPU allocation (GPU projects end in _g)
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 5
#SBATCH -n 20
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH -t 00:45:00                # nb32 grid ~7 min scan + load; raise for truth/robust_ad
#SBATCH -J tglfep_julia_gpu_5N
#SBATCH -o tglfep_julia_gpu_5N_%j.out
#SBATCH -e tglfep_julia_gpu_5N_%j.err
## For N_BASIS=48 extended-width runs: 80 GB A100 nodes and a halved team (see MPS_TEAM below)
##SBATCH -C gpu&hbm80g

set -uo pipefail

# -----------------------------------------------------------------------------
# CONFIG -- everything a user edits is in this block. Each is VAR="${VAR:-default}" so an
# environment variable at submit time overrides the value written here (SYSIMAGE uses
# "${SYSIMAGE-...}", so `SYSIMAGE="" sbatch ...` forces JIT).
# -----------------------------------------------------------------------------
# ---- Paths ----
TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"         # this checkout; auto when submitted from the repo root
CFS_DIR="${CFS_DIR:-/global/cfs/cdirs/m3739/TJLFEP}"             # shared m3739 tree: prebuilt sysimages, env_lean/env_full, depot
TGLFEP_JULIA_DEPOT="${TGLFEP_JULIA_DEPOT:-}"                                 # extra read-only depot; "" -> $CFS_DIR/depot (sysimage JLL artifacts)

# ---- Case / inputs ----
CASE_DIR="${CASE_DIR:-${TGLFEP_JULIA_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"   # directory holding input.gacode
GACODE_FILE="${GACODE_FILE:-${CASE_DIR}/input.gacode}"                      # equilibrium + profiles (EXPRO format)
TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_scan20_nb32.TGLFEP}"          # scan control. PHYSICS KNOBS LIVE HERE:
                                                                            #   SCAN_N (=20 here), N_BASIS, IRS, WIDTH_MIN/WIDTH_MAX,
                                                                            #   FACTOR_IN, PROCESS_IN, KY_MODEL, ... (see slurm/README.md)
OUT_DIR="${OUT_DIR:-${TGLFEP_JULIA_ROOT}/runs/gpu_5N_${SLURM_JOB_ID:-local}_tasks}"   # per-radius task_<i>.jls + merged profile; CASE_DIR is never written

# ---- Solver ----
SOLVER="${SOLVER:-grid}"                 # grid      Fortran-equivalent (kyhat x width x factor) sweep, w>=1 box     [verified vs Fortran]
                                         # ad        autodiff onset root-find; see AD_EXTEND_MODE                      [grid-independent]
                                         # robust_ad w>=1 grid-zoom + extended narrow-width locate                     [reference tier]
                                         # truth     robust_ad + N_BASIS ladder at the optimum                          [reference tier, slow]
AD_EXTEND_MODE="${AD_EXTEND_MODE:-locate}"   # SOLVER=ad only:  only   = w>=1 box, no width extension (fastest, approximates grid)
                                             #                  wide   = one log-seeded narrow-width pass (cheap, conservative)
                                             #                  locate = dense narrow-width locate (most accurate, production)
AD_WIDE_KDESC="${AD_WIDE_KDESC:-2}"          # ad :wide multistart breadth
AD_FAITHFUL_CONFIRM="${AD_FAITHFUL_CONFIRM:-1}"   # ad: 0 skips the faithful IFLUX confirm (pure AD, faster, less accurate)
REFINE_ROUNDS="${REFINE_ROUNDS:-1}"          # robust_ad / truth: (ky,w) window-narrowing rounds (0 = coarse grid min)
TGLFEP_JULIA_K_MAX="${TGLFEP_JULIA_K_MAX:-4}"            # grid: scale-factor refinement rounds
TGLFEP_JULIA_NMODES="${TGLFEP_JULIA_NMODES:-4}"          # eigenmodes kept per ky
TGLFEP_JULIA_NXGRID="${TGLFEP_JULIA_NXGRID:-}"           # "" -> max(N_BASIS, 32) Hermite quadrature points (debug knob)
TGLFEP_JULIA_PRINTOUT="${TGLFEP_JULIA_PRINTOUT:-0}"      # 1 also writes out.scalefactor_r### per radius into OUT_DIR
TGLFEP_JULIA_DEBUG="${TGLFEP_JULIA_DEBUG:-0}"            # 1 prints matched [*_DBG] lines (Fortran comparison)
TGLFEP_JULIA_PROBE="${TGLFEP_JULIA_PROBE:-0}"            # 1 prints per-worker eigensolve counts/times

# ---- Parallel layout ----
INNER="${INNER:-}"                           # within-radius parallelism: mps_team | threads. "" = auto:
                                             #   grid/robust_ad/truth -> mps_team (thousands of independent eigensolves amortize the team)
                                             #   ad                   -> threads  (few sequential Newton evals; a team only adds overhead)
MPS_TEAM="${MPS_TEAM:-8}"                    # MPS worker processes per GPU (8 fills an A100 for grid; use 4 at N_BASIS=48 to fit host RAM)
JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"   # CPU threads per worker (matrix assembly); 4 tasks x 8 workers x 2 = 64 cores/node
STAGE="${STAGE:-1}"                          # 1 = sbcast the sysimage to node-local /tmp (cuts the 20-way Lustre load tail); 0 = off
GPUS_PER_RADIUS=1                            # tied to the #SBATCH header above (4 tasks/node, 4 GPUs/node); do not edit

# ---- Sysimage ----
SYSIMAGE="${SYSIMAGE-${TGLFEP_JULIA_GPU_SYSIMAGE:-${CFS_DIR}/TGLFEP_JULIA_gpu_sysimage.so}}"   # prebuilt file-only GPU image (m3739 members).
                                             #   ""  -> JIT from this checkout (reproducible baseline, ~110 s/team slower per radius)
                                             #   build/TGLFEP_JULIA_gpu_sysimage.so -> your own bake (see build/README.md)
PROJECT="${PROJECT:-}"                       # julia --project. "" = auto: <sysimage dir>/env_lean with a sysimage, else $TGLFEP_JULIA_ROOT
# -----------------------------------------------------------------------------
# End of CONFIG. Nothing below needs editing.
# -----------------------------------------------------------------------------

if [[ -z "${INNER}" ]]; then
    if [[ "${SOLVER}" == "ad" ]]; then INNER=threads; else INNER=mps_team; fi
fi
DEVICE=gpu
EXPECT_NTASKS_EQ_SCAN_N=1
source "${TGLFEP_JULIA_ROOT}/slurm/common/tglfep_julia_env.inc.sh" \
    || { echo "ERROR: cannot source ${TGLFEP_JULIA_ROOT}/slurm/common/tglfep_julia_env.inc.sh -- submit from the repo root or export TGLFEP_JULIA_ROOT" >&2; exit 1; }
tglfep_julia_setup

cd "${TGLFEP_JULIA_ROOT}"
t_start=$(date +%s)
# One task per radius. mps-scan-wrapper.sh starts the per-node MPS daemon (daemon-first launch
# order is essential), pins this task's GPU, sets SCAN_INDEX = procid+1, then execs the driver.
srun --export=ALL --label -n "${SLURM_NTASKS}" --ntasks-per-node=4 --cpu-bind=cores \
    "${TGLFEP_JULIA_ROOT}/slurm/common/mps-scan-wrapper.sh" \
    stdbuf -oL -eL "${JULIA[@]}" -t "${JULIA_WORKER_THREADS}" \
    "${TGLFEP_JULIA_ROOT}/slurm/common/run_gacode_scan20_mps_task.jl"
scan_rc=$?
echo "=== scan step ended (exit code ${scan_rc}) after $(( $(date +%s) - t_start )) s (incl. worker spawn + image load) ==="

tglfep_julia_mps_quit
# Merge even after a failed scan step: its error names the radii whose task_<i>.jls is missing.
tglfep_julia_merge
merge_rc=$?
tglfep_julia_finish "=== done; per-radius task_*.jls + sfmin_scan.txt + alpha_*_crit.input in ${OUT_DIR} ===" \
    scan="${scan_rc}" merge="${merge_rc}"
