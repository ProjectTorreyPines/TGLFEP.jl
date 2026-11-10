#!/bin/bash -l
# =============================================================================
# TJLFEP on GPU, LOW FOOTPRINT (backfill): all SCAN_N radii on ONE GPU node.
#
#   4 Slurm tasks = 4 A100s. Each task spawns its MPS worker team once, then drains a shared
#   claim queue over all radii (BACKFILL_MODE=1: an atomic `mkdir OUT_DIR/.claims/r<i>` owns
#   radius i). A freed GPU immediately pulls the next radius, so straggler edge radii never
#   idle a node. This is the node-hours-minimal layout and the RECOMMENDED one for the
#   extended-width solvers (SOLVER=ad with AD_EXTEND_MODE=locate|wide, robust_ad, truth),
#   whose outer radii take much longer than the core. A 1-node job also backfills easily in
#   the Slurm queue. For uniform-cost solvers (grid, ad :only) the 5-node template gives the
#   same node-hours in ~1/5 of the wall.
#
# Usage (from the repo root, so the job can find this checkout):
#   cd TJLFEP
#   edit the CONFIG block below (at least: #SBATCH -A, TGLFEP_FILE)
#   sbatch slurm/tglfep_julia_gpu_backfill.sbatch
# Any CONFIG value can also be overridden at submit time:
#   SOLVER=ad AD_EXTEND_MODE=wide sbatch slurm/tglfep_julia_gpu_backfill.sbatch
#
# Any SCAN_N works (the queue has SCAN_N entries). Wall: nb32 :locate DIII-D ~1.5 h,
# UCP ~4 h; nb32 grid ~25 min. See slurm/README.md for the decision table and outputs.
# =============================================================================
#SBATCH -A m3739_g                 # EDIT: your NERSC GPU allocation (GPU projects end in _g)
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00                # 04:00:00 for UCP_complete :locate at nb32
#SBATCH -J tglfep_julia_gpu_backfill
#SBATCH -o tglfep_julia_gpu_backfill_%j.out
#SBATCH -e tglfep_julia_gpu_backfill_%j.err
## For N_BASIS=48 extended-width runs: 80 GB A100 nodes and a halved team (MPS_TEAM=4), else the host OOMs
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
                                                                            #   SCAN_N, N_BASIS, IRS, WIDTH_MIN/WIDTH_MAX,
                                                                            #   FACTOR_IN, PROCESS_IN, KY_MODEL, ... (see slurm/README.md)
OUT_DIR="${OUT_DIR:-${TGLFEP_JULIA_ROOT}/runs/gpu_backfill_${SLURM_JOB_ID:-local}_tasks}"   # shared by all 4 tasks (holds the .claims/ queue)

# ---- Solver ----
SOLVER="${SOLVER:-ad}"                   # grid      Fortran-equivalent (kyhat x width x factor) sweep, w>=1 box     [verified vs Fortran]
                                         # ad        autodiff onset root-find; see AD_EXTEND_MODE                      [grid-independent]
                                         # robust_ad w>=1 grid-zoom + extended narrow-width locate                     [reference tier]
                                         # truth     robust_ad + N_BASIS ladder at the optimum                          [reference tier, slow]
AD_EXTEND_MODE="${AD_EXTEND_MODE:-locate}"   # SOLVER=ad only:  locate = dense narrow-width locate (most accurate, production default)
                                             #                  wide   = one log-seeded narrow-width pass (~3x cheaper, conservative, bulk DBs)
                                             #                  only   = w>=1 box, no width extension (fastest; use the 5N template for this)
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
INNER="${INNER:-mps_team}"                   # within-radius parallelism: mps_team | threads.
                                             #   Measured on this template (UCP nb32, ad :locate, 2026-10-03): mps_team 8x2 = 1.49 node-h,
                                             #   threads x2 = 2.72, threads x16 = 2.60 -> keep mps_team for the extended-width solvers;
                                             #   threads only pays off for the few-eval SOLVER=ad AD_EXTEND_MODE=only (use the 5N template).
MPS_TEAM="${MPS_TEAM:-8}"                    # MPS worker processes per GPU (use 4 at N_BASIS=48 to fit the node's host RAM)
JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"   # CPU threads per worker; 4 tasks x 8 workers x 2 = 64 cores
BACKFILL_MODE=1                              # shared claim queue over all radii; tied to this template, do not edit
GPUS_PER_RADIUS=1                            # tied to the #SBATCH header (4 tasks, 4 GPUs); do not edit
STAGE=0                                      # single node: no sysimage broadcast needed

# ---- Sysimage ----
SYSIMAGE="${SYSIMAGE-${TGLFEP_JULIA_GPU_SYSIMAGE:-${CFS_DIR}/TGLFEP_JULIA_gpu_sysimage.so}}"   # prebuilt file-only GPU image (m3739 members).
                                             #   ""  -> JIT from this checkout (reproducible baseline, ~110 s/team slower, paid once per GPU here)
                                             #   build/TGLFEP_JULIA_gpu_sysimage.so -> your own bake (see build/README.md)
PROJECT="${PROJECT:-}"                       # julia --project. "" = auto: <sysimage dir>/env_lean with a sysimage, else $TGLFEP_JULIA_ROOT
# -----------------------------------------------------------------------------
# End of CONFIG. Nothing below needs editing.
# -----------------------------------------------------------------------------

DEVICE=gpu
EXPECT_NTASKS_EQ_SCAN_N=0
source "${TGLFEP_JULIA_ROOT}/slurm/common/tglfep_julia_env.inc.sh" \
    || { echo "ERROR: cannot source ${TGLFEP_JULIA_ROOT}/slurm/common/tglfep_julia_env.inc.sh -- submit from the repo root or export TGLFEP_JULIA_ROOT" >&2; exit 1; }
tglfep_julia_setup

cd "${TGLFEP_JULIA_ROOT}"
t_start=$(date +%s)
# 4 tasks, one per GPU. mps-scan-wrapper.sh starts the node's MPS daemon, pins each task's GPU,
# then execs the driver, which (BACKFILL_MODE=1) spawns its team once and drains the queue.
srun --export=ALL --label -N 1 -n 4 --ntasks-per-node=4 --cpu-bind=cores \
    "${TGLFEP_JULIA_ROOT}/slurm/common/mps-scan-wrapper.sh" \
    stdbuf -oL -eL "${JULIA[@]}" -t "${JULIA_WORKER_THREADS}" \
    "${TGLFEP_JULIA_ROOT}/slurm/common/run_gacode_scan20_mps_task.jl"
scan_rc=$?
echo "=== scan step ended (exit code ${scan_rc}) after $(( $(date +%s) - t_start )) s (incl. team spawn + image load) ==="

tglfep_julia_mps_quit
# Merge even after a failed scan step: its error names the radii whose task_<i>.jls is missing.
tglfep_julia_merge
merge_rc=$?
tglfep_julia_finish "=== done; per-radius task_*.jls + sfmin_scan.txt + alpha_*_crit.input in ${OUT_DIR} ===" \
    scan="${scan_rc}" merge="${merge_rc}"
