#!/bin/bash -l
# =============================================================================
# TJLFEP on CPU, HIGH THROUGHPUT: SCAN_N=20 radii on 10 CPU nodes, one worker per radius.
#
#   The master spawns 20 Julia workers (SlurmClusterManager, 2 per node x 64 threads) and
#   pmaps the radii over them; the master writes the merged SFmin profile itself. No GPU
#   allocation needed. The CPU grid path is ~6x slower than the Fortran reference at nb32
#   and ~25x slower than the GPU path (node-hours), so use the GPU templates when you can.
#
# Usage (from the repo root, so the job can find this checkout):
#   cd TJLFEP
#   edit the CONFIG block below (at least: #SBATCH -A, TGLFEP_FILE)
#   sbatch slurm/tglfep_julia_cpu_10N.sbatch
#
# SCAN_N in the input.TGLFEP MUST equal --ntasks below (20). Measured 10-node grid wall
# (DIII-D case): nb6 2 min, nb16 12 min, nb32 87 min, nb48 5.7 h -> set -t accordingly.
# =============================================================================
#SBATCH -A m3739                   # EDIT: your NERSC CPU allocation (no _g suffix)
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 10
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=128           # logical CPUs = 64 physical cores per worker
#SBATCH -t 02:00:00
#SBATCH -J tglfep_julia_cpu_10N
#SBATCH -o tglfep_julia_cpu_10N_%j.out
#SBATCH -e tglfep_julia_cpu_10N_%j.err

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
OUT_DIR="${OUT_DIR:-${TGLFEP_JULIA_ROOT}/runs/cpu_10N_${SLURM_JOB_ID:-local}}"     # out.TGLFEP, sfmin_scan.txt, alpha_*_crit.input

# ---- Solver ----
SOLVER="${SOLVER:-grid}"                 # grid      Fortran-equivalent (kyhat x width x factor) sweep, w>=1 box     [verified vs Fortran]
                                         # ad        autodiff onset root-find; see AD_EXTEND_MODE                      [grid-independent]
                                         # robust_ad w>=1 grid-zoom + extended narrow-width locate                     [reference tier]
                                         # truth     robust_ad + N_BASIS ladder at the optimum                          [reference tier, slow]
AD_EXTEND_MODE="${AD_EXTEND_MODE:-locate}"   # SOLVER=ad only: only | wide | locate (see the GPU templates / slurm/README.md)
AD_WIDE_KDESC="${AD_WIDE_KDESC:-2}"          # ad :wide multistart breadth
AD_FAITHFUL_CONFIRM="${AD_FAITHFUL_CONFIRM:-1}"   # ad: 0 skips the faithful IFLUX confirm
REFINE_ROUNDS="${REFINE_ROUNDS:-1}"          # robust_ad / truth: (ky,w) window-narrowing rounds
TGLFEP_JULIA_K_MAX="${TGLFEP_JULIA_K_MAX:-4}"            # grid: scale-factor refinement rounds
TGLFEP_JULIA_NMODES="${TGLFEP_JULIA_NMODES:-4}"          # eigenmodes kept per ky
TGLFEP_JULIA_NXGRID="${TGLFEP_JULIA_NXGRID:-}"           # "" -> max(N_BASIS, 32) Hermite quadrature points (debug knob)
TGLFEP_JULIA_DEBUG="${TGLFEP_JULIA_DEBUG:-0}"            # 1 prints matched [*_DBG] lines (Fortran comparison)

# ---- Parallel layout ----
# One Julia thread per PHYSICAL core: Slurm CPUs are hyperthreads here, so a worker with
# --cpus-per-task=128 owns 64 cores. Running 64 threads on 64 logical CPUs (32 cores) is ~2x slower.
JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-$(( ${SLURM_CPUS_PER_TASK:-128} / 2 ))}"   # threads per worker (the ky loop is threaded)
TGLFEP_JULIA_INSTANTIATE="${TGLFEP_JULIA_INSTANTIATE:-0}"   # 1 = Pkg.instantiate()+precompile() in the job first (JIT from a fresh checkout only;
                                                #     never against the read-only CFS env)

# ---- Sysimage ----
SYSIMAGE="${SYSIMAGE-${TGLFEP_JULIA_SYSIMAGE:-${CFS_DIR}/TGLFEP_JULIA_cpu_sysimage.so}}"   # prebuilt CPU image (m3739 members).
                                             #   ""  -> JIT from this checkout (slower startup on every worker)
                                             #   build/TGLFEP_JULIA_cpu_sysimage.so -> your own bake (see build/README.md)
PROJECT="${PROJECT:-}"                       # julia --project. "" = auto: <sysimage dir>/env_lean with a sysimage, else $TGLFEP_JULIA_ROOT
# -----------------------------------------------------------------------------
# End of CONFIG. Nothing below needs editing.
# -----------------------------------------------------------------------------

DEVICE=cpu
EXPECT_NTASKS_EQ_SCAN_N=1
source "${TGLFEP_JULIA_ROOT}/slurm/common/tglfep_julia_env.inc.sh" \
    || { echo "ERROR: cannot source ${TGLFEP_JULIA_ROOT}/slurm/common/tglfep_julia_env.inc.sh -- submit from the repo root or export TGLFEP_JULIA_ROOT" >&2; exit 1; }
tglfep_julia_setup

cd "${TGLFEP_JULIA_ROOT}"
t_start=$(date +%s)
# The master runs here; it spawns one worker per Slurm task via srun (SlurmClusterManager).
stdbuf -oL -eL "${JULIA[@]}" -t 8 "${TGLFEP_JULIA_ROOT}/slurm/common/run_scan_cpu_distributed.jl"
echo "=== done in $(( $(date +%s) - t_start )) s; out.TGLFEP + sfmin_scan.txt + alpha_*_crit.input in ${OUT_DIR} ==="
