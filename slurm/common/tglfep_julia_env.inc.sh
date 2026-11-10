# slurm/common/tglfep_julia_env.inc.sh -- shared boilerplate for the slurm/*.sbatch templates.
#
# A template sets its CONFIG variables, then:
#
#     TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
#     DEVICE=gpu            # or cpu
#     source "${TGLFEP_JULIA_ROOT}/slurm/common/tglfep_julia_env.inc.sh"
#     tglfep_julia_setup          # modules, depot, sysimage/project, input checks, env export, banner
#
# After tglfep_julia_setup the template has:
#     "${JULIA[@]}"          julia --startup-file=no --project=$PROJECT [--sysimage=$SYSIMAGE]
#     SCAN_N                 read from the input.TGLFEP
#     TGLFEP_JULIA_ROOT PROJECT SYSIMAGE OUT_DIR ...  exported for the Julia drivers
# and the helpers tglfep_julia_mps_quit (GPU), tglfep_julia_merge and tglfep_julia_finish (the job's
# exit code: a template must end on tglfep_julia_finish, never on a bare echo).
#
# Variables consumed (set them in the template's CONFIG block; all have fallbacks):
#     DEVICE                 gpu | cpu                                   (required)
#     TGLFEP_JULIA_ROOT            the TJLFEP checkout                          (default: SLURM_SUBMIT_DIR)
#     CFS_DIR                shared m3739 tree with images/envs/depot    (default below)
#     TGLFEP_JULIA_DEPOT           extra read-only depot to append              (default: $CFS_DIR/depot)
#     SYSIMAGE               prebuilt sysimage, "" = JIT
#     PROJECT                julia --project; "" = auto (see tglfep_julia_setup_sysimage)
#     STAGE                  1 = sbcast the sysimage to node-local /tmp (multi-node GPU)
#     CASE_DIR GACODE_FILE TGLFEP_FILE OUT_DIR
#     EXPECT_NTASKS_EQ_SCAN_N  1 = require SLURM_NTASKS == SCAN_N (one radius per task)
#     SOLVER AD_EXTEND_MODE AD_WIDE_KDESC AD_FAITHFUL_CONFIRM REFINE_ROUNDS
#     TGLFEP_JULIA_K_MAX TGLFEP_JULIA_NMODES TGLFEP_JULIA_NXGRID TGLFEP_JULIA_PRINTOUT TGLFEP_JULIA_DEBUG TGLFEP_JULIA_PROBE
#     INNER MPS_TEAM JULIA_WORKER_THREADS GPUS_PER_RADIUS BACKFILL_MODE TGLFEP_JULIA_INSTANTIATE
#
# Why SLURM_SUBMIT_DIR: sbatch copies the script to the slurmd spool directory before running
# it, so BASH_SOURCE / dirname "$0" do NOT point into the repo. Submit from the repo root (or
# export TGLFEP_JULIA_ROOT) and the templates find everything from there.

tglfep_julia_die() { echo "ERROR: $*" >&2; exit 1; }

# The knobs were renamed TJLFEP_* -> TGLFEP_JULIA_* (2026-11). Accept the old names for one release:
# copy every exported TJLFEP_<X> into an unset TGLFEP_JULIA_<X> and say so once.
tglfep_julia_migrate_legacy_env() {
    local v new moved=()
    for v in $(compgen -A export TJLFEP_ 2>/dev/null || true); do
        new="TGLFEP_JULIA_${v#TJLFEP_}"
        [[ -n "${!new:-}" ]] || { export "${new}=${!v}"; moved+=("${v}"); }
    done
    (( ${#moved[@]} )) && echo "WARNING: TJLFEP_* variables are deprecated, use TGLFEP_JULIA_*: ${moved[*]}" >&2
    return 0
}

tglfep_julia_resolve_root() {
    TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
    if [[ ! -f "${TGLFEP_JULIA_ROOT}/Project.toml" ]] || ! grep -q '^name = "TJLFEP"' "${TGLFEP_JULIA_ROOT}/Project.toml"; then
        tglfep_julia_die "TGLFEP_JULIA_ROOT='${TGLFEP_JULIA_ROOT}' is not a TJLFEP checkout. Submit from the repo root" \
                   "(cd TJLFEP && sbatch slurm/<template>.sbatch) or export TGLFEP_JULIA_ROOT=/path/to/TJLFEP."
    fi
    export TGLFEP_JULIA_ROOT
    CFS_DIR="${CFS_DIR:-/global/cfs/cdirs/m3739/TJLFEP}"
    export CFS_DIR
}

# cudatoolkit FIRST, then julia: the julia module stacks CUDA.jl preferences keyed on the
# cudatoolkit that is loaded at that moment. CUDA.jl is a hard TJLFEP dependency pinned to a
# 12.9 local toolkit, so the CPU path loads it too (otherwise a default-loaded cudatoolkit/13.x
# makes CUDA.jl log a toolkit-mismatch error at startup even though the CPU run ignores it).
tglfep_julia_load_modules() {
    module load "${CUDA_MODULE:-cudatoolkit/12.9}"
    module load "${JULIA_MODULE:-julia/1.11.7}"
    if [[ "${DEVICE}" == "gpu" ]]; then
        # Required: CUDA.jl's forward-compat driver shim hangs in cuInit for MPS clients on Perlmutter.
        export JULIA_CUDA_USE_COMPAT=false
    fi
}

# Depot search path. Order = precedence; the FIRST entry is the only writable one (precompile
# cache, logs). Never clobber the caller's depot ($HOME/.julia is Julia's default when unset):
# prepend/append instead. The shared CFS depot carries the JLL artifacts (OpenSSL_jll,
# CUDA_*_jll, ...) that a prebuilt sysimage resolves at load time; without it a shared
# sysimage aborts with `Artifact "OpenSSL" was not found`. It is only appended if readable.
tglfep_julia_setup_depot() {
    local depot="${JULIA_DEPOT_PATH:-$HOME/.julia}:${PSCRATCH:-$SCRATCH}/.julia"
    local shared="${TGLFEP_JULIA_DEPOT:-${CFS_DIR}/depot}"
    [[ -d "${shared}" ]] && depot="${depot}:${shared}"
    export JULIA_DEPOT_PATH="${depot}"
}

# Sysimage + project pairing.
#   SYSIMAGE set and present -> --sysimage, and PROJECT defaults to the published env that the
#       image was baked from: <dir>/env_lean (file-only GPU + CPU images) or <dir>/env_full
#       (*generic* image). A locally baked image in build/ pairs with build/sysimage/env_*.
#       Running a prebuilt image under a different project (e.g. your checkout) risks a
#       package-version mismatch between what is baked and what the manifest resolves.
#   SYSIMAGE empty or missing -> JIT; PROJECT defaults to the TJLFEP checkout.
tglfep_julia_setup_sysimage() {
    JULIA_SYSIMAGE_ARGS=()
    if [[ -n "${SYSIMAGE:-}" && -f "${SYSIMAGE}" ]]; then
        JULIA_SYSIMAGE_ARGS=(--sysimage="${SYSIMAGE}")
        if [[ "${DEVICE}" == "gpu" ]]; then export TGLFEP_JULIA_GPU_SYSIMAGE="${SYSIMAGE}"; unset TGLFEP_JULIA_SYSIMAGE
        else                               export TGLFEP_JULIA_SYSIMAGE="${SYSIMAGE}";     unset TGLFEP_JULIA_GPU_SYSIMAGE; fi
        if [[ -z "${PROJECT:-}" ]]; then
            local d env; d="$(dirname "${SYSIMAGE}")"
            case "$(basename "${SYSIMAGE}")" in *generic*) env=env_full ;; *) env=env_lean ;; esac
            PROJECT="${d}/${env}"
            [[ -f "${PROJECT}/Project.toml" ]] || PROJECT="${d}/sysimage/${env}"   # locally baked (build/)
        fi
        SYSIMAGE_DESC="${SYSIMAGE}"
    else
        [[ -n "${SYSIMAGE:-}" ]] && echo "WARNING: sysimage not found at '${SYSIMAGE}' -> running with JIT"
        SYSIMAGE=""
        unset TGLFEP_JULIA_GPU_SYSIMAGE TGLFEP_JULIA_SYSIMAGE
        PROJECT="${PROJECT:-${TGLFEP_JULIA_ROOT}}"
        SYSIMAGE_DESC="none (JIT)"
    fi
    [[ -f "${PROJECT}/Project.toml" ]] || tglfep_julia_die "no Project.toml in PROJECT='${PROJECT}'."
    if [[ -z "${SYSIMAGE}" && ! -f "${PROJECT}/Manifest.toml" ]]; then
        tglfep_julia_die "PROJECT='${PROJECT}' has no Manifest.toml. Instantiate it once on a login node:" \
                   "  julia --project='${PROJECT}' -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'"
    fi
    export PROJECT SYSIMAGE
    JULIA=(julia --startup-file=no --project="${PROJECT}" "${JULIA_SYSIMAGE_ARGS[@]}")
}

# Multi-node GPU runs: sbcast the sysimage to node-local /tmp once per node so the 4 radii x
# MPS_TEAM workers per node load it from local disk instead of all reading the ~1.3 GB image
# off Lustre at once (measured ~80 s less scan wall on 20 nodes; ~10 s one-time broadcast).
tglfep_julia_stage_sysimage() {
    [[ "${STAGE:-0}" == "1" && -n "${SYSIMAGE}" && "${SLURM_NNODES:-1}" -gt 1 ]] || return 0
    local staged="/tmp/tglfep_julia_gpusys_${SLURM_JOB_ID}.so" t0
    echo "STAGE=1: sbcast ${SYSIMAGE} -> ${staged} (all nodes)"
    t0=$(date +%s)
    if sbcast -f "${SYSIMAGE}" "${staged}"; then
        SYSIMAGE="${staged}"
        export SYSIMAGE TGLFEP_JULIA_GPU_SYSIMAGE="${staged}"
        JULIA_SYSIMAGE_ARGS=(--sysimage="${staged}")
        JULIA=(julia --startup-file=no --project="${PROJECT}" "${JULIA_SYSIMAGE_ARGS[@]}")
        SYSIMAGE_DESC="${staged} (staged copy of $(basename "${staged}"))"
        echo "sbcast done in $(( $(date +%s) - t0 )) s"
    else
        echo "WARNING: sbcast failed; falling back to the shared path ${SYSIMAGE}"
    fi
}

tglfep_julia_check_inputs() {
    [[ -f "${GACODE_FILE}" ]] || tglfep_julia_die "missing input.gacode at GACODE_FILE='${GACODE_FILE}'"
    [[ -f "${TGLFEP_FILE}" ]] || tglfep_julia_die "missing input.TGLFEP at TGLFEP_FILE='${TGLFEP_FILE}'"
    # input.TGLFEP lines are "<value>  <KEYWORD>"
    SCAN_N="$(awk '$2 == "SCAN_N" { print $1; exit }' "${TGLFEP_FILE}")"
    [[ -n "${SCAN_N}" ]] || tglfep_julia_die "could not read SCAN_N from ${TGLFEP_FILE}"
    if [[ "${EXPECT_NTASKS_EQ_SCAN_N:-0}" == "1" && -n "${SLURM_NTASKS:-}" && "${SLURM_NTASKS}" != "${SCAN_N}" ]]; then
        tglfep_julia_die "this template runs one radius per task, so SCAN_N in ${TGLFEP_FILE} (${SCAN_N})" \
                   "must equal the number of Slurm tasks (${SLURM_NTASKS}). Change SCAN_N or the #SBATCH -n/-N lines," \
                   "or use the backfill / 1-node template, which accepts any SCAN_N."
    fi
    mkdir -p "${OUT_DIR}"
    export SCAN_N CASE_DIR GACODE_FILE TGLFEP_FILE OUT_DIR
}

# Export the solver / layout knobs the Julia drivers read from ENV. Only non-empty values are
# exported so an unset knob falls back to the in-code default.
tglfep_julia_export_solver_env() {
    local v
    for v in SOLVER AD_EXTEND_MODE AD_WIDE_KDESC AD_FAITHFUL_CONFIRM REFINE_ROUNDS \
             TGLFEP_JULIA_K_MAX TGLFEP_JULIA_NMODES TGLFEP_JULIA_NXGRID TGLFEP_JULIA_PRINTOUT TGLFEP_JULIA_DEBUG TGLFEP_JULIA_PROBE \
             INNER MPS_TEAM JULIA_WORKER_THREADS GPUS_PER_RADIUS BACKFILL_MODE TGLFEP_JULIA_INSTANTIATE; do
        [[ -n "${!v:-}" ]] && export "${v}"
    done
    if [[ "${DEVICE}" == "gpu" ]]; then
        export USE_GPU=1
        export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.${SLURM_JOB_ID:-$$}"
        export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.${SLURM_JOB_ID:-$$}"
    else
        export USE_GPU=0
    fi
}

tglfep_julia_banner() {
    echo "=== TJLFEP ${DEVICE} scan | $(hostname) | $(date) | job ${SLURM_JOB_ID:-local} ==="
    echo "TGLFEP_JULIA_ROOT = ${TGLFEP_JULIA_ROOT}"
    echo "PROJECT     = ${PROJECT}"
    echo "SYSIMAGE    = ${SYSIMAGE_DESC}"
    echo "DEPOT       = ${JULIA_DEPOT_PATH}"
    echo "GACODE_FILE = ${GACODE_FILE}"
    echo "TGLFEP_FILE = ${TGLFEP_FILE}   (SCAN_N=${SCAN_N})"
    echo "OUT_DIR     = ${OUT_DIR}"
    echo "SOLVER=${SOLVER:-grid}$( [[ "${SOLVER:-grid}" == "ad" ]] && echo " AD_EXTEND_MODE=${AD_EXTEND_MODE:-locate}" )" \
         "REFINE_ROUNDS=${REFINE_ROUNDS:-1} INNER=${INNER:-} MPS_TEAM=${MPS_TEAM:-} JULIA_WORKER_THREADS=${JULIA_WORKER_THREADS:-}" \
         "nodes=${SLURM_NNODES:-?} tasks=${SLURM_NTASKS:-?} BACKFILL_MODE=${BACKFILL_MODE:-0}"
    [[ "${DEVICE}" == "gpu" ]] && { nvidia-smi -L 2>/dev/null | head -4 || true; }
    julia --version 2>/dev/null || true
}

# Everything a template needs before its srun line.
tglfep_julia_setup() {
    tglfep_julia_migrate_legacy_env
    [[ -n "${DEVICE:-}" ]] || tglfep_julia_die "set DEVICE=gpu or DEVICE=cpu before sourcing tglfep_julia_env.inc.sh"
    tglfep_julia_resolve_root
    tglfep_julia_load_modules
    tglfep_julia_setup_depot
    tglfep_julia_setup_sysimage
    [[ "${DEVICE}" == "gpu" ]] && tglfep_julia_stage_sysimage
    tglfep_julia_check_inputs
    tglfep_julia_export_solver_env
    tglfep_julia_banner
}

# Stop the per-node MPS control daemons started by mps-scan-wrapper.sh.
tglfep_julia_mps_quit() {
    srun --export=ALL -n "${SLURM_NNODES:-1}" --ntasks-per-node=1 \
        bash -c 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' || true
}

# Merge the per-radius task_*.jls in OUT_DIR into sfmin_scan.txt + alpha_*_crit.input.
tglfep_julia_merge() {
    echo "=== merging ${OUT_DIR} ==="
    USE_GPU=0 stdbuf -oL -eL "${JULIA[@]}" -t "${MERGE_THREADS:-8}" \
        "${TGLFEP_JULIA_ROOT}/slurm/common/merge_gacode_scan20_array.jl"
}

# Last line of every template: turn the exit codes of its steps into the job's exit code.
#     tglfep_julia_finish "<closing line>" <step>=<exit code> [<step>=<exit code> ...]
# All zero -> print the closing line. Otherwise say which step failed (in .out and .err) and
# exit with the first non-zero code, so Slurm records the job as FAILED and `afterok`
# dependents do not start. Without this a template ends on an echo and always exits 0.
tglfep_julia_finish() {
    local closing="$1" step code rc=0 failed=""
    shift
    for step in "$@"; do
        code="${step##*=}"
        [[ "${code}" == "0" ]] && continue
        failed="${failed:+${failed}, }${step%%=*} (exit code ${code})"
        [[ "${rc}" == "0" ]] && rc="${code}"
    done
    if [[ "${rc}" != "0" ]]; then
        echo "=== FAILED: ${failed}; outputs in ${OUT_DIR:-OUT_DIR} are incomplete ==="
        echo "ERROR: ${failed}; outputs in ${OUT_DIR:-OUT_DIR} are incomplete" >&2
        exit "${rc}"
    fi
    echo "${closing}"
}
