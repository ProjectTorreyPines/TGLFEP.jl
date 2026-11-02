#!/bin/bash
set -euo pipefail

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export FORTRAN_DIR="${FORTRAN_DIR:-${TGLFEP_JULIA_ROOT}/dev/fortran_runs/debug_nb6_scan20_10n_53171364}"
export JULIA_DIR="${JULIA_DIR:-${TGLFEP_JULIA_ROOT}/dev/debug_out_nb6_scan20_53171385_dist}"
export FILE_DIR="${FILE_DIR:-${TGLFEP_JULIA_ROOT}/dev/fileInput_nb6_scan20_10n_53171385}"
export PLOT_OUTDIR="${PLOT_OUTDIR:-${TGLFEP_JULIA_ROOT}/dev/compare_nb6_scan20_plots}"
export PLOT_TITLE="${PLOT_TITLE:-N_BASIS=6, SCAN_N=20 (10 nodes)}"

module load julia/1.11.7 2>/dev/null || true
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${PSCRATCH}/.julia}"

julia --project="${TGLFEP_JULIA_ROOT}" "${TGLFEP_JULIA_ROOT}/dev/verify/plot_nb6_scan20_compare.jl"
