module TJLFEP
# IMAS/FUSE entry points (`runTHD(::IMAS.dd)`, `preprocess_imas_inputs`, ...) live in the
# TJLFEPIMASExt package extension (ext/TJLFEPIMASExt.jl), which Julia loads automatically
# only when IMAS, GACODE, and TurbulentTransport are all present in the environment (e.g.
# under FUSE). Loaded standalone, TJLFEP provides only the file-based path
# (`runTHD(::String,...)` / `runTHD_from_gacode`) and stays light (no IMAS/HDF5/FUSE).
# This replaces the former `TGLFEP_JULIA_FILE_ONLY` ENV flag, which was fragile because ENV is
# not part of Julia's precompile cache key.

using Base.Threads
using LinearAlgebra
using SparseArrays
using Printf
using StaticArrays
using TJLF
using TJLF: InputTJLF  # consolidated single input type (re-exported below)
using Plots
import SimpleNonlinearSolve   # derivative-free bracketing roots (ITP/Brent) for the marginal-factor 1D root
import NLopt                  # derivative-free (ky,w) global/local search for the :nlopt solver

include("tjlfep_modules.jl")
include("tjlfep_inputs.jl")
include("tjlfep_ad_extensions.jl")
include("tjlfep_nls_extensions.jl")
include("tjlfep_read_inputs.jl")
include("tjlfep_debug.jl")
include("EXPROconst.jl")
include("tjlfep_pin_threads.jl")
include("tjlfep_ky.jl")
include("tjlfep_kwscale_scan.jl")
include("tjlfep_TM.jl")
include("tjlfep_ky_widthscan.jl")
include("tjlfep_ql_extract.jl")
include("mainsub.jl")
include("tjlfep_complete_output.jl")
include("run_tjlfep_file.jl")

include("tjlfep_generate_input.jl")

include("plotCritGrads.jl")

# Generic function bindings for the IMAS-path entry points implemented in
# ext/TJLFEPIMASExt.jl. Declaring them here lets the extension add methods, and lets
# TJLFEP export the names even when the extension is not loaded.

"""
    preprocess_imas_inputs(dd, rho, OptionsDict; kw...)

Build TJLFEP's `Options`/`profile` inputs directly from an IMAS data dictionary
(`dd`) on the requested `rho` grid, mirroring the `input.gacode` preprocessing.

Method provided by the `TJLFEPIMASExt` package extension, which loads only when
IMAS, GACODE, and TurbulentTransport are all present (e.g. under FUSE); calling
it without the extension raises a `MethodError`.
"""
function preprocess_imas_inputs end

"""
    save_imas_preprocessed_inputs(args...; kw...)

Write the IMAS-derived TJLFEP inputs to disk (the file-based `input.TGLFEP` /
`input.MTGLF` / `input.EXPRO` set) so an IMAS run can be reproduced through the
file path.

Method provided by the `TJLFEPIMASExt` package extension (loaded under
FUSE/IMAS); a `MethodError` is raised if the extension is not loaded.
"""
function save_imas_preprocessed_inputs end

"""
    remap_extraEP_for_fortran_save!(args...; kw...)

Remap the extra energetic-particle species bookkeeping into the layout the
Fortran-style file save expects (used when persisting IMAS-derived inputs).

Method provided by the `TJLFEPIMASExt` package extension (loaded under
FUSE/IMAS); a `MethodError` is raised if the extension is not loaded.
"""
function remap_extraEP_for_fortran_save! end

"""
    runTHD_dd_radius(args...; kw...)

Single-program-multiple-data (SPMD) per-radius entry point used by the MPS-team
multi-GPU layout: runs one radial point of an IMAS-`dd` scan on a worker.

Method provided by the `TJLFEPIMASExt` package extension (loaded under
FUSE/IMAS); a `MethodError` is raised if the extension is not loaded.
"""
function runTHD_dd_radius end

export profile, Options, InputTJLF  # InputTJLF is TJLF.InputTJLF (single consolidated type)
export WIDTH_FLOOR, width_floor, enforce_width_floor!
export readMTGLF, readTGLFEP, TJLF_map, readEXPRO, save_TGLFEP, save_MTGLF, save_EXPRO, save_all
export read_input_profile, readprofile, expro_dict_from_profile, ir_exp_from_scan, setup_fortran_file_inputs, setup_gacode_file_inputs, profile_from_gacode
export expro_vectors_from_gacode, preprocess_gacode_inputs
export expro_species_for_gacode_is_ep, read_gacode_scalar_field, read_gacode_ion_field
export expro_bound_deriv, expro_log_gradients, read_expro_for_alpha, compute_alpha_crit_profiles
export tjlfep_complete_output
export runTHD, runTHD_from_gacode
export TJLFEP_TM, TJLFEP_ky_widthscan
export gamma_dgamma_dfactor, gamma_grad, marginal_factor, marginal_factor_df, marginal_factor_faithful, critical_factor_grid, critical_factor_optimize, critical_factor_robust, critical_factor_confirm, critical_factor_profile, gamma_input_sensitivities, critical_factor_ad_f1seed, critical_factor_truth, critical_factor_triggered, critical_factor_multistart, critical_factor_nlopt
export MarginalQLData, extract_marginal_ql, build_alpha_ql_modes
export ql_flux_scan_at_marginal, diff_star_from_D_W
export populate_tjlfep_profile!
export run_gacode_scan_task, finalize_gacode_scan, slurm_array_task_id
export pin_threads_to_cores!

export TJLFEP_generate_input, readline_values

# IMAS-path entry points (methods provided by TJLFEPIMASExt when IMAS/GACODE/TurbulentTransport
# are loaded). InputTGLFEP now lives in TurbulentTransport (use `TurbulentTransport.InputTGLFEP`).
export preprocess_imas_inputs, save_imas_preprocessed_inputs, remap_extraEP_for_fortran_save!
export runTHD_dd_radius

export make_crit_grad_plots, read_crit_grad, write_crit_grad
export TJLF

# Environment knobs were renamed TJLFEP_* -> TGLFEP_JULIA_* (2026-11). Honor the old names for one
# release: at load time copy any TJLFEP_<X> into TGLFEP_JULIA_<X> that is not set, with one warning.
function _migrate_legacy_env!()
    legacy = [k for k in keys(ENV) if startswith(k, "TJLFEP_")]
    isempty(legacy) && return nothing
    moved = String[]
    for k in legacy
        new = "TGLFEP_JULIA_" * k[length("TJLFEP_")+1:end]
        haskey(ENV, new) || (ENV[new] = ENV[k]; push!(moved, k))
    end
    isempty(moved) || @warn "TJLFEP_* environment variables are deprecated; use TGLFEP_JULIA_* (copied for this session)" moved
    return nothing
end
__init__() = _migrate_legacy_env!()

# Collect the exported names for the auto-generated API reference (docs/make.jl).
# Mirrors TJLF: list public symbols, skipping the module itself and the
# re-exported `TJLF` submodule (documented in its own package).
const document = Dict()
document[Symbol(@__MODULE__)] = [
    name for name in Base.names(@__MODULE__; all=false, imported=false)
    if name != Symbol(@__MODULE__) && name != :TJLF
]

end 