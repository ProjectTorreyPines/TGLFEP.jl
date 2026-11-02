# Shared setup for the FUSE-dd MPS-team SPMD layout (Option A: matches the verified DIII-D
# gacode SPMD path, but driving FUSE.ActorTJLFEP -> TJLFEP.runTHD(dd) -> ALPHA).
#
# Three phases, one batch job:
#   1. prepare  (run_fuse_dd_prepare.jl)  -- master builds the `dd` once + serializes inputs
#   2. per-radius (run_fuse_dd_mps_task.jl) -- srun -n SCAN_N, 1 radius : 1 GPU + MPS team
#   3. merge    (run_fuse_dd_merge.jl)     -- master runs ActorTJLFEP reading precomputed radii
#
# This file (included by prepare + merge) builds the FUSE `act` with parameters set IDENTICALLY
# from the environment in both phases, so the OptionsDict serialized in phase 1 (consumed by the
# per-radius tasks) is byte-for-byte what the phase-3 actor rebuilds internally.

using Pkg

const TGLFEP_JULIA_ROOT = get(ENV, "TGLFEP_JULIA_ROOT", normpath(@__DIR__, "..", ".."))
const FUSE_ROOT = get(ENV, "FUSE_ROOT", normpath(TGLFEP_JULIA_ROOT, "..", "FUSE"))
Pkg.activate(FUSE_ROOT)
push!(LOAD_PATH, TGLFEP_JULIA_ROOT)

function logmsg(args...)
    println(args...)
    flush(stdout)
    flush(stderr)
end

const CASE = Symbol(get(ENV, "CASE", "ITER"))
const SCAN_N = parse(Int, get(ENV, "SCAN_N", "20"))
const N_BASIS = parse(Int, get(ENV, "N_BASIS", "32"))
const NGRID = parse(Int, get(ENV, "NGRID", "201"))
const ALPHA_SOLVER = Symbol(get(ENV, "ALPHA_SOLVER", "stiff"))
# Pin the critical-factor engine explicitly (the ActorTJLFEP default is now :ad). These FUSE-dd
# timing runs benchmark the grid path by default; override with SOLVER=ad|robust_ad|truth.
const SOLVER = Symbol(get(ENV, "SOLVER", "grid"))
# solver=:ad width-extension knobs (honored when SOLVER=ad; harmless otherwise).
const EXTEND_MODE = Symbol(get(ENV, "AD_EXTEND_MODE", "locate"))
const WIDE_KDESC = parse(Int, get(ENV, "AD_WIDE_KDESC", "2"))
const FAITHFUL_CONFIRM = get(ENV, "AD_FAITHFUL_CONFIRM", "1") != "0"
const INNER = Symbol(get(ENV, "INNER", "mps_team"))
const MPS_TEAM = parse(Int, get(ENV, "MPS_TEAM", "8"))
const OUT_DIR = get(() -> error("set TGLFEP_JULIA_OUT_DIR (SPMD run directory)"), ENV, "TGLFEP_JULIA_OUT_DIR")
const TASKS_DIR = joinpath(OUT_DIR, "tasks")
const DD_IN_JSON = joinpath(OUT_DIR, "dd_in.json")
const OPTIONSDICT_JLS = joinpath(OUT_DIR, "optionsdict.jls")
const RHOSCAN_JLS = joinpath(OUT_DIR, "rho_scan.jls")

import FUSE

"""Build `act` from `case_parameters(CASE)` and set the TGLF-EP scan parameters from ENV.
Identical in prepare and merge so `_optionsdict` is deterministic across phases."""
function build_act()
    ini, act = FUSE.case_parameters(CASE; init_from=:ods)
    ini.core_profiles.ngrid = NGRID
    act.ActorTJLFEP.rho_scan = collect(range(0.05, 0.95; length=SCAN_N))
    act.ActorTJLFEP.n_basis = N_BASIS
    act.ActorTJLFEP.use_gpu = true
    act.ActorTJLFEP.alpha_solver = ALPHA_SOLVER
    act.ActorTJLFEP.solver = SOLVER
    for (prop, val) in ((:inner, INNER), (:mps_team, MPS_TEAM),
                        (:extend_mode, EXTEND_MODE), (:wide_kdesc, WIDE_KDESC),
                        (:faithful_confirm, FAITHFUL_CONFIRM))
        try
            setproperty!(act.ActorTJLFEP, prop, val)
        catch err
            logmsg("note: ActorTJLFEP.$prop not set ($(typeof(err))); using actor default")
        end
    end
    return ini, act
end

"""OptionsDict + rho_scan exactly as `ActorTJLFEP._step` would build them from `act`."""
function optionsdict_and_rho(act)
    par = FUSE.OverrideParameters(act.ActorTJLFEP)
    return FUSE._optionsdict(par), collect(Float64, par.rho_scan)
end
