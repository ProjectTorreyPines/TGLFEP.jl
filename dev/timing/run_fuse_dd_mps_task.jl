# Phase 2 of the FUSE-dd MPS-team SPMD layout: ONE radius of the SCAN_N scan, on ONE GPU, with
# an MPS *team* (this is the verified DIII-D layout, run_gacode_scan20_mps_task.jl, but the inputs
# come from the FUSE `dd` instead of an input.gacode file).
#
# 1 SLURM task = 1 radius = 1 GPU (pinned via CUDA_VISIBLE_DEVICES by mps-scan-wrapper.sh) + an
# MPS_TEAM-sized pool of local worker processes that share that GPU as MPS clients (Hyper-Q). The
# team is spawned at TOP LEVEL here (addprocs + @everywhere) -- exactly as in the gacode task --
# and passed explicitly to runTHD_dd_radius, so there is no nested-master addprocs.
#
# Loads the phase-1 artifacts (dd_in.json + optionsdict.jls + rho_scan.jls) from TGLFEP_JULIA_OUT_DIR,
# runs the single radius SCAN_INDEX, and writes tasks/task_<SCAN_INDEX>.jls (the pmap-element the
# phase-3 merge reassembles).
#
# Env knobs:
#   SCAN_INDEX (set by mps-scan-wrapper.sh)  MPS_TEAM (8)  INNER (mps_team|threads)  USE_GPU (1)
#   TGLFEP_JULIA_OUT_DIR  TEAM_GPUS/CUDA_VISIBLE_DEVICES  JULIA_WORKER_THREADS (2)  TGLFEP_JULIA_GPU_SYSIMAGE
#   SOLVER (grid)  AD_EXTEND_MODE (locate)  AD_WIDE_KDESC (2)  AD_FAITHFUL_CONFIRM (1)

using Pkg

const TGLFEP_JULIA_ROOT = get(ENV, "TGLFEP_JULIA_ROOT", normpath(@__DIR__, "..", ".."))
const FUSE_ROOT = get(ENV, "FUSE_ROOT", normpath(TGLFEP_JULIA_ROOT, "..", "FUSE"))
# FUSE_PROJECT (fuse_actor_spmd.sbatch: the floored master env dev/timing/aps2026/fuse_env_f05) wins over the FUSE
# checkout; a bare Pkg.activate(FUSE_ROOT) here silently overrode --project and loaded the registered TJLFEP 2.0.18
# (no WIDTH_FLOOR) in all 20 masters of job 60626960 -> caught by the floor guard below.
Pkg.activate(get(ENV, "FUSE_PROJECT", FUSE_ROOT))
push!(LOAD_PATH, TGLFEP_JULIA_ROOT)

@assert get(ENV, "USE_GPU", "") == "1" "run_fuse_dd_mps_task.jl is GPU-only; set USE_GPU=1"

using CUDA
let rv = CUDA.runtime_version()
    rv >= v"12.6" || error("GPU run needs CUDA >= 12.6 for cusolverDnXgeev, but CUDA runtime is $rv. Load cudatoolkit/12.9.")
end

using Distributed

const NTEAM = parse(Int, get(ENV, "MPS_TEAM", "8"))
const THREADS_PER_WORKER = parse(Int, get(ENV, "JULIA_WORKER_THREADS", "2"))
const INNER = Symbol(get(ENV, "INNER", "mps_team"))
# Critical-factor engine + solver=:ad width-extension knobs (must match the phase-1/phase-3
# actor config so the per-radius result is consistent with the merge). Default :grid like the
# in-process timing path; run_tjlfep exports SOLVER/AD_* into the batch env.
const SOLVER = Symbol(get(ENV, "SOLVER", "grid"))
const EXTEND_MODE = Symbol(get(ENV, "AD_EXTEND_MODE", "locate"))
const WIDE_KDESC = parse(Int, get(ENV, "AD_WIDE_KDESC", "2"))
const FAITHFUL_CONFIRM = get(ENV, "AD_FAITHFUL_CONFIRM", "1") != "0"
const TEAM_GPUS = let s = get(ENV, "TEAM_GPUS", get(ENV, "CUDA_VISIBLE_DEVICES", "0"))
    String.(split(s, ',', keepempty=false))
end
const GPU_SYSIMAGE = get(ENV, "TGLFEP_JULIA_GPU_SYSIMAGE", "")
const _SYSIMG_FLAGS = (!isempty(GPU_SYSIMAGE) && isfile(GPU_SYSIMAGE)) ? `--sysimage=$(GPU_SYSIMAGE)` : ``
# Team-worker project. Default = the TJLFEP checkout (JIT). With a prebuilt sysimage the workers must run
# under the env the image was baked with (<image dir>/env_lean), otherwise a JIT fallback can recompile
# TJLFEP's deps on the compute node — SimulationParameters -> HDF5_jll fails there (OpenSSL clash,
# jobs 60382959/60464320, 2026-10-28/29). TGLFEP_JULIA_WORKER_PROJECT overrides explicitly.
const WORKER_PROJECT = get(ENV, "TGLFEP_JULIA_WORKER_PROJECT") do
    if !isempty(_SYSIMG_FLAGS)
        env = joinpath(dirname(GPU_SYSIMAGE), "env_lean")
        isfile(joinpath(env, "Project.toml")) ? env : TGLFEP_JULIA_ROOT
    else
        TGLFEP_JULIA_ROOT
    end
end

# Master loads IMAS + the TJLFEP IMAS extension (preprocess_imas_inputs/runTHD_dd_radius). Loading
# MUST happen BEFORE the team is spawned: Distributed's package callback replays every `using` the
# master performs after `addprocs` onto all workers by PkgId, so with this block after the team spawn
# the TJLFEP-project team workers were made to load (and precompile) IMAS -> HDF5_jll, which fails on
# the compute-node image (OpenSSL artifact 3.5 vs system libcrypto 3.1.4; job 60382959, 2026-10-28).
# IMAS + GACODE + TurbulentTransport triggers TJLFEPIMASExt without pulling all of FUSE at JIT;
# under the baked CFS sysimage these are all instant.
using IMAS
import GACODE
using TurbulentTransport
using TJLFEP
using TJLF
using LinearAlgebra
# Floor guard (2026-10-31): the per-radius driver runTHD_dd_radius -> _ad_width_floor executes in THIS (master)
# process, not on the team workers, so the master's TJLFEP must be the floored working tree. --project=FUSE resolves
# the registered 2.0.18 from the FUSE Manifest (no WIDTH_FLOOR) -> job 60622577 was cancelled. Opt out with
# TGLFEP_JULIA_ALLOW_UNFLOORED=1 (pre-floor comparison runs only).
const _MASTER_FLOOR = isdefined(TJLFEP, :WIDTH_FLOOR) ? TJLFEP.WIDTH_FLOOR : nothing
println("master TJLFEP=", pkgdir(TJLFEP), " v", pkgversion(TJLFEP), " WIDTH_FLOOR=", something(_MASTER_FLOOR, "none"))
if get(ENV, "TGLFEP_JULIA_ALLOW_UNFLOORED", "0") != "1" && _MASTER_FLOOR != 0.5
    error("master TJLFEP at $(pkgdir(TJLFEP)) (v$(pkgversion(TJLFEP))) has no WIDTH_FLOOR = 0.5; run with " *
          "--project=dev/timing/aps2026/fuse_env_f05 (FUSE_PROJECT in fuse_actor_spmd.sbatch) or set TGLFEP_JULIA_ALLOW_UNFLOORED=1")
end
BLAS.set_num_threads(1)

if INNER === :mps_team
    base_env = Dict{String,String}()
    for k in ("JULIA_DEPOT_PATH", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
              # JULIA_CUDA_USE_COMPAT=false is REQUIRED: CUDA.jl's forward-compat driver shim
              # hangs in cuInit when initializing as an MPS client on Perlmutter.
              "JULIA_CUDA_USE_COMPAT", "JULIA_CUDA_MEMORY_POOL")
        haskey(ENV, k) && (base_env[k] = ENV[k])
    end
    base_env["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    for w in 1:NTEAM
        env = copy(base_env)
        env["CUDA_VISIBLE_DEVICES"] = TEAM_GPUS[(w - 1) % length(TEAM_GPUS) + 1]
        # Team workers run the kw-scan eigensolves only -> TJLFEP project (CUDA/TJLF/TJLFEP),
        # NOT FUSE. Matches run_gacode_scan20_mps_task.jl.
        addprocs(1; exeflags=`--project=$(WORKER_PROJECT) -t $(THREADS_PER_WORKER) --startup-file=no $(_SYSIMG_FLAGS)`,
                 env=env)
    end
    @everywhere workers() begin
        using CUDA
        using TJLFEP
        using TJLF
        using LinearAlgebra
        BLAS.set_num_threads(1)
        CUDA.functional() && CUDA.device!(first(CUDA.devices()))
    end
end


const OUT_DIR = get(() -> error("set TGLFEP_JULIA_OUT_DIR"), ENV, "TGLFEP_JULIA_OUT_DIR")
const TASKS_DIR = joinpath(OUT_DIR, "tasks")
const DD_IN_JSON = joinpath(OUT_DIR, "dd_in.json")
const OPTIONSDICT_JLS = joinpath(OUT_DIR, "optionsdict.jls")
const RHOSCAN_JLS = joinpath(OUT_DIR, "rho_scan.jls")
@assert isfile(DD_IN_JSON) "missing $DD_IN_JSON (run prepare phase first)"

using Serialization
dd = IMAS.json2imas(DD_IN_JSON)
OptionsDict = Serialization.deserialize(OPTIONSDICT_JLS)
rho_scan = Serialization.deserialize(RHOSCAN_JLS)
scan_n = OptionsDict["SCAN_N"]

const SCAN_INDEX = haskey(ENV, "SCAN_INDEX") ? parse(Int, ENV["SCAN_INDEX"]) :
    (parse(Int, get(ENV, "SLURM_PROCID", "0")) + 1)
@assert 1 <= SCAN_INDEX <= scan_n "SCAN_INDEX=$SCAN_INDEX invalid for SCAN_N=$scan_n"

const TEAM = INNER === :mps_team ? workers() : nothing
println("=== fuse-dd scan task (inner=$INNER) ===")
println("worker sysimage=", isempty(_SYSIMG_FLAGS.exec) ? "<none, JIT>" : GPU_SYSIMAGE)
println("scan_index=$SCAN_INDEX / $scan_n  team=$(INNER === :mps_team ? nworkers() : 0) workers  ",
        "threads/worker=$THREADS_PER_WORKER  team_gpus=$(join(TEAM_GPUS, ','))")
println("OUT_DIR=$OUT_DIR")
println("CUDA_VISIBLE_DEVICES=$(get(ENV, "CUDA_VISIBLE_DEVICES", "<unset>")) MPS=$(get(ENV, "CUDA_MPS_PIPE_DIRECTORY", "<none>"))")
println("GPU=", CUDA.functional() ? CUDA.name(first(CUDA.devices())) : "n/a")
flush(stdout)

t0 = time()
task_file = TJLFEP.runTHD_dd_radius(dd, rho_scan, OptionsDict, SCAN_INDEX;
    out_dir=TASKS_DIR, use_gpu=true, inner=INNER, team=TEAM, solver=SOLVER,
    extend_mode=EXTEND_MODE, wide_kdesc=WIDE_KDESC, faithful_confirm=FAITHFUL_CONFIRM)
println("OK scan_index=$SCAN_INDEX -> $task_file in $(round(time() - t0; digits=1)) s")
flush(stdout)
