# Full FUSE ITER case on the 5-node / 20-GPU layout (1 radius : 1 A100), GPU.
#
# This drives the *FUSE* path -- FUSE.ActorTJLFEP -> TJLFEP.runTHD(dd, ...) -> ALPHA.run_alpha --
# at the production basis count (N_BASIS=32) over a SCAN_N=20 rho scan on GPU.
#
# Topology (differs from the gacode-file SPMD scan):
#   * The master process (this script) loads FUSE and builds the ITER `dd` once.
#   * It then addprocs(SlurmManager()) -> 20 worker tasks (4/node x 5 nodes), each PINNED to
#     one A100 via SLURM_LOCALID. runTHD(dd, ...) pmap's the 20 scan radii over workers(),
#     so each radius runs its full kw-scan on its own GPU. At 1 radius/GPU there is no GPU
#     oversubscription, so NO MPS is needed (unlike the gacode mps-team path).
#   * Workers load the baked GPU sysimage (TJLF/TJLFEP/CUDA) to skip the cold eigensolve JIT;
#     they do NOT need FUSE/IMAS (only per-radius TJLFEP structs are serialized to them).
#
# Env knobs:
#   SCAN_N (20)  N_BASIS (32)  NGRID (201)  ALPHA_SOLVER (stiff)  JULIA_WORKER_THREADS (8)
#   TGLFEP_JULIA_GPU_SYSIMAGE  FUSE_ROOT  TGLFEP_JULIA_ROOT

using Pkg

const TGLFEP_JULIA_ROOT = get(ENV, "TGLFEP_JULIA_ROOT", normpath(@__DIR__, "..", ".."))
const FUSE_ROOT = get(ENV, "FUSE_ROOT", normpath(TGLFEP_JULIA_ROOT, "..", "FUSE"))
Pkg.activate(FUSE_ROOT)
# Stack the TJLFEP environment so the master can `using SlurmClusterManager` (a TJLFEP dep,
# not a FUSE dep) while FUSE stays the active/primary project. FUSE/IMAS/ALPHA resolve from
# the active FUSE env; SlurmClusterManager falls through to TJLFEP. Non-invasive: no change
# to FUSE's Project.toml.
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
# Pin the critical-factor engine explicitly (the ActorTJLFEP default is now :ad). This FUSE-dd
# timing run benchmarks the grid path by default; override with SOLVER=ad|robust_ad|truth.
const SOLVER = Symbol(get(ENV, "SOLVER", "grid"))
# solver=:ad width-extension knobs (honored when SOLVER=ad; harmless otherwise).
const EXTEND_MODE = Symbol(get(ENV, "AD_EXTEND_MODE", "locate"))
const WIDE_KDESC = parse(Int, get(ENV, "AD_WIDE_KDESC", "2"))
const FAITHFUL_CONFIRM = get(ENV, "AD_FAITHFUL_CONFIRM", "1") != "0"
const INNER = Symbol(get(ENV, "INNER", "threads"))
const MPS_TEAM = parse(Int, get(ENV, "MPS_TEAM", "0"))
const OUT_DIR = get(ENV, "TGLFEP_JULIA_OUT_DIR", "")
const NNODES = get(ENV, "SLURM_NNODES", "?")
const NTASKS = get(ENV, "SLURM_NTASKS", "?")

job_t0 = time()
logmsg("TIMING_START backend=julia device=gpu path=fuse-dd-distributed nodes=$NNODES tasks=$NTASKS ",
    "SCAN_N=$SCAN_N N_BASIS=$N_BASIS NGRID=$NGRID alpha_solver=$ALPHA_SOLVER")

# ----------------------------------------------------------------------------------------------
# 1. Spin up the 20 GPU workers (1 radius : 1 GPU). Do this BEFORE building the dd so the
#    (expensive) FUSE.init on the master overlaps with worker spawn + GPU-context warmup.
# ----------------------------------------------------------------------------------------------
using Distributed
using SlurmClusterManager

const WORKER_THREADS = parse(Int, get(ENV, "JULIA_WORKER_THREADS", "8"))

# Worker sysimage is OFF by default: the master JIT-compiles TJLFEP from the dev source (via
# the FUSE project), and Distributed serializes the `runTHD` pmap *closure* (a gensym'd type
# like `#105#111`) to the workers. A baked sysimage built from an older TJLFEP source has a
# DIFFERENT closure numbering -> the workers can't deserialize it (UndefVarError #NNN#NNN in
# TJLFEP). JIT'ing TJLFEP from the same dev source on the workers keeps the closure types in
# sync. Opt in only when the sysimage is known to match the current source: TGLFEP_JULIA_GPU_SYSIMAGE=<path>.
const GPU_SYSIMG = get(ENV, "TGLFEP_JULIA_GPU_SYSIMAGE", "")
sysflag = (!isempty(GPU_SYSIMG) && isfile(GPU_SYSIMG)) ? `--sysimage=$(GPU_SYSIMG)` : ``
logmsg("worker sysimage = ", isempty(GPU_SYSIMG) ? "<none, JIT from dev source (closure-safe)>" : GPU_SYSIMG)
exeflags = `--project=$(TGLFEP_JULIA_ROOT) $(sysflag) -t $(WORKER_THREADS) --startup-file=no`

worker_env = Dict{String,String}()
haskey(ENV, "JULIA_DEPOT_PATH") && (worker_env["JULIA_DEPOT_PATH"] = ENV["JULIA_DEPOT_PATH"])
worker_env["JULIA_PROJECT"] = TGLFEP_JULIA_ROOT
worker_env["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
# REQUIRED on Perlmutter: CUDA.jl's forward-compat driver shim can hang in cuInit otherwise.
worker_env["JULIA_CUDA_USE_COMPAT"] = "false"

@assert haskey(ENV, "SLURM_NTASKS") "must run inside a SLURM allocation (SLURM_NTASKS unset)"
tw = time()
addprocs(SlurmManager(); exeflags=exeflags, env=worker_env)
logmsg("TIMING_RESULT backend=julia device=gpu path=fuse-dd-distributed phase=worker_spawn ",
    "seconds=$(round(time() - tw; digits=3)) workers=$(nworkers())")
@assert nworkers() == SCAN_N "expected $SCAN_N GPU workers (1 per radius), got $(nworkers())"

# Load the GPU eigensolve stack on the workers only and pin each to one A100.
@everywhere workers() begin
    using CUDA
    using TJLFEP
    using TJLF
    using LinearAlgebra
    BLAS.set_num_threads(1)
    if CUDA.functional()
        localid = parse(Int, get(ENV, "SLURM_LOCALID", "0"))
        ndev = length(collect(CUDA.devices()))
        CUDA.device!(localid % ndev)
    end
end
logmsg("TIMING_RESULT backend=julia device=gpu path=fuse-dd-distributed phase=worker_setup ",
    "seconds=$(round(time() - tw; digits=3)) workers=$(nworkers()) threads_per_worker=$WORKER_THREADS")

# Report the worker -> (host, GPU) mapping so we can confirm the 5x4 spread.
for w in workers()
    host, dev, gname = remotecall_fetch(w) do
        d = CUDA.functional() ? CUDA.deviceid(CUDA.device()) : -1
        g = CUDA.functional() ? CUDA.name(CUDA.device()) : "n/a"
        (gethostname(), d, g)
    end
    logmsg("  worker $w  host=$host  gpu=$dev  ($gname)")
end

# ----------------------------------------------------------------------------------------------
# 2. Build the ITER dd on the master (FUSE).
# ----------------------------------------------------------------------------------------------
import FUSE
import IMAS

td = time()
ini, act = FUSE.case_parameters(CASE; init_from=:ods)
ini.core_profiles.ngrid = NGRID
dd = IMAS.dd()
FUSE.init(dd, ini, act)
logmsg("TIMING_RESULT backend=julia device=gpu path=fuse-dd-distributed phase=dd_build ",
    "seconds=$(round(time() - td; digits=3)) NGRID=$NGRID")

# ----------------------------------------------------------------------------------------------
# 3. Configure ActorTJLFEP for the full GPU scan and run it (timed).
# ----------------------------------------------------------------------------------------------
act.ActorTJLFEP.rho_scan = collect(range(0.05, 0.95; length=SCAN_N))
act.ActorTJLFEP.n_basis = N_BASIS
act.ActorTJLFEP.use_gpu = true
act.ActorTJLFEP.alpha_solver = ALPHA_SOLVER
act.ActorTJLFEP.solver = SOLVER
# MPS-team concurrency on the FUSE-dd path (Section 8). Guarded so older ActorTJLFEP
# versions that lack these params still run (the timing harness sets neither var).
for (prop, val) in ((:inner, INNER), (:mps_team, MPS_TEAM),
                    (:extend_mode, EXTEND_MODE), (:wide_kdesc, WIDE_KDESC),
                    (:faithful_confirm, FAITHFUL_CONFIRM))
    try
        setproperty!(act.ActorTJLFEP, prop, val)
    catch err
        logmsg("note: ActorTJLFEP.$prop not set ($(typeof(err))); using actor default")
    end
end
logmsg("rho_scan = ", act.ActorTJLFEP.rho_scan)

ta = time()
actor = FUSE.ActorTJLFEP(dd, act)
actor_s = time() - ta
logmsg("TIMING_RESULT backend=julia device=gpu path=fuse-dd-distributed phase=actor ",
    "seconds=$(round(actor_s; digits=3)) SCAN_N=$SCAN_N N_BASIS=$N_BASIS workers=$(nworkers())")

# ----------------------------------------------------------------------------------------------
# 4. Report physics outputs (sanity + provenance).
# ----------------------------------------------------------------------------------------------
logmsg("SFmin      = ", round.(actor.SFmin; digits=4))
logmsg("width      = ", round.(actor.width; digits=4))
logmsg("kymark     = ", round.(actor.kymark; digits=4))
if actor.alpha !== nothing
    res = actor.alpha
    logmsg("ALPHA: n_EP[1]=", round(res.n_EP[1]; digits=4), " 10^19 m^-3   ",
        "p_EP[1]=", round(res.p_EP[1]; digits=4), " 10 kPa   rho_grid=", length(actor.rho_grid))
    ep = dd.core_profiles.profiles_1d[].ion[act.ActorTJLFEP.is_ep]
    logmsg("dd EP density_fast[1] = ", round(ep.density_fast[1]; digits=4), " m^-3")
end

# ----------------------------------------------------------------------------------------------
# 5. Persist results for run_tjlfep / load_tjlfep_results (when launched via run_tjlfep).
# ----------------------------------------------------------------------------------------------
if !isempty(OUT_DIR)
    using Serialization
    mkpath(OUT_DIR)
    n_EP = (actor.alpha !== nothing) ? actor.alpha.n_EP : Float64[]
    p_EP = (actor.alpha !== nothing) ? actor.alpha.p_EP : Float64[]
    Serialization.serialize(joinpath(OUT_DIR, "tglfep_julia_results.jls"),
        (; rho_scan=act.ActorTJLFEP.rho_scan, SFmin=actor.SFmin, width=actor.width,
            kymark=actor.kymark, n_EP=n_EP, p_EP=p_EP))
    try
        IMAS.imas2json(dd, joinpath(OUT_DIR, "dd_out.json"))
    catch err
        logmsg("note: imas2json(dd) failed ($(typeof(err))); skipping dd_out.json")
    end
    logmsg("persisted results to $OUT_DIR (tglfep_julia_results.jls + dd_out.json)")
end

total_s = time() - job_t0
logmsg("TIMING_RESULT backend=julia device=gpu path=fuse-dd-distributed phase=total_job ",
    "seconds=$(round(total_s; digits=3)) SCAN_N=$SCAN_N N_BASIS=$N_BASIS nodes=$NNODES tasks=$NTASKS")
logmsg("=== done ===")
