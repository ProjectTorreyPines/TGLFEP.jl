# CPU scan driver: one Julia master spawns one worker per Slurm task (SlurmClusterManager)
# and `pmap`s the SCAN_N radii over them. Fewer workers than radii is fine (the 1-node
# template runs 2 workers over 20 radii); the master writes the merged SFmin profile and
# alpha_*_crit.input files into OUT_DIR itself, so no separate merge step is needed.
#
# Run with an explicit `--project` (the slurm/ templates do): workers inherit the master's
# active project, so a prebuilt CFS sysimage is paired with its published env_lean project
# and a JIT run with the TJLFEP checkout. No Pkg.activate here on purpose.
#
# Env (all set by the slurm/ templates):
#   CASE_DIR / GACODE_FILE / TGLFEP_FILE   case inputs (input.gacode + input.TGLFEP)
#   OUT_DIR                                 output directory (created)
#   SOLVER, REFINE_ROUNDS                   grid | ad | robust_ad | truth ; robust_ad/truth rounds
#   JULIA_WORKER_THREADS                    threads per worker (default: SLURM_CPUS_PER_TASK)
#   TGLFEP_JULIA_SYSIMAGE                         optional CPU sysimage for the workers ("" = JIT)
#   TGLFEP_JULIA_INSTANTIATE=1                    run Pkg.instantiate()+precompile() first (never
#                                           against the read-only published CFS env)

function logmsg(args...)
    println(args...)
    flush(stdout)
    flush(stderr)
end

const TGLFEP_JULIA_ROOT = normpath(@__DIR__, "..", "..")
const PROJECT = dirname(Base.active_project())
const CASE_DIR = get(ENV, "CASE_DIR", joinpath(TGLFEP_JULIA_ROOT, "examples", "DIIID_202017C42_500ms_v3.1"))
const GACODE_FILE = get(ENV, "GACODE_FILE", joinpath(CASE_DIR, "input.gacode"))
const TGLFEP_FILE = get(ENV, "TGLFEP_FILE", joinpath(CASE_DIR, "input_scan20_nb32.TGLFEP"))
const OUT_DIR = get(ENV, "OUT_DIR", joinpath(TGLFEP_JULIA_ROOT, "runs", "tglfep_julia_cpu_$(get(ENV, "SLURM_JOB_ID", "local"))"))
const THREADS_PER_WORKER = parse(Int, get(ENV, "JULIA_WORKER_THREADS", get(ENV, "SLURM_CPUS_PER_TASK", "64")))

@assert isfile(GACODE_FILE) "missing $GACODE_FILE"
@assert isfile(TGLFEP_FILE) "missing $TGLFEP_FILE"
mkpath(OUT_DIR)

job_t0 = time()

if get(ENV, "TGLFEP_JULIA_INSTANTIATE", "0") == "1"
    using Pkg
    tp = time()
    Pkg.instantiate()
    Pkg.precompile()
    logmsg("TIMING_RESULT backend=julia device=cpu path=distributed phase=precompile seconds=$(round(time() - tp; digits=3))")
end

using Distributed
# SlurmClusterManager is a TJLFEP dependency but not a direct dep of the published env_lean
# project (used with the CFS sysimage), so load it through the manifest by UUID instead of a
# bare `using` from Main, which only resolves direct deps of the active project.
const SlurmClusterManager = Base.require(Base.PkgId(Base.UUID("c82cd089-7bf7-41d7-976b-6b5d413cbe0a"), "SlurmClusterManager"))
const SlurmManager = SlurmClusterManager.SlurmManager
using Printf
using TJLFEP
using TJLF
using LinearAlgebra

BLAS.set_num_threads(1)

opts, _, _ = preprocess_gacode_inputs(GACODE_FILE, TGLFEP_FILE)
const SCAN_N = opts.SCAN_N
const N_BASIS = opts.N_BASIS

logmsg("TIMING_START backend=julia device=cpu path=distributed nodes=$(get(ENV, "SLURM_NNODES", "?")) tasks=$(get(ENV, "SLURM_NTASKS", "?")) SCAN_N=$SCAN_N N_BASIS=$N_BASIS")

_sysimage = get(ENV, "TGLFEP_JULIA_SYSIMAGE", "")
exeflags = if !isempty(_sysimage) && isfile(_sysimage)
    `--project=$(PROJECT) --sysimage=$(_sysimage) -t $(THREADS_PER_WORKER) --startup-file=no`
else
    `--project=$(PROJECT) -t $(THREADS_PER_WORKER) --startup-file=no`
end
logmsg("project=", PROJECT, "  worker sysimage=", (!isempty(_sysimage) && isfile(_sysimage)) ? _sysimage : "none (JIT)")
worker_env = Dict{String,String}()
haskey(ENV, "JULIA_DEPOT_PATH") && (worker_env["JULIA_DEPOT_PATH"] = ENV["JULIA_DEPOT_PATH"])
worker_env["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

tw = time()
if haskey(ENV, "SLURM_JOB_ID") || haskey(ENV, "SLURM_JOBID")
    ntasks = parse(Int, get(ENV, "SLURM_NTASKS", string(SCAN_N)))
    ntasks >= 1 || error("SLURM_NTASKS=$ntasks: need at least one task for the workers")
    logmsg("SlurmClusterManager: ntasks=$ntasks workers over $SCAN_N radii (pmap load-balances), threads/worker=$THREADS_PER_WORKER")
    # launch_timeout: default 60s is too tight when the depot precompile cache is cold
    # (a Project/Manifest bump since the sysimage bake triggers a fresh precompile pass).
    addprocs(SlurmManager(launch_timeout=1200.0); exeflags=exeflags, env=worker_env)
else
    addprocs(min(SCAN_N, Sys.CPU_THREADS ÷ max(THREADS_PER_WORKER, 1), 4); exeflags=exeflags, env=worker_env)
end
@everywhere begin
    using TJLFEP
    using TJLF
    using LinearAlgebra
    BLAS.set_num_threads(1)
end
logmsg("TIMING_RESULT backend=julia device=cpu path=distributed phase=worker_setup seconds=$(round(time() - tw; digits=3)) workers=$(nworkers())")

solver = Symbol(get(ENV, "SOLVER", "grid"))
refine_rounds = parse(Int, get(ENV, "REFINE_ROUNDS", "1"))
logmsg("=== CPU distributed scan: solver=$solver refine_rounds=$refine_rounds workers=$(nworkers()) SCAN_N=$SCAN_N N_BASIS=$N_BASIS ===")
logmsg("OUT_DIR=$OUT_DIR")

tc = time()
width, kymark, SFmin, dpdr, dndr = cd(OUT_DIR) do
    runTHD_from_gacode(GACODE_FILE, TGLFEP_FILE; printout=true, use_gpu=false, parallel=:distributed,
                       solver=solver, refine_rounds=refine_rounds)
end
compute_s = time() - tc
total_s = time() - job_t0

logmsg("SFmin = ", SFmin)
# Same per-radius table the GPU templates' merge step writes.
open(joinpath(OUT_DIR, "sfmin_scan.txt"), "w") do io
    for (i, s) in enumerate(SFmin)
        println(io, i, " ", opts.IR_EXP[i], " ", s)
    end
end
logmsg(@sprintf("TIMING_RESULT backend=julia device=cpu solver=%s path=distributed phase=compute seconds=%.3f SCAN_N=%d N_BASIS=%d workers=%d threads_per_worker=%d",
    solver, compute_s, SCAN_N, N_BASIS, nworkers(), THREADS_PER_WORKER))
logmsg(@sprintf("TIMING_RESULT backend=julia device=cpu solver=%s path=distributed phase=total_job seconds=%.3f SCAN_N=%d N_BASIS=%d nodes=%s",
    solver, total_s, SCAN_N, N_BASIS, get(ENV, "SLURM_NNODES", "?")))
logmsg("=== done; outputs in $OUT_DIR ===")
