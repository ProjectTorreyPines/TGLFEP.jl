# SCAN_N=20 distributed CPU timing. Set N_BASIS via env (default 6).
get(ENV, "TGLFEP_JULIA_FILE_ONLY", "0") == "1" || (ENV["TGLFEP_JULIA_FILE_ONLY"] = "1")

using Pkg
Pkg.activate("..")

function logmsg(args...)
    println(args...)
    flush(stdout)
    flush(stderr)
end

const TGLFEP_JULIA_ROOT = normpath(@__DIR__, "..", "..")
const N_BASIS = parse(Int, get(ENV, "N_BASIS", "6"))
const CASE_DIR = get(ENV, "CASE_DIR", joinpath(TGLFEP_JULIA_ROOT, "examples", "DIIID_202017C42_500ms_v3.1"))
const FILE_DIR = get(ENV, "FILE_DIR", joinpath(TGLFEP_JULIA_ROOT, "dev", "fileInput_nb$(N_BASIS)_scan20_$(get(ENV, "SLURM_JOB_ID", "local"))"))
const SCAN_N = parse(Int, get(ENV, "SCAN_N", "20"))
const THREADS_PER_WORKER = parse(Int, get(ENV, "JULIA_WORKER_THREADS", get(ENV, "SLURM_CPUS_PER_TASK", "64")))
const TGLFEP_FILE = get(ENV, "TGLFEP_FILE", joinpath(CASE_DIR, "input_scan20_nb$(N_BASIS).TGLFEP"))
const GACODE_DUMP = get(ENV, "GACODE_DUMP", joinpath(CASE_DIR, "input.gacode"))
haskey(ENV, "GACODE_DUMP") || (ENV["GACODE_DUMP"] = GACODE_DUMP)

@assert isfile(joinpath(CASE_DIR, "dump.profile"))
@assert isfile(TGLFEP_FILE)
@assert isfile(GACODE_DUMP)

job_t0 = time()
logmsg("TIMING_START backend=julia device=cpu path=distributed nodes=$(get(ENV, "SLURM_NNODES", "?")) tasks=$(get(ENV, "SLURM_NTASKS", "?")) SCAN_N=$SCAN_N N_BASIS=$N_BASIS")

Pkg.instantiate()
tp = time()
Pkg.precompile()
logmsg("TIMING_RESULT backend=julia device=cpu path=distributed phase=precompile seconds=$(round(time() - tp; digits=3)) SCAN_N=$SCAN_N N_BASIS=$N_BASIS")

using Distributed
using SlurmClusterManager
using Printf
using TJLFEP
using TJLF
using LinearAlgebra

BLAS.set_num_threads(1)

project_path = TGLFEP_JULIA_ROOT
_sysimage = get(ENV, "TGLFEP_JULIA_SYSIMAGE", joinpath(TGLFEP_JULIA_ROOT, "dev", "TGLFEP_JULIA_cpu_sysimage.so"))
exeflags = if isfile(_sysimage)
    `--project=$(project_path) --sysimage=$(_sysimage) -t $(THREADS_PER_WORKER)`
else
    `--project=$(project_path) -t $(THREADS_PER_WORKER)`
end
logmsg("worker exeflags sysimage=", isfile(_sysimage) ? _sysimage : "none")
worker_env = Dict{String,String}()
haskey(ENV, "JULIA_DEPOT_PATH") && (worker_env["JULIA_DEPOT_PATH"] = ENV["JULIA_DEPOT_PATH"])
worker_env["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
worker_env["TGLFEP_JULIA_FILE_ONLY"] = "1"
worker_env["GACODE_DUMP"] = GACODE_DUMP

tw = time()
if haskey(ENV, "SLURM_JOB_ID") || haskey(ENV, "SLURM_JOBID")
    ntasks = parse(Int, get(ENV, "SLURM_NTASKS", string(SCAN_N)))
    @assert ntasks == SCAN_N
    logmsg("SlurmClusterManager: ntasks=$ntasks threads=$THREADS_PER_WORKER")
    # launch_timeout: default 60s is too tight when the depot precompile cache is cold
    # (a Project/Manifest bump since the sysimage bake triggers a fresh precompile pass).
    addprocs(SlurmManager(launch_timeout=1200.0); exeflags=exeflags, env=worker_env)
else
    addprocs(SCAN_N; exeflags=exeflags, env=worker_env)
end
@everywhere begin
    get(ENV, "TGLFEP_JULIA_FILE_ONLY", "0") == "1" || (ENV["TGLFEP_JULIA_FILE_ONLY"] = "1")
    using TJLFEP
    using TJLF
    using LinearAlgebra
    BLAS.set_num_threads(1)
end
logmsg("TIMING_RESULT backend=julia device=cpu path=distributed phase=worker_setup seconds=$(round(time() - tw; digits=3)) workers=$(nworkers())")

if !isfile(joinpath(FILE_DIR, "input.MTGLF"))
    setup_fortran_file_inputs(CASE_DIR, FILE_DIR; tglfep_file=TGLFEP_FILE)
end

use_gpu = false
tglfep = abspath(joinpath(FILE_DIR, "input.TGLFEP"))
mtglf = abspath(joinpath(FILE_DIR, "input.MTGLF"))
expro = abspath(joinpath(FILE_DIR, "input.EXPRO"))
outdir = abspath(joinpath(TGLFEP_JULIA_ROOT, "dev", "timing_out_scan20_nb$(N_BASIS)_cpu_$(get(ENV, "SLURM_JOB_ID", "local"))_dist"))
mkpath(outdir)

logmsg("workers=", nworkers(), " FILE_DIR=", FILE_DIR)
logmsg("output -> ", outdir)

solver = Symbol(get(ENV, "SOLVER", "grid"))
refine_rounds = parse(Int, get(ENV, "REFINE_ROUNDS", "1"))

tc = time()
cd(outdir) do
    runTHD(tglfep, mtglf, expro; printout=false, use_gpu=use_gpu, parallel=:distributed,
           solver=solver, refine_rounds=refine_rounds)
end
compute_s = time() - tc
total_s = time() - job_t0

logmsg(@sprintf("TIMING_RESULT backend=julia device=cpu solver=%s path=distributed phase=compute seconds=%.3f SCAN_N=%d N_BASIS=%d workers=%d threads_per_worker=%d",
    solver, compute_s, SCAN_N, N_BASIS, nworkers(), THREADS_PER_WORKER))
logmsg(@sprintf("TIMING_RESULT backend=julia device=cpu solver=%s path=distributed phase=total_job seconds=%.3f SCAN_N=%d N_BASIS=%d nodes=%s",
    solver, total_s, SCAN_N, N_BASIS, get(ENV, "SLURM_NNODES", "?")))
