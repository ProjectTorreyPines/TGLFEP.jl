# N_BASIS=6, SCAN_N=20, multi-node (SlurmClusterManager + pmap).
# Run: sbatch batch_debug_nb6_julia_scan20_10n.sh

get(ENV, "TGLFEP_JULIA_FILE_ONLY", "0") == "1" || (ENV["TGLFEP_JULIA_FILE_ONLY"] = "1")

using Pkg
Pkg.activate("..")

function logmsg(args...)
    println(args...)
    flush(stdout)
    flush(stderr)
    return nothing
end

const TGLFEP_JULIA_ROOT = normpath(@__DIR__, "..", "..")
const CASE_DIR = get(ENV, "CASE_DIR",
    joinpath(TGLFEP_JULIA_ROOT, "examples", "DIIID_202017C42_500ms_v3.1"))
const FILE_DIR = get(ENV, "FILE_DIR",
    joinpath(TGLFEP_JULIA_ROOT, "dev", "fileInput_nb6_scan20_$(get(ENV, "SLURM_JOB_ID", "local"))"))
const SCAN_N = parse(Int, get(ENV, "SCAN_N", "20"))
const THREADS_PER_WORKER = parse(Int, get(ENV, "JULIA_WORKER_THREADS",
    get(ENV, "SLURM_CPUS_PER_TASK", "64")))
const TGLFEP_FILE = get(ENV, "TGLFEP_FILE", joinpath(CASE_DIR, "input_scan20_nb6.TGLFEP"))
const GACODE_DUMP = get(ENV, "GACODE_DUMP", joinpath(CASE_DIR, "input.gacode"))
if !haskey(ENV, "GACODE_DUMP")
    ENV["GACODE_DUMP"] = GACODE_DUMP
end

@assert isfile(joinpath(CASE_DIR, "dump.profile"))
@assert isfile(TGLFEP_FILE)
@assert isfile(GACODE_DUMP)

logmsg("=== nb6 scan20 distributed (depot=", get(ENV, "JULIA_DEPOT_PATH", "<default>"), ") ===")
Pkg.instantiate()
t0 = time()
Pkg.precompile()
logmsg("Pkg.precompile done in ", round(time() - t0; digits=1), " s")

using Distributed
using SlurmClusterManager
using TJLFEP
using TJLF
using LinearAlgebra

BLAS.set_num_threads(1)
logmsg("TJLF: ", pathof(TJLF))
logmsg("TJLFEP: ", pathof(TJLFEP))

project_path = TGLFEP_JULIA_ROOT
_sysimage = get(ENV, "TGLFEP_JULIA_SYSIMAGE", joinpath(TGLFEP_JULIA_ROOT, "dev", "TGLFEP_JULIA_cpu_sysimage.so"))
exeflags = if isfile(_sysimage)
    `--project=$(project_path) --sysimage=$(_sysimage) -t $(THREADS_PER_WORKER)`
else
    `--project=$(project_path) -t $(THREADS_PER_WORKER)`
end
logmsg("worker exeflags sysimage=", isfile(_sysimage) ? _sysimage : "none")

worker_env = Dict{String,String}()
if haskey(ENV, "JULIA_DEPOT_PATH")
    worker_env["JULIA_DEPOT_PATH"] = ENV["JULIA_DEPOT_PATH"]
end
worker_env["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
worker_env["TGLFEP_JULIA_FILE_ONLY"] = "1"
worker_env["GACODE_DUMP"] = GACODE_DUMP

if haskey(ENV, "SLURM_JOB_ID") || haskey(ENV, "SLURM_JOBID")
    ntasks = parse(Int, get(ENV, "SLURM_NTASKS", string(SCAN_N)))
    @assert ntasks == SCAN_N "SLURM_NTASKS ($ntasks) must equal SCAN_N ($SCAN_N)"
    logmsg("SlurmClusterManager: ntasks=$ntasks threads=$THREADS_PER_WORKER")
    addprocs(SlurmManager(); exeflags=exeflags, env=worker_env)
else
    logmsg("Local test: addprocs($SCAN_N)")
    addprocs(SCAN_N; exeflags=exeflags, env=worker_env)
end

@everywhere begin
    get(ENV, "TGLFEP_JULIA_FILE_ONLY", "0") == "1" || (ENV["TGLFEP_JULIA_FILE_ONLY"] = "1")
    using TJLFEP
    using TJLF
    using LinearAlgebra
    BLAS.set_num_threads(1)
end

@everywhere println("worker ", myid(), " on ", gethostname(), " GACODE_DUMP=", get(ENV, "GACODE_DUMP", ""))

if !isfile(joinpath(FILE_DIR, "input.MTGLF"))
    setup_fortran_file_inputs(CASE_DIR, FILE_DIR; tglfep_file=TGLFEP_FILE)
end

use_gpu = TJLF.pick_device(:auto) === :gpu
tglfep = abspath(joinpath(FILE_DIR, "input.TGLFEP"))
mtglf = abspath(joinpath(FILE_DIR, "input.MTGLF"))
expro = abspath(joinpath(FILE_DIR, "input.EXPRO"))
outdir = abspath(joinpath(TGLFEP_JULIA_ROOT, "dev", "debug_out_nb6_scan20_$(get(ENV, "SLURM_JOB_ID", "local"))_dist"))
mkpath(outdir)

logmsg("workers=", nworkers(), " FILE_DIR=", FILE_DIR)
logmsg("output -> ", outdir)

t0 = time()
cd(outdir) do
    @time runTHD(tglfep, mtglf, expro; printout=true, use_gpu=use_gpu, parallel=:distributed)
    make_crit_grad_plots("neither"; dir=".", code="julia")
end
logmsg("OK in ", round(time() - t0; digits=1), " s")
