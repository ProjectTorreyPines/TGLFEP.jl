# Single-radius, file-based debug run (N_BASIS=6, SCAN_N=1).
# Run: TGLFEP_JULIA_DEBUG=1 julia --project=.. -t 8 debug_compare_nb6.jl

# Set before loading TJLFEP so IMAS/FUSE are not imported.
get(ENV, "TGLFEP_JULIA_FILE_ONLY", "0") == "1" || (ENV["TGLFEP_JULIA_FILE_ONLY"] = "1")

using Pkg
Pkg.activate("..")

function logmsg(args...)
    println(args...)
    flush(stdout)
    flush(stderr)
end

using TJLFEP
using TJLF
using LinearAlgebra

BLAS.set_num_threads(1)

const TGLFEP_JULIA_ROOT = normpath(@__DIR__, "..", "..")
const CASE_DIR = get(ENV, "CASE_DIR",
    joinpath(TGLFEP_JULIA_ROOT, "examples", "DIIID_202017C42_500ms_v3.1"))
const TGLFEP_FILE = get(ENV, "TGLFEP_FILE", joinpath(CASE_DIR, "input_singleradius_nb6.TGLFEP"))
const FILE_DIR = get(ENV, "FILE_DIR",
    joinpath(TGLFEP_JULIA_ROOT, "dev", "fileInput_nb6_$(get(ENV, "SLURM_JOB_ID", "local"))"))
# Physical ni/Ti and expro log-gradients (match Fortran expro_read).
const GACODE_DUMP = get(ENV, "GACODE_DUMP", joinpath(CASE_DIR, "input.gacode"))
if !haskey(ENV, "GACODE_DUMP")
    ENV["GACODE_DUMP"] = GACODE_DUMP
end

@assert isfile(joinpath(CASE_DIR, "dump.profile"))
@assert isfile(GACODE_DUMP)
@assert isfile(TGLFEP_FILE)

use_gpu = TJLF.pick_device(:auto) === :gpu
logmsg("TGLFEP_JULIA_DEBUG=", get(ENV, "TGLFEP_JULIA_DEBUG", "0"))
logmsg("device: ", use_gpu ? "GPU" : "CPU")
logmsg("CASE_DIR: ", CASE_DIR)
logmsg("TGLFEP_FILE: ", TGLFEP_FILE)
logmsg("FILE_DIR: ", FILE_DIR)

prof, ir_exp = setup_fortran_file_inputs(CASE_DIR, FILE_DIR; tglfep_file=TGLFEP_FILE)
logmsg("IR_EXP: ", ir_exp)

tglfep = abspath(joinpath(FILE_DIR, "input.TGLFEP"))
mtglf = abspath(joinpath(FILE_DIR, "input.MTGLF"))
expro = abspath(joinpath(FILE_DIR, "input.EXPRO"))
outdir = abspath(joinpath(TGLFEP_JULIA_ROOT, "dev", "debug_out_nb6_$(get(ENV, "SLURM_JOB_ID", "local"))"))
mkpath(outdir)

t0 = time()
cd(outdir) do
    logmsg("--- runTHD start (threads, 1 radius) ---")
    @time runTHD(tglfep, mtglf, expro; printout=true, use_gpu=use_gpu, parallel=:threads)
    make_crit_grad_plots("neither"; dir=".", code="julia")
end
logmsg("debug_compare OK in $(round(time() - t0; digits=1)) s -> $outdir")
