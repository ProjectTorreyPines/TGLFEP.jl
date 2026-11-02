# DIII-D validation without sysimage — Fortran file-based runTHD (dump.profile + input.TGLFEP).
# Run: module load julia/1.11.7; export JULIA_DEPOT_PATH=$PSCRATCH/.julia
#      julia --project=.. -t 32 validate_nosysimage.jl

using Pkg
Pkg.activate("..")

using TJLFEP
using TJLF
using LinearAlgebra

BLAS.set_num_threads(1)
use_gpu = TJLF.pick_device(:auto) === :gpu

const TGLFEP_JULIA_ROOT = normpath(@__DIR__, "..", "..")
const CASE_DIR = get(ENV, "CASE_DIR",
    joinpath(TGLFEP_JULIA_ROOT, "examples", "DIIID_202017C42_500ms_v3.1"))
const FILE_DIR = get(ENV, "FILE_DIR",
    joinpath(TGLFEP_JULIA_ROOT, "dev", "fileInput_$(get(ENV, "SLURM_JOB_ID", "local"))"))

@assert isfile(joinpath(CASE_DIR, "dump.profile")) "missing dump.profile in $CASE_DIR"
@assert isfile(joinpath(CASE_DIR, "input.TGLFEP")) "missing input.TGLFEP in $CASE_DIR"

println("device: ", use_gpu ? "GPU" : "CPU")
println("CASE_DIR: ", CASE_DIR)
println("FILE_DIR: ", FILE_DIR)

if !isfile(joinpath(FILE_DIR, "input.MTGLF")) || !isfile(joinpath(FILE_DIR, "input.EXPRO"))
    println("--- setup_fortran_file_inputs ---")
    setup_fortran_file_inputs(CASE_DIR, FILE_DIR)
end

tglfep = joinpath(FILE_DIR, "input.TGLFEP")
mtglf = joinpath(FILE_DIR, "input.MTGLF")
expro = joinpath(FILE_DIR, "input.EXPRO")

outdir = joinpath(TGLFEP_JULIA_ROOT, "dev", "validate_out_$(get(ENV, "SLURM_JOB_ID", "local"))_files")
mkpath(outdir)

t0 = time()
cd(outdir) do
    @time runTHD(tglfep, mtglf, expro; printout=true, use_gpu=use_gpu, parallel=:threads)
    make_crit_grad_plots("FILE"; dir=outdir)
end
println("validate OK in $(round(time() - t0; digits=1)) s -> $outdir")
