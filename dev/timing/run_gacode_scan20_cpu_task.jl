# CPU analog of run_gacode_scan20_mps_task.jl: one SLURM task = one radius, run threaded
# (inner=:threads) on the host CPU. Consumes input.gacode + input.TGLFEP directly and writes
# task_<i>.jls into OUT_DIR -- the SAME per-radius artifact that merge_gacode_scan20_array.jl
# reads -- so the CPU scan is byte-for-byte structurally identical to the GPU SPMD layout
# (srun -n SCAN_N, SCAN_INDEX = global procid + 1), just with no CUDA and no MPS team.
#
# Env:
#   SCAN_INDEX   1-based radius to run (default: SLURM_PROCID + 1)
#   SOLVER       :grid (default) | :ad | :robust_ad  -- must match the GPU run for a fair compare
#   TGLFEP_FILE / GACODE_FILE / CASE_DIR / OUT_DIR   as in the GPU task

ENV["TGLFEP_JULIA_FILE_ONLY"] = "1"

using Pkg
Pkg.activate(normpath(@__DIR__, "..", ".."))

using TJLFEP
using TJLF
using LinearAlgebra
# Each Julia thread runs its own combo (dense eigensolve), so pin BLAS to 1 thread to avoid
# oversubscribing cores -- the inner scan parallelism comes from Threads, not BLAS.
BLAS.set_num_threads(1)

const ROOT = normpath(@__DIR__, "..", "..")
# SOLVER selects the critical-factor engine: :grid (Fortran-equivalent kwscale_scan), :ad, or
# :robust_ad. Default :grid mirrors the production GPU scan.
const SOLVER = Symbol(get(ENV, "SOLVER", "grid"))
# REFINE_ROUNDS: accuracy/speed knob for SOLVER=:robust_ad (ignored by :grid/:ad).
const REFINE_ROUNDS = parse(Int, get(ENV, "REFINE_ROUNDS", "1"))

const CASE   = get(ENV, "CASE_DIR", joinpath(ROOT, "examples", "DIIID_202017C42_500ms_v3.1"))
const GACODE_PATH = get(ENV, "GACODE_FILE", joinpath(CASE, "input.gacode"))
const TGLFEP = get(ENV, "TGLFEP_FILE", joinpath(CASE, "input.TGLFEP"))
const OUT_DIR = get(ENV, "OUT_DIR", joinpath(ROOT, "dev", "gacode_scan20_cpu_$(get(ENV, "SLURM_JOB_ID", "local"))_tasks"))

@assert isfile(GACODE_PATH) "missing $GACODE_PATH"
@assert isfile(TGLFEP) "missing $TGLFEP"

opts, _, _ = preprocess_gacode_inputs(GACODE_PATH, TGLFEP)
scan_n = opts.SCAN_N

printout = get(ENV, "TGLFEP_JULIA_PRINTOUT", "0") == "1"

println("=== gacode scan CPU task (inner=threads solver=$SOLVER refine_rounds=$REFINE_ROUNDS) ===")
println("threads/task=$(Threads.nthreads())  scan_n=$scan_n  host=$(gethostname())")
println("OUT_DIR=$OUT_DIR")
flush(stdout)

function run_one(scan_index::Int)
    @assert 1 <= scan_index <= scan_n "scan_index=$scan_index invalid for SCAN_N=$scan_n"
    t0 = time()
    result = run_gacode_scan_task(
        GACODE_PATH, TGLFEP, scan_index;
        out_dir=OUT_DIR,
        use_gpu=false,
        printout=printout,
        inner=:threads,
        team=nothing,
        solver=SOLVER,
        refine_rounds=REFINE_ROUNDS,
    )
    println("OK scan_index=$(result.scan_index) ir=$(result.ir) sfmin=$(result.sfmin) in $(round(time() - t0; digits=1)) s")
    flush(stdout)
    return result
end

# One radius per task. SPMD: SCAN_INDEX from the wrapper env, else global procid + 1 (matches
# the GPU mps-scan-wrapper convention so the two runs cover the identical radius set).
scan_index = haskey(ENV, "SCAN_INDEX") ? parse(Int, ENV["SCAN_INDEX"]) :
             parse(Int, get(ENV, "SLURM_PROCID", "0")) + 1
run_one(scan_index)
