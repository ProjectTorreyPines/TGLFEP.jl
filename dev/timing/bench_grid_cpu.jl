# Single-radius CPU :grid benchmark with GC / allocation accounting.
#
# Measures what the production CPU layout does for ONE radius (one process, N threads,
# inner=:threads) and records enough output to prove a code change is bit-identical.
#
#   julia -t 64 --project=. dev/timing/bench_grid_cpu.jl
#
# Env:
#   CASE_DIR      example directory            (default: examples/DIIID_202017C42_500ms_v3.1)
#   TGLFEP_FILE   input.TGLFEP to time         (default: $CASE_DIR/input_scan20_nb32.TGLFEP)
#   WARM_FILE     cheap input for JIT warm-up  (default: $CASE_DIR/input_scan20_nb6.TGLFEP)
#   SCAN_INDEX    1-based radius               (default: 10)
#   BENCH_MODE    radius | combo               (default: radius)
#       radius: full kwscale_scan for the radius (k_max rounds x 256 combos), threaded
#       combo : BENCH_NCOMBO first-round combos run one after another on the calling
#               thread -> per-combo compute time with no thread or GC contention
#   BENCH_K_MAX   refinement rounds for radius mode (default: TGLFEP_JULIA_K_MAX or 4)
#   BENCH_NCOMBO  combos for combo mode        (default: 4)
#   BENCH_OUT     if set, serialize the results there (for == comparison between builds)
#   BENCH_TAG     free-text label echoed in the RESULT line

ENV["TGLFEP_JULIA_FILE_ONLY"] = "1"

using TJLFEP
using TJLF
using LinearAlgebra
using Serialization
using Printf

BLAS.set_num_threads(1)

const ROOT   = normpath(@__DIR__, "..", "..")
const CASE   = get(ENV, "CASE_DIR", joinpath(ROOT, "examples", "DIIID_202017C42_500ms_v3.1"))
const GACODE = get(ENV, "GACODE_FILE", joinpath(CASE, "input.gacode"))
const TGLFEP = get(ENV, "TGLFEP_FILE", joinpath(CASE, "input_scan20_nb32.TGLFEP"))
const WARM   = get(ENV, "WARM_FILE", joinpath(CASE, "input_scan20_nb6.TGLFEP"))
const SCAN_INDEX = parse(Int, get(ENV, "SCAN_INDEX", "10"))
const MODE   = get(ENV, "BENCH_MODE", "radius")
const K_MAX  = parse(Int, get(ENV, "BENCH_K_MAX", get(ENV, "TGLFEP_JULIA_K_MAX", "4")))
const NCOMBO = parse(Int, get(ENV, "BENCH_NCOMBO", "4"))
const TAG    = get(ENV, "BENCH_TAG", "")

# Same per-radius setup as TJLFEP.run_gacode_scan_task.
function radius_inputs(tglfep_file, scan_index)
    opts, prof, expro = preprocess_gacode_inputs(GACODE, tglfep_file)
    TJLFEP._apply_runthd_expro_setup!(opts, prof, expro)
    ep = deepcopy(opts)
    mt = deepcopy(prof)
    ep.IR = ep.IR_EXP[scan_index]
    ep.SUFFIX = "_r" * lpad(string(ep.IR), 3, '0')
    ep.FACTOR_IN = ep.FACTOR[scan_index]
    return ep, mt
end

function run_radius(tglfep_file, scan_index, k_max)
    ep, mt = radius_inputs(tglfep_file, scan_index)
    ret = TJLFEP.mainsub(ep, mt, true; use_gpu=false, inner=:threads, solver=:grid, k_max=k_max)
    growth, ep_out = ret[1][1], ret[1][2]
    return (growth=growth, sfmin=ep_out.FACTOR_IN, width=ep_out.WIDTH_IN, kymark=ep_out.KYMARK,
            scalefactor=ret[2][1])
end

# First-round combos exactly as kwscale_scan builds them (nkyhat=4, nefwid=8, nfactor=8).
function run_combos(tglfep_file, scan_index, ncombo)
    ep, mt = radius_inputs(tglfep_file, scan_index)
    ep.WIDTH_IN_FLAG = false; ep.MODE_IN = 2; ep.KY_MODEL = 3; ep.PROCESS_IN = 5
    nfactor, nefwid, nkyhat = 8, 8, 4
    f1 = ep.FACTOR_IN; w0 = ep.WIDTH_MIN; w1 = ep.WIDTH_MAX
    factor = [(f1 / nfactor) * i for i in 1:nfactor]
    efwid  = [((w1 - w0) / (nefwid - 1)) * (i - 1) + w0 for i in 1:nefwid]
    kyhat  = [(1.0 / nkyhat) * i for i in 1:nkyhat]
    nkwf = nfactor * nefwid * nkyhat
    # spread the sample over ky / width / factor
    idx = unique(round.(Int, range(1, nkwf; length=ncombo)))
    out = Any[]
    times = Float64[]
    for i in idx
        t = @elapsed r = TJLFEP._kw_combo(i, 1, 4, nfactor, nefwid, factor, efwid, kyhat, ep, mt,
                                          0, 0, 0, false; use_gpu=false, mode_in=2)
        push!(times, t)
        push!(out, (i=i, gamma=r.gamma_out, freq=r.freq_out, dep=r.dep_out, lkeep=r.LKEEP))
    end
    return (combos=out, times=times)
end

work(file, k_max) = MODE == "combo" ? run_combos(file, SCAN_INDEX, NCOMBO) : run_radius(file, SCAN_INDEX, k_max)

println("=== bench_grid_cpu mode=$MODE tag=$TAG ===")
println("threads=$(Threads.nthreads()) gcthreads=$(Threads.ngcthreads()) blas=$(BLAS.get_num_threads()) host=$(gethostname())")
println("input=$TGLFEP scan_index=$SCAN_INDEX k_max=$K_MAX")
flush(stdout)

t_warm = @elapsed (MODE == "combo" ? run_combos(WARM, SCAN_INDEX, 2) : run_radius(WARM, SCAN_INDEX, 1))
@printf("warm-up (JIT) %.1f s\n", t_warm); flush(stdout)

GC.gc()
g0 = Base.gc_num()
stats = @timed work(TGLFEP, K_MAX)
gd = Base.GC_Diff(Base.gc_num(), g0)
res = stats.value

wall = stats.time
@printf("RESULT tag=%s mode=%s threads=%d gcthreads=%d wall_s=%.2f gc_s=%.2f gc_pct=%.1f alloc_GiB=%.2f gc_pauses=%d full_sweeps=%d\n",
        TAG, MODE, Threads.nthreads(), Threads.ngcthreads(), wall, stats.gctime, 100 * stats.gctime / wall,
        stats.bytes / 2^30, gd.pause, gd.full_sweep)
@printf("RESULT maxrss_GiB=%.2f\n", Sys.maxrss() / 2^30)
if MODE == "combo"
    println("RESULT per_combo_s=", round.(res.times; digits=2))
else
    println("RESULT sfmin=", res.sfmin, " width=", res.width, " kymark=", res.kymark)
end

if haskey(ENV, "BENCH_OUT")
    mkpath(dirname(abspath(ENV["BENCH_OUT"])))
    serialize(ENV["BENCH_OUT"], res)
    println("wrote ", ENV["BENCH_OUT"])
end
