# One radius of SCAN_N=20, run with an MPS *team*: the radius's inner kw-scan (256
# combos x 4 k-rounds) is distributed across MPS_TEAM local worker processes that all
# share this task's single GPU. Each worker has its own CUDA context (MPS), so their
# Xgeev eigensolves overlap on the device via Hyper-Q — the within-GPU latency lever
# that intra-process concurrency could not unlock (it corrupted Xgeev).
#
# Layout: 1 SLURM task = 1 radius = 1 GPU (pinned via CUDA_VISIBLE_DEVICES set by the
# batch script) + MPS_TEAM coordinator-local workers on that GPU. The MPS control
# daemon must already be running on the node (see mps-wrapper.sh / the batch script).
#
# Env:
#   MPS_TEAM     worker processes per GPU (default 4; the slurm/ templates set 8)
#   SCAN_INDEX   1-based radius to run (default: SLURM array task id + 1)
#   USE_GPU      must be "1"
#   TGLFEP_FILE / GACODE_FILE / CASE_DIR / OUT_DIR  case inputs + per-radius output dir
#   BACKFILL_MODE=1  drain a shared claim queue over all radii instead of one radius/task
#
# Run it with an explicit `--project` (the slurm/ templates do): the MPS workers inherit the
# master's active project, so a prebuilt CFS sysimage is paired with its published env_lean
# project and a JIT run with the TJLFEP checkout. No Pkg.activate here on purpose.

@assert get(ENV, "USE_GPU", "") == "1" "run_gacode_scan20_mps_task.jl is GPU-only; set USE_GPU=1"

using CUDA
let rv = CUDA.runtime_version()
    rv >= v"12.6" || error("GPU run needs CUDA >= 12.6 for cusolverDnXgeev, but CUDA runtime is $rv. " *
                           "Load cudatoolkit/12.9.")
end

using Distributed

const ROOT = normpath(@__DIR__, "..", "..")
# Active project (set by `julia --project=...`); forwarded to the MPS workers.
const PROJECT = dirname(Base.active_project())
const NTEAM = parse(Int, get(ENV, "MPS_TEAM", "4"))
const THREADS_PER_WORKER = parse(Int, get(ENV, "JULIA_WORKER_THREADS", "2"))
# INNER selects the within-radius parallelism: :mps_team (default; addproc'd MPS clients) or
# :threads (single-process threaded baseline on this task's GPU, for speedup/timing comparison).
const INNER = Symbol(get(ENV, "INNER", "mps_team"))
# SOLVER selects the critical-factor engine: :grid (Fortran-equivalent kwscale_scan),
# :ad (fast autodiff AE-onset Newton + IFT descent), or :robust_ad (robust autodiff
# global-min over the (ky,w) grid). Default :grid preserves prior behavior.
const SOLVER = Symbol(get(ENV, "SOLVER", "grid"))
# REFINE_ROUNDS is the accuracy/speed knob for SOLVER=robust_ad: rounds of (ky,w)
# window narrowing around the running best (0=coarse grid min; higher=better off-node
# resolution at proportionally higher cost). Ignored by :grid and :ad.
const REFINE_ROUNDS = parse(Int, get(ENV, "REFINE_ROUNDS", "1"))
# Optional GPU pin list for the team workers, e.g. "0,1" to spread a radius across 2 GPUs.
# Defaults to this task's CUDA_VISIBLE_DEVICES (one GPU). Workers are round-robined over it.
const TEAM_GPUS = let s = get(ENV, "TEAM_GPUS", get(ENV, "CUDA_VISIBLE_DEVICES", "0"))
    String.(split(s, ',', keepempty=false))
end
# Optional GPU-worker sysimage (TGLFEP_JULIA_gpu_generic_sysimage.so): bakes the TJLF/TJLFEP/CUDA GPU
# eigensolve path so workers skip the ~110 s/team JIT on a cold spawn. Empty -> JIT as before.
const GPU_SYSIMAGE = get(ENV, "TGLFEP_JULIA_GPU_SYSIMAGE", "")
const _SYSIMG_FLAGS = (!isempty(GPU_SYSIMAGE) && isfile(GPU_SYSIMAGE)) ? `--sysimage=$(GPU_SYSIMAGE)` : ``

if INNER === :mps_team
    # Spawn the team. Workers inherit the MPS pipe dir and are pinned (round-robin) across
    # TEAM_GPUS, so each worker connects as an MPS client on its assigned physical GPU.
    base_env = Dict{String,String}()
    for k in ("JULIA_DEPOT_PATH", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
              # JULIA_CUDA_USE_COMPAT=false is REQUIRED: CUDA.jl's forward-compat driver shim
              # hangs in cuInit when initializing as an MPS client on Perlmutter. Without it,
              # every team worker's `using CUDA` blocks indefinitely.
              "JULIA_CUDA_USE_COMPAT", "JULIA_CUDA_MEMORY_POOL")
        haskey(ENV, k) && (base_env[k] = ENV[k])
    end
    base_env["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

    for w in 1:NTEAM
        env = copy(base_env)
        env["CUDA_VISIBLE_DEVICES"] = TEAM_GPUS[(w - 1) % length(TEAM_GPUS) + 1]
        addprocs(1; exeflags=`--project=$(PROJECT) -t $(THREADS_PER_WORKER) --startup-file=no $(_SYSIMG_FLAGS)`,
                 env=env)
    end

    @everywhere begin
        using CUDA
        using TJLFEP
        using TJLF
        using LinearAlgebra
        BLAS.set_num_threads(1)
        # touch the GPU so each worker's CUDA context (MPS client) is created up front
        CUDA.functional() && CUDA.device!(first(CUDA.devices()))
    end
end

using TJLFEP
using TJLF
using LinearAlgebra
BLAS.set_num_threads(1)

# NB: `GACODE_PATH`, not `GACODE` — the generic GPU sysimage bakes the `GACODE` *package*
# module into Main (it is listed in create_sysimage), so a top-level `const GACODE = ...`
# here collides with it ("invalid redefinition of constant Main.GACODE").
const CASE   = get(ENV, "CASE_DIR", joinpath(ROOT, "examples", "DIIID_202017C42_500ms_v3.1"))
const GACODE_PATH = get(ENV, "GACODE_FILE", joinpath(CASE, "input.gacode"))
const TGLFEP = get(ENV, "TGLFEP_FILE", joinpath(CASE, "input_scan20_nb32.TGLFEP"))
const OUT_DIR = get(ENV, "OUT_DIR", joinpath(ROOT, "runs", "tglfep_julia_scan_$(get(ENV, "SLURM_JOB_ID", "local"))_tasks"))

@assert isfile(GACODE_PATH) "missing $GACODE_PATH"
@assert isfile(TGLFEP) "missing $TGLFEP"

opts, _, _ = preprocess_gacode_inputs(GACODE_PATH, TGLFEP)
scan_n = opts.SCAN_N

printout = get(ENV, "TGLFEP_JULIA_PRINTOUT", "0") == "1"

const TEAM = INNER === :mps_team ? workers() : nothing
println("=== gacode scan task (inner=$INNER solver=$SOLVER refine_rounds=$REFINE_ROUNDS) ===")
println("worker sysimage=", isempty(_SYSIMG_FLAGS.exec) ? "<none, JIT>" : GPU_SYSIMAGE)
println("team=$(INNER === :mps_team ? nworkers() : 0) workers  " *
        "threads/worker=$THREADS_PER_WORKER  team_gpus=$(join(TEAM_GPUS, ','))  scan_n=$scan_n")
println("OUT_DIR=$OUT_DIR")
println("CUDA_VISIBLE_DEVICES=$(get(ENV, "CUDA_VISIBLE_DEVICES", "<unset>")) MPS=$(get(ENV, "CUDA_MPS_PIPE_DIRECTORY", "<none>"))")
println("GPU=", CUDA.functional() ? CUDA.name(first(CUDA.devices())) : "n/a")
flush(stdout)

# Run ONE radius reusing the persistent MPS team, write its result, optional per-worker probe dump.
# Hoisted into a function so BACKFILL_MODE can call it repeatedly (team spawn/JIT paid once per GPU).
function run_one(scan_index::Int)
    @assert 1 <= scan_index <= scan_n "scan_index=$scan_index invalid for SCAN_N=$scan_n"
    t0 = time()
    result = run_gacode_scan_task(
        GACODE_PATH, TGLFEP, scan_index;
        out_dir=OUT_DIR,
        use_gpu=true,
        printout=printout,
        inner=INNER,
        team=TEAM,
        solver=SOLVER,
        refine_rounds=REFINE_ROUNDS,
    )
    println("OK scan_index=$(result.scan_index) ir=$(result.ir) sfmin=$(result.sfmin) in $(round(time() - t0; digits=1)) s")
    flush(stdout)
    # Per-worker probe dump: each MPS team worker reports its own combo count and the time it
    # spent in TJLFEP_ky vs TJLF.run (eigensolve), so we can see whether the workers actually
    # overlapped on the GPU (high combo count per worker at low wall) or time-sliced/serialized.
    if INNER === :mps_team && get(ENV, "TGLFEP_JULIA_PROBE", "0") == "1"
        for w in workers()
            n, tky, trun = remotecall_fetch(w) do
                (TJLFEP._PROBE_N[], TJLFEP._PROBE_KY[], TJLFEP._PROBE_RUN[])
            end
            println("  [worker $w] combos=$n  sum(TJLFEP_ky)=$(round(tky; digits=1))s  sum(TJLF.run)=$(round(trun; digits=1))s")
        end
        flush(stdout)
    end
    return result
end

# BACKFILL_MODE (NN-DB / single-node backfill): the node's GPU-worker tasks share a directory-based
# ATOMIC claim queue over ALL radii 1..scan_n, so each freed GPU pulls the next radius and the
# team is REUSED across radii (spawn/JIT cost paid once per GPU). `mkdir` is atomic on POSIX/Lustre,
# so the first task to create `<OUT_DIR>/.claims/r$i` owns radius i — no lock files, works across
# tasks on the node (and across nodes if ever used multi-node). A radius whose run throws is logged
# and SKIPPED (its claim stays, so it is not retried) and the task keeps draining the queue, so one
# bad radius cannot abort the whole DB-gen job. Default (BACKFILL_MODE!=1) preserves the legacy
# one-task=one-radius behavior (SCAN_INDEX from the wrapper) — existing batch scripts are unchanged.
# Wrapped in a function so the nrun/nfail counters live in hard (function) scope — at top-level
# script scope a `for`-loop body is soft scope and `nrun += 1` would raise UndefVarError.
function drain_queue(qdir::String)
    mkpath(qdir)
    nrun = 0; nfail = 0
    for i in 1:scan_n
        claimed = try
            mkdir(joinpath(qdir, "r$i")); true
        catch
            false   # already claimed by a sibling GPU-worker task
        end
        claimed || continue
        try
            run_one(i); nrun += 1
        catch e
            nfail += 1
            println("ERROR scan_index=$i failed: $(sprint(showerror, e))")
            flush(stdout)
        end
    end
    return nrun, nfail
end

if get(ENV, "BACKFILL_MODE", "0") == "1"
    nrun, nfail = drain_queue(joinpath(OUT_DIR, ".claims"))
    println("BACKFILL done: this task ran $nrun radius(es) of $scan_n ($nfail failed)")
    flush(stdout)
else
    scan_index = haskey(ENV, "SCAN_INDEX") ? parse(Int, ENV["SCAN_INDEX"]) : slurm_array_task_id() + 1
    run_one(scan_index)
end
