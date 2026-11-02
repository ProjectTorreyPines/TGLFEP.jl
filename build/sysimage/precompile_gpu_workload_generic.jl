# Precompile workload for the GENERIC GPU sysimage (TJLFEP full / not file-only). Same GPU
# compute trace as precompile_gpu_workload.jl (one nb6 radius on the GPU -> tjlf_LS eigenvalue
# + eigenvector CUSOLVER paths), but does NOT set TGLFEP_JULIA_FILE_ONLY, so it runs in the full
# module context (IMAS/FUSE/TurbulentTransport loaded). The expensive JIT is the GPU per-combo
# path, which this traces; IMAS/FUSE glue is loaded (baked) but compiles on first actual use.

# Built against the registry-resolved "full" build env (see build_gpu_sysimage_generic.jl),
# where CUDA/TJLF/TJLFEP/IMAS/GACODE/TurbulentTransport are all direct deps. Do NOT
# Pkg.activate a different project here -- that would fight create_sysimage's build project.
using CUDA
using TJLF
using TJLFEP
using LinearAlgebra
BLAS.set_num_threads(1)

if !CUDA.functional()
    error("precompile_gpu_workload_generic.jl must run on a GPU node (CUDA.functional() == false)")
end
@info "generic precompile workload GPU" name=CUDA.name(first(CUDA.devices()))

# NB: keep these as LOCALS inside a `let` (not top-level `const`). PackageCompiler runs this
# file such that top-level Main bindings get baked into the image's Main; a baked `const GACODE`
# then collides with the MPS task script's own `const GACODE = ENV[...]` ("invalid redefinition
# of constant Main.GACODE" at run_gacode_scan20_mps_task.jl). The `let` keeps the compile trace
# identical while leaking nothing into Main.
let
    # Example inputs come from the installed TJLFEP package (registry copy), not a repo
    # checkout -- the traced code and its inputs stay from the same source.
    EX     = joinpath(pkgdir(TJLFEP), "examples", "DIIID_202017C42_500ms_v3.1")
    GACODE = joinpath(EX, "input.gacode")
    TGLFEP = joinpath(EX, "input_singleradius_nb6.TGLFEP")   # N_BASIS=6, SCAN_N=1

    @assert isfile(GACODE) "missing $GACODE"
    @assert isfile(TGLFEP) "missing $TGLFEP"

    mktempdir() do tmp
        res = run_gacode_scan_task(GACODE, TGLFEP, 1;
            out_dir=tmp, use_gpu=true, printout=false, inner=:threads, team=nothing)
        @info "generic precompile workload done" scan_index=res.scan_index ir=res.ir sfmin=res.sfmin
    end

    # Also bake the autodiff (solver=:ad) path on GPU: critical_factor_optimize with use_gpu=true
    # (Float64 keep_at evals on the GPU, Dual eigensolves on the GPU), reached via mainsub's
    # solver=:ad branch. Bakes runTHD/run_gacode_scan_task(solver=:ad) so AD runs don't JIT per task.
    mktempdir() do tmp
        res = run_gacode_scan_task(GACODE, TGLFEP, 1;
            out_dir=tmp, use_gpu=true, printout=false, inner=:threads, team=nothing, solver=:ad)
        @info "generic precompile AD workload done" scan_index=res.scan_index ir=res.ir sfmin=res.sfmin
    end

    # Bake the robust autodiff (solver=:robust_ad) path on GPU: critical_factor_robust — the
    # (ky,w) grid of faithful onsets + adaptive (ky,w) refinement — reached via mainsub's
    # solver=:robust_ad branch. refine_rounds>0 exercises the refinement loop so the zoom/trigger
    # code is baked too. Without this, robust_ad runs JIT ~15 min per task.
    mktempdir() do tmp
        res = run_gacode_scan_task(GACODE, TGLFEP, 1;
            out_dir=tmp, use_gpu=true, printout=false, inner=:threads, team=nothing,
            solver=:robust_ad, refine_rounds=1)
        @info "generic precompile robust_ad workload done" scan_index=res.scan_index ir=res.ir sfmin=res.sfmin
    end

    # Bake the physical-truth (solver=:truth) path on GPU: critical_factor_truth — the extended
    # log-width (ky,w) locate (cheap AE-edge rank -> :ad polish -> faithful confirm) + separable
    # nbasis convergence — reached via mainsub's solver=:truth branch. Without this the truth path
    # JITs the locate + nbasis sweep per task.
    mktempdir() do tmp
        res = run_gacode_scan_task(GACODE, TGLFEP, 1;
            out_dir=tmp, use_gpu=true, printout=false, inner=:threads, team=nothing, solver=:truth)
        @info "generic precompile truth workload done" scan_index=res.scan_index ir=res.ir sfmin=res.sfmin
    end
end
