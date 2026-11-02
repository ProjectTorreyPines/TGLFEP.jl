# Coverage for tjlfep_ad_extensions.jl: pure helpers plus a single-radius end-to-end
# :ad solve. The solve itself is fast (~seconds); the cost is one-time AD/ForwardDiff
# compilation, amortized by the CI depot cache. Uses the cheapest AD path
# (AD_EXTEND_MODE=:only, bare w>=1 box) to keep the wallclock modest.

using Test
using TJLFEP

@testset "AD extensions" begin

    @testset "_nbasis_extrapolate (pure)" begin
        # No finite samples -> NaN limit, not converged.
        r0 = TJLFEP._nbasis_extrapolate(Int[], Float64[])
        @test isnan(r0.limit) && r0.converged == false

        # Fewer than 3 samples -> return the finest measured value.
        r1 = TJLFEP._nbasis_extrapolate([6, 8], [1.0, 1.1])
        @test r1.limit == 1.1
        @test r1.nb_finest == 8
        @test r1.converged == false

        # Flat tail -> converged at the finest value.
        r2 = TJLFEP._nbasis_extrapolate([6, 8, 16], [1.0, 1.0, 1.0])
        @test r2.converged == true
        @test r2.limit == 1.0

        # Geometric shrink (ratio 0.5) -> Richardson limit, clamped to the band.
        r3 = TJLFEP._nbasis_extrapolate([6, 8, 16], [1.0, 1.5, 1.75])
        @test r3.converged == true
        @test r3.ratio ≈ 0.5
        @test r3.limit ≈ 2.0
    end

    @testset "_clamp_to / _knob_label (pure)" begin
        @test TJLFEP._clamp_to(5.0, 0.0, 3.0) == 3.0
        @test TJLFEP._clamp_to(-1.0, 0.0, 3.0) == 0.0
        @test TJLFEP._clamp_to(1.5, 0.0, 3.0) == 1.5

        @test TJLFEP._knob_label(:RLNS, nothing, 3) == "RLNS"
        @test TJLFEP._knob_label(:RLNS, 1, 3) == "RLNS[e]"
        @test TJLFEP._knob_label(:RLNS, 3, 3) == "RLNS[EP]"
        @test TJLFEP._knob_label(:RLNS, 2, 3) == "RLNS[i1]"
    end

    @testset "AD leaves honor mode_in_override (MODE_IN=2 vs 4)" begin
        # The AD building blocks default to forcing MODE_IN=2 (EP drive only), but must honor an
        # explicit override so the PROCESS_IN=6 (MODE_IN=4 thermal+EP / ITG-TEM) drive can be
        # threaded through them. Verify the override actually reaches TJLF_map/TJLFEP_ky by
        # checking the growth-rate spectrum changes (thermal gradients on + FILTER=2).
        root = normpath(@__DIR__, "..")
        gac  = joinpath(root, "examples", "DIIID_202017C42_500ms_v3.1", "input.gacode")
        tgl  = joinpath(@__DIR__, "fixtures", "scan2", "input_scan2_nb2.TGLFEP")
        @test isfile(gac) && isfile(tgl)

        opts, prof, _ = preprocess_gacode_inputs(gac, tgl)
        ep = deepcopy(opts)
        ep.IR = ep.IR_EXP[1]
        ep.FACTOR_IN = ep.FACTOR[1]
        ep.KYHAT_IN = 0.25
        ep.WIDTH_IN = 1.5

        g2 = TJLFEP.gamma_dgamma_dfactor(ep, prof; mode_in_override=2)
        g4 = TJLFEP.gamma_dgamma_dfactor(ep, prof; mode_in_override=4)
        # Different drive/filter => different eigenvalue spectrum (not silently identical).
        @test g2.gamma != g4.gamma

        # Faithful keep path (TJLFEP_ky) likewise honors the override.
        k2 = TJLFEP.keep_at(ep, prof, ep.FACTOR_IN; mode_in_override=2)
        k4 = TJLFEP.keep_at(ep, prof, ep.FACTOR_IN; mode_in_override=4)
        @test k2.gamma != k4.gamma
    end

    @testset "single-radius :ad solve (end-to-end)" begin
        root = normpath(@__DIR__, "..")
        cd = joinpath(root, "examples", "DIIID_202017C42_500ms_v3.1")
        gac = joinpath(cd, "input.gacode")
        tgl = joinpath(cd, "input_scan20_nb6.TGLFEP")
        @test isfile(gac) && isfile(tgl)

        saved = get(ENV, "AD_EXTEND_MODE", nothing)
        ENV["AD_EXTEND_MODE"] = "only"      # cheapest AD path (bare w>=1 box)
        try
            out = mktempdir()
            r = run_gacode_scan_task(gac, tgl, 2; out_dir=out, use_gpu=false,
                                     printout=false, solver=:ad)
            @test r.ir == 7
            @test isfinite(r.sfmin) && r.sfmin > 0
            @test isfinite(r.width) && r.width > 0
            rm(out; recursive=true, force=true)
        finally
            saved === nothing ? delete!(ENV, "AD_EXTEND_MODE") : (ENV["AD_EXTEND_MODE"] = saved)
        end
    end

    # Remaining production solver tiers on CPU at the cheapest setting (nb=2,
    # single radius). The one-time ForwardDiff compile is amortized by the :only
    # solve above, so each tier here adds only a few seconds. This covers the
    # width-extension branches of _mainsub_ad (:wide, :locate) and the otherwise
    # 0%-covered _mainsub_robust_ad path.
    @testset "AD tiers: :ad :wide / :ad :locate / :robust_ad (nb=2)" begin
        root = normpath(@__DIR__, "..")
        gac  = joinpath(root, "examples", "DIIID_202017C42_500ms_v3.1", "input.gacode")
        tgl  = joinpath(@__DIR__, "fixtures", "scan2", "input_scan2_nb2.TGLFEP")
        @test isfile(gac) && isfile(tgl)

        saved = get(ENV, "AD_EXTEND_MODE", nothing)
        try
            for mode in ("wide", "locate")
                ENV["AD_EXTEND_MODE"] = mode
                out = mktempdir()
                r = run_gacode_scan_task(gac, tgl, 1; out_dir=out, use_gpu=false,
                                         printout=false, solver=:ad)
                @test r.ir == 2
                @test isfinite(r.sfmin) && r.sfmin > 0
                @test isfinite(r.width) && r.width > 0
                rm(out; recursive=true, force=true)
            end
        finally
            saved === nothing ? delete!(ENV, "AD_EXTEND_MODE") : (ENV["AD_EXTEND_MODE"] = saved)
        end

        # Production adaptive-refinement solver (its own mainsub branch).
        out = mktempdir()
        r = run_gacode_scan_task(gac, tgl, 1; out_dir=out, use_gpu=false,
                                 printout=false, solver=:robust_ad, refine_rounds=1)
        @test r.ir == 2
        @test isfinite(r.sfmin) && r.sfmin > 0
        @test isfinite(r.width) && r.width > 0
        rm(out; recursive=true, force=true)
    end

    @testset "marginal_factor_df agrees with marginal_factor (derivative-free inner)" begin
        root = normpath(@__DIR__, "..")
        gac  = joinpath(root, "examples", "DIIID_202017C42_500ms_v3.1", "input.gacode")
        tgl  = joinpath(@__DIR__, "fixtures", "scan2", "input_scan2_nb2.TGLFEP")
        @test isfile(gac) && isfile(tgl)

        opts, prof, _ = preprocess_gacode_inputs(gac, tgl)
        ep = deepcopy(opts)
        ep.IR = ep.IR_EXP[1]
        ep.FACTOR_IN = ep.FACTOR[1]
        gth = TJLFEP._gamma_thresh_for(ep, prof)
        shi = Float64(ep.FACTOR_IN); slo = shi / 512.0

        # The bracketing ITP root must match the safeguarded-Newton root wherever an onset exists,
        # and both must agree that a point with no in-range onset is not converged. Check a couple
        # of (ky,w) points spanning feasible / infeasible.
        for (ky, w) in ((0.9, 1.9), (0.5, 1.5))
            e = deepcopy(ep); e.KYHAT_IN = ky; e.WIDTH_IN = w
            mfn = marginal_factor(e, prof; gamma_thresh=gth, ae_band=true, scan_lo=slo, scan_hi=shi)
            mfd = marginal_factor_df(e, prof; gamma_thresh=gth, ae_band=true, scan_lo=slo, scan_hi=shi)
            @test mfd.evals > 0
            @test mfn.converged == mfd.converged
            if mfn.converged
                @test isapprox(mfn.factor, mfd.factor; rtol=2e-2)
            end
        end
    end

    # The (kyhat, width) critical_factor_optimize reports must be the point whose AE-band root is
    # sfmin: re-solving marginal_factor at the reported point reproduces the reported factor. The
    # seed solves run concurrently (threads / team chunk map); a closure that bound the enclosing
    # function's `ky, w` instead of its own locals attributed one seed's root to another seed's
    # (ky, w) (ITER FUSE case, 2026-11-01), which this invariant catches whenever nthreads() > 1.
    @testset "critical_factor_optimize: reported (kyhat,width) reproduces sfmin" begin
        root = normpath(@__DIR__, "..")
        gac  = joinpath(root, "examples", "DIIID_202017C42_500ms_v3.1", "input.gacode")
        tgl  = joinpath(@__DIR__, "fixtures", "scan2", "input_scan2_nb2.TGLFEP")
        opts, prof, _ = preprocess_gacode_inputs(gac, tgl)
        ep = deepcopy(opts)
        ep.IR = ep.IR_EXP[1]
        ep.FACTOR_IN = ep.FACTOR[1]
        gth = TJLFEP._gamma_thresh_for(ep, prof)
        shi = Float64(ep.FACTOR_IN); slo = shi / 512.0
        r = critical_factor_optimize(ep, prof; gamma_thresh=gth, scan_lo=slo, faithful_confirm=false,
                                     extend_width=false, inner=:threads)
        if isfinite(r.sfmin)
            e = deepcopy(ep); e.KYHAT_IN = r.kyhat; e.WIDTH_IN = r.width
            mf = marginal_factor(e, prof; gamma_thresh=gth, ae_band=true, scan_lo=slo, scan_hi=shi)
            @test mf.converged
            @test isapprox(mf.factor, r.sfmin; rtol=1e-3)
            # with the confirm on, a candidate whose faithful sweep keeps no mode is infeasible (Inf),
            # never the cheap AE-band root
            rc = critical_factor_optimize(ep, prof; gamma_thresh=gth, scan_lo=slo, faithful_confirm=true,
                                          extend_width=false, inner=:threads)
            if rc.faithful !== nothing && rc.faithful.binding == :none
                @test !isfinite(rc.sfmin)
            elseif rc.faithful !== nothing
                @test rc.sfmin == rc.faithful.factor_faithful
            end
        else
            @test isnan(r.kyhat)
        end
    end

    # End-to-end coverage of the derivative-free (ky,w) solver branches (_mainsub_multistart /
    # _mainsub_nlopt + _finalize_nls_result!).
    # Seed grid / iteration knobs are shrunk via ENV so the nb=2 single-radius solve stays fast;
    # correctness (finite onset) is the assertion, not accuracy.
    @testset "derivative-free solvers: :multistart / :nlopt (nb=2)" begin
        root = normpath(@__DIR__, "..")
        gac  = joinpath(root, "examples", "DIIID_202017C42_500ms_v3.1", "input.gacode")
        tgl  = joinpath(@__DIR__, "fixtures", "scan2", "input_scan2_nb2.TGLFEP")
        @test isfile(gac) && isfile(tgl)

        saved = Dict(k => get(ENV, k, nothing) for k in
                     ("NLS_NSEED_KY", "NLS_NSEED_W", "NLS_KDESCEND", "NLS_LOCAL_EVALS", "NLOPT_MAXEVAL"))
        ENV["NLS_NSEED_KY"] = "3"; ENV["NLS_NSEED_W"] = "4"
        ENV["NLS_KDESCEND"] = "1"; ENV["NLS_LOCAL_EVALS"] = "4"
        ENV["NLOPT_MAXEVAL"] = "16"
        # Honor the GPU eigensolve path when the batch job requests it (TGLFEP_JULIA_TEST_USE_GPU=1);
        # defaults to the CPU path so the CI depot-cache run stays host-only.
        test_gpu = get(ENV, "TGLFEP_JULIA_TEST_USE_GPU", "0") == "1"
        try
            for solver in (:nlopt, :multistart)
                out = mktempdir()
                r = run_gacode_scan_task(gac, tgl, 1; out_dir=out, use_gpu=test_gpu,
                                         printout=false, solver=solver)
                @test r.ir == 2
                @test isfinite(r.sfmin) && r.sfmin > 0
                @test isfinite(r.width) && r.width > 0
                rm(out; recursive=true, force=true)
            end
        finally
            for (k, v) in saved
                v === nothing ? delete!(ENV, k) : (ENV[k] = v)
            end
        end
    end
end
