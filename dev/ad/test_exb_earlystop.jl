# ExB-shear (rotational-suppression) early-stop test.
#
# ExB shear in TJLFEP/TGLF-EP raises only GAMMA_THRESH (γ* = 0.15·|γ_E/ŝ|, Bass 2017);
# VEXB_SHEAR stays 0. A higher γ* can make the canonical w≥1 box infeasible, which used
# to defeat the width-extension early-stop (every narrow candidate got a full IFLUX=true
# faithful confirm). This script drives the DIII-D example through `critical_factor_robust`
# (the :robust_ad production path) over a sweep of forced γ* — exactly what
# ROTATIONAL_SUPPRESSION_FLAG=1 does — and reports cost (n_ext_confirm / evals / wall time)
# and correctness (sfmin / status) so the fix can be validated and compared pre/post.
#
# Usage (GPU, file-only):
#   module load cudatoolkit/12.9 julia/1.11.7 && export JULIA_DEPOT_PATH=$PSCRATCH/.julia
#   TGLFEP_JULIA_FILE_ONLY=1 USE_GPU=1 julia --project=. --threads=16 dev/ad/test_exb_earlystop.jl
#
# Env knobs: USE_GPU (default 0; 1 batches eigensolves on one A100), NB (basis, default 6),
#            SCAN_IS (comma list of scan indices, default "15,18"),
#            GTHS (comma list of γ*, default "1e-7,0.05,0.1,0.2,0.4").

using Printf, Dates
import TJLFEP
const TE = TJLFEP

const USE_GPU = get(ENV, "USE_GPU", "0") == "1"
if USE_GPU
    using CUDA
    @assert CUDA.functional() "USE_GPU=1 but no functional GPU"
    CUDA.device!(first(CUDA.devices()))
end

ex = joinpath(@__DIR__, "..", "..", "examples", "DIIID_202017C42_500ms_v3.1")
gacode = joinpath(ex, "input.gacode")
tglfep = joinpath(ex, "input.TGLFEP")

nb       = parse(Int, get(ENV, "NB", "6"))
scan_is  = parse.(Int, split(get(ENV, "SCAN_IS", "15,18"), ","))
gths     = parse.(Float64, split(get(ENV, "GTHS", "1e-7,0.05,0.1,0.2,0.4"), ","))

opts, prof, expro = TE.preprocess_gacode_inputs(gacode, tglfep)
TE._apply_runthd_expro_setup!(opts, prof, expro)

println("loaded DIII-D example: NR=", prof.NR, "  SCAN_N=", opts.SCAN_N, "  N_BASIS(test)=", nb,
        "  device=", USE_GPU ? "GPU" : "CPU"); flush(stdout)
println("IR_EXP = ", opts.IR_EXP); flush(stdout)

ts() = Dates.format(now(), "HH:MM:SS")

# Warmup: one cheap solve so the (very heavy) GPU/JIT kernel compilation is paid
# once, up front, and excluded from the per-γ* timings below.
let ep0 = deepcopy(opts)
    ep0.IR = ep0.IR_EXP[first(scan_is)]; ep0.FACTOR_IN = ep0.FACTOR[first(scan_is)]; ep0.N_BASIS = nb
    @printf("[%s] warmup (compile)...\n", ts()); flush(stdout)
    tw = @elapsed TE.critical_factor_robust(ep0, prof; gamma_thresh = 1e-7, use_gpu = USE_GPU,
                                            inner = :threads, refine_rounds = 1)
    @printf("[%s] warmup done in %.1fs\n", ts(), tw); flush(stdout)
end

for i in scan_is
    ep = deepcopy(opts)
    ep.IR = ep.IR_EXP[i]
    ep.FACTOR_IN = ep.FACTOR[i]
    ep.N_BASIS = nb

    # Natural ExB γ* this radius would get with ROTATIONAL_SUPPRESSION_FLAG=1.
    epc = deepcopy(ep); epc.ROTATIONAL_SUPPRESSION_FLAG = 1
    nat = TE._gamma_thresh_for(epc, prof)
    @printf("\n=== scan i=%d  IR=%d  FACTOR_IN=%.4g  (natural ExB γ*=%.3e) ===\n",
            i, ep.IR, ep.FACTOR_IN, nat)
    @printf("  %-9s %-10s %-10s %-13s %-10s %-11s %-6s %s\n",
            "gth", "sfmin", "status", "n_ext_confirm", "evals_full", "evals_eig", "npts", "wall"); flush(stdout)
    for gth in gths
        epx = deepcopy(ep)
        @printf("  [%s] start gth=%.2e ...\n", ts(), gth); flush(stdout)
        t0 = time()
        res = TE.critical_factor_robust(epx, prof; gamma_thresh = gth, use_gpu = USE_GPU,
                                        inner = :threads, refine_rounds = 1)
        dt = time() - t0
        @printf("  %-9.2e %-10.5g %-10s %-13d %-10d %-11d %-6d %.2fs\n",
                gth, res.sfmin, String(res.status), res.n_ext_confirm,
                res.total_evals_full, res.total_evals_eig, res.npts, dt)
        # Per-phase eigensolve breakdown (which stage burns the high-γ* eig cost).
        @printf("      eig: coarse=%d zoom=%d loc_cheap=%d loc_desc=%d ext_confirm=%d  (sum=%d)\n",
                res.eig_coarse, res.eig_zoom, res.eig_loc_cheap, res.eig_loc_desc, res.eig_ext_confirm,
                res.eig_coarse + res.eig_zoom + res.eig_loc_cheap + res.eig_loc_desc + res.eig_ext_confirm)
        flush(stdout)
    end
end
println("\ndone."); flush(stdout)
