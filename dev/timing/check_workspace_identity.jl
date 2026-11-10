# Bit-identity check for the reusable TJLF solver workspace in the threaded CPU :grid scan.
#
#   julia -t 16 --project=. dev/timing/check_workspace_identity.jl
#
# 1. One radius, K_MAX rounds, with the workspace pool off and on -> every output must be ==.
# 2. Single combos through a workspace whose buffers were poisoned with NaN beforehand
#    -> proves the solver overwrites every entry it later reads (no dependence on old content).
#
# Env: CASE_DIR, TGLFEP_FILE (default DIII-D nb6), SCAN_INDEX (default 10), CHECK_K_MAX (default 2)

using TJLFEP, TJLF, LinearAlgebra
BLAS.set_num_threads(1)

const ROOT   = normpath(@__DIR__, "..", "..")
const CASE   = get(ENV, "CASE_DIR", joinpath(ROOT, "examples", "DIIID_202017C42_500ms_v3.1"))
const GACODE = joinpath(CASE, "input.gacode")
const TGLFEP = get(ENV, "TGLFEP_FILE", joinpath(CASE, "input_scan20_nb6.TGLFEP"))
const SCAN_INDEX = parse(Int, get(ENV, "SCAN_INDEX", "10"))
const K_MAX  = parse(Int, get(ENV, "CHECK_K_MAX", "2"))

function radius_inputs()
    opts, prof, expro = preprocess_gacode_inputs(GACODE, TGLFEP)
    TJLFEP._apply_runthd_expro_setup!(opts, prof, expro)
    ep = deepcopy(opts); mt = deepcopy(prof)
    ep.IR = ep.IR_EXP[SCAN_INDEX]
    ep.SUFFIX = "_r" * lpad(string(ep.IR), 3, '0')
    ep.FACTOR_IN = ep.FACTOR[SCAN_INDEX]
    return ep, mt
end

function run_radius(use_ws::Bool)
    ENV["TGLFEP_JULIA_WORKSPACE"] = use_ws ? "1" : "0"
    ep, mt = radius_inputs()
    ret = TJLFEP.mainsub(ep, mt, true; use_gpu=false, inner=:threads, solver=:grid, k_max=K_MAX)
    growth, ep_out = ret[1][1], ret[1][2]
    return (growth=growth, sfmin=ep_out.FACTOR_IN, width=ep_out.WIDTH_IN, kymark=ep_out.KYMARK,
            scalefactor=ret[2][1])
end

println("threads=", Threads.nthreads(), " input=", TGLFEP, " scan_index=", SCAN_INDEX, " k_max=", K_MAX)
a = run_radius(false)
b = run_radius(true)
ok1 = isequal(a.growth, b.growth) && isequal(a.sfmin, b.sfmin) && isequal(a.width, b.width) &&
      isequal(a.kymark, b.kymark) && isequal(a.scalefactor, b.scalefactor)
println("radius: sfmin=", a.sfmin, " width=", a.width, " kymark=", a.kymark,
        " nonzero growth entries=", count(!iszero, a.growth), " scalefactor lines=", length(a.scalefactor))
println("CHECK radius workspace on == off : ", ok1)

# NaN-poison test on individual combos
ep, mt = radius_inputs()
ep.WIDTH_IN_FLAG = false; ep.MODE_IN = 2; ep.KY_MODEL = 3; ep.PROCESS_IN = 5
nfactor, nefwid, nkyhat = 8, 8, 4
factor = [(ep.FACTOR_IN / nfactor) * i for i in 1:nfactor]
efwid  = [((ep.WIDTH_MAX - ep.WIDTH_MIN) / (nefwid - 1)) * (i - 1) + ep.WIDTH_MIN for i in 1:nefwid]
kyhat  = [(1.0 / nkyhat) * i for i in 1:nkyhat]
combo(i, ws) = TJLFEP._kw_combo(i, 1, 4, nfactor, nefwid, factor, efwid, kyhat, ep, mt, 0, 0, 0, false;
                                use_gpu=false, mode_in=2, workspace=ws)
ws = TJLF.LSWorkspace{Float64}()
ok2 = true
nunstable = 0
for i in (1, 8, 37, 64, 100, 129, 200, 256)
    r0 = combo(i, nothing)
    for m in (ws.amat, ws.bmat, ws.amat_copy, ws.bmat_copy, ws.zmat)
        fill!(m, complex(NaN, NaN))
    end
    fill!(ws.eigenvector, complex(NaN, NaN))
    r1 = combo(i, ws)
    same = isequal(r0.gamma_out, r1.gamma_out) && isequal(r0.freq_out, r1.freq_out) &&
           isequal(r0.dep_out, r1.dep_out) && isequal(r0.ep_flux_out, r1.ep_flux_out) &&
           r0.LKEEP == r1.LKEEP && r0.LTEARING == r1.LTEARING && r0.L_QL_RATIO == r1.L_QL_RATIO
    global ok2 &= same
    global nunstable += count(>(0), r0.gamma_out)
    println("  combo ", i, ": gamma=", r0.gamma_out, " same=", same)
end
println("CHECK NaN-poisoned workspace == fresh : ", ok2, "  (unstable modes seen: ", nunstable, ")")
println(ok1 && ok2 ? "ALL IDENTICAL" : "MISMATCH")
exit(ok1 && ok2 ? 0 : 1)
