#!/usr/bin/env julia
# Compare Julia IMAS InputTGLFEP/extraEP vs GACODE EXPRO (expro_util.f90 logic) on a radial scan.
#
# Run (from the repo root):
#   TGLFEP_JULIA_FILE_ONLY=0 julia --startup-file=no --project=. utils/compare_imas_vs_expro_scan.jl
#
# Optional env:
#   CASE_DIR, IS_EP (IMAS index, default 1), RHO_SCAN (comma-separated)

ENV["TGLFEP_JULIA_FILE_ONLY"] = "0"

using Pkg
const TGLFEP_JULIA_ROOT = normpath(@__DIR__, "..")
Pkg.activate(TGLFEP_JULIA_ROOT)

using TJLFEP
using TJLFEP: InputTGLFEP, expro_bound_deriv, expro_log_gradients, read_gacode_scalar_field, read_gacode_ion_field
using IMAS
using GACODE
using Printf
using Statistics

const CASE_DIR = get(ENV, "CASE_DIR",
    joinpath(TGLFEP_JULIA_ROOT, "examples", "DIIID_202017C42_500ms_v3.1"))
const GACODE_FILE = joinpath(CASE_DIR, "dump.gacode")
const IS_EP_IMAS = parse(Int, get(ENV, "IS_EP", "1"))
const IS_EP_GACODE = parse(Int, get(ENV, "IS_EP_GACODE", string(IS_EP_IMAS + 1)))  # Fortran input.TGLFEP IS_EP=2

function parse_rho_scan()
    if haskey(ENV, "RHO_SCAN")
        return parse.(Float64, split(ENV["RHO_SCAN"], ','))
    end
    return [0.01, 0.06, 0.11, 0.16, 0.21, 0.27, 0.32, 0.37, 0.42, 0.47,
            0.53, 0.58, 0.63, 0.68, 0.73, 0.79, 0.84, 0.89, 0.94, 1.0]
end

"""Nearest index on monotonic grid."""
function nearest_index(grid::AbstractVector{<:Real}, x::Real)
    _, j = findmin(abs.(grid .- x))
    return j
end

"""Relative error; NaN if both ~0."""
function rel_err(a::Real, b::Real)
    denom = max(abs(a), abs(b), 1e-30)
    return abs(b - a) / denom
end

function read_gacode_header_int(lines::Vector{String}, tag::AbstractString)
    i = findfirst(l -> strip(l) == "# $tag", lines)
    i === nothing && error("header # $tag not found")
    return parse(Int, strip(lines[i + 1]))
end

"""Read full EXPRO-style profiles from dump.gacode and apply TGLFEP_read_EXPRO post-processing."""
function fortran_expro_profiles(gacode_file::AbstractString; is_ep::Int=1, tglfep_nion::Int=2)
    lines = readlines(gacode_file)
    nr = read_gacode_header_int(lines, "nexp")
    nion = read_gacode_header_int(lines, "nion")

    rho = read_gacode_scalar_field(gacode_file, "rho", nr)
    rmin = read_gacode_scalar_field(gacode_file, "rmin", nr)
    rmaj = read_gacode_scalar_field(gacode_file, "rmaj", nr)
    q = read_gacode_scalar_field(gacode_file, "q", nr)
    w0 = read_gacode_scalar_field(gacode_file, "w0", nr)
    ne = read_gacode_scalar_field(gacode_file, "ne", nr)
    te = read_gacode_scalar_field(gacode_file, "te", nr)

    ni = [read_gacode_ion_field(gacode_file, "ni", i, nr) for i in 1:nion]
    ti = [read_gacode_ion_field(gacode_file, "ti", i, nr) for i in 1:nion]

    # TGLFEP_read_EXPRO: axis offset
    rmin_work = copy(rmin)
    rmin_work .+= 1.0e-6

    a = rmin_work[end]
    eps_n = 1.0e-30

    dlnnedr = expro_bound_deriv(-log.(max.(ne, eps_n)), rmin_work)
    dlntedr = expro_bound_deriv(-log.(max.(te, eps_n)), rmin_work)
    dlnnidr = [expro_bound_deriv(-log.(max.(ni[i], eps_n)), rmin_work) for i in 1:nion]
    dlntidr = [expro_bound_deriv(-log.(max.(ti[i], eps_n)), rmin_work) for i in 1:nion]

    # TGLFEP quasineutrality among thermal ions (is_EP = GACODE ion index for fast species)
    1 <= is_ep <= nion || error("is_ep=$is_ep out of range 1..$nion")
    sum0 = zeros(nr)
    for i in 1:tglfep_nion
        i == is_ep && continue
        z_i = 1.0  # D D case
        sum0 .+= z_i .* ni[i] ./ ne
    end
    a_qn = (1.0 .- ni[is_ep] ./ ne) ./ sum0  # z_EP=1
    ni_qn = copy(ni)
    for i in 1:tglfep_nion
        i == is_ep && continue
        ni_qn[i] = a_qn .* ni[i]
    end

    # EP density gradient floor (TGLFEP_read_EXPRO)
    dlnnidr_ep = copy(dlnnidr[is_ep])
    dlnnidr_ep .= max.(dlnnidr_ep, 1.0)

    # Sound speed (expro_util.f90 CGS formula, Te in keV)
    k = 1.6022e-12
    mp = 2.0 * 1.6726e-24  # deuterium mass/2 in g (approx expro_mass_deuterium/2)
    cs = sqrt.(k .* (1e3 .* te) ./ (2.0 * mp)) ./ 1e2  # m/s

    w0p = expro_bound_deriv(w0, rmin_work)
    gamma_e_phys = -rmin_work ./ q .* w0p
    gamma_p_phys = -rmaj .* w0p
    gamma_e_norm = (a ./ cs) .* gamma_e_phys
    gamma_p_norm = (a ./ cs) .* gamma_p_phys

    rlns_e = dlnnedr .* a
    rlts_e = dlntedr .* a
    rlns_i = [dlnnidr[i] .* a for i in 1:nion]
    rlts_i = [dlntidr[i] .* a for i in 1:nion]
    rlns_i[is_ep] = dlnnidr_ep .* a

    return (
        nr=nr, nion=nion, rho=rho, rmin=rmin_work, rmaj=rmaj, q=q, w0=w0,
        ne=ne, te=te, ni=ni, ti=ti, ni_qn=ni_qn, a=a, cs=cs,
        dlnnedr=dlnnedr, dlntedr=dlntedr, dlnnidr=dlnnidr, dlntidr=dlntidr,
        rlns_e=rlns_e, rlts_e=rlts_e, rlns_i=rlns_i, rlts_i=rlts_i,
        gamma_e=gamma_e_norm, gamma_p=gamma_p_norm,
        is_ep=is_ep,
    )
end

function compare_field(name, ref, test, rho, ir_list; ref_is_full=true, test_is_full=true)
    errs = Float64[]
    for (k, ir) in enumerate(ir_list)
        rv = ref_is_full ? ref[ir] : ref[k]
        tv = test_is_full ? test[ir] : test[k]
        push!(errs, rel_err(rv, tv))
    end
    ir_worst = ir_list[argmax(errs)]
    rv_w = ref_is_full ? ref[ir_worst] : ref[argmax(errs)]
    tv_w = test_is_full ? test[ir_worst] : test[argmax(errs)]
    @printf("  %-22s max=%.3e mean=%.3e  (worst ρ=%.3f ref=%.6g test=%.6g)\n",
        name, maximum(errs), mean(errs), rho[argmax(errs)], rv_w, tv_w)
    return maximum(errs)
end

function main()
    @assert isfile(GACODE_FILE) "missing $GACODE_FILE"
    rho_scan = parse_rho_scan()

    println("=== IMAS extraEP vs Fortran EXPRO (dump.gacode + expro_util logic) ===")
    println("CASE_DIR: ", CASE_DIR)
    println("IS_EP (IMAS): ", IS_EP_IMAS, "  → GACODE ion index: ", IS_EP_GACODE)
    println("rho scan (n=$(length(rho_scan))): ", rho_scan)

    ft = fortran_expro_profiles(GACODE_FILE; is_ep=IS_EP_GACODE, tglfep_nion=2)

    inputFile = joinpath(CASE_DIR, "input.gacode")
    inputGACODE = GACODE.load(inputFile)
    dd = IMAS.dd(inputGACODE)
    _, extraEP = InputTGLFEP(dd, rho_scan; is_ep=IS_EP_IMAS)
    w0_imas = -dd.core_profiles.profiles_1d[].rotation_frequency_tor_sonic

    grid = extraEP["grid"]
    a_j = extraEP["RMIN"][end]
    ep_slot = extraEP["EP_SLOT"]
    ns = extraEP["NS"]

    # Map scan rho -> IMAS index and EXPRO index
    ir_imas = [nearest_index(grid, ρ) for ρ in rho_scan]
    ir_expro = [nearest_index(ft.rho, ρ) for ρ in rho_scan]

    println("\nGrid sizes: EXPRO nr=$(ft.nr), IMAS nr=$(length(grid))")
    @printf("At ρ=0.01: ir_expro=%d ir_imas=%d  rmin_ft=%.6f rmin_imas=%.6f\n",
        ir_expro[1], ir_imas[1], ft.rmin[ir_expro[1]], extraEP["RMIN"][ir_imas[1]])

    println("\n--- Raw profiles (no derivative) ---")
    compare_field("rmin [m]", ft.rmin, extraEP["RMIN"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("ne [1e19]", ft.ne, extraEP["DENS_1"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("te [keV]", ft.te, extraEP["TEMP_1"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("ni1 [1e19]", ft.ni[1], extraEP["DENS_2"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("ti1 [keV]", ft.ti[1], extraEP["TEMP_2"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("ni_EP [1e19]", ft.ni[ft.is_ep], extraEP["DENS_$ep_slot"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("ti_EP [keV]", ft.ti[ft.is_ep], extraEP["TEMP_$ep_slot"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("q (signed)", ft.q, extraEP["Q"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("|q|", abs.(ft.q), extraEP["Q"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("w0 [rad/s]", ft.w0, w0_imas[ir_imas], rho_scan, ir_expro; test_is_full=false)

    println("\n--- Log gradients / RLNS (Julia uses IMAS.calc_z; Fortran uses bound_deriv) ---")
    compare_field("RLNS_e", ft.rlns_e, a_j .* extraEP["DLNNDR_1"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("RLTS_e", ft.rlts_e, a_j .* extraEP["DLNTDR_1"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("RLNS_ion1", ft.rlns_i[1], a_j .* extraEP["DLNNDR_2"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("RLTS_ion1", ft.rlts_i[1], a_j .* extraEP["DLNTDR_2"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("RLNS_EP", ft.rlns_i[ft.is_ep], a_j .* extraEP["DLNNDR_$ep_slot"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("RLTS_EP", ft.rlts_i[ft.is_ep], a_j .* extraEP["DLNTDR_$ep_slot"][ir_imas], rho_scan, ir_expro; test_is_full=false)

    # Recompute Julia-side gradients with expro_bound_deriv on IMAS rmin (isolate derivative algorithm)
    rmin_j = extraEP["RMIN"]
    ne_j = extraEP["DENS_1"]
    te_j = extraEP["TEMP_1"]
    dlnn_bd = expro_bound_deriv(-log.(max.(ne_j, 1e-30)), rmin_j)
    dlnt_bd = expro_bound_deriv(-log.(max.(te_j, 1e-30)), rmin_j)
    dlnn_calcz = extraEP["DLNNDR_1"]
    println("\n--- Derivative method on same IMAS grid (ρ scan) ---")
    compare_field("dlnn_e bound_deriv", dlnn_bd, dlnn_calcz, rho_scan, ir_imas)
    compare_field("dlnt_e bound_deriv", dlnt_bd, extraEP["DLNTDR_1"], rho_scan, ir_imas)

    println("\n--- Rotation / flow (Fortran TGLFEP rotation_flag=0 → vpar=0 in driver; extraEP still fills gamma) ---")
    compare_field("gammaE (a/cs norm)", ft.gamma_e, extraEP["gammaE"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("gammap (a/cs norm)", ft.gamma_p, extraEP["gammap"][ir_imas], rho_scan, ir_expro; test_is_full=false)
    compare_field("cs [m/s]", ft.cs, extraEP["CS"][ir_imas] ./ 100, rho_scan, ir_expro; test_is_full=false)

    println("\n--- Normalized geometry in extraEP ---")
    rmin_norm_ft = ft.rmin ./ ft.a
    rmin_norm_im = extraEP["RMIN"][ir_imas] ./ a_j
    compare_field("RMIN/a", rmin_norm_ft, rmin_norm_im, rho_scan, ir_expro; test_is_full=false)

    println("\n--- Per-ρ table (selected fields) ---")
    @printf("%6s %6s %6s %12s %12s %12s %12s %12s\n",
        "rho", "ir_ex", "ir_im", "RLNS_e_ft", "RLNS_e_im", "RLNS_EP_ft", "RLNS_EP_im", "gammaE_ft")
    for (k, ρ) in enumerate(rho_scan)
        ir_e, ir_i = ir_expro[k], ir_imas[k]
        @printf("%6.3f %6d %6d %12.5f %12.5f %12.5f %12.5f %12.5g\n",
            ρ, ir_e, ir_i,
            ft.rlns_e[ir_e], a_j * extraEP["DLNNDR_1"][ir_i],
            ft.rlns_i[ft.is_ep][ir_e], a_j * extraEP["DLNNDR_$ep_slot"][ir_i],
            ft.gamma_e[ir_e])
    end

    println("\n=== done ===")
end

main()
