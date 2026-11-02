# Marginal ky/width/factor QL data for ALPHA quasi-linear diffusivity (Fortran gamma_star_hat / diff_star).

"""Per-radius marginal TGLF-EP QL mode data (one scan point)."""
struct MarginalQLData{T<:Real}
    gamma_star::Vector{T}   # physical growth rate [1/s], length NMODES
    diff_star::Vector{T}    # QL diffusivity weight D_W [m^2/s]
    dep::Vector{T}          # DEP = EP_QL_flux/(AS·RLNS) at reference scalefactor
    kymark::T
    width::T
    factor_mark::T
    unstable::Bool
    flux_scan_factors::Vector{T}  # n_sf values used for D_W (empty if fallback)
end

"""Opt-in env override for file/driver workflows (`TGLFEP_JULIA_QL_FLUX_SCAN=1`). FUSE gates via `ql_flux_scan` kwarg."""
_ql_flux_scan_env() = get(ENV, "TGLFEP_JULIA_QL_FLUX_SCAN", "0") == "1"

"""Env-var default for the number of eigenmodes kept (`TGLFEP_JULIA_NMODES`, default 4). Overridden by an explicit `nmodes` kwarg."""
_nmodes_env() = parse(Int, get(ENV, "TGLFEP_JULIA_NMODES", "4"))

"""Env-var default for the grid scale-factor iteration count (`TGLFEP_JULIA_K_MAX`, default 4). Overridden by an explicit `k_max` kwarg; affects the `:grid` solver only."""
_k_max_env() = parse(Int, get(ENV, "TGLFEP_JULIA_K_MAX", "4"))

function _gyrobohm_D(cs_cm::T, bunit::T, a_m::T) where {T<:Real}
    mp = T(1.6726219e-27)
    e = T(1.602176634e-19)
    cs = cs_cm / T(100)
    rho_s = cs * mp / (e * max(bunit, eps(T)))
    return rho_s^2 * cs / max(a_m, eps(T))
end

function _chi_gB_profile(inputsEP::Options{T}, inputsPR::profile{T}, ir::Int) where {T<:Real}
    a_m = inputsPR.RMIN[end]
    if !ismissing(inputsPR.RHO_STAR) && length(inputsPR.RHO_STAR) >= ir
        rho_s = inputsPR.RHO_STAR[ir] * a_m
        fr = inputsEP.F_REAL[ir]
        if !ismissing(inputsPR.RMAJ) && length(inputsPR.RMAJ) >= ir
            cs_m = fr * T(2π) * T(1e3) * inputsPR.RMAJ[ir]
            return rho_s^2 * cs_m / max(a_m, eps(T))
        end
        return rho_s^2 / max(a_m, eps(T))
    end
    return _gyrobohm_from_profile(inputsEP, inputsPR, ir) / max(a_m, eps(T))
end

function _gyrobohm_from_profile(inputsEP::Options{T}, inputsPR::profile{T}, ir::Int) where {T<:Real}
    fr = inputsEP.F_REAL[ir]
    a_m = inputsPR.RMIN[end] / T(100)
    if !ismissing(inputsPR.RHO_STAR) && length(inputsPR.RHO_STAR) >= ir
        rho_s = inputsPR.RHO_STAR[ir] * a_m
        if !ismissing(inputsPR.RMAJ) && length(inputsPR.RMAJ) >= ir
            cs_cm = fr * T(2π) * T(1e3) * inputsPR.RMAJ[ir] * T(100)
            cs = cs_cm / T(100)
            return rho_s^2 * cs / max(a_m, eps(T))
        end
        return rho_s^2 / max(a_m, eps(T))
    end
    if !ismissing(inputsPR.B_UNIT) && length(inputsPR.B_UNIT) >= ir && !ismissing(inputsPR.RMAJ)
        cs_cm = fr * T(2π) * T(1e3) * inputsPR.RMAJ[ir] * T(100)
        return _gyrobohm_D(cs_cm, inputsPR.B_UNIT[ir], a_m)
    end
    return T(1.0)
end

"""Reference n_e in 10^19 m^-3 for absolute flux scaling (profile lacks ne array)."""
function _ne_19_ref(::profile{T}, ::Int) where {T<:Real}
    return T(5.0)
end

"""EP slowing-down density gradient |dn_sd/dr| [10^19 m^-3 / m] from a/Ln and AS/ni."""
function _rg_n_sd_19(inputsPR::profile{T}, inputsEP::Options{T}, ir::Int) where {T<:Real}
    is = inputsEP.IS_EP + 1
    a_m = inputsPR.RMIN[end]
    rlns = inputsPR.RLNS[ir, is]
    n_ep = inputsPR.AS[ir, is] * _ne_19_ref(inputsPR, ir)
    return abs(rlns * n_ep / max(a_m, eps(T)))
end

"""TGLF EP QL particle flux → physical Γ [m^-2 s^-1] (Fortran `chi_gB` × QL flux scaling)."""
function _ep_particle_flux_phys(
    ep_ql::T,
    inputsEP::Options{T},
    inputsPR::profile{T},
    ir::Int,
) where {T<:Real}
    chi = _chi_gB_profile(inputsEP, inputsPR, ir)
    a_m = inputsPR.RMIN[end]
    ne_19 = _ne_19_ref(inputsPR, ir)
    return abs(chi * ep_ql * ne_19 * T(1e19) / max(a_m, eps(T)))
end

function _default_flux_scan_factors(fmark::T, fmax::T) where {T<:Real}
    f0 = max(fmark, T(0.05))
    fac = T[f0 * T(0.5), f0, f0 * T(1.5)]
    return [clamp(f, T(0.01), fmax) for f in fac]
end

function _slope_linear(y::AbstractVector{T}, x::AbstractVector{T}) where {T<:Real}
    n = length(x)
    n >= 2 || return zero(T)
    num = zero(T)
    den = zero(T)
    xm = sum(x) / n
    ym = sum(y) / n
    for i in 1:n
        dx = x[i] - xm
        num += dx * (y[i] - ym)
        den += dx * dx
    end
    return abs(den) < eps(T) ? zero(T) : num / den
end

"""
    ql_flux_scan_at_marginal(inputsEP, inputsPR; factors, use_gpu) -> (ep_flux, dep, factors)

Run `TJLFEP_ky` at fixed marginal `KYMARK`/`WIDTH_IN` for each scalefactor `n_sf`.
`MODE_IN=2` (EP drive only, thermal a/Ln → 1e-6) matches Fortran “thermal gradients off”.
"""
function ql_flux_scan_at_marginal(
    inputsEP::Options{T},
    inputsPR::profile{T};
    factors::AbstractVector{T}=T[],
    use_gpu::Bool=false,
) where {T<:Real}
    fmax = coalesce(inputsEP.FACTOR_MAX, T(10))
    facs = isempty(factors) ? _default_flux_scan_factors(inputsEP.FACTOR_IN, fmax) : collect(T, factors)
    nm = inputsEP.NMODES
    nf = length(facs)
    ep_flux = zeros(T, nf, nm)
    dep = zeros(T, nf, nm)
    local_ep = deepcopy(inputsEP)
    local_ep.MODE_IN = 2
    local_ep.KYHAT_IN = inputsEP.KYMARK
    local_ep.KY_MODEL = 3
    for (j, f) in enumerate(facs)
        local_ep.FACTOR_IN = f
        _, _, _, _, _, dep_j, flux_j = TJLFEP_ky(local_ep, inputsPR, "", 0; use_gpu=use_gpu)
        for n in 1:nm
            ep_flux[j, n] = flux_j[n]
            dep[j, n] = dep_j[n]
        end
    end
    return ep_flux, dep, facs
end

"""
    diff_star_from_D_W(ep_flux, factors, inputsEP, inputsPR) -> Vector

Fortran: `D_W_k = (dFlux_W_k/dn_sf) / (-d n_sd/dr)` with flux linear in `n_sf`.
"""
function diff_star_from_D_W(
    ep_flux::AbstractMatrix{T},
    factors::AbstractVector{T},
    inputsEP::Options{T},
    inputsPR::profile{T},
) where {T<:Real}
    ir = inputsEP.IR
    nm = size(ep_flux, 2)
    rg = max(_rg_n_sd_19(inputsPR, inputsEP, ir), eps(T))
    diff = zeros(T, nm)
    for n in 1:nm
        gvec = [_ep_particle_flux_phys(ep_flux[j, n], inputsEP, inputsPR, ir) for j in 1:length(factors)]
        dflux_dnsf = _slope_linear(gvec, factors)
        if isfinite(dflux_dnsf) && abs(dflux_dnsf) > eps(T)
            diff[n] = abs(dflux_dnsf) / rg
        end
    end
    return diff
end

"""
    extract_marginal_ql(inputsEP, inputsPR, growthrate, dep_scan, factor; ...) -> MarginalQLData

`gamma_star` from the stability scan; `diff_star` from the Fortran flux scan only when `use_flux_scan=true`.
"""
function extract_marginal_ql(
    inputsEP::Options{T},
    inputsPR::profile{T},
    growthrate::AbstractArray{T,4},
    dep_scan::AbstractArray{T,4},
    factor::AbstractVector{T};
    imark::AbstractMatrix{<:Integer},
    ikyhat_mark::Int,
    iefwid_mark::Int,
    imark_min::Int,
    nkyhat::Int,
    nefwid::Int,
    nfactor::Int,
    use_gpu::Bool=false,
    use_flux_scan::Bool=false,
) where {T<:Real}
    nm = inputsEP.NMODES
    ir = inputsEP.IR
    fr = inputsEP.F_REAL[ir]
    unstable = imark_min <= nfactor

    ik = unstable ? max(ikyhat_mark, 1) : 1
    iw = unstable ? max(iefwid_mark, 1) : 1
    ifa0 = unstable ? min(imark[ik, iw], nfactor) : 1
    ifa0 = max(ifa0, 1)

    gamma_star = zeros(T, nm)
    diff_star = zeros(T, nm)
    dep_out = zeros(T, nm)
    flux_factors = T[]
    for n in 1:nm
        dep_out[n] = dep_scan[ik, iw, ifa0, n]
        gamma_star[n] = fr * growthrate[ik, iw, ifa0, n]
    end

    fmark = unstable ? inputsEP.FACTOR_IN : factor[1]
    if unstable && use_flux_scan
        ep_flux, _, flux_factors = ql_flux_scan_at_marginal(inputsEP, inputsPR; use_gpu=use_gpu)
        diff_star .= diff_star_from_D_W(ep_flux, flux_factors, inputsEP, inputsPR)
        for n in 1:nm
            if !isfinite(diff_star[n]) || diff_star[n] <= zero(T)
                ifa1 = min(ifa0 + 1, nfactor)
                ddep = (dep_scan[ik, iw, ifa1, n] - dep_scan[ik, iw, ifa0, n]) /
                       max(factor[ifa1] - factor[ifa0], eps(T))
                diff_star[n] = _diff_star_fallback(ddep, dep_out[n], inputsEP, inputsPR, ir)
            end
        end
    else
        ifa1 = min(ifa0 + 1, nfactor)
        for n in 1:nm
            ddep = (dep_scan[ik, iw, ifa1, n] - dep_scan[ik, iw, ifa0, n]) /
                   max(factor[ifa1] - factor[ifa0], eps(T))
            diff_star[n] = _diff_star_fallback(ddep, dep_out[n], inputsEP, inputsPR, ir)
        end
    end

    return MarginalQLData{T}(
        gamma_star, diff_star, dep_out,
        unstable ? inputsEP.KYMARK : zero(T),
        unstable ? inputsEP.WIDTH_IN : inputsEP.WIDTH_MIN,
        fmark, unstable, flux_factors,
    )
end

function _diff_star_fallback(
    ddep_df::T,
    dep::T,
    inputsEP::Options{T},
    inputsPR::profile{T},
    ir::Int,
) where {T<:Real}
    D_gb = _gyrobohm_from_profile(inputsEP, inputsPR, ir)
    if isfinite(ddep_df) && abs(ddep_df) > eps(T)
        return max(abs(ddep_df), zero(T)) * D_gb
    elseif isfinite(dep) && abs(dep) > eps(T)
        return max(abs(dep), zero(T)) * D_gb
    else
        return zero(T)
    end
end

"""
    build_alpha_ql_modes(marginals, rho_scan, rho_grid, dndr_crit; km_max=5) -> Vector

Map per-scan `MarginalQLData` to a radial vector of named tuples for `ALPHA.QLModeInput`.
"""
function build_alpha_ql_modes(
    marginals::AbstractVector{<:Union{Nothing,MarginalQLData{T}}},
    rho_scan::AbstractVector,
    rho_grid::AbstractVector{T},
    dndr_crit::AbstractVector{T};
    km_max::Int=5,
) where {T<:Real}
    n = length(rho_grid)
    modes = Vector{Any}(undef, km_max)
    for km in 1:km_max
        gs = zeros(T, n)
        ds = zeros(T, n)
        for i in 1:n
            j = _nearest_scan_index(rho_grid[i], rho_scan)
            mq = marginals[j]
            if mq !== nothing && km <= length(mq.gamma_star)
                gs[i] = mq.gamma_star[km]
                ds[i] = max(mq.diff_star[km], zero(T))
                if !mq.unstable || !isfinite(gs[i]) || gs[i] <= zero(T)
                    gs[i] = T(0.1)
                    ds[i] = max(ds[i], T(1.0))
                end
            else
                gs[i] = T(0.1)
                ds[i] = T(1.0)
            end
        end
        modes[km] = (; gamma_star=gs, diff_star=ds, rg_n_crit=collect(T, dndr_crit),
                     crit_index_shift=km == 1 ? 0 : _island_shift(km), crit_scale=one(T))
    end
    return modes
end

function _island_shift(km::Int)
    delays = (0, 20, 5, 10, 15)
    return km <= length(delays) ? delays[km] : 0
end

function _nearest_scan_index(rho::Real, rho_scan::AbstractVector)
    j = 1
    dmin = abs(rho - rho_scan[1])
    for k in 2:length(rho_scan)
        d = abs(rho - rho_scan[k])
        if d < dmin
            dmin = d
            j = k
        end
    end
    return j
end
