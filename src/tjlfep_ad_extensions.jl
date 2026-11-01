# ─────────────────────────────────────────────────────────────────────────────
# TJLFEP AD helpers for ForwardDiff.Dual element types.
#
# The AD-compatible eigen rules (real-symmetric, complex, and the generalized
# complex eigenproblem A r = λ B r) now live in TJLF's TJLFForwardDiffExt and are
# dispatched on TJLF-owned functions (_sym_eigen / _herm_eigen /
# _generalized_eigenvalues) rather than pirating LinearAlgebra.eigen. TJLFEP no
# longer defines any eigen methods here; it only provides the Dual-safe run
# wrapper and a Float64 projection used by the wavefunction path.
# ─────────────────────────────────────────────────────────────────────────────

import ForwardDiff
import TJLF

# AD factor-search lower bound, tied to the grid's k_max so the AD solvers floor the same
# way the grid does. The grid `kwscale_scan` runs k_max rounds of nfactor-point bracketing
# (f1←2·fmark, f0←0), bottoming out at FACTOR_IN/(nfactor·4^(k_max-1)); with the canonical
# nfactor=8, k_max=4 that is FACTOR_IN/512 (the historical hard-coded floor). Generalizing to
# k_max lets a larger k_max deepen the AD factor search consistently with the grid, while
# k_max=4 reproduces the previous shi/512 default bit-for-bit.
_ad_factor_floor(shi::Real, k_max::Integer; nfactor::Integer = 8) = shi / (nfactor * 4.0^(k_max - 1))

# ── AD-safe run for non-Float64 element types (Dual, etc.) ───────────────────
#
# TJLF.run's single-ky branch hardcodes zeros(Float64,...)/zeros(ComplexF64,...)
# for result arrays, which fails for Dual. This function is identical to that
# branch but allocates with zeros(T,...). All internal TJLF calls are unchanged.
# Called from TJLFEP_ky via:  T === Float64 ? TJLF.run(x) : _tjlf_run_dual(x, T)
function _tjlf_run_dual(inputTJLF, ::Type{T}; use_gpu::Bool = false) where {T}
    TJLF.checkInput(inputTJLF)
    outputHermite = TJLF.gauss_hermite(inputTJLF)
    satParams     = TJLF.get_sat_params(inputTJLF)
    inputTJLF.KY_SPECTRUM .= TJLF.get_ky_spectrum(inputTJLF, satParams.grad_r0)

    ns     = inputTJLF.NS
    nmodes = inputTJLF.NMODES
    nbasis = inputTJLF.NBASIS_MAX

    nmodes_out, gamma_out, freq_out,
        particle_QL_out, energy_QL_out, stress_tor_QL_out, stress_par_QL_out, exchange_QL_out,
        _ft, field_weight_out_3d, _phi = TJLF.tjlf_LS(inputTJLF, satParams, outputHermite,
                                                       inputTJLF.KY, nbasis, inputTJLF.VEXB_SHEAR, 1;
                                                       use_gpu=use_gpu)

    eigenvalue       = zeros(T,          nmodes, 1, 2)
    QL_weights       = zeros(T,          3, ns, nmodes, 1, 5)
    field_weight_out = zeros(Complex{T}, 3, nbasis, nmodes, 1)

    eigenvalue[:, 1, 1]                      .= gamma_out
    eigenvalue[:, 1, 2]                      .= freq_out
    QL_weights[:, :, 1:nmodes_out, 1, 1]    .= particle_QL_out[:, :, 1:nmodes_out]
    QL_weights[:, :, 1:nmodes_out, 1, 2]    .= energy_QL_out[:, :, 1:nmodes_out]
    QL_weights[:, :, 1:nmodes_out, 1, 3]    .= stress_tor_QL_out[:, :, 1:nmodes_out]
    QL_weights[:, :, 1:nmodes_out, 1, 4]    .= stress_par_QL_out[:, :, 1:nmodes_out]
    QL_weights[:, :, 1:nmodes_out, 1, 5]    .= exchange_QL_out[:, :, 1:nmodes_out]
    field_weight_out[:, :, 1:nmodes_out, 1] .= field_weight_out_3d[:, :, 1:nmodes_out]

    QL_flux_out   = Array{T}(undef, 0, 0, 0)
    flux_spectrum = Array{T}(undef, 0, 0, 0)
    return (QL_weights=QL_weights, eigenvalue=eigenvalue, QL_flux_out=QL_flux_out,
            flux_spectrum=flux_spectrum, field_weight_out=field_weight_out)
end

# ── Strip Dual partials from InputTJLF → InputTJLF{Float64} ─────────────────
# Used before calling get_wavefunction, which has Float64-typed geometry arrays
# internally. The wavefunction shape is independent of the differentiated
# parameters, so extracting .value is exact (no information lost).
#
# NOTE: `src` may be either TJLFEP.InputTJLF or TJLF.InputTJLF; no type
# annotation so both are accepted. Runtime field copy works for either.
function _to_float64_input(src)
    dst = TJLF.InputTJLF{Float64}(src.NS, src.NKY)
    for fn in fieldnames(typeof(src))
        v = getfield(src, fn)
        if v isa ForwardDiff.Dual
            setfield!(dst, fn, ForwardDiff.value(v))
        elseif v isa AbstractArray && eltype(v) <: ForwardDiff.Dual
            setfield!(dst, fn, ForwardDiff.value.(v))
        elseif v isa AbstractArray && eltype(v) <: Complex && real(eltype(v)) <: ForwardDiff.Dual
            setfield!(dst, fn, map(x -> ComplexF64(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x))), v))
        else
            try; setfield!(dst, fn, v); catch; end
        end
    end
    return dst
end

# ── Promote a Float64 InputTJLF → InputTJLF{D} (D <: ForwardDiff.Dual) ────────
# Inverse of `_to_float64_input`: real scalars/arrays are widened to the Dual
# element type with zero partials (they are constants w.r.t. the seeded
# variable), complex arrays become Complex{D}, and Int/Bool/String/Missing
# fields are copied verbatim. The caller then overwrites the handful of fields
# that actually depend on the differentiated parameter with truly Dual values.
function _to_dual_input(src, ::Type{D}) where {D<:ForwardDiff.Dual}
    dst = TJLF.InputTJLF{D}(src.NS, src.NKY)
    for fn in fieldnames(typeof(src))
        v = getfield(src, fn)
        if v isa AbstractFloat
            setfield!(dst, fn, D(v))
        elseif v isa AbstractArray && eltype(v) <: AbstractFloat
            setfield!(dst, fn, D.(v))
        elseif v isa AbstractArray && eltype(v) <: Complex
            setfield!(dst, fn, map(x -> Complex{D}(D(real(x)), D(imag(x))), v))
        else
            try; setfield!(dst, fn, v); catch; end
        end
    end
    return dst
end

# ── Static InputTJLF configuration shared with TJLFEP_ky ─────────────────────
# Mirrors the in-place overrides applied right after `TJLF_map` inside
# `TJLFEP_ky` (single-ky path: transport model off, fixed width/basis, eigen
# solve on, TGLF-EP physics switches). Factored out so the AD building block
# below configures the eigensolve input identically to the production scan
# without re-running the rest of TJLFEP_ky. The validation script asserts that a
# Float64 input built this way reproduces TJLFEP_ky's growth rates exactly.
function _configure_inputTJLF_for_ky!(inputTJLF, inputsEP)
    inputTJLF.USE_TRANSPORT_MODEL = false
    inputTJLF.KYGRID_MODEL        = 0
    inputTJLF.NMODES              = inputsEP.NMODES
    inputTJLF.NBASIS_MIN          = inputsEP.N_BASIS
    inputTJLF.NBASIS_MAX          = inputsEP.N_BASIS
    # Match TJLFEP_ky (and the Fortran nb48 driver): the quadrature grid must
    # resolve the highest basis function, nxgrid = max(n_basis, 32).
    inputTJLF.NXGRID              = max(inputsEP.N_BASIS, 32)
    inputTJLF.WIDTH               = inputsEP.WIDTH_IN
    inputTJLF.FIND_WIDTH          = false
    inputTJLF.USE_AVE_ION_GRID    = false
    inputTJLF.WIDTH_SPECTRUM     .= inputTJLF.WIDTH
    inputTJLF.FIND_EIGEN          = true
    # AD only needs eigenvalues (growth rate): turn off the QL-flux / eigenvector
    # block in tjlf_LS, whose per-mode inverse-iteration LU (lu!(zmat)) is by far
    # the most expensive Dual operation and is unused by gamma_dgamma_dfactor.
    # The eigenvalue solve runs before that block, so γ is unchanged.
    inputTJLF.IFLUX               = false
    inputTJLF.RLNP_CUTOFF         = 18.0
    inputTJLF.BETA_LOC            = 0.0
    inputTJLF.DAMP_PSI            = 0.0
    inputTJLF.DAMP_SIG            = 0.0
    inputTJLF.WDIA_TRAPPED        = 0.0
    if inputTJLF.SAT_RULE == 2 || inputTJLF.SAT_RULE == 3
        inputTJLF.UNITS        = "CGYRO"
        inputTJLF.XNU_MODEL    = 3
        inputTJLF.WDIA_TRAPPED = 1.0
    end
    inputTJLF.KX0_LOC = 0.0
    return inputTJLF
end

# Resolve the TGLF-EP drive selector (MODE_IN) an AD building block should pass to
# TJLF_map / TJLFEP_ky. Explicit `override` wins; otherwise honor `inputsEP.MODE_IN`
# (set by mainsub: 2 = EP drive only for PROCESS_IN=5, 4 = thermal+EP / ITG-TEM for
# PROCESS_IN=6), defaulting to 2 when unset — reproducing the historical behavior where
# TJLF_map forced MODE_IN=2 for the AD path.
_resolve_ad_mode_in(inputsEP, override::Union{Nothing,Int}) =
    override !== nothing ? override : coalesce(inputsEP.MODE_IN, 2)

# ── Promote a Float64 value/array to the Dual element type (constants) ───────
_to_dual_val(v::AbstractFloat, ::Type{D}) where {D<:ForwardDiff.Dual} = D(v)
_to_dual_val(v::AbstractArray{<:AbstractFloat}, ::Type{D}) where {D<:ForwardDiff.Dual} = D.(v)
_to_dual_val(v, ::Type{D}) where {D<:ForwardDiff.Dual} = v   # Int/Bool/String/Missing/Vector{Int}/…

# Copy every field of `src` into `dst` (same struct, Dual element type), widening
# real scalars/arrays to `D` with zero partials. Fields named in `seed` are taken
# verbatim from `seed` instead (used to inject the truly-differentiated input).
function _promote_struct_to_dual!(dst, src, ::Type{D}, seed) where {D<:ForwardDiff.Dual}
    for fn in fieldnames(typeof(src))
        if haskey(seed, fn)
            setfield!(dst, fn, seed[fn])
        else
            setfield!(dst, fn, _to_dual_val(getfield(src, fn), D))
        end
    end
    return dst
end

# Options{Float64} → Options{D}, with FACTOR_IN replaced by the seeded Dual.
function _to_dual_options(src::Options{Float64}, factor_seed::D, nr::Int) where {D<:ForwardDiff.Dual}
    # Constructor dims only size scratch arrays that _promote_struct_to_dual!
    # immediately overwrites, so coalesce any `missing` dims to a safe value.
    dst = Options{D}(src.SCAN_N, src.WIDTH_IN_FLAG, coalesce(src.NN, 1), nr, coalesce(src.JTSCALE_MAX, 1), src.NMODES)
    return _promote_struct_to_dual!(dst, src, D, Dict{Symbol,Any}(:FACTOR_IN => factor_seed))
end

# Options{Float64} → Options{D} with an arbitrary set of seeded Dual fields.
# `seeds` maps EP field names (e.g. :FACTOR_IN, :KYHAT_IN, :WIDTH_IN) to the
# multi-partial Dual to inject; every other field is widened as a constant.
function _to_dual_options_seeded(src::Options{Float64}, seeds::AbstractDict, ::Type{D}, nr::Int) where {D<:ForwardDiff.Dual}
    dst = Options{D}(src.SCAN_N, src.WIDTH_IN_FLAG, coalesce(src.NN, 1), nr, coalesce(src.JTSCALE_MAX, 1), src.NMODES)
    return _promote_struct_to_dual!(dst, src, D, seeds)
end

# profile{Float64} → profile{D} (all entries are constants w.r.t. the factor).
function _to_dual_profile(src::profile{Float64}, ::Type{D}) where {D<:ForwardDiff.Dual}
    nr, ns = size(src.AS)
    dst = profile{D}(nr, ns)
    return _promote_struct_to_dual!(dst, src, D, Dict{Symbol,Any}())
end

"""
    gamma_dgamma_dfactor(inputsEP, inputsPR; use_gpu=false)

Single-point AD building block for the EP critical-gradient scan.

Evaluates the TGLF-EP dispersion at one `(KYHAT_IN, WIDTH_IN, FACTOR_IN, IR)`
operating point — exactly as the inner `kwscale_scan` combo does — and returns
both the per-mode growth rate `γ` and its exact derivative `dγ/d(FACTOR_IN)`,
computed in a single forward-mode AD pass by seeding a `ForwardDiff.Dual` on the
EP scale factor.

The derivative is taken *through `TJLF_map`* (by promoting `Options`/`profile` to
a Dual element type and seeding `FACTOR_IN`), so every channel by which the
factor reaches the TGLF input is captured automatically — not only the EP density
gradient `RLNS[is]` / density `AS`, but also the pressure-gradient (`P_PRIME_LOC`,
MHD-α) stabilization term, which for `PPRIME_METHOD ∈ {2,3}` varies with the
scaled EP gradient via the `sum*/sum1` ratio. Hand-replicating the factor→input
map is brittle for exactly this reason.

Returns a `NamedTuple`:
  - `gamma`            :: Vector{Float64}  per-mode growth rate (length NMODES)
  - `dgamma_dfactor`   :: Vector{Float64}  per-mode dγ/d(FACTOR_IN)
  - `freq`             :: Vector{Float64}  per-mode real frequency
  - `dfreq_dfactor`    :: Vector{Float64}  per-mode d(freq)/d(FACTOR_IN)
  - `factor`           :: Float64          the (clamped) factor the result is evaluated at

`inputsEP.IR`, `FACTOR_IN`, `KYHAT_IN`, and `WIDTH_IN` must be set on entry (as
the scan sets them). `inputsEP`/`inputsPR` are not mutated.
"""
function gamma_dgamma_dfactor(inputsEP::Options{Float64}, inputsPR::profile{Float64}; use_gpu::Bool = false,
                              mode_in_override::Union{Nothing,Int} = nothing)
    f0 = Float64(inputsEP.FACTOR_IN)
    mi = _resolve_ad_mode_in(inputsEP, mode_in_override)

    Tag = typeof(ForwardDiff.Tag(gamma_dgamma_dfactor, Float64))
    D   = ForwardDiff.Dual{Tag, Float64, 1}
    s   = ForwardDiff.Dual{Tag}(f0, one(Float64))   # seed: value=f0, ∂/∂factor=1

    nr = size(inputsPR.AS, 1)
    epD = _to_dual_options(inputsEP, s, nr)
    prD = _to_dual_profile(inputsPR, D)

    # Differentiate through the full factor→TGLF-input map. `mode_in_override` honors the
    # PROCESS_IN drive selector (2 = EP drive only, 4 = thermal+EP / ITG-TEM); without it
    # TJLF_map would force MODE_IN=2 regardless of what the caller set.
    inputD = TJLF_map(epD, prD; mode_in_override = mi)
    inputD isa Integer && error("gamma_dgamma_dfactor: TJLF_map rejected IR=$(inputsEP.IR) (out of range)")
    _configure_inputTJLF_for_ky!(inputD, epD)

    fclamp = ForwardDiff.value(epD.FACTOR_IN)   # post-clamp factor TJLF_map actually used
    if fclamp <= 0 || fclamp >= ForwardDiff.value(epD.FACTOR_MAX)
        @warn "gamma_dgamma_dfactor: FACTOR_IN=$fclamp is at/over the clamp range [0, $(ForwardDiff.value(epD.FACTOR_MAX))]; dγ/dfactor assumes an interior point"
    end

    # A singular TGLF dispersion matrix means no resolvable eigenmode — most commonly at the
    # separatrix scan point (ir≈NR) of a synthesized equilibrium, or at very narrow widths the
    # :wide seed grid probes. Mirror TJLFEP_ky's grid-path handling: treat the combo as AE-stable
    # (γ=0, ∂γ=0) instead of letting the SingularException crash the whole AD scan.
    local res
    try
        res = _tjlf_run_dual(inputD, D; use_gpu = use_gpu)
    catch err
        err isa SingularException || rethrow()
        @warn "gamma_dgamma_dfactor: singular TGLF dispersion matrix (ir=$(inputsEP.IR), ky=$(ForwardDiff.value(inputD.KY)), width=$(ForwardDiff.value(inputD.WIDTH))); treating combo as stable (γ=0, no AE mode)" maxlog = 5
        z = zeros(Float64, inputD.NMODES)
        return (gamma = z, dgamma_dfactor = copy(z), freq = copy(z), dfreq_dfactor = copy(z), factor = fclamp)
    end
    g = res.eigenvalue[:, 1, 1]   # [nmodes] growth rate (Dual)
    f = res.eigenvalue[:, 1, 2]   # [nmodes] frequency   (Dual)

    return (
        gamma          = ForwardDiff.value.(g),
        dgamma_dfactor = [ForwardDiff.partials(x, 1) for x in g],
        freq           = ForwardDiff.value.(f),
        dfreq_dfactor  = [ForwardDiff.partials(x, 1) for x in f],
        factor         = fclamp,
    )
end

# Operating-point variables that gamma_grad can seed. Each reaches the eigensolve
# through TJLF_map / _configure_inputTJLF_for_ky!:
#   FACTOR_IN → EP density/gradient + P_PRIME_LOC (MHD-α) channels
#   KYHAT_IN  → inputTJLF.KY  (TJLF_map KY_MODEL=3, line ~1092)
#   WIDTH_IN  → inputTJLF.WIDTH / WIDTH_SPECTRUM (_configure_inputTJLF_for_ky!)
const _GAMMA_GRAD_VARS = (:FACTOR_IN, :KYHAT_IN, :WIDTH_IN)

"""
    gamma_grad(inputsEP, inputsPR; vars=(:FACTOR_IN,:KYHAT_IN,:WIDTH_IN), use_gpu=false)

Multi-variable generalization of [`gamma_dgamma_dfactor`](@ref): one forward-mode
AD pass returning the per-mode growth rate `γ` and real frequency together with
their exact partials w.r.t. each operating-point variable in `vars` — any subset
of `(:FACTOR_IN, :KYHAT_IN, :WIDTH_IN)`. All partials are obtained in a single
N-partial `ForwardDiff.Dual` solve (one eigensolve), which is the building block
for the `(kyhat, width)` optimizer (Phase 3) and the freq/keep-condition root
finds (Phase 2).

Derivatives are taken *through `TJLF_map`* (by promoting `Options`/`profile` to
the Dual element type and seeding the chosen EP fields), so every channel by which
a variable reaches the TGLF input is captured automatically. `IFLUX=false`, so
only the eigenvalue block runs.

Returns a `NamedTuple`:
  - `gamma`  :: Vector{Float64}          per-mode growth rate (length NMODES)
  - `freq`   :: Vector{Float64}          per-mode real frequency
  - `dgamma` :: Matrix{Float64}          `NMODES × length(vars)`, `∂γ_m/∂vars[k]`
  - `dfreq`  :: Matrix{Float64}          `NMODES × length(vars)`, `∂freq_m/∂vars[k]`
  - `vars`   :: NTuple                   the variable order matching the columns

`inputsEP.IR`, `FACTOR_IN`, `KYHAT_IN`, `WIDTH_IN` must be set on entry (as the
scan sets them). `inputsEP`/`inputsPR` are not mutated.
"""
function gamma_grad(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                    vars::NTuple{N,Symbol} = _GAMMA_GRAD_VARS,
                    use_gpu::Bool = false,
                    mode_in_override::Union{Nothing,Int} = nothing) where {N}
    all(v -> v in _GAMMA_GRAD_VARS, vars) ||
        error("gamma_grad: vars must be a subset of $_GAMMA_GRAD_VARS; got $vars")
    mi = _resolve_ad_mode_in(inputsEP, mode_in_override)

    Tag = typeof(ForwardDiff.Tag(gamma_grad, Float64))
    D   = ForwardDiff.Dual{Tag, Float64, N}

    seeds = Dict{Symbol,Any}()
    for (i, v) in enumerate(vars)
        x0 = Float64(getfield(inputsEP, v))
        seeds[v] = ForwardDiff.Dual{Tag}(x0, ntuple(j -> j == i ? 1.0 : 0.0, N)...)
    end

    nr  = size(inputsPR.AS, 1)
    epD = _to_dual_options_seeded(inputsEP, seeds, D, nr)
    prD = _to_dual_profile(inputsPR, D)

    # See gamma_dgamma_dfactor: honor the MODE_IN drive selector rather than forcing 2.
    inputD = TJLF_map(epD, prD; mode_in_override = mi)
    inputD isa Integer && error("gamma_grad: TJLF_map rejected IR=$(inputsEP.IR) (out of range)")
    _configure_inputTJLF_for_ky!(inputD, epD)

    if :FACTOR_IN in vars
        fclamp = ForwardDiff.value(epD.FACTOR_IN)
        if fclamp <= 0 || fclamp >= ForwardDiff.value(epD.FACTOR_MAX)
            @warn "gamma_grad: FACTOR_IN=$fclamp is at/over the clamp range [0, $(ForwardDiff.value(epD.FACTOR_MAX))]; ∂γ/∂FACTOR_IN assumes an interior point"
        end
    end

    # Singular dispersion matrix → no resolvable eigenmode; degrade gracefully to AE-stable
    # (γ=0, all partials 0), matching TJLFEP_ky and gamma_dgamma_dfactor rather than crashing.
    local res
    try
        res = _tjlf_run_dual(inputD, D; use_gpu = use_gpu)
    catch err
        err isa SingularException || rethrow()
        @warn "gamma_grad: singular TGLF dispersion matrix (ir=$(inputsEP.IR), ky=$(ForwardDiff.value(inputD.KY)), width=$(ForwardDiff.value(inputD.WIDTH))); treating combo as stable (γ=0, no AE mode)" maxlog = 5
        NM = inputD.NMODES
        return (; gamma = zeros(Float64, NM), freq = zeros(Float64, NM),
                  dgamma = zeros(Float64, NM, N), dfreq = zeros(Float64, NM, N), vars)
    end
    g = res.eigenvalue[:, 1, 1]
    f = res.eigenvalue[:, 1, 2]
    NM = length(g)

    gamma  = ForwardDiff.value.(g)
    freq   = ForwardDiff.value.(f)
    dgamma = [ForwardDiff.partials(g[m], i) for m in 1:NM, i in 1:N]
    dfreq  = [ForwardDiff.partials(f[m], i) for m in 1:NM, i in 1:N]

    return (; gamma, freq, dgamma, dfreq, vars)
end

# AE-band upper frequency cut (FREQ_AE_UPPER = -|ω_GAM|): modes with freq below it
# are in the Alfvén-eigenmode band. Read from inputsEP if set (the standard read
# path sets it), else recomputed from the profile.
function _ae_band_upper(inputsEP::Options{Float64}, inputsPR::profile{Float64})
    fu = inputsEP.FREQ_AE_UPPER
    if !(fu === missing) && fu isa Real && !isnan(fu)
        return Float64(fu)
    end
    return -abs(inputsPR.omegaGAM[inputsEP.IR])
end

# Marginal growth-rate threshold γ* for the current radius, reproducing exactly
# what kwscale_scan/TJLF_map set on inputsEP.GAMMA_THRESH:
#   ROTATIONAL_SUPPRESSION_FLAG==1 : γ* = min(0.15·|γ_E/ŝ|, |γ_p|·2·min(1-r/a,r/a)/Rmaj)
#                                    (Bass PoP 2017 flow-shear AE suppression) — a
#                                    FINITE threshold ⇒ γ_keep(factor)=γ* is a smooth,
#                                    grid-independent root suitable for AD-Newton.
#   otherwise                      : γ* = 1e-7 (the "any positive AE growth" onset).
function _gamma_thresh_for(inputsEP::Options{Float64}, inputsPR::profile{Float64})
    ir = inputsEP.IR
    if coalesce(inputsEP.ROTATIONAL_SUPPRESSION_FLAG, 0) == 1
        r_over_a = inputsPR.RMIN[ir] / inputsPR.RMIN[end]
        gthr_max = abs(inputsPR.gammap[ir]) * 2.0 * (min(1.0 - r_over_a, r_over_a) / inputsPR.RMAJ[ir])
        gthr = 0.15 * abs(inputsPR.gammaE[ir] / inputsPR.SHEAR[ir])
        return min(gthr, gthr_max)
    else
        return 1.0e-7
    end
end

# Leading growth rate γ_lead and its dγ_lead/d(factor) at a given factor.
#
# `ae_band=false` (default): γ_lead = max over all modes — the raw indicator. This
# is dominated by background ITG/TEM modes that are unstable even at tiny EP
# factor and can be non-monotonic in factor, so it is unsuitable for an
# endpoint-only stability test.
#
# `ae_band=true`: γ_lead = max over only the modes whose real frequency lies in
# the AE band (freq < FREQ_AE_UPPER), mirroring the *primary* keep filter in
# TJLFEP_ky. This isolates the EP-driven Alfvén eigenmode (the instability the
# marginal scan actually tracks); if no mode is in-band the point is AE-stable and
# γ_lead is reported as 0. Uses only eigenvalues/frequencies, so it is compatible
# with the IFLUX=false fast path. (The secondary tearing/pinch/QL/θ² rejections
# need the wavefunction/QL weights, i.e. IFLUX=true, and are not applied here.)
function _gamma_lead_dfactor(inputsEP::Options{Float64}, inputsPR::profile{Float64}, f::Float64;
                             use_gpu::Bool = false, ae_band::Bool = false)
    ep = deepcopy(inputsEP)
    ep.FACTOR_IN = f
    r = gamma_dgamma_dfactor(ep, inputsPR; use_gpu = use_gpu)
    if ae_band
        fu = _ae_band_upper(inputsEP, inputsPR)
        cand = findall(<(fu), r.freq)
        isempty(cand) && return 0.0, 0.0, r   # no AE-band mode → AE-stable
        j = cand[argmax(@view r.gamma[cand])]
        return r.gamma[j], r.dgamma_dfactor[j], r
    end
    i = argmax(r.gamma)
    return r.gamma[i], r.dgamma_dfactor[i], r
end

"""
    marginal_factor(inputsEP, inputsPR; kwargs...) -> NamedTuple

AD-accelerated Newton root-find for the marginal EP scale factor at a fixed
`(IR, KYHAT_IN, WIDTH_IN)` operating point: the factor `f★` at which the leading
growth rate crosses the threshold,

    γ_lead(f★) = gamma_thresh .

This is the Julia-native, derivative-based complement to the Fortran 1-D factor
scan: instead of bracketing on an 8-point factor grid and taking a single secant
step, it runs a **safeguarded Newton** (Newton step with a bisection fallback)
using the exact `dγ_lead/d(factor)` from `gamma_dgamma_dfactor`, so it converges
quadratically and needs only a handful of TGLF eigensolves.

Keywords:
  - `gamma_thresh = nothing`        threshold; `nothing` ⇒ the case's own
                                    `GAMMA_THRESH` (rotational γ* if
                                    `ROTATIONAL_SUPPRESSION_FLAG=1`, else 1e-7)
  - `bracket = nothing`             `(f_lo, f_hi)` bracketing the sign change of
                                    `g(f)=γ_lead(f)-gamma_thresh`; auto-found if `nothing`
  - `scan_lo=1e-3, scan_hi=10.0, nscan=8`  geometric samples used to auto-bracket
  - `ae_band = false`               track only the AE-band-filtered γ (freq <
                                    `FREQ_AE_UPPER`) instead of the raw max-over-modes γ
  - `monotone_earlyout = true`      (raw γ_lead only; auto-disabled when `ae_band`)
                                    assume γ_lead↑ in `f` and probe `scan_hi`/`scan_lo`
                                    first, settling "always stable/unstable" (the
                                    high-γ*/ExB case) in ≤2 eigensolves instead of `nscan`
  - `tol = 1e-6`                    converged when `|g| < tol·(1+|γ_thresh|)` or step `< xtol`
  - `xtol = 1e-8`, `maxiter = 40`
  - `use_gpu = false`, `verbose = false`

Returns `(; factor, gamma_lead, gamma, freq, iters, evals, converged, bracket)`.
"""
function marginal_factor(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                         gamma_thresh::Union{Nothing,Float64} = nothing,
                         bracket::Union{Nothing,Tuple{Float64,Float64}} = nothing,
                         f_start::Union{Nothing,Float64} = nothing,
                         scan_lo::Float64 = 1.0e-3, scan_hi::Float64 = 10.0, nscan::Int = 8,
                         ae_band::Bool = false, monotone_earlyout::Bool = true,
                         tol::Float64 = 1.0e-6, xtol::Float64 = 1.0e-8, maxiter::Int = 40,
                         use_gpu::Bool = false, verbose::Bool = false)
    # nothing ⇒ use the case's own threshold (rotational γ* or 1e-7), matching kwscale_scan
    gamma_thresh = gamma_thresh === nothing ? _gamma_thresh_for(inputsEP, inputsPR) : gamma_thresh
    evals = Ref(0)
    g(f) = begin
        gl, dgl, r = _gamma_lead_dfactor(inputsEP, inputsPR, f; use_gpu = use_gpu, ae_band = ae_band)
        evals[] += 1
        verbose && @info "eval" factor=f gamma_lead=gl gobj=(gl - gamma_thresh) dgdf=dgl
        (gl - gamma_thresh, dgl, gl, r)
    end

    # ── Establish a bracket [a,b] with g(a) < 0 < g(b) ──
    local a, b, ga, gb
    a = b = NaN
    # Warm start: build a cheap LOCAL bracket by geometric expansion around f_start
    # (the previous root in a continuation/optimizer), avoiding the nscan global scan.
    if bracket === nothing && f_start !== nothing && scan_lo < f_start < scan_hi
        gs = g(f_start)[1]
        if gs >= 0           # at/above onset → step DOWN for a (g<0)
            b, gb = f_start, gs
            fa = f_start
            for _ in 1:8
                fa = max(scan_lo, fa / 1.5)
                gfa = g(fa)[1]
                if gfa < 0; a, ga = fa, gfa; break end
                fa <= scan_lo && break
            end
        else                 # below onset → step UP for b (g≥0)
            a, ga = f_start, gs
            fb = f_start
            for _ in 1:8
                fb = min(scan_hi, fb * 1.5)
                gfb = g(fb)[1]
                if gfb >= 0; b, gb = fb, gfb; break end
                fb >= scan_hi && break
            end
        end
    end
    if isnan(a) || isnan(b)
        a = b = NaN
        if bracket === nothing
            fs = exp.(range(log(scan_lo), log(scan_hi); length = nscan))
            # Top-of-range early-out, VALID ONLY for the raw (max-over-modes) γ_lead, which is
            # monotone-increasing in the EP drive factor — the same assumption rtsafe and the
            # f_start warm-start above already make. Probe scan_hi first: if even the largest
            # factor keeps γ_lead below threshold the mode is stable across all of [scan_lo,
            # scan_hi], settled in ONE eigensolve instead of the full nscan sweep (and detect
            # "always unstable" from the bottom point). DISABLED for ae_band=true: the
            # AE-band-filtered γ_AE(f) is a NARROW BUMP (rises above γ* in an interior window,
            # restabilises by scan_hi — see _ae_unstable_window), so a sub-threshold scan_hi
            # does NOT imply stability and the top-probe would wrongly drop a feasible bumped
            # seed. monotone_earlyout=false also forces the full sweep.
            eo = monotone_earlyout && !ae_band
            ghi = NaN
            if eo
                ghi = g(scan_hi)[1]
                if ghi < 0
                    return (; factor = scan_hi, gamma_lead = ghi + gamma_thresh,
                            gamma = Float64[], freq = Float64[], iters = 0,
                            evals = evals[], converged = false, bracket = (scan_lo, scan_hi))
                end
            end
            prev_f = fs[1]; prev_g = g(prev_f)[1]
            if eo && prev_g >= 0
                return (; factor = scan_lo, gamma_lead = prev_g + gamma_thresh,
                        gamma = Float64[], freq = Float64[], iters = 0,
                        evals = evals[], converged = false, bracket = (scan_lo, scan_hi))
            end
            for f in fs[2:end]
                # reuse the top-endpoint probe rather than re-evaluating g(scan_hi)
                gf = (eo && f == fs[end]) ? ghi : g(f)[1]
                if prev_g < 0 && gf >= 0
                    a, ga, b, gb = prev_f, prev_g, f, gf
                    break
                end
                prev_f, prev_g = f, gf
            end
            if isnan(a)
                @warn "marginal_factor: no sign change of γ_lead-thresh in [$scan_lo, $scan_hi] (always stable or always unstable); returning best endpoint"
                return (; factor = prev_g < 0 ? scan_hi : scan_lo, gamma_lead = prev_g + gamma_thresh,
                        gamma = Float64[], freq = Float64[], iters = 0, evals = evals[], converged = false, bracket = (scan_lo, scan_hi))
            end
        else
            a, b = bracket
            ga = g(a)[1]; gb = g(b)[1]
            ga < 0 < gb || error("marginal_factor: provided bracket ($a,$b) does not satisfy g(a)<0<g(b): g(a)=$ga g(b)=$gb")
        end
    end

    # ── Safeguarded Newton (rtsafe) ──
    f = 0.5 * (a + b)
    gf, dgf, gl, r = g(f)
    iters = 0
    converged = false
    for it in 1:maxiter
        iters = it
        # keep the bracket tight
        if gf < 0; a = f; else; b = f; end
        # Newton step, fall back to bisection if it leaves the bracket or dg≈0
        if abs(dgf) > 1e-30
            fn = f - gf / dgf
        else
            fn = NaN
        end
        if !(a < fn < b) || isnan(fn)
            fn = 0.5 * (a + b)   # bisection fallback
        end
        step = abs(fn - f)
        f = fn
        gf, dgf, gl, r = g(f)
        verbose && @info "newton" iter=it factor=f gobj=gf step=step
        if abs(gf) < tol * (1 + abs(gamma_thresh)) || step < xtol
            converged = true
            break
        end
    end

    return (; factor = f, gamma_lead = gl, gamma = r.gamma, freq = r.freq,
            iters = iters, evals = evals[], converged = converged, bracket = (a, b))
end

"""
    marginal_factor_df(inputsEP, inputsPR; kwargs...) -> NamedTuple

Derivative-FREE analogue of [`marginal_factor`](@ref): the marginal EP scale
factor `f★` at which the AE-band leading growth rate crosses threshold,

    γ_AE(f★) = gamma_thresh ,

found by **bracketing + a derivative-free root solver** (`SimpleNonlinearSolve`
`ITP`/`Brent`/`Bisection`) instead of the safeguarded Newton. This is the direct
replacement for the `:grid` engine's coarse 8-point factor scan + single linear
interpolation (see the "How `:grid` homes in on the threshold" analysis): it
converges to the true threshold at arbitrary precision in a handful of eigensolves
and — because it only ever *brackets* the up-crossing — it cannot overshoot the
narrow AE `γ_AE(f)` bump the way a monotone-assuming interpolation can.

It shares `marginal_factor`'s bracketing preamble (a warm `f_start` geometric
expansion for neighbor-radius continuation, else a log-spaced `nscan` scan for the
first `γ_AE(f)-γ*` up-crossing), but evaluates only the growth rate (the AD
`dγ/df` is never used by the root solver). Defaults to `ae_band=true` since the
derivative-free solvers track the AE band.

Keywords mirror [`marginal_factor`](@ref): `gamma_thresh`, `bracket`, `f_start`,
`scan_lo=1e-3`, `scan_hi=10.0`, `nscan=8`, `ae_band=true`, plus
`rootsolver=:itp` (`:itp` | `:brent` | `:bisection`), `tol=1e-6`, `maxiter=100`,
`use_gpu=false`, `verbose=false`.

Returns the same shape as [`marginal_factor`](@ref):
`(; factor, gamma_lead, gamma, freq, iters, evals, converged, bracket)`.
"""
function marginal_factor_df(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                            gamma_thresh::Union{Nothing,Float64} = nothing,
                            bracket::Union{Nothing,Tuple{Float64,Float64}} = nothing,
                            f_start::Union{Nothing,Float64} = nothing,
                            scan_lo::Float64 = 1.0e-3, scan_hi::Float64 = 10.0, nscan::Int = 8,
                            ae_band::Bool = true, rootsolver::Symbol = :itp,
                            tol::Float64 = 1.0e-6, maxiter::Int = 100,
                            use_gpu::Bool = false, verbose::Bool = false)
    gamma_thresh = gamma_thresh === nothing ? _gamma_thresh_for(inputsEP, inputsPR) : gamma_thresh
    evals = Ref(0)
    last_r = Ref{Any}(nothing)
    # γ-only evaluation (the root solver needs no derivative). We reuse the AD leaf so the
    # onset is numerically identical to `marginal_factor`/`_ae_unstable_window`; the unused
    # dγ/df is discarded (a fraction of one eigensolve), so the derivative-free win here is
    # solver ROBUSTNESS + precision, not per-eval cost.
    gl_of = f -> begin
        gl, _dgl, r = _gamma_lead_dfactor(inputsEP, inputsPR, f; use_gpu = use_gpu, ae_band = ae_band)
        evals[] += 1
        last_r[] = r
        verbose && @info "mf_df eval" factor=f gamma_lead=gl gobj=(gl - gamma_thresh)
        gl
    end
    g = f -> gl_of(f) - gamma_thresh

    # ── Establish a bracket [a,b] with g(a) < 0 < g(b) (γ-only; no derivative) ──
    local a, b, ga, gb
    a = b = NaN
    if bracket === nothing && f_start !== nothing && scan_lo < f_start < scan_hi
        # Warm continuation: geometric expansion around the previous radius's root.
        gs = g(f_start)
        if gs >= 0
            b, gb = f_start, gs
            fa = f_start
            for _ in 1:8
                fa = max(scan_lo, fa / 1.5)
                gfa = g(fa)
                if gfa < 0; a, ga = fa, gfa; break end
                fa <= scan_lo && break
            end
        else
            a, ga = f_start, gs
            fb = f_start
            for _ in 1:8
                fb = min(scan_hi, fb * 1.5)
                gfb = g(fb)
                if gfb >= 0; b, gb = fb, gfb; break end
                fb >= scan_hi && break
            end
        end
    end
    if isnan(a) || isnan(b)
        a = b = NaN
        if bracket === nothing
            # Log-spaced scan for the FIRST up-crossing of γ_AE(f)=γ*. No monotone
            # top-of-range early-out: the AE-band γ_AE(f) is a narrow bump (rises then
            # restabilises), so a sub-threshold scan_hi does NOT imply stability.
            fs = exp.(range(log(scan_lo), log(scan_hi); length = nscan))
            prev_f = fs[1]; prev_g = g(prev_f)
            for f in fs[2:end]
                gf = g(f)
                if prev_g < 0 && gf >= 0
                    a, ga, b, gb = prev_f, prev_g, f, gf
                    break
                end
                prev_f, prev_g = f, gf
            end
            if isnan(a)
                verbose && @warn "marginal_factor_df: no sign change of γ_AE-thresh in [$scan_lo,$scan_hi] (always stable/unstable); returning best endpoint"
                r = last_r[]
                return (; factor = prev_g < 0 ? scan_hi : scan_lo, gamma_lead = prev_g + gamma_thresh,
                        gamma = r === nothing ? Float64[] : r.gamma, freq = r === nothing ? Float64[] : r.freq,
                        iters = 0, evals = evals[], converged = false, bracket = (scan_lo, scan_hi))
            end
        else
            a, b = bracket
            ga = g(a); gb = g(b)
            ga < 0 < gb || error("marginal_factor_df: provided bracket ($a,$b) does not satisfy g(a)<0<g(b): g(a)=$ga g(b)=$gb")
        end
    end

    # ── derivative-free bracketing root solve on [a,b] ──
    alg = rootsolver === :brent     ? SimpleNonlinearSolve.Brent() :
          rootsolver === :bisection ? SimpleNonlinearSolve.Bisection() :
                                      SimpleNonlinearSolve.ITP()
    prob = SimpleNonlinearSolve.IntervalNonlinearProblem((u, _p) -> g(u), (a, b))
    sol = SimpleNonlinearSolve.solve(prob, alg; abstol = tol * (1 + abs(gamma_thresh)), maxiters = maxiter)
    f = sol.u
    gl = gl_of(f)   # refresh last_r at the root for the returned spectrum
    r = last_r[]
    converged = SimpleNonlinearSolve.SciMLBase.successful_retcode(sol)
    return (; factor = f, gamma_lead = gl, gamma = r === nothing ? Float64[] : r.gamma,
            freq = r === nothing ? Float64[] : r.freq,
            iters = 0, evals = evals[], converged = converged, bracket = (a, b))
end

"""
    critical_factor_grid(inputsEP, inputsPR; kwargs...) -> NamedTuple

AD analogue of the inner `kwscale_scan`: the critical EP scale factor is the
*minimum over the `(kyhat, width)` grid* of the per-point marginal factor. For
each grid point this calls `marginal_factor` (AD-Newton) instead of the
brute-force 8-factor × 4-round bracket, so it reaches `sfmin` in far fewer TGLF
eigensolves.

The `(kyhat, width)` grid mirrors `kwscale_scan`'s first round:
  - `kyhat[i] = (1/nkyhat)·i`,  i=1..nkyhat        (kyhat∈(0,1])
  - `width[i] = WIDTH_MIN + (WIDTH_MAX-WIDTH_MIN)/(nefwid-1)·(i-1)`

Keywords: `nkyhat=4, nefwid=8`, `gamma_thresh=1e-7`, `scan_lo=1e-3`,
`scan_hi=nothing` (defaults to the incoming `FACTOR_IN`), `threaded=true`,
`use_gpu=false`.

Returns `(; sfmin, kyhat, width, gamma, total_evals, results)` where `results`
is the per-grid-point table.
"""
function critical_factor_grid(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                              nkyhat::Int = 4, nefwid::Int = 8,
                              gamma_thresh::Union{Nothing,Float64} = nothing,
                              scan_lo::Float64 = 1.0e-3, scan_hi::Union{Nothing,Float64} = nothing,
                              ae_band::Bool = false,
                              threaded::Bool = true, use_gpu::Bool = false)
    # nothing ⇒ use the case's own threshold (rotational γ* or 1e-7); IR-dependent only,
    # so it is identical for every (kyhat,width) grid point.
    gamma_thresh = gamma_thresh === nothing ? _gamma_thresh_for(inputsEP, inputsPR) : gamma_thresh
    w0 = Float64(inputsEP.WIDTH_MIN); w1 = Float64(inputsEP.WIDTH_MAX)
    kyhats = [(1.0 / nkyhat) * i for i in 1:nkyhat]
    widths = [w0 + (w1 - w0) / (nefwid - 1) * (i - 1) for i in 1:nefwid]
    shi = scan_hi === nothing ? Float64(inputsEP.FACTOR_IN) : scan_hi

    pts = [(ky, w) for ky in kyhats for w in widths]
    npts = length(pts)
    out = Vector{Any}(undef, npts)

    work = function (idx)
        ky, w = pts[idx]
        ep = deepcopy(inputsEP); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
        r = marginal_factor(ep, inputsPR; gamma_thresh = gamma_thresh,
                            scan_lo = scan_lo, scan_hi = shi, ae_band = ae_band, use_gpu = use_gpu)
        out[idx] = (kyhat = ky, width = w, factor = r.factor, gamma = r.gamma,
                    converged = r.converged, evals = r.evals)
    end

    if threaded
        TJLF.with_blas_threads(1) do
            Threads.@threads for idx in 1:npts
                work(idx)
            end
        end
    else
        for idx in 1:npts
            work(idx)
        end
    end

    total_evals = sum(o.evals for o in out)
    best = (factor = Inf, kyhat = NaN, width = NaN, gamma = Float64[])
    for o in out
        if o.converged && o.factor < best.factor
            best = (factor = o.factor, kyhat = o.kyhat, width = o.width, gamma = o.gamma)
        end
    end

    return (; sfmin = best.factor, kyhat = best.kyhat, width = best.width,
            gamma = best.gamma, total_evals = total_evals, results = out)
end

# ── Faithful keep evaluation at one operating point ──────────────────────────
# Runs the FULL production single-ky path (TJLFEP_ky, IFLUX=true) so every keep
# filter is applied: band-entry (freq<FREQ_AE_UPPER) and growth (g>GAMMA_THRESH),
# plus the wavefunction/QL secondary rejections (tearing, ion/elec/thermal/EP
# pinch, QL-ratio, θ²). This is the SAME keep boolean kwscale_scan reduces to
# `imark`/`sfmin`, so it is the ground-truth target the AD root must match.
# (`inputsEP` is not mutated.)
function keep_at(inputsEP::Options{Float64}, inputsPR::profile{Float64}, factor::Float64;
                 kyhat::Float64 = Float64(inputsEP.KYHAT_IN),
                 width::Float64 = Float64(inputsEP.WIDTH_IN),
                 use_gpu::Bool = false,
                 mode_in_override::Union{Nothing,Int} = nothing)
    ep = deepcopy(inputsEP)
    ep.KYHAT_IN  = kyhat
    ep.WIDTH_IN  = width
    ep.FACTOR_IN = factor
    # Honor the MODE_IN drive selector through the faithful (IFLUX=true) single-ky path so
    # the keep filters see the correct drive; TJLFEP_ky/TJLF_map otherwise force MODE_IN=2.
    mi = _resolve_ad_mode_in(inputsEP, mode_in_override)
    g, f, = TJLFEP_ky(ep, inputsPR, "", 0; eigen_cache = nothing, use_gpu = use_gpu,
                      mode_in_override = mi)
    nm = ep.NMODES
    keep = collect(@view ep.LKEEP[1:nm])
    return (; factor, any_keep = any(keep), LKEEP = keep,
            gamma = g[1:nm], freq = f[1:nm],
            FREQ_AE_UPPER = Float64(ep.FREQ_AE_UPPER), GAMMA_THRESH = Float64(ep.GAMMA_THRESH),
            LTEARING = collect(@view ep.LTEARING[1:nm]),
            L_I_PINCH = collect(@view ep.L_I_PINCH[1:nm]),
            L_E_PINCH = collect(@view ep.L_E_PINCH[1:nm]),
            L_TH_PINCH = collect(@view ep.L_TH_PINCH[1:nm]),
            L_EP_PINCH = collect(@view ep.L_EP_PINCH[1:nm]),
            L_QL_RATIO = collect(@view ep.L_QL_RATIO[1:nm]),
            L_THETA_SQ = collect(@view ep.L_THETA_SQ[1:nm]))
end

# Which keep sub-condition rejects mode `n` at a faithful keep evaluation `kf`
# (the FIRST that fails, in TJLFEP_ky's evaluation order). Returns :kept if kept.
function _why_rejected(kf, n::Int)
    kf.LKEEP[n]                      && return :kept
    !(kf.freq[n] < kf.FREQ_AE_UPPER) && return :band          # not in AE band
    !(kf.gamma[n] > kf.GAMMA_THRESH) && return :below_thresh   # growth below γ*
    kf.LTEARING[n]                   && return :tearing
    kf.L_I_PINCH[n]                  && return :i_pinch
    kf.L_E_PINCH[n]                  && return :e_pinch
    kf.L_TH_PINCH[n]                 && return :th_pinch
    kf.L_EP_PINCH[n]                 && return :ep_pinch
    kf.L_QL_RATIO[n]                 && return :ql_ratio
    kf.L_THETA_SQ[n]                 && return :theta_sq
    return :unknown
end

# Parallel-map dispatcher for the AD path's independent-evaluation regions. With an MPS
# team (`inner=:mps_team`, non-empty `team`) it reuses kwscale_scan's chunked remotecall
# helper `_inner_team_map` (one round-trip per worker; each worker is a separate CUDA
# context so GPU solves overlap via Hyper-Q). Otherwise it runs in-process with threads
# (BLAS pinned to 1 so the dense LAPACK/CUSOLVER per-eval doesn't oversubscribe). Returns
# results in index order, `f(i)` for `i in 1:n`.
function _ad_pmap(f, n::Int; inner::Symbol = :threads,
                  team::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    if inner === :mps_team && team !== nothing && !isempty(team)
        return _inner_team_map(f, team, n)
    end
    res = Vector{Any}(undef, n)
    TJLF.with_blas_threads(1) do
        Threads.@threads for i in 1:n
            res[i] = f(i)
        end
    end
    return res
end

# AE-band unstable hull [f1,f2]: the factor range over which the leading AE-band
# (freq<FREQ_AE_UPPER) growth rate exceeds γ*, from a cheap eigenvalue-only scan
# (IFLUX=false). The keep window is a SUBSET of this (keep requires g>γ* and
# in-band), so a faithful sweep restricted to [f1,f2] is complete — and the AE
# γ(factor) is a narrow "bump", so an unbounded geometric sweep can step over the
# kept window entirely. `f1` is Newton-refined (smooth, grid-independent onset of
# instability) when the up-crossing is bracketed inside the scan; `f2` is padded
# to the next stable sample above the bump. `pinned_lo=true` flags that γ_AE>γ*
# already at scan_lo (no interior onset — the true edge is below the scan range).
# Returns `(; f1, f2, evals, unstable, pinned_lo)`.
function _ae_unstable_window(ep0::Options{Float64}, inputsPR::profile{Float64}, gth::Float64;
                             scan_lo::Float64, scan_hi::Float64, n_eig::Int,
                             threaded::Bool = true, use_gpu::Bool = false,
                             inner::Symbol = :threads,
                             team::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    fe  = exp.(range(log(scan_lo), log(scan_hi); length = n_eig))
    gae = Vector{Float64}(undef, n_eig)
    if threaded
        vals = _ad_pmap(i -> _gamma_lead_dfactor(ep0, inputsPR, fe[i]; use_gpu = use_gpu, ae_band = true)[1],
                        n_eig; inner = inner, team = team)
        for i in 1:n_eig
            gae[i] = vals[i]::Float64
        end
    else
        for i in 1:n_eig
            gae[i] = _gamma_lead_dfactor(ep0, inputsPR, fe[i]; use_gpu = use_gpu, ae_band = true)[1]
        end
    end
    unst = findall(>(gth), gae)
    isempty(unst) && return (; f1 = NaN, f2 = NaN, evals = n_eig, unstable = false, pinned_lo = false)
    i1 = first(unst); i2 = last(unst)
    evals = n_eig
    # Newton-refine the lower up-crossing of γ_AE(f)=γ* within [fe[i1-1],fe[i1]].
    f1 = fe[i1]
    pinned_lo = (i1 == 1)
    if i1 > 1
        rm = marginal_factor(ep0, inputsPR; gamma_thresh = gth, ae_band = true,
                             bracket = (fe[i1-1], fe[i1]), use_gpu = use_gpu)
        f1 = rm.factor
        evals += rm.evals
    end
    f2 = i2 < n_eig ? fe[i2+1] : fe[i2]   # pad to the next (stable) sample so the window brackets the bump
    return (; f1 = f1, f2 = f2, evals = evals, unstable = true, pinned_lo = pinned_lo)
end

"""
    marginal_factor_faithful(inputsEP, inputsPR; kwargs...) -> NamedTuple

Ground-truth marginal EP scale factor at a fixed `(IR, kyhat, width)` point that
matches the production keep definition (all secondary IFLUX=true filters), using
cheap AD eigensolves to localize the search and full evals only where needed.

Strategy:
  1. Cheap eigenvalue-only scan (`IFLUX=false`) maps the AE-band UNSTABLE hull
     `[f1,f2]` where `γ_AE(f) > γ*`, with `f1` Newton-refined. Because the keep
     window is a subset of this hull (keep ⇒ `g>γ*`), and `γ_AE(f)` is typically
     a narrow bump, restricting the expensive search to `[f1,f2]` is both
     complete and immune to stepping over the window.
  2. Faithful keep sweep (`IFLUX=true`) on a fine grid within `[f1,f2]`: the
     lowest kept sample is bisected against its unkept neighbor to give the lower
     edge of the kept window. The condition that fails just below that edge names
     the binding filter (band-entry/growth for DIII-D; ion/thermal pinch or
     QL-ratio for the rotational ITER case).

Keywords: `kyhat`, `width` (default to the struct's), `gamma_thresh=nothing`
(case γ*), `scan_lo=1e-3`, `scan_hi=10.0`, `n_eig=24` (eigenvalue scan points),
`n_fine=28` (faithful samples within the hull), `xtol=1e-4`, `maxbisect=30`,
`use_gpu=false`, `verbose=false`.

Returns `(; factor_faithful, factor_fast, window, binding, kept_modes,
evals_eig, evals_full, converged, keep)`. `factor_fast` is the cheap AD instability
onset `f1`; `window=(f1,f2)`. `binding` ∈ {`:ae_band_growth`, `:tearing`,
`:i_pinch`, `:e_pinch`, `:th_pinch`, `:ep_pinch`, `:ql_ratio`, `:theta_sq`,
`:none`}.
"""
function marginal_factor_faithful(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                                  kyhat::Float64 = Float64(inputsEP.KYHAT_IN),
                                  width::Float64 = Float64(inputsEP.WIDTH_IN),
                                  gamma_thresh::Union{Nothing,Float64} = nothing,
                                  scan_lo::Float64 = 1.0e-3, scan_hi::Float64 = 10.0,
                                  n_eig::Int = 24, n_fine::Int = 28,
                                  xtol::Float64 = 1.0e-4, maxbisect::Int = 30,
                                  threaded::Bool = true,
                                  inner::Symbol = :threads,
                                  team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                  use_gpu::Bool = false, verbose::Bool = false)
    ep0 = deepcopy(inputsEP); ep0.KYHAT_IN = kyhat; ep0.WIDTH_IN = width
    gth = gamma_thresh === nothing ? _gamma_thresh_for(ep0, inputsPR) : gamma_thresh

    # ── (1) cheap eigenvalue scan → AE-band unstable hull [f1,f2] ──
    win = _ae_unstable_window(ep0, inputsPR, gth; scan_lo = scan_lo, scan_hi = scan_hi,
                              n_eig = n_eig, threaded = threaded, use_gpu = use_gpu,
                              inner = inner, team = team)
    evals_eig = win.evals
    if !win.unstable
        verbose && @info "marginal_factor_faithful: AE-band γ never reaches γ* in [$scan_lo,$scan_hi]" gth
        return (; factor_faithful = NaN, factor_fast = NaN, window = (NaN, NaN),
                binding = :none, kept_modes = Int[], evals_eig, evals_full = 0,
                converged = false, keep = nothing, pinned_lo = false)
    end
    f1, f2 = win.f1, win.f2
    win.pinned_lo && verbose && @warn "marginal_factor_faithful: AE band already unstable at scan_lo=$scan_lo; the instability onset is below the scan range (true lower edge not bracketed)"
    verbose && @info "AE-band unstable hull" f1 f2 gth pinned_lo=win.pinned_lo

    nfull = Ref(0)
    keepf = f -> (nfull[] += 1; keep_at(ep0, inputsPR, f; kyhat = kyhat, width = width, use_gpu = use_gpu))

    # ── (2) faithful sweep within [f1,f2]: lowest kept sample + its unkept neighbor.
    #        Threaded: evaluate all n_fine samples in parallel (the keep window can sit
    #        anywhere in the hull), then locate the lowest kept index. ──
    fs = collect(range(f1, f2; length = n_fine))
    a = NaN; b = NaN; kept_b = nothing
    if threaded
        kfs = _ad_pmap(i -> keep_at(ep0, inputsPR, fs[i]; kyhat = kyhat, width = width, use_gpu = use_gpu),
                       n_fine; inner = inner, team = team)
        nfull[] += n_fine
        j = findfirst(kf -> kf.any_keep, kfs)
        if j !== nothing
            b = fs[j]; kept_b = kfs[j]
            a = j > 1 ? fs[j-1] : f1
        end
    else
        prev_f = NaN
        for (i, f) in enumerate(fs)
            kf = keepf(f)
            if kf.any_keep
                b = f; kept_b = kf
                a = i > 1 ? prev_f : f1
                break
            end
            prev_f = f
        end
    end
    if isnan(b)
        verbose && @info "marginal_factor_faithful: AE band unstable but NO mode kept in [f1,f2] (all rejected by secondary filters)"
        return (; factor_faithful = NaN, factor_fast = f1, window = (f1, f2),
                binding = :none, kept_modes = Int[], evals_eig, evals_full = nfull[],
                converged = false, keep = nothing, pinned_lo = win.pinned_lo)
    end

    # bisect [a (unkept), b (kept)] on any(LKEEP) for the lower edge
    fa, fb = a, b
    iters = 0
    while (fb - fa) > xtol && iters < maxbisect
        iters += 1
        fm = 0.5 * (fa + fb)
        kf = keepf(fm)
        if kf.any_keep
            fb = fm; kept_b = kf
        else
            fa = fm
        end
    end

    # binding condition = what fails on the unkept side just below the edge
    binding = :ae_band_growth
    ka = keepf(fa)
    for n in findall(kept_b.LKEEP)
        r = _why_rejected(ka, n)
        if r != :kept
            binding = (r == :band || r == :below_thresh) ? :ae_band_growth : r
            break
        end
    end

    return (; factor_faithful = fb, factor_fast = f1, window = (f1, f2), binding = binding,
            kept_modes = findall(kept_b.LKEEP), evals_eig, evals_full = nfull[],
            converged = true, keep = kept_b, pinned_lo = win.pinned_lo)
end

# Per-point marginal factor f★(ky,w) (AE-band growth root γ_AE=γ*) AND its exact
# gradient ∂f★/∂(ky,w) by the implicit function theorem. Differentiating
# γ_AE(f★(ky,w), ky, w) = γ* gives ∂f★/∂ky = -(∂γ/∂ky)/(∂γ/∂f) and likewise for w,
# with all three partials read from a SINGLE gamma_grad pass at (f★,ky,w). The
# binding mode is the AE-band leading mode (max γ among freq<FREQ_AE_UPPER) that
# defines the root. Returns `(; f, converged, grad, evals)`.
function _marginal_and_grad(ep0::Options{Float64}, inputsPR::profile{Float64}, ky::Float64, w::Float64;
                            gth::Float64, scan_lo::Float64, scan_hi::Float64, nscan::Int,
                            f_start::Union{Nothing,Float64} = nothing, use_gpu::Bool)
    ep = deepcopy(ep0); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
    mf = marginal_factor(ep, inputsPR; gamma_thresh = gth, ae_band = true, f_start = f_start,
                         scan_lo = scan_lo, scan_hi = scan_hi, nscan = nscan, use_gpu = use_gpu)
    !mf.converged && return (; f = Inf, converged = false, grad = (0.0, 0.0), evals = mf.evals)
    ep.FACTOR_IN = mf.factor
    fu = _ae_band_upper(ep, inputsPR)
    gg = gamma_grad(ep, inputsPR; vars = (:FACTOR_IN, :KYHAT_IN, :WIDTH_IN), use_gpu = use_gpu)
    cand = findall(<(fu), gg.freq)
    isempty(cand) && return (; f = mf.factor, converged = false, grad = (0.0, 0.0), evals = mf.evals + 1)
    j = cand[argmax(@view gg.gamma[cand])]
    dgdf, dgdky, dgdw = gg.dgamma[j, 1], gg.dgamma[j, 2], gg.dgamma[j, 3]
    if abs(dgdf) < 1e-30
        return (; f = mf.factor, converged = true, grad = (0.0, 0.0), evals = mf.evals + 1)
    end
    return (; f = mf.factor, converged = true, grad = (-dgdky / dgdf, -dgdw / dgdf), evals = mf.evals + 1)
end

_clamp_to(x, lo, hi) = min(max(x, lo), hi)

"""
    critical_factor_optimize(inputsEP, inputsPR; kwargs...) -> NamedTuple

AD analogue of the inner `kwscale_scan`: find the critical EP scale factor
`sfmin = min over (kyhat,width) of the marginal factor f★(kyhat,width)` by
**projected-gradient descent with implicit-function-theorem gradients**.

Each evaluated `(ky,w)` solves the AE-band growth root `f★` by AD-Newton
([`marginal_factor`](@ref)) and gets the exact `∂f★/∂(ky,w)` from one extra
`gamma_grad` pass (the implicit-function-theorem ratio `-(∂γ/∂x)/(∂γ/∂f)`). A
coarse `nseed_ky × nseed_w` seed grid (or a caller-supplied `seed`) locates a
feasible basin; descent then polishes it with backtracking line search,
projecting `(ky,w)` to `ky∈ky_bounds`, `w∈[WIDTH_MIN,WIDTH_MAX]`.

NOTE on cost: this yields the *continuous* optimum (more precise than any fixed
grid), but it is NOT necessarily cheaper than a `(ky,w)` grid scan — the AE-band
onset surface `f★(ky,w)` is bumpy/multimodal (the same narrow-bump physics the
keep window shows in factor), so descent can need comparably many TGLF solves and
warm seeds can land in a worse basin. The large AD win is per-evaluation
(eigenvalue-only `IFLUX=false` Newton vs the production `IFLUX=true` factor grid),
realized in [`marginal_factor`](@ref)/[`marginal_factor_faithful`](@ref), not in
collapsing the `(ky,w)` grid.

Keywords: `gamma_thresh=nothing` (case γ*), `nseed_ky=4`, `nseed_w=4`,
`scan_lo=1e-3`, `scan_hi=nothing`(→FACTOR_IN), `nscan=8`, `ky_bounds=(1e-3,1.0)`,
`w_bounds=nothing`(→struct), `maxiter=25`, `gtol=1e-4`, `xtol=1e-5`,
`step0=0.25`, `nbacktrack=12`, `faithful_confirm=false`, `use_gpu=false`,
`verbose=false`.

**`extend_width=false`** (opt-in; on by default in the `solver=:ad` scan path):
the canonical descent above only searches `w∈[WIDTH_MIN,WIDTH_MAX]` (`w≥1`), so it
structurally misses the narrow EP-driven AE modes (`w≪1`) that set the true
critical factor at near-marginal radii (the canonical `w≥1` box biases `sfmin`
high there — up to ~25× at the steep profile edge). When `extend_width=true` the
solver additionally runs [`_locate_extended`](@ref) — a cheap log-spaced `(ky,w)`
AE-onset rank (down to `w_lo` — default [`width_floor`](@ref) = 0.5, never below it — and `ky_lo=0.05`) → `:ad` descent → faithful
early-stop confirm — and folds any LOWER faithful onset into the result. This is
the SAME width-extension machinery `:robust_ad` uses, but seeded on the cheap `:ad`
descent instead of a full faithful `w≥1` grid-zoom, so it is much cheaper than
`:robust_ad` (no faithful grid) while recovering most of the narrow-width accuracy
gain over the Fortran/grid `w≥1` baseline. The `w≥1` descent is retained as a floor
(AD descent can under-resolve the dense core). Extension knobs: `w_lo=width_floor()` (0.5; lower
values are clamped up to the physics floor),
`ky_lo=0.05`, `nseed_ky_ext=6`, `nseed_w_ext=10`, `n_eig_seed=12`, `k_descend=6`.
It does NOT alter the default (`extend_width=false`) behavior, so
[`_locate_extended`](@ref)'s own inner `:ad` descents (which pass it unset)
remain `w≥1`.

`extend_mode` (default `:locate`) selects HOW the extension searches:
- `:locate` — the `_locate_extended` log-grid + `k_descend`-way multistart above
  (production; matches `:robust_ad` bitwise under `faithful_confirm`).
- `:wide` — a much cheaper single-pass scheme: widen the *main* descent box down
  to `w_lo` (and clamp the ky floor up to `ky_lo` to match `:locate`'s domain),
  seed it with a denser **log-spaced** `w` grid, descend the top `wide_kdesc`
  (default 2) well-separated seeds, drop any floor-pinned (w≈`w_lo`/ky≈`ky_lo`)
  corner optima when an interior optimum exists, and faithful-confirm the best.
  ~7× cheaper than `:locate`, recovers most of the narrow-width gain over the
  `w≥1` baseline, and stays *conservative* (confirmed keep onset ≥ truth) — it
  does NOT generally match `:robust_ad` bitwise. Good for fast NN-database
  generation. With `faithful_confirm=false` `:wide` reports the (unfiltered)
  AE-band onset and can under-predict, like the pure `:locate` path.

`extend_width` interacts with **`faithful_confirm`**:
- `faithful_confirm=true` (the `solver=:ad` scan default): the located narrow
  candidates are faithful-confirmed (`IFLUX=true`) with an early-stop bound, so
  `sfmin` is the faithful keep onset (min over the `w≥1` descent and the
  extension) and **matches `:robust_ad` bitwise** at the located mode.
- `faithful_confirm=false` ("pure AD" extended): NO faithful confirm anywhere —
  the solver folds in the lowest cheap AE-band/descended onset over the located
  candidates and reports that. Much cheaper (zero `IFLUX=true` evals), but the
  AE-band onset is ≤ the faithful keep onset, so it does **not** match
  `:robust_ad` bitwise (and can dip below it where keep filters would reject the
  marginal mode).

Returns `(; sfmin, kyhat, width, f_seedmin, ky_seed, w_seed, iters, evals,
converged, faithful, extended, n_ext_confirm)` where `faithful` is the
[`marginal_factor_faithful`](@ref) result at the winning optimum (or `nothing`),
`extended` echoes `extend_width`, and `n_ext_confirm` counts the narrow-width
faithful confirms (0 when `faithful_confirm=false`). With a faithful confirm,
`sfmin` is the faithful onset of the winner (min over the `w≥1` descent and the
narrow extension); otherwise the AE-band descent onset.
"""
function critical_factor_optimize(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                                  gamma_thresh::Union{Nothing,Float64} = nothing,
                                  seed::Union{Nothing,Tuple{Float64,Float64}} = nothing,
                                  nseed_ky::Int = 4, nseed_w::Int = 4,
                                  scan_lo::Float64 = 1.0e-3, scan_hi::Union{Nothing,Float64} = nothing,
                                  nscan::Int = 8,
                                  ky_bounds::Tuple{Float64,Float64} = (1.0e-3, 1.0),
                                  w_bounds::Union{Nothing,Tuple{Float64,Float64}} = nothing,
                                  maxiter::Int = 20, gtol::Float64 = 1.0e-4, xtol::Float64 = 1.0e-5,
                                  step0::Float64 = 0.25, nbacktrack::Int = 8,
                                  faithful_confirm::Bool = false,
                                  extend_width::Bool = false, w_lo::Float64 = width_floor(), ky_lo::Float64 = 0.05,
                                  nseed_ky_ext::Int = 6, nseed_w_ext::Int = 10,
                                  n_eig_seed::Int = 12, k_descend::Int = 6,
                                  extend_mode::Symbol = :locate, wide_kdesc::Int = 2,
                                  inner::Symbol = :threads,
                                  team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                  use_gpu::Bool = false, verbose::Bool = false)
    gth = gamma_thresh === nothing ? _gamma_thresh_for(inputsEP, inputsPR) : gamma_thresh
    shi = scan_hi === nothing ? Float64(inputsEP.FACTOR_IN) : scan_hi
    w_lo = floor_w_lo(w_lo)   # ballooning-width physics floor (WIDTH_FLOOR = 0.5)
    wlo, whi = w_bounds === nothing ? (Float64(inputsEP.WIDTH_MIN), Float64(inputsEP.WIDTH_MAX)) : w_bounds
    kylo, kyhi = ky_bounds

    # `extend_mode = :wide` (single-pass width extension): widen the descent/seed box down to `w_lo`
    # and run a small well-separated multistart of the SAME fast descent (+ faithful confirm) over the
    # whole width range, instead of the `:locate` log-grid + many-descent machinery. Much cheaper than
    # `:locate`, recovers most of the narrow-width accuracy, and stays conservative (it reports the
    # confirmed keep onset). `:locate` (default) keeps the production `:robust_ad`-matching behavior.
    wide = extend_width && extend_mode == :wide
    if wide
        w_bounds === nothing && (wlo = w_lo)
        # Match the :locate / :robust_ad descent domain: clamp the ky floor up to `ky_lo`. The default
        # `ky_bounds` reaches down to 1e-3, where the AE-band onset degenerates into a spurious ky→0
        # corner artifact that `_locate_extended` never explores (its descents are bounded ky ≥ ky_lo).
        kylo = max(kylo, ky_lo)
    end

    evals = Ref(0)
    fval = function (ky, w; f_start = nothing)
        ep = deepcopy(inputsEP); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
        mf = marginal_factor(ep, inputsPR; gamma_thresh = gth, ae_band = true, f_start = f_start,
                             scan_lo = scan_lo, scan_hi = shi, nscan = nscan, use_gpu = use_gpu)
        evals[] += mf.evals
        mf.converged ? mf.factor : Inf
    end
    fgrad = function (ky, w; f_start = nothing)
        r = _marginal_and_grad(inputsEP, inputsPR, ky, w; gth = gth, scan_lo = scan_lo,
                               scan_hi = shi, nscan = nscan, f_start = f_start, use_gpu = use_gpu)
        evals[] += r.evals
        r
    end

    # ── one projected-gradient descent (backtracking line search) from a single seed ──
    descend = function (ky0::Float64, w0::Float64, f0::Float64)
        ky, w, f = ky0, w0, f0
        iters = 0; converged = false
        for it in (isfinite(f0) ? (1:maxiter) : (1:0))
            iters = it
            r = fgrad(ky, w; f_start = isfinite(f) ? f : nothing)
            (!r.converged || !isfinite(r.f)) && break
            gky, gw = r.grad
            sky = (kyhi - kylo); sw = (whi - wlo)   # scale step by box so ky and w move comparably
            gnorm = sqrt((gky * sky)^2 + (gw * sw)^2)
            gnorm < gtol && (converged = true; break)
            α = step0; improved = false; f_cur = r.f
            for _ in 1:nbacktrack
                dky = -α * gky * sky^2 / max(gnorm, 1e-30)
                dw  = -α * gw  * sw^2  / max(gnorm, 1e-30)
                ky_t = _clamp_to(ky + dky, kylo, kyhi)
                w_t  = _clamp_to(w  + dw,  wlo, whi)
                f_pred = f_cur + gky * (ky_t - ky) + gw * (w_t - w)   # IFT first-order warm start
                f_t = fval(ky_t, w_t; f_start = isfinite(f_pred) && f_pred > 0 ? f_pred : nothing)
                if f_t < f_cur - 1e-10
                    small = abs(ky_t - ky) + abs(w_t - w) < xtol
                    ky, w, f = ky_t, w_t, f_t; improved = true
                    small && (converged = true)
                    break
                end
                α *= 0.5
            end
            verbose && @info "cfo" iter=it ky=ky w=w f=f grad=(gky, gw) gnorm=gnorm improved=improved
            improved || (converged = true; break)   # no descent direction within the box → local min
        end
        return (; ky, w, f, iters, converged)
    end

    # greedily pick up to `k` well-separated seeds (distinct basins) from a sorted-by-onset pool
    pick_starts = function (pool::Vector{Tuple{Float64,Float64,Float64}}, k::Int)
        picks = Tuple{Float64,Float64,Float64}[]
        for s in pool
            if all(p -> abs(log(s[2]) - log(p[2])) > 0.5 || abs(s[1] - p[1]) > 0.2, picks)
                push!(picks, s)
            end
            length(picks) >= k && break
        end
        isempty(picks) ? (isempty(pool) ? Tuple{Float64,Float64,Float64}[] : [pool[1]]) : picks
    end

    # ── seed: warm start from a caller-provided (ky,w) (e.g. neighbor radius), or
    #    a coarse grid (mirrors kwscale_scan's first round). The grid is the robust
    #    fallback if the warm seed is infeasible. ──
    best_f = Inf; best_ky = NaN; best_w = NaN
    seed_pool = Tuple{Float64,Float64,Float64}[]   # (ky, w, AE-band onset), ranked ascending below
    if seed !== nothing
        sky = _clamp_to(seed[1], kylo, kyhi); sw = _clamp_to(seed[2], wlo, whi)
        f = fval(sky, sw)
        if isfinite(f)
            best_f = f; best_ky = sky; best_w = sw; push!(seed_pool, (sky, sw, f))
        end
    end
    # `:wide` always sweeps the (log-w) grid so the multistart sees the narrow band, even with a warm seed.
    if !isfinite(best_f) || wide
        if wide
            # log-spaced w (spans ~decades; concentrates resolution at narrow w) + denser ky.
            kys = nseed_ky_ext <= 1 ? [0.5 * (kylo + kyhi)] :
                  [kylo + (kyhi - kylo) * (i - 1) / (nseed_ky_ext - 1) for i in 1:nseed_ky_ext]
            ws  = nseed_w_ext <= 1 ? [sqrt(wlo * whi)] :
                  [exp(log(wlo) + (log(whi) - log(wlo)) * (i - 1) / (nseed_w_ext - 1)) for i in 1:nseed_w_ext]
        else
            kys = [kylo + (kyhi - kylo) * (i - 0.5) / nseed_ky for i in 1:nseed_ky]
            ws  = nseed_w == 1 ? [0.5 * (wlo + whi)] : [wlo + (whi - wlo) * (i - 1) / (nseed_w - 1) for i in 1:nseed_w]
        end
        seedpts = [(ky, w) for ky in kys for w in ws]
        # Each seed solve is independent → saturate the team/threads. The eval-counting
        # `fval` closure mutates a coordinator-local Ref that remote workers can't update,
        # so the seed solver returns (factor, n_evals) and we reduce on the coordinator.
        seedsolve = function (idx)
            ky, w = seedpts[idx]
            ep = deepcopy(inputsEP); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
            mf = marginal_factor(ep, inputsPR; gamma_thresh = gth, ae_band = true,
                                 scan_lo = scan_lo, scan_hi = shi, nscan = nscan, use_gpu = use_gpu)
            (mf.converged ? mf.factor : Inf, mf.evals)
        end
        seedres = _ad_pmap(seedsolve, length(seedpts); inner = inner, team = team)
        for (idx, fr) in enumerate(seedres)
            f, ne = fr
            evals[] += ne
            isfinite(f) && push!(seed_pool, (seedpts[idx][1], seedpts[idx][2], f))
            if f < best_f
                best_f = f; best_ky = seedpts[idx][1]; best_w = seedpts[idx][2]
            end
        end
    end
    sort!(seed_pool, by = t -> t[3])
    f_seedmin = best_f; ky_seed = best_ky; w_seed = best_w
    # When `extend_width` is set we DON'T bail on an infeasible w≥1-box seed: a narrow
    # (w<WIDTH_MIN) mode may still be feasible and is recovered by the width extension below.
    if !isfinite(best_f) && !extend_width
        verbose && @warn "critical_factor_optimize: no feasible (instability-bearing) seed"
        return (; sfmin = Inf, kyhat = NaN, width = NaN, f_seedmin, ky_seed, w_seed,
                iters = 0, evals = evals[], converged = false, faithful = nothing,
                extended = false, n_ext_confirm = 0)
    end

    # ── local descent(s): one from the best seed, or a small well-separated multistart for `:wide`
    #    (descend the top-`wide_kdesc` distinct seeds so different narrow/core basins are explored). ──
    optima = NamedTuple[]
    if isfinite(best_f)
        starts = wide ? pick_starts(seed_pool, max(1, wide_kdesc)) : [(best_ky, best_w, best_f)]
        for s in starts
            push!(optima, descend(s[1], s[2], s[3]))
        end
        sort!(optima, by = o -> o.f)
        # Floor-pin guard (:wide): a descent that terminates exactly on the search-box floor
        # (w ≈ w_lo or ky ≈ ky_lo) is the spurious narrow/long-wavelength corner mode that
        # `:robust_ad` rejects in favor of the interior optimum. Drop floor-pinned optima, but ONLY
        # when a non-pinned feasible optimum exists (so a genuinely floor-bound mode still reports).
        if wide
            fpin = o -> isfinite(o.f) && (o.w <= wlo * (1 + 1e-3) || o.ky <= kylo * (1 + 1e-3))
            any(o -> isfinite(o.f) && !fpin(o), optima) && (optima = filter(!fpin, optima))
        end
    else
        push!(optima, (; ky = best_ky, w = best_w, f = best_f, iters = 0, converged = false))
    end
    ky = optima[1].ky; w = optima[1].w; f = optima[1].f
    iters = sum(o -> o.iters, optima); converged = optima[1].converged

    # ── canonical (w≥1) faithful confirm of the descent optimum ──
    # With `faithful_confirm` the running best below is carried in FAITHFUL (keep-onset) terms,
    # matching the candidate confirms and giving a valid early-stop bound. WITHOUT it the solver
    # stays on the cheap AE-band onset surface end-to-end (the "pure AD" extended path: no
    # `IFLUX=true` confirm anywhere — far cheaper, but it reports the AE-band onset, which is ≤ the
    # faithful keep onset and so does NOT match `:robust_ad` bitwise).
    faithful = nothing
    if faithful_confirm && isfinite(f)
        faithful = marginal_factor_faithful(inputsEP, inputsPR; kyhat = ky, width = w,
                                            scan_lo = scan_lo, scan_hi = shi, use_gpu = use_gpu,
                                            inner = inner, team = team)
    end

    # Running best (FAITHFUL terms when confirming; AE-band descent onset otherwise).
    win_f = (faithful !== nothing && faithful.binding != :none && isfinite(faithful.factor_faithful)) ?
            faithful.factor_faithful : f
    win_ky = ky; win_w = w; win_faithful = faithful
    n_ext_confirm = 0
    # `converged` must reflect whichever stage produced the WINNER, not just the initial canonical
    # (w≥1) descent. For a narrow-width EP-driven AE mode the w≥1 box is infeasible (best_f = Inf,
    # descent converged = false) and the real onset is supplied entirely by the width extension below;
    # reporting the w≥1 descent's flag then spuriously shows converged = false at every such radius.
    # Seed from the w≥1 descent/confirm and let the extension flip it true when it wins.
    win_converged = (faithful !== nothing && faithful.binding != :none && isfinite(faithful.factor_faithful)) ?
                    true : (isfinite(f) ? converged : false)

    # ── width extension: cheap log-spaced (ky,w) AE-onset rank → :ad descent → (optional) faithful
    # confirm. This is the SAME `_locate_extended` machinery `:robust_ad` uses, but seeded on the
    # cheap :ad descent above instead of a faithful w≥1 grid-zoom (so it is far cheaper than
    # `:robust_ad` — no full faithful grid). It captures the narrow EP-driven AE modes (w≪WIDTH_MIN)
    # the canonical w≥1 :ad descent structurally misses. The w≥1 descent is kept as a floor (the AD
    # descent can under-resolve the dense core min). ──
    if extend_width && wide
        # ── `:wide` extension: the multistart descents above already swept the narrow band. Fold the
        # remaining well-separated optima (sorted ascending by AE-band onset) into the running best.
        if faithful_confirm
            # Same early-stop bound as the `:locate` branch: prune candidates whose cheap onset is ≥
            # the incumbent `win_f`, or ≥ the scan ceiling `shi` (= FACTOR_IN) when the w≥1 box was
            # infeasible (win_f = Inf). The cheap onset lower-bounds the faithful onset and `optima`
            # is sorted ascending, so this is an exact prune that bounds the confirm count.
            for o in optima[2:end]
                stop_bound = isfinite(win_f) ? win_f : shi
                o.f >= stop_bound && break
                isfinite(o.f) || continue
                r = marginal_factor_faithful(inputsEP, inputsPR; kyhat = o.ky, width = o.w,
                        gamma_thresh = gth, scan_lo = scan_lo, scan_hi = shi, threaded = true,
                        inner = inner, team = team, use_gpu = use_gpu)
                n_ext_confirm += 1; evals[] += r.evals_full + r.evals_eig
                if r.binding != :none && isfinite(r.factor_faithful) &&
                   (!isfinite(win_f) || r.factor_faithful < win_f)
                    win_f = r.factor_faithful; win_ky = o.ky; win_w = o.w; win_faithful = r
                    win_converged = true
                end
            end
        else
            # pure `:wide` (no confirm): lowest descended AE-band onset (does NOT match :robust_ad).
            for o in optima
                if isfinite(o.f) && (!isfinite(win_f) || o.f < win_f)
                    win_f = o.f; win_ky = o.ky; win_w = o.w; win_faithful = nothing
                    win_converged = o.converged
                end
            end
        end
        verbose && @info "cfo wide-extended" ky=win_ky w=win_w sfmin=win_f n_ext_confirm confirm=faithful_confirm
    elseif extend_width
        loc = _locate_extended(inputsEP, inputsPR; gth = gth, slo = scan_lo, shi = shi, ky_lo = ky_lo,
                w_lo = w_lo, w_hi = whi, nseed_ky = nseed_ky_ext, nseed_w = nseed_w_ext,
                n_eig_seed = n_eig_seed, k_descend = k_descend, inner = inner, team = team,
                use_gpu = use_gpu)
        evals[] += loc.total_eig
        if faithful_confirm
            # faithful early-stop confirm: confirm candidates in increasing cheap order, stopping as
            # soon as the next cheap onset ≥ the pruning bound (the cheap edge lower-bounds faithful).
            # The bound is the running faithful incumbent `win_f`; when the canonical w≥1 box was
            # infeasible (win_f = Inf — e.g. ExB/rotational suppression raised γ* so no w≥1 mode clears
            # threshold), fall back to the scan ceiling `shi` (= FACTOR_IN). A candidate whose cheap
            # AE-band onset already exceeds FACTOR_IN cannot have a faithful onset below it (faithful ≥
            # cheap), so it can never be the binding critical factor and is pruned exactly — this keeps
            # the extension from faithful-confirming EVERY narrow candidate when the w≥1 box is
            # infeasible (the ExB-on slowdown). `win_f` itself stays Inf so `sfmin` is unaffected.
            for c in loc.cands
                stop_bound = isfinite(win_f) ? win_f : shi
                c[3] >= stop_bound && break
                r = marginal_factor_faithful(inputsEP, inputsPR; kyhat = c[1], width = c[2],
                        gamma_thresh = gth, scan_lo = scan_lo, scan_hi = shi, threaded = true,
                        inner = inner, team = team, use_gpu = use_gpu)
                n_ext_confirm += 1; evals[] += r.evals_full + r.evals_eig
                if r.binding != :none && isfinite(r.factor_faithful) &&
                   (!isfinite(win_f) || r.factor_faithful < win_f)
                    win_f = r.factor_faithful; win_ky = c[1]; win_w = c[2]; win_faithful = r
                    win_converged = true
                end
            end
        else
            # pure AD: fold in the lowest AE-band/descended onset over the located candidates (cands
            # are sorted ascending by onset), with NO faithful confirm anywhere.
            for c in loc.cands
                if isfinite(c[3]) && (!isfinite(win_f) || c[3] < win_f)
                    win_f = c[3]; win_ky = c[1]; win_w = c[2]; win_faithful = nothing
                    win_converged = true
                end
            end
        end
        verbose && @info "cfo width-extended" ky=win_ky w=win_w sfmin=win_f n_ext_confirm confirm=faithful_confirm
    end

    sfmin_out = isfinite(win_f) ? win_f : f
    return (; sfmin = sfmin_out, kyhat = win_ky, width = win_w, f_seedmin, ky_seed, w_seed,
            iters = iters, evals = evals[], converged = win_converged, faithful = win_faithful,
            extended = extend_width, n_ext_confirm = n_ext_confirm)
end

"""
    critical_factor_robust(inputsEP, inputsPR; kwargs...) -> NamedTuple

Robust, grid-faithful critical EP scale factor: the **global minimum over a
`(kyhat, width)` grid of the FAITHFUL (all-filter) marginal factor**, mirroring
exactly what the Fortran/`grid` `kwscale_scan` reduces to (`sfmin = min over
(kyhat,width) of the kept-window lower edge`). Unlike [`critical_factor_optimize`](@ref),
which gradient-descends the *bumpy/multimodal* AE-band onset surface `f★(ky,w)`
and can land in a shallower local basin (or on a mode the keep filters later
reject), this evaluates [`marginal_factor_faithful`](@ref) at **every** grid point
and takes the global min — so it reproduces the grid `sfmin` to within the
continuous-vs-discrete-factor difference, at the cost of scanning the full
`(ky,w)` grid (no `(ky,w)` collapse). The inner *factor* root is still the cheap
AD-Newton + faithful-bisect (`IFLUX=false` hull localization, `IFLUX=true` only
inside the AE band), so each grid point is far cheaper than the production
8-factor × 4-round brute force.

The coarse `(kyhat, width)` grid mirrors `kwscale_scan`'s first round:
  - `kyhat[i] = (1/nkyhat)·i`, i=1..nkyhat
  - `width[i] = WIDTH_MIN + (WIDTH_MAX-WIDTH_MIN)/(nefwid-1)·(i-1)`

`refine_rounds` then re-grids a shrinking `(ky,w)` box around the running best,
mirroring the Fortran window narrowing so binding points that lie *between* coarse
nodes (e.g. the plasma edge) are resolved rather than overestimated. With
`adaptive=true` (default), `refine_rounds` is a per-radius **budget**: a round runs
only when an a-priori trigger fires — the best is on a **box edge** (minimum
unbracketed), the feasible region is **sparse** (`≤2` or `< feasible_frac` of nodes
carried a kept mode), or the best onset is **near the cap** (`≥ cap_frac·scan_hi`) —
and stops early once a round's relative improvement drops below `improve_tol` and the
best is interior. So radii whose coarse min is already interior/dense/low spend **no**
refinement, while edge/sparse radii get up to `refine_rounds` zoom rounds. With
`adaptive=false`, exactly `refine_rounds` rounds always run.

`outer` selects how the off-node minimum is resolved once the coarse grid has bracketed
the basin:
  - `:grid_zoom` (default) — re-grid a shrinking `(ky,w)` box (the `refine_rounds` loop above);
  - `:hybrid` — when the zoom trigger is QUIET (best interior, feasibility dense, onset below
    cap), replace the grid-zoom round with a LOCAL AD-IFT polish ([`critical_factor_optimize`](@ref)
    seeded at the coarse best: projected-gradient on the cheap IFLUX=false AE-onset surface using
    exact `∂f★/∂(ky,w)`) followed by a FAITHFUL confirm of the optimum. The coarse grid still does
    the (multimodality-safe) global bracketing; AD only does the local descent it is good at. When
    the trigger FIRES (untrustworthy bracket), `:hybrid` falls back to the grid-zoom loop. This aims
    to recover the off-node accuracy of a refine round at a handful of solves instead of a full grid.

`extend_width` (default `true`) extends the search BELOW the canonical box to capture the
radially **narrow** EP-driven AE modes (`width ≪ WIDTH_MIN`) that `w∈[WIDTH_MIN,WIDTH_MAX]`
structurally misses. After the canonical grid-zoom (which is kept as a robust FLOOR), it runs
[`_locate_extended`](@ref) over a log-spaced width box down to `w_lo` — cheap `IFLUX=false`
AE-onset ranking → `:ad` (AD-IFT) descent of the top seeds → FAITHFUL confirm — and folds any
lower faithful onset into the global best. So `sfmin = min(canonical w≥1 grid-zoom, extended
narrow-width locate)`. This is the **width component** of the `grid→truth` gap and, on the
DIII-D edge, drops `sfmin` by 2–11× vs the `w≥1`-only value. Set `extend_width=false` to recover
the pure faithful `w≥1` minimum (≈ the `:grid` reference). The `nbasis`-converged tier is
[`critical_factor_truth`](@ref), which is this solver plus a `nbasis` ladder at the located optimum.

`confirm_grid` (default `false`) is an OPT-IN cheap-rank→few-confirm scheme for the `w≥1` grid
passes: rank every `(ky,w)` node on the cheap `IFLUX=false` AE-band onset and faithful-confirm only
the few that can still hold the minimum (the [`critical_factor_confirm`](@ref) logic, shared via
`_rank_confirm`). It is EXACT for a fixed node set (verified bitwise with `adaptive=false`) and cuts
`IFLUX=true` evals, but is kept OFF by default because on the GPU/MPS production route it is measurably
SLOWER (the early-stop confirm is serial — one GPU — while the brute path fans all nodes across the MPS
team; the keep filter is not the per-node bottleneck the eigensolve is) and, with `adaptive=true`, its
cheap (over-counted) feasibility count shifts the `sparse` zoom trigger so `sfmin` can differ from the
brute path at near-degenerate radii. Prefer it for CPU/serial runs or with `adaptive=false`.

Keywords: `nkyhat=4, nefwid=8`, `refine_rounds=1`, `adaptive=true`, `confirm_grid=false`, `outer=:grid_zoom`,
`feasible_frac=0.35`, `cap_frac=0.5`, `improve_tol=0.02`, `extend_width=true`, `ky_lo=0.05`,
`w_lo=width_floor()` (0.5, the physics floor; lower values are clamped), `nseed_ky=6`, `nseed_w=10`,
`n_eig_seed=12`, `k_descend=6`, `gamma_thresh=nothing`
(case γ*), `scan_lo=nothing`(→`scan_hi/(nfactor·4^(k_max-1))`, the grid floor; `scan_hi/512` at k_max=4),
`scan_hi=nothing`(→`FACTOR_IN`), `inner=:threads`, `team=nothing`, `use_gpu=false`,
`verbose=false`. The `(ky,w)` points are parallelized over the team/threads; each
point's inner faithful sweep runs serially to avoid nested oversubscription.

Returns `(; sfmin, sfmin_w1, kyhat, width, binding, status, scan_lo, scan_hi, refine_done,
polished, extended, n_ext_confirm, n_feasible_coarse, npts_coarse, total_evals_full,
total_evals_eig, npts, results)`, where `status ∈ (:ok, :no_onset, :cap)`, `sfmin_w1` is the
canonical `w≥1` grid-zoom minimum *before* the width extension (the well-converged Fortran-faithful
threshold, used by [`critical_factor_truth`](@ref) as a floor), `refine_done` is the number of
grid-zoom rounds performed, `polished` is `true` when the `:hybrid` AD-IFT polish ran, `extended`
reflects `extend_width`, and `n_ext_confirm` is the number of faithful confirms spent on the
width extension.
"""
function critical_factor_robust(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                                       nkyhat::Int = 4, nefwid::Int = 8,
                                       k_max::Int = _k_max_env(),
                                       refine_rounds::Int = 1,
                                       adaptive::Bool = true,
                                       confirm_grid::Bool = false,
                                       outer::Symbol = :grid_zoom,
                                       feasible_frac::Float64 = 0.35, cap_frac::Float64 = 0.5,
                                       improve_tol::Float64 = 0.02,
                                       extend_width::Bool = true,
                                       ky_lo::Float64 = 0.05, w_lo::Float64 = width_floor(),
                                       nseed_ky::Int = 6, nseed_w::Int = 10,
                                       n_eig_seed::Int = 12, k_descend::Int = 6,
                                       gamma_thresh::Union{Nothing,Float64} = nothing,
                                       scan_lo::Union{Nothing,Float64} = nothing, scan_hi::Union{Nothing,Float64} = nothing,
                                       inner::Symbol = :threads,
                                       team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                       use_gpu::Bool = false, verbose::Bool = false)
    outer in (:grid_zoom, :hybrid) ||
        error("critical_factor_robust: outer must be :grid_zoom or :hybrid (got $outer)")
    gth = gamma_thresh === nothing ? _gamma_thresh_for(inputsEP, inputsPR) : gamma_thresh
    shi = scan_hi === nothing ? Float64(inputsEP.FACTOR_IN) : scan_hi
    # Default scan_lo to the GRID's refinement floor so AD floors the same way the Fortran
    # kwscale_scan does: its k_max=4 rounds of nfactor=8-point bracketing (f1←2·fmark, f0←0)
    # bottom out at FACTOR_IN/(nfactor·4^(k_max-1)) = FACTOR_IN/512. A (ky,w) that is unstable
    # to arbitrarily small factor is reported by the grid at that floor (a discretization
    # artifact), so using the same floor keeps AD comparable instead of resolving below it.
    slo = scan_lo === nothing ? _ad_factor_floor(shi, k_max) : scan_lo
    wlo = Float64(inputsEP.WIDTH_MIN); whi = Float64(inputsEP.WIDTH_MAX)
    kylo = 0.0; kyhi = 1.0
    w_lo = floor_w_lo(w_lo)   # ballooning-width physics floor (WIDTH_FLOOR = 0.5)

    total_full = 0; total_eig = 0; npts_total = 0
    # Phase-level eigensolve breakdown (diagnostic: locate where the high-γ* eig cost lives).
    eig_coarse = 0; eig_zoom = 0; eig_loc_cheap = 0; eig_loc_desc = 0; eig_ext_confirm = 0
    all_res = Any[]
    best_f = Inf; best_ky = NaN; best_w = NaN; best_bind = :none

    # Evaluate the faithful onset on a (kyhat, width) grid spanning [kya,kyb]×[wa,wb], fold the
    # global min into the running best, and return per-pass stats used by the adaptive zoom
    # trigger. Parallelize the points over the team/threads; each point's inner faithful sweep
    # runs serially (`threaded=false`) to avoid nesting Threads.@threads / re-spawning the team.
    function eval_grid!(kya, kyb, wa, wb)
        kyhats = nkyhat == 1 ? [0.5 * (kya + kyb)] :
                 (kya <= 0.0 ? [(kyb / nkyhat) * i for i in 1:nkyhat] :
                  [kya + (kyb - kya) / (nkyhat - 1) * (i - 1) for i in 1:nkyhat])
        widths = nefwid == 1 ? [0.5 * (wa + wb)] : [wa + (wb - wa) / (nefwid - 1) * (i - 1) for i in 1:nefwid]
        pts = [(ky, w) for ky in kyhats for w in widths]
        np = length(pts)
        if confirm_grid
            # OPT-IN cheap-rank → few-confirm (confirm_grid=true; NOT the default — see below): rank all
            # nodes on the cheap AE-band edge and faithful-confirm only those that can still beat the
            # current GLOBAL incumbent `best_f`. EXACT for a FIXED node set (faithful ≥ cheap node-wise;
            # §5) — verified bitwise vs the brute path with `adaptive=false`. Two empirically-measured
            # caveats keep this OFF by default on the GPU/MPS production route:
            #   • SPEED: the early-stop confirm loop is inherently SERIAL (each confirm depends on the
            #     running incumbent), so on a multi-GPU MPS team it confirms on ONE GPU while the brute
            #     path fans all nkyhat·nefwid faithful solves across the whole team. Plus the cheap rank
            #     adds a full extra eigen-scan pass. Net on the 20-radius DIII-D 1-node sweep: ~7–18%
            #     SLOWER at nb=6..32 (the IFLUX=true keep filter is NOT the per-node bottleneck the GPU
            #     eigensolve is). It can still win on CPU/serial runs where the keep filter dominates.
            #   • ADAPTIVE PARITY: `nfeasible` below is the CHEAP AE-unstable count (over-counts vs
            #     faithful feasibility), fed to the `sparse` zoom trigger (§4.3). With `adaptive=true`
            #     this shifts which refine boxes are explored, so sfmin can differ from the brute path at
            #     near-degenerate radii (observed ~3–10% at a couple of nb<32 DIII-D radii). Use with
            #     `adaptive=false` for exact parity.
            rc = _rank_confirm(inputsEP, inputsPR, pts; gth = gth, slo = slo, shi = shi,
                    incumbent_f = best_f, n_eig = 24, inner = inner, team = team,
                    use_gpu = use_gpu, verbose = verbose)
            total_full += rc.total_evals_full; total_eig += rc.total_evals_eig
            append!(all_res, rc.confirmed)
            if rc.best_bind !== :none && isfinite(rc.best_f) && rc.best_f < best_f
                best_f = rc.best_f; best_ky = rc.best_ky; best_w = rc.best_w; best_bind = rc.best_bind
            end
            nfeasible = rc.n_cheap_feasible
            verbose && @info "confirm grid pass" np n_confirm=rc.n_confirm n_cheap_feasible=nfeasible sfmin=best_f
        else
            # DEFAULT brute-faithful path: confirm EVERY node faithfully, fanned across the team by
            # `_ad_pmap`. `nfeasible` is the confirmed-binding (faithful-feasible) count used by the
            # `sparse` zoom trigger. This is the production GPU/MPS route (fastest here; see above).
            res = _ad_pmap(idx -> begin
                    ky, w = pts[idx]
                    marginal_factor_faithful(inputsEP, inputsPR; kyhat = ky, width = w,
                                             gamma_thresh = gth, scan_lo = slo, scan_hi = shi,
                                             threaded = false, use_gpu = use_gpu)
                end, np; inner = inner, team = team)
            nfeasible = 0
            for (idx, r) in enumerate(res)
                total_full += r.evals_full; total_eig += r.evals_eig
                push!(all_res, r)
                if r.binding != :none && isfinite(r.factor_faithful)
                    nfeasible += 1
                    if r.factor_faithful < best_f
                        best_f = r.factor_faithful; best_ky = pts[idx][1]; best_w = pts[idx][2]; best_bind = r.binding
                    end
                end
                verbose && @info "faithful grid pt" ky=pts[idx][1] w=pts[idx][2] factor=r.factor_faithful binding=r.binding
            end
        end
        npts_total += np
        return (; nfeasible, np, kymin = minimum(kyhats), kymax = maximum(kyhats),
                wmin = minimum(widths), wmax = maximum(widths))
    end

    # A-priori zoom trigger: decide whether re-gridding around the current best is likely to
    # move the minimum, from cheap byproducts of the just-finished pass. Zoom when ANY of:
    #   • boundary — best (ky,w) sits at an edge of the searched box, so the true minimum is
    #     unbracketed / lies outside this box (the grid path keeps narrowing toward it);
    #   • sparse   — few nodes carried an unstable, kept mode (≤2 or < feasible_frac of the
    #     grid), so the feasible region is barely sampled and the off-node min is likely missed
    #     (this is the plasma-edge failure mode);
    #   • near-cap — the best onset is a large fraction of the search ceiling, a sign no
    #     well-resolved interior onset was found.
    # When none fire (interior best, densely feasible, low onset) the coarse min is already
    # trustworthy and refinement is skipped — so the ~per-round cost is spent only where needed.
    zoom_decision(bky, bw, st) = begin
        boundary = (bky <= st.kymin + 1e-12) || (bky >= st.kymax - 1e-12) ||
                   (bw  <= st.wmin  + 1e-12) || (bw  >= st.wmax  - 1e-12)
        sparse   = st.np > 0 && (st.nfeasible <= 2 || st.nfeasible < feasible_frac * st.np)
        nearcap  = isfinite(best_f) && best_f >= cap_frac * shi
        (; zoom = boundary || sparse || nearcap, boundary, sparse, nearcap)
    end

    # Coarse pass mirrors kwscale_scan's first round: kyhat∈(0,1], width∈[WIDTH_MIN,WIDTH_MAX].
    _e0 = total_eig
    st = eval_grid!(kylo, kyhi, wlo, whi)
    eig_coarse += total_eig - _e0
    n_feasible_coarse = st.nfeasible

    # ── Hybrid outer (outer=:hybrid) ─────────────────────────────────────────────
    # The coarse grid is a robust GLOBAL bracket (it enumerates every basin and picks the
    # winning (ky,w) node — the step local methods can't do safely on the multimodal f★).
    # When the zoom trigger is QUIET (best is interior, feasibility dense, onset well below
    # cap), the winning basin is bracketed, so pinpointing the off-node minimum is a LOCAL,
    # ~unimodal problem — exactly what AD-IFT descent is good at. Polish it with
    # `critical_factor_optimize` SEEDED at the coarse best (projected-gradient on the cheap
    # IFLUX=false AE-onset surface using exact ∂f★/∂(ky,w)=-(∂γ/∂x)/(∂γ/∂f)), then FAITHFUL-
    # confirm the optimum (keep filters can flip the binding off-node — gradient proposes,
    # faithful disposes). This replaces a full grid-zoom round with a handful of solves.
    # When the trigger FIRES (untrustworthy bracket: edge/sparse/near-cap), the basin is not
    # bracketed, so skip the polish and fall back to the grid-zoom loop below.
    polished = false
    if outer === :hybrid && best_bind !== :none && isfinite(best_f) &&
       !zoom_decision(best_ky, best_w, st).zoom
        opt = critical_factor_optimize(inputsEP, inputsPR; gamma_thresh = gth,
                seed = (best_ky, best_w), scan_lo = slo, scan_hi = shi,
                ky_bounds = (max(kylo, 1.0e-6), kyhi), w_bounds = (wlo, whi),
                faithful_confirm = true, inner = inner, team = team, use_gpu = use_gpu)
        total_eig += opt.evals
        if opt.faithful !== nothing
            total_full += opt.faithful.evals_full
            total_eig  += opt.faithful.evals_eig
            if opt.faithful.binding !== :none && isfinite(opt.faithful.factor_faithful) &&
               opt.faithful.factor_faithful < best_f
                best_f  = opt.faithful.factor_faithful
                best_ky = opt.kyhat; best_w = opt.width; best_bind = opt.faithful.binding
            end
        end
        polished = true
        verbose && @info "hybrid polish" ky=best_ky w=best_w sfmin=best_f binding=best_bind
    end

    # Refinement rounds mirror the Fortran (ky,w) window narrowing: re-grid a shrinking box
    # around the current best (ky,w) so the binding point can be resolved off the coarse nodes.
    # With `adaptive`, `refine_rounds` is a per-radius *budget*: each round runs only while the
    # zoom trigger fires, and stops early once a round barely improves the min (plateau) and the
    # best is no longer on a box edge. With `adaptive=false`, exactly `refine_rounds` rounds run.
    # outer=:hybrid skips grid-zoom entirely once it has polished a bracketed basin (budget 0).
    dky = nkyhat > 1 ? (kyhi - kylo) / (nkyhat - 1) : (kyhi - kylo)
    dw  = nefwid > 1 ? (whi - wlo) / (nefwid - 1) : (whi - wlo)
    refine_done = 0
    budget = polished ? 0 : max(0, refine_rounds)
    while refine_done < budget
        (best_bind === :none || !isfinite(best_f)) && break   # nothing to refine around
        if adaptive
            dec = zoom_decision(best_ky, best_w, st)
            dec.zoom || break                                 # coarse/current min already trustworthy
        end
        prev = best_f
        kya = max(kylo, best_ky - dky); kyb = min(kyhi, best_ky + dky)
        wa  = max(wlo,  best_w  - dw);  wb  = min(whi,  best_w  + dw)
        _ez = total_eig
        st = eval_grid!(kya, kyb, wa, wb)
        eig_zoom += total_eig - _ez
        refine_done += 1
        dky *= 2.0 / max(nkyhat - 1, 1)   # window already spans ±dky; tighten node spacing
        dw  *= 2.0 / max(nefwid - 1, 1)
        if adaptive
            rel = (isfinite(prev) && prev > 0) ? (prev - best_f) / prev : 0.0
            dec = zoom_decision(best_ky, best_w, st)
            (rel < improve_tol && !dec.boundary) && break     # plateau and bracketed → done
        end
    end

    # ── Width extension (extend_width, default on) ───────────────────────────────
    # The (ky,w) grid-zoom above resolves the dense canonical-box (w∈[WIDTH_MIN,WIDTH_MAX])
    # minimum and serves as a robust FLOOR. But the genuine EP-driven AE modes are radially
    # narrow (width ≪ WIDTH_MIN) and the canonical box misses them entirely — they are the
    # whole width component of the grid→truth gap. Extend the search to a log-spaced width box
    # down to `w_lo` via the cheap AE-onset rank → AD-IFT descent → faithful-confirm locator
    # (`_locate_extended`; the cheap ranking is what makes the larger box affordable), and fold
    # any LOWER faithful onset into the running best. Net: sfmin = min(canonical w≥1 grid-zoom,
    # extended narrow-width locate). The w≥1 zoom is retained as a floor precisely because the
    # AD descent can under-resolve the dense core min, so the two are complementary.
    # Snapshot the canonical w≥1 grid-zoom minimum BEFORE the extension lowers `best_f`. This is the
    # well-converged (at nb_work) Fortran-faithful threshold; `:truth` uses it as a final floor so a
    # razor-thin narrow win at nb_work that flips ABOVE w≥1 after nbasis convergence cannot make the
    # reported critical factor exceed the true min-over-modes value.
    sfmin_w1 = best_f; ky_w1 = best_ky; w_w1 = best_w; bind_w1 = best_bind
    n_ext_confirm = 0
    if extend_width
        loc = _locate_extended(inputsEP, inputsPR; gth = gth, slo = slo, shi = shi, ky_lo = ky_lo,
                w_lo = w_lo, w_hi = whi, nseed_ky = nseed_ky, nseed_w = nseed_w,
                n_eig_seed = n_eig_seed, k_descend = k_descend, inner = inner, team = team,
                use_gpu = use_gpu)
        total_eig += loc.total_eig
        eig_loc_cheap += loc.cheap_eig; eig_loc_desc += loc.desc_eig
        # Early-stop pruning bound: candidates are sorted ascending by their cheap/descended onset,
        # which lower-bounds the faithful keep onset, so once a candidate's cheap onset reaches the
        # bound none below it can win. The bound is the canonical w≥1 incumbent `best_f`; when that
        # box was infeasible (best_f = Inf — e.g. ExB/rotational suppression raised γ* so no w≥1 mode
        # clears threshold) fall back to the scan ceiling `shi` (= FACTOR_IN). A candidate whose cheap
        # onset already exceeds FACTOR_IN cannot have a faithful onset below it, so it is pruned
        # exactly — without this the extension faithful-confirms EVERY narrow candidate (the ExB-on
        # slowdown). `best_f` itself stays Inf so `sfmin`/`status` are unaffected.
        for c in loc.cands
            stop_bound = isfinite(best_f) ? best_f : shi
            c[3] >= stop_bound && break
            r = marginal_factor_faithful(inputsEP, inputsPR; kyhat = c[1], width = c[2],
                    gamma_thresh = gth, scan_lo = slo, scan_hi = shi, threaded = true,
                    inner = inner, team = team, use_gpu = use_gpu)
            n_ext_confirm += 1; total_full += r.evals_full; total_eig += r.evals_eig
            eig_ext_confirm += r.evals_eig
            if r.binding != :none && isfinite(r.factor_faithful) && r.factor_faithful < best_f
                best_f = r.factor_faithful; best_ky = c[1]; best_w = c[2]; best_bind = r.binding
            end
        end
        verbose && @info "width-extended" ky=best_ky w=best_w sfmin=best_f binding=best_bind n_ext_confirm
    end

    # status: :ok (genuine interior onset), :no_onset (no (ky,w) had a kept window in
    # [slo,shi]), or :cap (best onset sits at the search ceiling → likely no real onset
    # below FACTOR_IN). Callers can fall back to the grid solver on :no_onset/:cap.
    status = (best_bind === :none || !isfinite(best_f)) ? :no_onset :
             (best_f >= 0.999 * shi ? :cap : :ok)

    return (; sfmin = best_f, sfmin_w1 = sfmin_w1, ky_w1 = ky_w1, w_w1 = w_w1, bind_w1 = bind_w1,
            kyhat = best_ky, width = best_w, binding = best_bind,
            status = status, scan_lo = slo, scan_hi = shi, refine_done = refine_done,
            polished = polished, extended = extend_width, n_ext_confirm = n_ext_confirm,
            n_feasible_coarse = n_feasible_coarse, npts_coarse = nkyhat * nefwid,
            total_evals_full = total_full, total_evals_eig = total_eig, npts = npts_total,
            eig_coarse = eig_coarse, eig_zoom = eig_zoom, eig_loc_cheap = eig_loc_cheap,
            eig_loc_desc = eig_loc_desc, eig_ext_confirm = eig_ext_confirm, results = all_res)
end

# Cheap-rank → few-confirm over a FIXED (ky,w) point list. Ranks every node on the CHEAP
# eigenvalue-only AE-band lower edge (`_ae_unstable_window`, IFLUX=false), then faithful-confirms
# (`marginal_factor_faithful`, IFLUX=true) in increasing cheap order, early-stopping as soon as the
# next cheap onset ≥ the best faithful found. EXACT: the faithful keep onset f★ at any node is ≥ its
# cheap AE-band edge f1 (keep ⊆ AE-unstable hull), so no skipped node can hold a smaller f★ than the
# incumbent. Returns the updated incumbent + stats; shared by `critical_factor_confirm` and
# `critical_factor_robust`'s `eval_grid!`. `incumbent_f` lets a caller seed the early-stop bound with
# an already-found global best so even fewer nodes get the expensive confirm.
function _rank_confirm(ep0::Options{Float64}, prof::profile{Float64}, pts::Vector{Tuple{Float64,Float64}};
                       gth::Float64, slo::Float64, shi::Float64, incumbent_f::Float64,
                       n_eig::Int = 24, inner::Symbol = :threads,
                       team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                       use_gpu::Bool = false, verbose::Bool = false)
    # (1) cheap AE-band lower edge f1 at every node (IFLUX=false), parallel over team/threads
    cheap = _ad_pmap(idx -> begin
            ky, w = pts[idx]; ep = deepcopy(ep0); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
            win = _ae_unstable_window(ep, prof, gth; scan_lo = slo, scan_hi = shi,
                      n_eig = n_eig, threaded = false, use_gpu = use_gpu)
            (; ky = ky, w = w, f = (win.unstable ? win.f1 : Inf), evals = win.evals)
        end, length(pts); inner = inner, team = team)
    total_eig = sum(c.evals for c in cheap; init = 0)
    # (2) confirm in increasing cheap order, early stop when cheap ≥ best faithful
    best_f = incumbent_f; best_ky = NaN; best_w = NaN; best_bind = :none
    total_full = 0; n_confirm = 0; n_cheap_feasible = 0
    confirmed = Any[]
    for i in sortperm([c.f for c in cheap])
        c = cheap[i]
        isfinite(c.f) || break          # remaining nodes are AE-stable (Inf) → nothing to confirm
        n_cheap_feasible += 1           # cheap AE-unstable count (heuristic; over-counts vs faithful)
        c.f >= best_f && break          # bound: no remaining node can have faithful < best_f
        # Pin BLAS to 1 thread so this serial faithful confirm is BITWISE-identical to the brute
        # `eval_grid!` path, where the same call runs inside `_ad_pmap`'s `with_blas_threads(1)`
        # block (LAPACK reduction order is BLAS-thread-count dependent → ~1e-13 drift otherwise).
        r = TJLF.with_blas_threads(1) do
            marginal_factor_faithful(ep0, prof; kyhat = c.ky, width = c.w, gamma_thresh = gth,
                    scan_lo = slo, scan_hi = shi, threaded = false, use_gpu = use_gpu)
        end
        n_confirm += 1; total_full += r.evals_full; total_eig += r.evals_eig
        push!(confirmed, (; ky = c.ky, w = c.w, cheap = c.f, faithful = r.factor_faithful, binding = r.binding))
        if r.binding != :none && isfinite(r.factor_faithful) && r.factor_faithful < best_f
            best_f = r.factor_faithful; best_ky = c.ky; best_w = c.w; best_bind = r.binding
        end
        verbose && @info "rank_confirm" ky=c.ky w=c.w cheap=c.f faithful=r.factor_faithful binding=r.binding best=best_f
    end
    cheap_min = isempty(cheap) ? NaN : minimum(c.f for c in cheap)
    return (; best_f, best_ky, best_w, best_bind, n_confirm, n_cheap_feasible, cheap_min,
            total_evals_full = total_full, total_evals_eig = total_eig, confirmed)
end

"""
    critical_factor_confirm(inputsEP, inputsPR; kwargs...) -> NamedTuple

Cheap-search + few-confirm critical EP scale factor. Scans the `(kyhat, width)` grid on the
CHEAP eigenvalue-only AE-band onset (`IFLUX=false`), then applies the expensive faithful keep
filters (`IFLUX=true`, the tearing/pinch/QL-ratio/θ² block) at ONLY the few candidate nodes
that could hold the global minimum.

This is exact, not heuristic: the faithful keep onset at any `(ky,w)` is `≥` its AE-band onset
(keep ⊆ AE-unstable, since keep requires `γ>γ*` AND in-band AND surviving the secondary
filters). So candidates are confirmed in increasing cheap-onset order and the loop stops as
soon as the next candidate's cheap onset exceeds the best *faithful* onset found — no remaining
node can beat it. The result therefore EQUALS the faithful `(ky,w)`-grid minimum (the
[`critical_factor_robust`](@ref) refine=0 answer), but pays `IFLUX=true` on only ~1–few nodes
instead of all `nkyhat·nefwid`. The saving is large when the secondary filters rarely bind
(e.g. DIII-D, where the binding is AE-band growth and the pinch flags are off); when they bind
often (rotational ITER) more nodes are confirmed, degrading gracefully to the full faithful grid.

The `(ky,w)` grid mirrors `kwscale_scan`'s first round (`kyhat[i]=(1/nkyhat)·i`,
`width[i]=WIDTH_MIN+(WIDTH_MAX-WIDTH_MIN)/(nefwid-1)·(i-1)`). Keywords: `nkyhat=4, nefwid=8`,
`gamma_thresh=nothing` (case γ*), `scan_lo=nothing`(→`scan_hi/(nfactor·4^(k_max-1))`, the grid floor; `scan_hi/512` at k_max=4),
`scan_hi=nothing`(→`FACTOR_IN`), `inner=:threads`, `team=nothing`, `use_gpu=false`,
`verbose=false`.

Returns `(; sfmin, kyhat, width, binding, status, cheap_min, n_confirm, npts,
total_evals_full, total_evals_eig, results)`, where `status ∈ (:ok, :no_onset, :cap)`,
`cheap_min` is the lowest cheap AE-band onset over the grid, and `n_confirm` is the number of
nodes that actually got the `IFLUX=true` confirm.
"""
function critical_factor_confirm(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                                 nkyhat::Int = 4, nefwid::Int = 8,
                                 k_max::Int = _k_max_env(),
                                 gamma_thresh::Union{Nothing,Float64} = nothing,
                                 scan_lo::Union{Nothing,Float64} = nothing, scan_hi::Union{Nothing,Float64} = nothing,
                                 inner::Symbol = :threads,
                                 team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                 use_gpu::Bool = false, verbose::Bool = false)
    gth = gamma_thresh === nothing ? _gamma_thresh_for(inputsEP, inputsPR) : gamma_thresh
    shi = scan_hi === nothing ? Float64(inputsEP.FACTOR_IN) : scan_hi
    slo = scan_lo === nothing ? _ad_factor_floor(shi, k_max) : scan_lo
    wlo = Float64(inputsEP.WIDTH_MIN); whi = Float64(inputsEP.WIDTH_MAX)
    kyhats = [(1.0 / nkyhat) * i for i in 1:nkyhat]
    widths = nefwid == 1 ? [0.5 * (wlo + whi)] : [wlo + (whi - wlo) / (nefwid - 1) * (i - 1) for i in 1:nefwid]
    pts = [(ky, w) for ky in kyhats for w in widths]
    np = length(pts)

    # Cheap-rank → few-confirm over the whole canonical grid, seeding the early-stop bound with no
    # incumbent (Inf). The shared `_rank_confirm` (1) computes the cheap AE-band lower edge f1 at
    # every node (IFLUX=false; the AE-unstable hull edge, the correct lower bound since the faithful
    # keep onset is ≥ f1 node-wise), then (2) confirms candidates in increasing cheap order with the
    # expensive faithful keep sweep (IFLUX=true), stopping once the next cheap onset ≥ the best
    # faithful found. This is the same logic now reused inside `critical_factor_robust`.
    rc = _rank_confirm(inputsEP, inputsPR, pts; gth = gth, slo = slo, shi = shi,
            incumbent_f = Inf, n_eig = 24, inner = inner, team = team,
            use_gpu = use_gpu, verbose = verbose)

    status = (rc.best_bind === :none || !isfinite(rc.best_f)) ? :no_onset :
             (rc.best_f >= 0.999 * shi ? :cap : :ok)
    return (; sfmin = rc.best_f, kyhat = rc.best_ky, width = rc.best_w, binding = rc.best_bind,
            status = status, cheap_min = rc.cheap_min, n_confirm = rc.n_confirm, npts = np,
            total_evals_full = rc.total_evals_full, total_evals_eig = rc.total_evals_eig, results = rc.confirmed)
end

"""
    critical_factor_profile(inputsEP, inputsPR; kwargs...) -> NamedTuple

AD analogue of the radial driver loop (`tjlfep_driver` / `mainsub` over the
`SCAN_N` radii): the critical EP scale factor `sfmin(IR)` versus radius, computed
by [`critical_factor_optimize`](@ref) at each radius with **cross-radius
continuation** — the optimum `(kyhat,width)` of one radius warm-starts the next.
(Continuation skips the per-radius seed grid, but on the bumpy `f★(ky,w)` surface
it does not reliably reduce total solves and can land in a worse local basin; see
the cost note on `critical_factor_optimize`. It is most useful when the optimum
moves smoothly with radius.)

Keywords: `radii=nothing` (→ `Int.(IR_EXP[1:SCAN_N])`), `scan_lo=1e-3`,
`scan_hi=10.0`, `faithful=false` (confirm each optimum with
[`marginal_factor_faithful`](@ref)), `use_gpu=false`, `verbose=false`; remaining
kwargs forwarded to `critical_factor_optimize`.

Returns `(; radii, sfmin, kyhat, width, evals, converged, binding, total_evals)`.
When `faithful=true`, `sfmin` holds the faithful (all-filter) onset and `binding`
the per-radius binding condition; otherwise `sfmin` is the AE-band onset.
"""
function critical_factor_profile(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                                 radii::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                 scan_lo::Float64 = 1.0e-3, scan_hi::Float64 = 10.0,
                                 faithful::Bool = false,
                                 use_gpu::Bool = false, verbose::Bool = false, kwargs...)
    rs = radii === nothing ? Int.(inputsEP.IR_EXP[1:inputsEP.SCAN_N]) : collect(radii)
    n = length(rs)
    sfmin = fill(Inf, n); kyhat = fill(NaN, n); width = fill(NaN, n)
    evals = zeros(Int, n); conv = falses(n); binding = fill(:none, n)
    prev_seed = nothing
    for (i, ir) in enumerate(rs)
        ep = deepcopy(inputsEP); ep.IR = ir
        r = critical_factor_optimize(ep, inputsPR; seed = prev_seed, scan_lo = scan_lo, scan_hi = scan_hi,
                                     faithful_confirm = faithful, use_gpu = use_gpu, verbose = verbose, kwargs...)
        if faithful && r.faithful !== nothing
            sfmin[i] = r.faithful.factor_faithful
            binding[i] = r.faithful.binding
        else
            sfmin[i] = r.sfmin
        end
        kyhat[i] = r.kyhat; width[i] = r.width; evals[i] = r.evals; conv[i] = r.converged
        verbose && @info "radius" i ir sfmin=sfmin[i] kyhat=kyhat[i] width=width[i] evals=evals[i] converged=conv[i] binding=binding[i]
        # continuation: carry the optimum forward only if this radius converged
        if r.converged && isfinite(r.sfmin)
            prev_seed = (r.kyhat, r.width)
        end
    end
    return (; radii = rs, sfmin, kyhat, width, evals, converged = conv, binding, total_evals = sum(evals))
end

# Default sensitivity knobs for a TGLF-EP operating point: the per-species inverse
# scale lengths (density a/Ln = RLNS, temperature a/LT = RLTS) and densities (AS),
# plus the local safety factor Q_LOC — the standard AE drive/damping parameters.
# Species are labeled e/i1/i2/.../EP; the EP species (inputsEP.IS_EP) is the
# primary Alfvén-eigenmode drive.
function default_sensitivity_knobs(inputF)
    ns = inputF.NS
    knobs = Tuple{Symbol,Union{Int,Nothing}}[]
    for is in 1:ns
        push!(knobs, (:RLNS, is))
        push!(knobs, (:RLTS, is))
    end
    for is in 1:ns
        push!(knobs, (:AS, is))
    end
    push!(knobs, (:Q_LOC, nothing))
    return knobs
end

# `ep_slot` is the EP species SLOT in InputTJLF, i.e. IS_EP+1 (TJLF_map sets
# `is = inputsEP.IS_EP + 1`), not IS_EP itself. Slot 1 is the electron; the
# remaining ions are labeled i1,i2,… skipping the EP slot.
function _knob_label(fld::Symbol, idx, ep_slot::Int)
    idx === nothing && return string(fld)
    tag = idx == 1 ? "e" : (idx == ep_slot ? "EP" : "i$(idx-1)")
    return "$(fld)[$tag]"
end

"""
    gamma_input_sensitivities(inputsEP, inputsPR; knobs=nothing, use_gpu=false) -> NamedTuple

AD-exact sensitivities of the per-mode growth rate `γ` (and real frequency) to the
local TGLF plasma inputs, at a **fixed** operating point `(IR, KYHAT_IN, WIDTH_IN,
FACTOR_IN)`. This is the genuinely differentiable, grid-independent quantity the
Fortran TGLF-EP cannot provide — unlike the critical factor `sfmin`, which sits at
a discrete keep-flag transition and is not smoothly differentiable.

The growth rate is differentiated through TGLF's eigensolve in a single forward
pass by seeding multi-partial `ForwardDiff.Dual`s on the selected inputs of the
`InputTJLF` produced by `TJLF_map` (so the eigenvalue rule in TJLF's
`TJLFForwardDiffExt` provides exact `∂λ/∂input`). Uses `IFLUX=false` (eigenvalues
only), so it is cheap.

`knobs` is a vector of `(field::Symbol, idx)` pairs; `idx::Int` selects a species
entry of a vector field (`:RLNS,:RLTS,:TAUS,:AS,:VPAR,:VPAR_SHEAR`) and `idx=nothing`
a scalar field (`:Q_LOC,:Q_PRIME_LOC,:RMIN_LOC,...`). Defaults to per-species
a/Ln, a/LT, density, plus Q_LOC.

Returns `(; gamma, freq, dgamma, dfreq, logsens, knobs, labels, base)` where
`dgamma`/`dfreq` are `NMODES × Nknob` Jacobians, `base` the knob base values, and
`logsens[m,j] = base[j]·∂γ_m/∂knob_j` the dimensionless (logarithmic) sensitivity
used to rank drive (>0) vs damping (<0) terms.
"""
function gamma_input_sensitivities(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                                   knobs::Union{Nothing,AbstractVector} = nothing,
                                   use_gpu::Bool = false,
                                   mode_in_override::Union{Nothing,Int} = nothing)
    mi = _resolve_ad_mode_in(inputsEP, mode_in_override)
    inputF = TJLF_map(inputsEP, inputsPR; mode_in_override = mi)
    inputF isa Integer && error("gamma_input_sensitivities: TJLF_map rejected IR=$(inputsEP.IR)")
    _configure_inputTJLF_for_ky!(inputF, inputsEP)

    ks = knobs === nothing ? default_sensitivity_knobs(inputF) : knobs
    N = length(ks)
    # EP species slot in InputTJLF is IS_EP+1 (see TJLF_map: is = IS_EP + 1); 0 if unset.
    ep_slot = inputsEP.IS_EP === missing ? 0 : Int(inputsEP.IS_EP) + 1

    Tag = typeof(ForwardDiff.Tag(gamma_input_sensitivities, Float64))
    D   = ForwardDiff.Dual{Tag, Float64, N}
    inputD = _to_dual_input(inputF, D)

    base = zeros(Float64, N)
    labels = String[]
    for (i, (fld, idx)) in enumerate(ks)
        v = idx === nothing ? getfield(inputF, fld) : getfield(inputF, fld)[idx]
        base[i] = Float64(v)
        seed = ForwardDiff.Dual{Tag}(base[i], ntuple(j -> j == i ? 1.0 : 0.0, N)...)
        if idx === nothing
            setfield!(inputD, fld, seed)
        else
            getfield(inputD, fld)[idx] = seed
        end
        push!(labels, _knob_label(fld, idx, ep_slot))
    end

    # Singular dispersion matrix → no resolvable eigenmode; degrade to AE-stable (γ=0, zero
    # sensitivities) rather than crashing, matching TJLFEP_ky and the gamma_* helpers.
    local res
    try
        res = _tjlf_run_dual(inputD, D; use_gpu = use_gpu)
    catch err
        err isa SingularException || rethrow()
        @warn "gamma_input_sensitivities: singular TGLF dispersion matrix (ir=$(inputsEP.IR), ky=$(ForwardDiff.value(inputD.KY)), width=$(ForwardDiff.value(inputD.WIDTH))); treating combo as stable (γ=0, no AE mode)" maxlog = 5
        NM = inputD.NMODES
        z = zeros(Float64, NM, N)
        return (; gamma = zeros(Float64, NM), freq = zeros(Float64, NM),
                  dgamma = z, dfreq = copy(z), logsens = copy(z), knobs = ks, labels, base)
    end
    g = res.eigenvalue[:, 1, 1]
    f = res.eigenvalue[:, 1, 2]
    NM = length(g)

    gamma = ForwardDiff.value.(g)
    freq  = ForwardDiff.value.(f)
    dgamma = [ForwardDiff.partials(g[m], i) for m in 1:NM, i in 1:N]
    dfreq  = [ForwardDiff.partials(f[m], i) for m in 1:NM, i in 1:N]
    logsens = [base[i] * dgamma[m, i] for m in 1:NM, i in 1:N]

    return (; gamma, freq, dgamma, dfreq, logsens, knobs = ks, labels, base)
end

# ════════════════════════════════════════════════════════════════════════════════════════════════
# Extended-box + separable-nbasis "truth" protocol for the critical EP scale factor.
#
# NOTE (2026-10): the ballooning-width physics floor WIDTH_FLOOR = 0.5 now bounds every width search
# (w_lo defaults to width_floor() and lower requests are clamped); the w ≈ 0.1 optima described below were
# found with the pre-floor w_lo = 0.05 and lie outside the model's validity (historical record).
# Near-marginal radii hide their true sfmin minimum at NARROW width (w≈0.1), BELOW the canonical
# WIDTH_MIN=1.0, as an INTERIOR bowl (sfmin turns back up below w≈0.05); nbasis at that optimum
# converges geometrically (limit by nb≈48, the basis being rank-singular by nb≈64). The canonical
# w≥1 grid therefore biases sfmin high there (up to ~25× at the steep profile edge). The protocol:
#   1. LOCATE over an extended LOG-spaced (ky,width) seed grid (width down to ~0.05, WIDTH_MIN/MAX
#      anchored) via cheap AE-onset ranking → :ad polish → faithful confirm at a working nbasis.
#   2. SEPARABLE nbasis convergence AT the fixed located optimum (nbasis shifts the value, not the
#      location) — 3 evals + geometric extrapolation, capped nb≤56.
# Validated on DIII-D n_scan=20 (jobs 54569487/extbox5, 54574360): recovers IR=95 truth 0.215 at
# (ky=0.81,w=0.11) vs canonical 5.49 (25.6×); IR=48 0.0195 (2.8×); clean radii unchanged.
# ════════════════════════════════════════════════════════════════════════════════════════════════

# Pinned-aware f1 seed grid → :ad descent on interior basins → early-stop few-confirm. The cheap front
# end (the canonical fast solver): floor-pinned basins (onset at the scan floor) are SEEN via the
# AE-unstable-window lower edge f1, while interior basins are slid off-node by the :ad descent. Returns
# trust diagnostics (feasible_frac, cheap_gap) for the escalation gate. ky_lo=0.25 reproduces the
# canonical kwscale_scan kyhat floor {0.25,..,1.0} for nseed_ky=4.
"""
    critical_factor_ad_f1seed(ep0, prof; kwargs...)

Cheap "front-end" critical-factor solver: an `f1`-seed grid over `(kyhat, width)`
followed by an AD (`:ad`) descent on the interior basins and an early-stop
few-confirm pass. Floor-pinned basins (onset at the scan floor) are caught via
the AE-unstable-window lower-edge `f1`, while interior basins are slid off-node
by the descent. This is the fast tier feeding the `:ad :locate`/`robust_ad`
escalation ladder.

Key keywords: `gamma_thresh` (growth-rate threshold; defaults per case),
`scan_lo`/`scan_hi` (factor bracket; default `FACTOR_IN` and `FACTOR_IN/512`),
`nseed_ky`/`nseed_w`/`n_eig_seed` (seed-grid sizes), `k_descend` (interior
basins to descend), `ky_lo=0.25` (kyhat floor), and `inner`/`team`/`use_gpu`
for the parallel/GPU backend.

Returns the critical factor together with trust diagnostics (e.g.
`feasible_frac`, `cheap_gap`) used by the escalation gate.
"""
function critical_factor_ad_f1seed(ep0, prof; gamma_thresh=nothing,
                                   scan_lo=nothing, scan_hi=nothing, k_max::Int=_k_max_env(),
                                   nseed_ky::Int=4, nseed_w::Int=4, n_eig_seed::Int=12,
                                   k_descend::Int=4, ky_lo::Float64=0.25,
                                   inner::Symbol=:threads, team=nothing,
                                   use_gpu::Bool=false, verbose::Bool=false, descent_kwargs...)
    gth = gamma_thresh === nothing ? _gamma_thresh_for(ep0, prof) : gamma_thresh
    shi = scan_hi === nothing ? Float64(ep0.FACTOR_IN) : scan_hi
    slo = scan_lo === nothing ? _ad_factor_floor(shi, k_max) : scan_lo
    wlo, whi = Float64(ep0.WIDTH_MIN), Float64(ep0.WIDTH_MAX)

    kyhats = ky_lo <= 0.0 ? [(1.0/nseed_ky)*i for i in 1:nseed_ky] :
             [ky_lo + (1.0 - ky_lo)/(nseed_ky - 1)*(i - 1) for i in 1:nseed_ky]
    widths = nseed_w == 1 ? [0.5*(wlo+whi)] : [wlo + (whi-wlo)/(nseed_w-1)*(i-1) for i in 1:nseed_w]
    pts = [(ky, w) for ky in kyhats for w in widths]
    cheap = _ad_pmap(idx -> begin
            ky, w = pts[idx]; ep = deepcopy(ep0); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
            win = _ae_unstable_window(ep, prof, gth; scan_lo=slo, scan_hi=shi,
                      n_eig=n_eig_seed, threaded=false, use_gpu=use_gpu)
            (; ky=ky, w=w, f=(win.unstable ? win.f1 : Inf), pinned=win.pinned_lo, evals=win.evals)
        end, length(pts); inner=inner, team=team)
    total_eig = 0; for c in cheap; total_eig += c.evals; end
    feas = [c for c in cheap if isfinite(c.f)]
    if isempty(feas)
        return (; sfmin=Inf, kyhat=NaN, width=NaN, binding=:none, status=:no_onset,
                n_descend=0, n_confirm=0, feasible_frac=0.0, cheap_gap=Inf,
                nfeasible=0, npts=length(pts), total_evals_full=0, total_evals_eig=total_eig)
    end
    sort!(feas, by = c -> c.f)

    cands = Tuple{Float64,Float64,Float64}[]
    n_desc = 0
    for (j, c) in enumerate(feas)
        if j <= k_descend && !c.pinned
            r = critical_factor_optimize(ep0, prof; gamma_thresh=gth, seed=(c.ky, c.w),
                    scan_lo=slo, scan_hi=shi, ky_bounds=(max(ky_lo, 1.0e-6), 1.0),
                    inner=inner, team=team, use_gpu=use_gpu, descent_kwargs...)
            total_eig += r.evals; n_desc += 1
            if isfinite(r.sfmin)
                push!(cands, (r.kyhat, r.width, r.sfmin))   # descended (possibly off-node) optimum
                push!(cands, (c.ky, c.w, c.f))              # grid-floor guard: keep the original seed
            else
                push!(cands, (c.ky, c.w, c.f))              # descent failed → keep seed
            end
        else
            push!(cands, (c.ky, c.w, c.f))                  # pinned, or beyond k_descend
        end
    end

    sort!(cands, by = c -> c[3])
    best_f = Inf; best_ky=NaN; best_w=NaN; best_bind=:none; best_cheap=NaN
    total_full=0; n_confirm=0
    for c in cands
        c[3] >= best_f && break
        r = marginal_factor_faithful(ep0, prof; kyhat=c[1], width=c[2], gamma_thresh=gth,
                scan_lo=slo, scan_hi=shi, threaded=true, inner=inner, team=team, use_gpu=use_gpu)
        n_confirm += 1; total_full += r.evals_full; total_eig += r.evals_eig
        if r.binding != :none && isfinite(r.factor_faithful) && r.factor_faithful < best_f
            best_f = r.factor_faithful; best_ky=c[1]; best_w=c[2]; best_bind=r.binding; best_cheap=c[3]
        end
    end
    status = (best_bind===:none || !isfinite(best_f)) ? :no_onset : (best_f >= 0.999*shi ? :cap : :ok)
    feasible_frac = length(pts) == 0 ? 0.0 : length(feas) / length(pts)
    cheap_gap = (isfinite(best_f) && isfinite(best_cheap) && best_cheap > 0) ? best_f / best_cheap : Inf
    return (; sfmin=best_f, kyhat=best_ky, width=best_w, binding=best_bind, status=status,
            n_descend=n_desc, n_confirm=n_confirm, feasible_frac=feasible_frac, cheap_gap=cheap_gap,
            nfeasible=length(feas), npts=length(pts),
            total_evals_full=total_full, total_evals_eig=total_eig)
end

# Convergence test + (guarded) geometric extrapolation of an nbasis sequence at a FIXED optimum.
# The reported `limit` is what the caller uses as the nbasis-converged sfmin; it must never wander
# outside the measured envelope. Two guards keep the fragile Aitken formula `v3 + d2·r/(1-r)` from
# overshooting:
#   1. If the sequence is already flat to `rtol` (|d2|/|v3| ≤ rtol), it is CONVERGED — report the
#      finest measured value directly. (Avoids the r→1 blow-up: e.g. a +5% sequence with r=0.97 has
#      1/(1-r)≈36×, which would amplify a tiny increment into a 2× "limit".)
#   2. Otherwise extrapolate ONLY when increments shrink fast (0 < r ≤ r_max, tail ≤ 1/(1-r_max)),
#      and clamp the result into the measured band. r ≥ 1 (growing increments) ⇒ genuinely NOT
#      converged ⇒ report finest and flag it (the caller's min(grid,truth) then keeps grid).
function _nbasis_extrapolate(nbs::Vector{Int}, vals::Vector{Float64};
                             rtol::Float64=0.03, r_max::Float64=0.6)
    good = [(nbs[i], vals[i]) for i in eachindex(vals) if isfinite(vals[i])]
    length(good) == 0 && return (; limit=NaN, ratio=NaN, converged=false, finest=NaN, nb_finest=0)
    finest = good[end][2]; nb_finest = good[end][1]
    length(good) < 3 && return (; limit=finest, ratio=NaN, converged=false, finest=finest, nb_finest=nb_finest)
    v1, v2, v3 = good[end-2][2], good[end-1][2], good[end][2]
    d1 = v2 - v1; d2 = v3 - v2
    r = (d1 == 0.0) ? 0.0 : d2 / d1
    # (1) already flat: trust the finest measured value, no extrapolation
    if abs(d2) <= rtol * max(abs(v3), eps())
        return (; limit=finest, ratio=r, converged=true, finest=finest, nb_finest=nb_finest)
    end
    # (2) fast geometric shrink: extrapolate with a bounded tail, clamped to the measured band
    if 0.0 < r <= r_max
        lim = v3 + d2 * r / (1.0 - r)
        lo = min(v1, v2, v3) - abs(d2); hi = max(v1, v2, v3) + abs(d2)
        if lo <= lim <= hi
            return (; limit=lim, ratio=r, converged=true, finest=finest, nb_finest=nb_finest)
        end
    end
    return (; limit=finest, ratio=r, converged=false, finest=finest, nb_finest=nb_finest)
end

# Extended-box locate: log-width cheap-rank seed grid → :ad polish → candidate list (confirmed by the
# caller at nb_work). WIDTH_MIN/WIDTH_MAX are anchored into the grid so the canonical box (where
# floor-pinned modes live near w=WIDTH_MIN) is always covered.
function _locate_extended(ep0, prof; gth, slo, shi, ky_lo, w_lo, w_hi,
                          nseed_ky, nseed_w, n_eig_seed, k_descend,
                          inner, team, use_gpu)
    kyhats = nseed_ky == 1 ? [0.5*(ky_lo+1.0)] :
             [ky_lo + (1.0 - ky_lo)/(nseed_ky - 1)*(i - 1) for i in 1:nseed_ky]
    logw = nseed_w == 1 ? [sqrt(w_lo*w_hi)] : collect(exp.(range(log(w_lo), log(w_hi), length=nseed_w)))
    widths = sort(unique(vcat(logw, Float64(ep0.WIDTH_MIN), Float64(ep0.WIDTH_MAX))))
    pts = [(ky, w) for ky in kyhats for w in widths]

    cheap = _ad_pmap(idx -> begin
            ky, w = pts[idx]; ep = deepcopy(ep0); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
            win = _ae_unstable_window(ep, prof, gth; scan_lo=slo, scan_hi=shi,
                      n_eig=n_eig_seed, threaded=false, use_gpu=use_gpu)
            (; ky=ky, w=w, f=(win.unstable ? win.f1 : Inf), pinned=win.pinned_lo, evals=win.evals)
        end, length(pts); inner=inner, team=team)
    cheap_eig = 0; for c in cheap; cheap_eig += c.evals; end
    desc_eig = 0; total_eig = cheap_eig
    feas = [c for c in cheap if isfinite(c.f)]
    isempty(feas) && return (; cands=Tuple{Float64,Float64,Float64}[], feas=feas,
                             npts=length(pts), n_desc=0, total_eig=total_eig,
                             cheap_eig=cheap_eig, desc_eig=desc_eig)
    sort!(feas, by = c -> c.f)

    cands = Tuple{Float64,Float64,Float64}[]; n_desc = 0
    for (j, c) in enumerate(feas)
        if j <= k_descend && !c.pinned
            r = critical_factor_optimize(ep0, prof; gamma_thresh=gth, seed=(c.ky, c.w),
                    scan_lo=slo, scan_hi=shi, ky_bounds=(max(ky_lo, 1.0e-6), 1.0), w_bounds=(w_lo, w_hi),
                    inner=inner, team=team, use_gpu=use_gpu)
            total_eig += r.evals; desc_eig += r.evals; n_desc += 1
            isfinite(r.sfmin) && push!(cands, (r.kyhat, r.width, r.sfmin))   # descended interior optimum
            push!(cands, (c.ky, c.w, c.f))                                   # grid-floor guard: keep seed
        else
            push!(cands, (c.ky, c.w, c.f))                                   # pinned / beyond k_descend
        end
    end
    sort!(cands, by = c -> c[3])
    return (; cands=cands, feas=feas, npts=length(pts), n_desc=n_desc, total_eig=total_eig,
            cheap_eig=cheap_eig, desc_eig=desc_eig)
end

"""
    critical_factor_truth(inputsEP, inputsPR; ky_lo=0.05, w_lo=width_floor(), nseed_ky=6, nseed_w=10,
                          nb_work=32, nb_steps=[32,40,48,56], inner=:threads, team=nothing,
                          use_gpu=false) -> NamedTuple

The `nbasis`-converged accuracy tier. This is **`critical_factor_robust(extend_width=true)` plus a
`nbasis` ladder at the located optimum** — the third rung of the `grid → robust_ad → truth` cost/
fidelity ladder. It first calls [`critical_factor_robust`](@ref) (width-extended: `min` of the
canonical `w≥1` grid-zoom and the extended narrow-width locate) to find the production `(ky,w)` and
its faithful `sfmin` at `nb_work`, then re-evaluates the FAITHFUL marginal factor at that FIXED
optimum over `nb_steps` (geometric extrapolation, capped ≤56; `SingularException` at high
`(nb,narrow-w)` → `NaN`, not fatal). The `nbasis` correction is *adverse* (raises the threshold) at
the still-converging outer radii — exactly the `+nbasis` component of the `grid→truth` gap (the width
component, the 2–11× reduction, is already captured by `robust_ad`). The reported `sfmin` is the
`nbasis`-converged value **floored by `robust_ad`'s canonical `w≥1` minimum** (`sfmin_w1`): a
razor-thin narrow win at `nb_work` can flip *above* the `w≥1` threshold once converged, and the true
critical factor is the lowest threshold over ALL modes (the `w≥1` mode is already converged at
`nb_work`), so `sfmin = min(nbasis-converged narrow, w≥1)`. On most radii the narrow mode stays well
below `w≥1` and the floor is a no-op. Returns
`(; sfmin, sfmin_work, sfmin_conv, sfmin_w1, floored, kyhat, width, binding, status, nb_limit,
nb_ratio, nb_converged, nb_table, feasible_frac, n_confirm, total_evals_full, total_evals_eig)` where
`sfmin` is the floored production value, `sfmin_conv` the raw `nbasis`-converged narrow value before
the floor, `sfmin_work` the `robust_ad` value at `nb_work`, and `floored` whether the `w≥1` floor won.
"""
function critical_factor_truth(ep0, prof; gamma_thresh=nothing, scan_lo=nothing, scan_hi=nothing,
                               k_max::Int=_k_max_env(),
                               ky_lo::Float64=0.05, w_lo::Float64=width_floor(),
                               nseed_ky::Int=6, nseed_w::Int=10, n_eig_seed::Int=12, k_descend::Int=6,
                               confirm_grid::Bool=false,
                               nb_work::Int=32, nb_steps::Vector{Int}=[32, 40, 48, 56],
                               inner::Symbol=:threads, team=nothing,
                               use_gpu::Bool=false, verbose::Bool=false)
    @assert all(nb -> nb <= 56, nb_steps) "nb_steps must stay ≤56 (nb≥64 Hermite overlap matrix is singular)"
    gth = gamma_thresh === nothing ? _gamma_thresh_for(ep0, prof) : gamma_thresh
    shi = scan_hi === nothing ? Float64(ep0.FACTOR_IN) : scan_hi
    slo = scan_lo === nothing ? _ad_factor_floor(shi, k_max) : scan_lo
    w_lo = floor_w_lo(w_lo)   # ballooning-width physics floor (WIDTH_FLOOR = 0.5)

    # ── Tier-2: locate the production (ky,w) optimum with the width-extended robust solver at
    # nb_work. This is the SAME object as solver=:robust_ad — min of the canonical w≥1 grid-zoom
    # and the extended narrow-width locate — so truth differs from robust_ad ONLY by the nbasis
    # ladder below. (The earlier separate _locate_extended + w≥1 floor now live inside robust.)
    epw = deepcopy(ep0); epw.N_BASIS = nb_work
    rob = critical_factor_robust(epw, prof; gamma_thresh=gth, scan_lo=slo, scan_hi=shi,
            extend_width=true, confirm_grid=confirm_grid, ky_lo=ky_lo, w_lo=w_lo, nseed_ky=nseed_ky, nseed_w=nseed_w,
            n_eig_seed=n_eig_seed, k_descend=k_descend, inner=inner, team=team, use_gpu=use_gpu)
    total_full = rob.total_evals_full; total_eig = rob.total_evals_eig
    best_f = rob.sfmin; best_ky = rob.kyhat; best_w = rob.width; best_bind = rob.binding
    feasible_frac = rob.npts_coarse == 0 ? 0.0 : rob.n_feasible_coarse / rob.npts_coarse
    if rob.status === :no_onset || !isfinite(best_f)
        return (; sfmin=Inf, sfmin_work=Inf, sfmin_conv=Inf, sfmin_w1=rob.sfmin_w1, floored=false,
                kyhat=NaN, width=NaN, binding=:none, status=:no_onset,
                nb_limit=NaN, nb_ratio=NaN, nb_converged=false, nb_table=Int[]=>Float64[],
                feasible_frac=feasible_frac, n_confirm=rob.n_ext_confirm,
                total_evals_full=total_full, total_evals_eig=total_eig)
    end

    # ── Tier-3: separable nbasis convergence AT THE FIXED located optimum (re-optimising per nb
    # would corrupt the sequence by hopping basins); SingularException at high (nb,narrow-w) is
    # logged as NaN, not fatal. The converged value is the honest, conservative threshold — it can
    # exceed the nb_work robust_ad value (the nbasis correction is adverse at the outer edge); truth
    # reports it as-is (no flooring), which is exactly the +nbasis component of the grid→truth gap.
    nbs = Int[]; vals = Float64[]
    for nb in nb_steps
        v = if nb == nb_work
            best_f
        else
            epn = deepcopy(ep0); epn.N_BASIS = nb
            try
                r = marginal_factor_faithful(epn, prof; kyhat=best_ky, width=best_w, gamma_thresh=gth,
                        scan_lo=slo, scan_hi=shi, threaded=true, inner=inner, team=team, use_gpu=use_gpu)
                total_full += r.evals_full; total_eig += r.evals_eig
                (r.binding !== :none && isfinite(r.factor_faithful)) ? r.factor_faithful : NaN
            catch e
                NaN
            end
        end
        push!(nbs, nb); push!(vals, v)
    end
    ex = _nbasis_extrapolate(nbs, vals)
    finite_vals = [v for v in vals if isfinite(v)]
    sfmin_conv = ex.converged ? ex.limit : (isempty(finite_vals) ? best_f : finite_vals[end])

    # Final w≥1 floor (min-over-modes): a razor-thin narrow win at nb_work can flip ABOVE the
    # canonical w≥1 threshold once nbasis-converged (the nbasis correction is adverse). The true
    # critical factor is the lowest threshold over ALL modes, and the w≥1 mode is already converged at
    # nb_work, so cap the reported value at robust_ad's w≥1 grid-zoom min. When the floor wins, adopt
    # its (ky,w) too. (On most radii the narrow mode stays well below w≥1, so this is a no-op.)
    floored = isfinite(rob.sfmin_w1) && rob.sfmin_w1 < sfmin_conv
    sfmin = floored ? rob.sfmin_w1 : sfmin_conv
    out_ky  = floored ? rob.ky_w1   : best_ky
    out_w   = floored ? rob.w_w1    : best_w
    out_bind = floored ? rob.bind_w1 : best_bind

    verbose && @info "truth" ky=best_ky w=best_w sfmin_work=best_f w1=rob.sfmin_w1 nb_limit=ex.limit r=ex.ratio converged=ex.converged conv=sfmin_conv final=sfmin floored

    return (; sfmin=sfmin, sfmin_work=best_f, sfmin_conv=sfmin_conv, sfmin_w1=rob.sfmin_w1, floored=floored,
            kyhat=out_ky, width=out_w, binding=out_bind,
            status=(sfmin >= 0.999*shi ? :cap : :ok),
            nb_limit=ex.limit, nb_ratio=ex.ratio, nb_converged=ex.converged, nb_table=(nbs => vals),
            feasible_frac=feasible_frac, n_confirm=rob.n_ext_confirm,
            total_evals_full=total_full, total_evals_eig=total_eig)
end

"""
    critical_factor_triggered(inputsEP, inputsPR; w_floor_tol=0.05, gap_thresh=1.5, feas_thresh=0.25,
                              ky_lo=0.25, truth_kwargs=NamedTuple(), inner=:threads, team=nothing,
                              use_gpu=false) -> NamedTuple

Production policy: run the fast canonical [`critical_factor_ad_f1seed`](@ref) pass on
`w∈[WIDTH_MIN,WIDTH_MAX]`, and escalate to [`critical_factor_truth`](@ref) only when the canonical
optimum pins at the width floor (a narrow-width bowl likely lies below the box) or the trust gate
fires (`sparse` feasible fraction, `cheap_gap` keep-filter divergence, `no_onset`, `cap`). When
escalated, returns the lower (`min(canonical, truth)`) faithful onset. Returns
`(; sfmin, kyhat, width, binding, status, escalated, reasons, source, base_sfmin, truth_sfmin, ...)`.

Note: because narrower width is systematically more unstable, the canonical optimum pins at
`WIDTH_MIN` at most radii, so the `:width_floor` trigger is broad. Use `truth_kwargs` to forward
solver knobs (e.g. `nseed_w`, `nb_steps`) to the escalated search.
"""
function critical_factor_triggered(ep0, prof; gamma_thresh=nothing, scan_lo=nothing, scan_hi=nothing,
                                   w_floor_tol::Float64=0.05, gap_thresh::Float64=1.5, feas_thresh::Float64=0.25,
                                   ky_lo::Float64=0.25, nseed_ky::Int=4, nseed_w::Int=8, n_eig_seed::Int=12,
                                   truth_kwargs=NamedTuple(),
                                   inner::Symbol=:threads, team=nothing,
                                   use_gpu::Bool=false, verbose::Bool=false)
    base = critical_factor_ad_f1seed(ep0, prof; ky_lo=ky_lo, nseed_ky=nseed_ky, nseed_w=nseed_w,
            n_eig_seed=n_eig_seed, gamma_thresh=gamma_thresh, scan_lo=scan_lo, scan_hi=scan_hi,
            inner=inner, team=team, use_gpu=use_gpu)
    wmin = Float64(ep0.WIDTH_MIN)
    pinned = isfinite(base.width) && base.width <= wmin * (1.0 + w_floor_tol)
    reasons = Symbol[]
    pinned                            && push!(reasons, :width_floor)
    (base.status === :no_onset)       && push!(reasons, :no_onset)
    (base.status === :cap)            && push!(reasons, :cap)
    (base.cheap_gap > gap_thresh)     && push!(reasons, :cheap_gap)
    (base.feasible_frac < feas_thresh)&& push!(reasons, :sparse)
    flagged = !isempty(reasons)
    full = base.total_evals_full; eig = base.total_evals_eig
    if !flagged
        return (; sfmin=base.sfmin, kyhat=base.kyhat, width=base.width, binding=base.binding,
                status=base.status, escalated=false, reasons=reasons, source=:canonical,
                base_sfmin=base.sfmin, truth_sfmin=NaN, total_evals_full=full, total_evals_eig=eig)
    end
    t = critical_factor_truth(ep0, prof; gamma_thresh=gamma_thresh, scan_lo=scan_lo, scan_hi=scan_hi,
            inner=inner, team=team, use_gpu=use_gpu, truth_kwargs...)
    full += t.total_evals_full; eig += t.total_evals_eig
    use_t = isfinite(t.sfmin) && (!isfinite(base.sfmin) || t.sfmin < base.sfmin)
    verbose && @info "triggered" reasons=reasons base=base.sfmin truth=t.sfmin chosen=(use_t ? :truth : :canonical)
    return (; sfmin=(use_t ? t.sfmin : base.sfmin), kyhat=(use_t ? t.kyhat : base.kyhat),
            width=(use_t ? t.width : base.width), binding=(use_t ? t.binding : base.binding),
            status=(use_t ? t.status : base.status), escalated=true, reasons=reasons,
            source=(use_t ? :truth : :canonical), base_sfmin=base.sfmin, truth_sfmin=t.sfmin,
            truth_nb_limit=t.nb_limit, truth_nb_converged=t.nb_converged, truth_width=t.width,
            total_evals_full=full, total_evals_eig=eig)
end
