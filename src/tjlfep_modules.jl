# NOTE: the former TJLFEP-local `InputTGLF` and `InputTJLF` duplicate types were removed
# here -- TGLF inputs now come from `TurbulentTransport.InputTGLF` and `TJLF.InputTJLF`
# (single input type, consistent with TJLF.jl). See TurbulentTransport/src/tglf_ep.jl.

"""
    Options{T<:Real}

Scan-control / run settings for a TJLFEP calculation — the Julia counterpart of
the Fortran `TGLFEP` interface module populated from an `input.TGLFEP` file.

Holds the process/threshold selectors (`PROCESS_IN`, `THRESHOLD_FLAG`,
`SCAN_METHOD`, rejection-flag toggles), basis/mode sizes (`N_BASIS`, `NMODES`,
`NTOROIDAL`), the radial scan definition (`SCAN_N`, `IRS`, `IR_EXP`), the
EP/ion species descriptors (`M_I`/`Q_I`/`M_EP`/`Q_EP`, `IS_EP`), the width
controls (`WIDTH_IN_FLAG`, `WIDTH_IN`/`WIDTH_MIN`/`WIDTH_MAX`), and the
per-`(kyhat,width)` keep/reject result masks (`LKEEP`, `LTEARING`, the pinch
flags, ...) plus scan outputs (`FACTOR_OUT`, `WIDTH`, `F_REAL`).

Fields default to `missing`/`NaN` and are filled by the readers
([`readTGLFEP`](@ref), [`TJLFEP_generate_input`](@ref)); the inner constructor
`Options{T}(nscan_in, widthin, nn, nr, jtscale_max, nmodes)` allocates the
vector/matrix fields to the right sizes.
"""
mutable struct Options{T<:Real} # This acts as the interface module of Fortran, essentially. It reads the TGLFEP file
    PROCESS_IN::Union{Int, Missing} # May need to be a MODE_IN::Union{T, Missing} for PROCESS_IN <= 1
    THRESHOLD_FLAG::Union{Int, Missing}
    N_BASIS::Union{Int, Missing}
    SCAN_METHOD::Union{Int, Missing}
    REJECT_I_PINCH_FLAG::Union{Int, Missing} # 5
    REJECT_E_PINCH_FLAG::Union{Int, Missing}
    REJECT_TH_PINCH_FLAG::Union{Int, Missing}
    REJECT_EP_PINCH_FLAG::Union{Int, Missing}
    REJECT_TEARING_FLAG::Union{Int, Missing}
    ROTATIONAL_SUPPRESSION_FLAG::Union{Int, Missing} # 10
    PPRIME_METHOD::Union{Int, Missing} # pprime_method added 10-9-2024, EMB
    QL_RATIO_THRESH::Union{T, Missing}
    THETA_SQ_THRESH::Union{T, Missing}
    Q_SCALE::Union{T, Missing}
    WRITE_WAVEFUNCTION::Union{Int, Missing} # Maybe not needed for my purposes? Not sure yet
    KY_MODEL::Union{Int, Missing} #15
    SCAN_N::Union{Int} 
    IRS::Union{Int, Missing}
    FACTOR_IN_PROFILE::Union{Bool, Missing}
    FACTOR_IN::Union{T, Missing}
    WIDTH_IN_FLAG::Union{Bool} #20
    WIDTH_IN::Union{T, Missing} 
    WIDTH_MIN::Union{T, Missing}
    WIDTH_MAX::Union{T, Missing}
    INPUT_PROFILE_METHOD::Union{Int, Missing} # I very likely will not use this (?). It seems that INPUT_PROFILE_METHOD is meant for 
    # determining the type of input that will be used: input.profile or input.profiles. EXPRO is the method used to do the latter
    # but I cannot find the runction in Fortran for expro_read because it is imported from somewhere else. 
    IS_EP::Union{Int, Missing} #25
    M_I::Union{T, Missing} 
    Q_I::Union{T, Missing}
    M_EP::Union{T, Missing}
    Q_EP::Union{T, Missing}
    IR::Union{Int, Missing} #30
    NMODES::Union{Int}  
    MODE_IN::Union{Int, Missing} # Likely will not be used for now. There are some default settings dependent on the choice of process-in. = 2 for process_in = 5: the case we are interested in.
    FACTOR_MAX_PROFILE::Union{Vector{T}, Missing}
    FACTOR_MAX::Union{T, Missing}
    KY_IN::Union{T, Missing} #35
    KYMARK::Union{T, Missing} 
    # All of the next 7 vectors require the value nr to be recognized for the array. This is already done in the profile struct.
    # I need to decide what I want to do with this then. These are for process_in = 6 apparently.
    DEP_TRACE_LOC::Union{T, Missing}
    DEP_TRACE::Union{Vector{T}, Missing}
    DEP_TRACE_COMPLETE::Union{Vector{T}, Missing}
    DEP_TRACE_SCALE::Union{Matrix{T}, Missing} #40
    QL_RATIO_LOC::Union{T, Missing} 
    QL_RATIO::Union{Vector{T}, Missing}
    QL_RATIO_COMPLETE::Union{Vector{T}, Missing}
    QL_RATIO_SCAN::Union{Matrix{T}, Missing}
    CHI_GB::Union{Vector{T}, Missing} #45
    IR_EXP::Union{Vector{Int64}, Missing}  
    NBASIS::Union{Int, Missing}
    NTOROIDAL::Union{Int, Missing}
    KYHAT_IN::Union{T, Missing}
    SCAN_FACTOR::Union{T, Missing} #50 Default to 1
    JTSCALE::Union{Int, Missing} 
    JTSCALE_MAX::Union{Int, Missing} # Default to 1
    TSCALE_INTERVAL::Union{T, Missing} # Default to 1.0
    N_ION::Union{Int, Missing} # Equal to NS-1?
    FREQ_CUTOFF::Union{T, Missing} #55 Default to -0.35
    FREQ_AE_UPPER::Union{T, Missing} 
    GAMMA_THRESH::Union{T,Missing}   
    GAMMA_THRESH_MAX::Union{T, Missing} #58
    NN::Union{Int, Missing}
    FACTOR_OUT::Union{Vector{T}, Missing} #60 vector of nn elements
    # May or may not need id_2, np_2, id_3, np_3 ?
    # Ignoring suffix and str_wf_file right now.
    # As with l_print, l-WRITE_WAVEFUNCTION, l_wavefunction_out
    LKEEP::Vector{Bool} # All the L-starting parts have 4 dimensions. See kwscale_scan for that.
    LTEARING::Vector{Bool} # This needs to be defined somewhat differently then.
    L_TH_PINCH::Vector{Bool} 
    L_I_PINCH::Vector{Bool}
    L_E_PINCH::Vector{Bool}#65
    L_EP_PINCH::Vector{Bool} 
    L_QL_RATIO::Vector{Bool}
    L_THETA_SQ::Vector{Bool} 
    FACTOR::Union{Vector{T}, Missing}
    SUFFIX::Union{String, Missing} #70
    F_REAL::Union{Vector{T}, Missing} 
    REAL_FREQ::Int
    WIDTH::Union{Vector{T}, Missing}#73
    # I might want a default constructor for default values. I'm going to make one (no paramters)
    # which follows the 'default' case that TGLFEP used in my first test of TGLFEP (OMFIT).

    # There are a bunch of cases where the dimension of the fills or the existence of a value (WIDTH_IN and WIDTH_MIN/MAX)
    # depends on other input variables:

    # If WIDTH_IN_FLAG is true the fill needs (?) to be SCAN_N long of that same value and the min/max are missing
    # There is a bit of confusion on if the same goes for the FACTOR_IN_PROFILE. 

    # There are few things that depend on the profile reading: nr
    function Options{T}(nscan_in::Int64, widthin::Bool, nn::Int64, nr::Int64, jtscale_max::Int64, nmodes::Int64) where {T<:Real}
        if(widthin)
            new(missing, missing, missing, missing, missing, missing, missing, missing, missing, missing,
            missing,
            NaN, NaN, NaN, missing, missing, nscan_in, missing, missing, NaN, widthin, NaN,
            NaN, NaN, missing, missing, NaN, NaN, NaN, NaN, missing, nmodes, missing, fill(NaN, nscan_in),
            NaN, NaN, NaN, NaN, fill(NaN, nr), fill(NaN, nr), fill(NaN, (jtscale_max, nr)), NaN, fill(NaN, nr), fill(NaN, nr),
            fill(NaN, (jtscale_max, nr)), fill(NaN, nr), fill(0, nscan_in), missing, missing, NaN, 1, missing, 1,
            1.0, missing, -0.35, NaN, NaN, NaN, nn, fill(NaN, nn), fill(false, nmodes), fill(false, nmodes),
            fill(false, nmodes), fill(false, nmodes), fill(false, nmodes),
            fill(false, nmodes), fill(false, nmodes), fill(false, nmodes),
            fill(NaN, nscan_in), missing, fill(NaN, nr), 0, fill(NaN, nscan_in))
        else
            new(missing, missing, missing, missing, missing, missing, missing, missing, missing, missing,
            missing,
            NaN, NaN, NaN, missing, missing, nscan_in, missing, missing, NaN, widthin, 0.0,
            NaN, NaN, missing, missing, NaN, NaN, NaN, NaN, missing, nmodes, missing, fill(NaN, nscan_in),
            NaN, NaN, NaN, NaN, fill(NaN, nr), fill(NaN, nr), fill(NaN, (jtscale_max, nr)), NaN, fill(NaN, nr), fill(NaN, nr),
            fill(NaN, (jtscale_max, nr)), fill(NaN, nr), fill(0, nscan_in), missing, missing, NaN, 1, missing, 1,
            1.0, missing, -0.35, NaN, NaN, NaN, nn, fill(NaN, nn), fill(false, nmodes), fill(false, nmodes),
            fill(false, nmodes), fill(false, nmodes), fill(false, nmodes),
            fill(false, nmodes), fill(false, nmodes), fill(false, nmodes),
            fill(NaN, nscan_in), missing, fill(NaN, nr), 0, fill(NaN, nscan_in))
        end
    end
end

# ── Ballooning-width physics floor ──────────────────────────────────────────────────────────────
"""
    WIDTH_FLOOR = 0.5

Physics floor on the TGLF ballooning width (TGLF width units) for every TGLF-EP width search: the
`WIDTH_MIN` / `WIDTH_IN` controls read from `input.TGLFEP` and the narrow-width search floor `w_lo` of the
extended-width AD solvers (`:ad :locate` / `:wide`, `:robust_ad`, `:truth`, `:multistart`, `:nlopt`).

Basis: the eigenvalue-optimal TGLF-EP width stays at or above 0.5; narrower Gauss–Hermite envelopes do not
represent the gyrokinetic mode structure, and in the APS-2026 campaign the sub-0.5 located optima were not
`N_BASIS`-converged and could lock onto non-EP modes.
Widths below this floor are therefore outside the model's validity and every search is clamped to
`w ≥ WIDTH_FLOOR` ([`enforce_width_floor!`](@ref), [`floor_w_lo`](@ref), [`width_floor`](@ref)). The
Fortran TGLF-EP box `[WIDTH_MIN, WIDTH_MAX] = [1, 2]` lies above the floor, so Fortran-faithful (`:grid`)
runs with the standard inputs are unaffected.

For numerical studies ONLY, the environment variable `TJLFEP_WIDTH_FLOOR` overrides the value.
"""
const WIDTH_FLOOR = 0.5

"""
    width_floor() -> Float64

The effective ballooning-width floor: [`WIDTH_FLOOR`](@ref) unless `ENV["TJLFEP_WIDTH_FLOOR"]` overrides
it (numerical studies only; results below 0.5 are outside the TGLF-EP model's validity).
"""
width_floor() = parse(Float64, get(ENV, "TJLFEP_WIDTH_FLOOR", string(WIDTH_FLOOR)))

"""
    floor_w_lo(w_lo; what="w_lo") -> Float64

Clamp a narrow-width search floor up to [`width_floor`](@ref), warning once when a caller asked for less.
"""
function floor_w_lo(w_lo::Real; what::AbstractString = "w_lo")
    wf = width_floor()
    if w_lo < wf
        @warn "TJLFEP: $(what) = $(w_lo) is below the ballooning-width physics floor WIDTH_FLOOR = $(wf); using $(wf) (override with ENV TJLFEP_WIDTH_FLOOR for numerical studies only)" maxlog = 1
        return wf
    end
    return Float64(w_lo)
end

"""
    enforce_width_floor!(o::Options) -> o

Clamp the width controls read from `input.TGLFEP` up to [`width_floor`](@ref) in place, with a warning:
`WIDTH_IN` when `WIDTH_IN_FLAG` is set (the single fixed width), else `WIDTH_MIN`/`WIDTH_MAX` (the scan
box). The unused control (e.g. the `WIDTH_IN = 0` placeholder of a scan-box input) is left alone. Called
by the input readers ([`readTGLFEP`](@ref) and the IMAS extension).
"""
function enforce_width_floor!(o::Options)
    wf = width_floor()
    fields = (o.WIDTH_IN_FLAG === true) ? (:WIDTH_IN,) : (:WIDTH_MIN, :WIDTH_MAX)
    for f in fields
        v = getfield(o, f)
        (v === missing || (v isa AbstractFloat && isnan(v))) && continue
        if v < wf
            @warn "TJLFEP: input $(f) = $(v) is below the ballooning-width physics floor WIDTH_FLOOR = $(wf); clamping to $(wf) (override with ENV TJLFEP_WIDTH_FLOOR for numerical studies only)" maxlog = 1
            setfield!(o, f, oftype(v, wf))
        end
    end
    return o
end


"""
    profile{T<:Real}

Local plasma-equilibrium profiles on the TJLFEP radial grid — the intermediary
between the input readers and the per-radius TGLF/TJLF solves.

Stores the field signs (`SIGN_BT`, `SIGN_IT`), grid/species sizes (`NR`, `NS`,
`N_ION`), the per-species matrices (`ZS`, `MASS`, `AS`, `TAUS`, `RLNS`, `RLTS`,
`VPAR`, `VPAR_SHEAR`), the per-radius geometry/Miller quantities (`RMIN`, `RMAJ`,
`Q`, `SHEAR`, `Q_PRIME`, `P_PRIME`, `KAPPA`/`DELTA`/`ZETA` and their shears, ...),
and derived quantities (`ZEFF`, `BETAE`, `RHO_STAR`, `OMEGA_TAE`, `gammaE`,
`B_UNIT`, ...). Populated from an `input.gacode`/EXPRO or `input.profile` source
by the readers (e.g. [`read_input_profile`](@ref), `preprocess_gacode_inputs`).
"""
mutable struct profile{T<:Real}
    # This struct is an intermediary between the processes in read_inputs
    
    SIGN_BT::Union{T, Missing} #
    SIGN_IT::Union{T, Missing} #

    NR::Union{Int, Missing} #
    NS::Union{Int, Missing} #
    GEOMETRY_FLAG::Union{Int, Missing} #5   Constant 1 for TJLFEPnotanything else.
    ROTATION_FLAG::Union{Int, Missing} #

    ZS::Union{Matrix{T}, Missing} #
    MASS::Union{Vector{T}, Missing} #

    # This next section is especially why the profile struct must exist
    AS::Union{Matrix{T}, Missing} #
    TAUS::Union{Matrix{T}, Missing} #10
    RLNS::Union{Matrix{T}, Missing} #
    RLTS::Union{Matrix{T}, Missing} #
    VPAR::Union{Matrix{T}, Missing} #
    VPAR_SHEAR::Union{Matrix{T}, Missing} #

    RMIN::Union{Vector{T}, Missing}#15
    RMAJ::Union{Vector{T}, Missing}
    SHIFT::Union{Vector{T}, Missing}
    Q::Union{Vector{T}, Missing}
    SHEAR::Union{Vector{T}, Missing}
    ALPHA::Union{Vector{T}, Missing}#20
    Q_PRIME::Union{Vector{T}, Missing}
    P_PRIME::Union{Vector{T}, Missing}
    KAPPA::Union{Vector{T}, Missing}
    S_KAPPA::Union{Vector{T}, Missing}
    DELTA::Union{Vector{T}, Missing}#25
    S_DELTA::Union{Vector{T}, Missing}
    ZETA::Union{Vector{T}, Missing}
    S_ZETA::Union{Vector{T}, Missing}
    ZEFF::Union{Vector{T}, Missing}
    BETAE::Union{Vector{T}, Missing}#30
    RHO_STAR::Union{Vector{T}, Missing}
    OMEGA_TAE::Union{Vector{T}, Missing}
    omegaGAM::Union{Vector{T}, Missing}
    gammaE::Union{Vector{T}, Missing}
    gammap::Union{Vector{T}, Missing}#35
    B_UNIT::Union{Vector{T}, Missing}

    IS::Union{Int, Missing}
    IRS::Union{Int, Missing}

    A_QN::Union{T, Missing} # quasineutrality scale factor for non-EP species
    N_ION::Union{Int, Missing} #40

    DENS_1::Union{Vector{T},Missing} 
    TEMP_1::Union{Vector{T},Missing}
    DLNNDR_1::Union{Vector{T},Missing} 
    DLNTDR_1::Union{Vector{T},Missing} #44

    DENS_2::Union{Vector{T},Missing} 
    TEMP_2::Union{Vector{T},Missing} 
    DLNNDR_2::Union{Vector{T},Missing} 
    DLNTDR_2::Union{Vector{T},Missing} #48

    DENS_3::Union{Vector{T},Missing} 
    TEMP_3::Union{Vector{T},Missing}
    DLNNDR_3::Union{Vector{T},Missing}
    DLNTDR_3::Union{Vector{T},Missing}  #52

    DENS_4::Union{Vector{T},Missing} 
    TEMP_4::Union{Vector{T},Missing} 
    DLNNDR_4::Union{Vector{T},Missing} 
    DLNTDR_4::Union{Vector{T},Missing}  #56

    DENS_5::Union{Vector{T},Missing} 
    TEMP_5::Union{Vector{T},Missing} 
    DLNNDR_5::Union{Vector{T},Missing} 
    DLNTDR_5::Union{Vector{T},Missing}  # 60

    DENS_6::Union{Vector{T},Missing} 
    TEMP_6::Union{Vector{T},Missing} 
    DLNNDR_6::Union{Vector{T},Missing} 
    DLNTDR_6::Union{Vector{T},Missing}  #64

    DENS_7::Union{Vector{T},Missing} 
    TEMP_7::Union{Vector{T},Missing} 
    DLNNDR_7::Union{Vector{T},Missing} 
    DLNTDR_7::Union{Vector{T},Missing}  #68

    DENS_8::Union{Vector{T},Missing} 
    TEMP_8::Union{Vector{T},Missing} 
    DLNNDR_8::Union{Vector{T},Missing} 
    DLNTDR_8::Union{Vector{T},Missing}  #72

    DENS_9::Union{Vector{T},Missing} 
    TEMP_9::Union{Vector{T},Missing} 
    DLNNDR_9::Union{Vector{T},Missing} 
    DLNTDR_9::Union{Vector{T},Missing}  #76

    DENS_10::Union{Vector{T},Missing} 
    TEMP_10::Union{Vector{T},Missing} 
    DLNNDR_10::Union{Vector{T},Missing} 
    DLNTDR_10::Union{Vector{T},Missing}  #80

    CS::Union{Vector{T},Missing}  #81

    # function profile()
    #     return profile{Float64}()
    # end
    # As of right now, I don't believe there needs to be parameters, but the vectors
    # are probably the most of concern there. 
    function profile{T}(nr::Int, ns::Int) where (T<:Real)
        # new(NaN, NaN, nr, ns, 1, missing, fill(NaN, ns), fill(NaN, ns),
        new(NaN, NaN, missing, missing, 1, missing, fill(NaN, (nr,ns)), fill(NaN, (ns)),
        fill(NaN, (nr, ns)), fill(NaN, (nr, ns)), fill(NaN, (nr, ns)), fill(NaN, (nr, ns)), 
        fill(NaN, (nr, ns)), fill(NaN, (nr, ns)), fill(NaN, nr), fill(NaN, nr), fill(NaN, nr), 
        fill(NaN, nr), fill(NaN, nr), fill(NaN, nr), fill(NaN, nr), fill(NaN, nr), fill(NaN, nr), 
        fill(NaN, nr), fill(NaN, nr), fill(NaN, nr), fill(NaN, nr), fill(NaN, nr), fill(NaN, nr), 
        fill(NaN, nr), fill(NaN, nr), fill(NaN, nr), fill(NaN, nr), fill(1.0E-7, nr), fill(NaN, nr), 
        fill(NaN, nr), missing, missing, NaN, missing, fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)),
        fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)),
        fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)),
        fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)),
        fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)),
        fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)),
        fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)), fill(NaN,(ns)),
        fill(NaN,(ns)), fill(NaN,(nr)))
    end
end

# Removed (unused): `InputTJLFEP` bundled the now-removed duplicate `InputTJLF` with
# `Options`/`profile`. The live path passes `Options`/`profile` directly.

