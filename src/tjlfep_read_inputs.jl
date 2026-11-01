"""
readMTGLF: Directly converts a known MTGLF file into a profile struct. It used to do this directly to an inputTJLF struct, but I am
going to follow more directly what TGLFEP does first and then come back and make alterations later if I so decide.

Inputs: filepath of input.MTGLF

Outputs: InputTJLF struct to be run through TJLF (as is done in TGLFEP)
"""
function readMTGLF(filename::String)
    open(filename)
    lines = readlines(filename)

    # This will be all very similar to TJLF for the most part. I'll check for NS and NKY (? - That's not used in TGLFEP so maybe the defualt)
    dflt = true
    ns = -1
    nr = -1
    
    for line in lines[1:100]
        line = split(line, "\n")
        line = split(line[1],"=")
        for i in 1:2
            line[i] = strip(line[i])
        end
        
        if line[1] == "NS"
            ns = parse(Int, strip(line[2]))
        elseif line[1] == "NR"
            nr = parse(Int, strip(line[2]))
        end
    end
    # make sure ns is defined
    @assert ns!=-1 "did not find NS in $filename"
    @assert nr!=-1 "din not find NR in $filename"
    
    #Translating MTGLF to usable InputTJLF form:
    inputMTGLF = profile{Float64}(nr, ns)
    irexp::Vector{Int64} = []
    zs_temp = fill(NaN, ns)  # ZS is per-species in MTGLF file; broadcast to (NR,NS) matrix after reading
    
    for line in lines[1:length(lines)]
        line = split(line, "\n")
        line = split(line[1],"=")
        # Reformat for ease of access:
        for i in 1:2
            line[i] = strip(line[i])
        end

        check = (match(r"\d\d_\d", line[1]) !== nothing) || (match(r"\d\d\d_\d", line[1]) !== nothing) || (match(r"\d_\d", line[1]) !== nothing)
        vcheck = (match(r" \d", line[1]) !== nothing) || (match(r"_\d", line[1]) !== nothing)

        twoName = ["OMEGA_TAE", "RHO_STAR", "S_ZETA", "S_DELTA", "S_KAPPA", "P_PRIME", "Q_PRIME", "B_UNIT", "IR_EXP"]
        delFields = ["IR"]
        if (check == true) # Matrix values
            val = parse(Float64, line[2])
            line = split(line[1], "_")
            for i in 1:3
                line[i] = strip(line[i])
            end
            
            if (!contains(line[2], "SHEAR"))
                speciesField = Symbol(line[1])
                Index1 = line[2]
                Index2 = line[3]
            else # Any non-singular named will be here basically.
                speciesField = Symbol("VPAR_SHEAR")
                Index1 = line[3]
                Index2 = line[4]
            end
            getfield(inputMTGLF, speciesField)[parse(Int, Index1), parse(Int, Index2)] = val
            # Bam, done with that part. Next to vectors.

        elseif (check == false && vcheck == true) # Vector values
            val = parse(Float64, line[2])
            speciesField = strip(replace(replace(replace(line[1], r" \d"=>""), r"\d"=>""), r"\d"=>""), ['_', ' '])
            
            line = split(line[1], "_")
            
            if (speciesField ∈ twoName)
                speciesIndex = line[3]
            else
                speciesIndex = line[2]
            end
            if (speciesField == "ZS")
                zs_temp[parse(Int, speciesIndex)] = val
            elseif (speciesField != "IR_EXP")
                speciesField = Symbol(speciesField)
                getfield(inputMTGLF, speciesField)[parse(Int,speciesIndex)] = val
            else
                append!(irexp, Int(val))
            end
            # The ALPHA vector is only used for the input.profile method apparently. I cannot find it anywhere in TGLFEP besides
            # the method where it is listed in input.profile explicitly.
        else # Non-Vectors/Matrices

            field = line[1]
        
            if (contains(line[2], 'T') || contains(line[2], 'F'))
                val = lowercase(strip(line[2], ['\'','.'])) == "t"
            elseif (!contains(line[2], '.'))
                val = parse(Int64, line[2])
            else
                val = parse(Float64, line[2])
            end
            if (field ∈ delFields) continue
            else
                field = Symbol(field)
                setfield!(inputMTGLF, field, val)
            end
        end
    end
    # Broadcast per-species ZS values to all radial grid points
    for ir in 1:nr
        inputMTGLF.ZS[ir, :] = zs_temp
    end
    # That's all for now folks!
    return inputMTGLF, irexp
end
"""
    read_input_profile(filename::String) -> profile

Read Fortran/GACODE `input.profile` or `dump.profile` (fixed columns, NR radial points).
"""
function read_input_profile(filename::String)
    lines = readlines(filename)
    @assert length(lines) >= 5 "profile file too short: $filename"

    sign_bt = parse(Float64, strip(split(lines[1])[1]))
    sign_it = parse(Float64, strip(split(lines[2])[1]))
    nr = parse(Int, strip(split(lines[3])[1]))
    ns = parse(Int, strip(split(lines[4])[1]))
    geometry_flag = parse(Int, strip(split(lines[5])[1]))

    prof = profile{Float64}(nr, ns)
    prof.SIGN_BT = sign_bt
    prof.SIGN_IT = sign_it
    prof.NR = nr
    prof.NS = ns
    prof.GEOMETRY_FLAG = geometry_flag
    prof.ROTATION_FLAG = 0  # matches Fortran EXPRO path (rotation_flag = 0)
    prof.ZS = fill(NaN, nr, ns)
    prof.MASS = fill(NaN, ns)
    prof.AS = fill(NaN, nr, ns)
    prof.TAUS = fill(NaN, nr, ns)
    prof.RLNS = fill(NaN, nr, ns)
    prof.RLTS = fill(NaN, nr, ns)
    prof.VPAR = zeros(nr, ns)
    prof.VPAR_SHEAR = zeros(nr, ns)
    prof.AS[:, 1] .= 1.0
    prof.TAUS[:, 1] .= 1.0

    function read_floats!(target, i0::Int)
        vals = Float64[]
        i = i0
        while i <= length(lines) && length(vals) < nr
            s = strip(lines[i])
            i += 1
            isempty(s) && continue
            startswith(s, "#") && continue
            startswith(s, "---") && continue
            x = tryparse(Float64, s)
            x === nothing && continue
            push!(vals, x)
        end
        @assert length(vals) == nr "expected $nr values near line $i0 in $filename, got $(length(vals))"
        target .= vals
        return i
    end

    ispecies = 0
    i = 6
    while i <= length(lines)
        s = strip(lines[i])
        if startswith(s, "# electron species")
            ispecies = 1
            i += 1
            prof.ZS[:, 1] .= parse(Float64, strip(split(lines[i])[1]))
            prof.MASS[1] = parse(Float64, strip(split(lines[i + 1])[1]))
            i += 2
            continue
        elseif startswith(s, "# ion species")
            ispecies += 1
            i += 1
            prof.ZS[:, ispecies] .= parse(Float64, strip(split(strip(lines[i]))[1]))
            prof.MASS[ispecies] = parse(Float64, strip(split(strip(lines[i + 1]))[1]))
            i += 2
            continue
        elseif startswith(s, "# Geometry")
            ispecies = 0
            i += 1
            continue
        elseif startswith(s, "#") && ispecies > 0
            i += 1
            if occursin("density: as", s)
                i = read_floats!(view(prof.AS, :, ispecies), i)
            elseif occursin("temperature: taus", s)
                i = read_floats!(view(prof.TAUS, :, ispecies), i)
            elseif occursin("density gradients: rlns", s)
                i = read_floats!(view(prof.RLNS, :, ispecies), i)
            elseif occursin("temperature gradients: rlts", s)
                i = read_floats!(view(prof.RLTS, :, ispecies), i)
            end
            continue
        elseif startswith(s, "#") && ispecies == 0
            i += 1
            if occursin("minor radius: rmin", s)
                i = read_floats!(prof.RMIN, i)
            elseif occursin("major radius: rmaj", s)
                i = read_floats!(prof.RMAJ, i)
            elseif occursin("safety factor: q", s)
                i = read_floats!(prof.Q, i)
            elseif occursin("magnetic shear: shear", s)
                i = read_floats!(prof.SHEAR, i)
            elseif occursin("q_prime", s)
                i = read_floats!(prof.Q_PRIME, i)
            elseif occursin("p_prime", s)
                i = read_floats!(prof.P_PRIME, i)
            elseif occursin("shift", s)
                i = read_floats!(prof.SHIFT, i)
            elseif occursin("elogation: kappa", s)
                i = read_floats!(prof.KAPPA, i)
            elseif occursin("shear in elogation: s_kappa", s)
                i = read_floats!(prof.S_KAPPA, i)
            elseif occursin("triangularity: delta", s)
                i = read_floats!(prof.DELTA, i)
            elseif occursin("shear in triangularity: s_delta", s)
                i = read_floats!(prof.S_DELTA, i)
            elseif occursin("squareness: zeta", s)
                i = read_floats!(prof.ZETA, i)
            elseif occursin("shear in squareness: s_zeta", s)
                i = read_floats!(prof.S_ZETA, i)
            elseif occursin("effective ion charge: zeff", s)
                i = read_floats!(prof.ZEFF, i)
            elseif occursin("betae", s)
                i = read_floats!(prof.BETAE, i)
            end
            continue
        end
        i += 1
    end

    prof.gammaE = fill(1.0e-7, nr)
    prof.gammap = fill(1.0e-7, nr)
    prof.omegaGAM = similar(prof.RMAJ, nr)
    for ir in 1:nr
        prof.omegaGAM[ir] = (1.0 / prof.RMAJ[ir]) * sqrt(1.0 + prof.TAUS[ir, 2]) / (1.0 + 1.0 / (2.0 * prof.Q[ir]))
    end
    prof.RHO_STAR = fill(0.001, nr)
    prof.OMEGA_TAE = sqrt.(2.0 ./ prof.BETAE) ./ 2.0 ./ prof.Q ./ prof.RMAJ
    prof.B_UNIT = fill(1.0, nr)
    prof.IRS = 2
    prof.N_ION = max(ns - 1, 1)
    return prof
end

"""
    readprofile(filename::String) -> profile

Alias for [`read_input_profile`](@ref): read an `input.profile`-style file into a
[`profile`](@ref) struct.
"""
readprofile(filename::String) = read_input_profile(filename)

"""Map GACODE `is_EP` (ions only) to EXPRO species index when species 1 is electron."""
expro_species_for_gacode_is_ep(is_EP::Integer) = is_EP + 1

"""Build EXPRO dict for `save_EXPRO` from a `profile` (3-species Fortran layout)."""
function expro_dict_from_profile(prof::profile)
    nr, ns = prof.NR, prof.NS
    a_m = prof.RMIN[end]
    extraEP = Dict{String, Any}("NR" => nr, "NS" => ns)
    for is in 1:ns
        extraEP["DENS_$is"] = copy(prof.AS[:, is])
        extraEP["TEMP_$is"] = copy(prof.TAUS[:, is])
        # Fortran: EXPRO_dlnnidr = RLNS/a_m (TGLFEP_read_EXPRO.f90); not raw TGLF RLNS.
        extraEP["DLNNDR_$is"] = prof.RLNS[:, is] ./ a_m
        extraEP["DLNTDR_$is"] = prof.RLTS[:, is] ./ a_m
    end
    extraEP["CS"] = fill(1.0e7, nr)
    extraEP["RMIN"] = copy(prof.RMIN)
    extraEP["gammaE"] = copy(prof.gammaE)
    extraEP["gammap"] = copy(prof.gammap)
    extraEP["omegaGAM"] = copy(prof.omegaGAM)
    return extraEP
end

"""
    ir_exp_from_scan(nr::Int, irs::Int, scan_n::Int) -> Vector{Int}

Compute the `scan_n` experimental radial indices for a scan: evenly spaced from
the start index `irs` to `nr`. With `scan_n == 1` the single index `irs` is
returned.
"""
function ir_exp_from_scan(nr::Int, irs::Int, scan_n::Int)
    ir_exp = Vector{Int}(undef, scan_n)
    for i in 1:scan_n
        if scan_n != 1
            ir_exp[i] = irs + floor(Int, (i - 1) * (nr - irs) / (scan_n - 1))
        else
            ir_exp[i] = irs
        end
    end
    return ir_exp
end

function _read_gacode_header_int(lines::Vector{String}, tag::AbstractString)
    i = findfirst(l -> strip(l) == "# $tag", lines)
    i === nothing && error("header # $tag not found")
    return parse(Int, strip(lines[i + 1]))
end

function _read_gacode_header_float(lines::Vector{String}, tag::AbstractString)
    i = findfirst(l -> startswith(strip(l), "# $tag"), lines)
    i === nothing && error("header # $tag not found")
    return parse(Float64, strip(lines[i + 1]))
end

function _read_gacode_header_vector(lines::Vector{String}, tag::AbstractString, n::Int)
    i = findfirst(l -> strip(l) == "# $tag", lines)
    i === nothing && error("header # $tag not found")
    vals = parse.(Float64, split(strip(lines[i + 1])))
    length(vals) >= n || error("header #$tag: expected >= $n values, got $(length(vals))")
    return vals[1:n]
end

"""
    profile_from_gacode(gacode_file; is_ep=2, tglfep_nion=2, q_scale=1.0, rotation_flag=0)

Build a `profile` struct from `input.gacode` / `dump.gacode`, mirroring Fortran
`expro_read` + `expro_compute_derived` + `TGLFEP_read_EXPRO.f90` (species, geometry,
gradients). Intended for `INPUT_PROFILE_METHOD == 2` parity with Fortran TGLF-EP.
"""
function profile_from_gacode(
    gacode_file::AbstractString;
    is_ep::Int=2,
    tglfep_nion::Int=2,
    q_scale::Float64=1.0,
    rotation_flag::Int=0,
)
    lines = readlines(gacode_file)
    nr = _read_gacode_header_int(lines, "nexp")
    nion = _read_gacode_header_int(lines, "nion")
    masse = _read_gacode_header_float(lines, "masse")
    ion_mass = _read_gacode_header_vector(lines, "mass", nion)
    ze = _read_gacode_header_float(lines, "ze")
    ion_z = _read_gacode_header_vector(lines, "z", nion)
    torfluxa = _read_gacode_header_float(lines, "torfluxa")

    rmin = read_gacode_scalar_field(gacode_file, "rmin", nr)
    rmaj = read_gacode_scalar_field(gacode_file, "rmaj", nr)
    q = read_gacode_scalar_field(gacode_file, "q", nr)
    # w0 (toroidal rotation) is optional: some input.gacode files omit it. With rotation_flag=0
    # a missing w0 is equivalent to zero rotation (only feeds gammaE/gammap, unused unless
    # ROTATIONAL_SUPPRESSION_FLAG==1). Default to zeros so such cases run.
    w0 = read_gacode_scalar_field(gacode_file, "w0", nr; default=0.0)
    ne = read_gacode_scalar_field(gacode_file, "ne", nr)
    te = read_gacode_scalar_field(gacode_file, "te", nr)
    ptot = read_gacode_scalar_field(gacode_file, "ptot", nr)
    zeff = read_gacode_scalar_field(gacode_file, "z_eff", nr)
    kappa = read_gacode_scalar_field(gacode_file, "kappa", nr)
    delta = read_gacode_scalar_field(gacode_file, "delta", nr)
    zeta = read_gacode_scalar_field(gacode_file, "zeta", nr)
    rho_tor = read_gacode_scalar_field(gacode_file, "rho", nr)

    ni = [read_gacode_ion_field(gacode_file, "ni", i, nr) for i in 1:nion]
    ti = [read_gacode_ion_field(gacode_file, "ti", i, nr) for i in 1:nion]

    1 <= is_ep <= nion || error("is_ep=$is_ep out of range 1..$nion")
    ns = tglfep_nion + 1

    # TGLFEP_read_EXPRO axis fix
    rmin_work = rmin .+ 1.0e-6
    a_m = rmin_work[end]
    eps_n = 1.0e-30

    # expro_compute_derived: b_unit from torflux
    torflux = torfluxa .* rho_tor .^ 2
    bunit = expro_bound_deriv(torflux, 0.5 .* rmin_work .^ 2)

    # Gradients (expro_util.f90)
    dlnnedr = expro_bound_deriv(-log.(max.(ne, eps_n)), rmin_work)
    dlntedr = expro_bound_deriv(-log.(max.(te, eps_n)), rmin_work)
    dlnnidr = Vector{Vector{Float64}}(undef, nion)
    dlntidr = Vector{Vector{Float64}}(undef, nion)
    for i in 1:nion
        if minimum(ni[i]) > 0 && minimum(ti[i]) > 0
            dlnnidr[i] = expro_bound_deriv(-log.(ni[i]), rmin_work)
            dlntidr[i] = expro_bound_deriv(-log.(ti[i]), rmin_work)
        else
            dlnnidr[i] = zeros(nr)
            dlntidr[i] = zeros(nr)
        end
    end
    dlnptotdr = minimum(ptot) > 0 ?
        expro_bound_deriv(-log.(max.(ptot, eps_n)), rmin_work) :
        zeros(nr)

    # s, drmaj, shape s-derivatives
    temp = expro_bound_deriv(log.(abs.(q)), rmin_work)
    shear_phys = rmin_work .* temp
    drmaj = expro_bound_deriv(rmaj, rmin_work)
    skappa = (rmin_work ./ kappa) .* expro_bound_deriv(kappa, rmin_work)
    sdelta = rmin_work .* expro_bound_deriv(delta, rmin_work)
    szeta = rmin_work .* expro_bound_deriv(zeta, rmin_work)

    # cs, rhos (expro_util CGS → m)
    k_erg = 1.6022e-12
    mp_g = 2.0 * 1.6726e-24
    e_cgs = 4.8032e-10
    c_cgs = 2.9979e10
    cs = sqrt.(k_erg .* (1e3 .* te) ./ (2.0 * mp_g)) ./ 1e2
    rhos = cs ./ (e_cgs .* (1e4 .* bunit) ./ (2.0 * mp_g .* c_cgs)) ./ 1e2

    w0p = expro_bound_deriv(w0, rmin_work)
    gamma_e_phys = -rmin_work ./ q .* w0p
    gamma_p_phys = -rmaj .* w0p

    # TGLFEP quasineutrality (thermal ions only)
    sum0 = zeros(nr)
    for i in 1:tglfep_nion
        i == is_ep && continue
        sum0 .+= ion_z[i] .* ni[i] ./ ne
    end
    a_qn = (1.0 .- ion_z[is_ep] .* ni[is_ep] ./ ne) ./ sum0
    ni_qn = copy(ni)
    for i in 1:tglfep_nion
        i == is_ep && continue
        ni_qn[i] = a_qn .* ni[i]
    end

    dlnnidr_ep = copy(dlnnidr[is_ep])
    dlnnidr_ep .= max.(dlnnidr_ep, 1.0)

    prof = profile{Float64}(nr, ns)
    prof.SIGN_BT = -1.0
    prof.SIGN_IT = -1.0
    prof.NR = nr
    prof.NS = ns
    prof.GEOMETRY_FLAG = 1
    prof.ROTATION_FLAG = rotation_flag
    prof.N_ION = tglfep_nion
    # A_QN is radially varying in Fortran; profile struct stores scalar — skip here.

    prof.ZS = fill(NaN, nr, ns)
    prof.MASS = fill(NaN, ns)
    prof.AS = fill(NaN, nr, ns)
    prof.TAUS = fill(NaN, nr, ns)
    prof.RLNS = fill(NaN, nr, ns)
    prof.RLTS = fill(NaN, nr, ns)
    prof.VPAR = zeros(nr, ns)
    prof.VPAR_SHEAR = zeros(nr, ns)

    prof.ZS[:, 1] .= ze
    prof.MASS[1] = masse / 2.0
    prof.AS[:, 1] .= 1.0
    prof.TAUS[:, 1] .= 1.0
    prof.RLNS[:, 1] = dlnnedr .* a_m
    prof.RLTS[:, 1] = dlntedr .* a_m

    for i in 1:tglfep_nion
        s = i + 1
        prof.ZS[:, s] .= ion_z[i]
        prof.MASS[s] = ion_mass[i] / 2.0
        prof.AS[:, s] = ni_qn[i] ./ ne
        prof.TAUS[:, s] = ti[i] ./ te
        prof.RLNS[:, s] = dlnnidr[i] .* a_m
        prof.RLTS[:, s] = dlntidr[i] .* a_m
    end
    ep_slot = is_ep + 1
    prof.RLNS[:, ep_slot] = dlnnidr_ep .* a_m
    prof.RLTS[:, ep_slot] = dlntidr[is_ep] .* a_m

    if rotation_flag == 1
        for s in 1:ns
            prof.VPAR[:, s] = w0 .* rmaj ./ cs
            prof.VPAR_SHEAR[:, s] = -rmaj .* w0p .* a_m ./ cs
        end
    end

    rmin_norm = rmin_work ./ a_m
    rmaj_norm = rmaj ./ a_m
    q_out = q_scale .* q
    beta_unit = 2.0 * 4 * pi * 1.0e-7 .* ptot ./ (bunit .^ 2)
    betae = beta_unit .* (1.6022e3 .* ne .* te ./ ptot)

    prof.RMIN = rmin_norm
    prof.RMAJ = rmaj_norm
    prof.Q = q_out
    prof.SHEAR = shear_phys
    prof.SHIFT = drmaj
    prof.Q_PRIME = (q_out ./ rmin_norm) .^ 2 .* shear_phys
    prof.P_PRIME = -abs.(q_out) ./ rmin_norm .* beta_unit ./ (8 * pi) .* dlnptotdr .* a_m
    prof.KAPPA = kappa
    prof.S_KAPPA = skappa
    prof.DELTA = delta
    prof.S_DELTA = sdelta
    prof.ZETA = zeta
    prof.S_ZETA = szeta
    prof.ZEFF = zeff
    prof.BETAE = betae
    prof.RHO_STAR = rhos ./ a_m
    prof.OMEGA_TAE = sqrt.(2.0 ./ betae) ./ 2.0 ./ q_out ./ rmaj_norm
    taus2 = prof.TAUS[:, 2]
    prof.omegaGAM = (1.0 ./ rmaj_norm) .* sqrt.(1.0 .+ taus2) ./ (1.0 .+ 1.0 ./ (2.0 .* q_out))
    prof.gammaE = (a_m ./ cs) .* gamma_e_phys
    prof.gammap = (a_m ./ cs) .* gamma_p_phys
    prof.B_UNIT = bunit

    return prof
end

"""
    expro_vectors_from_gacode(prof, gacode_file, is_EP_gacode)

EP α-postprocessing vectors from `input.gacode` only (no `input.EXPRO` file).
`is_EP_gacode` is the GACODE ion index from `input.TGLFEP` (`IS_EP`).
"""
function expro_vectors_from_gacode(
    prof::profile,
    gacode_file::AbstractString,
    is_EP_gacode::Integer,
)
    nr = prof.NR
    rmin = read_gacode_scalar_field(gacode_file, "rmin", nr)
    rmin_work = rmin .+ 1.0e-6
    te = read_gacode_scalar_field(gacode_file, "te", nr)
    ni = read_gacode_ion_field(gacode_file, "ni", is_EP_gacode, nr)
    Ti = read_gacode_ion_field(gacode_file, "ti", is_EP_gacode, nr)
    dlnnidr, dlntidr = expro_log_gradients(ni, Ti, rmin_work)
    dlnnidr .= max.(dlnnidr, 1.0)

    # Match gacode expro_util.f90 exactly: EXPRO_cs = sqrt(k*(1e3*te)/(2*mp))/1e2 [m/s]
    # with mp the PROTON mass (2*mp = deuteron). The old code used mp_g = 2*1.6726e-24
    # (already the deuteron mass) inside /(2*mp_g) — an extra 1/sqrt(2) — and then
    # converted to cm/s (x100), making F_REAL (and all printed kHz) 100/sqrt(2) = 70.7x
    # too large vs Fortran TGLFEP. Physics was unaffected (F_REAL only scales printouts).
    k_erg = 1.6022e-12
    mp_g = 1.6726e-24
    cs = sqrt.(k_erg .* (1e3 .* te) ./ (2.0 * mp_g)) ./ 1e2  # m/s, = EXPRO_cs

    return (
        ni=ni,
        Ti=Ti,
        dlnnidr=dlnnidr,
        dlntidr=dlntidr,
        cs=cs,
        rmin_ex=rmin_work,
        gammaE=prof.gammaE,
        gammap=prof.gammap,
        omegaGAM=prof.omegaGAM,
    )
end

"""
    preprocess_gacode_inputs(gacode_file, tglfep_file)

Build in-memory `Options` and `profile` from `input.gacode` + `input.TGLFEP`.
No `dump.gacode`, `input.MTGLF`, or `input.EXPRO` required.

Returns `(Options, profile, expro_state)` where `expro_state` holds EP vectors for
α post-processing and `F_REAL` setup.
"""
function preprocess_gacode_inputs(
    gacode_file::AbstractString,
    tglfep_file::AbstractString;
    nmodes::Int = _nmodes_env(),
)
    @assert isfile(gacode_file) "missing gacode file: $gacode_file"
    @assert isfile(tglfep_file) "missing tglfep file: $tglfep_file"

    opts = readTGLFEP(tglfep_file, Int[]; nmodes=nmodes)
    prof = profile_from_gacode(gacode_file;
        is_ep=coalesce(opts.IS_EP, 2),
        tglfep_nion=opts.N_ION,
        q_scale=coalesce(opts.Q_SCALE, 1.0),
        rotation_flag=0)
    prof.IRS = opts.IRS
    prof.N_ION = opts.N_ION
    opts.IR_EXP = ir_exp_from_scan(prof.NR, prof.IRS, opts.SCAN_N)

    expro = expro_vectors_from_gacode(prof, gacode_file, opts.IS_EP)
    prof.gammaE = expro.gammaE
    prof.gammap = expro.gammap
    prof.omegaGAM = expro.omegaGAM

    return opts, prof, expro
end

"""
    setup_gacode_file_inputs(gacode_file, out_dir; tglfep_file, is_ep, ...)

Like `setup_fortran_file_inputs`, but build the profile from `input.gacode`
instead of a pre-generated `dump.profile`.
"""
function setup_gacode_file_inputs(
    gacode_file::AbstractString,
    out_dir::AbstractString;
    tglfep_file::Union{Nothing,String}=nothing,
    nmodes::Int = _nmodes_env(),
    kwargs...,
)
    tglfep_file = something(tglfep_file, joinpath(dirname(gacode_file), "input.TGLFEP"))
    @assert isfile(gacode_file) "missing $gacode_file"
    @assert isfile(tglfep_file) "missing $tglfep_file"
    mkpath(out_dir)

    opts = readTGLFEP(tglfep_file, Int[]; nmodes=nmodes)
    prof = profile_from_gacode(gacode_file;
        is_ep=coalesce(opts.IS_EP, 2),
        tglfep_nion=opts.N_ION,
        q_scale=coalesce(opts.Q_SCALE, 1.0),
        rotation_flag=0,
        kwargs...)
    prof.IRS = opts.IRS
    prof.N_ION = opts.N_ION
    ir_exp = ir_exp_from_scan(prof.NR, prof.IRS, opts.SCAN_N)

    save_MTGLF(prof, ir_exp, joinpath(out_dir, "input.MTGLF"))
    save_EXPRO(expro_dict_from_profile(prof), joinpath(out_dir, "input.EXPRO"))
    dest_tglfep = joinpath(out_dir, "input.TGLFEP")
    if abspath(tglfep_file) != abspath(dest_tglfep)
        isfile(dest_tglfep) && rm(dest_tglfep; force=true)
        symlink(abspath(tglfep_file), dest_tglfep)
    end
    return prof, ir_exp
end

"""Write `input.MTGLF`, `input.EXPRO` from Fortran `dump.profile` + `input.TGLFEP`."""
function setup_fortran_file_inputs(case_dir::AbstractString, out_dir::AbstractString;
        tglfep_file::Union{Nothing,String}=nothing, nmodes::Int = _nmodes_env())
    profile_file = joinpath(case_dir, "dump.profile")
    tglfep_file = something(tglfep_file, joinpath(case_dir, "input.TGLFEP"))
    @assert isfile(profile_file) "missing $profile_file"
    @assert isfile(tglfep_file) "missing $tglfep_file"
    mkpath(out_dir)

    prof = read_input_profile(profile_file)
    opts = readTGLFEP(tglfep_file, Int[]; nmodes=nmodes)
    prof.IRS = opts.IRS
    prof.N_ION = opts.N_ION
    ir_exp = ir_exp_from_scan(prof.NR, prof.IRS, opts.SCAN_N)

    prof.ROTATION_FLAG = coalesce(prof.ROTATION_FLAG, 0)
    save_MTGLF(prof, ir_exp, joinpath(out_dir, "input.MTGLF"))
    save_EXPRO(expro_dict_from_profile(prof), joinpath(out_dir, "input.EXPRO"))
    dest_tglfep = joinpath(out_dir, "input.TGLFEP")
    if abspath(tglfep_file) != abspath(dest_tglfep)
        if isfile(dest_tglfep) || islink(dest_tglfep)
            rm(dest_tglfep; force=true)
        end
        symlink(abspath(tglfep_file), dest_tglfep)
    end
    return prof, ir_exp
end

"""
readTGLFEP extracts the values needed from the input.TGLFEP file

Inputs: filename

Outputs: Options struct
"""
function readTGLFEP(filename::String, ir_exp::Vector{Int64}; nmodes::Int = _nmodes_env())
    open(filename)
    lines = readlines(filename)

    # Required values to extract BEFORE assignment:
    nscan_in = -1
    widthin = -1
    ky_model = -1 # For assigning n_toroidal
    process_in = -1 # For assigning nn
    threshold_flag = -1 # For assigning nn
    
    for line in lines[1:length(lines)]

        # This would need to be adjusted for WIDTH_IN_FLAG = false, which is
        # not a covered case in TGLF-EP (but is in OMFIT?)

        if contains(line, "") && !contains(line, " ") continue end
        line = split(line, "\n")

        if contains(line[1], "SCAN_N")
            line = split(line[1])
            nscan_in = parse(Int, strip(line[1]))
        elseif contains(line[1], "WIDTH_IN_FLAG")
            line = split(line[1])
            widthin = parse(Bool, strip(line[1], '.'))
        elseif contains(line[1], "KY_MODEL")
            line = split(line[1])
            ky_model = parse(Int, strip(line[1]))
        elseif contains(line[1], "PROCESS_IN")
            line = split(line[1])
            process_in = parse(Int, strip(line[1]))
        elseif contains(line[1], "THRESHOLD_FLAG")
            line = split(line[1])
            threshold_flag = parse(Int, strip(line[1]))
        end

    end
    
    @assert nscan_in != -1 "SCAN_N not found in TGLFEP input"
    @assert widthin != -1 "WIDTH_IN_FLAG not found in TGLFEP input"
    @assert ky_model != -1 "KY_MODEL not found in TGLFEP input"
    @assert process_in != -1 "PROCESS_IN not found in TGLFEP input"
    # THRESHOLD_FLAG is only read by the Fortran for process_in in {4,5,6}; the spectrum
    # (3) and ky/TM (0,1,2) inputs omit it. Require it only when it is actually used.
    if process_in == 4 || process_in == 5 || process_in == 6
        @assert threshold_flag != -1 "THRESHOLD_FLAG not found in TGLFEP input"
    elseif threshold_flag == -1
        threshold_flag = 0
    end

    # Now, nscan_in, widthin are assigned and ready.

    #=  # Assign nn:
    if ((process_in == 4) || (process_in == 5))
        if (threshold_flag == 0)
            nn = 5
        else
            nn = 15
        end
    end
    =#
    # See TGLFEP_interface.f90:
    jtscale_max = 1
    # nmodes (number of eigenmodes kept) is a user input: `nmodes` kwarg / TJLFEP_NMODES env
    # (default 4). Used to size the keep/reject vectors in the Options constructor below.

    # NR is derived from the profile! This means that in order to read TJLFEP in Julia, we have to call the previous inputMTGLF
    # function call's NR. It will now be a required input for this function. 
    nr = 201
    nn = 5
    inputTJLFEP = Options{Float64}(nscan_in, widthin, nn, nr, jtscale_max, nmodes)
    inputTJLFEP.IR_EXP = ir_exp
    inputTJLFEP.NMODES = nmodes
   


    for line in lines[1:length(lines)]
        line = replace(line, "   "=>" ") # Don't ask lol
        line = replace(line, "  "=>" ")

        line = split(line, "\n")
        if contains(line[1], "") && !contains(line[1], " ") continue end # && !contains(line[1], ".") may be needed for if WIDTH_IN can change in the vector. I am yet to see a case of this.

        line = split(line[1], " ")
        # Now I will have all input parameters I want. A space must exist between the fields. It is better if it is just one space but can be UP TO a tab (3 spaces in VSCode where I am editing this).
        
        vecFields = ["WIDTH", "FACTOR"] # For now...
        if line[2] ∈ vecFields 
            #if ()
            field = Symbol(line[2])
            println("field: $field")
            getfield(inputTJLFEP, field) .= [parse(Float64,line[1])]
        else
            field = Symbol(line[2])
            if line[1][1] == '.'
                val = lowercase(strip(line[1], ['\'','.'])) == "true"
            elseif !contains(line[1], '.')
                val = parse(Int, line[1])
            else
                val = parse(Float64, line[1])
            end
            try
                if contains(string(field),"THETA_2_THRESH")
                    setfield!(inputTJLFEP, Symbol("THETA_SQ_THRESH"), val)
                else
                    setfield!(inputTJLFEP, field, val)
                end
            catch
                throw(error(field))
            end
        end
    end
    # This function is for the most part done. There should be consideration for default conditions however...
    # Something similar to the checkInput function created for TJLF but with the option to define a default or throw an error back if it's not populated. In fact, some things will have to be
    # unpopulated depending on whether WIDTH_IN_FLAG is true or false, e.g..

    # There are a few fields that need to be manually set. I will list them here:

    if (ky_model == 0)
        inputTJLFEP.NTOROIDAL = 4
    else
        inputTJLFEP.NTOROIDAL = 3
    end
        
    if (process_in == 4 || process_in == 5)
        inputTJLFEP.NN = nn
    end

    if (!inputTJLFEP.FACTOR_IN_PROFILE)
        inputTJLFEP.FACTOR = fill(inputTJLFEP.FACTOR_IN, nscan_in)
    end
    inputTJLFEP.FACTOR_MAX_PROFILE = inputTJLFEP.FACTOR

    # The spectrum (process_in=3) and ky/TM (0,1,2) inputs omit the threshold/reject/scan
    # fields that only the threshold searches (4/5/6) read. The Fortran leaves those module
    # variables at their (zero) defaults; mirror that here so the spectrum + width-scan paths
    # run without them. (4/5/6 always supply these, so this never overrides a real value.)
    # Float fields in `Options` initialize to NaN (not `missing`), so treat NaN as "unset" too.
    _needs_default(x) = x === missing || (x isa AbstractFloat && isnan(x))
    _default_if_missing!(o, f, v) = (_needs_default(getfield(o, f)) && setfield!(o, f, v))
    _default_if_missing!(inputTJLFEP, :SCAN_METHOD, 0)
    # process_in=3 does not read PPRIME_METHOD; the Fortran then hits the default branch
    # ("reverting to fixed p_prime"). Use 0 here so TJLF_map's else-branch sets the raw
    # fixed p_prime(ir), matching the Fortran.
    _default_if_missing!(inputTJLFEP, :PPRIME_METHOD, 0)
    _default_if_missing!(inputTJLFEP, :REJECT_I_PINCH_FLAG, 0)
    _default_if_missing!(inputTJLFEP, :REJECT_E_PINCH_FLAG, 0)
    _default_if_missing!(inputTJLFEP, :REJECT_TH_PINCH_FLAG, 0)
    _default_if_missing!(inputTJLFEP, :REJECT_EP_PINCH_FLAG, 0)
    _default_if_missing!(inputTJLFEP, :REJECT_TEARING_FLAG, 0)
    _default_if_missing!(inputTJLFEP, :ROTATIONAL_SUPPRESSION_FLAG, 0)
    _default_if_missing!(inputTJLFEP, :WRITE_WAVEFUNCTION, 0)
    _default_if_missing!(inputTJLFEP, :N_BASIS, 32)
    _default_if_missing!(inputTJLFEP, :QL_RATIO_THRESH, 0.0)
    _default_if_missing!(inputTJLFEP, :THETA_SQ_THRESH, 1.0e10)
    _default_if_missing!(inputTJLFEP, :Q_SCALE, 1.0)

    enforce_width_floor!(inputTJLFEP)   # ballooning-width physics floor (WIDTH_FLOOR = 0.5)
    return inputTJLFEP
end

"""
TJLF_map: This directly maps any needed inputs from the InputTJLFEP and profile structs that were obtained
from the input.MTGLF (or input.profile) and input.TGLFEP files in tjlfep_read_inputs.jl (above). This is done so
TJLF can be run very easily. The InputTJLF struct is defined above as needed in order to perform this.

inputs: InputTJLFEP from TGLFEP input file; profile from MTGLF profile file

Outputs: InputTJLF struct ready for usage in running TJLF. 
"""
#Temporary using statements:
#include("../tjlf-ep/TJLFEP.jl")
#using .TJLFEP

"""
    default_input_tjlf(::Type{T}, ns::Int, nky::Int) -> TJLF.InputTJLF{T}

Build a `TJLF.InputTJLF` pre-populated with the TGLF-EP default values the TJLFEP
path relies on. This reproduces exactly what the former
`TJLFEP.InputTJLF{T}(ns, nky, true)` default constructor produced once copied into
`TJLF.InputTJLF` by `convert_input`, so the consolidated single-type path is
numerically identical. Fields not set here keep the `TJLF.InputTJLF` constructor
defaults (NaN/sentinel arrays + `SHAPE_*` harmonics), exactly as the previous
convert path did. `USE_AVE_ION_GRID`/`FIND_EIGEN` are left at TJLF defaults (they
were `missing` in the old default and are always overwritten by `TJLFEP_ky` before
TJLF runs).
"""
function default_input_tjlf(::Type{T}, ns::Int, nky::Int) where {T<:Real}
    input = TJLF.InputTJLF{T}(ns, nky)
    input.NKY = nky
    input.UNITS = "GYRO"
    input.USE_BPER = false
    input.USE_BPAR = false
    input.USE_MHD_RULE = true
    input.USE_BISECTION = true
    input.USE_INBOARD_DETRAPPED = false
    input.NEW_EIKONAL = true
    input.FIND_WIDTH = false
    input.IFLUX = true
    input.ADIABATIC_ELEC = false
    input.SAT_RULE = 0
    input.NMODES = 2
    input.NWIDTH = 21
    input.NBASIS_MAX = 4
    input.NBASIS_MIN = 4
    input.NXGRID = 32
    input.KYGRID_MODEL = 0
    input.XNU_MODEL = 2
    input.VPAR_MODEL = 0
    input.IBRANCH = -1
    input.SIGN_BT = 1
    input.SIGN_IT = 1
    input.KY = 0.3
    input.VEXB_SHEAR = 0.0
    input.BETAE = 0.0
    input.XNUE = 0.0
    input.ZEFF = 1.0
    input.DEBYE = 0.0
    input.ALPHA_MACH = 0.0
    input.ALPHA_E = 1.0
    input.ALPHA_P = 1.0
    input.ALPHA_QUENCH = 0
    input.ALPHA_ZF = 1.0
    input.XNU_FACTOR = 1.0
    input.DEBYE_FACTOR = 1.0
    input.ETG_FACTOR = 1.25
    input.RLNP_CUTOFF = 18.0
    input.WIDTH = 1.65
    input.WIDTH_MIN = 0.3
    input.RMIN_LOC = 0.5
    input.RMAJ_LOC = 3.0
    input.ZMAJ_LOC = 0.0
    input.DRMINDX_LOC = 1.0
    input.DRMAJDX_LOC = 0.0
    input.DZMAJDX_LOC = 0.0
    input.Q_LOC = 2.0
    input.KAPPA_LOC = 1.0
    input.S_KAPPA_LOC = 16.0
    input.DELTA_LOC = 0.0
    input.S_DELTA_LOC = 0.0
    input.ZETA_LOC = 0.0
    input.S_ZETA_LOC = 0.0
    input.P_PRIME_LOC = 0.0
    input.Q_PRIME_LOC = 16.0
    input.BETA_LOC = 0.0
    input.KX0_LOC = 0.0
    input.DAMP_PSI = 0.0
    input.DAMP_SIG = 0.0
    input.WDIA_TRAPPED = 0.0
    input.PARK = 1.0
    input.GHAT = 1.0
    input.GCHAT = 1.0
    input.WD_ZERO = 0.1
    input.LINSKER_FACTOR = 0.0
    input.GRADB_FACTOR = 0.0
    input.FILTER = 0.0
    input.THETA_TRAPPED = 0.7
    input.SMALL = 1.0e-13
    input.USE_TRANSPORT_MODEL = true
    return input
end

"""
    TJLF_map(inputsEP::Options, inputsPR::profile; mode_in_override=nothing, ky_model_override=nothing, nky=12) -> TJLF.InputTJLF

Build a `TJLF.InputTJLF` for a single radial point from TJLFEP's
[`Options`](@ref) (scan control) and [`profile`](@ref) (plasma profiles),
mirroring the Fortran `TGLFEP` → TGLF variable mapping.

`mode_in_override` / `ky_model_override` force the drive selector and `ky`-grid
model (used by the spectrum diagnostic), and `nky` sets the `ky`-grid size.
"""
function TJLF_map(inputsEP::Options{T}, inputsPR::profile{T};
                  mode_in_override::Union{Nothing,Int} = nothing,
                  ky_model_override::Union{Nothing,Int} = nothing,
                  nky::Int = 12) where {T<:Real}
    # Access the fields like this:
    # inputsOptions = inputsEP.Options
    # profile = inputsEP.profile   
     #=
    # Temp Defs:
    color = 0
    kyhat_in = 3
    # Temp Struct Inputs:
    filename = "/Users/benagnew/TJLF.jl/outputs/tglfep_tests/input.MTGLF"
    temp = readMTGLF(filename)
    inputsPR = temp[1]
    irexp2 = temp[2]
    filename = "/Users/benagnew/gacode_add/sample-rundir_2/input.TGLFEP"
    inputsEP = readTGLFEP(filename, irexp2)
    inputsEP.IR = inputsEP.IR_EXP[color+1]
    inputsEP.MODE_IN = 2
    inputsEP.KY_MODEL = 3
    =#
    # The grid/scalefactor path (process_in=5) forces EP-drive-only (mode_in=2) and the
    # kwscale_scan ky model (ky_model=3). The spectrum path (process_in=3) needs to vary
    # mode_in over {1,2,4} and keep its own ky_model, so callers can override here.
    inputsEP.MODE_IN  = mode_in_override === nothing  ? 2 : mode_in_override
    inputsEP.KY_MODEL = ky_model_override === nothing ? 3 : ky_model_override

    # Okay finally I can do this lol:
    inputTJLF = default_input_tjlf(T, inputsPR.NS, nky) # TJLF.InputTJLF seeded with TGLF-EP defaults
    if (inputsEP.IR < 1 || inputsEP.IR > inputsPR.NR)
        println("ir isn't within range")
        return 1
    end
    inputTJLF.SIGN_BT = inputsPR.SIGN_BT
    inputTJLF.SIGN_IT = inputsPR.SIGN_IT

    inputTJLF.SAT_RULE = 0

    inputTJLF.NS = inputsPR.NS
    ns = inputsPR.NS
    is = inputsEP.IS_EP + 1
    inputsPR.IS = is  # needed: profile.IS is read in runTHD after mainsub returns; all inner threads write the same value (IS_EP is a fixed input), so no actual race
    ir = inputsEP.IR

    #TJLF deletes GEOMETRY_FLAG so this is redundant:
    # inputTJLF.GEOMETRY_FLAG = inputsEP.GEOMETRY_FLAG

    inputTJLF.ZS = inputsPR.ZS[ir, :]
    inputTJLF.MASS = inputsPR.MASS
    inputTJLF.AS = inputsPR.AS[ir, :] # Check read_inputs for these
    inputTJLF.TAUS = inputsPR.TAUS[ir, :]

    # Prevent 0/0 = NaN in matrix (pol = sum(zs^2 * as/taus)) for zero-density
    # fast species where both AS=0 and TAUS=0. AS=0 already zeroes contribution.
    for i in 1:ns
        if inputTJLF.TAUS[i] <= 0.0
            inputTJLF.TAUS[i] = 1.0 # species w/ TAUS=0 will have AS=0, so setting to 1 doesn't affect physics
        end
    end

    inputTJLF.ZS[1] = -1.0
    # Prevent ZS=0 for zero-density fast species: ZS=0 → bb=taus*mass*(ky/0)²=Inf
    # → FLR_Hn(Inf)=0 → all-zero hn matrix → SingularException in inv()
    for i in 2:ns
        if inputTJLF.ZS[i] == 0.0
            inputTJLF.ZS[i] = 1.0  # placeholder; AS=0 already nullifies contribution
        end
    end
    
    # SCAN_METHOD is not read by the Fortran for the spectrum (process_in=3) input, so it
    # can be `missing` here; default to 0 (no factor-driven EP scaling) in that case.
    scan_method_eff = coalesce(inputsEP.SCAN_METHOD, 0)
    inputsEP.FACTOR_MAX = 0.5 / abs(inputTJLF.ZS[is]*inputsPR.AS[ir, is])
    if (scan_method_eff == 2)
        inputsEP.FACTOR_MAX = 1.0E3
    end
    inputsEP.FACTOR_IN
    if (inputsEP.FACTOR_IN < 0)
        inputsEP.FACTOR_IN = 0
    end
    if (inputsEP.FACTOR_IN > inputsEP.FACTOR_MAX)
        inputsEP.FACTOR_IN = inputsEP.FACTOR_MAX
    end

    # Factor_in is used to scale only the ion after the energetic species:
    if (scan_method_eff == 1)
        inputTJLF.AS[is] = inputsPR.AS[ir, is]*inputsEP.FACTOR_IN
    end
    # I believe this is a correction for quasineutrality? AS is the ratio of density of the species to the electron

    sum0 = 0.0
    for i = 2:ns
        if (i != is) 
            sum0 = sum0 + inputTJLF.ZS[i]*inputTJLF.AS[i]
        end
    end

    A_QN = (1.0 - inputTJLF.ZS[is]*inputTJLF.AS[is]) / sum0  # local variable; avoids race on shared inputsPR.A_QN in parallel nkwf loop
    inputsPR.A_QN = A_QN  # keep field updated for callers that read it outside TJLF_map (e.g. diagnostics)

    for i = 2:ns
        if (i != is)
            inputTJLF.AS[i] = A_QN*inputTJLF.AS[i]  # reads local A_QN, not shared inputsPR.A_QN
        end
    end

    if (inputsEP.MODE_IN == 2) # EP drive only
        for i = 1:ns
            if (i != is)
                inputTJLF.RLNS[i] = 1.0e-6
                inputTJLF.RLTS[i] = 1.0e-6
            else
                inputTJLF.RLNS[i] = inputsPR.RLNS[ir, i]
                inputTJLF.RLTS[i] = inputsPR.RLTS[ir, i]
            end
        end
    else
        for i = 1:ns
            inputTJLF.RLNS[i] = inputsPR.RLNS[ir, i]
            inputTJLF.RLTS[i] = inputsPR.RLTS[ir, i]
        end
    end

    # EP species
    # etc...

    if (inputsEP.MODE_IN == 3)
        for i = is:ns
            inputTJLF.RLNS[i] = 1.0e-5
            inputTJLF.RLTS[i] = 1.0e-5
        end
    else
        inputTJLF.RLNS[is] = inputsPR.RLNS[ir, is]*inputsEP.SCAN_FACTOR
        inputTJLF.FILTER = 0.0
    end

    if (scan_method_eff == 2) # SCAN_METHOD 1 or 2 is chosen for whether you are applying the scaling to the density gradient or the density.
        inputTJLF.RLNS[is] = inputsEP.FACTOR_IN*inputsPR.RLNS[ir, is]
    end

    # Geometry: TJLF can only recognize GEOMETRY_FLAG == 1 so I will skip geoflag == 0 for now

    geometry_flag = coalesce(inputsPR.GEOMETRY_FLAG, 1)
    if geometry_flag == 0
        
    end
    if geometry_flag == 1
        # Fortran read_EXPRO sets rmin(:)=EXPRO_rmin/a_meters before tglf_map; FUSE stores RMIN in metres.
        r_over_a = inputsPR.RMIN[ir] / inputsPR.RMIN[end]
        inputTJLF.RMIN_LOC = r_over_a
        inputTJLF.RMAJ_LOC = inputsPR.RMAJ[ir]
        inputTJLF.ZMAJ_LOC = 0.0
        inputTJLF.DRMAJDX_LOC = inputsPR.SHIFT[ir]
        inputTJLF.DZMAJDX_LOC = 0.0
        inputTJLF.KAPPA_LOC = inputsPR.KAPPA[ir]
        inputTJLF.S_KAPPA_LOC = inputsPR.S_KAPPA[ir]
        inputTJLF.DELTA_LOC = inputsPR.DELTA[ir]
        inputTJLF.S_DELTA_LOC = inputsPR.S_DELTA[ir]
        inputTJLF.ZETA_LOC = inputsPR.ZETA[ir]
        inputTJLF.S_ZETA_LOC = inputsPR.S_ZETA[ir]
        inputTJLF.Q_LOC = abs(inputsPR.Q[ir])
        inputTJLF.Q_PRIME_LOC = inputsPR.Q_PRIME[ir]

        sum0 = 0
        for i = 1:ns
            sum0 = sum0 + inputTJLF.AS[i]*inputTJLF.TAUS[i]*(inputTJLF.RLNS[i]+inputTJLF.RLTS[i])
        end
        sum1 = 0
        for i = 1:ns
            sum1 = sum1 + inputsPR.AS[ir, i]*inputsPR.TAUS[ir, i]*(inputsPR.RLNS[ir, i]+inputsPR.RLTS[ir, i])
        end
        # sum2: EP species uses scaled TGLF values; all other species use original profile values
        # (pprime_method added 10-9-2024, EMB)
        sum2 = 0
        for i = 1:ns
            if i == is
                sum2 = sum2 + inputTJLF.AS[i]*inputTJLF.TAUS[i]*(inputTJLF.RLNS[i]+inputTJLF.RLTS[i])
            else
                sum2 = sum2 + inputsPR.AS[ir, i]*inputsPR.TAUS[ir, i]*(inputsPR.RLNS[ir, i]+inputsPR.RLTS[ir, i])
            end
        end
        pprime_method = coalesce(inputsEP.PPRIME_METHOD, 2)
        if pprime_method == 1
            inputTJLF.P_PRIME_LOC = inputsPR.P_PRIME[ir]                    # Fixed beta_* stabilization
        elseif pprime_method == 2
            inputTJLF.P_PRIME_LOC = inputsPR.P_PRIME[ir]*sum0/sum1          # beta_* stabilization matches kinetic drives
        elseif pprime_method == 3
            inputTJLF.P_PRIME_LOC = inputsPR.P_PRIME[ir]*sum2/sum1          # beta_* stabilization preserves thermal piece, varies with EP
        else
            println("Invalid PPRIME_METHOD, reverting to fixed p_prime")
            inputTJLF.P_PRIME_LOC = inputsPR.P_PRIME[ir]
        end
        
    end

    if coalesce(inputsPR.ROTATION_FLAG, 0) == 1
        for i = 1:ns
            inputTJLF.VPAR[i] = inputsPR.VPAR[ir, i]
            inputTJLF.VPAR_SHEAR[i] = inputsPR.VPAR_SHEAR[ir, i]
        end
    else
        for i = 1:ns
            inputTJLF.VPAR[i] = 0.0
            inputTJLF.VPAR_SHEAR[i] = 0.0
        end
    end

    inputTJLF.USE_BPER = true
    inputTJLF.USE_BPAR = false

    inputTJLF.BETAE = inputsPR.BETAE[ir]
    inputTJLF.XNUE = 0.0
    inputTJLF.ZEFF = inputsPR.ZEFF[ir]


    if (inputsEP.MODE_IN == 4)
        inputTJLF.FILTER = 2.0
    end

    kym = inputsEP.KY_MODEL
    if (kym == 0)
        inputTJLF.KY = 0.01*inputsEP.NTOROIDAL
    elseif (kym == 1)
        inputTJLF.KY = inputsEP.NTOROIDAL*inputTJLF.Q_LOC/inputTJLF.RMIN_LOC*inputsPR.RHO_STAR[ir]
    elseif (kym == 2)
        # inputTJLF.KY = inputsEP.NTOROIDAL*0.1*abs(inputTJLF.ZS[is])/sqrt(inputTJLF.MASS[is]*inputTJLF.TAUS[is])
        inputTJLF.KY = inputsEP.NTOROIDAL*0.1*(inputTJLF.ZS[is])/sqrt(inputTJLF.MASS[is]*inputTJLF.TAUS[is])
    elseif (kym == 3)
        # This depends on a previous definition in kwscale_scan...
        # inputTJLF.KY = inputsEP.KYHAT_IN*abs(inputTJLF.ZS[is])/sqrt(inputTJLF.MASS[is]*inputTJLF.TAUS[is])
        inputTJLF.KY = inputsEP.KYHAT_IN*(inputTJLF.ZS[is])/sqrt(inputTJLF.MASS[is]*inputTJLF.TAUS[is])
    end

    # This is one of the only things that is ran to for inputTJLF:
    inputsEP.FREQ_AE_UPPER = -abs(inputsPR.omegaGAM[ir])
    if coalesce(inputsEP.ROTATIONAL_SUPPRESSION_FLAG, 0) == 1
        inputsEP.GAMMA_THRESH_MAX = abs(inputsPR.gammap[ir]) * 2.0 * (min(1.0 - r_over_a, r_over_a) / inputsPR.RMAJ[ir])
        inputsEP.GAMMA_THRESH = 0.15 * abs(inputsPR.gammaE[ir] / inputsPR.SHEAR[ir])   # Bass PoP 2017 flow-shear suppression of AEs
        inputsEP.GAMMA_THRESH = min(inputsEP.GAMMA_THRESH, inputsEP.GAMMA_THRESH_MAX)
    else
        
        inputsEP.GAMMA_THRESH = 1.0e-7
        # inputsEP.GAMMA_THRESH_MAX = 1.0e-7
    end

    debug_dump_tglf_map(inputsEP, inputsPR, inputTJLF)
    return inputTJLF
end
"""
    read_gacode_scalar_field(path, field, nr; default=nothing) -> Vector{Float64}

Read a single-column GACODE `dump.gacode` block (`ne`, `te`, `rmin`, …).

If the block is absent and `default` is given (not `nothing`), return `fill(default, nr)`
instead of erroring. Used for optional fields like `w0` (toroidal rotation), which some
`input.gacode` files omit; with `rotation_flag = 0` a missing `w0` is equivalent to zero
rotation (it only feeds `gammaE`/`gammap`, unused unless `ROTATIONAL_SUPPRESSION_FLAG == 1`).
"""
function read_gacode_scalar_field(path::AbstractString, field::AbstractString, nr::Integer; default=nothing)
    lines = readlines(path)
    header = "# $field |"
    start_i = findfirst(l -> startswith(strip(l), header), lines)
    if start_i === nothing
        default === nothing && error("block $header not found in $path")
        @info "GACODE block $header not found in $path; defaulting to $default (nr=$nr)"
        return fill(Float64(default), nr)
    end
    vals = fill(NaN, nr)
    for line in lines[(start_i + 1):end]
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && break
        parts = split(s)
        length(parts) < 2 && continue
        ir = parse(Int, parts[1])
        1 <= ir <= nr || continue
        vals[ir] = parse(Float64, parts[2])
    end
    any(isnan, vals) && error("incomplete $field in $path")
    return vals
end

"""
    expro_bound_deriv(f, r) -> Vector{Float64}

Match GACODE `expro_util.f90::bound_deriv` (Lagrange derivative on uniform radial grid).
"""
function expro_bound_deriv(f::AbstractVector{<:Real}, r::AbstractVector{<:Real})
    n = length(f)
    n == length(r) || error("expro_bound_deriv: length(f)=$(length(f)) != length(r)=$(length(r))")
    df = Vector{Float64}(undef, n)
    f = Float64.(f)
    r = Float64.(r)
    for i in 1:n
        if i == 1
            ra, r1, r2, r3 = r[1], r[1], r[2], r[3]
            f1, f2, f3 = f[1], f[2], f[3]
        elseif i == n
            ra, r1, r2, r3 = r[n], r[n - 2], r[n - 1], r[n]
            f1, f2, f3 = f[n - 2], f[n - 1], f[n]
        else
            ra, r1, r2, r3 = r[i], r[i - 1], r[i], r[i + 1]
            f1, f2, f3 = f[i - 1], f[i], f[i + 1]
        end
        df[i] = ((ra - r1) + (ra - r2)) / (r3 - r1) / (r3 - r2) * f3 +
                ((ra - r1) + (ra - r3)) / (r2 - r1) / (r2 - r3) * f2 +
                ((ra - r2) + (ra - r3)) / (r1 - r2) / (r1 - r3) * f1
    end
    return df
end

"""`dlnnidr = -d(ln n)/dr` and `dlntidr = -d(ln T)/dr` as in `expro_compute_derived`."""
function expro_log_gradients(ni::AbstractVector, ti::AbstractVector, rmin::AbstractVector)
    nr = length(ni)
    length(ti) == nr == length(rmin) || error("ni/ti/rmin length mismatch")
    eps_n = 1.0e-30
    dlnnidr = expro_bound_deriv(-log.(max.(ni, eps_n)), rmin)
    dlntidr = expro_bound_deriv(-log.(max.(ti, eps_n)), rmin)
    return dlnnidr, dlntidr
end

"""
    read_gacode_ion_field(path, field, ion_index, nr) -> Vector{Float64}

Read one radial column from a GACODE `dump.gacode` block (`ni`, `ti`, …).
`ion_index` is 1..nion (GACODE ion index, not EXPRO species).
"""
function read_gacode_ion_field(path::AbstractString, field::AbstractString, ion_index::Integer, nr::Integer)
    lines = readlines(path)
    header = "# $field |"
    # Require "|" so "# nion" does not match field "ni".
    start_i = findfirst(l -> startswith(strip(l), header), lines)
    start_i === nothing && error("block $header not found in $path")
    vals = fill(NaN, nr)
    for line in lines[(start_i + 1):end]
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && break
        parts = split(s)
        length(parts) < ion_index + 1 && continue
        ir = parse(Int, parts[1])
        1 <= ir <= nr || continue
        vals[ir] = parse(Float64, parts[ion_index + 1])
    end
    if any(isnan, vals)
        error("incomplete $field ion $ion_index in $path")
    end
    return vals
end

"""
    read_expro_for_alpha(expro_file, prof, is_EP_gacode; gacode_file=nothing)

EXPRO vectors for α post-processing matching Fortran `TGLFEP_read_EXPRO`:

- GACODE `is_EP` → fast-ion column in `dump.gacode` (`ni`, `ti` in 10^19/m^3 and keV).
- `dlnnidr`, `dlntidr` from `expro_bound_deriv(-log(n), rmin)` (not profile `RLNS`).
- `max(dlnnidr, 1.0)` on the EP species only.
"""
function read_expro_for_alpha(
    expro_file::AbstractString,
    prof::profile,
    is_EP_gacode::Integer;
    gacode_file::Union{Nothing,String} = nothing,
)
    nr = prof.NR
    a_m = prof.RMIN[end]
    expro_is = expro_species_for_gacode_is_ep(is_EP_gacode)
    1 <= expro_is <= prof.NS || error("is_EP=$is_EP_gacode -> EXPRO species $expro_is out of range NS=$(prof.NS)")

    # readEXPRO only exposes the first 4 species selectors (ni1..ni4), but its per-species
    # ni/Ti are DISCARDED below (recomputed from gacode_file, or prof.AS/TAUS). The values we
    # keep (cs, rmin_ex, gammaE, gammap, omegaGAM) are species-INDEPENDENT, so clamp the
    # selector into readEXPRO's supported 1..4 range to support cases with >4 EXPRO species
    # (e.g. e + 4 ions, IS_EP=4 -> expro_is=5).
    _, _Ti, _, _, cs, rmin_ex, gammaE, gammap, omegaGAM =
        readEXPRO(expro_file, clamp(expro_is, 1, 4))

    if gacode_file !== nothing && isfile(gacode_file)
        rmin = read_gacode_scalar_field(gacode_file, "rmin", nr)
        ne = read_gacode_scalar_field(gacode_file, "ne", nr)
        ni = read_gacode_ion_field(gacode_file, "ni", is_EP_gacode, nr)
        Ti = read_gacode_ion_field(gacode_file, "ti", is_EP_gacode, nr)
        # Fortran AS(is_EP+1) = EXPRO_ni(is_EP)/EXPRO_ne; use for cross-check only.
        as_norm = ni ./ max.(ne, 1.0e-30)
        max_rel_as = maximum(abs.(as_norm .- prof.AS[:, expro_is]) ./ max.(abs.(prof.AS[:, expro_is]), 1.0e-30))
        if max_rel_as > 0.05
            @debug "ni/ne vs profile AS mismatch" max_rel_as is_EP_gacode expro_is
        end
        dlnnidr, dlntidr = expro_log_gradients(ni, Ti, rmin)
        @. dlnnidr = max(dlnnidr, 1.0)
        rmin_ex = rmin
    else
        if gacode_file !== nothing && !isfile(gacode_file)
            @warn "gacode_file not found, using profile AS/RLNS for α inputs" gacode_file
        end
        ni = prof.AS[:, expro_is]
        Ti = prof.TAUS[:, expro_is]
        dlnnidr = max.(prof.RLNS[:, expro_is] ./ a_m, 1.0)
        dlntidr = prof.RLTS[:, expro_is] ./ a_m
    end
    return ni, Ti, dlnnidr, dlntidr, cs, rmin_ex, gammaE, gammap, omegaGAM
end

"""
    compute_alpha_crit_profiles(SFmin, Options, profile, ni, Ti, dlnnidr, dlntidr)

Recompute `alpha_dndr_crit` / `alpha_dpdr_crit` profiles from converged `SFmin` (same logic as `run_tjlfep_file`).
"""
function compute_alpha_crit_profiles(
    SFmin::AbstractVector{<:Real},
    Options::Options{T},
    profile::profile{T},
    ni::AbstractVector,
    Ti::AbstractVector,
    dlnnidr::AbstractVector,
    dlntidr::AbstractVector,
) where {T<:Real}
    scan_n = Options.SCAN_N
    nr = profile.NR
    length(SFmin) >= scan_n || error("SFmin length $(length(SFmin)) < SCAN_N=$scan_n")

    dndr_crit = fill(10000.0, scan_n)
    dpdr_crit = fill(10000.0, scan_n)
    dpdr_EP = Vector{Float64}(undef, nr)
    for i in 1:nr
        dpdr_EP[i] = ni[i] * Ti[i] * (dlnnidr[i] + dlntidr[i]) * 0.16022
    end

    for i in 1:scan_n
        ir = Int(Options.IR_EXP[i])
        if SFmin[i] < 9000.0
            dndr_crit[i] = SFmin[i] * ni[ir] * dlnnidr[ir]
            if Options.SCAN_METHOD == 1
                dpdr_scale = SFmin[i]
            elseif Options.SCAN_METHOD == 2
                denom = dlnnidr[ir] + dlntidr[ir]
                dpdr_scale = denom == 0 ? SFmin[i] :
                    (SFmin[i] * dlnnidr[ir] + dlntidr[ir]) / denom
            else
                dpdr_scale = SFmin[i]
            end
            dpdr_crit[i] = dpdr_scale * dpdr_EP[ir]
        end
    end

    _, dndr_out, _, _, _ = tjlfep_complete_output(dndr_crit, Options, profile)
    _, dpdr_out, _, _, _ = tjlfep_complete_output(dpdr_crit, Options, profile)
    return dndr_out, dpdr_out
end

"""
readEXPRO function is a temporary function just used to define any EXPRO constants that are needed.
`is_EP` here is the **EXPRO species index** (1-based, including electron as species 1 in file layout).
For GACODE energetic-ion index use `read_expro_for_alpha` instead.
"""
function readEXPRO(filename::String, is_EP::Int64)
    #filename = "/Users/benagnew/TJLF.jl/outputs/tglfep_tests/input.EXPRO"
    open(filename)

    lines = readlines(filename)

    ni1::Vector{Float64} = fill(NaN, 201)
    ni2::Vector{Float64} = fill(NaN, 201)
    ni3::Vector{Float64} = fill(NaN, 201)
    ni4::Vector{Float64} = fill(NaN, 201)
    Ti1::Vector{Float64} = fill(NaN, 201)
    Ti2::Vector{Float64} = fill(NaN, 201)
    Ti3::Vector{Float64} = fill(NaN, 201)
    Ti4::Vector{Float64} = fill(NaN, 201)
    dlnnidr1::Vector{Float64} = fill(NaN, 201)
    dlnnidr2::Vector{Float64} = fill(NaN, 201)
    dlnnidr3::Vector{Float64} = fill(NaN, 201)
    dlnnidr4::Vector{Float64} = fill(NaN, 201)
    dlntidr1::Vector{Float64} = fill(NaN, 201)
    dlntidr2::Vector{Float64} = fill(NaN, 201)
    dlntidr3::Vector{Float64} = fill(NaN, 201)
    dlntidr4::Vector{Float64} = fill(NaN, 201)
    cs::Vector{Float64} = fill(NaN, 201)
    rmin_ex::Vector{Float64} = fill(NaN, 201)
    gammaE::Vector{Float64} = fill(NaN, 201)
    gammap::Vector{Float64} = fill(NaN, 201)
    omegaGAM::Vector{Float64} = fill(NaN, 201)

    for line in lines[1:length(lines)]
        line = split(line, "=")
        val = line[2]
        line = split(line[1], "_")
        for i in eachindex(line)
            line[i] = strip(line[i])
        end

        exproname = line[2]
        if (length(line) == 4)
            isEPname = line[3]
        elseif (length(line) == 3)
            isEPname = ""
        end
        index = parse(Int64, String(line[end]))
        name = exproname*isEPname
        if (name == "ni1")
            ni1[index] = parse(Float64, String(val))
        elseif (name == "ni2")
            ni2[index] = parse(Float64, String(val))
        elseif (name == "ni3")
            ni3[index] = parse(Float64, String(val))
        elseif (name == "ni4")
            ni4[index] = parse(Float64, String(val))
        elseif (name == "Ti1") 
            Ti1[index] = parse(Float64, String(val))
        elseif (name == "Ti2")
            Ti2[index] = parse(Float64, String(val))
        elseif (name == "Ti3")
            Ti3[index] = parse(Float64, String(val))
        elseif (name == "Ti4")
            Ti4[index] = parse(Float64, String(val))
        elseif (name == "dlnnidr1")
            dlnnidr1[index] = parse(Float64, String(val))
        elseif (name == "dlnnidr2")
            dlnnidr2[index] = parse(Float64, String(val))
        elseif (name == "dlnnidr3")
            dlnnidr3[index] = parse(Float64, String(val))
        elseif (name == "dlnnidr4")
            dlnnidr4[index] = parse(Float64, String(val))
        elseif (name == "dlntidr1")
            dlntidr1[index] = parse(Float64, String(val))
        elseif (name == "dlntidr2")
            dlntidr2[index] = parse(Float64, String(val))
        elseif (name == "dlntidr3")
            dlntidr3[index] = parse(Float64, String(val))
        elseif (name == "dlntidr4")
            dlntidr4[index] = parse(Float64, String(val))
        elseif (name == "cs")
            cs[index] = parse(Float64, String(val))
        elseif (name == "rmin")
            rmin_ex[index] = parse(Float64, String(val))
        elseif (name == "gammaE")
            gammaE[index] = parse(Float64, String(val))
        elseif (name == "gammap")
            gammap[index] = parse(Float64, String(val))
        elseif (name == "omegaGAM")
            omegaGAM[index] = parse(Float64, String(val))
        end
    end

    # Diverge 4 is_EP values for each quatnity:
    # Match Fortran TGLFEP_read_EXPRO.f90: floor EP (dn/dr)/n at 1.0 m⁻¹ for α profiles.

    if (is_EP == 1)
        @. dlnnidr1 = max(dlnnidr1, 1.0)
        return ni1, Ti1, dlnnidr1, dlntidr1, cs, rmin_ex, gammaE, gammap, omegaGAM
    elseif (is_EP == 2)
        @. dlnnidr2 = max(dlnnidr2, 1.0)
        return ni2, Ti2, dlnnidr2, dlntidr2, cs, rmin_ex, gammaE, gammap, omegaGAM
    elseif (is_EP == 3)
        @. dlnnidr3 = max(dlnnidr3, 1.0)
        return ni3, Ti3, dlnnidr3, dlntidr3, cs, rmin_ex, gammaE, gammap, omegaGAM
    elseif (is_EP == 4)
        @. dlnnidr4 = max(dlnnidr4, 1.0)
        return ni4, Ti4, dlnnidr4, dlntidr4, cs, rmin_ex, gammaE, gammap, omegaGAM
    else
        # Loud failure: the old silent `return 1` sentinel surfaced two frames up as a
        # cryptic `BoundsError: attempt to access Int64 at index [2]` at the caller's
        # tuple destructuring (hit via IS_EP=4 -> expro species 5 before the
        # read_expro_for_alpha clamp existed).
        error("readEXPRO: species selector is_EP=$is_EP not in 1:4 for $filename; " *
              "for GACODE energetic-ion indices use read_expro_for_alpha instead")
    end

end 

"""
    save_TGLFEP(opts::Options, filename::AbstractString)

Write an Options struct to a file in input.TGLFEP format.
Format per line: `<value>  <FIELD_NAME>`
Scalars (Int, Float, Bool) are written; Vector fields and missing values are skipped,
as TGLFEP input files do not encode WIDTH/FACTOR scan vectors.
"""
function save_TGLFEP(opts::Options, filename::AbstractString)
    mkpath(dirname(abspath(filename)))
    open(filename, "w") do io
        for key in fieldnames(typeof(opts))
            value = getfield(opts, key)
            if ismissing(value)
                continue
            elseif isa(value, Bool)         # Bool before Int — Bool <: Int in Julia
                val_str = value ? ".true." : ".false."
                println(io, "$val_str  $(key)")
            elseif isa(value, Int)
                println(io, "$(value)  $(key)")
            elseif isa(value, AbstractFloat)
                isnan(value) && continue
                println(io, "$(value)  $(key)")
            # Vector/Matrix fields are not written to TGLFEP scalar input
            end
        end
    end
end

"""
    save_MTGLF(prof::profile, ir_exp::Vector{Int}, filename::AbstractString)

Write a profile struct to a file in input.MTGLF format.
  - Scalars (Int, Float):          `       FIELD_NAME= <value>`
  - Per-species vectors (length NS): `       FIELD_  <is>= <value>`
  - Radial vectors (length NR):    `       FIELD_  <ir>= <value>`
  - Matrices (NR×NS):              `       FIELD_  <ir>_<is>= <value>`
  - IR_EXP entries appended at end, one per scan radius.
"""
function save_MTGLF(prof::profile, ir_exp::Vector{Int}, filename::AbstractString)
    nr = ismissing(prof.NR) ? 0 : prof.NR
    ns = ismissing(prof.NS) ? 0 : prof.NS

    mkpath(dirname(abspath(filename)))
    open(filename, "w") do io
        for key in fieldnames(typeof(prof))
            value = getfield(prof, key)
            if ismissing(value)
                continue
            end
            kstr = String(key)
            if isa(value, Int)
                println(io, @sprintf("%15s=%d", kstr, value))
            elseif isa(value, AbstractFloat)
                isnan(value) && continue
                println(io, @sprintf("%15s=%#.17g", kstr, value))
            elseif isa(value, Vector) && length(value) == ns
                all(isnan, value) && continue
                # Per-species vector
                for is in 1:ns
                    println(io, @sprintf("%15s_%3d=%#.17g", kstr, is, Float64(value[is])))
                end
            elseif isa(value, Vector) && length(value) == nr
                all(isnan, value) && continue
                # Radial vector
                for ir in 1:nr
                    println(io, @sprintf("%15s_%3d=%#.17g", kstr, ir, Float64(value[ir])))
                end
            elseif kstr == "ZS" && isa(value, Matrix) && size(value) == (nr, ns)
                all(isnan, value) && continue
                # file version expects one value for each species- just take first for now (better to average?)
                for is in 1:ns
                    println(io, @sprintf("%15s_%3d=%#.17g", kstr, is, Float64(value[1, is])))
                end
            elseif isa(value, Matrix) && size(value) == (nr, ns)
                all(isnan, value) && continue
                # NR×NS matrix — written as FIELD_<ir>_<is>
                for is in 1:ns
                    for ir in 1:nr
                        println(io, @sprintf("%15s_%3d_%d=%#.17g", kstr, ir, is, Float64(value[ir, is])))
                    end
                end
            end
            # Any other type (e.g. Bool flags stored as Int elsewhere) is silently skipped
        end
        # IR_EXP is not a profile field but is read by readMTGLF; write one entry per scan radius.
        # Format: IR_EXP_  1_  <i>= <ir_value>  (twoName: speciesIndex = line[3], value appended)
        for i in eachindex(ir_exp)
            println(io, @sprintf("%15s_%3d_%3d=%#.17g", "IR_EXP", 1, i, Float64(ir_exp[i])))
        end
    end
end

"""
    save_EXPRO(extraEP::Dict, filename::AbstractString)

Write the extraEP dictionary to a file in input.EXPRO format.

Two-index (species + radial) fields — ni, Ti, dlnnidr, dlntidr:
    `      EXPRO_<field>_   <is>_   <ir>= <value>`

Single-index (radial) fields — cs, rmin, gammaE, gammap, omegaGAM:
    `      EXPRO_<field>_  <ir>= <value>`
"""
function save_EXPRO(extraEP::Dict, filename::AbstractString)
    nr = extraEP["NR"]
    ns = extraEP["NS"]

    # Fields written with two indices: EXPRO_field_   <is>_   <ir>
    two_index_fields = ["DENS", "TEMP", "DLNNDR", "DLNTDR"]
    # Map from extraEP key prefix to EXPRO field name (lowercase in file)
    two_index_map = Dict("DENS" => "ni", "TEMP" => "Ti", "DLNNDR" => "dlnnidr", "DLNTDR" => "dlntidr")

    # Fields written with one index: EXPRO_field_  <ir>
    one_index_map = Dict("CS" => "cs", "RMIN" => "rmin", "gammaE" => "gammaE",
                         "gammap" => "gammap", "omegaGAM" => "omegaGAM")

    mkpath(dirname(abspath(filename)))
    open(filename, "w") do io
        # Two-index fields, species outer loop
        for prefix in two_index_fields
            expro_name = two_index_map[prefix]
            for is in 1:ns
                key = "$(prefix)_$is"
                if !haskey(extraEP, key)
                    continue
                end
                vec = extraEP[key]
                for ir in 1:nr
                    println(io, @sprintf("      EXPRO_%s_%4d_%4d=%#.17g", expro_name, is, ir, vec[ir]))
                end
            end
        end

        # One-index fields
        for (key, expro_name) in one_index_map
            if !haskey(extraEP, key)
                continue
            end
            vec = extraEP[key]
            for ir in 1:nr
                println(io, @sprintf("EXPRO_%s_%d=%#.17g", expro_name, ir, vec[ir]))
            end
        end
    end
end

"""
    save_all(opts::Options, prof::profile, extraEP::Dict, dir::AbstractString)

Write all three input files (input.TGLFEP, input.MTGLF, input.EXPRO) to `dir`.
"""
function save_all(opts::Options, prof::profile, extraEP::Dict, dir::AbstractString)
    mkpath(dir)
    save_TGLFEP(opts, joinpath(dir, "input.TGLFEP"))
    save_MTGLF(prof, opts.IR_EXP, joinpath(dir, "input.MTGLF"))
    save_EXPRO(extraEP, joinpath(dir, "input.EXPRO"))
end
