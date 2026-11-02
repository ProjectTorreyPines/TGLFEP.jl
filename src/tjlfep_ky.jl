# Opt-in latency probe (TGLFEP_JULIA_PROBE=1). Per-process atomic accumulators: total time inside
# TJLFEP_ky vs the TJLF.run (eigensolve) sub-call, so kwscale_scan can report the
# eigensolve-vs-CPU split that bounds the MPS-team speedup (Amdahl).
# NOTE: read at RUNTIME (not a precompiled const) so it honors the env at run time.
_probe_on() = get(ENV, "TGLFEP_JULIA_PROBE", "0") == "1"
const _PROBE_RUN  = Threads.Atomic{Float64}(0.0)  # seconds in TJLF.run (eigensolve)
const _PROBE_KY   = Threads.Atomic{Float64}(0.0)  # seconds in the whole TJLFEP_ky
const _PROBE_N    = Threads.Atomic{Int}(0)
function _probe_reset!()
    _PROBE_RUN[] = 0.0; _PROBE_KY[] = 0.0; _PROBE_N[] = 0
end

function TJLFEP_ky(inputsEP::Options{T}, inputsPR::profile{T}, str_wf_file::String, l_wavefunction_out::Int;
                   eigen_cache::Union{Vector{<:Complex}, Nothing} = nothing,
                   use_gpu::Bool = false,
                   mode_in_override::Union{Nothing,Int} = nothing,
                   ky_model_override::Union{Nothing,Int} = nothing) where {T<:Real} #, factor_in::Int64, kyhat_in::Int64, width_in::Int64)
    _pb = _probe_on()
    _t_ky = _pb ? time_ns() : UInt64(0)
# function TJLFEP_ky(inputsEP::Options{T}, inputsPR::profile{T}, str_wf_file::String, l_wavefunction_out::Int) where {T<:Real} #, factor_in::Int64, kyhat_in::Int64, width_in::Int64)
    # Temp Defs:
    
    #color = 0
    #kyhat_in = 3
    # Temp Struct Inputs:
    #filename = "/Users/benagnew/TJLF.jl/outputs/tglfep_tests/input.MTGLF"
    #temp = readMTGLF(filename)
    #inputsPR = temp[1]
    #irexp2 = temp[2]
    #filename = "/Users/benagnew/TJLF.jl/outputs/tglfep_tests/input.TGLFEP"
    #inputsEP = readTGLFEP(filename, irexp2)
    #inputsEP.IR = inputsEP.IR_EXP[color+1]
    #inputsEP.MODE_IN = 2
    #inputsEP.KY_MODEL = 3

    #========================================#

    inputTJLF = TJLF_map(inputsEP, inputsPR; mode_in_override=mode_in_override, ky_model_override=ky_model_override)

    inputTJLF.USE_TRANSPORT_MODEL = false # single-ky path: bypasses full spectral transport model
    # inputTJLF.USE_TRANSPORT_MODEL = true
    
    inputTJLF.KYGRID_MODEL = 0

    inputTJLF.NMODES = inputsEP.NMODES

    inputTJLF.NBASIS_MIN = inputsEP.N_BASIS
    inputTJLF.NBASIS_MAX = inputsEP.N_BASIS

    # Match the Fortran TGLFEP nb48 driver: nxgrid = max(n_basis, 32), so the
    # Gauss-Hermite grid (nx = 2*nxgrid-1) resolves the highest basis function.
    # With the old hardcoded 32, NBASIS=48 runs were under-resolved (63 nodes vs
    # basis products of polynomial order ~94) and lost unstable AE branches.
    # Env override TGLFEP_JULIA_NXGRID is a debug knob for quadrature-resolution studies.
    inputTJLF.NXGRID = something(tryparse(Int, get(ENV, "TGLFEP_JULIA_NXGRID", "")),
                                 max(inputsEP.N_BASIS, 32))

    inputTJLF.WIDTH = inputsEP.WIDTH_IN
    inputTJLF.FIND_WIDTH = false

    # Corrections for TJLF specifically: (see main.jl after mainsub call)

    inputTJLF.USE_AVE_ION_GRID = false
    inputTJLF.WIDTH_SPECTRUM .= inputTJLF.WIDTH # see tjlf_read_input.jl from TJLF.jl.
    inputTJLF.FIND_EIGEN = true # in all inputs for tjlf this is set to true
    inputTJLF.RLNP_CUTOFF = 18.0 # in all inputs for tjlf this is set to 18.0
    inputTJLF.BETA_LOC = 0.0 # This one I am very unsure of. Some 0.0, some 1.0. 
    inputTJLF.DAMP_PSI = 0.0 # in all inputs for tjlf this is set to 0.0
    inputTJLF.DAMP_SIG = 0.0 # in all inputs for tjlf this is set to 0.0
    inputTJLF.WDIA_TRAPPED = 0.0

   
    # n_out::Int
    # EP_QL_e_flux::Float32
    # ef_phi_norm::Float32

    if inputTJLF.SAT_RULE == 2 || inputTJLF.SAT_RULE == 3 # From read_input, which is skipped over in this path of running TJLF
        inputTJLF.UNITS = "CGYRO"
        inputTJLF.XNU_MODEL = 3
        inputTJLF.WDIA_TRAPPED = 1.0
    end

    inputTJLF.KX0_LOC = 0.0

    # Consolidated onto TJLF.InputTJLF: TJLF_map returns a TJLF.InputTJLF directly, so we
    # run TJLF on inputTJLF in place (no convert_input/revert_input round-trip).
    # Seed EIGEN_SPECTRUM from cache so KrylovKit can be used instead of full geev!
    if eigen_cache !== nothing && !ismissing(inputTJLF.EIGEN_SPECTRUM) && length(eigen_cache) == length(inputTJLF.EIGEN_SPECTRUM)
        inputTJLF.EIGEN_SPECTRUM .= eigen_cache
    end

    # Run TJLF and return QLweight and eigenvalues:
    # TJLF.run hardcodes zeros(Float64,...) for result arrays, which fails for Dual.
    # _tjlf_run_dual is identical but uses zeros(T,...). T===Float64 is a compile-time
    # check so Julia eliminates the unused branch for each specialization.
    # The TGLF dispersion matrix can be singular for pathological inputs — most
    # commonly the separatrix scan point (ir = NR, rho ~ 1.0) of a synthesized
    # (e.g. FUSE) equilibrium where the edge gradients (RLTS/RLNS) blow up. A
    # singular matrix means there is no resolvable eigenmode, so treat the combo
    # as stable (no AE drive) instead of letting the SingularException crash the
    # whole scan.
    local result
    try
        if T === Float64
            if _pb
                _tr = time_ns()
                result = TJLF.run(inputTJLF; use_gpu=use_gpu)
                Threads.atomic_add!(_PROBE_RUN, (time_ns() - _tr) / 1e9)
            else
                result = TJLF.run(inputTJLF; use_gpu=use_gpu)
            end
        else
            quit()
            result = _tjlf_run_dual(inputTJLF, T; use_gpu=use_gpu)
        end
    catch err
        err isa SingularException || rethrow()
        @warn "TJLFEP_ky: singular TGLF dispersion matrix (ir=$(inputsEP.IR), ky=$(inputTJLF.KY), width=$(inputTJLF.WIDTH)); treating combo as stable (no AE mode)" maxlog = 5
        NM = inputTJLF.NMODES
        inputsEP.LKEEP .= false
        inputsEP.LTEARING .= false
        inputsEP.L_I_PINCH .= false
        inputsEP.L_E_PINCH .= false
        inputsEP.L_TH_PINCH .= false
        inputsEP.L_EP_PINCH .= false
        inputsEP.L_QL_RATIO .= false
        inputsEP.L_THETA_SQ .= false
        if _pb
            Threads.atomic_add!(_PROBE_KY, (time_ns() - _t_ky) / 1e9)
            Threads.atomic_add!(_PROBE_N, 1)
        end
        if _combo_log_level() >= 1
            println("[COMBO] ir=", inputsEP.IR, " ky^=", inputsEP.KYHAT_IN,
                    " w=", inputsEP.WIDTH_IN, " sf=", inputsEP.FACTOR_IN,
                    " SINGULAR (treated stable)")
            flush(stdout)
        end
        return fill(T(0), NM), fill(T(0), NM), inputTJLF, nothing, nothing, fill(T(NaN), NM), fill(T(NaN), NM)
    end
    gamma_out        = result.eigenvalue[:, 1, 1]   # [nmodes], ky=1
    freq_out         = result.eigenvalue[:, 1, 2]   # [nmodes], ky=1
    particle_QL_out  = result.QL_weights[:, :, :, 1, 1]  # [fields, species, nmodes], ky=1
    energy_QL_out    = result.QL_weights[:, :, :, 1, 2]  # [fields, species, nmodes], ky=1
    field_weight_out = result.field_weight_out[:, :, :, 1]  # [fields, basis, nmodes], ky=1
    satParams        = TJLF.get_sat_params(inputTJLF);
    eigen_out        = ismissing(inputTJLF.EIGEN_SPECTRUM) ? nothing : copy(inputTJLF.EIGEN_SPECTRUM)  # cache for next call
    
    g = fill(T(NaN), inputTJLF.NMODES)
    f = fill(T(NaN), inputTJLF.NMODES)
    for n = 1:inputTJLF.NMODES
        g[n] = gamma_out[n]
        f[n] = freq_out[n]
    end
    for n = 1:inputTJLF.NMODES
        debug_dump_ky_postrun(inputsEP, inputTJLF, g, f, n)
    end

    #GAMMA STILL NON-ZERO UP TO HERE

    # Establishes the lkeep vector. LKEEP is defaulted to all true as nothing is rejected yet.
    # This states that if the frequency that came from the converted test of TJLF is less than
    # the cutoff, the mode is kept. It also then requires that the growthrate is larger than 1e-7.
    inputsEP.LKEEP .= true

    for n = 1:inputTJLF.NMODES
        inputsEP.LKEEP[n] = (f[n] < inputsEP.FREQ_AE_UPPER)
        inputsEP.LKEEP[n] = (inputsEP.LKEEP[n] && (g[n] > inputsEP.GAMMA_THRESH))
    end

    # This function was translated within TJLF so as to get the wavefunction.
    ms = 128
    max_plot = Int(18*ms/8+1)
    # get_wavefunction requires ComplexF64 for field_weight_out and has Float64-typed
    # geometry arrays internally (xp, hp, plot_field_out). For Dual runs we strip
    # partials from all three args — wavefunction shape is independent of AD parameters.
    if T === Float64
        wavefunction, angle, nplot, nmodes_out = TJLF.get_wavefunction(inputTJLF, satParams, field_weight_out)
    else
        quit()
        ci_f64 = _to_float64_input(inputTJLF)
        sp_f64 = TJLF.get_sat_params(ci_f64)
        fw_f64 = map(x -> ComplexF64(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x))), field_weight_out)
        wavefunction, angle, nplot, nmodes_out = TJLF.get_wavefunction(ci_f64, sp_f64, fw_f64)
    end


    inputsEP.LTEARING .= false
    inputsEP.L_I_PINCH .= false
    inputsEP.L_E_PINCH .= false
    inputsEP.L_TH_PINCH .= false
    inputsEP.L_EP_PINCH .= false
    inputsEP.L_QL_RATIO .= false
    inputsEP.L_THETA_SQ .= false
    # inputsEP.L_MAX_OUTER_PANEL .= false
    x_tear_test = fill(zero(T), inputTJLF.NMODES)
    abswavefunction = abs.(wavefunction)

    

    #absdiffwavefunction = similar(wavefunction)
    ms = 128
    npi = 9
    np = Int(ms/8)
    nb = inputTJLF.NBASIS_MAX
    igeo = 1 # Hard-Coded for now as with the previous LS functions:
    max_plot = Int(18*ms/8+1) # 289 length vector
    maxmodes = inputTJLF.NMODES

    i_QL_cond_flux = fill(zero(T), inputTJLF.NMODES)
    e_QL_cond_flux = fill(zero(T), inputTJLF.NMODES)
    QL_flux_ratio = fill(zero(T), inputTJLF.NMODES)
    EP_conv_frac = fill(zero(T), inputTJLF.NMODES)
    theta_2_moment = fill(zero(T), inputTJLF.NMODES)
    # These per-mode diagnostic vectors are sized to NMODES (the keep-count), which is a
    # user input (env TGLFEP_JULIA_NMODES / nmodes kwarg, default 4) — no longer hard-coded to 4.
    DEP = fill(T(NaN), inputTJLF.NMODES)
    ep_ql_flux = fill(T(NaN), inputTJLF.NMODES)
    chi_th = fill(T(NaN), inputTJLF.NMODES)
    chi_i = fill(T(NaN), inputTJLF.NMODES)
    chi_i_cond = fill(T(NaN), inputTJLF.NMODES)
    chi_e = fill(T(NaN), inputTJLF.NMODES)
    chi_e_cond = fill(T(NaN), inputTJLF.NMODES)

    for n = 1:inputsEP.NMODES
        # The use of NMODES here is slightly confusing but needed as the Fortran uses assumed values for modes
        # which did not satisfy the criteria (nmodes_out). This means that the loop must continue for
        # those past nmodes_out so as to be consistent with the flags. This is why n <= nmodes_out is used
        # multiple times in this loop.

        #nul = abswavefunction
        wave_max = maximum(abs.(wavefunction[n,1,:]))+1.0E-3
        
        wave_max_loc = argmax(abs.(wavefunction[n,1,:]))
        n_balloon_pi = floor(Int, (max_plot-1)/9) # 32
        i_mid_plot = floor(Int, (max_plot-1)/2+1) # 145
        # inputsEP.L_MAX_OUTER_PANEL[n] = (wave_max_loc < (i_mid_plot-n_balloon_pi)) || (wave_max_loc > (i_mid_plot+n_balloon_pi))
        # So long as wave_max_loc is between 113 and 177, this isn't rejected for this.
        theta_2_moment[n] =0.0
        ef_phi_norm = 0.0


        if (n <= nmodes_out)#used to be nmodes_out
            for i = 1:max_plot # Finding the maximum value of this abs value of difference div wave_max
                absdiffwavefunction = abs(wavefunction[n,1,i]-wavefunction[n,1,max_plot+1-i])
                x_tear_test[n] = max(x_tear_test[n], absdiffwavefunction/wave_max)
                ef_phi_norm += abs(wavefunction[n,1,i])
                theta_2_moment[n] += (9 * π * (-1.0 + (2.0 * (i - 1)) / (max_plot - 1)))^2 * abs(wavefunction[n, 1, i])
            end
            theta_2_moment[n] /= ef_phi_norm

            if (x_tear_test[n] > 1.0E-1)
                inputsEP.LTEARING[n] = true
                
            end
        end 
        EP_QL_e_flux = 0.0
        EP_QL_flux = 0.0
        i_QL_flux = 0.0
        i_QL_cond_flux[n] = 0.0
        i_eff_grad = 0.0
        e_QL_flux = 0.0
        e_QL_cond_flux[n] = 0.0
        th_QL_flux = 0.0
        th_eff_grad = 0.0

        # Fortran TGLFEP_ky accumulates QL for all nmodes (no nmodes_out guard).
        for jfields = 1:3
            EP_QL_flux = EP_QL_flux + particle_QL_out[jfields, inputsEP.IS_EP + 1, n]
            EP_QL_e_flux = EP_QL_e_flux + energy_QL_out[jfields, inputsEP.IS_EP + 1, n]
            e_QL_flux = e_QL_flux + energy_QL_out[jfields, 1, n]
            e_QL_cond_flux[n] = e_QL_cond_flux[n] + energy_QL_out[jfields, 1, n] - 1.5*inputTJLF.TAUS[1]*particle_QL_out[jfields, 1, n]
            for j_ion = 2:inputsEP.IS_EP
                i_QL_flux = i_QL_flux + energy_QL_out[jfields, j_ion, n]
                i_QL_cond_flux[n] = i_QL_cond_flux[n] + energy_QL_out[jfields, j_ion, n] - 1.5*inputTJLF.TAUS[j_ion]*particle_QL_out[jfields, j_ion, n]
            end
        end

        for j_ion = 2:inputsEP.IS_EP
            i_eff_grad = i_eff_grad + inputTJLF.TAUS[j_ion]*inputTJLF.AS[j_ion]*inputTJLF.RLTS[j_ion]
        end

        th_eff_grad = i_eff_grad + inputTJLF.RLTS[1]*inputTJLF.AS[1]
        th_QL_flux = i_QL_cond_flux[n] + e_QL_cond_flux[n]
        ep_ql_flux[n] = EP_QL_flux
        DEP[n] = EP_QL_flux / (inputTJLF.AS[inputsEP.IS_EP+1]*inputTJLF.RLNS[inputsEP.IS_EP+1])
        chi_e[n] = e_QL_flux / (inputTJLF.RLTS[1]*inputTJLF.AS[1])
        chi_e_cond[n] = e_QL_cond_flux[n] / (inputTJLF.RLTS[1]*inputTJLF.AS[1])
        chi_i[n] = i_QL_flux / i_eff_grad
        chi_i_cond[n] = i_QL_cond_flux[n] / i_eff_grad
        chi_th[n] = th_QL_flux / th_eff_grad

        # QL_flux_ratio[n] = (EP_QL_flux/inputTJLF.AS[inputsEP.IS_EP+1])/(abs(i_QL_cond_flux[n])/(inputTJLF.AS[1]-inputTJLF.AS[inputsEP.IS_EP+1]))
        QL_flux_ratio[n] = EP_QL_e_flux / abs(i_QL_cond_flux[n])
        EP_conv_frac[n] = EP_QL_flux * 1.5 * inputTJLF.TAUS[inputsEP.IS_EP+1] / EP_QL_e_flux
        if n == 1
            dbgmsg("ky_ql n=1 QL_flux_ratio=", QL_flux_ratio[n], " EP_QL_e_flux=", EP_QL_e_flux,
                " i_QL_cond=", i_QL_cond_flux[n])
        end

        if (chi_i[n] < 0.0) inputsEP.L_I_PINCH[n] = true end
        if (chi_e[n] < 0.0) inputsEP.L_E_PINCH[n] = true end
        if (chi_th[n] < 0.0) inputsEP.L_TH_PINCH[n] = true end
        if (DEP[n] < 0.0) inputsEP.L_EP_PINCH[n] = true end
        if (QL_flux_ratio[n] < inputsEP.QL_RATIO_THRESH) inputsEP.L_QL_RATIO[n] = true end
        if (theta_2_moment[n] > inputsEP.THETA_SQ_THRESH) inputsEP.L_THETA_SQ[n] = true end
        
       
    end

    #=if (testid)
        println(chi_th)
    end=#
    
    # Runs over all modes (4) for the lkeep vector and checks if each flag is false. If lkeep false, all will be false. 
    # If lkeep is true, if the flag is false, lkeep will stay true; if lkeep is true and the flag is true, lkeep will be turned back to false
    # This essentiall means that if the rejection flag is turned on, anything marked for rejection will be rejected.
    for n = 1:inputTJLF.NMODES
        if (inputsEP.REJECT_TEARING_FLAG == 1) inputsEP.LKEEP[n] = (inputsEP.LKEEP[n] && !inputsEP.LTEARING[n]) end
        if (inputsEP.REJECT_I_PINCH_FLAG == 1) inputsEP.LKEEP[n] = (inputsEP.LKEEP[n] && !inputsEP.L_I_PINCH[n]) end
        if (inputsEP.REJECT_E_PINCH_FLAG == 1) inputsEP.LKEEP[n] = (inputsEP.LKEEP[n] && !inputsEP.L_E_PINCH[n]) end
        if (inputsEP.REJECT_TH_PINCH_FLAG == 1) inputsEP.LKEEP[n] = (inputsEP.LKEEP[n] && !inputsEP.L_TH_PINCH[n]) end
        if (inputsEP.REJECT_EP_PINCH_FLAG == 1) inputsEP.LKEEP[n] = (inputsEP.LKEEP[n] && !inputsEP.L_EP_PINCH[n]) end
        # if (inputsEP.ROTATIONAL_SUPPRESSION_FLAG == 1) inputsEP.LKEEP[n] = (inputsEP.LKEEP[n] && !inputsEP.L_MAX_OUTER_PANEL[n]) end
        inputsEP.LKEEP[n] = (inputsEP.LKEEP[n] && !inputsEP.L_QL_RATIO[n])
        inputsEP.LKEEP[n] = (inputsEP.LKEEP[n] && !inputsEP.L_THETA_SQ[n])
    end

    debug_dump_combo_full(inputsEP, inputTJLF, g, f, x_tear_test, QL_flux_ratio,
                          theta_2_moment, DEP, chi_i)

    # Next is writing the wavefunction files themselves:
    wavefunction_buffer = nothing
    if (l_wavefunction_out == 1) # nplot = max_plot_out; nfields = 1 by def.
        wavefunction_buffer = String[]
        nfields_out = size(wavefunction, 2)
        push!(wavefunction_buffer, "nmodes= $(nmodes_out)  nfields= $(nfields_out)  max_plot= $(nplot)")
        push!(wavefunction_buffer, "ky=$(inputTJLF.KY) width=$(inputTJLF.WIDTH)")
        push!(wavefunction_buffer, "theta     ((Re(field_i), Im(field_i),i=(1,nfields)),j=1,nmodes)")
        push!(wavefunction_buffer, "Tearing metric: $(x_tear_test)")
        push!(wavefunction_buffer, "DEP: $(DEP)")
        push!(wavefunction_buffer, "chi_th: $(chi_th)")
        push!(wavefunction_buffer, "chi_i: $(chi_i)")
        push!(wavefunction_buffer, "chi_i_cond: $(chi_i_cond)")
        push!(wavefunction_buffer, "chi_e: $(chi_e)")
        push!(wavefunction_buffer, "chi_e_cond: $(chi_e_cond)")
        push!(wavefunction_buffer, "i_QL_cond_flux: $(i_QL_cond_flux)")
        push!(wavefunction_buffer, "e_QL_cond_flux: $(e_QL_cond_flux)")
        push!(wavefunction_buffer, "QL_ratio: $(QL_flux_ratio)")
        push!(wavefunction_buffer, "EP QL convection fracton: $(EP_conv_frac)")
        push!(wavefunction_buffer, "<theta^2>: $(theta_2_moment)")
        push!(wavefunction_buffer, "lkeep: $(inputsEP.LKEEP)")
        # Renormalize and adjust phases:
        n_out = 0
        for n = 1:inputsEP.NMODES
            max_phi = maximum(abswavefunction[n,1,:])
            max_apar = maximum(abswavefunction[n,2,:])
            max_field = maximum([max_phi, max_apar])
            phase = atan(imag(wavefunction[n,1,Int((nplot+1)/2)]),real(wavefunction[n,1,Int((nplot+1)/2)]))
            for jfields = 1:2
                z = 0+1im
                wavefunction[n,jfields,:] .= wavefunction[n,jfields,:]/(max_field*exp(z*phase))
            end
            if n_out == 0 && inputsEP.LKEEP[n]
                n_out = n
            end
        end
        if n_out == 0
            n_out = 1
            push!(wavefunction_buffer, "No kept modes at nominal write parameters. Showing leading mode.")
        end
        #Write renormalized, re-phased eigenfunctions out to buffer
        for i = 1:max_plot
            push!(wavefunction_buffer, string(angle[i], " ", real(wavefunction[n_out,1,i]), " ", imag(wavefunction[n_out,1,i]), " ",
                    real(wavefunction[n_out,2,i]), " ", imag(wavefunction[n_out,2,i])))
        end
    end      
    if _pb
        Threads.atomic_add!(_PROBE_KY, (time_ns() - _t_ky) / 1e9)
        Threads.atomic_add!(_PROBE_N, 1)
    end
    # This is the end of ky.jl. Returning values will likely need to be changed later.
    return gamma_out, freq_out, inputTJLF, eigen_out, wavefunction_buffer, DEP, ep_ql_flux
end
