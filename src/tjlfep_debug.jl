# Gated debug logging for Julia parity checks (env TGLFEP_JULIA_DEBUG=1).

const TGLFEP_JULIA_DEBUG = get(ENV, "TGLFEP_JULIA_DEBUG", "0") == "1"

"""Print tagged debug line and flush (for Slurm logs)."""
function dbgmsg(args...)
    TGLFEP_JULIA_DEBUG || return nothing
    print("[TGLFEP_JULIA_DBG] ")
    println(args...)
    flush(stdout)
    flush(stderr)
    return nothing
end

"""After TJLF_map: mirror Fortran TGLFEP_tglf_map debug fields."""
function debug_dump_tglf_map(inputsEP, inputsPR, inputTJLF)
    TGLFEP_JULIA_DEBUG || return nothing
    ir = inputsEP.IR
    is = inputsEP.IS_EP + 1
    r_over_a = inputsPR.RMIN[ir] / inputsPR.RMIN[end]
    dbgmsg("tglf_map ir=", ir, " IS_EP=", inputsEP.IS_EP, " is=", is,
        " FACTOR_IN=", inputsEP.FACTOR_IN, " RMIN_LOC=", inputTJLF.RMIN_LOC,
        " KY=", inputTJLF.KY, " KYHAT_IN=", inputsEP.KYHAT_IN,
        " NBASIS_MAX=", inputTJLF.NBASIS_MAX, " NMODES=", inputTJLF.NMODES,
        " AS_e=", inputTJLF.AS[1], " AS_EP=", inputTJLF.AS[is],
        " ZS_e=", inputTJLF.ZS[1], " ZS_EP=", inputTJLF.ZS[is],
        " gamma_thresh=", inputsEP.GAMMA_THRESH,
        " r_over_a=", r_over_a)
    return nothing
end

"""After TJLF.run in TJLFEP_ky: mirror Fortran post-tglf_run debug."""
function debug_dump_ky_postrun(inputsEP, inputTJLF, gamma_out, freq_out, n::Int)
    TGLFEP_JULIA_DEBUG || return nothing
    n <= length(gamma_out) || return nothing
    dbgmsg("ky_postrun n=", n, " gamma=", gamma_out[n], " freq=", freq_out[n],
        " lkeep=", freq_out[n] < inputsEP.FREQ_AE_UPPER && gamma_out[n] > inputsEP.GAMMA_THRESH)
    return nothing
end

"""First kwscale combo only."""
function debug_dump_kw_combo(inputsEP, i::Int)
    TGLFEP_JULIA_DEBUG || return nothing
    i == 1 || return nothing
    dbgmsg("kwscale i=", i, " factor_in=", inputsEP.FACTOR_IN,
        " kyhat_in=", inputsEP.KYHAT_IN, " width_in=", inputsEP.WIDTH_IN)
    return nothing
end

"""Per-combo log level from env `TGLFEP_JULIA_COMBO_LOG` (read at RUNTIME, not baked into the
precompile cache): 0 = off (default), 1 = one line per (combo, mode) with eigenvalues and
every reject-filter value, 2 = additionally dump the mapped TJLF inputs per combo."""
_combo_log_level() = something(tryparse(Int, get(ENV, "TGLFEP_JULIA_COMBO_LOG", "0")), 0)

"""End of TJLFEP_ky: full per-combo, per-mode record — eigenvalue, keep decision, and the
raw filter metrics (tearing amplitude, QL ratio, <theta^2>, pinches) that produced it.
One parseable line per mode, prefixed `[COMBO]`."""
function debug_dump_combo_full(inputsEP, inputTJLF, g, f, x_tear_test, QL_flux_ratio,
                               theta_2_moment, DEP, chi_i)
    lvl = _combo_log_level()
    lvl >= 1 || return nothing
    io = IOBuffer()
    for n in 1:inputTJLF.NMODES
        print(io, "[COMBO] ir=", inputsEP.IR,
            " ky^=", inputsEP.KYHAT_IN,
            " w=", inputsEP.WIDTH_IN,
            " sf=", inputsEP.FACTOR_IN,
            " n=", n,
            " g=", g[n], " f=", f[n],
            " keep=", Int(inputsEP.LKEEP[n]),
            " tear=", Int(inputsEP.LTEARING[n]), " x_tear=", x_tear_test[n],
            " qlr=", Int(inputsEP.L_QL_RATIO[n]), " QLratio=", QL_flux_ratio[n],
            " th2=", Int(inputsEP.L_THETA_SQ[n]), " theta2=", theta_2_moment[n],
            " iP=", Int(inputsEP.L_I_PINCH[n]), " eP=", Int(inputsEP.L_E_PINCH[n]),
            " thP=", Int(inputsEP.L_TH_PINCH[n]), " epP=", Int(inputsEP.L_EP_PINCH[n]),
            " DEP=", DEP[n], " chi_i=", chi_i[n], "\n")
    end
    if lvl >= 2
        print(io, "[COMBO-IN] ir=", inputsEP.IR,
            " ky^=", inputsEP.KYHAT_IN, " w=", inputsEP.WIDTH_IN, " sf=", inputsEP.FACTOR_IN,
            " KY=", inputTJLF.KY, " WIDTH=", inputTJLF.WIDTH,
            " NBASIS=", inputTJLF.NBASIS_MAX,
            " AS=", inputTJLF.AS, " TAUS=", inputTJLF.TAUS,
            " ZS=", inputTJLF.ZS, " MASS=", inputTJLF.MASS,
            " RLNS=", inputTJLF.RLNS, " RLTS=", inputTJLF.RLTS,
            " BETAE=", inputTJLF.BETAE, " ZEFF=", inputTJLF.ZEFF,
            " Q_LOC=", inputTJLF.Q_LOC, " Q_PRIME=", inputTJLF.Q_PRIME_LOC,
            " P_PRIME=", inputTJLF.P_PRIME_LOC, " RMIN=", inputTJLF.RMIN_LOC,
            " RMAJ=", inputTJLF.RMAJ_LOC, " KAPPA=", inputTJLF.KAPPA_LOC,
            " S_KAPPA=", inputTJLF.S_KAPPA_LOC, " DELTA=", inputTJLF.DELTA_LOC,
            " SHIFT=", inputTJLF.DRMAJDX_LOC,
            " FREQ_AE_UPPER=", inputsEP.FREQ_AE_UPPER,
            " GAMMA_THRESH=", inputsEP.GAMMA_THRESH, "\n")
    end
    print(stdout, String(take!(io)))
    flush(stdout)
    return nothing
end
