# nb6 single-radius regression: assert the Julia critical scale factor (SFmin)
# for one radius of the DIII-D 202017C42 SCAN_N=20 case matches the Fortran
# TGLF-EP reference (build/fortran_runs/debug_nb6_scan20_10n_53171364, archived
# here in test/fixtures/202017C42_nb6/out.TGLFEP; n_basis=6).
#
# We run a single scan radius (default scan_index=2, ir=7) rather than the full
# 20-radius scan so the test is feasible as a smoke-level regression. The full
# SFmin/alpha vector match is validated by the SCAN_N=20 scripts under dev/verify/
# (see docs/REPRODUCE_FORTRAN_MATCH.md).

using Test
using TJLFEP
using TJLF

const _NB6_DIR   = @__DIR__
const _ROOT      = normpath(_NB6_DIR, "..")
const _CASE_DIR  = joinpath(_ROOT, "examples", "DIIID_202017C42_500ms_v3.1")
const _GACODE    = joinpath(_CASE_DIR, "input.gacode")
const _TGLFEP    = joinpath(_CASE_DIR, "input_scan20_nb6.TGLFEP")
const _GOLDEN    = joinpath(_NB6_DIR, "fixtures", "202017C42_nb6", "out.TGLFEP")

# Parse the `SFmin` block from a Fortran out.TGLFEP into a Vector{Float64}
# (one value per scan radius, in scan-index order).
function _read_golden_sfmin(path::AbstractString)
    lines = readlines(path)
    i = findfirst(l -> strip(l) == "SFmin", lines)
    i === nothing && error("no SFmin block in $path")
    vals = Float64[]
    for line in lines[i+1:end]
        x = tryparse(Float64, strip(line))
        x === nothing && break
        push!(vals, x)
    end
    return vals
end

@testset "nb6 single-radius SFmin vs Fortran" begin
    @test isfile(_GACODE)
    @test isfile(_TGLFEP)
    @test isfile(_GOLDEN)

    golden = _read_golden_sfmin(_GOLDEN)
    @test length(golden) == 20

    # scan_index is 1-based; default radius is robust (mid-range SFmin, far from a
    # bisection boundary). Override with TGLFEP_JULIA_TEST_SCAN_INDEX for ad-hoc checks.
    scan_index = parse(Int, get(ENV, "TGLFEP_JULIA_TEST_SCAN_INDEX", "2"))

    out_dir = mktempdir()
    r = run_gacode_scan_task(_GACODE, _TGLFEP, scan_index; out_dir=out_dir, use_gpu=false, printout=false)

    sf_julia  = r.sfmin
    sf_golden = golden[scan_index]
    @info "nb6 SFmin regression" scan_index ir=r.ir sf_julia sf_golden rel_err=abs(sf_julia - sf_golden) / abs(sf_golden)

    # Documented max relative error across all radii is ~0.03%; rtol=2e-3 (0.2%)
    # leaves margin while still catching a real divergence (a bisection step off
    # would jump the SFmin by a discrete factor, far exceeding this tolerance).
    @test isapprox(sf_julia, sf_golden; rtol=2e-3, atol=1e-3)
end
