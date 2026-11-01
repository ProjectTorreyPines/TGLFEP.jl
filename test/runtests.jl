# TJLFEP test entrypoint.
#
# File-based regression only: loaded standalone, TJLFEP never pulls in IMAS/FUSE (the IMAS
# entry points live in the TJLFEPIMASExt package extension, which stays dormant here).
# Run with:  module load julia/1.11.7
#            export JULIA_DEPOT_PATH="$HOME/.julia:$PSCRATCH/.julia${JULIA_DEPOT_PATH:+:$JULIA_DEPOT_PATH}"
#            julia --project=. -t 8 test/runtests.jl      (or Pkg.test())

using Test

# When a GPU run is requested (TJLFEP_TEST_USE_GPU=1, e.g. the premium-GPU batch job), CUDA must be
# imported HERE so TJLF's GPU-solve package extension activates for the whole session; otherwise the
# use_gpu=true solves throw "CUDA extension not loaded" (the extension is a weakdep, dormant until
# CUDA is loaded). No-op on the host-only CI path.
if get(ENV, "TJLFEP_TEST_USE_GPU", "0") == "1"
    using CUDA
    CUDA.functional() || error("TJLFEP_TEST_USE_GPU=1 but CUDA.functional() is false on this host")
    CUDA.device!(first(CUDA.devices()))
end

@testset "TJLFEP" begin
    # Fast, solve-free unit tests (parsers, structs, file IO) for broad coverage.
    include("unit_helpers.jl")
    include("unit_structs.jl")
    include("unit_io.jl")
    include("unit_extra.jl")
    include("unit_width_floor.jl")
    include("unit_ad_smoke.jl")

    # Small multi-radius scan + finalize (mirror of FUSE's ActorTJLFEP smoke test).
    include("unit_scan_finalize.jl")

    # Ballooning-width scan (PROCESS_IN=3 WIDTH_IN_FLAG=false branch).
    include("unit_widthscan.jl")

    # End-to-end regression against the Fortran TGLF-EP reference (one radius).
    include("runtests_regression_nb6.jl")
    include("runtests_regression_spectrum.jl")
end
