# Build the FILE-ONLY GPU sysimage: CUDA + TJLF + TJLFEP (standalone, no FUSE/IMAS), with the
# GPU eigensolve path precompiled. This is the image a TGLF-EP user running the file-based scan
# (run_gacode_scan_task / run_tjlfep on input.gacode + input.TGLFEP) gets before going
# FUSE-native -- leaner and faster-loading than TGLFEP_JULIA_gpu_generic_sysimage.so because the
# IMAS/FUSE stack is NOT baked.
#
# Builds against the registry-resolved "lean" environment created by setup_registry_env.jl
# (TGLFEP_JULIA_BUILD_ENV): released FuseRegistry versions only, reproducible by any user. The
# lean project never resolves the IMAS/GACODE/TurbulentTransport weak deps, so the
# TJLFEPIMASExt extension stays dormant and FUSE/IMAS are never pulled into the image.
#
# Run on a GPU node (the precompile workload needs a functional GPU).

using Pkg

build_env = get(ENV, "TGLFEP_JULIA_BUILD_ENV", "")
@assert !isempty(build_env) "TGLFEP_JULIA_BUILD_ENV must point at the project made by setup_registry_env.jl (variant: lean)"

Pkg.activate(build_env)
Pkg.instantiate()
using PackageCompiler

create_sysimage(
    [:CUDA, :TJLF, :TJLFEP];
    sysimage_path = normpath(@__DIR__, "..", "TGLFEP_JULIA_gpu_sysimage.so"),
    precompile_execution_file = normpath(@__DIR__, "precompile_gpu_workload_fileonly.jl"),
    project = build_env,
    cpu_target = PackageCompiler.default_app_cpu_target(),
)
