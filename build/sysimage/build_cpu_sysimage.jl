# Build the full CPU sysimage: TJLF + TJLFEP (file-only), with the CPU eigensolve path
# precompiled via precompile_cpu_workload.jl. This bakes in the per-radius JIT that cold
# distributed CPU workers otherwise pay on every spawn. The FUSE/IMAS stack is excluded by
# construction: the lean build project never resolves the IMAS/GACODE/TurbulentTransport
# weak deps, so TJLFEPIMASExt stays dormant. (CUDA is a hard TJLFEP dep and rides along in
# the manifest, but degrades gracefully on CPU nodes.)
#
# Builds against the registry-resolved "lean" environment created by setup_registry_env.jl
# (TGLFEP_JULIA_BUILD_ENV): released FuseRegistry versions only, reproducible by any user.
# Run on a CPU node.

using Pkg

build_env = get(ENV, "TGLFEP_JULIA_BUILD_ENV", "")
@assert !isempty(build_env) "TGLFEP_JULIA_BUILD_ENV must point at the project made by setup_registry_env.jl (variant: lean)"

Pkg.activate(build_env)
Pkg.instantiate()
using PackageCompiler

create_sysimage(
    [:TJLF, :TJLFEP];
    sysimage_path = get(ENV, "CPU_SYSIMAGE_OUT", normpath(@__DIR__, "..", "TGLFEP_JULIA_cpu_sysimage.so")),
    precompile_execution_file = normpath(@__DIR__, "precompile_cpu_workload.jl"),
    cpu_target = PackageCompiler.default_app_cpu_target(),
)
