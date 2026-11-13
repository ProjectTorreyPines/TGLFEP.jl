# Build the GENERIC GPU sysimage: CUDA + TJLF + TJLFEP (+ the TJLFEPIMASExt IMAS extension)
# + the FUSE/IMAS stack, with the GPU eigensolve path precompiled. Works for both the
# file-based scan (run_gacode_scan_task) and the IMAS/FUSE actor path on GPU. Because FUSE/IMAS
# are baked, the image still loads fast (the FUSE cost is compilation, which the image removes).
#
# Builds against the registry-resolved "full" environment created by setup_registry_env.jl
# (TGLFEP_JULIA_BUILD_ENV): every package at its released FuseRegistry version, so any user can
# reproduce the image — no maintainer dev tree involved. FUSE and the weak-dep trio are
# direct deps of that project, which is what lets create_sysimage bake them; listing
# IMAS/GACODE/TurbulentTransport makes Julia load them during the build, baking the
# TJLFEPIMASExt extension (the dd/FUSE actor entry points) into the image automatically.

using Pkg

build_env = get(ENV, "TGLFEP_JULIA_BUILD_ENV", "")
@assert !isempty(build_env) "TGLFEP_JULIA_BUILD_ENV must point at the project made by setup_registry_env.jl (variant: full)"

Pkg.activate(build_env)
Pkg.instantiate()

# An old FUSE (resolver fallback on a compat conflict) does not load against current IMAS;
# fail before the bake instead of an hour into it.
import TOML
let fuse = only(info for (_, info) in Pkg.dependencies() if info.name == "FUSE")
    vfiles = [joinpath(d, "registries", "FuseRegistry", "F", "FUSE", "Versions.toml") for d in DEPOT_PATH]
    vidx = findfirst(isfile, vfiles)
    @assert vidx !== nothing "no FuseRegistry Versions.toml for FUSE found in DEPOT_PATH"
    newest = maximum(VersionNumber.(keys(TOML.parsefile(vfiles[vidx]))))
    @assert fuse.version == newest "build env has FUSE $(fuse.version), newest registered is $newest; rerun setup_registry_env.jl (TGLFEP_JULIA_BUILD_CUDA >= 5.9)"
    println("FUSE $(fuse.version) (newest registered)")
end

using PackageCompiler

create_sysimage(
    [:TJLF, :TJLFEP, :FUSE, :IMAS, :GACODE, :TurbulentTransport];
    sysimage_path = normpath(@__DIR__, "..", "TGLFEP_JULIA_gpu_generic_sysimage.so"),
    precompile_execution_file = normpath(@__DIR__, "precompile_gpu_workload_generic.jl"),
    project = build_env,
    cpu_target = PackageCompiler.default_app_cpu_target(),
)
