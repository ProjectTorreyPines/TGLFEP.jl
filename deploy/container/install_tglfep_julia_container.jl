# Install TJLFEP + dependencies into a container image, with the CUDA runtime
# artifact baked in so the image works offline on GPU nodes.
#
# Adapted from FUSE/deploy/perlmutter-container/install_fuse_container.jl.
# Deltas: no sysimage (the GPU-traced sysimage is built later, in-image, on a
# GPU node — see gpu-sysimage/), no IJulia/Jupyter, explicit package sets
# instead of Makefile parsing, and a CUDA runtime version pin written BEFORE
# any CUDA-touching package is added (the build host has no GPU, so CUDA.jl
# cannot autodetect a driver to size the runtime against).
#
# Driven entirely by environment variables set in the Containerfile:
#   TGLFEP_JULIA_INSTALL_DIR   install/depot location   (default /opt/tglfep_julia)
#   TGLFEP_JULIA_VARIANT       "lean" | "imas"          (default lean)
#   JULIA_CPU_TARGET     pkgimage CPU target      (required)

@assert (Threads.nthreads() == 1) "Error: container install requires running Julia with one thread"
@assert ("JULIA_CPU_TARGET" in keys(ENV)) "Error: Must define JULIA_CPU_TARGET environment variable"

install_dir = get(ENV, "TGLFEP_JULIA_INSTALL_DIR", "/opt/tglfep_julia")
variant = get(ENV, "TGLFEP_JULIA_VARIANT", "lean")
@assert variant in ("lean", "imas") "Error: TGLFEP_JULIA_VARIANT must be \"lean\" or \"imas\", got \"$variant\""

import Pkg

println("### Setup main environment for installer (variant: $variant)")
Pkg.activate()
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ProjectTorreyPines/FuseRegistry.jl.git"))
Pkg.Registry.add("General")
Pkg.update()  # also flips Pkg's "registry updated this session" flag so the
              # Pkg.add() below will not re-fetch (and revert) the registry edits

println()
println("### Rewrite FuseRegistry SSH (git@github.com:) URLs to anonymous HTTPS")
# The FuseRegistry stores package repos as git@github.com: URLs. Julia's CLI git
# tries to clone those over SSH, which fails in a credential-free container
# (no ssh binary / no keys). Since all ProjectTorreyPines packages are public,
# rewrite the cloned registry's Package.toml repo fields to HTTPS so clones work
# anonymously. This is independent of git config / $HOME, so it always applies.
function rewrite_registry_ssh_to_https()
    n = 0
    for depot in DEPOT_PATH
        regroot = joinpath(depot, "registries")
        isdir(regroot) || continue
        for (root, _, files) in walkdir(regroot)
            for f in files
                f == "Package.toml" || continue
                path = joinpath(root, f)
                s = read(path, String)
                s2 = replace(s, "git@github.com:" => "https://github.com/")
                if s2 != s
                    chmod(path, 0o644)
                    write(path, s2)
                    n += 1
                end
            end
        end
    end
    return n
end
println("    rewrote ", rewrite_registry_ssh_to_https(), " Package.toml file(s)")

println()
println("### Pin CUDA runtime to 12.9 (artifact source) BEFORE adding CUDA packages")
# The build host has no GPU driver, so CUDA_Runtime_jll's platform augmentation
# would select no runtime ("none") and the image would try to download ~3 GB on
# first GPU use — impossible offline on compute nodes. Pinning `version` makes
# artifact selection driver-independent: Pkg.add below downloads and bakes the
# 12.9 runtime. 12.9 is the validated Perlmutter config (cudatoolkit/12.9 +
# CUDA.jl 5.8.5); TJLF's TJLFCUDAExt floor is 12.6 (cusolverDnXgeev); newer
# H100/H200-era drivers run a 12.9 runtime via CUDA minor-version compat.
# `local = "false"` = use the artifact, never a host toolkit.
# These are exactly the preferences CUDA.set_runtime_version!(v"12.9";
# local_toolkit=false) would write (stringified, on CUDA_Runtime_jll) — written
# by hand here because they must exist before CUDA precompiles, and Preferences
# needs the UUID in the project's [extras] to map the section name.
const CUDA_RUNTIME_JLL_UUID = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
mkpath(install_dir)
write(joinpath(install_dir, "Project.toml"),
    """
    [extras]
    CUDA_Runtime_jll = "$CUDA_RUNTIME_JLL_UUID"
    """)
write(joinpath(install_dir, "LocalPreferences.toml"),
    """
    [CUDA_Runtime_jll]
    version = "12.9"
    local = "false"
    """)
println("    wrote $(joinpath(install_dir, "LocalPreferences.toml"))")

println()
println("### Resolve TJLFEP version to install")
# TJLFEP MUST be added at an explicit version: TJLFEP/TJLF cap CUDA at 5.8.x,
# and with `CUDA` in the same unversioned Pkg.add the resolver prefers the
# newest CUDA from General and silently holds TJLFEP back (observed: v2.0.0 +
# TJLF v1.2.3 instead of v2.0.14 + v1.3.1). Pinning TJLFEP forces CUDA down to
# the compatible 5.8.x instead.
import TOML
tglfep_julia_version = String(lstrip(get(ENV, "TGLFEP_JULIA_BUILD_VERSION", ""), 'v'))
if isempty(tglfep_julia_version)
    # fall back to the highest version registered in the FuseRegistry clone
    vfile = joinpath(first(DEPOT_PATH), "registries", "FuseRegistry", "T", "TJLFEP", "Versions.toml")
    @assert isfile(vfile) "TGLFEP_JULIA_BUILD_VERSION unset and no registry Versions.toml at $vfile"
    tglfep_julia_version = string(maximum(VersionNumber.(keys(TOML.parsefile(vfile)))))
end
println("    TJLFEP $tglfep_julia_version")

println()
println("### Setup TJLFEP environment in ", install_dir)
Pkg.activate(install_dir)
# TJLF, CUDA, PackageCompiler are TJLFEP deps already; listing them makes them
# directly loadable/importable in the project (the gpu-sysimage bake and user
# scripts do `using TJLF, CUDA`). CUDA is pinned exactly: TJLFEP's caret compat
# ("5.8.5" = <6) admits newer CUDA.jl, but 5.8.5 is the config every Perlmutter
# GPU run and sysimage was validated against.
packages = [Pkg.PackageSpec(name="TJLFEP", version=tglfep_julia_version),
            Pkg.PackageSpec(name="TJLF"),
            Pkg.PackageSpec(name="CUDA", version="5.8.5"),
            Pkg.PackageSpec(name="PackageCompiler")]
if variant == "imas"
    append!(packages, [Pkg.PackageSpec(name="IMAS"), Pkg.PackageSpec(name="GACODE"),
                       Pkg.PackageSpec(name="TurbulentTransport")])
end
println("    ", [p.name for p in packages])
Pkg.add(packages)

println()
println("### Eagerly install lazy artifacts so the image runs offline")
# Pkg.add/instantiate skip artifacts marked `lazy = true`; those normally
# download on first @artifact_str access at runtime. Walk every
# (Julia)Artifacts.toml in the depot and download all artifacts for this
# platform, including lazy ones, so they are baked into the image.
import Pkg.Artifacts: select_downloadable_artifacts, ensure_artifact_installed
function install_all_artifacts(depot)
    installed = 0
    pkgsdir = joinpath(depot, "packages")
    isdir(pkgsdir) || return installed
    for (root, _, files) in walkdir(pkgsdir)
        for f in files
            (f == "Artifacts.toml" || f == "JuliaArtifacts.toml") || continue
            toml = joinpath(root, f)
            local arts
            try
                arts = select_downloadable_artifacts(toml; include_lazy=true)
            catch err
                @warn "Could not parse $toml" exception = err
                continue
            end
            for (name, meta) in arts
                try
                    ensure_artifact_installed(name, meta, toml)
                    installed += 1
                catch err
                    @warn "Failed to install artifact $name from $toml" exception = err
                end
            end
        end
    end
    return installed
end
println("    installed/verified ", install_all_artifacts(first(DEPOT_PATH)), " downloadable artifact(s)")

if variant == "imas"
    println()
    println("### Materialize TurbulentTransport Git LFS model files")
    # TurbulentTransport ships its NN weights via Git LFS; a Pkg install leaves
    # ~130-byte pointer stubs that the package downloads into its own package
    # dir on first use — impossible later in a read-only image (SIF).
    # Materialize every model now so the image is complete and works offline.
    import TurbulentTransport
    if isdefined(TurbulentTransport, :ensure_model_file!) && isdefined(TurbulentTransport, :is_lfs_pointer)
        let root = joinpath(pkgdir(TurbulentTransport), "models"), n = 0
            for (dir, _, files) in walkdir(root), f in files
                path = joinpath(dir, f)
                if TurbulentTransport.is_lfs_pointer(path)
                    TurbulentTransport.ensure_model_file!(path)
                    n += 1
                end
            end
            println("    materialized $n model file(s)")
        end
    else
        @warn "TurbulentTransport LFS helpers not found — models may remain pointer stubs in the image"
    end
end

println()
println("### Sanity checks")
# The CUDA 12.9 runtime artifact must be baked (selected via the preference pin
# even without a GPU/driver on the build host).
crj = Base.require(Base.PkgId(Base.UUID(CUDA_RUNTIME_JLL_UUID), "CUDA_Runtime_jll"))
@assert crj.is_available() "CUDA_Runtime_jll reports unavailable — the 12.9 preference pin did not take"
@assert isdir(crj.artifact_dir) "CUDA_Runtime_jll artifact_dir missing"
println("    CUDA runtime artifact: ", crj.artifact_dir)

import TJLFEP, TJLF, CUDA
println("    TJLFEP $(pkgversion(TJLFEP)), TJLF $(pkgversion(TJLF)), CUDA $(pkgversion(CUDA)) load clean")
@assert string(pkgversion(TJLFEP)) == tglfep_julia_version "resolver installed TJLFEP $(pkgversion(TJLFEP)), expected $tglfep_julia_version"
# No GPU at build time: CUDA must degrade gracefully, not throw.
@assert CUDA.functional() == false "unexpected: build host reports a functional GPU"
if variant == "imas"
    import IMAS, GACODE, TurbulentTransport
    @assert Base.get_extension(TJLFEP, :TJLFEPIMASExt) !== nothing "TJLFEPIMASExt did not load in the imas variant"
    println("    TJLFEPIMASExt loaded (IMAS $(pkgversion(IMAS)), GACODE $(pkgversion(GACODE)), TurbulentTransport $(pkgversion(TurbulentTransport)))")
else
    @assert Base.get_extension(TJLFEP, :TJLFEPIMASExt) === nothing "TJLFEPIMASExt unexpectedly loaded in the lean variant"
end

println()
println("### TJLFEP container install complete ($variant)")
