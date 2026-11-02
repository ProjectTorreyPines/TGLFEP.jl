# Create the registry-resolved sysimage BUILD ENVIRONMENT: a plain Julia project whose
# packages all come from FuseRegistry/General at released versions — no Pkg.develop'ed
# repos, no maintainer dev tree. Any user can run this; it is the same philosophy as the
# TJLFEP container install (deploy/container/install_tglfep_julia_container.jl), from which the
# version-resolution and pinning logic is adapted.
#
# Driven by environment variables (set by the batch_build_*.sh scripts):
#   TGLFEP_JULIA_BUILD_ENV       project dir to create/refresh          (required)
#   TGLFEP_JULIA_BUILD_VARIANT   "lean" | "full"                        (default lean)
#                          lean: TJLFEP + TJLF + CUDA (file-based path; the
#                                TJLFEPIMASExt weak-dep extension stays dormant)
#                          full: + FUSE/IMAS/GACODE/TurbulentTransport (generic
#                                image; the extension and the actor path bake in)
#   TGLFEP_JULIA_BUILD_VERSION   TJLFEP version to install (e.g. "2.0.14"); default =
#                          newest version registered in FuseRegistry
#
# CUDA runtime: pinned to the module-provided 12.9 toolkit (`local = "true"`), the
# validated Perlmutter config. The julia module stacks equivalent preferences keyed on
# the loaded cudatoolkit module, but PackageCompiler strips the load-path stack, so the
# pin must live INSIDE the project (see the NERSC CUDA-prefs notes in build/README.md).
# Run with `module load cudatoolkit/12.9 julia/1.11.7` (that order).

import Pkg
import TOML

build_env = get(ENV, "TGLFEP_JULIA_BUILD_ENV", "")
@assert !isempty(build_env) "TGLFEP_JULIA_BUILD_ENV must point at the build-project directory"
variant = get(ENV, "TGLFEP_JULIA_BUILD_VARIANT", "lean")
@assert variant in ("lean", "full") "TGLFEP_JULIA_BUILD_VARIANT must be \"lean\" or \"full\", got \"$variant\""

println("### Ensure registries (FuseRegistry + General)")
Pkg.activate()
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ProjectTorreyPines/FuseRegistry.jl.git"))
Pkg.Registry.add("General")
Pkg.update()  # also flips Pkg's "registry updated this session" flag so the
              # Pkg.add() below will not re-fetch (and revert) the registry edits

# The FuseRegistry stores package repos as git@github.com: URLs; users without GitHub SSH
# keys cannot clone those. All ProjectTorreyPines packages are public — rewrite to HTTPS.
println("### Rewrite FuseRegistry SSH (git@github.com:) URLs to anonymous HTTPS")
let n = 0
    for depot in DEPOT_PATH
        regroot = joinpath(depot, "registries")
        isdir(regroot) || continue
        for (root, _, files) in walkdir(regroot), f in files
            f == "Package.toml" || continue
            path = joinpath(root, f)
            s = read(path, String)
            s2 = replace(s, "git@github.com:" => "https://github.com/")
            if s2 != s
                try
                    chmod(path, 0o644)
                    write(path, s2)
                    n += 1
                catch err  # read-only shared depot: registry there is already usable
                    @warn "could not rewrite $path" exception = err
                end
            end
        end
    end
    println("    rewrote ", n, " Package.toml file(s)")
end

println("### Pin CUDA runtime to the 12.9 module toolkit BEFORE adding CUDA packages")
# Equivalent to CUDA.set_runtime_version!(v"12.9"; local_toolkit=true), written by hand
# because it must exist before CUDA precompiles, and Preferences needs the UUID in the
# project's [extras] to map the section name.
const CUDA_RUNTIME_JLL_UUID = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
mkpath(build_env)
write(joinpath(build_env, "Project.toml"),
    """
    [extras]
    CUDA_Runtime_jll = "$CUDA_RUNTIME_JLL_UUID"
    """)
write(joinpath(build_env, "LocalPreferences.toml"),
    """
    [CUDA_Runtime_jll]
    version = "12.9"
    local = "true"
    """)

println("### Resolve TJLFEP version to install")
# TJLFEP MUST be added at an explicit version: TJLFEP/TJLF cap CUDA at 5.8.x, and with
# `CUDA` in the same unversioned Pkg.add the resolver prefers the newest CUDA from
# General and silently holds TJLFEP back (observed: v2.0.0, which doesn't precompile).
tglfep_julia_version = String(lstrip(get(ENV, "TGLFEP_JULIA_BUILD_VERSION", ""), 'v'))
if isempty(tglfep_julia_version)
    # findfirst, not a bare for-loop assignment: at script top level the loop body is
    # soft scope, so `vfile = f` would bind a new local and leave the global "" empty
    vfiles = [joinpath(depot, "registries", "FuseRegistry", "T", "TJLFEP", "Versions.toml")
              for depot in DEPOT_PATH]
    vidx = findfirst(isfile, vfiles)
    vfile = vidx === nothing ? "" : vfiles[vidx]
    @assert !isempty(vfile) "TGLFEP_JULIA_BUILD_VERSION unset and no FuseRegistry Versions.toml found in DEPOT_PATH"
    tglfep_julia_version = string(maximum(VersionNumber.(keys(TOML.parsefile(vfile)))))
end
println("    TJLFEP $tglfep_julia_version")

println("### Setup $variant build environment in ", build_env)
Pkg.activate(build_env)
# TJLF, CUDA, PackageCompiler are TJLFEP deps already; listing them makes them direct
# deps, importable by name in the bake scripts. CUDA is pinned exactly: TJLFEP's caret
# compat ("5.8.5" = <6) admits newer CUDA.jl, but 5.8.5 is the config every Perlmutter
# GPU run and sysimage was validated against. The full variant lists FUSE and the
# IMAS/GACODE/TurbulentTransport weak-dep trio directly so create_sysimage can bake them
# and the TJLFEPIMASExt actor path compiles in.
packages = [Pkg.PackageSpec(name="TJLFEP", version=tglfep_julia_version),
            Pkg.PackageSpec(name="TJLF"),
            Pkg.PackageSpec(name="CUDA", version="5.8.5"),
            Pkg.PackageSpec(name="PackageCompiler")]
if variant == "full"
    append!(packages, [Pkg.PackageSpec(name="FUSE"), Pkg.PackageSpec(name="IMAS"),
                       Pkg.PackageSpec(name="GACODE"), Pkg.PackageSpec(name="TurbulentTransport")])
end
println("    ", [p.name for p in packages])
Pkg.add(packages)

println("### Sanity checks")
deps = Dict(info.name => info.version for (_, info) in Pkg.dependencies())
@assert string(deps["TJLFEP"]) == tglfep_julia_version "resolver installed TJLFEP $(deps["TJLFEP"]), expected $tglfep_julia_version (co-added dep held it back?)"
for p in ("TJLF", "CUDA", (variant == "full" ? ("FUSE", "IMAS", "GACODE", "TurbulentTransport") : ())...)
    println("    $p $(deps[p])")
end

println("### $variant build environment ready: ", build_env)
