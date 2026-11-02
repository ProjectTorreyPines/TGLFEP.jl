# Build the GPU-traced TJLFEP sysimage INSIDE the container, on a GPU node.
#
# Run via batch_bake_gpu_sysimage.sh — never in CI: the precompile workload
# executes real GPU solves, so it needs a functional A100 behind
# `podman-hpc run --gpu`. The depot paths baked into the .so match the final
# -gpu image because this runs in the very image the .so is layered onto
# (Containerfile.gpu); the container's writable overlay absorbs the temp
# objects and only $SYSIMAGE_PATH (a bind mount) persists.
#
# Invoked with plain `julia` (NOT the tglfep_julia launcher, which unsets
# JULIA_CPU_TARGET for JIT use — here we need the image's multi-target value).

import Pkg
Pkg.activate("/opt/tglfep_julia")
using PackageCompiler

# The image bakes a multi-target JULIA_CPU_TARGET (znver3/znver4/
# sapphirerapids/generic) so one .so serves Perlmutter and Defiant-era hosts.
# Do NOT fall back to PackageCompiler.default_app_cpu_target() — that traces
# the build host only (Milan), which is exactly the portability bug the
# in-repo fileonly builder has.
@assert haskey(ENV, "JULIA_CPU_TARGET") "JULIA_CPU_TARGET must be set (baked into the image ENV)"
cpu_target = ENV["JULIA_CPU_TARGET"]

sysimage_path = get(ENV, "SYSIMAGE_PATH", "/out/sys_tglfep_julia_gpu.so")
mkpath(dirname(sysimage_path))

# Two modes:
#   - default: run the GPU workload under --trace-compile as part of the bake
#     (single job on a GPU node — the Perlmutter flow)
#   - PRECOMPILE_STATEMENTS_FILE=...: reuse a statements file traced earlier on
#     a GPU node and compile WITHOUT a GPU (replaying precompile statements is
#     type-level only). This splits the bake into a short GPU trace job and a
#     CPU-partition compile job — essential on Defiant where GPU nodes are
#     scarce (see defiant_*.sbatch).
stmts = get(ENV, "PRECOMPILE_STATEMENTS_FILE", "")

println("### Baking GPU sysimage -> $sysimage_path")
println("    cpu_target: $cpu_target")
println("    precompile source: ", isempty(stmts) ? "execution file (GPU workload)" : "statements file $stmts")

if isempty(stmts)
    create_sysimage(
        [:CUDA, :TJLF, :TJLFEP];
        sysimage_path,
        precompile_execution_file = joinpath(@__DIR__, "precompile_gpu_workload_container.jl"),
        project = "/opt/tglfep_julia",
        cpu_target,
    )
else
    @assert isfile(stmts) "missing statements file $stmts"
    create_sysimage(
        [:CUDA, :TJLF, :TJLFEP];
        sysimage_path,
        precompile_statements_file = stmts,
        project = "/opt/tglfep_julia",
        cpu_target,
    )
end

println("### GPU sysimage bake complete: ", sysimage_path,
        " (", round(filesize(sysimage_path) / 2^30; digits=2), " GiB)")
