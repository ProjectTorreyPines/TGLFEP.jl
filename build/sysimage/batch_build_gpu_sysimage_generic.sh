#!/bin/bash -l
# Build the GENERIC GPU sysimage (CUDA + TJLF + TJLFEP full + FUSE/IMAS stack) on a GPU node.
# Works for both the file-based scan and the IMAS/FUSE actor path on GPU.
# Output: build/TGLFEP_JULIA_gpu_generic_sysimage.so
#
#   cd build && sbatch sysimage/batch_build_gpu_sysimage_generic.sh
#
# Every package resolves from FuseRegistry at released versions (setup_registry_env.jl),
# so ANY user can run this from their own TJLFEP checkout — no maintainer dev tree.
# Version overrides: TGLFEP_JULIA_BUILD_VERSION (default: newest registered).
#
#SBATCH -A m3739_g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -C gpu
#SBATCH -J TJLFEP_gpu_gen_sysimg
#SBATCH -o build_gpu_generic_sysimage_%j.out
#SBATCH -e build_gpu_generic_sysimage_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -euo pipefail
# New files/dirs (CFS publishes, depot writes) must stay m3739 group-readable.
umask 007

module load cudatoolkit/12.9
module load julia/1.11.7
# TGLFEP_JULIA_DEPOT: base depot for the bake. For group-shareable images this MUST be the CFS
# depot (paths are baked into the image); defaults to the private scratch depot for
# personal builds.
# TGLFEP_JULIA_BAKE_DEPOT: optional isolated primary depot for the bake (stacked in front of
# the shared depot, which stays read-only for packages/artifacts). Avoids cross-flavor
# precompile-cache collisions (pkgimages=no emit vs pkgimages=true shared caches, or two
# projects resolving different versions of the same package into one compiled/ dir).
DEPOT_BASE="${TGLFEP_JULIA_DEPOT:-${PSCRATCH}/.julia}"
if [[ -n "${TGLFEP_JULIA_BAKE_DEPOT:-}" ]]; then
    export JULIA_DEPOT_PATH="${TGLFEP_JULIA_BAKE_DEPOT}:${DEPOT_BASE}"
else
    export JULIA_DEPOT_PATH="${DEPOT_BASE}"
fi
mkdir -p "${JULIA_DEPOT_PATH%%:*}/compiled"

# JULIA_CUDA_USE_COMPAT=false matches the runtime env.
export JULIA_CUDA_USE_COMPAT=false

# Submitted from the repo's build/ dir (see usage above); the sysimage scripts and the
# output .so live relative to it. Works from any user's checkout.
BUILD_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${BUILD_DIR}"

# GENERIC image: build against the registry-resolved "full" environment. IMAS/GACODE/
# TurbulentTransport are direct deps there and load the TJLFEPIMASExt extension (the
# dd/FUSE actor path) automatically; the precompile_execution file `using`s them -> the
# ext + FUSE are baked into the image.
export TGLFEP_JULIA_BUILD_ENV="${TGLFEP_JULIA_BUILD_ENV:-${BUILD_DIR}/sysimage/env_full}"
export TGLFEP_JULIA_BUILD_VARIANT=full

echo "=== TJLFEP GENERIC GPU sysimage build ==="
echo "host: $(hostname)  date: $(date)"
julia --version
nvidia-smi -L 2>/dev/null | head -1 || true

# Create/refresh the registry-resolved build environment, then precompile it.
julia sysimage/setup_registry_env.jl
julia --project="${TGLFEP_JULIA_BUILD_ENV}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

julia -t "${SLURM_CPUS_PER_TASK:-32}" sysimage/build_gpu_sysimage_generic.jl

SO="${BUILD_DIR}/TGLFEP_JULIA_gpu_generic_sysimage.so"
if [[ ! -f "${SO}" ]]; then
    echo "ERROR: sysimage not found at ${SO}"
    exit 1
fi
ls -lh "${SO}"

# Self-check: load the freshly built image and assert it is GENERIC -- the TJLFEPIMASExt
# extension is loaded (dd/FUSE actor path), FUSE is baked, and runTHD has the ::IMAS.dd
# method. Fail loudly otherwise so we never ship a silently file-only "generic" image.
echo "=== verifying generic image ==="
julia --startup-file=no --sysimage="${SO}" --project="${TGLFEP_JULIA_BUILD_ENV}" -e '
    fuse = Base.PkgId(Base.UUID("e64856f0-3bb8-4376-b4b7-c03396503992"), "FUSE")
    ext = Base.get_extension(TJLFEP, :TJLFEPIMASExt)
    @assert ext !== nothing  "GENERIC build FAILED: TJLFEPIMASExt not loaded (IMAS/GACODE/TurbulentTransport not baked)"
    @assert haskey(Base.loaded_modules, fuse)  "GENERIC build FAILED: FUSE not baked into image"
    @assert length(methods(TJLFEP.runTHD)) >= 2  "GENERIC build FAILED: runTHD(::IMAS.dd) method missing (actor path not compiled in)"
    println("generic image OK: TJLFEPIMASExt loaded, FUSE baked, runTHD methods=", length(methods(TJLFEP.runTHD)))'

# Publish to the shared CFS location: the image, a .sha sidecar recording the resolved
# package versions + julia version, and the build environment itself (Project/Manifest/
# LocalPreferences) — users run with --project pointed at that published env, which
# resolves exactly the versions baked into the image.
CFS_DIR="${TGLFEP_JULIA_CFS_DIR:-/global/cfs/cdirs/m3739/TJLFEP}"
if mkdir -p "${CFS_DIR}" 2>/dev/null; then
    cp -f "${SO}" "${CFS_DIR}/"
    chgrp m3739 "${CFS_DIR}/$(basename "${SO}")" 2>/dev/null || true
    SHA_FILE="${CFS_DIR}/$(basename "${SO}").sha"
    {
        julia --project="${TGLFEP_JULIA_BUILD_ENV}" -e '
            import Pkg
            for n in ("TJLFEP", "TJLF", "CUDA", "FUSE", "IMAS", "GACODE", "TurbulentTransport")
                for (_, info) in Pkg.dependencies()
                    info.name == n && println(n, "=v", info.version)
                end
            end'
        printf 'julia=%s\n' "$(julia --version | awk '{print $NF}')"
    } > "${SHA_FILE}"
    chgrp m3739 "${SHA_FILE}" 2>/dev/null || true
    ENV_PUB="${CFS_DIR}/env_full"
    mkdir -p "${ENV_PUB}"
    cp -f "${TGLFEP_JULIA_BUILD_ENV}"/{Project.toml,Manifest.toml,LocalPreferences.toml} "${ENV_PUB}/"
    chgrp -R m3739 "${ENV_PUB}" 2>/dev/null || true
    echo "=== published to ${CFS_DIR}/$(basename "${SO}") (runtime project: ${ENV_PUB}) ==="
    cat "${SHA_FILE}"
else
    echo "WARNING: could not write ${CFS_DIR}; sysimage left at ${SO} only"
fi

echo "=== GENERIC GPU sysimage build OK: ${SO} ==="
