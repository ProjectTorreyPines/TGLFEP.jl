#!/bin/bash -l
# Build the FILE-ONLY GPU sysimage (CUDA + TJLF + TJLFEP standalone, no FUSE/IMAS) on a GPU node.
# This is what a TGLF-EP user running the file-based scan path gets before going FUSE-native:
# leaner + faster-loading than the generic image because the IMAS/FUSE stack is not baked.
# Output: build/TGLFEP_JULIA_gpu_sysimage.so
#
#   cd build && sbatch sysimage/batch_build_gpu_sysimage_fileonly.sh
#
# Every package resolves from FuseRegistry at released versions (setup_registry_env.jl),
# so ANY user can run this from their own TJLFEP checkout — no maintainer dev tree.
# Version overrides: TGLFEP_JULIA_BUILD_VERSION (default: newest registered).
#
#SBATCH -A m3739_g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -J TJLFEP_gpu_fo_sysimg
#SBATCH -o build_gpu_fileonly_sysimage_%j.out
#SBATCH -e build_gpu_fileonly_sysimage_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -euo pipefail
# New files/dirs (CFS publishes, depot writes) must stay m3739 group-readable.
umask 007

module load cudatoolkit/12.9
module load julia/1.11.7
# Drop the site environment the julia module appends: it switches packages to the system
# Cray MPI, and HDF5_jll then fails to precompile against the system OpenSSL. The CUDA
# runtime pin lives in the build project (setup_registry_env.jl).
export JULIA_LOAD_PATH="@:@v#.#:@stdlib"
# TGLFEP_JULIA_DEPOT: base depot for the bake. For group-shareable images this MUST be the CFS
# depot (paths are baked into the image); defaults to the private scratch depot.
export JULIA_DEPOT_PATH="${TGLFEP_JULIA_DEPOT:-${PSCRATCH}/.julia}"
mkdir -p "${JULIA_DEPOT_PATH}/compiled"

# JULIA_CUDA_USE_COMPAT=false matches the runtime worker env.
export JULIA_CUDA_USE_COMPAT=false

# Submitted from the repo's build/ dir (see usage above); works from any user's checkout.
BUILD_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${BUILD_DIR}"

# FILE-ONLY image: build against the registry-resolved "lean" environment, whose manifest
# never resolves the IMAS/GACODE/TurbulentTransport weak deps -- the TJLFEPIMASExt
# extension stays dormant and FUSE/IMAS are never pulled into the image.
export TGLFEP_JULIA_BUILD_ENV="${TGLFEP_JULIA_BUILD_ENV:-${BUILD_DIR}/sysimage/env_lean}"
export TGLFEP_JULIA_BUILD_VARIANT=lean

echo "=== TJLFEP FILE-ONLY GPU sysimage build ==="
echo "host: $(hostname)  date: $(date)"
julia --version
nvidia-smi -L 2>/dev/null | head -1 || true

# Create/refresh the registry-resolved build environment, then precompile it.
julia sysimage/setup_registry_env.jl
julia --project="${TGLFEP_JULIA_BUILD_ENV}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

julia -t "${SLURM_CPUS_PER_TASK:-32}" sysimage/build_gpu_sysimage_fileonly.jl

SO="${BUILD_DIR}/TGLFEP_JULIA_gpu_sysimage.so"
if [[ ! -f "${SO}" ]]; then
    echo "ERROR: sysimage not found at ${SO}"
    exit 1
fi
ls -lh "${SO}"

# Self-check: load the freshly built image and assert it is FILE-ONLY -- the TJLFEPIMASExt
# extension is NOT loaded and FUSE is NOT baked. Fail loudly otherwise so we never ship a
# silently generic image under the file-only name.
echo "=== verifying file-only image ==="
julia --startup-file=no --sysimage="${SO}" --project="${TGLFEP_JULIA_BUILD_ENV}" -e '
    fuse = Base.PkgId(Base.UUID("e64856f0-3bb8-4376-b4b7-c03396503992"), "FUSE")
    ext = Base.get_extension(TJLFEP, :TJLFEPIMASExt)
    @assert ext === nothing  "FILE-ONLY build FAILED: TJLFEPIMASExt is loaded (IMAS/GACODE/TurbulentTransport got baked)"
    @assert !haskey(Base.loaded_modules, fuse)  "FILE-ONLY build FAILED: FUSE is baked into image"
    println("file-only image OK: TJLFEPIMASExt dormant, FUSE not baked, runTHD methods=", length(methods(TJLFEP.runTHD)))'

# Publish to the shared CFS location: the image, a .sha sidecar with the resolved package
# versions (TJLF/TJLFEP/CUDA only: no FUSE/IMAS in this image), and the lean build env
# (users run with --project pointed at the published env).
CFS_DIR="${TGLFEP_JULIA_CFS_DIR:-/global/cfs/cdirs/m3739/TJLFEP}"
if mkdir -p "${CFS_DIR}" 2>/dev/null; then
    cp -f "${SO}" "${CFS_DIR}/"
    chgrp m3739 "${CFS_DIR}/$(basename "${SO}")" 2>/dev/null || true
    SHA_FILE="${CFS_DIR}/$(basename "${SO}").sha"
    {
        julia --project="${TGLFEP_JULIA_BUILD_ENV}" -e '
            import Pkg
            for n in ("TJLFEP", "TJLF", "CUDA")
                for (_, info) in Pkg.dependencies()
                    info.name == n && println(n, "=v", info.version)
                end
            end'
        printf 'julia=%s\n' "$(julia --version | awk '{print $NF}')"
    } > "${SHA_FILE}"
    chgrp m3739 "${SHA_FILE}" 2>/dev/null || true
    ENV_PUB="${CFS_DIR}/env_lean"
    mkdir -p "${ENV_PUB}"
    cp -f "${TGLFEP_JULIA_BUILD_ENV}"/{Project.toml,Manifest.toml,LocalPreferences.toml} "${ENV_PUB}/"
    chgrp -R m3739 "${ENV_PUB}" 2>/dev/null || true
    echo "=== published to ${CFS_DIR}/$(basename "${SO}") (runtime project: ${ENV_PUB}) ==="
    cat "${SHA_FILE}"
else
    echo "WARNING: could not write ${CFS_DIR}; sysimage left at ${SO} only"
fi

echo "=== FILE-ONLY GPU sysimage build OK: ${SO} ==="
