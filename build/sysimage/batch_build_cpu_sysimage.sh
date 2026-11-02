#!/bin/bash -l
# Build the full CPU sysimage (TJLF + TJLFEP, file-only) on a CPU node, with the CPU
# eigensolve path precompiled. Output: build/TGLFEP_JULIA_cpu_sysimage.so
#
#   cd build && sbatch sysimage/batch_build_cpu_sysimage.sh
#
# Every package resolves from FuseRegistry at released versions (setup_registry_env.jl),
# so ANY user can run this from their own TJLFEP checkout — no maintainer dev tree.
# Version overrides: TGLFEP_JULIA_BUILD_VERSION (default: newest registered).
#
#SBATCH -A m3739
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -C cpu
#SBATCH -J TJLFEP_cpu_sysimg
#SBATCH -o build_cpu_sysimage_%j.out
#SBATCH -e build_cpu_sysimage_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

set -euo pipefail
# New files/dirs (CFS publishes, depot writes) must stay m3739 group-readable.
umask 007

module load julia/1.11.7
# TGLFEP_JULIA_DEPOT: base depot for the bake. For group-shareable images this MUST be the CFS
# depot (paths are baked into the image); defaults to the private scratch depot.
export JULIA_DEPOT_PATH="${TGLFEP_JULIA_DEPOT:-${PSCRATCH}/.julia}${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"
mkdir -p "${JULIA_DEPOT_PATH%%:*}/compiled"

# Submitted from the repo's build/ dir (see usage above); works from any user's checkout.
BUILD_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${BUILD_DIR}"

# File-only image by construction: built against the registry-resolved "lean" environment,
# whose manifest never resolves the IMAS/GACODE/TurbulentTransport weak deps, so
# TJLFEPIMASExt stays dormant. (CUDA rides along as a hard TJLFEP dep; on a CPU node it
# simply reports non-functional. Don't run this and the file-only GPU build concurrently —
# they share the env_lean project dir.)
export TGLFEP_JULIA_BUILD_ENV="${TGLFEP_JULIA_BUILD_ENV:-${BUILD_DIR}/sysimage/env_lean}"
export TGLFEP_JULIA_BUILD_VARIANT=lean

echo "=== TJLFEP full CPU sysimage build (TJLF + TJLFEP) ==="
echo "host: $(hostname)  date: $(date)"
julia --version

# Create/refresh the registry-resolved build environment, then precompile it.
julia sysimage/setup_registry_env.jl
julia --project="${TGLFEP_JULIA_BUILD_ENV}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

julia -t "${SLURM_CPUS_PER_TASK:-128}" sysimage/build_cpu_sysimage.jl

SO="${CPU_SYSIMAGE_OUT:-${BUILD_DIR}/TGLFEP_JULIA_cpu_sysimage.so}"
if [[ ! -f "${SO}" ]]; then
    echo "ERROR: sysimage not found at ${SO}"
    exit 1
fi
ls -lh "${SO}"

# Self-check: load with the image and confirm TJLF + TJLFEP are baked (instant load).
echo "=== verifying CPU image ==="
julia --startup-file=no --sysimage="${SO}" --project="${TGLFEP_JULIA_BUILD_ENV}" -e '
    t=time(); using TJLF, TJLFEP
    println("load(using TJLF,TJLFEP)=", round(time()-t;digits=3), "s")
    @assert isdefined(TJLF, :run_tjlf) "TJLF not baked"
    @assert isdefined(TJLFEP, :run_gacode_scan_task) "TJLFEP not baked"
    heavy = filter(m -> nameof(m) in (:FUSE, :IMAS), Base.loaded_modules_array())
    println("file-only check: heavy deps loaded = ", isempty(heavy) ? "none" : nameof.(heavy))
    @assert isempty(heavy) "file-only image pulled in FUSE/IMAS"
    println("CPU sysimage OK: TJLF + TJLFEP baked (file-only)")'

# Publish to the shared CFS location: the image, a .sha sidecar with the resolved package
# versions (TJLF/TJLFEP only: no FUSE/IMAS in this image), and the lean build env
# (users run with --project pointed at the published env).
CFS_DIR="${TGLFEP_JULIA_CFS_DIR:-/global/cfs/cdirs/m3739/TJLFEP}"
if mkdir -p "${CFS_DIR}" 2>/dev/null; then
    cp -f "${SO}" "${CFS_DIR}/"
    chgrp m3739 "${CFS_DIR}/$(basename "${SO}")" 2>/dev/null || true
    SHA_FILE="${CFS_DIR}/$(basename "${SO}").sha"
    {
        julia --project="${TGLFEP_JULIA_BUILD_ENV}" -e '
            import Pkg
            for n in ("TJLFEP", "TJLF")
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

echo "=== full CPU sysimage build OK: ${SO} ==="
