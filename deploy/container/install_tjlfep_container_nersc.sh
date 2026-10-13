#!/bin/bash
# One-command TJLFEP container setup for NERSC Perlmutter.
#
# Gets the published TJLFEP image usable on this account (reusing a shared
# squash store when possible, otherwise pulling from ghcr.io). TJLFEP then
# runs with no Julia install and no compilation:
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/ProjectTorreyPines/TGLFEP.jl/master/deploy/container/install_tjlfep_container_nersc.sh)
#
# Optional:
#   TJLFEP_ENVIRONMENT=v2.0.14        image tag (append -imas / -gpu for those
#                                     variants; default: newest in FuseRegistry)
#   SQUASH_DIR=...                    force a specific shared image store

set -euo pipefail

shared_store="/global/cfs/cdirs/m3739/shared_images"
registry_image_base="ghcr.io/projecttorreypines/tjlfep"

command -v podman-hpc >/dev/null || { echo "ERROR: podman-hpc not found — run this on Perlmutter." >&2; exit 1; }

if [[ -n "${TJLFEP_ENVIRONMENT:-}" ]]; then
    version="$TJLFEP_ENVIRONMENT"
else
    # newest FuseRegistry-registered version (the repo publishes no GitHub releases)
    version="v$(curl -sf https://raw.githubusercontent.com/ProjectTorreyPines/FuseRegistry.jl/master/T/TJLFEP/Versions.toml \
        | sed -n 's/^\["\(.*\)"\]$/\1/p' | sort -V | tail -1)"
fi
[[ -n "$version" && "$version" != "v" ]] || { echo "ERROR: could not determine TJLFEP version (set TJLFEP_ENVIRONMENT)." >&2; exit 1; }

has_image() {  # has_image <extra podman-hpc args...>
    podman-hpc "$@" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
        | grep -qx "localhost/tjlfep:$version"
}

# --- pick an image store -----------------------------------------------------
# Prefer the shared store when it already holds the requested version; fall
# back to pulling from ghcr.io into the per-user store otherwise.
squash_dir="${SQUASH_DIR:-}"
if [[ -z "$squash_dir" ]]; then
    if ! id -nG | tr ' ' '\n' | grep -qx m3739 || [[ ! -r "$shared_store" ]]; then
        echo "### m3739 shared store not accessible — using your per-user image store"
    elif has_image --squash-dir "$shared_store"; then
        squash_dir="$shared_store"
        echo "### Using the shared m3739 image store (no pull needed): $squash_dir"
    else
        echo "### Shared m3739 store does not have tjlfep:$version yet — falling back to a registry pull"
    fi
fi

if [[ -n "$squash_dir" ]]; then
    has_image --squash-dir "$squash_dir" \
        || { echo "ERROR: localhost/tjlfep:$version not found in $squash_dir" >&2; exit 1; }
elif ! has_image; then
    echo "### Pulling $registry_image_base:$version (~9 GB, several minutes)"
    podman-hpc pull "$registry_image_base:$version"
    podman-hpc tag "$registry_image_base:$version" "localhost/tjlfep:$version"
    podman-hpc migrate "tjlfep:$version"
else
    echo "### localhost/tjlfep:$version already in your image store"
fi

sq=""; [[ -n "$squash_dir" ]] && sq=" --squash-dir $squash_dir"
echo
echo "### TJLFEP container ready."
echo "Interactive REPL (login node, CPU):"
echo "    podman-hpc$sq run --rm -it localhost/tjlfep:$version"
echo "GPU (inside a gpu allocation):"
echo "    podman-hpc$sq run --rm --gpu -it localhost/tjlfep:$version"
echo "Run a file-based scan task (mount your case directory):"
echo "    podman-hpc$sq run --rm --gpu -v \$PWD:/case localhost/tjlfep:$version tjlfep -e \\"
echo "        'import TJLFEP; TJLFEP.run_gacode_scan_task(\"/case/input.gacode\", \"/case/input.TGLFEP\", 1; out_dir=\"/case\", use_gpu=true)'"
