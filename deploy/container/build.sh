#!/bin/bash
# Build and migrate the TJLFEP podman-hpc image on Perlmutter.
#
# The build is moderate (precompiles the TJLFEP stack, ~3 GB CUDA runtime
# artifact download, no sysimage) and is fine on a login node:
#
#   ./deploy/container/build.sh                 # lean variant
#   TJLFEP_VARIANT=imas ./deploy/container/build.sh
#
# Override the image tag (defaults to the latest FuseRegistry-registered
# TJLFEP version — the repo publishes no GitHub releases) with:
#   TJLFEP_ENVIRONMENT=v2.0.14 ./deploy/container/build.sh
#
# Share the migrated image across the project (instead of personal SCRATCH):
#   SQUASH_DIR=/global/cfs/cdirs/m3739/shared_images ./deploy/container/build.sh

set -euo pipefail

scriptdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

variant="${TJLFEP_VARIANT:-lean}"
case "$variant" in
    lean|imas) ;;
    *) echo "ERROR: TJLFEP_VARIANT must be 'lean' or 'imas', got '$variant'" >&2; exit 1 ;;
esac

# Resolve the image tag: explicit override, else the highest version registered
# in FuseRegistry (TGLFEP.jl has no GitHub releases; FuseRegistry registers by
# git-tree-sha, so the registry is the source of truth for released versions).
if [[ -n "${TJLFEP_ENVIRONMENT:-}" ]]; then
    version="$TJLFEP_ENVIRONMENT"
else
    version="v$(curl -sf https://raw.githubusercontent.com/ProjectTorreyPines/FuseRegistry.jl/master/T/TJLFEP/Versions.toml \
        | sed -n 's/^\["\(.*\)"\]$/\1/p' | sort -V | tail -1)"
fi
if [[ -z "$version" || "$version" == "v" ]]; then
    echo "ERROR: could not determine TJLFEP version (set TJLFEP_ENVIRONMENT)." >&2
    exit 1
fi

suffix=""
[[ "$variant" == "imas" ]] && suffix="-imas"
image="tjlfep:${version}${suffix}"

if ! command -v podman-hpc >/dev/null 2>&1; then
    echo "ERROR: podman-hpc not found. Run this on Perlmutter." >&2
    exit 1
fi

echo "### Building $image (variant: $variant, context: $scriptdir)"
podman-hpc build -t "$image" \
    --build-arg TJLFEP_VERSION="$version" \
    --build-arg TJLFEP_VARIANT="$variant" \
    -f "$scriptdir/Containerfile" "$scriptdir"

echo "### Migrating $image to a squashed read-only image"
if [[ -n "${SQUASH_DIR:-}" ]]; then
    podman-hpc --squash-dir "$SQUASH_DIR" migrate "$image"
    # migrate writes with the caller's umask (600/700), which locks the shared
    # store to the publisher; open it up to the project group.
    echo "### Making $SQUASH_DIR group-readable"
    find "$SQUASH_DIR" \( ! -perm -g+r -o -type d ! -perm -g+x \) -exec chmod g+rX {} +
else
    podman-hpc migrate "$image"
fi

echo
echo "### Done. Image '$image' is built and migrated."
echo "Run the TJLFEP REPL with:"
echo "    podman-hpc run --rm -it $image"
echo "GPU (compute node):"
echo "    podman-hpc run --rm --gpu -it $image"
