#!/bin/bash
# Publish a locally-built TJLFEP container image to the GitHub Container
# Registry:
#
#     ghcr.io/projecttorreypines/tglfep_julia:<version><suffix>
#
# linux/amd64 only — no manifest lists or per-arch tags (simplification vs the
# FUSE publisher this is adapted from). Run AFTER build.sh (or the gpu-sysimage
# bake) on the same node: the image lives in the node-local podman store.
#
#     TGLFEP_JULIA_ENVIRONMENT=v2.0.14 ./deploy/container/publish_ghcr.sh
#     TGLFEP_JULIA_ENVIRONMENT=v2.0.14 TAG_SUFFIX=-imas ./deploy/container/publish_ghcr.sh
#     TGLFEP_JULIA_ENVIRONMENT=v2.0.14 TAG_SUFFIX=-gpu ./deploy/container/publish_ghcr.sh
#
# LATEST=1 additionally moves :latest (or :latest-imas) to this version. Never
# allowed for -gpu tags: `latest` must always point at a CI-reproducible
# pkgimages build, not a manually-baked sysimage image.
#
# Requires the `gh` CLI authenticated with the write:packages scope:
#
#     gh auth refresh --hostname github.com -s write:packages
#
# NOTE: the FIRST push creates the GHCR package with PRIVATE visibility. To let
# other sites pull without authentication (required for `apptainer pull` on
# Defiant), an org owner must make it public once at:
#   https://github.com/orgs/ProjectTorreyPines/packages/container/package/tglfep_julia
#   -> Package settings -> Change visibility.

set -euo pipefail

if [[ $# -gt 0 ]]; then
    echo "ERROR: this script takes no positional arguments. Set the version" >&2
    echo "       via TGLFEP_JULIA_ENVIRONMENT=$1 instead." >&2
    exit 1
fi

version="${TGLFEP_JULIA_ENVIRONMENT:-}"
[[ -n "$version" ]] || { echo "ERROR: set TGLFEP_JULIA_ENVIRONMENT (e.g. v2.0.14)" >&2; exit 1; }

suffix="${TAG_SUFFIX:-}"
case "$suffix" in
    ""|-imas|-gpu|-imas-gpu) ;;
    *) echo "ERROR: TAG_SUFFIX must be one of '', -imas, -gpu, -imas-gpu" >&2; exit 1 ;;
esac

if [[ "${LATEST:-0}" == "1" && "$suffix" == *-gpu ]]; then
    echo "ERROR: refusing to move :latest to a -gpu tag (latest stays on CI builds)" >&2
    exit 1
fi

dest="ghcr.io/projecttorreypines/tglfep_julia"
image="localhost/tglfep_julia:${version}${suffix}"

command -v podman-hpc >/dev/null || { echo "ERROR: podman-hpc not found — run on Perlmutter" >&2; exit 1; }

if ! command -v gh >/dev/null 2>&1 || ! gh auth status >/dev/null 2>&1; then
    echo "ERROR: gh CLI not authenticated. Run 'gh auth login' and add the" >&2
    echo "       write:packages scope (see header)." >&2
    exit 1
fi
user="$(gh api /user --jq .login)"

if ! podman-hpc image exists "$image"; then
    echo "ERROR: $image not found in the podman store on this node." >&2
    echo "       Run build.sh (or the gpu-sysimage bake) here first." >&2
    exit 1
fi

echo "### Logging in to ghcr.io as $user"
# GHCR_TOKEN_FILE overrides the gh token (e.g. when the gh oauth token lacks
# the write:packages scope). Otherwise: `gh auth token` first appeared in gh
# 2.17; older gh (e.g. the NERSC module) exposes the same token via
# `gh config get`.
if [[ -n "${GHCR_TOKEN_FILE:-}" ]]; then
    podman-hpc login ghcr.io -u "$user" --password-stdin < "$GHCR_TOKEN_FILE"
else
    { gh auth token 2>/dev/null || gh config get -h github.com oauth_token; } \
        | podman-hpc login ghcr.io -u "$user" --password-stdin
fi

# Always log out afterwards so no registry credential lingers on the node.
trap 'podman-hpc logout ghcr.io >/dev/null 2>&1 || true' EXIT

echo "### Pushing $image -> $dest:${version}${suffix} (several GB, takes a while)"
podman-hpc push "$image" "$dest:${version}${suffix}"

if [[ "${LATEST:-0}" == "1" ]]; then
    latest_tag="latest${suffix}"
    echo "### Moving $dest:$latest_tag -> ${version}${suffix}"
    podman-hpc push "$image" "$dest:$latest_tag"
fi

echo
echo "### Published $dest:${version}${suffix}${LATEST:+ (+ latest$suffix)}"
