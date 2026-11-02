#!/bin/bash
# Publish ghcr.io/projecttorreypines/tglfep_julia:<v>-gpu FROM DEFIANT by appending
# the sysimage layer to the already-published :<v> image with crane. This is
# exactly what Containerfile.gpu produces, but server-side: the base layers
# are cross-mounted in the registry, so nothing multi-GB moves over the wire
# and no local OCI store (podman-hpc) is needed.
#
#   TGLFEP_JULIA_ENVIRONMENT=v2.0.14 SYSIMAGE_OUT=$WORK/sysimage_out \
#       ./defiant_publish_gpu.sh
#
# Requires: crane (https://github.com/google/go-containerregistry) and the
# gh CLI authenticated with the write:packages scope:
#     gh auth refresh --hostname github.com -s write:packages
#
# Run only after defiant_smoke_gpu.sbatch passed on the same sysimage.

set -euo pipefail

version="${TGLFEP_JULIA_ENVIRONMENT:?set TGLFEP_JULIA_ENVIRONMENT (e.g. v2.0.14)}"
out="${SYSIMAGE_OUT:?set SYSIMAGE_OUT (dir holding sys_tglfep_julia_gpu.so)}"
ghcr="ghcr.io/projecttorreypines/tglfep_julia"

[[ -s "$out/sys_tglfep_julia_gpu.so" ]] || { echo "ERROR: no sysimage at $out/sys_tglfep_julia_gpu.so" >&2; exit 1; }
command -v crane >/dev/null || { echo "ERROR: crane not on PATH" >&2; exit 1; }

if ! command -v gh >/dev/null 2>&1 || ! gh auth status >/dev/null 2>&1; then
    echo "ERROR: gh CLI not authenticated (need write:packages — see header)" >&2
    exit 1
fi
user="$(gh api /user --jq .login)"

echo "### Logging in to ghcr.io as $user"
gh auth token | crane auth login ghcr.io -u "$user" --password-stdin
trap 'crane auth logout ghcr.io >/dev/null 2>&1 || true' EXIT

# Layer tar with the sysimage at the launcher's expected path, root-owned —
# identical content to the Containerfile.gpu COPY layer.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"; crane auth logout ghcr.io >/dev/null 2>&1 || true' EXIT
mkdir -p "$tmp/opt/tglfep_julia"
cp "$out/sys_tglfep_julia_gpu.so" "$tmp/opt/tglfep_julia/sys_tglfep_julia_gpu.so"
chmod 0644 "$tmp/opt/tglfep_julia/sys_tglfep_julia_gpu.so"
tar --owner=0 --group=0 --numeric-owner -C "$tmp" -cf "$tmp/layer.tar" opt

base_digest="$(crane digest "$ghcr:$version")"
echo "### Appending sysimage layer to $ghcr:$version ($base_digest) -> :$version-gpu"
crane append \
    --base "$ghcr@$base_digest" \
    --new_layer "$tmp/layer.tar" \
    --new_tag "$ghcr:$version-gpu"

echo "### Published $ghcr:$version-gpu ($(crane digest "$ghcr:$version-gpu"))"
