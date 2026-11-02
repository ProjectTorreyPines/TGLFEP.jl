#!/bin/bash
# Acceptance checks for the TJLFEP container.
#
#   TGLFEP_JULIA_ENVIRONMENT=v2.0.14 ./deploy/container/acceptance.sh
#   TGLFEP_JULIA_ENVIRONMENT=v2.0.14-imas ./deploy/container/acceptance.sh   # imas variant
#   TGLFEP_JULIA_ENVIRONMENT=v2.0.14-gpu ./deploy/container/acceptance.sh    # sysimage tag
#
# CPU-only checks — runs on a login node (GPU acceptance is a separate sbatch,
# see test_slurm_gpu.sbatch). Runs against the migrated podman-hpc image, or
# any docker/podman host via ENGINE:
#
#   ENGINE=docker TGLFEP_JULIA_ENVIRONMENT=v2.0.14 ./acceptance.sh
#
# Optional:
#   SQUASH_DIR=...   shared podman-hpc squash store
#   SKIP_SLOW=1      skip the single-radius physics solve
#   FULL_TESTS=1     also run the package test suite (network + time)
#
# Each check prints PASS/FAIL independently; the summary exits nonzero if any
# failed. Checks assert the precondition they are actually testing, so a check
# cannot pass while silently exercising nothing.

set -uo pipefail

version="${TGLFEP_JULIA_ENVIRONMENT:-}"
[[ -n "$version" ]] || { echo "ERROR: set TGLFEP_JULIA_ENVIRONMENT (e.g. v2.0.14, v2.0.14-imas, v2.0.14-gpu)" >&2; exit 1; }

# variant/sysimage expectations follow from the tag suffix
expect_imas=0; expect_sysimage=0
[[ "$version" == *-imas* ]] && expect_imas=1
[[ "$version" == *-gpu ]] && expect_sysimage=1

engine="${ENGINE:-podman-hpc}"
command -v "$engine" >/dev/null || { echo "ERROR: $engine not found" >&2; exit 1; }

image="localhost/tglfep_julia:$version"
out="${SCRATCH:-/tmp}/tglfep_julia_acceptance_${version}"
mkdir -p "$out"

podman=("$engine")
[[ "$engine" == "podman-hpc" && -n "${SQUASH_DIR:-}" ]] && podman=(podman-hpc --squash-dir "$SQUASH_DIR")

pass=0; fail=0; failed_names=()
check() {
    local name="$1"; shift
    echo "----- $name"
    if "$@"; then echo "PASS $name"; pass=$((pass+1))
    else echo "FAIL $name"; fail=$((fail+1)); failed_names+=("$name"); fi
}

echo "=== TJLFEP $version container acceptance — $(date -Is) on $(hostname)"
echo "=== image: $image (engine: $engine${SQUASH_DIR:+, squash-dir: $SQUASH_DIR})"
echo

# --- identity ---------------------------------------------------------------
base_version="${version%%-*}"
check "version-is-$base_version" bash -c "
    got=\$(${podman[*]} run --rm $image tglfep_julia -e 'import TJLFEP; print(pkgversion(TJLFEP))')
    echo \"  TJLFEP \$got\"; [ \"v\$got\" = \"$base_version\" ]"

if [[ "$expect_sysimage" == 1 ]]; then
    check "sysimage-loaded" bash -c "
        p=\$(${podman[*]} run --rm $image tglfep_julia -e 'print(unsafe_string(Base.JLOptions().image_file))')
        echo \"  sysimage: \$p\"; [ \"\$p\" = /opt/tglfep_julia/sys_tglfep_julia_gpu.so ]"
else
    check "no-stray-sysimage" bash -c "
        p=\$(${podman[*]} run --rm $image tglfep_julia -e 'print(unsafe_string(Base.JLOptions().image_file))')
        echo \"  sysimage: \$p\"; [ \"\${p##*/}\" != sys_tglfep_julia_gpu.so ]"
fi

# The `tglfep_julia` wrapper prepends $HOME/.julia_tjlfep_container when the baked
# depot is not writable (read-only squash rootfs / SIF). Either way, the
# effective first depot entry must be writable or runtime precompilation of
# new packages fails.
check "writable-depot" bash -c "
    ${podman[*]} run --rm $image tglfep_julia -e '
        d = first(DEPOT_PATH)
        println(\"  first depot: \", d)
        mkpath(d)
        p = joinpath(d, \".acceptance_write_test\")
        touch(p); rm(p)
        println(\"  depot is writable\")'"

check "core-package-versions" bash -c "
    ${podman[*]} run --rm $image tglfep_julia -e '
        import Pkg
        found = Dict(d.name => d.version for d in values(Pkg.dependencies()))
        for p in [\"TJLFEP\",\"TJLF\",\"CUDA\",\"PackageCompiler\"]
            println(\"  \", p, \" => \", get(found, p, \"NOT INSTALLED\"))
        end'"

# --- CUDA -------------------------------------------------------------------
# The 12.9 runtime artifact must be baked (offline GPU nodes cannot download),
# and on a host without --gpu / --nv the stack must degrade to CPU, not throw.
check "cuda-runtime-baked-12.9" bash -c "
    ${podman[*]} run --rm --network none $image tglfep_julia -e '
        crj = Base.require(Base.PkgId(Base.UUID(\"76a88914-d11a-5bdc-97e0-2f5a05c973a2\"), \"CUDA_Runtime_jll\"))
        crj.is_available() || (println(\"  CUDA_Runtime_jll unavailable\"); exit(1))
        d = crj.artifact_dir
        println(\"  artifact: \", d)
        isdir(d) || exit(1)
        import CUDA
        v = CUDA.runtime_version()
        println(\"  runtime version: \", v)
        (v.major == 12 && v.minor == 9) || exit(1)'"

check "cpu-fallback-graceful" bash -c "
    ${podman[*]} run --rm --network none $image tglfep_julia -e '
        import CUDA, TJLFEP
        f = CUDA.functional()
        println(\"  CUDA.functional() = \", f, \" (no GPU injected — expect false)\")
        f && exit(1)'"

# --- image hygiene ----------------------------------------------------------
check "registries-world-readable" bash -c "
    bad=\$(${podman[*]} run --rm $image bash -c 'find /opt/tglfep_julia/.julia/registries ! -perm -o+r | wc -l')
    echo \"  non-world-readable entries: \$bad\"; [ \"\$bad\" = 0 ]"

# --- variant ----------------------------------------------------------------
if [[ "$expect_imas" == 1 ]]; then
    check "imas-extension-loads" bash -c "
        ${podman[*]} run --rm $image tglfep_julia -e '
            import TJLFEP, IMAS, GACODE, TurbulentTransport
            ext = Base.get_extension(TJLFEP, :TJLFEPIMASExt)
            println(\"  TJLFEPIMASExt: \", ext)
            ext === nothing && exit(1)'"

    # Scan in Julia, not nested bash: through podman -> bash -c -> tglfep_julia -e the
    # quoting is three deep, and a quoting slip reads as "the image still has
    # LFS stubs".
    check "no-LFS-pointer-stubs" bash -c "
        ${podman[*]} run --rm $image tglfep_julia -e '
            import TurbulentTransport
            root = joinpath(pkgdir(TurbulentTransport), \"models\")
            isdir(root) || (println(\"  no models dir at \", root); exit(1))
            files = String[]; stubs = String[]
            for (d, _, fs) in walkdir(root), f in fs
                p = joinpath(d, f); push!(files, p)
                startswith(String(open(io -> read(io, 40), p)), \"version https://git-lfs\") && push!(stubs, p)
            end
            isempty(files) && (println(\"  scanned nothing — models dir empty\"); exit(1))
            println(\"  scanned \", length(files), \" model file(s); \", length(stubs), \" still LFS stubs\")
            for s in first(stubs, 5); println(\"    STUB \", s); end
            exit(isempty(stubs) ? 0 : 1)'"
else
    check "lean-has-no-imas" bash -c "
        ${podman[*]} run --rm $image tglfep_julia -e '
            import TJLFEP
            ext = Base.get_extension(TJLFEP, :TJLFEPIMASExt)
            println(\"  TJLFEPIMASExt: \", ext, \" (lean — expect nothing)\")
            ext === nothing || exit(1)'"
fi

# --- physics smoke ----------------------------------------------------------
# Single-radius CPU critical-gradient solve on the packaged DIII-D example,
# offline (--network none proves the image is self-contained).
if [[ "${SKIP_SLOW:-0}" != "1" ]]; then
    check "singleradius-cpu-solve-offline" bash -c "
        ${podman[*]} run --rm --network none --volume $out:/out $image tglfep_julia -e '
            import TJLFEP
            ex = joinpath(pkgdir(TJLFEP), \"examples\", \"DIIID_202017C42_500ms_v3.1\")
            TJLFEP.run_gacode_scan_task(
                joinpath(ex, \"input.gacode\"),
                joinpath(ex, \"input_singleradius_nb6.TGLFEP\"),
                1; out_dir=\"/out\", use_gpu=false, printout=true)
            println(\"  single-radius solve completed offline\")' \
        && ls -la $out/"
fi

if [[ "${FULL_TESTS:-0}" == "1" ]]; then
    check "package-test-suite" bash -c "
        ${podman[*]} run --rm $image tglfep_julia -e 'import Pkg; Pkg.test(\"TJLFEP\")'"
fi

echo
echo "=== acceptance summary: $pass passed, $fail failed"
if [[ "$fail" -gt 0 ]]; then
    printf '    FAILED: %s\n' "${failed_names[@]}"
    exit 1
fi
echo "=== ALL CHECKS PASSED — $image is good on $(hostname)"
