# Sysimages: building, sharing, and the depot rule

Moved here from `build/README.md`. The run templates in `slurm/` pick a sysimage up via their `SYSIMAGE` CONFIG line (`""` = JIT); this page covers how the images are built and why they must run under their published `env_lean`/`env_full` project.

Precompiled sysimages remove JIT cost (~110 s/team) for production runs. Images are
node-count agnostic and written to `build/`; keep the `.so` on a non-purged path
(CFS, not `$PSCRATCH`) for reuse across jobs. There are two GPU images, both baking
the identical GPU solver paths (grid / `:ad` / `:robust_ad` / `:truth`):

- **`TGLFEP_JULIA_gpu_generic_sysimage.so`** (~3.0 GB) bakes the full FUSE-native stack
  (CUDA + TJLF + TJLFEP + FUSE/IMAS); use it for the FUSE `dd` path and `ActorTJLFEP`.
- **`TGLFEP_JULIA_gpu_sysimage.so`** (~1.1 GB) is the **file-only** image (CUDA + TJLF +
  TJLFEP standalone, IMAS/FUSE *not* baked) — what a TGLF-EP user running the
  file-based scan (`runTHD` / `run_gacode_scan_task`) gets before going FUSE-native.
  It is leaner and faster to load: on an A100 node a worker process starts in
  **~5.0 s vs ~7.2 s** for the generic image (~2 s / ~30% faster per worker, warm
  cache; measured via `sysimage/batch_measure_sysimage_load.sh`). In a real 5-node
  / 20-task scan the win is larger because the slowest of the 20 concurrently
  launching workers sets the wall, and the leaner image shortens that load-tail:
  the `:ad :only` scan20 timing logs show file-only beating generic by **~6–24 s of
  scan wallclock** at N_BASIS 8/16/32 (identical compute). The exception is
  N_BASIS=6, whose file-only runs are reproducibly dominated by cold 20-way 1.1 GB
  load I/O (~150-170 s) rather than compute — which is why the timing plot pins
  nb6 to the generic image and nb8/16/32 to file-only. Build time is comparable
  (~29 min, same GPU precompile workload); the build self-checks that
  `TJLFEPIMASExt` stays dormant and FUSE is not baked.

The CPU image is `TGLFEP_JULIA_cpu_sysimage.so`.

```bash
sbatch sysimage/batch_build_gpu_sysimage_generic.sh    # -> build_gpu_sysimage_generic.jl  (+ precompile_gpu_workload_generic.jl)
sbatch sysimage/batch_build_gpu_sysimage_fileonly.sh   # -> build_gpu_sysimage_fileonly.jl (+ precompile_gpu_workload_fileonly.jl)
sbatch sysimage/batch_build_cpu_sysimage.sh            # -> build_cpu_sysimage.jl          (+ precompile_cpu_workload.jl)
```

Batch scripts auto-detect the image via `TGLFEP_JULIA_GPU_SYSIMAGE` / `TGLFEP_JULIA_SYSIMAGE`
(falling back to JIT if missing). `slurm/common/tglfep_julia_env.inc.sh` is the shared
helper that exports `JULIA_SYSIMAGE_ARGS`.

### Sharing a sysimage with other users (artifacts / `JULIA_DEPOT_PATH`)

A sysimage does **not** embed its JLL artifacts — at load time each baked JLL
(`OpenSSL_jll`, `CUDA_*_jll`, …) runs its `__init__` and resolves a **fixed**
artifact hash by searching every depot on `JULIA_DEPOT_PATH` for
`artifacts/<hash>`. If that hash isn't in any depot the process aborts before
`main`, e.g.:

```
InitError(mod=:OpenSSL_jll, error=ErrorException("Artifact "OpenSSL" was not found
  by looking in the path ".../.julia/artifacts/5aa05123…"))
```

The hash is pinned to the exact JLL **version** the image was built against, so a
colleague's own `Pkg.instantiate()` does **not** reliably fix it: if their
manifest resolves a different `OpenSSL_jll` version they fetch a *different*
artifact and the baked one is still missing. (You cannot embed artifacts into the
`.so`; `PackageCompiler.create_sysimage` resolves them from the depot at runtime.)

Artifacts are only half the story, though: the image (and Julia itself, for JIT
fallback and extension loading) also re-reads **package sources** — every JLL's
`Artifacts.toml` lives in the package's source dir, whose absolute path is baked
at build time. An image baked from a private scratch depot therefore fails for
everyone else no matter how they stack depots. The fix is baking *from
m3739-readable paths*: a shared depot on CFS holding all package sources
(registry-resolved — see below), plus the published build environments.

- Images: `/global/cfs/cdirs/m3739/TJLFEP/TGLFEP_JULIA_gpu_generic_sysimage.so`
  (+ file-only GPU / CPU), with `.sha` sidecars recording the resolved package
  versions each image was baked from
- Published build environments (`Project.toml` + `Manifest.toml` +
  `LocalPreferences.toml`): `/global/cfs/cdirs/m3739/TJLFEP/env_full` (generic
  image) and `env_lean` (file-only/CPU images) — these are the runtime
  `--project`s, resolving exactly the versions baked into the images
- **Full shared depot** (packages + artifacts + registries + compiled):
  `/global/cfs/cdirs/m3739/TJLFEP/depot`

Since 2026-08-26 the build environments are **registry-resolved**
(`sysimage/setup_registry_env.jl`): every package comes from FuseRegistry at its
released version — the same philosophy as the TJLFEP container
(`deploy/container/`) — so no dev repos are staged and any user can rebuild any
image from their own TJLFEP checkout. (Previously the images were baked from the
maintainer's `Pkg.develop` tree, staged to `src/` by the now-retired
`publish_build_tree.sh`.)

Any `m3739` member runs with:

```bash
module load cudatoolkit/12.9 julia/1.11.7
# own (writable) depot FIRST, then the shared read-only depot:
export JULIA_DEPOT_PATH="$SCRATCH/.julia:/global/cfs/cdirs/m3739/TJLFEP/depot"
export JULIA_CUDA_USE_COMPAT=false

# generic image / FUSE dd path (GACODE/IMAS/TurbulentTransport are TJLFEP weak
# deps -- only the published env_full project resolves them, so use it as the
# project in both sysimage and JIT mode):
julia --startup-file=no \
  --project=/global/cfs/cdirs/m3739/TJLFEP/env_full \
  --sysimage=/global/cfs/cdirs/m3739/TJLFEP/TGLFEP_JULIA_gpu_generic_sysimage.so \
  -e 'using TJLFEP, TurbulentTransport; println("OK")'
```

or, from a TJLFEP checkout, `sbatch slurm/tglfep_julia_gpu_backfill.sbatch` with the nb6 input (see `slurm/README.md`).

> Keep your own writable depot **first** in `JULIA_DEPOT_PATH` (precompile caches,
> logs, and scratchspaces are written to the first entry) and treat the CFS
> project/depot as read-only: do not run `Pkg.update`/`resolve`/`instantiate`
> against the published `env_*` projects — it would try to rewrite the shared
> manifest.

**Refreshing the shared images.** From a TJLFEP checkout, submit the three build
scripts (`cd build && sbatch sysimage/batch_build_*.sh`) with `TGLFEP_JULIA_DEPOT`
pointing at the CFS depot; each job creates/refreshes its registry-resolved build
env (`sysimage/setup_registry_env.jl`; pin a version with
`TGLFEP_JULIA_BUILD_VERSION`, default is the newest registered) and publishes the
`.so` + `.sha` + `env_*` project automatically. Never `rsync -a`/`cp -a` into
the share — preserving the maintainer's primary group is what once made
`depot/artifacts` unreadable to the group; the scripts run under `umask 007`
with an explicit `chgrp` pass instead.
