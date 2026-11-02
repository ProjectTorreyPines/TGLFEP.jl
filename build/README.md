# build/ — sysimage bakes

This directory only builds the precompiled sysimages that remove JIT cost from production
runs. **To run TJLFEP, go to [`slurm/`](../slurm/README.md)** — the templates there pick an
image up through their `SYSIMAGE` line, and m3739 members can use the prebuilt images on CFS
without building anything.

Three images, all baked from registry-resolved environments (`sysimage/setup_registry_env.jl`,
every package at its FuseRegistry release, so any user can rebuild from any checkout):

| Image | Bakes | Pairs with | Build |
|-------|-------|------------|-------|
| `TGLFEP_JULIA_gpu_sysimage.so` (file-only, ~1.3 GB) | CUDA + TJLF + TJLFEP | `env_lean` | `sbatch sysimage/batch_build_gpu_sysimage_fileonly.sh` |
| `TGLFEP_JULIA_gpu_generic_sysimage.so` (~2.4 GB) | + FUSE/IMAS/GACODE/TurbulentTransport (for the `dd` path / `ActorTJLFEP`) | `env_full` | `sbatch sysimage/batch_build_gpu_sysimage_generic.sh` |
| `TGLFEP_JULIA_cpu_sysimage.so` (~1.3 GB) | TJLF + TJLFEP, CPU only | `env_lean` | `sbatch sysimage/batch_build_cpu_sysimage.sh` |

```bash
cd build                                                  # submit from build/
sbatch sysimage/batch_build_gpu_sysimage_fileonly.sh      # -> build/TGLFEP_JULIA_gpu_sysimage.so  (~30 min, 1 GPU node)
```

Each job creates/refreshes its build env under `sysimage/env_lean` or `sysimage/env_full`
(pin a release with `TGLFEP_JULIA_BUILD_VERSION`), bakes the image into `build/`, self-checks it,
and — when `TGLFEP_JULIA_DEPOT` points at the shared CFS depot — publishes the `.so`, a `.sha`
sidecar with the resolved versions, and the env project to `/global/cfs/cdirs/m3739/TJLFEP/`.

Then point a template at it: `SYSIMAGE=build/TGLFEP_JULIA_gpu_sysimage.so` (the include pairs it
with `build/sysimage/env_lean` automatically). Keep images you rely on on CFS, not
`$PSCRATCH` (purged). A sysimage takes precedence over edited source: after changing `src/`,
rebuild or run with `SYSIMAGE=""`.

Why an image must run under its own `env_*` project, how JLL artifacts are resolved through
`JULIA_DEPOT_PATH`, and the rules for sharing images with other users are in
[`docs/SYSIMAGES.md`](../docs/SYSIMAGES.md). `sysimage/batch_measure_sysimage_load.sh`
benchmarks worker start-up time per image.
