# TJLFEP container

Self-contained TJLFEP images on `ghcr.io/projecttorreypines/tjlfep` for
**standalone** file-based runs (`input.TGLFEP` / `input.gacode` workflows,
`run_gacode_scan_task` / `runTHD_from_gacode`), with CPU **and** GPU capability
in one image. For the full-FUSE use case, use the
[FUSE container](https://github.com/ProjectTorreyPines/FUSE.jl/tree/master/deploy/perlmutter-container)
instead; for multi-node `SlurmClusterManager` scans, use the bare-metal CFS
install (see Limitations).

## Tags

| Tag | Contents | Built by |
|---|---|---|
| `:<v>`, `:latest` | lean: TJLFEP + TJLF + CUDA, pkgimages only | GitHub Actions (`.github/workflows/container.yml`) |
| `:<v>-imas`, `:latest-imas` | + IMAS/GACODE/TurbulentTransport (`TJLFEPIMASExt`) | GitHub Actions |
| `:<v>-gpu` | lean + GPU-traced multi-target sysimage | Manual, GPU node (`gpu-sysimage/`: Perlmutter A100 or Defiant H200) |

`latest` never points at a `-gpu` tag. All images are linux/amd64, CPU targets
`generic;znver3;znver4;sapphirerapids` (Perlmutter Milan, Defiant-era hosts,
generic fallback). Every image carries the **CUDA 12.9 runtime artifact**
(pinned via `CUDA_Runtime_jll` Preferences, CUDA.jl 5.8.5) and works fully
offline; without an injected GPU driver, `CUDA.functional()` is `false` and
everything falls back to the CPU path — one image serves both.

There is no file-only flag: the lean image simply lacks the IMAS trigger
packages, so the `TJLFEPIMASExt` extension never loads; in the `-imas` image it
loads automatically.

## NERSC Perlmutter (podman-hpc)

One-command setup:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/ProjectTorreyPines/TGLFEP.jl/master/deploy/container/install_tjlfep_container_nersc.sh)
```

or by hand:

```bash
podman-hpc pull ghcr.io/projecttorreypines/tjlfep:v2.0.14
podman-hpc tag ghcr.io/projecttorreypines/tjlfep:v2.0.14 localhost/tjlfep:v2.0.14
podman-hpc migrate tjlfep:v2.0.14

# CPU (any node)
podman-hpc run --rm -it localhost/tjlfep:v2.0.14

# GPU (inside a GPU allocation; --gpu injects the host driver)
podman-hpc run --rm --gpu -it localhost/tjlfep:v2.0.14
```

The in-image `tjlfep` launcher starts Julia against the baked project
(`/opt/tjlfep`), auto-loads the GPU sysimage when present (`-gpu` tags), and
prepends a writable `$HOME/.julia_tjlfep_container` depot when the image
depot is read-only.

Batch templates: `test_slurm_cpu.sbatch`, `test_slurm_gpu.sbatch`
(single-process), and `test_slurm_container_gpu_mps.sbatch` (CUDA MPS team run — a
working multi-process example of the daemon-first launch order and the
podman `-v`/`-e` passthrough, driving the shared `mps_team_driver.jl`).
Acceptance suite: `acceptance.sh` (CPU, login-node safe; see header for
options).

## ORNL Defiant (Apptainer)

Validated 2026-08 on Defiant H200 nodes (driver 580.126.20 / CUDA 13.0,
Xeon 8562Y+ — covered natively by the `sapphirerapids` sysimage target):

```bash
# set BOTH to scratch first — the OCI->SIF conversion of an ~8.6 GB image
# blows through a home quota otherwise
export APPTAINER_CACHEDIR=/path/to/scratch/cache APPTAINER_TMPDIR=/path/to/scratch/tmp
apptainer pull tjlfep.sif docker://ghcr.io/projecttorreypines/tjlfep:v2.0.14-gpu
apptainer exec --nv tjlfep.sif tjlfep -e 'import CUDA; @show CUDA.functional()'
```

- `--nv` injects the host driver; the baked 12.9 runtime runs cleanly on
  Defiant's CUDA 13.0 driver via forward compat (verified: real GPU solves,
  identical `sfmin` to the CPU path). If a future driver rejects 12.9, re-pin
  in your writable home depot:
  `tjlfep -e 'import CUDA; CUDA.set_runtime_version!(v"12.x")'` — the launcher's
  home-depot fallback makes the new preference stick.
- SIF images are read-only: the launcher automatically prepends
  `$HOME/.julia_tjlfep_container` as the first (writable) depot.
- Batch templates: `gpu-sysimage/defiant_smoke_gpu.sbatch` (single-process,
  `--cleanenv` — forward `CUDA_VISIBLE_DEVICES` and `JULIA_NUM_THREADS`
  explicitly) and `test_slurm_container_gpu_mps_defiant.sbatch` (CUDA MPS team run;
  deliberately NOT `--cleanenv` — the default env/`/tmp` forwarding is what
  makes the host-side MPS pipes visible in-container).

## Running TJLFEP

In a batch script, wrap the Julia invocation in the container — no
`module load`, no `JULIA_DEPOT_PATH`, no `--project`/`--sysimage` flags
(the `tjlfep` launcher supplies the baked project and, on `-gpu` tags, the
sysimage, and passes any remaining arguments to `julia`). With your case
directory holding `input.gacode` + `input.TGLFEP`:

```bash
# Defiant / Apptainer
srun apptainer exec --nv --bind "$CASE_DIR" tjlfep_v2.0.14-gpu.sif \
    tjlfep -t "$SLURM_CPUS_PER_TASK" -e '
        import CUDA, TJLFEP
        TJLFEP.run_gacode_scan_task(
            ENV["CASE_DIR"] * "/input.gacode", ENV["CASE_DIR"] * "/input.TGLFEP",
            1;                          # scan index (radius)
            out_dir=ENV["CASE_DIR"] * "/out",
            use_gpu=CUDA.functional())'

# Perlmutter / podman-hpc: same payload, different wrapper — podman shares
# nothing by default, so mount and pass env explicitly
srun podman-hpc run --rm --gpu -v "$CASE_DIR":"$CASE_DIR" -e CASE_DIR \
    localhost/tjlfep:v2.0.14-gpu tjlfep -t "$SLURM_CPUS_PER_TASK" -e '...'
```

Results land in `out_dir` (`task_<i>.jls` per scan index). Drop `--nv` /
`--gpu` and the identical solve runs on CPU threads. Apptainer forwards host
environment and binds `$HOME`/`/tmp` by default — which is also what makes
host-side CUDA MPS pipes visible in-container for SPMD team runs (start the
daemon on the host BEFORE the container runs; working examples:
`test_slurm_container_gpu_mps.sbatch` on Perlmutter,
`test_slurm_container_gpu_mps_defiant.sbatch` on Defiant, both driving the
shared `mps_team_driver.jl`). A packaged DIII-D example case ships in the image at
`joinpath(pkgdir(TJLFEP), "examples", "DIIID_202017C42_500ms_v3.1")` for
testing without any input files.

## Building

```bash
./deploy/container/build.sh                     # lean, local podman-hpc build
TJLFEP_VARIANT=imas ./deploy/container/build.sh # imas variant
TJLFEP_ENVIRONMENT=v2.0.14 ./deploy/container/acceptance.sh
TJLFEP_ENVIRONMENT=v2.0.14 ./deploy/container/publish_ghcr.sh
```

CI (`workflow_dispatch` or release) builds and pushes the lean + imas tags.
The version resolves from FuseRegistry's `Versions.toml` (this repo publishes
no GitHub releases). **First push only:** the GHCR package is created private —
an org owner must flip it public for unauthenticated pulls.

### GPU sysimage bake (`:<v>-gpu`)

CI cannot bake the sysimage (the trace executes real GPU solves). On a
Perlmutter login node:

```bash
TJLFEP_ENVIRONMENT=v2.0.14 ./deploy/container/gpu-sysimage/bake_and_publish.sh
```

pulls + migrates the CI image, runs the bake on an A100 (sbatch, m5377_g
premium, ~1–4 h), layers `sys_tjlfep_gpu.so` in with `Containerfile.gpu`
(COPY layer — reproducible, no `podman-hpc commit`), runs CPU acceptance, and
submits GPU acceptance. Publish after the GPU job passes:

```bash
TJLFEP_ENVIRONMENT=v2.0.14 TAG_SUFFIX=-gpu ./deploy/container/publish_ghcr.sh
```

**Defiant variant (Apptainer, no podman-hpc):** the bake is split so the GPU
job stays short enough to backfill Defiant's two busy H200 nodes — the trace
(`defiant_trace_gpu.sbatch`, ~20 min on batch-gpu, `--trace-compile` on the
real workload) and the sysimage compile
(`defiant_compile_sysimage.sbatch`, batch-cpu, no GPU needed: replaying
precompile statements is type-level only; `JULIA_CPU_TARGET` comes from the
image ENV so the result is host-independent). Chain them with
`--dependency=afterok`, GPU-accept with `defiant_smoke_gpu.sbatch` (binds the
fresh `.so` at the launcher's path), and publish with
`defiant_publish_gpu.sh` — `crane append` pushes just the sysimage layer onto
the already-published base, server-side. `tjlfep_gpu.def` builds a local
`-gpu` SIF; if `apptainer build` from a local SIF fails on Lustre
(unsquashfs uid/gid restore), build the layered tarball instead:
`crane append -b ghcr.io/...:<v> -f layer.tar -o out.tar` then
`apptainer build out.sif docker-archive://out.tar`.

## Limitations (v1)

- **Single-node only**: threads + local `addprocs` + CUDA MPS. Multi-node
  `SlurmClusterManager` runs would need every srun-spawned worker to start
  in-container — use the bare-metal CFS install for those.
- linux/amd64 only (no arm64/multi-arch manifests).
- The GPU sysimage is baked manually per release; plain tags run the same GPU
  code via JIT (slower first solve, identical throughput after).
