# slurm/ — how to run TJLFEP on Perlmutter

Four Slurm templates, one shared include. Pick a template by **device** (GPU or CPU) and
**layout** (high throughput on several nodes, or low footprint on one node), edit the
CONFIG block at the top of the file, submit from the repo root.

```bash
cd TJLFEP                                   # the job finds this checkout via SLURM_SUBMIT_DIR
$EDITOR slurm/tjlfep_gpu_backfill.sbatch    # at least: "#SBATCH -A <your allocation>" and TGLFEP_FILE
sbatch slurm/tjlfep_gpu_backfill.sbatch
```

| Template | Nodes | Layout | Use for |
|---|---|---|---|
| `tjlfep_gpu_5N.sbatch` | 5 GPU | 20 tasks = 20 radii in one wave, 1 A100 + MPS team per radius | **high throughput** for uniform-cost solvers: `grid`, `ad :only` |
| `tjlfep_gpu_backfill.sbatch` | 1 GPU | 4 tasks (one per A100) drain a shared radius queue | **low footprint**, and the right choice for the **extended-width solvers**: `ad :locate`, `ad :wide`, `robust_ad`, `truth` |
| `tjlfep_cpu_10N.sbatch` | 10 CPU | 20 workers, one per radius | high throughput without a GPU allocation |
| `tjlfep_cpu_1N.sbatch` | 1 CPU | 2 workers x 64 threads, pmap over radii | low footprint without a GPU allocation (slow: use `N_BASIS <= 16`) |
| `tjlfep_merge.sbatch` | 1 CPU | re-merge an existing output dir | only if a GPU job's final merge step failed |

`SCAN_N=20` is the default in every example input. The 5N/10N templates run **one radius per
task**, so `SCAN_N` in the `input.TGLFEP` must equal their `-n`/`--ntasks` (20); the job checks
and refuses otherwise. The backfill/1N templates accept any `SCAN_N`.

## Which solver, which template

`SOLVER` (and `AD_EXTEND_MODE` for `ad`) is the one physics-relevant run knob. Node-hours are
for the reactor-relevant `UCP_complete` case at `N_BASIS=32` (Fortran `-n 1280` on 10 CPU
nodes = 2.46 node-hours, measured 2026-10 on v2.0.18 with the `WIDTH_FLOOR = 0.5` default; full tables in [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md)).

| `SOLVER` | Width box | Template | `INNER` | node-h | What you get |
|---|---|---|---|---:|---|
| `grid` | `w >= 1` | gpu_5N | `mps_team` | 0.49 | The Fortran-equivalent `(kyhat x width x factor)` sweep. **Reproduces Fortran `SFmin`.** |
| `ad` + `AD_EXTEND_MODE=only` | `w >= 1` | gpu_5N | `threads` | 0.28 | De-quantized approximation of `grid`. Fast iteration only; misses the narrower `0.5 <= w < 1` modes. |
| `ad` + `AD_EXTEND_MODE=locate` **(default)** | extended | gpu_backfill | `mps_team` | 1.16 | Adds the narrower-width (`0.5 <= w < 1`, never below the physics floor `WIDTH_FLOOR = 0.5`) EP-driven Alfven modes Fortran misses; the faithful extended-width value. Production / `ActorTJLFEP` default. |
| `ad` + `AD_EXTEND_MODE=wide` | extended | gpu_backfill | `mps_team` | 0.39 | One log-seeded narrow-width pass: ~3x cheaper than `locate`, conservative (within 1-2x of it, never below). Bulk NN-database generation. |
| `robust_ad`, `truth` | extended | gpu_backfill | `mps_team` | higher | Reference tiers above `locate` (`truth` adds an `N_BASIS` ladder; ~34 min per DIII-D profile at nb32). Validation only. |
| `grid` on CPU | `w >= 1` | cpu_10N / cpu_1N | pmap | 13.6 | Same result as GPU `grid`, ~28x the node-hours. |

With the ballooning-width physics floor `WIDTH_FLOOR = 0.5` the extended solvers search a bounded box: they
land 1-4x below `grid` (not the up-to-16x of the earlier unbounded search) and cost 20-50 % less than before.

Why the split: `grid` and `ad :only` cost about the same at every radius, so spreading 20
radii over 5 nodes finishes in one wave at no extra node-hours. The extended-width solvers
spend far longer on the outer radii (that is where the `0.5 <= w < 1` locate triggers), so a fixed
20-task split would leave four nodes idle waiting on the edge; one node whose four GPUs keep
pulling the next radius from a queue gives the lowest node-hours, and a 1-node job also
backfills quickly in the Slurm queue.

`INNER` is set for you: the 5N template pairs `ad :only` with in-process threads and everything
else with an MPS worker team (`MPS_TEAM=8` processes per GPU); the backfill template uses
`mps_team` for every solver. That is not just inherited from the published numbers: measured
on the backfill template with UCP `N_BASIS=32`, `ad :locate` (2026-10-03, before the width floor), `mps_team` 8x2 took
1.49 node-hours, `threads` x2 2.72 and `threads` x16 2.60, with `SFmin` agreeing to within 0.3 %
(the two layouts partition the narrow-width search differently, so the located optimum can
shift slightly).
The "`ad` prefers threads" rule holds for the bare `:only` descent, not for the extended-width
locate, whose many independent narrow-width evaluations do amortize a team.

## Edit these lines

Open the template. Everything a user changes is inside the `CONFIG` block; every entry is
`VAR="${VAR:-default}"`, so you can also override at submit time
(`SOLVER=ad AD_EXTEND_MODE=wide sbatch slurm/tjlfep_gpu_backfill.sbatch`).

1. **`#SBATCH -A`** — your NERSC allocation (`<project>_g` for GPU, `<project>` for CPU).
   `-q regular` is the default; `debug` (30 min, 1-4 nodes) is fine for an nb6 smoke test.
2. **`TGLFEP_FILE`** — which `input.TGLFEP`. The physics lives in this file, not in the
   template: `SCAN_N`, `N_BASIS` (6/8/16/32, UCP also 40/48), `IRS`, `WIDTH_MIN`/`WIDTH_MAX`,
   `FACTOR_IN`, `PROCESS_IN`, `KY_MODEL`, the `REJECT_*` flags. The examples ship
   `input_scan20_nb{6,8,16,32}.TGLFEP`; copy one and edit `N_BASIS`.
3. **`SOLVER` / `AD_EXTEND_MODE`** — see the table above. Secondary knobs sit next to them
   (`AD_WIDE_KDESC`, `AD_FAITHFUL_CONFIRM`, `REFINE_ROUNDS`, `TJLFEP_K_MAX`, `TJLFEP_NMODES`,
   `TJLFEP_NXGRID`, `TJLFEP_PRINTOUT`), each with its allowed values in the comment.
4. **`SYSIMAGE`** — a prebuilt image (default: the shared m3739 image on CFS), or `""` for
   JIT from your checkout. See below.
5. **`MPS_TEAM`** (GPU) — 8 fills an A100 for `grid`; at `N_BASIS=48` use `MPS_TEAM=4` and
   uncomment `#SBATCH -C gpu&hbm80g`, or the 32-worker team exhausts the node's 256 GB.
6. **`-t`** — the headers carry the measured nb32 wall; raise it for `truth`, nb48, or
   `UCP_complete :locate` (~4 h on one node).

## Running your own case

A case is a directory with `input.gacode` (equilibrium + profiles, GACODE EXPRO format) and an
`input.TGLFEP` (scan control). Set `CASE_DIR`, or `GACODE_FILE`/`TGLFEP_FILE` individually.
Nothing is ever written into `CASE_DIR`.

## Outputs

Each run writes to `OUT_DIR` (default `runs/<template>_<jobid>_tasks/`, gitignored):

| File | Meaning |
|---|---|
| `task_<i>.jls` | per-radius result (GPU templates), `i = 1..SCAN_N` |
| `sfmin_scan.txt` | `i  IR  SFmin` per radius: the critical scale factor on the EP gradient |
| `alpha_dndr_crit.input`, `alpha_dpdr_crit.input` | critical EP density / pressure gradient profiles (the files TGLF-EP hands to transport) |
| `out.TGLFEP` | Fortran-style summary (CPU templates, and GPU with `TJLFEP_PRINTOUT=1`) |
| `out.scalefactor_r###` | per-radius scan trace (`TJLFEP_PRINTOUT=1`) |

Slurm logs `tjlfep_<template>_<jobid>.{out,err}` land where you submitted (the repo root;
gitignored). The `.out` starts with a banner echoing the resolved project, sysimage, inputs
and solver settings.

**Re-merging.** The GPU templates merge the `task_*.jls` at the end of the job. If that step
did not run, `OUT_DIR=runs/<run>_tasks sbatch slurm/tjlfep_merge.sbatch`, or on a login node:

```bash
module load julia/1.11.7
OUT_DIR=runs/<run>_tasks GACODE_FILE=<case>/input.gacode TGLFEP_FILE=<case>/input_scan20_nb32.TGLFEP \
  julia --project=. slurm/common/merge_gacode_scan20_array.jl
```

A `missing array task output: task_<i>.jls` error means radius `i` failed; look for its
`ERROR scan_index=<i>` line in the `.err`, fix, and re-run that radius
(`SCAN_INDEX=<i> OUT_DIR=<same dir> ... julia slurm/common/run_gacode_scan20_mps_task.jl`
inside a GPU allocation) before merging again.

## With or without a sysimage

A sysimage removes the JIT cost (~110 s per MPS team, paid once per radius on the 5N layout,
once per GPU on backfill). The `SYSIMAGE` line decides:

- **Prebuilt (default).** m3739 members get the shared images in
  `/global/cfs/cdirs/m3739/TJLFEP/`: `TJLFEP_gpu_sysimage.so` (file-only, what the GPU
  templates use), `TJLFEP_cpu_sysimage.so`, `TJLFEP_gpu_generic_sysimage.so` (FUSE/IMAS
  baked, for the `dd` path). The include pairs an image with the published project it was
  baked from (`env_lean`, or `env_full` for the generic image) and appends the shared CFS depot
  to `JULIA_DEPOT_PATH` so the baked JLL artifacts resolve. Do not run `Pkg` commands against
  those env projects; they are read-only.
- **`SYSIMAGE=""` (JIT).** Runs `--project=<your checkout>` with whatever your `Manifest.toml`
  resolves. Use this after editing `src/` (a sysimage silently runs the baked code), or if you
  are not in m3739. Instantiate once first:
  `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`.
- **Your own bake.** `cd build && sbatch sysimage/batch_build_gpu_sysimage_fileonly.sh`, then
  `SYSIMAGE=build/TJLFEP_gpu_sysimage.so` (see [`build/README.md`](../build/README.md)).

On the 5-node template `STAGE=1` (default) first `sbcast`s the image to node-local `/tmp` so
80 workers do not all read 1.3 GB off Lustre at once.

Depot rule used by every template: `JULIA_DEPOT_PATH` keeps your own depot first (writable;
`$HOME/.julia` if you had nothing set), then `$PSCRATCH/.julia`, then the shared CFS depot.
Your depot is never replaced. Details and the artifact mechanism: [`docs/SYSIMAGES.md`](../docs/SYSIMAGES.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `TJLFEP_ROOT='...' is not a TJLFEP checkout` | Submitted from another directory. `cd` to the repo root before `sbatch`, or `export TJLFEP_ROOT=/path/to/TJLFEP`. |
| `SCAN_N ... must equal the number of Slurm tasks` | 5N/10N run one radius per task. Use an `input_scan20_*` file, or the backfill/1N template. |
| `PROJECT=... has no Manifest.toml` | JIT from a fresh checkout: run the `Pkg.instantiate()` line above on a login node first. |
| `Artifact "OpenSSL" was not found` | A sysimage without its depot: keep the default `TJLFEP_DEPOT`/`CFS_DIR`, do not clobber `JULIA_DEPOT_PATH` in your shell init. |
| `GPU run needs CUDA >= 12.6` | Wrong module; the include loads `cudatoolkit/12.9`, check nothing in your shell init overrides it. |
| Workers hang in `using CUDA` | `JULIA_CUDA_USE_COMPAT` must be `false` for MPS clients; the include sets it, do not unset it. |
| Node OOM at `N_BASIS=48` | `MPS_TEAM=4` and `#SBATCH -C gpu&hbm80g`. |
| Results ignore a `src/` edit | A sysimage is in use. `SYSIMAGE=""` or rebuild. |

## What is in `common/`

| File | Role |
|---|---|
| `tjlfep_env.inc.sh` | sourced by every template: modules, depot order, root resolution, sysimage/project pairing, input checks, env export, merge |
| `run_gacode_scan20_mps_task.jl` | GPU driver: one radius per task (or the backfill queue), MPS worker team per GPU |
| `mps-scan-wrapper.sh` | per-node MPS daemon start (daemon-first), GPU pinning, `SCAN_INDEX` |
| `merge_gacode_scan20_array.jl` | merges `task_*.jls` into `sfmin_scan.txt` + `alpha_*_crit.input` |
| `run_scan_cpu_distributed.jl` | CPU driver: SlurmClusterManager workers + `pmap` over radii |

The Fortran analogue is `srun -n 1280 $TGLFEP_DIR/TGLFEP_driver` on the same
`input.TGLFEP` + `input.gacode`; the Julia drivers read the same two files and produce the
same `alpha_*_crit.input`. Spectrum runs (`PROCESS_IN=3`) do not use these templates; see
[`docs/SPECTRUM_DIAGNOSTIC.md`](../docs/SPECTRUM_DIAGNOSTIC.md).
