# Reproducing Fortran–Julia TGLF-EP match (DIII-D 202017C42)

Validated on Perlmutter CPU (premium QoS). See `FORTRAN_JULIA_COMPARISON.md` for physics fixes.

## Repositories

```bash
# TJLFEP depends on registered TJLF >= 1.2.4 (FuseRegistry) — no TJLF checkout needed.
git clone git@github.com:ProjectTorreyPines/TGLFEP.jl.git TJLFEP
cd TJLFEP
```

TJLF >= 1.2.4 provides the `use_gpu` kwarg and the `TJLFCUDAExt` /
`TJLFForwardDiffExt` extensions; `Pkg.instantiate()` pulls it from the registry.

## Julia environment

Julia **1.11+** (1.11.7 or the default NERSC `juliaup` 1.12+). On HPC, load the
module (e.g. `module load julia` on Perlmutter); optionally point
`JULIA_DEPOT_PATH` at scratch if your home directory is small or slow.

```bash
cd TJLFEP
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

File-based runs (no IMAS/FUSE):

```bash
export GACODE_DUMP=$PWD/examples/DIIID_202017C42_500ms_v3.1/input.gacode
```

## Fortran TGLF-EP

The verification scripts default to the shared, prebuilt Fortran `TGLFEP_driver`
on m3739:

```bash
export TGLFEP_DIR=/global/cfs/cdirs/m3739/gacode_add_d3d/TGLF-EP   # default in the scripts
# GACODE source providing shared/bin/gacode_setup + platform env (override for your install):
export GACODE_ROOT=.../gacode
export GACODE_PLATFORM=PERLMUTTER_CPU
```

To compare a local build instead, point `TGLFEP_DIR` (or `TGLFEP_DRIVER`) at it.

## Reference case

`examples/DIIID_202017C42_500ms_v3.1/` (`input.gacode`, `dump.profile`, scan-control
`input_scan20_nb{6,8,16,32}.TGLFEP`, and Fortran golden outputs `out.TGLFEP`,
`out.scalefactor_r*`, `alpha_*_crit.input`).

## SCAN_N=20 on GPU (5 nodes, gacode path)

> **CUDA >= 12.6 required.** The GPU eigensolver calls `cusolverDnXgeev`
> (`CUDA.CUSOLVER.Xgeev!`), which does not exist in CUDA 12.4 — a 12.4 runtime
> fails with *"This operation is not supported by the current CUDA version."*
> Load `cudatoolkit/12.9` (the batch scripts already do); do **not** follow the
> Lmod hint to load 12.4. If CUDA.jl was precompiled against a different runtime
> align it once with
> `using CUDA; CUDA.set_runtime_version!(v"12.9"; local_toolkit=true)`.
> GPU and CPU SFmin agree to machine precision.

Validated production layout: **20 radii on 5 GPU nodes** — `srun -n 20`,
**4 tasks/node**, **1 A100 + an MPS worker team** per radius. Scan and merge run
in one job.

```bash
cd dev
sbatch run/batch_run_scan20_5N.sh
# or the fully-documented, self-contained template (edit CONFIG paths at top):
sbatch run/submit_tjlfep_gpu_5N_example.sh
```

Inputs: `examples/DIIID_202017C42_500ms_v3.1/input.gacode` +
`input_scan20_nb32.TGLFEP` (`N_BASIS=32`, `SCAN_N=20`). Outputs per task:
`task_<i>.jls`, optional `out.scalefactor_r*`; after merge: `alpha_*_crit.input`,
`sfmin_scan.txt` (in `OUT_DIR`).

## N_BASIS=6, SCAN_N=20, 10 nodes (CPU, file-based verification)

| Step | Command |
|------|---------|
| Fortran | `cd dev && sbatch verify/batch_debug_nb6_fortran_scan20_10n.sh` |
| Julia | `cd dev && sbatch verify/batch_debug_nb6_julia_scan20_10n.sh` |
| Compare | `FORTRAN_DIR=fortran_runs/debug_nb6_scan20_10n_<FJOB> JULIA_DIR=debug_out_nb6_scan20_<JJOB>_dist FILE_DIR=fileInput_nb6_scan20_10n_<JJOB> ./verify/compare_debug_nb6_scan20.sh` |

Inputs: `examples/DIIID_202017C42_500ms_v3.1/input_scan20_nb6.TGLFEP`
(`N_BASIS=6`, `SCAN_N=20`, `IRS=2`).

Expected agreement at `IR_EXP` radii:

- SFmin: max relative error ~0.03%
- α(dn/dr), α(dp/dr): max relative error ~0.5%

Plots: `dev/compare_nb6_scan20_plots/` (created by the compare script; gitignored).

## Other verification scripts

| N_BASIS | Single radius | SCAN_N=20 (10 nodes) |
|---------|---------------|----------------------|
| 6 | `verify/batch_debug_nb6_{fortran,julia}.sh` + `verify/compare_debug_nb6.sh` | `verify/batch_debug_nb6_*_scan20_10n.sh` + `verify/compare_debug_nb6_scan20.sh` |

Larger-basis (nb16/nb32) verification variants from development are preserved in
`attic/build/` if needed.

Text diff of outputs: `examples/DIIID_202017C42_500ms_v3.1/compare_fortran_julia.jl`
with `FORTRAN_DIR` / `JULIA_DIR` set.
