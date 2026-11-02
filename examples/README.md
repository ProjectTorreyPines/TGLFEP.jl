# examples/

Canonical TGLF-EP cases used for verification, validation, and benchmarking.
Run everything from the repo root with the project active:

```bash
module load julia/1.11.7
export JULIA_DEPOT_PATH="$HOME/.julia:$PSCRATCH/.julia${JULIA_DEPOT_PATH:+:$JULIA_DEPOT_PATH}"   # never clobber ~/.julia
```

Full radial scans on Slurm (GPU or CPU) use the templates in [`slurm/`](../slurm/README.md),
which default to the DIII-D case below; point `CASE_DIR`/`TGLFEP_FILE` at `UCP_complete/`
or your own `input.gacode` + `input.TGLFEP` pair for anything else.

## DIIID_202017C42_500ms_v3.1/

DIII-D discharge 202017C42 at 500 ms — the primary Fortran-vs-Julia verification
and GPU-benchmark case.

Inputs:
- `input.gacode` — equilibrium + profiles (GACODE EXPRO format).
- `input.TGLFEP` — single-radius scan-control input.
- `input_singleradius_nb6.TGLFEP` — `N_BASIS=6`, `SCAN_N=1` (quick check).
- `input_scan20_nb{6,8,16,32}.TGLFEP` — `SCAN_N=20` scans at four basis sizes
  (the inputs swept by the timing-vs-N_BASIS benchmark).
- `dump.gacode`, `dump.profile`, `fileInput/` — preprocessed file-based inputs.

Fortran golden references (for trust-building comparisons):
- `out.TGLFEP` — reference `SFmin` profile.
- `out.scalefactor_r*` — per-radius scale-factor references.
- `alpha_dndr_crit.input`, `alpha_dpdr_crit.input` — critical-gradient references.

Case scripts:
- `DIIID_juliaValidation.jl` — IMAS-path validation driver for this case.
- `compare_fortran_julia.jl`, `diagnose_crit_grad.jl` — compare Julia output dirs
  against the Fortran references (`out.TGLFEP`, `alpha_*_crit.input`).
- `plotGrads.jl` — critical-gradient plots.
- `batch_TGLF-EP.sl` — the Fortran `TGLFEP_driver` submit this case was verified against.

Fortran-vs-Julia verification runs on this case live in `dev/verify/`
(see `docs/REPRODUCE_FORTRAN_MATCH.md`); production scans use `slurm/`.

## UCP_complete/

Reactor-relevant case (4 thermal ion species + energetic particles) behind the headline
benchmark in `docs/BENCHMARKS.md`; its per-`ky` eigenmatrix is ~1.7x larger than DIII-D's at
the same `N_BASIS`, which is where the GPU eigensolver pays off most.

Inputs: `input.gacode`, `input.TGLFEP` (single radius), `input_scan20_nb{6,8,16,32,40,48}.TGLFEP`
(`SCAN_N=20` scans; nb40/48 exceed stock Fortran TGLF's compile-time basis cap, Julia runs them
unmodified), `dump.gacode`, `dump.profile`, `Alpha_input`.
Fortran references: `alpha_dndr_crit.input`, `alpha_dpdr_crit.input`, `batch_TGLF-EP.sl`
(the 40-node `-n 5000` Fortran submit).

Extended-width solvers on this case need the long wall noted in `slurm/tglfep_julia_gpu_backfill.sbatch`
(`:ad :locate` at nb32 ~4 h on one node).

## ITER/

ITER case driven two ways through the **same** preprocessing/`runTHD` routines:

- `ITERfromFiles.jl` — file-based path from `input.{TGLFEP,MTGLF,EXPRO}`.
- `ITERstructExample.jl` — FUSE-native IMAS `dd` path: builds a `dd` via
  `FUSE.case_parameters(:ITER)` and calls `runTHD(dd, rho, OptionsDict; ...)`,
  exercising the same `expro_bound_deriv` gradient logic as the `input.gacode`
  path. This is the reference example for capability 6 (FUSE/IMAS `dd`).

```bash
julia --project=. examples/ITER/ITERfromFiles.jl       # file-based ITER run
julia --project=. examples/ITER/ITERstructExample.jl   # FUSE/IMAS dd ITER run
```
