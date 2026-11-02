# utils/

Developer utilities for verifying TJLFEP input preprocessing (not part of the
package API). Run from the repo root with `--project=.`.

## `compare_imas_vs_expro_scan.jl`

Compares the FUSE/IMAS `InputTGLFEP(dd, ...)` `extraEP` against the GACODE EXPRO
logic (`expro_util.f90`) on a radial scan, and isolates the derivative algorithm
(`expro_bound_deriv` vs `IMAS.calc_z`). This is the script used to validate that
the dd-path preprocessing reproduces the `input.gacode` path.

```bash
TGLFEP_JULIA_FILE_ONLY=0 julia --startup-file=no --project=. utils/compare_imas_vs_expro_scan.jl
# Optional env: CASE_DIR (default examples/DIIID_202017C42_500ms_v3.1), IS_EP, RHO_SCAN
```

## `compare_preprocess_inputs.jl`

File-diff of two preprocessed input sets (`input.MTGLF` / `input.EXPRO` /
`input.TGLFEP`), e.g. a Fortran/file reference vs IMAS-generated inputs, with
species-index remapping and per-field relative-error reporting.

```bash
REF_DIR=/path/to/ref TEST_DIR=/path/to/test \
  julia --startup-file=no --project=. utils/compare_preprocess_inputs.jl
```
