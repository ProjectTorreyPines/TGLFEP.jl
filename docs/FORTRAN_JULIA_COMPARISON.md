# TGLF-EP: Julia (TJLFEP) vs Fortran comparison results

This document records verification outcomes from the detailed comparison plan. The Fortran reference is the shared build at `/global/cfs/cdirs/m3739/gacode_add_d3d/TGLF-EP` (a snapshot of the original `TGLF-EP/` subdir is archived under `attic/`); Julia lives in `TJLFEP/src/`.

> **Scope: this verification covers the `grid` solver** (`mainsub(...; solver=:grid)`, the default `kwscale_scan` `(kyhat × width × factor)` sweep), which reproduces the Fortran `SFmin` to its printed precision. The autodiff **`ad` solver** (`solver=:ad`) is a deliberate, *grid-independent* departure: it finds the smooth instability/keep onset by a Newton root-find with exact AD derivatives instead of quantizing to the coarse EP-scale-factor grid, so its `sfmin` can differ from — and is more accurate than — the Fortran/grid value. The AD result is **not** expected to match Fortran bit-for-bit; on DIII-D the grid `sfmin` is effectively a coarse-grid / keep-flag-transition artifact that the AD onset resolves continuously. See the top-level `README.md` benchmark section.

## 1. RMIN_LOC normalization (`verify-rmin-norm`)

| Code | Assignment |
|------|------------|
| Fortran `TGLFEP_read_EXPRO` | `rmin(:) = EXPRO_rmin(:) / a_meters` then `tglf_rmin_loc_in = rmin(ir)` |
| Julia `TJLF_map` | `RMIN_LOC = RMIN[ir] / RMIN[end]` |
| Julia FUSE `context.jl` | `RMIN` stored in metres; `RMIN_LOC = rmin[ir]/a` |

**Conclusion:** Equivalent for the IMAS/FUSE path when `RMIN` is in physical units. No change required for `RMIN_LOC`.

**Additional fix:** `GAMMA_THRESH_MAX` used raw `RMIN[ir]` in metres; now uses `r_over_a = RMIN[ir]/RMIN[end]` to match Fortran’s dimensionless `rmin(ir)`.

## 2. QL_flux_ratio for `n > nmodes_out` (`verify-ql-ratio-guard`)

| Code | Behaviour |
|------|-----------|
| Fortran `TGLFEP_ky` | Accumulates QL for all `nmodes`; no `1e20` guard |
| Julia (before) | Skipped QL accumulation when `n > nmodes_out`; set `QL_flux_ratio[n] = 1e20` |
| Julia (after) | QL accumulation matches Fortran; no `1e20` branch |

**Conclusion:** Julia aligned with Fortran. For `nmodes_out < NMODES`, empty QL arrays yield `NaN` ratios; Fortran and Julia both treat `NaN < thresh` as false (no QL-ratio rejection).

## 3. TGLF defaults (`verify-tglf-defaults`)

Julia `TJLFEP_ky` sets explicitly:

- `USE_AVE_ION_GRID = false`
- `FIND_EIGEN = true`
- `RLNP_CUTOFF = 18.0`
- `BETA_LOC = 0.0`, `KX0_LOC = 0.0`

Fortran `TGLFEP_ky` relies on GACODE `tglf_run` defaults after `TGLFEP_tglf_map`. TJLF.jl module defaults: `BETA_LOC=0`, `KX0_LOC=0`, `FIND_EIGEN=true` via `tjlf_read_input.jl`.

**Conclusion:** Consistent with TJLF defaults used by the Fortran driver path.

## 4. FACTOR_IN rounding (`verify-factor-rounding`)

Julia had `FACTOR_IN = round(100000*FACTOR_IN)/100000` in `TJLF_map`; Fortran does not round.

**Change:** Rounding removed for Fortran parity.

## 5. QL weight axis ordering (`verify-ql-axis-order`)

| Accessor | Index order |
|----------|-------------|
| Fortran `get_QL_particle_flux(n, species, jfields)` | mode, species, field |
| Julia `particle_QL_out[jfields, species, n]` | field, species, mode |

Julia indexing `particle_QL_out[jfields, inputsEP.IS_EP + 1, n]` matches Fortran `get_QL_*(n, is_EP+1, jfields)`.

**Conclusion:** Axis order is consistent.

## 6. IR_EXP mapping (`diagnose-crit-grad-diff`)

| Code | `INPUT_PROFILE_METHOD == 2` |
|------|------------------------------|
| Fortran | `ir_exp(i) = IRS + floor((i-1)*(NR-IRS)/(SCAN_N-1))` |
| Julia (before) | `argmin(|grid - rho[i]|)` |
| Julia (after) | Fortran linear formula restored in `run_tjlfep.jl` |

**Change:** `IR_EXP` now matches Fortran for DIII-D validation.

**Note:** For `NR=101`, `IRS=2`, `SCAN_N=20`, and uniform `ρ` grid, `argmin` and the Fortran formula yield the **same index vector** `[2, 7, 12, …, 101]`. Plot shifts from IR_EXP alone are unlikely for this case; remaining differences are dominated by `SFmin` (TJLF vs GACODE TGLF) and the 3- vs 4-species layout.

## 7. IS_EP / species layout

| | Fortran (DIII-D reference) | Julia IMAS path |
|--|---------------------------|-----------------|
| `input.TGLFEP` | `IS_EP=2`, `N_ION=2` → `ns=3` | `InputTGLFEP(...; is_ep=1)` → fast D on 1st ion |
| TJLF EP index | `is = IS_EP+1 = 3` | `Options.IS_EP = ep_slot-1 = 3`, `is = 4` with `ns=4` |

Julia keeps an extra thermal ion slot (`ns = N_ION + 2`). `is_ep=1` in `DIIID_juliaValidation.jl` selects fast D in IMAS; this corresponds to Fortran’s 2nd EXPRO ion (`IS_EP=2`) in the 3-species layout.

Post-processing uses `DENS_$ep_slot` (last species), consistent with the EP slot in the 4-species TJLF grid.

## 8. DIII-D setup (`setup-diiid-run`)

- Paths: `EP_TJLF` → `TJLFEP` in `DIIID_juliaValidation.jl` and `batchRun.sh`.
- TJLF branch: use `origin/gpu_new` at `/pscratch/sd/t/tneiser/.julia/dev/TJLF`.
- Sysimage: `TJLFEP/build/TGLFEP_JULIA_cpu_sysimage.so` (optional).
- Comparison script: `examples/DIIID_202017C42_500ms_v3.1/compare_fortran_julia.jl`.

## 10. Debug workflow (N_BASIS=6)

| Slurm job name | Code | Script |
|----------------|------|--------|
| `TGLFEP_nb6` | Fortran TGLF-EP | `dev/verify/batch_debug_nb6_fortran.sh` |
| `TJLFEP_nb6` | Julia TJLFEP | `dev/verify/batch_debug_nb6_julia.sh` |

Set `TGLFEP_DEBUG=1` (Fortran) and `TGLFEP_JULIA_DEBUG=1` (Julia) for matched debug lines tagged `[TGLFEP_DBG]` and `[TGLFEP_JULIA_DBG]`. No sysimage is recommended while TJLFEP/TJLF are under active development.

## 11. Single-radius basis scans (SCAN_N=1, ir=2)

Same case as nb6; only `N_BASIS` changes in the scan-control input. The kept nb6
single-radius path uses `examples/DIIID_202017C42_500ms_v3.1/input_singleradius_nb6.TGLFEP`.

| N_BASIS | Fortran job | Julia job | Compare script |
|---------|-------------|-----------|----------------|
| 6 | `TGLFEP_nb6` | `TJLFEP_nb6` | `dev/verify/compare_debug_nb6.sh` |

Nb6 single-radius match (2026-05-18): SFmin 2.8125, n_EP/n_e 96.1%, β_crit 4.46%.

Larger-basis (nb16/nb32) single-radius and production variants from development
are archived under `attic/build/`.

## 12. N_BASIS=6 SCAN_N=20 on 10 nodes (2026-05-19)

| Job | Code | Script |
|-----|------|--------|
| `TGLFEP_nb6_s20_10n` | Fortran | `dev/verify/batch_debug_nb6_fortran_scan20_10n.sh` |
| `TJLFEP_nb6_s20_10n` | Julia | `dev/verify/batch_debug_nb6_julia_scan20_10n.sh` |

Compare: `dev/verify/compare_debug_nb6_scan20.sh` → `dev/compare_nb6_scan20_plots/`.

At scan radii (`IR_EXP`): SFmin max rel err ~0.03%; α(dn/dr) ~0.5%; α(dp/dr) ~0.4%. Full reproduction steps: `docs/REPRODUCE_FORTRAN_MATCH.md`.

## 13. N_BASIS=32 SCAN_N=20 GPU vs Fortran head-to-head (2026-06-02)

Full 20-radius production scan on Perlmutter A100 nodes (MPS team, GPU eigenvalue
**and** eigenvector solve), case `202017C42_500ms_v3.1`, `input_scan20_nb32.TGLFEP`.

**Correctness.** All three GPU layouts (20N / 10N / 5N) produced an *identical*
`SFmin` profile across all 20 radii, matching the Fortran reference `out.TGLFEP`
to its printed 4-decimal precision (e.g. `0.6249450…→0.6249`, `0.3124850…→0.3125`,
`1.2497672…→1.2498`). `scan_index=2` golden `SFmin = 0.6249450209778226`.

**Performance** (Fortran CPU reference: **25.8 min on 10 nodes**):

| Run | nodes | GPU/radius | team | wall | speedup | node-min |
|-----|-------|-----------|------|------|---------|----------|
| Fortran CPU            | 10 | – | –       | 25.8 min | 1.0× | 258 |
| Julia GPU **10N** (fair footprint) | 10 | 2 | 16w×2t | 5.3 min  | 4.8× | 53 |
| Julia GPU **5N** (best efficiency) |  5 | 1 | 8w×2t  | 4.9 min  | 5.3× | 25 |
| Julia GPU 20N         | 20 | 4 | 16w×4t | 4.8 min  | 5.4× | 161 |

On the same 10-node footprint the GPU port is **4.8× faster**; the most
efficient layout (**5N**) is **5.3× faster on half the nodes** (~10× fewer
node-hours). Each task spawns a fresh MPS worker team, so per-radius cost is
currently **JIT-compilation-bound** (~110 s/team; warm compute is ~35 s at 4
GPUs). That is why the 5N/10N/20N walls are ~tied — a GPU-worker **sysimage**
(removing the JIT) is the next lever and should push the scan toward ~2–3 min and
re-expose the more-GPUs/radius advantage.

Scripts: `slurm/common/mps-scan-wrapper.sh` + `slurm/tglfep_julia_gpu_5N.sbatch` (the
validated production layout; the 20N/10N layouts are archived under `attic/build/`,
generalized by `GPUS_PER_RADIUS`). The GPU eigenvector inverse-iteration solve
lives in TJLF (`_gpu_lu_solve!`, CUSOLVER getrf/getrs), gated by
`TJLF_GPU_EIGVEC` (default on): validated 1.85× (`:threads`) / 3.55× (`:mps_team`)
per-radius vs the host `lu!`/`ldiv!`, SFmin bit-exact.

## 14. dd/IMAS-path preprocessing parity with input.gacode (2026-06-03)

Goal: a `dd` converted from an `input.gacode` should reproduce the gacode-file
TJLFEP inputs. `InputTGLF_EP` (`src/context.jl`) was aligned to the file path
(`tjlfep_read_inputs.jl` / GACODE `expro_util.f90`):

| Quantity | Before (dd path) | After (matches file path) |
|----------|------------------|---------------------------|
| log-gradients (`dlnn/dlnt`) | `IMAS.calc_z(rmin, f, :backward)` | `expro_bound_deriv(-log(f), rmin_work)` (3-pt Lagrange, `rmin+1e-6`, `eps_n=1e-30`) |
| geometry slopes (`drmaj`, `s_kappa/delta/zeta`, `dpdr`) | `IMAS.gradient(rmin, …)` | `expro_bound_deriv(…, rmin_work)` |
| magnetic shear | `(rmin/q)·dq/dr` | `rmin·d(ln\|q\|)/dr` (expro_util log form) |
| sound speed `cs` | `GACODE.c_s` (mass `md`, cm/s) | expro_util CGS `sqrt(k·1e3·Te/(2·md))/1e2` (m/s); stored as cm/s |
| `gammaE`/`gammap` | `a/c_s` in m/(cm/s), `abs(q)` | `a/cs` in m/(m/s) with **signed** `q` |

**Verification** (`utils/compare_imas_vs_expro_scan.jl`, dd vs Fortran EXPRO on
`dump.gacode`): RLNS/RLTS match to ~1.7e-6; `cs` bit-exact; `|gammaE|`/`|gammap|`
bit-exact (previously off by `100×√2`). A residual overall sign on
`gammaE`/`gammap` (IMAS stores `q`>0 and flips `w0`) is irrelevant — the
threshold logic uses `abs(gammaE)`/`abs(gammap)` and only when
`ROTATIONAL_SUPPRESSION_FLAG=1`.

## 9. Running comparison (`test-agreement`)

```bash
cd examples/DIIID_202017C42_500ms_v3.1
module load julia/1.11.7   # Perlmutter
julia --project=../.. compare_fortran_julia.jl path/to/GPU_n32_*_20_*
julia --project=../.. diagnose_crit_grad.jl path/to/GPU_n32_*_20_* 4
```

Fortran reference outputs (in `examples/DIIID_202017C42_500ms_v3.1/`): `out.scalefactor_r*`, `alpha_*_crit.input`, `out.TGLFEP`.

The comparison parser was smoke-tested on Fortran reference files (self-comparison). Re-run `DIIID_juliaValidation.jl` after the code fixes above, then point the scripts at the new `GPU_*` or `CPU_*` output directory.
