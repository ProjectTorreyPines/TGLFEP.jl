# dev/ — maintainer material

Nothing in here is needed to run TJLFEP; the user-facing submit scripts are in
[`slurm/`](../slurm/README.md). This tree holds the scripts that produced the published
benchmarks and verification results:

| Subdir | Contents |
|--------|----------|
| `timing/` | Timing-vs-`N_BASIS` sweeps (Fortran CPU / Julia CPU / Julia GPU, every solver tier), the 1-node backfill node-hours sweep, collectors and plotters behind [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md). |
| `verify/` | Fortran-vs-Julia verification (batch runners, distributed drivers, overlay plotters), plus the old smoke/validate scripts. See [`docs/REPRODUCE_FORTRAN_MATCH.md`](../docs/REPRODUCE_FORTRAN_MATCH.md). |
| `ad/` | Autodiff-solver research: extended-width box experiments, batched/Krylov eigensolver attempts, validation and plotting scripts. See [`docs/AD_SOLVERS_AND_SEARCH_BOUNDS.md`](../docs/AD_SOLVERS_AND_SEARCH_BOUNDS.md). |
| `notes/` | Design notes and post-mortems (`PLAN_robust_ad_cheap_confirm.md`, `README_batched_GPU.md`). |

Conventions: submit from `dev/` (`cd dev && sbatch timing/<script>.sh`); the scripts `cd`
there and their Slurm logs and run output land in `dev/` (all gitignored). They default
`TGLFEP_JULIA_ROOT` to the maintainer's checkout and `-A m3739`/`m3739_g`, so expect to override
`TGLFEP_JULIA_ROOT`, the account, and `TGLFEP_JULIA_GPU_SYSIMAGE`/`TGLFEP_JULIA_SYSIMAGE` (the old local
`build/*.so` defaults no longer exist; the scripts fall back to JIT when the file is missing).
They share the production driver, MPS wrapper, and merge script in `slurm/common/`.
