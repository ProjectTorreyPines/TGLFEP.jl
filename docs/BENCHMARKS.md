# Benchmark: cost vs N_BASIS (UCP, SCAN_N=20)

Moved here from the top-level README. Reproduction scripts live in `dev/timing/`.

Timing/accuracy on the **reactor-relevant `UCP_complete`** case (4 thermal ion
species + energetic particles), a 20-radius scan on Perlmutter with sysimages.
This is the headline benchmark because the GPU eigensolver's payoff scales with
the eigenmatrix size (species × `N_BASIS`): UCP's per-`ky` eigenmatrix is `2400²`
(complex) at `N_BASIS=32` vs `1440²` for DIII-D, so each eigensolve is several×
heavier and the GPU pulls decisively ahead. The smaller DIII-D verification case,
where the margin is thinner, is in
[`docs/README_DIII-D_example.md`](docs/README_DIII-D_example.md).

**Cost is reported in node-hours** (nodes × wallclock), the fair,
layout-independent metric (Fortran `-n 1280` on 10 CPU nodes, the GPU tiers on 5 or
1). Each solver runs on its fastest parallel layout (**rule of thumb: `grid` → MPS
team, `ad` → in-process threads**; an MPS team only adds overhead to `ad`).

The solvers, with node-hours vs the fully MPI-parallel Fortran (`-n 1280`) at `N_BASIS=32`:

| Solver | vs Fortran | What it computes |
| --- | ---: | --- |
| **`:ad :only`** | **~8×** | Approximates `:grid`: a smooth, de-quantized version of the `w≥1` result (median `:only/grid ≈ 0.9`). Fast iteration only; misses the `w<1` edge modes. |
| **`:ad :wide`** | **~5.4×** | Adds the narrow `w<1` AE modes in one log-seeded pass; conservative (within ~1–2× of `:ad :locate`, never below it). Bulk NN-DB generation. |
| **`:grid`** | **~4.4×** | The verified Fortran-equivalent `(kyhat × width × factor)` sweep (thousands of eigensolves/radius). |
| **`:ad :locate`** *(default)* | **~1.7×** | Adds the narrow `w<1` AE modes; the faithful narrow-width value. Production default. |

The `ad` solvers are **grid-independent**: rather than reading `sfmin` off the coarse
factor grid, they locate the instability onset directly (Newton root-find with exact
forward-mode AD derivatives). `:grid`/`:ad :only` stay in the Fortran `w≥1` box;
`:ad :locate`/`:ad :wide` additionally resolve the narrow-width (`w<1`)
EP-driven modes it excludes (the *entire* `sfmin` reduction below grid, up to ~16× at
the edge). Higher-fidelity internal reference tiers (`robust_ad`, and the
`nbasis`-converged `truth`) sit above `:ad :locate`, which matches them essentially
bit-for-bit; they are documented for reference in
[`docs/AD_SOLVERS_AND_SEARCH_BOUNDS.md`](https://github.com/ProjectTorreyPines/TGLFEP.jl/blob/master/docs/AD_SOLVERS_AND_SEARCH_BOUNDS.md).

**Accuracy**: `sfmin(IR)` for all solvers at `N_BASIS=32`:

![UCP sfmin vs radius: Fortran vs grid vs :ad :only vs :ad :locate vs :ad :wide](https://raw.githubusercontent.com/ProjectTorreyPines/TGLFEP.jl/master/docs/plots/ucp_accuracy_nb32.png?v=1)

`:grid` (grey) reproduces the Fortran reference (blue) bit-for-bit, and `:ad :only`
(orange) tracks it closely (the de-quantized `w≥1` value). `:ad :locate` (green) and
`:ad :wide` (dark red) drop well below `grid` into the narrow-width `w<1` modes at the
outer radii (IR ≳ 40; ~10× below grid at IR≈117, up to ~16× at IR≈180). `:ad :wide`
stays within ~1–2× of `:ad :locate` and never below it. Colors and legend order match
the node-hours plot below for quick cross-reference.

![UCP node-hours vs N_BASIS](https://raw.githubusercontent.com/ProjectTorreyPines/TGLFEP.jl/master/docs/plots/ucp_scan20_timing_nodehours.png?v=2)

Absolute node-hours at `N_BASIS=32`, each solver on its fastest layout (Fortran
`-n 1280` on 10 CPU nodes; `grid`/`:ad :only` on 5 MPS GPU nodes; `:ad
:locate`/`:ad :wide` on a 1-node backfill with 4 GPU workers draining a 20-radius
queue): Fortran ≈2.47; `:ad :only` ≈0.31, `:ad :wide` ≈0.46, `grid` ≈0.57,
`:ad :locate` ≈1.49 — so **every GPU tier beats the fully-parallel Fortran**
(~8× / ~5.4× / ~4.4× / ~1.7×). Julia `:grid` on **CPU** (≈14.6, same 10 nodes as
Fortran) is ~6× *slower* than Fortran: the GPU eigensolver is what makes the Julia
port competitive. The GPU advantage *grows* with `N_BASIS` (Fortran is cheaper at
`N_BASIS ≤ 8`, break-even is near 16, and the GPU pulls ~4–8× ahead by 32) as the
eigenmatrix grows: at `N_BASIS=48` (Fortran ≈8.29 node-hours) the margins are
`:grid` **~8.0×** (≈1.03), `:ad :only` **~8.7×** (≈0.95), `:ad :wide` **~5.4×**
(≈1.53), `:ad :locate` **~2.1×** (≈3.97). `:ad :wide` is ~3× cheaper than
`:locate` at `N_BASIS=32`. The per-backend tables below give the raw wallclock
seconds.

**Fortran cannot run `N_BASIS>32` without recompiling.** Stock TGLF hard-caps the
Hermite basis at a compile-time `PARAMETER (nb=32)`; requesting more does **not**
error — a `put_switches` sanity check silently resets the run to the *default*
`nbasis=4`, returning garbage in seconds (the failure is only visible as an
impossibly fast run). The `N_BASIS=40/48` Fortran columns here required rebuilding
the TGLF library with `nb=48`, `nxm=95` (a full `make clean` rebuild — the gacode
Makefile does not track Fortran module dependencies). TJLF/TJLFEP allocate the
basis dynamically, so the Julia solvers run any `N_BASIS` unmodified.

**Best-throughput layout depends on the solver** (for these `SCAN_N=20` runs):
- **`:grid` and `:ad :only` → 5 GPU nodes.** Per-radius cost is uniform, so
  spreading the 20 radii across 5 nodes (~4 radii/node) minimizes wallclock with
  no wasted node-hours.
- **`:ad :locate` and `:ad :wide` → 1 GPU node, backfill.** Their edge radii take
  much longer (the narrow-width `w<1` locate is triggered there), so a fixed
  multi-node split would leave nodes idle waiting on the straggler edge radii.
  Running a single node with 4 workers draining a shared 20-radius claim queue
  keeps every GPU busy and gives the lowest node-hours. At `N_BASIS=48` the
  default 32-worker team OOMs the node's 256 GB host RAM, so the nb48
  `:locate`/`:wide` rows run a halved team (`MPS_TEAM=4`, 16 workers) on 80 GB
  A100 nodes (`-C gpu&hbm80g`) — the fastest layout that fits.

**Grid solver**: Fortran CPU (10 nodes, `-n 1280` = 128 ranks/node) vs Julia CPU
(10 nodes, SlurmClusterManager) vs Julia GPU (5 A100 nodes, **MPS team**):

| N_BASIS | Fortran CPU (s) | Julia CPU (s) | Julia GPU MPS (s) | GPU speedup vs Fortran (wallclock) |
|--------:|----------------:|--------------:|------------------:|-----------------------------------:|
| 6  | 20.3   | 128.2   | 178.4 | 0.11× |
| 8  | 25.4   | 179.5   | 184.3 | 0.14× |
| 16 | 112.2  | 739.0   | 197.0 | 0.57× |
| 32 | 888.8  | 5243.7  | 407.5 | **2.18×** |
| 40 | 1741.3 | 11082.4 | 518.9 | 3.36× |
| 48 | 2982.5 | 20489.1 | 743.9 | **4.01×** |

(Wallclock here compares a **5-node** GPU run to a **10-node** `-n 1280` Fortran run
— *half* the nodes — so it understates the GPU, yet it still wins 2.18× at
`N_BASIS=32` and 4.01× at 48; the fair node-count-normalized margins are ~4.4× and
~8.0×. Julia `:grid` on CPU, on the same 10 nodes as Fortran, is a steady ~6–7×
*slower* at every `N_BASIS`, so the GPU is doing the heavy lifting. The 40/48
Fortran times use the `nb=48` rebuild described above; the 40/48 Julia CPU rows ran
on a v2.0.13 CPU sysimage — the `N_BASIS≤32` rows used an earlier bake — so there is
a minor code-version seam in the CPU column only.)

**`:ad :only` (bare `w≥1` AD, no faithful confirm)**: Julia GPU (5 A100 nodes,
**in-process threads**), vs the grid GPU path (both 5-node, so this wallclock ratio is
apples-to-apples):

| N_BASIS | Grid GPU MPS (s) | `:ad :only` GPU threads (s) | `:only` vs grid-GPU (wallclock) |
|--------:|-----------------:|----------------------------:|--------------------------------:|
| 6  | 178.4 | 87.4  | 2.0× |
| 8  | 184.3 | 90.3  | 2.0× |
| 16 | 197.0 | 119.8 | 1.6× |
| 32 | 407.5 | 220.9 | **1.8×** |
| 40 | 518.9 | 320.8 | 1.6× |
| 48 | 743.9 | 681.7 | 1.1× |

`:ad :only` is ~1.6–2.0× faster than the grid MPS path in wallclock (narrowing to
~1.1× at `N_BASIS=48`, where the few-but-huge AD eigensolves stop amortizing their
serial descent): it replaces the
thousands of grid eigensolves/radius with a handful of AD Newton steps. Note this
is the *same* `ad` solver run two ways: on an **MPS team** instead of threads it is
*slower*, because the per-radius AD regions are small and the descent is sequential,
so team-spawn/remote-call overhead never amortizes. **Rule of thumb: `grid` → MPS,
`ad` → threads.**

Data: `docs/plots/ucp_scan20_timing.csv`. Reproduce with
`dev/timing/submit_ucp_scan20.sh` (Fortran `-n 1280` + all GPU tiers). The
smaller DIII-D verification case (bit-for-bit Fortran match) is in
[`docs/README_DIII-D_example.md`](docs/README_DIII-D_example.md).
