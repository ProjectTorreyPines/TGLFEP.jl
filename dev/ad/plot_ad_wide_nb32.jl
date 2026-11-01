#!/usr/bin/env julia
# Accuracy comparison at N_BASIS=32 over the 20-radius DIII-D 202017C42_500ms_v3.1 scan:
# sfmin (critical IR factor) vs radial grid index, in the SAME legend order and per-solver
# colors as the node-hours timing plot (dev/timing/plot_scan20_timing_wide.jl) so the two
# figures cross-reference at a glance:
#   * Fortran (CPU -n 1280 reference; SFmin block from timing_runs/fortran_scan20_n1280_nb32_<jobid>/out.TGLFEP)
#   * Julia :grid       the Fortran-equivalent w>=1 grid solver (reproduces Fortran bit-for-bit)
#   * Julia :ad :only   the bare w>=1 AD approximation of grid (fast-turnaround; no width extension)
#   * Julia :ad :locate the width-extended AD solver (multistart locate-grid; faithful extended-width value, default)
#   * Julia :ad :wide   the fast single-pass width-aware AD mode (extend_mode=:wide, wide_kdesc=2)
# Both extended solvers search down to the ballooning-width physics floor WIDTH_FLOOR = 0.5 (2026-10 runs).
# Sources are pinned per job id (dev/d3d_nb32_scan20_<solver>_<jobid>_tasks/sfmin_scan.txt, merged 20-radius
# outputs of slurm/tjlfep_gpu_{5N,backfill}.sbatch); override any of them with the environment variables below.
# Shows that :wide recovers nearly all of :ad :locate's narrow-width accuracy that the w>=1 grid
# misses, at a fraction of the cost (see node-hours-vs-nbasis plot). The internal robust_ad/truth
# reference tiers (matched bit-for-bit by :ad :locate) are documented in
# docs/AD_SOLVERS_AND_SEARCH_BOUNDS.md and intentionally omitted here to keep the figure on the
# 2x2 user-facing options.
#   julia --project=. dev/ad/plot_ad_wide_nb32.jl
using Pkg
Pkg.activate(normpath(@__DIR__, "..", ".."))
using Plots, Printf
using Plots.PlotMeasures: mm

const BUILD = normpath(@__DIR__, "..")
read_sfmin(p) = begin
    ir = Int[]; f = Float64[]
    for ln in eachline(p)
        t = split(strip(ln))
        length(t) >= 3 || continue
        push!(ir, parse(Int, t[2])); push!(f, parse(Float64, t[3]))
    end
    ir, f
end

# Fortran SFmin(IR) reference: the `SFmin` block in the nb=32 SCAN_N=20 out.TGLFEP.
const FORT = get(ENV, "D3D_FORTRAN_OUT", joinpath(BUILD, "timing_runs", "fortran_scan20_n1280_nb32_60226283", "out.TGLFEP"))
read_fortran_sfmin(p) = begin
    vals = Float64[]; inblk = false
    for ln in eachline(p)
        s = strip(ln)
        s == "SFmin" && (inblk = true; continue)
        if inblk
            startswith(s, "-") && break
            v = tryparse(Float64, s)
            v === nothing || push!(vals, v)
        end
    end
    vals
end

# Julia solvers, in the shared legend order / colors used by the timing plot.
series = [
    ("Julia :grid (GPU)", get(ENV, "D3D_GRID_SFMIN", joinpath(BUILD, "d3d_nb32_scan20_grid_60226408_tasks", "sfmin_scan.txt")),
        :gray55, :dash, :diamond),
    ("Julia :ad :only (GPU)", get(ENV, "D3D_ONLY_SFMIN", joinpath(BUILD, "d3d_nb32_scan20_ad_only_60639930_tasks", "sfmin_scan.txt")),
        :darkorange, :dot, :rect),
    ("Julia :ad :locate (GPU, default)", get(ENV, "D3D_LOCATE_SFMIN", joinpath(BUILD, "d3d_nb32_scan20_ad_locate_60416815_tasks", "sfmin_scan.txt")),
        :seagreen, :solid, :utriangle),
    ("Julia :ad :wide (GPU)", get(ENV, "D3D_WIDE_SFMIN", joinpath(BUILD, "d3d_nb32_scan20_ad_wide_60639942_tasks", "sfmin_scan.txt")),
        :firebrick, :solid, :star5),
]

default(legendfontsize=9, guidefontsize=11, tickfontsize=9, dpi=200, fontfamily="Computer Modern")
p = plot(xlabel="radial grid index IR", ylabel="sfmin (critical factor)",
         title="SCAN_N=20 accuracy @ N_BASIS=32: DIII-D 202017C42_500ms_v3.1",
         yscale=:log10, legend=:topleft, size=(900, 560),
         left_margin=4mm, bottom_margin=4mm)

# Fortran reference first (steelblue) so the legend reads Fortran, :grid, :only, :locate, :wide.
ir_ref, _ = read_sfmin(series[1][2])
fort = read_fortran_sfmin(FORT)
if length(fort) == length(ir_ref)
    plot!(p, ir_ref, fort; label="Fortran (CPU)", color=:steelblue, linestyle=:solid,
          marker=:circle, markersize=6, linewidth=3)
else
    @warn "Fortran SFmin length mismatch; skipping Fortran line" nfort=length(fort) nir=length(ir_ref)
end
for (lab, path, col, ls, mk) in series
    isfile(path) || (@warn "missing $path"; continue)
    ir, f = read_sfmin(path)
    plot!(p, ir, f; label=lab, color=col, linestyle=ls, marker=mk, markersize=5, linewidth=2)
end
out = joinpath(@__DIR__, "ad_wide_accuracy_nb32.png")
savefig(p, out)
println("Wrote ", out)

# Quick numeric table: Fortran vs grid (should match bit-for-bit) and grid/:wide vs :ad :locate.
irg, grd = read_sfmin(series[1][2])
_, loc = read_sfmin(series[3][2]); _, wid = read_sfmin(series[4][2])
println("\n IR   Fortran     grid    locate   (grid/loc)    wide    (wide/loc)")
for i in eachindex(irg)
    ft = i <= length(fort) ? fort[i] : NaN
    @printf("%4d  %8.4f  %8.4f  %8.4f  %7.2fx  %8.4f  %7.2fx\n",
        irg[i], ft, grd[i], loc[i], grd[i]/loc[i], wid[i], wid[i]/loc[i])
end
