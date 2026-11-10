#!/usr/bin/env julia
# Node-hours vs N_BASIS for the SCAN_N=20 DIII-D scan. Plots the 2x2 solver-selection
# options (+ the Fortran baseline) so the figure maps onto the README decision matrix:
#   * Fortran (CPU, reference baseline; its own node count from the CSV)
#   * Julia :grid       (julia_gpu_nh)         -- matches Fortran (w>=1 box)
#   * Julia :ad :only   (julia_gpu_ad_only_nh)   -- approximates :grid (bare w>=1 AD)
#   * Julia :ad :locate (julia_gpu_ad_locate_nh) -- extends Fortran with the 0.5 <= w < 1 AE modes (default)
#   * Julia :ad :wide   (julia_gpu_ad_wide_nh)   -- fast single-pass approximation of :locate
#   * Julia :grid on CPU (julia_cpu_nh)          -- the :grid solver without a GPU, same 10 nodes as Fortran
# All series come from timing_runs/scan20_timing.csv (copied to docs/plots/scan20_timing.csv), one column per
# series as in the UCP CSV; the extended-width rows use the WIDTH_FLOOR = 0.5 default (2026-10).
# The robust_ad/truth reference tiers are intentionally omitted: :ad :locate matches their
# accuracy at lower cost, so the plot guides users along the four matrix options.
# Node-hours (nodes x wallclock) is the fair metric regardless of node count (Fortran CPU on
# 10 nodes, the GPU tiers on 5), so the legend just tags (CPU)/(GPU).
#   julia --project=. dev/timing/plot_scan20_timing_wide.jl
using Pkg
Pkg.activate(normpath(@__DIR__, "..", ".."))
using Plots, Printf
using Plots.PlotMeasures: mm

const BUILD = normpath(@__DIR__, "..")
const CSV   = joinpath(BUILD, "timing_runs", "scan20_timing.csv")
const OUT   = joinpath(BUILD, "timing_runs", "scan20_timing_wide_lines.png")

# --- all series by column name from the node-hours CSV (same layout as docs/plots/ucp_scan20_timing.csv:
#     n_basis, <series>_s ..., <series>_nh ..., notes; node-hours = nodes x wallclock) ---
lines = readlines(CSV)
hdr   = split(lines[1], ",")
rows  = [split(l, ",") for l in lines[2:end] if !isempty(strip(l))]
col(name) = findfirst(==(name), hdr)
getcol(name) = [(i = col(name); (i === nothing || i > length(r) || isempty(strip(r[i]))) ? NaN : parse(Float64, r[i])) for r in rows]
basis  = [parse(Int, r[1]) for r in rows]
fort   = getcol("fortran_nh")              # Fortran -n 1280, 10 CPU nodes
grid   = getcol("julia_gpu_nh")            # :grid, 5-node MPS team
only   = getcol("julia_gpu_ad_only_nh")    # :ad :only, 5-node threads
locate = getcol("julia_gpu_ad_locate_nh")  # :ad :locate, 1-node backfill (WIDTH_FLOOR = 0.5)
wide   = getcol("julia_gpu_ad_wide_nh")    # :ad :wide, 1-node backfill (WIDTH_FLOOR = 0.5)
cpu    = getcol("julia_cpu_nh")            # :grid on CPU, same 10 nodes as Fortran

default(legendfontsize=12, guidefontsize=15, tickfontsize=12, titlefontsize=15, dpi=200, fontfamily="Computer Modern")
# Plot only the 2x2 solver-selection options (+ Fortran baseline): the reference tiers
# robust_ad/truth are omitted here because :ad :locate matches their accuracy at lower cost.
ys = filter(!isnan, vcat(fort, grid, locate, wide, only, cpu))
ymax = (isempty(ys) ? 1.0 : maximum(ys)) * 1.12
p = plot(xlabel="N_BASIS", ylabel="Node-hours (nodes × wallclock)",
         title="SCAN_N=20 node-hours vs N_BASIS: DIII-D",
         xticks=basis, legend=:topleft, ylim=(0, ymax), size=(900, 560),
         left_margin=4mm, bottom_margin=4mm)
plot!(p, basis, fort;   label="Fortran (CPU)",          marker=:circle,   linewidth=2, markersize=6, color=:steelblue)
any(!isnan, cpu) && plot!(p, basis, cpu; label="Julia :grid (CPU)", marker=:hexagon, linewidth=2, markersize=6, color=:purple, linestyle=:dashdot)
plot!(p, basis, grid;   label="Julia :grid (GPU)",      marker=:diamond,  linewidth=2, markersize=6, color=:gray55, linestyle=:dash)
any(!isnan, only) && plot!(p, basis, only; label="Julia :ad :only (GPU)", marker=:rect, linewidth=2, markersize=5, color=:darkorange, linestyle=:dot)
plot!(p, basis, locate; label="Julia :ad :locate (GPU)",marker=:utriangle,linewidth=2, markersize=6, color=:seagreen,   linestyle=:solid)
plot!(p, basis, wide;   label="Julia :ad :wide (GPU)", marker=:star5, linewidth=2.5, markersize=8, color=:firebrick)
savefig(p, OUT)
println("Wrote ", OUT)

println("\n N_BASIS  Fortran   :grid    :only    :locate   :wide   (locate/wide)")
for i in eachindex(basis)
    c(x) = isnan(x) ? "  —   " : @sprintf("%6.3f", x)
    rat = (!isnan(locate[i]) && !isnan(wide[i]) && wide[i] > 0) ? @sprintf("%5.2fx", locate[i]/wide[i]) : "  —  "
    @printf("  %4d   %s  %s  %s  %s  %s   %s\n", basis[i], c(fort[i]), c(grid[i]), c(only[i]), c(locate[i]), c(wide[i]), rat)
end
