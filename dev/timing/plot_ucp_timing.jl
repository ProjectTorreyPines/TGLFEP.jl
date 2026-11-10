#!/usr/bin/env julia
# Plot the UCP_complete SCAN_N=20 timing sweep (node-hours vs N_BASIS) from
# timing_runs/ucp_scan20_timing.csv, and print a wallclock + node-hours table.
# Colors match ad/plot_ucp_accuracy_nb32.jl for cross-reference.
#   cd dev && julia --project=.. timing/plot_ucp_timing.jl
using Pkg
Pkg.activate(normpath(@__DIR__, "..", ".."))
using Plots, Printf
using Plots.PlotMeasures: mm

const BUILD = normpath(@__DIR__, "..")
const CSV = joinpath(BUILD, "timing_runs", "ucp_scan20_timing.csv")
isfile(CSV) || error("missing $CSV -- run timing/collect_ucp_timing.jl first")

lines = readlines(CSV)
hdr = split(lines[1], ",")
col(name) = findfirst(==(name), hdr)
rows = [split(l, ",") for l in lines[2:end] if !isempty(strip(l))]
nb = [parse(Int, r[col("n_basis")]) for r in rows]
getcol(name) = [ (i = col(name); (i === nothing || isempty(r[i])) ? NaN : parse(Float64, r[i])) for r in rows]

# (csv label, legend, color, linestyle, marker)
plots = [
    ("fortran",             "Fortran (CPU -n 1280)",       :steelblue, :solid, :circle),
    ("julia_cpu",           "Julia :grid (CPU)",           :purple,    :dashdot, :hexagon),
    ("julia_gpu",           "Julia :grid (GPU)",           :gray55,    :dash,  :diamond),
    ("julia_gpu_ad_only",   "Julia :ad :only (GPU)",       :darkorange,:dot,   :rect),
    ("julia_gpu_ad_locate", "Julia :ad :locate (GPU)",     :seagreen,  :solid, :utriangle),
    ("julia_gpu_ad_wide",   "Julia :ad :wide (GPU)",       :firebrick, :solid, :star5),
]

default(legendfontsize=9, guidefontsize=11, tickfontsize=9, dpi=200, fontfamily="Computer Modern")
p = plot(xlabel="N_BASIS", ylabel="node-hours (nodes x wallclock)",
         title="UCP_complete SCAN_N=20 cost vs N_BASIS", xscale=:log2, yscale=:log10,
         legend=:topleft, size=(950, 560), left_margin=4mm, bottom_margin=4mm, xticks=(nb, string.(nb)))
for (name, lab, col_, ls, mk) in plots
    y = getcol("$(name)_nh")
    all(isnan, y) && continue
    plot!(p, nb, y; label=lab, color=col_, linestyle=ls, marker=mk, markersize=6, linewidth=2)
end
out = joinpath(BUILD, "timing_runs", "ucp_scan20_timing_nodehours.png")
savefig(p, out)
println("Wrote ", out)

# tables
println("\n=== Wallclock seconds ===")
@printf("%-6s", "nb"); for (n, l, _, _, _) in plots; @printf("%16s", l); end; println()
for (ri, b) in enumerate(nb)
    @printf("%-6d", b)
    for (name, _, _, _, _) in plots
        v = getcol("$(name)_s")[ri]; @printf("%16s", isnan(v) ? "-" : @sprintf("%.1f", v))
    end
    println()
end
println("\n=== Node-hours ===")
@printf("%-6s", "nb"); for (n, l, _, _, _) in plots; @printf("%16s", l); end; println()
for (ri, b) in enumerate(nb)
    @printf("%-6d", b)
    for (name, _, _, _, _) in plots
        v = getcol("$(name)_nh")[ri]; @printf("%16s", isnan(v) ? "-" : @sprintf("%.3f", v))
    end
    println()
end

# GPU-vs-Fortran node-hours ratio (the headline number)
println("\n=== node-hours speedup vs Fortran -n 1280 (Fortran_nh / series_nh) ===")
fort = getcol("fortran_nh")
for (name, lab, _, _, _) in plots[2:end]
    y = getcol("$(name)_nh")
    @printf("%-26s", lab)
    for ri in eachindex(nb)
        r = (isnan(fort[ri]) || isnan(y[ri]) || y[ri] == 0) ? NaN : fort[ri] / y[ri]
        @printf("  nb%d=%s", nb[ri], isnan(r) ? "-" : @sprintf("%.2fx", r))
    end
    println()
end
