# In-container CUDA MPS team driver, shared by test_slurm_container_gpu_mps.sbatch
# (Perlmutter, podman-hpc) and test_slurm_container_gpu_mps_defiant.sbatch (Defiant,
# Apptainer): one radius of the packaged DIII-D example whose inner kw-scan is
# distributed over MPS_TEAM local worker processes sharing this task's GPU.
#
# Mirrors slurm/common/run_gacode_scan20_mps_task.jl: workers inherit the MPS
# pipe dir + compat/depot env, get the launcher-selected sysimage when present
# (-gpu tags; addprocs does NOT propagate --sysimage on its own), and touch
# the GPU up front so every CUDA context is an MPS client before the scan.
# The MPS daemon must already be up on the HOST (the sbatch scripts do that).

ENV["TGLFEP_JULIA_FILE_ONLY"] = "1"
using Distributed
pipe = ENV["CUDA_MPS_PIPE_DIRECTORY"]
@assert ispath(joinpath(pipe, "control")) "MPS control pipe not visible in-container at $pipe"

nteam   = parse(Int, get(ENV, "MPS_TEAM", "4"))
tworker = parse(Int, get(ENV, "JULIA_WORKER_THREADS", "2"))
sysimg  = unsafe_string(Base.JLOptions().image_file)
exeflags = endswith(sysimg, "sys_tglfep_julia_gpu.so") ?
    `--startup-file=no -t $tworker --sysimage=$sysimg` :
    `--startup-file=no -t $tworker`

env = Dict{String,String}(k => ENV[k] for k in
    ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "JULIA_CUDA_USE_COMPAT",
     "JULIA_DEPOT_PATH", "JULIA_PROJECT", "TGLFEP_JULIA_FILE_ONLY", "CUDA_VISIBLE_DEVICES")
    if haskey(ENV, k))
env["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
addprocs(nteam; exeflags, env)

@everywhere begin
    ENV["TGLFEP_JULIA_FILE_ONLY"] = "1"
    using CUDA, TJLFEP, TJLF, LinearAlgebra
    BLAS.set_num_threads(1)
    CUDA.functional() || error("worker $(myid()): CUDA not functional — MPS client init failed")
    CUDA.device!(first(CUDA.devices()))   # create the MPS-client context up front
end

using CUDA, TJLFEP
println("team: $(nworkers()) workers x $tworker threads on ", CUDA.name(first(CUDA.devices())),
        "  sysimage: ", isempty(sysimg) ? "<none, JIT>" : sysimg)

ex = joinpath(pkgdir(TJLFEP), "examples", "DIIID_202017C42_500ms_v3.1")
@time res = run_gacode_scan_task(
    joinpath(ex, "input.gacode"),
    joinpath(ex, "input_singleradius_nb6.TGLFEP"),
    1; out_dir=mktempdir(), use_gpu=true, printout=true,
    inner=:mps_team, team=workers())
println("CONTAINER MPS TEAM OK  scan_index=", res.scan_index, " sfmin=", res.sfmin,
        "  TJLFEP ", pkgversion(TJLFEP))
