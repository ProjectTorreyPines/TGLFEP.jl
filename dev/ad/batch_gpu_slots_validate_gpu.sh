#!/bin/bash -l
# Phase 0/1 gate for the TJLF/TJLFEP speedup work. Runs the :grid solver over the 20-radius DIII-D
# case at nb=16 on one A100, twice:
#   (1) TJLF_GPU_SLOTS=1  -> legacy serialized per-device dispatch (must reproduce the golden)
#   (2) TJLF_GPU_SLOTS=N  -> concurrent per-device slots (must be BITWISE-identical to (1))
# then diffs both against dev/ad/golden_grid_nb16.csv with check_grid_exact.jl and prints the
# wall-time speedup. Nonzero exit if either exactness gate fails.
#
#   cd TJLFEP && sbatch dev/ad/batch_gpu_slots_validate_gpu.sh
# Env: NB (16), SLOTS (4), RADII (all).
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -J tglfep_julia_slots_gate
#SBATCH -o dev/ad/gpu_slots_validate_%j.out
#SBATCH -e dev/ad/gpu_slots_validate_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"
export JULIA_CUDA_USE_COMPAT=false

TGLFEP_JULIA_ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TGLFEP_JULIA_ROOT}"

export USE_GPU=1
export GKSwstype=nul
export NB="${NB:-16}"
export SOLVERS=grid
export RADII="${RADII:-}"
SLOTS="${SLOTS:-4}"
GOLDEN="dev/ad/golden_grid_nb${NB}.csv"
JL=(julia --startup-file=no --project="${TGLFEP_JULIA_ROOT}" -t 16)

echo "=== GPU slots exactness+speed gate (nb=${NB}, slots 1 vs ${SLOTS}) ==="
echo "host: $(hostname)  date: $(date)  job=${SLURM_JOB_ID:-?}"
nvidia-smi -L 2>/dev/null | head -1 || true

echo "### run 1/2: TJLF_GPU_SLOTS=1 (legacy serialized) ###"
TJLF_GPU_SLOTS=1 stdbuf -oL -eL "${JL[@]}" dev/ad/benchmark_nls_solvers.jl
cp "dev/ad/benchmark_nls_solvers_nb${NB}.csv" "dev/ad/grid_slots1_nb${NB}.csv"

echo "### run 2/2: TJLF_GPU_SLOTS=${SLOTS} (concurrent) ###"
TJLF_GPU_SLOTS="${SLOTS}" stdbuf -oL -eL "${JL[@]}" dev/ad/benchmark_nls_solvers.jl
cp "dev/ad/benchmark_nls_solvers_nb${NB}.csv" "dev/ad/grid_slots${SLOTS}_nb${NB}.csv"

rc=0
echo "### gate A: slots=1 vs golden (refactor must not change legacy) ###"
GOLDEN="${GOLDEN}" CANDIDATE="dev/ad/grid_slots1_nb${NB}.csv" "${JL[@]}" dev/ad/check_grid_exact.jl || rc=1
echo "### gate B: slots=${SLOTS} vs slots=1 (concurrency must be bitwise-identical) ###"
GOLDEN="dev/ad/grid_slots1_nb${NB}.csv" CANDIDATE="dev/ad/grid_slots${SLOTS}_nb${NB}.csv" "${JL[@]}" dev/ad/check_grid_exact.jl || rc=1

echo "### wall-time comparison (sum of grid wall_s) ###"
"${JL[@]}" -e '
nb=ENV["NB"]; s=ENV["SLOTS"]
sumwall(p)=sum(parse(Float64, split(l,",")[7]) for l in Iterators.drop(eachline(p),1) if !isempty(strip(l)) && split(l,",")[2]=="grid")
w1=sumwall("dev/ad/grid_slots1_nb$(nb).csv"); wN=sumwall("dev/ad/grid_slots$(s)_nb$(nb).csv")
println("  slots=1  Σwall=", round(w1,digits=1), "s")
println("  slots=", s, "  Σwall=", round(wN,digits=1), "s   speedup=", round(w1/wN,digits=2), "x")
' NB="${NB}" SLOTS="${SLOTS}"

echo "=== gate done (rc=${rc}) $(date) ==="
exit $rc
