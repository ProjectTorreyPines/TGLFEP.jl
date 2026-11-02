#!/bin/bash -l
# ExB-shear width-extension early-stop A/B test (premium, single GPU node).
#
# Drives the DIII-D example through critical_factor_robust over a sweep of forced
# gamma_thresh (= what ROTATIONAL_SUPPRESSION_FLAG=1 / ExB shear does) and reports
# cost (n_ext_confirm / evals / wall) + correctness (sfmin / status). Eigensolves
# batch onto one A100 (USE_GPU=1, inner=:threads).
#
#   POST-FIX : current HEAD (this checkout)  -> --project=$ROOT
#   PRE-FIX  : HEAD~1 via a throwaway git worktree -> --project=$WT
#
# Both run JIT (the committed GPU sysimage predates the fix, so it is NOT used —
# that keeps the A/B honest). The live source tree is never mutated.
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -C gpu
#SBATCH -J exb_earlystop_gpu
#SBATCH -o exb_earlystop_gpu_%j.out
#SBATCH -e exb_earlystop_gpu_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"
export JULIA_CUDA_USE_COMPAT=false
export TGLFEP_JULIA_FILE_ONLY=1
export USE_GPU=1

ROOT=/pscratch/sd/t/tneiser/.julia/dev/TJLFEP
TEST="${ROOT}/dev/ad/test_exb_earlystop.jl"
WT="${PSCRATCH}/.julia/dev/TJLFEP_prefix_${SLURM_JOB_ID}"

export NB="${NB:-6}"
export SCAN_IS="${SCAN_IS:-15,18}"
export GTHS="${GTHS:-1e-7,0.05,0.1,0.2,0.4}"
JL_THREADS="${JL_THREADS:-16}"

RUN_PREFIX="${RUN_PREFIX:-0}"   # 1 ⇒ also run the HEAD~1 worktree pre-fix phase

cd "$ROOT"

echo "================ config ================"
echo "host: $(hostname)  date: $(date)"
nvidia-smi -L 2>/dev/null | head -1 || true
echo "NB=${NB}  SCAN_IS=${SCAN_IS}  GTHS=${GTHS}  threads=${JL_THREADS}  USE_GPU=${USE_GPU}  RUN_PREFIX=${RUN_PREFIX}"
echo "current commit: $(git -C "$ROOT" rev-parse --short HEAD)  ($(git -C "$ROOT" log -1 --format=%s))"
echo "working tree:   $(git -C "$ROOT" diff --quiet -- src/tjlfep_ad_extensions.jl && echo clean || echo 'DIRTY (uncommitted marginal_factor early-out)')"

echo
echo "################################ CURRENT (HEAD + early-out) ################################"
stdbuf -oL -eL julia --startup-file=no --project="$ROOT" -t "$JL_THREADS" "$TEST"

if [[ "$RUN_PREFIX" == "1" ]]; then
    # Throwaway worktree pinned at the pre-fix commit; removed on exit.
    cleanup() { git -C "$ROOT" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"; git -C "$ROOT" worktree prune 2>/dev/null || true; }
    trap cleanup EXIT
    git -C "$ROOT" worktree add --detach "$WT" HEAD~1
    echo
    echo "################################ PRE-FIX (HEAD~1 worktree) #################################"
    stdbuf -oL -eL julia --startup-file=no --project="$WT" -t "$JL_THREADS" "$TEST"
fi

echo
echo "================ run finished ================"
