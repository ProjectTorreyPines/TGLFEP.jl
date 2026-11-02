#!/bin/bash -l
# NODE-HOURS vs N_BASIS for the full 20-radius inner=:batched_si scan (the README's timing metric).
# Radii are embarrassingly parallel, so we shard the 20 radii across all 4 A100s on the node (5
# radii/GPU) — the same "keep every GPU busy" layout the grid uses via MPS — and measure wallclock.
#   node-hours = (#nodes=1) * wall_seconds / 3600
# Also emits merged sfmin(IR) per N_BASIS for the accuracy overlay. Compare to grid/ad/Fortran
# node-hours in docs/plots/scan20_timing.csv.
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -J tglfep_julia_bsi_nodehours
#SBATCH -o dev/ad/bsi_nodehours_%j.out
#SBATCH -e dev/ad/bsi_nodehours_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4

set -uo pipefail
module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"
export JULIA_CUDA_USE_COMPAT=false
ROOT="${TGLFEP_JULIA_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${ROOT}"
SO="${ROOT}/build/TGLFEP_JULIA_gpu_sysimage.so"
SYSARG=(); [[ -f "${SO}" ]] && SYSARG=(--sysimage="${SO}")

CSV="${ROOT}/dev/ad/batched_si_nodehours.csv"
echo "nb,nodes,ngpu,wall_s,node_hours" > "${CSV}"
echo "=== batched_si node-hours vs nbasis (4-GPU sharded 20-radius scan)  $(date) ==="
nvidia-smi -L 2>/dev/null | head -4 || true

# 20 scan indices sharded across 4 GPUs: GPU g gets indices {g+1, g+5, g+9, ...} (round-robin).
SHARD0="1,5,9,13,17"; SHARD1="2,6,10,14,18"; SHARD2="3,7,11,15,19"; SHARD3="4,8,12,16,20"

for NB in 6 8 16 32; do
  echo ""; echo "######## nb=${NB}  4-GPU sharded 20-radius batched_si scan ########"
  t0=$(date +%s.%N)
  for g in 0 1 2 3; do
    case $g in 0) SH=$SHARD0;; 1) SH=$SHARD1;; 2) SH=$SHARD2;; 3) SH=$SHARD3;; esac
    CUDA_VISIBLE_DEVICES=$g NB=$NB USE_GPU=1 NFACTOR=8 NEFWID=8 NKYHAT=4 KMAX=4 \
      RADII_IDX="$SH" OUT="${ROOT}/dev/ad/bsi_sfmin_nb${NB}_g${g}.txt" \
      julia --startup-file=no "${SYSARG[@]}" --project="${ROOT}" -t 8 \
      dev/ad/run_batched_si_sfmin.jl > "${ROOT}/dev/ad/bsi_nb${NB}_g${g}.log" 2>&1 &
  done
  wait
  t1=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
  nh=$(awk "BEGIN{printf \"%.4f\", $wall/3600}")
  # merge shard sfmin -> sorted per-index file for the accuracy overlay
  cat "${ROOT}/dev/ad/bsi_sfmin_nb${NB}_g"*.txt 2>/dev/null | sort -n > "${ROOT}/dev/ad/batched_si_sfmin_nb${NB}.txt"
  nrad=$(wc -l < "${ROOT}/dev/ad/batched_si_sfmin_nb${NB}.txt")
  echo "nb=${NB}: wall=${wall}s  node_hours=${nh}  (radii merged=${nrad})"
  echo "${NB},1,4,${wall},${nh}" >> "${CSV}"
  grep -H TOTAL_WALL_S "${ROOT}/dev/ad/bsi_nb${NB}_g"*.log 2>/dev/null | sed 's/^/    /'
done

echo ""; echo "=== node-hours CSV ==="; cat "${CSV}"
echo "=== done $(date) ==="
