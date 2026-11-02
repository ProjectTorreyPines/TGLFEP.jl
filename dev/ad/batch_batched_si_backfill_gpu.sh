#!/bin/bash -l
# Node-hours of the 20-radius inner=:batched_si scan under a DYNAMIC BACKFILL layout, PROCESS-BASED:
# 4 worker julia processes (one GPU each, -t 8 like the fixed-shard runner) drain a shared radius
# queue by atomic-mkdir work-stealing. Directly comparable to the fixed-round-robin-shard node-hours
# (batched_si_nodehours.csv) — same per-GPU resourcing, only dynamic vs static radius assignment —
# to test whether backfill wins when per-radius cost is non-uniform (dense-fallback stragglers).
#
# NOTE (why this replaces the old in-process version): the previous single-process runner spawned the
# 4 GPU workers with Threads.@spawn, but kwscale_scan's dense endpoints call Threads.@threads, and
# nesting @threads inside @spawn on the shared thread pool crashed (job 55844344). Separate processes
# give each worker its own thread pool + GPU, mirroring the working fixed-shard template.
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 02:30:00
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -J tglfep_julia_bsi_backfill
#SBATCH -o dev/ad/bsi_backfill_%j.out
#SBATCH -e dev/ad/bsi_backfill_%j.err
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

NGPU=4
CSV="${ROOT}/dev/ad/batched_si_nodehours_backfill.csv"
echo "nb,layout,ngpu,wall_s,node_hours" > "${CSV}"
echo "=== batched_si BACKFILL node-hours (process-based work-stealing)  $(date) ==="
nvidia-smi -L 2>/dev/null | head -4 || true

for NB in 16 32; do
  echo ""; echo "######## nb=${NB}  backfill (${NGPU} GPU worker processes, shared mkdir queue) ########"
  QDIR="${ROOT}/dev/ad/bsi_backfill_q_nb${NB}_${SLURM_JOB_ID:-local}"
  rm -rf "${QDIR}"; mkdir -p "${QDIR}"
  t0=$(date +%s.%N)
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g NB=$NB NFACTOR=8 NEFWID=8 NKYHAT=4 KMAX=4 \
      QDIR="${QDIR}" WID=$g OUT="${ROOT}/dev/ad/bsi_bf_sfmin_nb${NB}_w${g}.txt" \
      julia --startup-file=no "${SYSARG[@]}" --project="${ROOT}" -t 8 \
      dev/ad/run_batched_si_backfill_worker.jl > "${ROOT}/dev/ad/bsi_bf_nb${NB}_w${g}.log" 2>&1 &
  done
  wait
  t1=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
  nh=$(awk "BEGIN{printf \"%.4f\", $wall/3600}")

  # merge worker shards (idx IR sfmin secs wid) -> sorted sfmin file (idx IR sfmin) for the overlay
  MERGED="${ROOT}/dev/ad/bsi_bf_all_nb${NB}.txt"
  cat "${ROOT}/dev/ad/bsi_bf_sfmin_nb${NB}_w"*.txt 2>/dev/null | sort -n > "${MERGED}"
  awk '{print $1, $2, $3}' "${MERGED}" > "${ROOT}/dev/ad/batched_si_sfmin_backfill_nb${NB}.txt"
  nrad=$(wc -l < "${MERGED}")

  # per-radius skew + parallel efficiency = (sum per-radius secs / NGPU) / wall
  read -r rmin rmax rsum <<< "$(awk 'NR==1{mn=$4;mx=$4} {s+=$4; if($4<mn)mn=$4; if($4>mx)mx=$4} END{printf "%.0f %.0f %.0f", mn, mx, s}' "${MERGED}")"
  peff=$(awk "BEGIN{printf \"%.2f\", ($rsum/${NGPU})/${wall}}")
  # per-worker radius counts (load balance)
  wcounts=$(awk '{c[$5]++} END{for(w=0;w<'"${NGPU}"';w++) printf "w%d=%d ", w, c[w]+0}' "${MERGED}")

  echo "nb=${NB}: wall=${wall}s  node_hours=${nh}  (radii=${nrad})"
  echo "  per-radius secs: min=${rmin} max=${rmax} sum=${rsum}   parallel_eff=${peff}   load: ${wcounts}"
  echo "${NB},backfill,${NGPU},${wall},${nh}" >> "${CSV}"
  rm -rf "${QDIR}"
done

echo ""; echo "=== backfill node-hours CSV ==="; cat "${CSV}"
echo "=== done $(date) ==="
