#!/bin/bash
#SBATCH -A m3739
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 02:30:00
#SBATCH -J tglfep_gridbench
#SBATCH -o tglfep_gridbench_%j.out
#SBATCH -e tglfep_gridbench_%j.out
#
# One-node GC / layout study of the CPU :grid path (see bench_grid_cpu.jl).
# Submit from the TJLFEP checkout:  sbatch dev/timing/batch_bench_grid_cpu.sh
#
# Env:
#   RUNS      space-separated run names (default: the full baseline set below)
#   BENCH_ID  label for the output directory (default: job id)

set -uo pipefail
ROOT="${TGLFEP_JULIA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT}"
module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${HOME}/.julia:${PSCRATCH}/.julia:/global/cfs/cdirs/m3739/TJLFEP/depot"
export OPENBLAS_NUM_THREADS=1

OUT="${PSCRATCH}/tglfep_gridbench/${BENCH_ID:-${SLURM_JOB_ID}}"
mkdir -p "${OUT}"
JL=(julia --startup-file=no --project="${ROOT}")
BENCH="${ROOT}/dev/timing/bench_grid_cpu.jl"
D3D="${ROOT}/examples/DIIID_202017C42_500ms_v3.1"
UCP="${ROOT}/examples/UCP_complete"
RUNS="${RUNS:-prod_k4 combo1 phys64 gc1 gc8 heap iso64 pencil ucp_combo1}"

echo "ROOT=${ROOT} OUT=${OUT} RUNS=${RUNS}"
git -C "${ROOT}" log --oneline -1
echo "TJLF: $("${JL[@]}" -e 'using TJLF; println(pkgdir(TJLF))' 2>/dev/null)"
"${JL[@]}" -e 'using Pkg; Pkg.precompile(); using TJLFEP' 2>&1 | tail -3

for r in ${RUNS}; do
  echo; echo "########## ${r} $(date +%T)"
  case "${r}" in
    # Production layout: one srun task with 64 logical CPUs, 64 Julia threads, 4 rounds.
    prod_k4)
      BENCH_TAG=${r} BENCH_OUT="${OUT}/d3d_nb32_r10_k4.jls" TGLFEP_JULIA_PROBE=1 \
        srun -N1 -n1 -c64 --exact "${JL[@]}" -t 64 "${BENCH}" ;;
    prod_k1)
      BENCH_TAG=${r} BENCH_K_MAX=1 BENCH_OUT="${OUT}/d3d_nb32_r10_k1.jls" TGLFEP_JULIA_PROBE=1 \
        srun -N1 -n1 -c64 --exact "${JL[@]}" -t 64 "${BENCH}" ;;
    # One thread, a few combos: per-combo compute with no contention at all.
    combo1)
      BENCH_TAG=${r} BENCH_MODE=combo BENCH_NCOMBO=4 BENCH_OUT="${OUT}/d3d_nb32_r10_combos.jls" \
        srun -N1 -n1 -c2 --exact "${JL[@]}" -t 1 "${BENCH}" ;;
    # 64 threads on 64 PHYSICAL cores (128 logical CPUs, one thread per core).
    phys64)
      BENCH_TAG=${r} BENCH_K_MAX=1 JULIA_EXCLUSIVE=0 \
        srun -N1 -n1 -c128 --exact --cpu-bind=cores "${JL[@]}" -t 64 "${BENCH}" ;;
    # as phys64 but with Julia threads pinned (thread i -> CPU i-1: one per core, socket 0)
    pin64)
      BENCH_TAG=${r} BENCH_K_MAX=1 JULIA_EXCLUSIVE=1 \
        srun -N1 -n1 -c128 --exact --cpu-bind=cores "${JL[@]}" -t 64 "${BENCH}" ;;
    phys64_k4)
      BENCH_TAG=${r} BENCH_OUT="${OUT}/d3d_nb32_r10_k4_phys.jls" \
        srun -N1 -n1 -c128 --exact --cpu-bind=cores "${JL[@]}" -t 64 "${BENCH}" ;;
    # two radii side by side, as a full node would run them (2 workers x 64 threads)
    pair64)
      BENCH_TAG=${r} BENCH_K_MAX=1 \
        srun -N1 -n2 -c128 --exact --cpu-bind=cores "${JL[@]}" -t 64 "${BENCH}" 2>&1 | grep -E "RESULT tag" ;;
    # 128 threads in one process on all 128 cores
    t128)
      BENCH_TAG=${r} BENCH_K_MAX=1 \
        srun -N1 -n1 -c256 --exact --cpu-bind=cores "${JL[@]}" -t 128 "${BENCH}" ;;
    # 16 processes x 8 threads, one radius round each (process-level isolation, 128 cores)
    p16t8)
      BENCH_TAG=${r} BENCH_K_MAX=1 \
        srun -N1 -n16 -c16 --exact --cpu-bind=cores "${JL[@]}" -t 8 "${BENCH}" 2>&1 | grep -E "RESULT tag" | sort | sed -n '1p;$p' ;;
    gc1)
      BENCH_TAG=${r} BENCH_K_MAX=1 srun -N1 -n1 -c64 --exact "${JL[@]}" -t 64 --gcthreads=1 "${BENCH}" ;;
    gc8)
      BENCH_TAG=${r} BENCH_K_MAX=1 srun -N1 -n1 -c64 --exact "${JL[@]}" -t 64 --gcthreads=8 "${BENCH}" ;;
    heap)
      BENCH_TAG=${r} BENCH_K_MAX=1 srun -N1 -n1 -c64 --exact "${JL[@]}" -t 64 --heap-size-hint=150G "${BENCH}" ;;
    # Fortran-like isolation: 64 single-thread processes at once, 4 combos each, one
    # physical core per process. Per-combo time under full memory-bandwidth load, no shared heap.
    iso64)
      BENCH_TAG=${r} BENCH_MODE=combo BENCH_NCOMBO=4 \
        srun -N1 -n64 -c2 --exact --cpu-bind=cores "${JL[@]}" -t 1 "${BENCH}" 2>&1 | grep -E "RESULT tag|per_combo" | sort | uniq -c | sort -rn | head -8
      ;;
    # Raw LAPACK: one dumped pencil, gesv!+geev! single thread, repeated.
    pencil)
      mkdir -p "${OUT}/pencils"
      TJLF_DUMP_PENCILS="${OUT}/pencils" TJLF_DUMP_PENCILS_MAX=2 TJLF_DUMP_PENCILS_MINSIZE=500 \
        BENCH_TAG=${r}_dump BENCH_MODE=combo BENCH_NCOMBO=2 srun -N1 -n1 -c2 --exact "${JL[@]}" -t 1 "${BENCH}" | grep RESULT
      srun -N1 -n1 -c2 --exact "${JL[@]}" -t 1 -e '
        using LinearAlgebra, Serialization
        using LinearAlgebra.LAPACK: gesv!, geev!, getrf!
        BLAS.set_num_threads(1)
        p = deserialize(joinpath(ARGS[1], "pencil_0001.jls")); n = size(p.A, 1)
        println("pencil n=", n, " BLAS=", BLAS.get_config())
        for rep in 1:3
            A = copy(p.A); B = copy(p.B)
            t1 = @elapsed gesv!(B, A)
            t2 = @elapsed geev!('"'"'N'"'"', '"'"'N'"'"', A)
            Z = copy(p.A); t3 = @elapsed getrf!(Z)
            t4 = @elapsed (copy(p.A); copy(p.B); copy(p.A))
            println("RESULT pencil rep=", rep, " gesv_s=", round(t1; digits=3), " geev_s=", round(t2; digits=3),
                    " getrf_s=", round(t3; digits=3), " three_copies_s=", round(t4; digits=3))
        end' "${OUT}/pencils" ;;
    ucp_combo1)
      CASE_DIR="${UCP}" BENCH_TAG=${r} BENCH_MODE=combo BENCH_NCOMBO=2 BENCH_OUT="${OUT}/ucp_nb32_r10_combos.jls" \
        srun -N1 -n1 -c2 --exact "${JL[@]}" -t 1 "${BENCH}" ;;
    ucp_k1)
      CASE_DIR="${UCP}" BENCH_TAG=${r} BENCH_K_MAX=1 BENCH_OUT="${OUT}/ucp_nb32_r10_k1.jls" TGLFEP_JULIA_PROBE=1 \
        srun -N1 -n1 -c64 --exact "${JL[@]}" -t 64 "${BENCH}" ;;
    *) echo "unknown run ${r}" ;;
  esac
done
echo "ALL DONE $(date +%T)"
