#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CATFLOW_SEEDS=${CATFLOW_SEEDS:-"0 1 2"}
CATFLOW_WL_ITERS="1 2 3"
CATFLOW_QM9_ROOT=${CATFLOW_QM9_ROOT:-"benchmark_runs/catflow_qm9_declared_multiseed"}
CATFLOW_BATCH_SIZE_WL1=${CATFLOW_BATCH_SIZE_WL1:-256}
CATFLOW_BATCH_SIZE_WL2=${CATFLOW_BATCH_SIZE_WL2:-128}
CATFLOW_BATCH_SIZE_WL3=${CATFLOW_BATCH_SIZE_WL3:-128}

if [[ -z "${CATFLOW_SEEDS//[[:space:]]/}" ]]; then
  echo "CATFLOW_SEEDS must contain at least one integer seed." >&2
  exit 2
fi
if [[ -z "${CATFLOW_WL_ITERS//[[:space:]]/}" ]]; then
  echo "CATFLOW_WL_ITERS must contain at least one positive integer." >&2
  exit 2
fi

mkdir -p "${CATFLOW_QM9_ROOT}"

for seed in ${CATFLOW_SEEDS}; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid CatFlow seed ${seed@Q}; expected non-negative integers." >&2
    exit 2
  fi

  for wl_iters in ${CATFLOW_WL_ITERS}; do
    if [[ ! "${wl_iters}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid WL iteration count ${wl_iters@Q}; expected positive integers." >&2
      exit 2
    fi

    case "${wl_iters}" in
      1) batch_size=${CATFLOW_BATCH_SIZE_WL1} ;;
      2) batch_size=${CATFLOW_BATCH_SIZE_WL2} ;;
      3) batch_size=${CATFLOW_BATCH_SIZE_WL3} ;;
      *) echo "Unsupported CatFlow QM9 WL level: ${wl_iters}" >&2; exit 2 ;;
    esac

    echo "CatFlow QM9: WL${wl_iters}, seed ${seed}, batch size ${batch_size}"
    SEED="${seed}" \
    N_WL_ITERS="${wl_iters}" \
    BATCH_SIZE="${batch_size}" \
    BENCHMARK_SUITE=catflow_2406_04843 \
    BENCH_DATASETS=QM9 \
    RUN_ABLATIONS=0 \
    BENCH_ROOT="${CATFLOW_QM9_ROOT}/wl${wl_iters}/seed${seed}" \
      bash scripts/run_all_benchmarks.sh
  done
done

"${PYBIN}" src/aggregate_catflow_molecule_metrics.py \
  --root "${CATFLOW_QM9_ROOT}" \
  --dataset QM9 \
  --seeds ${CATFLOW_SEEDS} \
  --wl-iters ${CATFLOW_WL_ITERS} \
  --output-dir "${CATFLOW_QM9_ROOT}/catflow_aggregate"

cat > "${CATFLOW_QM9_ROOT}/declared_seeds.txt" <<EOF
Dataset: QM9
Independent training/generation seeds: ${CATFLOW_SEEDS}
WL iteration counts: ${CATFLOW_WL_ITERS}
Default batch sizes: WL1=${CATFLOW_BATCH_SIZE_WL1}, WL2=${CATFLOW_BATCH_SIZE_WL2}, WL3=${CATFLOW_BATCH_SIZE_WL3}
Each (WL, seed) run has its own checkpoints and 10,000-attempt CatFlow evaluation.
EOF
