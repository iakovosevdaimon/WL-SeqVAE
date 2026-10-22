#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
SEED=${SEED:-0}
ROOT=${ROOT:-benchmark_runs/corrected_graphs_wl1_wl2_wl3_seed${SEED}}
TRACE_ROOT=${TRACE_ROOT:-trace_cache/corrected_graphs_wl1_wl2_wl3_seed${SEED}}
DATA_ROOT=${DATA_ROOT:-benchmark_data/all_corrected}
TU_DATA_ROOT=${TU_DATA_ROOT:-${DATA_ROOT}/tu}
PAPER_DATA_ROOT=${PAPER_DATA_ROOT:-${DATA_ROOT}/paper_graphs}

TU_DATASETS=${TU_DATASETS:-"MUTAG NCI1"}
PAPER_GRAPH_DATASETS=${PAPER_GRAPH_DATASETS:-"EGO_SMALL COMMUNITY_SMALL ENZYMES GRID PLANAR SBM PROTEINS"}
WL_ITERS=${WL_ITERS:-"1 2 3"}

EPOCHS=${EPOCHS:-50}
# "auto" selects a dataset/WL-specific batch in scripts/run_all_benchmarks.sh.
BATCH_SIZE=${BATCH_SIZE:-auto}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}
EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS:-15}
EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA:-0.001}
REFERENCE_SPLIT=${REFERENCE_SPLIT:-test}
MAX_RESTARTS=${MAX_RESTARTS:-1}
EVAL_COMPLETION_MODES=${EVAL_COMPLETION_MODES:-"strict allow_incomplete"}
METRIC_MAX_TRAIN_GRAPHS=${METRIC_MAX_TRAIN_GRAPHS:-20000}
DIAGNOSTIC_MAX_TRAIN_GRAPHS=${DIAGNOSTIC_MAX_TRAIN_GRAPHS:-20000}
MMD_MAX_GRAPHS=${MMD_MAX_GRAPHS:-2000}

mkdir -p "${ROOT}" "${TRACE_ROOT}"

run_group() {
  local wl_iters="$1"
  local group="$2"
  local datasets="$3"
  local sample_mode="$4"

  env \
    "PYBIN=${PYBIN}" \
    "SEED=${SEED}" \
    "MODEL_NAME=budget_ring_wl_vae_corrected" \
    "BENCHMARK_SUITE=all_papers" \
    "BENCH_ROOT=${ROOT}/wl${wl_iters}/${group}" \
    "TRACE_ROOT=${TRACE_ROOT}/wl${wl_iters}/${group}" \
    "TU_DATA_ROOT=${TU_DATA_ROOT}" \
    "PAPER_DATA_ROOT=${PAPER_DATA_ROOT}" \
    "BENCH_DATASETS=${datasets}" \
    "N_WL_ITERS=${wl_iters}" \
    "N_SAMPLES=${sample_mode}" \
    "REFERENCE_SPLIT=${REFERENCE_SPLIT}" \
    "EPOCHS=${EPOCHS}" \
    "BATCH_SIZE=${BATCH_SIZE}" \
    "EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}" \
    "EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS}" \
    "EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA}" \
    "MAX_SEQ_LEN=auto" \
    "MAX_RESTARTS=${MAX_RESTARTS}" \
    "EVAL_COMPLETION_MODES=${EVAL_COMPLETION_MODES}" \
    "METRIC_MAX_TRAIN_GRAPHS=${METRIC_MAX_TRAIN_GRAPHS}" \
    "DIAGNOSTIC_MAX_TRAIN_GRAPHS=${DIAGNOSTIC_MAX_TRAIN_GRAPHS}" \
    "MMD_MAX_GRAPHS=${MMD_MAX_GRAPHS}" \
    "COMPUTE_MMD_RATIO=1" \
    "RUN_ABLATIONS=0" \
    "REUSE_PAPER_ZINC_CKPT=0" \
    "FCD_COMMAND=" \
    ./scripts/run_all_benchmarks.sh
}

for wl_iters in ${WL_ITERS}; do
  case "${wl_iters}" in
    1|2|3) ;;
    *)
      echo "Unsupported WL iteration count: ${wl_iters}; expected one of 1, 2, 3." >&2
      exit 2
      ;;
  esac
  run_group "${wl_iters}" tu "${TU_DATASETS}" reference
  run_group "${wl_iters}" papers "${PAPER_GRAPH_DATASETS}" paper
done

"${PYBIN}" src/benchmark_report.py --root "${ROOT}" --out "${ROOT}/summary"

echo
echo "Corrected generic-graph WL1/WL2/WL3 sweep finished."
echo "Root: ${ROOT}"
echo "Summary: ${ROOT}/summary/benchmark_summary.md"
