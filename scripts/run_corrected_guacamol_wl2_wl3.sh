#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
SEED=${SEED:-0}
ROOT=${ROOT:-benchmark_runs/corrected_guacamol_wl2_wl3_seed${SEED}}
TRACE_ROOT=${TRACE_ROOT:-trace_cache/corrected_guacamol_wl2_wl3_seed${SEED}}
DATA_ROOT=${DATA_ROOT:-benchmark_data/all_corrected}
BENCHMARK_DATA_DIR=${BENCHMARK_DATA_DIR:-${DATA_ROOT}/molecule_benchmarks}
WL_ITERS=${WL_ITERS:-"2 3"}

EPOCHS=${EPOCHS:-50}
# "auto" selects batch 128 for WL2 and batch 64 for WL3.
BATCH_SIZE=${BATCH_SIZE:-auto}
N_SAMPLES=${N_SAMPLES:-10000}
# Default to exactly 50 epochs; set a positive patience to opt into early stopping.
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}
EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS:-15}
EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA:-0.001}
REFERENCE_SPLIT=${REFERENCE_SPLIT:-test}
MOL_PREPROCESS=${MOL_PREPROCESS:-catflow}
FCD_REFERENCE_SPLIT=${FCD_REFERENCE_SPLIT:-test}
FCD_DEVICE=${FCD_DEVICE:-cuda}
FCD_COMMAND=${FCD_COMMAND:-"${PYBIN} src/compute_fcd.py --ref {ref} --gen {gen} --out {out} --device ${FCD_DEVICE}"}
FCD_TIMEOUT=${FCD_TIMEOUT:-1200}
MAX_RESTARTS=${MAX_RESTARTS:-1}
EVAL_COMPLETION_MODES=${EVAL_COMPLETION_MODES:-"strict allow_incomplete"}
AUTO_DOWNLOAD_BENCHMARK_DATA=${AUTO_DOWNLOAD_BENCHMARK_DATA:-1}
INSTALL_BENCHMARK_PACKAGES=${INSTALL_BENCHMARK_PACKAGES:-0}
BENCHMARK_METRIC_SAMPLES=${BENCHMARK_METRIC_SAMPLES:-10000}
MOSES_BACKEND=${MOSES_BACKEND:-auto}

mkdir -p "${ROOT}" "${TRACE_ROOT}" "${BENCHMARK_DATA_DIR}"

batch_size_for_wl() {
  local wl_iters="$1"
  local specific_name="BATCH_SIZE_GUACAMOL_WL${wl_iters}"
  local value="${!specific_name:-}"
  if [[ -z "${value}" && "${BATCH_SIZE}" != "auto" ]]; then
    value="${BATCH_SIZE}"
  fi
  if [[ -z "${value}" ]]; then
    case "${wl_iters}" in
      2) value=128 ;;
      3) value=64 ;;
    esac
  fi
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid GuacaMol batch size ${value@Q} for WL${wl_iters}." >&2
    return 2
  fi
  printf '%s\n' "${value}"
}

for wl_iters in ${WL_ITERS}; do
  case "${wl_iters}" in
    2|3) ;;
    *)
      echo "Unsupported WL iteration count: ${wl_iters}; expected 2 or 3." >&2
      exit 2
      ;;
  esac
  wl_batch_size="$(batch_size_for_wl "${wl_iters}")"
  echo "GuacaMol WL${wl_iters}: batch_size=${wl_batch_size}"
  env \
    "PYBIN=${PYBIN}" \
    "SEED=${SEED}" \
    "MODE=full" \
    "EXP_ROOT=${ROOT}/wl${wl_iters}" \
    "TRACE_ROOT=${TRACE_ROOT}/wl${wl_iters}" \
    "RUN_QM9_NO_H=0" \
    "RUN_QM9_H=0" \
    "RUN_ZINC=0" \
    "RUN_MOSES=0" \
    "RUN_GUACAMOL=1" \
    "RUN_RETRAIN=1" \
    "RUN_RETRAIN_ZINC=0" \
    "SKIP_EXISTING_TRAINING=1" \
    "N_WL_ITERS=${wl_iters}" \
    "EPOCHS=${EPOCHS}" \
    "BATCH_SIZE=${wl_batch_size}" \
    "N_SAMPLES=${N_SAMPLES}" \
    "EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}" \
    "EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS}" \
    "EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA}" \
    "REFERENCE_SPLIT=${REFERENCE_SPLIT}" \
    "MOL_PREPROCESS=${MOL_PREPROCESS}" \
    "FCD_REFERENCE_SPLIT=${FCD_REFERENCE_SPLIT}" \
    "FCD_COMMAND=${FCD_COMMAND}" \
    "FCD_TIMEOUT=${FCD_TIMEOUT}" \
    "MAX_RESTARTS=${MAX_RESTARTS}" \
    "EVAL_COMPLETION_MODES=${EVAL_COMPLETION_MODES}" \
    "AUTO_DOWNLOAD_BENCHMARK_DATA=${AUTO_DOWNLOAD_BENCHMARK_DATA}" \
    "INSTALL_BENCHMARK_PACKAGES=${INSTALL_BENCHMARK_PACKAGES}" \
    "BENCHMARK_DATA_DIR=${BENCHMARK_DATA_DIR}" \
    "BENCHMARK_METRIC_SAMPLES=${BENCHMARK_METRIC_SAMPLES}" \
    "RUN_MOSES_METRICS=0" \
    "RUN_GUACAMOL_METRICS=1" \
    "MOSES_BACKEND=${MOSES_BACKEND}" \
    ./scripts/run_moses_guacamol_experiments.sh
done

"${PYBIN}" src/benchmark_report.py --root "${ROOT}" --out "${ROOT}/summary"

echo
echo "Corrected GuacaMol WL2/WL3 sweep finished."
echo "Epochs per model: ${EPOCHS}"
echo "Root: ${ROOT}"
echo "Summary: ${ROOT}/summary/benchmark_summary.md"
