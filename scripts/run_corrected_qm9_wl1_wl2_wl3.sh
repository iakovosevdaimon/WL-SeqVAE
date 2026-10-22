#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
SEED=${SEED:-0}
ROOT=${ROOT:-benchmark_runs/corrected_qm9_wl1_wl2_wl3_seed${SEED}}
TRACE_ROOT=${TRACE_ROOT:-trace_cache/corrected_qm9_wl1_wl2_wl3_seed${SEED}}
DATA_ROOT=${DATA_ROOT:-benchmark_data/all_corrected}
QM9_DATA_ROOT=${QM9_DATA_ROOT:-${DATA_ROOT}/QM9}
QM9_DATASETS=${QM9_DATASETS:-"QM9 QM9_H"}
WL_ITERS=${WL_ITERS:-"1 2 3"}

EPOCHS=${EPOCHS:-50}
# "auto" selects QM9/QM9_H- and WL-specific batches.
BATCH_SIZE=${BATCH_SIZE:-auto}
N_SAMPLES=${N_SAMPLES:-10000}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}
EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS:-15}
EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA:-0.001}
REFERENCE_SPLIT=${REFERENCE_SPLIT:-test}
MOL_PREPROCESS=${MOL_PREPROCESS:-catflow}
MOL_PREPROCESS_QM9=${MOL_PREPROCESS_QM9:-catflow}
MOL_PREPROCESS_QM9_H=${MOL_PREPROCESS_QM9_H:-repo}
QM9_PROTOCOL=${QM9_PROTOCOL:-gdss_catflow}
QM9_H_PROTOCOL=${QM9_H_PROTOCOL:-legacy_local}
FCD_REFERENCE_SPLIT=${FCD_REFERENCE_SPLIT:-test}
FCD_DEVICE=${FCD_DEVICE:-cuda}
FCD_COMMAND=${FCD_COMMAND:-"${PYBIN} src/compute_fcd.py --ref {ref} --gen {gen} --out {out} --device ${FCD_DEVICE}"}
FCD_TIMEOUT=${FCD_TIMEOUT:-1200}
MAX_RESTARTS=${MAX_RESTARTS:-1}
EVAL_COMPLETION_MODES=${EVAL_COMPLETION_MODES:-"strict allow_incomplete"}
METRIC_MAX_TRAIN_GRAPHS=${METRIC_MAX_TRAIN_GRAPHS:-20000}
DIAGNOSTIC_MAX_TRAIN_GRAPHS=${DIAGNOSTIC_MAX_TRAIN_GRAPHS:-20000}
MMD_MAX_GRAPHS=${MMD_MAX_GRAPHS:-2000}
PREPARE_QM9=${PREPARE_QM9:-1}

mkdir -p "${ROOT}" "${TRACE_ROOT}" "${QM9_DATA_ROOT}"

download_verified() {
  local url="$1"
  local destination="$2"
  local expected_sha256="$3"
  local actual_sha256=""
  if [[ -f "${destination}" ]]; then
    actual_sha256="$(sha256sum "${destination}" | awk '{print $1}')"
    if [[ "${actual_sha256}" == "${expected_sha256}" ]]; then
      return 0
    fi
    echo "Refusing to overwrite checksum-mismatched QM9 asset: ${destination}" >&2
    return 2
  fi
  mkdir -p "$(dirname "${destination}")"
  local temporary
  temporary="$(mktemp "${destination}.tmp.XXXXXX")"
  if ! curl --fail --location --retry 3 --output "${temporary}" "${url}"; then
    rm -f "${temporary}"
    return 1
  fi
  actual_sha256="$(sha256sum "${temporary}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "Checksum mismatch for ${url}: received ${actual_sha256}, expected ${expected_sha256}" >&2
    rm -f "${temporary}"
    return 2
  fi
  mv "${temporary}" "${destination}"
}

if [[ "${PREPARE_QM9}" == "1" && " ${QM9_DATASETS} " == *" QM9 "* ]]; then
  GDSS_REVISION=24cc490e0c5b39cbd265fda33ab2bc8b528d4454
  download_verified \
    "https://raw.githubusercontent.com/harryjo97/GDSS/${GDSS_REVISION}/data/qm9.csv" \
    "${QM9_DATA_ROOT}/raw/gdss_qm9.csv" \
    "b7d055641f34c2803d0461806af46d263ebc6dec879bf0766657b1d3efd7ff8f"
  download_verified \
    "https://raw.githubusercontent.com/harryjo97/GDSS/${GDSS_REVISION}/data/valid_idx_qm9.json" \
    "${QM9_DATA_ROOT}/raw/valid_idx_qm9.json" \
    "889fd232a33d8a2c00f057c807b119e72abd5baa48f6fab93e8d8b7a79340f4d"
fi

if [[ "${PREPARE_QM9}" == "1" && " ${QM9_DATASETS} " == *" QM9_H "* && ! -s "${QM9_DATA_ROOT}/raw/gdb9.sdf" ]]; then
  "${PYBIN}" - "${QM9_DATA_ROOT}" <<'PY'
import sys
from pathlib import Path

from torch_geometric.datasets import QM9

root = Path(sys.argv[1]).resolve()
dataset = QM9(root=str(root))
raw_sdf = root / "raw" / "gdb9.sdf"
if not raw_sdf.is_file() or raw_sdf.stat().st_size == 0:
    raise SystemExit(f"QM9 preparation did not create {raw_sdf}")
print(f"Prepared QM9: {len(dataset)} PyG records; raw SDF={raw_sdf}")
PY
fi

for wl_iters in ${WL_ITERS}; do
  case "${wl_iters}" in
    1|2|3) ;;
    *)
      echo "Unsupported WL iteration count: ${wl_iters}; expected one of 1, 2, 3." >&2
      exit 2
      ;;
  esac
  env \
    "PYBIN=${PYBIN}" \
    "SEED=${SEED}" \
    "MODEL_NAME=budget_ring_wl_vae_corrected" \
    "BENCHMARK_SUITE=all_papers" \
    "BENCH_ROOT=${ROOT}/wl${wl_iters}" \
    "TRACE_ROOT=${TRACE_ROOT}/wl${wl_iters}" \
    "QM9_DATA_ROOT=${QM9_DATA_ROOT}" \
    "QM9_FORCE_RAW_FALLBACK=1" \
    "BENCH_DATASETS=${QM9_DATASETS}" \
    "N_WL_ITERS=${wl_iters}" \
    "N_SAMPLES=${N_SAMPLES}" \
    "REFERENCE_SPLIT=${REFERENCE_SPLIT}" \
    "MOL_PREPROCESS=${MOL_PREPROCESS}" \
    "MOL_PREPROCESS_QM9=${MOL_PREPROCESS_QM9}" \
    "MOL_PREPROCESS_QM9_H=${MOL_PREPROCESS_QM9_H}" \
    "QM9_PROTOCOL=${QM9_PROTOCOL}" \
    "QM9_H_PROTOCOL=${QM9_H_PROTOCOL}" \
    "FCD_REFERENCE_SPLIT=${FCD_REFERENCE_SPLIT}" \
    "FCD_COMMAND=${FCD_COMMAND}" \
    "FCD_TIMEOUT=${FCD_TIMEOUT}" \
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
    ./scripts/run_all_benchmarks.sh
done

"${PYBIN}" src/benchmark_report.py --root "${ROOT}" --out "${ROOT}/summary"

echo
echo "Corrected QM9/QM9_H WL1/WL2/WL3 sweep finished."
echo "QM9 policy: GDSS/CatFlow heavy-atom data, max 9 nodes, 4 atom types, CatFlow canonicalization."
echo "QM9_H policy: explicit-H generation/primary uniqueness plus automatic H-stripped secondary metrics."
echo "Root: ${ROOT}"
echo "Summary: ${ROOT}/summary/benchmark_summary.md"
