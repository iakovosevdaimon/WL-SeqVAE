#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
SEED=${SEED:-0}
ROOT=${ROOT:-benchmark_runs/corrected_zinc250k_wl2_wl3_seed${SEED}}
TRACE_ROOT=${TRACE_ROOT:-trace_cache/corrected_zinc250k_wl2_wl3_seed${SEED}}
DATA_ROOT=${DATA_ROOT:-benchmark_data/all_corrected}
ZINC_DATA_ROOT=${ZINC_DATA_ROOT:-${DATA_ROOT}/ZINC250K}
WL_ITERS=${WL_ITERS:-"2 3"}

EPOCHS=${EPOCHS:-50}
# "auto" selects batch 64 for WL2 and batch 32 for WL3.
BATCH_SIZE=${BATCH_SIZE:-auto}
N_SAMPLES=${N_SAMPLES:-10000}
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
METRIC_MAX_TRAIN_GRAPHS=${METRIC_MAX_TRAIN_GRAPHS:-20000}
DIAGNOSTIC_MAX_TRAIN_GRAPHS=${DIAGNOSTIC_MAX_TRAIN_GRAPHS:-20000}
MMD_MAX_GRAPHS=${MMD_MAX_GRAPHS:-2000}
PREPARE_ZINC=${PREPARE_ZINC:-1}

mkdir -p "${ROOT}" "${TRACE_ROOT}" "${ZINC_DATA_ROOT}"

zinc_ready=1
for required_path in \
  "${ZINC_DATA_ROOT}/full/processed/train.pt" \
  "${ZINC_DATA_ROOT}/full/processed/val.pt" \
  "${ZINC_DATA_ROOT}/full/processed/test.pt" \
  "${ZINC_DATA_ROOT}/raw/atom_dict.pickle"
do
  if [[ ! -f "${required_path}" ]]; then
    zinc_ready=0
  fi
done

if [[ "${zinc_ready}" == "0" ]]; then
  if [[ "${PREPARE_ZINC}" != "1" ]]; then
    echo "ZINC250K data is incomplete under ${ZINC_DATA_ROOT}; set PREPARE_ZINC=1 or provide ZINC_DATA_ROOT." >&2
    exit 2
  fi
  "${PYBIN}" - "${ZINC_DATA_ROOT}" <<'PY'
import sys
from pathlib import Path

from torch_geometric.datasets import ZINC

root = Path(sys.argv[1]).resolve()
for split in ("train", "val", "test"):
    dataset = ZINC(root=str(root), subset=False, split=split)
    print(f"Prepared ZINC250K {split}: {len(dataset)} graphs")
required = [
    root / "full" / "processed" / f"{split}.pt"
    for split in ("train", "val", "test")
] + [root / "raw" / "atom_dict.pickle"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("ZINC preparation did not create: " + ", ".join(missing))
PY
fi

for wl_iters in ${WL_ITERS}; do
  case "${wl_iters}" in
    2|3) ;;
    *)
      echo "Unsupported WL iteration count: ${wl_iters}; expected 2 or 3." >&2
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
    "ZINC_DATA_ROOT=${ZINC_DATA_ROOT}" \
    "BENCH_DATASETS=ZINC250K" \
    "N_WL_ITERS=${wl_iters}" \
    "N_SAMPLES=${N_SAMPLES}" \
    "REFERENCE_SPLIT=${REFERENCE_SPLIT}" \
    "MOL_PREPROCESS=${MOL_PREPROCESS}" \
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
echo "Corrected ZINC250K WL2/WL3 sweep finished."
echo "Root: ${ROOT}"
echo "Summary: ${ROOT}/summary/benchmark_summary.md"
