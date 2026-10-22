#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-50}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}
EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS:-15}
EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA:-0.001}
MAX_RESTARTS=${MAX_RESTARTS:-1}
BATCH_SIZE_PROTEINS=${BATCH_SIZE_PROTEINS:-32}
BATCH_SIZE_PROTEINS_WL3=${BATCH_SIZE_PROTEINS_WL3:-16}
SYNTHETIC_SPLIT_PROTOCOL=${SYNTHETIC_SPLIT_PROTOCOL:-digress_seed0}
ROOT=${ROOT:-benchmark_runs/official_${SYNTHETIC_SPLIT_PROTOCOL}_graphs_wl1_wl2_wl3_seed${SEED}_e${EPOCHS}}
TRACE_ROOT=${TRACE_ROOT:-trace_cache/official_${SYNTHETIC_SPLIT_PROTOCOL}_graphs_wl1_wl2_wl3_seed${SEED}_e${EPOCHS}}
SPECTRE_DATA_ROOT=${SPECTRE_DATA_ROOT:-benchmark_data/official_spectre}
PROTEINS_DATA_ROOT=${PROTEINS_DATA_ROOT:-${SPECTRE_DATA_ROOT}}
EXTENDED_DATA_ROOT=${EXTENDED_DATA_ROOT:-benchmark_data/all_corrected/paper_graphs}
CORE_GRAPH_DATASETS=${CORE_GRAPH_DATASETS-"COMMUNITY_SMALL PLANAR SBM"}
EXTENDED_GRAPH_DATASETS=${EXTENDED_GRAPH_DATASETS-"EGO_SMALL ENZYMES GRID"}
PROTEINS_DATASETS=${PROTEINS_DATASETS-"PROTEINS"}
WL_ITERS=${WL_ITERS:-"1 2 3"}
ORCA=${ORCA:-third_party/orca/orca}
EVAL_COMPLETION_MODES=${EVAL_COMPLETION_MODES:-allow_incomplete}

case "${SYNTHETIC_SPLIT_PROTOCOL}" in
  digress_seed0|spectre_seed1234) ;;
  *) echo "Unsupported SYNTHETIC_SPLIT_PROTOCOL=${SYNTHETIC_SPLIT_PROTOCOL}" >&2; exit 2 ;;
esac

if [[ -n "${CORE_GRAPH_DATASETS}" && -n "${PROTEINS_DATASETS}" && \
      "$(realpath -m "${PROTEINS_DATA_ROOT}")" == "$(realpath -m "${SPECTRE_DATA_ROOT}")" ]]; then
  "${PYBIN}" src/prepare_official_spectre_data.py \
    --output-dir "${SPECTRE_DATA_ROOT}" \
    --datasets ${CORE_GRAPH_DATASETS} ${PROTEINS_DATASETS}
elif [[ -n "${CORE_GRAPH_DATASETS}" && -n "${PROTEINS_DATASETS}" ]]; then
  "${PYBIN}" src/prepare_official_spectre_data.py \
    --output-dir "${SPECTRE_DATA_ROOT}" \
    --datasets ${CORE_GRAPH_DATASETS}
  "${PYBIN}" src/prepare_official_spectre_data.py \
    --output-dir "${PROTEINS_DATA_ROOT}" \
    --datasets ${PROTEINS_DATASETS}
elif [[ -n "${CORE_GRAPH_DATASETS}" ]]; then
  "${PYBIN}" src/prepare_official_spectre_data.py \
    --output-dir "${SPECTRE_DATA_ROOT}" \
    --datasets ${CORE_GRAPH_DATASETS}
elif [[ -n "${PROTEINS_DATASETS}" ]]; then
  "${PYBIN}" src/prepare_official_spectre_data.py \
    --output-dir "${PROTEINS_DATA_ROOT}" \
    --datasets ${PROTEINS_DATASETS}
fi
"${PYBIN}" - <<'PY'
import warnings
warnings.filterwarnings("ignore", message="Error importing cairo.*")
warnings.filterwarnings("ignore", message="Error importing draw module.*")
try:
    import graph_tool.all  # noqa: F401
    import pyemd  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "Exact SBM V.U.N. and Community MMD require graph-tool and pyemd in PYBIN. "
        "Run scripts/setup_exact_graph_tool_env.sh first. Original error: " + repr(exc)
    )
print("Exact graph-tool SBM and pyemd Community backends are available.")
PY

if [[ ! -x "${ORCA}" ]]; then
  echo "Exact ORCA executable is missing or not executable: ${ORCA}" >&2
  exit 2
fi
echo "ORCA sha256: $(sha256sum "${ORCA}" | awk '{print $1}')"

if [[ -n "${EXTENDED_GRAPH_DATASETS}" ]]; then
  "${PYBIN}" - "${EXTENDED_DATA_ROOT}" "${SYNTHETIC_SPLIT_PROTOCOL}" ${EXTENDED_GRAPH_DATASETS} <<'PY'
import json
import sys

from gpt_datasets import load_graph_dataset_bundle


data_root = sys.argv[1]
protocol = sys.argv[2]
datasets = sys.argv[3:]
for dataset in datasets:
    bundle = load_graph_dataset_bundle(
        dataset,
        data_root=data_root,
        seed=0,
        paper_split_protocol=protocol,
    )
    provenance = bundle.provenance
    if provenance.get("official_digress_split") is not False:
        raise SystemExit(
            f"{dataset}: extended split must be explicitly marked non-official"
        )
    if protocol == "digress_seed0" and provenance.get("split_seed") != 0:
        raise SystemExit(f"{dataset}: expected Torch seed-0 split provenance")
    print(json.dumps({
        "dataset": dataset,
        "asset_file": provenance.get("asset_file"),
        "asset_sha256": provenance.get("asset_sha256"),
        "split_sizes": provenance.get("split_sizes"),
        "split_seed": provenance.get("split_seed"),
        "official_digress_split": provenance.get("official_digress_split"),
        "comparability_scope": provenance.get("comparability_scope"),
    }, sort_keys=True))
print("Pinned extended graph assets and split provenance are valid.")
PY
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Graph benchmark preflight passed; PREFLIGHT_ONLY=1, so no training was started."
  exit 0
fi

mkdir -p "${ROOT}" "${TRACE_ROOT}"

run_group() {
  local wl="$1"
  local group="$2"
  local datasets="$3"
  local data_root="$4"
  local split_protocol="$5"
  local sbm_backend="$6"
  local benchmark_suite="$7"

  env \
    "PYBIN=${PYBIN}" \
    "SEED=${SEED}" \
    "MODEL_NAME=budget_ring_wl_vae_official" \
    "BENCHMARK_SUITE=${benchmark_suite}" \
    "BENCH_ROOT=${ROOT}/wl${wl}/${group}" \
    "TRACE_ROOT=${TRACE_ROOT}/wl${wl}/${group}" \
    "PAPER_DATA_ROOT=${data_root}" \
    "PAPER_SPLIT_PROTOCOL=${split_protocol}" \
    "VERIFY_DATA_PROVENANCE=1" \
    "SBM_VALIDITY_BACKEND=${sbm_backend}" \
    "SBM_REFINEMENT_STEPS=100" \
    "SBM_VUN_REFINEMENT_STEPS=1000" \
    "BENCH_DATASETS=${datasets}" \
    "N_WL_ITERS=${wl}" \
    "N_SAMPLES=paper" \
    "REFERENCE_SPLIT=test" \
    "EPOCHS=${EPOCHS}" \
    "EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}" \
    "EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS}" \
    "EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA}" \
    "MAX_RESTARTS=${MAX_RESTARTS}" \
    "BATCH_SIZE_PROTEINS=${BATCH_SIZE_PROTEINS}" \
    "BATCH_SIZE_PROTEINS_WL3=${BATCH_SIZE_PROTEINS_WL3}" \
    "BATCH_SIZE=auto" \
    "MAX_SEQ_LEN=auto" \
    "EVAL_COMPLETION_MODES=${EVAL_COMPLETION_MODES}" \
    "COMPUTE_MMD_RATIO=1" \
    "RUN_ABLATIONS=0" \
    "REUSE_PAPER_ZINC_CKPT=0" \
    "FCD_COMMAND=" \
    ./scripts/run_all_benchmarks.sh
}

for wl in ${WL_ITERS}; do
  case "${wl}" in
    1|2|3) ;;
    *) echo "Unsupported WL iteration ${wl}; expected 1, 2, or 3." >&2; exit 2 ;;
  esac
  if [[ -n "${CORE_GRAPH_DATASETS}" ]]; then
    run_group "${wl}" synthetic "${CORE_GRAPH_DATASETS}" "${SPECTRE_DATA_ROOT}" "${SYNTHETIC_SPLIT_PROTOCOL}" exact spectre_2204_01613
  fi
  if [[ -n "${EXTENDED_GRAPH_DATASETS}" ]]; then
    run_group "${wl}" digress_style "${EXTENDED_GRAPH_DATASETS}" "${EXTENDED_DATA_ROOT}" "${SYNTHETIC_SPLIT_PROTOCOL}" auto digress_style_extended
  fi
  if [[ -n "${PROTEINS_DATASETS}" ]]; then
    run_group "${wl}" spectre "${PROTEINS_DATASETS}" "${PROTEINS_DATA_ROOT}" spectre_seed1234 auto spectre_2204_01613
  fi
done

"${PYBIN}" src/benchmark_report.py --root "${ROOT}" --out "${ROOT}/summary"
"${PYBIN}" src/recompute_digress_spectre_metrics.py \
  --run-root "${ROOT}" \
  --output-dir "${ROOT}/digress_spectre_compatible" \
  --markdown "${ROOT}/official_digress_spectre_metrics.md" \
  --orca "${ORCA}"
"${PYBIN}" src/generate_current_best_results_tex.py \
  --metrics "${ROOT}/digress_spectre_compatible/metrics.json" \
  --ledger "${ROOT}/official_digress_spectre_metrics.md" \
  --output "${ROOT}/official_best_benchmark_results.tex"

echo
echo "SPECTRE core plus DiGress-style extended graph rerun finished."
echo "Root: ${ROOT}"
echo "Summary: ${ROOT}/summary/benchmark_summary.md"
echo "Exact-compatible metrics: ${ROOT}/official_digress_spectre_metrics.md"
echo "LaTeX table report: ${ROOT}/official_best_benchmark_results.tex"
