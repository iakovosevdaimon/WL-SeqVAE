#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CATFLOW_SEEDS=${CATFLOW_SEEDS:-"0 1 2"}
CATFLOW_WL_ITERS=${CATFLOW_WL_ITERS:-"1 2 3"}
CATFLOW_RUN_MAIN=${CATFLOW_RUN_MAIN:-1}
CATFLOW_RUN_EXTRA=${CATFLOW_RUN_EXTRA:-0}
CATFLOW_RUN_PROTEINS=${CATFLOW_RUN_PROTEINS:-1}
CATFLOW_ROOT=${CATFLOW_ROOT:-"benchmark_runs/catflow_declared_multiseed"}
SPECTRE_DATA_ROOT=${SPECTRE_DATA_ROOT:-benchmark_data/official_spectre}
ORCA=${ORCA:-third_party/orca/orca}
CATFLOW_MAIN_GRAPH_DATASETS="EGO_SMALL COMMUNITY_SMALL ENZYMES GRID"

if [[ -z "${CATFLOW_SEEDS//[[:space:]]/}" ]]; then
  echo "CATFLOW_SEEDS must contain at least one integer seed." >&2
  exit 2
fi
if [[ -z "${CATFLOW_WL_ITERS//[[:space:]]/}" ]]; then
  echo "CATFLOW_WL_ITERS must contain at least one positive integer." >&2
  exit 2
fi
if [[ "${CATFLOW_RUN_MAIN}" != "1" && "${CATFLOW_RUN_EXTRA}" != "1" && \
      "${CATFLOW_RUN_PROTEINS}" != "1" ]]; then
  echo "At least one benchmark group must be enabled." >&2
  exit 2
fi

mkdir -p "${CATFLOW_ROOT}"

if [[ "${CATFLOW_RUN_PROTEINS}" == "1" ]]; then
  if [[ ! -x "${ORCA}" ]]; then
    echo "Exact ORCA executable is missing or not executable: ${ORCA}" >&2
    exit 2
  fi
  "${PYBIN}" src/prepare_official_spectre_data.py \
    --output-dir "${SPECTRE_DATA_ROOT}" \
    --datasets PROTEINS
fi

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

    if [[ "${CATFLOW_RUN_MAIN}" == "1" ]]; then
      echo "CatFlow main-table WL${wl_iters} seed ${seed}"
      SEED="${seed}" \
      N_WL_ITERS="${wl_iters}" \
      BENCHMARK_SUITE=catflow_2406_04843 \
      BENCH_DATASETS="${CATFLOW_MAIN_GRAPH_DATASETS}" \
      BENCH_ROOT="${CATFLOW_ROOT}/main/wl${wl_iters}/seed${seed}" \
        bash scripts/run_all_benchmarks.sh
    fi

    if [[ "${CATFLOW_RUN_EXTRA}" == "1" ]]; then
      echo "CatFlow appendix Planar/SBM WL${wl_iters} seed ${seed}"
      SEED="${seed}" \
      N_WL_ITERS="${wl_iters}" \
      BENCHMARK_SUITE=catflow_extra_2406_04843 \
      BENCH_ROOT="${CATFLOW_ROOT}/extra/wl${wl_iters}/seed${seed}" \
        bash scripts/run_all_benchmarks.sh
    fi

    if [[ "${CATFLOW_RUN_PROTEINS}" == "1" ]]; then
      echo "Official SPECTRE PROTEINS WL${wl_iters} model seed ${seed}"
      PYBIN="${PYBIN}" \
      SEED="${seed}" \
      MODEL_NAME=budget_ring_wl_vae_official \
      N_WL_ITERS="${wl_iters}" \
      BENCHMARK_SUITE=spectre_2204_01613 \
      BENCH_DATASETS=PROTEINS \
      BENCH_ROOT="${CATFLOW_ROOT}/proteins_spectre/wl${wl_iters}/seed${seed}" \
      TRACE_ROOT="trace_cache/proteins_spectre/wl${wl_iters}/seed${seed}" \
      PAPER_DATA_ROOT="${SPECTRE_DATA_ROOT}" \
      PAPER_SPLIT_PROTOCOL=spectre_seed1234 \
      VERIFY_DATA_PROVENANCE=1 \
      N_SAMPLES=paper \
      REFERENCE_SPLIT=test \
      EARLY_STOPPING_PATIENCE=0 \
      MAX_RESTARTS=1 \
      BATCH_SIZE=auto \
      BATCH_SIZE_PROTEINS=32 \
      BATCH_SIZE_PROTEINS_WL3=16 \
      MAX_SEQ_LEN=auto \
      EVAL_COMPLETION_MODES=allow_incomplete \
      COMPUTE_MMD_RATIO=1 \
      RUN_ABLATIONS=0 \
      REUSE_PAPER_ZINC_CKPT=0 \
      FCD_COMMAND= \
        bash scripts/run_all_benchmarks.sh
    fi
  done
done

if [[ "${CATFLOW_RUN_PROTEINS}" == "1" ]]; then
  "${PYBIN}" src/recompute_digress_spectre_metrics.py \
    --run-root "${CATFLOW_ROOT}/proteins_spectre" \
    --output-dir "${CATFLOW_ROOT}/proteins_spectre/digress_spectre_compatible" \
    --markdown "${CATFLOW_ROOT}/proteins_spectre/official_spectre_metrics.md" \
    --orca "${ORCA}"
fi

completed_datasets=""
if [[ "${CATFLOW_RUN_MAIN}" == "1" ]]; then
  completed_datasets="${CATFLOW_MAIN_GRAPH_DATASETS}"
fi
if [[ "${CATFLOW_RUN_EXTRA}" == "1" ]]; then
  completed_datasets="${completed_datasets:+${completed_datasets} }PLANAR SBM"
fi
if [[ "${CATFLOW_RUN_PROTEINS}" == "1" ]]; then
  completed_datasets="${completed_datasets:+${completed_datasets} }PROTEINS"
fi

cat > "${CATFLOW_ROOT}/aggregation_required.txt" <<EOF
Completed declared CatFlow seeds: ${CATFLOW_SEEDS}
Completed WL iteration counts: ${CATFLOW_WL_ITERS}
Datasets: ${completed_datasets}

Before publication, aggregate like-for-like checkpoint roles and WL settings
with arithmetic mean and population standard deviation. Retain the individual
seed reports and state n explicitly. CatFlow does not publish its exact seed
list or aggregation count, so do not claim exact error-bar reproduction.

PROTEINS uses the official filtered DD collection, fixed Torch seed-1234
SPECTRE split, test reference, and 184 generated samples. CATFLOW_SEEDS controls
model-training seeds only; it does not alter the fixed PROTEINS data split.
EOF
