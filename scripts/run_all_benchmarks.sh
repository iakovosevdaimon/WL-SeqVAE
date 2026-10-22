#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
SEED=${SEED:-0}
N_WL_ITERS=${N_WL_ITERS:-3}
TRACE_ROOT=${TRACE_ROOT:-trace_cache}

# Main paper model.  Other model variants are ablations and are opt-in.
MODEL_NAME=${MODEL_NAME:-budget_ring_wl_vae}
BENCHMARK_SUITE=${BENCHMARK_SUITE:-local}
if [[ "${BENCHMARK_SUITE}" == "local" ]]; then
  DEFAULT_BENCH_ROOT="benchmark_runs/${MODEL_NAME}_seed${SEED}"
else
  DEFAULT_BENCH_ROOT="benchmark_runs/${MODEL_NAME}_${BENCHMARK_SUITE}_seed${SEED}"
fi
BENCH_ROOT=${BENCH_ROOT:-${DEFAULT_BENCH_ROOT}}
if [[ -z "${BENCH_DATASETS+x}" ]]; then
  case "${BENCHMARK_SUITE}" in
    local)
      BENCH_DATASETS="ZINC250K"
      ;;
    catflow_2406_04843)
      BENCH_DATASETS="EGO_SMALL COMMUNITY_SMALL ENZYMES GRID"
      ;;
    catflow_extra_2406_04843)
      BENCH_DATASETS="SBM PLANAR"
      ;;
    spectre_2204_01613)
      BENCH_DATASETS="COMMUNITY_SMALL QM9 PLANAR SBM PROTEINS"
      ;;
    digress_style_extended)
      BENCH_DATASETS="EGO_SMALL ENZYMES GRID"
      ;;
    papers|all_papers)
      BENCH_DATASETS="EGO_SMALL COMMUNITY_SMALL ENZYMES GRID QM9 ZINC250K PLANAR SBM PROTEINS"
      ;;
    *)
      echo "Unknown BENCHMARK_SUITE=${BENCHMARK_SUITE}" >&2
      exit 2
      ;;
  esac
fi
RUN_ABLATIONS=${RUN_ABLATIONS:-0}
ABLATION_VARIANTS=${ABLATION_VARIANTS:-"base action latent_plan"}
REUSE_PAPER_ZINC_CKPT=${REUSE_PAPER_ZINC_CKPT:-1}

if [[ -z "${N_SAMPLES+x}" ]]; then
  if [[ "${BENCHMARK_SUITE}" == "local" ]]; then
    N_SAMPLES=10000
  else
    N_SAMPLES=paper
  fi
fi
if [[ -z "${REFERENCE_SPLIT+x}" ]]; then
  if [[ "${BENCHMARK_SUITE}" == "local" ]]; then
    REFERENCE_SPLIT=train
  else
    REFERENCE_SPLIT=test
  fi
fi
if [[ -z "${MOL_PREPROCESS+x}" ]]; then
  if [[ "${BENCHMARK_SUITE}" == "catflow_2406_04843" ]]; then
    MOL_PREPROCESS=catflow
  else
    MOL_PREPROCESS=repo
  fi
fi
MOL_PREPROCESS_QM9=${MOL_PREPROCESS_QM9:-${MOL_PREPROCESS}}
MOL_PREPROCESS_QM9_H=${MOL_PREPROCESS_QM9_H:-repo}
if [[ -z "${QM9_PROTOCOL+x}" ]]; then
  if [[ "${BENCHMARK_SUITE}" == "catflow_2406_04843" ]]; then
    QM9_PROTOCOL=gdss_catflow
  else
    QM9_PROTOCOL=legacy_local
  fi
fi
QM9_H_PROTOCOL=${QM9_H_PROTOCOL:-legacy_local}
if [[ -z "${ZINC250K_PROTOCOL+x}" ]]; then
  if [[ "${BENCHMARK_SUITE}" == "catflow_2406_04843" ]]; then
    ZINC250K_PROTOCOL=gdss_catflow
  else
    ZINC250K_PROTOCOL=legacy_pyg
  fi
fi
FCD_REFERENCE_SPLIT=${FCD_REFERENCE_SPLIT:-${REFERENCE_SPLIT}}
FCD_DEDUPLICATE_REFERENCE=${FCD_DEDUPLICATE_REFERENCE:-0}
if [[ -z "${FCD_DEDUPLICATE_GENERATED+x}" ]]; then
  if [[ "${BENCHMARK_SUITE}" == "catflow_2406_04843" ]]; then
    FCD_DEDUPLICATE_GENERATED=1
  else
    FCD_DEDUPLICATE_GENERATED=0
  fi
fi
EPOCHS=${EPOCHS:-50}
if [[ ! "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer, received ${EPOCHS@Q}." >&2
  exit 2
fi
if (( EPOCHS > 50 )); then
  echo "Capping requested EPOCHS=${EPOCHS} to the benchmark maximum of 50."
  EPOCHS=50
fi
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-10}
EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS:-15}
EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA:-0.001}
BATCH_SIZE=${BATCH_SIZE:-32}
BATCH_SIZE_PROTEINS=${BATCH_SIZE_PROTEINS:-32}
BATCH_SIZE_PROTEINS_WL3=${BATCH_SIZE_PROTEINS_WL3:-16}
HIDDEN_DIM=${HIDDEN_DIM:-192}
LATENT_DIM=${LATENT_DIM:-64}
N_LAYERS=${N_LAYERS:-4}
N_HEADS=${N_HEADS:-4}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-auto}
LR=${LR:-0.0003}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
DROPOUT=${DROPOUT:-0.1}
KL_BETA=${KL_BETA:-0.1}
KL_WARMUP_EPOCHS=${KL_WARMUP_EPOCHS:-10}
GRAD_CLIP=${GRAD_CLIP:-1.0}
DECODER_COLOR_MIN_FREQUENCY=${DECODER_COLOR_MIN_FREQUENCY:-5}
DECODER_RARE_FRACTION_THRESHOLD=${DECODER_RARE_FRACTION_THRESHOLD:-0.0}
STRUCTURAL_VALIDATION_INTERVAL=${STRUCTURAL_VALIDATION_INTERVAL:-5}
STRUCTURAL_VALIDATION_SAMPLES=${STRUCTURAL_VALIDATION_SAMPLES:-8}

NODE_BUDGET_FACTOR=${NODE_BUDGET_FACTOR:-1.0}
NODE_BUDGET_MARGIN=${NODE_BUDGET_MARGIN:-0}
METRIC_MAX_TRAIN_GRAPHS=${METRIC_MAX_TRAIN_GRAPHS:-20000}
DIAGNOSTIC_MAX_TRAIN_GRAPHS=${DIAGNOSTIC_MAX_TRAIN_GRAPHS:-20000}
MMD_MAX_GRAPHS=${MMD_MAX_GRAPHS:-2000}
if [[ -z "${COMPUTE_MMD_RATIO+x}" ]]; then
  if [[ "${BENCHMARK_SUITE}" == "catflow_2406_04843" ]]; then
    COMPUTE_MMD_RATIO=0
  else
    COMPUTE_MMD_RATIO=1
  fi
fi
MAX_RESTARTS=${MAX_RESTARTS:-200}
if [[ -z "${EVAL_COMPLETION_MODES+x}" ]]; then
  if [[ "${BENCHMARK_SUITE}" == "catflow_2406_04843" || "${BENCHMARK_SUITE}" == "catflow_extra_2406_04843" ]]; then
    EVAL_COMPLETION_MODES=allow_incomplete
  else
    EVAL_COMPLETION_MODES=strict
  fi
fi
# External FCD/ChemNet command. It may use {ref}, {gen}, and {out}
# placeholders, or accept reference/generated SMILES files as final args.
# CatFlow suites default to the repository wrapper; other suites keep FCD opt-in.
if [[ -z "${FCD_COMMAND+x}" ]]; then
  if [[ "${BENCHMARK_SUITE}" == "catflow_2406_04843" ]]; then
    FCD_COMMAND="${PYBIN} src/compute_fcd.py --ref {ref} --gen {gen} --out {out} --device cpu --n-jobs 8 --batch-size 512 --preprocess none"
  else
    FCD_COMMAND=
  fi
fi
FCD_TIMEOUT=${FCD_TIMEOUT:-1200}
ZINC_DATA_ROOT=${ZINC_DATA_ROOT:-}
TU_DATA_ROOT=${TU_DATA_ROOT:-}
PAPER_DATA_ROOT=${PAPER_DATA_ROOT:-}
QM9_DATA_ROOT=${QM9_DATA_ROOT:-}
if [[ -z "${PAPER_SPLIT_PROTOCOL+x}" ]]; then
  case "${BENCHMARK_SUITE}" in
    catflow_2406_04843) PAPER_SPLIT_PROTOCOL=catflow_gdss ;;
    catflow_extra_2406_04843) PAPER_SPLIT_PROTOCOL=digress_seed0 ;;
    *) PAPER_SPLIT_PROTOCOL=legacy_local ;;
  esac
fi
if [[ -z "${VERIFY_DATA_PROVENANCE+x}" ]]; then
  case "${BENCHMARK_SUITE}" in
    catflow_2406_04843|catflow_extra_2406_04843) VERIFY_DATA_PROVENANCE=1 ;;
    *) VERIFY_DATA_PROVENANCE=0 ;;
  esac
fi
if [[ -z "${SBM_VALIDITY_BACKEND+x}" ]]; then
  if [[ "${BENCHMARK_SUITE}" == "catflow_extra_2406_04843" ]]; then
    SBM_VALIDITY_BACKEND=exact
  else
    SBM_VALIDITY_BACKEND=auto
  fi
fi
SBM_REFINEMENT_STEPS=${SBM_REFINEMENT_STEPS:-100}
SBM_VUN_REFINEMENT_STEPS=${SBM_VUN_REFINEMENT_STEPS:-1000}

ACTION_TEMPERATURE=${ACTION_TEMPERATURE:-1.0}
ACTION_CONNECT_WEIGHT=${ACTION_CONNECT_WEIGHT:-3.0}
ACTION_CREATE_WEIGHT=${ACTION_CREATE_WEIGHT:-1.0}
ACTION_STOP_WEIGHT=${ACTION_STOP_WEIGHT:-5.0}
COMPLETION_LOSS_WEIGHT=${COMPLETION_LOSS_WEIGHT:-0.1}
COLOR_BALANCE_WEIGHT=${COLOR_BALANCE_WEIGHT:-0.01}

PLAN_LOSS_WEIGHT=${PLAN_LOSS_WEIGHT:-1.0}
PLAN_DROPOUT=${PLAN_DROPOUT:-0.10}
PLAN_PRIOR_WEIGHT=${PLAN_PRIOR_WEIGHT:-3.0}
PLAN_TEMPERATURE=${PLAN_TEMPERATURE:-1.0}

EMPIRICAL_PRIOR_WEIGHT=${EMPIRICAL_PRIOR_WEIGHT:-0.0}
ADAPTIVE_PRIOR_WEIGHT=${ADAPTIVE_PRIOR_WEIGHT:-0.0}
ADAPTIVE_PRIOR_PSEUDOCOUNT=${ADAPTIVE_PRIOR_PSEUDOCOUNT:-1000.0}
CONNECT_EXISTING_FIRST=${CONNECT_EXISTING_FIRST:-1}

download_verified_asset() {
  local url="$1"
  local destination="$2"
  local expected_sha256="$3"
  local actual_sha256=""
  if [[ -f "${destination}" ]]; then
    actual_sha256="$(sha256sum "${destination}" | awk '{print $1}')"
    if [[ "${actual_sha256}" == "${expected_sha256}" ]]; then
      return 0
    fi
    echo "Refusing to overwrite checksum-mismatched benchmark asset: ${destination}" >&2
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

CATFLOW_GDSS_REVISION=24cc490e0c5b39cbd265fda33ab2bc8b528d4454

has_catflow_main_graph_dataset() {
  local requested_dataset
  for requested_dataset in ${BENCH_DATASETS}; do
    case "${requested_dataset}" in
      EGO_SMALL|COMMUNITY_SMALL|ENZYMES|GRID)
        return 0
        ;;
    esac
  done
  return 1
}

if [[ "${BENCHMARK_SUITE}" == "catflow_2406_04843" ]]; then
  if [[ "${PAPER_SPLIT_PROTOCOL}" != "catflow_gdss" ]]; then
    echo "CatFlow main-table runs require PAPER_SPLIT_PROTOCOL=catflow_gdss; received ${PAPER_SPLIT_PROTOCOL@Q}." >&2
    exit 2
  fi
  if has_catflow_main_graph_dataset; then
    if [[ -z "${PAPER_DATA_ROOT}" ]]; then
      PAPER_DATA_ROOT="benchmark_data/official_catflow"
    fi
    "${PYBIN}" src/prepare_official_catflow_data.py --output-dir "${PAPER_DATA_ROOT}"
  fi
fi

if [[ "${BENCHMARK_SUITE}" == "catflow_extra_2406_04843" ]]; then
  if [[ "${PAPER_SPLIT_PROTOCOL}" != "digress_seed0" ]]; then
    echo "CatFlow appendix Planar/SBM runs require PAPER_SPLIT_PROTOCOL=digress_seed0; received ${PAPER_SPLIT_PROTOCOL@Q}." >&2
    exit 2
  fi
  if [[ -z "${PAPER_DATA_ROOT}" ]]; then
    PAPER_DATA_ROOT="benchmark_data/official_spectre"
  fi
  "${PYBIN}" src/prepare_official_spectre_data.py \
    --output-dir "${PAPER_DATA_ROOT}" \
    --datasets PLANAR SBM
fi

if [[ "${QM9_PROTOCOL}" == "gdss_catflow" && " ${BENCH_DATASETS} " == *" QM9 "* ]]; then
  if [[ -z "${QM9_DATA_ROOT}" ]]; then
    QM9_DATA_ROOT="benchmark_data/all_corrected/QM9"
  fi
  download_verified_asset \
    "https://raw.githubusercontent.com/harryjo97/GDSS/${CATFLOW_GDSS_REVISION}/data/qm9.csv" \
    "${QM9_DATA_ROOT}/raw/gdss_qm9.csv" \
    "b7d055641f34c2803d0461806af46d263ebc6dec879bf0766657b1d3efd7ff8f"
  download_verified_asset \
    "https://raw.githubusercontent.com/harryjo97/GDSS/${CATFLOW_GDSS_REVISION}/data/valid_idx_qm9.json" \
    "${QM9_DATA_ROOT}/raw/valid_idx_qm9.json" \
    "889fd232a33d8a2c00f057c807b119e72abd5baa48f6fab93e8d8b7a79340f4d"
fi

if [[ "${ZINC250K_PROTOCOL}" == "gdss_catflow" && " ${BENCH_DATASETS} " == *" ZINC250K "* ]]; then
  if [[ -z "${ZINC_DATA_ROOT}" ]]; then
    ZINC_DATA_ROOT="benchmark_data/official_catflow/ZINC250K"
  fi
  download_verified_asset \
    "https://raw.githubusercontent.com/harryjo97/GDSS/${CATFLOW_GDSS_REVISION}/data/zinc250k.csv" \
    "${ZINC_DATA_ROOT}/raw/gdss_zinc250k.csv" \
    "8dfc7f364bdd0c6f89d9dbaf9d1812f58d5f8b55dac9b970b26bbe597e5115f2"
  download_verified_asset \
    "https://raw.githubusercontent.com/harryjo97/GDSS/${CATFLOW_GDSS_REVISION}/data/valid_idx_zinc250k.json" \
    "${ZINC_DATA_ROOT}/raw/valid_idx_zinc250k.json" \
    "cf1af3a8588c947493ff0a2b60178db6726e9a4dbb445235b149cd3bbe6831df"
fi

mkdir -p "${BENCH_ROOT}"

cat > "${BENCH_ROOT}/paper_benchmark_status.txt" <<EOF
Main model:
  budget_ring_wl_vae = budget-v2 Ring WL-VAE with correlated node/edge
  budgets, requested-WL encoder context, reusable WL1 decoder backoff,
  and dataset-aware constrained closure. This is the paper/report model.

Selected benchmark suite:
  ${BENCHMARK_SUITE}

Selected datasets:
  ${BENCH_DATASETS}

Reference split:
  ${REFERENCE_SPLIT}

Molecule preprocessing:
  default=${MOL_PREPROCESS}, QM9=${MOL_PREPROCESS_QM9}, QM9_H=${MOL_PREPROCESS_QM9_H}

QM9 data protocols:
  QM9=${QM9_PROTOCOL}, QM9_H=${QM9_H_PROTOCOL}

ZINC250K data protocol:
  ${ZINC250K_PROTOCOL}

Dataset roots:
  TU: ${TU_DATA_ROOT:-default}
  paper graphs: ${PAPER_DATA_ROOT:-default}
  paper split protocol: ${PAPER_SPLIT_PROTOCOL}
  QM9/QM9_H: ${QM9_DATA_ROOT:-default}
  ZINC: ${ZINC_DATA_ROOT:-default}

FCD reference split:
  ${FCD_REFERENCE_SPLIT}

FCD reference duplicate handling:
  FCD_DEDUPLICATE_REFERENCE=${FCD_DEDUPLICATE_REFERENCE}
  FCD_DEDUPLICATE_GENERATED=${FCD_DEDUPLICATE_GENERATED}

Sample count mode:
  ${N_SAMPLES}
  For paper suites, N_SAMPLES=paper matches the paper evaluation size:
  QM9/ZINC/MOSES=10000, Ego-small=40, Community-small=20, Enzymes=117,
  Grid=20, Planar/SBM=40, Proteins/DD=184.
  N_SAMPLES=reference matches the selected REFERENCE_SPLIT size.

WL iterations:
  ${N_WL_ITERS}

Training profile:
  epochs=${EPOCHS}, early_stopping_patience=${EARLY_STOPPING_PATIENCE},
  early_stopping_min_epochs=${EARLY_STOPPING_MIN_EPOCHS},
  early_stopping_min_delta=${EARLY_STOPPING_MIN_DELTA}
  hidden=${HIDDEN_DIM}, latent=${LATENT_DIM}, layers=${N_LAYERS}, heads=${N_HEADS},
  lr=${LR}, weight_decay=${WEIGHT_DECAY}, dropout=${DROPOUT},
  kl_beta=${KL_BETA}, kl_warmup_epochs=${KL_WARMUP_EPOCHS}, grad_clip=${GRAD_CLIP}
  decoder_color_min_frequency=${DECODER_COLOR_MIN_FREQUENCY},
  decoder_rare_fraction_threshold=${DECODER_RARE_FRACTION_THRESHOLD}
  structural_validation_interval=${STRUCTURAL_VALIDATION_INTERVAL},
  structural_validation_samples=${STRUCTURAL_VALIDATION_SAMPLES}

Implemented local loaders:
  QM9, ZINC250K, MOSES, ZINC, MUTAG, NCI1,
  Ego-small, Community-small, Enzymes, Grid, Planar, SBM, Proteins/DD.

Paper suites:
  catflow_2406_04843:
    Ego-small, Community-small, Enzymes, Grid, QM9, ZINC250K.
    Use catflow_extra_2406_04843 for the paper's extra SBM/Planar table.
  spectre_2204_01613:
    Community-small, QM9, Planar, SBM, Proteins/DD.
  digress_style_extended:
    Ego-small, Enzymes, and Grid with a full-universe Torch split selected by
    PAPER_SPLIT_PROTOCOL. This is a DiGress metric/seed-style extension, not an
    official DiGress dataset suite and not a source of published denominators.

Metrics produced by this runner:
  Complete/ring-complete rate, plan match when applicable, closure edges,
  node/edge/degree/WL-color TV, degree MMD, clustering MMD, spectral MMD,
  15-orbit graphlet MMD for induced graphlets up to 4 nodes,
  ratio MMD when COMPUTE_MMD_RATIO=1,
  connected/planar/grid/SBM-valid graph fractions,
  exact-isomorphism graph uniqueness/novelty/VUN fractions,
  molecule validity/uniqueness/novelty, and FCD when FCD_COMMAND is set.

Exact SBM validity:
  backend=${SBM_VALIDITY_BACKEND}, accuracy_refinement_steps=${SBM_REFINEMENT_STEPS},
  vun_refinement_steps=${SBM_VUN_REFINEMENT_STEPS}

Metrics not implemented internally:
  the ORCA binary itself,
  SPECTRE wavelet MMD, FCD/ChemNet, and the original paper baseline model families.
  The Python orbit fallback is exact for Ego-small/Community-small and
  sampled for larger graphs.

Checkpoint evaluation:
  Every trained budget-v2 dataset is sampled three times: validation-only
  structural best (best_ring_vae.pt), legacy warmup-weighted validation-loss best
  (best_legacy_ring_vae.pt), and last epoch (last_ring_vae.pt).
EOF

dataset_slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '_'
}

qm9_protocol_for_dataset() {
  case "$1" in
    QM9) printf '%s\n' "${QM9_PROTOCOL}" ;;
    QM9_H) printf '%s\n' "${QM9_H_PROTOCOL}" ;;
    *) printf '%s\n' legacy_local ;;
  esac
}

mol_preprocess_for_dataset() {
  case "$1" in
    QM9) printf '%s\n' "${MOL_PREPROCESS_QM9}" ;;
    QM9_H) printf '%s\n' "${MOL_PREPROCESS_QM9_H}" ;;
    *) printf '%s\n' "${MOL_PREPROCESS}" ;;
  esac
}

data_args_for_dataset() {
  local dataset="$1"
  case "${dataset}" in
    MUTAG|NCI1)
      if [[ -n "${TU_DATA_ROOT}" ]]; then
        printf '%s\n' --data-root "${TU_DATA_ROOT}"
      fi
      ;;
    EGO_SMALL|COMMUNITY_SMALL|ENZYMES|GRID|PLANAR|SBM|PROTEINS)
      if [[ -n "${PAPER_DATA_ROOT}" ]]; then
        printf '%s\n' --data-root "${PAPER_DATA_ROOT}"
      fi
      printf '%s\n' --paper-split-protocol "${PAPER_SPLIT_PROTOCOL}"
      ;;
    QM9|QM9_H)
      if [[ -n "${QM9_DATA_ROOT}" ]]; then
        printf '%s\n' --data-root "${QM9_DATA_ROOT}"
      fi
      printf '%s\n' --qm9-protocol "$(qm9_protocol_for_dataset "${dataset}")"
      ;;
    ZINC250K)
      if [[ -n "${ZINC_DATA_ROOT}" ]]; then
        printf '%s\n' --data-root "${ZINC_DATA_ROOT}"
      fi
      printf '%s\n' --zinc-subset large
      printf '%s\n' --zinc250k-protocol "${ZINC250K_PROTOCOL}"
      ;;
    ZINC)
      if [[ -n "${ZINC_DATA_ROOT}" ]]; then
        printf '%s\n' --data-root "${ZINC_DATA_ROOT}"
      fi
      printf '%s\n' --zinc-subset small
      ;;
    MOSES)
      if [[ -n "${MOSES_SMILES_FILE:-}" ]]; then
        printf '%s\n' --smiles-file "${MOSES_SMILES_FILE}"
      fi
      ;;
    GUACAMOL)
      if [[ -n "${GUACAMOL_TRAIN_FILE:-}" ]]; then
        printf '%s\n' --smiles-file "${GUACAMOL_TRAIN_FILE}"
      fi
      ;;
  esac
}

paper_sample_count_for_dataset() {
  local dataset="$1"
  case "${dataset}" in
    QM9|QM9_H|ZINC250K|ZINC|MOSES|GUACAMOL)
      printf '%s\n' 10000
      ;;
    EGO_SMALL)
      printf '%s\n' 40
      ;;
    COMMUNITY_SMALL|GRID)
      printf '%s\n' 20
      ;;
    ENZYMES)
      printf '%s\n' 117
      ;;
    PLANAR|SBM)
      printf '%s\n' 40
      ;;
    PROTEINS)
      printf '%s\n' 184
      ;;
    *)
      printf '%s\n' 10000
      ;;
  esac
}

reference_sample_count_for_dataset() {
  local dataset="$1"
  local split="$2"
  local extra_data_args=()
  mapfile -t extra_data_args < <(data_args_for_dataset "${dataset}")

  "${PYBIN}" - "${dataset}" "${split}" "${SEED}" "${extra_data_args[@]}" <<'PY'
import sys

from gpt_datasets import load_graph_dataset_bundle


dataset = sys.argv[1]
split = sys.argv[2]
seed = int(sys.argv[3])
extra = sys.argv[4:]
kwargs = {}
i = 0
while i < len(extra):
    key = extra[i]
    if key == "--data-root":
        kwargs["data_root"] = extra[i + 1]
        i += 2
    elif key == "--zinc-subset":
        kwargs["zinc_subset"] = extra[i + 1]
        i += 2
    elif key == "--smiles-file":
        kwargs["smiles_file"] = extra[i + 1]
        i += 2
    elif key == "--paper-split-protocol":
        kwargs["paper_split_protocol"] = extra[i + 1]
        i += 2
    elif key == "--qm9-protocol":
        kwargs["qm9_protocol"] = extra[i + 1]
        i += 2
    elif key == "--moses-split":
        kwargs["moses_split"] = extra[i + 1]
        i += 2
    else:
        raise SystemExit(f"Unsupported data arg for N_SAMPLES=reference: {key}")

bundle = load_graph_dataset_bundle(dataset, seed=seed, **kwargs)
graphs_by_split = {
    "train": bundle.train_graphs,
    "val": bundle.val_graphs,
    "test": bundle.test_graphs,
}
print(len(graphs_by_split[split]))
PY
}

samples_for_dataset() {
  local dataset="$1"
  if [[ "${N_SAMPLES}" == "paper" ]]; then
    paper_sample_count_for_dataset "${dataset}"
  elif [[ "${N_SAMPLES}" == "reference" || "${N_SAMPLES}" == "ref" ]]; then
    reference_sample_count_for_dataset "${dataset}" "${REFERENCE_SPLIT}"
  else
    printf '%s\n' "${N_SAMPLES}"
  fi
}

max_seq_len_for_dataset() {
  local dataset="$1"
  if [[ "${MAX_SEQ_LEN}" != "auto" ]]; then
    printf '%s\n' "${MAX_SEQ_LEN}"
    return
  fi
  case "${dataset}" in
    NCI1)
      printf '%s\n' 512
      ;;
    GRID|PROTEINS)
      printf '%s\n' 768
      ;;
    SBM)
      printf '%s\n' 384
      ;;
    *)
      printf '%s\n' 256
      ;;
  esac
}

batch_size_for_dataset() {
  local dataset="$1"
  local dataset_key="${dataset^^}"
  dataset_key="${dataset_key//-/_}"
  local wl_specific_name="BATCH_SIZE_${dataset_key}_WL${N_WL_ITERS}"
  local dataset_specific_name="BATCH_SIZE_${dataset_key}"
  local value="${!wl_specific_name:-}"

  if [[ -z "${value}" ]]; then
    value="${!dataset_specific_name:-}"
  fi
  if [[ -z "${value}" && "${BATCH_SIZE}" != "auto" ]]; then
    value="${BATCH_SIZE}"
  fi
  if [[ -z "${value}" ]]; then
    case "${dataset}:${N_WL_ITERS}" in
      MUTAG:*|EGO_SMALL:*|GRID:*) value=32 ;;
      COMMUNITY_SMALL:1|COMMUNITY_SMALL:2) value=32 ;;
      COMMUNITY_SMALL:3) value=16 ;;
      ENZYMES:1) value=64 ;;
      ENZYMES:2|ENZYMES:3) value=32 ;;
      PLANAR:1|SBM:1|PROTEINS:1) value=64 ;;
      PLANAR:2|SBM:2|PROTEINS:2) value=32 ;;
      PLANAR:3|SBM:3|PROTEINS:3) value=16 ;;
      NCI1:1) value=128 ;;
      NCI1:2) value=64 ;;
      NCI1:3) value=32 ;;
      QM9:1) value=512 ;;
      QM9:2) value=256 ;;
      QM9:3) value=128 ;;
      QM9_H:1) value=256 ;;
      QM9_H:2) value=256 ;;
      QM9_H:3) value=128 ;;
      ZINC250K:1|ZINC:1) value=128 ;;
      ZINC250K:2|ZINC:2) value=64 ;;
      ZINC250K:3|ZINC:3) value=32 ;;
      MOSES:1|GUACAMOL:1) value=256 ;;
      MOSES:2|GUACAMOL:2) value=128 ;;
      MOSES:3|GUACAMOL:3) value=64 ;;
      *) value=32 ;;
    esac
  fi

  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid batch size ${value@Q} for ${dataset} WL${N_WL_ITERS}." >&2
    return 2
  fi
  printf '%s\n' "${value}"
}

train_args_for_variant() {
  local variant="$1"
  case "${variant}" in
    budget)
      printf '%s\n' \
        --budget-context \
        --edge-budget-context \
        --decoder-color-min-frequency "${DECODER_COLOR_MIN_FREQUENCY}" \
        --decoder-rare-fraction-threshold "${DECODER_RARE_FRACTION_THRESHOLD}" \
        --structural-validation-interval "${STRUCTURAL_VALIDATION_INTERVAL}" \
        --structural-validation-samples "${STRUCTURAL_VALIDATION_SAMPLES}" \
        --no-action-head \
        --no-completion-head
      ;;
    base)
      printf '%s\n' \
        --no-action-head \
        --no-completion-head
      ;;
    action)
      printf '%s\n' \
        --budget-context \
        --completion-loss-weight "${COMPLETION_LOSS_WEIGHT}" \
        --color-balance-weight "${COLOR_BALANCE_WEIGHT}" \
        --action-connect-weight "${ACTION_CONNECT_WEIGHT}" \
        --action-create-weight "${ACTION_CREATE_WEIGHT}" \
        --action-stop-weight "${ACTION_STOP_WEIGHT}"
      ;;
    latent_plan)
      printf '%s\n' \
        --budget-context \
        --plan-context \
        --latent-plan \
        --plan-loss-weight "${PLAN_LOSS_WEIGHT}" \
        --plan-dropout "${PLAN_DROPOUT}"
      ;;
    *)
      echo "Unknown benchmark variant: ${variant}" >&2
      return 2
      ;;
  esac
}

run_args_for_variant() {
  local variant="$1"
  case "${variant}" in
    latent_plan)
      printf '%s\n' \
        --plan-temperature "${PLAN_TEMPERATURE}" \
        --plan-prior-weight "${PLAN_PRIOR_WEIGHT}"
      ;;
    *)
      true
      ;;
  esac
}

checkpoint_compatible_for_variant() {
  local checkpoint="$1"
  local dataset="$2"
  local variant="$3"
  local feature_args=()
  case "${variant}" in
    budget)
      feature_args+=(--budget-context on --edge-budget-context on --action-head off --completion-head off --plan-context off --latent-plan off)
      ;;
    base)
      feature_args+=(--budget-context off --action-head off --completion-head off --plan-context off --latent-plan off)
      ;;
    action)
      feature_args+=(--budget-context on --action-head on --completion-head on --plan-context off --latent-plan off)
      ;;
    latent_plan)
      feature_args+=(--budget-context on --action-head on --completion-head on --plan-context on --latent-plan on)
      ;;
    *)
      echo "Unknown benchmark variant: ${variant}" >&2
      return 2
      ;;
  esac
  local provenance_args=()
  local expected_batch_size
  expected_batch_size="$(batch_size_for_dataset "${dataset}")"
  local expected_max_seq_len
  expected_max_seq_len="$(max_seq_len_for_dataset "${dataset}")"
  local training_args=(
    --expected-training-arg "seed=${SEED}"
    --expected-training-arg "epochs=${EPOCHS}"
    --expected-training-arg "batch_size=${expected_batch_size}"
    --expected-training-arg "hidden_dim=${HIDDEN_DIM}"
    --expected-training-arg "latent_dim=${LATENT_DIM}"
    --expected-training-arg "n_layers=${N_LAYERS}"
    --expected-training-arg "n_heads=${N_HEADS}"
    --expected-training-arg "max_seq_len=${expected_max_seq_len}"
    --expected-training-arg "lr=${LR}"
    --expected-training-arg "weight_decay=${WEIGHT_DECAY}"
    --expected-training-arg "dropout=${DROPOUT}"
    --expected-training-arg "kl_beta=${KL_BETA}"
    --expected-training-arg "kl_warmup_epochs=${KL_WARMUP_EPOCHS}"
    --expected-training-arg "early_stopping_patience=${EARLY_STOPPING_PATIENCE}"
    --expected-training-arg "early_stopping_min_epochs=${EARLY_STOPPING_MIN_EPOCHS}"
    --expected-training-arg "early_stopping_min_delta=${EARLY_STOPPING_MIN_DELTA}"
    --expected-training-arg "grad_clip=${GRAD_CLIP}"
  )
  if [[ "${variant}" == "budget" ]]; then
    training_args+=(
      --expected-training-arg "decoder_color_min_frequency=${DECODER_COLOR_MIN_FREQUENCY}"
      --expected-training-arg "decoder_rare_fraction_threshold=${DECODER_RARE_FRACTION_THRESHOLD}"
      --expected-training-arg "edge_budget_context=true"
      --expected-training-arg "structural_validation_interval=${STRUCTURAL_VALIDATION_INTERVAL}"
      --expected-training-arg "structural_validation_samples=${STRUCTURAL_VALIDATION_SAMPLES}"
    )
  fi
  if [[ "${dataset}" == "QM9" || "${dataset}" == "QM9_H" ]]; then
    local dataset_qm9_protocol
    dataset_qm9_protocol="$(qm9_protocol_for_dataset "${dataset}")"
    training_args+=(--expected-training-arg "qm9_protocol=\"${dataset_qm9_protocol}\"")
    if [[ "${VERIFY_DATA_PROVENANCE}" == "1" ]]; then
      provenance_args+=(--verify-data-provenance --seed "${SEED}" --qm9-protocol "${dataset_qm9_protocol}")
      if [[ -n "${QM9_DATA_ROOT}" ]]; then
        provenance_args+=(--data-root "${QM9_DATA_ROOT}")
      fi
    fi
  fi
  if [[ "${dataset}" == "ZINC250K" ]]; then
    training_args+=(
      --expected-training-arg "zinc250k_protocol=\"${ZINC250K_PROTOCOL}\""
    )
    if [[ "${VERIFY_DATA_PROVENANCE}" == "1" ]]; then
      provenance_args+=(
        --verify-data-provenance
        --seed "${SEED}"
        --zinc250k-protocol "${ZINC250K_PROTOCOL}"
      )
      if [[ -n "${ZINC_DATA_ROOT}" ]]; then
        provenance_args+=(--data-root "${ZINC_DATA_ROOT}")
      fi
    fi
  fi
  if [[ "${VERIFY_DATA_PROVENANCE}" == "1" ]]; then
    case "${dataset}" in
      EGO_SMALL|COMMUNITY_SMALL|ENZYMES|GRID|PLANAR|SBM|PROTEINS)
        provenance_args+=(
          --verify-data-provenance
          --seed "${SEED}"
          --paper-split-protocol "${PAPER_SPLIT_PROTOCOL}"
        )
        if [[ -n "${PAPER_DATA_ROOT}" ]]; then
          provenance_args+=(--data-root "${PAPER_DATA_ROOT}")
        fi
        ;;
    esac
  fi
  [[ -f "${checkpoint}" ]] && "${PYBIN}" src/check_ring_vae_checkpoint.py \
    --checkpoint "${checkpoint}" --dataset "${dataset}" --n-wl-iters "${N_WL_ITERS}" \
    "${provenance_args[@]}" "${training_args[@]}" "${feature_args[@]}"
}

checkpoint_set_compatible_for_variant() {
  local checkpoint_dir="$1"
  local dataset="$2"
  local variant="$3"
  local filename
  for filename in best_ring_vae.pt best_legacy_ring_vae.pt last_ring_vae.pt; do
    checkpoint_compatible_for_variant "${checkpoint_dir}/${filename}" "${dataset}" "${variant}" || return 1
  done
}

run_dataset_variant() {
  local dataset="$1"
  local variant="$2"
  local lower
  lower="$(dataset_slug "${dataset}")"
  local ckpt_dir="${BENCH_ROOT}/${lower}/${variant}/checkpoints"
  local out_dir="${BENCH_ROOT}/${lower}/${variant}/samples"
  local trace_suffix=""
  if [[ "${N_WL_ITERS}" != "3" ]]; then
    trace_suffix="_wl${N_WL_ITERS}"
  fi
  local trace_cache="${TRACE_ROOT}/benchmark_${variant}_${lower}${trace_suffix}_seed${SEED}.pt"
  local best_ckpt="${ckpt_dir}/best_ring_vae.pt"
  local legacy_best_ckpt="${ckpt_dir}/best_legacy_ring_vae.pt"
  local last_ckpt="${ckpt_dir}/last_ring_vae.pt"
  local paper_ckpt_dir="checkpoints/ring_vae_budget/zinc250k_seed${SEED}"
  local using_paper_checkpoints=0
  local dataset_samples
  dataset_samples="$(samples_for_dataset "${dataset}")"
  local dataset_max_seq_len
  dataset_max_seq_len="$(max_seq_len_for_dataset "${dataset}")"
  local dataset_batch_size
  dataset_batch_size="$(batch_size_for_dataset "${dataset}")"
  local dataset_mol_preprocess
  dataset_mol_preprocess="$(mol_preprocess_for_dataset "${dataset}")"

  local extra_data_args=()
  mapfile -t extra_data_args < <(data_args_for_dataset "${dataset}")
  local extra_train_args=()
  mapfile -t extra_train_args < <(train_args_for_variant "${variant}")
  local extra_run_args=()
  mapfile -t extra_run_args < <(run_args_for_variant "${variant}")

  if [[ "${CONNECT_EXISTING_FIRST}" == "0" ]]; then
    extra_run_args+=(--no-connect-existing-first)
  fi
  if [[ "${COMPUTE_MMD_RATIO}" == "1" ]]; then
    extra_run_args+=(--compute-mmd-ratio)
  fi
  if [[ "${VERIFY_DATA_PROVENANCE}" == "1" ]]; then
    extra_run_args+=(--verify-data-provenance)
  fi
  if [[ "${dataset}" == "SBM" ]]; then
    extra_run_args+=(
      --sbm-validity-backend "${SBM_VALIDITY_BACKEND}"
      --sbm-refinement-steps "${SBM_REFINEMENT_STEPS}"
      --sbm-vun-refinement-steps "${SBM_VUN_REFINEMENT_STEPS}"
    )
  fi
  if [[ "${FCD_DEDUPLICATE_REFERENCE}" == "1" ]]; then
    extra_run_args+=(--fcd-deduplicate-reference)
  fi
  if [[ "${FCD_DEDUPLICATE_GENERATED}" == "1" ]]; then
    extra_run_args+=(--fcd-deduplicate-generated)
  fi
  if [[ -n "${FCD_COMMAND}" ]]; then
    extra_run_args+=(--fcd-command "${FCD_COMMAND}" --fcd-timeout "${FCD_TIMEOUT}")
  fi

  echo
  echo "=== Benchmark: dataset=${dataset} variant=${variant} WL=${N_WL_ITERS} batch_size=${dataset_batch_size} mol_preprocess=${dataset_mol_preprocess} ==="

  if [[ "${variant}" == "budget" && "${dataset}" == "ZINC250K" && "${REUSE_PAPER_ZINC_CKPT}" == "1" ]]; then
    if checkpoint_set_compatible_for_variant "${paper_ckpt_dir}" "${dataset}" "${variant}"; then
      best_ckpt="${paper_ckpt_dir}/best_ring_vae.pt"
      legacy_best_ckpt="${paper_ckpt_dir}/best_legacy_ring_vae.pt"
      last_ckpt="${paper_ckpt_dir}/last_ring_vae.pt"
      using_paper_checkpoints=1
      echo "Using complete paper checkpoint set: ${paper_ckpt_dir}"
    elif [[ -f "${paper_ckpt_dir}/best_ring_vae.pt" ]]; then
      echo "Paper checkpoint set is incomplete or incompatible; training a local three-checkpoint set."
    fi
  fi

  if [[ "${using_paper_checkpoints}" == "0" ]] \
     && ! checkpoint_set_compatible_for_variant "${ckpt_dir}" "${dataset}" "${variant}"; then
    echo "Training because the local checkpoint set is missing or incompatible."
    "${PYBIN}" src/train_ring_vae.py \
      --dataset "${dataset}" \
      "${extra_data_args[@]}" \
      --seed "${SEED}" \
      --n-wl-iters "${N_WL_ITERS}" \
      --epochs "${EPOCHS}" \
      --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
      --early-stopping-min-epochs "${EARLY_STOPPING_MIN_EPOCHS}" \
      --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
      --batch-size "${dataset_batch_size}" \
      --hidden-dim "${HIDDEN_DIM}" \
      --latent-dim "${LATENT_DIM}" \
      --n-layers "${N_LAYERS}" \
      --n-heads "${N_HEADS}" \
      --max-seq-len "${dataset_max_seq_len}" \
      --lr "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --dropout "${DROPOUT}" \
      --kl-beta "${KL_BETA}" \
      --kl-warmup-epochs "${KL_WARMUP_EPOCHS}" \
      --grad-clip "${GRAD_CLIP}" \
      "${extra_train_args[@]}" \
      --trace-cache "${trace_cache}" \
      --output-dir "${ckpt_dir}"
  elif [[ "${using_paper_checkpoints}" == "0" ]]; then
    echo "Found complete compatible checkpoint set in ${ckpt_dir}; skipping training."
  fi

  local primary_checkpoint_label="fixed_beta_best"
  if [[ "${variant}" == "budget" ]]; then
    primary_checkpoint_label="structural_validation_best"
  fi
  local checkpoint_labels=("${primary_checkpoint_label}" legacy_val_loss_best last_epoch)
  local checkpoint_paths=("${best_ckpt}" "${legacy_best_ckpt}" "${last_ckpt}")
  local checkpoint_output_dirs=(
    "${out_dir}"
    "${BENCH_ROOT}/${lower}/${variant}_legacy_val_loss_best/samples"
    "${BENCH_ROOT}/${lower}/${variant}_last_epoch/samples"
  )
  local checkpoint_idx
  for checkpoint_idx in "${!checkpoint_paths[@]}"; do
    local checkpoint_label="${checkpoint_labels[checkpoint_idx]}"
    local checkpoint_path="${checkpoint_paths[checkpoint_idx]}"
    local checkpoint_base_out_dir="${checkpoint_output_dirs[checkpoint_idx]}"
    if [[ ! -f "${checkpoint_path}" ]]; then
      echo "Missing ${checkpoint_label} checkpoint: ${checkpoint_path}" >&2
      return 2
    fi
    local completion_mode
    for completion_mode in ${EVAL_COMPLETION_MODES}; do
      local checkpoint_out_dir
      local completion_args=()
      case "${completion_mode}" in
        strict|require_complete)
          checkpoint_out_dir="${checkpoint_base_out_dir}"
          completion_args+=(--require-complete)
          ;;
        allow_incomplete|permissive)
          checkpoint_out_dir="${checkpoint_base_out_dir%/samples}_allow_incomplete/samples"
          completion_args+=(--allow-incomplete)
          ;;
        *)
          echo "Unknown completion evaluation mode: ${completion_mode}" >&2
          return 2
          ;;
      esac
      echo "Sampling checkpoint role=${checkpoint_label} completion_mode=${completion_mode}: ${checkpoint_path}"
      "${PYBIN}" src/run_ring_vae.py \
        --dataset "${dataset}" \
        "${extra_data_args[@]}" \
        --checkpoint "${checkpoint_path}" \
        --seed "${SEED}" \
        --n-samples "${dataset_samples}" \
        --reference-split "${REFERENCE_SPLIT}" \
        --mol-preprocess "${dataset_mol_preprocess}" \
        --fcd-reference-split "${FCD_REFERENCE_SPLIT}" \
        --node-budget-factor "${NODE_BUDGET_FACTOR}" \
        --node-budget-margin "${NODE_BUDGET_MARGIN}" \
        --action-temperature "${ACTION_TEMPERATURE}" \
        --empirical-prior-weight "${EMPIRICAL_PRIOR_WEIGHT}" \
        --adaptive-prior-weight "${ADAPTIVE_PRIOR_WEIGHT}" \
        --adaptive-prior-pseudocount "${ADAPTIVE_PRIOR_PSEUDOCOUNT}" \
        --max-restarts "${MAX_RESTARTS}" \
        --metric-max-train-graphs "${METRIC_MAX_TRAIN_GRAPHS}" \
        --diagnostic-max-train-graphs "${DIAGNOSTIC_MAX_TRAIN_GRAPHS}" \
        --mmd-max-graphs "${MMD_MAX_GRAPHS}" \
        "${completion_args[@]}" \
        "${extra_run_args[@]}" \
        --output-dir "${checkpoint_out_dir}"
    done
  done
}

for dataset in ${BENCH_DATASETS}; do
  if [[ "${dataset}" == "MOSES" && -z "${MOSES_SMILES_FILE:-}" ]]; then
    if ! "${PYBIN}" - <<'PY' >/dev/null 2>&1
import moses  # noqa: F401
PY
    then
      echo "Skipping MOSES: set MOSES_SMILES_FILE or install the optional moses package."
      continue
    fi
  fi

  run_dataset_variant "${dataset}" budget

  if [[ "${RUN_ABLATIONS}" == "1" ]]; then
    for variant in ${ABLATION_VARIANTS}; do
      run_dataset_variant "${dataset}" "${variant}"
    done
  fi
done

"${PYBIN}" src/benchmark_report.py \
  --root "${BENCH_ROOT}" \
  --out "${BENCH_ROOT}/summary"

if [[ "${BENCHMARK_SUITE}" == "catflow_2406_04843" ]] \
   && has_catflow_main_graph_dataset; then
  "${PYBIN}" src/recompute_catflow_metrics.py \
    --run-root "${BENCH_ROOT}" \
    --data-root "${PAPER_DATA_ROOT}" \
    --output-dir "${BENCH_ROOT}/catflow_compatible" \
    --markdown "${BENCH_ROOT}/catflow_compatible/catflow_metrics.md"
fi

if [[ "${BENCHMARK_SUITE}" == "catflow_extra_2406_04843" ]]; then
  "${PYBIN}" src/recompute_digress_spectre_metrics.py \
    --run-root "${BENCH_ROOT}" \
    --output-dir "${BENCH_ROOT}/catflow_extra_compatible" \
    --markdown "${BENCH_ROOT}/catflow_extra_compatible/catflow_extra_metrics.md" \
    --datasets PLANAR SBM
fi
