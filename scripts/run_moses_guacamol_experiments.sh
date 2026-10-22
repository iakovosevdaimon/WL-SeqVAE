#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
SEED=${SEED:-0}
MODE=${MODE:-smoke}
EXP_ROOT=${EXP_ROOT:-benchmark_runs/moses_guacamol_seed${SEED}}
LOG_ROOT="${EXP_ROOT}/logs"
TRACE_ROOT=${TRACE_ROOT:-trace_cache/moses_guacamol_seed${SEED}}

if [[ "${MODE}" == "full" ]]; then
  EPOCHS=${EPOCHS:-50}
  N_SAMPLES=${N_SAMPLES:-10000}
  MAX_TRAIN_GRAPHS=${MAX_TRAIN_GRAPHS:-}
  MAX_VAL_GRAPHS=${MAX_VAL_GRAPHS:-}
else
  EPOCHS=${EPOCHS:-20}
  N_SAMPLES=${N_SAMPLES:-1000}
  MAX_TRAIN_GRAPHS=${MAX_TRAIN_GRAPHS:-50000}
  MAX_VAL_GRAPHS=${MAX_VAL_GRAPHS:-5000}
fi

if [[ ! "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer, received ${EPOCHS@Q}." >&2
  exit 2
fi
if (( EPOCHS > 50 )); then
  echo "Capping requested EPOCHS=${EPOCHS} to the benchmark maximum of 50."
  EPOCHS=50
fi

BATCH_SIZE=${BATCH_SIZE:-64}
HIDDEN_DIM=${HIDDEN_DIM:-192}
LATENT_DIM=${LATENT_DIM:-64}
N_LAYERS=${N_LAYERS:-4}
N_HEADS=${N_HEADS:-4}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-256}
KL_BETA=${KL_BETA:-0.05}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-10}
EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS:-15}
EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA:-0.001}
DECODER_COLOR_MIN_FREQUENCY=${DECODER_COLOR_MIN_FREQUENCY:-5}
DECODER_RARE_FRACTION_THRESHOLD=${DECODER_RARE_FRACTION_THRESHOLD:-0.0}
STRUCTURAL_VALIDATION_INTERVAL=${STRUCTURAL_VALIDATION_INTERVAL:-5}
STRUCTURAL_VALIDATION_SAMPLES=${STRUCTURAL_VALIDATION_SAMPLES:-8}
N_WL_ITERS=${N_WL_ITERS:-}
MOL_PREPROCESS=${MOL_PREPROCESS:-catflow}
MOL_PREPROCESS_QM9=${MOL_PREPROCESS_QM9:-${MOL_PREPROCESS}}
MOL_PREPROCESS_QM9_H=${MOL_PREPROCESS_QM9_H:-repo}
FCD_REFERENCE_SPLIT=${FCD_REFERENCE_SPLIT:-test}
REFERENCE_SPLIT=${REFERENCE_SPLIT:-test}
FCD_COMMAND=${FCD_COMMAND:-"${PYBIN} src/compute_fcd.py --ref {ref} --gen {gen} --out {out} --device cuda"}
FCD_TIMEOUT=${FCD_TIMEOUT:-1200}
MAX_RESTARTS=${MAX_RESTARTS:-200}
EVAL_COMPLETION_MODES=${EVAL_COMPLETION_MODES:-strict}

# Dataset toggles.
RUN_QM9_NO_H=${RUN_QM9_NO_H:-1}
RUN_QM9_H=${RUN_QM9_H:-1}
RUN_ZINC=${RUN_ZINC:-1}
RUN_MOSES=${RUN_MOSES:-0}
RUN_GUACAMOL=${RUN_GUACAMOL:-0}

if [[ -z "${N_WL_ITERS}" ]]; then
  if [[ "${RUN_QM9_NO_H}" == "0" && "${RUN_QM9_H}" == "0" && "${RUN_ZINC}" == "0" ]] \
    && [[ "${RUN_MOSES}" == "1" || "${RUN_GUACAMOL}" == "1" ]]; then
    N_WL_ITERS=3
  else
    N_WL_ITERS=2
  fi
fi

# Training toggles. Set RUN_RETRAIN=0 to evaluate existing checkpoints only.
RUN_RETRAIN=${RUN_RETRAIN:-1}
RUN_RETRAIN_ZINC=${RUN_RETRAIN_ZINC:-0}
SKIP_EXISTING_TRAINING=${SKIP_EXISTING_TRAINING:-0}

# Optional data/package locations.
MOSES_SMILES_FILE=${MOSES_SMILES_FILE:-}
MOSES_TRAIN_FILE=${MOSES_TRAIN_FILE:-}
MOSES_TEST_FILE=${MOSES_TEST_FILE:-}
MOSES_TEST_SCAFFOLDS_FILE=${MOSES_TEST_SCAFFOLDS_FILE:-}
GUACAMOL_TRAIN_FILE=${GUACAMOL_TRAIN_FILE:-}
GUACAMOL_TEST_FILE=${GUACAMOL_TEST_FILE:-}
BENCHMARK_DATA_DIR=${BENCHMARK_DATA_DIR:-datasets/molecule_benchmarks}
AUTO_DOWNLOAD_BENCHMARK_DATA=${AUTO_DOWNLOAD_BENCHMARK_DATA:-0}
INSTALL_BENCHMARK_PACKAGES=${INSTALL_BENCHMARK_PACKAGES:-0}
RUN_MOSES_METRICS=${RUN_MOSES_METRICS:-1}
RUN_GUACAMOL_METRICS=${RUN_GUACAMOL_METRICS:-1}
MOSES_BACKEND=${MOSES_BACKEND:-auto}
BENCHMARK_METRIC_SAMPLES=${BENCHMARK_METRIC_SAMPLES:-2000}

mkdir -p "${EXP_ROOT}" "${LOG_ROOT}" "${TRACE_ROOT}"

cat > "${EXP_ROOT}/README.md" <<EOF
# MOSES / GuacaMol Extra Experiments

Mode: ${MODE}
Seed: ${SEED}
Samples: ${N_SAMPLES}
Epochs: ${EPOCHS}
Early stopping: patience=${EARLY_STOPPING_PATIENCE}, min_epochs=${EARLY_STOPPING_MIN_EPOCHS}, relative_min_delta=${EARLY_STOPPING_MIN_DELTA}
WL iterations: ${N_WL_ITERS}
Molecule preprocessing: ${MOL_PREPROCESS}
FCD reference split: ${FCD_REFERENCE_SPLIT}
Checkpoint evaluation: structural-validation best, legacy validation-loss best, and last epoch.

This runner can retrain and evaluate:
- QM9 no-H / heavy atoms
- QM9_H / original explicit-H setting
- ZINC250K
- MOSES, when MOSES_SMILES_FILE is set or the moses package is installed
- GuacaMol/ChEMBL, when GUACAMOL_TRAIN_FILE is set

MOSES/GuacaMol metrics are optional because their packages and standardized
reference datasets are external to this repository.
EOF

maybe_install_packages() {
  if [[ "${INSTALL_BENCHMARK_PACKAGES}" != "1" ]]; then
    return 0
  fi
  {
    echo "Installing GuacaMol only."
    echo "Skipping pip install moses: the old MOSES package depends on pomegranate, which is not reliable on Python 3.11."
    echo "The evaluator will use the local RDKit-based MOSES-style fallback unless an official compatible MOSES install is already present."
  } | tee "${LOG_ROOT}/install_molecule_benchmarks.log"
  "${PYBIN}" -m pip install guacamol 2>&1 | tee -a "${LOG_ROOT}/install_molecule_benchmarks.log"
}

prepare_benchmark_data() {
  if [[ "${AUTO_DOWNLOAD_BENCHMARK_DATA}" == "1" ]]; then
    DATA_DIR="${BENCHMARK_DATA_DIR}" ./scripts/download_moses_guacamol_data.sh 2>&1 | tee "${LOG_ROOT}/download_moses_guacamol_data.log"
  fi

  if [[ -z "${MOSES_SMILES_FILE}" && -s "${BENCHMARK_DATA_DIR}/moses/train.csv" ]]; then
    MOSES_SMILES_FILE="${BENCHMARK_DATA_DIR}/moses/train.csv"
  fi
  if [[ -z "${MOSES_TRAIN_FILE}" && -s "${BENCHMARK_DATA_DIR}/moses/train.csv" ]]; then
    MOSES_TRAIN_FILE="${BENCHMARK_DATA_DIR}/moses/train.csv"
  fi
  if [[ -z "${MOSES_TEST_FILE}" && -s "${BENCHMARK_DATA_DIR}/moses/test.csv" ]]; then
    MOSES_TEST_FILE="${BENCHMARK_DATA_DIR}/moses/test.csv"
  fi
  if [[ -z "${MOSES_TEST_SCAFFOLDS_FILE}" && -s "${BENCHMARK_DATA_DIR}/moses/test_scaffolds.csv" ]]; then
    MOSES_TEST_SCAFFOLDS_FILE="${BENCHMARK_DATA_DIR}/moses/test_scaffolds.csv"
  fi
  if [[ -z "${GUACAMOL_TRAIN_FILE}" && -s "${BENCHMARK_DATA_DIR}/guacamol/guacamol_v1_train.smiles" ]]; then
    GUACAMOL_TRAIN_FILE="${BENCHMARK_DATA_DIR}/guacamol/guacamol_v1_train.smiles"
  fi
  if [[ -z "${GUACAMOL_TEST_FILE}" && -s "${BENCHMARK_DATA_DIR}/guacamol/guacamol_v1_test.smiles" ]]; then
    GUACAMOL_TEST_FILE="${BENCHMARK_DATA_DIR}/guacamol/guacamol_v1_test.smiles"
  fi
}

max_graph_args() {
  local args=()
  if [[ -n "${MAX_TRAIN_GRAPHS}" ]]; then
    args+=(--max-train-graphs "${MAX_TRAIN_GRAPHS}")
  fi
  if [[ -n "${MAX_VAL_GRAPHS}" ]]; then
    args+=(--max-val-graphs "${MAX_VAL_GRAPHS}")
  fi
  printf '%s\n' "${args[@]}"
}

train_model() {
  local name="$1"
  local dataset="$2"
  shift 2
  local ckpt_dir="${EXP_ROOT}/${name}/checkpoints"
  local trace_cache="${TRACE_ROOT}/${name}.pt"
  local log="${LOG_ROOT}/${name}_train.log"
  local expected_max_train=null
  local expected_max_val=null
  if [[ -n "${MAX_TRAIN_GRAPHS}" ]]; then
    expected_max_train="${MAX_TRAIN_GRAPHS}"
  fi
  if [[ -n "${MAX_VAL_GRAPHS}" ]]; then
    expected_max_val="${MAX_VAL_GRAPHS}"
  fi
  local compatibility_args=(
    --dataset "${dataset}"
    --n-wl-iters "${N_WL_ITERS}"
    --budget-context on
    --edge-budget-context on
    --action-head off
    --completion-head off
    --plan-context off
    --latent-plan off
    --expected-training-arg "seed=${SEED}"
    --expected-training-arg "epochs=${EPOCHS}"
    --expected-training-arg "batch_size=${BATCH_SIZE}"
    --expected-training-arg "hidden_dim=${HIDDEN_DIM}"
    --expected-training-arg "latent_dim=${LATENT_DIM}"
    --expected-training-arg "n_layers=${N_LAYERS}"
    --expected-training-arg "n_heads=${N_HEADS}"
    --expected-training-arg "max_seq_len=${MAX_SEQ_LEN}"
    --expected-training-arg "kl_beta=${KL_BETA}"
    --expected-training-arg "early_stopping_patience=${EARLY_STOPPING_PATIENCE}"
    --expected-training-arg "early_stopping_min_epochs=${EARLY_STOPPING_MIN_EPOCHS}"
    --expected-training-arg "early_stopping_min_delta=${EARLY_STOPPING_MIN_DELTA}"
    --expected-training-arg "decoder_color_min_frequency=${DECODER_COLOR_MIN_FREQUENCY}"
    --expected-training-arg "decoder_rare_fraction_threshold=${DECODER_RARE_FRACTION_THRESHOLD}"
    --expected-training-arg "structural_validation_interval=${STRUCTURAL_VALIDATION_INTERVAL}"
    --expected-training-arg "structural_validation_samples=${STRUCTURAL_VALIDATION_SAMPLES}"
    --expected-training-arg "max_train_graphs=${expected_max_train}"
    --expected-training-arg "max_val_graphs=${expected_max_val}"
  )

  mkdir -p "${ckpt_dir}"
  if [[ "${SKIP_EXISTING_TRAINING}" == "1" \
     && -f "${ckpt_dir}/best_ring_vae.pt" \
     && -f "${ckpt_dir}/best_legacy_ring_vae.pt" \
     && -f "${ckpt_dir}/last_ring_vae.pt" ]] \
     && "${PYBIN}" src/check_ring_vae_checkpoint.py --checkpoint "${ckpt_dir}/best_ring_vae.pt" \
       "${compatibility_args[@]}" \
     && "${PYBIN}" src/check_ring_vae_checkpoint.py --checkpoint "${ckpt_dir}/best_legacy_ring_vae.pt" \
       "${compatibility_args[@]}" \
     && "${PYBIN}" src/check_ring_vae_checkpoint.py --checkpoint "${ckpt_dir}/last_ring_vae.pt" \
       "${compatibility_args[@]}"; then
    echo "Found complete compatible checkpoint set in ${ckpt_dir}; skipping completed training for ${name}." | tee "${log}"
    return 0
  fi
  {
    echo
    echo "=== train ${name} ==="
  } | tee "${log}"

  local args=(
    "${PYBIN}" src/train_ring_vae.py
    --dataset "${dataset}"
    --seed "${SEED}"
    --n-wl-iters "${N_WL_ITERS}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --hidden-dim "${HIDDEN_DIM}"
    --latent-dim "${LATENT_DIM}"
    --n-layers "${N_LAYERS}"
    --n-heads "${N_HEADS}"
    --max-seq-len "${MAX_SEQ_LEN}"
    --kl-beta "${KL_BETA}"
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
    --early-stopping-min-epochs "${EARLY_STOPPING_MIN_EPOCHS}"
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}"
    --budget-context
    --edge-budget-context
    --decoder-color-min-frequency "${DECODER_COLOR_MIN_FREQUENCY}"
    --decoder-rare-fraction-threshold "${DECODER_RARE_FRACTION_THRESHOLD}"
    --structural-validation-interval "${STRUCTURAL_VALIDATION_INTERVAL}"
    --structural-validation-samples "${STRUCTURAL_VALIDATION_SAMPLES}"
    --no-action-head
    --no-completion-head
    --trace-cache "${trace_cache}"
    --output-dir "${ckpt_dir}"
  )
  while IFS= read -r item; do
    [[ -n "${item}" ]] && args+=("${item}")
  done < <(max_graph_args)
  args+=("$@")

  printf 'Command:' | tee -a "${log}"
  printf ' %q' "${args[@]}" | tee -a "${log}"
  echo | tee -a "${log}"
  "${args[@]}" 2>&1 | tee -a "${log}"
}

checkpoint_for_role() {
  local name="$1"
  local fallback="$2"
  local filename="$3"
  local trained="${EXP_ROOT}/${name}/checkpoints/${filename}"
  if [[ "${RUN_RETRAIN}" == "1" && -f "${trained}" ]]; then
    printf '%s\n' "${trained}"
  else
    printf '%s\n' "$(dirname "${fallback}")/${filename}"
  fi
}

eval_model() {
  local name="$1"
  local dataset="$2"
  local checkpoint="$3"
  shift 3
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Skipping ${name}: missing checkpoint ${checkpoint}" | tee -a "${LOG_ROOT}/skipped.log"
    return 0
  fi
  local out_dir="${EXP_ROOT}/${name}/samples"
  local log="${LOG_ROOT}/${name}_eval.log"
  local dataset_mol_preprocess="${MOL_PREPROCESS}"
  if [[ "${dataset}" == "QM9" ]]; then
    dataset_mol_preprocess="${MOL_PREPROCESS_QM9}"
  elif [[ "${dataset}" == "QM9_H" ]]; then
    dataset_mol_preprocess="${MOL_PREPROCESS_QM9_H}"
  fi
  mkdir -p "${out_dir}"

  local args=(
    "${PYBIN}" src/run_ring_vae.py
    --dataset "${dataset}"
    --checkpoint "${checkpoint}"
    --seed "${SEED}"
    --n-samples "${N_SAMPLES}"
    --reference-split "${REFERENCE_SPLIT}"
    --mol-preprocess "${dataset_mol_preprocess}"
    --fcd-reference-split "${FCD_REFERENCE_SPLIT}"
    --fcd-command "${FCD_COMMAND}"
    --fcd-timeout "${FCD_TIMEOUT}"
    --max-restarts "${MAX_RESTARTS}"
    --compute-mmd-ratio
    --metric-max-train-graphs 20000
    --diagnostic-max-train-graphs 20000
    --mmd-max-graphs 2000
  )
  if [[ "${dataset}" == "ZINC250K" ]]; then
    args+=(--zinc-subset large)
  fi
  args+=("$@" --output-dir "${out_dir}")

  {
    echo
    echo "=== eval ${name} ==="
    printf 'Command:'
    printf ' %q' "${args[@]}"
    echo
  } | tee "${log}"
  "${args[@]}" 2>&1 | tee -a "${log}"
}

eval_checkpoint_set() {
  local eval_name="$1"
  local dataset="$2"
  local train_name="$3"
  local fallback_best="$4"
  shift 4

  local checkpoint_filenames=(best_ring_vae.pt best_legacy_ring_vae.pt last_ring_vae.pt)
  local eval_suffixes=("" _legacy_val_loss_best _last_epoch)
  local checkpoint_idx
  for checkpoint_idx in "${!checkpoint_filenames[@]}"; do
    local filename="${checkpoint_filenames[checkpoint_idx]}"
    local checkpoint
    checkpoint="$(checkpoint_for_role "${train_name}" "${fallback_best}" "${filename}")"
    local checkpoint_eval_name="${eval_name}${eval_suffixes[checkpoint_idx]}"
    local completion_mode
    for completion_mode in ${EVAL_COMPLETION_MODES}; do
      case "${completion_mode}" in
        strict|require_complete)
          eval_model "${checkpoint_eval_name}" "${dataset}" "${checkpoint}" \
            --require-complete "$@"
          ;;
        allow_incomplete|permissive)
          eval_model "${checkpoint_eval_name}_allow_incomplete" "${dataset}" "${checkpoint}" \
            --allow-incomplete "$@"
          ;;
        *)
          echo "Unknown completion evaluation mode: ${completion_mode}" >&2
          return 2
          ;;
      esac
    done
  done
}

run_qm9_no_h() {
  local name=qm9_no_h_wl2_retrain_beta005
  local fallback="benchmark_runs/budget_ring_wl_vae_qm9_no_h_wl2_seed${SEED}/qm9/budget/checkpoints/best_ring_vae.pt"
  if [[ "${RUN_RETRAIN}" == "1" ]]; then
    QM9_FORCE_RAW_FALLBACK=1 train_model "${name}" QM9
  fi
  eval_checkpoint_set "${name}_adapt05" QM9 "${name}" "${fallback}" \
    --adaptive-prior-weight 0.5 \
    --node-budget-factor 1.0
}

run_qm9_h() {
  local name=qm9_h_wl2_retrain_beta005
  local fallback="benchmark_runs/budget_ring_wl_vae_qm9_original_wl2_seed${SEED}/qm9_h/budget/checkpoints/best_ring_vae.pt"
  if [[ "${RUN_RETRAIN}" == "1" ]]; then
    QM9_FORCE_RAW_FALLBACK=1 train_model "${name}" QM9_H
  fi
  eval_checkpoint_set "${name}_budget085" QM9_H "${name}" "${fallback}" \
    --node-budget-factor 0.85
}

run_zinc() {
  local name=zinc_wl2_retrain_beta005
  local fallback="checkpoints/ring_vae_budget/zinc250k_seed${SEED}/best_ring_vae.pt"
  if [[ "${RUN_RETRAIN}" == "1" && "${RUN_RETRAIN_ZINC}" == "1" ]]; then
    train_model "${name}" ZINC250K --zinc-subset large
  fi
  eval_checkpoint_set "${name}_adapt05" ZINC250K "${name}" "${fallback}" \
    --adaptive-prior-weight 0.5 \
    --node-budget-factor 1.0
}

run_moses_dataset() {
  local name="moses_wl${N_WL_ITERS}_retrain_beta005"
  local extra=()
  if [[ -n "${MOSES_SMILES_FILE}" ]]; then
    extra+=(--smiles-file "${MOSES_SMILES_FILE}")
  else
    if ! "${PYBIN}" - <<'PY' >/dev/null 2>&1
import moses  # noqa: F401
PY
    then
      echo "Skipping MOSES training/eval: set MOSES_SMILES_FILE or install moses." | tee -a "${LOG_ROOT}/skipped.log"
      return 0
    fi
  fi
  if [[ "${RUN_RETRAIN}" == "1" ]]; then
    train_model "${name}" MOSES "${extra[@]}"
  fi
  local fallback="${EXP_ROOT}/${name}/checkpoints/best_ring_vae.pt"
  eval_checkpoint_set "${name}_base" MOSES "${name}" "${fallback}" \
    "${extra[@]}" --node-budget-factor 1.0
}

run_guacamol_dataset() {
  local name="guacamol_wl${N_WL_ITERS}_retrain_beta005"
  if [[ -z "${GUACAMOL_TRAIN_FILE}" ]]; then
    echo "Skipping GuacaMol training/eval: set GUACAMOL_TRAIN_FILE to the standardized ChEMBL train SMILES." | tee -a "${LOG_ROOT}/skipped.log"
    return 0
  fi
  if [[ "${RUN_RETRAIN}" == "1" ]]; then
    train_model "${name}" GUACAMOL --smiles-file "${GUACAMOL_TRAIN_FILE}"
  fi
  local fallback="${EXP_ROOT}/${name}/checkpoints/best_ring_vae.pt"
  eval_checkpoint_set "${name}_base" GUACAMOL "${name}" "${fallback}" \
    --smiles-file "${GUACAMOL_TRAIN_FILE}" --node-budget-factor 1.0
}

run_extra_metrics() {
  local suite_args=()
  if [[ "${RUN_MOSES_METRICS}" == "1" ]]; then
    suite_args+=(--suite moses)
  fi
  if [[ "${RUN_GUACAMOL_METRICS}" == "1" ]]; then
    suite_args+=(--suite guacamol)
  fi
  if [[ "${#suite_args[@]}" == "0" ]]; then
    return 0
  fi

  local ref_mode=run
  local ref_args=()
  if [[ -n "${MOSES_TRAIN_FILE}" || -n "${MOSES_TEST_FILE}" || -n "${GUACAMOL_TRAIN_FILE}" ]]; then
    ref_mode=external
  fi
  if [[ -n "${MOSES_TRAIN_FILE}" ]]; then
    ref_args+=(--train-smiles "${MOSES_TRAIN_FILE}")
  elif [[ -n "${GUACAMOL_TRAIN_FILE}" ]]; then
    ref_args+=(--train-smiles "${GUACAMOL_TRAIN_FILE}")
  fi
  if [[ -n "${MOSES_TEST_FILE}" ]]; then
    ref_args+=(--test-smiles "${MOSES_TEST_FILE}")
  fi
  if [[ -n "${MOSES_TEST_SCAFFOLDS_FILE}" ]]; then
    ref_args+=(--test-scaffolds-smiles "${MOSES_TEST_SCAFFOLDS_FILE}")
  fi
  if [[ -n "${GUACAMOL_TRAIN_FILE}" ]]; then
    ref_args+=(--guacamol-train-smiles "${GUACAMOL_TRAIN_FILE}")
  fi

  "${PYBIN}" src/evaluate_molecule_benchmark_suites.py \
    --runs-root "${EXP_ROOT}" \
    --out-dir "${EXP_ROOT}/summary/molecule_benchmark_suites" \
    --reference-mode "${ref_mode}" \
    --n-samples "${BENCHMARK_METRIC_SAMPLES}" \
    --device cuda \
    --n-jobs 4 \
    --moses-backend "${MOSES_BACKEND}" \
    --local-moses-max-pairwise 1000 \
    --local-moses-max-reference 2000 \
    "${suite_args[@]}" \
    "${ref_args[@]}" 2>&1 | tee "${LOG_ROOT}/molecule_benchmark_suites.log"
}

maybe_install_packages
prepare_benchmark_data

if [[ "${RUN_QM9_NO_H}" == "1" ]]; then
  run_qm9_no_h
fi
if [[ "${RUN_QM9_H}" == "1" ]]; then
  run_qm9_h
fi
if [[ "${RUN_ZINC}" == "1" ]]; then
  run_zinc
fi
if [[ "${RUN_MOSES}" == "1" ]]; then
  run_moses_dataset
fi
if [[ "${RUN_GUACAMOL}" == "1" ]]; then
  run_guacamol_dataset
fi

"${PYBIN}" src/benchmark_report.py \
  --root "${EXP_ROOT}" \
  --out "${EXP_ROOT}/summary"

run_extra_metrics

echo
echo "MOSES/GuacaMol experiment runner finished."
echo "Root: ${EXP_ROOT}"
echo "Summary: ${EXP_ROOT}/summary/benchmark_summary.md"
echo "Extra metrics: ${EXP_ROOT}/summary/molecule_benchmark_suites/molecule_benchmark_metrics.md"
