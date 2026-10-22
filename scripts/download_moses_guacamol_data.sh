#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATA_DIR=${DATA_DIR:-datasets/molecule_benchmarks}
MOSES_DIR="${DATA_DIR}/moses"
GUACAMOL_DIR="${DATA_DIR}/guacamol"

mkdir -p "${MOSES_DIR}" "${GUACAMOL_DIR}"

download_if_missing() {
  local url="$1"
  local out="$2"
  if [[ -s "${out}" ]]; then
    echo "Found ${out}; skipping download."
    return 0
  fi
  echo "Downloading ${out}"
  curl -L --fail --retry 3 --retry-delay 5 --output "${out}.tmp" "${url}"
  mv "${out}.tmp" "${out}"
}

count_lines() {
  local path="$1"
  if [[ -s "${path}" ]]; then
    wc -l < "${path}" | tr -d ' '
  else
    printf '0'
  fi
}

download_if_missing \
  "https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv" \
  "${MOSES_DIR}/train.csv"
download_if_missing \
  "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv" \
  "${MOSES_DIR}/test.csv"
download_if_missing \
  "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv" \
  "${MOSES_DIR}/test_scaffolds.csv"

download_if_missing \
  "https://ndownloader.figshare.com/files/13612760" \
  "${GUACAMOL_DIR}/guacamol_v1_train.smiles"
download_if_missing \
  "https://ndownloader.figshare.com/files/13612766" \
  "${GUACAMOL_DIR}/guacamol_v1_valid.smiles"
download_if_missing \
  "https://ndownloader.figshare.com/files/13612757" \
  "${GUACAMOL_DIR}/guacamol_v1_test.smiles"
download_if_missing \
  "https://ndownloader.figshare.com/files/13612745" \
  "${GUACAMOL_DIR}/guacamol_v1_all.smiles"

echo
echo "Prepared MOSES / GuacaMol data under ${DATA_DIR}"
echo "MOSES train lines:          $(count_lines "${MOSES_DIR}/train.csv")"
echo "MOSES test lines:           $(count_lines "${MOSES_DIR}/test.csv")"
echo "MOSES scaffold-test lines:  $(count_lines "${MOSES_DIR}/test_scaffolds.csv")"
echo "GuacaMol train lines:       $(count_lines "${GUACAMOL_DIR}/guacamol_v1_train.smiles")"
echo "GuacaMol valid lines:       $(count_lines "${GUACAMOL_DIR}/guacamol_v1_valid.smiles")"
echo "GuacaMol test lines:        $(count_lines "${GUACAMOL_DIR}/guacamol_v1_test.smiles")"
echo
echo "Use:"
echo "  MOSES_SMILES_FILE=${MOSES_DIR}/train.csv"
echo "  MOSES_TRAIN_FILE=${MOSES_DIR}/train.csv"
echo "  MOSES_TEST_FILE=${MOSES_DIR}/test.csv"
echo "  MOSES_TEST_SCAFFOLDS_FILE=${MOSES_DIR}/test_scaffolds.csv"
echo "  GUACAMOL_TRAIN_FILE=${GUACAMOL_DIR}/guacamol_v1_train.smiles"
echo "  GUACAMOL_TEST_FILE=${GUACAMOL_DIR}/guacamol_v1_test.smiles"
