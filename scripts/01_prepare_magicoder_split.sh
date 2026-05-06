#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/train/datasets/Magicoder}"
INPUT_FILE="${INPUT_FILE:-${DATA_DIR}/data-evol_instruct-decontaminated.json}"
TRAIN_OUT="${TRAIN_OUT:-${DATA_DIR}/magicoder_train.jsonl}"
VAL_OUT="${VAL_OUT:-${DATA_DIR}/magicoder_val.jsonl}"
SEED="${SEED:-42}"
VAL_SIZE="${VAL_SIZE:-1024}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/prepare_magicoder_split.py" \
  --input "${INPUT_FILE}" \
  --train_out "${TRAIN_OUT}" \
  --val_out "${VAL_OUT}" \
  --seed "${SEED}" \
  --val_size "${VAL_SIZE}"

echo "[done] split files generated."
