#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-.../models/Llama-3.2-3B}"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/train/datasets/Magicoder/magicoder_train.jsonl}"
VALIDATION_FILE="${VALIDATION_FILE:-${PROJECT_ROOT}/train/datasets/Magicoder/magicoder_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/dualsft_llama32_3b_magicoder}"

require_configured_path() {
  local name="$1"
  local value="${!name}"
  if [[ "${value}" == *"..."* ]]; then
    echo "[ERROR] Replace the placeholder in ${name}: ${value}"
    exit 1
  fi
}

require_configured_path MODEL_NAME_OR_PATH

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/run_dualsft.py" \
  --stage warmup \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --validation_file "${VALIDATION_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --prompt_field instruction \
  --response_field response \
  --max_length 1024 \
  --seed 42 \
  --device cuda \
  --teacher_device cuda \
  --num_workers 8 \
  --prefetch_factor 2 \
  --learning_rate 2e-5 \
  --warmup_epochs 1 \
  --warmup_batch_size 4 \
  --warmup_grad_accum_steps 4 \
  --warmup_ratio 0.1 \
  --anchor_ratio 0.02 \
  --score_pool_ratio 1 \
  --score_dtype float16 \
  --quota_enable \
  --quota_layer_min_ratio 0.20 \
  --quota_module_min_ratio 0.20 \
  --bf16
