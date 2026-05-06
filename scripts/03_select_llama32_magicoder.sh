#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_MODEL_DIR="${MODEL_NAME_OR_PATH:-.../models/Llama-3.2-3B}"
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

require_configured_path BASE_MODEL_DIR
mkdir -p "${OUTPUT_DIR}"

fix_tok_dir () {
  local d="$1"
  [[ -d "$d" ]] || return 0
  echo "[INFO] fix tokenizer in $d"
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model; do
    [[ -f "${BASE_MODEL_DIR}/${f}" ]] && cp -f "${BASE_MODEL_DIR}/${f}" "${d}/${f}" || true
  done
  [[ -f "${d}/tokenizer_config.json" ]] && \
    sed -i 's/"tokenizer_class":[[:space:]]*"TokenizersBackend"/"tokenizer_class":"PreTrainedTokenizerFast"/g' "${d}/tokenizer_config.json" || true
}

fix_tok_dir "${OUTPUT_DIR}"
for wd in "${OUTPUT_DIR}/warmup" "${OUTPUT_DIR}/warmup_model" "${OUTPUT_DIR}/stage1_warmup" "${OUTPUT_DIR}/stage1_warmup_model" "${OUTPUT_DIR}/stage1/warmup_model"; do
  fix_tok_dir "${wd}"
done
find "${OUTPUT_DIR}" -maxdepth 3 -type f -name "tokenizer_config.json" | while read -r f; do
  fix_tok_dir "$(dirname "$f")"
done

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/run_dualsft.py" \
  --stage select \
  --model_name_or_path "${BASE_MODEL_DIR}" \
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
  --eval_batch_size 4 \
  --score_batch_size 8 \
  --lambda_new 1.0 \
  --lambda_prior 0.2 \
  --tau 2.0 \
  --confidence_mode max_prob \
  --score_method ghost_linear \
  --data_rerank_topm 30000 \
  --score_dtype float16 \
  --save_param_scores \
  --save_data_scores \
  --quota_enable \
  --quota_layer_min_ratio 0.20 \
  --quota_module_min_ratio 0.20 \
  --bf16
