#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-.../models/Gemma-3-4B}"
DATASET_DIR="${DATASET_DIR:-${TRAIN_ROOT}/datasets/Magicoder}"
DATASET_NAME="${DATASET_NAME:-data-evol_instruct-decontaminated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_ROOT}/outputs/gemma_full_sft}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-.../LlamaFactory-main/examples/deepspeed/ds_z2_config.json}"

if [[ "${MODEL_NAME_OR_PATH}" == *"..."* ]]; then
  echo "[ERROR] Replace the placeholder in MODEL_NAME_OR_PATH: ${MODEL_NAME_OR_PATH}"
  exit 1
fi
if [[ "${DEEPSPEED_CONFIG}" == *"..."* ]]; then
  echo "[ERROR] Replace the placeholder in DEEPSPEED_CONFIG: ${DEEPSPEED_CONFIG}"
  exit 1
fi

MODEL_TAG="$(basename "${MODEL_NAME_OR_PATH}")"

llamafactory-cli train \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --stage sft \
    --do_train True \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --preprocessing_num_workers 32 \
    --finetuning_type full \
    --template gemma \
    --flash_attn auto \
    --dataset_dir "${DATASET_DIR}" \
    --dataset "${DATASET_NAME}" \
    --cutoff_len 1024 \
    --learning_rate 2e-05 \
    --num_train_epochs 3.0 \
    --max_samples 1000000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir "${OUTPUT_ROOT}/${MODEL_TAG}/${DATASET_NAME}" \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch
