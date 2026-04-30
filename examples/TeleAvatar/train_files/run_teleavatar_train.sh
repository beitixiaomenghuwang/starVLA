#!/bin/bash
# Training script for starVLA on TeleAvatar pick-marker task
# Run from project root:
#   bash examples/TeleAvatar/train_files/run_teleavatar_train.sh

# ── User settings ──────────────────────────────────────────────────────────────
# Local path to Qwen3-VL-4B-Instruct weights
MODEL_PATH=/DATA/disk0/xueyang/model/Qwen3-VL-4B-Instruct

# Dataset directory. Its basename must match the teleavatar_pick_marker mixture.
DATA_ROOT_DIR=/DATA/disk0/xueyang/Data/pick_marker_put_into_cup_20251113_with_progress

RUN_ROOT_DIR=/DATA/disk0/xueyang/model/starvla_teleavatar
RUN_ID=teleavatar_pick_marker_$(date +%m%d_%H%M)

# GPU configuration (adjust CUDA_VISIBLE_DEVICES and --num_processes to your setup)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# W&B settings (optional – set WANDB_MODE=disabled to skip)
# export WANDB_MODE=disabled
WANDB_PROJECT=starvla_teleavatar
WANDB_ENTITY=gxy1000h-jilin-university   # ← replace or set via env var

# ── Advanced overrides (can also be edited in the YAML) ────────────────────────
PER_DEVICE_BS=8
MAX_TRAIN_STEPS=30000
SAVE_INTERVAL=5000
GRADIENT_ACC=1

# ── Setup ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR=${RUN_ROOT_DIR}/${RUN_ID}
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"   # archive this script with the run

echo "============================================"
echo "  Run ID: ${RUN_ID}"
echo "  Model:  ${MODEL_PATH}"
echo "  Data:   ${DATA_ROOT_DIR}"
echo "  GPUs:   ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} processes)"
echo "============================================"

# ── Launch ─────────────────────────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH}"

accelerate launch \
  --num_processes ${NUM_GPUS} \
  --main_process_port 29501 \
  --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/TeleAvatar/train_files/QwenMLP_teleavatar.yaml \
  --framework.qwenvl.base_vlm "${MODEL_PATH}" \
  --datasets.vla_data.data_root_dir "$(dirname "${DATA_ROOT_DIR}")" \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BS} \
  --trainer.max_train_steps ${MAX_TRAIN_STEPS} \
  --trainer.save_interval ${SAVE_INTERVAL} \
  --trainer.gradient_accumulation_steps ${GRADIENT_ACC} \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  "$@"
