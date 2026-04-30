#!/bin/bash
# Inference server startup script for NeuroVLA on TeleAvatar dual-arm robot.
#
# This script loads a trained checkpoint and starts a WebSocket policy server.
# The server accepts observation batches and returns normalized action chunks.
#
# Action output layout (20-dim):
#   [pos_L(3) | rot6d_L(6) | grip_L(1) | pos_R(3) | rot6d_R(6) | grip_R(1)]
# State input layout (18-dim):
#   [pos_L(3) | rot6d_L(6) | pos_R(3) | rot6d_R(6)]
#
# Run from project root:
#   bash examples/TeleAvatar/start_server.sh
# Or pass checkpoint path as first argument:
#   bash examples/TeleAvatar/start_server.sh /path/to/checkpoint.pt

# ── User settings ──────────────────────────────────────────────────────────────

# Path to the trained checkpoint (.pt file or directory containing pytorch_model.pt)
# Override via first positional argument: bash start_server.sh /your/ckpt.pt
CKPT_PATH=${1:-/media/caslx/1635-A2D7/weight/neurovla_pick_marker/final_model/pytorch_model.pt}

# WebSocket port the inference server will listen on
PORT=10093

# GPU to use for inference (single GPU recommended)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Python interpreter (activate your conda env before running, or set this explicitly)
# e.g. PYTHON=~/miniconda3/envs/neurovla/bin/python
PYTHON=${PYTHON:-python}

# ── Launch ─────────────────────────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "============================================"
echo "  NeuroVLA TeleAvatar Inference Server"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  Port:       ${PORT}"
echo "  GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

${PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${PORT}" \
    --use_bf16
