#!/bin/bash
# ============================================================
#  Stage-1 Keypoint Head Training  (Single GPU)
#
#  - Freezes QwenVL backbone and action model
#  - Trains only the KeypointPredHead (CrossAttentionPool + 2 MLPs)
#  - Loads pretrained VLA weights from Qwen3-VL-OFT-RoboTwin2-All
#
#  Output:
#    results/Checkpoints/${run_id}/keypoint_head_final.pt  ← keypoint head weights
#    results/Checkpoints/${run_id}/full_model_final.pt     ← full model (use for stage 2)
# ============================================================

###########################################################################################
# === Please modify the following paths according to your environment ===
config_yaml=./examples/Robotwin/train_files/starvla_keypoint_stage1.yaml
run_root_dir=./results/Checkpoints
run_id=keypoint_stage1_freeze_backbone

data_dir=/home/caslx/Robotics/RoboTwin/data/aloha-agilex_randomized_500
base_vlm=Qwen/Qwen3-VL-4B-Instruct   # or local path e.g. ./playground/Pretrained_models/Qwen3-VL-4B-Instruct
pretrained_checkpoint=./checkpoints/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt

# GPU selection (single GPU only)
export CUDA_VISIBLE_DEVICES=0
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

python starVLA/training/train_keypoint_stage1.py \
    --config_yaml ${config_yaml} \
    --framework.qwenvl.base_vlm ${base_vlm} \
    --datasets.keypoint_data.data_dir ${data_dir} \
    --datasets.keypoint_data.per_device_batch_size 4 \
    --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
    --trainer.max_train_steps 10000 \
    --trainer.save_interval 2000 \
    --trainer.logging_frequency 50 \
    --run_root_dir ${run_root_dir} \
    --run_id ${run_id}
    # --wandb_project starVLA_Keypoint \
    # --wandb_entity your_entity
