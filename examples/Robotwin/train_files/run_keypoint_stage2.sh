#!/bin/bash
# ============================================================
#  Stage-2 Full Joint Training  (8 GPU, DeepSpeed ZeRO-2)
#
#  - All modules trainable: QwenVL backbone, action model, keypoint head
#  - Joint loss: action_loss + keypoint_loss
#  - Keypoint gradient updates Qwen backbone intermediate and earlier layers
#  - Recommended: run Stage-1 first and use its full_model_final.pt here
#
#  Output:
#    results/Checkpoints/${run_id}/checkpoints/steps_*_pytorch_model.pt
#    results/Checkpoints/${run_id}/final_model/pytorch_model.pt
# ============================================================

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000   # 1 hour timeout

###########################################################################################
# === Please modify the following paths according to your environment ===
config_yaml=./examples/Robotwin/train_files/starvla_keypoint_stage2.yaml
run_root_dir=./results/Checkpoints
run_id=keypoint_stage2_full_training

data_dir=/home/caslx/Robotics/RoboTwin/data/aloha-agilex_randomized_500
base_vlm=Qwen/Qwen3-VL-4B-Instruct   # or local path

# Full model from Stage-1 (includes warm keypoint_head)
pretrained_checkpoint=./results/Checkpoints/keypoint_stage1_freeze_backbone/full_model_final.pt

# Alternative (if Stage-1 not done yet): original VLA ckpt + keypoint_head separately
# pretrained_checkpoint=./checkpoints/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt
# keypoint_head_checkpoint=./results/Checkpoints/keypoint_stage1_freeze_backbone/keypoint_head_final.pt
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 8 \
    starVLA/training/train_keypoint_stage2.py \
    --config_yaml ${config_yaml} \
    --framework.qwenvl.base_vlm ${base_vlm} \
    --datasets.keypoint_data.data_dir ${data_dir} \
    --datasets.keypoint_data.per_device_batch_size 4 \
    --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
    --trainer.max_train_steps 50000 \
    --trainer.save_interval 5000 \
    --trainer.logging_frequency 100 \
    --run_root_dir ${run_root_dir} \
    --run_id ${run_id}
    # Uncomment for Stage-1 head + original VLA ckpt (alternative init):
    # --trainer.keypoint_head_checkpoint ${keypoint_head_checkpoint}
    # --wandb_project starVLA_Keypoint \
    # --wandb_entity your_entity


##### Multi-Server Multi-GPU (SLURM) template #####
# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port $MASTER_PORT \
#   --machine_rank $SLURM_PROCID \
#   --num_machines $SLURM_NNODES \
#   --num_processes=${TOTAL_GPUS} \
#   starVLA/training/train_keypoint_stage2.py \
#   --config_yaml ${config_yaml} \
#   ...
##### Multi-Server Multi-GPU template end #####
