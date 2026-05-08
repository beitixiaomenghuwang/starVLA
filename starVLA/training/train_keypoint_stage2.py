#!/usr/bin/env python3
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").

"""
Stage-2 full joint training — action loss + keypoint loss, all modules trainable.

Uses Accelerate + DeepSpeed ZeRO-2 for 8-GPU distributed training.
Keypoint loss gradient flows through the QwenVL backbone, updating intermediate
and earlier layers (as required).

Recommended starting point: Stage-1 full_model_final.pt, which already has
the pretrained VLA weights AND a warm-started keypoint_head.

Usage (8 GPUs):
    accelerate launch \\
        --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \\
        --num_processes 8 \\
        starVLA/training/train_keypoint_stage2.py \\
        --config_yaml examples/Robotwin/train_files/starvla_keypoint_stage2.yaml \\
        [--key value ...]
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from starVLA.dataloader.keypoint_hdf5_dataset import (
    KeypointHDF5Dataset,
    collate_keypoint_batch,
)
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
logger = get_logger(__name__)


def main(cfg) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    set_seed(int(cfg.get("seed", 42)) + rank)

    output_dir = Path(cfg.output_dir)
    if accelerator.is_main_process:
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    dist.barrier() if dist.is_initialized() else None

    # ── Dataset ──────────────────────────────────────────────────────────
    kp_cfg = cfg.datasets.keypoint_data
    dataset = KeypointHDF5Dataset(
        data_dir=kp_cfg.data_dir,
        image_size=tuple(kp_cfg.get("image_size", [224, 224])),
        action_horizon=int(cfg.framework.action_model.action_horizon),
        cameras=list(kp_cfg.get("cameras", ["cam_high", "cam_left_wrist", "cam_right_wrist"])),
        frame_sample_strategy=kp_cfg.get("frame_sample_strategy", "random"),
        stats_path=kp_cfg.get("stats_path", None),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(kp_cfg.per_device_batch_size),
        shuffle=True,
        collate_fn=collate_keypoint_batch,
        num_workers=4,
        pin_memory=True,
    )
    accelerator.print(
        f"Dataset: {len(dataset)} episodes  |  "
        f"per-device batch={kp_cfg.per_device_batch_size}  |  "
        f"total batch={kp_cfg.per_device_batch_size * accelerator.num_processes}"
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = build_framework(cfg)

    # Load checkpoint (Stage-1 full_model_final.pt recommended for warm keypoint_head)
    pretrained_ckpt = cfg.trainer.get("pretrained_checkpoint", None)
    if pretrained_ckpt:
        model = TrainerUtils.load_pretrained_backbones(model, pretrained_ckpt, reload_modules=None)
        accelerator.print(f"Loaded checkpoint: {pretrained_ckpt}")

    # Optional: additionally load just the keypoint_head from Stage-1 if using
    #           the original VLA checkpoint as pretrained_checkpoint
    kp_head_ckpt = cfg.trainer.get("keypoint_head_checkpoint", None)
    if kp_head_ckpt:
        kp_state = torch.load(kp_head_ckpt, map_location="cpu")
        model.keypoint_head.load_state_dict(kp_state, strict=True)
        accelerator.print(f"Loaded keypoint_head from: {kp_head_ckpt}")

    # ── Optimizer ────────────────────────────────────────────────────────
    # Builds per-module LR groups: qwen_vl_interface, action_model, keypoint_head, base
    # Add "keypoint_head" to cfg.trainer.learning_rate to give it a custom LR
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(cfg.trainer.learning_rate.base),
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=float(cfg.trainer.optimizer.weight_decay),
        eps=float(cfg.trainer.optimizer.eps),
    )

    total_steps = int(cfg.trainer.max_train_steps)
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(cfg.trainer.num_warmup_steps),
        num_training_steps=total_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    # ── Accelerate prepare ───────────────────────────────────────────────
    accelerator.dataloader_config.dispatch_batches = False
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # ── WandB ────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        wandb.init(
            name=cfg.run_id,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            dir=str(output_dir / "wandb"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Training loop ────────────────────────────────────────────────────
    completed_steps = 0
    data_iter = iter(loader)
    grad_clip = cfg.trainer.get("gradient_clipping", None)
    log_freq = int(cfg.trainer.logging_frequency)
    save_interval = int(cfg.trainer.save_interval)

    pbar = tqdm(
        range(total_steps),
        desc="[Stage-2] full joint training",
        disable=not accelerator.is_local_main_process,
    )

    while completed_steps < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        t0 = time.perf_counter()

        with accelerator.accumulate(model):
            optimizer.zero_grad()

            out = model.forward(batch)
            total_loss   = out["total_loss"]
            action_loss  = out["action_loss"]
            kp_loss      = out["keypoint_loss"]
            xyz_loss     = out["xyz_loss"]
            contact_loss = out["contact_loss"]

            accelerator.backward(total_loss)

            if grad_clip is not None:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            lr_scheduler.step()

        if accelerator.sync_gradients:
            completed_steps += 1
            pbar.update(1)
            pbar.set_postfix(
                {
                    "act": f"{action_loss.item():.4f}",
                    "kp": f"{kp_loss.item():.4f}",
                }
            )

        if completed_steps % log_freq == 0 and accelerator.is_main_process:
            wandb.log(
                {
                    "action_loss":   action_loss.item(),
                    "keypoint_loss": kp_loss.item(),
                    "xyz_loss":      xyz_loss.item(),
                    "contact_loss":  contact_loss.item(),
                    "total_loss":    total_loss.item(),
                    "lr":            lr_scheduler.get_last_lr()[0],
                    "step_time_s":   time.perf_counter() - t0,
                },
                step=completed_steps,
            )

        if completed_steps > 0 and completed_steps % save_interval == 0:
            if accelerator.is_main_process:
                ckpt_path = (
                    output_dir / "checkpoints" / f"steps_{completed_steps}_pytorch_model.pt"
                )
                state_dict = accelerator.get_state_dict(model)
                torch.save(state_dict, ckpt_path)
                # Append to summary.jsonl
                with open(output_dir / "summary.jsonl", "a") as f:
                    f.write(json.dumps({"steps": completed_steps}) + "\n")
                logger.info(f"✅ Checkpoint saved → {ckpt_path}")
            if dist.is_initialized():
                dist.barrier()

        if completed_steps >= total_steps:
            break

    # ── Final save ───────────────────────────────────────────────────────
    if accelerator.is_main_process:
        final_dir = output_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        torch.save(accelerator.get_state_dict(model), final_dir / "pytorch_model.pt")
        logger.info(f"✅ Final model saved → {final_dir / 'pytorch_model.pt'}")
        wandb.finish()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-2 full joint training (8 GPU)")
    parser.add_argument("--config_yaml", type=str, required=True)
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)

    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)

    main(cfg)
