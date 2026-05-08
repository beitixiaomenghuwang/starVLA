#!/usr/bin/env python3
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").

"""
Stage-1 keypoint head training — single GPU, frozen backbone.

Only trains KeypointPredHead (CrossAttentionPool + two MLPs).
The QwenVL backbone and action model are fully frozen.
The pretrained VLA checkpoint provides the frozen feature extractor.

Usage:
    python starVLA/training/train_keypoint_stage1.py \\
        --config_yaml examples/Robotwin/train_files/starvla_keypoint_stage1.yaml \\
        [--key value ...]
"""

import argparse
import os
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from starVLA.dataloader.keypoint_hdf5_dataset import (
    KeypointHDF5Dataset,
    collate_keypoint_batch,
)
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_checkpoint_single_gpu(model, ckpt_path: str) -> None:
    """Load checkpoint without torch.distributed (single-GPU safe)."""
    print(f"📦 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    kp_missing = [k for k in missing if "keypoint_head" in k]
    other_missing = [k for k in missing if "keypoint_head" not in k]
    print(f"   Missing  (keypoint_head — expected, newly added): {len(kp_missing)}")
    if other_missing:
        print(f"   Missing  (other): {other_missing[:5]}")
    if unexpected:
        print(f"   Unexpected keys: {unexpected[:5]}")
    print("✅ Checkpoint loaded")


def _freeze_backbone(model) -> None:
    """Freeze VLM and action model; keep keypoint_head trainable.

    After this call:
        qwen_vl_interface  → all params frozen (requires_grad=False)
        action_model       → all params frozen (requires_grad=False)
        keypoint_head      → all params trainable (requires_grad=True)
    """
    for p in model.qwen_vl_interface.parameters():
        p.requires_grad_(False)
    for p in model.action_model.parameters():
        p.requires_grad_(False)
    # keypoint_head params remain at their default requires_grad=True


def _print_freeze_status(model) -> None:
    """Print per-top-module freeze status so you can verify what's frozen."""
    from collections import defaultdict

    status: dict = defaultdict(lambda: {"frozen": 0, "trainable": 0})
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        if p.requires_grad:
            status[top]["trainable"] += p.numel()
        else:
            status[top]["frozen"] += p.numel()

    print("\n🔒 Module freeze status:")
    print(f"  {'Module':<30} {'Frozen params':>15} {'Trainable params':>17}")
    print("  " + "-" * 64)
    for mod, counts in status.items():
        frozen_m = counts["frozen"] / 1e6
        train_m  = counts["trainable"] / 1e6
        state = "❄️  FROZEN" if counts["trainable"] == 0 else (
                "🔥 TRAINABLE" if counts["frozen"] == 0 else "⚠️  MIXED")
        print(f"  {mod:<30} {frozen_m:>12.2f}M {train_m:>14.2f}M  {state}")
    print()


def _count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train(cfg) -> None:
    output_dir = Path(cfg.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"Dataset: {len(dataset)} episodes  |  batch_size={kp_cfg.per_device_batch_size}")

    # ── Model ────────────────────────────────────────────────────────────
    model = build_framework(cfg)

    # Load pretrained VLA checkpoint (strict=False: keypoint_head not in checkpoint)
    pretrained_ckpt = cfg.trainer.get("pretrained_checkpoint", None)
    if pretrained_ckpt:
        _load_checkpoint_single_gpu(model, pretrained_ckpt)

    _freeze_backbone(model)
    _print_freeze_status(model)   # ← shows exactly which modules are frozen
    model = model.to(device)

    trainable, total = _count_params(model)
    print(
        f"Parameters: {total / 1e6:.1f}M total, "
        f"{trainable / 1e6:.3f}M trainable ({100 * trainable / total:.2f}%)\n"
    )

    # ── Optimizer (keypoint_head params only) ────────────────────────────
    kp_params = list(model.keypoint_head.parameters())
    lr = cfg.trainer.learning_rate.get("keypoint_head", 1e-4)
    optimizer = AdamW(
        kp_params,
        lr=lr,
        betas=tuple(cfg.trainer.optimizer.get("betas", [0.9, 0.95])),
        weight_decay=float(cfg.trainer.optimizer.get("weight_decay", 1e-6)),
        eps=float(cfg.trainer.optimizer.get("eps", 1e-8)),
    )
    total_steps = int(cfg.trainer.max_train_steps)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(cfg.trainer.get("num_warmup_steps", 200)),
        num_training_steps=total_steps,
    )

    # ── WandB ────────────────────────────────────────────────────────────
    wandb.init(
        name=cfg.run_id,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        dir=str(output_dir / "wandb"),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    data_iter = iter(loader)
    grad_clip = cfg.trainer.get("gradient_clipping", None)
    log_freq = int(cfg.trainer.logging_frequency)
    save_interval = int(cfg.trainer.save_interval)

    pbar = tqdm(range(total_steps), desc="[Stage-1] keypoint head")

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        optimizer.zero_grad()

        # action_loss_weight=0 → only keypoint loss propagates gradients
        out = model.forward(batch)
        kp_loss      = out["keypoint_loss"]
        xyz_loss     = out["xyz_loss"]
        contact_loss = out["contact_loss"]

        kp_loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(kp_params, grad_clip)

        optimizer.step()
        lr_scheduler.step()

        if step % log_freq == 0:
            metrics = {
                "keypoint_loss": kp_loss.item(),
                "xyz_loss":      xyz_loss.item(),
                "contact_loss":  contact_loss.item(),
                "lr":            lr_scheduler.get_last_lr()[0],
            }
            wandb.log(metrics, step=step)
            pbar.set_postfix({
                "xyz": f"{xyz_loss.item():.4f}",
                "ctt": f"{contact_loss.item():.4f}",
            })

        if step > 0 and step % save_interval == 0:
            ckpt_path = output_dir / "checkpoints" / f"step_{step}_keypoint_head.pt"
            torch.save(model.keypoint_head.state_dict(), ckpt_path)
            print(f"\n✅ [Step {step}] keypoint_head saved → {ckpt_path}")

    # ── Final save ───────────────────────────────────────────────────────
    final_kp_path = output_dir / "keypoint_head_final.pt"
    torch.save(model.keypoint_head.state_dict(), final_kp_path)

    # Also save full model state dict for Stage-2 warm-start
    full_model_path = output_dir / "full_model_final.pt"
    torch.save(model.state_dict(), full_model_path)

    print(f"\n✅ Training complete.")
    print(f"   keypoint_head → {final_kp_path}")
    print(f"   full model    → {full_model_path}  (use as Stage-2 pretrained_checkpoint)")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-1 keypoint head training (single GPU)")
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)

    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)

    train(cfg)
