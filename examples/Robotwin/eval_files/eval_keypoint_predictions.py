#!/usr/bin/env python3
"""
Evaluate trained keypoint prediction head against ground truth.

Loads a QwenOFTKeypoint checkpoint, runs model.predict_keypoints() on
training-data episodes, then reports per-arm contact accuracy and xyz
Euclidean error for active arms.

Usage:
    python examples/Robotwin/eval_files/eval_keypoint_predictions.py \\
        --config_yaml examples/Robotwin/train_files/starvla_keypoint_stage1.yaml \\
        --checkpoint  results/Checkpoints/keypoint_stage1_freeze_backbone/keypoint_head_final.pt \\
        [--full_model_checkpoint  path/to/full_model_final.pt]  \\
        [--num_samples 100]   \\
        [--batch_size 4]      \\
        [--frame_strategy contact]   \\
        [--seed 0]

Notes:
    --checkpoint loads *only* the keypoint_head weights (output of stage-1).
    --full_model_checkpoint loads the full model state dict.
    If --full_model_checkpoint is given, --checkpoint is ignored.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Make project root importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from starVLA.dataloader.keypoint_hdf5_dataset import (
    KeypointHDF5Dataset,
    collate_keypoint_batch,
)
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ──────────────────────────────────────────────────────────────────────────────
#  Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_full_model(model, ckpt_path: str) -> None:
    print(f"Loading full model checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"  Missing keys : {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected   : {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print("  Full model loaded.")


def _load_keypoint_head_only(model, ckpt_path: str) -> None:
    print(f"Loading keypoint_head checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.keypoint_head.load_state_dict(ckpt, strict=True)
    if missing:
        print(f"  Missing keys : {missing}")
    if unexpected:
        print(f"  Unexpected   : {unexpected}")
    print("  keypoint_head loaded.")


# ──────────────────────────────────────────────────────────────────────────────
#  Metrics accumulator
# ──────────────────────────────────────────────────────────────────────────────

class ArmMetrics:
    """Accumulates per-arm contact classification and xyz error."""

    def __init__(self, name: str):
        self.name = name
        self.tp = self.fp = self.tn = self.fn = 0
        self.xyz_errors: list = []   # Euclidean L2 (metres) for active arms
        self.xyz_per_axis: list = [] # (3,) absolute per-axis errors, active arms only

    def update(
        self,
        pred_contact: np.ndarray,  # (B,) bool
        pred_xyz: np.ndarray,      # (B, 3)
        gt_contact: np.ndarray,    # (B,) bool
        gt_xyz: np.ndarray,        # (B, 3)
    ):
        for pc, px, gc, gx in zip(pred_contact, pred_xyz, gt_contact, gt_xyz):
            if gc and pc:
                self.tp += 1
            elif gc and not pc:
                self.fn += 1
            elif not gc and pc:
                self.fp += 1
            else:
                self.tn += 1

            if gc:  # only evaluate xyz error for active (ground-truth contact) arms
                err = np.abs(px - gx)
                self.xyz_per_axis.append(err)
                self.xyz_errors.append(float(np.linalg.norm(px - gx)))

    @property
    def n_contact(self):
        return self.tp + self.fn

    @property
    def n_total(self):
        return self.tp + self.fp + self.tn + self.fn

    def contact_accuracy(self):
        return (self.tp + self.tn) / max(self.n_total, 1)

    def contact_precision(self):
        return self.tp / max(self.tp + self.fp, 1)

    def contact_recall(self):
        return self.tp / max(self.tp + self.fn, 1)

    def contact_f1(self):
        p, r = self.contact_precision(), self.contact_recall()
        return 2 * p * r / max(p + r, 1e-9)

    def mean_l2_m(self):
        return float(np.mean(self.xyz_errors)) if self.xyz_errors else float("nan")

    def mean_l2_cm(self):
        return self.mean_l2_m() * 100

    def mean_per_axis_cm(self):
        if not self.xyz_per_axis:
            return np.array([float("nan")] * 3)
        return np.mean(self.xyz_per_axis, axis=0) * 100   # m → cm

    def report(self):
        ax = self.mean_per_axis_cm()
        lines = [
            f"  {self.name} arm  ({self.n_contact} active / {self.n_total} total samples)",
            f"    Contact  acc={self.contact_accuracy():.3f}  "
            f"prec={self.contact_precision():.3f}  "
            f"recall={self.contact_recall():.3f}  "
            f"F1={self.contact_f1():.3f}",
            f"    XYZ L2   mean={self.mean_l2_cm():.2f} cm  "
            f"(x={ax[0]:.2f} y={ax[1]:.2f} z={ax[2]:.2f} cm)",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(cfg, model, loader, device, contact_threshold: float):
    model.eval()
    left_m  = ArmMetrics("left")
    right_m = ArmMetrics("right")

    for batch in tqdm(loader, desc="Evaluating"):
        result = model.predict_keypoints(batch, contact_threshold=contact_threshold)

        gt_left_contact  = np.array([e["left_contact"]  > 0.5 for e in batch])
        gt_right_contact = np.array([e["right_contact"] > 0.5 for e in batch])
        gt_left_xyz      = np.stack([e["left_kp"]  for e in batch])   # (B, 3)
        gt_right_xyz     = np.stack([e["right_kp"] for e in batch])   # (B, 3)

        left_m.update(
            pred_contact=result["left_contact"],
            pred_xyz=result["left_xyz"],
            gt_contact=gt_left_contact,
            gt_xyz=gt_left_xyz,
        )
        right_m.update(
            pred_contact=result["right_contact"],
            pred_xyz=result["right_xyz"],
            gt_contact=gt_right_contact,
            gt_xyz=gt_right_xyz,
        )

    return left_m, right_m


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to keypoint_head_final.pt (head weights only)")
    parser.add_argument("--full_model_checkpoint", default=None,
                        help="Path to full_model_final.pt (overrides --checkpoint)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of episodes to evaluate (default: all)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--frame_strategy", default="contact",
                        choices=["contact", "random", "first"],
                        help="Frame sampling strategy (default: contact — deterministic)")
    parser.add_argument("--contact_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args, clipargs = parser.parse_known_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = apply_config_compat(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    kp_cfg = cfg.datasets.keypoint_data
    dataset = KeypointHDF5Dataset(
        data_dir=kp_cfg.data_dir,
        image_size=tuple(kp_cfg.get("image_size", [224, 224])),
        action_horizon=int(cfg.framework.action_model.action_horizon),
        cameras=list(kp_cfg.get("cameras", ["cam_high", "cam_left_wrist", "cam_right_wrist"])),
        frame_sample_strategy=args.frame_strategy,
        stats_path=kp_cfg.get("stats_path", None),
    )

    if args.num_samples is not None and args.num_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.num_samples, replace=False).tolist()
        dataset = Subset(dataset, sorted(indices))
        print(f"Evaluating {args.num_samples} episodes (randomly sampled).")
    else:
        print(f"Evaluating all {len(dataset)} episodes.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_keypoint_batch,
        num_workers=2,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # build_framework 已从 HuggingFace/本地路径加载 Qwen backbone，无需再 load 预训练 VLA
    model = build_framework(cfg)

    if args.full_model_checkpoint:
        _load_full_model(model, args.full_model_checkpoint)
    elif args.checkpoint:
        _load_keypoint_head_only(model, args.checkpoint)
    else:
        print("WARNING: no --checkpoint or --full_model_checkpoint provided — "
              "evaluating with randomly initialized keypoint head.")

    model = model.to(device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    left_m, right_m = evaluate(cfg, model, loader, device, args.contact_threshold)

    # ── Report ────────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Keypoint Prediction Evaluation Results")
    print(f"  frame_strategy={args.frame_strategy}  "
          f"contact_threshold={args.contact_threshold}")
    print(sep)
    print(left_m.report())
    print()
    print(right_m.report())
    print(sep)

    # Combined xyz error across both arms (active only)
    all_l2 = left_m.xyz_errors + right_m.xyz_errors
    if all_l2:
        print(f"\n  Combined active-arm L2 error : "
              f"mean={np.mean(all_l2)*100:.2f} cm  "
              f"median={np.median(all_l2)*100:.2f} cm  "
              f"p90={np.percentile(all_l2,90)*100:.2f} cm")
    print()


if __name__ == "__main__":
    main()
