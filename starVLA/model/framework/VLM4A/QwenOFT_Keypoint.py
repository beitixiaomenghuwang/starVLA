# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").

"""
QwenOFTKeypoint — QwenOFT extended with a unified keypoint prediction head.

The head jointly predicts:
  - 3D contact xyz for left and right arms (shared MLP)
  - Per-arm contact probability (binary classification)

Loss design:
  contact_loss = BCE(contact_logit, contact_label)   — all arms always supervised
  xyz_loss     = contact_label * L1(xyz_pred, xyz_gt) — active arms only (masked)

  where: contact_label=1 / xyz_gt=actual_xyz  when arm is in contact
         contact_label=0 / xyz_gt=zeros(3)    when arm is NOT in contact

Training modes:
  Stage 1: keypoint_loss_weight=1.0, action_loss_weight=0.0
           Freeze qwen_vl_interface + action_model, train keypoint_head only.
  Stage 2: both loss weights > 0, all modules trainable.
           Keypoint gradient flows through entire Qwen backbone.

Input data dict per sample:
    image:          List[PIL.Image]
    lang:           str
    action:         np.ndarray (T, action_dim)
    left_kp:        np.ndarray (3,) — actual xyz if active, zeros(3) if inactive
    right_kp:       np.ndarray (3,) — actual xyz if active, zeros(3) if inactive
    left_contact:   float           — 1.0 if left arm contacts, 0.0 otherwise
    right_contact:  float           — 1.0 if right arm contacts, 0.0 otherwise
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from starVLA.model.modules.keypoint_head.keypoint_head import KeypointPredHead
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# Qwen3-VL image-pad token ID (the <|image_pad|> placeholder in input_ids)
IMAGE_TOKEN_INDEX = 151655


@FRAMEWORK_REGISTRY.register("QwenOFTKeypoint")
class Qwenvl_OFT_Keypoint(Qwenvl_OFT):
    """QwenOFT + unified CrossAttention-pooled keypoint prediction head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        d_model = self.qwen_vl_interface.model.config.hidden_size
        self.keypoint_head = KeypointPredHead(d_model=d_model, hidden_dim=d_model)

        fw_cfg = self.config.framework
        self.keypoint_loss_weight  = float(getattr(fw_cfg, "keypoint_loss_weight", 1.0))
        self.action_loss_weight    = float(getattr(fw_cfg, "action_loss_weight", 1.0))
        # Weight for contact BCE loss relative to xyz L1 loss
        self.contact_loss_weight   = float(getattr(fw_cfg, "contact_loss_weight", 1.0))

        self.bce_loss = nn.BCEWithLogitsLoss()

    # ──────────────────────────────────────────────────────────────────────
    #  Vision token extraction
    # ──────────────────────────────────────────────────────────────────────

    def _extract_vision_tokens(
        self,
        last_hidden: torch.Tensor,   # (B, L, H)
        input_ids: torch.Tensor,     # (B, L)
    ) -> torch.Tensor:
        """
        Extract vision-token hidden states via IMAGE_TOKEN_INDEX positions.

        Requires all samples in the batch to have the same number of vision tokens
        (guaranteed when all input images share the same resolution).

        Returns:
            V: (B, N, H) where N = total vision tokens per sample
        """
        vis_mask = input_ids == IMAGE_TOKEN_INDEX     # (B, L)
        N = int(vis_mask[0].sum().item())
        if N == 0:
            raise RuntimeError(
                "No vision tokens found in input_ids. "
                "Verify IMAGE_TOKEN_INDEX=151655 matches your Qwen3-VL tokenizer."
            )
        B, L, H = last_hidden.shape
        V = last_hidden[vis_mask].view(B, N, H)
        return V

    # ──────────────────────────────────────────────────────────────────────
    #  Forward (training)
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        """
        Joint forward for action + keypoint prediction.

        Returns dict with keys:
            action_loss, xyz_loss, contact_loss, keypoint_loss, total_loss
        """
        batch_images  = [e["image"]  for e in examples]
        instructions  = [e["lang"]   for e in examples]
        actions       = [e["action"] for e in examples]
        state = [e["state"] for e in examples] if "state" in examples[0] else None

        if state is not None:
            instructions = self.add_discretized_state_to_instruction(instructions, state)

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = (
            f" Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}<action>."
        )
        instructions = [instr + prompt_suffix for instr in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]   # (B, L, H)

        input_ids = qwen_inputs.get("input_ids", None)

        with torch.autocast("cuda", dtype=torch.float32):
            # ── Action prediction ────────────────────────────────────────
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions  = self.action_model.predict_action(action_queries)
            actions_t     = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )
            actions_target = actions_t[:, -self.action_horizon:, :]
            action_loss    = self.l1_loss(pred_actions, actions_target)

            # ── Keypoint prediction ──────────────────────────────────────
            V = self._extract_vision_tokens(last_hidden, input_ids)   # (B, N, H)
            xyz_pred, contact_logit, _ = self.keypoint_head(V.to(torch.float32))
            # xyz_pred:      (B, 2, 3) — [:, 0]=left, [:, 1]=right
            # contact_logit: (B, 2)

            xyz_loss, contact_loss = self._compute_keypoint_loss(
                examples, xyz_pred, contact_logit
            )
            keypoint_loss = xyz_loss + self.contact_loss_weight * contact_loss

        total_loss = (
            self.action_loss_weight   * action_loss
            + self.keypoint_loss_weight * keypoint_loss
        )

        return {
            "action_loss":   action_loss,
            "xyz_loss":      xyz_loss,
            "contact_loss":  contact_loss,
            "keypoint_loss": keypoint_loss,
            "total_loss":    total_loss,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  Keypoint loss
    # ──────────────────────────────────────────────────────────────────────

    def _compute_keypoint_loss(
        self,
        examples: List[dict],
        xyz_pred:      torch.Tensor,  # (B, 2, 3)
        contact_logit: torch.Tensor,  # (B, 2)
    ):
        """
        Compute xyz L1 loss and contact BCE loss.

        xyz_loss:
            Only computed for arms where contact_label=1 (active arm).
            Inactive arms are excluded from xyz supervision to avoid polluting
            the xyz head with the arbitrary [0, 0, 0] target.

        contact_loss:
            Computed for ALL arms (both active and inactive).
            This is the primary signal for the model to learn "is there a contact?"

        Each example must contain:
            left_kp:       np.ndarray (3,) — zeros if inactive
            right_kp:      np.ndarray (3,) — zeros if inactive
            left_contact:  float 1.0 or 0.0
            right_contact: float 1.0 or 0.0
        """
        device  = xyz_pred.device
        dtype   = xyz_pred.dtype
        B = len(examples)

        # Build GT tensors from the batch  (B, 2, 3) and (B, 2)
        gt_xyz     = torch.zeros(B, 2, 3, device=device, dtype=dtype)
        gt_contact = torch.zeros(B, 2,    device=device, dtype=dtype)

        for i, ex in enumerate(examples):
            gt_xyz[i, 0] = torch.tensor(ex["left_kp"],  dtype=dtype, device=device)
            gt_xyz[i, 1] = torch.tensor(ex["right_kp"], dtype=dtype, device=device)
            gt_contact[i, 0] = float(ex["left_contact"])
            gt_contact[i, 1] = float(ex["right_contact"])

        # ── Contact BCE (all arms) ────────────────────────────────────────
        contact_loss = self.bce_loss(contact_logit, gt_contact)

        # ── xyz L1 (active arms only, weighted by contact_label) ──────────
        # gt_contact: (B, 2) → (B, 2, 1) for broadcasting with (B, 2, 3)
        mask      = gt_contact.unsqueeze(-1)                         # (B, 2, 1)
        xyz_err   = torch.abs(xyz_pred - gt_xyz)                     # (B, 2, 3)
        xyz_loss_all = (mask * xyz_err).sum(-1)                      # (B, 2) mean over xyz
        n_active  = gt_contact.sum().clamp(min=1.0)
        xyz_loss  = xyz_loss_all.sum() / n_active / 3.0              # normalise over xyz dims

        return xyz_loss, contact_loss

    # ──────────────────────────────────────────────────────────────────────
    #  Inference helper (optional — for deployment)
    # ──────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict_keypoints(self, examples: List[dict], contact_threshold: float = 0.5):
        """
        Inference: predict contact xyz and whether each arm is in contact.

        Returns dict:
            left_xyz:     (B, 3) predicted left contact xyz
            right_xyz:    (B, 3) predicted right contact xyz
            left_contact: (B,)   bool — True if left arm predicted in contact
            right_contact:(B,)   bool — True if right arm predicted in contact
        """
        batch_images = [e["image"] for e in examples]
        instructions = [e["lang"]  for e in examples]

        action_tokens  = self.action_token * self.chunk_len
        prompt_suffix  = (
            f" Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}<action>."
        )
        instructions = [instr + prompt_suffix for instr in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(
                **qwen_inputs, output_hidden_states=True, return_dict=True
            )
        last_hidden = out.hidden_states[-1]
        input_ids   = qwen_inputs["input_ids"]

        with torch.autocast("cuda", dtype=torch.float32):
            V = self._extract_vision_tokens(last_hidden, input_ids)
            xyz_pred, _, contact_prob = self.keypoint_head(V.to(torch.float32))

        return {
            "left_xyz":      xyz_pred[:, 0, :].cpu().numpy(),
            "right_xyz":     xyz_pred[:, 1, :].cpu().numpy(),
            "left_contact":  (contact_prob[:, 0] >= contact_threshold).cpu().numpy(),
            "right_contact": (contact_prob[:, 1] >= contact_threshold).cpu().numpy(),
        }
