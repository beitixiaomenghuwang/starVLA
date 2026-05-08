"""
Learnable cross-attention keypoint prediction head for dual-arm robot contact prediction.

Design (v2 — unified head with contact classification):
  Input:  V ∈ R^{N × d}  — all vision token features from Qwen backbone

  Step 1: CrossAttentionPool
    Q = [q_left, q_right] ∈ R^{2 × d}  (learnable, random init)
    A = Softmax(Q · V^T / sqrt(d))       ∈ R^{2 × N}
    F = A · V                            ∈ R^{2 × d}
      F[:, 0] = left-arm geometric feature
      F[:, 1] = right-arm geometric feature

  Step 2: Unified predictions from F (applied to both arms simultaneously)
    P_pred = MLP_xyz(F)          ∈ R^{2 × 3}   — left/right contact xyz
    c_pred = sigmoid(Linear(F))  ∈ R^{2}        — left/right contact probability

Loss design:
  contact_loss = BCE(c_pred, contact_label)          — all arms
  xyz_loss     = contact_label * |P_pred - P_gt|     — active arms only (masked)
  total = xyz_loss + λ * contact_loss

  Rationale for masking xyz_loss:
    Setting inactive arm target to [0, 0, 0] is arbitrary (origin has no semantic
    meaning in robot workspace).  Masking ensures the xyz head is only supervised
    by geometrically meaningful coordinates.  The contact head still learns to output
    near-zero for inactive arms, providing the "no contact" signal during inference.
"""

import torch
import torch.nn as nn

from starVLA.model.modules.action_model.MLP_ActionHeader import MLPResNet


class CrossAttentionPool(nn.Module):
    """Pool N vision tokens into 2 arm-specific vectors via learnable cross-attention.

    Q (learnable) acts as task queries for left/right arm respectively.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Two learnable query vectors: index 0 = left, index 1 = right
        self.queries = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.scale = d_model ** -0.5

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        """
        Args:
            V: (B, N, d) — vision token features
        Returns:
            F: (B, 2, d) — F[:, 0]=left feature, F[:, 1]=right feature
        """
        B = V.size(0)
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)               # (B, 2, d)
        A = torch.softmax(Q @ V.transpose(1, 2) * self.scale, dim=-1) # (B, 2, N)
        return A @ V                                                    # (B, 2, d)


class KeypointPredHead(nn.Module):
    """
    Unified dual-arm contact-point prediction head.

    One shared MLP predicts xyz for both arms simultaneously.
    A lightweight linear head predicts per-arm contact probability.

    Architecture:
        pool      : CrossAttentionPool  V (B, N, d) → F (B, 2, d)
        mlp_xyz   : MLPResNet           F reshaped → xyz_pred  (B, 2, 3)
        contact   : Linear + sigmoid    F reshaped → contact_prob (B, 2)
    """

    def __init__(self, d_model: int, hidden_dim: int = None):
        """
        Args:
            d_model:    Vision token / backbone hidden dimension.
            hidden_dim: MLP hidden dim. Defaults to d_model.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model
        self.pool = CrossAttentionPool(d_model)
        # Shared MLP: processes F[:, 0] and F[:, 1] with the same weights
        self.mlp_xyz = MLPResNet(
            num_blocks=2, input_dim=d_model, hidden_dim=hidden_dim, output_dim=3
        )
        # Contact prediction: one logit per arm
        self.contact_head = nn.Linear(d_model, 1)

    def forward(self, V: torch.Tensor):
        """
        Args:
            V: (B, N, d) vision token features
        Returns:
            xyz_pred:      (B, 2, 3)  — predicted contact xyz for [left, right]
            contact_logit: (B, 2)     — raw logits for contact probability
            contact_prob:  (B, 2)     — sigmoid of logits (for inference)
        """
        F = self.pool(V)                      # (B, 2, d)
        B, _, d = F.shape

        # Flatten arms into batch dimension so one MLP serves both arms
        F_flat = F.reshape(B * 2, d)          # (B*2, d)

        xyz_flat = self.mlp_xyz(F_flat)                      # (B*2, 3)
        xyz_pred = xyz_flat.view(B, 2, 3)                    # (B, 2, 3)

        contact_logit = self.contact_head(F_flat).view(B, 2) # (B, 2)
        contact_prob  = torch.sigmoid(contact_logit)         # (B, 2)

        return xyz_pred, contact_logit, contact_prob
