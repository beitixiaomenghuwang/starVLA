# Copyright 2025 NeuroVLA community. All rights reserved.
"""
TeleAvatar-specific dataset subclass.

Applies two transformations on top of the raw LeRobot parquet data:

  1. Quaternion (xyzw) → Rotate6D (Zhou et al., 2019)
     - avoids the double-cover / gradient-discontinuity problem of quaternions
     - rot6d components are bounded in [-1, 1] by construction

  2. q99 normalization
     - EE position (x,y,z): normalize to [-1,1] using the 1st/99th percentile of
       the dataset's own statistics (computed once from parquet files)
     - Rot6D: NO normalization (already bounded, statistics over quat are not applicable)
     - Gripper effort: q99 normalization

Output action layout  (20-dim per timestep):
  [pos_L(3) | rot6d_L(6) | grip_L(1) | pos_R(3) | rot6d_R(6) | grip_R(1)]

Output state layout  (18-dim per timestep):
  [pos_L(3) | rot6d_L(6) | pos_R(3) | rot6d_R(6)]

Gripper effort note
-------------------
The raw parquet values ARE NOT standard 0/1 binary:
  left  arm: open = +2.5,  close = -0.5
  right arm: open = -2.5,  close = +0.5

After q99 normalization both grippers map cleanly to {-1, +1}:
  left  q01 ≈ -0.5, q99 ≈ 2.5  →  close → -1,  open → +1
  right q01 ≈ -2.5, q99 ≈ 0.5  →  open  → -1, close → +1

The semantic convention is inverted between arms, but the DiT diffusion model
learns from visual+language context and does NOT need cross-arm semantic parity.
Do NOT use 'binary' normalization (threshold=0.5): for the right arm close=0.5
would map to 0 (same as open=-2.5), making the two states indistinguishable.
"""

import numpy as np
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


# ──────────────────────────────────────────────────────────────────────────────
# Pure-numpy helpers (no torch dependency, fast in DataLoader workers)
# ──────────────────────────────────────────────────────────────────────────────

def quat_xyzw_to_rot6d(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (xyzw, scalar-last) to Rotate6D representation.

    Takes the first two *columns* of the rotation matrix and flattens them
    in C (row-major) order:
        mat[..., :, :2]  shape (..., 3, 2)
        → reshape → (..., 6)  = [r00, r01, r10, r11, r20, r21]

    All output components lie in [-1, 1].

    Args:
        q: (..., 4) quaternion array in (x, y, z, w) order.

    Returns:
        (..., 6) rotation-6D array.
    """
    from scipy.spatial.transform import Rotation
    mat = Rotation.from_quat(q).as_matrix()          # (..., 3, 3)
    return mat[..., :, :2].reshape(q.shape[:-1] + (6,))


def q99_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """
    Normalize array to [-1, 1] using 1st / 99th percentile clipping.

    Mirrors the formula in Normalizer.forward("q99") in transform/state_action.py:
        out = 2 * (x - q01) / (q99 - q01) - 1,  clamped to [-1, 1]
    Dimensions where q01 == q99 are left unchanged (avoids divide-by-zero).

    Args:
        x:   (..., D) array to normalize.
        q01: (D,)  1st percentile per dimension.
        q99: (D,)  99th percentile per dimension.
    """
    eps = 1e-8
    denom = q99 - q01
    mask = denom > eps                                   # (D,) bool
    out = x.copy()
    out[..., mask] = (
        2.0 * (x[..., mask] - q01[mask]) / (denom[mask] + eps) - 1.0
    )
    return np.clip(out, -1.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# TeleAvatarDataset
# ──────────────────────────────────────────────────────────────────────────────

class TeleAvatarDataset(LeRobotSingleDataset):
    """
    LeRobotSingleDataset subclass for the TeleAvatar dual-arm robot.

    Overrides __getitem__ to apply:
      - Multi-camera image loading (left_color, right_color, chest_camera)
      - Quaternion → Rotate6D conversion for EE orientations
      - q99 normalization for EE positions and gripper efforts
    """

    def _get_ee_stats(self, modality: str, key: str):
        """Return (q01[:3], q99[:3]) position stats and (q01, q99) full ee stats."""
        stats = getattr(self.metadata.statistics, modality)[key]
        q01 = np.array(stats.q01, dtype=np.float32)
        q99 = np.array(stats.q99, dtype=np.float32)
        return q01[:3], q99[:3]   # position part only (indices 0-2)

    def _get_grip_stats(self, modality: str, key: str):
        """Return (q01, q99) for a scalar gripper effort key."""
        stats = getattr(self.metadata.statistics, modality)[key]
        return (
            np.array(stats.q01, dtype=np.float32),
            np.array(stats.q99, dtype=np.float32),
        )

    def _convert_ee(
        self,
        raw_ee: np.ndarray,     # [T, 7]:  pos(3) + quat_xyzw(4)
        q01_pos: np.ndarray,    # [3]
        q99_pos: np.ndarray,    # [3]
    ) -> np.ndarray:
        """[T,7] → [T,9]: q99-normalized pos(3) + rot6d(6, already bounded)."""
        pos   = raw_ee[..., :3]                       # [T, 3]
        quat  = raw_ee[..., 3:]                       # [T, 4]  xyzw
        rot6d = quat_xyzw_to_rot6d(quat)             # [T, 6]
        pos_n = q99_normalize(pos, q01_pos, q99_pos)  # [T, 3] → [-1, 1]
        return np.concatenate([pos_n, rot6d], axis=-1) # [T, 9]

    def get_training_sample(self, trajectory_id: int, base_index: int) -> dict:
        data = self.get_step_data(trajectory_id, base_index)

        # ── Images: all cameras in order (left_color, right_color, chest_camera) ──
        images = [
            Image.fromarray(data[vk][0]).resize((224, 224))
            for vk in self.modality_keys["video"]
        ]
        language = data[self.modality_keys["language"][0]][0]

        # ── Position statistics (from dataset stats on raw 7-dim ee_pose) ────
        al_q01, al_q99 = self._get_ee_stats("action", "left_ee_pose")
        ar_q01, ar_q99 = self._get_ee_stats("action", "right_ee_pose")
        sl_q01, sl_q99 = self._get_ee_stats("state",  "left_ee_pose")
        sr_q01, sr_q99 = self._get_ee_stats("state",  "right_ee_pose")

        # ── Gripper statistics ────────────────────────────────────────────────
        lg_q01, lg_q99 = self._get_grip_stats("action", "left_gripper_effort")
        rg_q01, rg_q99 = self._get_grip_stats("action", "right_gripper_effort")

        # ── Action [T_future, 20] ─────────────────────────────────────────────
        l_ee9 = self._convert_ee(data["action.left_ee_pose"],  al_q01, al_q99)
        r_ee9 = self._convert_ee(data["action.right_ee_pose"], ar_q01, ar_q99)
        l_grip = q99_normalize(data["action.left_gripper_effort"],  lg_q01, lg_q99)
        r_grip = q99_normalize(data["action.right_gripper_effort"], rg_q01, rg_q99)

        # Layout: [pos_L(3) rot6d_L(6) grip_L(1) | pos_R(3) rot6d_R(6) grip_R(1)]
        action = np.concatenate(
            [l_ee9, l_grip, r_ee9, r_grip], axis=-1
        ).astype(np.float32)  # [T_future, 20]

        # ── State [T_past, 18] ────────────────────────────────────────────────
        # Layout: [pos_L(3) rot6d_L(6) | pos_R(3) rot6d_R(6)]
        ls_ee9 = self._convert_ee(data["state.left_ee_pose"],  sl_q01, sl_q99)
        rs_ee9 = self._convert_ee(data["state.right_ee_pose"], sr_q01, sr_q99)
        state = np.concatenate([ls_ee9, rs_ee9], axis=-1).astype(np.float32)  # [T_past, 18]

        return dict(action=action, image=images, lang=language, language=language, state=state)

    def __getitem__(self, index: int) -> dict:
        trajectory_id, base_index = self.all_steps[index]
        return self.get_training_sample(trajectory_id, base_index)
