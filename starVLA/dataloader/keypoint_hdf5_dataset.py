"""
KeypointHDF5Dataset — reads raw RoboTwin HDF5 episode files for keypoint training.

Expected directory layout:
    data_dir/
        data/
            episode0.hdf5
            episode1.hdf5
            ...
        keypoints.json          per-episode keypoint metadata
        instructions/
            episode0.json       seen/unseen language variations
            episode1.json
            ...

Each HDF5 episode contains:
    observation/{camera}/rgb   bytes per frame (JPEG-encoded), shape (T,) dtype=|S*
    joint_action/vector        raw joint actions, shape (T, 14)
    keypoints/                 group with attrs {active_arm, frame, time_s}
        left_contact_xyz       shape (3,) float64
        right_contact_xyz      shape (3,) float64
        contact_xyz            shape (3,) float64 — active arm's xyz

Supervision signal (v2 — unified head with contact classification):
    active_arm "left":
        left_kp  = left_contact_xyz,  left_contact  = 1.0
        right_kp = zeros(3),          right_contact = 0.0
    active_arm "right":
        left_kp  = zeros(3),          left_contact  = 0.0
        right_kp = right_contact_xyz, right_contact = 1.0
    active_arm "both":
        left_kp  = left_contact_xyz,  left_contact  = 1.0
        right_kp = right_contact_xyz, right_contact = 1.0

The xyz head is only supervised for active arms (via contact_label mask in loss).
The contact head is supervised for all arms.
"""

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# Maps LeRobot-style camera names → HDF5 group names
_CAMERA_HDF5 = {
    "cam_high": "front_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
    "cam_head": "head_camera",
}


def _decode_jpeg(raw: bytes) -> np.ndarray:
    """Decode JPEG bytes → (H, W, 3) uint8 array."""
    return np.array(Image.open(io.BytesIO(raw)))


class KeypointHDF5Dataset(Dataset):
    """
    Dataset that pairs visual observations with per-episode keypoint labels.

    __getitem__ returns:
        image:         List[PIL.Image]       — one per camera, resized to image_size
        lang:          str                   — sampled from seen instruction variations
        action:        np.ndarray (horizon, action_dim) — optionally normalized
        left_kp:       np.ndarray (3,) float32 — left contact xyz (zeros if inactive)
        right_kp:      np.ndarray (3,) float32 — right contact xyz (zeros if inactive)
        left_contact:  float  1.0 = left arm contacts,  0.0 = does not
        right_contact: float  1.0 = right arm contacts, 0.0 = does not
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        action_horizon: int = 49,
        cameras: Sequence[str] = ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        frame_sample_strategy: str = "random",
        stats_path: Optional[str] = None,
    ):
        """
        Args:
            data_dir:               Root directory (contains data/, keypoints.json, instructions/).
            image_size:             PIL resize target (W, H).
            action_horizon:         Action chunk length.
            cameras:                Camera names (LeRobot naming convention).
            frame_sample_strategy:  "random"  — sample uniformly from [0, contact_frame);
                                    "first"   — always frame 0;
                                    "contact" — frame of first arm contact.
            stats_path:             Path to dataset_statistics.json for action normalization.
                                    If None or missing, actions are returned in raw joint space.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.action_horizon = action_horizon
        self.cameras = list(cameras)
        self.frame_sample_strategy = frame_sample_strategy

        # Per-episode keypoint JSON (fallback if HDF5 attrs are missing)
        kp_json = self.data_dir / "keypoints.json"
        self.kp_meta: Dict[str, dict] = {}
        if kp_json.exists():
            with open(kp_json) as f:
                self.kp_meta = json.load(f)

        # Discover and sort episode HDF5 files
        ep_files = sorted(
            (self.data_dir / "data").glob("episode*.hdf5"),
            key=lambda p: int(p.stem.replace("episode", "")),
        )
        self.episode_files = [str(p) for p in ep_files]
        if not self.episode_files:
            raise FileNotFoundError(
                f"No episode*.hdf5 files found under {self.data_dir / 'data'}"
            )

        # Optional action normalization (mean/std from dataset_statistics.json)
        self.action_mean: Optional[np.ndarray] = None
        self.action_std: Optional[np.ndarray] = None
        if stats_path:
            self._load_action_stats(stats_path)

    # ──────────────────────────────────────────────────────────────────────
    #  Action normalization helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_action_stats(self, stats_path: str) -> None:
        """Load action normalization stats from dataset_statistics.json.

        Accepts two formats:
            {"action": {"mean": [...], "std": [...]}}              (flat)
            {"<task_key>": {"action": {"mean": [...], "std": [...]}}}  (nested)
        """
        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[KeypointHDF5Dataset] Warning: could not load stats from {stats_path}: {e}")
            return

        # Try flat format first
        if "action" in stats and "mean" in stats["action"]:
            action_stats = stats["action"]
        else:
            # Nested: grab first entry with "action" key
            action_stats = {}
            for val in stats.values():
                if isinstance(val, dict) and "action" in val:
                    a = val["action"]
                    if "mean" in a and "std" in a:
                        action_stats = a
                        break

        if action_stats:
            self.action_mean = np.array(action_stats["mean"], dtype=np.float32)
            self.action_std = np.array(action_stats["std"], dtype=np.float32).clip(min=1e-8)
            print(f"[KeypointHDF5Dataset] Loaded action normalization stats from {stats_path}")

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_mean is not None:
            return (action - self.action_mean) / self.action_std
        return action

    # ──────────────────────────────────────────────────────────────────────
    #  Dataset interface
    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.episode_files)

    def __getitem__(self, idx: int) -> dict:
        ep_path = self.episode_files[idx]

        with h5py.File(ep_path, "r") as f:
            T = f["joint_action"]["vector"].shape[0]

            # Contact metadata from HDF5 attributes
            kp_attrs = f["keypoints"].attrs
            contact_frame = int(kp_attrs.get("frame", T // 2))
            active_arm = str(kp_attrs.get("active_arm", "left"))

            # Sample observation frame
            t = self._sample_frame(T, contact_frame)

            # Decode camera images
            images = []
            for cam in self.cameras:
                hdf5_cam = _CAMERA_HDF5.get(cam, cam)
                raw_bytes = f["observation"][hdf5_cam]["rgb"][t].tobytes()
                arr = _decode_jpeg(raw_bytes)
                img = Image.fromarray(arr).resize(self.image_size, Image.BILINEAR)
                images.append(img)

            # Ground-truth keypoint xyz
            left_xyz_raw  = f["keypoints"]["left_contact_xyz"][:].astype(np.float32)
            right_xyz_raw = f["keypoints"]["right_contact_xyz"][:].astype(np.float32)

            # Raw actions for action chunk
            all_actions = f["joint_action"]["vector"][:]   # (T, 14)

        # Contact labels (float for BCE loss compatibility)
        left_active  = active_arm in ("left",  "both")
        right_active = active_arm in ("right", "both")

        # xyz GT: actual xyz for active arm, zeros for inactive.
        # The loss function masks xyz supervision by contact_label, so
        # these zeros are never used for xyz supervision.
        left_kp  = left_xyz_raw  if left_active  else np.zeros(3, dtype=np.float32)
        right_kp = right_xyz_raw if right_active else np.zeros(3, dtype=np.float32)

        # Build padded action chunk
        t_end = min(t + self.action_horizon, T)
        chunk = all_actions[t:t_end].astype(np.float32)
        if len(chunk) < self.action_horizon:
            pad = np.tile(chunk[-1:], (self.action_horizon - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk = self._normalize_action(chunk)

        return {
            "image":          images,
            "lang":           self._get_instruction(idx),
            "action":         chunk,                        # (action_horizon, 14)
            "left_kp":        left_kp,                     # (3,) float32
            "right_kp":       right_kp,                    # (3,) float32
            "left_contact":   float(left_active),          # 1.0 or 0.0
            "right_contact":  float(right_active),         # 1.0 or 0.0
        }

    # ──────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _sample_frame(self, T: int, contact_frame: int) -> int:
        if self.frame_sample_strategy == "first":
            return 0
        elif self.frame_sample_strategy == "contact":
            return min(contact_frame, T - 1)
        else:   # random — sample from [0, contact_frame) to expose pre-contact states
            hi = max(1, min(contact_frame, T))
            return int(np.random.randint(0, hi))

    def _get_instruction(self, idx: int) -> str:
        instr_path = self.data_dir / "instructions" / f"episode{idx}.json"
        if instr_path.exists():
            try:
                with open(instr_path) as f:
                    data = json.load(f)
                seen = data.get("seen", [])
                if seen:
                    return str(np.random.choice(seen))
            except (json.JSONDecodeError, KeyError):
                pass
        return "complete the robot manipulation task"


def collate_keypoint_batch(batch: List[dict]) -> List[dict]:
    """Pass-through collate — returns a list of sample dicts.

    QwenOFTKeypoint.forward() expects List[dict], so no tensor-stacking is done here.
    PIL images stay as-is; the VLM processor handles batching internally.
    """
    return batch
