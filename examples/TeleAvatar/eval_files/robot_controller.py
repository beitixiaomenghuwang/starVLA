#!/usr/bin/env python3
"""
NeuroVLA closed-loop controller for the TeleAvatar dual-arm robot.

Pipeline per control step:
  1. Receive observation from ROS2 (left_color image)
  2. Send left_color image + language instruction to the NeuroVLA inference server
  3. Receive normalized action chunk  [B=1, T=16, D=20]
  4. Decode each step:
       q99-denormalize  → positions (idx 0:3 L, 10:13 R) and grippers (idx 9 L, 19 R)
       rot6d → quaternion → EE orientations (idx 3:9 L, 13:19 R)
  5. Execute all chunk_size steps, then re-query for the next chunk
  6. Publish decoded 16-dim actions to ROS2 topics

Action layout (model output, 20-dim normalized):
  [pos_L(3) | rot6d_L(6) | grip_L(1) | pos_R(3) | rot6d_R(6) | grip_R(1)]

Action layout (ROS2 Pose + JointState, 16-dim):
  [pos_L(3) | quat_L_xyzw(4) | grip_L(1) | pos_R(3) | quat_R_xyzw(4) | grip_R(1)]

Gripper denormalization note:
  Training uses q99_normalize: y = 2*(x-q01)/(q99-q01) - 1
  Inverse:                      x = 0.5*(y+1)*(q99-q01) + q01
  Left  arm: q01≈-0.5, q99≈2.5  → close(-1)→-0.5,  open(+1)→+2.5   (physically correct)
  Right arm: q01≈-2.5, q99≈0.5  → open(-1) →-2.5, close(+1)→+0.5   (physically correct)

Control rate:
  Each chunk of `chunk_size` steps is executed back-to-back without sleep.
  The natural rate is determined by robot execution time per step plus
  the DDIM inference latency amortized over the chunk.

Usage (from project root):
  python examples/TeleAvatar/robot_controller.py \
      --ckpt_path /media/caslx/1635-A2D7/weight/neurovla_pick_marker/final_model/pytorch_model.pt \
      --host 127.0.0.1 \
      --port 10093 \
      --instruction "pick up the marker and put it into the cup" \
      --chunk_size 16

NOTE: The inference server (start_server.sh) must be running before starting this script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import logging
import time
from threading import Event
from typing import Dict, Optional

import numpy as np
import rclpy
from PIL import Image
from scipy.spatial.transform import Rotation

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.TeleAvatar.ros2_interface_endeffector import TeleavatarEndEffectorROS2Interface

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State normalization stats  (for GRU-edit state conditioning, state_dim=18)
# ─────────────────────────────────────────────────────────────────────────────

class TeleAvatarStateNormStats:
    """
    Loads *state* q01/q99 statistics from dataset_statistics.json.

    The dataset_statistics.json uses a flat 14-dim array for state:
      "<unnorm_key>": {"state": {"q01": [14 floats], "q99": [14 floats]}}
    Layout: [left_ee_pose(7): xyz(0:3)+quat(3:7) | right_ee_pose(7): xyz(7:10)+quat(10:14)]

    Only the xyz position slices are used; rot6d is bounded by construction.
    """

    def __init__(self, ckpt_path: str, unnorm_key: Optional[str] = None):
        ckpt_path  = Path(ckpt_path)
        stats_file = ckpt_path.parents[1] / "dataset_statistics.json"
        if not stats_file.exists():
            raise FileNotFoundError(f"dataset_statistics.json not found at {stats_file}")

        with open(stats_file) as f:
            all_stats = json.load(f)

        if unnorm_key is None:
            unnorm_key = next(iter(all_stats))
        elif unnorm_key not in all_stats:
            raise KeyError(f"unnorm_key '{unnorm_key}' not found in {stats_file}")

        ss  = all_stats[unnorm_key]["state"]
        q01 = ss["q01"]
        q99 = ss["q99"]

        # flat 14-dim: [left_xyz(0:3), left_quat(3:7), right_xyz(7:10), right_quat(10:14)]
        self.sl_q01 = np.array(q01[0:3],  dtype=np.float32)
        self.sl_q99 = np.array(q99[0:3],  dtype=np.float32)
        self.sr_q01 = np.array(q01[7:10], dtype=np.float32)
        self.sr_q99 = np.array(q99[7:10], dtype=np.float32)
        logger.info(
            f"Loaded STATE norm stats from '{unnorm_key}':\n"
            f"  left  pos q01={self.sl_q01}  q99={self.sl_q99}\n"
            f"  right pos q01={self.sr_q01}  q99={self.sr_q99}"
        )


def _build_state_18d(
    ee_pose_l: np.ndarray,
    ee_pose_r: np.ndarray,
    stats: TeleAvatarStateNormStats,
) -> np.ndarray:
    """
    Convert two raw 7-dim ee poses to the 18-dim normalized state vector.

    Args:
        ee_pose_l / ee_pose_r: (7,) [x,y,z,qx,qy,qz,qw]
        stats: TeleAvatarStateNormStats

    Returns:
        (18,) float32 [pos_L_norm(3) | rot6d_L(6) | pos_R_norm(3) | rot6d_R(6)]
    """
    def _norm_pos(pos, q01, q99):
        denom = q99 - q01
        mask  = denom > 1e-8
        out   = pos.copy()
        out[mask] = 2.0 * (pos[mask] - q01[mask]) / denom[mask] - 1.0
        return np.clip(out, -1.0, 1.0)

    def _quat_to_rot6d(q):
        mat = Rotation.from_quat(q).as_matrix()  # (3,3)
        return mat[:, :2].reshape(6)              # row-major first two columns

    pos_l  = _norm_pos(ee_pose_l[:3], stats.sl_q01, stats.sl_q99)
    rot6_l = _quat_to_rot6d(ee_pose_l[3:7])
    pos_r  = _norm_pos(ee_pose_r[:3], stats.sr_q01, stats.sr_q99)
    rot6_r = _quat_to_rot6d(ee_pose_r[3:7])
    return np.concatenate([pos_l, rot6_l, pos_r, rot6_r], dtype=np.float32)  # (18,)


# ─────────────────────────────────────────────────────────────────────────────
# Rotation conversion helpers  (pure numpy, no torch)
# ─────────────────────────────────────────────────────────────────────────────

def rot6d_to_quat_xyzw(v6: np.ndarray) -> np.ndarray:
    """
    Decode Rotate6D representation back to quaternion (xyzw, scalar-last).

    The storage layout from quat_xyzw_to_rot6d (in teleavatar_dataset.py):
        mat[..., :, :2].reshape((..., 6))
        → flattened in C order: [m00, m01, m10, m11, m20, m21]
          where m_ij is the (i,j) element (row i, column j)
        → first column (col 0):  v6[..., 0::2]  = [m00, m10, m20]
        → second column (col 1): v6[..., 1::2]  = [m01, m11, m21]

    Gram-Schmidt orthonormalization recovers the full rotation matrix,
    then converts to quaternion via scipy.

    Args:
        v6: (..., 6) float array in the training-convention layout above.

    Returns:
        (..., 4) quaternion array in xyzw (scalar-last) format.
    """
    v6 = np.asarray(v6, dtype=np.float64)
    assert v6.shape[-1] == 6, f"Expected last dim = 6, got {v6.shape[-1]}"

    # Recover the two raw column vectors from the flattened storage
    # Storage: [m00, m01, m10, m11, m20, m21]  (row-major scan of a 3×2 block)
    # col0 occupies even indices, col1 occupies odd indices
    col0 = v6[..., 0::2]  # [m00, m10, m20]  → shape (..., 3)
    col1 = v6[..., 1::2]  # [m01, m11, m21]  → shape (..., 3)

    # Gram-Schmidt: orthonormalize the two column vectors
    b1 = col0 / (np.linalg.norm(col0, axis=-1, keepdims=True) + 1e-8)
    b2 = col1 - np.sum(b1 * col1, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)

    # Assemble rotation matrix: columns are b1, b2, b3  →  shape (..., 3, 3)
    rot_mat = np.stack([b1, b2, b3], axis=-1)
    return Rotation.from_matrix(rot_mat).as_quat()  # xyzw


def q99_denormalize(x_norm: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """
    Inverse of q99_normalize:  x = 0.5*(x_norm + 1)*(q99 - q01) + q01

    Args:
        x_norm: (..., D) normalized values in [-1, 1].
        q01:    (D,)  1st-percentile per dimension.
        q99:    (D,)  99th-percentile per dimension.

    Returns:
        (..., D) values in original scale.
    """
    x_norm = np.clip(x_norm, -1.0, 1.0)
    return 0.5 * (x_norm + 1.0) * (q99 - q01) + q01


# ─────────────────────────────────────────────────────────────────────────────
# Statistics loader
# ─────────────────────────────────────────────────────────────────────────────

class TeleAvatarNormStats:
    """
    Loads per-component q01/q99 statistics from the run's dataset_statistics.json.

    Supports two JSON layouts:

    1. Per-component (legacy):
         "<unnorm_key>": {"action": {"left_ee_pose": {"q01": [...], "q99": [...]}, ...}}

    2. Flat 16-dim array (current training output):
         "<unnorm_key>": {"action": {"q01": [16 floats], "q99": [16 floats]}}
         Layout: [left_xyz(0:3), left_quat(3:7), left_grip(7),
                  right_xyz(8:11), right_quat(11:15), right_grip(15)]

    Rot6D is NOT normalized, so no stats are needed for orientation.
    """

    def __init__(self, ckpt_path: str, unnorm_key: Optional[str] = None):
        ckpt_path = Path(ckpt_path)
        # run_dir layout: <run_dir>/final_model/<name>.pt  →  parents[1] = run_dir
        # also supports: <run_dir>/checkpoints/<name>.pt
        run_dir = ckpt_path.parents[1]
        stats_file = run_dir / "dataset_statistics.json"

        if not stats_file.exists():
            raise FileNotFoundError(
                f"dataset_statistics.json not found at {stats_file}. "
                "Make sure the checkpoint path follows the layout: "
                "<run_dir>/checkpoints/<name>.pt  or  <run_dir>/final_model/<name>.pt"
            )

        with open(stats_file) as f:
            all_stats = json.load(f)

        # Resolve unnorm_key: if not given, take the first (and usually only) key
        if unnorm_key is None:
            unnorm_key = next(iter(all_stats))
            logger.info(f"unnorm_key not specified – using '{unnorm_key}' from stats file")
        elif unnorm_key not in all_stats:
            available = list(all_stats.keys())
            raise KeyError(
                f"unnorm_key '{unnorm_key}' not found in {stats_file}. "
                f"Available keys: {available}"
            )

        action_stats = all_stats[unnorm_key]["action"]

        if "left_ee_pose" in action_stats:
            # Per-component layout (legacy)
            self.al_q01 = np.array(action_stats["left_ee_pose"]["q01"][:3],  dtype=np.float32)
            self.al_q99 = np.array(action_stats["left_ee_pose"]["q99"][:3],  dtype=np.float32)
            self.ar_q01 = np.array(action_stats["right_ee_pose"]["q01"][:3], dtype=np.float32)
            self.ar_q99 = np.array(action_stats["right_ee_pose"]["q99"][:3], dtype=np.float32)
            self.lg_q01 = np.array(action_stats["left_gripper_effort"]["q01"],  dtype=np.float32)
            self.lg_q99 = np.array(action_stats["left_gripper_effort"]["q99"],  dtype=np.float32)
            self.rg_q01 = np.array(action_stats["right_gripper_effort"]["q01"], dtype=np.float32)
            self.rg_q99 = np.array(action_stats["right_gripper_effort"]["q99"], dtype=np.float32)
        else:
            # Flat 16-dim layout (save_dataset_statistics: non-gripper first, gripper last):
            #   [left_ee_pose(7): xyz(0:3)+quat(3:7) |
            #    right_ee_pose(7): xyz(7:10)+quat(10:14) |
            #    left_gripper_effort(1): [14] |
            #    right_gripper_effort(1): [15]]
            # Verified from actual stats file:
            #   [14] left grip  q01=-0.5  q99=+2.5  (close→-0.5, open→+2.5)
            #   [15] right grip q01=-2.5  q99=+0.5  (open→-2.5, close→+0.5)
            q01 = action_stats["q01"]
            q99 = action_stats["q99"]
            self.al_q01 = np.array(q01[0:3],   dtype=np.float32)   # left  pos
            self.al_q99 = np.array(q99[0:3],   dtype=np.float32)
            self.ar_q01 = np.array(q01[7:10],  dtype=np.float32)   # right pos
            self.ar_q99 = np.array(q99[7:10],  dtype=np.float32)
            self.lg_q01 = np.array([q01[14]],  dtype=np.float32)   # left  grip
            self.lg_q99 = np.array([q99[14]],  dtype=np.float32)
            self.rg_q01 = np.array([q01[15]],  dtype=np.float32)   # right grip
            self.rg_q99 = np.array([q99[15]],  dtype=np.float32)

        logger.info(
            f"Loaded TeleAvatar norm stats from '{unnorm_key}':\n"
            f"  left  pos  q01={self.al_q01}  q99={self.al_q99}\n"
            f"  right pos  q01={self.ar_q01}  q99={self.ar_q99}\n"
            f"  left  grip q01={self.lg_q01}  q99={self.lg_q99}\n"
            f"  right grip q01={self.rg_q01}  q99={self.rg_q99}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Action decoder:  20-dim normalized  →  16-dim quaternion
# ─────────────────────────────────────────────────────────────────────────────

class TeleAvatarActionDecoder:
    """
    Decodes a batch of normalized 20-dim actions into 16-dim quaternion actions.

    Normalized model output layout (20-dim):
      idx  0: 3   left  EE position  (q99-normalized)
      idx  3: 9   left  EE rot6d     (not normalized; bounded in [-1,1])
      idx  9:10   left  gripper effort (q99-normalized)
      idx 10:13   right EE position  (q99-normalized)
      idx 13:19   right EE rot6d     (not normalized)
      idx 19:20   right gripper effort (q99-normalized)

    Robot action layout (16-dim, quaternion):
      idx  0: 3   left  EE position  (raw metres)
      idx  3: 7   left  EE orientation (xyzw quaternion)
      idx  7: 8   left  gripper effort (raw N)
      idx  8:11   right EE position  (raw metres)
      idx 11:15   right EE orientation (xyzw quaternion)
      idx 15:16   right gripper effort (raw N)
    """

    def __init__(self, stats: TeleAvatarNormStats):
        self.stats = stats

    def decode_chunk(self, norm_actions: np.ndarray) -> np.ndarray:
        """
        Args:
            norm_actions: (T, 20) normalized action chunk from the server.

        Returns:
            (T, 16) robot actions in physical units + quaternion orientation.
        """
        assert norm_actions.ndim == 2 and norm_actions.shape[1] == 20, (
            f"Expected shape (T, 20), got {norm_actions.shape}"
        )
        T = norm_actions.shape[0]
        out = np.zeros((T, 16), dtype=np.float32)

        s = self.stats

        # ── Left arm ────────────────────────────────────────────────────────
        pos_l_norm  = norm_actions[:, 0:3]   # (T, 3)
        rot6d_l     = norm_actions[:, 3:9]   # (T, 6)
        grip_l_norm = norm_actions[:, 9:10]  # (T, 1)

        pos_l  = q99_denormalize(pos_l_norm,  s.al_q01, s.al_q99)  # (T, 3)
        quat_l = rot6d_to_quat_xyzw(rot6d_l).astype(np.float32)    # (T, 4)
        grip_l = q99_denormalize(grip_l_norm, s.lg_q01, s.lg_q99)  # (T, 1)

        out[:, 0:3] = pos_l
        out[:, 3:7] = quat_l
        out[:, 7:8] = grip_l

        # ── Right arm ───────────────────────────────────────────────────────
        pos_r_norm  = norm_actions[:, 10:13]  # (T, 3)
        rot6d_r     = norm_actions[:, 13:19]  # (T, 6)
        grip_r_norm = norm_actions[:, 19:20]  # (T, 1)

        pos_r  = q99_denormalize(pos_r_norm,  s.ar_q01, s.ar_q99)  # (T, 3)
        quat_r = rot6d_to_quat_xyzw(rot6d_r).astype(np.float32)    # (T, 4)
        grip_r = q99_denormalize(grip_r_norm, s.rg_q01, s.rg_q99)  # (T, 1)

        out[:, 8:11]  = pos_r
        out[:, 11:15] = quat_r
        out[:, 15:16] = grip_r

        return out  # (T, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Main controller
# ─────────────────────────────────────────────────────────────────────────────

class TeleAvatarController:
    """
    Closed-loop controller that connects the NeuroVLA inference server to the
    TeleAvatar ROS2 interface.

    Control strategy: temporal chunking with periodic re-query.
      - Query the inference server for a fresh 16-step action chunk.
      - Execute the first `chunk_size` steps back-to-back (no artificial sleep;
        the robot's own execution latency sets the pace).
      - Re-query when the active chunk is exhausted.

    chunk_size should match the training action_horizon (16) to fully exploit
    the predicted trajectory. It can be reduced to re-query more frequently
    for faster error correction at the cost of higher server load.
    """

    def __init__(
        self,
        ros_node: TeleavatarEndEffectorROS2Interface,
        client: WebsocketClientPolicy,
        decoder: TeleAvatarActionDecoder,
        state_stats: TeleAvatarStateNormStats,
        instruction: str,
        chunk_size: int = 16,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 4,
    ):
        self.ros_node = ros_node
        self.client = client
        self.decoder = decoder
        self.state_stats = state_stats
        self.instruction = instruction
        self.chunk_size = min(chunk_size, 16)  # model produces max 16 steps
        self.cfg_scale = cfg_scale
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps

        self._active_chunk: Optional[np.ndarray] = None  # (T, 16)

        logger.info(
            f"Controller ready | instruction='{instruction}' chunk_size={chunk_size}"
        )

    def _query_server(self, obs: Dict) -> np.ndarray:
        """
        Send observation to the inference server and return decoded (T, 16) actions.
        """
        # Visual inputs: left_color, right_color, chest_camera (training order)
        # Send as uint8 numpy arrays; server calls to_pil_preserve internally.
        def _resize_np(arr):
            return np.array(Image.fromarray(arr).resize((224, 224)), dtype=np.uint8)

        img_left  = _resize_np(obs["images"]["left_color"])
        img_right = _resize_np(obs["images"]["right_color"])
        img_chest = _resize_np(obs["images"]["chest_camera"])

        # State input: 18-dim normalized ee poses from current observation.
        # obs["state"] is the 62-dim vector; [48:55]=left ee, [55:62]=right ee.
        raw_state = obs["state"]  # (62,)
        state_18d = _build_state_18d(raw_state[48:55], raw_state[55:62], self.state_stats)

        payload = {
            "batch_images":     [[img_left, img_right, img_chest]],  # [B=1, [3 cams]] uint8 np
            "instructions":     [self.instruction],
            "states":           [[[float(v) for v in state_18d]]],   # [B=1, T=1, 18]
            "cfg_scale":        self.cfg_scale,
            "use_ddim":         self.use_ddim,
            "num_ddim_steps":   self.num_ddim_steps,
        }

        t0 = time.time()
        response = self.client.infer(payload)
        elapsed = time.time() - t0

        if not response.get("ok", True):
            raise RuntimeError(f"Server returned error: {response}")

        # normalized_actions: (B=1, T=16, D=20) → take first batch element
        norm_actions = np.array(response["data"]["normalized_actions"][0])  # (T, 20)
        logger.debug(f"Inference latency: {elapsed*1000:.1f} ms")

        return self.decoder.decode_chunk(norm_actions)  # (T, 16)

    def _get_fresh_chunk(self, obs: Dict) -> None:
        """Query the server and store the decoded action chunk."""
        self._active_chunk = self._query_server(obs)

    def run(self, stop_event: Optional[Event] = None, max_steps: int = 0) -> None:
        """
        Main control loop.

        Execution model: chunk-level loop.
          - Query server → get (T=16, 20) chunk → decode to (T=16, 16).
          - Execute all chunk_size steps back-to-back.
          - Re-query immediately for the next chunk.
          - Repeat until stop_event is set or max_steps is reached.

        The effective control rate is set by the robot's own step execution
        time plus DDIM inference latency amortized over the chunk.

        Args:
            stop_event: threading.Event; set from another thread to stop gracefully.
            max_steps:  Hard limit on control steps (0 = run until Ctrl-C).
        """
        if stop_event is None:
            stop_event = Event()

        logger.info("Waiting for initial sensor data …")
        if not self.ros_node.wait_for_initial_data(timeout=15.0):
            logger.error("Timed out waiting for sensor data. Aborting.")
            return

        logger.info("Starting control loop.")
        step = 0

        while not stop_event.is_set():
            if max_steps > 0 and step >= max_steps:
                logger.info(f"Reached max_steps={max_steps}. Stopping.")
                break

            # Check FSM mode — only act in RL/AUTOMODE
            fsm = self.ros_node._fsm_mode
            if fsm is not None and fsm not in ("RLMODE", "AUTOMODE", "rl", "auto"):
                logger.info(f"FSM mode '{fsm}' — waiting for RLMODE.")
                time.sleep(0.1)
                continue

            # Snapshot observation at the chunk boundary for the server query
            obs = self.ros_node.get_observation()
            if obs is None:
                logger.warning("Observation not ready yet, retrying.")
                time.sleep(0.05)
                continue

            # Query server for a fresh action chunk
            try:
                self._get_fresh_chunk(obs)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                time.sleep(0.1)
                continue

            # Execute all chunk_size steps back-to-back
            for i in range(self.chunk_size):
                if stop_event.is_set():
                    break
                if max_steps > 0 and step >= max_steps:
                    break
                self.ros_node.publish_action(self._active_chunk[i])
                step += 1

        logger.info("Control loop stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NeuroVLA closed-loop controller for TeleAvatar",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ckpt_path", required=True, type=str,
        help="Path to the trained .pt checkpoint. "
             "Must follow layout: <run_dir>/checkpoints/<name>.pt "
             "(dataset_statistics.json lives in <run_dir>/).",
    )
    p.add_argument("--host", default="127.0.0.1", type=str,
                   help="Inference server hostname or IP.")
    p.add_argument("--port", default=10093, type=int,
                   help="Inference server WebSocket port.")
    p.add_argument(
        "--unnorm_key", default=None, type=str,
        help="Key in dataset_statistics.json (e.g. 'teleavatar_pick_marker'). "
             "Auto-detected from the file when omitted.",
    )
    p.add_argument(
        "--instruction", default="pick up the marker and put it into the cup",
        type=str, help="Natural language task instruction.",
    )
    p.add_argument(
        "--chunk_size", default=16, type=int,
        help="Steps to execute per chunk before re-querying the server (max 16). "
             "Matches the training action_horizon=16; reduce to re-query more frequently.",
    )
    p.add_argument("--cfg_scale", default=1.5, type=float,
                   help="Classifier-free guidance scale (>1 = guided).")
    p.add_argument("--num_ddim_steps", default=4, type=int,
                   help="Number of DDIM diffusion steps for inference.")
    p.add_argument("--max_steps", default=0, type=int,
                   help="Hard limit on control steps (0 = run until Ctrl-C).")
    return p


def main():
    args = build_argparser().parse_args()

    # ── Load normalization statistics ────────────────────────────────────────
    stats       = TeleAvatarNormStats(args.ckpt_path, unnorm_key=args.unnorm_key)
    state_stats = TeleAvatarStateNormStats(args.ckpt_path, unnorm_key=args.unnorm_key)
    decoder     = TeleAvatarActionDecoder(stats)

    # ── Connect to inference server ──────────────────────────────────────────
    logger.info(f"Connecting to inference server at {args.host}:{args.port} …")
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    logger.info("Connected to inference server.")

    # ── Initialize ROS2 ──────────────────────────────────────────────────────
    rclpy.init()
    ros_node = TeleavatarEndEffectorROS2Interface()

    import threading
    ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_spin_thread.start()

    # ── Run controller ───────────────────────────────────────────────────────
    controller = TeleAvatarController(
        ros_node=ros_node,
        client=client,
        decoder=decoder,
        state_stats=state_stats,
        instruction=args.instruction,
        chunk_size=args.chunk_size,
        cfg_scale=args.cfg_scale,
        use_ddim=True,
        num_ddim_steps=args.num_ddim_steps,
    )

    stop_event = Event()
    try:
        controller.run(stop_event=stop_event, max_steps=args.max_steps)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stop_event.set()
        client.close()
        ros_node.destroy_node()
        rclpy.shutdown()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
