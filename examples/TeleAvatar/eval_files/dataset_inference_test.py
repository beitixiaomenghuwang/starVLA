#!/usr/bin/env python3
"""
Offline dataset-based inference test for NeuroVLA / TeleAvatar.

Replaces the live ROS2 interface with direct reads from a LeRobot dataset
episode so the model can be evaluated without a physical robot.

What is read from the dataset per step
───────────────────────────────────────
  • left_color, right_color, chest_camera frames (mp4)  → model visual inputs
    Sent in training order: [left_color, right_color, chest_camera]
  • observation.state[48:55]  left  ee pose  [x,y,z,qx,qy,qz,qw]
  • observation.state[55:62]  right ee pose  [x,y,z,qx,qy,qz,qw]
    → normalized & converted to rot6d → 18-dim model state input
  • action[48:55], action[39], action[55:62], action[47]
    → ground-truth 16-dim quaternion action for comparison

NOTE on gripper in state
────────────────────────
The model's GRU-edit state is 18-dim: [pos_L(3)+rot6d_L(6)+pos_R(3)+rot6d_R(6)].
Gripper effort is part of the predicted action, NOT the state conditioning.

Output
──────
  <output_dir>/ep<N>_action_diff.txt  – per-step pred vs GT action difference

Usage
─────
  conda run -n neurovla \\
    python examples/TeleAvatar/dataset_inference_test.py \\
      --dataset_root pick_marker_put_into_cup_20251113 \\
      --ckpt_path    /DATA/.../checkpoints/steps_XXXXXX_pytorch_model.pt \\
      --episode      0 \\
      --host 127.0.0.1 --port 10093

Optional ROS2 publishing (requires running ROS2 environment):
  Add --publish_ros to also send decoded actions to the robot topics in real time.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

# ── video reading ─────────────────────────────────────────────────────────────
try:
    import decord
    _HAVE_DECORD = True
except ImportError:
    _HAVE_DECORD = False

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

# ── project imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.TeleAvatar.robot_controller import (
    rot6d_to_quat_xyzw,
    q99_denormalize,
    TeleAvatarNormStats,
    TeleAvatarActionDecoder,
)
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State normalization stats  (separate from action stats in robot_controller.py)
# ─────────────────────────────────────────────────────────────────────────────

class TeleAvatarStateNormStats:
    """
    Loads *state* q01/q99 statistics for EE position normalization.

    The dataset_statistics.json uses a flat 14-dim array for state:
      "<unnorm_key>": {"state": {"q01": [14 floats], "q99": [14 floats]}}
    Layout: [left_ee_pose(7): xyz(0:3)+quat(3:7) | right_ee_pose(7): xyz(7:10)+quat(10:14)]

    Only the xyz position slices are used; rot6d is bounded by construction
    and does not require normalization statistics.
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
            raise KeyError(f"unnorm_key '{unnorm_key}' not in {stats_file}")

        ss = all_stats[unnorm_key]["state"]
        q01 = ss["q01"]
        q99 = ss["q99"]

        # flat 14-dim: [left_xyz(0:3), left_quat(3:7), right_xyz(7:10), right_quat(10:14)]
        self.sl_q01 = np.array(q01[0:3], dtype=np.float32)
        self.sl_q99 = np.array(q99[0:3], dtype=np.float32)
        self.sr_q01 = np.array(q01[7:10], dtype=np.float32)
        self.sr_q99 = np.array(q99[7:10], dtype=np.float32)

        logger.info(
            f"Loaded TeleAvatar STATE norm stats from '{unnorm_key}':\n"
            f"  left  pos q01={self.sl_q01}  q99={self.sl_q99}\n"
            f"  right pos q01={self.sr_q01}  q99={self.sr_q99}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build the 18-dim model state from raw parquet ee_pose data
# ─────────────────────────────────────────────────────────────────────────────

def _quat_xyzw_to_rot6d(q: np.ndarray) -> np.ndarray:
    """(4,) xyzw quaternion → (6,) rot6d (first two columns of rotation matrix, row-major)."""
    from scipy.spatial.transform import Rotation
    mat = Rotation.from_quat(q).as_matrix()   # (3, 3)
    return mat[:, :2].reshape(6)              # [r00,r01,r10,r11,r20,r21]


def _q99_normalize_pos(pos: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Normalize a (3,) position to [-1,1] using q01/q99 clipping."""
    denom = q99 - q01
    mask  = denom > 1e-8
    out   = pos.copy()
    out[mask] = 2.0 * (pos[mask] - q01[mask]) / denom[mask] - 1.0
    return np.clip(out, -1.0, 1.0)


def build_state_18d(
    raw_state_62d: np.ndarray,
    stats: "TeleAvatarStateNormStats",
) -> np.ndarray:
    """
    Convert the 62-dim parquet observation.state vector into the 18-dim
    model state used by the GRU-edit conditioning module.

    Layout:
      raw_state_62d[48:55] = left  ee pose [x,y,z,qx,qy,qz,qw]
      raw_state_62d[55:62] = right ee pose [x,y,z,qx,qy,qz,qw]

    Output (18-dim):
      [pos_L_norm(3) | rot6d_L(6) | pos_R_norm(3) | rot6d_R(6)]
    """
    l_pose = raw_state_62d[48:55]   # (7,)
    r_pose = raw_state_62d[55:62]   # (7,)

    pos_l  = _q99_normalize_pos(l_pose[:3], stats.sl_q01, stats.sl_q99)  # (3,)
    rot6_l = _quat_xyzw_to_rot6d(l_pose[3:7])                            # (6,)
    pos_r  = _q99_normalize_pos(r_pose[:3], stats.sr_q01, stats.sr_q99)  # (3,)
    rot6_r = _quat_xyzw_to_rot6d(r_pose[3:7])                            # (6,)

    return np.concatenate([pos_l, rot6_l, pos_r, rot6_r], dtype=np.float32)  # (18,)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset reader
# ─────────────────────────────────────────────────────────────────────────────

class LeRobotEpisodeReader:
    """
    Reads one episode from a LeRobot dataset stored on disk.

    Camera frames are loaded lazily (one at a time via decord or cv2).
    Parquet state/action arrays are loaded fully at construction.
    """

    # Slices into the 62-dim action vector (from modality.json)
    _L_EE_POSE  = slice(48, 55)   # [x,y,z,qx,qy,qz,qw]
    _L_GRIP     = slice(39, 40)   # effort scalar
    _R_EE_POSE  = slice(55, 62)   # [x,y,z,qx,qy,qz,qw]
    _R_GRIP     = slice(47, 48)   # effort scalar

    def __init__(self, dataset_root: str, episode_index: int):
        self.root = Path(dataset_root)
        self.ep   = episode_index

        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)

        with open(self.root / "meta" / "tasks.jsonl") as f:
            tasks = {d["task_index"]: d["task"] for d in (json.loads(l) for l in f)}

        chunk = episode_index // self.info["chunks_size"]
        parquet_path = self.root / self.info["data_path"].format(
            episode_chunk=chunk, episode_index=episode_index
        )
        table = pq.read_table(str(parquet_path))

        self._action = np.stack(table["action"].to_pylist()).astype(np.float32)         # (T,62)
        self._state  = np.stack(table["observation.state"].to_pylist()).astype(np.float32)  # (T,62)
        self.instruction = tasks[table["task_index"].to_pylist()[0]]
        self.num_frames  = len(self._action)

        # Build video path map
        cam_map = {
            "left_color":   "observation.images.left_color",
            "right_color":  "observation.images.right_color",
            "head_camera":  "observation.images.head_camera",
            "chest_camera": "observation.images.chest_camera",
        }
        self._video_paths: dict[str, str] = {}
        for cam_key, video_key in cam_map.items():
            p = self.root / self.info["video_path"].format(
                episode_chunk=chunk, episode_index=episode_index, video_key=video_key
            )
            if p.exists():
                self._video_paths[cam_key] = str(p)
            else:
                logger.warning(f"Video not found (skipped): {p}")

        logger.info(
            f"Episode {episode_index}: {self.num_frames} frames | "
            f"cameras={list(self._video_paths.keys())} | "
            f"instruction='{self.instruction}'"
        )

    def get_frame_rgb(self, frame_idx: int, camera: str = "left_color") -> np.ndarray:
        """Return a single (H, W, 3) uint8 RGB frame."""
        path = self._video_paths.get(camera)
        if path is None:
            raise KeyError(f"Camera '{camera}' not available for episode {self.ep}")
        return _read_video_frame(path, frame_idx)

    def get_state_62d(self, frame_idx: int) -> np.ndarray:
        """Return the raw 62-dim observation.state vector at frame_idx."""
        return self._state[frame_idx].copy()

    def get_gt_action_16d(self, frame_idx: int) -> np.ndarray:
        """
        Extract the ground-truth 16-dim action from the 62-dim parquet action.

        Output: [pos_L(3) | quat_L_xyzw(4) | grip_L(1) | pos_R(3) | quat_R_xyzw(4) | grip_R(1)]
        """
        a = self._action[frame_idx]
        return np.concatenate([
            a[self._L_EE_POSE][:3],   # pos_L  (x,y,z)
            a[self._L_EE_POSE][3:7],  # quat_L (qx,qy,qz,qw)
            a[self._L_GRIP],          # grip_L
            a[self._R_EE_POSE][:3],   # pos_R  (x,y,z)
            a[self._R_EE_POSE][3:7],  # quat_R (qx,qy,qz,qw)
            a[self._R_GRIP],          # grip_R
        ]).astype(np.float32)  # (16,)


# ─────────────────────────────────────────────────────────────────────────────
# Video frame helper
# ─────────────────────────────────────────────────────────────────────────────

def _read_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Read one RGB uint8 frame using decord (preferred) or cv2 (fallback)."""
    if _HAVE_DECORD:
        vr = decord.VideoReader(video_path)
        return vr[frame_idx].asnumpy()  # RGB
    elif _HAVE_CV2:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"cv2: cannot read frame {frame_idx} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise RuntimeError("Install decord or cv2 for video reading.")


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

# Human-readable names for the 16-dim action components
_ACTION_COMPONENT_NAMES = [
    "pos_L_x", "pos_L_y", "pos_L_z",
    "quat_L_x", "quat_L_y", "quat_L_z", "quat_L_w",
    "grip_L",
    "pos_R_x", "pos_R_y", "pos_R_z",
    "quat_R_x", "quat_R_y", "quat_R_z", "quat_R_w",
    "grip_R",
]


def run_episode(
    reader: LeRobotEpisodeReader,
    client: WebsocketClientPolicy,
    decoder: TeleAvatarActionDecoder,
    state_stats: TeleAvatarStateNormStats,
    instruction: Optional[str],
    chunk_size: int,
    step_stride: int,
    cfg_scale: float,
    num_ddim_steps: int,
    max_steps: int,
    ros_publisher=None,  # optional TeleavatarEndEffectorROS2Interface
) -> list[dict]:
    """
    Iterate over the episode in chunks, query the inference server, decode
    the predicted actions, and compare with ground-truth.

    Returns a list of per-step result dicts:
      {
        "frame_idx":  int,
        "pred":       np.ndarray (16,),
        "gt":         np.ndarray (16,),
        "diff":       np.ndarray (16,),
        "latency_ms": float (only nonzero at chunk start),
      }
    """
    instr = instruction or reader.instruction
    results = []
    step = 0
    frame_idx = 0

    while frame_idx < reader.num_frames:
        if max_steps > 0 and step >= max_steps:
            logger.info(f"Reached max_steps={max_steps}. Stopping.")
            break

        # ── Build observation ──────────────────────────────────────────────
        # Visual inputs in training order: [left_color, right_color, chest_camera]
        # Send as uint8 numpy arrays (H,W,3); server calls to_pil_preserve internally.
        def _resize_np(arr: np.ndarray) -> np.ndarray:
            return np.array(Image.fromarray(arr).resize((224, 224)), dtype=np.uint8)

        img_left  = _resize_np(reader.get_frame_rgb(frame_idx, "left_color"))
        img_right = _resize_np(reader.get_frame_rgb(frame_idx, "right_color"))
        img_chest = _resize_np(reader.get_frame_rgb(frame_idx, "chest_camera"))

        # State input: 18-dim normalized ee poses [B=1, T=1, 18]
        raw_state = reader.get_state_62d(frame_idx)
        state_18d = build_state_18d(raw_state, state_stats)  # (18,)

        payload = {
            "batch_images":   [[img_left, img_right, img_chest]],  # [B=1, [3 cams]] uint8 np
            "instructions":   [instr],
            "states":         [[[float(v) for v in state_18d]]],   # [B=1, T=1, 18]
            "cfg_scale":      cfg_scale,
            "use_ddim":       True,
            "num_ddim_steps": num_ddim_steps,
        }

        # ── Inference ─────────────────────────────────────────────────────
        t0 = time.time()
        response = client.infer(payload)
        latency_ms = (time.time() - t0) * 1000.0

        if not response.get("ok", True):
            logger.error(f"Server error at frame {frame_idx}: {response}")
            frame_idx += chunk_size * step_stride
            continue

        # (B=1, T=16, D=20) → (T=16, D=20)
        norm_actions = np.array(response["data"]["normalized_actions"][0])
        pred_chunk   = decoder.decode_chunk(norm_actions)   # (T=16, 16)

        logger.info(
            f"[frame {frame_idx:4d}/{reader.num_frames}]  "
            f"latency={latency_ms:.1f} ms  |  "
            f"pred_pos_L={pred_chunk[0, 0:3].round(4)}"
        )

        # ── Collect chunk steps ────────────────────────────────────────────
        for i in range(chunk_size):
            fi = frame_idx + i * step_stride
            if fi >= reader.num_frames:
                break
            if max_steps > 0 and step >= max_steps:
                break

            pred_16d = pred_chunk[i]
            gt_16d   = reader.get_gt_action_16d(fi)
            diff_16d = pred_16d - gt_16d

            results.append({
                "frame_idx":  fi,
                "pred":       pred_16d.copy(),
                "gt":         gt_16d.copy(),
                "diff":       diff_16d.copy(),
                "latency_ms": latency_ms if i == 0 else 0.0,
            })

            # Optional: publish decoded action to ROS2 topics
            if ros_publisher is not None:
                ros_publisher.publish_action(pred_16d)

            step += 1

        frame_idx += chunk_size * step_stride

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_diff_txt(results: list[dict], output_path: str) -> None:
    """
    Write predicted-vs-GT action differences to a plain-text file.

    Format (per step):
      step <N>  frame <F>  latency=<L> ms
        <component>  pred=<v>  gt=<v>  diff=<v>
      ...
      === SUMMARY ===
      MAE per component  ...
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n = len(results)
    all_diff = np.stack([r["diff"] for r in results])   # (N, 16)
    all_pred = np.stack([r["pred"] for r in results])
    all_gt   = np.stack([r["gt"]   for r in results])

    with open(output_path, "w") as f:
        f.write(f"# NeuroVLA TeleAvatar – Offline Inference Action Differences\n")
        f.write(f"# Total steps: {n}\n")
        f.write(f"# Columns: step | frame | component | pred | gt | diff\n\n")

        for step_i, r in enumerate(results):
            lat = f"  latency={r['latency_ms']:.1f} ms" if r["latency_ms"] > 0 else ""
            f.write(f"step {step_i:4d}  frame {r['frame_idx']:4d}{lat}\n")
            for j, name in enumerate(_ACTION_COMPONENT_NAMES):
                f.write(
                    f"  {name:<12}  pred={r['pred'][j]:+.6f}  "
                    f"gt={r['gt'][j]:+.6f}  diff={r['diff'][j]:+.6f}\n"
                )
            f.write("\n")

        # ── Summary ────────────────────────────────────────────────────────
        mae = np.abs(all_diff).mean(axis=0)   # (16,)
        f.write("=" * 60 + "\n")
        f.write("SUMMARY – Mean Absolute Error (MAE) per component\n")
        f.write("=" * 60 + "\n")
        for j, name in enumerate(_ACTION_COMPONENT_NAMES):
            f.write(f"  {name:<12}  MAE={mae[j]:.6f}\n")
        f.write(f"\n  Overall MAE = {mae.mean():.6f}\n")

        # Inference latency
        lats = np.array([r["latency_ms"] for r in results])
        nonzero = lats[lats > 0]
        if len(nonzero):
            f.write(
                f"\nInference latency: mean={nonzero.mean():.1f} ms  "
                f"min={nonzero.min():.1f} ms  max={nonzero.max():.1f} ms\n"
            )

    logger.info(f"Saved action differences to '{output_path}'  (steps={n})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline dataset-based inference test for NeuroVLA / TeleAvatar",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_root", required=True, type=str,
                   help="LeRobot dataset root (contains meta/, data/, videos/).")
    p.add_argument("--ckpt_path", required=True, type=str,
                   help="Checkpoint .pt path. Used to locate dataset_statistics.json "
                        "under <run_dir>/dataset_statistics.json.")
    p.add_argument("--episode", default=0, type=int,
                   help="Episode index to replay.")
    p.add_argument("--host", default="127.0.0.1", type=str,
                   help="Inference server hostname.")
    p.add_argument("--port", default=10093, type=int,
                   help="Inference server WebSocket port.")
    p.add_argument("--unnorm_key", default=None, type=str,
                   help="Key in dataset_statistics.json. Auto-detected when omitted.")
    p.add_argument("--instruction", default=None, type=str,
                   help="Override language instruction (default: from tasks.jsonl).")
    p.add_argument("--chunk_size", default=16, type=int,
                   help="Steps per chunk, max 16 (must match training action_horizon).")
    p.add_argument("--step_stride", default=1, type=int,
                   help="Frame stride within a chunk (1 = every frame).")
    p.add_argument("--cfg_scale", default=1.5, type=float,
                   help="Classifier-free guidance scale.")
    p.add_argument("--num_ddim_steps", default=4, type=int,
                   help="DDIM diffusion steps.")
    p.add_argument("--max_steps", default=0, type=int,
                   help="Hard step limit (0 = full episode).")
    p.add_argument("--output_dir", default="results", type=str,
                   help="Directory to save the output txt file.")
    p.add_argument("--publish_ros", action="store_true",
                   help="Also publish decoded actions to ROS2 topics "
                        "(requires a running ROS2 environment).")
    return p


def main():
    args = build_argparser().parse_args()

    # ── Norm stats ───────────────────────────────────────────────────────────
    action_stats = TeleAvatarNormStats(args.ckpt_path, unnorm_key=args.unnorm_key)
    state_stats  = TeleAvatarStateNormStats(args.ckpt_path, unnorm_key=args.unnorm_key)
    decoder      = TeleAvatarActionDecoder(action_stats)

    # ── Dataset reader ───────────────────────────────────────────────────────
    reader = LeRobotEpisodeReader(args.dataset_root, args.episode)

    # ── Inference server ─────────────────────────────────────────────────────
    logger.info(f"Connecting to inference server at {args.host}:{args.port} …")
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    logger.info("Connected.")

    # ── Optional ROS2 publisher ──────────────────────────────────────────────
    ros_publisher = None
    ros_spin_thread = None
    if args.publish_ros:
        import threading
        import rclpy
        from examples.TeleAvatar.ros2_interface_endeffector import (
            TeleavatarEndEffectorROS2Interface,
        )
        rclpy.init()
        ros_publisher = TeleavatarEndEffectorROS2Interface()
        ros_spin_thread = threading.Thread(
            target=rclpy.spin, args=(ros_publisher,), daemon=True
        )
        ros_spin_thread.start()
        logger.info("ROS2 publisher initialized.")

    # ── Run ──────────────────────────────────────────────────────────────────
    try:
        results = run_episode(
            reader=reader,
            client=client,
            decoder=decoder,
            state_stats=state_stats,
            instruction=args.instruction,
            chunk_size=min(args.chunk_size, 16),
            step_stride=max(args.step_stride, 1),
            cfg_scale=args.cfg_scale,
            num_ddim_steps=args.num_ddim_steps,
            max_steps=args.max_steps,
            ros_publisher=ros_publisher,
        )
    finally:
        client.close()
        if args.publish_ros:
            import rclpy
            ros_publisher.destroy_node()
            rclpy.shutdown()

    # ── Save differences ─────────────────────────────────────────────────────
    out_path = Path(args.output_dir) / f"ep{args.episode:03d}_action_diff.txt"
    save_diff_txt(results, str(out_path))


if __name__ == "__main__":
    main()
