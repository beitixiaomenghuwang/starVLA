#!/usr/bin/env python3
"""Add coarse/fine wavelet phase labels to a LeRobot dataset.

The generated per-frame column is:
    phase_label: 0 = coarse, 1 = fine
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pywt


def iter_episode_paths(dataset_root: Path) -> list[Path]:
    data_root = dataset_root / "data"
    return sorted(data_root.glob("chunk-*/episode_*.parquet"))


def read_action(path: Path, action_column: str) -> np.ndarray:
    table = pq.read_table(path, columns=[action_column])
    return np.asarray(table[action_column].to_pylist(), dtype=np.float32)


def robust_standardize(action: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale dimensions for phase detection without changing stored actions."""
    median = np.nanmedian(action, axis=0)
    q25 = np.nanpercentile(action, 25, axis=0)
    q75 = np.nanpercentile(action, 75, axis=0)
    scale = q75 - q25
    std = np.nanstd(action, axis=0)
    scale = np.where(scale > eps, scale, std)
    keep = scale > eps
    if not np.any(keep):
        return np.zeros((action.shape[0], 1), dtype=np.float32)
    return ((action[:, keep] - median[keep]) / scale[keep]).astype(np.float32)


def reconstruct_components(
    signal: np.ndarray,
    wavelet: str,
    level: int,
    mode: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return cA_level and detail components cD1..cD_level at original length."""
    coeffs = pywt.wavedec(signal, wavelet=wavelet, mode=mode, level=level)
    length = signal.shape[0]

    approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    approx = pywt.waverec(approx_coeffs, wavelet=wavelet, mode=mode)[:length]

    details_by_user_order: list[np.ndarray] = []
    for user_level in range(1, level + 1):
        coeff_index = level - user_level + 1
        detail_coeffs = [np.zeros_like(c) for c in coeffs]
        detail_coeffs[coeff_index] = coeffs[coeff_index]
        detail = pywt.waverec(detail_coeffs, wavelet=wavelet, mode=mode)[:length]
        details_by_user_order.append(detail)

    return approx, details_by_user_order


def energy_ratio(
    action: np.ndarray,
    wavelet: str,
    level: int,
    mode: str,
    standardize: bool,
    eps: float = 1e-8,
) -> np.ndarray:
    x = robust_standardize(action) if standardize else action.astype(np.float32)
    if x.shape[0] < 2:
        return np.zeros((x.shape[0],), dtype=np.float32)

    max_level = pywt.dwt_max_level(x.shape[0], pywt.Wavelet(wavelet).dec_len)
    use_level = max(1, min(level, max_level))

    approx_all = []
    detail_all = []
    for dim in range(x.shape[1]):
        approx, details = reconstruct_components(x[:, dim], wavelet, use_level, mode)
        approx_all.append(approx)
        detail_all.extend(details)

    approx_arr = np.stack(approx_all, axis=1)
    detail_arr = np.stack(detail_all, axis=1)

    high_freq = np.sqrt(np.mean(np.square(detail_arr), axis=1))
    low_freq = np.sqrt(np.mean(np.square(approx_arr), axis=1))
    return (high_freq / (low_freq + high_freq + eps)).astype(np.float32)


def median_filter_binary(labels: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return labels.astype(np.int64)
    if window % 2 == 0:
        raise ValueError("median filter window must be odd")
    radius = window // 2
    padded = np.pad(labels.astype(np.int64), (radius, radius), mode="edge")
    out = np.empty_like(labels, dtype=np.int64)
    for i in range(labels.shape[0]):
        out[i] = int(np.median(padded[i : i + window]))
    return out


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Compute a global 1D Otsu threshold without extra dependencies."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    value_min = float(values.min())
    value_max = float(values.max())
    if value_min == value_max:
        return value_min

    hist, edges = np.histogram(values, bins=bins, range=(value_min, value_max))
    centers = (edges[:-1] + edges[1:]) * 0.5
    hist = hist.astype(np.float64)

    weight_low = np.cumsum(hist)
    weight_high = np.cumsum(hist[::-1])[::-1]
    mean_low = np.cumsum(hist * centers) / np.maximum(weight_low, 1.0)
    mean_high = (
        np.cumsum((hist * centers)[::-1]) / np.maximum(weight_high[::-1], 1.0)
    )[::-1]

    variance = weight_low[:-1] * weight_high[1:] * np.square(mean_low[:-1] - mean_high[1:])
    return float(centers[int(np.argmax(variance))])


def choose_threshold(values: np.ndarray, method: str, percentile: float, otsu_bins: int) -> float:
    if method == "percentile":
        return float(np.percentile(values, percentile))
    if method == "otsu":
        return otsu_threshold(values, bins=otsu_bins)
    raise ValueError(f"Unsupported threshold method: {method}")


def merge_short_segments(labels: np.ndarray, min_frames: int) -> np.ndarray:
    """Merge very short islands into the longer neighboring phase."""
    if min_frames <= 1 or labels.size == 0:
        return labels.astype(np.int64)

    out = labels.astype(np.int64).copy()
    changed = True
    while changed:
        changed = False
        segments = binary_segments(out)
        if len(segments) <= 1:
            break
        for idx, segment in enumerate(segments):
            if segment["length"] >= min_frames:
                continue
            if idx == 0:
                fill = segments[idx + 1]["label"]
            elif idx == len(segments) - 1:
                fill = segments[idx - 1]["label"]
            else:
                left = segments[idx - 1]
                right = segments[idx + 1]
                fill = left["label"] if left["length"] >= right["length"] else right["label"]
            out[segment["start_frame"] : segment["end_frame"] + 1] = fill
            changed = True
            break
    return out


def expand_fine_segments(labels: np.ndarray, pre_frames: int, post_frames: int) -> np.ndarray:
    """Expand fine islands so event peaks also cover preparation/alignment frames."""
    if labels.size == 0 or (pre_frames <= 0 and post_frames <= 0):
        return labels.astype(np.int64)

    out = labels.astype(np.int64).copy()
    fine_indices = np.flatnonzero(labels == 1)
    if fine_indices.size == 0:
        return out

    for segment in binary_segments(labels):
        if segment["label"] != 1:
            continue
        start = max(0, segment["start_frame"] - pre_frames)
        end = min(labels.shape[0] - 1, segment["end_frame"] + post_frames)
        out[start : end + 1] = 1
    return out


def close_short_gaps(labels: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Fill short coarse gaps between two fine segments."""
    if labels.size == 0 or max_gap_frames <= 0:
        return labels.astype(np.int64)

    out = labels.astype(np.int64).copy()
    segments = binary_segments(out)
    for idx, segment in enumerate(segments):
        if segment["label"] != 0 or segment["length"] > max_gap_frames:
            continue
        if idx == 0 or idx == len(segments) - 1:
            continue
        if segments[idx - 1]["label"] == 1 and segments[idx + 1]["label"] == 1:
            out[segment["start_frame"] : segment["end_frame"] + 1] = 1
    return out


def binary_segments(labels: np.ndarray) -> list[dict]:
    if labels.size == 0:
        return []

    segments = []
    start = 0
    current = int(labels[0])
    for i in range(1, labels.shape[0]):
        value = int(labels[i])
        if value == current:
            continue
        segments.append(
            {
                "label": current,
                "phase": "fine" if current == 1 else "coarse",
                "start_frame": start,
                "end_frame": i - 1,
                "length": i - start,
            }
        )
        start = i
        current = value

    segments.append(
        {
            "label": current,
            "phase": "fine" if current == 1 else "coarse",
            "start_frame": start,
            "end_frame": int(labels.shape[0] - 1),
            "length": int(labels.shape[0] - start),
        }
    )
    return segments


def replace_or_append_column(table: pa.Table, name: str, values: Iterable[int]) -> pa.Table:
    array = pa.array(list(values), type=pa.int64())
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, array)
    return table.append_column(name, array)


def update_info_json(dataset_root: Path, label_column: str) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    features = info.setdefault("features", {})
    features[label_column] = {
        "dtype": "int64",
        "shape": [1],
        "names": ["coarse_fine_phase"],
        "description": "Wavelet phase label: 0=coarse, 1=fine",
    }

    tmp_path = info_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, info_path)


def write_summary(dataset_root: Path, summary: dict) -> None:
    out_path = dataset_root / "meta" / "wavelet_phase_labels.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--action-column", default="action")
    parser.add_argument("--label-column", default="phase_label")
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--mode", default="symmetric")
    parser.add_argument("--threshold-method", choices=["otsu", "percentile"], default="otsu")
    parser.add_argument("--percentile", type=float, default=65.0)
    parser.add_argument("--otsu-bins", type=int, default=256)
    parser.add_argument("--median-window", type=int, default=31)
    parser.add_argument("--fine-pre-frames", type=int, default=45)
    parser.add_argument("--fine-post-frames", type=int, default=15)
    parser.add_argument("--merge-gap-frames", type=int, default=30)
    parser.add_argument("--min-segment-frames", type=int, default=15)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    paths = iter_episode_paths(dataset_root)
    if not paths:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_root / 'data'}")

    ratios_by_path: dict[Path, np.ndarray] = {}
    all_ratios = []
    for path in paths:
        action = read_action(path, args.action_column)
        ratio = energy_ratio(
            action,
            wavelet=args.wavelet,
            level=args.level,
            mode=args.mode,
            standardize=not args.no_standardize,
        )
        ratios_by_path[path] = ratio
        all_ratios.append(ratio)

    flat_ratios = np.concatenate(all_ratios)
    tau = choose_threshold(
        flat_ratios,
        method=args.threshold_method,
        percentile=args.percentile,
        otsu_bins=args.otsu_bins,
    )

    fine_frames = 0
    total_frames = 0
    episode_summaries = []
    for path in paths:
        ratio = ratios_by_path[path]
        raw = (ratio > tau).astype(np.int64)
        labels = median_filter_binary(raw, args.median_window)
        labels = expand_fine_segments(labels, args.fine_pre_frames, args.fine_post_frames)
        labels = close_short_gaps(labels, args.merge_gap_frames)
        labels = merge_short_segments(labels, args.min_segment_frames)
        fine_frames += int(labels.sum())
        total_frames += int(labels.shape[0])
        episode_summaries.append(
            {
                "episode_file": str(path.relative_to(dataset_root)),
                "frames": int(labels.shape[0]),
                "fine_frames": int(labels.sum()),
                "fine_ratio": float(labels.mean()) if labels.size else 0.0,
                "segments": binary_segments(labels),
            }
        )

        if not args.dry_run:
            table = pq.read_table(path)
            table = replace_or_append_column(table, args.label_column, labels)
            tmp_path = path.with_suffix(".parquet.tmp")
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, path)

    summary = {
        "label_column": args.label_column,
        "action_column": args.action_column,
        "wavelet": args.wavelet,
        "requested_level": args.level,
        "mode": args.mode,
        "threshold_method": args.threshold_method,
        "percentile": args.percentile,
        "otsu_bins": args.otsu_bins,
        "tau_phase": tau,
        "median_window": args.median_window,
        "fine_pre_frames": args.fine_pre_frames,
        "fine_post_frames": args.fine_post_frames,
        "merge_gap_frames": args.merge_gap_frames,
        "min_segment_frames": args.min_segment_frames,
        "standardized_per_episode": not args.no_standardize,
        "episodes": len(paths),
        "total_frames": total_frames,
        "fine_frames": fine_frames,
        "fine_ratio_after_median": float(fine_frames / total_frames),
        "episode_summaries": episode_summaries,
    }

    if not args.dry_run:
        update_info_json(dataset_root, args.label_column)
        write_summary(dataset_root, summary)

    print(json.dumps({k: v for k, v in summary.items() if k != "episode_summaries"}, indent=2))


if __name__ == "__main__":
    main()
