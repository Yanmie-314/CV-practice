from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import resolve_data_root, save_json


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def sample_paths(path: Path, limit: int | None) -> list[Path]:
    paths = sorted(path.glob("*.jpg"))
    if limit is not None:
        return paths[:limit]
    return paths


def image_features(rgb_path: Path, tir_path: Path) -> dict[str, float]:
    rgb = imread_unicode(rgb_path, cv2.IMREAD_COLOR)
    tir = imread_unicode(tir_path, cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    if tir is None:
        raise FileNotFoundError(tir_path)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return {
        "rgb_mean": float(rgb.mean()),
        "rgb_std": float(rgb.std()),
        "rgb_b": float(rgb[:, :, 0].mean()),
        "rgb_g": float(rgb[:, :, 1].mean()),
        "rgb_r": float(rgb[:, :, 2].mean()),
        "sat_mean": float(hsv[:, :, 1].mean()),
        "value_mean": float(hsv[:, :, 2].mean()),
        "gray_lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "tir_mean": float(tir.mean()),
        "tir_std": float(tir.std()),
        "tir_lap_var": float(cv2.Laplacian(tir, cv2.CV_64F).var()),
    }


def summarize(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = records[0].keys()
    summary = {}
    for key in keys:
        values = np.array([record[key] for record in records], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }
    return summary


def collect_split(data_root: Path, split: str, limit: int | None) -> dict[str, dict[str, float]]:
    rgb_dir = data_root / split / "rgb"
    tir_dir = data_root / split / "tir"
    records = []
    for rgb_path in sample_paths(rgb_dir, limit):
        records.append(image_features(rgb_path, tir_dir / rgb_path.name))
    return summarize(records)


def ratio(test_value: float, train_value: float, default: float = 1.0) -> float:
    if abs(train_value) < 1e-6:
        return default
    return float(test_value / train_value)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_aug_config(stats: dict) -> dict:
    train = stats["train"]
    test = stats["test"]
    brightness_delta = test["value_mean"]["mean"] - train["value_mean"]["mean"]
    tir_brightness_delta = test["tir_mean"]["mean"] - train["tir_mean"]["mean"]
    contrast_ratio = ratio(test["rgb_std"]["mean"], train["rgb_std"]["mean"])
    tir_contrast_ratio = ratio(test["tir_std"]["mean"], train["tir_std"]["mean"])
    blur_ratio = ratio(test["gray_lap_var"]["mean"], train["gray_lap_var"]["mean"])

    return {
        "seed": 2026,
        "fusion": "rgbt",
        "probabilities": {
            "brightness_contrast_gamma": 0.75,
            "tir_contrast": 0.60,
            "blur": 0.30,
            "noise": 0.25,
            "jpeg": 0.25,
        },
        "brightness_delta_range": [
            int(clamp(brightness_delta - 12, -45, 45)),
            int(clamp(brightness_delta + 12, -45, 45)),
        ],
        "contrast_range": [
            round(clamp(contrast_ratio * 0.90, 0.70, 1.30), 3),
            round(clamp(contrast_ratio * 1.10, 0.70, 1.30), 3),
        ],
        "gamma_range": [0.85, 1.20],
        "tir_brightness_delta_range": [
            int(clamp(tir_brightness_delta - 10, -40, 40)),
            int(clamp(tir_brightness_delta + 10, -40, 40)),
        ],
        "tir_contrast_range": [
            round(clamp(tir_contrast_ratio * 0.90, 0.70, 1.35), 3),
            round(clamp(tir_contrast_ratio * 1.10, 0.70, 1.35), 3),
        ],
        "blur_kernel_choices": [3, 5] if blur_ratio < 0.85 else [3],
        "noise_std_range": [1.0, 6.0],
        "jpeg_quality_range": [65, 92],
    }


def write_report(stats: dict, aug_config: dict, out_path: Path) -> None:
    lines = [
        "# Domain Shift Report",
        "",
        "## Summary",
        "",
        "| feature | train mean | test mean | test/train |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in stats["train"].keys():
        train_mean = stats["train"][key]["mean"]
        test_mean = stats["test"][key]["mean"]
        lines.append(f"| {key} | {train_mean:.3f} | {test_mean:.3f} | {ratio(test_mean, train_mean):.3f} |")
    lines.extend(
        [
            "",
            "## Generated Augmentation Config",
            "",
            "```json",
            json.dumps(aug_config, ensure_ascii=False, indent=2),
            "```",
            "",
            "All selected transforms are pixel-only, so bounding boxes remain unchanged.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze train/test image domain shift.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--sample-limit", type=int, default=2000)
    parser.add_argument("--stats-out", default="analysis/domain_stats.json")
    parser.add_argument("--config-out", default="configs/domain_aug_rgbt.json")
    parser.add_argument("--report-out", default="analysis/domain_shift_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    stats = {
        "train": collect_split(data_root, "train", args.sample_limit),
        "test": collect_split(data_root, "test", args.sample_limit),
    }
    aug_config = build_aug_config(stats)
    save_json(stats, args.stats_out)
    save_json(aug_config, args.config_out)
    write_report(stats, aug_config, Path(args.report_out))
    print(f"Wrote stats: {args.stats_out}")
    print(f"Wrote augmentation config: {args.config_out}")
    print(f"Wrote report: {args.report_out}")


if __name__ == "__main__":
    main()
