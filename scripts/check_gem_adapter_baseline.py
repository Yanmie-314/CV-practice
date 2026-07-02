from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import resolve_data_root
from datasets.rgbt_dual_dataset import image_to_tensor, imread_unicode, letterbox
from models.gem_adapter_yolo import GEMAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify GEM adapter baseline matches the pretrained RGB-T fused-image input."
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--fused-root", default="datasets/gaiic_yolo_rgbt_aug_domain")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument(
        "--report-candidates",
        action="store_true",
        help="Also report simple alternative channel orders for diagnosis.",
    )
    return parser.parse_args()


def load_pair(data_root: Path, split: str, name: str, imgsz: int) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = imread_unicode(data_root / split / "rgb" / name, cv2.IMREAD_COLOR)
    tir = imread_unicode(data_root / split / "tir" / name, cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise FileNotFoundError(data_root / split / "rgb" / name)
    if tir is None:
        raise FileNotFoundError(data_root / split / "tir" / name)
    if rgb.shape[:2] != tir.shape[:2]:
        tir = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    rgb_lb, _, _ = letterbox(rgb, imgsz)
    tir_lb, _, _ = letterbox(tir, imgsz)
    return image_to_tensor(rgb_lb, channels=3), image_to_tensor(tir_lb, channels=1)


def load_fused(fused_root: Path, split: str, name: str, imgsz: int) -> torch.Tensor:
    fused = imread_unicode(fused_root / "images" / split / name, cv2.IMREAD_COLOR)
    if fused is None:
        raise FileNotFoundError(fused_root / "images" / split / name)
    fused_lb, _, _ = letterbox(fused, imgsz)
    return image_to_tensor(fused_lb, channels=3)


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    fused_root = Path(args.fused_root)
    names = sorted(path.name for path in (data_root / args.split / "rgb").glob("*.jpg"))
    if args.max_images is not None:
        names = names[: args.max_images]
    if not names:
        raise SystemExit(f"No images found in {data_root / args.split / 'rgb'}")

    max_abs_error = 0.0
    mean_abs_error = 0.0
    candidate_stats = {
        "R,G,TIR": [0.0, 0.0],
        "TIR,G,R": [0.0, 0.0],
        "R,G,B": [0.0, 0.0],
        "B,G,R": [0.0, 0.0],
    }
    for name in names:
        rgb, tir = load_pair(data_root, args.split, name, args.imgsz)
        expected = load_fused(fused_root, args.split, name, args.imgsz)
        actual = GEMAdapter.baseline_rgbt(rgb.unsqueeze(0), tir.unsqueeze(0)).squeeze(0)
        diff = (actual - expected).abs()
        max_abs_error = max(max_abs_error, float(diff.max()))
        mean_abs_error += float(diff.mean())
        if args.report_candidates:
            candidates = {
                "R,G,TIR": actual,
                "TIR,G,R": torch.cat((tir, rgb[1:2], rgb[0:1]), dim=0),
                "R,G,B": rgb,
                "B,G,R": torch.cat((rgb[2:3], rgb[1:2], rgb[0:1]), dim=0),
            }
            for order, candidate in candidates.items():
                candidate_diff = (candidate - expected).abs()
                candidate_stats[order][0] += float(candidate_diff.mean())
                candidate_stats[order][1] = max(candidate_stats[order][1], float(candidate_diff.max()))
    mean_abs_error /= len(names)

    print(
        f"Checked {len(names)} images | baseline_tensor_order={GEMAdapter.baseline_tensor_order} "
        f"| max_abs_error={max_abs_error:.8f} | mean_abs_error={mean_abs_error:.8f}"
    )
    if args.report_candidates:
        for order, (mean_error, max_error) in candidate_stats.items():
            print(f"  {order:<8} mean_abs_error={mean_error / len(names):.8f} max_abs_error={max_error:.8f}")

    if mean_abs_error > 0.02:
        raise SystemExit("Baseline mismatch mean error is too large.")


if __name__ == "__main__":
    main()
