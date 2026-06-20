from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import (
    CLASS_NAMES,
    COCO_TO_YOLO_CATEGORY,
    RGBT_DATASET_NAME,
    load_json,
    resolve_data_root,
)


def yolo_bbox(coco_bbox: list[float], width: float, height: float) -> tuple[float, ...]:
    x, y, w, h = coco_bbox
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    return xc, yc, w / width, h / height


def write_labels(coco: dict, labels_dir: Path, image_names: set[str] | None = None, verbose: bool = True) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_ids = None
    if image_names is not None:
        image_ids = {image["id"] for image in coco["images"] if image["file_name"] in image_names}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if image_ids is not None and ann["image_id"] not in image_ids:
            continue
        if ann.get("iscrowd", 0):
            continue
        if ann["category_id"] not in COCO_TO_YOLO_CATEGORY:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        annotations_by_image[ann["image_id"]].append(ann)

    count = 0
    for image in coco["images"]:
        if image_names is not None and image["file_name"] not in image_names:
            continue
        width = float(image["width"])
        height = float(image["height"])
        lines = []
        for ann in annotations_by_image.get(image["id"], []):
            class_id = COCO_TO_YOLO_CATEGORY[ann["category_id"]]
            bbox = [min(1.0, max(0.0, value)) for value in yolo_bbox(ann["bbox"], width, height)]
            lines.append(f"{class_id} " + " ".join(f"{value:.8f}" for value in bbox))
        (labels_dir / f"{Path(image['file_name']).stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        count += 1
        if verbose and count % 1000 == 0:
            print(f"  [Labels] Written {count} label files")
    if verbose:
        print(f"  [Labels] Total label files written: {count}")


def read_pair(rgb_path: Path, tir_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = imread_unicode(rgb_path, cv2.IMREAD_COLOR)
    tir = imread_unicode(tir_path, cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    if tir is None:
        raise FileNotFoundError(tir_path)
    if rgb.shape[:2] != tir.shape[:2]:
        tir = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return rgb, tir


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def compute_brightness_stats(rgb: np.ndarray) -> dict:
    """计算图像亮度统计值，用于自适应权重计算。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    median = float(np.median(gray))
    brightness_ratio = mean / 255.0
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "brightness_ratio": brightness_ratio,
    }


def compute_adaptive_weights(brightness_ratio: float) -> tuple[float, float]:
    """根据亮度比例计算 RGB 和 TIR 的自适应权重。
    brightness_ratio: 0.0 ~ 1.0 (0=全黑, 1=全白)
    返回 (rgb_weight, tir_weight)，满足 rgb_weight + tir_weight = 1.0
    """
    k = 8.0
    median = 0.45
    rgb_weight = 1.0 / (1.0 + np.exp(-k * (brightness_ratio - median)))
    rgb_weight = float(np.clip(rgb_weight, 0.2, 0.8))
    tir_weight = 1.0 - rgb_weight
    return rgb_weight, tir_weight


def fuse_rgbt(rgb: np.ndarray, tir: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rgbt":
        b, g, r = cv2.split(rgb)
        return cv2.merge([tir, g, r])
    if mode == "tir_rgb_gray":
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        return cv2.merge([tir, gray, gray])
    if mode == "weighted":
        tir_bgr = cv2.cvtColor(tir, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(rgb, 0.65, tir_bgr, 0.35, 0.0)
    if mode == "adaptive":
        stats = compute_brightness_stats(rgb)
        rgb_w, tir_w = compute_adaptive_weights(stats["brightness_ratio"])
        tir_bgr = cv2.cvtColor(tir, cv2.COLOR_GRAY2BGR)
        fused = cv2.addWeighted(rgb, rgb_w, tir_bgr, tir_w, 0.0)
        return fused
    raise ValueError(f"Unsupported fusion mode: {mode}")


def convert_images(
    data_root: Path,
    out_dir: Path,
    split: str,
    image_names: list[str],
    mode: str,
    overwrite: bool,
    verbose: bool = True,
    progress_interval: int = 100,
) -> None:
    dst_dir = out_dir / "images" / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    weight_log = []
    total = len(image_names)
    start_time = time.time()
    
    if verbose:
        print(f"\n[========== {split.upper()} ==========]")
        print(f"  Total images to process: {total}")
        print(f"  Fusion mode: {mode}")
        print(f"  Output directory: {dst_dir}")
        print(f"  Overwrite: {overwrite}")
        print(f"  Starting at {time.strftime('%H:%M:%S')}")
        print(f"  Progress interval: every {progress_interval} images")
        print(f"{'':2}----------------------------------------")
    
    for idx, name in enumerate(image_names, start=1):
        dst = dst_dir / name
        if dst.exists() and not overwrite:
            if verbose and idx % progress_interval == 0:
                print(f"  [{split}] {idx:>6}/{total}  {name}  (skipped, already exists)")
            continue
        
        # Step 1: Read image pair
        rgb, tir = read_pair(data_root / split / "rgb" / name, data_root / split / "tir" / name)
        
        # Step 2: Fusion
        fused = fuse_rgbt(rgb, tir, mode)
        
        # Step 3: Write
        imwrite_unicode(dst, fused)
        
        # Step 4: Log (adaptive mode)
        if mode == "adaptive":
            stats = compute_brightness_stats(rgb)
            rgb_w, tir_w = compute_adaptive_weights(stats["brightness_ratio"])
            weight_log.append({
                "image": name,
                "rgb_weight": round(rgb_w, 4),
                "tir_weight": round(tir_w, 4),
                "brightness_mean": round(stats["mean"], 2),
                "brightness_ratio": round(stats["brightness_ratio"], 4),
            })
            if verbose and idx % progress_interval == 0:
                elapsed = time.time() - start_time
                speed = idx / elapsed if elapsed > 0 else 0
                eta = (total - idx) / speed if speed > 0 else 0
                print(f"  [{split}] {idx:>6}/{total}  {name}  "
                      f"BR={stats['brightness_ratio']:.3f}  RGBw={rgb_w:.3f}  TIRw={tir_w:.3f}  "
                      f"Speed={speed:.1f}img/s  ETA={eta:.0f}s")
        else:
            if verbose and idx % progress_interval == 0:
                elapsed = time.time() - start_time
                speed = idx / elapsed if elapsed > 0 else 0
                eta = (total - idx) / speed if speed > 0 else 0
                print(f"  [{split}] {idx:>6}/{total}  {name}  "
                      f"Speed={speed:.1f}img/s  ETA={eta:.0f}s")
    
    elapsed_total = time.time() - start_time
    
    if verbose:
        print(f"{'':2}----------------------------------------")
        print(f"  [{split}] Completed {total} images in {elapsed_total:.1f}s ({total/elapsed_total:.1f} img/s)")
    
    # Save adaptive weight log and summary
    if mode == "adaptive" and weight_log:
        import json
        log_path = out_dir / f"adaptive_weights_{split}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(weight_log, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"  [{split}] Saved adaptive weights log: {log_path}")
        
        rgb_weights = [w["rgb_weight"] for w in weight_log]
        tir_weights = [w["tir_weight"] for w in weight_log]
        brightness_ratios = [w["brightness_ratio"] for w in weight_log]
        if verbose:
            print(f"\n  [{split}] Adaptive weights summary:")
            print(f"    RGB weight: mean={np.mean(rgb_weights):.4f}, std={np.std(rgb_weights):.4f}, "
                  f"min={np.min(rgb_weights):.4f}, max={np.max(rgb_weights):.4f}")
            print(f"    TIR weight: mean={np.mean(tir_weights):.4f}, std={np.std(tir_weights):.4f}, "
                  f"min={np.min(tir_weights):.4f}, max={np.max(tir_weights):.4f}")
            print(f"    Brightness ratio: mean={np.mean(brightness_ratios):.4f}, std={np.std(brightness_ratios):.4f}")
            # 统计白天/夜间/黄昏分布
            night = sum(1 for b in brightness_ratios if b < 0.3)
            dusk = sum(1 for b in brightness_ratios if 0.3 <= b < 0.6)
            day = sum(1 for b in brightness_ratios if b >= 0.6)
            print(f"    Distribution: Night(<0.3)={night}  Dusk(0.3-0.6)={dusk}  Day(>=0.6)={day}")


def convert_labeled_split(
    data_root: Path, out_dir: Path, split: str, mode: str, overwrite: bool, 
    max_images: int | None, verbose: bool, progress_interval: int
) -> None:
    if verbose:
        print(f"\n>>> Processing {split.upper()} split...")
    coco = load_json(data_root / split / f"{split}.json")
    image_names = [image["file_name"] for image in coco["images"]]
    if max_images is not None:
        image_names = image_names[:max_images]
        if verbose:
            print(f"  Limited to {max_images} images (smoke test mode)")
    convert_images(data_root, out_dir, split, image_names, mode, overwrite, verbose, progress_interval)
    write_labels(coco, out_dir / "labels" / split, set(image_names), verbose)
    if verbose:
        print(f">>> {split.upper()} split done.\n")


def convert_test_split(
    data_root: Path, out_dir: Path, mode: str, overwrite: bool, 
    max_images: int | None, verbose: bool, progress_interval: int
) -> None:
    if verbose:
        print(f"\n>>> Processing TEST split...")
    image_names = sorted(path.name for path in (data_root / "test" / "rgb").glob("*.jpg"))
    if max_images is not None:
        image_names = image_names[:max_images]
        if verbose:
            print(f"  Limited to {max_images} images (smoke test mode)")
    convert_images(data_root, out_dir, "test", image_names, mode, overwrite, verbose, progress_interval)
    if verbose:
        print(f">>> TEST split done.\n")


def write_dataset_yaml(dataset_dir: Path, yaml_path: Path, verbose: bool = True) -> None:
    lines = [
        f"path: {dataset_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if verbose:
        print(f">>> Wrote YAML config: {yaml_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-stream RGB-T YOLO dataset by fusing paired RGB/TIR images."
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default=f"datasets/{RGBT_DATASET_NAME}")
    parser.add_argument("--yaml-out", default=f"configs/{RGBT_DATASET_NAME}.yaml")
    parser.add_argument(
        "--fusion",
        choices=["rgbt", "tir_rgb_gray", "weighted", "adaptive"],
        default="rgbt",
        help="rgbt: [TIR, G, R]; tir_rgb_gray: [TIR, gray, gray]; weighted: fixed 0.65/0.35; adaptive: illumination-aware dynamic weights",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional smoke-test limit per split. Do not use for full training.",
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Print progress to terminal (default: True)")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress terminal output")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N images (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = args.verbose and not args.quiet
    data_root = resolve_data_root(args.data_root)
    out_dir = Path(args.out_dir).resolve()
    
    if verbose:
        print("=" * 60)
        print("  RGB-T YOLO Dataset Preparation")
        print("=" * 60)
        print(f"  Data root: {data_root}")
        print(f"  Output dir: {out_dir}")
        print(f"  Fusion mode: {args.fusion}")
        print(f"  Progress interval: every {args.progress_interval} images")
        print(f"  Start time: {time.strftime('%H:%M:%S')}")
        print("=" * 60)
        sys.stdout.flush()

    for split in ("train", "val"):
        convert_labeled_split(
            data_root, out_dir, split, args.fusion, args.overwrite, 
            args.max_images, verbose, args.progress_interval
        )
    convert_test_split(
        data_root, out_dir, args.fusion, args.overwrite, 
        args.max_images, verbose, args.progress_interval
    )
    write_dataset_yaml(out_dir, Path(args.yaml_out), verbose)

    if verbose:
        print("=" * 60)
        print(f"  ALL DONE")
        print(f"  Dataset: {out_dir}")
        print(f"  Config: {args.yaml_out}")
        print(f"  End time: {time.strftime('%H:%M:%S')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
