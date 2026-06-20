from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import CLASS_NAMES, COCO_TO_YOLO_CATEGORY, load_json, resolve_data_root
from prepare_rgbt_yolo_dataset import imread_unicode, imwrite_unicode


def yolo_bbox(coco_bbox: list[float], width: float, height: float) -> tuple[float, ...]:
    x, y, w, h = coco_bbox
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    return xc, yc, w / width, h / height


def write_labels(coco: dict, labels_dir: Path, image_names: set[str] | None = None) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_ids = None
    if image_names is not None:
        image_ids = {
            image["id"] for image in coco["images"] if image["file_name"] in image_names
        }

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if image_ids is not None and ann["image_id"] not in image_ids:
            continue
        if ann.get("iscrowd", 0) or ann["category_id"] not in COCO_TO_YOLO_CATEGORY:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        annotations_by_image[ann["image_id"]].append(ann)

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


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, (1, 99))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    out = (image - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def dct_filter_channel(channel: np.ndarray, keep_ratio: float, mode: str) -> np.ndarray:
    if channel.ndim != 2:
        raise ValueError("DCT input must be a single channel image.")
    h, w = channel.shape
    dct = cv2.dct(channel.astype(np.float32) / 255.0)
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy / max(h, 1)) ** 2 + (xx / max(w, 1)) ** 2)
    threshold = float(keep_ratio)
    if mode == "low":
        mask = radius <= threshold
    elif mode == "high":
        mask = radius > threshold
    else:
        raise ValueError(f"Unsupported DCT mode: {mode}")
    filtered = cv2.idct(dct * mask.astype(np.float32))
    return normalize_uint8(filtered)


def read_pair(data_root: Path, split: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = imread_unicode(data_root / split / "rgb" / name, cv2.IMREAD_COLOR)
    tir = imread_unicode(data_root / split / "tir" / name, cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise FileNotFoundError(data_root / split / "rgb" / name)
    if tir is None:
        raise FileNotFoundError(data_root / split / "tir" / name)
    if rgb.shape[:2] != tir.shape[:2]:
        tir = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return rgb, tir


def fd2_fuse(
    rgb: np.ndarray,
    tir: np.ndarray,
    low_keep: float,
    high_keep: float,
    base_rgb_weight: float,
) -> np.ndarray:
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    tir_low = dct_filter_channel(tir, low_keep, "low")
    rgb_high = dct_filter_channel(rgb_gray, high_keep, "high")
    base = cv2.addWeighted(rgb_gray, base_rgb_weight, tir, 1.0 - base_rgb_weight, 0.0)

    # BGR file order. YOLO sees a standard 3-channel image, while each channel
    # carries a frequency-decomposed RGB-T cue.
    return cv2.merge([tir_low, rgb_high, base])


def convert_images(
    data_root: Path,
    out_dir: Path,
    split: str,
    image_names: list[str],
    low_keep: float,
    high_keep: float,
    base_rgb_weight: float,
    overwrite: bool,
) -> None:
    dst_dir = out_dir / "images" / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(image_names, start=1):
        dst = dst_dir / name
        if dst.exists() and not overwrite:
            continue
        rgb, tir = read_pair(data_root, split, name)
        fused = fd2_fuse(rgb, tir, low_keep, high_keep, base_rgb_weight)
        imwrite_unicode(dst, fused)
        if idx % 1000 == 0 or idx == len(image_names):
            print(f"[{split}] FD2-fused {idx}/{len(image_names)}")


def convert_labeled_split(
    data_root: Path,
    out_dir: Path,
    split: str,
    low_keep: float,
    high_keep: float,
    base_rgb_weight: float,
    overwrite: bool,
    max_images: int | None,
) -> None:
    coco = load_json(data_root / split / f"{split}.json")
    image_names = [image["file_name"] for image in coco["images"]]
    if max_images is not None:
        image_names = image_names[:max_images]
    convert_images(
        data_root,
        out_dir,
        split,
        image_names,
        low_keep,
        high_keep,
        base_rgb_weight,
        overwrite,
    )
    write_labels(coco, out_dir / "labels" / split, set(image_names))


def convert_test_split(
    data_root: Path,
    out_dir: Path,
    low_keep: float,
    high_keep: float,
    base_rgb_weight: float,
    overwrite: bool,
    max_images: int | None,
) -> None:
    image_names = sorted(path.name for path in (data_root / "test" / "rgb").glob("*.jpg"))
    if max_images is not None:
        image_names = image_names[:max_images]
    convert_images(
        data_root,
        out_dir,
        "test",
        image_names,
        low_keep,
        high_keep,
        base_rgb_weight,
        overwrite,
    )


def write_dataset_yaml(dataset_dir: Path, yaml_path: Path) -> None:
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


def copy_preview(out_dir: Path) -> None:
    preview_dir = Path("visualizations") / "fd2_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for image in sorted((out_dir / "images" / "train").glob("*.jpg"))[:8]:
        shutil.copy2(image, preview_dir / image.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an FD2-style RGB-T YOLO dataset. The output encodes TIR low "
            "frequency, RGB high frequency, and base luminance fusion as three channels."
        )
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default="datasets/gaiic_yolo_rgbt_fd2")
    parser.add_argument("--yaml-out", default="configs/gaiic_yolo_rgbt_fd2.yaml")
    parser.add_argument("--low-keep", type=float, default=0.12)
    parser.add_argument("--high-keep", type=float, default=0.18)
    parser.add_argument("--base-rgb-weight", type=float, default=0.55)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    out_dir = Path(args.out_dir).resolve()

    for split in ("train", "val"):
        convert_labeled_split(
            data_root,
            out_dir,
            split,
            args.low_keep,
            args.high_keep,
            args.base_rgb_weight,
            args.overwrite,
            args.max_images,
        )
    convert_test_split(
        data_root,
        out_dir,
        args.low_keep,
        args.high_keep,
        args.base_rgb_weight,
        args.overwrite,
        args.max_images,
    )
    write_dataset_yaml(out_dir, Path(args.yaml_out))
    if args.preview:
        copy_preview(out_dir)

    print(f"Prepared FD2 RGB-T YOLO dataset: {out_dir}")
    print(f"Wrote config: {args.yaml_out}")
    print(
        "FD2 channels: B=TIR low-frequency, G=RGB high-frequency, "
        "R=RGB/TIR base luminance fusion"
    )


if __name__ == "__main__":
    main()
