from __future__ import annotations

import argparse
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


def write_labels(coco: dict, labels_dir: Path, image_names: set[str] | None = None) -> None:
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
    raise ValueError(f"Unsupported fusion mode: {mode}")


def convert_images(
    data_root: Path,
    out_dir: Path,
    split: str,
    image_names: list[str],
    mode: str,
    overwrite: bool,
) -> None:
    dst_dir = out_dir / "images" / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(image_names, start=1):
        dst = dst_dir / name
        if dst.exists() and not overwrite:
            continue
        rgb, tir = read_pair(data_root / split / "rgb" / name, data_root / split / "tir" / name)
        fused = fuse_rgbt(rgb, tir, mode)
        imwrite_unicode(dst, fused)
        if idx % 1000 == 0:
            print(f"[{split}] fused {idx}/{len(image_names)}")


def convert_labeled_split(
    data_root: Path,
    out_dir: Path,
    split: str,
    mode: str,
    overwrite: bool,
    max_images: int | None,
) -> None:
    coco = load_json(data_root / split / f"{split}.json")
    image_names = [image["file_name"] for image in coco["images"]]
    if max_images is not None:
        image_names = image_names[:max_images]
    convert_images(data_root, out_dir, split, image_names, mode, overwrite)
    write_labels(coco, out_dir / "labels" / split, set(image_names))


def convert_test_split(
    data_root: Path, out_dir: Path, mode: str, overwrite: bool, max_images: int | None
) -> None:
    image_names = sorted(path.name for path in (data_root / "test" / "rgb").glob("*.jpg"))
    if max_images is not None:
        image_names = image_names[:max_images]
    convert_images(data_root, out_dir, "test", image_names, mode, overwrite)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-stream RGB-T YOLO dataset by fusing paired RGB/TIR images."
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default=f"datasets/{RGBT_DATASET_NAME}")
    parser.add_argument("--yaml-out", default=f"configs/{RGBT_DATASET_NAME}.yaml")
    parser.add_argument(
        "--fusion",
        choices=["rgbt", "tir_rgb_gray", "weighted"],
        default="rgbt",
        help="rgbt stores channels as [TIR, G, R] in BGR file order, decoded as RGB-like 3-channel input by YOLO.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional smoke-test limit per split. Do not use for full training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    out_dir = Path(args.out_dir).resolve()

    for split in ("train", "val"):
        convert_labeled_split(
            data_root, out_dir, split, args.fusion, args.overwrite, args.max_images
        )
    convert_test_split(data_root, out_dir, args.fusion, args.overwrite, args.max_images)
    write_dataset_yaml(out_dir, Path(args.yaml_out))

    print(f"Prepared RGB-T YOLO dataset: {out_dir}")
    print(f"Wrote config: {args.yaml_out}")


if __name__ == "__main__":
    main()
