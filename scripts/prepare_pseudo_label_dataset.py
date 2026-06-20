from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from common import CLASS_NAMES, COCO_TO_YOLO_CATEGORY, load_json


IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 512.0


def yolo_bbox(coco_bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in coco_bbox]
    x = max(0.0, min(IMAGE_WIDTH, x))
    y = max(0.0, min(IMAGE_HEIGHT, y))
    w = max(0.0, min(IMAGE_WIDTH - x, w))
    h = max(0.0, min(IMAGE_HEIGHT - y, h))
    xc = (x + w / 2.0) / IMAGE_WIDTH
    yc = (y + h / 2.0) / IMAGE_HEIGHT
    return xc, yc, w / IMAGE_WIDTH, h / IMAGE_HEIGHT


def copy_tree_files(src: Path, dst: Path, overwrite: bool) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(src.glob("*")):
        if not path.is_file():
            continue
        target = dst / path.name
        if target.exists() and not overwrite:
            count += 1
            continue
        shutil.copy2(path, target)
        count += 1
    return count


def write_pseudo_labels(
    predictions: list[dict],
    src_images: Path,
    dst_images: Path,
    dst_labels: Path,
    score_thr: float,
    max_per_image: int,
    overwrite: bool,
) -> tuple[int, int, Counter]:
    by_image: dict[int, list[dict]] = defaultdict(list)
    for det in predictions:
        score = float(det["score"])
        category_id = int(det["category_id"])
        if score < score_thr or category_id not in COCO_TO_YOLO_CATEGORY:
            continue
        by_image[int(det["image_id"])].append(det)

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    per_class: Counter = Counter()
    image_count = 0
    box_count = 0
    for image_id, detections in sorted(by_image.items()):
        name = f"{image_id:05d}.jpg"
        src = src_images / name
        if not src.exists():
            raise FileNotFoundError(src)
        pseudo_name = f"pseudo_{name}"
        image_dst = dst_images / pseudo_name
        if overwrite or not image_dst.exists():
            shutil.copy2(src, image_dst)

        lines = []
        kept = sorted(detections, key=lambda det: float(det["score"]), reverse=True)[
            :max_per_image
        ]
        for det in kept:
            x, y, w, h = yolo_bbox(det["bbox"])
            if w <= 0 or h <= 0:
                continue
            class_id = COCO_TO_YOLO_CATEGORY[int(det["category_id"])]
            per_class[class_id] += 1
            box_count += 1
            lines.append(f"{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
        label_dst = dst_labels / f"pseudo_{image_id:05d}.txt"
        label_dst.write_text("\n".join(lines), encoding="utf-8")
        image_count += 1
    return image_count, box_count, per_class


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
        description="Build a YOLO dataset with high-confidence test pseudo labels."
    )
    parser.add_argument("--base-dir", default="datasets/gaiic_yolo_rgbt_aug_domain")
    parser.add_argument("--pred", required=True, help="COCO detection JSON for test.")
    parser.add_argument("--out-dir", default="datasets/gaiic_yolo_rgbt_pseudo_wbf035")
    parser.add_argument("--yaml-out", default="configs/gaiic_yolo_rgbt_pseudo_wbf035.yaml")
    parser.add_argument("--score-thr", type=float, default=0.35)
    parser.add_argument("--max-per-image", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    train_images = copy_tree_files(
        base_dir / "images" / "train", out_dir / "images" / "train", args.overwrite
    )
    train_labels = copy_tree_files(
        base_dir / "labels" / "train", out_dir / "labels" / "train", args.overwrite
    )
    val_images = copy_tree_files(
        base_dir / "images" / "val", out_dir / "images" / "val", args.overwrite
    )
    val_labels = copy_tree_files(
        base_dir / "labels" / "val", out_dir / "labels" / "val", args.overwrite
    )
    test_images = copy_tree_files(
        base_dir / "images" / "test", out_dir / "images" / "test", args.overwrite
    )

    pseudo_images, pseudo_boxes, per_class = write_pseudo_labels(
        load_json(args.pred),
        base_dir / "images" / "test",
        out_dir / "images" / "train",
        out_dir / "labels" / "train",
        args.score_thr,
        args.max_per_image,
        args.overwrite,
    )
    write_dataset_yaml(out_dir, Path(args.yaml_out))

    print(f"Copied train images/labels: {train_images}/{train_labels}")
    print(f"Copied val images/labels: {val_images}/{val_labels}")
    print(f"Copied test images: {test_images}")
    print(f"Added pseudo images: {pseudo_images}")
    print(f"Added pseudo boxes: {pseudo_boxes}")
    print("Pseudo boxes per class:")
    for class_id, count in sorted(per_class.items()):
        print(f"  {class_id} {CLASS_NAMES[class_id]}: {count}")
    print(f"Prepared pseudo-label dataset: {out_dir.resolve()}")
    print(f"Wrote config: {args.yaml_out}")


if __name__ == "__main__":
    main()
