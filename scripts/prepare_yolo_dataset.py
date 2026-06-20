from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from common import CLASS_NAMES, COCO_TO_YOLO_CATEGORY, load_json, resolve_data_root


def yolo_bbox(coco_bbox: list[float], width: float, height: float) -> tuple[float, ...]:
    x, y, w, h = coco_bbox
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    return xc, yc, w / width, h / height


def write_labels(coco: dict, labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    images = {image["id"]: image for image in coco["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)

    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        if ann["category_id"] not in COCO_TO_YOLO_CATEGORY:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        annotations_by_image[ann["image_id"]].append(ann)

    for image in coco["images"]:
        width = float(image["width"])
        height = float(image["height"])
        stem = Path(image["file_name"]).stem
        lines = []
        for ann in annotations_by_image.get(image["id"], []):
            class_id = COCO_TO_YOLO_CATEGORY[ann["category_id"]]
            bbox = yolo_bbox(ann["bbox"], width, height)
            clipped = [
                min(1.0, max(0.0, value))
                for value in bbox
            ]
            lines.append(
                f"{class_id} "
                + " ".join(f"{value:.8f}" for value in clipped)
            )
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def link_or_copy_images(src_dir: Path, dst_dir: Path, image_names: list[str], mode: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in image_names:
        src = src_dir / name
        dst = dst_dir / name
        if dst.exists():
            continue
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "hardlink":
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        else:
            try:
                dst.symlink_to(src)
            except OSError:
                try:
                    dst.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)


def write_dataset_yaml(dataset_dir: Path, yaml_path: Path) -> None:
    # Use absolute path so Ultralytics can locate the dataset regardless of
    # where the training command is invoked from.
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


def convert_split(data_root: Path, output_dir: Path, modality: str, split: str, mode: str) -> None:
    ann_path = data_root / split / f"{split}.json"
    coco = load_json(ann_path)
    image_names = [image["file_name"] for image in coco["images"]]
    link_or_copy_images(
        src_dir=data_root / split / modality,
        dst_dir=output_dir / "images" / split,
        image_names=image_names,
        mode=mode,
    )
    write_labels(coco, output_dir / "labels" / split)


def prepare_test_images(data_root: Path, output_dir: Path, modality: str, mode: str) -> None:
    src_dir = data_root / "test" / modality
    image_names = sorted(p.name for p in src_dir.glob("*.jpg"))
    link_or_copy_images(src_dir, output_dir / "images" / "test", image_names, mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the GAIIC COCO dataset to an Ultralytics YOLO dataset."
    )
    parser.add_argument("--data-root", default=None, help="Path to GAIIC2024 dataset root.")
    parser.add_argument("--modality", choices=["rgb", "tir"], required=True)
    parser.add_argument("--out-dir", default=None, help="Output dataset directory.")
    parser.add_argument(
        "--image-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="Use hardlinks by default to avoid duplicating images on Windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    out_dir = Path(args.out_dir or f"datasets/gaiic_yolo_{args.modality}").resolve()

    for split in ("train", "val"):
        convert_split(data_root, out_dir, args.modality, split, args.image_mode)
    prepare_test_images(data_root, out_dir, args.modality, args.image_mode)
    write_dataset_yaml(out_dir, Path("configs") / f"gaiic_yolo_{args.modality}.yaml")

    print(f"Prepared YOLO dataset: {out_dir}")
    print(f"Wrote config: configs/gaiic_yolo_{args.modality}.yaml")


if __name__ == "__main__":
    main()
