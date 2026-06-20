from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import CLASS_NAMES, COCO_CATEGORY_IDS, load_json, resolve_data_root


def count_images(path: Path) -> int:
    return len(list(path.glob("*.jpg")))


def check_split(data_root: Path, split: str) -> None:
    coco = load_json(data_root / split / f"{split}.json")
    images = coco["images"]
    annotations = coco.get("annotations", [])
    image_names = {image["file_name"] for image in images}
    image_ids = {image["id"] for image in images}

    rgb_names = {path.name for path in (data_root / split / "rgb").glob("*.jpg")}
    tir_names = {path.name for path in (data_root / split / "tir").glob("*.jpg")}
    missing_rgb = sorted(image_names - rgb_names)
    missing_tir = sorted(image_names - tir_names)
    extra_rgb = sorted(rgb_names - image_names)
    extra_tir = sorted(tir_names - image_names)

    categories = Counter()
    invalid = 0
    for ann in annotations:
        categories[ann["category_id"]] += 1
        x, y, w, h = ann["bbox"]
        if ann["image_id"] not in image_ids or w <= 0 or h <= 0:
            invalid += 1

    print(f"[{split}] images={len(images)} annotations={len(annotations)}")
    print(f"[{split}] rgb_files={len(rgb_names)} tir_files={len(tir_names)}")
    for cat_id, name in zip(COCO_CATEGORY_IDS, CLASS_NAMES):
        print(f"[{split}] {cat_id}:{name}={categories[cat_id]}")
    if invalid:
        print(f"[{split}] invalid_annotations={invalid}")
    if missing_rgb or missing_tir or extra_rgb or extra_tir:
        print(f"[{split}] missing_rgb={len(missing_rgb)} missing_tir={len(missing_tir)}")
        print(f"[{split}] extra_rgb={len(extra_rgb)} extra_tir={len(extra_tir)}")
        raise SystemExit(f"{split} split failed image pairing checks.")


def check_test(data_root: Path) -> None:
    rgb_names = {path.name for path in (data_root / "test" / "rgb").glob("*.jpg")}
    tir_names = {path.name for path in (data_root / "test" / "tir").glob("*.jpg")}
    print(f"[test] rgb_files={len(rgb_names)} tir_files={len(tir_names)}")
    if rgb_names != tir_names:
        print(f"[test] only_rgb={len(rgb_names - tir_names)} only_tir={len(tir_names - rgb_names)}")
        raise SystemExit("test split failed RGB/TIR pairing checks.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GAIIC RGB-T COCO dataset integrity.")
    parser.add_argument("--data-root", default=None)
    return parser.parse_args()


def main() -> None:
    data_root = resolve_data_root(parse_args().data_root)
    check_split(data_root, "train")
    check_split(data_root, "val")
    check_test(data_root)
    print("Dataset checks passed.")


if __name__ == "__main__":
    main()
