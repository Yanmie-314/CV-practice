from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2

from common import CLASS_NAMES, COCO_TO_YOLO_CATEGORY, load_json, resolve_data_root


COLORS = [
    (45, 220, 90),
    (40, 150, 255),
    (255, 170, 40),
    (220, 80, 220),
    (80, 220, 220),
]


def draw_box(image, ann: dict) -> None:
    class_idx = COCO_TO_YOLO_CATEGORY[ann["category_id"]]
    color = COLORS[class_idx % len(COLORS)]
    x, y, w, h = ann["bbox"]
    p1 = int(round(x)), int(round(y))
    p2 = int(round(x + w)), int(round(y + h))
    cv2.rectangle(image, p1, p2, color, 2)
    label = CLASS_NAMES[class_idx]
    cv2.putText(
        image,
        label,
        (p1[0], max(12, p1[1] - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize COCO annotations on RGB/TIR images.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--modality", choices=["rgb", "tir"], default="rgb")
    parser.add_argument("--out-dir", default="outputs/vis")
    parser.add_argument("--num", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    coco = load_json(data_root / args.split / f"{args.split}.json")
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann["category_id"] in COCO_TO_YOLO_CATEGORY:
            anns_by_image[ann["image_id"]].append(ann)

    out_dir = Path(args.out_dir) / args.split / args.modality
    out_dir.mkdir(parents=True, exist_ok=True)
    for image in coco["images"][: args.num]:
        src = data_root / args.split / args.modality / image["file_name"]
        img = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(src)
        for ann in anns_by_image.get(image["id"], []):
            draw_box(img, ann)
        dst = out_dir / image["file_name"]
        cv2.imwrite(str(dst), img)
        print(dst)


if __name__ == "__main__":
    main()
