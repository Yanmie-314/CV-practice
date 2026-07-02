from __future__ import annotations

import argparse
from pathlib import Path

from common import load_json, save_json


def parse_category_source(items: list[str]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Expected CATEGORY:INPUT_INDEX, got {item!r}")
        category, input_idx = item.split(":", 1)
        mapping[int(category)] = int(input_idx)
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine COCO detection files by selecting a source file per category."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--category-source",
        nargs="+",
        required=True,
        help="Category-to-input mapping using 0-based input indexes, e.g. 1:0 2:1.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-per-image", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_sets = [load_json(path) for path in args.inputs]
    category_source = parse_category_source(args.category_source)

    by_image: dict[int, list[dict]] = {}
    kept = 0
    for category_id, input_idx in category_source.items():
        if input_idx < 0 or input_idx >= len(result_sets):
            raise IndexError(f"Invalid input index {input_idx} for category {category_id}")
        for det in result_sets[input_idx]:
            if int(det["category_id"]) != category_id:
                continue
            by_image.setdefault(int(det["image_id"]), []).append(det)
            kept += 1

    output: list[dict] = []
    for image_id in sorted(by_image):
        detections = sorted(
            by_image[image_id],
            key=lambda det: float(det["score"]),
            reverse=True,
        )
        output.extend(detections[: args.max_per_image])
    output.sort(key=lambda det: (int(det["image_id"]), int(det["category_id"]), -float(det["score"])))
    save_json(output, Path(args.out))
    print(f"Wrote {len(output)} detections from {kept} selected detections: {args.out}")


if __name__ == "__main__":
    main()
