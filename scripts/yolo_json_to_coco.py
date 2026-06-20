from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import YOLO_TO_COCO_CATEGORY, load_json, resolve_data_root, save_json


def load_image_lookup(data_root: Path, split: str) -> dict[str, dict]:
    if split in {"train", "val"}:
        coco = load_json(data_root / split / f"{split}.json")
        return {Path(image["file_name"]).name: image for image in coco["images"]}

    image_dir = data_root / "test" / "rgb"
    return {
        path.name: {
            "id": idx,
            "file_name": path.name,
            "width": 640,
            "height": 512,
        }
        for idx, path in enumerate(sorted(image_dir.glob("*.jpg")), start=1)
    }


def convert_prediction(pred: dict, image_lookup: dict[str, dict]) -> list[dict]:
    image_name = Path(pred["image_id"]).name
    image = image_lookup.get(image_name)
    if image is None:
        raise KeyError(f"Prediction image not found in split metadata: {image_name}")

    width = float(image["width"])
    height = float(image["height"])
    results = []
    for box in pred.get("bboxes", []):
        x1, y1, x2, y2 = [float(v) for v in box["box"]]
        x1 = min(width, max(0.0, x1))
        y1 = min(height, max(0.0, y1))
        x2 = min(width, max(0.0, x2))
        y2 = min(height, max(0.0, y2))
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        yolo_class = int(box["class"])
        results.append(
            {
                "image_id": int(image["id"]),
                "category_id": YOLO_TO_COCO_CATEGORY[yolo_class],
                "bbox": [
                    round(x1, 3),
                    round(y1, 3),
                    round(w, 3),
                    round(h, 3),
                ],
                "score": round(float(box["confidence"]), 6),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Ultralytics JSON predictions to COCO detection results."
    )
    parser.add_argument("--pred-json", required=True, help="Ultralytics predictions.json path.")
    parser.add_argument("--out", required=True, help="Output COCO detection JSON path.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    image_lookup = load_image_lookup(data_root, args.split)
    predictions = json.loads(Path(args.pred_json).read_text(encoding="utf-8"))

    coco_results = []
    for pred in predictions:
        coco_results.extend(convert_prediction(pred, image_lookup))

    save_json(coco_results, args.out)
    print(f"Wrote {len(coco_results)} detections: {args.out}")


if __name__ == "__main__":
    main()
