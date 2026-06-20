from __future__ import annotations

import argparse
from pathlib import Path

from common import YOLO_TO_COCO_CATEGORY, load_json, resolve_data_root, save_json


def image_lookup(data_root: Path, split: str) -> dict[str, dict]:
    if split in {"train", "val"}:
        coco = load_json(data_root / split / f"{split}.json")
        return {Path(image["file_name"]).name: image for image in coco["images"]}
    image_dir = data_root / "test" / "rgb"
    return {
        path.name: {"id": idx, "file_name": path.name, "width": 640, "height": 512}
        for idx, path in enumerate(sorted(image_dir.glob("*.jpg")), start=1)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO prediction on RGB-T fused images and export COCO results."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, help="Fused RGB-T image directory.")
    parser.add_argument("--out", required=True, help="Output COCO detection JSON.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from exc

    data_root = resolve_data_root(args.data_root)
    lookup = image_lookup(data_root, args.split)
    source = Path(args.source)
    images = sorted(source.glob("*.jpg"))
    model = YOLO(args.model)

    output = []
    for start in range(0, len(images), args.batch_size):
        batch = images[start : start + args.batch_size]
        results = model.predict(
            source=[str(path) for path in batch],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            stream=False,
            verbose=False,
        )
        for result_idx, result in enumerate(results):
            image_name = Path(result.path).name
            if image_name not in lookup:
                image_name = batch[result_idx].name
            image = lookup[image_name]
            width = float(image["width"])
            height = float(image["height"])
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            for box, score, cls_id in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = [float(v) for v in box]
                x1 = min(width, max(0.0, x1))
                y1 = min(height, max(0.0, y1))
                x2 = min(width, max(0.0, x2))
                y2 = min(height, max(0.0, y2))
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                output.append(
                    {
                        "image_id": int(image["id"]),
                        "category_id": YOLO_TO_COCO_CATEGORY[int(cls_id)],
                        "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)],
                        "score": round(float(score), 6),
                    }
                )
        if (start + len(batch)) % 100 == 0 or start + len(batch) == len(images):
            print(f"Predicted {start + len(batch)}/{len(images)} images")

    save_json(output, args.out)
    print(f"Wrote {len(output)} detections: {args.out}")


if __name__ == "__main__":
    main()
