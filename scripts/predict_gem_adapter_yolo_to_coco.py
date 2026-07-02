from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import YOLO_TO_COCO_CATEGORY, load_json, resolve_data_root, save_json
from datasets.rgbt_dual_dataset import image_to_tensor, imread_unicode, letterbox
from models.gem_adapter_yolo import GEMAdapterYOLO


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
    parser = argparse.ArgumentParser(description="Run GEM Adapter + pretrained YOLO prediction to COCO JSON.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--yolo-weights", default=None, help="Override the YOLO weights path saved in the checkpoint.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--multi-label", action="store_true")
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def load_model(
    checkpoint: str | Path,
    device: torch.device,
    yolo_weights: str | None = None,
) -> tuple[GEMAdapterYOLO, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    model_meta = ckpt.get("model_meta", {})
    saved_order = model_meta.get("baseline_tensor_order")
    if saved_order and saved_order != "R,G,TIR":
        raise ValueError(
            f"Checkpoint baseline_tensor_order={saved_order!r} is incompatible with the current adapter."
        )
    if not saved_order:
        print(
            "Warning: checkpoint has no baseline_tensor_order metadata. "
            "It may be a legacy GEM adapter checkpoint; retrain after the channel-order fix for reliable results."
        )
    weights = yolo_weights or model_meta.get("yolo_weights") or saved_args.get(
        "yolo_weights", "work_dirs/yolo11s_rgbt_aug_domain-2/weights/best.pt"
    )
    model = GEMAdapterYOLO(
        weights,
        adapter_hidden=int(model_meta.get("adapter_hidden", saved_args.get("adapter_hidden", 24))),
        residual_scale=float(model_meta.get("residual_scale", saved_args.get("residual_scale", 0.10))),
        freeze_yolo=False,
    ).to(device)
    model.adapter.load_state_dict(ckpt["adapter"])
    if "yolo" in ckpt:
        model.yolo.load_state_dict(ckpt["yolo"])
    model.eval()
    return model, saved_args


def main() -> None:
    args = parse_args()
    if args.device.lower() == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError as exc:
        raise SystemExit("Ultralytics non_max_suppression is required.") from exc

    model, saved_args = load_model(args.checkpoint, device, args.yolo_weights)
    imgsz = args.imgsz or int(saved_args.get("imgsz", 640))
    data_root = resolve_data_root(args.data_root or saved_args.get("data_root"))
    lookup = image_lookup(data_root, args.split)
    image_paths = sorted((data_root / args.split / "rgb").glob("*.jpg"))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    output = []
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        rgb_tensors = []
        tir_tensors = []
        metas = []
        for rgb_path in batch_paths:
            name = rgb_path.name
            tir_path = data_root / args.split / "tir" / name
            rgb = imread_unicode(rgb_path, cv2.IMREAD_COLOR)
            tir = imread_unicode(tir_path, cv2.IMREAD_GRAYSCALE)
            if rgb is None:
                raise FileNotFoundError(rgb_path)
            if tir is None:
                raise FileNotFoundError(tir_path)
            if rgb.shape[:2] != tir.shape[:2]:
                tir = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            rgb_lb, scale, (pad_x, pad_y) = letterbox(rgb, imgsz)
            tir_lb, _, _ = letterbox(tir, imgsz)
            rgb_tensors.append(image_to_tensor(rgb_lb, channels=3))
            tir_tensors.append(image_to_tensor(tir_lb, channels=1))
            metas.append((name, scale, pad_x, pad_y))

        rgb_batch = torch.stack(rgb_tensors).to(device)
        tir_batch = torch.stack(tir_tensors).to(device)
        with torch.no_grad():
            prediction = model(rgb_batch, tir_batch)[0]
            detections = non_max_suppression(
                prediction,
                conf_thres=args.conf,
                iou_thres=args.iou,
                agnostic=args.agnostic_nms,
                multi_label=args.multi_label,
                max_det=args.max_det,
                nc=5,
                max_time_img=2.0,
            )

        for det, (name, scale, pad_x, pad_y) in zip(detections, metas):
            image = lookup[name]
            width = float(image["width"])
            height = float(image["height"])
            if det is None or det.numel() == 0:
                continue
            det = det.detach().cpu()
            for row in det.tolist():
                x1, y1, x2, y2, score, cls_id = row[:6]
                x1 = (x1 - pad_x) / scale
                y1 = (y1 - pad_y) / scale
                x2 = (x2 - pad_x) / scale
                y2 = (y2 - pad_y) / scale
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
        done = start + len(batch_paths)
        if done % 100 == 0 or done == len(image_paths):
            print(f"Predicted {done}/{len(image_paths)} images")

    save_json(output, args.out)
    print(f"Wrote {len(output)} detections: {args.out}")


if __name__ == "__main__":
    main()
