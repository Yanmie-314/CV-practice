from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from torchvision.ops import nms as torchvision_nms
except Exception:
    torchvision_nms = None

from common import YOLO_TO_COCO_CATEGORY, load_json, resolve_data_root, save_json
from datasets.rgbt_dual_dataset import image_to_tensor, imread_unicode, letterbox
from models.gem_yolo import build_gem_yolo_from_dict


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def image_lookup(data_root: Path, split: str) -> dict[str, dict]:
    if split in {"train", "val"}:
        coco = load_json(data_root / split / f"{split}.json")
        return {Path(image["file_name"]).name: image for image in coco["images"]}
    image_dir = data_root / "test" / "rgb"
    return {
        path.name: {"id": idx, "file_name": path.name, "width": 640, "height": 512}
        for idx, path in enumerate(sorted(image_dir.glob("*.jpg")), start=1)
    }


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(dim=1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=1)


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_b = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter).clamp(min=1e-9)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    if torchvision_nms is not None:
        return torchvision_nms(boxes, scores, iou_thr)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thr]
    if not keep:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    return torch.stack(keep)


def postprocess(
    pred: torch.Tensor,
    conf_thr: float,
    iou_thr: float,
    max_det: int,
    multi_label: bool,
    pre_nms_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred.transpose(0, 1)
    anchor_boxes = xywh_to_xyxy(pred[:, :4])
    scores_all = pred[:, 4:]
    if multi_label:
        anchor_idx, classes = (scores_all >= conf_thr).nonzero(as_tuple=True)
        boxes = anchor_boxes[anchor_idx]
        scores = scores_all[anchor_idx, classes]
    else:
        scores, classes = scores_all.max(dim=1)
        mask = scores >= conf_thr
        boxes = anchor_boxes[mask]
        scores = scores[mask]
        classes = classes[mask]
    if pre_nms_topk > 0 and scores.numel() > pre_nms_topk:
        keep = scores.argsort(descending=True)[:pre_nms_topk]
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
    keep_all = []
    for cls in classes.unique():
        cls_mask = classes == cls
        cls_indices = cls_mask.nonzero(as_tuple=False).flatten()
        cls_keep = nms(boxes[cls_mask], scores[cls_mask], iou_thr)
        keep_all.append(cls_indices[cls_keep])
    if keep_all:
        keep = torch.cat(keep_all)
        keep = keep[scores[keep].argsort(descending=True)[:max_det]]
        return boxes[keep], scores[keep], classes[keep]
    return boxes[:0], scores[:0], classes[:0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEM-YOLO dual-stream prediction and export COCO results.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Defaults to the config stored in checkpoint.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test image limit.")
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="Export every class above conf for each anchor instead of only the top class.",
    )
    parser.add_argument(
        "--pre-nms-topk",
        type=int,
        default=30000,
        help="Keep only the top K candidates per image before NMS. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.lower() == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = load_yaml(args.config) if args.config else ckpt.get("config", {})
    data_root = resolve_data_root(args.data_root or cfg.get("data_root"))
    lookup = image_lookup(data_root, args.split)

    model = build_gem_yolo_from_dict(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    names = sorted((data_root / args.split / "rgb").glob("*.jpg"))
    if args.max_images is not None:
        names = names[: args.max_images]
    output = []
    for start in range(0, len(names), args.batch_size):
        batch_paths = names[start : start + args.batch_size]
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
            rgb_lb, scale, (pad_x, pad_y) = letterbox(rgb, args.imgsz)
            tir_lb, _, _ = letterbox(tir, args.imgsz)
            rgb_tensors.append(image_to_tensor(rgb_lb, channels=3))
            tir_tensors.append(image_to_tensor(tir_lb, channels=1))
            metas.append((name, scale, pad_x, pad_y))
        rgb_batch = torch.stack(rgb_tensors).to(device)
        tir_batch = torch.stack(tir_tensors).to(device)
        with torch.no_grad():
            preds = model(rgb_batch, tir_batch)[0].detach().cpu()
        for pred, (name, scale, pad_x, pad_y) in zip(preds, metas):
            image = lookup[name]
            width = float(image["width"])
            height = float(image["height"])
            boxes, scores, classes = postprocess(
                pred,
                args.conf,
                args.iou,
                args.max_det,
                args.multi_label,
                args.pre_nms_topk,
            )
            for box, score, cls_id in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
                x1, y1, x2, y2 = [float(v) for v in box]
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
        if (start + len(batch_paths)) % 100 == 0 or start + len(batch_paths) == len(names):
            print(f"Predicted {start + len(batch_paths)}/{len(names)} images")

    save_json(output, args.out)
    print(f"Wrote {len(output)} detections: {args.out}")


if __name__ == "__main__":
    main()
