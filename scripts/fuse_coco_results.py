from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common import COCO_CATEGORY_IDS, load_json, save_json


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def weighted_fuse_cluster(cluster: list[dict]) -> dict:
    weights = [max(1e-6, det["score"]) for det in cluster]
    total = sum(weights)
    fused_xyxy = [0.0, 0.0, 0.0, 0.0]
    for det, weight in zip(cluster, weights):
        box = xywh_to_xyxy(det["bbox"])
        for idx in range(4):
            fused_xyxy[idx] += box[idx] * weight / total
    best_score = max(det["score"] for det in cluster)
    score = min(1.0, best_score * (1.0 + 0.05 * (len(cluster) - 1)))
    first = cluster[0]
    return {
        "image_id": first["image_id"],
        "category_id": first["category_id"],
        "bbox": [round(v, 3) for v in xyxy_to_xywh(fused_xyxy)],
        "score": round(score, 6),
    }


def fuse_detections(detections: list[dict], iou_thr: float) -> list[dict]:
    remaining = sorted(detections, key=lambda det: det["score"], reverse=True)
    fused = []
    while remaining:
        seed = remaining.pop(0)
        seed_box = xywh_to_xyxy(seed["bbox"])
        cluster = [seed]
        rest = []
        for det in remaining:
            if iou(seed_box, xywh_to_xyxy(det["bbox"])) >= iou_thr:
                cluster.append(det)
            else:
                rest.append(det)
        remaining = rest
        fused.append(weighted_fuse_cluster(cluster))
    return fused


def group_detections(*result_sets: list[dict], weights: list[float]) -> dict[tuple[int, int], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for detections, weight in zip(result_sets, weights):
        for det in detections:
            if det["category_id"] not in COCO_CATEGORY_IDS:
                continue
            scaled = dict(det)
            scaled["score"] = float(scaled["score"]) * weight
            grouped[(int(det["image_id"]), int(det["category_id"]))].append(scaled)
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse RGB and TIR COCO detection results with per-class weighted boxes."
    )
    parser.add_argument("--rgb", required=True, help="RGB COCO detection JSON.")
    parser.add_argument("--tir", required=True, help="TIR COCO detection JSON.")
    parser.add_argument("--out", required=True, help="Output fused JSON.")
    parser.add_argument("--iou-thr", type=float, default=0.55)
    parser.add_argument("--rgb-weight", type=float, default=1.0)
    parser.add_argument("--tir-weight", type=float, default=1.0)
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--max-per-image", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rgb = load_json(args.rgb)
    tir = load_json(args.tir)
    grouped = group_detections(rgb, tir, weights=[args.rgb_weight, args.tir_weight])

    by_image: dict[int, list[dict]] = defaultdict(list)
    for detections in grouped.values():
        for det in fuse_detections(detections, args.iou_thr):
            if det["score"] >= args.score_thr:
                by_image[det["image_id"]].append(det)

    output = []
    for image_id, detections in by_image.items():
        output.extend(
            sorted(detections, key=lambda det: det["score"], reverse=True)[
                : args.max_per_image
            ]
        )
    output.sort(key=lambda det: (det["image_id"], det["category_id"], -det["score"]))
    save_json(output, args.out)
    print(f"Wrote {len(output)} fused detections: {args.out}")


if __name__ == "__main__":
    main()
