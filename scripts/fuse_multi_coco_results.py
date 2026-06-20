from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common import COCO_CATEGORY_IDS, load_json, save_json


IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 512.0


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(IMAGE_WIDTH, x1))
    y1 = max(0.0, min(IMAGE_HEIGHT, y1))
    x2 = max(0.0, min(IMAGE_WIDTH, x2))
    y2 = max(0.0, min(IMAGE_HEIGHT, y2))
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


def normalize_weights(weights: list[float]) -> list[float]:
    if not weights:
        return []
    max_weight = max(weights)
    if max_weight <= 0:
        raise ValueError("At least one fusion weight must be positive.")
    return [weight / max_weight for weight in weights]


def group_detections(
    result_sets: list[list[dict]],
    weights: list[float],
    score_thr: float,
) -> dict[tuple[int, int], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for model_idx, detections in enumerate(result_sets):
        weight = weights[model_idx]
        for det in detections:
            category_id = int(det["category_id"])
            if category_id not in COCO_CATEGORY_IDS:
                continue
            score = float(det["score"])
            if score < score_thr:
                continue
            grouped[(int(det["image_id"]), category_id)].append(
                {
                    "image_id": int(det["image_id"]),
                    "category_id": category_id,
                    "bbox": [float(v) for v in det["bbox"]],
                    "score": score,
                    "weighted_score": score * weight,
                    "model_idx": model_idx,
                    "model_weight": weight,
                }
            )
    return grouped


def has_primary_vote(cluster: list[dict], primary_count: int) -> bool:
    return primary_count <= 0 or any(
        int(det["model_idx"]) < primary_count for det in cluster
    )


def nms_fuse(detections: list[dict], iou_thr: float, primary_count: int = 0) -> list[dict]:
    remaining = sorted(detections, key=lambda det: det["weighted_score"], reverse=True)
    kept: list[dict] = []
    while remaining:
        seed = remaining.pop(0)
        seed_box = xywh_to_xyxy(seed["bbox"])
        rest = []
        for det in remaining:
            if iou(seed_box, xywh_to_xyxy(det["bbox"])) < iou_thr:
                rest.append(det)
        remaining = rest
        if not has_primary_vote([seed], primary_count):
            continue
        kept.append(
            {
                "image_id": seed["image_id"],
                "category_id": seed["category_id"],
                "bbox": [round(v, 3) for v in seed["bbox"]],
                "score": round(min(1.0, seed["weighted_score"]), 6),
            }
        )
    return kept


def weighted_box(cluster: list[dict], num_models: int) -> dict:
    total = sum(max(1e-6, det["weighted_score"]) for det in cluster)
    fused_xyxy = [0.0, 0.0, 0.0, 0.0]
    for det in cluster:
        box = xywh_to_xyxy(det["bbox"])
        weight = max(1e-6, det["weighted_score"])
        for idx in range(4):
            fused_xyxy[idx] += box[idx] * weight / total

    model_votes = {det["model_idx"] for det in cluster}
    best_score = max(det["weighted_score"] for det in cluster)
    avg_score = sum(det["weighted_score"] for det in cluster) / len(cluster)
    vote_boost = 1.0 + 0.08 * (len(model_votes) - 1)
    cluster_boost = min(1.10, 1.0 + 0.02 * (len(cluster) - 1))
    coverage_penalty = 0.85 + 0.15 * (len(model_votes) / max(1, num_models))
    score = min(1.0, max(best_score, avg_score) * vote_boost * cluster_boost * coverage_penalty)

    first = cluster[0]
    return {
        "image_id": first["image_id"],
        "category_id": first["category_id"],
        "bbox": [round(v, 3) for v in xyxy_to_xywh(fused_xyxy)],
        "score": round(score, 6),
    }


def wbf_fuse(
    detections: list[dict],
    iou_thr: float,
    num_models: int,
    primary_count: int = 0,
) -> list[dict]:
    remaining = sorted(detections, key=lambda det: det["weighted_score"], reverse=True)
    fused: list[dict] = []
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
        if not has_primary_vote(cluster, primary_count):
            continue
        fused.append(weighted_box(cluster, num_models))
    return fused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse any number of COCO detection result JSON files with NMS or WBF."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["wbf", "nms"], default="wbf")
    parser.add_argument("--iou-thr", type=float, default=0.65)
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--max-per-image", type=int, default=300)
    parser.add_argument(
        "--primary-count",
        type=int,
        default=0,
        help=(
            "Require each fused cluster to include at least one detection from the "
            "first N inputs. Use this to make weak auxiliary models unable to add "
            "standalone boxes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [Path(path) for path in args.inputs]
    result_sets = [load_json(path) for path in inputs]
    weights = args.weights or [1.0] * len(inputs)
    if len(weights) != len(inputs):
        raise SystemExit("--weights must have the same length as --inputs.")
    weights = normalize_weights(weights)

    grouped = group_detections(result_sets, weights, args.score_thr)
    by_image: dict[int, list[dict]] = defaultdict(list)
    for detections in grouped.values():
        if args.method == "wbf":
            fused = wbf_fuse(detections, args.iou_thr, len(inputs), args.primary_count)
        else:
            fused = nms_fuse(detections, args.iou_thr, args.primary_count)
        for det in fused:
            if det["score"] >= args.score_thr and det["bbox"][2] > 0 and det["bbox"][3] > 0:
                by_image[int(det["image_id"])].append(det)

    output = []
    for image_id in sorted(by_image):
        detections = sorted(by_image[image_id], key=lambda det: det["score"], reverse=True)
        output.extend(detections[: args.max_per_image])
    output.sort(key=lambda det: (det["image_id"], det["category_id"], -det["score"]))

    save_json(output, args.out)
    print(f"Method: {args.method}")
    print(f"Primary count: {args.primary_count}")
    print(f"Inputs: {len(inputs)}")
    for path, weight in zip(inputs, weights):
        print(f"  {path} weight={weight:.3f}")
    print(f"Wrote {len(output)} fused detections: {args.out}")


if __name__ == "__main__":
    main()
