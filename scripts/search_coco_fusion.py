from __future__ import annotations

import argparse
import contextlib
import csv
import itertools
import io
from pathlib import Path

from common import load_json, resolve_data_root, save_json
from fuse_multi_coco_results import (
    group_detections,
    nms_fuse,
    normalize_weights,
    wbf_fuse,
)


def parse_float_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_grid(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_weight_tuples(value: str) -> list[tuple[float, ...]]:
    tuples: list[tuple[float, ...]] = []
    for group in value.split(";"):
        items = [float(item) for item in group.split(",") if item.strip()]
        if items:
            tuples.append(tuple(items))
    return tuples


def fuse_results(
    result_sets: list[list[dict]],
    weights: list[float],
    method: str,
    iou_thr: float,
    score_thr: float,
    max_per_image: int,
    primary_count: int,
) -> list[dict]:
    grouped = group_detections(result_sets, normalize_weights(weights), score_thr)
    by_image: dict[int, list[dict]] = {}
    for detections in grouped.values():
        if method == "wbf":
            fused = wbf_fuse(detections, iou_thr, len(result_sets), primary_count)
        else:
            fused = nms_fuse(detections, iou_thr, primary_count)
        for det in fused:
            if det["score"] < score_thr or det["bbox"][2] <= 0 or det["bbox"][3] <= 0:
                continue
            by_image.setdefault(int(det["image_id"]), []).append(det)

    output: list[dict] = []
    for image_id in sorted(by_image):
        detections = sorted(by_image[image_id], key=lambda det: det["score"], reverse=True)
        output.extend(detections[:max_per_image])
    output.sort(key=lambda det: (det["image_id"], det["category_id"], -det["score"]))
    return output


def evaluate(
    coco_gt,
    detections: list[dict],
    category_id: int | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        if category_id is not None:
            evaluator.params.catIds = [category_id]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return (
        float(stats[0]),
        float(stats[1]),
        float(stats[2]),
        float(stats[3]),
        float(stats[4]),
        float(stats[5]),
        float(stats[8]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search COCO result fusion on validation data.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--aux-weight-grid",
        default="0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated weights for each non-primary input. First input weight is fixed to 1.0.",
    )
    parser.add_argument(
        "--weight-tuples",
        default=None,
        help=(
            "Optional explicit semicolon-separated weight tuples, e.g. "
            "'1,1.1,0.2;1,1.1,0.4'. Overrides --aux-weight-grid."
        ),
    )
    parser.add_argument("--iou-grid", default="0.65,0.70,0.75,0.80,0.85")
    parser.add_argument("--score-grid", default="0.001")
    parser.add_argument("--method-grid", default="wbf")
    parser.add_argument("--primary-count-grid", default="1")
    parser.add_argument("--max-per-image", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--category-id",
        type=int,
        default=None,
        help="If set, search and evaluate only this COCO category id.",
    )
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--save-best", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise SystemExit("Install pycocotools first: pip install pycocotools") from exc

    input_paths = [Path(path) for path in args.inputs]
    result_sets = [load_json(path) for path in input_paths]
    if args.category_id is not None:
        result_sets = [
            [det for det in result_set if int(det["category_id"]) == args.category_id]
            for result_set in result_sets
        ]
    data_root = resolve_data_root(args.data_root)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(data_root / "val" / "val.json"))

    aux_weight_grid = parse_float_grid(args.aux_weight_grid)
    iou_grid = parse_float_grid(args.iou_grid)
    score_grid = parse_float_grid(args.score_grid)
    method_grid = [item.strip() for item in args.method_grid.split(",") if item.strip()]
    primary_count_grid = parse_int_grid(args.primary_count_grid)

    if args.weight_tuples:
        weight_grid = parse_weight_tuples(args.weight_tuples)
        bad = [weights for weights in weight_grid if len(weights) != len(result_sets)]
        if bad:
            raise SystemExit("--weight-tuples entries must match the number of inputs.")
    elif len(result_sets) == 1:
        weight_grid = [(1.0,)]
    else:
        weight_grid = [
            (1.0, *weights)
            for weights in itertools.product(aux_weight_grid, repeat=len(result_sets) - 1)
        ]

    rows = []
    best_output: list[dict] | None = None
    best_row: dict[str, float | int | str] | None = None
    total = len(weight_grid) * len(iou_grid) * len(score_grid) * len(method_grid) * len(primary_count_grid)
    done = 0
    for method, iou_thr, score_thr, primary_count, weights in itertools.product(
        method_grid, iou_grid, score_grid, primary_count_grid, weight_grid
    ):
        done += 1
        fused = fuse_results(
            result_sets,
            list(weights),
            method,
            iou_thr,
            score_thr,
            args.max_per_image,
            primary_count,
        )
        ap, ap50, ap75, aps, apm, apl, ar100 = evaluate(coco_gt, fused, args.category_id)
        row: dict[str, float | int | str] = {
            "rank": 0,
            "ap": ap,
            "ap50": ap50,
            "ap75": ap75,
            "aps": aps,
            "apm": apm,
            "apl": apl,
            "ar100": ar100,
            "detections": len(fused),
            "method": method,
            "iou_thr": iou_thr,
            "score_thr": score_thr,
            "primary_count": primary_count,
            "weights": " ".join(f"{weight:.3f}" for weight in weights),
            "inputs": " | ".join(str(path) for path in input_paths),
        }
        if args.category_id is not None:
            row["category_id"] = args.category_id
        rows.append(row)
        if best_row is None or ap > float(best_row["ap"]):
            best_row = row
            best_output = fused
        if done % 10 == 0 or done == total:
            print(f"searched {done}/{total}; best AP={float(best_row['ap']):.6f}")

    rows.sort(key=lambda row: float(row["ap"]), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.save_best and best_output is not None:
        save_json(best_output, args.save_best)

    print("Top results:")
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows[: args.top_k])


if __name__ == "__main__":
    main()
