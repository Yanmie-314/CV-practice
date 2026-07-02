from __future__ import annotations

import argparse
import contextlib
import csv
import io
from pathlib import Path

from common import CLASS_NAMES, COCO_CATEGORY_IDS, resolve_data_root


def evaluate_file(coco_gt, pred: Path, classwise: bool = False) -> dict[str, float | int | str]:
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(str(pred))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = evaluator.stats
    row: dict[str, float | int | str] = {
        "path": str(pred),
        "detections": len(coco_dt.dataset.get("annotations", [])),
        "ap": float(stats[0]),
        "ap50": float(stats[1]),
        "ap75": float(stats[2]),
        "aps": float(stats[3]),
        "apm": float(stats[4]),
        "apl": float(stats[5]),
        "ar100": float(stats[8]),
    }
    if classwise:
        for cat_id, name in zip(COCO_CATEGORY_IDS, CLASS_NAMES):
            with contextlib.redirect_stdout(io.StringIO()):
                cls_eval = COCOeval(coco_gt, coco_dt, "bbox")
                cls_eval.params.catIds = [cat_id]
                cls_eval.evaluate()
                cls_eval.accumulate()
                cls_eval.summarize()
            row[f"ap_{name}"] = float(cls_eval.stats[0])
    return row


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        if matches:
            paths.extend(matches)
            continue
        path = Path(pattern)
        if path.exists():
            paths.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate COCO detection result files.")
    parser.add_argument("inputs", nargs="+", help="JSON files or glob patterns.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--classwise", action="store_true")
    parser.add_argument("--out", default=None, help="Optional CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise SystemExit("Install pycocotools first: pip install pycocotools") from exc

    data_root = resolve_data_root(args.data_root)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(data_root / "val" / "val.json"))

    rows = [
        evaluate_file(coco_gt, path, classwise=args.classwise)
        for path in expand_inputs(args.inputs)
        if "smoke" not in path.name
    ]
    rows.sort(key=lambda row: float(row["ap"]), reverse=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
