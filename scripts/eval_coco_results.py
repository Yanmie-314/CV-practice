from __future__ import annotations

import argparse
from pathlib import Path

from common import resolve_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate COCO detection results on val.")
    parser.add_argument("--pred", required=True, help="COCO detection results JSON.")
    parser.add_argument("--data-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise SystemExit("Install pycocotools first: pip install pycocotools") from exc

    data_root = resolve_data_root(args.data_root)
    ann_file = data_root / "val" / "val.json"
    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(str(Path(args.pred)))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()


if __name__ == "__main__":
    main()
