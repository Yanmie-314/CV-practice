from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter COCO detection results by score and per-image top-k."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--max-per-image", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detections = load_json(args.input)
    by_image: dict[int, list[dict]] = defaultdict(list)

    for det in detections:
        if float(det["score"]) < args.score_thr:
            continue
        by_image[int(det["image_id"])].append(det)

    output = []
    for image_id in sorted(by_image):
        image_dets = sorted(
            by_image[image_id], key=lambda item: float(item["score"]), reverse=True
        )
        output.extend(image_dets[: args.max_per_image])

    output.sort(key=lambda det: (int(det["image_id"]), -float(det["score"])))
    save_json(output, Path(args.out))
    print(f"Kept {len(output)} / {len(detections)} detections.")
    print(f"score_thr={args.score_thr}, max_per_image={args.max_per_image}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
