from __future__ import annotations

import argparse
from pathlib import Path

from common import COCO_CATEGORY_IDS, load_json, save_json


def format_detection(
    det: dict, bbox_decimals: int, score_decimals: int | None
) -> dict:
    score = float(det["score"])
    if score_decimals is not None:
        score = round(score, score_decimals)
    return {
        "image_id": int(det["image_id"]),
        "category_id": int(det["category_id"]),
        "bbox": [round(float(v), bbox_decimals) for v in det["bbox"]],
        "score": score,
    }


def format_submission(
    results: list, bbox_decimals: int = 1, score_decimals: int | None = None
) -> list[dict]:
    return [format_detection(det, bbox_decimals, score_decimals) for det in results]


def validate_submission(results: list) -> list[str]:
    errors = []
    if not isinstance(results, list):
        errors.append("Submission must be a JSON list.")
        return errors

    required_keys = {"image_id", "category_id", "bbox", "score"}
    for idx, det in enumerate(results):
        missing = required_keys - set(det.keys())
        if missing:
            errors.append(f"Detection {idx} missing keys: {missing}")
            continue
        extra = set(det.keys()) - required_keys
        if extra:
            errors.append(f"Detection {idx} has extra keys: {extra}")
        if not isinstance(det["image_id"], int):
            errors.append(f"Detection {idx} has non-int image_id: {det['image_id']}")
        if det["category_id"] not in COCO_CATEGORY_IDS:
            errors.append(
                f"Detection {idx} has invalid category_id: {det['category_id']}"
            )
        bbox = det["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            errors.append(f"Detection {idx} has invalid bbox: {bbox}")
        score = det["score"]
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            errors.append(f"Detection {idx} has invalid score: {score}")
        if idx >= 1000 and errors:
            # Avoid huge error logs; report first 1000 problematic rows.
            break
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and finalize a COCO detection submission JSON."
    )
    parser.add_argument("--input", required=True, help="Fused COCO detection JSON.")
    parser.add_argument(
        "--out",
        default="submissions/final_submission.json",
        help="Final submission output path.",
    )
    parser.add_argument(
        "--bbox-decimals",
        type=int,
        default=1,
        help="Number of decimal places kept in bbox values.",
    )
    parser.add_argument(
        "--score-decimals",
        type=int,
        default=None,
        help="Optional number of decimal places kept in score values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = format_submission(load_json(args.input), args.bbox_decimals, args.score_decimals)
    errors = validate_submission(results)
    if errors:
        print(f"Found {len(errors)} validation errors (showing first 10):")
        for err in errors[:10]:
            print(f"  - {err}")
        raise SystemExit("Submission validation failed.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out_path)
    print(f"Validated {len(results)} detections.")
    print(f"Final submission saved to: {out_path}")


if __name__ == "__main__":
    main()
