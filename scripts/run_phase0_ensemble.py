from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnsembleMember:
    name: str
    model: Path
    source: Path
    weight: float


def run_command(cmd: list[str], step_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"STEP: {step_name}")
    print("Command:", " ".join(cmd))
    print(f"Start: {time.strftime('%H:%M:%S')}")
    print(f"{'=' * 60}")
    sys.stdout.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"{step_name} failed with exit code {proc.returncode}.")


def default_members(split: str) -> list[EnsembleMember]:
    return [
        EnsembleMember(
            name="rgb",
            model=Path("work_dirs/yolo11s_rgb/weights/best.pt"),
            source=Path(f"datasets/gaiic_yolo_rgb/images/{split}"),
            weight=0.30,
        ),
        EnsembleMember(
            name="tir",
            model=Path("work_dirs/yolo11s_tir/weights/best.pt"),
            source=Path(f"datasets/gaiic_yolo_tir/images/{split}"),
            weight=0.30,
        ),
        EnsembleMember(
            name="rgbt_aug_domain",
            model=Path("work_dirs/yolo11s_rgbt_aug_domain/weights/best.pt"),
            source=Path(f"datasets/gaiic_yolo_rgbt_aug_domain/images/{split}"),
            weight=0.40,
        ),
    ]


def parse_member(spec: str, split: str) -> EnsembleMember:
    parts = spec.split("=")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--member must use name=model=source=weight, for example "
            "rgb=work_dirs/yolo11s_rgb/weights/best.pt=datasets/gaiic_yolo_rgb/images/val=0.3"
        )
    name, model, source, weight = parts
    return EnsembleMember(
        name=name,
        model=Path(model),
        source=Path(source.format(split=split)),
        weight=float(weight),
    )


def validate_members(members: list[EnsembleMember]) -> None:
    for member in members:
        if not member.model.exists():
            raise FileNotFoundError(f"Model not found for {member.name}: {member.model}")
        if not member.source.exists():
            raise FileNotFoundError(f"Source directory not found for {member.name}: {member.source}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 0 RGB/TIR/RGBT prediction, WBF fusion, and optional val evaluation."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child scripts.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out-dir", default="submissions/phase0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--augment", action="store_true", help="Enable TTA for every member prediction.")
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--fuse-iou-thr", type=float, default=0.55)
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--max-per-image", type=int, default=300)
    parser.add_argument("--primary-count", type=int, default=0)
    parser.add_argument("--method", choices=["wbf", "nms"], default="wbf")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--skip-predict", action="store_true", help="Reuse existing member JSON files.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip COCO evaluation on val.")
    parser.add_argument(
        "--member",
        action="append",
        default=None,
        help=(
            "Optional member override. Format: name=model=source=weight. "
            "Use {split} in source if needed. Repeat for multiple models."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    members = (
        [parse_member(spec, args.split) for spec in args.member]
        if args.member
        else default_members(args.split)
    )
    validate_members(members)

    pred_paths: list[Path] = []
    for member in members:
        pred_path = out_dir / f"{member.name}_{args.split}.json"
        pred_paths.append(pred_path)
        if args.skip_predict:
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing reused prediction JSON: {pred_path}")
            continue
        cmd = [
            args.python,
            "scripts/predict_rgbt_to_coco.py",
            "--model",
            str(member.model),
            "--source",
            str(member.source),
            "--out",
            str(pred_path),
            "--split",
            args.split,
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--max-det",
            str(args.max_det),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
        ]
        if args.data_root:
            cmd.extend(["--data-root", args.data_root])
        if args.augment:
            cmd.append("--augment")
        if args.agnostic_nms:
            cmd.append("--agnostic-nms")
        run_command(cmd, f"Predict {member.name}")

    fused_path = out_dir / f"phase0_{args.method}_{args.split}.json"
    fuse_cmd = [
        args.python,
        "scripts/fuse_multi_coco_results.py",
        "--inputs",
        *[str(path) for path in pred_paths],
        "--weights",
        *[str(member.weight) for member in members],
        "--out",
        str(fused_path),
        "--method",
        args.method,
        "--iou-thr",
        str(args.fuse_iou_thr),
        "--score-thr",
        str(args.score_thr),
        "--max-per-image",
        str(args.max_per_image),
        "--primary-count",
        str(args.primary_count),
    ]
    run_command(fuse_cmd, "Fuse predictions")

    if args.split == "val" and not args.skip_eval:
        eval_cmd = [
            args.python,
            "scripts/eval_coco_results.py",
            "--pred",
            str(fused_path),
        ]
        if args.data_root:
            eval_cmd.extend(["--data-root", args.data_root])
        run_command(eval_cmd, "Evaluate fused val results")

    print(f"\nPhase 0 output: {fused_path}")


if __name__ == "__main__":
    main()
