from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one YOLO model on fused RGB-T images.")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--data", default="configs/gaiic_yolo_rgbt.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="D:/cv/work_dirs")
    parser.add_argument("--name", default="yolo11s_rgbt")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--close-mosaic", type=int, default=None)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(Path(args.data)),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "amp": True,
        "verbose": True,
        "plots": True,
        "patience": args.patience,
        "cos_lr": args.cos_lr,
        "seed": args.seed,
    }
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.close_mosaic is not None:
        train_kwargs["close_mosaic"] = args.close_mosaic
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
