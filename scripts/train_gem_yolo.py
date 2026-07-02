from __future__ import annotations

import argparse
import atexit
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from datasets.rgbt_dual_dataset import RGBTDualDataset, collate_rgbt
from models.gem_yolo import build_gem_yolo_from_dict


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight GEM-YOLO RGB-T detector.")
    parser.add_argument("--config", default="configs/gem_yolo_rgbt.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", default=None, help="Optional checkpoint path.")
    parser.add_argument("--max-images", type=int, default=None, help="Smoke-test image limit per split.")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=20,
        help="Print progress every N training steps. Use 1 for very verbose live logs.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log path. Defaults to work_dirs/<name>/train.log.",
    )
    parser.add_argument("--no-log-file", action="store_true", help="Disable file logging.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP. Disabled by default for fusion stability.")
    return parser.parse_args()


def pick(value, default):
    return default if value is None else value


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def setup_logging(run_dir: Path, log_file: str | None, no_log_file: bool) -> Path | None:
    if no_log_file:
        return None
    log_path = Path(log_file) if log_file else run_dir / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    atexit.register(log_handle.close)
    sys.stdout = TeeStream(sys.__stdout__, log_handle)
    sys.stderr = TeeStream(sys.__stderr__, log_handle)
    return log_path


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})

    imgsz = int(pick(args.imgsz, train_cfg.get("imgsz", 640)))
    epochs = int(pick(args.epochs, train_cfg.get("epochs", 150)))
    batch_size = int(pick(args.batch, train_cfg.get("batch", 16)))
    workers = int(pick(args.workers, train_cfg.get("workers", 8)))
    lr0 = float(pick(args.lr0, train_cfg.get("lr0", 0.004)))
    momentum = float(pick(args.momentum, train_cfg.get("momentum", 0.937)))
    weight_decay = float(pick(args.weight_decay, train_cfg.get("weight_decay", 0.0005)))
    project = Path(pick(args.project, train_cfg.get("project", "work_dirs")))
    name = str(pick(args.name, train_cfg.get("name", "gem_yolo_rgbt")))
    data_root = pick(args.data_root, cfg.get("data_root"))
    amp = bool(args.amp or train_cfg.get("amp", False))
    device_arg = pick(args.device, train_cfg.get("device", "0"))
    if str(device_arg).lower() == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_arg}")

    run_dir = project / name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(run_dir, args.log_file, args.no_log_file)

    train_ds = RGBTDualDataset(data_root=data_root, split="train", imgsz=imgsz, max_images=args.max_images)
    val_ds = RGBTDualDataset(data_root=data_root, split="val", imgsz=imgsz, max_images=args.max_images)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rgbt,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rgbt,
        drop_last=False,
    )

    model = build_gem_yolo_from_dict(cfg).to(device)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    start_epoch = 0
    best_loss = math.inf
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_loss = float(ckpt.get("best_loss", best_loss))

    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    print(f"Training GEM-YOLO on {device}")
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)} | Batch: {batch_size} | AMP: {amp}")
    print(f"Run dir: {run_dir}")
    if log_path:
        print(f"Log file: {log_path}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        seen = 0
        start = time.time()
        for batch_i, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                total_loss, loss_items = model.loss(batch)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_n = int(batch["rgb"].shape[0])
            running += float(total_loss.detach()) * batch_n
            seen += batch_n
            if batch_i % args.progress_interval == 0 or batch_i == len(train_loader):
                loss_text = " ".join(f"{v:.4f}" for v in loss_items.detach().cpu().tolist())
                print(
                    f"epoch {epoch + 1}/{epochs} step {batch_i}/{len(train_loader)} "
                    f"loss={running / max(1, seen):.4f} items=[{loss_text}]"
                )

        val_loss = 0.0
        val_seen = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                total_loss, _ = model.loss(batch)
                batch_n = int(batch["rgb"].shape[0])
                val_loss += float(total_loss.detach()) * batch_n
                val_seen += batch_n
        mean_val_loss = val_loss / max(1, val_seen)
        elapsed = time.time() - start
        print(
            f"epoch {epoch + 1}/{epochs} complete "
            f"train_loss={running / max(1, seen):.4f} val_loss={mean_val_loss:.4f} time={elapsed:.1f}s"
        )

        save_checkpoint(weights_dir / "last.pt", model, optimizer, epoch, best_loss, cfg)
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            save_checkpoint(weights_dir / "best.pt", model, optimizer, epoch, best_loss, cfg)
            print(f"saved best checkpoint: {weights_dir / 'best.pt'}")
        if args.save_period > 0 and (epoch + 1) % args.save_period == 0:
            save_checkpoint(weights_dir / f"epoch{epoch + 1}.pt", model, optimizer, epoch, best_loss, cfg)

    print(f"Training complete. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
