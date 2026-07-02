from __future__ import annotations

import argparse
import atexit
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from datasets.rgbt_dual_dataset import RGBTDualDataset, collate_rgbt
from models.gem_adapter_yolo import GEMAdapterYOLO


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except ValueError:
                pass
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GEM input adapter on top of a pretrained YOLO detector.")
    parser.add_argument("--yolo-weights", default="work_dirs/yolo11s_rgbt_aug_domain-2/weights/best.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="work_dirs")
    parser.add_argument("--name", default="gem_adapter_yolo")
    parser.add_argument("--adapter-hidden", type=int, default=24)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--yolo-lr-scale", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze-epoch", type=int, default=-1, help="1-based epoch to unfreeze YOLO. -1 keeps it frozen.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--no-log-file", action="store_true")
    return parser.parse_args()


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


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
    return moved


def scalar_loss(loss: Tensor) -> Tensor:
    return loss.sum() if loss.ndim > 0 else loss


def save_checkpoint(
    path: Path,
    model: GEMAdapterYOLO,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "GEMAdapterYOLO",
            "epoch": epoch,
            "best_loss": best_loss,
            "adapter": model.adapter.state_dict(),
            "yolo": model.yolo.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "model_meta": {
                "yolo_weights": model.yolo_weights,
                "adapter_hidden": model.adapter_hidden,
                "residual_scale": model.residual_scale,
                "baseline_tensor_order": model.adapter.baseline_tensor_order,
            },
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: GEMAdapterYOLO,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.adapter.load_state_dict(ckpt["adapter"])
    if "yolo" in ckpt:
        model.yolo.load_state_dict(ckpt["yolo"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["epoch"]) + 1, float(ckpt.get("best_loss", float("inf")))


def set_yolo_trainable(model: GEMAdapterYOLO, trainable: bool) -> None:
    model.freeze_yolo(not trainable)
    state = "unfrozen" if trainable else "frozen"
    print(f"YOLO detector is {state}.")


def apply_current_learning_rates(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> None:
    if optimizer.param_groups:
        optimizer.param_groups[0]["lr"] = args.lr0
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["lr"] = args.lr0 * args.yolo_lr_scale


def main() -> None:
    args = parse_args()
    if args.device.lower() == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    run_dir = Path(args.project) / args.name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(run_dir, args.log_file, args.no_log_file)

    train_ds = RGBTDualDataset(data_root=args.data_root, split="train", imgsz=args.imgsz, max_images=args.max_images)
    val_ds = RGBTDualDataset(data_root=args.data_root, split="val", imgsz=args.imgsz, max_images=args.max_images)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rgbt,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rgbt,
        drop_last=False,
    )

    model = GEMAdapterYOLO(
        args.yolo_weights,
        adapter_hidden=args.adapter_hidden,
        residual_scale=args.residual_scale,
        freeze_yolo=True,
    ).to(device)
    model.yolo.criterion = None
    optimizer = torch.optim.AdamW(
        [
            {"params": model.adapter.parameters(), "lr": args.lr0},
            {"params": model.yolo.parameters(), "lr": args.lr0 * args.yolo_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        start_epoch, best_loss = load_checkpoint(args.resume, model, optimizer, device)
        apply_current_learning_rates(optimizer, args)

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    print(f"Training GEM Adapter + pretrained YOLO on {device}")
    print(f"YOLO weights: {args.yolo_weights}")
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)} | Batch: {args.batch} | AMP: {args.amp}")
    print(f"Run dir: {run_dir}")
    if log_path:
        print(f"Log file: {log_path}")
    set_yolo_trainable(model, False)

    for epoch in range(start_epoch, args.epochs):
        if args.unfreeze_epoch > 0 and epoch + 1 >= args.unfreeze_epoch:
            set_yolo_trainable(model, True)

        model.train()
        train_loss = 0.0
        seen = 0
        start = time.time()
        for batch_i, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss, loss_items = model.loss(batch)
                total_loss = scalar_loss(loss)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_n = int(batch["rgb"].shape[0])
            train_loss += float(total_loss.detach()) * batch_n
            seen += batch_n
            if batch_i % args.progress_interval == 0 or batch_i == len(train_loader):
                items = " ".join(f"{v:.4f}" for v in loss_items.detach().cpu().tolist())
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {batch_i}/{len(train_loader)} "
                    f"loss={train_loss / max(1, seen):.4f} items=[{items}]"
                )

        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                loss, _ = model.loss(batch)
                total_loss = scalar_loss(loss)
                batch_n = int(batch["rgb"].shape[0])
                val_loss += float(total_loss.detach()) * batch_n
                val_seen += batch_n
        mean_val_loss = val_loss / max(1, val_seen)
        print(
            f"epoch {epoch + 1}/{args.epochs} complete "
            f"train_loss={train_loss / max(1, seen):.4f} val_loss={mean_val_loss:.4f} "
            f"time={time.time() - start:.1f}s"
        )

        save_checkpoint(weights_dir / "last.pt", model, optimizer, epoch, best_loss, args)
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            save_checkpoint(weights_dir / "best.pt", model, optimizer, epoch, best_loss, args)
            print(f"saved best checkpoint: {weights_dir / 'best.pt'}")
        if args.save_period > 0 and (epoch + 1) % args.save_period == 0:
            save_checkpoint(weights_dir / f"epoch{epoch + 1}.pt", model, optimizer, epoch, best_loss, args)

    print(f"Training complete. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
