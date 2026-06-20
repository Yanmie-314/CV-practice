from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOSS_COLUMNS = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
]


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]
    return df


def plot_single(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160)
    pairs = [
        ("box_loss", "train/box_loss", "val/box_loss"),
        ("cls_loss", "train/cls_loss", "val/cls_loss"),
        ("dfl_loss", "train/dfl_loss", "val/dfl_loss"),
    ]
    for ax, (name, train_col, val_col) in zip(axes, pairs):
        ax.plot(df["epoch"], df[train_col], label="train", linewidth=1.8)
        ax.plot(df["epoch"], df[val_col], label="val", linewidth=1.8)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_compare(base: pd.DataFrame, aug: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=160)
    pairs = [
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Cls Loss"),
        ("train/dfl_loss", "Train DFL Loss"),
        ("val/box_loss", "Val Box Loss"),
        ("val/cls_loss", "Val Cls Loss"),
        ("val/dfl_loss", "Val DFL Loss"),
    ]
    for ax, (column, title) in zip(axes.flat, pairs):
        ax.plot(base["epoch"], base[column], label="baseline", linewidth=1.8)
        ax.plot(aug["epoch"], aug[column], label="aug_domain", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("Training Loss Comparison")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(base: pd.DataFrame, aug: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for name, df in [("baseline", base), ("aug_domain", aug)]:
        first = df.iloc[0]
        last = df.iloc[-1]
        row = {"run": name, "first_epoch": int(first["epoch"]), "last_epoch": int(last["epoch"])}
        for column in LOSS_COLUMNS:
            row[f"{column}_start"] = float(first[column])
            row[f"{column}_end"] = float(last[column])
            row[f"{column}_delta"] = float(last[column] - first[column])
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot YOLO training loss curves.")
    parser.add_argument("--baseline", default="work_dirs/yolo11s_rgbt/results.csv")
    parser.add_argument("--aug", default="work_dirs/yolo11s_rgbt_aug_domain/results.csv")
    parser.add_argument("--out-dir", default="analysis/loss_curves")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    base = load_results(Path(args.baseline))
    aug = load_results(Path(args.aug))
    plot_single(base, "Baseline RGB-T Loss Curves", out_dir / "baseline_loss.png")
    plot_single(aug, "Augmented RGB-T Loss Curves", out_dir / "aug_domain_loss.png")
    plot_compare(base, aug, out_dir / "loss_comparison.png")
    write_summary(base, aug, out_dir / "loss_summary.csv")
    print(f"Wrote: {out_dir / 'baseline_loss.png'}")
    print(f"Wrote: {out_dir / 'aug_domain_loss.png'}")
    print(f"Wrote: {out_dir / 'loss_comparison.png'}")
    print(f"Wrote: {out_dir / 'loss_summary.csv'}")


if __name__ == "__main__":
    main()
