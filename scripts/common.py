from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CLASS_NAMES = ["car", "truck", "bus", "van", "freight_car"]
COCO_CATEGORY_IDS = [1, 2, 3, 4, 5]
YOLO_TO_COCO_CATEGORY = {idx: cat_id for idx, cat_id in enumerate(COCO_CATEGORY_IDS)}
COCO_TO_YOLO_CATEGORY = {cat_id: idx for idx, cat_id in YOLO_TO_COCO_CATEGORY.items()}
RGBT_DATASET_NAME = "gaiic_yolo_rgbt"


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    if data_root:
        path = Path(data_root).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Data root does not exist: {path}")
        return path

    cwd = Path.cwd()
    candidates = [
        p
        for p in cwd.iterdir()
        if p.is_dir()
        and (p / "train" / "train.json").exists()
        and (p / "val" / "val.json").exists()
        and (p / "test").exists()
    ]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-detect data root. Pass --data-root explicitly."
        )
    names = ", ".join(str(p) for p in candidates)
    raise RuntimeError(f"Multiple data roots found, pass --data-root: {names}")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def split_annotation_path(data_root: Path, split: str) -> Path:
    return data_root / split / f"{split}.json"
