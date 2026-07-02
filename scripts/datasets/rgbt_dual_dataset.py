from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from common import COCO_TO_YOLO_CATEGORY, load_json, resolve_data_root


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def letterbox(image: np.ndarray, imgsz: int, pad_value: int = 114) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = image.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = imgsz - new_w
    pad_h = imgsz - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    return padded, scale, (float(left), float(top))


def image_to_tensor(image: np.ndarray, channels: int) -> Tensor:
    if channels == 1:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[None, :, :]
    else:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)


class RGBTDualDataset(Dataset):
    """Paired RGB/TIR dataset that returns tensors for dual-stream detectors."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: str = "train",
        imgsz: int = 640,
        max_images: int | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("RGBTDualDataset currently supports train/val splits with COCO labels.")
        self.data_root = resolve_data_root(data_root)
        self.split = split
        self.imgsz = imgsz
        self.split_dir = self.data_root / split
        coco = load_json(self.split_dir / f"{split}.json")
        images = coco["images"]
        if max_images is not None:
            images = images[:max_images]
        keep_names = {image["file_name"] for image in images}
        image_ids = {image["id"] for image in images}
        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            if ann["image_id"] not in image_ids or ann.get("iscrowd", 0):
                continue
            if ann["category_id"] not in COCO_TO_YOLO_CATEGORY:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            anns_by_image[ann["image_id"]].append(ann)
        self.items = [
            {
                "image": image,
                "annotations": anns_by_image.get(image["id"], []),
            }
            for image in images
            if image["file_name"] in keep_names
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        item = self.items[index]
        image = item["image"]
        name = image["file_name"]
        rgb = imread_unicode(self.split_dir / "rgb" / name, cv2.IMREAD_COLOR)
        tir = imread_unicode(self.split_dir / "tir" / name, cv2.IMREAD_GRAYSCALE)
        if rgb is None:
            raise FileNotFoundError(self.split_dir / "rgb" / name)
        if tir is None:
            raise FileNotFoundError(self.split_dir / "tir" / name)
        if rgb.shape[:2] != tir.shape[:2]:
            tir = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        rgb_lb, scale, (pad_x, pad_y) = letterbox(rgb, self.imgsz)
        tir_lb, _, _ = letterbox(tir, self.imgsz)
        labels = []
        for ann in item["annotations"]:
            cls = COCO_TO_YOLO_CATEGORY[int(ann["category_id"])]
            x, y, w, h = [float(v) for v in ann["bbox"]]
            x1 = x * scale + pad_x
            y1 = y * scale + pad_y
            x2 = (x + w) * scale + pad_x
            y2 = (y + h) * scale + pad_y
            x1 = min(self.imgsz, max(0.0, x1))
            y1 = min(self.imgsz, max(0.0, y1))
            x2 = min(self.imgsz, max(0.0, x2))
            y2 = min(self.imgsz, max(0.0, y2))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            labels.append(
                [
                    float(cls),
                    ((x1 + x2) * 0.5) / self.imgsz,
                    ((y1 + y2) * 0.5) / self.imgsz,
                    bw / self.imgsz,
                    bh / self.imgsz,
                ]
            )

        if labels:
            label_tensor = torch.tensor(labels, dtype=torch.float32)
            cls_tensor = label_tensor[:, 0:1]
            bbox_tensor = label_tensor[:, 1:5]
        else:
            cls_tensor = torch.zeros((0, 1), dtype=torch.float32)
            bbox_tensor = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "rgb": image_to_tensor(rgb_lb, channels=3),
            "tir": image_to_tensor(tir_lb, channels=1),
            "cls": cls_tensor,
            "bboxes": bbox_tensor,
            "im_file": str(self.split_dir / "rgb" / name),
            "ori_shape": torch.tensor([image["height"], image["width"]], dtype=torch.float32),
            "resized_shape": torch.tensor([self.imgsz, self.imgsz], dtype=torch.float32),
        }


def collate_rgbt(batch: list[dict]) -> dict[str, Tensor | list[str]]:
    rgb = torch.stack([sample["rgb"] for sample in batch])
    tir = torch.stack([sample["tir"] for sample in batch])
    cls_values = []
    bbox_values = []
    batch_indices = []
    for batch_idx, sample in enumerate(batch):
        cls = sample["cls"]
        bboxes = sample["bboxes"]
        if cls.numel() == 0:
            continue
        cls_values.append(cls)
        bbox_values.append(bboxes)
        batch_indices.append(torch.full((cls.shape[0],), batch_idx, dtype=torch.float32))
    if cls_values:
        cls_out = torch.cat(cls_values, dim=0)
        bbox_out = torch.cat(bbox_values, dim=0)
        batch_idx_out = torch.cat(batch_indices, dim=0)
    else:
        cls_out = torch.zeros((0, 1), dtype=torch.float32)
        bbox_out = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx_out = torch.zeros((0,), dtype=torch.float32)
    return {
        "rgb": rgb,
        "tir": tir,
        "img": rgb,
        "cls": cls_out,
        "bboxes": bbox_out,
        "batch_idx": batch_idx_out,
        "im_file": [sample["im_file"] for sample in batch],
        "ori_shape": torch.stack([sample["ori_shape"] for sample in batch]),
        "resized_shape": torch.stack([sample["resized_shape"] for sample in batch]),
    }
