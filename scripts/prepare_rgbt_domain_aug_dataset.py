from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import CLASS_NAMES, load_json, resolve_data_root
from prepare_rgbt_yolo_dataset import fuse_rgbt, imread_unicode, imwrite_unicode


def apply_brightness_contrast_gamma(
    image: np.ndarray,
    rng: np.random.Generator,
    brightness_range: list[float],
    contrast_range: list[float],
    gamma_range: list[float],
) -> np.ndarray:
    brightness = rng.uniform(brightness_range[0], brightness_range[1])
    contrast = rng.uniform(contrast_range[0], contrast_range[1])
    gamma = rng.uniform(gamma_range[0], gamma_range[1])
    out = image.astype(np.float32) * contrast + brightness
    out = np.clip(out, 0, 255) / 255.0
    out = np.power(out, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_tir_adjust(
    image: np.ndarray,
    rng: np.random.Generator,
    brightness_range: list[float],
    contrast_range: list[float],
) -> np.ndarray:
    out = image.copy().astype(np.float32)
    brightness = rng.uniform(brightness_range[0], brightness_range[1])
    contrast = rng.uniform(contrast_range[0], contrast_range[1])
    out[:, :, 0] = out[:, :, 0] * contrast + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_noise(image: np.ndarray, rng: np.random.Generator, std_range: list[float]) -> np.ndarray:
    std = rng.uniform(std_range[0], std_range[1])
    noise = rng.normal(0.0, std, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_blur(image: np.ndarray, rng: np.random.Generator, choices: list[int]) -> np.ndarray:
    kernel = int(rng.choice(choices))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def apply_jpeg(image: np.ndarray, rng: np.random.Generator, quality_range: list[int]) -> np.ndarray:
    quality = int(rng.integers(quality_range[0], quality_range[1] + 1))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image


def augment(image: np.ndarray, config: dict, rng: np.random.Generator) -> np.ndarray:
    out = image
    probs = config["probabilities"]
    if rng.random() < probs["brightness_contrast_gamma"]:
        out = apply_brightness_contrast_gamma(
            out,
            rng,
            config["brightness_delta_range"],
            config["contrast_range"],
            config["gamma_range"],
        )
    if rng.random() < probs["tir_contrast"]:
        out = apply_tir_adjust(
            out,
            rng,
            config["tir_brightness_delta_range"],
            config["tir_contrast_range"],
        )
    if rng.random() < probs["blur"]:
        out = apply_blur(out, rng, config["blur_kernel_choices"])
    if rng.random() < probs["noise"]:
        out = apply_noise(out, rng, config["noise_std_range"])
    if rng.random() < probs["jpeg"]:
        out = apply_jpeg(out, rng, config["jpeg_quality_range"])
    return out


def read_pair(data_root: Path, split: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = imread_unicode(data_root / split / "rgb" / name, cv2.IMREAD_COLOR)
    tir = imread_unicode(data_root / split / "tir" / name, cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise FileNotFoundError(data_root / split / "rgb" / name)
    if tir is None:
        raise FileNotFoundError(data_root / split / "tir" / name)
    if rgb.shape[:2] != tir.shape[:2]:
        tir = cv2.resize(tir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return rgb, tir


def copy_labels(src: Path, dst: Path, names: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        stem = Path(name).stem
        shutil.copy2(src / f"{stem}.txt", dst / f"{stem}.txt")


def write_dataset_yaml(dataset_dir: Path, yaml_path: Path) -> None:
    lines = [
        f"path: {dataset_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_train(data_root: Path, out_dir: Path, config: dict, max_images: int | None, overwrite: bool) -> None:
    coco = load_json(data_root / "train" / "train.json")
    names = [image["file_name"] for image in coco["images"]]
    if max_images is not None:
        names = names[:max_images]
    rng = np.random.default_rng(int(config.get("seed", 2026)))
    dst_dir = out_dir / "images" / "train"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(names, start=1):
        dst = dst_dir / name
        if dst.exists() and not overwrite:
            continue
        rgb, tir = read_pair(data_root, "train", name)
        fused = fuse_rgbt(rgb, tir, config.get("fusion", "rgbt"))
        imwrite_unicode(dst, augment(fused, config, rng))
        if idx % 1000 == 0:
            print(f"[train] augmented {idx}/{len(names)}")
    copy_labels(Path("datasets/gaiic_yolo_rgbt/labels/train"), out_dir / "labels" / "train", names)


def copy_split_from_rgbt(split: str, out_dir: Path, max_images: int | None, overwrite: bool) -> None:
    src_images = Path("datasets/gaiic_yolo_rgbt/images") / split
    src_labels = Path("datasets/gaiic_yolo_rgbt/labels") / split
    names = sorted(path.name for path in src_images.glob("*.jpg"))
    if max_images is not None:
        names = names[:max_images]
    dst_images = out_dir / "images" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    for name in names:
        dst = dst_images / name
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(src_images / name, dst)
    if src_labels.exists():
        copy_labels(src_labels, out_dir / "labels" / split, names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RGB-T test-domain augmented YOLO dataset.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--config", default="configs/domain_aug_rgbt.json")
    parser.add_argument("--out-dir", default="datasets/gaiic_yolo_rgbt_aug_domain")
    parser.add_argument("--yaml-out", default="configs/gaiic_yolo_rgbt_aug_domain.yaml")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir).resolve()
    process_train(data_root, out_dir, config, args.max_images, args.overwrite)
    copy_split_from_rgbt("val", out_dir, args.max_images, args.overwrite)
    copy_split_from_rgbt("test", out_dir, args.max_images, args.overwrite)
    write_dataset_yaml(out_dir, Path(args.yaml_out))
    print(f"Prepared augmented RGB-T dataset: {out_dir}")
    print(f"Wrote config: {args.yaml_out}")


if __name__ == "__main__":
    main()
