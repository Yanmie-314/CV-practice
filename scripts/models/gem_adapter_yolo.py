from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class GEMAdapter(nn.Module):
    """Input-level GEM adapter that preserves the pretrained RGB-T YOLO input."""

    baseline_tensor_order = "R,G,TIR"

    def __init__(self, hidden: int = 24, residual_scale: float = 0.10) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.rgb_proj = nn.Sequential(ConvBNAct(3, hidden), ConvBNAct(hidden, hidden))
        self.tir_proj = nn.Sequential(ConvBNAct(1, hidden), ConvBNAct(hidden, hidden))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 2, max(8, hidden // 2), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, hidden // 2), 2, 1),
            nn.Softmax(dim=1),
        )
        self.out = nn.Sequential(
            ConvBNAct(hidden, hidden),
            nn.Conv2d(hidden, 3, kernel_size=1),
            nn.Tanh(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        final_conv = self.out[1]
        if isinstance(final_conv, nn.Conv2d):
            nn.init.zeros_(final_conv.weight)
            nn.init.zeros_(final_conv.bias)

    @staticmethod
    def baseline_rgbt(rgb: Tensor, tir: Tensor) -> Tensor:
        # Fused JPEGs are written as BGR [TIR, G, R], then decoded to RGB
        # tensors by Ultralytics/OpenCV preprocessing: [R, G, TIR].
        red = rgb[:, 0:1]
        green = rgb[:, 1:2]
        return torch.cat((red, green, tir), dim=1)

    def forward(self, rgb: Tensor, tir: Tensor) -> Tensor:
        rgb_feat = self.rgb_proj(rgb)
        tir_feat = self.tir_proj(tir)
        weights = self.gate(torch.cat((rgb_feat, tir_feat), dim=1))
        fused_feat = weights[:, 0:1] * rgb_feat + weights[:, 1:2] * tir_feat
        residual = self.out(fused_feat) * self.residual_scale
        return (self.baseline_rgbt(rgb, tir) + residual).clamp(0.0, 1.0)


class GEMAdapterYOLO(nn.Module):
    """Wrap a pretrained Ultralytics DetectionModel with a trainable RGB/TIR adapter."""

    def __init__(
        self,
        yolo_weights: str | Path,
        adapter_hidden: int = 24,
        residual_scale: float = 0.10,
        freeze_yolo: bool = True,
    ) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("GEMAdapterYOLO requires ultralytics.") from exc
        self.adapter = GEMAdapter(hidden=adapter_hidden, residual_scale=residual_scale)
        self.adapter_hidden = adapter_hidden
        self.residual_scale = residual_scale
        self.yolo_weights = str(yolo_weights)
        self.yolo = YOLO(str(yolo_weights)).model
        self.yolo.float()
        self._normalize_yolo_args()
        self.freeze_yolo(freeze_yolo)

    def _normalize_yolo_args(self) -> None:
        try:
            from ultralytics.utils import DEFAULT_CFG
            from ultralytics.utils import IterableSimpleNamespace
        except ImportError:
            return
        if isinstance(getattr(self.yolo, "args", None), dict):
            merged = vars(DEFAULT_CFG).copy()
            merged.update(self.yolo.args)
            self.yolo.args = IterableSimpleNamespace(**merged)

    def freeze_yolo(self, freeze: bool = True) -> None:
        for param in self.yolo.parameters():
            param.requires_grad = not freeze
        if freeze:
            self.yolo.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(param.requires_grad for param in self.yolo.parameters()):
            self.yolo.eval()
        return self

    def forward(self, rgb: Tensor | dict[str, Tensor], tir: Tensor | None = None):
        if isinstance(rgb, dict):
            tir = rgb["tir"]
            rgb = rgb["rgb"]
        if tir is None:
            raise ValueError("GEMAdapterYOLO.forward requires rgb and tir tensors.")
        return self.yolo(self.adapter(rgb, tir))

    def loss(self, batch: dict[str, Tensor], preds=None):
        fused = self.adapter(batch["rgb"], batch["tir"])
        yolo_batch = dict(batch)
        yolo_batch["img"] = fused
        return self.yolo.loss(yolo_batch, preds=preds)
