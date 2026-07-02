from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import Tensor, nn


def autopad(kernel_size: int, padding: int | None = None, dilation: int = 1) -> int:
    if padding is not None:
        return padding
    return dilation * (kernel_size - 1) // 2


class ConvBNAct(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size,
            stride,
            padding=autopad(kernel_size),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class GatedFusionModule(nn.Module):
    """Global-context RGB/TIR gating from GEM-YOLO."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, (channels * 2) // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, rgb: Tensor, tir: Tensor) -> Tensor:
        if rgb.shape != tir.shape:
            raise ValueError(f"GFM expects matching feature shapes, got {rgb.shape} and {tir.shape}.")
        weights = self.gate(torch.cat((rgb, tir), dim=1))
        rgb_weight = weights[:, 0:1]
        tir_weight = weights[:, 1:2]
        return rgb_weight * rgb + tir_weight * tir


class SpaceToDepth(nn.Module):
    def __init__(self, block_size: int = 2) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        s = self.block_size
        if h % s != 0 or w % s != 0:
            x = nn.functional.pad(x, (0, w % s, 0, h % s))
            b, c, h, w = x.shape
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


class SPDConv(nn.Module):
    """Space-to-depth downsampling followed by depthwise separable convolution."""

    def __init__(self, c1: int, c2: int, block_size: int = 2) -> None:
        super().__init__()
        expanded = c1 * block_size * block_size
        self.space_to_depth = SpaceToDepth(block_size)
        self.depthwise = ConvBNAct(expanded, expanded, kernel_size=3, groups=expanded)
        self.pointwise = ConvBNAct(expanded, c2, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(self.space_to_depth(x)))


class GhostConv(nn.Module):
    def __init__(self, c1: int, c2: int, kernel_size: int = 1, ratio: int = 2) -> None:
        super().__init__()
        hidden = max(1, (c2 + ratio - 1) // ratio)
        cheap = c2 - hidden
        self.primary = ConvBNAct(c1, hidden, kernel_size)
        self.cheap = (
            ConvBNAct(hidden, cheap, kernel_size=3, groups=hidden)
            if cheap > 0
            else nn.Identity()
        )
        self.out_channels = c2

    def forward(self, x: Tensor) -> Tensor:
        primary = self.primary(x)
        if isinstance(self.cheap, nn.Identity):
            return primary[:, : self.out_channels]
        return torch.cat((primary, self.cheap(primary)), dim=1)[:, : self.out_channels]


class GhostBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, expansion: float = 0.5, shortcut: bool = True) -> None:
        super().__init__()
        hidden = max(8, int(c2 * expansion))
        self.block = nn.Sequential(
            GhostConv(c1, hidden, kernel_size=1),
            ConvBNAct(hidden, hidden, kernel_size=3, groups=hidden),
            GhostConv(hidden, c2, kernel_size=1),
        )
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        y = self.block(x)
        return x + y if self.shortcut else y


class GSConv(nn.Module):
    """GSConv-style light convolution with channel shuffle."""

    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__()
        half = c2 // 2
        self.conv = ConvBNAct(c1, half, kernel_size=1, stride=stride)
        self.dw = ConvBNAct(half, c2 - half, kernel_size=5, groups=half)

    @staticmethod
    def channel_shuffle(x: Tensor, groups: int = 2) -> Tensor:
        b, c, h, w = x.shape
        if c % groups != 0:
            return x
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv(x)
        return self.channel_shuffle(torch.cat((x1, self.dw(x1)), dim=1))


class GemStream(nn.Module):
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_channels, width, kernel_size=3, stride=2)
        self.stage1 = nn.Sequential(
            SPDConv(width, width * 2),
            GhostBottleneck(width * 2, width * 2),
        )
        self.stage2 = nn.Sequential(
            SPDConv(width * 2, width * 4),
            GhostBottleneck(width * 4, width * 4),
            GhostBottleneck(width * 4, width * 4),
        )
        self.stage3 = nn.Sequential(
            SPDConv(width * 4, width * 8),
            GhostBottleneck(width * 8, width * 8),
            GhostBottleneck(width * 8, width * 8),
        )
        self.stage4 = nn.Sequential(
            SPDConv(width * 8, width * 16),
            GhostBottleneck(width * 16, width * 16),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


class GhostPAN(nn.Module):
    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        c3, c4, c5 = channels
        self.p5_reduce = GhostConv(c5, c4)
        self.p4_fuse = GhostBottleneck(c4 + c4, c4, shortcut=False)
        self.p4_reduce = GhostConv(c4, c3)
        self.p3_fuse = GhostBottleneck(c3 + c3, c3, shortcut=False)
        self.p3_down = GSConv(c3, c4, stride=2)
        self.n4_fuse = GhostBottleneck(c4 + c4, c4, shortcut=False)
        self.p4_down = GSConv(c4, c5, stride=2)
        self.n5_fuse = GhostBottleneck(c5 + c5, c5, shortcut=False)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> list[Tensor]:
        p3, p4, p5 = features
        p5_up = nn.functional.interpolate(self.p5_reduce(p5), size=p4.shape[-2:], mode="nearest")
        n4 = self.p4_fuse(torch.cat((p5_up, p4), dim=1))
        n4_up = nn.functional.interpolate(self.p4_reduce(n4), size=p3.shape[-2:], mode="nearest")
        n3 = self.p3_fuse(torch.cat((n4_up, p3), dim=1))
        n3_down = self.p3_down(n3)
        out4 = self.n4_fuse(torch.cat((n3_down, n4), dim=1))
        out4_down = self.p4_down(out4)
        out5 = self.n5_fuse(torch.cat((out4_down, p5), dim=1))
        return [n3, out4, out5]


@dataclass(frozen=True)
class GemYoloConfig:
    num_classes: int = 5
    width: int = 32
    rgb_channels: int = 3
    tir_channels: int = 1
    reg_max: int = 16


class GemYoloDetector(nn.Module):
    """Pure-PyTorch GEM-YOLO-style RGB-T detector.

    The model returns Ultralytics Detect outputs, so the same v8 detection loss can
    be used by train_gem_yolo.py.
    """

    def __init__(self, config: GemYoloConfig | None = None) -> None:
        super().__init__()
        self.config = config or GemYoloConfig()
        w = self.config.width
        channels = (w * 4, w * 8, w * 16)
        self.rgb_stream = GemStream(self.config.rgb_channels, w)
        self.tir_stream = GemStream(self.config.tir_channels, w)
        self.fuse_p3 = GatedFusionModule(channels[0])
        self.fuse_p4 = GatedFusionModule(channels[1])
        self.fuse_p5 = GatedFusionModule(channels[2])
        self.neck = GhostPAN(channels)

        try:
            from ultralytics.nn.modules.head import Detect
            from ultralytics.utils import DEFAULT_CFG
        except ImportError as exc:
            raise ImportError("GemYoloDetector requires ultralytics for the Detect head.") from exc

        self.detect = Detect(nc=self.config.num_classes, reg_max=self.config.reg_max, ch=channels)
        self.model = nn.ModuleList([self.rgb_stream, self.tir_stream, self.neck, self.detect])
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.detect.stride = self.stride
        self.detect.bias_init()
        self.args = DEFAULT_CFG
        self.nc = self.config.num_classes
        self.names = {idx: name for idx, name in enumerate(["car", "truck", "bus", "van", "freight_car"])}
        self.criterion = None

    def extract_features(self, rgb: Tensor, tir: Tensor) -> list[Tensor]:
        rgb_features = self.rgb_stream(rgb)
        tir_features = self.tir_stream(tir)
        fused = (
            self.fuse_p3(rgb_features[0], tir_features[0]),
            self.fuse_p4(rgb_features[1], tir_features[1]),
            self.fuse_p5(rgb_features[2], tir_features[2]),
        )
        return self.neck(fused)

    def forward(self, rgb: Tensor | dict[str, Tensor], tir: Tensor | None = None):
        if isinstance(rgb, dict):
            tir = rgb["tir"]
            rgb = rgb["rgb"]
        if tir is None:
            raise ValueError("GemYoloDetector.forward requires rgb and tir tensors.")
        return self.detect(self.extract_features(rgb, tir))

    def init_criterion(self):
        try:
            from ultralytics.utils.loss import v8DetectionLoss
        except ImportError as exc:
            raise ImportError("GemYoloDetector loss requires ultralytics.") from exc
        return v8DetectionLoss(self)

    def loss(self, batch: dict[str, Tensor], preds=None):
        if self.criterion is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["rgb"], batch["tir"])
        loss, loss_items = self.criterion(preds, batch)
        if loss.ndim > 0:
            return loss.sum(), loss_items
        return loss, loss_items


def build_gem_yolo_from_dict(config: dict) -> GemYoloDetector:
    model_cfg = config.get("model", config)
    gem_cfg = GemYoloConfig(
        num_classes=int(model_cfg.get("num_classes", model_cfg.get("nc", 5))),
        width=int(model_cfg.get("width", 32)),
        rgb_channels=int(model_cfg.get("rgb_channels", 3)),
        tir_channels=int(model_cfg.get("tir_channels", 1)),
        reg_max=int(model_cfg.get("reg_max", 16)),
    )
    return GemYoloDetector(gem_cfg)
