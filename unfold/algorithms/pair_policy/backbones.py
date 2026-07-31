from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class DenseBackboneBase(nn.Module):
    out_channels: int
    stride: int

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class BilinearUNetBackbone(DenseBackboneBase):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        channels: list[int] | tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"bilinear_unet expects 4 channel stages, got {channels}")
        c1, c2, c3, c4 = (int(c) for c in channels)
        self.stem = ConvBlock(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)
        self.proj = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.out_channels = int(out_channels)
        self.stride = 1

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        y2 = self.up2(x3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)
        feat = self.proj(y0)
        return {
            "feat": feat,
            "pyramid": OrderedDict(
                p0=feat,
                p1=x1,
                p2=x2,
                p3=x3,
            ),
        }


class ToyUNetBackbone(BilinearUNetBackbone):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, out_channels: int = 64):
        c1 = int(base_channels)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(c1, c1 * 2, c1 * 4, c1 * 8),
        )


class TorchvisionResNetFPNBackbone(DenseBackboneBase):
    def __init__(
        self,
        in_channels: int = 3,
        backbone_name: str = "resnet18",
        out_channels: int = 64,
        trainable_layers: int = 3,
        returned_layers: list[int] | None = None,
        weights: str | None = None,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("torchvision_resnet_fpn currently supports in_channels=3 only")
        try:
            from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        except ModuleNotFoundError as exc:
            raise RuntimeError("torchvision is required for torchvision_resnet_fpn backbone") from exc

        weights_enum = None
        if weights is not None:
            from torchvision.models import get_model_weights

            weights_enum = get_model_weights(backbone_name).DEFAULT if str(weights).upper() == "DEFAULT" else None

        self.body = resnet_fpn_backbone(
            backbone_name=backbone_name,
            weights=weights_enum,
            trainable_layers=int(trainable_layers),
            returned_layers=returned_layers,
        )
        self.out_channels = int(out_channels)
        self.stride = 4
        if int(self.body.out_channels) != self.out_channels:
            self.feat_proj = nn.Conv2d(int(self.body.out_channels), self.out_channels, kernel_size=1)
        else:
            self.feat_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pyramid = self.body(x)
        first_key = sorted(pyramid.keys())[0]
        feat = self.feat_proj(pyramid[first_key])
        return {
            "feat": feat,
            "pyramid": pyramid,
        }


class MonaiUNetBackbone(DenseBackboneBase):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        channels: list[int] | tuple[int, ...] = (32, 64, 128, 256),
        strides: list[int] | tuple[int, ...] = (2, 2, 2),
        num_res_units: int = 2,
        norm: str = "BATCH",
        dropout: float = 0.0,
    ):
        super().__init__()
        try:
            from monai.networks.nets import UNet
        except ModuleNotFoundError as exc:
            raise RuntimeError("monai is required for monai_unet backbone") from exc

        self.body = UNet(
            spatial_dims=2,
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            channels=tuple(int(x) for x in channels),
            strides=tuple(int(x) for x in strides),
            num_res_units=int(num_res_units),
            norm=str(norm),
            dropout=float(dropout),
        )
        self.out_channels = int(out_channels)
        self.stride = 1

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.body(x)
        return {
            "feat": feat,
            "pyramid": OrderedDict(
                p0=feat,
            ),
        }


class SMPUnetPlusPlusBackbone(DenseBackboneBase):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        encoder_name: str = "resnet50",
        encoder_weights: str | None = "imagenet",
        decoder_channels: list[int] | tuple[int, ...] = (256, 128, 64, 32, 16),
        decoder_attention_type: str | None = None,
    ):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ModuleNotFoundError as exc:
            raise RuntimeError("segmentation-models-pytorch is required for smp_unetplusplus backbone") from exc

        self.body = smp.UnetPlusPlus(
            encoder_name=str(encoder_name),
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=int(out_channels),
            decoder_channels=tuple(int(x) for x in decoder_channels),
            decoder_attention_type=decoder_attention_type,
        )
        self.out_channels = int(out_channels)
        self.stride = 1

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.body(x)
        return {
            "feat": feat,
            "pyramid": OrderedDict(
                p0=feat,
            ),
        }


class SMPSegformerBackbone(DenseBackboneBase):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        encoder_name: str = "mit_b4",
        encoder_weights: str | None = "imagenet",
        decoder_segmentation_channels: int = 256,
    ):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ModuleNotFoundError as exc:
            raise RuntimeError("segmentation-models-pytorch is required for smp_segformer backbone") from exc

        self.body = smp.Segformer(
            encoder_name=str(encoder_name),
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=int(out_channels),
            decoder_segmentation_channels=int(decoder_segmentation_channels),
        )
        self.out_channels = int(out_channels)
        self.stride = 1

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.body(x)
        return {
            "feat": feat,
            "pyramid": OrderedDict(
                p0=feat,
            ),
        }


def build_backbone(cfg: dict[str, Any], in_channels: int = 3) -> DenseBackboneBase:
    name = str(cfg.get("name", "toy_unet"))
    params = dict(cfg.get("params", {}) or {})
    if "in_channels" not in params:
        params["in_channels"] = int(in_channels)

    if name == "toy_unet":
        return ToyUNetBackbone(**params)
    if name == "bilinear_unet":
        return BilinearUNetBackbone(**params)
    if name == "torchvision_resnet_fpn":
        return TorchvisionResNetFPNBackbone(**params)
    if name == "monai_unet":
        return MonaiUNetBackbone(**params)
    if name == "smp_unetplusplus":
        return SMPUnetPlusPlusBackbone(**params)
    if name == "smp_segformer":
        return SMPSegformerBackbone(**params)
    raise ValueError(f"Unknown backbone: {name}")
