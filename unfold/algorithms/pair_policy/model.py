from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import build_backbone


def _make_dense_head(in_channels: int, hidden_channels: int) -> nn.Sequential:
    mid_channels = max(hidden_channels // 2, 16)
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, 1, kernel_size=1),
    )


class PairPolicyNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 64,
        num_x1_samples: int = 4,
        condition_sigma: float = 5.0,
        backbone: dict | None = None,
    ):
        super().__init__()
        self.num_x1_samples = int(num_x1_samples)
        self.condition_sigma = float(condition_sigma)
        self.backbone = build_backbone(backbone or {"name": "toy_unet", "params": {"out_channels": feature_dim}}, in_channels=in_channels)
        backbone_channels = int(self.backbone.out_channels)
        self.feature_proj = nn.Identity() if backbone_channels == int(feature_dim) else nn.Conv2d(backbone_channels, feature_dim, kernel_size=1)
        head_channels = max(int(feature_dim), 32)
        self.a1_head = _make_dense_head(feature_dim, head_channels)
        self.a2_head = _make_dense_head(feature_dim * 2 + 1, head_channels)

    def _sample_point_features(self, feat_map: torch.Tensor, point_xy: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = feat_map.shape
        if point_xy.shape[1] == 0:
            return feat_map.new_zeros((bsz, 0, channels))
        x = point_xy[..., 0] / max(width - 1, 1) * 2.0 - 1.0
        y = point_xy[..., 1] / max(height - 1, 1) * 2.0 - 1.0
        grid = torch.stack([x, y], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(feat_map, grid, mode="bilinear", align_corners=True)
        return sampled.squeeze(-1).transpose(1, 2)

    def _build_gaussian_condition_maps(
        self,
        height: int,
        width: int,
        sampled_x1_xy: torch.Tensor,
        sampled_x1_valid: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_queries = sampled_x1_xy.shape[:2]
        cond = sampled_x1_xy.new_zeros((bsz, num_queries, height, width))
        if num_queries == 0:
            return cond

        sigma = max(self.condition_sigma, 1e-6)
        sigma_sq = float(sigma * sigma)
        xs = torch.arange(width, device=sampled_x1_xy.device, dtype=sampled_x1_xy.dtype).view(1, 1, 1, width)
        ys = torch.arange(height, device=sampled_x1_xy.device, dtype=sampled_x1_xy.dtype).view(1, 1, height, 1)
        x0 = sampled_x1_xy[..., 0].view(bsz, num_queries, 1, 1)
        y0 = sampled_x1_xy[..., 1].view(bsz, num_queries, 1, 1)
        dist_sq = (xs - x0).square() + (ys - y0).square()
        cond = torch.exp(-0.5 * dist_sq / sigma_sq)
        if sampled_x1_valid.numel() > 0:
            cond = cond * sampled_x1_valid[:, :, None, None].to(dtype=cond.dtype)
        return cond

    def forward(
        self,
        image: torch.Tensor,
        sampled_x1_xy: torch.Tensor,
        sampled_x1_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out_h, out_w = image.shape[-2:]
        backbone_out = self.backbone(image)
        feat_map = self.feature_proj(backbone_out["feat"])
        a1_logits = self.a1_head(feat_map)
        bsz, channels, height, width = feat_map.shape
        query_feat = self._sample_point_features(feat_map, sampled_x1_xy)
        g1_map = self._build_gaussian_condition_maps(height, width, sampled_x1_xy, sampled_x1_valid)

        feat_map_expand = feat_map[:, None].expand(-1, query_feat.shape[1], -1, -1, -1)
        g1_map_feat = g1_map[:, :, None, :, :]
        query_feat_map = query_feat[:, :, :, None, None].expand(-1, -1, -1, height, width)
        a2_input = torch.cat([feat_map_expand, query_feat_map, g1_map_feat], dim=2)
        a2_logits = self.a2_head(a2_input.flatten(0, 1)).view(bsz, query_feat.shape[1], height, width)
        if (height, width) != (out_h, out_w):
            a1_logits = F.interpolate(a1_logits, size=(out_h, out_w), mode="bicubic", align_corners=False)
            a2_logits = F.interpolate(
                a2_logits.flatten(0, 1).unsqueeze(1),
                size=(out_h, out_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1).view(bsz, query_feat.shape[1], out_h, out_w)
        if sampled_x1_valid.numel() > 0:
            a2_logits = a2_logits.masked_fill(~sampled_x1_valid[:, :, None, None], float("-inf"))

        return {
            "feature_map": feat_map,
            "backbone_features": backbone_out.get("pyramid"),
            "a1_logits": a1_logits.squeeze(1),
            "query_feat": query_feat,
            "g1_map": g1_map,
            "a2_logits": a2_logits,
        }
