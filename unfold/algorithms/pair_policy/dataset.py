from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from unfold.algorithms.pair_policy.augment import (
    GeometricAugmentConfig,
    MaskAugmentConfig,
    RGBAugmentConfig,
    perturb_geometry,
    perturb_mask,
    perturb_rgb,
)
from unfold.algorithms.supervision.projection import masked_softmax_heatmap


_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PairPolicyBatch:
    image: torch.Tensor
    input_mask: torch.Tensor
    gt_mask: torch.Tensor
    a1_target: torch.Tensor
    a1_target_mask: torch.Tensor
    a1_negative_mask: torch.Tensor
    a1_valid_mask: torch.Tensor
    sampled_x1_xy: torch.Tensor
    sampled_x1_valid: torch.Tensor
    a2_target: torch.Tensor
    a2_target_mask: torch.Tensor
    a2_negative_mask: torch.Tensor
    a2_valid_mask: torch.Tensor
    a2_target_valid: torch.Tensor
    sample_id: list[str]
    asset_id: list[str]
    mask_area_ratio: torch.Tensor


def _image_minmax_heatmap(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    heat = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask) > 0
    out = np.zeros_like(heat, dtype=np.float32)
    if not np.any(valid):
        return out
    x = np.asarray(heat[valid], dtype=np.float32)
    vmin = float(np.min(x))
    vmax = float(np.max(x))
    if vmax <= vmin + 1e-12:
        out[valid] = 1.0
    else:
        out[valid] = (x - vmin) / (vmax - vmin)
    return out


def _image_exp_heatmap(values: np.ndarray, valid_mask: np.ndarray, tau: float) -> np.ndarray:
    heat = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask) > 0
    out = np.zeros_like(heat, dtype=np.float32)
    if not np.any(valid):
        return out
    x = np.asarray(heat[valid], dtype=np.float64)
    x = (x - float(np.max(x))) / float(max(tau, 1e-6))
    mapped = np.exp(x)
    vmax = float(np.max(mapped))
    if vmax > 1e-12:
        mapped = mapped / vmax
    out[valid] = mapped.astype(np.float32, copy=False)
    return out


def build_top_region_mask(
    *,
    value_map: np.ndarray,
    valid_mask: np.ndarray,
    top_ratio: float,
) -> np.ndarray:
    valid = np.asarray(valid_mask) > 0
    out = np.zeros_like(valid, dtype=np.uint8)
    if not np.any(valid):
        return out
    ratio = float(np.clip(top_ratio, 0.0, 1.0))
    if ratio <= 0.0:
        return out

    values = np.asarray(value_map, dtype=np.float32)
    valid_values = values[valid]
    take = max(1, int(np.ceil(valid_values.shape[0] * ratio)))
    take = min(take, int(valid_values.shape[0]))
    if take >= int(valid_values.shape[0]):
        out[valid] = 1
        return out

    threshold = float(np.partition(valid_values, valid_values.shape[0] - take)[valid_values.shape[0] - take])
    selected = valid & (values >= threshold)
    selected_flat = np.flatnonzero(selected.reshape(-1))
    if selected_flat.shape[0] > take:
        values_flat = values.reshape(-1)[selected_flat]
        keep = selected_flat[np.argsort(values_flat)[::-1][:take]]
        out.reshape(-1)[keep] = 1
        return out
    out[selected] = 1
    return out


def build_bottom_region_mask(
    *,
    value_map: np.ndarray,
    valid_mask: np.ndarray,
    bottom_ratio: float,
) -> np.ndarray:
    valid = np.asarray(valid_mask) > 0
    out = np.zeros_like(valid, dtype=np.uint8)
    if not np.any(valid):
        return out
    ratio = float(np.clip(bottom_ratio, 0.0, 1.0))
    if ratio <= 0.0:
        return out

    values = np.asarray(value_map, dtype=np.float32)
    valid_values = values[valid]
    take = max(1, int(np.ceil(valid_values.shape[0] * ratio)))
    take = min(take, int(valid_values.shape[0]))
    if take >= int(valid_values.shape[0]):
        out[valid] = 1
        return out

    threshold = float(np.partition(valid_values, take - 1)[take - 1])
    selected = valid & (values <= threshold)
    selected_flat = np.flatnonzero(selected.reshape(-1))
    if selected_flat.shape[0] > take:
        values_flat = values.reshape(-1)[selected_flat]
        keep = selected_flat[np.argsort(values_flat)[:take]]
        out.reshape(-1)[keep] = 1
        return out
    out[selected] = 1
    return out


def build_training_target(
    *,
    value_map: np.ndarray,
    valid_mask: np.ndarray,
    target_name: str,
    tau: float,
    top_ratio: float = 1.0,
) -> np.ndarray:
    name = str(target_name)
    if name == "masked_softmax":
        return masked_softmax_heatmap(value_map, valid_mask, tau=float(tau)).astype(np.float32, copy=False)
    if name == "topk_masked_softmax":
        return masked_softmax_heatmap(value_map, valid_mask, tau=float(tau)).astype(np.float32, copy=False)
    if name == "image_minmax":
        return _image_minmax_heatmap(value_map, valid_mask).astype(np.float32, copy=False)
    if name == "image_exp":
        return _image_exp_heatmap(value_map, valid_mask, tau=float(tau)).astype(np.float32, copy=False)
    raise ValueError(f"unsupported pair-policy target_name: {target_name}")


class PairPolicyDataset(Dataset):
    def __init__(
        self,
        *,
        index_payload: dict[str, Any],
        split: str,
        num_x1_samples: int = 4,
        target_name: str = "masked_softmax",
        a1_target_tau: float = 1.0,
        a2_target_tau: float = 1.0,
        top_ratio: float = 0.15,
        bottom_ratio: float = 0.5,
        train: bool = True,
        geom_aug_cfg: GeometricAugmentConfig | None = None,
        mask_aug_cfg: MaskAugmentConfig | None = None,
        rgb_aug_cfg: RGBAugmentConfig | None = None,
        resize_width: int | None = None,
        resize_height: int | None = None,
        input_normalization: str = "none",
        supervision_mask_mode: str = "heatmap_valid",
        loss_name: str = "masked_kl",
        min_a1_std: float = 0.0,
        min_a1_margin: float = 0.0,
        min_reward_row_margin: float = 0.0,
        seed: int = 42,
    ):
        self.split = str(split)
        self.train = bool(train)
        self.num_x1_samples = int(num_x1_samples)
        self.target_name = str(target_name)
        self.a1_target_tau = float(a1_target_tau)
        self.a2_target_tau = float(a2_target_tau)
        self.top_ratio = float(top_ratio)
        self.bottom_ratio = float(bottom_ratio)
        self.geom_aug_cfg = geom_aug_cfg or GeometricAugmentConfig(enabled=False)
        self.mask_aug_cfg = mask_aug_cfg or MaskAugmentConfig(enabled=train)
        self.rgb_aug_cfg = rgb_aug_cfg or RGBAugmentConfig(enabled=False)
        self.resize_width = int(resize_width) if resize_width else None
        self.resize_height = int(resize_height) if resize_height else None
        self.input_normalization = str(input_normalization)
        self.supervision_mask_mode = str(supervision_mask_mode)
        self.loss_name = str(loss_name)
        self._rng = random.Random(seed)
        self._shard_handles: dict[str, h5py.File] = {}
        self.rows = []
        for row in index_payload["rows"]:
            if row["split"] != self.split:
                continue
            if float(row.get("a1_std", 0.0)) < float(min_a1_std):
                continue
            if float(row.get("a1_top1_margin", 0.0)) < float(min_a1_margin):
                continue
            if float(row.get("reward_row_top1_margin", 0.0)) < float(min_reward_row_margin):
                continue
            self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No samples available for split={self.split}")

    def close(self) -> None:
        for handle in self._shard_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._shard_handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        handles = state.get("_shard_handles")
        if isinstance(handles, dict):
            for handle in handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
        state["_shard_handles"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._shard_handles = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _get_shard_handle(self, shard_path: str) -> h5py.File:
        shard_path = str(shard_path)
        handle = self._shard_handles.get(shard_path)
        if handle is None:
            handle = h5py.File(shard_path, "r")
            self._shard_handles[shard_path] = handle
        return handle

    def _resize_sample(
        self,
        *,
        rgb: np.ndarray,
        mask_bool: np.ndarray,
        a1_target: np.ndarray,
        a1_valid_mask: np.ndarray,
        sampled_x1_xy: np.ndarray,
        a2_target: np.ndarray,
        a2_valid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.resize_width is None or self.resize_height is None:
            return rgb, mask_bool, a1_target, a1_valid_mask, sampled_x1_xy, a2_target, a2_valid_mask

        src_h, src_w = int(mask_bool.shape[0]), int(mask_bool.shape[1])
        dst_h, dst_w = int(self.resize_height), int(self.resize_width)
        if src_h == dst_h and src_w == dst_w:
            return rgb, mask_bool, a1_target, a1_valid_mask, sampled_x1_xy, a2_target, a2_valid_mask

        image_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        a1_t = torch.from_numpy(a1_target.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        a1_valid_t = torch.from_numpy(a1_valid_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        a2_t = torch.from_numpy(a2_target.astype(np.float32)).unsqueeze(0)
        a2_valid_t = torch.from_numpy(a2_valid_mask.astype(np.float32)).unsqueeze(0)

        rgb_resized = (
            F.interpolate(image_t, size=(dst_h, dst_w), mode="bilinear", align_corners=False)
            .round()
            .clamp_(0.0, 255.0)
            .squeeze(0)
            .permute(1, 2, 0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        mask_resized = (
            F.interpolate(mask_t, size=(dst_h, dst_w), mode="nearest")
            .squeeze(0)
            .squeeze(0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        a1_resized = (
            F.interpolate(a1_t, size=(dst_h, dst_w), mode="bilinear", align_corners=False)
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        a1_valid_resized = (
            F.interpolate(a1_valid_t, size=(dst_h, dst_w), mode="nearest")
            .squeeze(0)
            .squeeze(0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        a2_resized = (
            F.interpolate(a2_t, size=(dst_h, dst_w), mode="bilinear", align_corners=False)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        a2_valid_resized = (
            F.interpolate(a2_valid_t, size=(dst_h, dst_w), mode="nearest")
            .squeeze(0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

        sampled_x1_xy_resized = np.asarray(sampled_x1_xy, dtype=np.float32).copy()
        valid = (sampled_x1_xy_resized[:, 0] >= 0.0) & (sampled_x1_xy_resized[:, 1] >= 0.0)
        sampled_x1_xy_resized[valid, 0] *= float(dst_w) / float(src_w)
        sampled_x1_xy_resized[valid, 1] *= float(dst_h) / float(src_h)
        return rgb_resized, mask_resized, a1_resized, a1_valid_resized, sampled_x1_xy_resized, a2_resized, a2_valid_resized

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if not row.get("shard_path"):
            raise ValueError(
                "PairPolicyDataset now requires shard-backed training data. "
                f"Missing shard_path for sample_id={row.get('sample_id')}"
            )
        shard = self._get_shard_handle(str(row["shard_path"]))
        shard_index = int(row["shard_index"])
        rgb = np.asarray(shard["image"][shard_index], dtype=np.uint8)
        mask_bool = np.asarray(shard["gt_mask"][shard_index], dtype=np.uint8)
        a1_value_map = np.asarray(shard["a1_value_map"][shard_index], dtype=np.float32)
        a1_valid_mask = np.asarray(shard["a1_valid_mask"][shard_index], dtype=np.uint8)
        sampled_x1_xy = np.asarray(shard["sampled_x1_xy"][shard_index], dtype=np.float32)
        a2_value_map = np.asarray(shard["a2_value_map"][shard_index], dtype=np.float32)
        a2_valid_mask = np.asarray(shard["a2_valid_mask"][shard_index], dtype=np.uint8)
        a2_target_valid = np.asarray(shard["a2_target_valid"][shard_index], dtype=np.bool_)

        rgb, mask_bool, a1_value_map, a1_valid_mask, sampled_x1_xy, a2_value_map, a2_valid_mask = self._resize_sample(
            rgb=rgb,
            mask_bool=mask_bool,
            a1_target=a1_value_map,
            a1_valid_mask=a1_valid_mask,
            sampled_x1_xy=sampled_x1_xy,
            a2_target=a2_value_map,
            a2_valid_mask=a2_valid_mask,
        )

        if self.train:
            rgb, mask_bool, a1_value_map, a1_valid_mask, sampled_x1_xy, a2_value_map, a2_valid_mask = perturb_geometry(
                rgb=rgb,
                mask=mask_bool,
                a1_value_map=a1_value_map,
                a1_valid_mask=a1_valid_mask,
                sampled_x1_xy=sampled_x1_xy,
                a2_value_map=a2_value_map,
                a2_valid_mask=a2_valid_mask,
                cfg=self.geom_aug_cfg,
                rng=self._rng,
            )
            sampled_x1_valid_after_geom = (sampled_x1_xy[:, 0] >= 0.0) & (sampled_x1_xy[:, 1] >= 0.0)
            a2_target_valid = np.asarray(a2_target_valid, dtype=np.bool_) & sampled_x1_valid_after_geom
            perturbed_mask, area_ratio = perturb_mask(mask_bool, self.mask_aug_cfg, self._rng)
            rgb = perturb_rgb(rgb, self.rgb_aug_cfg, self._rng).astype(np.uint8)
        else:
            perturbed_mask, area_ratio = mask_bool.copy(), 1.0

        if self.supervision_mask_mode == "input_mask":
            a1_supervision_mask = np.asarray(perturbed_mask, dtype=np.uint8)
            a2_supervision_mask = np.broadcast_to(
                np.asarray(perturbed_mask, dtype=np.uint8)[None, ...],
                a2_valid_mask.shape,
            ).copy()
        elif self.supervision_mask_mode == "heatmap_valid":
            a1_supervision_mask = np.asarray(a1_valid_mask, dtype=np.uint8)
            a2_supervision_mask = np.asarray(a2_valid_mask, dtype=np.uint8)
        else:
            raise ValueError(f"unsupported pair-policy supervision_mask_mode: {self.supervision_mask_mode}")

        a1_target_mask = (
            build_top_region_mask(value_map=a1_value_map, valid_mask=a1_supervision_mask, top_ratio=self.top_ratio)
            if self.target_name == "topk_masked_softmax"
            else np.asarray(a1_supervision_mask, dtype=np.uint8)
        )
        a1_negative_mask = (
            build_bottom_region_mask(value_map=a1_value_map, valid_mask=a1_supervision_mask, bottom_ratio=self.bottom_ratio)
            if self.target_name == "topk_masked_softmax"
            else np.zeros_like(a1_supervision_mask, dtype=np.uint8)
        )
        a1_target = build_training_target(
            value_map=a1_value_map,
            valid_mask=a1_target_mask,
            target_name=self.target_name,
            tau=self.a1_target_tau,
            top_ratio=self.top_ratio,
        )
        a2_target = np.zeros_like(a2_value_map, dtype=np.float32)
        a2_target_mask = np.zeros_like(a2_valid_mask, dtype=np.uint8)
        a2_negative_mask = np.zeros_like(a2_valid_mask, dtype=np.uint8)
        for k in range(int(a2_value_map.shape[0])):
            if bool(a2_target_valid[k]):
                a2_target_mask[k] = (
                    build_top_region_mask(
                        value_map=a2_value_map[k],
                        valid_mask=a2_supervision_mask[k],
                        top_ratio=self.top_ratio,
                    )
                    if self.target_name == "topk_masked_softmax"
                    else np.asarray(a2_supervision_mask[k], dtype=np.uint8)
                )
                a2_negative_mask[k] = (
                    build_bottom_region_mask(
                        value_map=a2_value_map[k],
                        valid_mask=a2_supervision_mask[k],
                        bottom_ratio=self.bottom_ratio,
                    )
                    if self.target_name == "topk_masked_softmax"
                    else np.zeros_like(a2_supervision_mask[k], dtype=np.uint8)
                )
                a2_target[k] = build_training_target(
                    value_map=a2_value_map[k],
                    valid_mask=a2_target_mask[k],
                    target_name=self.target_name,
                    tau=self.a2_target_tau,
                    top_ratio=self.top_ratio,
                )

        input_rgb = rgb.astype(np.float32) / 255.0
        if self.input_normalization == "imagenet":
            input_rgb = (input_rgb - _IMAGENET_MEAN[None, None, :]) / _IMAGENET_STD[None, None, :]
        elif self.input_normalization not in {"none", ""}:
            raise ValueError(f"unsupported pair-policy input_normalization: {self.input_normalization}")
        input_rgb = input_rgb * perturbed_mask[..., None].astype(np.float32)

        return {
            "image": torch.from_numpy(input_rgb.transpose(2, 0, 1)).float(),
            "input_mask": torch.from_numpy(perturbed_mask.astype(np.float32)),
            "gt_mask": torch.from_numpy(mask_bool.astype(np.float32)),
            "a1_target": torch.from_numpy(a1_target.astype(np.float32)),
            "a1_target_mask": torch.from_numpy(a1_target_mask.astype(np.float32)),
            "a1_negative_mask": torch.from_numpy(a1_negative_mask.astype(np.float32)),
            "a1_valid_mask": torch.from_numpy(a1_supervision_mask.astype(np.float32)),
            "sampled_x1_xy": torch.from_numpy(sampled_x1_xy.astype(np.float32)),
            "a2_target": torch.from_numpy(a2_target.astype(np.float32)),
            "a2_target_mask": torch.from_numpy(a2_target_mask.astype(np.float32)),
            "a2_negative_mask": torch.from_numpy(a2_negative_mask.astype(np.float32)),
            "a2_valid_mask": torch.from_numpy(a2_supervision_mask.astype(np.float32)),
            "a2_target_valid": torch.from_numpy(a2_target_valid.astype(np.bool_)),
            "sample_id": str(row["sample_id"]),
            "asset_id": str(row["asset_id"]),
            "mask_area_ratio": torch.tensor(float(area_ratio), dtype=torch.float32),
        }


def collate_pair_policy_batch(items: list[dict[str, Any]]) -> PairPolicyBatch:
    bsz = len(items)
    image = torch.stack([item["image"] for item in items], dim=0)
    input_mask = torch.stack([item["input_mask"] for item in items], dim=0)
    gt_mask = torch.stack([item["gt_mask"] for item in items], dim=0)
    a1_target = torch.stack([item["a1_target"] for item in items], dim=0)
    a1_target_mask = torch.stack([item["a1_target_mask"] for item in items], dim=0)
    a1_negative_mask = torch.stack([item["a1_negative_mask"] for item in items], dim=0)
    a1_valid_mask = torch.stack([item["a1_valid_mask"] for item in items], dim=0)
    mask_area_ratio = torch.stack([item["mask_area_ratio"] for item in items], dim=0)
    num_x1 = int(items[0]["sampled_x1_xy"].shape[0])
    sampled_x1_xy = torch.stack([item["sampled_x1_xy"] for item in items], dim=0)
    sampled_x1_valid = (sampled_x1_xy[..., 0] >= 0) & (sampled_x1_xy[..., 1] >= 0)
    a2_target = torch.stack([item["a2_target"] for item in items], dim=0)
    a2_target_mask = torch.stack([item["a2_target_mask"] for item in items], dim=0)
    a2_negative_mask = torch.stack([item["a2_negative_mask"] for item in items], dim=0)
    a2_valid_mask = torch.stack([item["a2_valid_mask"] for item in items], dim=0)
    a2_target_valid = torch.stack([item["a2_target_valid"] for item in items], dim=0)

    return PairPolicyBatch(
        image=image,
        input_mask=input_mask,
        gt_mask=gt_mask,
        a1_target=a1_target,
        a1_target_mask=a1_target_mask,
        a1_negative_mask=a1_negative_mask,
        a1_valid_mask=a1_valid_mask,
        sampled_x1_xy=sampled_x1_xy,
        sampled_x1_valid=sampled_x1_valid,
        a2_target=a2_target,
        a2_target_mask=a2_target_mask,
        a2_negative_mask=a2_negative_mask,
        a2_valid_mask=a2_valid_mask,
        a2_target_valid=a2_target_valid,
        sample_id=[item["sample_id"] for item in items],
        asset_id=[item["asset_id"] for item in items],
        mask_area_ratio=mask_area_ratio,
    )
