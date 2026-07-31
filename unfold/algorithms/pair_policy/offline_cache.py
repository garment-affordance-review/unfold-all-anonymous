from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from unfold.algorithms.supervision.projection import (
    build_projection_valid_mask,
    build_dense_a1_heatmap,
    build_dense_a2_heatmap_for_pixel,
    fill_invalid_with_margin,
    masked_softmax_heatmap,
)


def sample_x1_pixels(
    *,
    a1_distribution: np.ndarray,
    valid_mask: np.ndarray,
    num_x1_samples: int,
    tau: float,
    train: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    valid_yx = np.argwhere(np.asarray(valid_mask) > 0)
    if valid_yx.size == 0:
        return np.full((int(num_x1_samples), 2), -1.0, dtype=np.float32)

    weights = np.asarray(a1_distribution[np.asarray(valid_mask) > 0], dtype=np.float32)
    if bool(train):
        probs = np.clip(weights, a_min=0.0, a_max=None)
        probs_sum = float(probs.sum())
        if probs_sum <= 0.0:
            probs = np.full((valid_yx.shape[0],), 1.0 / float(valid_yx.shape[0]), dtype=np.float32)
        else:
            probs = probs / probs_sum
        take = min(int(num_x1_samples), int(valid_yx.shape[0]))
        chosen = rng.choice(valid_yx.shape[0], size=take, replace=False, p=probs)
    else:
        order = np.argsort(weights)[::-1]
        chosen = order[: min(int(num_x1_samples), int(order.shape[0]))]

    out = np.full((int(num_x1_samples), 2), -1.0, dtype=np.float32)
    for dst, src in enumerate(np.asarray(chosen).tolist()):
        y, x = valid_yx[int(src)]
        out[dst] = np.array([float(x), float(y)], dtype=np.float32)
    return out


def top_ratio_mask(
    *,
    values: np.ndarray,
    valid_mask: np.ndarray,
    top_ratio: float,
) -> np.ndarray:
    valid = np.asarray(valid_mask) > 0
    out = np.zeros_like(valid, dtype=np.uint8)
    if not np.any(valid):
        return out
    ratio = float(top_ratio)
    if ratio >= 1.0:
        out[valid] = 1
        return out
    vals = np.asarray(values, dtype=np.float32)[valid]
    k = max(1, int(np.ceil(vals.shape[0] * max(ratio, 0.0))))
    thresh = float(np.partition(vals, -k)[-k])
    out[(np.asarray(values, dtype=np.float32) >= thresh) & valid] = 1
    return out


def _stable_row_seed(*, sample_id: str, asset_id: str, base_seed: int) -> int:
    payload = f"{asset_id}:{sample_id}:{int(base_seed)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def stable_row_seed(*, sample_id: str, asset_id: str, base_seed: int) -> int:
    return _stable_row_seed(sample_id=sample_id, asset_id=asset_id, base_seed=base_seed)


def _resize_cache_sample(
    *,
    rgb: np.ndarray,
    mask_bool: np.ndarray,
    a1_target: np.ndarray,
    a1_valid_mask: np.ndarray,
    sampled_x1_xy: np.ndarray,
    a2_target: np.ndarray,
    a2_valid_mask: np.ndarray,
    resize_width: int | None,
    resize_height: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if resize_width is None or resize_height is None:
        return rgb, mask_bool, a1_target, a1_valid_mask, sampled_x1_xy, a2_target, a2_valid_mask

    src_h, src_w = int(mask_bool.shape[0]), int(mask_bool.shape[1])
    dst_h, dst_w = int(resize_height), int(resize_width)
    if src_h == dst_h and src_w == dst_w:
        return rgb, mask_bool, a1_target, a1_valid_mask, sampled_x1_xy, a2_target, a2_valid_mask

    image = Image.fromarray(rgb, mode="RGB").resize((dst_w, dst_h), resample=Image.BILINEAR)
    mask = Image.fromarray(mask_bool.astype(np.uint8), mode="L").resize((dst_w, dst_h), resample=Image.NEAREST)

    def _resize_heatmap(arr: np.ndarray) -> np.ndarray:
        return np.asarray(
            Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F").resize((dst_w, dst_h), resample=Image.BILINEAR),
            dtype=np.float32,
        )

    def _resize_mask(arr: np.ndarray) -> np.ndarray:
        return np.asarray(
            Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L").resize((dst_w, dst_h), resample=Image.NEAREST),
            dtype=np.uint8,
        )

    a2_target_resized = np.stack([_resize_heatmap(a2_target[k]) for k in range(int(a2_target.shape[0]))], axis=0)
    a2_valid_resized = np.stack([_resize_mask(a2_valid_mask[k]) for k in range(int(a2_valid_mask.shape[0]))], axis=0)
    sampled_x1_xy = np.asarray(sampled_x1_xy, dtype=np.float32).copy()
    valid = (sampled_x1_xy[:, 0] >= 0.0) & (sampled_x1_xy[:, 1] >= 0.0)
    sampled_x1_xy[valid, 0] *= float(dst_w) / float(src_w)
    sampled_x1_xy[valid, 1] *= float(dst_h) / float(src_h)
    return (
        np.asarray(image, dtype=np.uint8),
        np.asarray(mask, dtype=np.uint8),
        _resize_heatmap(a1_target),
        _resize_mask(a1_valid_mask),
        sampled_x1_xy,
        a2_target_resized,
        a2_valid_resized,
    )


def build_pair_policy_cache_sample(
    *,
    sample_id: str,
    asset_id: str,
    rgb: np.ndarray,
    mask_bool: np.ndarray,
    face_index: np.ndarray,
    face_vertex_ids: np.ndarray,
    barycentric_weights: np.ndarray,
    candidate_raw_vid: np.ndarray,
    a1_logits: np.ndarray,
    reward_matrix: np.ndarray,
    num_x1_samples: int,
    a1_tau: float,
    a1_top_ratio: float = 1.0,
    target_type: str = "masked_softmax",
    train: bool,
    seed: int,
    resize_width: int | None = None,
    resize_height: int | None = None,
) -> dict[str, np.ndarray]:
    if str(target_type) != "masked_softmax":
        raise ValueError(f"unsupported pair-policy target_type: {target_type}")

    a1_value_map = build_dense_a1_heatmap(
        mask_np=mask_bool,
        face_index=face_index,
        face_vertex_ids=face_vertex_ids,
        barycentric_weights=barycentric_weights,
        candidate_raw_vid=candidate_raw_vid,
        a1_logits=a1_logits,
    ).astype(np.float32, copy=False)
    a1_valid_mask = build_projection_valid_mask(
        mask_np=mask_bool,
        face_index=face_index,
        face_vertex_ids=face_vertex_ids,
        barycentric_weights=barycentric_weights,
        candidate_raw_vid=candidate_raw_vid,
    ).astype(np.uint8, copy=False)
    a1_value_map = fill_invalid_with_margin(a1_value_map, a1_valid_mask).astype(np.float32, copy=False)
    a1_sampling_mask = top_ratio_mask(
        values=a1_value_map,
        valid_mask=a1_valid_mask,
        top_ratio=float(a1_top_ratio),
    ).astype(np.uint8, copy=False)
    a1_sampling_dist = masked_softmax_heatmap(
        a1_value_map,
        a1_sampling_mask,
        tau=float(a1_tau),
    ).astype(np.float16, copy=False)

    row_seed = _stable_row_seed(sample_id=str(sample_id), asset_id=str(asset_id), base_seed=int(seed))
    sampled_x1_xy = sample_x1_pixels(
        a1_distribution=np.asarray(a1_sampling_dist, dtype=np.float32),
        valid_mask=a1_sampling_mask,
        num_x1_samples=int(num_x1_samples),
        tau=float(a1_tau),
        train=bool(train),
        rng=np.random.default_rng(row_seed),
    )

    a2_value_map = np.zeros((int(num_x1_samples), *mask_bool.shape), dtype=np.float16)
    a2_target_valid = np.zeros((int(num_x1_samples),), dtype=np.bool_)
    a2_valid_mask = np.zeros((int(num_x1_samples), *mask_bool.shape), dtype=np.uint8)
    for k, xy in enumerate(sampled_x1_xy):
        x = int(round(float(xy[0])))
        y = int(round(float(xy[1])))
        if x < 0 or y < 0:
            continue
        heat, _, valid = build_dense_a2_heatmap_for_pixel(
            x=x,
            y=y,
            mask_np=mask_bool,
            face_index=face_index,
            face_vertex_ids=face_vertex_ids,
            barycentric_weights=barycentric_weights,
            candidate_raw_vid=candidate_raw_vid,
            reward_matrix=reward_matrix,
        )
        a2_target_valid[k] = bool(valid)
        if bool(valid):
            a2_valid_mask[k] = a1_valid_mask
            a2_value_map[k] = np.asarray(
                fill_invalid_with_margin(np.asarray(heat, dtype=np.float32), a2_valid_mask[k]),
                dtype=np.float16,
            )

    rgb, mask_bool, a1_value_map, a1_valid_mask, sampled_x1_xy, a2_value_map, a2_valid_mask = _resize_cache_sample(
        rgb=np.asarray(rgb, dtype=np.uint8),
        mask_bool=np.asarray(mask_bool, dtype=np.uint8),
        a1_target=np.asarray(a1_value_map, dtype=np.float32),
        a1_valid_mask=np.asarray(a1_valid_mask, dtype=np.uint8),
        sampled_x1_xy=np.asarray(sampled_x1_xy, dtype=np.float32),
        a2_target=np.asarray(a2_value_map, dtype=np.float32),
        a2_valid_mask=np.asarray(a2_valid_mask, dtype=np.uint8),
        resize_width=resize_width,
        resize_height=resize_height,
    )
    a1_value_map = fill_invalid_with_margin(np.asarray(a1_value_map, dtype=np.float32), a1_valid_mask).astype(np.float32, copy=False)
    for k in range(int(a2_value_map.shape[0])):
        if bool(a2_target_valid[k]):
            a2_value_map[k] = fill_invalid_with_margin(
                np.asarray(a2_value_map[k], dtype=np.float32),
                a2_valid_mask[k],
            ).astype(np.float32, copy=False)
    return {
        "image": np.asarray(rgb, dtype=np.uint8),
        "gt_mask": np.asarray(mask_bool, dtype=np.uint8),
        "a1_value_map": np.asarray(a1_value_map, dtype=np.float16),
        "a1_valid_mask": np.asarray(a1_valid_mask, dtype=np.uint8),
        "sampled_x1_xy": np.asarray(sampled_x1_xy, dtype=np.float32),
        "a2_value_map": np.asarray(a2_value_map, dtype=np.float16),
        "a2_valid_mask": np.asarray(a2_valid_mask, dtype=np.uint8),
        "a2_target_valid": np.asarray(a2_target_valid, dtype=np.bool_),
    }
def _write_hdf5_shard(
    *,
    shard_path: Path,
    rows: list[dict[str, Any]],
    samples: list[dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = len(samples)
    if sample_count == 0:
        return []

    image_shape = samples[0]["image"].shape
    a1_shape = samples[0]["a1_value_map"].shape
    sampled_x1_shape = samples[0]["sampled_x1_xy"].shape
    a2_shape = samples[0]["a2_value_map"].shape

    with tempfile.NamedTemporaryFile(dir=str(shard_path.parent), prefix=shard_path.stem + ".", suffix=".h5.tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("image", shape=(sample_count, *image_shape), dtype=np.uint8)
            f.create_dataset("gt_mask", shape=(sample_count, *a1_shape), dtype=np.uint8)
            f.create_dataset("a1_value_map", shape=(sample_count, *a1_shape), dtype=np.float16)
            f.create_dataset("a1_valid_mask", shape=(sample_count, *a1_shape), dtype=np.uint8)
            f.create_dataset("sampled_x1_xy", shape=(sample_count, *sampled_x1_shape), dtype=np.float32)
            f.create_dataset("a2_value_map", shape=(sample_count, *a2_shape), dtype=np.float16)
            f.create_dataset("a2_valid_mask", shape=(sample_count, *a2_shape), dtype=np.uint8)
            f.create_dataset("a2_target_valid", shape=(sample_count, sampled_x1_shape[0]), dtype=np.bool_)
            for idx, sample in enumerate(samples):
                f["image"][idx] = sample["image"]
                f["gt_mask"][idx] = sample["gt_mask"]
                f["a1_value_map"][idx] = sample["a1_value_map"]
                f["a1_valid_mask"][idx] = sample["a1_valid_mask"]
                f["sampled_x1_xy"][idx] = sample["sampled_x1_xy"]
                f["a2_value_map"][idx] = sample["a2_value_map"]
                f["a2_valid_mask"][idx] = sample["a2_valid_mask"]
                f["a2_target_valid"][idx] = sample["a2_target_valid"]
        os.replace(tmp_path, shard_path)
        os.chmod(shard_path, 0o664)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    shard_rows = []
    for idx, row in enumerate(rows):
        shard_row = dict(row)
        shard_row["shard_path"] = str(shard_path)
        shard_row["shard_index"] = int(idx)
        shard_rows.append(shard_row)
    return shard_rows


def write_pair_policy_hdf5_shard(
    *,
    shard_path: Path,
    rows: list[dict[str, Any]],
    samples: list[dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    return _write_hdf5_shard(shard_path=shard_path, rows=rows, samples=samples)


def build_offline_pair_policy_cache(
    *,
    index_payload: dict[str, Any],
    out_dir: str | Path,
    out_index_path: str | Path,
    num_x1_samples: int,
    a1_tau: float,
    seed: int = 42,
    rebuild: bool = False,
    progress_every: int = 100,
    num_workers: int = 1,
    resize_width: int | None = None,
    resize_height: int | None = None,
    shard_size: int = 512,
) -> dict[str, Any]:
    raise RuntimeError(
        "build_offline_pair_policy_cache has been removed. "
        "Generate direct shard-backed training data from render_supervision instead."
    )
