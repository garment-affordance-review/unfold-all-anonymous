from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Optional

import numpy as np
import torch
import yaml
import h5py
from PIL import Image

from unfold.algorithms.pair_policy.offline_cache import (
    build_pair_policy_cache_sample,
    write_pair_policy_hdf5_shard,
)
from unfold.algorithms.pair_policy.index import _asset_ordered_split
from unfold.algorithms.supervision.targets import (
    build_a1_from_reward_matrix,
    build_reward_matrix,
    build_a1_from_reward_matrix_torch,
    build_reward_matrix_torch,
    symmetrize_reward_matrix_torch,
)
from unfold.algorithms.supervision.projection import compute_reward_row_margin, compute_top1_margin
from unfold.algorithms.supervision.teacher_pointcept import TeacherRewardInfer
from unfold.algorithms.supervision.visualize import save_supervision_visuals


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.full_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _estimate_sample_bytes(sample: dict[str, np.ndarray]) -> int:
    total = 0
    for value in sample.values():
        total += int(np.asarray(value).nbytes)
    return int(total)


def _resolve_shard_size(
    *,
    requested_shard_size: int,
    target_shard_bytes: int,
    sample_bytes: int,
) -> int:
    if int(requested_shard_size) > 0:
        return int(requested_shard_size)
    if int(sample_bytes) <= 0:
        raise ValueError("sample_bytes must be positive when shard_size is auto")
    if int(target_shard_bytes) <= 0:
        raise ValueError("target_shard_bytes must be positive when shard_size is auto")
    return max(1, int(math.floor(float(target_shard_bytes) / float(sample_bytes))))


def _normalize_rel_path(p: str) -> str:
    s = str(Path(p)).replace("\\", "/")
    for marker in ("/cloth/", "/assets/"):
        if marker in s:
            s = s.split(marker, 1)[1]
    if s.startswith("cloth/"):
        s = s[len("cloth/"):]
    if s.startswith("./"):
        return s[2:]
    return s


def _usd_match_keys(usd_rel: str) -> list[str]:
    norm = _normalize_rel_path(usd_rel)
    keys = [norm]
    parts = Path(norm).parts
    for start in range(len(parts)):
        suffix = Path(*parts[start:]).as_posix()
        if suffix not in keys:
            keys.append(suffix)
    m = re.search(r"(Dress|Tops|Trousers)/.*", norm)
    if m:
        key = m.group(0)
        if key not in keys:
            keys.append(key)
    return keys


def _load_pointcept_manifest(pointcept_manifest: Path) -> list[dict[str, Any]]:
    if pointcept_manifest.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with pointcept_manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    obj = json.loads(pointcept_manifest.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        assets = obj.get("assets", [])
    elif isinstance(obj, list):
        assets = obj
    else:
        raise ValueError(f"Unsupported Pointcept manifest format: {pointcept_manifest}")
    if not isinstance(assets, list):
        raise ValueError(f"Pointcept manifest assets is not a list: {pointcept_manifest}")
    return assets


def _resolve_pointcept_asset_by_usd(assets: list[dict[str, Any]], usd_rel: str) -> dict[str, Any]:
    usd_keys = _usd_match_keys(usd_rel)
    manifest_pairs = []
    for a in assets:
        usd_value = a.get("usd")
        if not usd_value:
            usd_value = a.get("asset_path", "")
        manifest_pairs.append((a, _normalize_rel_path(str(usd_value))))

    for key in usd_keys:
        exact = [a for a, manifest_usd in manifest_pairs if manifest_usd == key]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise ValueError(f"USD exact key matched multiple Pointcept assets: usd={usd_rel} key={key} matches={len(exact)}")

    for key in usd_keys:
        suffix = [a for a, manifest_usd in manifest_pairs if manifest_usd.endswith(key) or key.endswith(manifest_usd)]
        if len(suffix) == 1:
            return suffix[0]
        if len(suffix) > 1:
            raise ValueError(f"USD suffix matched multiple Pointcept assets: usd={usd_rel} key={key} matches={len(suffix)}")

    raise ValueError(f"USD not found in Pointcept manifest: usd={usd_rel} keys={usd_keys}")


def _load_vertex_index_map(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f"vertex index map must be 1-D, got {arr.shape} from {path}")
    return arr.astype(np.int64, copy=False)


def _resolve_index_map_path(
    *,
    vertex_index_map_path: Optional[Path],
    pointcept_asset: dict[str, Any],
    asset_dir: Path,
    mapping_file_key: str,
) -> Path:
    if vertex_index_map_path is not None:
        return vertex_index_map_path
    candidates: list[str] = []
    rel = pointcept_asset.get(mapping_file_key)
    if rel is not None:
        candidates.append(str(rel))
    candidates.extend(["raw2coord.npy", "raw_to_down.npy"])
    for rel_path in candidates:
        p = asset_dir / rel_path
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Mapping file missing for asset_id={pointcept_asset.get('asset_id')}: key={mapping_file_key} candidates={candidates}"
    )


def _sorted_unique_with_first(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, first_idx = np.unique(values, return_index=True)
    order = np.argsort(first_idx)
    return uniq[order], first_idx[order]


def _candidate_cap_indices(num_items: int, max_items: int) -> np.ndarray:
    if max_items <= 0 or num_items <= max_items:
        return np.arange(num_items, dtype=np.int64)
    return np.linspace(0, num_items - 1, num=max_items, dtype=np.int64)


def _ordered_pair_count(num_candidates: int) -> int:
    if num_candidates <= 1:
        return 0
    return int(num_candidates) * int(num_candidates - 1)


def _sample_ordered_pair_indices(num_pairs_full: int, pair_budget: int) -> np.ndarray | None:
    if pair_budget <= 0 or num_pairs_full <= pair_budget:
        return None
    rng = np.random.default_rng(0)
    keep = rng.choice(num_pairs_full, size=pair_budget, replace=False)
    return np.sort(keep.astype(np.int64, copy=False))


def _ordered_pairs_from_flat_indices(num_candidates: int, flat_indices: np.ndarray) -> np.ndarray:
    if flat_indices.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs_per_row = num_candidates - 1
    ii = flat_indices // pairs_per_row
    jj = flat_indices % pairs_per_row
    jj += (jj >= ii).astype(np.int64)
    return np.stack([ii, jj], axis=1).astype(np.int64, copy=False)


def _iter_ordered_pair_blocks(
    *,
    num_candidates: int,
    pair_indices: np.ndarray | None,
    block_size: int,
):
    num_pairs_full = _ordered_pair_count(num_candidates)
    if num_pairs_full == 0:
        return
    if pair_indices is None:
        for start in range(0, num_pairs_full, block_size):
            stop = min(start + block_size, num_pairs_full)
            flat = np.arange(start, stop, dtype=np.int64)
            yield _ordered_pairs_from_flat_indices(num_candidates, flat)
        return

    for start in range(0, pair_indices.shape[0], block_size):
        yield _ordered_pairs_from_flat_indices(num_candidates, pair_indices[start : start + block_size])


def _resolve_asset_path(
    row: dict[str, Any],
    data_root: Path,
) -> str:
    for key in ("usd", "usd_rel", "asset_usd", "asset_path", "usd_path"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return _normalize_rel_path(v)

    meta_rel = row.get("paths", {}).get("meta")
    if meta_rel:
        meta_path = data_root / meta_rel
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for key in ("usd", "usd_rel", "asset_usd", "asset_path", "usd_path"):
                v = meta.get(key)
                if isinstance(v, str) and v:
                    return _normalize_rel_path(v)

    raise ValueError(f"sample_id={row.get('id')} missing asset/usd path in manifest or meta.json")


def _iter_selected_rows(
    rows: list[dict[str, Any]],
    *,
    sample_id: Optional[str],
    all_samples: bool,
    max_samples: Optional[int],
) -> list[dict[str, Any]]:
    if sample_id is not None:
        sample_key = str(sample_id)
        selected = [
            r for r in rows
            if (
                str(r.get("id")) == sample_key
                or str(r.get("global_sample_id", "")) == sample_key
                or f"{r.get('asset_id')}:{r.get('id')}" == sample_key
            )
        ]
        if not selected:
            raise ValueError(f"sample_id={sample_id} not found in manifest")
        if len(selected) > 1:
            raise ValueError(
                f"sample_id={sample_id} matched multiple rows; use global_sample_id or asset_id:sample_id"
            )
        return selected[:1]

    selected = rows if all_samples else rows[:1]
    if max_samples is not None:
        selected = selected[: max_samples]
    return selected


def _candidate_xy_from_anchor_map(
    candidate_ids: np.ndarray,
    anchor_xy: dict[int, tuple[float, float]],
) -> np.ndarray:
    xy = np.zeros((candidate_ids.shape[0], 2), dtype=np.float32)
    for i, cid in enumerate(candidate_ids.tolist()):
        pt = anchor_xy.get(int(cid))
        if pt is not None:
            xy[i] = np.asarray(pt, dtype=np.float32)
    return xy


class _OfflineTeacherPolicy:
    def __init__(
        self,
        *,
        teacher: TeacherRewardInfer,
        pointcept_assets: list[dict[str, Any]],
        pointcept_data_root: Path,
        vertex_index_map_path: Optional[Path],
        raw_to_teacher_map_key: str,
        pair_chunk_size: int,
        num_candidates: int,
        teacher_min_candidate_dist: float,
    ):
        self.teacher = teacher
        self.pointcept_assets = pointcept_assets
        self.pointcept_data_root = pointcept_data_root
        self.vertex_index_map_path = vertex_index_map_path
        self.raw_to_teacher_map_key = raw_to_teacher_map_key
        self.pair_chunk_size = int(pair_chunk_size)
        self.num_candidates = int(num_candidates)
        self.teacher_min_candidate_dist = float(teacher_min_candidate_dist)
        self._asset_cache: dict[str, dict[str, Any]] = {}

    def resolve_asset_bundle(self, asset_path: str) -> dict[str, Any]:
        cache_key = str(asset_path)
        if cache_key in self._asset_cache:
            return self._asset_cache[cache_key]

        pointcept_asset = _resolve_pointcept_asset_by_usd(self.pointcept_assets, usd_rel=asset_path)
        asset_dir_name = str(pointcept_asset.get("asset_dir", f"asset_{int(pointcept_asset['asset_id']):04d}"))
        asset_dir = self.pointcept_data_root / "assets" / asset_dir_name
        coord_path = asset_dir / "coord.npy"
        normal_path = asset_dir / "normal.npy"
        map_path = _resolve_index_map_path(
            vertex_index_map_path=self.vertex_index_map_path,
            pointcept_asset=pointcept_asset,
            asset_dir=asset_dir,
            mapping_file_key=self.raw_to_teacher_map_key,
        )
        if not coord_path.exists():
            raise FileNotFoundError(f"Pointcept coord.npy not found: {coord_path}")
        bundle = {
            "asset_id": asset_dir_name,
            "coord_path": str(coord_path),
            "normal_path": str(normal_path) if normal_path.exists() else None,
            "index_map": _load_vertex_index_map(map_path),
            "map_path": str(map_path),
        }
        self._asset_cache[cache_key] = bundle
        return bundle

    def encode_teacher_asset(self, bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None, Any, Any]:
        teacher_coord = np.load(bundle["coord_path"]).astype(np.float32)
        teacher_normal = np.load(bundle["normal_path"]).astype(np.float32) if bundle["normal_path"] else None
        feat, point_offset = self.teacher.encode_points(coord=teacher_coord, normal=teacher_normal)
        return teacher_coord, teacher_normal, feat, point_offset

    def build_candidates(
        self,
        *,
        face_index: np.ndarray,
        face_vertex_ids: np.ndarray,
        barycentric_weights: np.ndarray,
        mask_np: np.ndarray,
        index_map: np.ndarray,
        teacher_nv: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[float, float]]]:
        mask_bool = np.asarray(mask_np).astype(bool)
        pix_valid = (face_index >= 0) & mask_bool
        raw_tri = face_vertex_ids[pix_valid]
        raw_valid = (raw_tri >= 0) & (raw_tri < index_map.shape[0])
        teacher_tri = np.full(raw_tri.shape, -1, dtype=np.int64)
        teacher_tri[raw_valid] = index_map[raw_tri[raw_valid]]
        visible_raw_vid = np.unique(raw_tri[raw_valid]).astype(np.int64) if raw_tri.size > 0 else np.zeros((0,), dtype=np.int64)
        teacher_valid = (teacher_tri >= 0) & (teacher_tri < teacher_nv)
        if not np.any(teacher_valid):
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), visible_raw_vid, {}

        # Use visible raw render vertices as the dense support set, then map them
        # to teacher vertices only for reward inference. This keeps the raster
        # support aligned with face_vertex_ids during barycentric projection.
        yx = np.argwhere(pix_valid)
        yx_rep = np.repeat(yx, repeats=face_vertex_ids.shape[-1], axis=0)
        raw_flat = raw_tri.reshape(-1)
        teacher_flat = teacher_tri.reshape(-1)
        valid_flat = teacher_valid.reshape(-1)

        raw_flat = raw_flat[valid_flat].astype(np.int64, copy=False)
        teacher_flat = teacher_flat[valid_flat].astype(np.int64, copy=False)
        yx_rep = yx_rep[valid_flat]

        raw_vid, first_idx = _sorted_unique_with_first(raw_flat)
        rep_yx = yx_rep[first_idx]
        rep_teacher = teacher_flat[first_idx]
        keep = _candidate_cap_indices(raw_vid.shape[0], self.num_candidates)
        raw_vid = raw_vid[keep]
        rep_teacher = rep_teacher[keep]
        rep_yx = rep_yx[keep]
        anchor_xy = {
            int(rvid): (float(rep_yx[i, 1]), float(rep_yx[i, 0]))
            for i, rvid in enumerate(raw_vid.tolist())
        }
        return rep_teacher.astype(np.int64), raw_vid.astype(np.int64), visible_raw_vid, anchor_xy

    def downsample_candidates_by_teacher_xy(
        self,
        *,
        teacher_coord_xy: np.ndarray,
        candidate_teacher_vid: np.ndarray,
        candidate_raw_vid: np.ndarray,
        anchor_xy: dict[int, tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[float, float]]]:
        min_dist = float(self.teacher_min_candidate_dist)
        if min_dist <= 0.0 or candidate_teacher_vid.size <= 2:
            return candidate_teacher_vid, candidate_raw_vid, anchor_xy

        coord_xy = teacher_coord_xy[candidate_teacher_vid].astype(np.float32, copy=False)
        n = int(coord_xy.shape[0])
        if n <= 2:
            return candidate_teacher_vid, candidate_raw_vid, anchor_xy

        center = np.mean(coord_xy, axis=0, keepdims=True)
        start = int(np.argmax(np.linalg.norm(coord_xy - center, axis=1)))
        order: list[int] = [start]
        min_sq = np.sum((coord_xy - coord_xy[start]) ** 2, axis=1)
        chosen = np.zeros((n,), dtype=bool)
        chosen[start] = True
        for _ in range(1, n):
            min_sq[chosen] = -1.0
            nxt = int(np.argmax(min_sq))
            if min_sq[nxt] < 0:
                break
            order.append(nxt)
            chosen[nxt] = True
            d_sq = np.sum((coord_xy - coord_xy[nxt]) ** 2, axis=1)
            min_sq = np.minimum(min_sq, d_sq)

        keep: list[int] = []
        min_dist_sq = float(min_dist * min_dist)
        for idx in order:
            if not keep:
                keep.append(idx)
                continue
            d_sq = np.sum((coord_xy[keep] - coord_xy[idx]) ** 2, axis=1)
            if float(np.min(d_sq)) >= min_dist_sq:
                keep.append(idx)

        if len(keep) < 2:
            keep = order[: min(2, n)]
        keep_idx = np.array(keep, dtype=np.int64)
        if self.num_candidates > 0 and keep_idx.size > self.num_candidates:
            keep_idx = keep_idx[: self.num_candidates]

        new_teacher_vid = candidate_teacher_vid[keep_idx]
        new_raw_vid = candidate_raw_vid[keep_idx]
        new_anchor_xy = {int(t): anchor_xy[int(t)] for t in new_teacher_vid.tolist() if int(t) in anchor_xy}
        return new_teacher_vid, new_raw_vid, new_anchor_xy


def _group_rows_by_asset(
    *,
    rows: list[dict[str, Any]],
    data_root: Path,
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        asset_path = _resolve_asset_path(row=row, data_root=data_root)
        if asset_path not in grouped:
            order.append(asset_path)
        row_copy = dict(row)
        row_copy["_resolved_asset_path"] = asset_path
        grouped[asset_path].append(row_copy)
    return [(asset_path, grouped[asset_path]) for asset_path in order]


def _existing_shard_sample_counts(shards_dir: Path) -> list[int]:
    counts: list[int] = []
    shard_id = 0
    while True:
        shard_path = shards_dir / f"shard_{shard_id:05d}.h5"
        if not shard_path.exists():
            break
        with h5py.File(shard_path, "r") as f:
            if "image" not in f:
                raise ValueError(f"existing shard missing image dataset: {shard_path}")
            counts.append(int(f["image"].shape[0]))
        shard_id += 1

    extra = sorted(shards_dir.glob("shard_*.h5"))
    if len(extra) != len(counts):
        expected = {shards_dir / f"shard_{idx:05d}.h5" for idx in range(len(counts))}
        unexpected = [str(p) for p in extra if p not in expected]
        raise ValueError(f"existing shards are not contiguous from shard_00000.h5: {unexpected[:5]}")
    return counts


def _resume_meta_from_row(
    *,
    row: dict[str, Any],
    data_root: Path,
    asset_path: str,
    bundle: dict[str, Any],
    sample_split: str,
    shard_path: Path,
    shard_index: int,
    shard_id: int,
    final_dataset_root: Path,
    pair_budget: int,
) -> dict[str, Any]:
    sid = str(row["id"])
    paths = row["paths"]
    render_asset_dir = str(row.get("asset_dir", row.get("asset_id", "unknown_asset")))
    meta = {
        "sample_id": sid,
        "global_sample_id": row.get("global_sample_id"),
        "render_asset_id": row.get("asset_id"),
        "render_asset_dir": render_asset_dir,
        "asset_path": asset_path,
        "teacher_asset_id": bundle["asset_id"],
        "split": str(sample_split) if sample_split else None,
        "vertex_index_map": bundle["map_path"],
        "num_vertices_render": -1,
        "num_vertices_teacher": -1,
        "num_visible_raw": -1,
        "num_candidates": -1,
        "num_pairs_full": -1,
        "num_pairs_used": -1,
        "pair_budget": int(pair_budget),
        "pair_nan_count": 0,
        "pair_inf_count": 0,
        "a1_std": 0.0,
        "a1_top1_margin": 0.0,
        "reward_row_top1_margin": 0.0,
        "schema_version": 2,
        "vis_files": [],
        "source": {
            "rgb": str(data_root / paths["rgb"]),
            "mask": str(data_root / paths["mask"]),
            "vertices": str(data_root / paths["vertices"]),
            "face_index": str(data_root / paths["face_index"]),
            "face_vertex_ids": str(data_root / paths["face_vertex_ids"]),
            "barycentric_weights": str(data_root / paths["barycentric_weights"]),
        },
        "shard_path": str(shard_path),
        "shard_index": int(shard_index),
        "shard_id": int(shard_id),
        "train_cache_root": str(final_dataset_root),
        "resumed_without_shard_metadata": True,
    }
    return meta


def _prepare_sample_query(
    *,
    row: dict[str, Any],
    data_root: Path,
    policy: _OfflineTeacherPolicy,
    pair_budget: int,
    teacher_coord_xy: np.ndarray,
) -> dict[str, Any]:
    sid = str(row["id"])
    paths = row["paths"]
    render_asset_dir = str(row.get("asset_dir", row.get("asset_id", "unknown_asset")))

    rgb_path = data_root / paths["rgb"]
    mask_path = data_root / paths["mask"]
    vertices_path = data_root / paths["vertices"]
    face_index_path = data_root / paths["face_index"]
    face_vertex_ids_path = data_root / paths["face_vertex_ids"]
    barycentric_weights_path = data_root / paths["barycentric_weights"]

    asset_path = str(row.get("_resolved_asset_path") or _resolve_asset_path(row=row, data_root=data_root))
    bundle = policy.resolve_asset_bundle(asset_path)

    mask = np.array(Image.open(mask_path))
    face_index = np.load(face_index_path).astype(np.int64)
    face_vertex_ids = np.load(face_vertex_ids_path).astype(np.int64)
    barycentric_weights = np.load(barycentric_weights_path).astype(np.float32)
    vertices_render = np.load(vertices_path).astype(np.float32)

    teacher_nv = int(np.load(bundle["coord_path"], mmap_mode="r").shape[0])
    candidate_teacher_vid, candidate_raw_vid, visible_raw_vid, anchor_xy = policy.build_candidates(
        face_index=face_index,
        face_vertex_ids=face_vertex_ids,
        barycentric_weights=barycentric_weights,
        mask_np=(mask > 0),
        index_map=bundle["index_map"],
        teacher_nv=teacher_nv,
    )
    candidate_teacher_vid, candidate_raw_vid, anchor_xy = policy.downsample_candidates_by_teacher_xy(
        teacher_coord_xy=teacher_coord_xy,
        candidate_teacher_vid=candidate_teacher_vid,
        candidate_raw_vid=candidate_raw_vid,
        anchor_xy=anchor_xy,
    )
    if candidate_teacher_vid.size == 0:
        raise RuntimeError(f"no_visible_teacher_vertices: sample_id={sid} asset={asset_path}")

    candidate_xy = _candidate_xy_from_anchor_map(candidate_raw_vid, anchor_xy)
    num_pairs_full = _ordered_pair_count(int(candidate_teacher_vid.shape[0]))
    pair_indices = _sample_ordered_pair_indices(num_pairs_full=num_pairs_full, pair_budget=pair_budget)
    num_pairs_used = int(pair_indices.shape[0]) if pair_indices is not None else int(num_pairs_full)
    return {
        "sid": sid,
        "row": row,
        "bundle": bundle,
        "render_asset_dir": render_asset_dir,
        "asset_path": asset_path,
        "rgb_path": rgb_path,
        "mask_path": mask_path,
        "vertices_path": vertices_path,
        "face_index_path": face_index_path,
        "face_vertex_ids_path": face_vertex_ids_path,
        "barycentric_weights_path": barycentric_weights_path,
        "mask": mask,
        "face_index": face_index,
        "face_vertex_ids": face_vertex_ids,
        "barycentric_weights": barycentric_weights,
        "vertices_render": vertices_render,
        "teacher_nv": teacher_nv,
        "candidate_teacher_vid": candidate_teacher_vid,
        "candidate_raw_vid": candidate_raw_vid,
        "visible_raw_vid": visible_raw_vid,
        "candidate_xy": candidate_xy,
        "num_pairs_full": int(num_pairs_full),
        "num_pairs_used": int(num_pairs_used),
        "pair_indices": pair_indices,
    }


def _score_and_finalize_sample(
    *,
    query: dict[str, Any],
    policy: _OfflineTeacherPolicy,
    feat: Any,
    point_offset: Any,
    out_root: Path,
    pair_process_batch_size: int,
    pair_budget: int,
    a1_reduce: str,
    a1_reduce_topk: int,
    topk_cond: int,
    heat_sigma: float,
    save_visuals: bool,
    visuals_root: Path,
    matrix_device: torch.device,
    sample_split: str,
    train_cache_cfg: dict[str, Any],
    reward_symmetrize: str,
) -> tuple[dict[str, Any], dict[str, float], dict[str, np.ndarray]]:
    sid = str(query["sid"])
    bundle = query["bundle"]

    score_t0 = time.perf_counter()
    num_candidates = int(query["candidate_teacher_vid"].shape[0])
    candidate_teacher_vid = query["candidate_teacher_vid"].astype(np.int64, copy=False)
    reward_matrix_t = torch.full(
        (num_candidates, num_candidates),
        -float("inf"),
        dtype=torch.float32,
        device=matrix_device,
    )
    pair_nan_count = 0
    pair_inf_count = 0
    pairs_scored = 0

    for local_pairs in _iter_ordered_pair_blocks(
        num_candidates=num_candidates,
        pair_indices=query["pair_indices"],
        block_size=pair_process_batch_size,
    ):
        pair_vertex_ids = np.stack(
            [
                candidate_teacher_vid[local_pairs[:, 0]],
                candidate_teacher_vid[local_pairs[:, 1]],
            ],
            axis=1,
        ).astype(np.int64, copy=False)
        pair_rewards_t = policy.teacher.infer_pairs_from_features_torch(
            feat=feat,
            point_offset=point_offset,
            pairs=pair_vertex_ids,
            max_pairs_per_forward=policy.pair_chunk_size,
        ).to(device=matrix_device, dtype=torch.float32)
        pair_nan_count += int(torch.isnan(pair_rewards_t).sum().item()) if pair_rewards_t.numel() > 0 else 0
        pair_inf_count += int(torch.isinf(pair_rewards_t).sum().item()) if pair_rewards_t.numel() > 0 else 0
        local_pairs_t = torch.from_numpy(local_pairs).to(device=matrix_device, dtype=torch.int64)
        flat_idx = local_pairs_t[:, 0] * num_candidates + local_pairs_t[:, 1]
        reward_matrix_t.view(-1).scatter_reduce_(0, flat_idx, pair_rewards_t, reduce="amax", include_self=True)
        pairs_scored += int(local_pairs.shape[0])
        del pair_vertex_ids, pair_rewards_t, local_pairs_t, flat_idx

    if num_candidates > 0:
        diag_idx = torch.arange(num_candidates, device=matrix_device)
        reward_matrix_t[diag_idx, diag_idx] = -float("inf")
    reward_matrix_t = symmetrize_reward_matrix_torch(
        reward_matrix_t,
        mode=reward_symmetrize,
        diagonal_value=-float("inf"),
    )
    score_sec = time.perf_counter() - score_t0

    matrix_t0 = time.perf_counter()
    a1_logits_t = build_a1_from_reward_matrix_torch(
        reward_matrix=reward_matrix_t,
        reduce=a1_reduce,
        topk=int(a1_reduce_topk),
    )
    matrix_sec = time.perf_counter() - matrix_t0

    reward_matrix_save = reward_matrix_t.to(dtype=torch.float16).cpu().numpy()
    a1_logits_save = a1_logits_t.to(dtype=torch.float16).cpu().numpy()
    best_x1_idx = int(torch.argmax(a1_logits_t).item()) if a1_logits_t.numel() > 0 else -1
    a1_std = float(np.std(a1_logits_save.astype(np.float32, copy=False))) if a1_logits_save.size else 0.0
    a1_top1_margin = float(compute_top1_margin(a1_logits_save.astype(np.float32, copy=False)))
    reward_row_top1_margin = (
        float(compute_reward_row_margin(reward_matrix_save.astype(np.float32, copy=False), best_x1_idx))
        if best_x1_idx >= 0
        else 0.0
    )

    cache_t0 = time.perf_counter()
    rgb_np = np.asarray(Image.open(query["rgb_path"]).convert("RGB"), dtype=np.uint8)
    cache_sample = build_pair_policy_cache_sample(
        sample_id=sid,
        asset_id=str(bundle["asset_id"]),
        rgb=rgb_np,
        mask_bool=np.asarray(query["mask"] > 0, dtype=np.uint8),
        face_index=query["face_index"],
        face_vertex_ids=query["face_vertex_ids"],
        barycentric_weights=query["barycentric_weights"],
        candidate_raw_vid=query["candidate_raw_vid"].astype(np.int64, copy=False),
        a1_logits=a1_logits_save.astype(np.float32, copy=False),
        reward_matrix=reward_matrix_save.astype(np.float32, copy=False),
        num_x1_samples=int(train_cache_cfg.get("num_x1_samples", 4)),
        a1_tau=float(train_cache_cfg.get("a1_tau", 1.0)),
        a1_top_ratio=float(train_cache_cfg.get("a1_top_ratio", 1.0)),
        target_type=str(train_cache_cfg.get("target_type", "masked_softmax")),
        train=(sample_split == "train"),
        seed=int(train_cache_cfg.get("seed", 42)),
        resize_width=train_cache_cfg.get("resize_width"),
        resize_height=train_cache_cfg.get("resize_height"),
    )
    cache_sec = time.perf_counter() - cache_t0

    vis_files: list[str] = []
    vis_sec = 0.0
    if save_visuals:
        vis_t0 = time.perf_counter()
        reward_matrix_vis = reward_matrix_t.cpu().numpy()
        a1_logits_vis = a1_logits_t.cpu().numpy()
        vis_prefix = f"{bundle['asset_id']}__{sid}__"
        vis_files = save_supervision_visuals(
            rgb_path=query["rgb_path"],
            mask_path=query["mask_path"],
            out_dir=visuals_root,
            filename_prefix=vis_prefix,
            candidate_xy=query["candidate_xy"],
            candidate_teacher_vid=query["candidate_teacher_vid"],
            candidate_raw_vid=query["candidate_raw_vid"],
            a1_logits=a1_logits_vis,
            reward_matrix=reward_matrix_vis,
            mask_index=None,
            render_to_teacher=bundle["index_map"],
            face_vertex_ids=query["face_vertex_ids"],
            barycentric_weights=query["barycentric_weights"],
            face_index=query["face_index"],
            sigma=heat_sigma,
            target_tau=float(train_cache_cfg.get("target_tau", 1.0)),
        )
        vis_sec = time.perf_counter() - vis_t0

    meta_t0 = time.perf_counter()
    meta = {
        "sample_id": sid,
        "global_sample_id": query["row"].get("global_sample_id"),
        "render_asset_id": query["row"].get("asset_id"),
        "render_asset_dir": query["render_asset_dir"],
        "asset_path": query["asset_path"],
        "teacher_asset_id": bundle["asset_id"],
        "split": str(sample_split) if sample_split else None,
        "vertex_index_map": bundle["map_path"],
        "num_vertices_render": int(query["vertices_render"].shape[0]),
        "num_vertices_teacher": int(query["teacher_nv"]),
        "num_visible_raw": int(query["visible_raw_vid"].shape[0]),
        "num_candidates": int(query["candidate_teacher_vid"].shape[0]),
        "num_pairs_full": int(query["num_pairs_full"]),
        "num_pairs_used": int(query["num_pairs_used"]),
        "pair_budget": int(pair_budget),
        "pair_nan_count": pair_nan_count,
        "pair_inf_count": pair_inf_count,
        "a1_std": a1_std,
        "a1_top1_margin": a1_top1_margin,
        "reward_row_top1_margin": reward_row_top1_margin,
        "schema_version": 2,
        "vis_files": vis_files,
        "source": {
            "rgb": str(query["rgb_path"]),
            "mask": str(query["mask_path"]),
            "vertices": str(query["vertices_path"]),
            "face_index": str(query["face_index_path"]),
            "face_vertex_ids": str(query["face_vertex_ids_path"]),
            "barycentric_weights": str(query["barycentric_weights_path"]),
        },
    }
    meta_sec = time.perf_counter() - meta_t0
    print(
        f"[INFO] sample={sid} asset={bundle['asset_id']} "
        f"visible_raw={query['visible_raw_vid'].shape[0]} candidates={query['candidate_teacher_vid'].shape[0]} "
        f"pairs={pairs_scored}"
    )
    timing = {
        "score_sec": float(score_sec),
        "matrix_sec": float(matrix_sec),
        "cache_sec": float(cache_sec),
        "vis_sec": float(vis_sec),
        "meta_sec": float(meta_sec),
    }
    del reward_matrix_t, a1_logits_t
    return meta, timing, cache_sample


def run(config_path: str | Path, *, resume: bool = False) -> None:
    cfg_path = Path(config_path).resolve()
    cfg = _load_yaml(cfg_path)

    data_root = Path(cfg["data_root"]).resolve()
    manifest_path = Path(cfg.get("manifest", data_root / "manifest.jsonl")).resolve()
    out_root = Path(cfg.get("output_dir", "logs/render_supervision")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    asset_timing_path = out_root / "asset_timings.jsonl"
    if asset_timing_path.exists() and not resume:
        asset_timing_path.unlink()

    pointcept_data_root = Path(cfg["pointcept_data_root"]).resolve()
    pointcept_manifest = Path(cfg.get("pointcept_manifest", pointcept_data_root / "manifest.json")).resolve()
    pointcept_assets = _load_pointcept_manifest(pointcept_manifest)
    vertex_index_map = cfg.get("vertex_index_map")
    vertex_index_map_path = Path(vertex_index_map).resolve() if vertex_index_map else None

    rows = _load_manifest(manifest_path)
    selected = _iter_selected_rows(
        rows,
        sample_id=cfg.get("sample_id"),
        all_samples=bool(cfg.get("all_samples", True)),
        max_samples=cfg.get("max_samples"),
    )
    print(f"[INFO] config={cfg_path} total_rows={len(rows)} selected={len(selected)}")

    teacher = TeacherRewardInfer(
        teacher_cfg=cfg["teacher_cfg"],
        teacher_ckpt=cfg["teacher_ckpt"],
        device=str(cfg.get("device", "cuda")),
        pointcept_code_root=cfg.get("pointcept_code_root"),
    )
    policy = _OfflineTeacherPolicy(
        teacher=teacher,
        pointcept_assets=pointcept_assets,
        pointcept_data_root=pointcept_data_root,
        vertex_index_map_path=vertex_index_map_path,
        raw_to_teacher_map_key=str(cfg.get("raw_to_teacher_map_key", "raw_to_down")),
        pair_chunk_size=int(cfg.get("pair_chunk_size", 65536)),
        num_candidates=int(cfg.get("num_candidates", 0)),
        teacher_min_candidate_dist=float(cfg.get("teacher_min_candidate_dist", 0.0)),
    )
    matrix_device = policy.teacher.device
    pair_process_batch_size = max(
        policy.pair_chunk_size,
        int(cfg.get("pair_process_batch_size", policy.pair_chunk_size * 4)),
    )
    grouped_rows = _group_rows_by_asset(rows=selected, data_root=data_root)
    train_cache_cfg = dict(cfg.get("train_cache", {}) or {})
    train_cache_enabled = bool(train_cache_cfg.get("enabled", False))
    if not train_cache_enabled:
        raise ValueError("render_supervision now requires train_cache.enabled=true")
    split_cfg = dict(cfg.get("split", {}) or {})
    split_seed = int(split_cfg.get("seed", 42))
    split_val_ratio = float(split_cfg.get("val_ratio", 0.125))
    split_val_assets: set[str] = set()
    all_teacher_asset_ids = []
    for asset_path, _asset_rows in grouped_rows:
        bundle = policy.resolve_asset_bundle(asset_path)
        all_teacher_asset_ids.append(str(bundle["asset_id"]))
    _, split_val_assets = _asset_ordered_split(
        all_teacher_asset_ids,
        val_ratio=split_val_ratio,
        seed=split_seed,
    )
    train_cache_cfg["num_x1_samples"] = int(train_cache_cfg.get("num_x1_samples", 4))
    train_cache_cfg["a1_tau"] = float(train_cache_cfg.get("a1_tau", 1.0))
    train_cache_cfg["a1_top_ratio"] = float(train_cache_cfg.get("a1_top_ratio", 1.0))
    train_cache_cfg["target_type"] = str(train_cache_cfg.get("target_type", "masked_softmax"))
    train_cache_cfg["target_tau"] = float(train_cache_cfg.get("target_tau", 1.0))
    train_cache_cfg["seed"] = int(train_cache_cfg.get("seed", 42))
    train_cache_cfg["resize_width"] = (
        int(train_cache_cfg["resize_width"]) if train_cache_cfg.get("resize_width") else None
    )
    train_cache_cfg["resize_height"] = (
        int(train_cache_cfg["resize_height"]) if train_cache_cfg.get("resize_height") else None
    )
    train_cache_cfg["shard_size"] = int(train_cache_cfg.get("shard_size", 0))
    train_cache_cfg["target_shard_size_mb"] = int(train_cache_cfg.get("target_shard_size_mb", 512))
    final_dataset_root = out_root
    final_shards_dir = final_dataset_root / "shards"
    final_shards_dir.mkdir(parents=True, exist_ok=True)
    train_cache_cfg["out_dir"] = str(final_dataset_root)
    visuals_root = (out_root / "visuals").resolve()
    visuals_root.mkdir(parents=True, exist_ok=True)

    existing_shard_counts = _existing_shard_sample_counts(final_shards_dir) if resume else []
    resume_sample_count = int(sum(existing_shard_counts))
    if resume_sample_count:
        print(
            f"[INFO] resume enabled: found {len(existing_shard_counts)} existing shards "
            f"with {resume_sample_count} samples",
            flush=True,
        )

    metas = []
    pending_rows: list[dict[str, Any]] = []
    pending_samples: list[dict[str, np.ndarray]] = []
    pending_meta_refs: list[dict[str, Any]] = []
    written_samples: list[dict[str, Any]] = []
    shard_id = len(existing_shard_counts)
    estimated_sample_bytes = 0
    resolved_shard_size: int | None = existing_shard_counts[0] if existing_shard_counts else None
    if existing_shard_counts:
        first_shard = final_shards_dir / "shard_00000.h5"
        estimated_sample_bytes = max(1, int(first_shard.stat().st_size // max(1, existing_shard_counts[0])))

    def _flush_pending() -> None:
        nonlocal shard_id, pending_rows, pending_samples, pending_meta_refs, written_samples
        if not pending_samples:
            return
        shard_path = final_shards_dir / f"shard_{shard_id:05d}.h5"
        shard_rows = write_pair_policy_hdf5_shard(
            shard_path=shard_path,
            rows=pending_rows,
            samples=pending_samples,
        )
        for meta_ref, shard_row in zip(pending_meta_refs, shard_rows):
            meta_ref["shard_path"] = shard_row["shard_path"]
            meta_ref["shard_index"] = shard_row["shard_index"]
            meta_ref["shard_id"] = int(shard_id)
            meta_ref["train_cache_root"] = str(final_dataset_root)
            written_samples.append(meta_ref)
        shard_id += 1
        pending_rows = []
        pending_samples = []
        pending_meta_refs = []

    global_sample_index = 0
    resume_shard_cursor = 0
    resume_shard_start = 0

    for asset_path, asset_rows in grouped_rows:
        asset_t0 = time.perf_counter()
        bundle = policy.resolve_asset_bundle(asset_path)
        asset_split = "val" if str(bundle["asset_id"]) in split_val_assets else "train"
        if resume and global_sample_index + len(asset_rows) <= resume_sample_count:
            for row in asset_rows:
                while resume_shard_cursor < len(existing_shard_counts) and (
                    global_sample_index >= resume_shard_start + existing_shard_counts[resume_shard_cursor]
                ):
                    resume_shard_start += existing_shard_counts[resume_shard_cursor]
                    resume_shard_cursor += 1
                if resume_shard_cursor >= len(existing_shard_counts):
                    raise ValueError("resume shard cursor exceeded existing shard count")
                resume_meta = _resume_meta_from_row(
                    row=row,
                    data_root=data_root,
                    asset_path=asset_path,
                    bundle=bundle,
                    sample_split=asset_split,
                    shard_path=final_shards_dir / f"shard_{resume_shard_cursor:05d}.h5",
                    shard_index=int(global_sample_index - resume_shard_start),
                    shard_id=int(resume_shard_cursor),
                    final_dataset_root=final_dataset_root,
                    pair_budget=int(cfg.get("pair_budget", 0)),
                )
                written_samples.append(resume_meta)
                metas.append(resume_meta)
                global_sample_index += 1
            continue

        encode_t0 = time.perf_counter()
        teacher_coord, _teacher_normal, feat, point_offset = policy.encode_teacher_asset(bundle)
        teacher_coord_xy = teacher_coord[:, :2].astype(np.float32, copy=False)
        encode_sec = time.perf_counter() - encode_t0

        prepare_sec = 0.0
        score_sec = 0.0
        matrix_sec = 0.0
        cache_sec = 0.0
        vis_sec = 0.0
        meta_sec = 0.0
        total_pairs = 0
        unique_pair_count = 0
        processed_samples = 0
        finalize_t0 = time.perf_counter()
        for row in asset_rows:
            if resume and global_sample_index < resume_sample_count:
                while resume_shard_cursor < len(existing_shard_counts) and (
                    global_sample_index >= resume_shard_start + existing_shard_counts[resume_shard_cursor]
                ):
                    resume_shard_start += existing_shard_counts[resume_shard_cursor]
                    resume_shard_cursor += 1
                if resume_shard_cursor >= len(existing_shard_counts):
                    raise ValueError("resume shard cursor exceeded existing shard count")
                resume_meta = _resume_meta_from_row(
                    row=row,
                    data_root=data_root,
                    asset_path=asset_path,
                    bundle=bundle,
                    sample_split=asset_split,
                    shard_path=final_shards_dir / f"shard_{resume_shard_cursor:05d}.h5",
                    shard_index=int(global_sample_index - resume_shard_start),
                    shard_id=int(resume_shard_cursor),
                    final_dataset_root=final_dataset_root,
                    pair_budget=int(cfg.get("pair_budget", 0)),
                )
                written_samples.append(resume_meta)
                metas.append(resume_meta)
                global_sample_index += 1
                continue

            sample_prepare_t0 = time.perf_counter()
            query = _prepare_sample_query(
                row=row,
                data_root=data_root,
                policy=policy,
                pair_budget=int(cfg.get("pair_budget", 0)),
                teacher_coord_xy=teacher_coord_xy,
            )
            prepare_sec += time.perf_counter() - sample_prepare_t0
            total_pairs += int(query["num_pairs_used"])
            unique_pair_count += int(query["num_pairs_used"])
            processed_samples += 1

            meta, sample_timing, cache_sample = _score_and_finalize_sample(
                query=query,
                policy=policy,
                feat=feat,
                point_offset=point_offset,
                out_root=out_root,
                pair_process_batch_size=pair_process_batch_size,
                pair_budget=int(cfg.get("pair_budget", 0)),
                a1_reduce=str(cfg.get("a1_reduce", "max")),
                a1_reduce_topk=int(cfg.get("a1_reduce_topk", 8)),
                topk_cond=int(cfg.get("topk_cond", 8)),
                heat_sigma=float(cfg.get("heat_sigma", 18.0)),
                save_visuals=(row is asset_rows[0]),
                visuals_root=visuals_root,
                matrix_device=matrix_device,
                sample_split=asset_split,
                train_cache_cfg=train_cache_cfg,
                reward_symmetrize=str(cfg.get("reward_symmetrize", "none")),
            )
            metas.append(meta)
            score_sec += float(sample_timing["score_sec"])
            matrix_sec += float(sample_timing["matrix_sec"])
            cache_sec += float(sample_timing.get("cache_sec", 0.0))
            vis_sec += float(sample_timing["vis_sec"])
            meta_sec += float(sample_timing["meta_sec"])
            if cache_sample is not None:
                if resolved_shard_size is None:
                    estimated_sample_bytes = _estimate_sample_bytes(cache_sample)
                    resolved_shard_size = _resolve_shard_size(
                        requested_shard_size=int(train_cache_cfg.get("shard_size", 0)),
                        target_shard_bytes=int(train_cache_cfg.get("target_shard_size_mb", 512)) * 1024 * 1024,
                        sample_bytes=int(estimated_sample_bytes),
                    )
                    est_mb = float(estimated_sample_bytes) / float(1024 * 1024)
                    approx_shard_mb = float(estimated_sample_bytes * resolved_shard_size) / float(1024 * 1024)
                    print(
                        f"[INFO] direct train_v2 shards: shard_size={resolved_shard_size} "
                        f"sample_est={est_mb:.2f}MB target_shard={approx_shard_mb:.1f}MB",
                        flush=True,
                    )
                pending_rows.append(meta)
                pending_samples.append(cache_sample)
                pending_meta_refs.append(meta)
                if len(pending_samples) >= int(resolved_shard_size):
                    _flush_pending()
            del query
            global_sample_index += 1
        finalize_sec = time.perf_counter() - finalize_t0
        asset_sec = time.perf_counter() - asset_t0
        timing = {
            "asset_id": bundle["asset_id"],
            "render_asset_dir": str(asset_rows[0].get("asset_dir", "")),
            "render_asset_id": int(asset_rows[0].get("asset_id", -1)),
            "teacher_asset_id": bundle["asset_id"],
            "asset_path": asset_path,
            "num_samples": int(processed_samples),
            "total_pairs": int(total_pairs),
            "unique_pairs": int(unique_pair_count),
            "reuse_ratio": float(unique_pair_count / total_pairs) if total_pairs > 0 else 0.0,
            "encode_sec": float(encode_sec),
            "prepare_sec": float(prepare_sec),
            "score_sec": float(score_sec),
            "matrix_sec": float(matrix_sec),
            "cache_sec": float(cache_sec),
            "vis_sec": float(vis_sec),
            "meta_sec": float(meta_sec),
            "finalize_sec": float(finalize_sec),
            "asset_sec": float(asset_sec),
        }
        with asset_timing_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(timing, ensure_ascii=False) + "\n")
        print(
            f"[INFO] asset={bundle['asset_id']} samples={processed_samples} "
            f"total_pairs={total_pairs} unique_pairs={unique_pair_count} "
            f"encode_sec={encode_sec:.2f} prepare_sec={prepare_sec:.2f} "
            f"score_sec={score_sec:.2f} cache_sec={cache_sec:.2f} "
            f"finalize_sec={finalize_sec:.2f} asset_sec={asset_sec:.2f}"
        )

    _flush_pending()

    index = {
        "version": 2,
        "format": "pair_policy_train_v2",
        "config": str(cfg_path),
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "pointcept_data_root": str(pointcept_data_root),
        "pointcept_manifest": str(pointcept_manifest),
        "root": str(final_dataset_root),
        "num_samples": len(written_samples),
        "num_shards": int(shard_id),
        "shard_size": int(resolved_shard_size or 0),
        "target_shard_size_mb": int(train_cache_cfg.get("target_shard_size_mb", 512)),
        "estimated_sample_size_bytes": int(estimated_sample_bytes),
        "fields": [
            "image",
            "gt_mask",
            "a1_target",
            "a1_valid_mask",
            "sampled_x1_xy",
            "a2_target",
            "a2_valid_mask",
            "a2_target_valid",
        ],
        "samples": written_samples,
    }
    (out_root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {out_root / 'index.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build teacher supervision heatmaps from a rendered dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/pair_policy/configs/render/supervision.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing contiguous HDF5 shards in output_dir and continue from the next sample.",
    )
    args = parser.parse_args()
    run(args.config, resume=bool(args.resume))


if __name__ == "__main__":
    main()
