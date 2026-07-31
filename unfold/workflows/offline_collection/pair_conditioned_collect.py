#!/usr/bin/env python3
"""Pair-conditioned offline data collection to Pointcept clothes_with_map format."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.app import AppLauncher

from unfold.platform.camera import rotmat_to_quat_wxyz

from .cli import build_pair_conditioned_collect_parser
from .common import PROJECT_ROOT, configure_runtime_warnings, install_exit_signal_handlers


@dataclass(frozen=True)
class PairCandidate:
    coord_id1: int
    coord_id2: int
    raw_id1: int
    raw_id2: int
    quat_wxyz: torch.Tensor
    rotated_midpoint: torch.Tensor
    distance: float
    bin_idx: int


@dataclass
class AssetPointCloud:
    asset_path: str
    source: str
    raw_coord: np.ndarray
    raw_faces: np.ndarray
    coord: np.ndarray
    normal: np.ndarray
    segment: np.ndarray
    raw2coord: np.ndarray
    coord2raw: np.ndarray
    coord2raw_indptr: np.ndarray
    coord2raw_indices: np.ndarray
    anchor_coord_ids: np.ndarray


@dataclass
class AssetMonitorRecord:
    asset_id: int
    pair_count: int
    elapsed_sec: float
    rewards: np.ndarray
    bin_indices: np.ndarray


class RollingMonitor:
    def __init__(self, bin_count: int, report_every: int = 20):
        self.bin_count = max(1, int(bin_count))
        self.report_every = max(1, int(report_every))
        self._saved_records: list[AssetMonitorRecord] = []
        self._failed_since_report = 0
        self._saved_since_report = 0

    def add_success(self, record: AssetMonitorRecord) -> None:
        self._saved_records.append(record)
        self._saved_since_report += 1

    def add_failure(self) -> None:
        self._failed_since_report += 1

    def should_report(self) -> bool:
        return (self._saved_since_report + self._failed_since_report) >= self.report_every

    def render_summary(self) -> str:
        total = self._saved_since_report + self._failed_since_report
        saved = self._saved_since_report
        failed = self._failed_since_report
        success_rate = saved / total if total > 0 else 0.0

        if self._saved_records:
            rewards = np.concatenate([r.rewards for r in self._saved_records if r.rewards.size > 0], axis=0)
            bins = np.concatenate([r.bin_indices for r in self._saved_records if r.bin_indices.size > 0], axis=0)
            pair_total = int(sum(r.pair_count for r in self._saved_records))
            elapsed_total = float(sum(r.elapsed_sec for r in self._saved_records))
            rolling_rate = pair_total / max(elapsed_total, 1e-6)
        else:
            rewards = np.zeros((0,), dtype=np.float32)
            bins = np.zeros((0,), dtype=np.int64)
            rolling_rate = 0.0

        if rewards.size > 0:
            finite = np.isfinite(rewards)
            rewards = rewards[finite]
        if rewards.size > 0:
            p10, p50, p90 = np.percentile(rewards, [10, 50, 90])
            reward_mean = float(np.mean(rewards))
        else:
            reward_mean = float("nan")
            p10 = p50 = p90 = float("nan")

        bucket_parts: list[str] = []
        for bin_idx in range(self.bin_count):
            mask = bins == bin_idx
            count = int(mask.sum())
            frac = count / max(int(bins.shape[0]), 1)
            if count > 0 and rewards.size > 0:
                # bins is constructed one-to-one with kept rewards.
                bin_mean = float(np.mean(np.concatenate([r.rewards[r.bin_indices == bin_idx] for r in self._saved_records if r.rewards.size > 0], axis=0)))
                bucket_parts.append(f"b{bin_idx}:{count}({frac:.0%},{bin_mean:.3f})")
            else:
                bucket_parts.append(f"b{bin_idx}:0(0%,nan)")

        return (
            f"[MONITOR] window={total} success={saved} fail={failed} success_rate={success_rate:.1%} "
            f"pair_rate={rolling_rate:.2f}/s reward_mean={reward_mean:.4f} "
            f"p10={p10:.4f} p50={p50:.4f} p90={p90:.4f} "
            f"bins={' '.join(bucket_parts)}"
        )

    def reset_window(self) -> None:
        self._saved_records.clear()
        self._failed_since_report = 0
        self._saved_since_report = 0


class PairBank:
    def __init__(self, bins: list[list[PairCandidate]], rng: np.random.Generator):
        self.bins = [bucket for bucket in bins if bucket]
        self.rng = rng

    def pop_distinct(self, count: int) -> list[PairCandidate]:
        selected: list[PairCandidate] = []
        used_pairs: set[tuple[int, int]] = set()
        bin_cursor = 0
        while len(selected) < count and self.bins:
            bucket = self.bins[bin_cursor % len(self.bins)]
            if not bucket:
                self.bins.pop(bin_cursor % len(self.bins))
                if not self.bins:
                    break
                continue
            sample_idx = int(self.rng.integers(0, len(bucket)))
            candidate = bucket.pop(sample_idx)
            key = (candidate.coord_id1, candidate.coord_id2)
            if key not in used_pairs:
                used_pairs.add(key)
                selected.append(candidate)
            bin_cursor += 1
        return selected

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self.bins)

    def preview(self, per_bin: int = 1, max_total: int = 6) -> list[PairCandidate]:
        preview: list[PairCandidate] = []
        for bucket in self.bins:
            for candidate in bucket[:per_bin]:
                preview.append(candidate)
                if len(preview) >= max_total:
                    return preview
        return preview[:max_total]


def load_pair_conditioned_env_and_cfg(args):
    import gymnasium as gym
    from unfold.platform.config_utils import parse_yaml_config
    from unfold.simulation.env import EnvCfg

    if args.task != "UnfoldAll-Cloth-Direct-v0":
        raise ValueError(f"Unknown task: {args.task}")

    yaml_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    env_cfg = parse_yaml_config(yaml_path, device=args.device if args.device else "cuda:0", env_cfg_class=EnvCfg)
    if getattr(args, "assets_manifest", None):
        env_cfg.assets_manifest = str(args.assets_manifest)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = int(args.num_envs)
        env_cfg.num_envs = int(args.num_envs)
    if getattr(args, "env_spacing", None) is not None:
        env_cfg.scene.env_spacing = float(args.env_spacing)
    if not isinstance(getattr(env_cfg, "ground_size_m", None), (list, tuple)) or len(getattr(env_cfg, "ground_size_m", [])) < 2:
        spacing = float(getattr(env_cfg.scene, "env_spacing", 2.0) or 2.0)
        grid_dim = max(1, math.ceil(math.sqrt(int(env_cfg.scene.num_envs))))
        default_size = max(4.0, (grid_dim + 1) * spacing)
        env_cfg.ground_size_m = [default_size, default_size]
    env_cfg.steps_per_episode = max(2, int(getattr(env_cfg, "steps_per_episode", 1)))

    cfg_pc = getattr(env_cfg, "pair_conditioned_collection", {}) or {}
    setattr(env_cfg, "pair_conditioned_collection", cfg_pc)

    env = gym.make(args.task, cfg=env_cfg)
    return env, env_cfg


def _safe_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return None
    return vec / norm


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    chosen: dict[tuple[int, int, int], tuple[int, float]] = {}
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for raw_idx, (vx, vy, vz) in enumerate(voxel_indices):
        key = (int(vx), int(vy), int(vz))
        center = np.array(
            [(vx + 0.5) * voxel_size, (vy + 0.5) * voxel_size, (vz + 0.5) * voxel_size],
            dtype=np.float32,
        )
        dist2 = float(np.sum((points[raw_idx] - center) ** 2))
        grouped.setdefault(key, []).append(raw_idx)
        best = chosen.get(key)
        if best is None or dist2 < best[1]:
            chosen[key] = (raw_idx, dist2)

    ordered_keys = list(chosen.keys())
    coord = np.stack([points[chosen[key][0]] for key in ordered_keys], axis=0).astype(np.float32, copy=False)
    raw2coord = np.empty((points.shape[0],), dtype=np.int64)
    coord2raw = np.empty((len(ordered_keys),), dtype=np.int64)
    indptr = np.zeros((len(ordered_keys) + 1,), dtype=np.int64)
    indices_parts: list[np.ndarray] = []

    for down_idx, key in enumerate(ordered_keys):
        raw_ids = np.array(grouped[key], dtype=np.int64)
        raw2coord[raw_ids] = down_idx
        coord2raw[down_idx] = chosen[key][0]
        indices_parts.append(raw_ids)
        indptr[down_idx + 1] = indptr[down_idx] + raw_ids.shape[0]

    indices = np.concatenate(indices_parts, axis=0) if indices_parts else np.zeros((0,), dtype=np.int64)
    return coord, raw2coord, coord2raw, indptr, indices


def _compute_vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(points, dtype=np.float32)
    if faces.size == 0:
        return normals
    tri = points[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norm.squeeze(-1) > 1e-12
    normals[valid] = normals[valid] / norm[valid]
    normals[~valid] = 0.0
    return normals.astype(np.float32, copy=False)


def _aggregate_normals_to_coord(raw_normals: np.ndarray, indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    coord_normals = np.zeros((indptr.shape[0] - 1, 3), dtype=np.float32)
    for down_idx in range(coord_normals.shape[0]):
        start = int(indptr[down_idx])
        end = int(indptr[down_idx + 1])
        raw_ids = indices[start:end]
        if raw_ids.size == 0:
            continue
        n = raw_normals[raw_ids].sum(axis=0)
        nn = float(np.linalg.norm(n))
        if nn > 1e-12:
            coord_normals[down_idx] = n / nn
    return coord_normals


def _farthest_point_sample_np(points: np.ndarray, npoint: int, rng: np.random.Generator) -> np.ndarray:
    n = int(points.shape[0])
    if n <= npoint:
        return np.arange(n, dtype=np.int64)
    selected = np.zeros((npoint,), dtype=np.int64)
    first = int(rng.integers(0, n))
    selected[0] = first
    dist2 = np.sum((points - points[first]) ** 2, axis=1)
    for i in range(1, npoint):
        next_idx = int(np.argmax(dist2))
        selected[i] = next_idx
        d = np.sum((points - points[next_idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d)
    return selected


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _save_sampling_overview(out_path: Path, pointcloud: AssetPointCloud, preview_candidates: list[PairCandidate]) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 8))
    raw_xy = pointcloud.raw_coord[:, :2]
    coord_xy = pointcloud.coord[:, :2]
    anchor_xy = pointcloud.coord[pointcloud.anchor_coord_ids][:, :2]
    ax.scatter(raw_xy[:, 0], raw_xy[:, 1], s=4, c="#d9d9d9", alpha=0.45, label="raw")
    ax.scatter(coord_xy[:, 0], coord_xy[:, 1], s=8, c="#1f77b4", alpha=0.65, label="coord")
    ax.scatter(anchor_xy[:, 0], anchor_xy[:, 1], s=18, c="#d62728", alpha=0.9, label="anchor")

    palette = ["#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]
    for idx, candidate in enumerate(preview_candidates):
        c = palette[idx % len(palette)]
        p1 = pointcloud.coord[candidate.coord_id1, :2]
        p2 = pointcloud.coord[candidate.coord_id2, :2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, linewidth=1.6, alpha=0.9)
        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=42, c=c)

    ax.set_title("Sampling Overview: raw / coord / anchor / preview pairs")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_reward_examples(out_path: Path, pointcloud: AssetPointCloud, pairs: np.ndarray, rewards: np.ndarray) -> None:
    if pairs.shape[0] == 0:
        return
    plt = _setup_matplotlib()
    finite = np.isfinite(rewards)
    if not np.any(finite):
        return
    valid_idx = np.flatnonzero(finite)
    rewards_valid = rewards[valid_idx]
    order = valid_idx[np.argsort(rewards_valid)]
    pick_bottom = int(order[0])
    pick_mid = int(order[len(order) // 2])
    pick_top = int(order[-1])
    chosen = [("Low", pick_bottom), ("Mid", pick_mid), ("High", pick_top)]

    coord_xy = pointcloud.coord[:, :2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    for ax, (label, idx) in zip(axes[0], chosen):
        pair = pairs[idx]
        p1 = coord_xy[pair[0]]
        p2 = coord_xy[pair[1]]
        ax.scatter(coord_xy[:, 0], coord_xy[:, 1], s=8, c="#bdbdbd", alpha=0.55)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#1f77b4", linewidth=2.0)
        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=70, c="#d62728")
        ax.set_title(f"{label} Reward = {float(rewards[idx]):.4f}")
        ax.set_aspect("equal", adjustable="box")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


class PointceptAssetWriter:
    REQUIRED_FILES = (
        "coord.npy",
        "normal.npy",
        "segment.npy",
        "pairs.npy",
        "reward.npy",
        "raw2coord.npy",
        "coord2raw.npy",
        "coord2raw_all.npz",
    )

    def __init__(self, output_root: Path, overwrite: bool, resume: bool, worker_id: int | None = None):
        self.output_root = output_root
        self.assets_root = self.output_root / "assets"
        self.assets_root.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite
        self.resume = resume
        self.worker_id = worker_id
        if worker_id is None:
            self.index_path = self.output_root / "asset_index.jsonl"
        else:
            self.index_path = self.output_root / f"asset_index.worker_{worker_id}.jsonl"
        self._index_records: dict[str, dict[str, Any]] = {}
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._index_records[record["asset_dir"]] = record

    def asset_dir(self, asset_id: int) -> Path:
        return self.assets_root / f"asset_{asset_id:04d}"

    def is_complete(self, asset_id: int) -> bool:
        asset_dir = self.asset_dir(asset_id)
        return asset_dir.exists() and all((asset_dir / name).exists() for name in self.REQUIRED_FILES)

    def should_skip(self, asset_id: int) -> bool:
        if self.overwrite:
            return False
        if self.resume and self.is_complete(asset_id):
            return True
        return False

    def write_asset(
        self,
        asset_id: int,
        asset_path: str,
        pointcloud: AssetPointCloud,
        pairs: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        asset_dir = self.asset_dir(asset_id)
        asset_dir.mkdir(parents=True, exist_ok=True)
        np.save(asset_dir / "coord.npy", pointcloud.coord.astype(np.float32, copy=False))
        np.save(asset_dir / "normal.npy", pointcloud.normal.astype(np.float32, copy=False))
        np.save(asset_dir / "segment.npy", pointcloud.segment.astype(np.int32, copy=False))
        np.save(asset_dir / "pairs.npy", pairs.astype(np.int64, copy=False))
        np.save(asset_dir / "reward.npy", rewards.astype(np.float32, copy=False))
        np.save(asset_dir / "raw2coord.npy", pointcloud.raw2coord.astype(np.int64, copy=False))
        np.save(asset_dir / "coord2raw.npy", pointcloud.coord2raw.astype(np.int64, copy=False))
        np.savez(
            asset_dir / "coord2raw_all.npz",
            indptr=pointcloud.coord2raw_indptr.astype(np.int64, copy=False),
            indices=pointcloud.coord2raw_indices.astype(np.int64, copy=False),
        )

        record = {"asset_dir": asset_dir.name, "asset_id": int(asset_id), "asset_path": asset_path}
        self._index_records[asset_dir.name] = record
        with self.index_path.open("w", encoding="utf-8") as f:
            for key in sorted(self._index_records.keys()):
                f.write(json.dumps(self._index_records[key], ensure_ascii=False) + "\n")


class PairConditionedOfflineCollector:
    def __init__(self, env, env_cfg, args):
        self.env = env
        self.env_cfg = env_cfg
        self.args = args
        self.cfg_pc = getattr(env_cfg, "pair_conditioned_collection", {}) or {}
        self.device = env.unwrapped.device
        self.vis_root = Path(self.args.vis_dir)
        self._save_debug_vis = bool(self.cfg_pc.get("save_debug_vis", False))
        if self._save_debug_vis:
            self.vis_root.mkdir(parents=True, exist_ok=True)
        seed = int(getattr(env_cfg, "seed", 42) or 42)
        self.rng = np.random.default_rng(seed)
        self.writer = PointceptAssetWriter(
            output_root=Path(self.cfg_pc["output_root"]),
            overwrite=bool(args.overwrite),
            resume=bool(self.cfg_pc.get("resume", False)),
            worker_id=getattr(args, "worker_id", None),
        )
        self.monitor = RollingMonitor(bin_count=int(self.cfg_pc["pair_distance_bins"]), report_every=20)
        self.start_time = time.time()
        self._pair_bank_cache: dict[str, PairBank] = {}
        self._pointcloud_cache: dict[str, AssetPointCloud] = {}
        self._asset_paths = list(env.unwrapped._asset_pool._pool)
        self._asset_ids = list(env.unwrapped._asset_pool._asset_ids)

    def _assign_single_asset_batch(self, asset_idx: int) -> None:
        pool = self.env.unwrapped._asset_pool
        repeated = [int(asset_idx)] * int(self.env_cfg.scene.num_envs)
        pool._batches = [repeated]
        pool.idx = 0

    def _reset_env_without_randomization(self, options: dict[str, Any]) -> None:
        env = self.env.unwrapped
        prev_randomize = bool(getattr(env.cfg, "randomize_on_reset", True))
        env.cfg.randomize_on_reset = False
        try:
            env.reset(options=options)
        finally:
            env.cfg.randomize_on_reset = prev_randomize

    def _reset_single_asset(self, asset_idx: int, asset_seq_id: int) -> None:
        self._assign_single_asset_batch(asset_idx)
        epoch_info = {"epoch": 1, "total_epochs": 1, "batch": asset_seq_id + 1, "total_batches": len(self._asset_paths)}
        self._reset_env_without_randomization({"switch_asset": True, "epoch_info": epoch_info})

    def _prepare_pointcloud(self, env_idx: int) -> AssetPointCloud:
        manager = self.env.unwrapped._garment_manager
        asset_path = manager._env_usd_paths[env_idx]
        if asset_path in self._pointcloud_cache:
            return self._pointcloud_cache[asset_path]

        count = int(self.env.unwrapped._garment_manager._num_particles_per_env_dict.get(env_idx, 0))
        template = manager._template_pos_per_env.get(env_idx)
        if template is None or count <= 0:
            raise RuntimeError(f"Missing USD-derived template positions for asset {asset_path}")
        raw_coord = template[:count].detach().cpu().numpy().astype(np.float32, copy=False)
        raw_faces = self.env.unwrapped._unfold.action_manager.get_mesh_faces(env_idx).detach().cpu().numpy().astype(np.int64)
        valid_faces = np.all((raw_faces >= 0) & (raw_faces < raw_coord.shape[0]), axis=1)
        raw_faces = raw_faces[valid_faces]

        coord, raw2coord, coord2raw, indptr, indices = _voxel_downsample(
            raw_coord,
            voxel_size=float(self.cfg_pc["coord_voxel_size"]),
        )
        raw_normals = _compute_vertex_normals(raw_coord, raw_faces)
        coord_normals = _aggregate_normals_to_coord(raw_normals, indptr, indices)
        segment = np.full((coord.shape[0],), -1, dtype=np.int32)
        anchor_coord_ids = _farthest_point_sample_np(
            coord,
            npoint=min(int(self.cfg_pc["anchor_fps_count"]), int(coord.shape[0])),
            rng=self.rng,
        )

        pointcloud = AssetPointCloud(
            asset_path=asset_path,
            source="template_pos_per_env",
            raw_coord=raw_coord,
            raw_faces=raw_faces,
            coord=coord,
            normal=coord_normals,
            segment=segment,
            raw2coord=raw2coord,
            coord2raw=coord2raw,
            coord2raw_indptr=indptr,
            coord2raw_indices=indices,
            anchor_coord_ids=anchor_coord_ids.astype(np.int64, copy=False),
        )
        self._pointcloud_cache[asset_path] = pointcloud
        return pointcloud

    def _apply_coord_reward_sampling_mask(self, pointcloud: AssetPointCloud) -> None:
        """Use coord representatives as the reward sampling set.

        Reward is evaluated on coord representatives instead of the legacy sample_mask subset.
        """

        manager = self.env.unwrapped._garment_manager
        sampling_mask = manager.get_sampling_mask().clone()
        sampling_mask.zero_()

        raw_ids = pointcloud.coord2raw.astype(np.int64, copy=False)
        for env_idx, asset_path in enumerate(manager._env_usd_paths):
            if asset_path != pointcloud.asset_path:
                continue
            count = int(pointcloud.raw_coord.shape[0])
            if count > 0:
                sampling_mask[env_idx, raw_ids, 0] = 1.0
                if count < sampling_mask.shape[1]:
                    sampling_mask[env_idx, count:, 0] = 0.0

        manager._sampling_mask = sampling_mask

    def _build_pair_candidate(
        self,
        pointcloud: AssetPointCloud,
        coord_id1: int,
        coord_id2: int,
        distance: float,
        bin_idx: int,
    ) -> PairCandidate | None:
        raw_id1 = int(pointcloud.coord2raw[coord_id1])
        raw_id2 = int(pointcloud.coord2raw[coord_id2])
        p1 = pointcloud.raw_coord[raw_id1]
        p2 = pointcloud.raw_coord[raw_id2]
        midpoint = 0.5 * (p1 + p2)
        pair_dir = _safe_normalize(p2 - p1)
        if pair_dir is None:
            return None

        rel = pointcloud.raw_coord - midpoint[None, :]
        proj = rel - np.outer(rel @ pair_dir, pair_dir)
        centroid = proj.mean(axis=0)
        centroid = centroid - float(np.dot(centroid, pair_dir)) * pair_dir
        body_dir = _safe_normalize(centroid)
        if body_dir is None:
            return None
        normal = _safe_normalize(np.cross(pair_dir, body_dir))
        if normal is None:
            return None
        body_dir = _safe_normalize(np.cross(normal, pair_dir))
        if body_dir is None:
            return None

        basis = np.stack([pair_dir, body_dir, normal], axis=1).astype(np.float32, copy=False)
        rotation = torch.from_numpy(basis.T.copy()).to(device=self.device, dtype=torch.float32)
        quat = rotmat_to_quat_wxyz(rotation.unsqueeze(0))[0]
        midpoint_t = torch.as_tensor(midpoint, device=self.device, dtype=torch.float32)
        rotated_midpoint = rotation @ midpoint_t
        return PairCandidate(
            coord_id1=int(coord_id1),
            coord_id2=int(coord_id2),
            raw_id1=raw_id1,
            raw_id2=raw_id2,
            quat_wxyz=quat,
            rotated_midpoint=rotated_midpoint,
            distance=float(distance),
            bin_idx=int(bin_idx),
        )

    def _build_pair_bank(self, env_idx: int) -> PairBank:
        pointcloud = self._prepare_pointcloud(env_idx)
        if pointcloud.asset_path in self._pair_bank_cache:
            return self._pair_bank_cache[pointcloud.asset_path]

        anchor_ids = pointcloud.anchor_coord_ids
        if anchor_ids.size < 2:
            raise RuntimeError(f"Not enough anchors for asset {pointcloud.asset_path}")
        anchor_points = pointcloud.coord[anchor_ids]
        bin_count = max(1, int(self.cfg_pc["pair_distance_bins"]))

        pair_entries: list[tuple[int, int, float]] = []
        for i in range(anchor_ids.shape[0] - 1):
            p1 = anchor_points[i]
            d = np.linalg.norm(anchor_points[i + 1 :] - p1, axis=1)
            for j_offset, dist in enumerate(d, start=i + 1):
                if not np.isfinite(dist) or dist <= 1e-8:
                    continue
                pair_entries.append((int(anchor_ids[i]), int(anchor_ids[j_offset]), float(dist)))

        if not pair_entries:
            raise RuntimeError(f"No valid pair entries for asset {pointcloud.asset_path}")

        distances = np.array([entry[2] for entry in pair_entries], dtype=np.float32)
        quantiles = np.linspace(0.0, 1.0, bin_count + 1)
        edges = np.quantile(distances, quantiles)
        edges[0] = min(edges[0], distances.min())
        edges[-1] = max(edges[-1], distances.max() + 1e-6)

        bins: list[list[PairCandidate]] = [[] for _ in range(bin_count)]
        for coord_id1, coord_id2, dist in pair_entries:
            bin_idx = min(bin_count - 1, max(0, int(np.searchsorted(edges, dist, side="right") - 1)))
            candidate = self._build_pair_candidate(pointcloud, coord_id1, coord_id2, dist, bin_idx)
            if candidate is not None:
                bins[bin_idx].append(candidate)

        non_empty = [bucket for bucket in bins if bucket]
        if not non_empty:
            raise RuntimeError(f"No valid pair-conditioned candidates for asset {pointcloud.asset_path}")

        bank = PairBank(non_empty, rng=self.rng)
        self._pair_bank_cache[pointcloud.asset_path] = bank
        return bank

    def _apply_pair_conditioned_poses(
        self,
        candidates: list[PairCandidate],
        rot_noise_deg: tuple[float, float, float] | None = None,
    ) -> dict[str, Any]:
        env = self.env.unwrapped
        env_ids = torch.arange(self.env_cfg.scene.num_envs, device=self.device, dtype=torch.long)
        env_ids_list = env_ids.cpu().tolist()
        env._unfold.reset(env_ids_list)
        env_origins = env.scene.env_origins
        spawn_center = torch.tensor(self.env_cfg.spawn_cfg["center"], device=self.device, dtype=torch.float32)
        spawn_center = spawn_center.clone()
        spawn_center[2] = float(self.cfg_pc["pair_target_height"])

        root_pos = torch.zeros((self.env_cfg.scene.num_envs, 3), device=self.device, dtype=torch.float32)
        root_rot = torch.zeros((self.env_cfg.scene.num_envs, 4), device=self.device, dtype=torch.float32)
        rot_noise_deg = rot_noise_deg or (0.0, 0.0, 0.0)
        noise_x, noise_y, noise_z = [float(v) for v in rot_noise_deg]
        for env_idx, candidate in enumerate(candidates):
            target_midpoint = env_origins[env_idx] + spawn_center
            quat = candidate.quat_wxyz
            if noise_x > 0.0 or noise_y > 0.0 or noise_z > 0.0:
                from isaaclab.utils.math import quat_from_euler_xyz

                euler_deg = torch.tensor(
                    [
                        self.rng.uniform(-noise_x, noise_x),
                        self.rng.uniform(-noise_y, noise_y),
                        self.rng.uniform(-noise_z, noise_z),
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                euler_rad = torch.deg2rad(euler_deg)
                noise_quat = quat_from_euler_xyz(euler_rad[0], euler_rad[1], euler_rad[2]).view(4)
                quat = _quat_mul_wxyz(quat.view(1, 4), noise_quat.view(1, 4))[0]
            root_rot[env_idx] = quat
            root_pos[env_idx] = target_midpoint - candidate.rotated_midpoint

        env._garment_manager.reset_to_poses(env_ids, root_pos, root_rot)
        env.scene.write_data_to_sim()
        if env.sim.has_gui():
            env.sim.render()
        env._unfold.action_manager.set_sequence_phase("idle")
        env.episode_length = 0
        env.extras = {}
        obs = env._get_observations()
        env.obs = obs
        return obs

    def _collect_single_asset(self, asset_idx: int, asset_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        self._reset_single_asset(asset_idx, asset_id)
        bank = self._build_pair_bank(0)
        pointcloud = self._prepare_pointcloud(0)
        self._apply_coord_reward_sampling_mask(pointcloud)
        vis_asset_dir = self.vis_root / f"asset_{asset_id:04d}"
        preview_candidates = bank.preview(per_bin=1, max_total=6)
        if self._save_debug_vis and preview_candidates:
            _save_sampling_overview(vis_asset_dir / "sampling_overview.png", pointcloud, preview_candidates)
        target_pairs = int(self.cfg_pc["pairs_per_asset"])
        collected_pairs: list[list[int]] = []
        collected_rewards: list[float] = []
        collected_bins: list[int] = []

        while len(collected_pairs) < target_pairs:
            batch_n = int(self.env_cfg.scene.num_envs)
            candidates = bank.pop_distinct(batch_n)
            if len(candidates) < batch_n:
                break

            actions = torch.full((self.env_cfg.scene.num_envs, 2), -1, dtype=torch.long, device=self.device)
            for env_idx, candidate in enumerate(candidates):
                actions[env_idx, 0] = candidate.raw_id1
                actions[env_idx, 1] = candidate.raw_id2

            self._apply_pair_conditioned_poses(candidates)
            self.env.unwrapped.progress_info = {
                "epoch": 1,
                "total_epochs": 1,
                "step_in_epoch": len(collected_pairs) + 1,
                "steps_per_episode": target_pairs,
                "step": len(collected_pairs) + 1,
                "total_steps": target_pairs,
            }
            obs, rewards, *_ = self.env.unwrapped.step(actions)
            del obs
            reward_list = rewards.detach().cpu().view(-1).tolist() if torch.is_tensor(rewards) else list(rewards)

            for env_idx, candidate in enumerate(candidates):
                reward = float(reward_list[env_idx])
                if not math.isfinite(reward):
                    continue
                collected_pairs.append([candidate.coord_id1, candidate.coord_id2])
                collected_rewards.append(reward)
                collected_bins.append(int(candidate.bin_idx))
                if len(collected_pairs) >= target_pairs:
                    break

        pairs_np = np.asarray(collected_pairs, dtype=np.int64)
        rewards_np = np.asarray(collected_rewards, dtype=np.float32)
        bins_np = np.asarray(collected_bins, dtype=np.int64)
        if self._save_debug_vis and pairs_np.shape[0] > 0:
            _save_reward_examples(vis_asset_dir / "reward_examples.png", pointcloud, pairs_np, rewards_np)

        return (
            pairs_np,
            rewards_np,
            bins_np,
            pointcloud.asset_path,
        )

    def run(self, simulation_app):
        processed = 0
        failed: list[dict[str, Any]] = []
        total_assets = len(self._asset_paths)
        print(f"[INFO] Pair-conditioned collection started. assets={total_assets}", flush=True)

        try:
            for asset_seq_id, asset_path in enumerate(self._asset_paths):
                asset_id = int(self._asset_ids[asset_seq_id])
                if self.writer.should_skip(asset_id):
                    print(f"[SKIP] asset_id={asset_id:04d} path={asset_path}", flush=True)
                    processed += 1
                    continue

                try:
                    t0 = time.time()
                    pairs, rewards, bins_np, loaded_asset_path = self._collect_single_asset(asset_seq_id, asset_id)
                    if pairs.shape[0] < int(self.cfg_pc["pairs_per_asset"]):
                        raise RuntimeError(
                            f"asset_id={asset_id:04d} collected only {pairs.shape[0]} / {int(self.cfg_pc['pairs_per_asset'])} pairs"
                        )
                    pointcloud = self._pointcloud_cache[loaded_asset_path]
                    self.writer.write_asset(asset_id, loaded_asset_path, pointcloud, pairs, rewards)
                    elapsed = time.time() - t0
                    rate = pairs.shape[0] / max(elapsed, 1e-6)
                    print(
                        f"[SAVE] asset_id={asset_id:04d} pairs={pairs.shape[0]} coord={pointcloud.coord.shape[0]} "
                        f"time={elapsed:.2f}s rate={rate:.2f} pair/s path={loaded_asset_path}",
                        flush=True,
                    )
                    self.monitor.add_success(
                        AssetMonitorRecord(
                            asset_id=asset_id,
                            pair_count=int(pairs.shape[0]),
                            elapsed_sec=float(elapsed),
                            rewards=rewards,
                            bin_indices=bins_np,
                        )
                    )
                    if self.monitor.should_report():
                        print(self.monitor.render_summary(), flush=True)
                        self.monitor.reset_window()
                    processed += 1
                except Exception as exc:
                    failed.append({"asset_id": asset_id, "asset_path": asset_path, "error": str(exc)})
                    print(f"[FAIL] asset_id={asset_id:04d} path={asset_path} error={exc}", flush=True)
                    self.monitor.add_failure()
                    if self.monitor.should_report():
                        print(self.monitor.render_summary(), flush=True)
                        self.monitor.reset_window()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.", flush=True)
        finally:
            if getattr(self.args, "worker_id", None) is None:
                fail_path = self.writer.output_root / "failed_assets.json"
            else:
                fail_path = self.writer.output_root / f"failed_assets.worker_{int(self.args.worker_id)}.json"
            fail_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
            if self.monitor._saved_since_report > 0 or self.monitor._failed_since_report > 0:
                print(self.monitor.render_summary(), flush=True)
                self.monitor.reset_window()
            self.env.close()
            simulation_app.close()
            elapsed = time.time() - self.start_time
            print(
                f"\n[INFO] Pair-conditioned collection completed. processed={processed} "
                f"failed={len(failed)} elapsed={elapsed / 3600.0:.2f}h",
                flush=True,
            )


def run(args) -> None:
    configure_runtime_warnings()
    install_exit_signal_handlers("Signal received, stopping pair-conditioned collection...")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import carb
    import unfold  # noqa: F401

    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    carb.log_warn = lambda *a, **k: None

    env, env_cfg = load_pair_conditioned_env_and_cfg(args)
    cfg_pc = getattr(env_cfg, "pair_conditioned_collection", {}) or {}

    print(
        "[INFO] Pair-conditioned collection | "
        f"task={args.task} num_envs={env_cfg.scene.num_envs} "
        f"output_root={cfg_pc['output_root']} voxel={cfg_pc['coord_voxel_size']} "
        f"anchors={cfg_pc['anchor_fps_count']} pairs_per_asset={cfg_pc['pairs_per_asset']} "
        f"bins={cfg_pc['pair_distance_bins']} reward_sampling=coord",
        flush=True,
    )

    collector = PairConditionedOfflineCollector(env, env_cfg, args)
    collector.run(simulation_app)


def main() -> None:
    parser = build_pair_conditioned_collect_parser()
    args = parser.parse_args()
    run(args)
