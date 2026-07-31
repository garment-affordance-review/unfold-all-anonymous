from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn


class OfflineReplayPolicy(nn.Module):
    """
    Visibility-aware random replay:
    - Filter samples with |reward| > max_reward_abs, and optionally filter reward == 0.
    - Shuffle all offline samples into a pool; each step randomly selects a valid pair
      whose vertices are visible in the current frame.
    - Remove selected samples from the pool to avoid repeats. If the pool is empty,
      fall back to random visible vertices or fully random vertices.
    """

    def __init__(
        self,
        manager,
        offline_file: str,
        device: str = "cuda",
        cache_assets: bool = True,
        max_reward_abs: float = 2.0,
        filter_zero: bool = True,
        seed: int = 42,
        visible_resolution: int = 128,
    ):
        super().__init__()
        self.manager = manager
        self.offline_file = Path(offline_file)
        if not self.offline_file.exists():
            raise FileNotFoundError(f"Offline file not found: {self.offline_file}")
        self.device = device
        self.cache_assets = cache_assets
        self.max_reward_abs = float(max_reward_abs)
        self.filter_zero = bool(filter_zero)
        self.rng = np.random.default_rng(seed)
        self.visible_resolution = int(visible_resolution)

        self._file: Optional[h5py.File] = None
        # cache key by (cat, inst, max_abs, filter_zero)
        self._asset_cache: Dict[Tuple[str, str, float, bool], Optional[Dict[str, Any]]] = {}
        self.last_offline_info: List[Dict[str, Any]] = []

    # -------------------- helpers -------------------- #
    def _parse_asset_key(self, asset_path: str) -> Tuple[str, str]:
        parts = Path(asset_path).parts
        if "cloth" in parts:
            idx = parts.index("cloth")
            category = "_".join(parts[idx + 1 : -2]) if idx + 3 < len(parts) else "Unknown"
            instance = parts[-2]
            return category, instance
        if len(parts) >= 3:
            return parts[-3], parts[-2]
        return "Unknown", "Unknown"

    def _ensure_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.offline_file, "r")
        return self._file

    def _load_asset(self, asset_path: str) -> Optional[Dict[str, Any]]:
        cat, inst = self._parse_asset_key(asset_path)
        key = (cat, inst, self.max_reward_abs, self.filter_zero)
        if key in self._asset_cache:
            return self._asset_cache[key]

        h5 = self._ensure_file()
        if cat not in h5 or inst not in h5[cat]:
            self._asset_cache[key] = None
            return None

        grp = h5[cat][inst]
        id1 = grp["id1"][:]
        id2 = grp["id2"][:]
        rewards = grp["rewards"][:]

        mask = np.abs(rewards) <= self.max_reward_abs
        if self.filter_zero:
            mask &= rewards != 0

        if not mask.any():
            self._asset_cache[key] = None
            return None

        id1_f = id1[mask]
        id2_f = id2[mask]
        rewards_f = rewards[mask]

        indices = np.arange(len(rewards_f))
        self.rng.shuffle(indices)
        pool = [
            {"pair": (int(id1_f[i]), int(id2_f[i])), "reward": float(rewards_f[i])}
            for i in indices
        ]
        data = {"pool": pool}
        if self.cache_assets:
            self._asset_cache[key] = data
        return data

    def _compute_visible_mask(
        self,
        pos_env: torch.Tensor,
        faces_env: Optional[torch.Tensor],
        v_count: int,
    ) -> Optional[torch.Tensor]:
        try:
            if faces_env is None or faces_env.numel() == 0:
                return None
            from unfold.platform.perception import compute_visible_vertices_cuda

            grid, _ = compute_visible_vertices_cuda(
                positions_tensor=pos_env,
                faces_long=faces_env,
                resolution=self.visible_resolution,
            )
            vids = grid[grid > 0].long() - 1
            if vids.numel() == 0:
                return None
            mask = torch.zeros(v_count, dtype=torch.bool, device=pos_env.device)
            mask[vids.clamp_max(v_count - 1)] = True
            return mask
        except Exception:
            return None

    def _next_sample(
        self,
        asset_data: Optional[Dict[str, Any]],
        num_vertices: int,
        visible_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        if asset_data is None:
            return None, None
        pool: List[Dict[str, Any]] = asset_data.get("pool", [])
        if not pool:
            return None, None

        visible_indices: List[int] = []
        for idx, sample in enumerate(pool):
            a, b = sample["pair"]
            if not (0 <= a < num_vertices and 0 <= b < num_vertices and a != b):
                continue
            if visible_mask is not None and not (visible_mask[a] and visible_mask[b]):
                continue
            visible_indices.append(idx)

        if not visible_indices:
            return None, None

        pick_idx = visible_indices[self.rng.integers(0, len(visible_indices))]
        sample = pool.pop(pick_idx)
        a, b = sample["pair"]
        return (a, b), float(sample["reward"])

    # -------------------- forward -------------------- #
    def forward(self, obs: Dict[str, Any], explore: bool = False) -> torch.Tensor:  # type: ignore[override]
        pos_list = obs.get("pos", [])
        pos_mask = obs.get("pos_mask", None)
        faces_list = obs.get("faces", None)
        faces_mask = obs.get("faces_mask", None)

        num_envs = len(pos_list) if not torch.is_tensor(pos_list) else pos_list.shape[0]
        actions = torch.full((num_envs, 2), -1, dtype=torch.long, device=self.device)
        self.last_offline_info = [{"pair": None, "reward": None, "asset": None} for _ in range(num_envs)]

        for env_idx in range(num_envs):
            pos = pos_list[env_idx] if torch.is_tensor(pos_list) else pos_list[env_idx]
            if pos is None:
                continue

            if pos_mask is not None:
                v_mask = pos_mask[env_idx]
                v_count = int(v_mask[..., 0].sum().item())
            else:
                v_count = pos.shape[0]
            pos_env = pos[:v_count]
            if pos_env.numel() == 0:
                continue

            faces_env = None
            if faces_list is not None and len(faces_list) > env_idx:
                faces_env = faces_list[env_idx]
                if faces_mask is not None and len(faces_mask) > env_idx and faces_mask[env_idx] is not None:
                    fm = faces_mask[env_idx]
                    faces_env = faces_env[fm.squeeze() > 0]
                faces_env = faces_env.long()

            asset_paths = getattr(self.manager, "_env_usd_paths", None)
            asset_path = asset_paths[env_idx] if asset_paths and env_idx < len(asset_paths) else None
            asset_data = self._load_asset(asset_path) if asset_path else None

            visible_mask = self._compute_visible_mask(pos_env, faces_env, v_count)
            best_pair, offline_reward = self._next_sample(asset_data, v_count, visible_mask)

            if best_pair is None:
                # Fallback: random visible vertices, otherwise fully random vertices.
                if visible_mask is not None:
                    vis_idx = visible_mask.nonzero(as_tuple=False).squeeze(-1)
                    if vis_idx.numel() >= 2:
                        actions[env_idx] = vis_idx[torch.randperm(vis_idx.numel(), device=self.device)[:2]]
                if actions[env_idx].min().item() < 0 and v_count >= 2:
                    actions[env_idx] = torch.randperm(v_count, device=self.device)[:2]
                self.last_offline_info[env_idx] = {"pair": None, "reward": None, "asset": asset_path}
                continue

            a, b = best_pair
            actions[env_idx, 0] = a
            actions[env_idx, 1] = b
            self.last_offline_info[env_idx] = {
                "pair": (a, b),
                "reward": offline_reward,
                "asset": asset_path,
            }

        return actions

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None

    def __del__(self):
        self.close()
