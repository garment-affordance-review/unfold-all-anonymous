import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Sequence, List, Union, Any

# These helper functions are expected to remain available.
from unfold.platform.assets import absolute_asset_path, load_assets_from_json

class AssetPool:
    """Manages a pool of garment USD assets."""

    def __init__(self, cfg, assets_root: Path, device: str):
        self.cfg = cfg
        self.assets_root = assets_root
        self.garment_root = assets_root / "cloth"
        self.device = device
        
        self._pool: List[str] = []
        self._asset_ids: List[int] = []
        self._batches: List[List[int]] = [] 
        self._current_indices: Optional[np.ndarray] = None
        base_seed = getattr(self.cfg, "seed", None)
        self._rng = np.random.default_rng(None if base_seed is None else int(base_seed) + 11)
        
        # Initialize state pointers.
        self.idx = 0 
        self.num_envs = getattr(self.cfg.scene, 'num_envs', 1)  # Get the default environment count.
        
        self.build_pool()
        self.make_batches(self.num_envs)

    def build_pool(self) -> None:
        """Builds the asset pool from validation JSON."""
        self._pool.clear()
        self._asset_ids.clear()
        
        print(f"\n[ASSET_POOL] Building asset pool...", file=sys.stdout, flush=True)
        
        manifest_path = getattr(self.cfg, "assets_manifest", None)
        if manifest_path:
            manifest_json = Path(manifest_path).expanduser().resolve()
            try:
                with open(manifest_json, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception as e:
                print(f"[ASSET_POOL] Error loading assets manifest {manifest_json}: {e}", flush=True)
                manifest_data = []
            print(f"[ASSET_POOL] Loaded {len(manifest_data)} assets from {manifest_json.name}", flush=True)
        else:
            validation_json = self.garment_root / "valid_assets.json"
            manifest_data = load_assets_from_json(validation_json)
            print(f"[ASSET_POOL] Loaded {len(manifest_data)} assets from {validation_json.name}", flush=True)
        
        # Filter by categories
        records: list[dict[str, Any]] = []
        for idx, item in enumerate(manifest_data):
            if isinstance(item, dict):
                asset_path = item.get("asset_path") or item.get("path") or item.get("usd")
                if asset_path is None:
                    continue
                asset_id = int(item.get("asset_id", idx))
            else:
                asset_path = str(item)
                asset_id = idx
            records.append({"asset_id": asset_id, "asset_path": str(asset_path)})

        if records and hasattr(self.cfg, 'garment_categories') and self.cfg.garment_categories:
            categories = set(self.cfg.garment_categories)
            records = [r for r in records if str(r["asset_path"]).split('/')[0] in categories]
            print(f"[ASSET_POOL] Filtered to {len(records)} assets (categories: {categories})", flush=True)
        
        # Resolve absolute paths
        for record in records:
            self._pool.append(absolute_asset_path(record["asset_path"], self.garment_root))
            self._asset_ids.append(int(record["asset_id"]))
        
        print(f"[ASSET_POOL] Pool ready: {len(self._pool)} assets", flush=True)

    def make_batches(self, num_envs: int, shuffle: bool = True) -> None:
        """Organizes the pool into fixed-size batches.
        
        Guarantees that every batch has exactly `num_envs` items.
        If pool_size is not divisible by num_envs, or pool_size < num_envs,
        it pads using data from the beginning of the shuffled list.
        """
        if not self._pool: 
            print("[ASSET_POOL] Warning: Pool is empty!", flush=True)
            return

        pool_size = len(self._pool)
        self.num_envs = num_envs
        
        # 1. Generate base indices.
        if shuffle:
            indices = self._rng.permutation(pool_size)
        else:
            indices = np.arange(pool_size)
        
        self._batches = []
        
        # 2. Compute the required batches.
        # If the asset pool is smaller than the number of environments, create
        # at least one padded batch.
        if pool_size < num_envs:
            # Edge case: very few assets. Pad by repeating with resize.
            # Example: indices=[0,1], num_envs=4 -> batch=[0,1,0,1].
            padded_batch = np.resize(indices, num_envs).tolist()
            self._batches.append(padded_batch)
        
        else:
            # Standard case: split into complete batches.
            num_full_batches = pool_size // num_envs
            for i in range(num_full_batches):
                batch = indices[i * num_envs : (i + 1) * num_envs].tolist()
                self._batches.append(batch)
            
            # Handle the remainder.
            remainder = pool_size % num_envs
            if remainder > 0:
                # Take the final leftover chunk.
                last_chunk = indices[-remainder:]
                # Compute how many entries are needed to fill the batch.
                needed = num_envs - remainder
                # Fill from the beginning. Resize keeps this valid even when needed > pool_size.
                fill = np.resize(indices, needed)
                
                final_batch = np.concatenate([last_chunk, fill])
                self._batches.append(final_batch.tolist())
            
        self.idx = 0
        print(f"[ASSET_POOL] Created {len(self._batches)} batches (batch_size={num_envs}, shuffle={shuffle})", flush=True)

    def shuffle(self) -> None:
        """Reshuffles the asset pool and rebuilds batches."""
        if self.num_envs > 0:
             self.make_batches(self.num_envs, shuffle=True)

    def sample_indices(self) -> np.ndarray:
        """Returns the next batch of indices sequentially.
        
        It automatically loops back to the first batch when exhausted.
        Use `make_batches` to change batch size or shuffle mode.
        """
        if not self._batches:
             raise RuntimeError("Asset pool is empty or batches not initialized.")
            
        # Use modulo indexing for an infinite iterator.
        current_batch_idx = self.idx % len(self._batches)
        indices = np.array(self._batches[current_batch_idx])
        
        self._current_indices = indices
        self.idx += 1

        return indices

    def get_paths(self, indices: Sequence[int]) -> List[str]:
        """Returns USD paths for given indices."""
        # Add a bounds check to avoid out-of-range indices.
        valid_indices = [i for i in indices if i < len(self._pool)]
        return [self._pool[i] for i in valid_indices]

    def get_asset_ids(self, indices: Sequence[int]) -> List[int]:
        valid_indices = [i for i in indices if i < len(self._asset_ids)]
        return [self._asset_ids[i] for i in valid_indices]

    @property
    def size(self) -> int:
        return len(self._pool)
    
    @property
    def num_batches(self) -> int:
        return len(self._batches)
