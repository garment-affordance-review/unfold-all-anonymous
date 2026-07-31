from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np


def asset_key_from_path(asset_path: str) -> Tuple[str, str]:
    """
    Mirror StructuredHDF5Storage._parse_from_parts to build (category, instance) key.
    """
    parts = Path(asset_path).parts
    if "cloth" in parts:
        idx = parts.index("cloth")
        category = "_".join(parts[idx + 1 : -2]) if idx + 3 < len(parts) else "Unknown"
        instance = parts[-2]
        return category, instance

    if len(parts) >= 3:
        return parts[-3], parts[-2]
    return "Unknown", "Unknown"


def scan_offline_groups(h5_path: str) -> List[Tuple[str, str, int]]:
    """
    Returns list of (category, instance, count) tuples.
    """
    results: List[Tuple[str, str, int]] = []
    with h5py.File(h5_path, "r") as f:
        for cat in f.keys():
            if cat == "metadata":
                continue
            cat_grp = f[cat]
            for inst in cat_grp.keys():
                g = cat_grp[inst]
                results.append((cat, inst, g["id1"].shape[0]))
    return results


def load_offline_index(
    h5_path: str, allowed_keys: Sequence[Tuple[str, str]] | None = None
) -> List[Tuple[Tuple[str, str], np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load offline data into memory (id1, id2, reward per asset key).
    """
    allowed = set(allowed_keys) if allowed_keys is not None else None
    out: List[Tuple[Tuple[str, str], np.ndarray, np.ndarray, np.ndarray]] = []
    with h5py.File(h5_path, "r") as f:
        for cat in f.keys():
            if cat == "metadata":
                continue
            cat_grp = f[cat]
            for inst in cat_grp.keys():
                key = (cat, inst)
                if allowed is not None and key not in allowed:
                    continue
                g = cat_grp[inst]
                id1 = g["id1"][:]
                id2 = g["id2"][:]
                rewards = g["rewards"][:]
                out.append((key, id1, id2, rewards))
    return out


def load_vertex_usage(h5_path: str) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Collect unique vertex ids used per asset in offline data.
    """
    usage: Dict[Tuple[str, str], np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for cat in f.keys():
            if cat == "metadata":
                continue
            cat_grp = f[cat]
            for inst in cat_grp.keys():
                g = cat_grp[inst]
                ids = np.concatenate([g["id1"][:], g["id2"][:]], axis=0)
                usage[(cat, inst)] = np.unique(ids)
    return usage
