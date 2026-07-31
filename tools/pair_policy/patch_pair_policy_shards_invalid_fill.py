#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from unfold.algorithms.supervision.projection import fill_invalid_with_margin


def _patch_shard(shard_path: Path, *, margin_ratio: float, min_margin: float) -> tuple[int, int]:
    patched_a1 = 0
    patched_a2 = 0
    with h5py.File(shard_path, "r+") as f:
        a1_value_map = f["a1_value_map"]
        a1_valid_mask = f["a1_valid_mask"]
        a2_value_map = f["a2_value_map"]
        a2_valid_mask = f["a2_valid_mask"]
        a2_target_valid = f["a2_target_valid"]

        sample_count = int(a1_value_map.shape[0])
        for idx in range(sample_count):
            a1 = np.asarray(a1_value_map[idx], dtype=np.float32)
            a1_mask = np.asarray(a1_valid_mask[idx], dtype=np.uint8)
            a1_patched = fill_invalid_with_margin(
                a1,
                a1_mask,
                margin_ratio=margin_ratio,
                min_margin=min_margin,
            ).astype(np.float16)
            if not np.array_equal(a1_value_map[idx], a1_patched):
                a1_value_map[idx] = a1_patched
                patched_a1 += 1

            valid_queries = np.asarray(a2_target_valid[idx], dtype=np.bool_)
            for q, is_valid in enumerate(valid_queries.tolist()):
                if not bool(is_valid):
                    continue
                a2 = np.asarray(a2_value_map[idx, q], dtype=np.float32)
                a2_mask = np.asarray(a2_valid_mask[idx, q], dtype=np.uint8)
                a2_patched = fill_invalid_with_margin(
                    a2,
                    a2_mask,
                    margin_ratio=margin_ratio,
                    min_margin=min_margin,
                ).astype(np.float16)
                if not np.array_equal(a2_value_map[idx, q], a2_patched):
                    a2_value_map[idx, q] = a2_patched
                    patched_a2 += 1
    return patched_a1, patched_a2


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch pair-policy shard invalid values to be below valid minimum.")
    parser.add_argument("--shards-dir", type=Path, required=True, help="Directory containing shard_*.h5 files")
    parser.add_argument("--margin-ratio", type=float, default=0.1, help="Fill margin as a fraction of valid range")
    parser.add_argument("--min-margin", type=float, default=1e-3, help="Minimum absolute fill margin")
    args = parser.parse_args()

    shard_paths = sorted(args.shards_dir.glob("shard_*.h5"))
    if not shard_paths:
        raise SystemExit(f"no shard_*.h5 files found in {args.shards_dir}")

    total_a1 = 0
    total_a2 = 0
    for i, shard_path in enumerate(shard_paths, start=1):
        patched_a1, patched_a2 = _patch_shard(
            shard_path,
            margin_ratio=float(args.margin_ratio),
            min_margin=float(args.min_margin),
        )
        total_a1 += patched_a1
        total_a2 += patched_a2
        print(
            f"[{i}/{len(shard_paths)}] {shard_path.name}: patched_a1={patched_a1} patched_a2={patched_a2}",
            flush=True,
        )
    print(
        f"done: shards={len(shard_paths)} patched_a1={total_a1} patched_a2={total_a2}",
        flush=True,
    )


if __name__ == "__main__":
    main()
