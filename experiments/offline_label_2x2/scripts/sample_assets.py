#!/usr/bin/env python3
"""Sample a fixed asset manifest for the offline-label 2x2 experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AssetRecord:
    asset_index: int
    asset_id: int
    asset_path: str
    coord_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample assets stratified by point-count distribution.")
    parser.add_argument("--valid-assets-json", type=str, default="data/assets/cloth/valid_assets.json")
    parser.add_argument("--clothes-root", type=str, default="data/clothes/assets")
    parser.add_argument("--num-assets", type=int, default=100)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/offline_label_2x2/manifests/assets_pointcount_100_seed0.json",
    )
    return parser


def _load_records(valid_assets_json: Path, clothes_root: Path) -> list[AssetRecord]:
    asset_paths = json.loads(valid_assets_json.read_text(encoding="utf-8"))
    records: list[AssetRecord] = []
    for asset_index, asset_path in enumerate(asset_paths):
        asset_dir = clothes_root / f"asset_{asset_index:04d}"
        coord_path = asset_dir / "coord.npy"
        if not coord_path.exists():
            continue
        coord = np.load(coord_path, mmap_mode="r")
        records.append(
            AssetRecord(
                asset_index=int(asset_index),
                asset_id=int(asset_index),
                asset_path=str(asset_path),
                coord_count=int(coord.shape[0]),
            )
        )
    return records


def _allocate_counts(bin_sizes: list[int], total_count: int) -> list[int]:
    total_available = sum(bin_sizes)
    if total_count > total_available:
        raise ValueError(f"Requested {total_count} assets but only {total_available} are available.")
    raw = np.asarray(bin_sizes, dtype=np.float64) / float(total_available) * float(total_count)
    alloc = np.floor(raw).astype(np.int64)
    alloc = np.minimum(alloc, np.asarray(bin_sizes, dtype=np.int64))
    remaining = int(total_count - int(alloc.sum()))
    if remaining <= 0:
        return alloc.tolist()

    frac_order = np.argsort(-(raw - alloc))
    for idx in frac_order.tolist():
        if remaining <= 0:
            break
        if alloc[idx] >= bin_sizes[idx]:
            continue
        alloc[idx] += 1
        remaining -= 1

    if remaining > 0:
        for idx in np.argsort(-np.asarray(bin_sizes)).tolist():
            if remaining <= 0:
                break
            spare = int(bin_sizes[idx] - alloc[idx])
            if spare <= 0:
                continue
            take = min(spare, remaining)
            alloc[idx] += take
            remaining -= take

    if remaining != 0:
        raise RuntimeError(f"Failed to allocate all samples, remaining={remaining}")
    return alloc.tolist()


def run(args) -> None:
    valid_assets_json = Path(args.valid_assets_json)
    clothes_root = Path(args.clothes_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_records(valid_assets_json, clothes_root)
    if not records:
        raise RuntimeError("No valid asset records found.")

    counts = np.asarray([record.coord_count for record in records], dtype=np.int64)
    bin_count = max(1, min(int(args.num_bins), len(records)))
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.quantile(counts.astype(np.float32), quantiles)
    edges[0] = min(edges[0], float(counts.min()))
    edges[-1] = max(edges[-1], float(counts.max()) + 1e-6)

    bins: list[list[AssetRecord]] = [[] for _ in range(bin_count)]
    for record in records:
        bin_idx = min(bin_count - 1, max(0, int(np.searchsorted(edges, record.coord_count, side="right") - 1)))
        bins[bin_idx].append(record)

    alloc = _allocate_counts([len(bucket) for bucket in bins], int(args.num_assets))
    rng = np.random.default_rng(int(args.seed))
    selected: list[AssetRecord] = []
    summary_bins: list[dict] = []
    for bin_idx, (bucket, take_n) in enumerate(zip(bins, alloc, strict=True)):
        if take_n <= 0:
            summary_bins.append({"bin_idx": bin_idx, "available": len(bucket), "selected": 0})
            continue
        order = rng.permutation(len(bucket))[:take_n]
        chosen = [bucket[int(i)] for i in order]
        chosen.sort(key=lambda item: item.asset_index)
        selected.extend(chosen)
        summary_bins.append({"bin_idx": bin_idx, "available": len(bucket), "selected": take_n})

    selected.sort(key=lambda item: item.asset_index)
    payload = {
        "seed": int(args.seed),
        "num_assets": len(selected),
        "num_bins": bin_count,
        "valid_assets_json": str(valid_assets_json),
        "clothes_root": str(clothes_root),
        "bins": summary_bins,
        "assets": [record.__dict__ for record in selected],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote asset manifest with {len(selected)} assets to {out_path}")


if __name__ == "__main__":
    run(build_parser().parse_args())
