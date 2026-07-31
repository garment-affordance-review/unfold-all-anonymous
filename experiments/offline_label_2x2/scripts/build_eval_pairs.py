#!/usr/bin/env python3
"""Build a fixed ordered-pair evaluation set for the offline-label 2x2 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed ordered evaluation pairs for sampled assets.")
    parser.add_argument(
        "--assets-manifest",
        type=str,
        default="experiments/offline_label_2x2/manifests/assets_pointcount_100_seed0.json",
    )
    parser.add_argument("--clothes-root", type=str, default="data/clothes/assets")
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--pair-distance-bins", type=int, default=4)
    parser.add_argument("--pairs-per-bin", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/offline_label_2x2/manifests/eval_pairs_100assets_128anchors_4x8_seed0.json",
    )
    return parser


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


def _build_asset_pairs(asset_dir: Path, anchor_count: int, bin_count: int, pairs_per_bin: int, rng: np.random.Generator) -> dict:
    coord = np.load(asset_dir / "coord.npy")
    coord2raw = np.load(asset_dir / "coord2raw.npy")
    if coord.shape[0] < 2:
        raise RuntimeError(f"Not enough coord points in {asset_dir}")

    anchor_ids = _farthest_point_sample_np(coord, npoint=min(int(anchor_count), int(coord.shape[0])), rng=rng).astype(np.int64, copy=False)
    anchor_points = coord[anchor_ids]
    pair_entries: list[dict] = []
    for i in range(anchor_ids.shape[0]):
        p1 = anchor_points[i]
        for j in range(anchor_ids.shape[0]):
            if i == j:
                continue
            coord_id1 = int(anchor_ids[i])
            coord_id2 = int(anchor_ids[j])
            dist = float(np.linalg.norm(coord[coord_id2] - p1))
            if not np.isfinite(dist) or dist <= 1e-8:
                continue
            pair_entries.append(
                {
                    "coord_id1": coord_id1,
                    "coord_id2": coord_id2,
                    "raw_id1": int(coord2raw[coord_id1]),
                    "raw_id2": int(coord2raw[coord_id2]),
                    "distance": dist,
                }
            )

    if not pair_entries:
        raise RuntimeError(f"No valid ordered pairs for {asset_dir}")

    distances = np.asarray([entry["distance"] for entry in pair_entries], dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.quantile(distances, quantiles)
    edges[0] = min(edges[0], float(distances.min()))
    edges[-1] = max(edges[-1], float(distances.max()) + 1e-6)

    bins: list[list[dict]] = [[] for _ in range(bin_count)]
    for entry in pair_entries:
        bin_idx = min(bin_count - 1, max(0, int(np.searchsorted(edges, entry["distance"], side="right") - 1)))
        item = dict(entry)
        item["bin_idx"] = int(bin_idx)
        bins[bin_idx].append(item)

    selected_pairs: list[dict] = []
    bin_summary: list[dict] = []
    for bin_idx, bucket in enumerate(bins):
        if len(bucket) < pairs_per_bin:
            raise RuntimeError(
                f"Asset {asset_dir.name} bin {bin_idx} only has {len(bucket)} ordered pairs, needs {pairs_per_bin}."
            )
        order = rng.permutation(len(bucket))[:pairs_per_bin]
        chosen = [bucket[int(i)] for i in order]
        chosen.sort(key=lambda item: (item["coord_id1"], item["coord_id2"]))
        selected_pairs.extend(chosen)
        bin_summary.append({"bin_idx": bin_idx, "available": len(bucket), "selected": pairs_per_bin})

    return {
        "coord_count": int(coord.shape[0]),
        "anchor_count": int(anchor_ids.shape[0]),
        "anchor_coord_ids": anchor_ids.tolist(),
        "bin_summary": bin_summary,
        "pairs": selected_pairs,
    }


def run(args) -> None:
    manifest_path = Path(args.assets_manifest)
    clothes_root = Path(args.clothes_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        assets = payload.get("assets", [])
    elif isinstance(payload, list):
        assets = []
        for asset_index, item in enumerate(payload):
            if isinstance(item, dict):
                asset_path = item.get("asset_path") or item.get("path") or item.get("usd")
                asset_id = int(item.get("asset_id", asset_index))
            else:
                asset_path = str(item)
                asset_id = int(asset_index)
            assets.append(
                {
                    "asset_index": int(asset_index),
                    "asset_id": int(asset_id),
                    "asset_path": str(asset_path),
                }
            )
    else:
        raise TypeError(f"Unsupported assets manifest format: {type(payload).__name__}")
    if not assets:
        raise RuntimeError(f"No assets found in manifest: {manifest_path}")

    rng = np.random.default_rng(int(args.seed))
    result_assets: list[dict] = []
    for local_asset_index, asset in enumerate(assets):
        asset_index = int(asset["asset_index"])
        asset_dir = clothes_root / f"asset_{asset_index:04d}"
        entry = _build_asset_pairs(
            asset_dir=asset_dir,
            anchor_count=int(args.anchor_count),
            bin_count=int(args.pair_distance_bins),
            pairs_per_bin=int(args.pairs_per_bin),
            rng=rng,
        )
        entry.update(
            {
                "local_asset_index": int(local_asset_index),
                "asset_index": asset_index,
                "asset_id": int(asset.get("asset_id", asset_index)),
                "asset_path": str(asset.get("asset_path", "")),
            }
        )
        result_assets.append(entry)

    result = {
        "seed": int(args.seed),
        "anchor_count": int(args.anchor_count),
        "pair_distance_bins": int(args.pair_distance_bins),
        "pairs_per_bin": int(args.pairs_per_bin),
        "assets_manifest": str(manifest_path),
        "clothes_root": str(clothes_root),
        "assets": result_assets,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote eval-pairs manifest to {out_path}")


if __name__ == "__main__":
    run(build_parser().parse_args())
