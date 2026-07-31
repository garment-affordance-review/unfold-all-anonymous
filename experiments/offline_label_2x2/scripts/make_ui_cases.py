#!/usr/bin/env python3
"""Create one-asset one-pair manifests for UI debugging from an existing pilot run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create UI-debug manifests for best/worst conditioned-vs-random cases.")
    parser.add_argument(
        "--pair-summary",
        type=str,
        default="experiments/offline_label_2x2/runs/full_v1/pair_summary.json",
    )
    parser.add_argument(
        "--pairs-manifest",
        type=str,
        default="experiments/offline_label_2x2/manifests/full_v1_pairs_100assets_128anchors_4x8_seed0.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/offline_label_2x2/manifests/ui_cases",
    )
    parser.add_argument(
        "--mode",
        choices=["fling", "y"],
        default="fling",
        help="Compare conditioned vs random under this loading mode.",
    )
    return parser


def _find_extreme_cases(pair_summary: list[dict], mode: str) -> tuple[dict, dict]:
    cond_protocol = f"cond_{mode}"
    rand_protocol = f"random_{mode}"

    by_pair: dict[tuple[int, int, int], dict[str, dict]] = defaultdict(dict)
    for row in pair_summary:
        key = (int(row["asset_index"]), int(row["coord_id1"]), int(row["coord_id2"]))
        by_pair[key][str(row["protocol"])] = row

    best_better: dict | None = None
    best_worse: dict | None = None
    for key, rows in by_pair.items():
        if cond_protocol not in rows or rand_protocol not in rows:
            continue
        cond = rows[cond_protocol]
        rand = rows[rand_protocol]
        diff = float(cond["mean_reward"]) - float(rand["mean_reward"])
        item = {
            "asset_index": key[0],
            "coord_id1": key[1],
            "coord_id2": key[2],
            "cond": cond,
            "rand": rand,
            "diff": diff,
        }
        if best_better is None or diff > float(best_better["diff"]):
            best_better = item
        if best_worse is None or diff < float(best_worse["diff"]):
            best_worse = item

    if best_better is None or best_worse is None:
        raise RuntimeError(f"Could not find comparable {mode} cases in pair summary.")
    return best_better, best_worse


def _pair_index_map(pairs_manifest: dict) -> tuple[dict[tuple[int, int, int], int], dict[int, dict]]:
    pair_idx_map: dict[tuple[int, int, int], int] = {}
    asset_meta: dict[int, dict] = {}
    for asset in pairs_manifest["assets"]:
        local_asset_index = int(asset["local_asset_index"])
        asset_meta[local_asset_index] = asset
        for pair_idx, pair in enumerate(asset["pairs"]):
            pair_idx_map[(local_asset_index, int(pair["coord_id1"]), int(pair["coord_id2"]))] = int(pair_idx)
    return pair_idx_map, asset_meta


def _write_case(case: dict, *, label: str, mode: str, pair_idx_map: dict, asset_meta: dict, output_dir: Path) -> None:
    asset_index = int(case["asset_index"])
    coord_id1 = int(case["coord_id1"])
    coord_id2 = int(case["coord_id2"])
    pair_idx = pair_idx_map[(asset_index, coord_id1, coord_id2)]
    asset = asset_meta[asset_index]

    pair_row = None
    for row in asset["pairs"]:
        if int(row["coord_id1"]) == coord_id1 and int(row["coord_id2"]) == coord_id2:
            pair_row = row
            break
    if pair_row is None:
        raise RuntimeError(f"Missing pair row for asset_index={asset_index} coord=({coord_id1},{coord_id2})")

    asset_manifest = [
        {
            "asset_id": int(asset["asset_id"]),
            "asset_path": str(asset["asset_path"]),
        }
    ]
    pair_manifest = {
        "assets": [
            {
                "local_asset_index": 0,
                "asset_index": int(asset["asset_id"]),
                "asset_id": int(asset["asset_id"]),
                "asset_path": str(asset["asset_path"]),
                "pairs": [
                    {
                        "coord_id1": int(pair_row["coord_id1"]),
                        "coord_id2": int(pair_row["coord_id2"]),
                        "raw_id1": int(pair_row["raw_id1"]),
                        "raw_id2": int(pair_row["raw_id2"]),
                        "distance": float(pair_row["distance"]),
                        "bin_idx": int(pair_row["bin_idx"]),
                    }
                ],
            }
        ]
    }
    meta = {
        "mode": mode,
        "label": label,
        "pilot_local_asset_index": asset_index,
        "pilot_pair_index": pair_idx,
        "asset_id": int(asset["asset_id"]),
        "asset_path": str(asset["asset_path"]),
        "coord_id1": coord_id1,
        "coord_id2": coord_id2,
        "raw_id1": int(pair_row["raw_id1"]),
        "raw_id2": int(pair_row["raw_id2"]),
        "distance": float(pair_row["distance"]),
        "bin_idx": int(pair_row["bin_idx"]),
        "cond_mean_reward": float(case["cond"]["mean_reward"]),
        "rand_mean_reward": float(case["rand"]["mean_reward"]),
        "diff": float(case["diff"]),
    }

    stem = f"{mode}_{label}"
    (output_dir / f"{stem}_asset.json").write_text(json.dumps(asset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{stem}_pair.json").write_text(json.dumps(pair_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{stem}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{label}] asset_id={meta['asset_id']} pair_idx={pair_idx} coord=({coord_id1},{coord_id2}) diff={meta['diff']:.6f}")
    print(f"  asset: {output_dir / f'{stem}_asset.json'}")
    print(f"  pair : {output_dir / f'{stem}_pair.json'}")


def run(args) -> None:
    pair_summary = json.loads(Path(args.pair_summary).read_text(encoding="utf-8"))
    pairs_manifest = json.loads(Path(args.pairs_manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_better, best_worse = _find_extreme_cases(pair_summary, args.mode)
    pair_idx_map, asset_meta = _pair_index_map(pairs_manifest)
    _write_case(best_better, label="cond_better", mode=args.mode, pair_idx_map=pair_idx_map, asset_meta=asset_meta, output_dir=output_dir)
    _write_case(best_worse, label="cond_worse", mode=args.mode, pair_idx_map=pair_idx_map, asset_meta=asset_meta, output_dir=output_dir)


if __name__ == "__main__":
    run(build_parser().parse_args())
