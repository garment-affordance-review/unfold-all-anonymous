#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ClothMate and pair-policy aRTF zero-shot summaries.")
    parser.add_argument(
        "--pair-policy-dir",
        type=Path,
        default=Path("experiments/aRTFClothes/analysis/pair_policy_keypoint_eval_full"),
    )
    parser.add_argument(
        "--clothmate-dir",
        type=Path,
        default=Path("experiments/aRTFClothes/analysis/clothmate_keypoint_eval_full"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/aRTFClothes/analysis/clothmate_vs_pair_policy_full"),
    )
    return parser.parse_args()


def load_summary(root: Path, split: str) -> dict[str, Any]:
    path = root / f"summary_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(model: str, split: str, category: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "split": split,
        "category": category,
        "images_used": stats.get("images_used", 0),
        "norm_dist_x1_mean": stats.get("norm_dist_x1_mean"),
        "norm_dist_x1_median": stats.get("norm_dist_x1_median"),
        "norm_dist_x2_mean": stats.get("norm_dist_x2_mean"),
        "norm_dist_x2_median": stats.get("norm_dist_x2_median"),
        "min_dist_x1_mean": stats.get("min_dist_x1_mean"),
        "min_dist_x2_mean": stats.get("min_dist_x2_mean"),
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    comparison: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for split in ["train", "test", "combined"]:
        pair_summary = load_summary(args.pair_policy_dir, split)
        cloth_summary = load_summary(args.clothmate_dir, split)
        comparison[split] = {
            "pair_policy": pair_summary,
            "clothmate": cloth_summary,
        }
        rows.append(metric_row("pair_policy", split, "total", pair_summary["total"]))
        rows.append(metric_row("clothmate", split, "total", cloth_summary["total"]))
        categories = sorted(set(pair_summary.get("categories", {}).keys()) | set(cloth_summary.get("categories", {}).keys()))
        for category in categories:
            if category in pair_summary.get("categories", {}):
                rows.append(metric_row("pair_policy", split, category, pair_summary["categories"][category]))
            if category in cloth_summary.get("categories", {}):
                rows.append(metric_row("clothmate", split, category, cloth_summary["categories"][category]))

    (args.out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    with (args.out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
