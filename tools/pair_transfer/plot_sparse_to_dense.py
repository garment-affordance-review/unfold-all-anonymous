#!/usr/bin/env python3
"""Plot concise figures for sparse-to-dense pair transfer results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sparse-to-dense transfer results.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory with per_asset_metrics.jsonl and summary.json")
    return parser.parse_args()


def _load_rows(run_dir: Path) -> list[dict]:
    rows_path = run_dir / "per_asset_metrics.jsonl"
    return [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _boxplot_best_rewards(rows: list[dict], out_path: Path) -> None:
    data = [
        [r["offline_best_reward"] for r in rows],
        [r["random_best_mean"] for r in rows],
        [r["model_top1_reward"] for r in rows],
        [r["model_topk_best_reward"] for r in rows],
        [r["oracle_top1_reward"] for r in rows],
    ]
    labels = ["Offline best\n(budget=64)", "Random best\n(hidden top-10)", "Model top-1", "Model top-10\nbest", "Oracle top-1"]
    colors = ["#9aa1a9", "#d98e04", "#4e79a7", "#2a9d8f", "#264653"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("True reward (higher is better)")
    ax.set_title("Per-asset best reward comparison on unseen assets")
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _gain_hist(rows: list[dict], out_path: Path) -> None:
    gain_offline = np.asarray([r["model_gain_vs_offline_best"] for r in rows], dtype=np.float32)
    gain_random = np.asarray([r["model_gain_vs_random_best"] for r in rows], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(gain_offline, bins=30, color="#2a9d8f", alpha=0.85)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Model top-10 best minus offline best")
    axes[0].set_xlabel("Reward gain")
    axes[0].set_ylabel("Asset count")

    axes[1].hist(gain_random, bins=30, color="#4e79a7", alpha=0.85)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Model top-10 best minus random hidden top-10")
    axes[1].set_xlabel("Reward gain")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _win_bar(rows: list[dict], out_path: Path) -> None:
    total = len(rows)
    wins = {
        "Model > offline best": sum(r["model_gain_vs_offline_best"] > 0 for r in rows),
        "Model > random hidden top-10": sum(r["model_gain_vs_random_best"] > 0 for r in rows),
        "Model top-1 > offline best": sum(r["model_top1_reward"] > r["offline_best_reward"] for r in rows),
    }
    labels = list(wins.keys())
    values = [wins[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#2a9d8f", "#4e79a7", "#8ab17d"])
    ax.set_ylim(0, total)
    ax.set_ylabel(f"Assets (out of {total})")
    ax.set_title("How often the model wins on unseen assets")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value}/{total}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    rows = _load_rows(run_dir)
    _boxplot_best_rewards(rows, run_dir / "best_reward_comparison.png")
    _gain_hist(rows, run_dir / "gain_histograms.png")
    _win_bar(rows, run_dir / "win_counts.png")
    print(f"Saved plots under {run_dir}")


if __name__ == "__main__":
    main()
