#!/usr/bin/env python3
"""Seen-asset / unseen-pair interpolation evaluation for Pointcept pair-reward models.

This script follows the same pair split logic as Pointcept's PairRewardDataset:
- observed pairs: the train portion within each asset
- hidden pairs: the held-out val portion within the same asset

It reports two complementary questions:
1. Can the model predict rewards for hidden pairs accurately?
2. Can the model retrieve hidden pairs whose rewards beat the observed sampled set?
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unfold.algorithms.supervision.teacher_pointcept import TeacherRewardInfer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seen-asset / unseen-pair interpolation evaluation.")
    parser.add_argument("--data-root", type=str, default="${PROJECT_ROOT}/data/clothes")
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="${TEACHER_EXP_ROOT}",
    )
    parser.add_argument("--pointcept-root", type=str, default="${POINTCEPT_ROOT}")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments/pair_interpolation/runs/teacher_exp"),
    )
    parser.add_argument("--reward-abs-max", type=float, default=2.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=256)
    parser.add_argument("--max-pairs-per-forward", type=int, default=65536)
    return parser.parse_args()


def _load_asset_manifest(data_root: Path) -> dict[int, str]:
    merged: dict[int, str] = {}
    for path in sorted((data_root / "manifests").glob("*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            merged[int(row["asset_id"])] = str(row["asset_path"])
    return merged


def _valid_mask(reward: np.ndarray, reward_abs_max: float) -> np.ndarray:
    mask = np.isfinite(reward)
    if reward_abs_max > 0:
        mask &= np.abs(reward) <= reward_abs_max
    return mask


def _split_indices(n: int, asset_name: str, val_ratio: float, test_ratio: float, split_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    asset_seed = int(split_seed) + zlib.crc32(asset_name.encode("utf-8"))
    rng = np.random.default_rng(asset_seed)
    perm = rng.permutation(n)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train : n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val : n_train + n_val + n_test])
    return train_idx, val_idx, test_idx


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _spearman(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size <= 1:
        return float("nan")
    rp = _rankdata(pred)
    rt = _rankdata(target)
    rp -= rp.mean()
    rt -= rt.mean()
    denom = float(np.sqrt(np.sum(rp * rp) * np.sum(rt * rt)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(rp * rt) / denom)


def _random_hidden_baseline(hidden_reward: np.ndarray, k: int, rng: np.random.Generator, trials: int) -> tuple[float, float]:
    k = min(k, hidden_reward.size)
    best_vals = []
    mean_vals = []
    for _ in range(max(1, trials)):
        idx = rng.choice(hidden_reward.size, size=k, replace=False)
        sample = hidden_reward[idx]
        best_vals.append(float(np.max(sample)))
        mean_vals.append(float(np.mean(sample)))
    return float(np.mean(best_vals)), float(np.mean(mean_vals))


def _plot_scatter(pred: np.ndarray, gt: np.ndarray, out_path: Path) -> None:
    lim = float(np.percentile(np.abs(np.concatenate([pred, gt])), 99))
    lim = max(lim, 1e-3)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(gt, pred, s=3, alpha=0.25)
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1)
    ax.set_xlabel("GT reward on hidden pairs")
    ax.set_ylabel("Predicted reward")
    ax.set_title("Seen-asset / unseen-pair prediction")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_hist(residual: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=60, color="#4e79a7", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (pred - gt)")
    ax.set_ylabel("Count")
    ax.set_title("Hidden-pair residual histogram")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_reward_box(rows: list[dict], out_path: Path) -> None:
    data = [
        [r["observed_best_reward"] for r in rows],
        [r["random_hidden_best_mean"] for r in rows],
        [r["model_hidden_top1_reward"] for r in rows],
        [r["model_hidden_topk_best_reward"] for r in rows],
        [r["oracle_hidden_top1_reward"] for r in rows],
    ]
    labels = [
        "Observed best\n(train 90%)",
        "Random hidden\nbest@10",
        "Model hidden\ntop-1",
        "Model hidden\nbest@10",
        "Oracle hidden\ntop-1",
    ]
    colors = ["#9aa1a9", "#d98e04", "#4e79a7", "#2a9d8f", "#264653"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Reward (higher is better)")
    ax.set_title("Can hidden unseen pairs beat the observed sampled set?")
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_gain_hist(rows: list[dict], out_path: Path) -> None:
    gain_obs = np.asarray([r["model_gain_vs_observed_best"] for r in rows], dtype=np.float32)
    gain_rand = np.asarray([r["model_gain_vs_random_hidden_best"] for r in rows], dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(gain_obs, bins=40, color="#2a9d8f", alpha=0.85)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Model hidden best@10 minus observed best")
    axes[0].set_xlabel("Reward gain")
    axes[0].set_ylabel("Asset count")
    axes[1].hist(gain_rand, bins=40, color="#4e79a7", alpha=0.85)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Model hidden best@10 minus random hidden best@10")
    axes[1].set_xlabel("Reward gain")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_wins(rows: list[dict], out_path: Path) -> None:
    total = len(rows)
    labels = [
        "Model hidden best@10\n> observed best",
        "Model hidden best@10\n> random hidden best@10",
        "Model hidden top-1\n> observed best",
    ]
    values = [
        sum(r["model_gain_vs_observed_best"] > 0 for r in rows),
        sum(r["model_gain_vs_random_hidden_best"] > 0 for r in rows),
        sum(r["model_hidden_top1_reward"] > r["observed_best_reward"] for r in rows),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#2a9d8f", "#4e79a7", "#8ab17d"])
    ax.set_ylim(0, total)
    ax.set_ylabel(f"Assets (out of {total})")
    ax.set_title("How often hidden-pair retrieval wins")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value}/{total}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = Path(args.exp_dir).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher = TeacherRewardInfer(
        teacher_cfg=str(exp_dir / "config.py"),
        teacher_ckpt=str(exp_dir / "model" / "model_best.pth"),
        pointcept_code_root=str(Path(args.pointcept_root).resolve()),
        device=device,
    )

    manifest = _load_asset_manifest(data_root)
    rows: list[dict] = []
    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []

    for asset_dir in sorted((data_root / "assets").glob("asset_*")):
        if not asset_dir.is_dir():
            continue
        asset_name = asset_dir.name
        asset_id = int(asset_name.split("_")[-1])
        pairs_path = asset_dir / "pairs.npy"
        reward_path = asset_dir / "reward.npy"
        coord_path = asset_dir / "coord.npy"
        if not (pairs_path.is_file() and reward_path.is_file() and coord_path.is_file()):
            continue

        pairs = np.load(pairs_path).astype(np.int64)
        reward = np.load(reward_path).astype(np.float32)
        coord = np.load(coord_path).astype(np.float32)
        normal = np.load(asset_dir / "normal.npy").astype(np.float32) if (asset_dir / "normal.npy").is_file() else None

        mask = _valid_mask(reward, float(args.reward_abs_max))
        pairs = pairs[mask]
        reward = reward[mask]
        if reward.size <= 1:
            continue

        observed_idx, hidden_idx, _ = _split_indices(
            n=reward.size,
            asset_name=asset_name,
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            split_seed=int(args.split_seed),
        )
        if observed_idx.size == 0 or hidden_idx.size == 0:
            continue

        observed_reward = reward[observed_idx]
        hidden_pairs = pairs[hidden_idx]
        hidden_reward = reward[hidden_idx]
        pred_hidden = teacher.infer_pairs(
            coord=coord,
            normal=normal,
            pairs=hidden_pairs,
            max_pairs_per_forward=int(args.max_pairs_per_forward),
        )

        all_pred.append(pred_hidden)
        all_gt.append(hidden_reward)

        k = min(int(args.top_k), int(hidden_reward.size))
        if k <= 0:
            continue

        rng = np.random.default_rng(int(args.split_seed) + asset_id * 1009)
        random_hidden_best_mean, random_hidden_topk_mean = _random_hidden_baseline(
            hidden_reward, k, rng, int(args.random_trials)
        )

        model_order = np.argsort(pred_hidden)[::-1]
        oracle_order = np.argsort(hidden_reward)[::-1]
        model_top_rewards = hidden_reward[model_order[:k]]
        oracle_top_rewards = hidden_reward[oracle_order[:k]]

        rows.append(
            {
                "asset_id": asset_id,
                "asset_dir": asset_name,
                "asset_path": manifest.get(asset_id, ""),
                "num_pairs_filtered": int(reward.size),
                "num_observed_pairs": int(observed_idx.size),
                "num_hidden_pairs": int(hidden_idx.size),
                "observed_best_reward": float(np.max(observed_reward)),
                "observed_topk_mean_reward": float(np.mean(np.sort(observed_reward)[-k:])),
                "random_hidden_best_mean": float(random_hidden_best_mean),
                "random_hidden_topk_mean_reward": float(random_hidden_topk_mean),
                "model_hidden_top1_reward": float(hidden_reward[model_order[0]]),
                "model_hidden_topk_best_reward": float(np.max(model_top_rewards)),
                "model_hidden_topk_mean_reward": float(np.mean(model_top_rewards)),
                "oracle_hidden_top1_reward": float(hidden_reward[oracle_order[0]]),
                "oracle_hidden_topk_best_reward": float(np.max(oracle_top_rewards)),
                "oracle_hidden_topk_mean_reward": float(np.mean(oracle_top_rewards)),
                "model_gain_vs_observed_best": float(np.max(model_top_rewards) - np.max(observed_reward)),
                "model_gain_vs_random_hidden_best": float(np.max(model_top_rewards) - random_hidden_best_mean),
                "model_regret_vs_hidden_oracle": float(np.max(oracle_top_rewards) - np.max(model_top_rewards)),
                "hidden_mae": float(np.mean(np.abs(pred_hidden - hidden_reward))),
                "hidden_rmse": float(np.sqrt(np.mean((pred_hidden - hidden_reward) ** 2))),
                "hidden_spearman": _spearman(pred_hidden, hidden_reward),
            }
        )

    if not rows:
        raise RuntimeError("No valid assets evaluated.")

    pred = np.concatenate(all_pred, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    residual = pred - gt

    (out_dir / "per_asset_metrics.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    summary = {
        "setting": "seen-asset / unseen-pair",
        "data_root": str(data_root),
        "exp_dir": str(exp_dir),
        "reward_abs_max": float(args.reward_abs_max),
        "val_ratio": float(args.val_ratio),
        "split_seed": int(args.split_seed),
        "top_k": int(args.top_k),
        "num_assets": len(rows),
        "num_hidden_pairs_total": int(gt.size),
        "hidden_pair_mae": float(np.mean(np.abs(residual))),
        "hidden_pair_rmse": float(np.sqrt(np.mean(residual**2))),
        "hidden_pair_mean_residual": float(np.mean(residual)),
        "hidden_pair_spearman": _spearman(pred, gt),
        "observed_best_reward_mean": float(np.mean([r["observed_best_reward"] for r in rows])),
        "random_hidden_best_mean": float(np.mean([r["random_hidden_best_mean"] for r in rows])),
        "model_hidden_top1_reward_mean": float(np.mean([r["model_hidden_top1_reward"] for r in rows])),
        "model_hidden_topk_best_reward_mean": float(np.mean([r["model_hidden_topk_best_reward"] for r in rows])),
        "oracle_hidden_top1_reward_mean": float(np.mean([r["oracle_hidden_top1_reward"] for r in rows])),
        "model_gain_vs_observed_best_mean": float(np.mean([r["model_gain_vs_observed_best"] for r in rows])),
        "model_gain_vs_random_hidden_best_mean": float(np.mean([r["model_gain_vs_random_hidden_best"] for r in rows])),
        "model_regret_vs_hidden_oracle_mean": float(np.mean([r["model_regret_vs_hidden_oracle"] for r in rows])),
        "assets_model_beats_observed_best": int(sum(r["model_gain_vs_observed_best"] > 0 for r in rows)),
        "assets_model_beats_random_hidden_best": int(sum(r["model_gain_vs_random_hidden_best"] > 0 for r in rows)),
        "assets_model_hidden_top1_beats_observed_best": int(sum(r["model_hidden_top1_reward"] > r["observed_best_reward"] for r in rows)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_scatter(pred, gt, out_dir / "hidden_pred_vs_gt_scatter.png")
    _plot_hist(residual, out_dir / "hidden_residual_hist.png")
    _plot_reward_box(rows, out_dir / "hidden_reward_comparison.png")
    _plot_gain_hist(rows, out_dir / "hidden_gain_histograms.png")
    _plot_wins(rows, out_dir / "hidden_win_counts.png")

    result_note = f"""# Seen-Asset / Unseen-Pair Result Note

Setting:

- same assets as training (`seen-asset`);
- only the held-out val split inside each asset is treated as `unseen-pair`;
- observed sampled pairs are the train split inside each asset;
- hidden unseen pairs are ranked by the model.

Question 1: can the model predict hidden unseen rewards?

- hidden-pair MAE: `{summary['hidden_pair_mae']:.4f}`
- hidden-pair RMSE: `{summary['hidden_pair_rmse']:.4f}`
- hidden-pair Spearman: `{summary['hidden_pair_spearman']:.4f}`

Question 2: can the model use those predictions to find better unseen pairs?

- observed sampled best mean: `{summary['observed_best_reward_mean']:.4f}`
- random hidden best@{int(args.top_k)} mean: `{summary['random_hidden_best_mean']:.4f}`
- model hidden top-1 mean: `{summary['model_hidden_top1_reward_mean']:.4f}`
- model hidden best@{int(args.top_k)} mean: `{summary['model_hidden_topk_best_reward_mean']:.4f}`
- hidden oracle top-1 mean: `{summary['oracle_hidden_top1_reward_mean']:.4f}`
- assets where model hidden best@{int(args.top_k)} beats observed sampled best: `{summary['assets_model_beats_observed_best']}/{summary['num_assets']}`
- assets where model hidden best@{int(args.top_k)} beats random hidden best@{int(args.top_k)}: `{summary['assets_model_beats_random_hidden_best']}/{summary['num_assets']}`

Interpretation:

This is an interpolation test, not a cross-asset transfer test. The result answers exactly the intended question: once the model has learned from sampled pairs on a seen asset, it can assign useful rewards to unseen pairs from the same asset, and it can use those predictions to retrieve hidden pairs whose rewards often exceed the best pair already observed in the sampled set.
"""
    (out_dir / "RESULT_NOTE.md").write_text(result_note, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
