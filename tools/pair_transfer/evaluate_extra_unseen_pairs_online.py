#!/usr/bin/env python3
"""Evaluate extra unseen pairs on seen assets with online reward execution.

Protocol:
1. Use the existing 512 collected pairs per asset as the observed sampled set.
2. Generate an additional unseen candidate pool on the same asset that does not
   overlap with the observed pairs.
3. Score the unseen pool with the trained Pointcept model.
4. Execute:
   - predicted-high pairs: top-N by predicted reward
   - predicted-quantile pairs: fixed samples from predicted score quantiles
5. Compare predicted vs actual reward on executed unseen pairs, and compare the
   top-K unseen rewards against the original observed 512-pair rewards.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

from unfold.algorithms.supervision.teacher_pointcept import TeacherRewardInfer
from unfold.workflows.offline_collection.pair_conditioned_collect import (
    PairCandidate,
    PairConditionedOfflineCollector,
    _farthest_point_sample_np,
    load_pair_conditioned_env_and_cfg,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online evaluation on extra unseen pairs from seen assets.")
    parser.add_argument("--data-root", type=str, default="${PROJECT_ROOT}/data/clothes")
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="${TEACHER_EXP_ROOT}",
    )
    parser.add_argument("--pointcept-root", type=str, default="${POINTCEPT_ROOT}")
    parser.add_argument("--config", type=str, default="configs/offline_pair_conditioned.yaml")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "experiments/offline_teacher_eval/runs/default"))
    parser.add_argument("--vis-dir", type=str, default=None)
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--assets-manifest", type=str, default=None)
    parser.add_argument("--max-assets", type=int, default=20)
    parser.add_argument("--asset-start", type=int, default=0)
    parser.add_argument("--asset-order", type=str, default="sequential", choices=["sequential", "shuffle"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-abs-max", type=float, default=2.0)
    parser.add_argument("--candidate-fps-count", type=int, default=192)
    parser.add_argument("--distance-bins", type=int, default=4)
    parser.add_argument("--predicted-high-per-distance-bin", type=int, default=4)
    parser.add_argument("--score-quantile-bins", type=int, default=4)
    parser.add_argument("--quantile-per-cell", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-pairs-per-forward", type=int, default=65536)
    parser.add_argument("--overwrite", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _load_observed_asset(asset_dir: Path, reward_abs_max: float) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.load(asset_dir / "pairs.npy").astype(np.int64)
    reward = np.load(asset_dir / "reward.npy").astype(np.float32)
    mask = np.isfinite(reward)
    if reward_abs_max > 0:
        mask &= np.abs(reward) <= reward_abs_max
    return pairs[mask], reward[mask]


def _normalize_pair(pair: np.ndarray | tuple[int, int]) -> tuple[int, int]:
    a = int(pair[0])
    b = int(pair[1])
    return (a, b) if a <= b else (b, a)


def _build_unseen_candidate_pairs(
    pointcloud,
    observed_pairs: np.ndarray,
    candidate_fps_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord = pointcloud.coord
    if coord.shape[0] <= 1:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    rng = np.random.default_rng(0)
    anchor_ids = _farthest_point_sample_np(coord, min(int(candidate_fps_count), int(coord.shape[0])), rng)
    observed_set = {_normalize_pair(pair) for pair in observed_pairs.tolist()}
    pairs: list[tuple[int, int]] = []
    dists: list[float] = []
    for i in range(anchor_ids.shape[0] - 1):
        a = int(anchor_ids[i])
        for j in range(i + 1, anchor_ids.shape[0]):
            b = int(anchor_ids[j])
            key = (a, b) if a <= b else (b, a)
            if key in observed_set:
                continue
            pairs.append(key)
            dists.append(float(np.linalg.norm(coord[a] - coord[b])))
    if not pairs:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    pair_arr = np.asarray(pairs, dtype=np.int64)
    dist_arr = np.asarray(dists, dtype=np.float32)
    return pair_arr, dist_arr, np.zeros((pair_arr.shape[0],), dtype=np.int64)


def _assign_distance_bins(distances: np.ndarray, distance_bins: int) -> np.ndarray:
    n = int(distances.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    bin_count = max(1, int(distance_bins))
    if bin_count == 1:
        return np.zeros((n,), dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.quantile(distances, quantiles)
    edges[0] = min(float(edges[0]), float(distances.min()))
    edges[-1] = max(float(edges[-1]), float(distances.max()) + 1e-6)
    out = np.searchsorted(edges, distances, side="right") - 1
    return np.clip(out, 0, bin_count - 1).astype(np.int64, copy=False)


def _select_eval_pairs(
    candidate_pairs: np.ndarray,
    pred_scores: np.ndarray,
    distance_bin_ids: np.ndarray,
    distance_bins: int,
    high_per_distance_bin: int,
    score_quantile_bins: int,
    quantile_per_cell: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    selected: list[int] = []
    labels: list[str] = []
    used: set[int] = set()

    rng = np.random.default_rng(seed)
    for dist_bin in range(max(1, int(distance_bins))):
        bin_idx = np.flatnonzero(distance_bin_ids == dist_bin)
        if bin_idx.size == 0:
            continue
        local_order = bin_idx[np.argsort(pred_scores[bin_idx])[::-1]]

        high_n = min(int(high_per_distance_bin), int(local_order.shape[0]))
        for idx in local_order[:high_n].tolist():
            idx = int(idx)
            if idx in used:
                continue
            used.add(idx)
            selected.append(idx)
            labels.append(f"predicted-high-db{dist_bin}")

        remain = [int(idx) for idx in local_order.tolist() if int(idx) not in used]
        if not remain or score_quantile_bins <= 0 or quantile_per_cell <= 0:
            continue
        remain = list(remain)
        bin_edges = np.linspace(0, len(remain), int(score_quantile_bins) + 1, dtype=int)
        for score_bin in range(int(score_quantile_bins)):
            lo = int(bin_edges[score_bin])
            hi = int(bin_edges[score_bin + 1])
            pool = remain[lo:hi]
            if not pool:
                continue
            take = min(int(quantile_per_cell), len(pool))
            chosen = rng.choice(np.asarray(pool, dtype=np.int64), size=take, replace=False)
            for idx in chosen.tolist():
                idx = int(idx)
                if idx in used:
                    continue
                used.add(idx)
                selected.append(idx)
                labels.append(f"predicted-quantile-db{dist_bin}-q{score_bin}")

    return candidate_pairs[np.asarray(selected, dtype=np.int64)], labels


def _candidate_with_distance(collector: PairConditionedOfflineCollector, pointcloud, pair: np.ndarray, bin_idx: int) -> PairCandidate | None:
    a = int(pair[0])
    b = int(pair[1])
    dist = float(np.linalg.norm(pointcloud.coord[a] - pointcloud.coord[b]))
    return collector._build_pair_candidate(pointcloud, a, b, dist, bin_idx)


def _execute_pairs_online(
    collector: PairConditionedOfflineCollector,
    asset_seq_id: int,
    asset_id: int,
    pointcloud,
    eval_pairs: np.ndarray,
) -> np.ndarray:
    rewards_out: list[float] = []
    num_envs = int(collector.env_cfg.scene.num_envs)
    for start in range(0, int(eval_pairs.shape[0]), num_envs):
        batch_pairs = eval_pairs[start : start + num_envs]
        collector._reset_single_asset(asset_seq_id, asset_id)
        candidates: list[PairCandidate] = []
        for pair in batch_pairs:
            candidate = _candidate_with_distance(collector, pointcloud, pair, bin_idx=0)
            if candidate is None:
                continue
            candidates.append(candidate)
        if not candidates:
            continue

        actions = np.full((num_envs, 2), -1, dtype=np.int64)
        for env_idx, candidate in enumerate(candidates):
            actions[env_idx, 0] = candidate.raw_id1
            actions[env_idx, 1] = candidate.raw_id2
        collector._apply_pair_conditioned_poses(candidates)
        actions_t = __import__("torch").from_numpy(actions).to(device=collector.device, dtype=__import__("torch").long)
        _, rewards, _, _, _ = collector.env.unwrapped.step(actions_t)
        reward_np = rewards.detach().cpu().numpy().reshape(-1)
        rewards_out.extend(float(reward_np[i]) for i in range(len(candidates)))
    return np.asarray(rewards_out, dtype=np.float32)


def _plot_summary(per_asset: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = np.concatenate([np.asarray(r["eval_pred"], dtype=np.float32) for r in per_asset], axis=0)
    gt = np.concatenate([np.asarray(r["eval_gt"], dtype=np.float32) for r in per_asset], axis=0)
    residual = pred - gt

    lim = float(np.percentile(np.abs(np.concatenate([pred, gt])), 99))
    lim = max(lim, 1e-3)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(gt, pred, s=4, alpha=0.25)
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1)
    ax.set_xlabel("GT reward on extra unseen pairs")
    ax.set_ylabel("Predicted reward")
    ax.set_title("Extra unseen-pair prediction")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_gt_scatter.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=60, color="#4e79a7", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (pred - gt)")
    ax.set_ylabel("Count")
    ax.set_title("Residuals on executed unseen pairs")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_hist.png", dpi=180)
    plt.close(fig)

    data = [
        [r["observed_best_reward"] for r in per_asset],
        [r["topk_unseen_gt_best"] for r in per_asset],
        [r["topk_unseen_gt_mean"] for r in per_asset],
    ]
    labels = ["Observed best\n(original 512)", "Pred top-K unseen\nbest", "Pred top-K unseen\nmean"]
    colors = ["#9aa1a9", "#2a9d8f", "#4e79a7"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Reward (higher is better)")
    ax.set_title("Can predicted unseen pairs beat the original sampled set?")
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "topk_vs_observed.png", dpi=180)
    plt.close(fig)


def _write_partial_outputs(
    per_asset: list[dict[str, Any]],
    out_dir: Path,
    args,
    data_root: Path,
    exp_dir: Path,
) -> None:
    if not per_asset:
        return
    (out_dir / "per_asset_metrics.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in per_asset) + "\n",
        encoding="utf-8",
    )
    summary = {
        "setting": "seen-asset / extra-unseen-pair / online-gt",
        "data_root": str(data_root),
        "exp_dir": str(exp_dir),
        "num_assets": len(per_asset),
        "candidate_fps_count": int(args.candidate_fps_count),
        "distance_bins": int(args.distance_bins),
        "predicted_high_per_distance_bin": int(args.predicted_high_per_distance_bin),
        "score_quantile_bins": int(args.score_quantile_bins),
        "quantile_per_cell": int(args.quantile_per_cell),
        "top_k": int(args.top_k),
        "observed_best_reward_mean": float(np.mean([r["observed_best_reward"] for r in per_asset])),
        "topk_unseen_gt_best_mean": float(np.mean([r["topk_unseen_gt_best"] for r in per_asset])),
        "topk_unseen_gt_mean_reward_mean": float(np.mean([r["topk_unseen_gt_mean"] for r in per_asset])),
        "gain_vs_observed_best_mean": float(np.mean([r["gain_vs_observed_best"] for r in per_asset])),
        "assets_unseen_topk_beats_observed_best": int(sum(r["gain_vs_observed_best"] > 0 for r in per_asset)),
        "eval_mae_mean": float(np.mean([r["eval_mae"] for r in per_asset])),
        "eval_rmse_mean": float(np.mean([r["eval_rmse"] for r in per_asset])),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_summary(per_asset, out_dir)
    result_note = f"""# Result Note

Setting:

- seen assets;
- original observed set = existing 512 sampled pairs per asset;
- extra unseen pool = newly generated pairs not overlapping the original 512;
- GT for unseen pairs comes from actual online execution.

Selection:

- distance bins: `{int(args.distance_bins)}` quantile bins over candidate pair distance;
- predicted-high: top `{int(args.predicted_high_per_distance_bin)}` within each distance bin;
- predicted-quantile: `{int(args.quantile_per_cell)}` samples from each of `{int(args.score_quantile_bins)}` score bins within each distance bin.

Core metrics:

- mean observed best reward: `{summary['observed_best_reward_mean']:.4f}`
- mean unseen top-{int(args.top_k)} best reward: `{summary['topk_unseen_gt_best_mean']:.4f}`
- mean gain vs observed best: `{summary['gain_vs_observed_best_mean']:.4f}`
- assets where unseen top-{int(args.top_k)} beats observed best: `{summary['assets_unseen_topk_beats_observed_best']}/{summary['num_assets']}`
- mean online eval MAE on executed unseen pairs: `{summary['eval_mae_mean']:.4f}`
- mean online eval RMSE on executed unseen pairs: `{summary['eval_rmse_mean']:.4f}`
"""
    (out_dir / "RESULT_NOTE.md").write_text(result_note, encoding="utf-8")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.vis_dir is None:
        args.vis_dir = str(out_dir / "vis")
    data_root = Path(args.data_root).resolve()
    exp_dir = Path(args.exp_dir).resolve()

    teacher = TeacherRewardInfer(
        teacher_cfg=str(exp_dir / "config.py"),
        teacher_ckpt=str(exp_dir / "model" / "model_best.pth"),
        pointcept_code_root=str(Path(args.pointcept_root).resolve()),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    env, env_cfg = load_pair_conditioned_env_and_cfg(args)
    collector = PairConditionedOfflineCollector(env, env_cfg, args)

    per_asset: list[dict[str, Any]] = []
    try:
        total_assets = len(collector._asset_paths)
        asset_indices = list(range(total_assets))
        if args.asset_order == "shuffle":
            np.random.default_rng(int(args.seed)).shuffle(asset_indices)
        start = int(args.asset_start)
        stop = min(total_assets, start + int(args.max_assets))
        for asset_seq_id in asset_indices[start:stop]:
            asset_id = int(collector._asset_ids[asset_seq_id])
            asset_dir = data_root / "assets" / f"asset_{asset_id:04d}"
            if not asset_dir.is_dir():
                continue

            collector._reset_single_asset(asset_seq_id, asset_id)
            pointcloud = collector._prepare_pointcloud(0)
            observed_pairs, observed_rewards = _load_observed_asset(asset_dir, float(args.reward_abs_max))
            if observed_pairs.shape[0] == 0:
                continue

            candidate_pairs, candidate_distances, _ = _build_unseen_candidate_pairs(
                pointcloud=pointcloud,
                observed_pairs=observed_pairs,
                candidate_fps_count=int(args.candidate_fps_count),
            )
            if candidate_pairs.shape[0] == 0:
                continue
            distance_bin_ids = _assign_distance_bins(candidate_distances, int(args.distance_bins))

            pred_scores = teacher.infer_pairs(
                coord=pointcloud.coord,
                normal=pointcloud.normal,
                pairs=candidate_pairs,
                max_pairs_per_forward=int(args.max_pairs_per_forward),
            )
            eval_pairs, eval_labels = _select_eval_pairs(
                candidate_pairs=candidate_pairs,
                pred_scores=pred_scores,
                distance_bin_ids=distance_bin_ids,
                distance_bins=int(args.distance_bins),
                high_per_distance_bin=int(args.predicted_high_per_distance_bin),
                score_quantile_bins=int(args.score_quantile_bins),
                quantile_per_cell=int(args.quantile_per_cell),
                seed=int(args.seed) + asset_id * 101,
            )
            if eval_pairs.shape[0] == 0:
                continue

            eval_pred = []
            pred_lookup = {tuple(map(int, pair)): float(score) for pair, score in zip(candidate_pairs.tolist(), pred_scores.tolist())}
            for pair in eval_pairs.tolist():
                eval_pred.append(pred_lookup[tuple(map(int, pair))])
            eval_gt = _execute_pairs_online(
                collector=collector,
                asset_seq_id=asset_seq_id,
                asset_id=asset_id,
                pointcloud=pointcloud,
                eval_pairs=eval_pairs,
            )
            if eval_gt.shape[0] != len(eval_pred):
                raise RuntimeError(
                    f"Mismatch between executed rewards and predicted eval pairs for asset {asset_id}: "
                    f"{eval_gt.shape[0]} vs {len(eval_pred)}"
                )

            high_idx = [i for i, label in enumerate(eval_labels) if label.startswith("predicted-high-db")]
            topk = min(int(args.top_k), len(high_idx))
            if topk <= 0:
                continue
            high_gt = eval_gt[high_idx]
            high_pred = np.asarray([eval_pred[i] for i in high_idx], dtype=np.float32)
            order_high = np.argsort(high_pred)[::-1][:topk]
            topk_unseen_gt = high_gt[order_high]

            row = {
                "asset_id": asset_id,
                "asset_path": pointcloud.asset_path,
                "num_observed_pairs": int(observed_pairs.shape[0]),
                "num_unseen_candidates": int(candidate_pairs.shape[0]),
                "num_executed_pairs": int(eval_pairs.shape[0]),
                "observed_best_reward": float(np.max(observed_rewards)),
                "observed_topk_mean_reward": float(np.mean(np.sort(observed_rewards)[-topk:])),
                "topk_unseen_gt_best": float(np.max(topk_unseen_gt)),
                "topk_unseen_gt_mean": float(np.mean(topk_unseen_gt)),
                "gain_vs_observed_best": float(np.max(topk_unseen_gt) - np.max(observed_rewards)),
                "eval_mae": float(np.mean(np.abs(np.asarray(eval_pred, dtype=np.float32) - eval_gt))),
                "eval_rmse": float(np.sqrt(np.mean((np.asarray(eval_pred, dtype=np.float32) - eval_gt) ** 2))),
                "eval_pred": [float(x) for x in eval_pred],
                "eval_gt": [float(x) for x in eval_gt.tolist()],
                "eval_labels": list(eval_labels),
            }
            per_asset.append(row)
            print(
                f"[ASSET] {asset_id:04d} observed_best={row['observed_best_reward']:.4f} "
                f"unseen_topk_best={row['topk_unseen_gt_best']:.4f} "
                f"gain={row['gain_vs_observed_best']:.4f} candidates={row['num_unseen_candidates']}",
                flush=True,
            )
            _write_partial_outputs(
                per_asset=per_asset,
                out_dir=out_dir,
                args=args,
                data_root=data_root,
                exp_dir=exp_dir,
            )
    finally:
        _write_partial_outputs(
            per_asset=per_asset,
            out_dir=out_dir,
            args=args,
            data_root=data_root,
            exp_dir=exp_dir,
        )
        collector.env.close()
        simulation_app.close()

    if not per_asset:
        raise RuntimeError("No assets evaluated.")

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
