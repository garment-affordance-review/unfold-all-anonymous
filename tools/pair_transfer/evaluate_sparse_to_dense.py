#!/usr/bin/env python3
"""Evaluate whether a cross-asset reward model can beat sparse offline sampling.

Protocol:
1. Hold out test assets by asset id.
2. For each test asset, treat a small random subset of labeled pairs as the
   "offline budget" already collected.
3. Score the remaining hidden pairs with the trained Pointcept reward model.
4. Compare:
   - best reward already seen in the offline budget
   - random hidden top-K (Monte Carlo expectation)
   - model-ranked hidden top-K
   - oracle hidden top-K

This is an offline decision-quality experiment on unseen assets. It does not
re-run Isaac Sim; instead it uses stored pair rewards as the evaluation oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unfold.algorithms.supervision.teacher_pointcept import TeacherRewardInfer


@dataclass
class AssetEvalResult:
    asset_id: int
    asset_dir: str
    asset_path: str
    num_pairs_total: int
    num_pairs_visible_eval: int
    offline_budget: int
    hidden_pool: int
    top_k: int
    offline_best_reward: float
    offline_mean_reward: float
    random_best_mean: float
    random_topk_mean_reward: float
    model_top1_reward: float
    model_topk_best_reward: float
    model_topk_mean_reward: float
    oracle_top1_reward: float
    oracle_topk_best_reward: float
    oracle_topk_mean_reward: float
    model_gain_vs_offline_best: float
    model_gain_vs_random_best: float
    model_regret_vs_oracle_best: float
    spearman_hidden: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse-to-dense pair transfer evaluation.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="${PROJECT_ROOT}/data/clothes",
        help="Pointcept-format clothes dataset root.",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="${TEACHER_EXP_ROOT}",
        help="Trained Pointcept experiment directory.",
    )
    parser.add_argument(
        "--pointcept-root",
        type=str,
        default="${POINTCEPT_ROOT}",
        help="Pointcept repository root for direct imports.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments/pair_transfer/runs/teacher_exp"),
        help="Directory to save manifests and evaluation outputs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--offline-budget",
        type=int,
        default=64,
        help="How many per-asset pairs are treated as already collected offline.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=256)
    parser.add_argument(
        "--max-test-assets",
        type=int,
        default=None,
        help="Optional cap for faster iteration.",
    )
    parser.add_argument(
        "--reward-abs-max",
        type=float,
        default=2.0,
        help="Filter out labels with |reward| above this threshold. <=0 disables.",
    )
    parser.add_argument(
        "--max-pairs-per-forward",
        type=int,
        default=65536,
        help="Chunk size for teacher forward.",
    )
    return parser.parse_args()


def _load_asset_manifest(data_root: Path) -> dict[int, str]:
    manifest_dir = data_root / "manifests"
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"Manifest directory not found: {manifest_dir}")
    merged: dict[int, str] = {}
    for path in sorted(manifest_dir.glob("*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            merged[int(row["asset_id"])] = str(row["asset_path"])
    if not merged:
        raise RuntimeError(f"No asset records found under {manifest_dir}")
    return merged


def _discover_assets(data_root: Path, reward_abs_max: float) -> list[dict[str, Any]]:
    asset_map = _load_asset_manifest(data_root)
    out: list[dict[str, Any]] = []
    assets_root = data_root / "assets"
    for asset_dir in sorted(assets_root.glob("asset_*")):
        if not asset_dir.is_dir():
            continue
        asset_id = int(asset_dir.name.split("_")[-1])
        pairs_path = asset_dir / "pairs.npy"
        reward_path = asset_dir / "reward.npy"
        coord_path = asset_dir / "coord.npy"
        if not (pairs_path.is_file() and reward_path.is_file() and coord_path.is_file()):
            continue
        reward = np.load(reward_path, mmap_mode="r")
        mask = np.isfinite(reward)
        if reward_abs_max > 0:
            mask &= np.abs(reward) <= reward_abs_max
        valid = int(mask.sum())
        if valid <= 1:
            continue
        out.append(
            {
                "asset_id": asset_id,
                "asset_dir": asset_dir,
                "asset_path": asset_map.get(asset_id, ""),
                "valid_pairs": valid,
            }
        )
    if not out:
        raise RuntimeError(f"No valid assets found under {assets_root}")
    return out


def _split_assets(assets: list[dict[str, Any]], test_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(assets))
    ordered = [assets[int(i)] for i in perm]
    n_test = max(1, int(len(ordered) * test_ratio))
    test_assets = ordered[:n_test]
    train_assets = ordered[n_test:]
    return train_assets, test_assets


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
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    denom = float(np.sqrt(np.sum(rp * rp) * np.sum(rt * rt)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(rp * rt) / denom)


def _random_baseline(
    hidden_reward: np.ndarray,
    top_k: int,
    rng: np.random.Generator,
    trials: int,
) -> tuple[float, float]:
    if hidden_reward.size == 0:
        return float("nan"), float("nan")
    k = min(int(top_k), int(hidden_reward.size))
    best_values: list[float] = []
    mean_values: list[float] = []
    for _ in range(max(1, int(trials))):
        idx = rng.choice(hidden_reward.size, size=k, replace=False)
        sample = hidden_reward[idx]
        best_values.append(float(np.max(sample)))
        mean_values.append(float(np.mean(sample)))
    return float(np.mean(best_values)), float(np.mean(mean_values))


def _evaluate_asset(
    teacher: TeacherRewardInfer,
    asset_row: dict[str, Any],
    offline_budget: int,
    top_k: int,
    reward_abs_max: float,
    max_pairs_per_forward: int,
    random_trials: int,
    seed: int,
) -> AssetEvalResult | None:
    asset_dir = Path(asset_row["asset_dir"])
    coord = np.load(asset_dir / "coord.npy").astype(np.float32)
    normal_path = asset_dir / "normal.npy"
    normal = np.load(normal_path).astype(np.float32) if normal_path.is_file() else None
    pairs = np.load(asset_dir / "pairs.npy").astype(np.int64)
    reward = np.load(asset_dir / "reward.npy").astype(np.float32)

    valid_mask = np.isfinite(reward)
    if reward_abs_max > 0:
        valid_mask &= np.abs(reward) <= reward_abs_max
    pairs = pairs[valid_mask]
    reward = reward[valid_mask]
    if reward.size <= offline_budget:
        return None

    asset_seed = seed + int(asset_row["asset_id"]) * 1009
    rng = np.random.default_rng(asset_seed)
    perm = rng.permutation(reward.size)

    observed_idx = perm[:offline_budget]
    hidden_idx = perm[offline_budget:]
    if hidden_idx.size < top_k:
        return None

    hidden_pairs = pairs[hidden_idx]
    hidden_reward = reward[hidden_idx]
    pred_hidden = teacher.infer_pairs(
        coord=coord,
        normal=normal,
        pairs=hidden_pairs,
        max_pairs_per_forward=max_pairs_per_forward,
    )

    model_order = np.argsort(pred_hidden)[::-1]
    oracle_order = np.argsort(hidden_reward)[::-1]
    k = min(int(top_k), int(hidden_reward.size))
    model_top_idx = model_order[:k]
    oracle_top_idx = oracle_order[:k]

    offline_reward = reward[observed_idx]
    random_best_mean, random_topk_mean = _random_baseline(hidden_reward, k, rng, random_trials)

    model_top_rewards = hidden_reward[model_top_idx]
    oracle_top_rewards = hidden_reward[oracle_top_idx]

    return AssetEvalResult(
        asset_id=int(asset_row["asset_id"]),
        asset_dir=asset_dir.name,
        asset_path=str(asset_row.get("asset_path", "")),
        num_pairs_total=int(np.load(asset_dir / "reward.npy", mmap_mode="r").shape[0]),
        num_pairs_visible_eval=int(reward.size),
        offline_budget=int(offline_budget),
        hidden_pool=int(hidden_reward.size),
        top_k=int(k),
        offline_best_reward=float(np.max(offline_reward)),
        offline_mean_reward=float(np.mean(offline_reward)),
        random_best_mean=float(random_best_mean),
        random_topk_mean_reward=float(random_topk_mean),
        model_top1_reward=float(hidden_reward[model_order[0]]),
        model_topk_best_reward=float(np.max(model_top_rewards)),
        model_topk_mean_reward=float(np.mean(model_top_rewards)),
        oracle_top1_reward=float(hidden_reward[oracle_order[0]]),
        oracle_topk_best_reward=float(np.max(oracle_top_rewards)),
        oracle_topk_mean_reward=float(np.mean(oracle_top_rewards)),
        model_gain_vs_offline_best=float(np.max(model_top_rewards) - np.max(offline_reward)),
        model_gain_vs_random_best=float(np.max(model_top_rewards) - random_best_mean),
        model_regret_vs_oracle_best=float(np.max(oracle_top_rewards) - np.max(model_top_rewards)),
        spearman_hidden=_spearman(pred_hidden, hidden_reward),
    )


def _summarize(results: list[AssetEvalResult], meta: dict[str, Any]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("No assets were evaluated.")
    keys = [
        "offline_best_reward",
        "offline_mean_reward",
        "random_best_mean",
        "random_topk_mean_reward",
        "model_top1_reward",
        "model_topk_best_reward",
        "model_topk_mean_reward",
        "oracle_top1_reward",
        "oracle_topk_best_reward",
        "oracle_topk_mean_reward",
        "model_gain_vs_offline_best",
        "model_gain_vs_random_best",
        "model_regret_vs_oracle_best",
        "spearman_hidden",
    ]
    summary = dict(meta)
    summary["num_assets_evaluated"] = len(results)
    for key in keys:
        values = np.asarray([getattr(r, key) for r in results], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = float(np.mean(finite)) if finite.size > 0 else None
        summary[f"{key}_median"] = float(np.median(finite)) if finite.size > 0 else None
    summary["assets_model_beats_offline_best"] = int(sum(r.model_gain_vs_offline_best > 0 for r in results))
    summary["assets_model_beats_random_best"] = int(sum(r.model_gain_vs_random_best > 0 for r in results))
    summary["assets_model_top1_beats_offline_best"] = int(sum(r.model_top1_reward > r.offline_best_reward for r in results))
    return summary


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root).resolve()
    exp_dir = Path(args.exp_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = _discover_assets(data_root, reward_abs_max=float(args.reward_abs_max))
    train_assets, test_assets = _split_assets(assets, test_ratio=float(args.test_ratio), seed=int(args.seed))
    if args.max_test_assets is not None:
        test_assets = test_assets[: int(args.max_test_assets)]

    split_manifest = {
        "seed": int(args.seed),
        "test_ratio": float(args.test_ratio),
        "offline_budget": int(args.offline_budget),
        "top_k": int(args.top_k),
        "reward_abs_max": float(args.reward_abs_max),
        "train_asset_ids": [int(row["asset_id"]) for row in train_assets],
        "test_asset_ids": [int(row["asset_id"]) for row in test_assets],
    }
    split_text = json.dumps(split_manifest, ensure_ascii=False, indent=2)
    (out_dir / "asset_split.json").write_text(
        split_text,
        encoding="utf-8",
    )
    manifests_dir = PROJECT_ROOT / "experiments/pair_transfer/manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    split_name = (
        f"asset_split_seed{int(args.seed)}"
        f"_testratio{str(args.test_ratio).replace('.', 'p')}"
        f"_budget{int(args.offline_budget)}_top{int(args.top_k)}.json"
    )
    (manifests_dir / split_name).write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    teacher = TeacherRewardInfer(
        teacher_cfg=str(exp_dir / "config.py"),
        teacher_ckpt=str(exp_dir / "model" / "model_best.pth"),
        pointcept_code_root=str(Path(args.pointcept_root).resolve()),
        device=device,
        )

    results: list[AssetEvalResult] = []
    jsonl_path = out_dir / "per_asset_metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, asset_row in enumerate(test_assets, start=1):
            result = _evaluate_asset(
                teacher=teacher,
                asset_row=asset_row,
                offline_budget=int(args.offline_budget),
                top_k=int(args.top_k),
                reward_abs_max=float(args.reward_abs_max),
                max_pairs_per_forward=int(args.max_pairs_per_forward),
                random_trials=int(args.random_trials),
                seed=int(args.seed),
            )
            if result is None:
                continue
            results.append(result)
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            print(
                f"[{idx}/{len(test_assets)}] asset={result.asset_dir} "
                f"offline_best={result.offline_best_reward:.4f} "
                f"model_top1={result.model_top1_reward:.4f} "
                f"model_topk_best={result.model_topk_best_reward:.4f} "
                f"gain_vs_offline={result.model_gain_vs_offline_best:.4f}"
            )

    summary = _summarize(
        results,
        meta={
            "data_root": str(data_root),
            "exp_dir": str(exp_dir),
            "seed": int(args.seed),
            "test_ratio": float(args.test_ratio),
            "offline_budget": int(args.offline_budget),
            "top_k": int(args.top_k),
            "random_trials": int(args.random_trials),
            "reward_abs_max": float(args.reward_abs_max),
        },
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
