#!/usr/bin/env python3
"""Evaluate reward repeatability for fixed pair-conditioned actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

from .pair_conditioned_collect import PairConditionedOfflineCollector, load_pair_conditioned_env_and_cfg


def build_pair_repeatability_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pair repeatability under small pose noise.")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0", help="Gym task id.")
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel environments used as repeat rollouts.")
    parser.add_argument("--asset-id", type=int, default=0, help="0-based asset index in the pool.")
    parser.add_argument("--num-pairs", type=int, default=8, help="Number of distinct pairs to evaluate.")
    parser.add_argument("--repeats-per-pair", type=int, default=16, help="Number of repeated rollouts for each pair.")
    parser.add_argument("--rot-noise-deg", type=float, default=3.0, help="Uniform Euler noise range in degrees per axis.")
    parser.add_argument("--output-dir", type=str, default="logs/pair_repeatability", help="Output directory for CSV/plots.")
    parser.add_argument("--vis-dir", type=str, default="logs/pair_repeatability_visuals", help="Unused debug visual root for shared collector setup.")
    parser.add_argument("--overwrite", action="store_true", help="Unused compatibility flag for shared collector setup.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_outputs(out_dir: Path, records: list[dict], summary: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    import csv

    with (out_dir / "repeatability_records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_idx", "coord_id1", "coord_id2", "repeat_idx", "reward"],
        )
        writer.writeheader()
        writer.writerows(records)

    with (out_dir / "repeatability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plt = _setup_matplotlib()
    labels = [f"{item['coord_id1']}-{item['coord_id2']}" for item in summary]
    values = [item["rewards"] for item in summary]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.boxplot(values, tick_labels=labels, showfliers=True)
    ax.set_title("Pair Repeatability Reward Distribution")
    ax.set_xlabel("coord pair")
    ax.set_ylabel("reward")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "repeatability_boxplot.png", dpi=160)
    plt.close(fig)


def run(args) -> None:
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import carb
    import unfold  # noqa: F401

    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    carb.log_warn = lambda *a, **k: None

    env, env_cfg = load_pair_conditioned_env_and_cfg(args)
    collector = PairConditionedOfflineCollector(env, env_cfg, args)
    out_dir = Path(args.output_dir)

    try:
        if args.asset_id < 0 or args.asset_id >= len(collector._asset_paths):
            raise ValueError(f"asset-id out of range: {args.asset_id}")

        collector._reset_single_asset(args.asset_id, args.asset_id)
        pointcloud = collector._prepare_pointcloud(0)
        collector._apply_coord_reward_sampling_mask(pointcloud)
        bank = collector._build_pair_bank(0)
        chosen_pairs = bank.pop_distinct(int(args.num_pairs))
        if not chosen_pairs:
            raise RuntimeError("No pair candidates available for repeatability evaluation.")

        rot_noise = (float(args.rot_noise_deg),) * 3
        records: list[dict] = []
        summary: list[dict] = []

        for pair_idx, candidate in enumerate(chosen_pairs):
            rewards_acc: list[float] = []
            repeat_idx = 0
            while repeat_idx < int(args.repeats_per_pair):
                batch_n = min(int(env_cfg.scene.num_envs), int(args.repeats_per_pair) - repeat_idx)
                candidates = [candidate] * batch_n
                actions = torch.full((env_cfg.scene.num_envs, 2), -1, dtype=torch.long, device=collector.device)
                for env_i in range(batch_n):
                    actions[env_i, 0] = candidate.raw_id1
                    actions[env_i, 1] = candidate.raw_id2

                collector._apply_pair_conditioned_poses(candidates, rot_noise_deg=rot_noise)
                obs, rewards, *_ = env.unwrapped.step(actions)
                del obs
                reward_list = rewards.detach().cpu().view(-1).tolist()

                for env_i in range(batch_n):
                    reward = float(reward_list[env_i])
                    rewards_acc.append(reward)
                    records.append(
                        {
                            "pair_idx": pair_idx,
                            "coord_id1": int(candidate.coord_id1),
                            "coord_id2": int(candidate.coord_id2),
                            "repeat_idx": repeat_idx,
                            "reward": reward,
                        }
                    )
                    repeat_idx += 1

            rewards_np = np.asarray(rewards_acc, dtype=np.float32)
            summary.append(
                {
                    "pair_idx": pair_idx,
                    "coord_id1": int(candidate.coord_id1),
                    "coord_id2": int(candidate.coord_id2),
                    "mean_reward": float(rewards_np.mean()),
                    "std_reward": float(rewards_np.std()),
                    "min_reward": float(rewards_np.min()),
                    "max_reward": float(rewards_np.max()),
                    "repeats": int(rewards_np.shape[0]),
                    "rewards": rewards_np.round(6).tolist(),
                }
            )
            print(
                f"[PAIR_REPEAT] pair={candidate.coord_id1}-{candidate.coord_id2} "
                f"mean={rewards_np.mean():.4f} std={rewards_np.std():.4f} "
                f"min={rewards_np.min():.4f} max={rewards_np.max():.4f}",
                flush=True,
            )

        _write_outputs(out_dir, records, summary)
        print(f"[OK] Saved repeatability outputs to {out_dir}", flush=True)
    finally:
        env.close()
        simulation_app.close()


def main() -> None:
    parser = build_pair_repeatability_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
