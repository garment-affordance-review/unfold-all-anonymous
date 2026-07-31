#!/usr/bin/env python3
"""Generate paper-friendly summary plots for the offline-label 2x2 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROTOCOL_ORDER = ["random_fling", "random_y", "cond_fling", "cond_y"]
DISPLAY_LABELS = {
    "random_fling": "Random + Fling",
    "random_y": "Random + Y-gravity",
    "cond_fling": "Conditioned + Fling",
    "cond_y": "Conditioned + Y-gravity",
}
COLORS = {
    "random_fling": "#B54A3A",
    "random_y": "#D08B2E",
    "cond_fling": "#3D7EA6",
    "cond_y": "#2F8F5B",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot offline-label 2x2 summary figures.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="experiments/offline_label_2x2/runs/full_v1_gpu1_manual",
        help="Run directory containing pair_summary.json and protocol_summary.csv.",
    )
    return parser


def _load_pair_summary(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_protocol_summary(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["protocol"]: row for row in rows}


def _metric_groups(pair_summary: list[dict], key: str) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for protocol in PROTOCOL_ORDER:
        values = [float(item[key]) for item in pair_summary if item["protocol"] == protocol]
        groups.append(np.asarray(values, dtype=np.float64))
    return groups


def _asset_level_metrics(pair_summary: list[dict]) -> dict[str, list[dict[str, float]]]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in pair_summary:
        key = (int(row["asset_index"]), str(row["protocol"]))
        grouped.setdefault(key, []).append(row)

    result: dict[str, list[dict[str, float]]] = {protocol: [] for protocol in PROTOCOL_ORDER}
    for (asset_index, protocol), rows in grouped.items():
        mean_rewards = sorted(float(item["mean_reward"]) for item in rows)
        std_rewards = [float(item["std_reward"]) for item in rows]
        n = len(mean_rewards)
        median_reward = float(np.median(mean_rewards))
        top10_start = max(0, math.ceil(0.9 * n) - 1)
        top10 = mean_rewards[top10_start:]
        result[protocol].append(
            {
                "asset_index": float(asset_index),
                "noise": float(np.mean(std_rewards)),
                "quality": float(np.mean(mean_rewards)),
                "selection_margin": float(np.mean(top10) - median_reward),
            }
        )
    return result


def _records_by_protocol(records_path: Path) -> list[dict]:
    with records_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _asset_win_rates(records: list[dict]) -> dict[str, list[float]]:
    by = {}
    for row in records:
        key = (
            int(row["asset_index"]),
            int(row["pair_idx"]),
            int(row["repeat_idx"]),
            str(row["protocol"]),
        )
        by[key] = float(row["reward"])

    result: dict[str, list[float]] = {"random": [], "cond": []}
    for init in ("random", "cond"):
        asset_stats: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for (asset, pair_idx, repeat_idx, protocol), reward_y in by.items():
            if protocol != f"{init}_y":
                continue
            other_key = (asset, pair_idx, repeat_idx, f"{init}_fling")
            if other_key not in by:
                continue
            reward_fling = by[other_key]
            asset_stats[asset][1] += 1
            if reward_y > reward_fling:
                asset_stats[asset][0] += 1
        result[init] = [wins / total for wins, total in asset_stats.values() if total > 0]
    return result


def _asset_superiority_prob(records: list[dict]) -> dict[str, list[float]]:
    by_pair: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in records:
        key = (int(row["asset_index"]), int(row["pair_idx"]), str(row["protocol"]))
        by_pair[key].append(float(row["reward"]))

    def superiority(a: list[float], b: list[float]) -> float:
        win = 0.0
        total = 0
        for x in a:
            for y in b:
                total += 1
                if x > y:
                    win += 1.0
                elif x == y:
                    win += 0.5
        return win / total if total else 0.5

    result: dict[str, list[float]] = {"random": [], "cond": []}
    for init in ("random", "cond"):
        asset_scores: dict[int, list[float]] = defaultdict(list)
        for (asset, pair_idx, protocol), ys in by_pair.items():
            if protocol != f"{init}_y":
                continue
            fs = by_pair.get((asset, pair_idx, f"{init}_fling"))
            if not fs:
                continue
            asset_scores[asset].append(superiority(ys, fs))
        result[init] = [float(np.mean(scores)) for scores in asset_scores.values() if scores]
    return result


def _style_boxplot(ax, groups: list[np.ndarray], ylabel: str, title: str) -> None:
    box = ax.boxplot(
        groups,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "#1A1A1A", "linewidth": 1.3},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        boxprops={"linewidth": 1.1, "edgecolor": "#333333"},
        flierprops={
            "marker": ".",
            "markersize": 2.0,
            "markerfacecolor": "#666666",
            "markeredgecolor": "#666666",
            "alpha": 0.2,
        },
    )
    for patch, protocol in zip(box["boxes"], PROTOCOL_ORDER, strict=True):
        patch.set_facecolor(COLORS[protocol])
        patch.set_alpha(0.85)

    means = [float(np.mean(values)) for values in groups]
    ax.scatter(
        np.arange(1, len(groups) + 1),
        means,
        marker="D",
        s=26,
        color="#111111",
        zorder=3,
        label="Mean",
    )
    ax.set_xticks(np.arange(1, len(PROTOCOL_ORDER) + 1))
    ax.set_xticklabels([DISPLAY_LABELS[p] for p in PROTOCOL_ORDER], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)


def _write_main_table(path: Path, protocol_summary: dict[str, dict]) -> None:
    lines = [
        "| Protocol | Mean Reward | Reward Std | Best-of-N Reward | Num Asset-Pairs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for protocol in PROTOCOL_ORDER:
        row = protocol_summary[protocol]
        lines.append(
            "| "
            f"{DISPLAY_LABELS[protocol]} | "
            f"{float(row['mean_reward']):.4f} | "
            f"{float(row['mean_std_reward']):.4f} | "
            f"{float(row['mean_best_reward']):.4f} | "
            f"{int(float(row['num_asset_pairs']))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_tradeoff(protocol_summary: dict[str, dict], out_path_png: Path, out_path_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)

    xs = [float(protocol_summary[p]["mean_std_reward"]) for p in PROTOCOL_ORDER]
    ys = [float(protocol_summary[p]["mean_best_reward"]) for p in PROTOCOL_ORDER]

    for protocol, x, y in zip(PROTOCOL_ORDER, xs, ys, strict=True):
        ax.scatter(
            x,
            y,
            s=180,
            color=COLORS[protocol],
            edgecolors="#222222",
            linewidths=1.0,
            zorder=3,
        )
        ax.annotate(
            DISPLAY_LABELS[protocol],
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
        )

    ax.axvline(float(np.mean(xs)), color="#777777", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(float(np.mean(ys)), color="#777777", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(
        0.02,
        0.98,
        "Better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    ax.annotate(
        "",
        xy=(0.08, 0.90),
        xytext=(0.20, 0.90),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#222222"},
    )
    ax.annotate(
        "",
        xy=(0.08, 0.90),
        xytext=(0.08, 0.78),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#222222"},
    )

    ax.set_xlabel("Noise: mean per-pair reward std (lower is better)")
    ax.set_ylabel("Potential: mean best-of-8 reward (higher is better)")
    ax.set_title("Offline Label 2x2: Noise vs Potential")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)

    fig.savefig(out_path_png, dpi=240)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def _plot_quality_vs_noise(protocol_summary: dict[str, dict], out_path_png: Path, out_path_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)

    xs = [float(protocol_summary[p]["mean_std_reward"]) for p in PROTOCOL_ORDER]
    ys = [float(protocol_summary[p]["mean_reward"]) for p in PROTOCOL_ORDER]

    for protocol, x, y in zip(PROTOCOL_ORDER, xs, ys, strict=True):
        ax.scatter(
            x,
            y,
            s=190,
            color=COLORS[protocol],
            edgecolors="#222222",
            linewidths=1.0,
            zorder=3,
        )
        ax.annotate(
            DISPLAY_LABELS[protocol],
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
        )

    ax.axvline(float(np.mean(xs)), color="#777777", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(float(np.mean(ys)), color="#777777", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(
        0.02,
        0.98,
        "Better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    ax.annotate(
        "",
        xy=(0.08, 0.90),
        xytext=(0.20, 0.90),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#222222"},
    )
    ax.annotate(
        "",
        xy=(0.08, 0.90),
        xytext=(0.08, 0.78),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#222222"},
    )

    ax.set_xlabel("Noise: mean per-pair reward std (lower is better)")
    ax.set_ylabel("Quality: mean pair reward (higher is better)")
    ax.set_title("Offline Label 2x2: Label Quality vs Noise")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)

    fig.savefig(out_path_png, dpi=240)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def _plot_asset_margin_vs_noise(pair_summary: list[dict], out_path_png: Path, out_path_pdf: Path) -> None:
    asset_metrics = _asset_level_metrics(pair_summary)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)

    label_offsets = {
        "random_fling": (8, -14),
        "random_y": (8, 10),
        "cond_fling": (-72, 10),
        "cond_y": (-72, -16),
    }

    for protocol in PROTOCOL_ORDER:
        rows = asset_metrics[protocol]
        xs = np.asarray([row["noise"] for row in rows], dtype=np.float64)
        ys = np.asarray([row["selection_margin"] for row in rows], dtype=np.float64)
        ax.scatter(
            xs,
            ys,
            s=26,
            alpha=0.28,
            color=COLORS[protocol],
            edgecolors="none",
            rasterized=True,
        )
        cx, cy = float(xs.mean()), float(ys.mean())
        ax.scatter(
            [cx],
            [cy],
            s=220,
            marker="D",
            color=COLORS[protocol],
            edgecolors="#111111",
            linewidths=1.1,
            zorder=4,
        )
        dx, dy = label_offsets[protocol]
        ax.annotate(
            DISPLAY_LABELS[protocol],
            (cx, cy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.2},
            zorder=5,
        )

    ax.set_xlabel("Mean per-pair reward std")
    ax.set_ylabel("Selection margin: top 10% pair mean - median pair mean")
    ax.set_title("Asset-Level Discriminability vs Label Noise")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_axisbelow(True)

    fig.savefig(out_path_png, dpi=260)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def _plot_asset_quality_vs_noise(pair_summary: list[dict], out_path_png: Path, out_path_pdf: Path) -> None:
    asset_metrics = _asset_level_metrics(pair_summary)
    fig, ax = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)

    label_offsets = {
        "random_fling": (10, -14),
        "random_y": (10, 10),
        "cond_fling": (-86, 10),
        "cond_y": (-86, -16),
    }

    for protocol in PROTOCOL_ORDER:
        rows = asset_metrics[protocol]
        xs = np.asarray([row["noise"] for row in rows], dtype=np.float64)
        ys = np.asarray([row["quality"] for row in rows], dtype=np.float64)
        ax.scatter(
            xs,
            ys,
            s=30,
            alpha=0.32,
            color=COLORS[protocol],
            edgecolors="none",
            rasterized=True,
        )

        cx, cy = float(xs.mean()), float(ys.mean())
        ax.scatter(
            [cx],
            [cy],
            s=240,
            marker="D",
            color=COLORS[protocol],
            edgecolors="#111111",
            linewidths=1.1,
            zorder=4,
        )

        dx, dy = label_offsets[protocol]
        ax.annotate(
            DISPLAY_LABELS[protocol],
            (cx, cy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.2},
            zorder=5,
        )

    ax.set_xlabel("Asset-level mean per-pair reward std")
    ax.set_ylabel("Asset-level mean pair reward")
    ax.set_title("Asset-Level Label Quality vs Noise")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_axisbelow(True)

    fig.savefig(out_path_png, dpi=260)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def _plot_y_over_fling_win_rate(records: list[dict], out_path_png: Path, out_path_pdf: Path) -> None:
    win_rates = _asset_win_rates(records)
    fig, ax = plt.subplots(figsize=(7.0, 5.8), constrained_layout=True)

    groups = [np.asarray(win_rates["random"], dtype=np.float64), np.asarray(win_rates["cond"], dtype=np.float64)]
    positions = [1, 2]
    labels = ["Random Init", "Conditioned Init"]
    colors = [COLORS["random_y"], COLORS["cond_y"]]

    box = ax.boxplot(
        groups,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        boxprops={"linewidth": 1.1, "edgecolor": "#333333"},
        flierprops={"marker": "", "markersize": 0},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)

    rng = np.random.default_rng(0)
    for x0, values, color in zip(positions, groups, colors, strict=True):
        jitter = rng.uniform(-0.12, 0.12, size=values.shape[0])
        ax.scatter(
            np.full(values.shape[0], x0) + jitter,
            values,
            s=26,
            color=color,
            alpha=0.38,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        mean_v = float(values.mean())
        ax.scatter([x0], [mean_v], s=170, marker="D", color=color, edgecolors="#111111", linewidths=1.0, zorder=4)
        ax.annotate(f"{mean_v:.3f}", (x0, mean_v), xytext=(8, 8), textcoords="offset points", fontsize=10)

    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.1, alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Asset-level win rate: P(reward_y > reward_fling)")
    ax.set_title("How Often Y-Gravity Beats Fling")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_axisbelow(True)

    fig.savefig(out_path_png, dpi=260)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def _plot_y_over_fling_superiority(records: list[dict], out_path_png: Path, out_path_pdf: Path) -> None:
    superiority = _asset_superiority_prob(records)
    fig, ax = plt.subplots(figsize=(7.0, 5.8), constrained_layout=True)

    groups = [np.asarray(superiority["random"], dtype=np.float64), np.asarray(superiority["cond"], dtype=np.float64)]
    positions = [1, 2]
    labels = ["Random Init", "Conditioned Init"]
    colors = [COLORS["random_y"], COLORS["cond_y"]]

    box = ax.boxplot(
        groups,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        boxprops={"linewidth": 1.1, "edgecolor": "#333333"},
        flierprops={"marker": "", "markersize": 0},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)

    rng = np.random.default_rng(1)
    for x0, values, color in zip(positions, groups, colors, strict=True):
        jitter = rng.uniform(-0.12, 0.12, size=values.shape[0])
        ax.scatter(
            np.full(values.shape[0], x0) + jitter,
            values,
            s=26,
            color=color,
            alpha=0.38,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        mean_v = float(values.mean())
        ax.scatter([x0], [mean_v], s=170, marker="D", color=color, edgecolors="#111111", linewidths=1.0, zorder=4)
        ax.annotate(f"{mean_v:.3f}", (x0, mean_v), xytext=(8, 8), textcoords="offset points", fontsize=10)

    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.1, alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Asset-level pooled superiority P(R_y > R_fling)")
    ax.set_title("How Often Y-Gravity Beats Fling in Pooled Repeats")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_axisbelow(True)

    fig.savefig(out_path_png, dpi=260)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def _plot_asset_delta_bars(pair_summary: list[dict], out_path_png: Path, out_path_pdf: Path) -> None:
    asset_metrics = _asset_level_metrics(pair_summary)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.4), constrained_layout=True, sharex=True)
    pairs = [
        ("random", "random_fling", "random_y", "Random Init: y - fling"),
        ("cond", "cond_fling", "cond_y", "Conditioned Init: y - fling"),
    ]

    for ax, (_, fling_key, y_key, title) in zip(axes, pairs, strict=True):
        fling_rows = {int(row["asset_index"]): row for row in asset_metrics[fling_key]}
        y_rows = {int(row["asset_index"]): row for row in asset_metrics[y_key]}
        asset_ids = sorted(set(fling_rows) & set(y_rows))
        delta_rows = [
            (asset_id, float(y_rows[asset_id]["quality"] - fling_rows[asset_id]["quality"]))
            for asset_id in asset_ids
        ]
        delta_rows.sort(key=lambda item: item[1], reverse=True)
        sorted_asset_ids = [asset_id for asset_id, _ in delta_rows]
        deltas = [delta for _, delta in delta_rows]

        colors = ["#2F8F5B" if d >= 0 else "#B54A3A" for d in deltas]
        ax.bar(np.arange(len(sorted_asset_ids)), deltas, color=colors, width=0.82, linewidth=0)
        ax.axhline(0.0, color="#222222", linewidth=1.0)
        ax.set_ylabel("Delta mean reward")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("Asset index")
    axes[-1].set_xticks(np.arange(0, len(sorted_asset_ids), 5))
    axes[-1].set_xticklabels([str(sorted_asset_ids[i]) for i in range(0, len(sorted_asset_ids), 5)], rotation=0)

    fig.savefig(out_path_png, dpi=260)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def run(args) -> None:
    run_dir = Path(args.run_dir)
    pair_summary = _load_pair_summary(run_dir / "pair_summary.json")
    protocol_summary = _load_protocol_summary(run_dir / "protocol_summary.csv")
    records = _records_by_protocol(run_dir / "records.csv")

    plots_dir = run_dir / "plots"
    summary_dir = run_dir / "summary"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    reward_std_groups = _metric_groups(pair_summary, "std_reward")
    best_reward_groups = _metric_groups(pair_summary, "best_reward")

    fig1, ax1 = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    _style_boxplot(ax1, reward_std_groups, ylabel="Per-pair reward std", title="Repeatability Across 2x2 Protocols")
    fig1.savefig(plots_dir / "reward_std_boxplot.png", dpi=220)
    fig1.savefig(plots_dir / "reward_std_boxplot.pdf")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    _style_boxplot(ax2, best_reward_groups, ylabel="Per-pair best-of-8 reward", title="Best-of-N Reward Across 2x2 Protocols")
    fig2.savefig(plots_dir / "best_of_n_boxplot.png", dpi=220)
    fig2.savefig(plots_dir / "best_of_n_boxplot.pdf")
    plt.close(fig2)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True)
    _style_boxplot(axes[0], reward_std_groups, ylabel="Per-pair reward std", title="Noise Metric")
    _style_boxplot(axes[1], best_reward_groups, ylabel="Per-pair best-of-8 reward", title="Potential Metric")
    fig.savefig(plots_dir / "offline_label_2x2_main_figure.png", dpi=240)
    fig.savefig(plots_dir / "offline_label_2x2_main_figure.pdf")
    plt.close(fig)

    _plot_tradeoff(
        protocol_summary,
        plots_dir / "noise_vs_potential_tradeoff.png",
        plots_dir / "noise_vs_potential_tradeoff.pdf",
    )
    _plot_quality_vs_noise(
        protocol_summary,
        plots_dir / "quality_vs_noise_tradeoff.png",
        plots_dir / "quality_vs_noise_tradeoff.pdf",
    )
    _plot_asset_margin_vs_noise(
        pair_summary,
        plots_dir / "asset_margin_vs_noise.png",
        plots_dir / "asset_margin_vs_noise.pdf",
    )
    _plot_asset_quality_vs_noise(
        pair_summary,
        plots_dir / "asset_quality_vs_noise.png",
        plots_dir / "asset_quality_vs_noise.pdf",
    )
    _plot_y_over_fling_win_rate(
        records,
        plots_dir / "y_over_fling_win_rate.png",
        plots_dir / "y_over_fling_win_rate.pdf",
    )
    _plot_y_over_fling_superiority(
        records,
        plots_dir / "y_over_fling_superiority.png",
        plots_dir / "y_over_fling_superiority.pdf",
    )
    _plot_asset_delta_bars(
        pair_summary,
        plots_dir / "asset_delta_y_minus_fling.png",
        plots_dir / "asset_delta_y_minus_fling.pdf",
    )

    _write_main_table(summary_dir / "main_table.md", protocol_summary)
    print(f"[OK] Wrote plots to {plots_dir}")
    print(f"[OK] Wrote summary table to {summary_dir / 'main_table.md'}")


if __name__ == "__main__":
    run(build_parser().parse_args())
