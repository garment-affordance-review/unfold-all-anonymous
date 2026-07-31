#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single bar chart for aRTF affordance behavior analysis.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("experiments/aRTFClothes/analysis/keypoint_eval_gtmask_amp/keypoint_eval_test.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/aRTFClothes/analysis/affordance_behavior_metrics"),
    )
    parser.add_argument(
        "--min-percent",
        type=float,
        default=5.0,
        help="Only report pair types whose within-category percentage is at least this threshold.",
    )
    return parser.parse_args()


def collapse_semantic(category: str, keypoint: str) -> str:
    if category == "tshirts":
        if keypoint.startswith("shoulder_"):
            return "shoulder"
        if keypoint.startswith("sleeve_") and keypoint.endswith("_top"):
            return "sleeve top"
        if keypoint.startswith("sleeve_") and keypoint.endswith("_bottom"):
            return "sleeve bottom"
        if keypoint.startswith("neck_"):
            return "neck"
        if keypoint.startswith("armpit_"):
            return "armpit"
        if keypoint.startswith("waist_"):
            return "waist"
    if category == "shorts":
        if keypoint.startswith("waist_"):
            return "waist"
        if keypoint.startswith("pipe_") and keypoint.endswith("_outer"):
            return "pipe outer"
        if keypoint.startswith("pipe_") and keypoint.endswith("_inner"):
            return "pipe inner"
        if keypoint == "crotch":
            return "crotch"
    return keypoint.replace("_", " ")


def towel_pair_type(k1: str, k2: str) -> str:
    if k1 == k2:
        return "same corner"
    i = int(k1.replace("corner", ""))
    j = int(k2.replace("corner", ""))
    diff = abs(i - j) % 4
    if diff in (1, 3):
        return "adjacent"
    return "diagonal"


def pair_type(category: str, k1: str, k2: str) -> str:
    if category == "towels":
        return towel_pair_type(k1, k2)
    c1 = collapse_semantic(category, k1)
    c2 = collapse_semantic(category, k2)
    return "/".join(sorted((c1, c2)))


def display_label(category: str, name: str) -> str:
    if category == "tshirts":
        replacements = {
            "shoulder/sleeve top": "shoulder/sleeve",
            "neck/shoulder": "neck/shoulder",
            "shoulder/shoulder": "shoulder/shoulder",
        }
        return replacements.get(name, name)
    if category == "shorts":
        replacements = {
            "waist/waist": "waist/waist",
            "pipe outer/waist": "outer leg/waist",
            "pipe inner/pipe outer": "inner/outer leg",
        }
        return replacements.get(name, name)
    return name


def build_stats(rows: list[dict[str, str]], min_percent: float) -> dict[str, dict[str, dict[str, float]]]:
    per_cat: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        cat = row["category"]
        per_cat[cat][pair_type(cat, row["nearest_kp_x1"], row["nearest_kp_x2"])] += 1

    stats: dict[str, dict[str, dict[str, float]]] = {}
    threshold = min_percent / 100.0
    for cat, counter in per_cat.items():
        total = sum(counter.values())
        kept: dict[str, dict[str, float]] = {}
        for name, count in counter.most_common():
            rate = count / total
            if rate < threshold:
                continue
            kept[name] = {"count": int(count), "percent": float(rate * 100.0)}
        stats[cat] = kept
    return stats


def plot(stats: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    category_order = ["towels", "tshirts", "shorts"]
    category_names = {"towels": "Towel", "tshirts": "T-shirt", "shorts": "Shorts"}
    # Warm-to-cool palette inspired by the user's reference figure.
    gradient_colors = [
        "#C8102E",
        "#F04E37",
        "#F79A4A",
        "#E69F00",
        "#F2C66D",
        "#CDEAF7",
        "#8EC5E8",
        "#3D63B8",
    ]
    title_colors = {"towels": "#C23B2A", "tshirts": "#A77B18", "shorts": "#4478C7"}

    # Match the final LaTeX single-column footprint so font sizes survive PDF placement.
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    labels: list[str] = []
    values: list[float] = []
    bar_colors: list[str] = []
    x_positions: list[float] = []
    separators: list[float] = []
    category_ranges: dict[str, tuple[int, int]] = {}
    x = 0.0
    group_gap = 0.42
    for idx, cat in enumerate(category_order):
        items = list(stats.get(cat, {}).items())
        start_idx = len(x_positions)
        for item_idx, (name, payload) in enumerate(items):
            labels.append(display_label(cat, name))
            values.append(payload["percent"])
            bar_colors.append(gradient_colors[len(bar_colors)])
            x_positions.append(x)
            x += 0.92
        end_idx = len(x_positions)
        if end_idx > start_idx:
            category_ranges[cat] = (start_idx, end_idx)
        if idx < len(category_order) - 1:
            separators.append(x - 0.46)
            x += group_gap

    fig, ax = plt.subplots(figsize=(3.45, 2.47), constrained_layout=True)
    bars = ax.bar(x_positions, values, color=bar_colors, width=0.72, edgecolor="#454545", linewidth=0.9)
    ylim_top = max(values) * 1.08 if values else 1.0
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, ylim_top)
    ax.set_xticks(x_positions, labels, rotation=27, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=1.0, length=0)
    ax.tick_params(axis="y", pad=1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for sep in separators:
        ax.axvline(sep, color="#C7C7C7", linewidth=0.8, linestyle=":")

    for cat in category_order:
        if cat not in category_ranges:
            continue
        start_idx, end_idx = category_ranges[cat]
        cat_positions = x_positions[start_idx:end_idx]
        center = (cat_positions[0] + cat_positions[-1]) / 2.0
        ax.text(
            center,
            1.008,
            category_names[cat],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=title_colors[cat],
            clip_on=False,
        )

    for rect, value in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            value + ylim_top * 0.008,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(args.csv.open(newline="", encoding="utf-8")))
    stats = build_stats(rows, min_percent=args.min_percent)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "pair_distribution_bars_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    png_path = args.out_dir / "pair_distribution_bars.png"
    pdf_path = args.out_dir / "pair_distribution_bars.pdf"
    plot(stats, png_path)
    plot(stats, pdf_path)
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
