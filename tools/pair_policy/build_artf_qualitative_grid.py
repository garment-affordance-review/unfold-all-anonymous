#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose aRTF qualitative panels into a row-labeled grid.")
    parser.add_argument(
        "--panel-root",
        type=Path,
        required=True,
        help="Root directory containing category subdirectories of panel PNGs.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="towels,tshirts,shorts",
        help="Comma-separated category order by row.",
    )
    parser.add_argument("--samples-per-row", type=int, default=4)
    parser.add_argument("--panel-suffix", type=str, default=".png")
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path, default=None)
    parser.add_argument("--margin", type=int, default=28)
    parser.add_argument("--gap", type=int, default=10)
    parser.add_argument("--row-label-width", type=int, default=52)
    parser.add_argument("--font-size", type=int, default=42)
    parser.add_argument(
        "--panel-size",
        type=int,
        default=228,
        help="Resize every square panel to this size before composition, matching single-column layout.",
    )
    return parser.parse_args()


def display_name(category: str) -> str:
    mapping = {
        "towels": "Towel",
        "tshirts": "T-shirt",
        "shorts": "Shorts",
    }
    return mapping.get(category, category)


def load_panels(panel_root: Path, categories: list[str], samples_per_row: int) -> dict[str, list[Path]]:
    rows: dict[str, list[Path]] = {}
    for category in categories:
        paths = sorted((panel_root / category).glob(f"*{'.png'}"))
        if len(paths) < samples_per_row:
            raise FileNotFoundError(
                f"Category {category} only has {len(paths)} panels under {panel_root / category},"
                f" need {samples_per_row}."
            )
        rows[category] = paths[:samples_per_row]
    return rows


def build_grid(
    rows: dict[str, list[Path]],
    margin: int,
    gap: int,
    row_label_width: int,
    font_size: int,
    panel_size: int,
) -> Image.Image:
    panel_w = panel_size
    panel_h = panel_size
    categories = list(rows.keys())
    cols = len(next(iter(rows.values())))
    canvas_w = margin * 2 + row_label_width + cols * panel_w + (cols - 1) * gap
    canvas_h = margin * 2 + len(categories) * panel_h + (len(categories) - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    for row_idx, category in enumerate(categories):
        y = margin + row_idx * (panel_h + gap)
        for col_idx, panel_path in enumerate(rows[category]):
            x = margin + row_label_width + col_idx * (panel_w + gap)
            panel = Image.open(panel_path).convert("RGB").resize((panel_w, panel_h), resample=Image.BILINEAR)
            canvas.paste(panel, (x, y))

    return canvas


def _configure_matplotlib() -> None:
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Nimbus Roman No9 L",
                "DejaVu Serif",
            ],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_with_vector_labels(
    grid: Image.Image,
    categories: list[str],
    margin: int,
    gap: int,
    row_label_width: int,
    font_size: int,
    panel_size: int,
    out_png: Path,
    out_pdf: Path | None,
) -> None:
    _configure_matplotlib()

    canvas_w, canvas_h = grid.size
    dpi = 300
    fig = plt.figure(figsize=(canvas_w / dpi, canvas_h / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(grid)
    ax.set_axis_off()

    panel_h = panel_size
    for row_idx, category in enumerate(categories):
        y = margin + row_idx * (panel_h + gap)
        row_center_px = y + panel_h / 2.0
        x_center_px = margin + row_label_width * 0.55
        ax.text(
            x_center_px / canvas_w,
            1.0 - row_center_px / canvas_h,
            display_name(category),
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=font_size_to_points(font_size),
            fontweight="bold",
            color="black",
            clip_on=False,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0)
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def font_size_to_points(px: int, dpi: int = 300) -> float:
    return px * 72.0 / dpi


def main() -> int:
    args = parse_args()
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    rows = load_panels(args.panel_root, categories, args.samples_per_row)
    grid = build_grid(
        rows=rows,
        margin=args.margin,
        gap=args.gap,
        row_label_width=args.row_label_width,
        font_size=args.font_size,
        panel_size=args.panel_size,
    )
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    save_with_vector_labels(
        grid=grid,
        categories=categories,
        margin=args.margin,
        gap=args.gap,
        row_label_width=args.row_label_width,
        font_size=args.font_size,
        panel_size=args.panel_size,
        out_png=args.out_png,
        out_pdf=args.out_pdf,
    )
    print(args.out_png)
    if args.out_pdf is not None:
        print(args.out_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
