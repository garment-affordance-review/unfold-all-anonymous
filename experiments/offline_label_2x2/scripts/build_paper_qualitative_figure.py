#!/usr/bin/env python3
"""Build the final paper qualitative figure from trusted raw frames.

This script generates the final single-column paper figure for Sec. 4.2.
It uses matplotlib so that text size and margins can be controlled in
paper-space units, which is much more reliable than raster-only compositing.

Current preset:
- asset 93, side view
- rows: Cond-D / Cond-F / Rand-D
- raw frame indices:
  Cond-D: 0, 4, 17, 21, 24
  Cond-F: 0, 12, 15, 16, 18
  Rand-D: 0, 13, 29, 31, 35
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image


ROOT = Path("${PROJECT_ROOT}")
BASE = ROOT / "experiments/offline_label_2x2/runs/paper_qualitative_trusted/asset93_shared_best_restyle"

ROWS = [
    (
        "Cond-D",
        BASE / "cond_y/qualitative/asset_0093/pair_00/cond_y/side/frames",
        [0, 4, 17, 21, 24],
    ),
    (
        "Cond-F",
        BASE / "cond_fling/qualitative/asset_0093/pair_00/cond_fling/side/frames",
        [0, 12, 15, 16, 18],
    ),
    (
        "Rand-D",
        BASE / "random_y_seed1024/qualitative/asset_0093/pair_00/random_y/side/frames",
        [0, 13, 29, 31, 35],
    ),
]

OUT_PNG = BASE / "asset93_restyle_side_paper_final.png"
OUT_PDF = BASE / "asset93_restyle_side_paper_final.pdf"
OUT_CROP = BASE / "asset93_restyle_side_paper_final_crop.pdf"


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


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _build_matplotlib():
    _configure_matplotlib()
    sample = _load_image(sorted(ROWS[0][1].glob("frame_*.png"))[0])
    tile_px_w, tile_px_h = sample.size
    aspect = tile_px_h / tile_px_w

    cols = 5
    nrows = len(ROWS)

    # IEEE/RA-L single-column width is roughly 3.45in.  Build the layout in
    # physical units so cell size and spacing stay predictable.
    fig_width = 3.45
    label_col_w = 0.115
    label_gap = 0.006
    gap_x = 0.018
    gap_y = 0.054
    outer_pad = 0.003

    tile_w = (
        fig_width
        - 2 * outer_pad
        - label_col_w
        - label_gap
        - (cols - 1) * gap_x
    ) / cols
    tile_h = tile_w * aspect
    fig_height = 2 * outer_pad + nrows * tile_h + (nrows - 1) * gap_y

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300, facecolor="white")

    for r, (label, frame_dir, indices) in enumerate(ROWS):
        y0_in = fig_height - outer_pad - (r + 1) * tile_h - r * gap_y
        files = sorted(frame_dir.glob("frame_*.png"))

        row_center = (y0_in + tile_h / 2.0) / fig_height
        fig.text(
            (outer_pad + label_col_w * 0.54) / fig_width,
            row_center,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color="black",
            clip_on=False,
        )

        for c, idx in enumerate(indices):
            x0_in = outer_pad + label_col_w + label_gap + c * (tile_w + gap_x)
            ax = fig.add_axes(
                [x0_in / fig_width, y0_in / fig_height, tile_w / fig_width, tile_h / fig_height]
            )
            ax.imshow(_load_image(files[idx]))
            ax.set_axis_off()
            ax.text(
                0.97,
                0.97,
                f"#{c + 1}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="white",
                fontsize=8.0,
                fontweight="bold",
            )

    return fig


def main() -> None:
    fig = _build_matplotlib()
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, dpi=300, facecolor="white")
    plt.close(fig)
    try:
        subprocess.run(
            ["pdfcrop", str(OUT_PDF), str(OUT_CROP)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")
    if OUT_CROP.exists():
        print(f"wrote {OUT_CROP}")


if __name__ == "__main__":
    main()
