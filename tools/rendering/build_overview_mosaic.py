#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont


def _load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _extract_view_idx(meta: dict) -> int:
    camera_path = str(meta.get("camera_path", ""))
    if "view_" in camera_path:
        try:
            return int(camera_path.split("view_")[1].split("/")[0])
        except Exception:
            return 0
    return 0


def _extract_category(meta: dict, asset_dir_name: str) -> str:
    asset_path = str(meta.get("asset_path", ""))
    parts = [p for p in asset_path.split("/") if p]
    if len(parts) >= 2:
        return parts[1]
    return asset_dir_name


def _discover_samples(
    root: Path,
    sample_count: int,
    max_asset_dirs: int | None = None,
    seed: int = 0,
) -> tuple[list[tuple[Path, dict]], list[tuple[str, str, int, int, bool]]]:
    rng = random.Random(seed)
    candidates: list[tuple[Path, dict]] = []
    asset_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if max_asset_dirs is not None and max_asset_dirs > 0:
        asset_dirs = asset_dirs[:max_asset_dirs]

    for asset_dir in asset_dirs:
        sample_dirs = sorted(p for p in asset_dir.iterdir() if p.is_dir())
        for sample_dir in sample_dirs:
            meta_path = sample_dir / "meta.json"
            rgb_path = sample_dir / "rgb.png"
            if not meta_path.exists() or not rgb_path.exists():
                continue
            meta = _load_meta(meta_path)
            candidates.append((rgb_path, meta))

    if not candidates:
        return [], []

    rng.shuffle(candidates)
    chosen = candidates[:sample_count]
    samples: list[tuple[Path, dict]] = []
    summary: list[tuple[str, str, int, int, bool]] = []
    for rgb_path, meta in chosen:
        samples.append((rgb_path, meta))
        summary.append(
            (
                str(meta.get("asset_name", "unknown")),
                _extract_category(meta, "unknown"),
                int(meta.get("global_step", 0)),
                _extract_view_idx(meta),
                bool(meta.get("external_texture_applied", False)),
            )
        )
    return samples, summary


def _make_tile(
    rgb_path: Path,
    meta: dict,
    tile_w: int,
    tile_h: int,
    font: ImageFont.ImageFont,
    show_labels: bool,
    apply_mask_black_bg: bool,
) -> Image.Image:
    img = Image.open(rgb_path).convert("RGB")

    if apply_mask_black_bg:
        mask_path = rgb_path.with_name("mask.png")
        if mask_path.exists():
            # Render masks are stored as 0/1 labels; convert to a real binary
            # alpha mask before compositing or the whole tile turns nearly black.
            mask = Image.open(mask_path).convert("L").point(lambda p: 255 if p > 0 else 0)
            bg = Image.new("RGB", img.size, (0, 0, 0))
            img = Image.composite(img, bg, mask)

    img = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)

    if show_labels:
        draw = ImageDraw.Draw(img, "RGBA")
        label_h = 34
        draw.rectangle((0, tile_h - label_h, tile_w, tile_h), fill=(0, 0, 0, 150))

        asset_name = str(meta.get("asset_name", meta.get("asset_dir", "unknown")))
        camera_path = str(meta.get("camera_path", ""))
        view_label = "view?"
        if "view_" in camera_path:
            try:
                view_idx = int(camera_path.split("view_")[1].split("/")[0])
                view_label = f"view {view_idx}"
            except Exception:
                pass
        tex_label = "tex on" if bool(meta.get("external_texture_applied", False)) else "tex off"
        text = f"{asset_name} | {view_label} | {tex_label}"
        draw.text((8, tile_h - label_h + 8), text, fill=(255, 255, 255), font=font)
    return img


def build_mosaic(
    input_root: Path,
    output_path: Path,
    sample_count: int,
    max_asset_dirs: int | None,
    tile_w: int,
    tile_h: int,
    margin_x: int,
    margin_y: int,
    columns: int | None,
    seed: int,
    show_labels: bool,
    bg_color: tuple[int, int, int] | tuple[int, int, int, int],
    apply_mask_black_bg: bool,
) -> Path:
    samples, summary = _discover_samples(
        input_root,
        sample_count=sample_count,
        max_asset_dirs=max_asset_dirs,
        seed=seed,
    )
    if not samples:
        raise RuntimeError(f"No render samples found under {input_root}")

    cols = columns or min(len(samples), 4)
    rows = math.ceil(len(samples) / cols)
    font = _safe_font(16)

    canvas_w = cols * tile_w + (cols + 1) * margin_x
    canvas_h = rows * tile_h + (rows + 1) * margin_y
    mode = "RGBA" if len(bg_color) == 4 else "RGB"
    canvas = Image.new(mode, (canvas_w, canvas_h), color=bg_color)

    for idx, (rgb_path, meta) in enumerate(samples):
        row = idx // cols
        col = idx % cols
        x = margin_x + col * (tile_w + margin_x)
        y = margin_y + row * (tile_h + margin_y)
        tile = _make_tile(
            rgb_path,
            meta,
            tile_w=tile_w,
            tile_h=tile_h,
            font=font,
            show_labels=show_labels,
            apply_mask_black_bg=apply_mask_black_bg,
        )
        if canvas.mode == "RGBA":
            canvas.paste(tile.convert("RGBA"), (x, y))
        else:
            canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print("[INFO] Selected samples:")
    for asset_name, category, step, view, tex in summary:
        print(f"  - category={category:12s} asset={asset_name:20s} step={step:02d} view={view} texture={int(tex)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a rendered-views mosaic for the paper overview figure.")
    parser.add_argument("--input-root", type=Path, default=Path("data/renders"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_figures/figures/rendered_views_mosaic.png"))
    parser.add_argument("--rows", type=int, default=4, help="Number of mosaic rows.")
    parser.add_argument("--cols", type=int, default=4, help="Number of mosaic columns.")
    parser.add_argument("--max-asset-dirs", type=int, default=80, help="Limit how many asset dirs to scan for faster iteration.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sample selection.")
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=240)
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument("--margin-x", type=int, default=None)
    parser.add_argument("--margin-y", type=int, default=None)
    parser.add_argument("--show-labels", action="store_true", default=False, help="Overlay per-tile text labels.")
    parser.add_argument("--bg", type=str, default="black", choices=["black", "white", "transparent"], help="Canvas background color.")
    parser.add_argument("--mask-black-bg", action="store_true", default=False, help="Replace each tile background with black using mask.png.")
    args = parser.parse_args()

    sample_count = int(args.rows) * int(args.cols)
    if args.bg == "black":
        bg_color = (0, 0, 0)
    elif args.bg == "white":
        bg_color = (255, 255, 255)
    else:
        bg_color = (0, 0, 0, 0)
    margin_x = int(args.margin if args.margin_x is None else args.margin_x)
    margin_y = int(args.margin if args.margin_y is None else args.margin_y)
    output = build_mosaic(
        input_root=args.input_root,
        output_path=args.output,
        sample_count=sample_count,
        max_asset_dirs=None if args.max_asset_dirs <= 0 else args.max_asset_dirs,
        tile_w=args.tile_width,
        tile_h=args.tile_height,
        margin_x=margin_x,
        margin_y=margin_y,
        columns=int(args.cols),
        seed=int(args.seed),
        show_labels=bool(args.show_labels),
        bg_color=bg_color,
        apply_mask_black_bg=bool(args.mask_black_bg),
    )
    print(f"[OK] Saved mosaic to {output}")


if __name__ == "__main__":
    main()
