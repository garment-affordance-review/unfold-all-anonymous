#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _crop_white_margin(img: Image.Image, threshold: int = 245, pad: int = 0) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    xs: list[int] = []
    ys: list[int] = []
    for y in range(h):
        for x in range(w):
            r, g, b = rgb.getpixel((x, y))
            if not (r >= threshold and g >= threshold and b >= threshold):
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return img
    left = max(min(xs) - pad, 0)
    top = max(min(ys) - pad, 0)
    right = min(max(xs) + 1 + pad, w)
    bottom = min(max(ys) + 1 + pad, h)
    return img.crop((left, top, right, bottom))


def _fit(img: Image.Image, width: int, height: int) -> Image.Image:
    src = img.convert("RGBA")
    scale = min(width / src.width, height / src.height)
    new_size = (max(1, int(src.width * scale)), max(1, int(src.height * scale)))
    return src.resize(new_size, Image.Resampling.LANCZOS)


def _load_meta(sample_dir: Path) -> dict:
    return json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))


def _masked_rgb(rgb_path: Path, mask_path: Path, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rgb = Image.open(rgb_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    mask = Image.open(mask_path).convert("L").resize(size, Image.Resampling.NEAREST)
    mask_np = (np.asarray(mask) > 0).astype(np.float32)
    rgb_np = np.asarray(rgb).astype(np.float32)
    rgb_np[mask_np <= 0] = 0.0
    return rgb_np, mask_np


def _prepare_center(result_path: Path, rgb_np: np.ndarray, mask_np: np.ndarray) -> Image.Image:
    result = np.asarray(_crop_white_margin(Image.open(result_path), pad=2).convert("RGB").resize((rgb_np.shape[1], rgb_np.shape[0]), Image.Resampling.LANCZOS)).astype(np.float32)
    result[mask_np <= 0] = 0.0
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")


def _white_to_alpha(img: Image.Image, threshold: int = 245) -> Image.Image:
    rgba = img.convert("RGBA")
    out = []
    for r, g, b, a in rgba.getdata():
        if r >= threshold and g >= threshold and b >= threshold:
            out.append((r, g, b, 0))
        else:
            out.append((r, g, b, a))
    rgba.putdata(out)
    return rgba


def _render_clean_overlay(overlay_path: Path, rgb_np: np.ndarray, mask_np: np.ndarray) -> Image.Image:
    overlay = _crop_white_margin(Image.open(overlay_path), pad=2).convert("RGBA").resize((rgb_np.shape[1], rgb_np.shape[0]), Image.Resampling.LANCZOS)
    overlay = _white_to_alpha(overlay)
    return overlay


def build_triptych(sample_dir: Path, output_path: Path, width: int = 1500, height: int = 500) -> Path:
    meta = _load_meta(sample_dir)
    rgb_path = Path(meta["source"]["rgb"])
    mask_path = Path(meta["source"]["mask"])

    slot_w = width // 3
    slot_h = height
    rgb_np, mask_np = _masked_rgb(rgb_path, mask_path, (slot_w, slot_h))

    center = _prepare_center(sample_dir / "result.png", rgb_np, mask_np)
    left = _render_clean_overlay(sample_dir / "a1.png", rgb_np, mask_np)
    right = _render_clean_overlay(sample_dir / "a2.png", rgb_np, mask_np)

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for idx, panel in enumerate([left, center, right]):
        fitted = _fit(panel, slot_w, slot_h)
        x = idx * slot_w + max((slot_w - fitted.width) // 2, 0)
        y = max((slot_h - fitted.height) // 2, 0)
        canvas.alpha_composite(fitted, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean seamless A1-RGB-A2 triptych.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=500)
    args = parser.parse_args()

    out = build_triptych(args.sample_dir, args.output, width=args.width, height=args.height)
    print(f"[OK] Saved triptych to {out}")


if __name__ == "__main__":
    main()
