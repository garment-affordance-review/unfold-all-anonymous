#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_meta(sample_dir: Path) -> dict:
    return json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))


def _apply_mask_to_image(img_path: Path, mask_path: Path) -> Image.Image:
    img = Image.open(img_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L").resize(img.size, Image.Resampling.NEAREST)
    mask_np = np.asarray(mask)
    alpha = np.where(mask_np > 0, 255, 0).astype(np.uint8)
    arr = np.asarray(img).copy()
    arr[..., 3] = np.minimum(arr[..., 3], alpha)
    return Image.fromarray(arr, mode="RGBA")


def _fit(img: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / img.width, height / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def build_triptych(sample_dir: Path, output_path: Path, width: int = 1500, height: int = 500, gap: int = 0) -> Path:
    meta = _load_meta(sample_dir)
    mask_path = Path(meta["source"]["mask"])

    images = [
        _apply_mask_to_image(sample_dir / "a1.png", mask_path),
        _apply_mask_to_image(sample_dir / "result.png", mask_path),
        _apply_mask_to_image(sample_dir / "a2.png", mask_path),
    ]

    slot_w = (width - 2 * gap) // 3
    slot_h = height
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    for idx, image in enumerate(images):
        fitted = _fit(image, slot_w, slot_h)
        x0 = idx * (slot_w + gap)
        x = x0 + max((slot_w - fitted.width) // 2, 0)
        y = max((slot_h - fitted.height) // 2, 0)
        canvas.alpha_composite(fitted, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a raw triptych masked by the original render mask.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--gap", type=int, default=0)
    args = parser.parse_args()

    out = build_triptych(args.sample_dir, args.output, width=args.width, height=args.height, gap=args.gap)
    print(f"[OK] Saved triptych to {out}")


if __name__ == "__main__":
    main()
