#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


THEMES = {
    "white": (255, 255, 255, 255),
    "warm": (242, 238, 230, 255),
    "charcoal": (20, 22, 26, 255),
    "transparent": (0, 0, 0, 0),
}


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


def _remove_white_to_alpha(img: Image.Image, threshold: int = 245) -> Image.Image:
    rgba = img.convert("RGBA")
    out = []
    for r, g, b, a in rgba.getdata():
        if r >= threshold and g >= threshold and b >= threshold:
            out.append((r, g, b, 0))
        else:
            out.append((r, g, b, a))
    rgba.putdata(out)
    return rgba


def _crop_transparent_margin(img: Image.Image, pad: int = 0) -> Image.Image:
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return rgba
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(rgba.width, right + pad)
    bottom = min(rgba.height, bottom + pad)
    return rgba.crop((left, top, right, bottom))


def build_panel(
    result_path: Path,
    a1_path: Path,
    a2_path: Path,
    output_path: Path,
    margin: int = 16,
    gap: int = 12,
    bg_theme: str = "white",
    remove_white_alpha: bool = True,
    layout: str = "horizontal",
) -> Path:
    result = _crop_white_margin(Image.open(result_path), pad=2)
    a1 = _crop_white_margin(Image.open(a1_path), pad=2)
    a2 = _crop_white_margin(Image.open(a2_path), pad=2)
    if remove_white_alpha:
        result = _remove_white_to_alpha(result)
        a1 = _remove_white_to_alpha(a1)
        a2 = _remove_white_to_alpha(a2)

    images = [a1, result, a2]

    if layout == "vertical":
        panel_w = 560
        slot_w = panel_w
        fitted_images = [_fit(img, slot_w, 460) for img in images]
        panel_h = sum(img.height for img in fitted_images) + 2 * gap
    else:
        panel_w = 1500
        panel_h = 500
        slot_w = (panel_w - 2 * gap) // 3
        slot_h = panel_h

    canvas = Image.new("RGBA", (panel_w, panel_h), THEMES[bg_theme])

    if layout == "vertical":
        y_cursor = 0
        for fitted in fitted_images:
            x0 = 0
            y0 = y_cursor
            x = x0 + max((slot_w - fitted.width) // 2, 0)
            y = y0
            canvas.alpha_composite(fitted, (x, y))
            y_cursor += fitted.height + gap
    else:
        for idx, image in enumerate(images):
            fitted = _fit(image, slot_w, slot_h)
            x0 = idx * (slot_w + gap)
            y0 = 0
            x = x0 + max((slot_w - fitted.width) // 2, 0)
            y = y0 + max((slot_h - fitted.height) // 2, 0)
            canvas.alpha_composite(fitted, (x, y))

    canvas = _crop_transparent_margin(canvas, pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean RGB+A1+A2 detail panel for the paper figure.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--bg-theme", type=str, default="white", choices=sorted(THEMES.keys()))
    parser.add_argument("--keep-original-bg", action="store_true", default=False)
    parser.add_argument("--layout", type=str, default="horizontal", choices=["horizontal", "vertical"])
    args = parser.parse_args()

    out = build_panel(
        result_path=args.sample_dir / "result.png",
        a1_path=args.sample_dir / "a1.png",
        a2_path=args.sample_dir / "a2.png",
        output_path=args.output,
        margin=args.margin,
        gap=args.gap,
        bg_theme=args.bg_theme,
        remove_white_alpha=not bool(args.keep_original_bg),
        layout=args.layout,
    )
    print(f"[OK] Saved panel to {out}")


if __name__ == "__main__":
    main()
