from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine as tv_affine


@dataclass(frozen=True)
class MaskAugmentConfig:
    enabled: bool = True
    min_area_ratio: float = 0.45
    max_area_ratio: float = 1.35
    max_retries: int = 8
    morph_radius_min: int = 0
    morph_radius_max: int = 6
    morph_mode: str = "random"
    blur_prob: float = 0.35
    blur_radius_max: float = 2.5
    hole_count_max: int = 16
    hole_radius_max: int = 18
    patch_count_max: int = 8
    patch_radius_max: int = 26
    exterior_patch_prob: float = 0.35
    contour_dropout_prob: float = 0.35
    contour_band_width: int = 5


@dataclass(frozen=True)
class RGBAugmentConfig:
    enabled: bool = False
    brightness_delta: float = 0.08
    contrast_delta: float = 0.08
    saturation_delta: float = 0.06
    hue_delta: float = 0.0
    gamma_delta: float = 0.0
    blur_prob: float = 0.15
    blur_radius_max: float = 1.2
    noise_prob: float = 0.15
    noise_std: float = 0.02
    jpeg_prob: float = 0.0
    jpeg_quality_min: int = 75
    jpeg_quality_max: int = 95


@dataclass(frozen=True)
class GeometricAugmentConfig:
    enabled: bool = False
    prob: float = 0.35
    max_rotate_deg: float = 10.0
    min_scale: float = 0.96
    max_scale: float = 1.04
    max_translate_frac_x: float = 0.03
    max_translate_frac_y: float = 0.03


def _ellipse_draw(mask: np.ndarray, cx: int, cy: int, rx: int, ry: int, value: int) -> None:
    h, w = mask.shape
    x0 = max(0, cx - rx)
    x1 = min(w, cx + rx + 1)
    y0 = max(0, cy - ry)
    y1 = min(h, cy + ry + 1)
    if x0 >= x1 or y0 >= y1:
        return
    # Tiny radius edits on the native mask grid can quantize into star-like artifacts.
    # Rasterize on a small supersampled canvas first, then downsample back to the mask grid.
    supersample = 4
    pad = 1
    local_h = y1 - y0
    local_w = x1 - x0
    hi_h = max(1, local_h * supersample)
    hi_w = max(1, local_w * supersample)
    hi = Image.new("L", (hi_w, hi_h), color=0)
    draw = ImageDraw.Draw(hi)

    cx_local = (float(cx - x0) + 0.5) * supersample
    cy_local = (float(cy - y0) + 0.5) * supersample
    rx_hi = max(float(rx) * supersample, 1.0)
    ry_hi = max(float(ry) * supersample, 1.0)
    draw.ellipse(
        (
            cx_local - rx_hi - pad,
            cy_local - ry_hi - pad,
            cx_local + rx_hi + pad,
            cy_local + ry_hi + pad,
        ),
        fill=255,
    )
    lo = hi.resize((local_w, local_h), resample=Image.Resampling.BOX)
    lo_mask = np.asarray(lo, dtype=np.uint8) >= 127
    mask[y0:y1, x0:x1][lo_mask] = value


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return (np.array(img.filter(ImageFilter.MaxFilter(size=radius * 2 + 1))) > 0).astype(np.uint8)


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return (np.array(img.filter(ImageFilter.MinFilter(size=radius * 2 + 1))) > 0).astype(np.uint8)


def _boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    outer = _binary_dilate(mask, width)
    inner = _binary_erode(mask, width)
    return ((outer > 0) & ~(inner > 0)).astype(np.uint8)


def _mask_size_caps(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return 0, 2, 2, 1
    box_h = int(ys.max() - ys.min() + 1)
    box_w = int(xs.max() - xs.min() + 1)
    short_side = max(1, min(box_h, box_w))
    morph_cap = max(0, short_side // 40)
    hole_radius_cap = max(2, short_side // 20)
    patch_radius_cap = max(2, short_side // 24)
    hole_count_cap = max(1, short_side // 28)
    return morph_cap, hole_radius_cap, patch_radius_cap, hole_count_cap


def perturb_mask(mask: np.ndarray, cfg: MaskAugmentConfig, rng: random.Random) -> tuple[np.ndarray, float]:
    base = (np.asarray(mask) > 0).astype(np.uint8)
    if not cfg.enabled:
        return base, 1.0

    base_area = int(base.sum())
    if base_area <= 0:
        return base, 1.0

    h, w = base.shape
    morph_cap, hole_radius_cap, patch_radius_cap, hole_count_cap = _mask_size_caps(base)
    for _ in range(max(cfg.max_retries, 1)):
        aug = base.copy()
        effective_morph_max = min(int(cfg.morph_radius_max), morph_cap)
        effective_morph_min = min(int(cfg.morph_radius_min), effective_morph_max)
        radius = rng.randint(effective_morph_min, effective_morph_max) if effective_morph_max > 0 else 0
        if radius > 0:
            if cfg.morph_mode == "erode":
                aug = _binary_erode(aug, radius)
            elif cfg.morph_mode == "dilate":
                aug = _binary_dilate(aug, radius)
            else:
                if rng.random() < 0.5:
                    aug = _binary_erode(aug, radius)
                else:
                    aug = _binary_dilate(aug, radius)

        if rng.random() < cfg.contour_dropout_prob:
            band = _boundary_band(aug, cfg.contour_band_width)
            band_yx = np.argwhere(band > 0)
            max_drop_count = min(int(cfg.patch_count_max), max(1, band_yx.shape[0] // 200 + 1))
            if band_yx.size > 0 and max_drop_count > 0:
                drop_count = rng.randint(1, max_drop_count)
                for _ in range(drop_count):
                    py, px = band_yx[rng.randrange(band_yx.shape[0])]
                    rr = rng.randint(4, max(5, cfg.patch_radius_max))
                    _ellipse_draw(aug, int(px), int(py), rr, rr, 0)

        effective_hole_count_max = min(int(cfg.hole_count_max), hole_count_cap)
        hole_count = rng.randint(0, effective_hole_count_max)
        fg_yx = np.argwhere(aug > 0)
        for _ in range(hole_count):
            if fg_yx.size == 0:
                break
            py, px = fg_yx[rng.randrange(fg_yx.shape[0])]
            max_hole_radius = min(int(cfg.hole_radius_max), hole_radius_cap)
            rx = rng.randint(2, max(2, max_hole_radius))
            ry = rng.randint(2, max(2, max_hole_radius))
            _ellipse_draw(aug, int(px), int(py), rx, ry, 0)

        patch_count = rng.randint(0, cfg.patch_count_max)
        fg_yx = np.argwhere(aug > 0)
        outer_band = (_boundary_band(aug, max(1, cfg.contour_band_width)) > 0) & (aug == 0)
        outer_yx = np.argwhere(outer_band)
        for _ in range(patch_count):
            use_outer = outer_yx.size > 0 and rng.random() < cfg.exterior_patch_prob
            if use_outer:
                py, px = outer_yx[rng.randrange(outer_yx.shape[0])]
            else:
                if fg_yx.size == 0:
                    break
                py, px = fg_yx[rng.randrange(fg_yx.shape[0])]
            max_patch_radius = min(int(cfg.patch_radius_max), patch_radius_cap)
            rx = rng.randint(2, max(2, max_patch_radius))
            ry = rng.randint(2, max(2, max_patch_radius))
            _ellipse_draw(aug, int(px), int(py), rx, ry, 1)

        if rng.random() < cfg.blur_prob:
            blur_radius = rng.uniform(0.4, max(cfg.blur_radius_max, 0.5))
            img = Image.fromarray(aug.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(radius=blur_radius))
            aug = (np.array(img) >= 127).astype(np.uint8)

        area_ratio = float(aug.sum()) / float(base_area)
        if cfg.min_area_ratio <= area_ratio <= cfg.max_area_ratio and int(aug.sum()) > 0:
            return aug, area_ratio

    return base, 1.0


def perturb_rgb(rgb: np.ndarray, cfg: RGBAugmentConfig, rng: random.Random) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32)
    if not cfg.enabled:
        return image

    x = np.clip(image / 255.0, 0.0, 1.0)

    if cfg.brightness_delta > 0.0:
        x = np.clip(x * rng.uniform(1.0 - cfg.brightness_delta, 1.0 + cfg.brightness_delta), 0.0, 1.0)

    if cfg.contrast_delta > 0.0:
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = np.clip((x - mean) * rng.uniform(1.0 - cfg.contrast_delta, 1.0 + cfg.contrast_delta) + mean, 0.0, 1.0)

    if cfg.saturation_delta > 0.0:
        gray = np.dot(x, np.asarray([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float32)[..., None]
        x = np.clip(
            gray + (x - gray) * rng.uniform(1.0 - cfg.saturation_delta, 1.0 + cfg.saturation_delta),
            0.0,
            1.0,
        )

    if cfg.hue_delta > 0.0:
        shift = rng.uniform(-cfg.hue_delta, cfg.hue_delta)
        hsv = np.asarray(
            Image.fromarray((x * 255.0).round().astype(np.uint8), mode="RGB").convert("HSV"),
            dtype=np.uint8,
        ).copy()
        hue_shift = int(round(shift * 255.0))
        hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + hue_shift) % 256).astype(np.uint8)
        x = np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB"), dtype=np.float32) / 255.0

    if cfg.gamma_delta > 0.0:
        gamma = rng.uniform(1.0 - cfg.gamma_delta, 1.0 + cfg.gamma_delta)
        gamma = max(gamma, 1e-3)
        x = np.clip(np.power(x, gamma, dtype=np.float32), 0.0, 1.0)

    if cfg.blur_prob > 0.0 and rng.random() < cfg.blur_prob:
        blur_radius = rng.uniform(0.2, max(cfg.blur_radius_max, 0.25))
        x = np.asarray(
            Image.fromarray((x * 255.0).round().astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=blur_radius)),
            dtype=np.float32,
        ) / 255.0

    if cfg.jpeg_prob > 0.0 and rng.random() < cfg.jpeg_prob:
        qmin = int(min(cfg.jpeg_quality_min, cfg.jpeg_quality_max))
        qmax = int(max(cfg.jpeg_quality_min, cfg.jpeg_quality_max))
        quality = int(rng.randint(qmin, qmax))
        buffer = io.BytesIO()
        Image.fromarray((x * 255.0).round().astype(np.uint8), mode="RGB").save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=2,
        )
        buffer.seek(0)
        x = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255.0

    if cfg.noise_prob > 0.0 and rng.random() < cfg.noise_prob and cfg.noise_std > 0.0:
        x = np.clip(x + np.random.normal(0.0, cfg.noise_std, size=x.shape).astype(np.float32), 0.0, 1.0)

    return (x * 255.0).round().clip(0.0, 255.0).astype(np.float32)


def _affine_image(
    image: np.ndarray,
    *,
    angle: float,
    translate: tuple[int, int],
    scale: float,
    interpolation: InterpolationMode,
    fill: float | list[float],
) -> np.ndarray:
    tensor = torch.from_numpy(image)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).float()
        out = tv_affine(
            tensor,
            angle=angle,
            translate=[int(translate[0]), int(translate[1])],
            scale=float(scale),
            shear=[0.0, 0.0],
            interpolation=interpolation,
            fill=[float(fill)],
        )
        return out.squeeze(0).cpu().numpy()

    tensor = tensor.permute(2, 0, 1).float()
    fill_values = fill if isinstance(fill, list) else [float(fill)] * int(tensor.shape[0])
    out = tv_affine(
        tensor,
        angle=angle,
        translate=[int(translate[0]), int(translate[1])],
        scale=float(scale),
        shear=[0.0, 0.0],
        interpolation=interpolation,
        fill=fill_values,
    )
    return out.permute(1, 2, 0).cpu().numpy()


def _transform_points(
    sampled_x1_xy: np.ndarray,
    *,
    width: int,
    height: int,
    angle: float,
    translate: tuple[int, int],
    scale: float,
) -> np.ndarray:
    points = np.asarray(sampled_x1_xy, dtype=np.float32).copy()
    valid = (points[:, 0] >= 0.0) & (points[:, 1] >= 0.0)
    if not np.any(valid):
        return points

    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    theta = np.deg2rad(float(angle))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    # Match torchvision.affine's positive-angle image-space rotation.
    rotation = np.asarray([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    shift = np.asarray([cx + float(translate[0]), cy + float(translate[1])], dtype=np.float32)

    xy = points[valid] - np.asarray([cx, cy], dtype=np.float32)[None, :]
    transformed = (float(scale) * (xy @ rotation.T)) + shift[None, :]
    in_bounds = (
        (transformed[:, 0] >= 0.0)
        & (transformed[:, 0] <= float(width - 1))
        & (transformed[:, 1] >= 0.0)
        & (transformed[:, 1] <= float(height - 1))
    )
    points[valid] = transformed
    invalid_idx = np.flatnonzero(valid)[~in_bounds]
    points[invalid_idx] = -1.0
    return points


def perturb_geometry(
    *,
    rgb: np.ndarray,
    mask: np.ndarray,
    a1_value_map: np.ndarray,
    a1_valid_mask: np.ndarray,
    sampled_x1_xy: np.ndarray,
    a2_value_map: np.ndarray,
    a2_valid_mask: np.ndarray,
    cfg: GeometricAugmentConfig,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not cfg.enabled or rng.random() >= float(cfg.prob):
        return rgb, mask, a1_value_map, a1_valid_mask, sampled_x1_xy, a2_value_map, a2_valid_mask

    height, width = int(mask.shape[0]), int(mask.shape[1])
    angle = rng.uniform(-float(cfg.max_rotate_deg), float(cfg.max_rotate_deg))
    scale = rng.uniform(float(cfg.min_scale), float(cfg.max_scale))
    tx = int(round(rng.uniform(-float(cfg.max_translate_frac_x), float(cfg.max_translate_frac_x)) * float(width)))
    ty = int(round(rng.uniform(-float(cfg.max_translate_frac_y), float(cfg.max_translate_frac_y)) * float(height)))
    translate = (tx, ty)

    rgb_aug = _affine_image(
        np.asarray(rgb, dtype=np.float32),
        angle=angle,
        translate=translate,
        scale=scale,
        interpolation=InterpolationMode.BILINEAR,
        fill=[0.0, 0.0, 0.0],
    ).round().clip(0.0, 255.0).astype(np.uint8)
    mask_aug = (
        _affine_image(
            np.asarray(mask, dtype=np.float32),
            angle=angle,
            translate=translate,
            scale=scale,
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )
        >= 0.5
    ).astype(np.uint8)
    a1_value_aug = _affine_image(
        np.asarray(a1_value_map, dtype=np.float32),
        angle=angle,
        translate=translate,
        scale=scale,
        interpolation=InterpolationMode.BILINEAR,
        fill=float(np.min(a1_value_map)),
    ).astype(np.float32, copy=False)
    a1_valid_aug = (
        _affine_image(
            np.asarray(a1_valid_mask, dtype=np.float32),
            angle=angle,
            translate=translate,
            scale=scale,
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )
        >= 0.5
    ).astype(np.uint8)

    a2_value_aug = np.empty_like(a2_value_map, dtype=np.float32)
    a2_valid_aug = np.empty_like(a2_valid_mask, dtype=np.uint8)
    for idx in range(int(a2_value_map.shape[0])):
        a2_value_aug[idx] = _affine_image(
            np.asarray(a2_value_map[idx], dtype=np.float32),
            angle=angle,
            translate=translate,
            scale=scale,
            interpolation=InterpolationMode.BILINEAR,
            fill=float(np.min(a2_value_map[idx])),
        ).astype(np.float32, copy=False)
        a2_valid_aug[idx] = (
            _affine_image(
                np.asarray(a2_valid_mask[idx], dtype=np.float32),
                angle=angle,
                translate=translate,
                scale=scale,
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )
            >= 0.5
        ).astype(np.uint8)

    sampled_x1_xy_aug = _transform_points(
        sampled_x1_xy,
        width=width,
        height=height,
        angle=angle,
        translate=translate,
        scale=scale,
    )
    return rgb_aug, mask_aug, a1_value_aug, a1_valid_aug, sampled_x1_xy_aug, a2_value_aug, a2_valid_aug
