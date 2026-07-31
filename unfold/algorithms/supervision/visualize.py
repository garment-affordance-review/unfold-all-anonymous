from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from unfold.algorithms.supervision.projection import (
    build_dense_a1_heatmap,
    build_dense_a2_heatmap_for_pixel,
    masked_softmax_heatmap,
)

matplotlib.use("Agg")


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float32)
    if not np.any(finite):
        return out
    v = x[finite]
    vmin = float(v.min())
    vmax = float(v.max())
    if vmax <= vmin + 1e-12:
        out[finite] = 1.0
    else:
        out[finite] = (v - vmin) / (vmax - vmin)
    return out


def _gaussian_splat_heatmap(
    height: int,
    width: int,
    points_xy: np.ndarray,
    point_weights: np.ndarray,
    sigma: float = 18.0,
    valid_mask: Optional[np.ndarray] = None,
    composite: str = "max",
) -> np.ndarray:
    heat = np.zeros((height, width), dtype=np.float32)
    if points_xy.shape[0] == 0:
        return heat

    w = np.asarray(point_weights, dtype=np.float32)
    p = np.asarray(points_xy, dtype=np.float32)
    s2 = float(max(sigma, 1.0) ** 2)
    radius = int(max(3.0, 3.0 * sigma))

    for i in range(p.shape[0]):
        px = int(round(float(p[i, 0])))
        py = int(round(float(p[i, 1])))
        ww = float(w[i])
        if ww <= 0:
            continue
        x0 = max(0, px - radius)
        x1 = min(width, px + radius + 1)
        y0 = max(0, py - radius)
        y1 = min(height, py + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - px) ** 2 + (yy - py) ** 2
        g = np.exp(-0.5 * d2 / s2, dtype=np.float32)
        patch = ww * g
        if composite == "sum":
            heat[y0:y1, x0:x1] += patch
        elif composite == "max":
            heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], patch)
        else:
            raise ValueError(f"Unknown composite mode: {composite}")

    if valid_mask is not None:
        m = np.asarray(valid_mask) > 0
        heat[~m] = 0.0

    vmax = float(heat.max()) if heat.size else 0.0
    if vmax > 1e-12:
        heat /= vmax
    return heat


def _estimate_uniform_step(points_xy: np.ndarray) -> float:
    """
    Estimate candidate spacing in image plane (pixels) from nearest-neighbor distance.
    """
    p = np.asarray(points_xy, dtype=np.float32)
    n = p.shape[0]
    if n < 2:
        return 12.0
    # O(N^2) is acceptable for current candidate sizes (~128-256).
    d2 = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    finite = np.isfinite(nn)
    if not np.any(finite):
        return 12.0
    return float(np.median(nn[finite]))


def _resolve_sigma(points_xy: np.ndarray, sigma: float) -> float:
    """
    If sigma > 0: use as absolute pixel sigma.
    If sigma <= 0: auto-set sigma from uniform sampling step.
    """
    if sigma > 0:
        return float(max(1.0, sigma))
    step = _estimate_uniform_step(points_xy)
    # Neighborhood enhancement only (not broad overlap): around 1/3~1/2 local step.
    return float(np.clip(step * 0.35, 2.0, 10.0))


def _save_heat_overlay(
    image: np.ndarray,
    heat: np.ndarray,
    out_path: Path,
    title: str,
    x1_xy: Optional[np.ndarray] = None,
    x2_xy: Optional[np.ndarray] = None,
) -> None:
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    display_heat = np.asarray(heat, dtype=np.float32)
    vmax = float(np.max(display_heat)) if display_heat.size else 0.0
    if vmax > 1e-12:
        # Training targets are probability distributions after masked softmax;
        # rescale by per-sample peak for visualization while preserving ranking.
        display_heat = display_heat / vmax
    heat_alpha = np.clip((display_heat - 0.08) / 0.92, 0.0, 1.0) * 0.80
    ax.imshow(display_heat, cmap="jet", alpha=heat_alpha, vmin=0.0, vmax=1.0)

    if x1_xy is not None and x1_xy.size == 2:
        ax.scatter(
            [x1_xy[0]],
            [x1_xy[1]],
            s=240,
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=3.6,
        )
    if x2_xy is not None and x2_xy.size == 2:
        ax.scatter(
            [x2_xy[0]],
            [x2_xy[1]],
            s=240,
            marker="o",
            facecolors="none",
            edgecolors="yellow",
            linewidths=3.6,
        )
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_result_image(
    image: np.ndarray,
    out_path: Path,
    x1_xy: Optional[np.ndarray],
    x2_xy: Optional[np.ndarray],
) -> None:
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    if x1_xy is not None and x1_xy.size == 2:
        ax.scatter(
            [x1_xy[0]],
            [x1_xy[1]],
            s=240,
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=3.6,
        )
    if x2_xy is not None and x2_xy.size == 2:
        ax.scatter(
            [x2_xy[0]],
            [x2_xy[1]],
            s=240,
            marker="o",
            facecolors="none",
            edgecolors="yellow",
            linewidths=3.6,
        )
    if x1_xy is not None and x2_xy is not None and x1_xy.size == 2 and x2_xy.size == 2:
        ax.plot([x1_xy[0], x2_xy[0]], [x1_xy[1], x2_xy[1]], color="white", linewidth=2.2, alpha=0.95)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _pin_peak_at_point(heat: np.ndarray, xy: Optional[np.ndarray], radius: int = 5) -> np.ndarray:
    if xy is None or xy.size != 2:
        return heat
    h, w = heat.shape
    px = int(round(float(xy[0])))
    py = int(round(float(xy[1])))
    x0 = max(0, px - radius)
    x1 = min(w, px + radius + 1)
    y0 = max(0, py - radius)
    y1 = min(h, py + radius + 1)
    heat2 = heat.copy()
    heat2[y0:y1, x0:x1] = 1.0
    return heat2


def save_supervision_visuals(
    rgb_path: str | Path,
    mask_path: str | Path,
    out_dir: str | Path,
    candidate_xy: np.ndarray,
    candidate_teacher_vid: np.ndarray,
    candidate_raw_vid: Optional[np.ndarray],
    a1_logits: np.ndarray,
    reward_matrix: np.ndarray,
    mask_index: Optional[np.ndarray] = None,
    render_to_teacher: Optional[np.ndarray] = None,
    face_vertex_ids: Optional[np.ndarray] = None,
    barycentric_weights: Optional[np.ndarray] = None,
    face_index: Optional[np.ndarray] = None,
    sigma: float = 18.0,
    target_tau: float = 1.0,
    filename_prefix: str = "",
) -> list[str]:
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    valid_mask = (mask > 0).astype(np.uint8)
    masked_rgb = rgb.copy()
    masked_rgb[valid_mask == 0] = 0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(filename_prefix)

    xy = np.asarray(candidate_xy, dtype=np.float32)
    n = xy.shape[0]
    a1 = np.asarray(a1_logits, dtype=np.float32)
    r = np.asarray(reward_matrix, dtype=np.float32)
    cand_tvid = np.asarray(candidate_teacher_vid, dtype=np.int64)
    cand_raw = np.asarray(candidate_raw_vid, dtype=np.int64) if candidate_raw_vid is not None else None

    best_x1_idx = int(np.nanargmax(np.where(np.isfinite(a1), a1, -np.inf))) if n > 0 else -1
    best_x2_idx = -1
    if 0 <= best_x1_idx < n and r.shape == (n, n):
        row = r[best_x1_idx].copy()
        if best_x1_idx < row.shape[0]:
            row[best_x1_idx] = -np.inf
        if np.isfinite(row).any():
            best_x2_idx = int(np.nanargmax(np.where(np.isfinite(row), row, -np.inf)))

    best_x1_xy = xy[best_x1_idx] if 0 <= best_x1_idx < n else None
    best_x2_xy = xy[best_x2_idx] if 0 <= best_x2_idx < n else None

    a1_w = _normalize_scores(a1)
    use_dense_soft = (
        face_vertex_ids is not None
        and barycentric_weights is not None
        and cand_raw is not None
        and cand_raw.size == n
        and r.shape == (n, n)
    )
    use_dense = mask_index is not None and render_to_teacher is not None and cand_tvid.size == n
    if use_dense_soft:
        fvid = np.asarray(face_vertex_ids, dtype=np.int64)
        if fvid.ndim != 3 or fvid.shape[-1] != 3:
            use_dense_soft = False
        bw = np.asarray(barycentric_weights, dtype=np.float32)
        if bw.shape != fvid.shape:
            use_dense_soft = False
        if use_dense_soft:
            fi = np.asarray(face_index, dtype=np.int64) if face_index is not None else np.full(fvid.shape[:2], 0, dtype=np.int64)
            pix_valid = (valid_mask > 0) & (fi >= 0)
            a1_value_map = build_dense_a1_heatmap(
                mask_np=valid_mask,
                face_index=fi,
                face_vertex_ids=fvid,
                barycentric_weights=bw,
                candidate_raw_vid=cand_raw,
                a1_logits=a1,
            )
            a1_heat = masked_softmax_heatmap(a1_value_map, pix_valid, tau=target_tau)

            a2_heat = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            if np.any(pix_valid):
                masked = np.where(pix_valid, a1_heat, -np.inf)
                flat = int(np.nanargmax(masked))
                y1, x1 = np.unravel_index(flat, a1_heat.shape)
                a2_value_map, _, ok = build_dense_a2_heatmap_for_pixel(
                    x=int(x1),
                    y=int(y1),
                    mask_np=valid_mask,
                    face_index=fi,
                    face_vertex_ids=fvid,
                    barycentric_weights=bw,
                    candidate_raw_vid=cand_raw,
                    reward_matrix=r,
                )
                if ok:
                    a2_heat = masked_softmax_heatmap(a2_value_map, pix_valid, tau=target_tau)
                best_x1_xy = np.array([float(x1), float(y1)], dtype=np.float32)
                if ok:
                    sel = np.where(pix_valid, a2_heat, -np.inf)
                    rad = 3
                    y0 = max(0, y1 - rad)
                    y1b = min(a2_heat.shape[0], y1 + rad + 1)
                    x0 = max(0, x1 - rad)
                    x1b = min(a2_heat.shape[1], x1 + rad + 1)
                    sel[y0:y1b, x0:x1b] = -np.inf
                    if np.any(np.isfinite(sel)):
                        flat2 = int(np.nanargmax(sel))
                        y2, x2 = np.unravel_index(flat2, a2_heat.shape)
                        best_x2_xy = np.array([float(x2), float(y2)], dtype=np.float32)
            use_dense = False
    if use_dense_soft:
        pass
    elif use_dense:
        mi = np.asarray(mask_index, dtype=np.int64)
        r2t = np.asarray(render_to_teacher, dtype=np.int64)
        tmax = int(max(int(r2t.max(initial=-1)), int(cand_tvid.max(initial=-1)))) + 1
        tmax = max(tmax, 1)

        # teacher vertex score table (only candidate teacher ids are assigned valid scores)
        score_t_a1 = np.full((tmax,), -1.0, dtype=np.float32)
        score_t_a1[cand_tvid] = a1_w

        a1_heat = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        valid_render = (mi >= 0) & (mi < r2t.shape[0])
        teacher_pix = np.full(mi.shape, -1, dtype=np.int64)
        teacher_pix[valid_render] = r2t[mi[valid_render]]
        valid_teacher = (teacher_pix >= 0) & (teacher_pix < tmax)
        a1_heat[valid_teacher] = score_t_a1[teacher_pix[valid_teacher]]
        a1_heat[~(valid_mask > 0)] = 0.0

        a2_heat = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        if 0 <= best_x1_idx < n and r.shape == (n, n):
            row = r[best_x1_idx].copy()
            if best_x1_idx < row.shape[0]:
                row[best_x1_idx] = -np.inf
            a2_w = _normalize_scores(row)
            score_t_a2 = np.full((tmax,), -1.0, dtype=np.float32)
            score_t_a2[cand_tvid] = a2_w
            a2_heat[valid_teacher] = score_t_a2[teacher_pix[valid_teacher]]
            a2_heat[~(valid_mask > 0)] = 0.0
    else:
        sigma_eff = _resolve_sigma(points_xy=xy, sigma=sigma)
        a1_heat = _gaussian_splat_heatmap(
            height=rgb.shape[0],
            width=rgb.shape[1],
            points_xy=xy,
            point_weights=a1_w,
            sigma=sigma_eff,
            valid_mask=valid_mask,
            composite="max",
        )
        a2_heat = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        if 0 <= best_x1_idx < n and r.shape == (n, n):
            row = r[best_x1_idx].copy()
            if best_x1_idx < row.shape[0]:
                row[best_x1_idx] = -np.inf
            a2_w = _normalize_scores(row)
            a2_heat = _gaussian_splat_heatmap(
                height=rgb.shape[0],
                width=rgb.shape[1],
                points_xy=xy,
                point_weights=a2_w,
                sigma=sigma_eff,
                valid_mask=valid_mask,
                composite="max",
            )

    out_files: list[str] = []
    p1 = out_dir / f"{prefix}a1.png"
    _save_heat_overlay(
        image=masked_rgb,
        heat=a1_heat,
        out_path=p1,
        title="A1 Heatmap",
        x1_xy=best_x1_xy,
    )
    out_files.append(str(p1))

    p2 = out_dir / f"{prefix}a2.png"
    _save_heat_overlay(
        image=masked_rgb,
        heat=a2_heat,
        out_path=p2,
        title="A2 Heatmap",
        x2_xy=best_x2_xy,
    )
    out_files.append(str(p2))

    p3 = out_dir / f"{prefix}result.png"
    _save_result_image(
        image=masked_rgb,
        out_path=p3,
        x1_xy=best_x1_xy,
        x2_xy=best_x2_xy,
    )
    out_files.append(str(p3))
    return out_files
