from __future__ import annotations

from typing import Optional

import numpy as np


def normalize_finite_scores(values: np.ndarray) -> np.ndarray:
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


def finite_scores_with_fill(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float32)
    if not np.any(finite):
        return out
    out[finite] = x[finite]
    fill = float(np.min(x[finite]))
    out[~finite] = fill
    return out


def masked_softmax_heatmap(values: np.ndarray, valid_mask: np.ndarray, tau: float = 1.0) -> np.ndarray:
    heat = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask) > 0
    out = np.zeros_like(heat, dtype=np.float32)
    if not np.any(valid):
        return out
    x = np.asarray(heat[valid], dtype=np.float64) / float(max(tau, 1e-6))
    x = x - float(np.max(x))
    expv = np.exp(x)
    z = float(np.sum(expv))
    if z <= 0.0:
        out[valid] = 1.0 / float(np.count_nonzero(valid))
    else:
        out[valid] = (expv / z).astype(np.float32, copy=False)
    return out


def fill_invalid_with_margin(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    margin_ratio: float = 0.1,
    min_margin: float = 1e-3,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask) > 0
    out = x.copy()
    if not np.any(valid):
        out[...] = 0.0
        return out
    valid_values = x[valid]
    vmin = float(np.min(valid_values))
    vmax = float(np.max(valid_values))
    vrange = float(vmax - vmin)
    margin = max(float(min_margin), float(margin_ratio) * max(vrange, 0.0))
    fill_value = vmin - margin
    out[~valid] = fill_value
    return out


def _visible_face_local_tris(
    *,
    face_index: np.ndarray,
    face_vertex_ids: np.ndarray,
    candidate_raw_vid: np.ndarray,
) -> np.ndarray:
    face_index = np.asarray(face_index, dtype=np.int64)
    face_vertex_ids = np.asarray(face_vertex_ids, dtype=np.int64)
    cand_raw = np.asarray(candidate_raw_vid, dtype=np.int64)
    pix_valid = face_index >= 0
    if not np.any(pix_valid) or cand_raw.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    tri_raw = face_vertex_ids[pix_valid]
    face_ids = face_index[pix_valid]
    _, first_idx = np.unique(face_ids, return_index=True)
    tri_raw = tri_raw[first_idx]

    rmax = int(max(int(tri_raw.max(initial=-1)), int(cand_raw.max(initial=-1)))) + 1
    rmax = max(rmax, 1)
    r2local = np.full((rmax,), -1, dtype=np.int64)
    valid_cand = (cand_raw >= 0) & (cand_raw < rmax)
    r2local[cand_raw[valid_cand]] = np.arange(cand_raw.shape[0], dtype=np.int64)[valid_cand]

    tri_valid = (tri_raw >= 0) & (tri_raw < rmax)
    tri_local = np.full_like(tri_raw, -1, dtype=np.int64)
    tri_local[tri_valid] = r2local[tri_raw[tri_valid]]
    keep = np.all(tri_local >= 0, axis=1)
    return tri_local[keep].astype(np.int64, copy=False)


def _smooth_vertex_scores(
    *,
    values: np.ndarray,
    face_index: np.ndarray,
    face_vertex_ids: np.ndarray,
    candidate_raw_vid: np.ndarray,
    iters: int = 2,
    self_weight: float = 0.6,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if x.size == 0 or iters <= 0:
        return x
    tris = _visible_face_local_tris(
        face_index=face_index,
        face_vertex_ids=face_vertex_ids,
        candidate_raw_vid=candidate_raw_vid,
    )
    if tris.shape[0] == 0:
        return x

    edges = np.concatenate(
        [
            tris[:, [0, 1]],
            tris[:, [1, 2]],
            tris[:, [2, 0]],
            tris[:, [1, 0]],
            tris[:, [2, 1]],
            tris[:, [0, 2]],
        ],
        axis=0,
    ).astype(np.int64, copy=False)
    keep = edges[:, 0] != edges[:, 1]
    edges = edges[keep]
    if edges.shape[0] == 0:
        return x

    out = x.astype(np.float32, copy=True)
    sw = float(np.clip(self_weight, 0.0, 1.0))
    for _ in range(int(iters)):
        nb_sum = np.zeros_like(out, dtype=np.float32)
        nb_deg = np.zeros_like(out, dtype=np.float32)
        np.add.at(nb_sum, edges[:, 0], out[edges[:, 1]])
        np.add.at(nb_deg, edges[:, 0], 1.0)
        nb_mean = np.where(nb_deg > 0, nb_sum / np.maximum(nb_deg, 1.0), out)
        out = sw * out + (1.0 - sw) * nb_mean
    return out


def row_softmax_masked(logits: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64) / float(max(tau, 1e-6))
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    if not np.any(finite):
        return out.astype(np.float32)
    xf = x[finite]
    m = float(np.max(xf))
    expv = np.exp(xf - m)
    z = float(np.sum(expv))
    if z <= 0.0:
        return out.astype(np.float32)
    out[finite] = expv / z
    return out.astype(np.float32)


def compute_top1_margin(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float32)
    finite = v[np.isfinite(v)]
    if finite.size < 2:
        return 0.0
    order = np.sort(finite)
    return float(order[-1] - order[-2])


def compute_reward_row_margin(reward_matrix: np.ndarray, x1_idx: int) -> float:
    if reward_matrix.ndim != 2 or reward_matrix.shape[0] == 0:
        return 0.0
    row = np.asarray(reward_matrix[x1_idx], dtype=np.float32).copy()
    if 0 <= x1_idx < row.shape[0]:
        row[x1_idx] = -np.inf
    return compute_top1_margin(row)


def build_projection_valid_mask(
    *,
    mask_np: np.ndarray,
    face_index: np.ndarray,
    face_vertex_ids: np.ndarray,
    barycentric_weights: np.ndarray,
    candidate_raw_vid: np.ndarray,
) -> np.ndarray:
    mask_bool = np.asarray(mask_np) > 0
    face_index = np.asarray(face_index, dtype=np.int64)
    face_vertex_ids = np.asarray(face_vertex_ids, dtype=np.int64)
    barycentric_weights = np.asarray(barycentric_weights, dtype=np.float32)
    cand_raw = np.asarray(candidate_raw_vid, dtype=np.int64)

    if face_vertex_ids.ndim != 3 or face_vertex_ids.shape[-1] != 3:
        raise ValueError(f"face_vertex_ids must be [H,W,3], got {face_vertex_ids.shape}")
    if barycentric_weights.shape != face_vertex_ids.shape:
        raise ValueError(
            f"barycentric_weights shape must match face_vertex_ids: {barycentric_weights.shape} vs {face_vertex_ids.shape}"
        )

    rmax = int(max(int(face_vertex_ids.max(initial=-1)), int(cand_raw.max(initial=-1)))) + 1
    rmax = max(rmax, 1)
    cand_valid = (cand_raw >= 0) & (cand_raw < rmax)
    cand_raw = cand_raw[cand_valid]
    cand_set = np.zeros((rmax,), dtype=np.bool_)
    cand_set[cand_raw] = True

    tri_vertex_valid = (face_vertex_ids >= 0) & (face_vertex_ids < rmax)
    tri_in_candidates = np.zeros_like(face_vertex_ids, dtype=np.bool_)
    tri_in_candidates[tri_vertex_valid] = cand_set[face_vertex_ids[tri_vertex_valid]]
    tri_positive_weight = barycentric_weights > 1e-8
    pixel_supported = np.any(tri_in_candidates & tri_positive_weight, axis=-1)
    pix_valid = mask_bool & (face_index >= 0)
    return (pix_valid & pixel_supported).astype(np.uint8, copy=False)


def build_dense_a1_heatmap(
    *,
    mask_np: np.ndarray,
    face_index: np.ndarray,
    face_vertex_ids: np.ndarray,
    barycentric_weights: np.ndarray,
    candidate_raw_vid: np.ndarray,
    a1_logits: np.ndarray,
) -> np.ndarray:
    mask_bool = np.asarray(mask_np) > 0
    face_index = np.asarray(face_index, dtype=np.int64)
    face_vertex_ids = np.asarray(face_vertex_ids, dtype=np.int64)
    barycentric_weights = np.asarray(barycentric_weights, dtype=np.float32)
    cand_raw = np.asarray(candidate_raw_vid, dtype=np.int64)
    a1_w = finite_scores_with_fill(a1_logits)
    a1_w = _smooth_vertex_scores(
        values=a1_w,
        face_index=face_index,
        face_vertex_ids=face_vertex_ids,
        candidate_raw_vid=cand_raw,
    )

    if face_vertex_ids.ndim != 3 or face_vertex_ids.shape[-1] != 3:
        raise ValueError(f"face_vertex_ids must be [H,W,3], got {face_vertex_ids.shape}")
    if barycentric_weights.shape != face_vertex_ids.shape:
        raise ValueError(
            f"barycentric_weights shape must match face_vertex_ids: {barycentric_weights.shape} vs {face_vertex_ids.shape}"
        )
    if cand_raw.shape[0] != a1_w.shape[0]:
        raise ValueError(f"candidate_raw_vid and a1_logits must align: {cand_raw.shape[0]} vs {a1_w.shape[0]}")

    rmax = int(max(int(face_vertex_ids.max(initial=-1)), int(cand_raw.max(initial=-1)))) + 1
    rmax = max(rmax, 1)
    valid_cand = (cand_raw >= 0) & (cand_raw < rmax)
    score_raw = np.zeros((rmax,), dtype=np.float32)
    score_raw[cand_raw[valid_cand]] = a1_w[valid_cand]

    tri_valid = (face_vertex_ids >= 0) & (face_vertex_ids < rmax)
    tri_score_a1 = np.zeros_like(barycentric_weights, dtype=np.float32)
    tri_score_a1[tri_valid] = score_raw[face_vertex_ids[tri_valid]]

    pix_valid = build_projection_valid_mask(
        mask_np=mask_np,
        face_index=face_index,
        face_vertex_ids=face_vertex_ids,
        barycentric_weights=barycentric_weights,
        candidate_raw_vid=cand_raw,
    ) > 0
    a1_heat = (tri_score_a1 * barycentric_weights).sum(axis=-1)
    a1_heat[~pix_valid] = 0.0
    return a1_heat.astype(np.float32, copy=False)


def build_dense_a2_heatmap_from_row(
    *,
    mask_np: np.ndarray,
    face_index: np.ndarray,
    face_vertex_ids: np.ndarray,
    barycentric_weights: np.ndarray,
    candidate_raw_vid: np.ndarray,
    row_logits: np.ndarray,
) -> np.ndarray:
    mask_bool = np.asarray(mask_np) > 0
    face_index = np.asarray(face_index, dtype=np.int64)
    face_vertex_ids = np.asarray(face_vertex_ids, dtype=np.int64)
    barycentric_weights = np.asarray(barycentric_weights, dtype=np.float32)
    cand_raw = np.asarray(candidate_raw_vid, dtype=np.int64)
    row_w = finite_scores_with_fill(row_logits)
    row_w = _smooth_vertex_scores(
        values=row_w,
        face_index=face_index,
        face_vertex_ids=face_vertex_ids,
        candidate_raw_vid=cand_raw,
    )

    rmax = int(max(int(face_vertex_ids.max(initial=-1)), int(cand_raw.max(initial=-1)))) + 1
    rmax = max(rmax, 1)
    valid_cand = (cand_raw >= 0) & (cand_raw < rmax)
    score_raw = np.zeros((rmax,), dtype=np.float32)
    score_raw[cand_raw[valid_cand]] = row_w[valid_cand]

    tri_valid = (face_vertex_ids >= 0) & (face_vertex_ids < rmax)
    tri_score = np.zeros_like(barycentric_weights, dtype=np.float32)
    tri_score[tri_valid] = score_raw[face_vertex_ids[tri_valid]]

    pix_valid = build_projection_valid_mask(
        mask_np=mask_np,
        face_index=face_index,
        face_vertex_ids=face_vertex_ids,
        barycentric_weights=barycentric_weights,
        candidate_raw_vid=cand_raw,
    ) > 0
    heat = (tri_score * barycentric_weights).sum(axis=-1)
    heat[~pix_valid] = 0.0
    return heat.astype(np.float32, copy=False)


def build_dense_a2_heatmap_for_pixel(
    *,
    x: int,
    y: int,
    mask_np: np.ndarray,
    face_index: np.ndarray,
    face_vertex_ids: np.ndarray,
    barycentric_weights: np.ndarray,
    candidate_raw_vid: np.ndarray,
    reward_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    mask_bool = np.asarray(mask_np) > 0
    face_index = np.asarray(face_index, dtype=np.int64)
    face_vertex_ids = np.asarray(face_vertex_ids, dtype=np.int64)
    barycentric_weights = np.asarray(barycentric_weights, dtype=np.float32)
    cand_raw = np.asarray(candidate_raw_vid, dtype=np.int64)
    reward_matrix = np.asarray(reward_matrix, dtype=np.float32)

    h, w = mask_bool.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), dtype=np.float32), np.full((cand_raw.shape[0],), -np.inf, dtype=np.float32), False
    if not bool(mask_bool[y, x]) or int(face_index[y, x]) < 0:
        return np.zeros((h, w), dtype=np.float32), np.full((cand_raw.shape[0],), -np.inf, dtype=np.float32), False

    rmax = int(max(int(face_vertex_ids.max(initial=-1)), int(cand_raw.max(initial=-1)))) + 1
    rmax = max(rmax, 1)
    r2local = np.full((rmax,), -1, dtype=np.int64)
    valid_cand = (cand_raw >= 0) & (cand_raw < rmax)
    r2local[cand_raw[valid_cand]] = np.arange(cand_raw.shape[0], dtype=np.int64)[valid_cand]

    raw_tri = face_vertex_ids[y, x].astype(np.int64, copy=False)
    bary = barycentric_weights[y, x].astype(np.float32, copy=False)
    mixed_row = np.zeros((cand_raw.shape[0],), dtype=np.float32)
    total_w = 0.0
    for raw_vid, wv in zip(raw_tri.tolist(), bary.tolist()):
        if raw_vid < 0 or raw_vid >= r2local.shape[0] or wv <= 0.0:
            continue
        local_idx = int(r2local[raw_vid])
        if local_idx < 0 or local_idx >= reward_matrix.shape[0]:
            continue
        mixed_row += float(wv) * reward_matrix[local_idx]
        total_w += float(wv)

    if total_w <= 1e-8:
        return np.zeros((h, w), dtype=np.float32), np.full((cand_raw.shape[0],), -np.inf, dtype=np.float32), False

    mixed_row = mixed_row / total_w
    heat = build_dense_a2_heatmap_from_row(
        mask_np=mask_np,
        face_index=face_index,
        face_vertex_ids=face_vertex_ids,
        barycentric_weights=barycentric_weights,
        candidate_raw_vid=candidate_raw_vid,
        row_logits=mixed_row,
    )
    return heat, mixed_row.astype(np.float32, copy=False), True


def best_candidate_xy(candidate_xy: np.ndarray, logits: np.ndarray) -> Optional[np.ndarray]:
    xy = np.asarray(candidate_xy, dtype=np.float32)
    score = np.asarray(logits, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return None
    masked = np.where(np.isfinite(score), score, -np.inf)
    if not np.isfinite(masked).any():
        return None
    idx = int(np.nanargmax(masked))
    return xy[idx].astype(np.float32, copy=False)
