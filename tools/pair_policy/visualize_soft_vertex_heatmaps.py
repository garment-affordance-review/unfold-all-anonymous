#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from unfold.algorithms.supervision.targets import build_a1_from_reward_matrix, build_reward_matrix
from unfold.algorithms.supervision.teacher_pointcept import TeacherRewardInfer

matplotlib.use("Agg")


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    finite = np.isfinite(x)
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


def _save_overlay(
    rgb: np.ndarray,
    heat: np.ndarray,
    out_path: Path,
    title: str,
    best_x1_xy: np.ndarray | None = None,
    best_x2_xy: np.ndarray | None = None,
) -> None:
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    ax.imshow(rgb)
    heat_alpha = np.clip((np.asarray(heat, dtype=np.float32) - 0.08) / 0.92, 0.0, 1.0) * 0.80
    ax.imshow(heat, cmap="jet", alpha=heat_alpha, vmin=0.0, vmax=1.0)
    if best_x1_xy is not None and best_x1_xy.size == 2:
        ax.scatter(
            [best_x1_xy[0]],
            [best_x1_xy[1]],
            s=240,
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=3.6,
        )
    if best_x2_xy is not None and best_x2_xy.size == 2:
        ax.scatter(
            [best_x2_xy[0]],
            [best_x2_xy[1]],
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


def _save_compare(
    rgb: np.ndarray,
    heat_hard: np.ndarray,
    heat_soft: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, heat, tt in [
        (axes[0], heat_hard, "Hard Map"),
        (axes[1], heat_soft, "Soft Vertex"),
    ]:
        ax.imshow(rgb)
        heat_alpha = np.clip((np.asarray(heat, dtype=np.float32) - 0.08) / 0.92, 0.0, 1.0) * 0.80
        ax.imshow(heat, cmap="jet", alpha=heat_alpha, vmin=0.0, vmax=1.0)
        ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_result(rgb: np.ndarray, out_path: Path, best_x1_xy: np.ndarray | None, best_x2_xy: np.ndarray | None) -> None:
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    ax.imshow(rgb)
    if best_x1_xy is not None and best_x1_xy.size == 2:
        ax.scatter(
            [best_x1_xy[0]],
            [best_x1_xy[1]],
            s=240,
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=3.6,
        )
    if best_x2_xy is not None and best_x2_xy.size == 2:
        ax.scatter(
            [best_x2_xy[0]],
            [best_x2_xy[1]],
            s=240,
            marker="o",
            facecolors="none",
            edgecolors="yellow",
            linewidths=3.6,
        )
    if best_x1_xy is not None and best_x2_xy is not None:
        ax.plot([best_x1_xy[0], best_x2_xy[0]], [best_x1_xy[1], best_x2_xy[1]], color="white", linewidth=2.2, alpha=0.95)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _sample_unique_visible_vertices(mask: np.ndarray, mask_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = (mask > 0) & (mask_index >= 0)
    yx = np.argwhere(valid)
    if yx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    vids = mask_index[valid].astype(np.int64)
    uniq_vid, first_idx = np.unique(vids, return_index=True)
    rep_yx = yx[first_idx]
    rep_xy = rep_yx[:, [1, 0]].astype(np.float32)
    return uniq_vid.astype(np.int64), rep_xy


def _build_pairs(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    ii, jj = np.where(~np.eye(n, dtype=bool))
    return np.stack([ii, jj], axis=1).astype(np.int64)


def _compute_tv(heat: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    dx = np.abs(heat[:, 1:] - heat[:, :-1])
    dy = np.abs(heat[1:, :] - heat[:-1, :])
    mx = m[:, 1:] & m[:, :-1]
    my = m[1:, :] & m[:-1, :]
    vx = float(dx[mx].mean()) if np.any(mx) else 0.0
    vy = float(dy[my].mean()) if np.any(my) else 0.0
    return 0.5 * (vx + vy)


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    interior = np.zeros_like(m, dtype=bool)
    interior[1:-1, 1:-1] = (
        m[1:-1, 1:-1]
        & m[:-2, 1:-1]
        & m[2:, 1:-1]
        & m[1:-1, :-2]
        & m[1:-1, 2:]
    )
    return m & (~interior)


def _boundary_variance(heat: np.ndarray, mask: np.ndarray) -> float:
    b = _boundary_mask(mask)
    if not np.any(b):
        return 0.0
    return float(np.var(heat[b]))


def _best_point_percentile(heat: np.ndarray, mask: np.ndarray, xy: np.ndarray | None) -> float:
    if xy is None or xy.size != 2:
        return 0.0
    m = mask.astype(bool)
    vals = heat[m]
    if vals.size == 0:
        return 0.0
    px = int(round(float(xy[0])))
    py = int(round(float(xy[1])))
    if py < 0 or py >= heat.shape[0] or px < 0 or px >= heat.shape[1]:
        return 0.0
    v = float(heat[py, px])
    return float(np.mean(vals <= v))


def _load_rows(data_root: Path) -> list[dict]:
    manifest = data_root / "manifest.jsonl"
    if manifest.exists():
        rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(rows) > 0:
            return rows
    rows = []
    for sd in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        meta_path = sd / "meta.json"
        if not meta_path.exists():
            continue
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append({"id": sd.name, "paths": m.get("paths", {}), "meta": m})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize A1/A2 with soft-vertex projection (k=3).")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--teacher-cfg", type=str, required=True)
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--pointcept-data-root", type=str, required=True)
    parser.add_argument("--usd-rel", type=str, required=True)
    parser.add_argument("--mapping-file", type=str, default=None, help="Optional render->teacher map .npy")
    parser.add_argument("--pointcept-code-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pair-chunk-size", type=int, default=65536)
    parser.add_argument("--a1-reduce", type=str, default="max", choices=["max", "logsumexp"])
    parser.add_argument("--max-samples", type=int, default=13)
    parser.add_argument("--out-dir", type=str, default="logs/pair_policy_soft_vertex_vis")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(data_root)
    rows = rows[: args.max_samples]
    if len(rows) == 0:
        raise RuntimeError(f"No samples found in {data_root}")

    teacher = TeacherRewardInfer(
        teacher_cfg=args.teacher_cfg,
        teacher_ckpt=args.teacher_ckpt,
        pointcept_code_root=args.pointcept_code_root,
        device=args.device,
    )

    p_manifest = json.loads((Path(args.pointcept_data_root) / "manifest.json").read_text(encoding="utf-8"))["assets"]
    p_asset = None
    for a in p_manifest:
        if str(Path(a["usd"]).as_posix()) == str(Path(args.usd_rel).as_posix()):
            p_asset = a
            break
    if p_asset is None:
        raise RuntimeError(f"usd_rel not found in Pointcept manifest: {args.usd_rel}")
    asset_dir = Path(args.pointcept_data_root) / "assets" / p_asset["asset_id"]
    teacher_coord = np.load(asset_dir / "coord.npy").astype(np.float32)
    normal_path = asset_dir / "normal.npy"
    teacher_normal = np.load(normal_path).astype(np.float32) if normal_path.exists() else None

    if args.mapping_file:
        render_to_teacher = np.load(Path(args.mapping_file)).astype(np.int64)
    else:
        map_rel = p_asset.get("raw2coord", "raw2coord.npy")
        map_path = asset_dir / map_rel
        if not map_path.exists():
            map_path = asset_dir / "raw_to_down.npy"
        render_to_teacher = np.load(map_path).astype(np.int64)
        # In some captures, render vertex space is a compact prefix subset of raw USD vertices.
        # If so, allow prefix trim to render-space length.
        first_paths = rows[0]["paths"]
        render_nv = int(np.load(data_root / first_paths["vertices"], mmap_mode="r").shape[0])
        if render_to_teacher.shape[0] > render_nv:
            render_to_teacher = render_to_teacher[:render_nv]

    integrity = []
    for row in rows[:5]:
        sid = str(row["id"])
        paths = row["paths"]
        face_index = np.load(data_root / paths["face_index"]).astype(np.int64)
        face_vid = np.load(data_root / paths["face_vertex_ids"]).astype(np.int64)
        bary_w = np.load(data_root / paths["barycentric_weights"]).astype(np.float32)
        vtx = np.load(data_root / paths["vertices"], mmap_mode="r")
        valid = face_index >= 0
        bw_sum = bary_w.sum(axis=-1)
        integrity.append(
            {
                "sample_id": sid,
                "face_index_shape": list(face_index.shape),
                "face_vertex_ids_shape": list(face_vid.shape),
                "barycentric_weights_shape": list(bary_w.shape),
                "vertex_count": int(vtx.shape[0]),
                "face_min": int(face_index.min()),
                "face_max": int(face_index.max()),
                "valid_pixel_count": int(valid.sum()),
                "bary_weight_sum_max_abs_err": float(np.max(np.abs(bw_sum[valid] - 1.0))) if np.any(valid) else 0.0,
            }
        )

    per_sample = []
    hard_t = 0.0
    soft_t = 0.0

    for row in rows:
        sid = str(row["id"])
        paths = row["paths"]

        rgb = np.array(Image.open(data_root / paths["rgb"]).convert("RGB"))
        mask = np.array(Image.open(data_root / paths["mask"]))
        mask_bool = mask > 0
        masked_rgb = rgb.copy()
        masked_rgb[~mask_bool] = 0
        face_index = np.load(data_root / paths["face_index"]).astype(np.int64)
        face_vertex_ids = np.load(data_root / paths["face_vertex_ids"]).astype(np.int64)
        bary_w = np.load(data_root / paths["barycentric_weights"]).astype(np.float32)
        if face_vertex_ids.ndim != 3 or face_vertex_ids.shape[-1] != 3:
            raise ValueError(f"face_vertex_ids must be (H,W,3), got {face_vertex_ids.shape} sid={sid}")
        if bary_w.shape != face_vertex_ids.shape:
            raise ValueError(f"barycentric_weights shape mismatch sid={sid}: {bary_w.shape} vs {face_vertex_ids.shape}")
        if face_index.shape != face_vertex_ids.shape[:2]:
            raise ValueError(f"face_index shape mismatch sid={sid}: {face_index.shape} vs {face_vertex_ids.shape[:2]}")

        hard_slot = np.argmax(bary_w, axis=-1)
        render_vertex_map = np.take_along_axis(face_vertex_ids, hard_slot[..., None], axis=-1)[..., 0]
        render_vertex_map[face_index < 0] = -1
        render_vid, candidate_xy = _sample_unique_visible_vertices(mask=mask, mask_index=render_vertex_map)
        if render_to_teacher.shape[0] < int(render_vid.max(initial=-1)) + 1:
            raise ValueError(
                f"render_to_teacher map too short sid={sid}: map_len={render_to_teacher.shape[0]} max_render_vid={int(render_vid.max(initial=-1))}"
            )
        mapped = render_to_teacher[render_vid]
        valid = (mapped >= 0) & (mapped < teacher_coord.shape[0])
        render_vid = render_vid[valid]
        candidate_xy = candidate_xy[valid]
        candidate_tvid = mapped[valid].astype(np.int64)
        uniq_t, first = np.unique(candidate_tvid, return_index=True)
        candidate_tvid = uniq_t
        candidate_xy = candidate_xy[first]

        n = int(candidate_tvid.shape[0])
        local_pairs = _build_pairs(n)
        pair_vid = np.stack([candidate_tvid[local_pairs[:, 0]], candidate_tvid[local_pairs[:, 1]]], axis=1).astype(np.int64)
        pair_rewards = teacher.infer_pairs(
            coord=teacher_coord,
            pairs=pair_vid,
            normal=teacher_normal,
            max_pairs_per_forward=args.pair_chunk_size,
        )
        rmat = build_reward_matrix(
            num_candidates=n,
            local_pairs=local_pairs,
            rewards=pair_rewards,
            fill_value=-np.inf,
            diagonal_value=-np.inf,
        )
        a1 = build_a1_from_reward_matrix(rmat, reduce=args.a1_reduce)
        best_x1_idx = int(np.nanargmax(np.where(np.isfinite(a1), a1, -np.inf))) if n > 0 else -1
        best_x2_idx = -1
        row_a2 = np.full((n,), -np.inf, dtype=np.float32)
        if 0 <= best_x1_idx < n:
            row_a2 = rmat[best_x1_idx].copy()
            row_a2[best_x1_idx] = -np.inf
            if np.isfinite(row_a2).any():
                best_x2_idx = int(np.nanargmax(np.where(np.isfinite(row_a2), row_a2, -np.inf)))

        best_x1_xy = candidate_xy[best_x1_idx] if 0 <= best_x1_idx < n else None
        best_x2_xy = candidate_xy[best_x2_idx] if 0 <= best_x2_idx < n else None

        score_a1 = _normalize_scores(a1)
        score_a2 = _normalize_scores(row_a2)
        score_table_a1 = np.zeros((teacher_coord.shape[0],), dtype=np.float32)
        score_table_a2 = np.zeros((teacher_coord.shape[0],), dtype=np.float32)
        score_table_a1[candidate_tvid] = score_a1
        score_table_a2[candidate_tvid] = score_a2

        # Hard dense map timing
        t0 = time.perf_counter()
        hard_a1 = np.zeros(render_vertex_map.shape, dtype=np.float32)
        hard_a2 = np.zeros(render_vertex_map.shape, dtype=np.float32)
        valid_render = (render_vertex_map >= 0) & (render_vertex_map < render_to_teacher.shape[0]) & mask_bool
        teacher_pix = np.full(render_vertex_map.shape, -1, dtype=np.int64)
        teacher_pix[valid_render] = render_to_teacher[render_vertex_map[valid_render]]
        valid_teacher = (teacher_pix >= 0) & (teacher_pix < teacher_coord.shape[0]) & mask_bool
        hard_a1[valid_teacher] = score_table_a1[teacher_pix[valid_teacher]]
        hard_a2[valid_teacher] = score_table_a2[teacher_pix[valid_teacher]]
        hard_t += time.perf_counter() - t0

        # Soft map timing (k=3 full-res)
        t1 = time.perf_counter()
        valid_soft_render = (face_vertex_ids >= 0) & (face_vertex_ids < render_to_teacher.shape[0])
        soft_teacher_ids = np.full_like(face_vertex_ids, -1, dtype=np.int64)
        soft_teacher_ids[valid_soft_render] = render_to_teacher[face_vertex_ids[valid_soft_render]]
        valid_soft = (soft_teacher_ids >= 0) & (soft_teacher_ids < teacher_coord.shape[0])
        soft_a1 = (np.where(valid_soft, score_table_a1[soft_teacher_ids], 0.0) * bary_w).sum(axis=-1)
        soft_a2 = (np.where(valid_soft, score_table_a2[soft_teacher_ids], 0.0) * bary_w).sum(axis=-1)
        soft_a1[~mask_bool] = 0.0
        soft_a2[~mask_bool] = 0.0
        soft_t += time.perf_counter() - t1

        sample_out = out_dir / sid
        sample_out.mkdir(parents=True, exist_ok=True)
        _save_overlay(masked_rgb, soft_a1, sample_out / "a1.png", "A1 Soft-Vertex Heatmap", best_x1_xy=best_x1_xy)
        _save_overlay(masked_rgb, soft_a2, sample_out / "a2.png", "A2 Soft-Vertex Heatmap", best_x2_xy=best_x2_xy)
        _save_result(masked_rgb, sample_out / "result.png", best_x1_xy=best_x1_xy, best_x2_xy=best_x2_xy)
        _save_compare(masked_rgb, hard_a1, soft_a1, sample_out / "a1_compare.png", "A1 Hard vs Soft")
        _save_compare(masked_rgb, hard_a2, soft_a2, sample_out / "a2_compare.png", "A2 Hard vs Soft")

        per_sample.append(
            {
                "sample_id": sid,
                "num_candidates": n,
                "tv_hard_a1": _compute_tv(hard_a1, mask_bool),
                "tv_soft_a1": _compute_tv(soft_a1, mask_bool),
                "tv_hard_a2": _compute_tv(hard_a2, mask_bool),
                "tv_soft_a2": _compute_tv(soft_a2, mask_bool),
                "boundary_var_hard_a1": _boundary_variance(hard_a1, mask_bool),
                "boundary_var_soft_a1": _boundary_variance(soft_a1, mask_bool),
                "boundary_var_hard_a2": _boundary_variance(hard_a2, mask_bool),
                "boundary_var_soft_a2": _boundary_variance(soft_a2, mask_bool),
                "best_x1_percentile_hard": _best_point_percentile(hard_a1, mask_bool, best_x1_xy),
                "best_x1_percentile_soft": _best_point_percentile(soft_a1, mask_bool, best_x1_xy),
                "best_x2_percentile_hard": _best_point_percentile(hard_a2, mask_bool, best_x2_xy),
                "best_x2_percentile_soft": _best_point_percentile(soft_a2, mask_bool, best_x2_xy),
                "best_x1_xy": best_x1_xy.tolist() if best_x1_xy is not None else None,
                "best_x2_xy": best_x2_xy.tolist() if best_x2_xy is not None else None,
            }
        )
        print(f"[INFO] sample={sid} candidates={n} saved={sample_out}")

    report = {
        "data_root": str(data_root),
        "num_samples": len(rows),
        "integrity_first5": integrity,
        "projection_timing": {
            "hard_total_sec": float(hard_t),
            "soft_total_sec": float(soft_t),
            "hard_per_sample_ms": float(1000.0 * hard_t / max(len(rows), 1)),
            "soft_per_sample_ms": float(1000.0 * soft_t / max(len(rows), 1)),
            "soft_over_hard_ratio": float(soft_t / max(hard_t, 1e-12)),
        },
        "per_sample_metrics": per_sample,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] wrote report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
