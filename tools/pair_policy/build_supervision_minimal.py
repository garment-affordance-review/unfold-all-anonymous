#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

from unfold.algorithms.supervision.teacher_pointcept import TeacherRewardInfer
from unfold.algorithms.supervision.targets import (
    build_a1_from_reward_matrix,
    build_a2_conditional_topk,
    build_reward_matrix,
    symmetrize_reward_matrix,
)
from unfold.algorithms.supervision.visualize import save_supervision_visuals


def load_manifest(manifest_path: Path) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_pointcept_manifest(pointcept_manifest: Path) -> list[dict]:
    obj = json.loads(pointcept_manifest.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        assets = obj.get("assets", [])
    elif isinstance(obj, list):
        assets = obj
    else:
        raise ValueError(f"Unsupported Pointcept manifest format: {pointcept_manifest}")
    if not isinstance(assets, list):
        raise ValueError(f"Pointcept manifest assets is not a list: {pointcept_manifest}")
    return assets


def _normalize_rel_path(p: str) -> str:
    return str(Path(p)).replace("\\", "/").lstrip("./")


def _resolve_usd_for_sample(
    row: dict,
    data_root: Path,
    usd_rel_override: str | None,
    sample_usd_map: dict[str, str] | None,
) -> str:
    if usd_rel_override:
        return _normalize_rel_path(usd_rel_override)

    sid = str(row.get("id"))
    if sample_usd_map and sid in sample_usd_map:
        return _normalize_rel_path(sample_usd_map[sid])

    for key in ("usd", "usd_rel", "asset_usd"):
        if key in row and row[key]:
            return _normalize_rel_path(str(row[key]))

    meta_rel = row.get("paths", {}).get("meta")
    if meta_rel:
        meta_path = data_root / meta_rel
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for key in ("usd", "usd_rel", "asset_usd", "asset_path", "usd_path"):
                v = meta.get(key)
                if isinstance(v, str) and v:
                    return _normalize_rel_path(v)

    raise ValueError(
        f"sample_id={sid} missing USD source. "
        f"Provide --usd-rel or --sample-usd-map, or add usd/usd_path into sample meta."
    )


def _resolve_pointcept_asset_by_usd(assets: list[dict], usd_rel: str) -> dict:
    usd_norm = _normalize_rel_path(usd_rel)
    exact = [a for a in assets if _normalize_rel_path(str(a.get("usd", ""))) == usd_norm]
    if exact:
        return exact[0]

    suffix = [a for a in assets if _normalize_rel_path(str(a.get("usd", ""))).endswith(usd_norm)]
    if len(suffix) == 1:
        return suffix[0]
    if len(suffix) > 1:
        raise ValueError(f"USD suffix matched multiple Pointcept assets: usd={usd_rel} matches={len(suffix)}")
    raise ValueError(f"USD not found in Pointcept manifest: usd={usd_rel}")


def _load_vertex_index_map(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f"vertex index map must be 1-D, got {arr.shape} from {path}")
    return arr.astype(np.int64, copy=False)


def _resolve_index_map_path(
    *,
    vertex_index_map_path: Path | None,
    pointcept_asset: dict,
    asset_dir: Path,
    mapping_file_key: str,
) -> Path:
    if vertex_index_map_path is not None:
        return vertex_index_map_path
    candidates: list[str] = []
    rel = pointcept_asset.get(mapping_file_key)
    if rel is not None:
        candidates.append(str(rel))
    candidates.extend(["raw2coord.npy", "raw_to_down.npy"])
    for rel_path in candidates:
        p = asset_dir / rel_path
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Mapping file missing for asset_id={pointcept_asset.get('asset_id')}: key={mapping_file_key} candidates={candidates}"
    )


def iter_selected_samples(rows: list[dict], sample_id: str | None, all_samples: bool, max_samples: int | None) -> list[dict]:
    if sample_id is not None:
        selected = [r for r in rows if str(r.get("id")) == str(sample_id)]
        if not selected:
            raise ValueError(f"sample_id={sample_id} not found in manifest")
        return selected[:1]

    if all_samples:
        selected = rows
    else:
        selected = rows[:1]
    if max_samples is not None:
        selected = selected[: max_samples]
    return selected


def _uniform_grid_select(
    xy: np.ndarray,
    target_n: int,
) -> np.ndarray:
    """
    Deterministic uniform selection in image space using 2D grid cells.
    """
    n = xy.shape[0]
    if target_n <= 0 or n <= target_n:
        return np.arange(n, dtype=np.int64)

    g = int(np.ceil(np.sqrt(target_n)))
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    xspan = float(max(xmax - xmin, 1.0))
    yspan = float(max(ymax - ymin, 1.0))

    gx = np.clip(((xy[:, 0] - xmin) / xspan * g).astype(np.int64), 0, g - 1)
    gy = np.clip(((xy[:, 1] - ymin) / yspan * g).astype(np.int64), 0, g - 1)
    cell = gy * g + gx

    selected: list[int] = []
    for c in range(g * g):
        ids = np.nonzero(cell == c)[0]
        if ids.size == 0:
            continue
        cx = xmin + (float(c % g) + 0.5) * xspan / g
        cy = ymin + (float(c // g) + 0.5) * yspan / g
        pts = xy[ids]
        d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
        selected.append(int(ids[int(np.argmin(d2))]))
        if len(selected) >= target_n:
            break

    if len(selected) < target_n:
        picked = np.zeros((n,), dtype=bool)
        picked[np.array(selected, dtype=np.int64)] = True
        rest = np.nonzero(~picked)[0]
        rest_sorted = rest[np.lexsort((xy[rest, 0], xy[rest, 1]))]
        selected.extend(rest_sorted[: target_n - len(selected)].tolist())

    return np.array(selected[:target_n], dtype=np.int64)


def sample_candidate_vertices(
    mask: np.ndarray,
    render_vertex_map: np.ndarray,
    num_candidates: int,
    sampling_mode: str = "all_unique",
) -> tuple[np.ndarray, np.ndarray]:
    valid = (mask > 0) & (render_vertex_map >= 0)
    yx = np.argwhere(valid)
    if yx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

    vids = render_vertex_map[valid].astype(np.int64)
    uniq_vids, first_idx = np.unique(vids, return_index=True)
    rep_yx = yx[first_idx]  # (U,2): y,x

    if num_candidates > 0 and uniq_vids.shape[0] > num_candidates:
        if sampling_mode == "grid_uniform":
            rep_xy_full = rep_yx[:, [1, 0]].astype(np.float32)
            keep = _uniform_grid_select(rep_xy_full, target_n=num_candidates)
        elif sampling_mode == "all_unique":
            # No grid partition. Deterministically keep first num_candidates in unique-id order.
            keep = np.arange(num_candidates, dtype=np.int64)
        else:
            raise ValueError(f"Unknown sampling_mode={sampling_mode}")
        uniq_vids = uniq_vids[keep]
        rep_yx = rep_yx[keep]

    # convert to x,y for plotting and output consistency
    rep_xy = rep_yx[:, [1, 0]].astype(np.float32)
    return uniq_vids.astype(np.int64), rep_xy


def build_ordered_local_pairs(num_candidates: int) -> np.ndarray:
    if num_candidates <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    ii, jj = np.where(~np.eye(num_candidates, dtype=bool))
    return np.stack([ii, jj], axis=1).astype(np.int64)


def maybe_subsample_pairs(local_pairs: np.ndarray, pair_budget: int, rng: np.random.Generator) -> np.ndarray:
    if pair_budget <= 0 or local_pairs.shape[0] <= pair_budget:
        return local_pairs
    keep = rng.choice(local_pairs.shape[0], size=pair_budget, replace=False)
    return local_pairs[keep]


def run_one_sample(
    row: dict,
    data_root: Path,
    pointcept_data_root: Path,
    pointcept_assets: list[dict],
    teacher_input_source: str,
    usd_rel_override: str | None,
    sample_usd_map: dict[str, str] | None,
    vertex_index_map_path: Path | None,
    mapping_file_key: str,
    teacher: TeacherRewardInfer,
    out_root: Path,
    num_candidates: int,
    sampling_mode: str,
    pair_budget: int,
    a1_reduce: str,
    reward_symmetrize: str,
    tau: float,
    topk_cond: int,
    pair_chunk_size: int,
    heat_sigma: float,
) -> dict:
    sid = str(row["id"])
    sample_dir = data_root / sid
    paths = row["paths"]

    rgb_path = data_root / paths["rgb"]
    mask_path = data_root / paths["mask"]
    face_index_path = data_root / paths["face_index"]
    face_vertex_ids_path = data_root / paths["face_vertex_ids"]
    barycentric_weights_path = data_root / paths["barycentric_weights"]
    vertices_path = data_root / paths["vertices"]

    mask = np.array(Image.open(mask_path))
    face_index = np.load(face_index_path).astype(np.int64)
    face_vertex_ids = np.load(face_vertex_ids_path).astype(np.int64)
    barycentric_weights = np.load(barycentric_weights_path).astype(np.float32)
    if face_vertex_ids.ndim != 3 or face_vertex_ids.shape[-1] != 3:
        raise ValueError(f"face_vertex_ids must be (H,W,3), got {face_vertex_ids.shape} from {face_vertex_ids_path}")
    if barycentric_weights.shape != face_vertex_ids.shape:
        raise ValueError(
            f"barycentric_weights shape mismatch: {barycentric_weights.shape} vs {face_vertex_ids.shape} "
            f"paths=({barycentric_weights_path}, {face_vertex_ids_path})"
        )
    if face_index.shape != face_vertex_ids.shape[:2]:
        raise ValueError(f"face_index shape mismatch: {face_index.shape} vs {face_vertex_ids.shape[:2]}")
    hard_slot = np.argmax(barycentric_weights, axis=-1)
    render_vertex_map = np.take_along_axis(face_vertex_ids, hard_slot[..., None], axis=-1)[..., 0]
    render_vertex_map[face_index < 0] = -1
    vertices_render = np.load(vertices_path).astype(np.float32)

    usd_rel = None
    pointcept_asset = None
    teacher_coord = None
    teacher_normal = None
    index_map = None

    if teacher_input_source != "pointcept_asset":
        raise ValueError("Only --teacher-input-source pointcept_asset is supported.")

    usd_rel = _resolve_usd_for_sample(
        row=row,
        data_root=data_root,
        usd_rel_override=usd_rel_override,
        sample_usd_map=sample_usd_map,
    )
    pointcept_asset = _resolve_pointcept_asset_by_usd(pointcept_assets, usd_rel=usd_rel)
    asset_id = pointcept_asset["asset_id"]
    asset_dir = pointcept_data_root / "assets" / asset_id
    coord_path = asset_dir / "coord.npy"
    normal_path = asset_dir / "normal.npy"
    map_path = _resolve_index_map_path(
        vertex_index_map_path=vertex_index_map_path,
        pointcept_asset=pointcept_asset,
        asset_dir=asset_dir,
        mapping_file_key=mapping_file_key,
    )
    if not coord_path.exists():
        raise FileNotFoundError(f"Pointcept coord.npy not found: {coord_path}")
    teacher_coord = np.load(coord_path).astype(np.float32)
    teacher_normal = np.load(normal_path).astype(np.float32) if normal_path.exists() else None
    index_map = _load_vertex_index_map(map_path)
    if index_map.shape[0] < vertices_render.shape[0]:
        raise ValueError(
            f"index map shorter than render vertices: map={index_map.shape[0]} render={vertices_render.shape[0]} map={map_path}"
        )

    # Build visible set from full rasterized triangles (not hard single-vertex map):
    # raw visible vertices -> downsample (teacher) visible vertices.
    mask_bool = (mask > 0)
    pix_valid = (face_index >= 0) & mask_bool
    raw_tri = face_vertex_ids[pix_valid]
    raw_valid = (raw_tri >= 0) & (raw_tri < index_map.shape[0])
    teacher_tri = np.full(raw_tri.shape, -1, dtype=np.int64)
    teacher_tri[raw_valid] = index_map[raw_tri[raw_valid]]
    teacher_valid = (teacher_tri >= 0) & (teacher_tri < int(teacher_coord.shape[0]))
    visible_teacher_vid = np.unique(teacher_tri[teacher_valid]).astype(np.int64)
    if visible_teacher_vid.size == 0:
        raise ValueError(f"No visible teacher vertices after raw->down mapping: sample_id={sid}")

    # Representative 2D anchor per teacher vertex for visualization/scatter.
    hard_teacher_map = np.full(render_vertex_map.shape, -1, dtype=np.int64)
    valid_render = (render_vertex_map >= 0) & (render_vertex_map < index_map.shape[0]) & mask_bool
    hard_teacher_map[valid_render] = index_map[render_vertex_map[valid_render]]
    yx = np.argwhere((hard_teacher_map >= 0) & (hard_teacher_map < int(teacher_coord.shape[0])))
    vids = hard_teacher_map[(hard_teacher_map >= 0) & (hard_teacher_map < int(teacher_coord.shape[0]))].astype(np.int64)
    uniq_vids, first_idx = np.unique(vids, return_index=True)
    rep_yx = yx[first_idx]
    rep_xy = rep_yx[:, [1, 0]].astype(np.float32)

    candidate_vid = uniq_vids
    candidate_xy = rep_xy
    n_candidate_before_map = int(candidate_vid.shape[0])
    n_candidate_after_map_before_dedup = int(candidate_vid.shape[0])
    n_candidate_after_dedup = int(candidate_vid.shape[0])
    candidate_vid_render = np.full(candidate_vid.shape, -1, dtype=np.int64)

    # Optional cap for speed; default should keep all visible vertices.
    if num_candidates > 0 and candidate_vid.shape[0] > num_candidates:
        if sampling_mode == "grid_uniform":
            keep = _uniform_grid_select(candidate_xy, target_n=num_candidates)
        elif sampling_mode == "all_unique":
            keep = np.arange(num_candidates, dtype=np.int64)
        else:
            raise ValueError(f"Unknown sampling_mode={sampling_mode}")
        candidate_vid = candidate_vid[keep]
        candidate_xy = candidate_xy[keep]
        candidate_vid_render = candidate_vid_render[keep]

    n = int(candidate_vid.shape[0])
    local_pairs_full = build_ordered_local_pairs(n)
    # deterministic pair subsampling for reproducible supervision indices
    local_pairs = maybe_subsample_pairs(
        local_pairs_full,
        pair_budget=pair_budget,
        rng=np.random.default_rng(0),
    )

    if local_pairs.shape[0] > 0:
        pair_vertex_ids = np.stack(
            [candidate_vid[local_pairs[:, 0]], candidate_vid[local_pairs[:, 1]]],
            axis=1,
        ).astype(np.int64)
        pair_rewards = teacher.infer_pairs(
            coord=teacher_coord,
            pairs=pair_vertex_ids,
            normal=teacher_normal,
            max_pairs_per_forward=pair_chunk_size,
        )
    else:
        pair_vertex_ids = np.zeros((0, 2), dtype=np.int64)
        pair_rewards = np.zeros((0,), dtype=np.float32)

    reward_matrix = build_reward_matrix(
        num_candidates=n,
        local_pairs=local_pairs,
        rewards=pair_rewards,
        fill_value=-np.inf,
        diagonal_value=-np.inf,
    )
    reward_matrix = symmetrize_reward_matrix(
        reward_matrix=reward_matrix,
        mode=reward_symmetrize,
        diagonal_value=-np.inf,
    )
    a1_logits = build_a1_from_reward_matrix(reward_matrix=reward_matrix, reduce=a1_reduce)
    topk_x1_idx, a2_logits_topk, a2_probs_topk = build_a2_conditional_topk(
        reward_matrix=reward_matrix,
        a1_logits=a1_logits,
        topk=topk_cond,
        tau=tau,
        exclude_self=True,
    )

    sample_out = out_root / sid
    sample_out.mkdir(parents=True, exist_ok=True)
    sup_path = sample_out / "supervision.npz"
    np.savez_compressed(
        sup_path,
        sample_id=np.array([sid]),
        rgb_path=np.array([str(rgb_path)]),
        mask_path=np.array([str(mask_path)]),
        vertices_path=np.array([str(vertices_path)]),
        teacher_input_source=np.array([teacher_input_source]),
        teacher_usd_rel=np.array([usd_rel if usd_rel is not None else ""]),
        teacher_asset_id=np.array([pointcept_asset["asset_id"] if pointcept_asset else ""]),
        candidate_vertex_ids=candidate_vid.astype(np.int64),
        candidate_vertex_ids_render=candidate_vid_render.astype(np.int64),
        candidate_xy=candidate_xy.astype(np.float32),
        local_pairs=local_pairs.astype(np.int64),
        pair_vertex_ids=pair_vertex_ids.astype(np.int64),
        pair_rewards=pair_rewards.astype(np.float32),
        reward_matrix=reward_matrix.astype(np.float32),
        a1_logits=a1_logits.astype(np.float32),
        topk_x1_idx=topk_x1_idx.astype(np.int64),
        a2_logits_topk=a2_logits_topk.astype(np.float32),
        a2_probs_topk=a2_probs_topk.astype(np.float32),
    )

    vis_files = save_supervision_visuals(
        rgb_path=rgb_path,
        mask_path=mask_path,
        out_dir=sample_out,
        candidate_xy=candidate_xy,
        candidate_teacher_vid=candidate_vid,
        a1_logits=a1_logits,
        reward_matrix=reward_matrix,
        mask_index=render_vertex_map,
        render_to_teacher=index_map,
        face_vertex_ids=face_vertex_ids,
        barycentric_weights=barycentric_weights,
        face_index=face_index,
        sigma=heat_sigma,
    )

    best_x1_idx = int(np.nanargmax(np.where(np.isfinite(a1_logits), a1_logits, -np.inf))) if n > 0 else -1
    best_x2_idx = -1
    if 0 <= best_x1_idx < n:
        row = reward_matrix[best_x1_idx].copy()
        row[best_x1_idx] = -np.inf
        if np.isfinite(row).any():
            best_x2_idx = int(np.nanargmax(np.where(np.isfinite(row), row, -np.inf)))

    # Numeric checks
    has_nan = bool(np.isnan(pair_rewards).any()) if pair_rewards.size > 0 else False
    has_inf = bool(np.isinf(pair_rewards).any()) if pair_rewards.size > 0 else False
    a2_row_sums = a2_probs_topk.sum(axis=1) if a2_probs_topk.size > 0 else np.zeros((0,), dtype=np.float32)

    meta = {
        "sample_id": sid,
        "num_vertices_render": int(vertices_render.shape[0]),
        "num_vertices_teacher": int(teacher_coord.shape[0]),
        "num_candidates": int(n),
        "num_candidates_before_map": int(n_candidate_before_map),
        "num_candidates_after_map_before_dedup": int(n_candidate_after_map_before_dedup),
        "num_candidates_after_dedup": int(n_candidate_after_dedup),
        "num_pairs_full": int(local_pairs_full.shape[0]),
        "num_pairs_used": int(local_pairs.shape[0]),
        "pair_budget": int(pair_budget),
        "pair_nan_count": int(np.isnan(pair_rewards).sum()) if pair_rewards.size else 0,
        "pair_inf_count": int(np.isinf(pair_rewards).sum()) if pair_rewards.size else 0,
        "a2_softmax_max_abs_err": float(np.max(np.abs(a2_row_sums - 1.0))) if a2_row_sums.size else 0.0,
        "supervision_path": str(sup_path),
        "best_x1_idx": int(best_x1_idx),
        "best_x2_idx": int(best_x2_idx),
        "best_x1_xy": candidate_xy[best_x1_idx].tolist() if 0 <= best_x1_idx < n else None,
        "best_x2_xy": candidate_xy[best_x2_idx].tolist() if 0 <= best_x2_idx < n else None,
        "vis_files": vis_files,
        "source": {
            "rgb": str(rgb_path),
            "mask": str(mask_path),
            "face_index": str(face_index_path),
            "face_vertex_ids": str(face_vertex_ids_path),
            "barycentric_weights": str(barycentric_weights_path),
            "vertices": str(vertices_path),
            "teacher_input_source": teacher_input_source,
            "teacher_usd_rel": usd_rel,
            "teacher_asset_id": pointcept_asset["asset_id"] if pointcept_asset else None,
            "vertex_index_map": str(map_path) if map_path else None,
        },
    }
    with (sample_out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"[INFO] sample={sid} candidates={n} pairs={local_pairs.shape[0]} "
        f"teacher_nv={teacher_coord.shape[0]} source={teacher_input_source} "
        f"asset={pointcept_asset['asset_id'] if pointcept_asset else 'N/A'} "
        f"pair_nan={has_nan} pair_inf={has_inf} a2_err={meta['a2_softmax_max_abs_err']:.3e}"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build minimal pair-policy supervision from rendered cloth images.")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root containing manifest.jsonl and sample dirs")
    parser.add_argument("--manifest", type=str, default=None, help="Optional manifest path; defaults to <data-root>/manifest.jsonl")
    parser.add_argument("--sample-id", type=str, default=None, help="Single sample id to process (e.g., 00000000)")
    parser.add_argument("--all-samples", action="store_true", help="Process all samples in manifest")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap number of processed samples")

    parser.add_argument("--teacher-cfg", type=str, required=True)
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--pointcept-code-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pair-chunk-size", type=int, default=65536)
    parser.add_argument(
        "--teacher-input-source",
        type=str,
        default="pointcept_asset",
        choices=["pointcept_asset"],
        help="Only pointcept_asset is supported.",
    )
    parser.add_argument(
        "--pointcept-data-root",
        type=str,
        default="${POINTCEPT_ROOT}/data/clothes",
        help="Pointcept clothes dataset root containing manifest.json and assets/asset_****.",
    )
    parser.add_argument(
        "--pointcept-manifest",
        type=str,
        default=None,
        help="Optional Pointcept manifest path; defaults to <pointcept-data-root>/manifest.json",
    )
    parser.add_argument("--usd-rel", type=str, default=None, help="Force one USD relative path for all selected samples.")
    parser.add_argument(
        "--sample-usd-map",
        type=str,
        default=None,
        help="JSON file path: {\"sample_id\": \"usd/relative/path.usd\", ...}",
    )
    parser.add_argument(
        "--vertex-index-map",
        type=str,
        default=None,
        help="Optional explicit .npy map from render vertex id -> Pointcept downsampled vertex id. "
        "If omitted, read mapping file from Pointcept asset via --mapping-file-key.",
    )
    parser.add_argument(
        "--mapping-file-key",
        type=str,
        default="raw2coord",
        help="Manifest key for mapping file in Pointcept asset (default: raw2coord).",
    )

    parser.add_argument("--num-candidates", type=int, default=0, help="<=0 keeps all visible downsample vertices")
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="all_unique",
        choices=["all_unique", "grid_uniform"],
        help="Candidate selection mode when num_candidates > 0.",
    )
    parser.add_argument("--pair-budget", type=int, default=0, help="<=0 uses all ordered pairs")
    parser.add_argument("--candidate-seed", type=int, default=0)
    parser.add_argument("--a1-reduce", type=str, default="max", choices=["max", "logsumexp"])
    parser.add_argument("--reward-symmetrize", type=str, default="none", choices=["none", "max_swap"])
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--topk-cond", type=int, default=8)
    parser.add_argument(
        "--heat-sigma",
        type=float,
        default=-1.0,
        help="Gaussian sigma in pixels. <=0 enables auto sigma from candidate spacing.",
    )
    parser.add_argument("--out-dir", type=str, default="logs/pair_policy_minimal")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else (data_root / "manifest.jsonl")
    pointcept_data_root = Path(args.pointcept_data_root).resolve()
    pointcept_manifest = Path(args.pointcept_manifest).resolve() if args.pointcept_manifest else (pointcept_data_root / "manifest.json")
    pointcept_assets = load_pointcept_manifest(pointcept_manifest)
    sample_usd_map = None
    if args.sample_usd_map:
        sample_usd_map = json.loads(Path(args.sample_usd_map).read_text(encoding="utf-8"))
    vertex_index_map_path = Path(args.vertex_index_map).resolve() if args.vertex_index_map else None
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    selected = iter_selected_samples(
        rows=rows,
        sample_id=args.sample_id,
        all_samples=args.all_samples,
        max_samples=args.max_samples,
    )
    print(f"[INFO] Loaded manifest={manifest_path} total={len(rows)} selected={len(selected)}")

    teacher = TeacherRewardInfer(
        teacher_cfg=args.teacher_cfg,
        teacher_ckpt=args.teacher_ckpt,
        pointcept_code_root=args.pointcept_code_root,
        device=args.device,
    )
    metas: list[dict] = []
    for row in selected:
        meta = run_one_sample(
            row=row,
            data_root=data_root,
            pointcept_data_root=pointcept_data_root,
            pointcept_assets=pointcept_assets,
            teacher_input_source=args.teacher_input_source,
            usd_rel_override=args.usd_rel,
            sample_usd_map=sample_usd_map,
            vertex_index_map_path=vertex_index_map_path,
            mapping_file_key=args.mapping_file_key,
            teacher=teacher,
            out_root=out_root,
            num_candidates=args.num_candidates,
            sampling_mode=args.sampling_mode,
            pair_budget=args.pair_budget,
            a1_reduce=args.a1_reduce,
            reward_symmetrize=args.reward_symmetrize,
            tau=args.tau,
            topk_cond=args.topk_cond,
            pair_chunk_size=args.pair_chunk_size,
            heat_sigma=args.heat_sigma,
        )
        metas.append(meta)

    index = {
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "pointcept_data_root": str(pointcept_data_root),
        "pointcept_manifest": str(pointcept_manifest),
        "teacher_input_source": args.teacher_input_source,
        "usd_rel_override": args.usd_rel,
        "sample_usd_map": str(args.sample_usd_map) if args.sample_usd_map else None,
        "vertex_index_map": str(vertex_index_map_path) if vertex_index_map_path else None,
        "mapping_file_key": args.mapping_file_key,
        "sampling_mode": args.sampling_mode,
        "num_samples": len(metas),
        "samples": metas,
    }
    with (out_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote index: {out_root / 'index.json'}")


if __name__ == "__main__":
    main()
