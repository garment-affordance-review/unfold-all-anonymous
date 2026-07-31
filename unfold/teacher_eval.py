#!/usr/bin/env python3
"""Online evaluation pipeline for the Pointcept teacher."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Pointcept teacher online in UnfoldAll.")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0", help="Gym task id.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Environment YAML config.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of passes over the asset pool.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard cap on evaluated states.")
    parser.add_argument("--teacher-cfg", type=str, required=True, help="Teacher config.py path.")
    parser.add_argument("--teacher-ckpt", type=str, required=True, help="Teacher checkpoint path.")
    parser.add_argument("--teacher-python", type=str, default=None, help="Optional Python executable for teacher subprocess.")
    parser.add_argument("--pointcept-code-root", type=str, default=None, help="Optional Pointcept code root.")
    parser.add_argument(
        "--pointcept-data-root",
        type=str,
        default="${POINTCEPT_ROOT}/data/clothes",
        help="Pointcept clothes dataset root.",
    )
    parser.add_argument(
        "--pointcept-manifest",
        type=str,
        default=None,
        help="Optional Pointcept manifest path; defaults to <pointcept-data-root>/manifest.json.",
    )
    parser.add_argument("--vertex-index-map", type=str, default=None, help="Optional explicit raw->teacher map .npy path.")
    parser.add_argument(
        "--raw-to-teacher-map-key",
        type=str,
        default="raw2coord.npy",
        help="Pointcept manifest key for the raw->teacher index map.",
    )
    parser.add_argument("--pair-chunk-size", type=int, default=65536, help="Max ordered pairs per teacher forward.")
    parser.add_argument("--num-candidates", type=int, default=0, help="Optional cap on visible teacher candidates.")
    parser.add_argument(
        "--teacher-min-candidate-dist",
        type=float,
        default=0.0,
        help="Optional minimum XY distance (in teacher coord space) between candidate teacher vertices.",
    )
    parser.add_argument("--output-dir", type=str, default="logs/teacher_eval", help="Output directory.")
    parser.add_argument("--results-csv", type=str, default="results.csv", help="CSV file name under output-dir.")
    parser.add_argument("--results-h5", type=str, default="results.h5", help="HDF5 file name under output-dir.")
    parser.add_argument("--summary-name", type=str, default="summary.json", help="Summary file name under output-dir.")
    parser.add_argument("--write-threshold", type=int, default=64, help="HDF5 buffered write threshold.")
    parser.add_argument("--camera-width", type=int, default=640, help="Top-down render width.")
    parser.add_argument("--camera-height", type=int, default=480, help="Top-down render height.")
    parser.add_argument("--topdown-min-height", type=float, default=2.0, help="Minimum camera height above cloth center.")
    parser.add_argument(
        "--topdown-extent-scale",
        type=float,
        default=3.0,
        help="Scale factor from cloth XY extent to camera height.",
    )
    parser.add_argument(
        "--dynamic-camera-pose",
        dest="fixed_camera_pose",
        action="store_false",
        help="Disable fixed camera and recompute top-down camera pose per step.",
    )
    parser.add_argument(
        "--fixed-camera-height",
        type=float,
        default=2.5,
        help="Camera height above each env origin when fixed camera pose is enabled.",
    )
    parser.add_argument(
        "--projection-overlap-threshold",
        type=float,
        default=0.5,
        help="Projection consistency threshold for face rasterization.",
    )
    parser.add_argument(
        "--visibility-source",
        type=str,
        default="mesh",
        choices=["seg", "mesh"],
        help="Visibility gating source: 'seg' uses rendered cloth mask; 'mesh' uses projected mesh only.",
    )
    parser.set_defaults(fixed_camera_pose=True)
    parser.add_argument("--spp", type=int, default=1, help="Renderer samples per pixel.")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="performance",
        choices=["performance", "balanced", "quality"],
        help="Renderer mode.",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="FXAA",
        choices=["Off", "FXAA", "DLSS", "TAA", "DLAA"],
        help="Antialiasing mode.",
    )
    parser.add_argument(
        "--disable-rp-between-captures",
        action="store_true",
        help="Disable render products between captures for performance.",
    )
    parser.add_argument("--no-dome-bg", action="store_true", default=True, help="Disable dome HDR background randomization.")
    parser.add_argument("--no-extra-lights", action="store_true", default=True, help="Disable extra light randomization.")
    parser.add_argument("--no-ground-color", action="store_true", default=True, help="Disable ground color randomization.")
    parser.add_argument("--no-material-rand", action="store_true", default=True, help="Disable cloth material randomization.")
    parser.add_argument("--no-cam-intrinsics", action="store_true", default=True, help="Disable camera intrinsics randomization.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic candidate capping.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _normalize_rel_path(p: str) -> str:
    return str(Path(p)).replace("\\", "/").lstrip("./")


def _usd_match_keys(usd_path: str) -> list[str]:
    """Build normalized candidate keys for matching env USD path to Pointcept manifest USD."""
    norm = _normalize_rel_path(usd_path)
    keys: list[str] = []

    def add(v: str) -> None:
        vv = _normalize_rel_path(v)
        if vv and vv not in keys:
            keys.append(vv)

    add(norm)

    # Keep a stable relative form when the env path includes repo-specific cloth roots.
    for marker in ("data/assets/cloth/", "assets/cloth/"):
        idx = norm.find(marker)
        if idx >= 0:
            add(norm[idx + len(marker) :])

    # Pointcept manifest USD entries start from category root (Dress/Tops/Trousers).
    for marker in ("/Dress/", "/Tops/", "/Trousers/"):
        idx = norm.find(marker)
        if idx >= 0:
            add(norm[idx + 1 :])
    for marker in ("Dress/", "Tops/", "Trousers/"):
        idx = norm.find(marker)
        if idx >= 0:
            add(norm[idx:])

    return keys


def _load_pointcept_manifest(pointcept_manifest: Path) -> list[dict]:
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


def _resolve_pointcept_asset_by_usd(assets: list[dict], usd_rel: str) -> dict:
    usd_keys = _usd_match_keys(usd_rel)
    manifest_pairs = [(a, _normalize_rel_path(str(a.get("usd", "")))) for a in assets]

    for key in usd_keys:
        exact = [a for a, manifest_usd in manifest_pairs if manifest_usd == key]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise ValueError(f"USD exact key matched multiple Pointcept assets: usd={usd_rel} key={key} matches={len(exact)}")

    for key in usd_keys:
        suffix = [a for a, manifest_usd in manifest_pairs if manifest_usd.endswith(key) or key.endswith(manifest_usd)]
        if len(suffix) == 1:
            return suffix[0]
        if len(suffix) > 1:
            raise ValueError(f"USD suffix matched multiple Pointcept assets: usd={usd_rel} key={key} matches={len(suffix)}")

    raise ValueError(f"USD not found in Pointcept manifest: usd={usd_rel} keys={usd_keys}")


def _load_vertex_index_map(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f"vertex index map must be 1-D, got {arr.shape} from {path}")
    return arr.astype(np.int64, copy=False)


def _resolve_index_map_path(
    *,
    vertex_index_map_path: Optional[Path],
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


def _extract_env_idx_from_camera_path(cam_path: str) -> int:
    match = re.search(r"/env_(\d+)/view_0/cam$", cam_path)
    return int(match.group(1)) if match else 0


def _collect_observations(env) -> dict[str, Any]:
    obs = env.unwrapped._get_observations()
    return {
        "pos": obs["pos"],
        "pos_mask": obs["pos_mask"],
        "init_pos": obs["init_pos"],
        "faces": obs["faces"],
        "faces_mask": obs["faces_mask"],
        "env_origins": env.unwrapped.scene.env_origins,
    }


def _extract_faces_for_env(obs: dict[str, Any], env_idx: int) -> np.ndarray:
    faces_env = obs["faces"][env_idx]
    faces_mask = obs["faces_mask"][env_idx].detach().cpu().numpy().astype(bool).squeeze(-1)
    if faces_mask.size > 0:
        faces_env = faces_env[faces_mask]
    return faces_env.detach().cpu().numpy().astype(np.int64)


def _extract_vertices_for_env(obs: dict[str, Any], env_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices_local = obs["pos"][env_idx].detach().cpu().numpy().astype(np.float32)
    vertex_mask = obs["pos_mask"][env_idx].detach().cpu().numpy().astype(bool).squeeze(-1)
    init_pos_local = obs["init_pos"][env_idx].detach().cpu().numpy().astype(np.float32)
    env_origin = obs["env_origins"][env_idx].detach().cpu().numpy().astype(np.float32)
    return vertices_local, vertex_mask, init_pos_local, env_origin


def _sorted_unique_with_first(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, first_idx = np.unique(values, return_index=True)
    order = np.argsort(uniq)
    return uniq[order], first_idx[order]


def _candidate_cap_indices(n_items: int, num_candidates: int) -> np.ndarray:
    if num_candidates <= 0 or n_items <= num_candidates:
        return np.arange(n_items, dtype=np.int64)
    return np.linspace(0, n_items - 1, num=num_candidates, dtype=np.int64)


def _save_rgb_image(rgb_np: Optional[np.ndarray], path: Path) -> None:
    if rgb_np is None:
        return
    rgb = np.asarray(rgb_np)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def _draw_overlay(
    rgb_np: Optional[np.ndarray],
    out_path: Path,
    text_lines: list[str],
    points_xy: list[tuple[float, float]],
) -> None:
    if rgb_np is None:
        return
    rgb = np.asarray(rgb_np)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    image = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(image)

    colors = [(255, 64, 64), (64, 255, 255)]
    if len(points_xy) == 2:
        draw.line([points_xy[0], points_xy[1]], fill=(255, 255, 0), width=3)
    for idx, xy in enumerate(points_xy[:2]):
        x, y = float(xy[0]), float(xy[1])
        r = 6
        draw.ellipse((x - r, y - r, x + r, y + r), outline=colors[idx], width=3)

    text = "\n".join(text_lines)
    draw.rectangle((8, 8, 420, 22 + 18 * max(len(text_lines), 1)), fill=(0, 0, 0))
    draw.text((14, 12), text, fill=(255, 255, 255))
    image.save(out_path)


def _save_xy_projection_png(
    *,
    init_pos: np.ndarray,
    other_pos: np.ndarray,
    vertex_mask: np.ndarray,
    out_path: Path,
    title: str,
    distance: Optional[float] = None,
) -> None:
    from unfold.platform.reward_vis import save_scatter_png

    mask = np.asarray(vertex_mask).astype(bool).reshape(-1)
    if mask.size == 0:
        return
    init_xy = np.asarray(init_pos, dtype=np.float32)[mask, :2]
    other_xy = np.asarray(other_pos, dtype=np.float32)[mask, :2]
    if init_xy.size == 0 or other_xy.size == 0:
        return
    save_scatter_png(
        ref_points=init_xy,
        other_points=other_xy,
        out_path=out_path,
        title=title,
        ref_label="Init",
        other_label="Current",
        distance=distance,
    )


@dataclass
class EvalResult:
    status: str
    skip_reason: Optional[str]
    asset_path: str
    asset_id: Optional[str]
    map_path: Optional[str]
    raw_pair: Optional[tuple[int, int]]
    teacher_pair: Optional[tuple[int, int]]
    predicted_reward: Optional[float]
    visible_raw_vid: np.ndarray
    teacher_anchor_xy: dict[int, tuple[float, float]]
    num_visible_teacher: int
    num_pairs_evaluated: int
    before_meta: dict[str, Any]


class TeacherWorkerClient:
    def __init__(
        self,
        *,
        teacher_python: str,
        teacher_cfg: str,
        teacher_ckpt: str,
        pointcept_code_root: Optional[str],
        device: str,
        pair_chunk_size: int,
    ):
        worker_script = Path(__file__).resolve().parents[1] / "teacher_eval_worker.py"
        cmd = [
            teacher_python,
            str(worker_script),
            "--teacher-cfg",
            teacher_cfg,
            "--teacher-ckpt",
            teacher_ckpt,
            "--device",
            device,
            "--pair-chunk-size",
            str(pair_chunk_size),
        ]
        if pointcept_code_root:
            cmd.extend(["--pointcept-code-root", pointcept_code_root])
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def infer_best(
        self,
        *,
        coord_path: str,
        normal_path: Optional[str],
        candidate_teacher_vid: np.ndarray,
        pair_chunk_size: int,
    ) -> tuple[Optional[tuple[int, int]], Optional[float], int]:
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        req = {
            "cmd": "infer_best",
            "coord_path": coord_path,
            "normal_path": normal_path,
            "candidate_teacher_vid": candidate_teacher_vid.astype(np.int64).tolist(),
            "pair_chunk_size": int(pair_chunk_size),
        }
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = ""
            if self.proc.stderr is not None:
                err = self.proc.stderr.read()
            raise RuntimeError(f"teacher worker exited unexpectedly: {err}")
        resp = json.loads(line)
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("error", "unknown teacher worker error"))
        best_pair = resp.get("best_pair")
        return (
            (int(best_pair[0]), int(best_pair[1])) if best_pair is not None else None,
            resp.get("predicted_reward"),
            int(resp.get("num_pairs_evaluated", 0)),
        )

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class TeacherEvalPolicy:
    def __init__(
        self,
        *,
        teacher,
        worker: Optional[TeacherWorkerClient],
        pointcept_assets: list[dict],
        pointcept_data_root: Path,
        vertex_index_map_path: Optional[Path],
        raw_to_teacher_map_key: str,
        pair_chunk_size: int,
        num_candidates: int,
        teacher_min_candidate_dist: float,
    ):
        self.teacher = teacher
        self.worker = worker
        self.pointcept_assets = pointcept_assets
        self.pointcept_data_root = pointcept_data_root
        self.vertex_index_map_path = vertex_index_map_path
        self.raw_to_teacher_map_key = raw_to_teacher_map_key
        self.pair_chunk_size = int(pair_chunk_size)
        self.num_candidates = int(num_candidates)
        self.teacher_min_candidate_dist = float(teacher_min_candidate_dist)
        self._asset_cache: dict[str, dict[str, Any]] = {}

    def _resolve_asset_bundle(self, asset_path: str) -> dict[str, Any]:
        cache_key = str(asset_path)
        if cache_key in self._asset_cache:
            return self._asset_cache[cache_key]

        pointcept_asset = _resolve_pointcept_asset_by_usd(self.pointcept_assets, usd_rel=asset_path)
        asset_id = str(pointcept_asset["asset_id"])
        asset_dir = self.pointcept_data_root / "assets" / asset_id
        coord_path = asset_dir / "coord.npy"
        normal_path = asset_dir / "normal.npy"
        map_path = _resolve_index_map_path(
            vertex_index_map_path=self.vertex_index_map_path,
            pointcept_asset=pointcept_asset,
            asset_dir=asset_dir,
            mapping_file_key=self.raw_to_teacher_map_key,
        )
        if not coord_path.exists():
            raise FileNotFoundError(f"Pointcept coord.npy not found: {coord_path}")
        index_map = _load_vertex_index_map(map_path)
        map_path_str = str(map_path)

        bundle = {
            "asset_id": asset_id,
            "coord_path": str(coord_path),
            "normal_path": str(normal_path) if normal_path.exists() else None,
            "index_map": index_map,
            "map_path": map_path_str,
        }
        self._asset_cache[cache_key] = bundle
        return bundle

    def _build_candidates(
        self,
        *,
        face_index: np.ndarray,
        face_vertex_ids: np.ndarray,
        barycentric_weights: np.ndarray,
        mask_np: np.ndarray,
        index_map: np.ndarray,
        teacher_nv: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[float, float]]]:
        mask_bool = np.asarray(mask_np).astype(bool)
        hard_slot = np.argmax(barycentric_weights, axis=-1)
        render_vertex_map = np.take_along_axis(face_vertex_ids, hard_slot[..., None], axis=-1)[..., 0]
        render_vertex_map[face_index < 0] = -1

        pix_valid = (face_index >= 0) & mask_bool
        raw_tri = face_vertex_ids[pix_valid]
        raw_valid = (raw_tri >= 0) & (raw_tri < index_map.shape[0])
        teacher_tri = np.full(raw_tri.shape, -1, dtype=np.int64)
        teacher_tri[raw_valid] = index_map[raw_tri[raw_valid]]
        teacher_valid = (teacher_tri >= 0) & (teacher_tri < teacher_nv)
        visible_teacher_vid = np.unique(teacher_tri[teacher_valid]).astype(np.int64)
        visible_raw_vid = np.unique(raw_tri[raw_valid]).astype(np.int64) if raw_tri.size > 0 else np.zeros((0,), dtype=np.int64)

        hard_teacher_map = np.full(render_vertex_map.shape, -1, dtype=np.int64)
        valid_render = (render_vertex_map >= 0) & (render_vertex_map < index_map.shape[0]) & mask_bool
        hard_teacher_map[valid_render] = index_map[render_vertex_map[valid_render]]
        teacher_ok = (hard_teacher_map >= 0) & (hard_teacher_map < teacher_nv)
        if not np.any(teacher_ok):
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), visible_raw_vid, {}

        yx = np.argwhere(teacher_ok)
        teacher_pix = hard_teacher_map[teacher_ok].astype(np.int64)
        raw_pix = render_vertex_map[teacher_ok].astype(np.int64)
        teacher_vid, first_idx = _sorted_unique_with_first(teacher_pix)
        rep_yx = yx[first_idx]
        rep_raw = raw_pix[first_idx]
        keep = _candidate_cap_indices(teacher_vid.shape[0], self.num_candidates)
        teacher_vid = teacher_vid[keep]
        rep_raw = rep_raw[keep]
        rep_yx = rep_yx[keep]
        anchor_xy = {
            int(tvid): (float(rep_yx[i, 1]), float(rep_yx[i, 0]))
            for i, tvid in enumerate(teacher_vid.tolist())
        }
        return teacher_vid, rep_raw.astype(np.int64), visible_raw_vid, anchor_xy

    def _downsample_candidates_by_teacher_xy(
        self,
        *,
        coord_path: str,
        candidate_teacher_vid: np.ndarray,
        candidate_raw_vid: np.ndarray,
        anchor_xy: dict[int, tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[float, float]]]:
        min_dist = float(self.teacher_min_candidate_dist)
        if min_dist <= 0.0 or candidate_teacher_vid.size <= 2:
            return candidate_teacher_vid, candidate_raw_vid, anchor_xy

        coord_xy = np.load(coord_path, mmap_mode="r")[candidate_teacher_vid, :2].astype(np.float32, copy=False)
        n = int(coord_xy.shape[0])
        if n <= 2:
            return candidate_teacher_vid, candidate_raw_vid, anchor_xy

        # Farthest-point ordering to prioritize spatially spread vertices.
        center = np.mean(coord_xy, axis=0, keepdims=True)
        start = int(np.argmax(np.linalg.norm(coord_xy - center, axis=1)))
        order: list[int] = [start]
        min_sq = np.sum((coord_xy - coord_xy[start]) ** 2, axis=1)
        chosen = np.zeros((n,), dtype=bool)
        chosen[start] = True
        for _ in range(1, n):
            min_sq[chosen] = -1.0
            nxt = int(np.argmax(min_sq))
            if min_sq[nxt] < 0:
                break
            order.append(nxt)
            chosen[nxt] = True
            d_sq = np.sum((coord_xy - coord_xy[nxt]) ** 2, axis=1)
            min_sq = np.minimum(min_sq, d_sq)

        keep: list[int] = []
        min_dist_sq = float(min_dist * min_dist)
        for idx in order:
            if not keep:
                keep.append(idx)
                continue
            d_sq = np.sum((coord_xy[keep] - coord_xy[idx]) ** 2, axis=1)
            if float(np.min(d_sq)) >= min_dist_sq:
                keep.append(idx)

        # Avoid degenerating into zero/one-point candidate set.
        if len(keep) < 2:
            keep = order[: min(2, n)]

        keep_idx = np.array(keep, dtype=np.int64)
        if self.num_candidates > 0 and keep_idx.size > self.num_candidates:
            keep_idx = keep_idx[: self.num_candidates]

        new_teacher_vid = candidate_teacher_vid[keep_idx]
        new_raw_vid = candidate_raw_vid[keep_idx]
        new_anchor_xy = {int(t): anchor_xy[int(t)] for t in new_teacher_vid.tolist() if int(t) in anchor_xy}
        return new_teacher_vid, new_raw_vid, new_anchor_xy

    def _find_best_pair(
        self,
        *,
        teacher_coord: np.ndarray,
        teacher_normal: Optional[np.ndarray],
        candidate_teacher_vid: np.ndarray,
    ) -> tuple[Optional[tuple[int, int]], Optional[float], int]:
        n = int(candidate_teacher_vid.shape[0])
        if n <= 1:
            return None, None, 0

        idx = np.arange(n, dtype=np.int64)
        src_idx = np.repeat(idx, n)
        dst_idx = np.tile(idx, n)
        valid = src_idx != dst_idx
        if not np.any(valid):
            return None, None, 0

        pairs = np.stack(
            [
                candidate_teacher_vid[src_idx[valid]].astype(np.int64, copy=False),
                candidate_teacher_vid[dst_idx[valid]].astype(np.int64, copy=False),
            ],
            axis=1,
        )
        pair_count = int(pairs.shape[0])
        scores = self.teacher.infer_pairs(
            coord=teacher_coord,
            pairs=pairs,
            normal=teacher_normal,
            max_pairs_per_forward=self.pair_chunk_size,
        )
        best_idx = int(np.argmax(scores))
        best_pair = (int(pairs[best_idx, 0]), int(pairs[best_idx, 1]))
        best_score = float(scores[best_idx])
        return best_pair, best_score, pair_count

    def evaluate(
        self,
        *,
        asset_path: str,
        face_index: np.ndarray,
        face_vertex_ids: np.ndarray,
        barycentric_weights: np.ndarray,
        mask_np: np.ndarray,
    ) -> EvalResult:
        bundle = self._resolve_asset_bundle(asset_path)

        index_map = bundle["index_map"]
        teacher_nv = int(np.load(bundle["coord_path"], mmap_mode="r").shape[0])
        candidate_teacher_vid, candidate_raw_vid, visible_raw_vid, anchor_xy = self._build_candidates(
            face_index=face_index,
            face_vertex_ids=face_vertex_ids,
            barycentric_weights=barycentric_weights,
            mask_np=mask_np,
            index_map=index_map,
            teacher_nv=teacher_nv,
        )
        candidate_teacher_vid, candidate_raw_vid, anchor_xy = self._downsample_candidates_by_teacher_xy(
            coord_path=bundle["coord_path"],
            candidate_teacher_vid=candidate_teacher_vid,
            candidate_raw_vid=candidate_raw_vid,
            anchor_xy=anchor_xy,
        )
        if candidate_teacher_vid.size == 0:
            raise RuntimeError(f"no_visible_teacher_vertices: asset={asset_path}")

        if self.worker is not None:
            best_teacher_pair, predicted_reward, pair_count = self.worker.infer_best(
                coord_path=bundle["coord_path"],
                normal_path=bundle["normal_path"],
                candidate_teacher_vid=candidate_teacher_vid,
                pair_chunk_size=self.pair_chunk_size,
            )
        else:
            teacher_coord = np.load(bundle["coord_path"]).astype(np.float32)
            teacher_normal = np.load(bundle["normal_path"]).astype(np.float32) if bundle["normal_path"] else None
            best_teacher_pair, predicted_reward, pair_count = self._find_best_pair(
                teacher_coord=teacher_coord,
                teacher_normal=teacher_normal,
                candidate_teacher_vid=candidate_teacher_vid,
            )
        if best_teacher_pair is None:
            raise RuntimeError(
                f"no_candidate_pairs: asset={asset_path} num_visible_teacher={int(candidate_teacher_vid.size)}"
            )

        teacher_to_raw = {int(tid): int(rid) for tid, rid in zip(candidate_teacher_vid.tolist(), candidate_raw_vid.tolist())}
        raw_pair = (teacher_to_raw[int(best_teacher_pair[0])], teacher_to_raw[int(best_teacher_pair[1])])
        return EvalResult(
            status="ok",
            skip_reason=None,
            asset_path=asset_path,
            asset_id=bundle["asset_id"],
            map_path=bundle["map_path"],
            raw_pair=raw_pair,
            teacher_pair=(int(best_teacher_pair[0]), int(best_teacher_pair[1])),
            predicted_reward=float(predicted_reward),
            visible_raw_vid=visible_raw_vid,
            teacher_anchor_xy=anchor_xy,
            num_visible_teacher=int(candidate_teacher_vid.size),
            num_pairs_evaluated=pair_count,
            before_meta={"num_visible_raw": int(visible_raw_vid.size)},
        )


def _prepare_env_cfg(args):
    import isaaclab.sim as sim_utils
    from unfold.platform.config_utils import parse_yaml_config
    from unfold.simulation.env import EnvCfg

    cfg_path = Path(args.config).resolve()
    env_cfg: EnvCfg = parse_yaml_config(
        cfg_path,
        device=(args.device if getattr(args, "device", None) else "cuda:0"),
        env_cfg_class=EnvCfg,
    )
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.num_envs = int(args.num_envs)
    env_cfg.steps_per_episode = 2
    env_cfg.mv_num_views = 1
    env_cfg.camera_res = [int(args.camera_width), int(args.camera_height)]
    env_cfg.cloth_root = "/World/Cloth"
    if not isinstance(getattr(env_cfg, "ground_size_m", None), (list, tuple)) or len(getattr(env_cfg, "ground_size_m", [])) < 2:
        env_cfg.ground_size_m = [3.0, 3.0]
    env_cfg.sim.render = sim_utils.RenderCfg(
        rendering_mode=args.render_mode,
        antialiasing_mode=args.aa,
        dlss_mode=2,
        enable_dl_denoiser=False,
        samples_per_pixel=int(args.spp),
        enable_reflections=True,
        enable_global_illumination=True,
    )
    return env_cfg


def _set_topdown_camera_poses(env, cameras, args) -> None:
    import omni.usd
    from unfold.platform.camera import compute_centers_world, set_camera_prims_look_at

    if bool(getattr(args, "fixed_camera_pose", False)):
        if bool(getattr(args, "_fixed_camera_pose_applied", False)):
            return
        obs = env.unwrapped._get_observations()
        centers_world = compute_centers_world(obs["pos"], obs["pos_mask"], env.unwrapped.scene.env_origins)
        centers_np = centers_world.detach().cpu().numpy().astype(np.float64)
        eyes = centers_np.copy()
        eyes[:, 2] = centers_np[:, 2] + float(args.fixed_camera_height)
        setattr(args, "_fixed_camera_pose_applied", True)
    else:
        obs = env.unwrapped._get_observations()
        centers_world = compute_centers_world(obs["pos"], obs["pos_mask"], env.unwrapped.scene.env_origins)
        pos_np = obs["pos"].detach().cpu().numpy()
        mask_np = obs["pos_mask"].detach().cpu().numpy().astype(bool)
        centers_np = centers_world.detach().cpu().numpy().astype(np.float64)
        eyes = centers_np.copy()
        for env_idx in range(env.unwrapped.num_envs):
            valid = mask_np[env_idx].squeeze(-1)
            max_extent = 0.0
            if np.any(valid):
                pts = pos_np[env_idx][valid]
                extent_xy = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
                max_extent = float(np.max(extent_xy))
            height = max(float(args.topdown_min_height), float(args.topdown_extent_scale) * max_extent)
            eyes[env_idx, 2] = centers_np[env_idx, 2] + height

    ordered_cameras = sorted(
        [c[0] if isinstance(c, list) else c for c in cameras],
        key=_extract_env_idx_from_camera_path,
    )
    stage = omni.usd.get_context().get_stage()
    set_camera_prims_look_at(ordered_cameras, eyes, centers_np, stage=stage)
    env.unwrapped.sim.render()


def _collect_camera_params_only(cameras, env_cfg) -> dict[str, Any]:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    cam_res = getattr(env_cfg, "camera_res", [1024, 1024])
    image_width, image_height = int(cam_res[0]), int(cam_res[1])
    out: dict[str, Any] = {}

    for cam_path in cameras:
        if isinstance(cam_path, list):
            cam_path = cam_path[0]
        prim = stage.GetPrimAtPath(cam_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"camera prim invalid: {cam_path}")
        cam_geom = UsdGeom.Camera(prim)
        if not cam_geom:
            raise RuntimeError(f"camera geom invalid: {cam_path}")

        focal_length = cam_geom.GetFocalLengthAttr().Get()
        h_aperture = cam_geom.GetHorizontalApertureAttr().Get()
        v_aperture = cam_geom.GetVerticalApertureAttr().Get()
        fx = (focal_length / h_aperture) * image_width
        fy = (focal_length / v_aperture) * image_height
        cx = image_width / 2.0
        cy = image_height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(0)
        c2w = np.array(world_transform).reshape(4, 4).T
        w2c = np.linalg.inv(c2w).astype(np.float32)

        out[str(cam_path)] = {
            "rgb": None,
            "depth": None,
            "seg": None,
            "intrinsics": K,
            "extrinsics": w2c,
            "image_hw": (image_height, image_width),
            "camera_distance_m": None,
        }
    return out


def _capture_frames(env, env_cfg, args, cameras, render_products, annotators) -> dict[str, Any]:
    from unfold.workflows.rendering.pipeline import _capture_one_frame

    _set_topdown_camera_poses(env, cameras, args)
    if str(getattr(args, "visibility_source", "mesh")).lower() == "mesh" and bool(getattr(args, "fixed_camera_pose", False)):
        return _collect_camera_params_only(cameras, env_cfg)
    return _capture_one_frame(cameras, render_products, annotators, env_cfg, args, env)


def _build_face_raster(
    frame: dict[str, Any],
    obs: dict[str, Any],
    env_idx: int,
    projection_overlap_threshold: float,
    visibility_source: str,
) -> dict[str, Any]:
    from unfold.workflows.rendering.geometry import _rasterize_face_index_and_barycentric_from_mesh

    seg = frame.get("seg")
    use_seg = str(visibility_source).lower() == "seg"
    if use_seg and seg is None:
        raise RuntimeError("capture missing cloth mask")

    image_hw: Optional[tuple[int, int]] = None
    if seg is not None:
        m = np.asarray(seg)
        image_hw = (int(m.shape[0]), int(m.shape[1]))
    elif frame.get("rgb") is not None:
        rgb = np.asarray(frame["rgb"])
        image_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
    elif frame.get("depth") is not None:
        depth = np.asarray(frame["depth"])
        image_hw = (int(depth.shape[0]), int(depth.shape[1]))
    elif frame.get("image_hw") is not None:
        h, w = frame["image_hw"]
        image_hw = (int(h), int(w))
    if image_hw is None:
        raise RuntimeError("capture missing image size for rasterization")

    vertices_local, vertex_mask, init_pos_local, env_origin = _extract_vertices_for_env(obs, env_idx)
    faces = _extract_faces_for_env(obs, env_idx)
    if faces.ndim != 2 or faces.shape[0] == 0:
        raise RuntimeError("observation faces invalid or empty")

    face_index, face_vertex_ids, barycentric_weights, _ = _rasterize_face_index_and_barycentric_from_mesh(
        mask_np=seg if use_seg else None,
        vertices_local=vertices_local,
        init_pos_local=init_pos_local,
        vertex_valid_mask=vertex_mask,
        faces=faces,
        K=frame.get("intrinsics"),
        w2c=frame.get("extrinsics"),
        env_origin=env_origin,
        image_hw=image_hw,
        projection_overlap_threshold=projection_overlap_threshold,
    )
    if face_index is None or face_vertex_ids is None or barycentric_weights is None:
        raise RuntimeError("face rasterization returned no valid pixels")
    return {
        "face_index": np.asarray(face_index).astype(np.int64),
        "face_vertex_ids": np.asarray(face_vertex_ids).astype(np.int64),
        "barycentric_weights": np.asarray(barycentric_weights).astype(np.float32),
    }


def _build_visible_raw_anchor_map(face_index: np.ndarray, face_vertex_ids: np.ndarray, barycentric_weights: np.ndarray) -> dict[int, tuple[float, float]]:
    hard_slot = np.argmax(barycentric_weights, axis=-1)
    render_vertex_map = np.take_along_axis(face_vertex_ids, hard_slot[..., None], axis=-1)[..., 0]
    render_vertex_map[face_index < 0] = -1
    valid = render_vertex_map >= 0
    if not np.any(valid):
        return {}
    yx = np.argwhere(valid)
    raw_pix = render_vertex_map[valid].astype(np.int64)
    raw_vid, first_idx = _sorted_unique_with_first(raw_pix)
    rep_yx = yx[first_idx]
    return {int(rid): (float(rep_yx[i, 1]), float(rep_yx[i, 0])) for i, rid in enumerate(raw_vid.tolist())}


def _write_sample_outputs(
    *,
    sample_dir: Path,
    init_pos: np.ndarray,
    before_pos: np.ndarray,
    after_pos: np.ndarray,
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    deformable_compare: Optional[tuple[np.ndarray, np.ndarray, float]],
    rigid_compare: Optional[tuple[np.ndarray, np.ndarray, float]],
    raw_pair: Optional[tuple[int, int]],
    teacher_pair: Optional[tuple[int, int]],
    predicted_reward: Optional[float],
    actual_reward: Optional[float],
    error: Optional[float],
    meta: dict[str, Any],
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    _save_xy_projection_png(
        init_pos=init_pos,
        other_pos=before_pos,
        vertex_mask=before_mask,
        out_path=sample_dir / "before_xy_projection.png",
        title=f"Before XY | raw_pair={raw_pair} teacher_pair={teacher_pair}",
        distance=predicted_reward if predicted_reward is not None else None,
    )
    _save_xy_projection_png(
        init_pos=init_pos,
        other_pos=after_pos,
        vertex_mask=after_mask,
        out_path=sample_dir / "after_xy_projection.png",
        title=f"After XY | raw_pair={raw_pair}",
        distance=actual_reward if actual_reward is not None else None,
    )

    if deformable_compare is not None:
        goal_xy, icp_verts, deformable_dist = deformable_compare
        from unfold.platform.reward_vis import save_scatter_png

        save_scatter_png(
            ref_points=goal_xy,
            other_points=icp_verts,
            out_path=sample_dir / "deformable_projection.png",
            title="Deformable (Goal vs ICP-Aligned Current)",
            ref_label="Goal",
            other_label="Aligned Current",
            distance=deformable_dist,
        )
    if rigid_compare is not None:
        goal_xy, reverse_goal_xy, rigid_dist = rigid_compare
        from unfold.platform.reward_vis import save_scatter_png

        save_scatter_png(
            ref_points=goal_xy,
            other_points=reverse_goal_xy,
            out_path=sample_dir / "rigid_projection.png",
            title="Rigid (Goal vs Aligned Goal)",
            ref_label="Goal",
            other_label="Aligned Goal",
            distance=rigid_dist,
        )

    (sample_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [r for r in rows if r["status"] == "ok"]
    skip_rows = [r for r in rows if r["status"] != "ok"]
    pred = [float(r["predicted_reward"]) for r in success_rows if r["predicted_reward"] is not None]
    actual = [float(r["actual_reward"]) for r in success_rows if r["actual_reward"] is not None]
    abs_err = [float(r["abs_error"]) for r in success_rows if r["abs_error"] is not None]
    skip_reasons: dict[str, int] = {}
    for row in skip_rows:
        key = str(row.get("skip_reason") or "unknown")
        skip_reasons[key] = skip_reasons.get(key, 0) + 1
    return {
        "num_rows": len(rows),
        "num_success": len(success_rows),
        "num_skipped": len(skip_rows),
        "mean_predicted_reward": float(np.mean(pred)) if pred else None,
        "mean_actual_reward": float(np.mean(actual)) if actual else None,
        "mae": float(np.mean(abs_err)) if abs_err else None,
        "skip_reasons": skip_reasons,
    }


def run(args) -> None:
    # Replicator/camera extensions are loaded by AppLauncher only when cameras are enabled.
    args.enable_cameras = True
    # Mesh-only visibility does not require semantic segmentation annotators.
    args.enable_semantic_seg = str(getattr(args, "visibility_source", "seg")).lower() == "seg"
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import carb
    import gymnasium as gym
    import torch
    import unfold  # noqa: F401
    import omni.kit.app
    from unfold.data.storage.structured_hdf5 import StructuredHDF5Storage
    from unfold.workflows.rendering.pipeline import _setup_replicator

    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    carb.log_warn = lambda *a, **k: None
    # Keep parity with rendering pipeline: enable experimental material extension used by ground randomizer.
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.core.experimental.materials", True)

    np.random.seed(int(args.seed))
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    pointcept_data_root = Path(args.pointcept_data_root).resolve()
    pointcept_manifest = Path(args.pointcept_manifest).resolve() if args.pointcept_manifest else (pointcept_data_root / "manifest.json")
    pointcept_assets = _load_pointcept_manifest(pointcept_manifest)
    vertex_index_map_path = Path(args.vertex_index_map).resolve() if args.vertex_index_map else None

    teacher = None
    worker = None
    if args.teacher_python:
        worker = TeacherWorkerClient(
            teacher_python=args.teacher_python,
            teacher_cfg=args.teacher_cfg,
            teacher_ckpt=args.teacher_ckpt,
            pointcept_code_root=args.pointcept_code_root,
            device=args.device if getattr(args, "device", None) else "cuda",
            pair_chunk_size=args.pair_chunk_size,
        )
    else:
        from unfold.algorithms.supervision.teacher_pointcept import TeacherRewardInfer

        teacher = TeacherRewardInfer(
            teacher_cfg=args.teacher_cfg,
            teacher_ckpt=args.teacher_ckpt,
            device=args.device if getattr(args, "device", None) else "cuda",
            pointcept_code_root=args.pointcept_code_root,
        )
    policy = TeacherEvalPolicy(
        teacher=teacher,
        worker=worker,
        pointcept_assets=pointcept_assets,
        pointcept_data_root=pointcept_data_root,
        vertex_index_map_path=vertex_index_map_path,
        raw_to_teacher_map_key=args.raw_to_teacher_map_key,
        pair_chunk_size=args.pair_chunk_size,
        num_candidates=args.num_candidates,
        teacher_min_candidate_dist=args.teacher_min_candidate_dist,
    )

    env_cfg = _prepare_env_cfg(args)
    env = gym.make(args.task, cfg=env_cfg)
    cameras, render_products, annotators = _setup_replicator(env, env_cfg, args)

    pool = env.unwrapped._asset_pool
    num_batches = int(getattr(pool, "num_batches", 1))
    total_steps = max(int(args.epochs), 1) * num_batches * int(env_cfg.episodes_per_asset_batch)
    if args.max_steps is not None:
        total_steps = int(args.max_steps)

    csv_path = out_dir / args.results_csv
    h5_path = out_dir / args.results_h5
    summary_path = out_dir / args.summary_name
    storage = StructuredHDF5Storage(str(h5_path), feature_dim=0, write_threshold=args.write_threshold, overwrite=True)

    rows: list[dict[str, Any]] = []
    episodes_in_batch = 0
    batch_counter = 0
    current_epoch = 1
    step_idx = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "step",
                "env",
                "status",
                "skip_reason",
                "asset",
                "asset_id",
                "raw_pair_a",
                "raw_pair_b",
                "teacher_pair_a",
                "teacher_pair_b",
                "predicted_reward",
                "actual_reward",
                "error",
                "abs_error",
                "num_visible_teacher",
                "num_pairs_evaluated",
                "sample_dir",
            ],
        )
        writer.writeheader()

        try:
            env.unwrapped.reset(
                options={
                    "switch_asset": True,
                    "epoch_info": {"epoch": current_epoch, "total_epochs": args.epochs, "batch": 1, "total_batches": num_batches},
                }
            )
            setattr(args, "_fixed_camera_pose_applied", False)

            while simulation_app.is_running() and step_idx < total_steps:
                step_idx += 1
                obs_before = _collect_observations(env)
                before_frames = _capture_frames(env, env_cfg, args, cameras, render_products, annotators)

                actions = np.full((env.unwrapped.num_envs, 2), -1, dtype=np.int64)
                per_env_state: list[dict[str, Any]] = []
                asset_paths = getattr(env.unwrapped._garment_manager, "_env_usd_paths", None) or []

                for env_idx in range(env.unwrapped.num_envs):
                    cam_path = next(cp for cp in before_frames.keys() if _extract_env_idx_from_camera_path(str(cp)) == env_idx)
                    frame_before = before_frames[cam_path]
                    asset_path = str(asset_paths[env_idx]) if env_idx < len(asset_paths) else ""
                    before_raster = _build_face_raster(
                        frame_before,
                        obs_before,
                        env_idx,
                        args.projection_overlap_threshold,
                        args.visibility_source,
                    )
                    candidate_mask = frame_before["seg"]
                    if str(args.visibility_source).lower() == "mesh":
                        candidate_mask = np.ones_like(before_raster["face_index"], dtype=bool)
                    eval_result = policy.evaluate(
                        asset_path=asset_path,
                        face_index=before_raster["face_index"],
                        face_vertex_ids=before_raster["face_vertex_ids"],
                        barycentric_weights=before_raster["barycentric_weights"],
                        mask_np=candidate_mask,
                    )
                    selected_pair = eval_result.raw_pair
                    if selected_pair is None:
                        raise RuntimeError(f"strict_mode_violation: eval_result.raw_pair is None for asset={asset_path}")
                    actions[env_idx] = np.asarray(selected_pair, dtype=np.int64)
                    per_env_state.append(
                        {
                            "env_idx": env_idx,
                            "cam_path": cam_path,
                            "frame_before": frame_before,
                            "before_raster": before_raster,
                            "eval_result": eval_result,
                            "selected_pair": selected_pair,
                        }
                    )

                actions_t = torch.from_numpy(actions).to(device=env.unwrapped.device, dtype=torch.long)
                _, rewards, _, _, info = env.unwrapped.step(actions_t)
                obs_after = _collect_observations(env)

                reward_list = rewards.detach().cpu().tolist()
                extras = info.get("rewards_extras", info) if isinstance(info, dict) else {}
                l2_val = extras.get("l2_distance")
                icp_val = extras.get("icp_distance")

                for state in per_env_state:
                    env_idx = state["env_idx"]
                    eval_result: EvalResult = state["eval_result"]
                    frame_before = state["frame_before"]
                    raw_pair = tuple(int(x) for x in state["selected_pair"])
                    sample_id = f"{step_idx:08d}_env{env_idx:02d}"
                    sample_dir = samples_dir / sample_id

                    init_pos_np = obs_before["init_pos"][env_idx].detach().cpu().numpy().astype(np.float32)
                    before_pos_np = obs_before["pos"][env_idx].detach().cpu().numpy().astype(np.float32)
                    after_pos_np = obs_after["pos"][env_idx].detach().cpu().numpy().astype(np.float32)
                    before_mask_np = obs_before["pos_mask"][env_idx].detach().cpu().numpy().astype(bool).squeeze(-1)
                    after_mask_np = obs_after["pos_mask"][env_idx].detach().cpu().numpy().astype(bool).squeeze(-1)

                    actual_reward = None
                    error = None
                    abs_error = None
                    deformable_dist_val = None
                    rigid_dist_val = None
                    if eval_result.status == "ok":
                        actual_reward = float(reward_list[env_idx])
                        error = float(actual_reward - float(eval_result.predicted_reward))
                        abs_error = abs(error)

                    deformable_tensor = extras.get("deformable_distance", l2_val)
                    rigid_tensor = extras.get("rigid_distance", icp_val)
                    if deformable_tensor is not None:
                        dv = deformable_tensor[env_idx]
                        deformable_dist_val = float(dv.item() if hasattr(dv, "item") else dv)
                    if rigid_tensor is not None:
                        rv = rigid_tensor[env_idx]
                        rigid_dist_val = float(rv.item() if hasattr(rv, "item") else rv)

                    deformable_compare = None
                    rigid_compare = None
                    goal_xy_t = extras.get("goal_xy")
                    reverse_goal_xy_t = extras.get("reverse_goal_xy")
                    icp_verts_t = extras.get("icp_verts")
                    padding_mask_t = extras.get("padding_mask")
                    if (
                        goal_xy_t is not None
                        and reverse_goal_xy_t is not None
                        and icp_verts_t is not None
                        and padding_mask_t is not None
                    ):
                        goal_xy_np = goal_xy_t[env_idx].detach().cpu().numpy()
                        reverse_goal_xy_np = reverse_goal_xy_t[env_idx].detach().cpu().numpy()
                        icp_verts_np = icp_verts_t[env_idx].detach().cpu().numpy()
                        padding_mask_np = padding_mask_t[env_idx].detach().cpu().numpy()
                        valid = padding_mask_np[:, 0] > 0.5
                        if np.any(valid):
                            goal_xy_use = goal_xy_np[valid]
                            reverse_goal_xy_use = reverse_goal_xy_np[valid]
                            icp_verts_use = icp_verts_np[valid]
                            if deformable_dist_val is not None:
                                deformable_compare = (goal_xy_use, icp_verts_use, deformable_dist_val)
                            if rigid_dist_val is not None:
                                rigid_compare = (goal_xy_use, reverse_goal_xy_use, rigid_dist_val)

                    meta = {
                        "step": step_idx,
                        "env_idx": env_idx,
                        "status": eval_result.status,
                        "skip_reason": eval_result.skip_reason,
                        "asset_path": eval_result.asset_path,
                        "asset_id": eval_result.asset_id,
                        "vertex_index_map": eval_result.map_path,
                        "raw_pair": list(raw_pair),
                        "teacher_pair": list(eval_result.teacher_pair) if eval_result.teacher_pair is not None else None,
                        "predicted_reward": eval_result.predicted_reward,
                        "actual_reward": actual_reward,
                        "error": error,
                        "deformable_distance": deformable_dist_val,
                        "rigid_distance": rigid_dist_val,
                        "num_visible_teacher": eval_result.num_visible_teacher,
                        "num_pairs_evaluated": eval_result.num_pairs_evaluated,
                    }
                    meta.update(eval_result.before_meta)
                    _write_sample_outputs(
                        sample_dir=sample_dir,
                        init_pos=init_pos_np,
                        before_pos=before_pos_np,
                        after_pos=after_pos_np,
                        before_mask=before_mask_np,
                        after_mask=after_mask_np,
                        deformable_compare=deformable_compare,
                        rigid_compare=rigid_compare,
                        raw_pair=raw_pair if eval_result.status == "ok" else None,
                        teacher_pair=eval_result.teacher_pair,
                        predicted_reward=eval_result.predicted_reward,
                        actual_reward=actual_reward,
                        error=error,
                        meta=meta,
                    )

                    row = {
                        "step": step_idx,
                        "env": env_idx,
                        "status": eval_result.status,
                        "skip_reason": eval_result.skip_reason,
                        "asset": eval_result.asset_path,
                        "asset_id": eval_result.asset_id,
                        "raw_pair_a": raw_pair[0] if eval_result.status == "ok" else None,
                        "raw_pair_b": raw_pair[1] if eval_result.status == "ok" else None,
                        "teacher_pair_a": eval_result.teacher_pair[0] if eval_result.teacher_pair is not None else None,
                        "teacher_pair_b": eval_result.teacher_pair[1] if eval_result.teacher_pair is not None else None,
                        "predicted_reward": eval_result.predicted_reward,
                        "actual_reward": actual_reward,
                        "error": error,
                        "abs_error": abs_error,
                        "num_visible_teacher": eval_result.num_visible_teacher,
                        "num_pairs_evaluated": eval_result.num_pairs_evaluated,
                        "sample_dir": str(sample_dir),
                    }
                    writer.writerow(row)
                    rows.append(row)

                    if eval_result.status == "ok" and actual_reward is not None:
                        l2_use = 0.0
                        icp_use = 0.0
                        if l2_val is not None:
                            l2_use = float(l2_val[env_idx].item() if hasattr(l2_val[env_idx], "item") else l2_val[env_idx])
                        if icp_val is not None:
                            icp_use = float(icp_val[env_idx].item() if hasattr(icp_val[env_idx], "item") else icp_val[env_idx])
                        storage.add(
                            asset_path=eval_result.asset_path,
                            id1=raw_pair[0],
                            id2=raw_pair[1],
                            reward=actual_reward,
                            l2_dist=l2_use,
                            icp_dist=icp_use,
                        )

                summary_path.write_text(json.dumps(_summary_from_rows(rows), ensure_ascii=False, indent=2), encoding="utf-8")
                episodes_in_batch += 1
                switch_asset = episodes_in_batch >= int(env_cfg.episodes_per_asset_batch)
                if switch_asset:
                    episodes_in_batch = 0
                    batch_counter += 1
                    current_epoch = (batch_counter // max(num_batches, 1)) + 1
                next_batch = (batch_counter % max(num_batches, 1)) + 1
                env.unwrapped.reset(
                    options={
                        "switch_asset": switch_asset,
                        "epoch_info": {
                            "epoch": current_epoch,
                            "total_epochs": args.epochs,
                            "batch": next_batch,
                            "total_batches": num_batches,
                        },
                    }
                )
                setattr(args, "_fixed_camera_pose_applied", False)
        finally:
            storage.close()
            if worker is not None:
                worker.close()
            env.close()

    summary = _summary_from_rows(rows)
    summary["total_steps_requested"] = total_steps
    summary["pointcept_manifest"] = str(pointcept_manifest)
    summary["output_dir"] = str(out_dir)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    simulation_app.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
