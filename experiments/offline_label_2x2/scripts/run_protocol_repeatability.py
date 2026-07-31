#!/usr/bin/env python3
"""Run repeated fixed-pair evaluations for the offline-label 2x2 protocols."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import resource
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher
from unfold.platform.camera import rotmat_to_quat_wxyz

if TYPE_CHECKING:
    import torch

    from unfold.workflows.offline_collection.pair_conditioned_collect import (
        PairCandidate,
        PairConditionedOfflineCollector,
    )


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    init_mode: str
    loading_mode: str


PROTOCOLS: dict[str, ProtocolSpec] = {
    "random_fling": ProtocolSpec("random_fling", init_mode="random", loading_mode="fling"),
    "random_y": ProtocolSpec("random_y", init_mode="random", loading_mode="y_gravity"),
    "cond_fling": ProtocolSpec("cond_fling", init_mode="conditioned", loading_mode="fling"),
    "cond_y": ProtocolSpec("cond_y", init_mode="conditioned", loading_mode="y_gravity"),
}


@dataclass
class ClothSnapshot:
    positions: torch.Tensor
    velocities: torch.Tensor
    label: str


class StateSnapshotRecorder:
    """Records sparse cloth states without requesting rendering during rollout."""

    def __init__(self, env, *, sample_hz: float, physics_dt: float):
        self.env = env
        self.sample_every_steps = max(1, int(round(1.0 / max(float(sample_hz), 1e-6) / float(physics_dt))))
        self.global_step = 0
        self.snapshots: list[ClothSnapshot] = []
        self.render_every_step = False
        self.should_stop = False

    def wants_capture(self, *, phase_type: str, step_idx: int) -> bool:
        del phase_type, step_idx
        return False

    def capture(self, *, phase_type: str, step_idx: int) -> None:
        del step_idx
        if (self.global_step % self.sample_every_steps) == 0:
            garment = self.env.unwrapped._garment_manager
            self.snapshots.append(
                ClothSnapshot(
                    positions=garment._get_particle_positions().detach().cpu().clone(),
                    velocities=garment._get_particle_velocities().detach().cpu().clone(),
                    label=f"{phase_type}:{self.global_step}",
                )
            )
        self.global_step += 1

    def append_final(self, *, label: str = "final") -> None:
        garment = self.env.unwrapped._garment_manager
        self.snapshots.append(
            ClothSnapshot(
                positions=garment._get_particle_positions().detach().cpu().clone(),
                velocities=garment._get_particle_velocities().detach().cpu().clone(),
                label=label,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeatability runner for the offline-label 2x2 protocols.")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0", help="Gym task id.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/offline_label_2x2/configs/offline_label_2x2.yaml",
        help="YAML config path.",
    )
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel environments used as repeat rollouts.")
    parser.add_argument(
        "--protocol",
        type=str,
        default="all",
        choices=["all", *sorted(PROTOCOLS.keys())],
        help="Protocol to run. Use 'all' to sweep the full 2x2 matrix on the same asset/pair set.",
    )
    parser.add_argument(
        "--asset-indices",
        type=str,
        default="0",
        help="Comma-separated 0-based asset indices in the asset pool.",
    )
    parser.add_argument("--num-pairs", type=int, default=32, help="Number of ordered pairs per asset.")
    parser.add_argument("--repeats-per-pair", type=int, default=8, help="Repeated executions per (asset, pair, protocol).")
    parser.add_argument(
        "--rot-noise-deg",
        type=float,
        default=0.0,
        help="Optional conditioned-init Euler noise range in degrees per axis.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/offline_label_2x2/runs/pilot",
        help="Output directory for CSV/JSON summaries.",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="experiments/offline_label_2x2/runs/pilot/visuals",
        help="Root directory for shared collector debug visuals.",
    )
    parser.add_argument("--assets-manifest", type=str, default=None, help="Optional asset manifest JSON overriding valid_assets.json.")
    parser.add_argument(
        "--pairs-manifest",
        type=str,
        default=None,
        help="Optional fixed evaluation-pairs JSON. When set, uses the listed assets/pairs instead of sampling on the fly.",
    )
    parser.add_argument("--relift-height-min", type=float, default=0.8, help="Random init relift minimum height.")
    parser.add_argument("--relift-height-max", type=float, default=1.2, help="Random init relift maximum height.")
    parser.add_argument("--relift-xy-jitter", type=float, default=0.05, help="Random init relift xy jitter magnitude.")
    parser.add_argument("--debug-protocol-trace", action="store_true", help="Print key protocol-stage height/pose summaries.")
    parser.add_argument("--debug-stretch-trace", action="store_true", help="Print stretch termination reasons.")
    parser.add_argument("--debug-loop-trace", action="store_true", help="Print runner loop progress around pair rebuild/init/step.")
    parser.add_argument(
        "--debug-stack-dump-seconds",
        type=int,
        default=0,
        help="When >0, dump all Python thread stacks to stderr at this interval for diagnosing stalls.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing records.csv in output-dir if present.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite experiment outputs if the run directory exists.")
    parser.add_argument(
        "--flush-every-batches",
        type=int,
        default=8,
        help="Flush records/progress/summary to disk every N simulation batches instead of every batch.",
    )
    parser.add_argument(
        "--convex-hull-max-points",
        type=int,
        default=512,
        help="Maximum points per env used for CPU convex-hull diagnostics. Use 0 for exact full-mask hulls.",
    )
    parser.add_argument("--export-qualitative", action="store_true", help="Export trusted qualitative keyframes for a single asset/pair/protocol setup.")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory for trusted qualitative exports. Defaults to output-dir/qualitative.")
    parser.add_argument("--export-sample-fps", type=float, default=4.0, help="Sparse rollout-state sampling rate used for trusted qualitative export.")
    parser.add_argument("--export-video-width", type=int, default=1024)
    parser.add_argument("--export-video-height", type=int, default=1024)
    parser.add_argument("--export-ground-size", type=float, default=8.0)
    parser.add_argument("--export-warmup-renders", type=int, default=4)
    parser.add_argument("--export-keyframes-per-view", type=int, default=16)
    parser.add_argument("--export-cloth-color", type=float, nargs=3, default=(0.42, 0.84, 0.90))
    parser.add_argument("--export-fixed-cameras", action="store_true", default=True)
    parser.add_argument("--export-disable-fixed-cameras", dest="export_fixed_cameras", action="store_false")
    parser.add_argument("--export-top-eye", type=float, nargs=3, default=(0.0, 0.55, 3.0))
    parser.add_argument("--export-top-target", type=float, nargs=3, default=(0.0, 0.55, 0.12))
    parser.add_argument("--export-side-eye", type=float, nargs=3, default=(1.2, 0.55, 2.45))
    parser.add_argument("--export-side-target", type=float, nargs=3, default=(0.0, 0.55, 0.88))
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_asset_indices(raw: str) -> list[int]:
    values = [part.strip() for part in str(raw).split(",")]
    indices = [int(part) for part in values if part]
    if not indices:
        raise ValueError("asset-indices must contain at least one index.")
    if len(set(indices)) != len(indices):
        raise ValueError("asset-indices contains duplicates.")
    return indices


def _selected_protocols(name: str) -> list[ProtocolSpec]:
    if name == "all":
        return [PROTOCOLS[key] for key in ("random_fling", "random_y", "cond_fling", "cond_y")]
    return [PROTOCOLS[name]]


def _fixed_asset_samples_complete(
    *,
    asset_index: int,
    protocols: list[ProtocolSpec],
    pairs: list["FixedPair"],
    repeats_per_pair: int,
    done_keys: set[tuple[int, str, int, int]],
) -> bool:
    if not pairs:
        return False
    for spec in protocols:
        for pair_idx in range(len(pairs)):
            for repeat_idx in range(int(repeats_per_pair)):
                if (asset_index, spec.name, pair_idx, repeat_idx) not in done_keys:
                    return False
    return True


def _debug_trace(args, message: str) -> None:
    if not getattr(args, "debug_loop_trace", False):
        return
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is KiB on Linux.
    rss_mb = float(usage.ru_maxrss) / 1024.0
    cuda = ""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        cuda_alloc_mb = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
        cuda_reserved_mb = torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
        cuda = f" cuda_alloc_mb={cuda_alloc_mb:.1f} cuda_reserved_mb={cuda_reserved_mb:.1f}"
    print(f"[TRACE] {message} rss_max_mb={rss_mb:.1f}{cuda}", flush=True)


@dataclass(frozen=True)
class FixedPair:
    coord_id1: int
    coord_id2: int
    raw_id1: int
    raw_id2: int
    distance: float
    bin_idx: int


def _load_pairs_manifest(path: str | None) -> dict[int, list[FixedPair]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result: dict[int, list[FixedPair]] = {}
    for asset_entry in payload.get("assets", []):
        local_asset_index = int(asset_entry["local_asset_index"])
        result[local_asset_index] = [
            FixedPair(
                coord_id1=int(item["coord_id1"]),
                coord_id2=int(item["coord_id2"]),
                raw_id1=int(item["raw_id1"]),
                raw_id2=int(item["raw_id2"]),
                distance=float(item["distance"]),
                bin_idx=int(item["bin_idx"]),
            )
            for item in asset_entry.get("pairs", [])
        ]
    return result


def _configure_protocol(env, env_cfg, spec: ProtocolSpec, args) -> None:
    random_mode = spec.init_mode == "random"
    env_cfg.randomize_on_reset = random_mode
    env.cfg.randomize_on_reset = random_mode

    spawn_cfg_env = getattr(env_cfg, "spawn_cfg", {}) or {}
    spawn_cfg_runtime = getattr(env.cfg, "spawn_cfg", {}) or {}
    relift_cfg = {
        "enabled": random_mode,
        "height_range": [float(args.relift_height_min), float(args.relift_height_max)],
        "xy_jitter": [float(args.relift_xy_jitter), float(args.relift_xy_jitter)],
    }
    spawn_cfg_env["predrop_relift"] = relift_cfg
    spawn_cfg_runtime["predrop_relift"] = dict(relift_cfg)

    action_cfg_env = getattr(env_cfg, "action_sequence", {}) or {}
    action_cfg_runtime = getattr(env.cfg, "action_sequence", {}) or {}
    action_cfg_env["init_mode"] = spec.init_mode
    action_cfg_runtime["init_mode"] = spec.init_mode
    action_cfg_env["loading_mode"] = spec.loading_mode
    action_cfg_runtime["loading_mode"] = spec.loading_mode
    env._unfold.init_mode = spec.init_mode
    env._unfold.loading_mode = spec.loading_mode

    debug_cfg_env = getattr(env_cfg, "debug", None)
    debug_cfg_runtime = getattr(env.cfg, "debug", None)

    def _set_debug_flag(container, name: str, value: bool):
        if container is None:
            return None
        if isinstance(container, dict):
            container[name] = value
            return container
        try:
            container[name] = value
            return container
        except Exception:
            try:
                setattr(container, name, value)
                return container
            except Exception:
                return container

    if debug_cfg_env is None:
        debug_cfg_env = SimpleNamespace()
        env_cfg.debug = debug_cfg_env
    if debug_cfg_runtime is None:
        debug_cfg_runtime = SimpleNamespace()
        env.cfg.debug = debug_cfg_runtime

    debug_cfg_env = _set_debug_flag(debug_cfg_env, "protocol_trace", bool(args.debug_protocol_trace))
    debug_cfg_runtime = _set_debug_flag(debug_cfg_runtime, "protocol_trace", bool(args.debug_protocol_trace))
    debug_cfg_env = _set_debug_flag(debug_cfg_env, "stretch_trace", bool(args.debug_stretch_trace))
    debug_cfg_runtime = _set_debug_flag(debug_cfg_runtime, "stretch_trace", bool(args.debug_stretch_trace))


def _build_full_actions(device: "torch.device", num_envs: int, candidate: "PairCandidate") -> "torch.Tensor":
    import torch

    actions = torch.full((num_envs, 2), -1, dtype=torch.long, device=device)
    actions[:, 0] = int(candidate.raw_id1)
    actions[:, 1] = int(candidate.raw_id2)
    return actions


def _build_full_actions_from_raw_ids(device: "torch.device", num_envs: int, raw_id1: int, raw_id2: int) -> "torch.Tensor":
    import torch

    actions = torch.full((num_envs, 2), -1, dtype=torch.long, device=device)
    actions[:, 0] = int(raw_id1)
    actions[:, 1] = int(raw_id2)
    return actions


def _apply_init(
    collector: "PairConditionedOfflineCollector",
    *,
    spec: ProtocolSpec,
    candidates: list["PairCandidate"],
    rot_noise_deg: tuple[float, float, float],
    random_reset_seed: int | None = None,
) -> None:
    if spec.init_mode == "conditioned":
        collector._apply_pair_conditioned_poses(candidates, rot_noise_deg=rot_noise_deg)
        return
    env = collector.env.unwrapped
    prev_env_seed = getattr(env.cfg, "seed", None)
    prev_cfg_seed = getattr(collector.env_cfg, "seed", None)
    if random_reset_seed is not None:
        env.cfg.seed = int(random_reset_seed)
        collector.env_cfg.seed = int(random_reset_seed)
    try:
        collector.env.reset()
    finally:
        env.cfg.seed = prev_env_seed
        collector.env_cfg.seed = prev_cfg_seed


def _repeat_random_seed(base_seed: int, asset_index: int, pair_idx: int, repeat_start: int) -> int:
    # Use a deterministic mixed integer seed so random_y and random_fling share
    # the same reset state for the same (asset, pair, repeat batch).
    seed = int(base_seed) & 0x7FFFFFFF
    seed = (seed * 1000003 + int(asset_index)) & 0x7FFFFFFF
    seed = (seed * 1000003 + int(pair_idx)) & 0x7FFFFFFF
    seed = (seed * 1000003 + int(repeat_start)) & 0x7FFFFFFF
    return seed


def _extract_component_list(extras: dict, key: str, batch_n: int) -> list[float]:
    rewards_extras = extras.get("rewards_extras", {}) if isinstance(extras, dict) else {}
    tensor = rewards_extras.get(key)
    if tensor is None:
        return [float("nan")] * batch_n
    if hasattr(tensor, "detach"):
        values = tensor.detach().cpu().view(-1).tolist()
    else:
        values = list(tensor)
    if len(values) < batch_n:
        values = values + [float("nan")] * (batch_n - len(values))
    return [float(values[i]) for i in range(batch_n)]


def _safe_normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return vec / norm


def _rebuild_pair_candidate_inline(pointcloud, pair_entry, *, device: torch.device):
    from unfold.workflows.offline_collection.pair_conditioned_collect import PairCandidate

    coord_id1 = int(pair_entry.coord_id1)
    coord_id2 = int(pair_entry.coord_id2)
    raw_id1 = int(pointcloud.coord2raw[coord_id1])
    raw_id2 = int(pointcloud.coord2raw[coord_id2])
    p1 = pointcloud.raw_coord[raw_id1]
    p2 = pointcloud.raw_coord[raw_id2]
    midpoint = 0.5 * (p1 + p2)
    pair_dir = _safe_normalize(p2 - p1)
    if pair_dir is None:
        return None
    rel = pointcloud.raw_coord - midpoint[None, :]
    proj = rel - np.outer(rel @ pair_dir, pair_dir)
    centroid = proj.mean(axis=0)
    centroid = centroid - float(np.dot(centroid, pair_dir)) * pair_dir
    body_dir = _safe_normalize(centroid)
    if body_dir is None:
        return None
    normal = _safe_normalize(np.cross(pair_dir, body_dir))
    if normal is None:
        return None
    body_dir = _safe_normalize(np.cross(normal, pair_dir))
    if body_dir is None:
        return None
    basis = np.stack([pair_dir, body_dir, normal], axis=1).astype(np.float32, copy=False)
    rotation = torch.from_numpy(basis.T.copy()).to(device=device, dtype=torch.float32)
    quat = rotmat_to_quat_wxyz(rotation.unsqueeze(0))[0]
    midpoint_t = torch.as_tensor(midpoint, device=device, dtype=torch.float32)
    rotated_midpoint = rotation @ midpoint_t
    return PairCandidate(
        coord_id1=coord_id1,
        coord_id2=coord_id2,
        raw_id1=raw_id1,
        raw_id2=raw_id2,
        quat_wxyz=quat,
        rotated_midpoint=rotated_midpoint,
        distance=float(pair_entry.distance),
        bin_idx=int(pair_entry.bin_idx),
    )


def _load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _export_keyframe_strip(*, frames_dir: Path, output_path: Path, title: str, max_frames: int) -> None:
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        return
    count = max(1, min(int(max_frames), len(frame_paths)))
    if count == 1:
        chosen = [frame_paths[-1]]
    else:
        indices = np.linspace(0, len(frame_paths) - 1, count).round().astype(int).tolist()
        chosen = [frame_paths[idx] for idx in indices]
    images = [Image.open(path).convert("RGB") for path in chosen]
    thumb_w = 240
    thumbs = []
    for img in images:
        scale = thumb_w / float(img.width)
        thumb_h = max(1, int(round(img.height * scale)))
        thumbs.append(img.resize((thumb_w, thumb_h)))
    pad = 14
    title_h = 34
    canvas_w = pad * (len(thumbs) + 1) + thumb_w * len(thumbs)
    canvas_h = title_h + pad * 2 + max(img.height for img in thumbs)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (252, 252, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), title, fill=(20, 20, 20), font=_load_font(18))
    x = pad
    y = title_h + pad
    for idx, img in enumerate(thumbs):
        canvas.paste(img, (x, y))
        draw.rectangle([x - 1, y - 1, x + img.width + 1, y + img.height + 1], outline=(35, 35, 35), width=1)
        draw.text((x + 6, y + 6), f"{idx + 1}", fill=(255, 255, 255), font=_load_font(16))
        x += thumb_w + pad
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _convex_hull_area_xy(points_xy: np.ndarray, *, max_points: int = 512) -> float:
    """2D convex hull area via Andrew monotonic chain + shoelace formula."""
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        return 0.0
    max_points = int(max_points)
    if max_points > 0 and pts.shape[0] > max_points:
        indices = np.linspace(0, pts.shape[0] - 1, num=max_points, dtype=np.int64)
        pts = pts[indices]
    # Deduplicate to keep hull construction stable.
    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = np.vstack((lower[:-1], upper[:-1]))
    if hull.shape[0] < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def _export_trusted_snapshots(
    *,
    env,
    args,
    out_dir: Path,
    asset_index: int,
    pair_idx: int,
    protocol_name: str,
    snapshots: list[ClothSnapshot],
) -> None:
    from experiments.offline_label_2x2.scripts.record_protocol_videos import (
        _apply_recording_scene_style,
        _set_fixed_camera_poses,
        _setup_headless_cameras,
        _update_camera_pose,
        _warmup_render,
    )
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    def _apply_fixed_cloth_color() -> None:
        stage = env.unwrapped.sim.stage
        garment = env.unwrapped._garment_manager
        mesh_path = "/World/Cloth/env_0/garment/mesh"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim or not mesh_prim.IsValid():
            return

        color = tuple(float(x) for x in args.export_cloth_color)
        material_path = Sdf.Path("/World/Cloth/TrustedQualitativeClothMaterial")
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        subsets = [child for child in mesh_prim.GetChildren() if child.IsA(UsdGeom.Subset)]
        if subsets:
            for subset in subsets:
                UsdShade.MaterialBindingAPI.Apply(subset).Bind(material)
        else:
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)
        setattr(garment.env_cfg, "debug_fixed_color", color)

    protocol_dir = out_dir / f"asset_{asset_index:04d}" / f"pair_{pair_idx:02d}" / protocol_name
    protocol_dir.mkdir(parents=True, exist_ok=True)

    _apply_recording_scene_style(env, ground_size=float(args.export_ground_size))
    _apply_fixed_cloth_color()
    cameras = _setup_headless_cameras(
        env,
        width=int(args.export_video_width),
        height=int(args.export_video_height),
    )

    export_cam_args = SimpleNamespace(
        top_eye=tuple(float(x) for x in args.export_top_eye),
        top_target=tuple(float(x) for x in args.export_top_target),
        side_eye=tuple(float(x) for x in args.export_side_eye),
        side_target=tuple(float(x) for x in args.export_side_target),
    )
    if bool(args.export_fixed_cameras):
        _set_fixed_camera_poses(export_cam_args, cameras)
    else:
        for view_name in ("top", "side"):
            _update_camera_pose(env, cameras[view_name]["cam_path"], view_name=view_name)
    _warmup_render(env, int(args.export_warmup_renders))

    garment = env.unwrapped._garment_manager
    env_ids_long = torch.tensor([0], device=env.unwrapped.device, dtype=torch.long)
    env.unwrapped._unfold.action_manager.stop_all_control()
    try:
        for view_name, camera_info in cameras.items():
            frames_dir = protocol_dir / view_name / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for snap_idx, snapshot in enumerate(snapshots):
                garment._set_particle_positions(snapshot.positions.to(env.unwrapped.device), env_ids_long)
                garment._set_particle_velocities(snapshot.velocities.to(env.unwrapped.device), env_ids_long)
                env.unwrapped.scene.write_data_to_sim()
                env.unwrapped.sim.render()
                rgb = camera_info["annotator"].get_data()
                frame = np.asarray(rgb, dtype=np.uint8)
                if frame.ndim == 3 and frame.shape[-1] >= 3:
                    frame = frame[..., :3]
                imageio.imwrite(frames_dir / f"frame_{snap_idx:05d}.png", frame)
            _export_keyframe_strip(
                frames_dir=frames_dir,
                output_path=protocol_dir / f"{protocol_name}_{view_name}_keyframes.png",
                title=f"{protocol_name} | {view_name}",
                max_frames=int(args.export_keyframes_per_view),
            )
    finally:
        for camera_info in cameras.values():
            try:
                camera_info["annotator"].detach()
            except Exception:
                pass
            try:
                camera_info["render_product"].destroy()
            except Exception:
                pass


def _extract_convex_hull_metrics(obs: dict, batch_n: int, *, max_points: int = 512) -> tuple[list[float], list[float]]:
    if not isinstance(obs, dict):
        return [float("nan")] * batch_n, [float("nan")] * batch_n
    pos = obs.get("pos")
    init_pos = obs.get("init_pos")
    sampling_mask = obs.get("pos_mask_sampled", obs.get("pos_mask"))
    if pos is None or init_pos is None or sampling_mask is None:
        return [float("nan")] * batch_n, [float("nan")] * batch_n

    pos_np = pos.detach().cpu().numpy()
    init_np = init_pos.detach().cpu().numpy()
    mask_np = sampling_mask.detach().cpu().numpy().squeeze(-1) > 0.5
    areas: list[float] = []
    ratios: list[float] = []
    for env_i in range(batch_n):
        mask = mask_np[env_i]
        if not np.any(mask):
            areas.append(float("nan"))
            ratios.append(float("nan"))
            continue
        current_xy = pos_np[env_i, mask, :2]
        init_xy = init_np[env_i, mask, :2]
        current_area = _convex_hull_area_xy(current_xy, max_points=max_points)
        init_area = _convex_hull_area_xy(init_xy, max_points=max_points)
        ratio = current_area / max(init_area, 1e-6)
        areas.append(float(current_area))
        ratios.append(float(ratio))
    return areas, ratios


def _existing_done_keys(path: Path) -> set[tuple[int, str, int, int]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        return {
            (
                int(row["asset_index"]),
                str(row["protocol"]),
                int(row["pair_idx"]),
                int(row["repeat_idx"]),
            )
            for row in rows
        }


def _load_existing_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_progress(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _flush_pending_outputs(
    *,
    out_dir: Path,
    metadata: dict,
    all_records: list[dict],
    pending_records: list[dict],
    done_keys: set[tuple[int, str, int, int]],
    total_samples: int | None,
    last_asset_index: int | None,
    last_protocol: str | None,
    last_batch_start: int | None = None,
    last_batch_size: int | None = None,
) -> None:
    if not pending_records:
        return
    _write_outputs(out_dir, metadata, pending_records, None, append_records=True)
    progress_payload = {
        "completed_samples": len(done_keys),
        "total_samples": total_samples,
        "last_asset_index": last_asset_index,
        "last_protocol": last_protocol,
    }
    if last_batch_start is not None:
        progress_payload["last_batch_start"] = int(last_batch_start)
    if last_batch_size is not None:
        progress_payload["last_batch_size"] = int(last_batch_size)
    _write_progress(out_dir, progress_payload)
    pending_records.clear()


def _summarize_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_deformable: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_rigid: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_real_l2: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_xy_major_dispersion: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_xy_minor_dispersion: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_xy_major_dispersion_norm: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_xy_minor_dispersion_norm: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_xy_bbox_area_ratio: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_xy_convex_hull_area: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_xy_convex_hull_area_ratio: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_z_thickness: dict[tuple[int, int, int, int, str], list[float]] = {}
    grouped_z_thickness_norm: dict[tuple[int, int, int, int, str], list[float]] = {}
    meta: dict[tuple[int, int, int, int, str], dict] = {}
    for record in records:
        key = (
            int(record["asset_index"]),
            int(record["asset_id"]),
            int(record["coord_id1"]),
            int(record["coord_id2"]),
            str(record["protocol"]),
        )
        grouped.setdefault(key, []).append(float(record["reward"]))
        grouped_deformable.setdefault(key, []).append(float(record.get("deformable_distance", "nan")))
        grouped_rigid.setdefault(key, []).append(float(record.get("rigid_distance", "nan")))
        grouped_real_l2.setdefault(key, []).append(float(record.get("real_l2_distance", "nan")))
        grouped_xy_major_dispersion.setdefault(key, []).append(float(record.get("xy_major_dispersion", "nan")))
        grouped_xy_minor_dispersion.setdefault(key, []).append(float(record.get("xy_minor_dispersion", "nan")))
        grouped_xy_major_dispersion_norm.setdefault(key, []).append(float(record.get("xy_major_dispersion_norm", "nan")))
        grouped_xy_minor_dispersion_norm.setdefault(key, []).append(float(record.get("xy_minor_dispersion_norm", "nan")))
        grouped_xy_bbox_area_ratio.setdefault(key, []).append(float(record.get("xy_bbox_area_ratio", "nan")))
        grouped_xy_convex_hull_area.setdefault(key, []).append(float(record.get("xy_convex_hull_area", "nan")))
        grouped_xy_convex_hull_area_ratio.setdefault(key, []).append(float(record.get("xy_convex_hull_area_ratio", "nan")))
        grouped_z_thickness.setdefault(key, []).append(float(record.get("z_thickness", "nan")))
        grouped_z_thickness_norm.setdefault(key, []).append(float(record.get("z_thickness_norm", "nan")))
        meta[key] = record

    summary: list[dict] = []
    for key in sorted(grouped.keys()):
        rewards = np.asarray(grouped[key], dtype=np.float32)
        deformable_vals = np.asarray(grouped_deformable[key], dtype=np.float32)
        rigid_vals = np.asarray(grouped_rigid[key], dtype=np.float32)
        real_l2_vals = np.asarray(grouped_real_l2[key], dtype=np.float32)
        xy_major_vals = np.asarray(grouped_xy_major_dispersion[key], dtype=np.float32)
        xy_minor_vals = np.asarray(grouped_xy_minor_dispersion[key], dtype=np.float32)
        xy_major_norm_vals = np.asarray(grouped_xy_major_dispersion_norm[key], dtype=np.float32)
        xy_minor_norm_vals = np.asarray(grouped_xy_minor_dispersion_norm[key], dtype=np.float32)
        xy_bbox_ratio_vals = np.asarray(grouped_xy_bbox_area_ratio[key], dtype=np.float32)
        xy_hull_area_vals = np.asarray(grouped_xy_convex_hull_area[key], dtype=np.float32)
        xy_hull_area_ratio_vals = np.asarray(grouped_xy_convex_hull_area_ratio[key], dtype=np.float32)
        z_thickness_vals = np.asarray(grouped_z_thickness[key], dtype=np.float32)
        z_thickness_norm_vals = np.asarray(grouped_z_thickness_norm[key], dtype=np.float32)
        item = meta[key]
        summary.append(
            {
                "asset_index": int(item["asset_index"]),
                "asset_id": int(item["asset_id"]),
                "protocol": str(item["protocol"]),
                "coord_id1": int(item["coord_id1"]),
                "coord_id2": int(item["coord_id2"]),
                "raw_id1": int(item["raw_id1"]),
                "raw_id2": int(item["raw_id2"]),
                "bin_idx": int(item["bin_idx"]),
                "distance": float(item["distance"]),
                "repeats": int(rewards.shape[0]),
                "mean_reward": float(rewards.mean()),
                "std_reward": float(rewards.std()),
                "best_reward": float(rewards.max()),
                "min_reward": float(rewards.min()),
                "mean_deformable_distance": float(np.nanmean(deformable_vals)),
                "std_deformable_distance": float(np.nanstd(deformable_vals)),
                "mean_rigid_distance": float(np.nanmean(rigid_vals)),
                "std_rigid_distance": float(np.nanstd(rigid_vals)),
                "mean_real_l2_distance": float(np.nanmean(real_l2_vals)),
                "std_real_l2_distance": float(np.nanstd(real_l2_vals)),
                "mean_xy_major_dispersion": float(np.nanmean(xy_major_vals)),
                "std_xy_major_dispersion": float(np.nanstd(xy_major_vals)),
                "mean_xy_minor_dispersion": float(np.nanmean(xy_minor_vals)),
                "std_xy_minor_dispersion": float(np.nanstd(xy_minor_vals)),
                "mean_xy_major_dispersion_norm": float(np.nanmean(xy_major_norm_vals)),
                "std_xy_major_dispersion_norm": float(np.nanstd(xy_major_norm_vals)),
                "mean_xy_minor_dispersion_norm": float(np.nanmean(xy_minor_norm_vals)),
                "std_xy_minor_dispersion_norm": float(np.nanstd(xy_minor_norm_vals)),
                "mean_xy_bbox_area_ratio": float(np.nanmean(xy_bbox_ratio_vals)),
                "std_xy_bbox_area_ratio": float(np.nanstd(xy_bbox_ratio_vals)),
                "mean_xy_convex_hull_area": float(np.nanmean(xy_hull_area_vals)),
                "std_xy_convex_hull_area": float(np.nanstd(xy_hull_area_vals)),
                "mean_xy_convex_hull_area_ratio": float(np.nanmean(xy_hull_area_ratio_vals)),
                "std_xy_convex_hull_area_ratio": float(np.nanstd(xy_hull_area_ratio_vals)),
                "mean_z_thickness": float(np.nanmean(z_thickness_vals)),
                "std_z_thickness": float(np.nanstd(z_thickness_vals)),
                "mean_z_thickness_norm": float(np.nanmean(z_thickness_norm_vals)),
                "std_z_thickness_norm": float(np.nanstd(z_thickness_norm_vals)),
            }
        )
    return summary


def _write_outputs(
    out_dir: Path,
    metadata: dict,
    records: list[dict],
    summary: list[dict] | None,
    *,
    append_records: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "asset_index",
        "asset_id",
        "protocol",
        "pair_idx",
        "coord_id1",
        "coord_id2",
        "raw_id1",
        "raw_id2",
        "bin_idx",
        "distance",
        "repeat_idx",
        "reward",
        "deformable_distance",
        "rigid_distance",
        "real_l2_distance",
        "xy_major_dispersion",
        "xy_minor_dispersion",
        "xy_major_dispersion_norm",
        "xy_minor_dispersion_norm",
        "xy_bbox_area_ratio",
        "xy_convex_hull_area",
        "xy_convex_hull_area_ratio",
        "z_thickness",
        "z_thickness_norm",
    ]
    records_path = out_dir / "records.csv"
    write_header = (not append_records) or (not records_path.exists()) or records_path.stat().st_size == 0
    mode = "a" if append_records else "w"
    with records_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        if records:
            writer.writerows(records)
        f.flush()

    if summary is None:
        return

    with (out_dir / "pair_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (out_dir / "protocol_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "protocol",
            "mean_reward",
            "mean_std_reward",
            "mean_best_reward",
            "mean_deformable_distance",
            "mean_std_deformable_distance",
            "mean_rigid_distance",
            "mean_std_rigid_distance",
            "mean_real_l2_distance",
            "mean_std_real_l2_distance",
            "mean_xy_major_dispersion",
            "mean_std_xy_major_dispersion",
            "mean_xy_minor_dispersion",
            "mean_std_xy_minor_dispersion",
            "mean_xy_major_dispersion_norm",
            "mean_std_xy_major_dispersion_norm",
            "mean_xy_minor_dispersion_norm",
            "mean_std_xy_minor_dispersion_norm",
            "mean_xy_bbox_area_ratio",
            "mean_std_xy_bbox_area_ratio",
            "mean_xy_convex_hull_area",
            "mean_std_xy_convex_hull_area",
            "mean_xy_convex_hull_area_ratio",
            "mean_std_xy_convex_hull_area_ratio",
            "mean_z_thickness",
            "mean_std_z_thickness",
            "mean_z_thickness_norm",
            "mean_std_z_thickness_norm",
            "num_asset_pairs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for protocol in sorted({item["protocol"] for item in summary}):
            rows = [item for item in summary if item["protocol"] == protocol]
            writer.writerow(
                {
                    "protocol": protocol,
                    "mean_reward": float(np.mean([item["mean_reward"] for item in rows])),
                    "mean_std_reward": float(np.mean([item["std_reward"] for item in rows])),
                    "mean_best_reward": float(np.mean([item["best_reward"] for item in rows])),
                    "mean_deformable_distance": float(np.nanmean([item["mean_deformable_distance"] for item in rows])),
                    "mean_std_deformable_distance": float(np.nanmean([item["std_deformable_distance"] for item in rows])),
                    "mean_rigid_distance": float(np.nanmean([item["mean_rigid_distance"] for item in rows])),
                    "mean_std_rigid_distance": float(np.nanmean([item["std_rigid_distance"] for item in rows])),
                    "mean_real_l2_distance": float(np.nanmean([item["mean_real_l2_distance"] for item in rows])),
                    "mean_std_real_l2_distance": float(np.nanmean([item["std_real_l2_distance"] for item in rows])),
                    "mean_xy_major_dispersion": float(np.nanmean([item["mean_xy_major_dispersion"] for item in rows])),
                    "mean_std_xy_major_dispersion": float(np.nanmean([item["std_xy_major_dispersion"] for item in rows])),
                    "mean_xy_minor_dispersion": float(np.nanmean([item["mean_xy_minor_dispersion"] for item in rows])),
                    "mean_std_xy_minor_dispersion": float(np.nanmean([item["std_xy_minor_dispersion"] for item in rows])),
                    "mean_xy_major_dispersion_norm": float(np.nanmean([item["mean_xy_major_dispersion_norm"] for item in rows])),
                    "mean_std_xy_major_dispersion_norm": float(np.nanmean([item["std_xy_major_dispersion_norm"] for item in rows])),
                    "mean_xy_minor_dispersion_norm": float(np.nanmean([item["mean_xy_minor_dispersion_norm"] for item in rows])),
                    "mean_std_xy_minor_dispersion_norm": float(np.nanmean([item["std_xy_minor_dispersion_norm"] for item in rows])),
                    "mean_xy_bbox_area_ratio": float(np.nanmean([item["mean_xy_bbox_area_ratio"] for item in rows])),
                    "mean_std_xy_bbox_area_ratio": float(np.nanmean([item["std_xy_bbox_area_ratio"] for item in rows])),
                    "mean_xy_convex_hull_area": float(np.nanmean([item["mean_xy_convex_hull_area"] for item in rows])),
                    "mean_std_xy_convex_hull_area": float(np.nanmean([item["std_xy_convex_hull_area"] for item in rows])),
                    "mean_xy_convex_hull_area_ratio": float(np.nanmean([item["mean_xy_convex_hull_area_ratio"] for item in rows])),
                    "mean_std_xy_convex_hull_area_ratio": float(np.nanmean([item["std_xy_convex_hull_area_ratio"] for item in rows])),
                    "mean_z_thickness": float(np.nanmean([item["mean_z_thickness"] for item in rows])),
                    "mean_std_z_thickness": float(np.nanmean([item["std_z_thickness"] for item in rows])),
                    "mean_z_thickness_norm": float(np.nanmean([item["mean_z_thickness_norm"] for item in rows])),
                    "mean_std_z_thickness_norm": float(np.nanmean([item["std_z_thickness_norm"] for item in rows])),
                    "num_asset_pairs": len(rows),
                }
            )


def run(args) -> None:
    if bool(getattr(args, "export_qualitative", False)):
        args.enable_cameras = True

    if int(getattr(args, "debug_stack_dump_seconds", 0) or 0) > 0:
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
        faulthandler.dump_traceback_later(
            int(args.debug_stack_dump_seconds),
            repeat=True,
            file=sys.stderr,
            exit=False,
        )

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import carb
    import omni.kit.app
    import unfold  # noqa: F401
    from unfold.workflows.offline_collection.pair_conditioned_collect import (
        PairConditionedOfflineCollector,
        load_pair_conditioned_env_and_cfg,
    )

    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    carb.log_warn = lambda *a, **k: None
    if bool(getattr(args, "export_qualitative", False)):
        try:
            manager = omni.kit.app.get_app().get_extension_manager()
            manager.set_extension_enabled_immediate("isaacsim.core.experimental.materials", True)
        except Exception as exc:
            print(f"[trusted-export] materials extension enable skipped: {exc}", flush=True)

    env, env_cfg = load_pair_conditioned_env_and_cfg(args)
    collector = PairConditionedOfflineCollector(env, env_cfg, args)
    fixed_pairs = _load_pairs_manifest(args.pairs_manifest)
    if fixed_pairs:
        asset_indices = sorted(fixed_pairs.keys())
    else:
        asset_indices = _parse_asset_indices(args.asset_indices)
    protocols = _selected_protocols(args.protocol)
    rot_noise = (float(args.rot_noise_deg),) * 3
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite and not args.resume:
        raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        records_path = out_dir / "records.csv"
        done_keys = _existing_done_keys(records_path) if args.resume else set()
        all_records: list[dict] = _load_existing_records(records_path) if args.resume else []
        pending_records: list[dict] = []
        batches_since_flush = 0
        flush_every_batches = max(1, int(getattr(args, "flush_every_batches", 8)))
        num_envs = int(env_cfg.scene.num_envs)
        base_seed = int(getattr(env_cfg, "seed", 42) or 42)
        total_samples = 0
        for asset_index in asset_indices:
            total_samples += len(protocols) * len(fixed_pairs[asset_index] if fixed_pairs else []) * int(args.repeats_per_pair)
        metadata = {
            "asset_indices": asset_indices,
            "asset_ids": [int(collector._asset_ids[idx]) for idx in asset_indices],
            "protocols": [spec.name for spec in protocols],
            "num_pairs": int(args.num_pairs),
            "repeats_per_pair": int(args.repeats_per_pair),
            "num_envs": int(env_cfg.scene.num_envs),
            "rot_noise_deg": float(args.rot_noise_deg),
            "resume_enabled": bool(args.resume),
            "total_samples": int(total_samples) if fixed_pairs else None,
        }
        _write_outputs(out_dir, metadata, [], None, append_records=args.resume)
        _write_progress(
            out_dir,
            {
                "completed_samples": len(done_keys),
                "total_samples": metadata["total_samples"],
                "last_asset_index": None,
                "last_protocol": None,
            },
        )
        for asset_index in asset_indices:
            if asset_index < 0 or asset_index >= len(collector._asset_paths):
                raise ValueError(f"asset index out of range: {asset_index}")
            if fixed_pairs and args.resume and _fixed_asset_samples_complete(
                asset_index=asset_index,
                protocols=protocols,
                pairs=fixed_pairs[asset_index],
                repeats_per_pair=int(args.repeats_per_pair),
                done_keys=done_keys,
            ):
                print(f"[RESUME] skip completed asset_index={asset_index}", flush=True)
                continue

            _debug_trace(args, f"asset_reset_start asset={asset_index}")
            collector._reset_single_asset(asset_index, asset_index)
            _debug_trace(args, f"asset_reset_done asset={asset_index}")
            _debug_trace(args, f"pointcloud_start asset={asset_index}")
            pointcloud = collector._prepare_pointcloud(0)
            _debug_trace(
                args,
                f"pointcloud_done asset={asset_index} coord_count={pointcloud.coord.shape[0]} "
                f"raw_count={pointcloud.raw_coord.shape[0]}",
            )
            collector._apply_coord_reward_sampling_mask(pointcloud)
            if fixed_pairs:
                chosen_pairs = fixed_pairs[asset_index]
            else:
                bank = collector._build_pair_bank(0)
                chosen_pairs = bank.pop_distinct(int(args.num_pairs))
                if len(chosen_pairs) < int(args.num_pairs):
                    raise RuntimeError(
                        f"Requested {args.num_pairs} pairs for asset_index={asset_index}, got {len(chosen_pairs)}."
                    )

            asset_id = int(collector._asset_ids[asset_index])
            for spec in protocols:
                _debug_trace(args, f"protocol_start asset={asset_index} protocol={spec.name}")
                _configure_protocol(env.unwrapped, env_cfg, spec, args)
                protocol_samples = []
                for pair_idx, candidate in enumerate(chosen_pairs):
                    for repeat_idx in range(int(args.repeats_per_pair)):
                        key = (asset_index, spec.name, pair_idx, repeat_idx)
                        if key in done_keys:
                            continue
                        protocol_samples.append((pair_idx, repeat_idx, candidate))
                _debug_trace(
                    args,
                    f"protocol_samples asset={asset_index} protocol={spec.name} remaining={len(protocol_samples)}",
                )

                sample_cursor = 0
                while sample_cursor < len(protocol_samples):
                    _debug_trace(
                        args,
                        f"batch_loop_enter asset={asset_index} protocol={spec.name} sample_cursor={sample_cursor}",
                    )
                    batch = protocol_samples[sample_cursor : sample_cursor + num_envs]
                    batch_n = len(batch)
                    actions = torch.full((num_envs, 2), -1, dtype=torch.long, device=collector.device)
                    init_candidates = []
                    batch_records: list[dict] = []
                    first_pair_idx = int(batch[0][0])
                    first_repeat_idx = int(batch[0][1])
                    for env_i, (pair_idx, repeat_idx, candidate) in enumerate(batch):
                        if isinstance(candidate, FixedPair):
                            _debug_trace(
                                args,
                                f"fixed_pair_rebuild_start asset={asset_index} protocol={spec.name} pair_idx={pair_idx}",
                            )
                            built = _rebuild_pair_candidate_inline(
                                pointcloud,
                                candidate,
                                device=collector.device,
                            )
                            _debug_trace(
                                args,
                                f"fixed_pair_rebuild_done asset={asset_index} protocol={spec.name} pair_idx={pair_idx} ok={built is not None}",
                            )
                            if built is None:
                                raise RuntimeError(
                                    f"Failed to rebuild pair-conditioned pose for asset_index={asset_index} pair={pair_idx}"
                                )
                            init_candidate = built
                        else:
                            init_candidate = candidate
                        init_candidates.append(init_candidate)
                        actions[env_i, 0] = int(init_candidate.raw_id1)
                        actions[env_i, 1] = int(init_candidate.raw_id2)
                        batch_records.append(
                            {
                                "asset_index": asset_index,
                                "asset_id": asset_id,
                                "protocol": spec.name,
                                "pair_idx": int(pair_idx),
                                "coord_id1": int(candidate.coord_id1),
                                "coord_id2": int(candidate.coord_id2),
                                "raw_id1": int(init_candidate.raw_id1),
                                "raw_id2": int(init_candidate.raw_id2),
                                "bin_idx": int(candidate.bin_idx),
                                "distance": float(candidate.distance),
                                "repeat_idx": int(repeat_idx),
                            }
                        )

                    if batch_n < num_envs and batch_records:
                        pad_record = batch_records[-1]
                        pad_candidate = init_candidates[-1]
                        for env_i in range(batch_n, num_envs):
                            init_candidates.append(pad_candidate)
                            actions[env_i, 0] = int(pad_candidate.raw_id1)
                            actions[env_i, 1] = int(pad_candidate.raw_id2)

                    random_reset_seed = None
                    if spec.init_mode == "random":
                        random_reset_seed = _repeat_random_seed(
                            base_seed=base_seed,
                            asset_index=asset_index,
                            pair_idx=first_pair_idx,
                            repeat_start=first_repeat_idx,
                        )
                    if args.debug_loop_trace:
                        print(
                            f"[LOOP] begin asset={asset_index} protocol={spec.name} "
                            f"batch_start={sample_cursor} batch_n={batch_n} seed={random_reset_seed}",
                            flush=True,
                        )
                    _debug_trace(
                        args,
                        f"apply_init_start asset={asset_index} protocol={spec.name} "
                        f"batch_start={sample_cursor} batch_n={batch_n}",
                    )
                    _apply_init(
                        collector,
                        spec=spec,
                        candidates=init_candidates,
                        rot_noise_deg=rot_noise,
                        random_reset_seed=random_reset_seed,
                    )
                    _debug_trace(
                        args,
                        f"apply_init_done asset={asset_index} protocol={spec.name} batch_start={sample_cursor}",
                    )
                    _debug_trace(
                        args,
                        f"env_step_start asset={asset_index} protocol={spec.name} batch_start={sample_cursor}",
                    )
                    state_recorder = None
                    if bool(getattr(args, "export_qualitative", False)) and batch_n == 1:
                        state_recorder = StateSnapshotRecorder(
                            env,
                            sample_hz=float(args.export_sample_fps),
                            physics_dt=float(env.unwrapped.physics_dt),
                        )
                        env.unwrapped._unfold.frame_recorder = state_recorder
                    obs, rewards, _, _, extras = env.unwrapped.step(actions)
                    if state_recorder is not None:
                        env.unwrapped._unfold.frame_recorder = None
                        state_recorder.append_final(label="final")
                    _debug_trace(
                        args,
                        f"env_step_done asset={asset_index} protocol={spec.name} batch_start={sample_cursor}",
                    )
                    reward_list = rewards.detach().cpu().view(-1).tolist()
                    deformable_list = _extract_component_list(extras, "deformable_distance", batch_n)
                    rigid_list = _extract_component_list(extras, "rigid_distance", batch_n)
                    real_l2_list = _extract_component_list(extras, "real_l2_distance", batch_n)
                    xy_major_list = _extract_component_list(extras, "xy_major_dispersion", batch_n)
                    xy_minor_list = _extract_component_list(extras, "xy_minor_dispersion", batch_n)
                    xy_major_norm_list = _extract_component_list(extras, "xy_major_dispersion_norm", batch_n)
                    xy_minor_norm_list = _extract_component_list(extras, "xy_minor_dispersion_norm", batch_n)
                    xy_bbox_area_ratio_list = _extract_component_list(extras, "xy_bbox_area_ratio", batch_n)
                    _debug_trace(
                        args,
                        f"convex_hull_start asset={asset_index} protocol={spec.name} "
                        f"batch_start={sample_cursor} max_points={int(args.convex_hull_max_points)}",
                    )
                    xy_convex_hull_area_list, xy_convex_hull_area_ratio_list = _extract_convex_hull_metrics(
                        obs,
                        batch_n,
                        max_points=int(args.convex_hull_max_points),
                    )
                    _debug_trace(
                        args,
                        f"convex_hull_done asset={asset_index} protocol={spec.name} batch_start={sample_cursor}",
                    )
                    z_thickness_list = _extract_component_list(extras, "z_thickness", batch_n)
                    z_thickness_norm_list = _extract_component_list(extras, "z_thickness_norm", batch_n)
                    for env_i in range(batch_n):
                        batch_records[env_i]["reward"] = float(reward_list[env_i])
                        batch_records[env_i]["deformable_distance"] = float(deformable_list[env_i])
                        batch_records[env_i]["rigid_distance"] = float(rigid_list[env_i])
                        batch_records[env_i]["real_l2_distance"] = float(real_l2_list[env_i])
                        batch_records[env_i]["xy_major_dispersion"] = float(xy_major_list[env_i])
                        batch_records[env_i]["xy_minor_dispersion"] = float(xy_minor_list[env_i])
                        batch_records[env_i]["xy_major_dispersion_norm"] = float(xy_major_norm_list[env_i])
                        batch_records[env_i]["xy_minor_dispersion_norm"] = float(xy_minor_norm_list[env_i])
                        batch_records[env_i]["xy_bbox_area_ratio"] = float(xy_bbox_area_ratio_list[env_i])
                        batch_records[env_i]["xy_convex_hull_area"] = float(xy_convex_hull_area_list[env_i])
                        batch_records[env_i]["xy_convex_hull_area_ratio"] = float(xy_convex_hull_area_ratio_list[env_i])
                        batch_records[env_i]["z_thickness"] = float(z_thickness_list[env_i])
                        batch_records[env_i]["z_thickness_norm"] = float(z_thickness_norm_list[env_i])
                        done_keys.add(
                            (
                                int(batch_records[env_i]["asset_index"]),
                                str(batch_records[env_i]["protocol"]),
                                int(batch_records[env_i]["pair_idx"]),
                                int(batch_records[env_i]["repeat_idx"]),
                            )
                        )
                    if state_recorder is not None:
                        export_dir = Path(args.export_dir) if args.export_dir else (out_dir / "qualitative")
                        _export_trusted_snapshots(
                            env=env,
                            args=args,
                            out_dir=export_dir,
                            asset_index=int(asset_index),
                            pair_idx=int(batch_records[0]["pair_idx"]),
                            protocol_name=str(spec.name),
                            snapshots=state_recorder.snapshots,
                        )
                    del obs
                    all_records.extend(batch_records)
                    pending_records.extend(batch_records)
                    batches_since_flush += 1
                    if batches_since_flush >= flush_every_batches:
                        _debug_trace(
                            args,
                            f"flush_start asset={asset_index} protocol={spec.name} batch_start={sample_cursor}",
                        )
                        _flush_pending_outputs(
                            out_dir=out_dir,
                            metadata=metadata,
                            all_records=all_records,
                            pending_records=pending_records,
                            done_keys=done_keys,
                            total_samples=metadata["total_samples"],
                            last_asset_index=int(asset_index),
                            last_protocol=str(spec.name),
                            last_batch_start=int(sample_cursor),
                            last_batch_size=int(batch_n),
                        )
                        _debug_trace(
                            args,
                            f"flush_done asset={asset_index} protocol={spec.name} batch_start={sample_cursor}",
                        )
                        batches_since_flush = 0
                    sample_cursor += batch_n

                _flush_pending_outputs(
                    out_dir=out_dir,
                    metadata=metadata,
                    all_records=all_records,
                    pending_records=pending_records,
                    done_keys=done_keys,
                    total_samples=metadata["total_samples"],
                    last_asset_index=int(asset_index),
                    last_protocol=str(spec.name),
                )
                batches_since_flush = 0
                print(
                    f"[PROTO] asset_index={asset_index} asset_id={asset_id} "
                    f"protocol={spec.name} completed_samples={len(protocol_samples)}",
                    flush=True,
                )

        _flush_pending_outputs(
            out_dir=out_dir,
            metadata=metadata,
            all_records=all_records,
            pending_records=pending_records,
            done_keys=done_keys,
            total_samples=metadata["total_samples"],
            last_asset_index=None,
            last_protocol=None,
        )
        summary = _summarize_records(all_records)
        _write_outputs(out_dir, metadata, [], summary, append_records=True)
        print(f"[OK] Saved outputs to {out_dir}", flush=True)
    finally:
        env.close()
        simulation_app.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
