#!/usr/bin/env python3
"""Record headless protocol videos for the offline-label 2x2 protocols."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

from experiments.offline_label_2x2.scripts.run_protocol_repeatability import (
    PROTOCOLS,
    FixedPair,
    _apply_init,
    _build_full_actions,
    _build_full_actions_from_raw_ids,
    _configure_protocol,
    _load_pairs_manifest,
    _parse_asset_indices,
    _selected_protocols,
)
from tools.offline_collection.export_parallel_snapshot import _apply_snapshot_scene_style


@dataclass(frozen=True)
class RecordedPairCandidate:
    coord_id1: int
    coord_id2: int
    raw_id1: int
    raw_id2: int
    quat_wxyz: object
    rotated_midpoint: object
    distance: float
    bin_idx: int


class FrameRecorder:
    def __init__(
        self,
        annotator,
        *,
        frames_dir: Path,
        video_path: Path,
        fps: int,
        frame_stride: int,
        image_frame_stride: int,
    ):
        self.annotator = annotator
        self.frames_dir = frames_dir
        self.video_path = video_path
        self.fps = int(fps)
        self.frame_stride = max(1, int(frame_stride))
        self.image_frame_stride = max(1, int(image_frame_stride))
        self.frame_idx = 0
        self.saved_frames = 0
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(video_path), fps=self.fps) if video_path is not None else None
        self.render_every_step = self._writer is not None

    def capture(self, *, phase_type: str, step_idx: int) -> None:
        del phase_type
        if (self.frame_idx % self.frame_stride) != 0:
            self.frame_idx += 1
            return
        rgb = self.annotator.get_data()
        if rgb is None:
            self.frame_idx += 1
            return
        frame = np.asarray(rgb)
        if frame.ndim != 3:
            self.frame_idx += 1
            return
        if frame.shape[-1] >= 3:
            frame = frame[..., :3]
        frame = np.asarray(frame, dtype=np.uint8)
        if self._writer is not None:
            self._writer.append_data(frame)
        if (self.frame_idx % self.image_frame_stride) == 0:
            frame_path = self.frames_dir / f"frame_{self.saved_frames:05d}.png"
            imageio.imwrite(frame_path, frame)
            self.saved_frames += 1
        self.frame_idx += 1

    def wants_capture(self, *, phase_type: str, step_idx: int) -> bool:
        del phase_type
        if self._writer is not None:
            return True
        return (self.frame_idx % self.image_frame_stride) == 0

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


class MultiFrameRecorder:
    def __init__(self, recorders: list[FrameRecorder]):
        self.recorders = list(recorders)

    def capture(self, *, phase_type: str, step_idx: int) -> None:
        for recorder in self.recorders:
            recorder.capture(phase_type=phase_type, step_idx=step_idx)

    def close(self) -> None:
        for recorder in self.recorders:
            recorder.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record headless mp4 videos for the offline-label 2x2 protocols.")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0", help="Gym task id.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/offline_label_2x2/configs/offline_label_2x2.yaml",
        help="YAML config path.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Recording should use a single environment.")
    parser.add_argument(
        "--protocol",
        type=str,
        default="all",
        choices=["all", *sorted(PROTOCOLS.keys())],
        help="Protocol to record. Use 'all' to record the full 2x2 set.",
    )
    parser.add_argument("--asset-indices", type=str, default="0", help="Single 0-based asset index.")
    parser.add_argument("--num-pairs", type=int, default=4, help="Number of candidate pairs sampled before selecting pair-index.")
    parser.add_argument("--pair-index", type=int, default=0, help="Index within the sampled candidate list to record.")
    parser.add_argument("--pairs-manifest", type=str, default=None, help="Optional fixed evaluation-pairs manifest.")
    parser.add_argument("--rot-noise-deg", type=float, default=0.0, help="Optional conditioned-init Euler noise range in degrees per axis.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/offline_label_2x2/runs/recordings",
        help="Directory where mp4 videos and metadata are saved.",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="experiments/offline_label_2x2/runs/recordings/visuals",
        help="Collector debug visualization directory required by the existing collector interface.",
    )
    parser.add_argument("--assets-manifest", type=str, default=None, help="Optional asset manifest JSON overriding valid_assets.json.")
    parser.add_argument("--relift-height-min", type=float, default=0.8, help="Random init relift minimum height.")
    parser.add_argument("--relift-height-max", type=float, default=1.2, help="Random init relift maximum height.")
    parser.add_argument("--relift-xy-jitter", type=float, default=0.05, help="Random init relift xy jitter magnitude.")
    parser.add_argument("--video-width", type=int, default=1024, help="Render product width.")
    parser.add_argument("--video-height", type=int, default=1024, help="Render product height.")
    parser.add_argument("--video-fps", type=int, default=30, help="Encoded mp4 frame rate.")
    parser.add_argument("--disable-video", action="store_true", help="Do not encode mp4; export sparse PNG frames only.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Save every Nth rendered frame.")
    parser.add_argument(
        "--image-save-fps",
        type=float,
        default=1.0,
        help="Approximate PNG keyframe export rate in Hz. Videos still encode at --video-fps.",
    )
    parser.add_argument("--warmup-renders", type=int, default=4, help="Render warmup frames before recording.")
    parser.add_argument("--enable-textures", action="store_true", default=True, help="Prefer asset textures for recording renders.")
    parser.add_argument("--disable-textures", dest="enable_textures", action="store_false", help="Disable asset textures and fall back to debug materials.")
    parser.add_argument("--fixed-cameras", action="store_true", default=True, help="Use fixed top/side cameras instead of recentering on cloth every protocol.")
    parser.add_argument("--disable-fixed-cameras", dest="fixed_cameras", action="store_false", help="Recenter top/side cameras from cloth observations.")
    parser.add_argument("--ground-size", type=float, default=8.0, help="Target square ground size in meters for qualitative renders.")
    parser.add_argument("--top-eye", type=float, nargs=3, default=(0.0, 0.55, 2.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--top-target", type=float, nargs=3, default=(0.0, 0.55, 0.12), metavar=("X", "Y", "Z"))
    parser.add_argument("--side-eye", type=float, nargs=3, default=(2.0, 0.55, 2.45), metavar=("X", "Y", "Z"))
    parser.add_argument("--side-target", type=float, nargs=3, default=(0.0, 0.55, 0.88), metavar=("X", "Y", "Z"))
    parser.add_argument("--keyframes-per-view", type=int, default=5, help="Export this many evenly spaced keyframes per recorded view.")
    parser.add_argument("--debug-protocol-trace", action="store_true", help="Print key protocol-stage height/pose summaries.")
    parser.add_argument("--debug-stretch-trace", action="store_true", help="Print stretch termination reasons.")
    parser.add_argument("--debug-loop-trace", action="store_true", help="Print runner loop progress around pair rebuild/init/step.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it already contains files.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


VIEW_SPECS = {
    "side": {},
    "top": {},
}


def _setup_headless_cameras(env, *, width: int, height: int):
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom

    stage = env.unwrapped.sim.stage
    cameras = {}
    for view_name in VIEW_SPECS:
        cam_path = f"/World/RecordCamera_{view_name}"
        cam_geom = UsdGeom.Camera.Define(stage, cam_path)
        focal_length = 18.0 if view_name == "side" else 18.0
        cam_geom.CreateFocalLengthAttr().Set(float(focal_length))
        cam_geom.CreateFocusDistanceAttr().Set(400.0)
        cam_geom.CreateFStopAttr().Set(0.0)
        cam_geom.CreateHorizontalApertureAttr().Set(20.955)
        cam_geom.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))
        render_product = rep.create.render_product(cam_path, resolution=(int(width), int(height)))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(render_product)
        cameras[view_name] = {
            "cam_path": cam_path,
            "render_product": render_product,
            "annotator": annotator,
        }
    return cameras


def _set_fixed_camera_poses(args, cameras) -> None:
    from unfold.platform.camera import set_camera_prims_look_at

    fixed_specs = {
        "top": (np.asarray(args.top_eye, dtype=np.float64), np.asarray(args.top_target, dtype=np.float64)),
        "side": (np.asarray(args.side_eye, dtype=np.float64), np.asarray(args.side_target, dtype=np.float64)),
    }
    for view_name, (eye, target) in fixed_specs.items():
        set_camera_prims_look_at([cameras[view_name]["cam_path"]], np.asarray([eye]), np.asarray([target]))


def _update_camera_pose(env, cam_path: str, *, view_name: str) -> None:
    from unfold.platform.camera import compute_centers_world, set_camera_prims_look_at

    obs = env.unwrapped._get_observations()
    centers_world = compute_centers_world(obs["pos"], obs["pos_mask"], env.unwrapped.scene.env_origins)
    center = centers_world[0].detach().cpu().numpy()
    pos = obs["pos"][0].detach().cpu().numpy()
    mask = obs["pos_mask"][0, :, 0].detach().cpu().numpy() > 0.5
    env_origin = env.unwrapped.scene.env_origins[0].detach().cpu().numpy()
    world_pos = pos[mask] + env_origin[None, :]
    if world_pos.shape[0] == 0:
        world_pos = center[None, :]

    bbox_min = world_pos.min(axis=0)
    bbox_max = world_pos.max(axis=0)
    bbox_extent = np.maximum(bbox_max - bbox_min, 1e-3)

    if view_name == "side":
        target = np.asarray(
            [
                center[0],
                center[1],
                bbox_min[2] + 0.22 * bbox_extent[2],
            ],
            dtype=np.float64,
        )
        eye = np.asarray(
            [
                bbox_max[0] + max(4.8, 3.0 * bbox_extent[1], 2.4 * bbox_extent[2]),
                center[1] + 0.28 * bbox_extent[1],
                bbox_max[2] + max(1.5, 1.8 * bbox_extent[2], 1.2 * bbox_extent[1]),
            ],
            dtype=np.float64,
        )
    else:
        target = np.asarray(
            [
                center[0],
                center[1],
                bbox_min[2] + 0.15 * bbox_extent[2],
            ],
            dtype=np.float64,
        )
        eye = np.asarray(
            [
                center[0],
                center[1] + 0.05 * bbox_extent[1],
                bbox_max[2] + max(4.5, 3.0 * max(bbox_extent[0], bbox_extent[1])),
            ],
            dtype=np.float64,
        )

    set_camera_prims_look_at([cam_path], np.asarray([eye]), np.asarray([target]))


def _warmup_render(env, count: int) -> None:
    for _ in range(max(0, int(count))):
        env.unwrapped.sim.render()


def _apply_recording_scene_style(env, *, ground_size: float) -> None:
    stage = env.unwrapped.sim.stage
    env_cfg = env.unwrapped.cfg
    side = max(4.0, float(ground_size))
    env_cfg.ground_size_m = [side, side]
    env.unwrapped._set_ground_size("/World/Environment/Ground")
    ground_visual = stage.GetPrimAtPath("/World/Environment/GroundVisual")
    if ground_visual and ground_visual.IsValid():
        stage.RemovePrim("/World/Environment/GroundVisual")
    env.unwrapped._create_ground_visual_plane("/World/Environment/GroundVisual")
    _apply_snapshot_scene_style(env)


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


def _safe_normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return vec / norm


def _build_candidate_from_raw_ids(collector, pointcloud, pair_entry: FixedPair) -> RecordedPairCandidate:
    from unfold.platform.camera import rotmat_to_quat_wxyz
    import torch

    raw_id1 = int(pair_entry.raw_id1)
    raw_id2 = int(pair_entry.raw_id2)
    if raw_id1 < 0 or raw_id1 >= int(pointcloud.raw_coord.shape[0]):
        raise IndexError(f"raw_id1 out of range: {raw_id1}")
    if raw_id2 < 0 or raw_id2 >= int(pointcloud.raw_coord.shape[0]):
        raise IndexError(f"raw_id2 out of range: {raw_id2}")

    p1 = pointcloud.raw_coord[raw_id1]
    p2 = pointcloud.raw_coord[raw_id2]
    midpoint = 0.5 * (p1 + p2)
    pair_dir = _safe_normalize(p2 - p1)
    if pair_dir is None:
        raise RuntimeError(f"Degenerate raw-id pair: ({raw_id1}, {raw_id2})")

    rel = pointcloud.raw_coord - midpoint[None, :]
    proj = rel - np.outer(rel @ pair_dir, pair_dir)
    centroid = proj.mean(axis=0)
    centroid = centroid - float(np.dot(centroid, pair_dir)) * pair_dir
    body_dir = _safe_normalize(centroid)
    if body_dir is None:
        raise RuntimeError(f"Failed to build body axis for raw-id pair: ({raw_id1}, {raw_id2})")
    normal = _safe_normalize(np.cross(pair_dir, body_dir))
    if normal is None:
        raise RuntimeError(f"Failed to build normal axis for raw-id pair: ({raw_id1}, {raw_id2})")
    body_dir = _safe_normalize(np.cross(normal, pair_dir))
    if body_dir is None:
        raise RuntimeError(f"Failed to finalize body axis for raw-id pair: ({raw_id1}, {raw_id2})")

    rotation = np.stack([pair_dir, body_dir, normal], axis=1).astype(np.float32, copy=False)
    rotation_tensor = torch.from_numpy(rotation.T.copy()).to(device=collector.device, dtype=torch.float32)
    quat = rotmat_to_quat_wxyz(rotation_tensor.unsqueeze(0))[0]
    midpoint_t = torch.as_tensor(midpoint, device=collector.device, dtype=torch.float32)
    rotated_midpoint = rotation_tensor @ midpoint_t
    return RecordedPairCandidate(
        coord_id1=int(pair_entry.coord_id1),
        coord_id2=int(pair_entry.coord_id2),
        raw_id1=raw_id1,
        raw_id2=raw_id2,
        quat_wxyz=quat,
        rotated_midpoint=rotated_midpoint,
        distance=float(pair_entry.distance),
        bin_idx=int(pair_entry.bin_idx),
    )


def run(args) -> None:
    from unfold.workflows.offline_collection.pair_conditioned_collect import (
        PairConditionedOfflineCollector,
        load_pair_conditioned_env_and_cfg,
    )

    args.enable_cameras = True
    if not hasattr(args, "debug_protocol_trace"):
        args.debug_protocol_trace = False
    if not hasattr(args, "debug_stretch_trace"):
        args.debug_stretch_trace = False
    if not hasattr(args, "debug_loop_trace"):
        args.debug_loop_trace = False
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import carb
    import omni.kit.app
    import unfold  # noqa: F401

    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    carb.log_warn = lambda *a, **k: None
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.core.experimental.materials", True)

    env, env_cfg = load_pair_conditioned_env_and_cfg(args)
    env_cfg.scene.num_envs = 1
    env_cfg.num_envs = 1
    collector = PairConditionedOfflineCollector(env, env_cfg, args)
    asset_indices = _parse_asset_indices(args.asset_indices)
    if len(asset_indices) != 1:
        raise ValueError("Recording expects exactly one asset index.")
    protocols = _selected_protocols(args.protocol)
    rot_noise = (float(args.rot_noise_deg),) * 3
    out_dir = Path(args.output_dir)
    fixed_pairs = _load_pairs_manifest(args.pairs_manifest)

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = None
    recorder = None
    try:
        asset_index = int(asset_indices[0])
        if asset_index < 0 or asset_index >= len(collector._asset_paths):
            raise ValueError(f"asset index out of range: {asset_index}")

        collector._reset_single_asset(asset_index, asset_index)
        pointcloud = collector._prepare_pointcloud(0)
        collector._apply_coord_reward_sampling_mask(pointcloud)
        pair_index = int(args.pair_index)
        print(f"[DEBUG] prepared asset_index={asset_index} pair_index={pair_index}", flush=True)
        if fixed_pairs:
            print("[DEBUG] entering fixed_pairs branch", flush=True)
            chosen_pairs = fixed_pairs.get(asset_index, [])
            print(f"[DEBUG] fixed_pairs count for asset_index={asset_index}: {len(chosen_pairs)}", flush=True)
            if pair_index < 0 or pair_index >= len(chosen_pairs):
                raise ValueError(f"pair-index out of range: {pair_index}, only {len(chosen_pairs)} fixed pairs available.")
            pair_entry = chosen_pairs[pair_index]
            print(f"[DEBUG] fixed pair entry type={type(pair_entry).__name__}", flush=True)
            if not isinstance(pair_entry, FixedPair):
                raise TypeError(f"Unexpected fixed pair type: {type(pair_entry).__name__}")
            print("[DEBUG] rebuilding fixed pair candidate via coord ids", flush=True)
            candidate = collector._build_pair_candidate(
                pointcloud=pointcloud,
                coord_id1=int(pair_entry.coord_id1),
                coord_id2=int(pair_entry.coord_id2),
                distance=float(pair_entry.distance),
                bin_idx=int(pair_entry.bin_idx),
            )
            if candidate is None:
                raise RuntimeError(
                    f"Failed to rebuild pair-conditioned pose for asset_index={asset_index} pair={pair_index}"
                )
            print(
                f"[DEBUG] rebuilt fixed pair asset_index={asset_index} pair_index={pair_index} "
                f"coord=({candidate.coord_id1},{candidate.coord_id2}) raw=({candidate.raw_id1},{candidate.raw_id2})",
                flush=True,
            )
        else:
            bank = collector._build_pair_bank(0)
            chosen_pairs = bank.pop_distinct(int(args.num_pairs))
            if pair_index < 0 or pair_index >= len(chosen_pairs):
                raise ValueError(f"pair-index out of range: {pair_index}, only {len(chosen_pairs)} pairs sampled.")
            candidate = chosen_pairs[pair_index]
            print(
                f"[DEBUG] sampled pair asset_index={asset_index} pair_index={pair_index} "
                f"coord=({candidate.coord_id1},{candidate.coord_id2}) raw=({candidate.raw_id1},{candidate.raw_id2})",
                flush=True,
            )

        video_dir = out_dir / f"asset_{asset_index:04d}" / f"pair_{pair_index:02d}"
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] video_dir={video_dir}", flush=True)

        captures = []
        if fixed_pairs:
            actions = _build_full_actions_from_raw_ids(
                collector.device,
                int(env_cfg.scene.num_envs),
                int(candidate.raw_id1),
                int(candidate.raw_id2),
            )
        else:
            actions = _build_full_actions(collector.device, int(env_cfg.scene.num_envs), candidate)
        print("[DEBUG] actions ready", flush=True)
        for spec in protocols:
            print(f"[DEBUG] begin protocol={spec.name}", flush=True)
            print(f"[DEBUG] configure protocol={spec.name}", flush=True)
            _configure_protocol(env.unwrapped, env_cfg, spec, args)
            print(f"[DEBUG] apply init protocol={spec.name}", flush=True)
            _apply_init(collector, spec=spec, candidates=[candidate], rot_noise_deg=rot_noise)
            if bool(args.enable_textures):
                env_cfg.enable_textures = True
                env.unwrapped.cfg.enable_textures = True
                try:
                    env.unwrapped._garment_manager._apply_preview_materials()
                except Exception as exc:
                    print(f"[WARN] failed to apply preview textures: {exc}", flush=True)
            _apply_recording_scene_style(env, ground_size=float(args.ground_size))
            if cameras is None:
                print("[DEBUG] setting up headless cameras", flush=True)
                cameras = _setup_headless_cameras(
                    env,
                    width=int(args.video_width),
                    height=int(args.video_height),
                )
                print("[DEBUG] cameras ready", flush=True)
            print(f"[DEBUG] update cameras protocol={spec.name}", flush=True)
            if bool(args.fixed_cameras):
                _set_fixed_camera_poses(args, cameras)
            else:
                for view_name in VIEW_SPECS:
                    _update_camera_pose(
                        env,
                        cameras[view_name]["cam_path"],
                        view_name=view_name,
                    )
            print(f"[DEBUG] warmup render protocol={spec.name}", flush=True)
            _warmup_render(env, int(args.warmup_renders))

            protocol_dir = video_dir / spec.name
            protocol_dir.mkdir(parents=True, exist_ok=True)
            single_recorders = []
            image_frame_stride = max(1, int(round(float(args.video_fps) / max(float(args.image_save_fps), 1e-6))))
            for view_name, camera_info in cameras.items():
                view_dir = protocol_dir / view_name
                frames_dir = view_dir / "frames"
                view_dir.mkdir(parents=True, exist_ok=True)
                video_path = None if bool(args.disable_video) else (view_dir / f"{spec.name}_{view_name}.mp4")
                single_recorders.append(
                    FrameRecorder(
                        camera_info["annotator"],
                        frames_dir=frames_dir,
                        video_path=video_path,
                        fps=int(args.video_fps),
                        frame_stride=int(args.frame_stride),
                        image_frame_stride=image_frame_stride,
                    )
                )
            recorder = MultiFrameRecorder(single_recorders)
            env.unwrapped._unfold.frame_recorder = recorder
            print(
                f"[RECORD] asset_index={asset_index} protocol={spec.name} "
                f"coord=({candidate.coord_id1},{candidate.coord_id2})",
                flush=True,
            )
            env.unwrapped.step(actions)
            env.unwrapped._unfold.frame_recorder = None
            recorder.close()
            for view_name in VIEW_SPECS:
                frames_dir = protocol_dir / view_name / "frames"
                _export_keyframe_strip(
                    frames_dir=frames_dir,
                    output_path=protocol_dir / f"{spec.name}_{view_name}_keyframes.png",
                    title=f"{spec.name} | {view_name}",
                    max_frames=int(args.keyframes_per_view),
                )
            captures.append(
                {
                    "protocol": spec.name,
                    "views": {
                        view_name: {
                            "video": str(protocol_dir / view_name / f"{spec.name}_{view_name}.mp4"),
                            "num_frames": int(single_recorders[idx].saved_frames),
                            "keyframes": str(protocol_dir / f"{spec.name}_{view_name}_keyframes.png"),
                        }
                        for idx, view_name in enumerate(VIEW_SPECS.keys())
                    },
                    "frame_stride": int(args.frame_stride),
                    "image_save_fps": float(args.image_save_fps),
                    "image_frame_stride": int(image_frame_stride),
                }
            )
            recorder = None

        metadata = {
            "asset_index": asset_index,
            "asset_id": int(collector._asset_ids[asset_index]),
            "pair_index": pair_index,
            "coord_id1": int(candidate.coord_id1),
            "coord_id2": int(candidate.coord_id2),
            "raw_id1": int(candidate.raw_id1),
            "raw_id2": int(candidate.raw_id2),
            "distance": float(candidate.distance),
            "protocols": [spec.name for spec in protocols],
            "captures": captures,
        }
        with (video_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved recordings to {video_dir}", flush=True)
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                pass
        if getattr(env.unwrapped._unfold, "frame_recorder", None) is not None:
            env.unwrapped._unfold.frame_recorder = None
        if cameras is not None:
            for camera_info in cameras.values():
                try:
                    camera_info["annotator"].detach()
                except Exception:
                    pass
                try:
                    camera_info["render_product"].destroy()
                except Exception:
                    pass
        env.close()
        simulation_app.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
