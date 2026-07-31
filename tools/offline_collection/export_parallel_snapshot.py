#!/usr/bin/env python3
"""Export a headless overview snapshot of parallel Isaac cloth environments during stretch."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a parallel Isaac physics snapshot during stretch+y phase.")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0")
    parser.add_argument("--config", type=str, default="configs/offline_standard.yaml")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--env-spacing", type=float, default=2.0)
    parser.add_argument("--assets-manifest", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/paper_figures/figures/isaac_parallel_snapshot.png")
    parser.add_argument("--metadata", type=str, default="outputs/paper_figures/figures/isaac_parallel_snapshot.json")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--camera-view",
        type=str,
        default="overview",
        choices=["overview", "side", "top", "oblique", "plus_y_oblique", "plus_y_same_radius"],
        help="Camera framing mode. 'side' and 'top' follow the offline_label recording style. 'oblique' is a 45-degree XY+ overview for parallel envs. 'plus_y_oblique' looks from +Y toward the origin with a moderate downward tilt. 'plus_y_same_radius' keeps the default overview pitch but rotates the camera to the +Y side at the same radius.",
    )
    parser.add_argument("--camera-eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--camera-radius-scale",
        type=float,
        default=1.0,
        help="Scale the eye-target offset while preserving the view direction. Values < 1 move the camera closer.",
    )
    parser.add_argument(
        "--camera-y-offset",
        type=float,
        default=0.0,
        help="Add a signed offset to the auto-computed camera eye along world +Y/-Y.",
    )
    parser.add_argument(
        "--camera-z-offset",
        type=float,
        default=0.0,
        help="Add an absolute z offset to the computed camera eye after view selection.",
    )
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=None,
        help="Override the downward pitch angle in degrees for +Y-facing auto views. Larger values look down more.",
    )
    parser.add_argument("--init-mode", type=str, default="random", choices=["random", "conditioned"])
    parser.add_argument("--asset-index", type=int, default=0, help="Asset index used for conditioned initialization.")
    parser.add_argument("--pair-index", type=int, default=0, help="Pair index within sampled/fixed candidate list.")
    parser.add_argument("--num-pairs", type=int, default=32, help="Number of candidate pairs sampled before selecting pair-index.")
    parser.add_argument("--pairs-manifest", type=str, default=None, help="Optional fixed pair manifest for conditioned init.")
    parser.add_argument("--rot-noise-deg", type=float, default=0.0, help="Conditioned-init Euler noise range in degrees per axis.")
    parser.add_argument("--vis-dir", type=str, default="logs/snapshot_visuals", help="Debug directory required by the pair-conditioned collector.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite semantics required by some collector paths.")
    parser.add_argument("--debug-protocol-trace", action="store_true", help="Forwarded debug flag for conditioned protocol setup.")
    parser.add_argument("--debug-stretch-trace", action="store_true", help="Forwarded debug flag for conditioned stretch tracing.")
    parser.add_argument("--debug-loop-trace", action="store_true", help="Forwarded debug flag for conditioned loop tracing.")
    parser.add_argument("--capture-step", type=int, default=None, help="Explicit stretch-step index to capture (0-based).")
    parser.add_argument("--capture-ratio", type=float, default=0.35, help="Fallback stretch-step ratio if --capture-step is unset.")
    parser.add_argument("--video-path", type=str, default=None, help="Optional mp4 path for recording stretch frames.")
    parser.add_argument("--frames-dir", type=str, default=None, help="Optional directory for saving raw stretch frames.")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--warmup-renders", type=int, default=12)
    parser.add_argument("--high-quality", action="store_true", help="Apply a high-quality single-frame rendering preset based on the main rendering pipeline defaults.")
    parser.add_argument("--pipeline-renderer", type=str, default=None, choices=["RayTracedLighting", "PathTracing"])
    parser.add_argument("--render-mode", type=str, default=None, choices=["performance", "balanced", "quality"])
    parser.add_argument("--aa", type=str, default=None, choices=["Off", "FXAA", "DLSS", "TAA", "DLAA"])
    parser.add_argument("--dlss-mode", type=int, default=None, choices=[0, 1, 2, 3])
    parser.add_argument("--spp", type=int, default=None)
    parser.add_argument("--denoise", action="store_true", default=False)
    parser.add_argument("--pt-spp-per-frame", type=int, default=None)
    parser.add_argument("--pt-total-spp", type=int, default=None)
    parser.add_argument("--pt-max-bounces", type=int, default=None)
    parser.add_argument("--pt-denoise", action="store_true", default=False)
    parser.add_argument("--rt-subframes", type=int, default=None)
    parser.add_argument(
        "--column-color-style",
        action="store_true",
        default=False,
        help="Apply a static column-wise cloth color sweep for showcase renders. This is visual-only and may be incompatible with dynamic cloth simulation on some runs.",
    )
    print("[snapshot] importing AppLauncher for CLI args", flush=True)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser


class SnapshotRecorder:
    def __init__(
        self,
        annotator,
        *,
        output_path: Path | None,
        phase: str,
        target_step: int,
        video_path: Path | None = None,
        frames_dir: Path | None = None,
        fps: int = 30,
        frame_stride: int = 1,
        pre_capture_hook=None,
        render_after_hook=None,
    ):
        self.annotator = annotator
        self.output_path = output_path
        self.phase = str(phase)
        self.target_step = int(target_step)
        self.captured = False
        self.capture_info: dict[str, int | str] = {}
        self.video_path = video_path
        self.frames_dir = frames_dir
        self.fps = int(fps)
        self.frame_stride = max(1, int(frame_stride))
        self.frame_idx = 0
        self.saved_frames = 0
        self.pre_capture_hook = pre_capture_hook
        self.render_after_hook = render_after_hook
        self.stop_after_capture = True
        self._writer = None
        if self.video_path is not None:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(str(self.video_path), fps=self.fps)
        if self.frames_dir is not None:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, *, phase_type: str, step_idx: int) -> None:
        if str(phase_type) != self.phase:
            return
        if int(step_idx) < self.target_step:
            return
        print(f"[snapshot] capture hook phase={phase_type} step={step_idx}", flush=True)
        if self.pre_capture_hook is not None:
            try:
                self.pre_capture_hook()
            except Exception as exc:
                print(f"[snapshot] pre_capture_hook failed: {exc}", flush=True)
        if self.render_after_hook is not None:
            try:
                self.render_after_hook()
            except Exception as exc:
                print(f"[snapshot] render_after_hook failed: {exc}", flush=True)
        rgb = self.annotator.get_data()
        if rgb is None:
            print("[snapshot] capture hook got no rgb data", flush=True)
            return
        frame = np.asarray(rgb)
        if frame.ndim != 3:
            print(f"[snapshot] capture hook unexpected frame ndim={frame.ndim}", flush=True)
            return
        if frame.shape[-1] >= 3:
            frame = frame[..., :3]
        frame = np.asarray(frame, dtype=np.uint8)
        if (self.frame_idx % self.frame_stride) == 0:
            if self._writer is not None:
                self._writer.append_data(frame)
            if self.frames_dir is not None:
                frame_path = self.frames_dir / f"frame_{self.saved_frames:05d}.png"
                Image.fromarray(frame).save(frame_path)
            self.saved_frames += 1
        self.frame_idx += 1
        if not self.captured and self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(self.output_path)
            print(f"[snapshot] saved snapshot image to {self.output_path.resolve()}", flush=True)
            self.captured = True
            self.capture_info = {
                "phase_type": str(phase_type),
                "step_idx": int(step_idx),
            }

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    @property
    def should_stop(self) -> bool:
        return bool(self.stop_after_capture and self.captured)


def _load_env(args):
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    from unfold.platform.config_utils import parse_yaml_config
    from unfold.simulation.env import EnvCfg

    yaml_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    env_cfg = parse_yaml_config(yaml_path, device=args.device if args.device else "cuda:0", env_cfg_class=EnvCfg)
    if args.assets_manifest:
        env_cfg.assets_manifest = str(args.assets_manifest)
    if bool(getattr(args, "column_color_style", False)):
        setattr(env_cfg, "debug_color_mode", "x_gradient")
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.num_envs = int(args.num_envs)
    env_cfg.scene.env_spacing = float(args.env_spacing)
    if bool(getattr(args, "high_quality", False)):
        env_cfg.sim.render = sim_utils.RenderCfg(
            rendering_mode=args.render_mode or "quality",
            antialiasing_mode=args.aa or "DLAA",
            dlss_mode=args.dlss_mode if args.dlss_mode is not None else 2,
            enable_dl_denoiser=bool(getattr(args, "denoise", False) or getattr(args, "pt_denoise", False)),
            samples_per_pixel=int(args.spp if args.spp is not None else 64),
            enable_reflections=True,
            enable_global_illumination=True,
        )
        env_cfg.renderer = args.pipeline_renderer or "PathTracing"
        env_cfg.pt_spp_per_frame = int(args.pt_spp_per_frame if args.pt_spp_per_frame is not None else 64)
        env_cfg.pt_total_spp = int(args.pt_total_spp if args.pt_total_spp is not None else 256)
        env_cfg.pt_max_bounces = int(args.pt_max_bounces if args.pt_max_bounces is not None else 8)
        env_cfg.pt_denoise = bool(getattr(args, "pt_denoise", False))
    spacing = float(env_cfg.scene.env_spacing)
    grid_dim = max(1, math.ceil(math.sqrt(int(env_cfg.scene.num_envs))))
    env_cfg.ground_size_m = [max(6.0, (grid_dim + 1) * spacing), max(6.0, (grid_dim + 1) * spacing)]
    env = gym.make(args.task, cfg=env_cfg)
    return env, env_cfg


def _load_conditioned_env(args):
    from unfold.workflows.offline_collection.pair_conditioned_collect import load_pair_conditioned_env_and_cfg

    env, env_cfg = load_pair_conditioned_env_and_cfg(args)
    if bool(getattr(args, "column_color_style", False)):
        setattr(env_cfg, "debug_color_mode", "x_gradient")
    if getattr(args, "env_spacing", None) is not None:
        env_cfg.scene.env_spacing = float(args.env_spacing)
        env_cfg.num_envs = int(args.num_envs)
        env_cfg.scene.num_envs = int(args.num_envs)
        spacing = float(env_cfg.scene.env_spacing)
        grid_dim = max(1, math.ceil(math.sqrt(int(env_cfg.scene.num_envs))))
        env_cfg.ground_size_m = [max(6.0, (grid_dim + 1) * spacing), max(6.0, (grid_dim + 1) * spacing)]
    return env, env_cfg


def _setup_overview_camera(env, *, width: int, height: int):
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom

    stage = env.unwrapped.sim.stage
    cam_path = "/World/ParallelSnapshotCamera"
    cam_geom = UsdGeom.Camera.Define(stage, cam_path)
    cam_geom.CreateFocalLengthAttr().Set(28.0)
    cam_geom.CreateFocusDistanceAttr().Set(400.0)
    cam_geom.CreateFStopAttr().Set(0.0)
    cam_geom.CreateHorizontalApertureAttr().Set(20.955)
    cam_geom.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))
    render_product = rep.create.render_product(cam_path, resolution=(int(width), int(height)))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    return cam_path, render_product, annotator


def _apply_snapshot_scene_style(env) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    stage = env.unwrapped.sim.stage

    print("[snapshot] scene_style: begin", flush=True)
    try:
        # Make the render ground more matte to suppress the bright specular hotspot.
        ground_mesh = stage.GetPrimAtPath("/World/Environment/GroundVisual/mesh")
        if ground_mesh and ground_mesh.IsValid():
            material_path = Sdf.Path("/World/Environment/SnapshotGroundVisualMaterial")
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.04, 0.04, 0.04))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(ground_mesh).Bind(material)
        print("[snapshot] scene_style: ground material ok", flush=True)
    except Exception as exc:
        print(f"[snapshot] scene_style: ground material skipped: {exc}", flush=True)

    try:
        # Remove local lights so the frame is dominated by a single global dome light.
        light_types = (
            UsdLux.SphereLight,
            UsdLux.RectLight,
            UsdLux.DiskLight,
            UsdLux.CylinderLight,
            UsdLux.DistantLight,
        )
        remove_paths: list[str] = []
        for prim in stage.Traverse():
            if not prim or not prim.IsValid():
                continue
            if prim.GetPath().pathString == "/World/ParallelSnapshotDomeLight":
                continue
            if any(prim.IsA(t) for t in light_types):
                remove_paths.append(prim.GetPath().pathString)
        for path in remove_paths:
            stage.RemovePrim(path)
        print(f"[snapshot] scene_style: removed {len(remove_paths)} local lights", flush=True)
    except Exception as exc:
        print(f"[snapshot] scene_style: remove lights skipped: {exc}", flush=True)

    try:
        dome = UsdLux.DomeLight.Define(stage, "/World/ParallelSnapshotDomeLight")
        dome.CreateIntensityAttr().Set(820.0)
        dome.CreateExposureAttr().Set(0.0)
        dome.CreateColorAttr().Set(Gf.Vec3f(0.96, 0.96, 0.96))
        dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool, custom=False).Set(False)
        print("[snapshot] scene_style: dome light ok", flush=True)
    except Exception as exc:
        print(f"[snapshot] scene_style: dome light skipped: {exc}", flush=True)

    # Dome-only lighting for paper qualitative renders. This keeps weak ambient
    # shading while suppressing directional shadows that distract from cloth shape.


def _apply_column_gradient_cloth_style(env) -> None:
    from colorsys import hsv_to_rgb
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = env.unwrapped.sim.stage
    env_origins = env.unwrapped.scene.env_origins.detach().cpu().numpy()
    if env_origins.size == 0:
        return

    xs = np.round(env_origins[:, 0], 4)
    unique_x = np.unique(xs)
    unique_x.sort()
    x_to_col = {float(v): idx for idx, v in enumerate(unique_x.tolist())}
    num_cols = max(1, len(unique_x))

    root_path = "/World/ParallelSnapshotClothStyle"
    UsdGeom.Xform.Define(stage, root_path)

    for env_idx in range(env_origins.shape[0]):
        col_idx = x_to_col.get(float(xs[env_idx]), 0)
        # Keep color constant for the full depth line at the same x-position, and
        # sweep hue smoothly from left to right in image space.
        t = col_idx / max(1, num_cols - 1)
        hue = (0.22 + 0.62 * t) % 1.0
        sat = 0.52
        val = 0.88
        rgb = hsv_to_rgb(hue, sat, val)
        color = Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))

        material_path = Sdf.Path(f"{root_path}/env_{env_idx:04d}_material")
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        mesh_prim = stage.GetPrimAtPath(f"/World/Cloth/env_{env_idx}/garment/mesh")
        if not mesh_prim or not mesh_prim.IsValid():
            continue
        subsets = [child for child in mesh_prim.GetChildren() if child.IsA(UsdGeom.Subset)]
        if subsets:
            for subset in subsets:
                UsdShade.MaterialBindingAPI.Apply(subset).Bind(material)
        else:
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)


def _create_grasp_markers(env, actions: torch.Tensor, *, radius: float = 0.04):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = env.unwrapped.sim.stage
    root_path = "/World/ParallelSnapshotMarkers"
    root = UsdGeom.Xform.Define(stage, root_path)
    del root

    obs = env.unwrapped._get_observations()
    pos = obs["pos"].detach().cpu().numpy()
    env_origins = env.unwrapped.scene.env_origins.detach().cpu().numpy()

    material_path = Sdf.Path(f"{root_path}/WhiteMarkerMaterial")
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    markers: list[tuple[int, int, any]] = []
    num_envs = min(int(actions.shape[0]), int(pos.shape[0]))
    for env_idx in range(num_envs):
        for pick_idx in range(2):
            vid = int(actions[env_idx, pick_idx].item())
            if vid < 0 or vid >= pos.shape[1]:
                continue
            world = pos[env_idx, vid] + env_origins[env_idx]
            world = np.asarray(world, dtype=np.float64)
            sphere_path = f"{root_path}/env_{env_idx:03d}_pick_{pick_idx}"
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.CreateRadiusAttr(float(radius))
            xform = UsdGeom.Xformable(sphere.GetPrim())
            translate_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(float(world[0]), float(world[1]), float(world[2])))
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
            markers.append((int(env_idx), int(vid), translate_op))
    return markers


def _update_grasp_markers(env, markers) -> None:
    if not markers:
        return
    all_pos = env.unwrapped._garment_manager._get_particle_positions().detach().cpu().numpy()
    for env_idx, vid, translate_op in markers:
        if env_idx >= all_pos.shape[0] or vid < 0 or vid >= all_pos.shape[1]:
            continue
        world = all_pos[env_idx, vid]
        translate_op.Set((float(world[0]), float(world[1]), float(world[2])))


def _update_capture_view(
    env,
    markers,
    *,
    cam_path: str,
    camera_view: str,
    camera_eye=None,
    camera_target=None,
    camera_radius_scale: float = 1.0,
    camera_y_offset: float = 0.0,
    camera_z_offset: float = 0.0,
    camera_pitch_deg: float | None = None,
) -> None:
    _update_grasp_markers(env, markers)
    _set_overview_pose(
        env,
        cam_path,
        camera_view=camera_view,
        camera_eye=camera_eye,
        camera_target=camera_target,
        camera_radius_scale=camera_radius_scale,
        camera_y_offset=camera_y_offset,
        camera_z_offset=camera_z_offset,
        camera_pitch_deg=camera_pitch_deg,
    )


def _set_overview_pose(
    env,
    cam_path: str,
    *,
    camera_view: str = "overview",
    camera_eye=None,
    camera_target=None,
    camera_radius_scale: float = 1.0,
    camera_y_offset: float = 0.0,
    camera_z_offset: float = 0.0,
    camera_pitch_deg: float | None = None,
) -> None:
    from unfold.platform.camera import set_camera_prims_look_at

    if camera_eye is not None and camera_target is not None:
        eye = np.asarray(camera_eye, dtype=np.float64)
        target = np.asarray(camera_target, dtype=np.float64)
        set_camera_prims_look_at([cam_path], np.asarray([eye]), np.asarray([target]))
        return

    obs = env.unwrapped._get_observations()
    pos = obs["pos"].detach().cpu().numpy()
    pos_mask = obs["pos_mask"].detach().cpu().numpy()[..., 0] > 0.5
    env_origins = env.unwrapped.scene.env_origins.detach().cpu().numpy()

    world_points = []
    for env_idx in range(pos.shape[0]):
        if np.any(pos_mask[env_idx]):
            world = pos[env_idx, pos_mask[env_idx]] + env_origins[env_idx][None, :]
            world_points.append(world)
    if not world_points:
        center = env_origins.mean(axis=0)
        extent = np.array([4.0, 4.0, 1.0], dtype=np.float64)
    else:
        world_all = np.concatenate(world_points, axis=0)
        bbox_min = world_all.min(axis=0)
        bbox_max = world_all.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)
        extent = np.maximum(bbox_max - bbox_min, 1e-3)

    bbox_min = center - 0.5 * extent
    bbox_max = center + 0.5 * extent
    max_xy = float(max(extent[0], extent[1]))
    if camera_view == "side":
        target = np.asarray(
            [
                center[0],
                center[1],
                bbox_min[2] + 0.4 * extent[2],
            ],
            dtype=np.float64,
        )
        eye = np.asarray(
            [
                bbox_max[0] + max(4.5, 2.8 * extent[1], 2.2 * extent[2]),
                center[1] + 0.15 * extent[1],
                bbox_min[2] + 1.0,
            ],
            dtype=np.float64,
        )
    elif camera_view == "oblique":
        # Frame the entire env grid from the +x,+y quadrant with a moderate downward tilt.
        target = np.asarray(
            [
                center[0],
                center[1],
                bbox_min[2] + 0.20 * extent[2],
            ],
            dtype=np.float64,
        )
        eye = np.asarray(
            [
                center[0] + max(5.0, 0.90 * max_xy + 0.65 * extent[0]),
                center[1] + max(5.0, 0.90 * max_xy + 0.65 * extent[1]),
                bbox_max[2] + max(4.0, 1.05 * max_xy),
            ],
            dtype=np.float64,
        )
    elif camera_view == "plus_y_oblique":
        target = np.asarray(
            [
                center[0],
                center[1] + 0.34 * extent[1],
                bbox_min[2] + 0.18 * extent[2],
            ],
            dtype=np.float64,
        )
        eye = np.asarray(
            [
                center[0],
                center[1] + max(1.9, 0.40 * max_xy + 0.24 * extent[1]),
                bbox_max[2] + max(2.60, 0.46 * max_xy),
            ],
            dtype=np.float64,
        )
    elif camera_view == "top":
        target = np.asarray(
            [
                center[0],
                center[1],
                bbox_min[2] + 0.15 * extent[2],
            ],
            dtype=np.float64,
        )
        eye = np.asarray(
            [
                center[0],
                center[1] + 0.05 * extent[1],
                bbox_max[2] + max(4.5, 3.0 * max_xy),
            ],
            dtype=np.float64,
        )
    elif camera_view == "plus_y_same_radius":
        target = np.asarray(
            [
                center[0],
                center[1],
                center[2] + 0.10 * extent[2],
            ],
            dtype=np.float64,
        )
        default_eye = np.asarray(
            [
                center[0] + 0.55 * extent[0] + 0.8,
                center[1] - 1.45 * max_xy - 1.8,
                center[2] + 1.15 * max_xy + 2.0,
            ],
            dtype=np.float64,
        )
        offset = default_eye - target
        horizontal_radius = float(np.linalg.norm(offset[:2]))
        eye = np.asarray(
            [
                target[0],
                target[1] + horizontal_radius,
                target[2] + offset[2],
            ],
            dtype=np.float64,
        )
    else:
        eye = np.asarray(
            [
                center[0] + 0.55 * extent[0] + 0.8,
                center[1] - 1.45 * max_xy - 1.8,
                center[2] + 1.15 * max_xy + 2.0,
            ],
            dtype=np.float64,
        )
        target = np.asarray(
            [
                center[0],
                center[1],
                center[2] + 0.10 * extent[2],
            ],
            dtype=np.float64,
        )
    radius_scale = max(0.05, float(camera_radius_scale))
    offset = eye - target
    if abs(radius_scale - 1.0) > 1e-6:
        eye = target + radius_scale * offset
        offset = eye - target
    if abs(float(camera_y_offset)) > 1e-6:
        eye = np.asarray(eye, dtype=np.float64).copy()
        eye[1] += float(camera_y_offset)
        offset = eye - target
    if abs(float(camera_z_offset)) > 1e-6:
        eye = np.asarray(eye, dtype=np.float64).copy()
        eye[2] += float(camera_z_offset)
        offset = eye - target

    if camera_pitch_deg is not None and camera_view in {"plus_y_same_radius", "plus_y_oblique"}:
        # Keep the camera on the +Y side and preserve its current Y-distance,
        # then solve target.z from the desired downward pitch angle.
        dy = max(1e-6, float(eye[1] - target[1]))
        pitch_rad = math.radians(float(camera_pitch_deg))
        target = np.asarray(target, dtype=np.float64).copy()
        target[2] = float(eye[2] - math.tan(pitch_rad) * dy)

    set_camera_prims_look_at([cam_path], np.asarray([eye]), np.asarray([target]))


def _warmup_render(env, count: int) -> None:
    for _ in range(max(0, int(count))):
        env.unwrapped.sim.render()


def _prepare_conditioned_snapshot(env, env_cfg, args):
    import traceback

    from experiments.offline_label_2x2.scripts.run_protocol_repeatability import (
        PROTOCOLS,
        _apply_init,
        _build_full_actions,
        _configure_protocol,
        _load_pairs_manifest,
    )
    from unfold.workflows.offline_collection.pair_conditioned_collect import PairConditionedOfflineCollector

    print("[snapshot] conditioned: creating collector", flush=True)
    collector = PairConditionedOfflineCollector(env, env_cfg, args)
    asset_index = int(args.asset_index)
    if asset_index < 0 or asset_index >= len(collector._asset_paths):
        raise ValueError(f"asset_index out of range: {asset_index}")

    print(f"[snapshot] conditioned: reset single asset asset_index={asset_index}", flush=True)
    collector._reset_single_asset(asset_index, asset_index)
    print("[snapshot] conditioned: asset reset done", flush=True)
    pointcloud = collector._prepare_pointcloud(0)
    print(f"[snapshot] conditioned: pointcloud ready coord={pointcloud.coord.shape[0]} raw={pointcloud.raw_coord.shape[0]}", flush=True)
    collector._apply_coord_reward_sampling_mask(pointcloud)
    print("[snapshot] conditioned: sampling mask applied", flush=True)

    fixed_pairs = _load_pairs_manifest(args.pairs_manifest)
    pair_index = int(args.pair_index)
    if fixed_pairs:
        chosen_pairs = fixed_pairs.get(asset_index, [])
        if pair_index < 0 or pair_index >= len(chosen_pairs):
            raise ValueError(f"pair-index out of range: {pair_index}, only {len(chosen_pairs)} fixed pairs available.")
        pair_entry = chosen_pairs[pair_index]
        candidate = collector._build_pair_candidate(
            pointcloud=pointcloud,
            coord_id1=int(pair_entry.coord_id1),
            coord_id2=int(pair_entry.coord_id2),
            distance=float(pair_entry.distance),
            bin_idx=int(pair_entry.bin_idx),
        )
        if candidate is None:
            raise RuntimeError(
                f"Failed to rebuild conditioned pose for asset_index={asset_index} pair_index={pair_index}"
            )
    else:
        print("[snapshot] conditioned: building pair bank", flush=True)
        bank = collector._build_pair_bank(0)
        print("[snapshot] conditioned: pair bank ready", flush=True)
        chosen_pairs = bank.pop_distinct(int(args.num_pairs))
        if pair_index < 0 or pair_index >= len(chosen_pairs):
            raise ValueError(f"pair-index out of range: {pair_index}, only {len(chosen_pairs)} pairs sampled.")
        candidate = chosen_pairs[pair_index]

    spec = PROTOCOLS["cond_y"]
    print("[snapshot] conditioned: configuring cond_y protocol", flush=True)
    try:
        _configure_protocol(env.unwrapped, env_cfg, spec, args)
    except Exception as exc:
        print(f"[snapshot] conditioned: _configure_protocol failed: {exc!r}", flush=True)
        traceback.print_exc()
        raise
    rot_noise = (float(args.rot_noise_deg),) * 3
    print("[snapshot] conditioned: applying init", flush=True)
    try:
        _apply_init(collector, spec=spec, candidate=candidate, rot_noise_deg=rot_noise)
    except Exception as exc:
        print(f"[snapshot] conditioned: _apply_init failed: {exc!r}", flush=True)
        traceback.print_exc()
        raise
    print("[snapshot] conditioned: init applied", flush=True)
    actions = _build_full_actions(collector.device, int(env_cfg.scene.num_envs), candidate)
    print("[snapshot] conditioned: actions built", flush=True)
    return collector, candidate, actions


def run(args) -> None:
    from unfold.workflows.offline_collection.common import configure_runtime_warnings
    from unfold.workflows.rendering.app import _apply_renderer_settings

    print("[snapshot] starting export_parallel_snapshot", flush=True)
    print("[snapshot] importing AppLauncher runtime", flush=True)
    from isaaclab.app import AppLauncher

    args.enable_cameras = True
    configure_runtime_warnings()
    print("[snapshot] launching Isaac app", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    if bool(getattr(args, "high_quality", False)):
        _apply_renderer_settings(args)

    import carb
    import unfold  # noqa: F401

    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    carb.log_warn = lambda *a, **k: None

    env = None
    collector = None
    candidate = None
    render_product = None
    annotator = None
    try:
        print("[snapshot] loading env", flush=True)
        if str(args.init_mode) == "conditioned":
            env, env_cfg = _load_conditioned_env(args)
        else:
            env, env_cfg = _load_env(args)
        print(f"[snapshot] env loaded num_envs={env_cfg.scene.num_envs} spacing={env_cfg.scene.env_spacing}", flush=True)
        if str(args.init_mode) == "conditioned":
            print("[snapshot] preparing conditioned initialization", flush=True)
            collector, candidate, actions = _prepare_conditioned_snapshot(env, env_cfg, args)
            obs = env.unwrapped._get_observations()
            print(
                f"[snapshot] conditioned init ready asset_index={int(args.asset_index)} pair_index={int(args.pair_index)} "
                f"coord=({int(candidate.coord_id1)},{int(candidate.coord_id2)}) raw=({int(candidate.raw_id1)},{int(candidate.raw_id2)})",
                flush=True,
            )
        else:
            obs, _info = env.unwrapped.reset(options={"switch_asset": True, "epoch_info": {"epoch": 1, "batch": 1}})
            print("[snapshot] env reset complete", flush=True)
        print("[snapshot] applying snapshot scene style", flush=True)
        _apply_snapshot_scene_style(env)
        print("[snapshot] snapshot scene style applied", flush=True)
        if str(args.init_mode) != "conditioned":
            print("[snapshot] importing RandomPolicy", flush=True)
            from unfold.algorithms.policies.random_policy import RandomPolicy

            policy = RandomPolicy(
                manager=env.unwrapped._garment_manager,
                cfg=getattr(env.unwrapped.cfg, "random_policy", {}),
                device=env.unwrapped.device,
            )
            actions = policy(obs)
            print("[snapshot] random policy actions built", flush=True)
        print("[snapshot] creating grasp markers", flush=True)
        marker_radius = 0.025 if int(args.num_envs) >= 64 else (0.03 if int(args.num_envs) >= 16 else 0.04)
        markers = _create_grasp_markers(env, actions, radius=marker_radius)
        print("[snapshot] grasp markers ready", flush=True)

        stretch_duration = int(
            env.unwrapped._unfold._calculate_steps_from_velocity(
                env_cfg.stretch_max_distance / 2,
                env_cfg.stretch_motion_velocity,
            )
        )
        target_step = int(args.capture_step) if args.capture_step is not None else max(1, int(stretch_duration * float(args.capture_ratio)))

        cam_path, render_product, annotator = _setup_overview_camera(env, width=int(args.width), height=int(args.height))
        _set_overview_pose(
            env,
            cam_path,
            camera_view=str(args.camera_view),
            camera_eye=args.camera_eye,
            camera_target=args.camera_target,
            camera_radius_scale=float(args.camera_radius_scale),
            camera_y_offset=float(args.camera_y_offset),
            camera_z_offset=float(args.camera_z_offset),
            camera_pitch_deg=args.camera_pitch_deg,
        )
        _warmup_render(env, int(args.warmup_renders))
        print(f"[snapshot] camera ready target_step={target_step} stretch_duration={stretch_duration}", flush=True)

        recorder = SnapshotRecorder(
            annotator,
            output_path=Path(args.output) if args.output else None,
            phase="stretch",
            target_step=target_step,
            video_path=Path(args.video_path) if args.video_path else None,
            frames_dir=Path(args.frames_dir) if args.frames_dir else None,
            fps=int(args.video_fps),
            frame_stride=int(args.frame_stride),
            pre_capture_hook=lambda: _update_capture_view(
                env,
                markers,
                cam_path=cam_path,
                camera_view=str(args.camera_view),
                camera_eye=args.camera_eye,
                camera_target=args.camera_target,
                camera_radius_scale=float(args.camera_radius_scale),
                camera_y_offset=float(args.camera_y_offset),
                camera_z_offset=float(args.camera_z_offset),
                camera_pitch_deg=args.camera_pitch_deg,
            ),
            render_after_hook=lambda: env.unwrapped.sim.render(),
        )
        env.unwrapped._unfold.frame_recorder = recorder
        print("[snapshot] executing one env step", flush=True)
        env.unwrapped.step(actions)
        env.unwrapped._unfold.frame_recorder = None
        print(f"[snapshot] env step finished captured={recorder.captured} info={recorder.capture_info}", flush=True)

        if not recorder.captured:
            raise RuntimeError(f"Snapshot was not captured during stretch phase (target_step={target_step}).")

        asset_pool = getattr(env.unwrapped, "_asset_pool", None)
        current_asset_indices = getattr(env.unwrapped, "current_asset_indices", None)
        asset_names: list[str] = []
        if asset_pool is not None and current_asset_indices is not None:
            for idx in current_asset_indices.detach().cpu().tolist():
                try:
                    asset_names.append(str(asset_pool.asset_paths[int(idx)]))
                except Exception:
                    asset_names.append(str(idx))

        meta = {
            "output": str(Path(args.output).resolve()),
            "init_mode": str(args.init_mode),
            "num_envs": int(env_cfg.scene.num_envs),
            "env_spacing": float(env_cfg.scene.env_spacing),
            "capture_phase": "stretch",
            "capture_step": int(recorder.capture_info.get("step_idx", target_step)),
            "target_step": int(target_step),
            "stretch_duration_steps": int(stretch_duration),
            "asset_paths": asset_names,
        }
        if candidate is not None:
            meta.update(
                {
                    "asset_index": int(args.asset_index),
                    "pair_index": int(args.pair_index),
                    "coord_id1": int(candidate.coord_id1),
                    "coord_id2": int(candidate.coord_id2),
                    "raw_id1": int(candidate.raw_id1),
                    "raw_id2": int(candidate.raw_id2),
                    "distance": float(candidate.distance),
                    "bin_idx": int(candidate.bin_idx),
                }
            )
        meta_path = Path(args.metadata)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Saved snapshot to {args.output}")
    finally:
        if env is not None and getattr(env.unwrapped._unfold, "frame_recorder", None) is not None:
            env.unwrapped._unfold.frame_recorder = None
        if annotator is not None:
            try:
                annotator.detach()
            except Exception:
                pass
        if 'recorder' in locals() and recorder is not None:
            recorder.close()
        if render_product is not None:
            try:
                render_product.destroy()
            except Exception:
                pass
        if env is not None:
            env.close()
        simulation_app.close()


def main() -> None:
    parser = build_parser()
    print("[snapshot] parser constructed", flush=True)
    args = parser.parse_args()
    print("[snapshot] args parsed", flush=True)
    run(args)


if __name__ == "__main__":
    main()
