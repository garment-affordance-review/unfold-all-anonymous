#!/usr/bin/env python3
"""Export trusted qualitative protocol snapshots without perturbing rollout dynamics.

This script follows the formal run_protocol_repeatability rollout path:
- build/rebuild fixed pairs exactly as in the evaluation runner
- apply protocol init exactly as in the evaluation runner
- run env.unwrapped.step(actions) without any render-triggering recorder
- record sparse cloth states during rollout
- replay the recorded states after rollout for offline rendering
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

from experiments.offline_label_2x2.scripts.record_protocol_videos import (
    _apply_recording_scene_style,
    _export_keyframe_strip,
    _load_font,
    _set_fixed_camera_poses,
    _setup_headless_cameras,
    _update_camera_pose,
    _warmup_render,
)
from experiments.offline_label_2x2.scripts.run_protocol_repeatability import (
    PROTOCOLS,
    _apply_init,
    _build_full_actions,
    _configure_protocol,
    _load_pairs_manifest,
    _parse_asset_indices,
    _selected_protocols,
)
from unfold.platform.camera import rotmat_to_quat_wxyz
from unfold.workflows.offline_collection.pair_conditioned_collect import PairCandidate


def _safe_normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return vec / norm


def _rebuild_candidate_inline(pointcloud, pair_entry, *, device: torch.device) -> PairCandidate | None:
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


@dataclass
class ClothSnapshot:
    positions: torch.Tensor
    velocities: torch.Tensor
    label: str


class StateSnapshotRecorder:
    """Records sparse cloth states without requesting any rendering during rollout."""

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
            all_pos = garment._get_particle_positions().detach().cpu()
            all_vel = garment._get_particle_velocities().detach().cpu()
            self.snapshots.append(
                ClothSnapshot(
                    positions=all_pos.clone(),
                    velocities=all_vel.clone(),
                    label=f"{phase_type}:{self.global_step}",
                )
            )
        self.global_step += 1

    def append_final(self, *, label: str = "final") -> None:
        garment = self.env.unwrapped._garment_manager
        all_pos = garment._get_particle_positions().detach().cpu()
        all_vel = garment._get_particle_velocities().detach().cpu()
        self.snapshots.append(
            ClothSnapshot(
                positions=all_pos.clone(),
                velocities=all_vel.clone(),
                label=label,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trusted sparse qualitative export for offline-label 2x2 protocols.")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/offline_label_2x2/configs/offline_label_2x2.yaml",
    )
    parser.add_argument("--asset-indices", type=str, default="96")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--pairs-manifest", type=str, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--protocol", type=str, default="all", choices=["all", *sorted(PROTOCOLS.keys())])
    parser.add_argument("--rot-noise-deg", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/offline_label_2x2_trusted_snapshots",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="logs/offline_label_2x2_trusted_snapshots/visuals",
    )
    parser.add_argument("--assets-manifest", type=str, default=None)
    parser.add_argument("--relift-height-min", type=float, default=0.8)
    parser.add_argument("--relift-height-max", type=float, default=1.2)
    parser.add_argument("--relift-xy-jitter", type=float, default=0.05)
    parser.add_argument("--video-width", type=int, default=1024)
    parser.add_argument("--video-height", type=int, default=1024)
    parser.add_argument("--warmup-renders", type=int, default=4)
    parser.add_argument("--enable-textures", action="store_true", default=True)
    parser.add_argument("--disable-textures", dest="enable_textures", action="store_false")
    parser.add_argument("--fixed-cameras", action="store_true", default=True)
    parser.add_argument("--disable-fixed-cameras", dest="fixed_cameras", action="store_false")
    parser.add_argument("--ground-size", type=float, default=8.0)
    parser.add_argument("--top-eye", type=float, nargs=3, default=(0.0, 0.55, 2.0))
    parser.add_argument("--top-target", type=float, nargs=3, default=(0.0, 0.55, 0.12))
    parser.add_argument("--side-eye", type=float, nargs=3, default=(2.0, 0.55, 2.45))
    parser.add_argument("--side-target", type=float, nargs=3, default=(0.0, 0.55, 0.88))
    parser.add_argument("--image-save-fps", type=float, default=1.0, help="Sparse rollout state sampling rate.")
    parser.add_argument("--keyframes-per-view", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug-protocol-trace", action="store_true")
    parser.add_argument("--debug-stretch-trace", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _save_frame(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.asarray(frame, dtype=np.uint8))


def _render_snapshot_set(
    env,
    cameras,
    snapshots: list[ClothSnapshot],
    *,
    protocol_dir: Path,
    keyframes_per_view: int,
) -> None:
    garment = env.unwrapped._garment_manager
    env_ids_long = torch.tensor([0], device=env.unwrapped.device, dtype=torch.long)
    action_manager = env.unwrapped._unfold.action_manager
    action_manager.stop_all_control()

    for view_name, camera_info in cameras.items():
        frames_dir = protocol_dir / view_name / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for snap_idx, snapshot in enumerate(snapshots):
            garment._set_particle_positions(snapshot.positions.to(env.unwrapped.device), env_ids_long)
            garment._set_particle_velocities(snapshot.velocities.to(env.unwrapped.device), env_ids_long)
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.render()
            rgb = camera_info["annotator"].get_data()
            frame = np.asarray(rgb)[..., :3]
            _save_frame(frame, frames_dir / f"frame_{snap_idx:05d}.png")
        _export_keyframe_strip(
            frames_dir=frames_dir,
            output_path=protocol_dir / f"{protocol_dir.name}_{view_name}_keyframes.png",
            title=f"{protocol_dir.name} | {view_name}",
            max_frames=int(keyframes_per_view),
        )


def run(args) -> None:
    from unfold.workflows.offline_collection.pair_conditioned_collect import (
        PairConditionedOfflineCollector,
        load_pair_conditioned_env_and_cfg,
    )

    args.enable_cameras = True
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
        raise ValueError("Trusted export expects exactly one asset index.")
    asset_index = int(asset_indices[0])
    protocols = _selected_protocols(args.protocol)
    fixed_pairs = _load_pairs_manifest(args.pairs_manifest)
    if asset_index not in fixed_pairs:
        raise ValueError(f"No fixed pairs found for asset_index={asset_index}")
    pair_entry = fixed_pairs[asset_index][int(args.pair_index)]
    rot_noise = (float(args.rot_noise_deg),) * 3
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = None
    try:
        print(f"[TRUSTED] reset asset_index={asset_index}", flush=True)
        collector._reset_single_asset(asset_index, asset_index)
        print(f"[TRUSTED] prepare pointcloud asset_index={asset_index}", flush=True)
        pointcloud = collector._prepare_pointcloud(0)
        collector._apply_coord_reward_sampling_mask(pointcloud)
        print(f"[TRUSTED] rebuild candidate asset_index={asset_index}", flush=True)
        candidate = _rebuild_candidate_inline(
            pointcloud,
            pair_entry,
            device=collector.device,
        )
        if candidate is None:
            raise RuntimeError("Failed to rebuild trusted fixed pair candidate.")
        print(
            f"[TRUSTED] candidate coord=({candidate.coord_id1},{candidate.coord_id2}) "
            f"raw=({candidate.raw_id1},{candidate.raw_id2})",
            flush=True,
        )
        actions = _build_full_actions(collector.device, int(env_cfg.scene.num_envs), candidate)

        if bool(args.enable_textures):
            env_cfg.enable_textures = True
            env.unwrapped.cfg.enable_textures = True
            try:
                env.unwrapped._garment_manager._apply_preview_materials()
            except Exception as exc:
                print(f"[WARN] failed to apply preview textures: {exc}", flush=True)
        print("[TRUSTED] apply scene style", flush=True)
        _apply_recording_scene_style(env, ground_size=float(args.ground_size))
        print("[TRUSTED] setup cameras", flush=True)
        cameras = _setup_headless_cameras(env, width=int(args.video_width), height=int(args.video_height))
        if bool(args.fixed_cameras):
            _set_fixed_camera_poses(args, cameras)
        else:
            for view_name in ("top", "side"):
                _update_camera_pose(env, cameras[view_name]["cam_path"], view_name=view_name)
        print("[TRUSTED] warmup render", flush=True)
        _warmup_render(env, int(args.warmup_renders))

        asset_dir = out_dir / f"asset_{asset_index:04d}" / f"pair_{int(args.pair_index):02d}"
        asset_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "asset_index": asset_index,
            "asset_id": int(collector._asset_ids[asset_index]),
            "pair_index": int(args.pair_index),
            "coord_id1": int(candidate.coord_id1),
            "coord_id2": int(candidate.coord_id2),
            "raw_id1": int(candidate.raw_id1),
            "raw_id2": int(candidate.raw_id2),
            "distance": float(candidate.distance),
            "protocols": [],
        }

        for spec in protocols:
            print(f"[TRUSTED] protocol_start={spec.name}", flush=True)
            _configure_protocol(env.unwrapped, env_cfg, spec, args)
            print(f"[TRUSTED] apply_init={spec.name}", flush=True)
            _apply_init(collector, spec=spec, candidates=[candidate], rot_noise_deg=rot_noise)
            state_recorder = StateSnapshotRecorder(
                env,
                sample_hz=float(args.image_save_fps),
                physics_dt=float(env.unwrapped.physics_dt),
            )
            env.unwrapped._unfold.frame_recorder = state_recorder
            print(f"[TRUSTED] env_step={spec.name}", flush=True)
            obs, rewards, _, _, extras = env.unwrapped.step(actions)
            env.unwrapped._unfold.frame_recorder = None
            state_recorder.append_final(label="final")
            print(f"[TRUSTED] replay_render={spec.name} snapshots={len(state_recorder.snapshots)}", flush=True)

            protocol_dir = asset_dir / spec.name
            protocol_dir.mkdir(parents=True, exist_ok=True)
            _render_snapshot_set(
                env,
                cameras,
                state_recorder.snapshots,
                protocol_dir=protocol_dir,
                keyframes_per_view=int(args.keyframes_per_view),
            )
            summary["protocols"].append(
                {
                    "protocol": spec.name,
                    "reward": float(rewards.detach().cpu().view(-1)[0].item()),
                    "num_snapshots": len(state_recorder.snapshots),
                    "deformable_distance": float(
                        extras.get("rewards_extras", {}).get("deformable_distance", torch.tensor([float("nan")]))[0]
                    ),
                    "rigid_distance": float(
                        extras.get("rewards_extras", {}).get("rigid_distance", torch.tensor([float("nan")]))[0]
                    ),
                }
            )

        with (asset_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved trusted snapshots to {asset_dir}", flush=True)
    finally:
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
