#!/usr/bin/env python3
"""Export a single protocol snapshot from the validated offline_label_2x2 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
)
from tools.offline_collection.export_parallel_snapshot import (
    SnapshotRecorder,
    _apply_snapshot_scene_style,
    _create_grasp_markers,
    _set_overview_pose,
    _setup_overview_camera,
    _update_capture_view,
    _warmup_render,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a single snapshot using the offline_label_2x2 protocol path.")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/offline_label_2x2/configs/offline_label_2x2.yaml",
        help="YAML config path.",
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--env-spacing", type=float, default=1.45)
    parser.add_argument(
        "--protocol",
        type=str,
        default="cond_y",
        choices=sorted(PROTOCOLS.keys()),
        help="Protocol mode to snapshot.",
    )
    parser.add_argument("--asset-indices", type=str, default="0", help="Single 0-based asset index.")
    parser.add_argument("--num-pairs", type=int, default=32)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--pairs-manifest", type=str, default=None)
    parser.add_argument("--rot-noise-deg", type=float, default=0.0)
    parser.add_argument("--vis-dir", type=str, default="logs/protocol_snapshot_visuals")
    parser.add_argument("--assets-manifest", type=str, default=None)
    parser.add_argument("--relift-height-min", type=float, default=0.8)
    parser.add_argument("--relift-height-max", type=float, default=1.2)
    parser.add_argument("--relift-xy-jitter", type=float, default=0.05)
    parser.add_argument("--debug-protocol-trace", action="store_true")
    parser.add_argument("--debug-stretch-trace", action="store_true")
    parser.add_argument("--debug-loop-trace", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--camera-view", type=str, default="plus_y_same_radius")
    parser.add_argument("--camera-eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-radius-scale", type=float, default=1.0)
    parser.add_argument("--camera-y-offset", type=float, default=0.0)
    parser.add_argument("--camera-z-offset", type=float, default=0.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=None)
    parser.add_argument("--column-color-style", action="store_true", default=False)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--warmup-renders", type=int, default=4)
    parser.add_argument("--capture-step", type=int, default=8)
    parser.add_argument("--output", type=str, default="logs/protocol_snapshot.png")
    parser.add_argument("--metadata", type=str, default="logs/protocol_snapshot.json")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def run(args) -> None:
    from unfold.workflows.offline_collection.pair_conditioned_collect import (
        PairConditionedOfflineCollector,
        load_pair_conditioned_env_and_cfg,
    )

    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import carb
    import unfold  # noqa: F401

    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    carb.log_warn = lambda *a, **k: None

    env = None
    cameras = None
    recorder = None
    try:
        env, env_cfg = load_pair_conditioned_env_and_cfg(args)
        if bool(getattr(args, "column_color_style", False)):
            setattr(env_cfg, "debug_color_mode", "x_gradient")
        collector = PairConditionedOfflineCollector(env, env_cfg, args)
        asset_indices = _parse_asset_indices(args.asset_indices)
        if len(asset_indices) != 1:
            raise ValueError("Snapshot expects exactly one asset index.")
        asset_index = int(asset_indices[0])
        if asset_index < 0 or asset_index >= len(collector._asset_paths):
            raise ValueError(f"asset index out of range: {asset_index}")

        collector._reset_single_asset(asset_index, asset_index)
        pointcloud = collector._prepare_pointcloud(0)
        collector._apply_coord_reward_sampling_mask(pointcloud)

        pair_index = int(args.pair_index)
        fixed_pairs = _load_pairs_manifest(args.pairs_manifest)
        if fixed_pairs:
            chosen_pairs = fixed_pairs.get(asset_index, [])
            if pair_index < 0 or pair_index >= len(chosen_pairs):
                raise ValueError(f"pair-index out of range: {pair_index}, only {len(chosen_pairs)} fixed pairs available.")
            pair_entry = chosen_pairs[pair_index]
            if not isinstance(pair_entry, FixedPair):
                raise TypeError(f"Unexpected fixed pair type: {type(pair_entry).__name__}")
            candidate = collector._build_pair_candidate(
                pointcloud=pointcloud,
                coord_id1=int(pair_entry.coord_id1),
                coord_id2=int(pair_entry.coord_id2),
                distance=float(pair_entry.distance),
                bin_idx=int(pair_entry.bin_idx),
            )
            if candidate is None:
                raise RuntimeError(
                    f"Failed to rebuild pair-conditioned pose for asset_index={asset_index} pair_index={pair_index}"
                )
            actions = _build_full_actions_from_raw_ids(
                collector.device,
                int(env_cfg.scene.num_envs),
                int(candidate.raw_id1),
                int(candidate.raw_id2),
            )
        else:
            bank = collector._build_pair_bank(0)
            chosen_pairs = bank.pop_distinct(int(args.num_pairs))
            if pair_index < 0 or pair_index >= len(chosen_pairs):
                raise ValueError(f"pair-index out of range: {pair_index}, only {len(chosen_pairs)} pairs sampled.")
            candidate = chosen_pairs[pair_index]
            actions = _build_full_actions(collector.device, int(env_cfg.scene.num_envs), candidate)

        spec = PROTOCOLS[str(args.protocol)]
        _configure_protocol(env.unwrapped, env_cfg, spec, args)
        rot_noise = (float(args.rot_noise_deg),) * 3
        _apply_init(collector, spec=spec, candidate=candidate, rot_noise_deg=rot_noise)

        _apply_snapshot_scene_style(env)
        print(f"[PROTO_SNAPSHOT] envs={env_cfg.scene.num_envs} spacing={env_cfg.scene.env_spacing} protocol={spec.name}", flush=True)
        print(
            f"[PROTO_SNAPSHOT] asset_index={asset_index} pair_index={pair_index} "
            f"coord=({candidate.coord_id1},{candidate.coord_id2}) raw=({candidate.raw_id1},{candidate.raw_id2})",
            flush=True,
        )
        markers = _create_grasp_markers(
            env,
            actions,
            radius=0.025 if int(args.num_envs) >= 64 else (0.03 if int(args.num_envs) >= 16 else 0.04),
        )

        cam_path, render_product, annotator = _setup_overview_camera(env, width=int(args.width), height=int(args.height))
        cameras = {"cam_path": cam_path, "render_product": render_product, "annotator": annotator}
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

        recorder = SnapshotRecorder(
            annotator,
            output_path=Path(args.output),
            phase="stretch",
            target_step=int(args.capture_step),
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
        env.unwrapped.step(actions)
        env.unwrapped._unfold.frame_recorder = None
        recorder.close()
        recorder = None

        if not Path(args.output).exists():
            raise RuntimeError("Protocol snapshot did not produce an output image.")

        meta = {
            "output": str(Path(args.output).resolve()),
            "protocol": spec.name,
            "num_envs": int(env_cfg.scene.num_envs),
            "env_spacing": float(env_cfg.scene.env_spacing),
            "capture_step": int(args.capture_step),
            "asset_index": asset_index,
            "pair_index": pair_index,
            "coord_id1": int(candidate.coord_id1),
            "coord_id2": int(candidate.coord_id2),
            "raw_id1": int(candidate.raw_id1),
            "raw_id2": int(candidate.raw_id2),
            "distance": float(candidate.distance),
            "bin_idx": int(candidate.bin_idx),
        }
        meta_path = Path(args.metadata)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Saved protocol snapshot to {args.output}", flush=True)
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                pass
        if env is not None and getattr(env.unwrapped._unfold, "frame_recorder", None) is not None:
            env.unwrapped._unfold.frame_recorder = None
        if cameras is not None:
            try:
                cameras["annotator"].detach()
            except Exception:
                pass
            try:
                cameras["render_product"].destroy()
            except Exception:
                pass
        if env is not None:
            env.close()
        simulation_app.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
