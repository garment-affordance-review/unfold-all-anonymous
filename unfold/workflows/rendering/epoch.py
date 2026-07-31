#!/usr/bin/env python3
"""Epoch loop and sample post-processing for replicator v2."""

import json
import importlib.util
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .geometry import (
    _compute_side_bit_visualization,
    _rasterize_face_index_and_barycentric_from_mesh,
)
from .io import _save_sample


def _extract_env_idx_from_camera_path(cam_path: str) -> int:
    env_idx = 0
    for part in cam_path.split('/'):
        if part.startswith('env_'):
            try:
                env_idx = int(part.split('_')[-1])
            except Exception:
                pass
    return env_idx


def _extract_view_idx_from_camera_path(cam_path: str) -> int:
    match = re.search(r"/view_(\d+)/cam$", str(cam_path))
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return 0
    return 0


def _default_texture_meta(args):
    return {
        "texture_id": None,
        "patch_size_m": float(getattr(args, "patch_size_m", 0.2)),
        "size_ref_m": None,
        "tile_base": None,
        "texture_scale": None,
        "texture_rotate": None,
        "texture_translate": None,
        "external_texture_applied": False,
        "texture_apply_failure_reason": None,
    }


def _asset_dir_name(asset_meta: dict) -> str:
    asset_id = asset_meta.get("asset_id")
    if asset_id is None:
        raise RuntimeError(f"asset_id missing for rendered sample: meta={asset_meta}")
    return f"asset_{int(asset_id):04d}"


def _collect_current_batch_asset_dirs(env) -> list[str]:
    env_unwrapped = env.unwrapped
    current_asset_indices = getattr(env_unwrapped, "current_asset_indices", None)
    asset_pool = getattr(env_unwrapped, "_asset_pool", None)
    if current_asset_indices is None or asset_pool is None or not hasattr(asset_pool, "get_asset_ids"):
        return []

    asset_dirs: list[str] = []
    seen: set[str] = set()
    for pool_idx in current_asset_indices:
        try:
            pool_idx_i = int(pool_idx)
        except Exception:
            continue
        asset_ids = asset_pool.get_asset_ids([pool_idx_i])
        if not asset_ids:
            continue
        asset_dir = f"asset_{int(asset_ids[0]):04d}"
        if asset_dir in seen:
            continue
        seen.add(asset_dir)
        asset_dirs.append(asset_dir)
    return asset_dirs


def _collect_capture_observations(env):
    obs_current = env.unwrapped._get_observations()
    return {
        "pos_world": obs_current["pos"],
        "pos_mask": obs_current["pos_mask"],
        "init_pos": obs_current.get("init_pos", None),
        "faces": obs_current.get("faces", None),
        "faces_mask": obs_current.get("faces_mask", None),
        "env_origins": env.unwrapped.scene.env_origins,
    }


def _collect_asset_metadata(env, env_idx: int) -> dict:
    env_unwrapped = env.unwrapped
    garment_manager = getattr(env_unwrapped, "_garment_manager", None)
    asset_pool = getattr(env_unwrapped, "_asset_pool", None)
    assets_root = getattr(env_unwrapped, "_assets_root", None)

    asset_path_abs = None
    usd_paths = getattr(garment_manager, "_env_usd_paths", None)
    if isinstance(usd_paths, (list, tuple)) and 0 <= env_idx < len(usd_paths):
        usd = usd_paths[env_idx]
        if usd is not None:
            asset_path_abs = str(usd)

    asset_path = asset_path_abs
    if asset_path_abs and assets_root is not None:
        try:
            asset_path = str(Path(asset_path_abs).resolve().relative_to(Path(assets_root).resolve()))
        except Exception:
            asset_path = asset_path_abs

    asset_id = None
    asset_name = None
    current_asset_indices = getattr(env_unwrapped, "current_asset_indices", None)
    if current_asset_indices is not None and 0 <= env_idx < len(current_asset_indices):
        try:
            pool_index = int(current_asset_indices[env_idx])
        except Exception:
            pool_index = None
        if pool_index is not None:
            if asset_pool is not None and hasattr(asset_pool, "get_asset_ids"):
                asset_ids = asset_pool.get_asset_ids([pool_index])
                if asset_ids:
                    asset_id = int(asset_ids[0])
            if asset_path:
                asset_name = Path(asset_path).parent.name

    return {
        "asset_path": asset_path,
        "usd_path": asset_path,
        "asset_id": asset_id,
        "asset_name": asset_name,
    }


def _build_meta_extra(
    current_epoch,
    step_in_epoch,
    step_count,
    v_idx,
    env_idx,
    cam_path,
    data,
    masked_vertices,
    face_index,
    face_vertex_ids,
    barycentric_weights,
    nocs_vis,
    side_bit,
    side_rgb,
    texture_meta,
    asset_meta,
    asset_dir,
    asset_sample_id,
    global_sample_id,
):
    meta_extra = {
        'epoch': current_epoch,
        'step_in_epoch': step_in_epoch,
        'global_step': step_count,
        'global_sample_id': global_sample_id,
        'asset_sample_id': asset_sample_id,
        'asset_dir': asset_dir,
        'view_idx': v_idx,
        'env_idx': env_idx,
        'camera_path': cam_path,
        'randomized': True,
        'camera_intrinsics': data['intrinsics'],
        'camera_extrinsics': data['extrinsics'],
        'camera_distance_m': data.get('camera_distance_m'),
        'vertex_positions': masked_vertices,
        'face_index': face_index,
        'face_vertex_ids': face_vertex_ids,
        'barycentric_weights': barycentric_weights,
        'nocs_visualization': nocs_vis,
        'side_bit': side_bit,
        'side_visualization': side_rgb,
        'texture_id': texture_meta.get("texture_id"),
        'patch_size_m': texture_meta.get("patch_size_m"),
        'size_ref_m': texture_meta.get("size_ref_m"),
        'tile_base': texture_meta.get("tile_base"),
        'texture_scale': texture_meta.get("texture_scale"),
        'texture_rotate': texture_meta.get("texture_rotate"),
        'texture_translate': texture_meta.get("texture_translate"),
        'external_texture_applied': texture_meta.get("external_texture_applied"),
        'texture_apply_failure_reason': texture_meta.get("texture_apply_failure_reason"),
        'asset_path': asset_meta.get("asset_path"),
        'usd_path': asset_meta.get("usd_path"),
        'asset_id': asset_meta.get("asset_id"),
        'asset_name': asset_meta.get("asset_name"),
    }
    return meta_extra


def _process_and_save_camera_sample(
    env,
    cam_path,
    data,
    obs_ctx,
    out_dir,
    total_samples,
    args,
    current_epoch,
    step_in_epoch,
    step_count,
    v_idx,
    fabric_meta_by_env,
    mf,
    barycentric_weight_dtype,
    projection_overlap_threshold,
    asset_sample_counts,
):
    if data['rgb'] is None:
        return total_samples, None

    seg_error = data.get("seg_error")
    if seg_error:
        print(
            f"[SEMANTIC_SKIP] sample={total_samples:08d} cam={cam_path} reason={seg_error}",
            flush=True,
        )
        return total_samples, None

    env_idx = _extract_env_idx_from_camera_path(cam_path)
    pos_world = obs_ctx["pos_world"]
    pos_mask = obs_ctx["pos_mask"]
    init_pos = obs_ctx["init_pos"]
    faces = obs_ctx["faces"]
    faces_mask = obs_ctx["faces_mask"]
    env_origins = obs_ctx["env_origins"]

    mask = pos_mask[env_idx].cpu().numpy().astype(bool).squeeze()
    vertices = pos_world[env_idx].cpu().numpy()
    masked_vertices = vertices[mask]

    face_index = None
    face_vertex_ids = None
    barycentric_weights = None
    nocs_vis = None
    side_bit = None
    side_rgb = None
    env_origin = env_origins[env_idx].detach().cpu().numpy().astype(np.float32)
    if data.get('seg') is not None and masked_vertices.size > 0:
        if data.get('intrinsics') is None or data.get('extrinsics') is None:
            raise RuntimeError(
                f"face-rasterization failed sample={total_samples:08d} cam={cam_path} env={env_idx}: "
                "camera intrinsics/extrinsics missing"
            )
        if init_pos is None:
            raise RuntimeError(
                f"face-rasterization failed sample={total_samples:08d} cam={cam_path} env={env_idx}: "
                "init_pos missing"
            )

        init_pos_env = init_pos[env_idx].detach().cpu().numpy().astype(np.float32)
        if faces is None:
            raise RuntimeError(
                f"face-rasterization failed sample={total_samples:08d} cam={cam_path} env={env_idx}: "
                "obs faces missing"
            )
        faces_env = faces[env_idx]
        if faces_mask is not None:
            fmask_env = faces_mask[env_idx].cpu().numpy().astype(bool).squeeze()
            faces_env = faces_env[fmask_env]
        mesh_faces = faces_env.detach().cpu().numpy().astype(np.int64)

        mesh_faces = np.asarray(mesh_faces, dtype=np.int64)
        n_vtx = int(vertices.shape[0])
        if mesh_faces.ndim != 2 or mesh_faces.shape[1] != 3 or mesh_faces.shape[0] == 0:
            raise RuntimeError(
                f"face-rasterization failed sample={total_samples:08d} cam={cam_path} env={env_idx}: "
                "obs faces invalid or empty"
            )
        in_range_face = (
            (mesh_faces[:, 0] >= 0) & (mesh_faces[:, 0] < n_vtx) &
            (mesh_faces[:, 1] >= 0) & (mesh_faces[:, 1] < n_vtx) &
            (mesh_faces[:, 2] >= 0) & (mesh_faces[:, 2] < n_vtx)
        )
        mesh_faces_kept = mesh_faces[in_range_face]
        if mesh_faces_kept.shape[0] == 0:
            raise RuntimeError(
                f"face-rasterization failed sample={total_samples:08d} cam={cam_path} env={env_idx}: "
                "obs faces all out of particle range"
            )

        face_index, face_vertex_ids, barycentric_weights, nocs_vis = (
            _rasterize_face_index_and_barycentric_from_mesh(
                mask_np=data['seg'],
                vertices_local=vertices,
                init_pos_local=init_pos_env,
                vertex_valid_mask=mask,
                faces=mesh_faces_kept,
                K=data.get('intrinsics'),
                w2c=data.get('extrinsics'),
                env_origin=env_origin,
                projection_overlap_threshold=projection_overlap_threshold,
            )
        )

        if face_index is None or face_vertex_ids is None or barycentric_weights is None:
            raise RuntimeError(
                f"face-rasterization failed sample={total_samples:08d} cam={cam_path} env={env_idx}: "
                "PyTorch3D mesh rasterization returned no valid pixels"
            )
        if str(barycentric_weight_dtype) == "float16":
            barycentric_weights = barycentric_weights.astype(np.float16)
        else:
            barycentric_weights = barycentric_weights.astype(np.float32)
        side_mask = np.asarray(data['seg'])
        while side_mask.ndim > 2:
            side_mask = side_mask[..., 0]
        side_bit, side_rgb = _compute_side_bit_visualization(
            vertices_world=vertices,
            faces=mesh_faces_kept,
            face_index=face_index,
            mask_bool=side_mask > 0,
            w2c=data.get('extrinsics'),
        )
    else:
        print(
            f"[SEMANTIC_SKIP] sample={total_samples:08d} cam={cam_path} env={env_idx} "
            "reason=cloth mask missing or no valid cloth vertices",
            flush=True,
        )
        return total_samples, None

    texture_meta = fabric_meta_by_env.get(env_idx, _default_texture_meta(args))
    asset_meta = _collect_asset_metadata(env, env_idx)
    asset_dir = _asset_dir_name(asset_meta)
    asset_sample_idx = int(asset_sample_counts.get(asset_dir, 0))
    sp_id = f"{asset_sample_idx:08d}"
    sp_dir = out_dir / asset_dir / sp_id
    global_sample_id = f"{int(total_samples):08d}"
    meta_extra = _build_meta_extra(
        current_epoch=current_epoch,
        step_in_epoch=step_in_epoch,
        step_count=step_count,
        v_idx=v_idx,
        env_idx=env_idx,
        cam_path=cam_path,
        data=data,
        masked_vertices=masked_vertices,
        face_index=face_index,
        face_vertex_ids=face_vertex_ids,
        barycentric_weights=barycentric_weights,
        nocs_vis=nocs_vis,
        side_bit=side_bit,
        side_rgb=side_rgb,
        texture_meta=texture_meta,
        asset_meta=asset_meta,
        asset_dir=asset_dir,
        asset_sample_id=sp_id,
        global_sample_id=global_sample_id,
    )

    entry = _save_sample(
        sample_id=sp_id,
        sample_dir=sp_dir,
        rgb_np=data['rgb'],
        depth_np=data['depth'],
        mask_np=data['seg'],
        meta_extra=meta_extra,
        manifest_base=out_dir,
    )
    mf.write(json.dumps(entry, ensure_ascii=False) + '\n')
    asset_sample_counts[asset_dir] = asset_sample_idx + 1
    return total_samples + 1, {
        "asset_dir": asset_dir,
        "asset_sample_id": sp_id,
        "global_sample_id": global_sample_id,
    }


def run_epochs(
    env,
    cfg,
    args,
    cameras,
    render_products,
    annotators,
    cloth_material_switcher,
    *,
    runtime_facade,
    randomize_camera_poses_fn,
    randomize_ground_material_fn,
    randomize_cloth_material_fn,
    capture_one_frame_fn,
    refresh_capture_resources_fn,
    apply_semantic_labels_fn,
    cloth_mesh_path_fn,
    event_randomize_lights,
    event_randomize_dome_background,
    event_randomize_camera_intrinsics,
):
    def _build_camera_mappings(cameras, render_products):
        camera_to_rp = {}
        for idx, cam_path in enumerate(cameras):
            cam_key = cam_path[0] if isinstance(cam_path, list) else cam_path
            if idx < len(render_products):
                camera_to_rp[cam_key] = render_products[idx]
        ordered_cameras = sorted(
            [(c[0] if isinstance(c, list) else c) for c in cameras],
            key=lambda cp: (_extract_env_idx_from_camera_path(cp), _extract_view_idx_from_camera_path(cp)),
        )
        return camera_to_rp, ordered_cameras

    import omni.replicator.core as rep

    if runtime_facade is None:
        raise RuntimeError("runtime facade is not initialized")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / args.manifest
    with open(manifest_path, 'w'):
        pass

    pool = env.unwrapped._asset_pool
    num_batches = pool.num_batches
    capture_rounds_per_step = max(1, int(getattr(cfg, "capture_rounds_per_step", 1)))
    views_per_asset_per_step = max(1, int(getattr(cfg, "mv_num_views", 1))) * capture_rounds_per_step
    samples_per_asset = max(1, int(getattr(args, "samples_per_asset", getattr(cfg, "samples_per_asset", views_per_asset_per_step))))
    steps_per_asset_batch = int(np.ceil(float(samples_per_asset) / float(views_per_asset_per_step)))
    steps_per_epoch = num_batches * steps_per_asset_batch
    total_steps = int(args.epochs) * steps_per_epoch
    total_batches_target = int(args.epochs) * num_batches

    barycentric_weight_dtype = str(getattr(args, "barycentric_weight_dtype", "float32"))
    if barycentric_weight_dtype not in ("float16", "float32"):
        raise ValueError(
            f"--barycentric-weight-dtype must be float16|float32, got {barycentric_weight_dtype}"
        )
    projection_overlap_threshold = float(getattr(args, "projection_overlap_threshold", 0.5))
    projection_overlap_threshold = float(np.clip(projection_overlap_threshold, 0.0, 1.0))

    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("pytorch3d") is None:
        raise RuntimeError(
            "PyTorch3D face rasterization is enabled by default. Missing dependency: "
            "please install both 'torch' and 'pytorch3d' in the active environment."
        )

    camera_to_rp, ordered_cameras = _build_camera_mappings(cameras, render_products)

    step_count = 0
    total_samples = 0
    capture_index = 0
    completed_batches = 0
    asset_sample_counts: dict[str, int] = {}
    current_batch_asset_dirs = _collect_current_batch_asset_dirs(env)
    current_batch_saved_counts = {asset_dir: 0 for asset_dir in current_batch_asset_dirs}
    start_time = time.time()
    log_interval = max(1, steps_per_epoch // 10) if steps_per_epoch > 0 else 1
    rep_cam_cfg = getattr(cfg, "replicator", {}).get("camera", {})
    use_event_intrinsics = (str(rep_cam_cfg.get("intrinsics_mode", "synchronized")).lower() == "event") and (not args.no_cam_intrinsics)
    randomization_periods = getattr(cfg, "replicator", {}).get("randomization_periods", {})
    fabric_cfg = getattr(cfg, "replicator", {}).get("fabric_texture", {})
    keep_original_views = max(0, int(fabric_cfg.get("keep_original_views", 0)))
    random_prob_after_original = float(np.clip(
        fabric_cfg.get("random_prob_after_original", getattr(args, "fabric_texture_prob", 0.5)),
        0.0,
        1.0,
    ))

    def _period(key: str, default: int) -> int:
        try:
            p = int(randomization_periods.get(key, default))
        except Exception:
            p = int(default)
        return p

    def _should_trigger(step: int, period: int) -> bool:
        return period > 0 and (step % period == 0)

    lights_period = _period("lights", 3)
    dome_period = _period("dome_background", 5)
    intrinsics_period = _period("camera_intrinsics", 7)
    cloth_material_period = _period("cloth_material", 11)

    print("========================================")
    print(f"[SDG] Starting specific task: {args.task}")
    print(f"[SDG] steps/epoch={steps_per_epoch} total={total_steps} batches={num_batches}")
    print(f"[SDG] Output dir: {out_dir}")
    print(f"[SDG] samples_per_asset={samples_per_asset}")
    print("[SDG] face mapping mode: pytorch3d_rasterization")
    print(
        f"[SDG] barycentric weight dtype={barycentric_weight_dtype} "
        f"projection_overlap_threshold={projection_overlap_threshold:.2f}"
    )
    print(
        "[SDG] randomization periods: "
        f"lights={lights_period}, dome={dome_period}, "
        f"intrinsics={intrinsics_period}, cloth_material={cloth_material_period}"
    )
    print(
        "[SDG] fabric texture policy: "
        f"keep_original_views={keep_original_views}, random_prob_after_original={random_prob_after_original:.2f}"
    )
    if args.max_samples is not None:
        print(f"[SDG] max_samples={int(args.max_samples)}")
    print("========================================")

    manager = getattr(env.unwrapped, "_garment_manager", None)
    usd_paths = getattr(manager, "_env_usd_paths", None) if manager is not None else None
    if not isinstance(usd_paths, (list, tuple)) or len(usd_paths) == 0:
        epoch_info = {"epoch": 1, "total_epochs": int(args.epochs), "batch": 1, "total_batches": num_batches}
        env.unwrapped.reset(options={"switch_asset": True, "epoch_info": epoch_info})
        apply_semantic_labels_fn(env)
        current_batch_asset_dirs = _collect_current_batch_asset_dirs(env)
        current_batch_saved_counts = {asset_dir: 0 for asset_dir in current_batch_asset_dirs}
    if cloth_material_switcher is not None:
        cloth_material_switcher.refresh_original_bindings()

    with open(manifest_path, 'a', encoding='utf-8') as mf:
        while completed_batches < total_batches_target:
            if args.max_samples is not None and total_samples >= int(args.max_samples):
                print(f"[SDG] Reached max_samples={int(args.max_samples)}. Stopping collection.")
                break

            current_epoch = completed_batches // num_batches + 1
            step_in_epoch = step_count % steps_per_epoch + 1

            render_interval = max(1, int(getattr(cfg.sim, "render_interval", 1)))
            switch_after_capture = False

            if step_count % render_interval == 0:
                capture_rounds = capture_rounds_per_step

                for round_idx in range(int(capture_rounds)):
                    if _should_trigger(capture_index, lights_period):
                        runtime_facade.send_event(rep, event_randomize_lights)
                    if _should_trigger(capture_index, dome_period):
                        runtime_facade.send_event(rep, event_randomize_dome_background)
                    if use_event_intrinsics and _should_trigger(capture_index, intrinsics_period):
                        runtime_facade.send_event(rep, event_randomize_camera_intrinsics)
                    randomize_camera_poses_fn(env, cfg, cameras, args=args)

                    for cam_order_idx, cam_path in enumerate(ordered_cameras):
                        if args.max_samples is not None and total_samples >= int(args.max_samples):
                            break

                        cam_view_idx = _extract_view_idx_from_camera_path(cam_path)
                        fabric_meta_by_env = {}
                        if cloth_material_switcher is not None and not args.no_fabric_texture:
                            fabric_prob = 0.0 if cam_view_idx < keep_original_views else random_prob_after_original
                            fabric_meta_by_env = cloth_material_switcher.apply_for_capture(
                                probability=fabric_prob,
                                capture_tag=(
                                    f"epoch{current_epoch}_step{step_in_epoch}_round{round_idx+1}_"
                                    f"cam{cam_view_idx}"
                                ),
                            )
                        if randomize_cloth_material_fn is not None and not args.no_material_rand:
                            if _should_trigger(capture_index, cloth_material_period):
                                randomize_cloth_material_fn(
                                    capture_tag=(
                                        f"epoch{current_epoch}_step{step_in_epoch}_round{round_idx+1}_"
                                        f"cam{cam_view_idx}"
                                    )
                                )
                        if randomize_ground_material_fn is not None and not args.no_ground_color:
                            randomize_ground_material_fn(
                                capture_tag=(
                                    f"epoch{current_epoch}_step{step_in_epoch}_round{round_idx+1}_"
                                    f"cam{cam_view_idx}"
                                )
                            )

                        print(
                            f"[{current_epoch}/{args.epochs}][Step {step_in_epoch}/{steps_per_epoch}]"
                            f"[Round {round_idx+1}/{capture_rounds}]"
                            f"[Camera {cam_order_idx+1}/{len(ordered_cameras)}: view_{cam_view_idx}] Capturing frame..."
                        )

                        rp = camera_to_rp.get(cam_path)
                        rp_list = [rp] if rp is not None else render_products
                        data_dict = capture_one_frame_fn([cam_path], rp_list, annotators, cfg, args, env)
                        obs_ctx = _collect_capture_observations(env)
                        data = data_dict.get(cam_path)
                        if data is None:
                            capture_index += 1
                            continue
                        total_samples, saved_info = _process_and_save_camera_sample(
                            env,
                            cam_path=cam_path,
                            data=data,
                            obs_ctx=obs_ctx,
                            out_dir=out_dir,
                            total_samples=total_samples,
                            args=args,
                            current_epoch=current_epoch,
                            step_in_epoch=step_in_epoch,
                            step_count=step_count,
                            v_idx=cam_view_idx,
                            fabric_meta_by_env=fabric_meta_by_env,
                            mf=mf,
                            barycentric_weight_dtype=barycentric_weight_dtype,
                            projection_overlap_threshold=projection_overlap_threshold,
                            asset_sample_counts=asset_sample_counts,
                        )
                        if saved_info is not None:
                            saved_asset_dir = str(saved_info["asset_dir"])
                            if saved_asset_dir in current_batch_saved_counts:
                                current_batch_saved_counts[saved_asset_dir] += 1
                            if current_batch_saved_counts and all(
                                count >= samples_per_asset for count in current_batch_saved_counts.values()
                            ):
                                switch_after_capture = True
                        capture_index += 1
                        if switch_after_capture:
                            break

                    if switch_after_capture or (args.max_samples is not None and total_samples >= int(args.max_samples)):
                        break

                mf.flush()
                if args.max_samples is not None and total_samples >= int(args.max_samples):
                    break
                if switch_after_capture:
                    completed_batches += 1
                    if completed_batches >= total_batches_target:
                        break
                    next_batch_seq = completed_batches + 1
                    epoch_info = {
                        "epoch": (next_batch_seq - 1) // num_batches + 1,
                        "total_epochs": int(args.epochs),
                        "batch": (next_batch_seq - 1) % num_batches + 1,
                        "total_batches": num_batches,
                    }
                    env.unwrapped.reset(
                        seed=getattr(cfg, "seed", None),
                        options={"switch_asset": True, "epoch_info": epoch_info},
                    )
                    cameras, render_products, annotators = refresh_capture_resources_fn(
                        env, cfg, args, cameras, render_products, annotators
                    )
                    camera_to_rp, ordered_cameras = _build_camera_mappings(cameras, render_products)
                    if cloth_material_switcher is not None:
                        cloth_material_switcher.refresh_original_bindings()
                    current_batch_asset_dirs = _collect_current_batch_asset_dirs(env)
                    current_batch_saved_counts = {asset_dir: 0 for asset_dir in current_batch_asset_dirs}
                    continue

            env.unwrapped.reset(options={"switch_asset": False})
            step_count += 1

            if (step_count % log_interval == 0) or (step_count >= total_steps):
                elapsed = max(time.time() - start_time, 1e-6)
                rate = total_samples / elapsed
                print(
                    f"[STAT] epoch {current_epoch}/{args.epochs} "
                    f"step {step_in_epoch}/{steps_per_epoch} "
                    f"global {step_count}/{total_steps} "
                    f"samples {total_samples} | {rate:.1f} s/s",
                    flush=True,
                )

    if cloth_material_switcher is not None:
        stats = cloth_material_switcher.get_failure_stats()
        stats_path = out_dir / "fabric_texture_stats.json"
        with open(stats_path, "w", encoding="utf-8") as sf:
            json.dump(stats, sf, ensure_ascii=False, indent=2)
        print(f"[FABRIC_SWITCH] failure stats: {stats}")
        print(f"[FABRIC_SWITCH] stats saved: {stats_path}")

    elapsed_sec = max(time.time() - start_time, 1e-6)
    run_summary = {
        "task": str(args.task),
        "output_dir": str(out_dir),
        "manifest": str(manifest_path),
        "start_time_unix": float(start_time),
        "end_time_unix": float(start_time + elapsed_sec),
        "start_time_utc": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
        "end_time_utc": datetime.fromtimestamp(start_time + elapsed_sec, tz=timezone.utc).isoformat(),
        "elapsed_sec": float(elapsed_sec),
        "elapsed_min": float(elapsed_sec / 60.0),
        "num_samples": int(total_samples),
        "num_assets": int(len(asset_sample_counts)),
        "samples_per_second": float(total_samples / elapsed_sec),
        "seconds_per_sample": float(elapsed_sec / max(total_samples, 1)),
        "epochs": int(args.epochs),
        "num_batches": int(num_batches),
        "completed_batches": int(completed_batches),
        "samples_per_asset_target": int(samples_per_asset),
        "mv_num_views": int(getattr(cfg, "mv_num_views", 1)),
        "capture_rounds_per_step": int(capture_rounds_per_step),
        "renderer": str(getattr(cfg, "renderer", getattr(args, "pipeline_renderer", "unknown"))),
        "render_mode": str(getattr(args, "render_mode", getattr(cfg, "render_mode", "unknown"))),
        "camera_res": list(getattr(cfg, "camera_res", [])),
        "seed": getattr(cfg, "seed", None),
        "fabric_texture_stats_path": str(out_dir / "fabric_texture_stats.json") if cloth_material_switcher is not None else None,
    }
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(run_summary, sf, ensure_ascii=False, indent=2)

    print("========================================")
    print(f"All done! Total samples: {total_samples}")
    print(f"Total time elapsed: {elapsed_sec:.2f} s")
    print(f"Saved manifest to: {manifest_path}")
    print(f"Saved run summary to: {summary_path}")
