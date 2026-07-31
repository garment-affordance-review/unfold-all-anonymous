#!/usr/bin/env python3
"""Replicator runtime helpers for the rendering pipeline."""

import numpy as np

from .camera import CameraPosePipeline
from .epoch import run_epochs
from .fabric import ClothMaterialSwitcher, FabricTextureCatalog
from .randomization import DomainRandomizationFacade
from .runtime import RenderRuntimeFacade

_CAMERA_DISTANCE_BY_PATH = {}
_runtime_facade = None
_camera_pose_pipeline = None
_domain_randomization_facade = None

EVENT_RANDOMIZE_LIGHTS = "randomize_lights"
EVENT_RANDOMIZE_DOME_BACKGROUND = "randomize_dome_background"
EVENT_RANDOMIZE_CAMERA_INTRINSICS = "randomize_camera_intrinsics"


def _cloth_mesh_path(cloth_root: str, env_idx: int) -> str:
    root = str(cloth_root).rstrip("/") or "/World/Cloth"
    return f"{root}/env_{env_idx}/garment/mesh"

def _extract_camera_parameters(cam_path, stage, image_width, image_height):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(cam_path)
    if not prim.IsValid():
        return None, None

    cam_geom = UsdGeom.Camera(prim)
    if not cam_geom:
        return None, None

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
    w2c = np.linalg.inv(c2w)

    return K, w2c


def _apply_semantic_labels(env):
    """Add class:cloth semantic labels for garment root and mesh prims."""
    from pxr import UsdSemantics

    stage = env.unwrapped.sim.stage
    cloth_root = str(getattr(env.unwrapped.cfg, "cloth_root", "/World/Cloth"))

    labeled = 0
    for env_idx in range(env.unwrapped.num_envs):
        prim_paths = [
            f"{cloth_root.rstrip('/')}/env_{env_idx}/garment",
            _cloth_mesh_path(cloth_root, env_idx),
        ]
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                continue
            schema_name = "class"
            has_labels_api = any(
                applied_schema == f"SemanticsLabelsAPI:{schema_name}"
                for applied_schema in prim.GetAppliedSchemas()
            )
            labels_api = (
                UsdSemantics.LabelsAPI(prim, schema_name)
                if has_labels_api
                else UsdSemantics.LabelsAPI.Apply(prim, schema_name)
            )
            labels_attr = labels_api.GetLabelsAttr()
            if not labels_attr or not labels_attr.IsDefined():
                labels_attr = labels_api.CreateLabelsAttr()
            labels_attr.Set(["cloth"])
            labeled += 1
    print(f"[REPLICATOR] Applied 'class:cloth' semantic labels to {labeled} garment prims")


def _setup_native_randomization(env, cfg, cam_group, args):
    global _domain_randomization_facade
    if _domain_randomization_facade is None:
        _domain_randomization_facade = DomainRandomizationFacade(
            {
                "lights": EVENT_RANDOMIZE_LIGHTS,
                "dome_background": EVENT_RANDOMIZE_DOME_BACKGROUND,
                "camera_intrinsics": EVENT_RANDOMIZE_CAMERA_INTRINSICS,
            }
        )
    _domain_randomization_facade.setup_native_randomization(env, cfg, cam_group, args)


def _randomize_camera_poses_usd(env, cfg, cameras, args=None):
    global _CAMERA_DISTANCE_BY_PATH, _camera_pose_pipeline
    if _camera_pose_pipeline is None:
        _camera_pose_pipeline = CameraPosePipeline(seed=getattr(cfg, "seed", None))
    _CAMERA_DISTANCE_BY_PATH = _camera_pose_pipeline.randomize_camera_poses_usd(env, cfg, cameras, args=args)


def _randomize_ground_material(capture_tag=None):
    global _domain_randomization_facade
    if _domain_randomization_facade is None:
        return
    _domain_randomization_facade.randomize_ground_material(capture_tag=capture_tag)


def _randomize_cloth_material(capture_tag=None):
    global _domain_randomization_facade
    if _domain_randomization_facade is None:
        return
    _domain_randomization_facade.randomize_cloth_material_once(capture_tag=capture_tag)


def _setup_replicator(env, cfg, args):
    global _runtime_facade
    import omni.replicator.core as rep

    rl_env = env.unwrapped if hasattr(env, "unwrapped") else env

    _runtime_facade = RenderRuntimeFacade(
        cfg,
        args,
        extract_camera_parameters=_extract_camera_parameters,
    )
    cloth_root = str(getattr(args, "cloth_root", "/World/Cloth"))

    cameras_all = _runtime_facade.create_camera_prims(rl_env, cloth_root)
    if not cameras_all:
        raise ValueError("[ERROR] No cameras spawned.")

    cam_group = rep.get.prims(path_pattern=f"{cloth_root.rstrip('/')}/env_.*/view_.*/cam")
    if bool(getattr(args, "enable_semantic_seg", True)):
        _apply_semantic_labels(env)
    render_products, all_annotators = _runtime_facade.create_render_products_and_annotators(cameras_all)
    _setup_native_randomization(env, cfg, cam_group, args)
    _runtime_facade.warmup(env, num_frames=3)
    return cameras_all, render_products, all_annotators


def _refresh_replicator_capture_resources(env, cfg, args, cameras, render_products, annotators):
    global _runtime_facade
    if _runtime_facade is None:
        raise RuntimeError("runtime facade is not initialized")

    rl_env = env.unwrapped if hasattr(env, "unwrapped") else env
    cloth_root = str(getattr(args, "cloth_root", "/World/Cloth"))

    _runtime_facade.destroy_render_products_and_annotators(render_products, annotators)
    cameras_all = _runtime_facade.create_camera_prims(rl_env, cloth_root)
    if bool(getattr(args, "enable_semantic_seg", True)):
        _apply_semantic_labels(env)
    render_products, all_annotators = _runtime_facade.create_render_products_and_annotators(cameras_all)
    _runtime_facade.warmup(env, num_frames=3)
    return cameras_all, render_products, all_annotators


def _set_render_products_updates(render_products, enabled: bool):
    global _runtime_facade
    if _runtime_facade is None:
        raise RuntimeError("runtime facade is not initialized")
    _runtime_facade.set_render_products_updates(render_products, enabled=enabled)


def _capture_one_frame(cameras, render_products, annotators, cfg, args, env):
    global _runtime_facade
    import omni.usd

    if args.disable_rp_between_captures:
        _set_render_products_updates(render_products, enabled=True)

    if _runtime_facade is None:
        raise RuntimeError("runtime facade is not initialized")
    _runtime_facade.render_subframes(env)

    stage = omni.usd.get_context().get_stage()
    cam_res = getattr(cfg, "camera_res", [1024, 1024])
    image_width, image_height = cam_res[0], cam_res[1]

    out_dict = {}
    for cam_path in cameras:
        if isinstance(cam_path, list):
            cam_path = cam_path[0]
        anns = annotators[cam_path]
        rgb, depth, mask_np, K, w2c, seg_error = _runtime_facade.collect_camera_frame_data(
            cam_path, anns, stage, image_width, image_height
        )
        out_dict[cam_path] = {
            "rgb": np.array(rgb) if rgb is not None else None,
            "depth": np.array(depth) if depth is not None else None,
            "seg": mask_np,
            "seg_error": seg_error,
            "intrinsics": K,
            "extrinsics": w2c,
            "camera_distance_m": _CAMERA_DISTANCE_BY_PATH.get(cam_path),
        }

    if args.disable_rp_between_captures:
        _set_render_products_updates(render_products, enabled=False)
    return out_dict


def _run_epochs(env, cfg, args, cameras, render_products, annotators, cloth_material_switcher=None):
    global _runtime_facade
    if _runtime_facade is None:
        raise RuntimeError("runtime facade is not initialized")

    return run_epochs(
        env,
        cfg,
        args,
        cameras,
        render_products,
        annotators,
        cloth_material_switcher,
        runtime_facade=_runtime_facade,
        randomize_camera_poses_fn=_randomize_camera_poses_usd,
        randomize_ground_material_fn=_randomize_ground_material,
        randomize_cloth_material_fn=_randomize_cloth_material,
        capture_one_frame_fn=_capture_one_frame,
        refresh_capture_resources_fn=_refresh_replicator_capture_resources,
        apply_semantic_labels_fn=_apply_semantic_labels,
        cloth_mesh_path_fn=_cloth_mesh_path,
        event_randomize_lights=EVENT_RANDOMIZE_LIGHTS,
        event_randomize_dome_background=EVENT_RANDOMIZE_DOME_BACKGROUND,
        event_randomize_camera_intrinsics=EVENT_RANDOMIZE_CAMERA_INTRINSICS,
    )
