#!/usr/bin/env python3
"""High-level camera pipeline for pose randomization and diagnostics."""

import math
import re

import numpy as np


def _normalize_np(vec, eps=1e-9):
    v = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0, n
    return v / n, n


def _quat_look_at_world_up(camera_pos, target_pos, world_up=(0.0, 0.0, 1.0)):
    cam = np.asarray(camera_pos, dtype=np.float64)
    tgt = np.asarray(target_pos, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)

    fwd, nf = _normalize_np(tgt - cam)
    if nf < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), up.astype(np.float32), fwd.astype(np.float32)

    right, nr = _normalize_np(np.cross(fwd, up))
    if nr < 1e-9:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fwd, fallback))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right, _ = _normalize_np(np.cross(fwd, fallback))

    cam_up, _ = _normalize_np(np.cross(right, fwd))
    R = np.eye(3, dtype=np.float64)
    R[:, 0] = right
    R[:, 1] = cam_up
    R[:, 2] = -fwd

    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q, _ = _normalize_np(q)
    return q.astype(np.float32), cam_up.astype(np.float32), fwd.astype(np.float32)


def _compute_roll_deg_from_world_up(forward_world, cam_up_world, world_up=(0.0, 0.0, 1.0)):
    fwd, _ = _normalize_np(forward_world)
    cup, _ = _normalize_np(cam_up_world)
    wup, _ = _normalize_np(world_up)
    if np.linalg.norm(fwd) < 1e-9 or np.linalg.norm(cup) < 1e-9 or np.linalg.norm(wup) < 1e-9:
        return 0.0

    wup_proj = wup - np.dot(wup, fwd) * fwd
    cup_proj = cup - np.dot(cup, fwd) * fwd
    wup_proj, n1 = _normalize_np(wup_proj)
    cup_proj, n2 = _normalize_np(cup_proj)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0

    sin_a = float(np.dot(fwd, np.cross(wup_proj, cup_proj)))
    cos_a = float(np.clip(np.dot(wup_proj, cup_proj), -1.0, 1.0))
    return math.degrees(math.atan2(sin_a, cos_a))


def _set_transform_attributes(prim, location=None, orientation=None):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xformable(prim)
    x_ops = x.GetOrderedXformOps()

    if location is not None:
        t_ops = [o for o in x_ops if o.GetOpName().endswith("xformOp:translate")]
        t_op = t_ops[0] if t_ops else UsdGeom.Xformable(prim).AddTranslateOp()
        loc = np.asarray(location, dtype=np.float64).reshape(-1)
        if loc.size >= 3:
            t_op.Set(Gf.Vec3d(float(loc[0]), float(loc[1]), float(loc[2])))
        else:
            t_op.Set(location)

    if orientation is not None:
        o_ops = [o for o in x_ops if o.GetOpName().endswith("xformOp:orient")]
        o_op = o_ops[0] if o_ops else UsdGeom.Xformable(prim).AddOrientOp()
        o_op.Set(orientation)


class CameraPosePipeline:
    def __init__(self, seed: int | None = None):
        self._camera_distance_by_path = {}
        self._cam_fov_cache = {}
        self._seed = None if seed is None else int(seed)
        self._rng = np.random.default_rng(self._seed)

    def _ensure_seed(self, seed: int | None):
        next_seed = None if seed is None else int(seed)
        if next_seed != self._seed:
            self._seed = next_seed
            self._rng = np.random.default_rng(self._seed)

    def get_pose_debug(self):
        return dict(self._camera_distance_by_path)

    def randomize_camera_poses_usd(self, env, cfg, cameras, args=None):
        from pxr import Gf, UsdGeom
        import omni.usd
        from unfold.platform.camera import compute_centers_world

        stage = omni.usd.get_context().get_stage()
        device = env.unwrapped.device
        obs = env.unwrapped._get_observations()
        cam_res = getattr(cfg, "camera_res", [1024, 1024])
        self._camera_distance_by_path = {}
        self._ensure_seed(getattr(cfg, "seed", None))

        centers = compute_centers_world(obs["pos"], obs["pos_mask"], env.unwrapped.scene.env_origins).to(device)

        bbox_cache = {}
        try:
            pos = obs.get("pos") if isinstance(obs, dict) else None
            pos_mask = obs.get("pos_mask") if isinstance(obs, dict) else None
            if pos is not None and pos_mask is not None:
                pos_np = pos.detach().cpu().numpy()
                mask_np = pos_mask.detach().cpu().numpy()
                for env_idx in range(pos_np.shape[0]):
                    mask = np.array(mask_np[env_idx]).astype(bool).squeeze()
                    if mask.size == 0 or not mask.any():
                        continue
                    pts = pos_np[env_idx][mask]
                    if pts.size == 0:
                        continue
                    bbox_min = pts.min(axis=0)
                    bbox_max = pts.max(axis=0)
                    extent = bbox_max - bbox_min
                    bbox_cache[env_idx] = {
                        "min": bbox_min,
                        "max": bbox_max,
                        "extent": extent,
                        "diag": float(np.linalg.norm(extent)),
                        "center": (bbox_min + bbox_max) / 2.0,
                    }
        except Exception:
            pass

        rep_cam_cfg = getattr(cfg, "replicator", {}).get("camera", {})
        camera_adapt_cfg = getattr(cfg, "replicator", {}).get("camera_adapt", {})
        orbit_mode = str(rep_cam_cfg.get("orbit_mode", "random_sphere")).lower()
        r_jitter = float(rep_cam_cfg.get("radius_jitter", 0.0))
        look_jitter = float(rep_cam_cfg.get("look_at_jitter", 0.0))
        phi_min_deg = float(rep_cam_cfg.get("min_elevation_deg", 30.0))
        phi_max_deg = float(rep_cam_cfg.get("max_elevation_deg", 65.0))
        ring_elevation_mode = str(rep_cam_cfg.get("uniform_ring_elevation_mode", "stratified")).lower()
        elev_jitter_deg = float(rep_cam_cfg.get("elevation_jitter_deg", 0.0))
        yaw_jitter_deg = float(rep_cam_cfg.get("yaw_jitter_deg", 0.0))
        roll_max_deg = float(rep_cam_cfg.get("roll_max_deg", 5.0))
        intrinsics_mode = str(rep_cam_cfg.get("intrinsics_mode", "synchronized")).lower()

        f_range = rep_cam_cfg.get("focal_length_range", [24.0, 35.0])
        if not isinstance(f_range, (list, tuple)) or len(f_range) < 2:
            f_range = [24.0, 35.0]
        focal_min, focal_max = float(f_range[0]), float(f_range[1])
        if focal_max < focal_min:
            focal_min, focal_max = focal_max, focal_min

        target_ratio_range = camera_adapt_cfg.get(
            "target_mask_ratio_range", camera_adapt_cfg.get("target_ratio_range", [0.6, 0.8])
        )
        if not isinstance(target_ratio_range, (list, tuple)) or len(target_ratio_range) < 2:
            target_ratio_range = [0.6, 0.8]
        ratio_min = float(target_ratio_range[0])
        ratio_max = float(target_ratio_range[1])
        if ratio_max < ratio_min:
            ratio_min, ratio_max = ratio_max, ratio_min

        target_metric = str(camera_adapt_cfg.get("target_metric", "area")).lower()
        long_side_ratio_range = camera_adapt_cfg.get(
            "target_long_side_ratio_range",
            [0.7, 0.9],
        )
        if not isinstance(long_side_ratio_range, (list, tuple)) or len(long_side_ratio_range) < 2:
            long_side_ratio_range = [0.7, 0.9]
        long_side_min = float(long_side_ratio_range[0])
        long_side_max = float(long_side_ratio_range[1])
        if long_side_max < long_side_min:
            long_side_min, long_side_max = long_side_max, long_side_min

        max_resample = max(1, int(camera_adapt_cfg.get("max_resample", 6)))
        if phi_max_deg <= phi_min_deg:
            phi_max_deg = phi_min_deg + 1e-3

        cloth_root = str(getattr(args, "cloth_root", "/World/Cloth")) if args is not None else "/World/Cloth"
        pattern = rf"{re.escape(cloth_root.rstrip('/'))}/env_(\d+)/view_(\d+)/cam"
        disable_intrinsics = bool(getattr(args, "no_cam_intrinsics", False)) if args is not None else False
        num_views = max(1, int(getattr(cfg, "mv_num_views", 1)))
        per_env_yaw_phase = {}
        env_focal_lengths = {}

        def sample_focal_length() -> float:
            return float(self._rng.uniform(focal_min, focal_max))

        for cam_path in cameras:
            match = re.search(pattern, cam_path)
            env_id = int(match.group(1)) if match else 0
            view_id = int(match.group(2)) if match else 0
            base_target = centers[env_id].detach().cpu().numpy()
            bbox = bbox_cache.get(env_id)
            prim = stage.GetPrimAtPath(cam_path)
            if not prim.IsValid():
                continue

            cam_geom = UsdGeom.Camera(prim)
            if cam_geom and (not disable_intrinsics):
                if intrinsics_mode == "synchronized":
                    focal_value = env_focal_lengths.get(env_id)
                    if focal_value is None:
                        focal_value = sample_focal_length()
                        env_focal_lengths[env_id] = focal_value
                    cam_geom.GetFocalLengthAttr().Set(focal_value)
                elif intrinsics_mode == "per_view":
                    cam_geom.GetFocalLengthAttr().Set(sample_focal_length())

            d_clamp = camera_adapt_cfg.get("d_clamp", [0.1, 10.0])
            if not isinstance(d_clamp, (list, tuple)) or len(d_clamp) < 2:
                d_clamp = [0.1, 10.0]
            d_min, d_max = float(d_clamp[0]), float(d_clamp[1])
            min_h = float(rep_cam_cfg.get("min_height", 0.6))

            selected = None
            for attempt in range(max_resample):
                target_loc = base_target.copy()
                if look_jitter > 0.0:
                    target_loc += self._rng.uniform(-look_jitter, look_jitter, 3)

                if target_metric == "long_side":
                    linear_ratio_target = float(self._rng.uniform(long_side_min, long_side_max))
                    linear_ratio_target = float(np.clip(linear_ratio_target, 1e-3, 0.98))
                    area_ratio_target = float(linear_ratio_target * linear_ratio_target)
                else:
                    area_ratio_target = float(self._rng.uniform(ratio_min, ratio_max))
                    linear_ratio_target = float(np.sqrt(np.clip(area_ratio_target, 1e-4, 1.0)))
                if camera_adapt_cfg.get("mode") == "fov_backsolve":
                    mv_radius = self._compute_adaptive_distance(
                        stage=stage,
                        cam_path=cam_path,
                        cam_geom=cam_geom,
                        bbox=bbox,
                        cam_res=cam_res,
                        cfg=cfg,
                        camera_adapt_cfg=camera_adapt_cfg,
                        linear_ratio_target=linear_ratio_target,
                        cameras=cameras,
                    )
                else:
                    mv_radius = getattr(cfg, "mv_radius", 2.0)

                r_cur = mv_radius * (1.0 + self._rng.uniform(-r_jitter, r_jitter))
                r_cur = float(np.clip(r_cur, d_min, d_max))
                r_cur = max(r_cur, 0.1)

                if orbit_mode == "uniform_ring":
                    if env_id not in per_env_yaw_phase:
                        per_env_yaw_phase[env_id] = float(self._rng.uniform(0.0, 2.0 * math.pi))
                    theta_base = per_env_yaw_phase[env_id] + (2.0 * math.pi * float(view_id) / float(num_views))
                    theta = theta_base + math.radians(float(self._rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)))
                    if ring_elevation_mode in ("stratified", "uniform_by_view"):
                        if num_views <= 1:
                            phi_base_deg = float(self._rng.uniform(phi_min_deg, phi_max_deg))
                        else:
                            t0 = float(view_id) / float(num_views)
                            t1 = float(view_id + 1) / float(num_views)
                            u = float(self._rng.uniform(t0, t1))
                            phi_base_deg = float(phi_min_deg + (phi_max_deg - phi_min_deg) * u)
                    else:
                        phi_base_deg = float(self._rng.uniform(phi_min_deg, phi_max_deg))
                    phi_deg = phi_base_deg + float(self._rng.uniform(-elev_jitter_deg, elev_jitter_deg))
                    phi_deg = float(np.clip(phi_deg, phi_min_deg, phi_max_deg))
                    phi = math.radians(phi_deg)
                else:
                    theta = self._rng.uniform(0.0, 2 * math.pi)
                    phi = self._rng.uniform(math.radians(phi_min_deg), math.radians(phi_max_deg))
                ce, se = math.cos(phi), math.sin(phi)
                cy, sy = math.cos(theta), math.sin(theta)
                offset = np.array([r_cur * ce * cy, r_cur * ce * sy, r_cur * se], dtype=np.float32)

                cam_loc = target_loc + offset
                cam_loc[2] = max(cam_loc[2], target_loc[2] + min_h)

                quat_wxyz, cam_up_world, forward_world = _quat_look_at_world_up(
                    camera_pos=cam_loc,
                    target_pos=target_loc,
                    world_up=(0.0, 0.0, 1.0),
                )
                roll_deg = _compute_roll_deg_from_world_up(
                    forward_world=forward_world,
                    cam_up_world=cam_up_world,
                    world_up=(0.0, 0.0, 1.0),
                )
                dist = float(np.linalg.norm(target_loc - cam_loc))
                elev_deg = math.degrees(math.asin(np.clip(float(offset[2] / max(r_cur, 1e-6)), -1.0, 1.0)))
                selected = (cam_loc, quat_wxyz, dist)
                if abs(roll_deg) <= roll_max_deg:
                    break

            if selected is None:
                continue

            cam_loc, quat_wxyz, dist = selected
            quat = Gf.Quatf(float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))
            _set_transform_attributes(prim, location=cam_loc, orientation=quat)

            if cam_geom:
                cam_geom.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
                cam_geom.GetFocusDistanceAttr().Set(dist)
                cam_geom.GetFStopAttr().Set(0.0)
            self._camera_distance_by_path[cam_path] = float(dist)

        return self.get_pose_debug()

    def _compute_adaptive_distance(
        self,
        *,
        stage,
        cam_path,
        cam_geom,
        bbox,
        cam_res,
        cfg,
        camera_adapt_cfg,
        linear_ratio_target,
        cameras,
    ):
        from pxr import UsdGeom

        if bbox is None:
            return getattr(cfg, "mv_radius", 2.0)

        extent = bbox.get("extent")
        diag = float(bbox.get("diag", 0.0) or 0.0)
        if extent is None or not np.all(np.isfinite(extent)) or diag <= 1e-6:
            return getattr(cfg, "mv_radius", 2.0)

        linear_ratio_target = max(1e-3, float(linear_ratio_target))
        target_metric = str(camera_adapt_cfg.get("target_metric", "area")).lower()

        enable_cache = camera_adapt_cfg.get("enable_cache", True)
        focal_for_key = None
        if cam_geom is not None:
            try:
                focal_for_key = float(cam_geom.GetFocalLengthAttr().Get())
            except Exception:
                focal_for_key = None

        cache_key = f"{cam_path}|f={focal_for_key:.6f}" if focal_for_key is not None else (cam_path if cam_path else "default")
        fov_x = fov_y = None
        if enable_cache and cache_key in self._cam_fov_cache:
            fov_x, fov_y = self._cam_fov_cache[cache_key]

        if fov_x is None or fov_y is None:
            try:
                camera_geom = cam_geom
                if camera_geom is None:
                    cam_path_ref = cam_path[0] if isinstance(cam_path, list) else cam_path
                    if not cam_path_ref and cameras:
                        first_cam = cameras[0]
                        cam_path_ref = first_cam[0] if isinstance(first_cam, list) else first_cam
                    if cam_path_ref:
                        cam_prim = stage.GetPrimAtPath(cam_path_ref)
                        camera_geom = UsdGeom.Camera(cam_prim) if cam_prim and cam_prim.IsValid() else None
                if camera_geom:
                    focal_len = camera_geom.GetFocalLengthAttr().Get()
                    h_ap = camera_geom.GetHorizontalApertureAttr().Get()
                    v_ap = camera_geom.GetVerticalApertureAttr().Get()
                    if (not v_ap or v_ap <= 0) and h_ap:
                        v_ap = h_ap * float(cam_res[1]) / max(float(cam_res[0]), 1.0)
                    if focal_len and h_ap and v_ap:
                        fov_x = 2.0 * math.atan(0.5 * h_ap / float(focal_len))
                        fov_y = 2.0 * math.atan(0.5 * v_ap / float(focal_len))
                        if enable_cache:
                            self._cam_fov_cache[cache_key] = (fov_x, fov_y)
            except Exception:
                pass

        if fov_x is None or fov_y is None:
            h_ap_default = 20.955
            focal_default = 24.0
            v_ap_default = h_ap_default * float(cam_res[1]) / max(float(cam_res[0]), 1.0)
            fov_x = 2.0 * math.atan(0.5 * h_ap_default / focal_default)
            fov_y = 2.0 * math.atan(0.5 * v_ap_default / focal_default)

        fov_x = max(fov_x, 1e-3)
        fov_y = max(fov_y, 1e-3)

        horiz_extent = float(math.sqrt(float(extent[0]) ** 2 + float(extent[1]) ** 2))
        vert_extent = float(abs(extent[2])) if len(extent) > 2 else 0.0

        candidates = []
        if horiz_extent > 0:
            candidates.append(horiz_extent / (2.0 * linear_ratio_target * math.tan(fov_x / 2.0)))
        if vert_extent > 0:
            candidates.append(vert_extent / (2.0 * linear_ratio_target * math.tan(fov_y / 2.0)))
        if diag > 0 and target_metric != "long_side":
            fov_min = min(fov_x, fov_y)
            candidates.append(diag / (2.0 * linear_ratio_target * math.tan(fov_min / 2.0)))

        base_radius = max(candidates) if candidates else getattr(cfg, "mv_radius", 2.0)
        distance_scale = float(camera_adapt_cfg.get("distance_scale", 1.0))
        if distance_scale > 0.0:
            base_radius *= distance_scale

        d_clamp = camera_adapt_cfg.get("d_clamp", [0.1, 10.0])
        if not isinstance(d_clamp, (list, tuple)) or len(d_clamp) < 2:
            d_clamp = [0.1, 10.0]
        d_min, d_max = float(d_clamp[0]), float(d_clamp[1])

        base_radius = float(np.clip(base_radius, d_min, d_max))
        dist_jitter = float(camera_adapt_cfg.get("dist_jitter", 0.0))
        if dist_jitter > 0.0:
            base_radius *= (1.0 + self._rng.uniform(-dist_jitter, dist_jitter))

        return float(np.clip(base_radius, d_min, d_max))
