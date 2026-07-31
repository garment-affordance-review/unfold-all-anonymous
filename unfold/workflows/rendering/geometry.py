#!/usr/bin/env python3
"""Geometry and rasterization helpers for the rendering pipeline."""

import numpy as np


def _mask_to_bool(mask_np):
    m = np.asarray(mask_np)
    while m.ndim > 2:
        m = m[..., 1]
    return m.astype(bool)


def _pick_projection_axis_signs(mask_np, cam_xyz, depth, K):
    """
    Pick projection axis signs (sx, sy) from {+1,-1}^2 by maximizing overlap with cloth mask.
    """
    sx_best, sy_best = 1.0, 1.0
    if mask_np is None:
        return sx_best, sy_best
    m = _mask_to_bool(mask_np)
    if not m.any():
        return sx_best, sy_best

    h, w = m.shape[:2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X = cam_xyz[:, 0]
    Y = cam_xyz[:, 1]
    Z = np.maximum(depth, 1e-8)

    best_score = -1
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            u = fx * (sx * X / Z) + cx
            v = fy * (sy * Y / Z) + cy
            ui = np.rint(u).astype(np.int32)
            vi = np.rint(v).astype(np.int32)
            inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & np.isfinite(u) & np.isfinite(v)
            score = int(m[vi[inb], ui[inb]].sum()) if inb.any() else 0
            if score > best_score:
                best_score = score
                sx_best, sy_best = sx, sy
    return sx_best, sy_best


def _project_vertices_to_image(vertices_world, K, w2c, mask_np=None):
    """Project world-space vertices to image plane, returning pixel xy and camera-space depth."""
    if vertices_world is None or K is None or w2c is None:
        return None, None
    v = np.asarray(vertices_world, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
        return None, None

    ones = np.ones((v.shape[0], 1), dtype=np.float32)
    v_h = np.concatenate([v, ones], axis=1)
    cam = (v_h @ np.asarray(w2c, dtype=np.float32).T)[:, :3]
    z = cam[:, 2]
    eps = 1e-6
    pos_count = int((z > eps).sum())
    neg_count = int((z < -eps).sum())
    if pos_count == 0 and neg_count == 0:
        return None, None

    depth = (-z) if neg_count > pos_count else z
    valid = depth > eps
    if not valid.any():
        return None, None

    cam = cam[valid]
    depth = depth[valid]
    K = np.asarray(K, dtype=np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    sx, sy = _pick_projection_axis_signs(mask_np, cam, depth, K)
    uv = np.empty((cam.shape[0], 2), dtype=np.float32)
    uv[:, 0] = fx * (sx * cam[:, 0] / depth) + cx
    uv[:, 1] = fy * (sy * cam[:, 1] / depth) + cy
    return uv.astype(np.float32), valid


def _count_mask_overlap(mask_bool, uv):
    if uv is None or uv.shape[0] == 0:
        return 0
    h, w = mask_bool.shape[:2]
    ui = np.rint(uv[:, 0]).astype(np.int32)
    vi = np.rint(uv[:, 1]).astype(np.int32)
    inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & np.isfinite(uv).all(axis=1)
    if not inb.any():
        return 0
    return int(mask_bool[vi[inb], ui[inb]].sum())


def _compute_vertex_nocs(init_pos_local, vertex_valid_mask):
    init_local = np.asarray(init_pos_local, dtype=np.float32)
    if init_local.ndim != 2 or init_local.shape[1] != 3:
        return None
    n_v = int(init_local.shape[0])
    valid_v = np.ones((n_v,), dtype=bool)
    if vertex_valid_mask is not None:
        valid_v = np.asarray(vertex_valid_mask).astype(bool).squeeze()
        if valid_v.shape[0] != n_v:
            valid_v = np.ones((n_v,), dtype=bool)
    if valid_v.any():
        init_valid = init_local[valid_v]
    else:
        init_valid = init_local
    if init_valid.shape[0] == 0:
        return np.zeros_like(init_local, dtype=np.float32)
    i_min = init_valid.min(axis=0)
    i_max = init_valid.max(axis=0)
    i_den = np.maximum(i_max - i_min, 1e-8)
    return np.clip((init_local - i_min[None, :]) / i_den[None, :], 0.0, 1.0).astype(np.float32)


def _resolve_projection_config(vertices_local, vertex_valid_mask, K, w2c, env_origin, mask_bool):
    v_local = np.asarray(vertices_local, dtype=np.float32)
    if v_local.ndim != 2 or v_local.shape[1] != 3 or v_local.shape[0] == 0:
        return None

    n_v = int(v_local.shape[0])

    env_origin = np.asarray(env_origin, dtype=np.float32).reshape(1, 3)
    world_candidates = (v_local + env_origin, v_local)

    K = np.asarray(K, dtype=np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    best = None
    eps = 1e-6
    for use_local_frame, v_world in ((False, world_candidates[0]), (True, world_candidates[1])):
        ones = np.ones((n_v, 1), dtype=np.float32)
        v_h = np.concatenate([v_world, ones], axis=1)
        cam = (v_h @ np.asarray(w2c, dtype=np.float32).T)[:, :3]
        z = cam[:, 2]

        for depth_sign in (1.0, -1.0):
            depth = depth_sign * z
            front = depth > eps
            visible = front
            if not visible.any():
                continue

            sx, sy = _pick_projection_axis_signs(mask_bool, cam[visible], depth[visible], K)
            uv = np.empty((n_v, 2), dtype=np.float32)
            uv[:, 0] = fx * (sx * cam[:, 0] / np.maximum(depth, eps)) + cx
            uv[:, 1] = fy * (sy * cam[:, 1] / np.maximum(depth, eps)) + cy
            score = _count_mask_overlap(mask_bool, uv[visible])
            cand = {
                "score": score,
                "use_local_frame": use_local_frame,
                "depth_sign": depth_sign,
                "sx": sx,
                "sy": sy,
                "v_world": v_world,
                "cam": cam,
                "depth": depth,
                "visible": visible,
                "uv": uv,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
    return best


def _build_nocs_from_face_bary(face_vertex_ids, barycentric_weights, vertex_nocs, valid_pixels):
    h, w = face_vertex_ids.shape[:2]
    nocs_img = np.zeros((h, w, 3), dtype=np.float32)
    if not np.any(valid_pixels):
        return np.zeros((h, w, 3), dtype=np.uint8)

    tri_vid = face_vertex_ids[valid_pixels]
    tri_nocs = vertex_nocs[tri_vid]  # (N,3,3)
    bw = barycentric_weights[valid_pixels][..., None]  # (N,3,1)
    pix_nocs = (bw * tri_nocs).sum(axis=1)
    nocs_img[valid_pixels] = np.clip(pix_nocs, 0.0, 1.0)
    return np.clip(nocs_img * 255.0, 0.0, 255.0).astype(np.uint8)


def _compute_side_bit_visualization(
    *,
    vertices_world,
    faces,
    face_index,
    mask_bool,
    w2c,
):
    """
    Estimate whether the visible cloth surface corresponds to the authored outer side.

    Convention:
      - side_bit == 1 when the visible face normal points toward the camera
      - side_bit == 0 otherwise
      - background == 255

    This assumes mesh winding is consistent for the garment asset.
    """
    if vertices_world is None or faces is None or face_index is None or w2c is None:
        return None, None

    v_world = np.asarray(vertices_world, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    face_idx = np.asarray(face_index, dtype=np.int64)
    valid_pix = np.asarray(mask_bool).astype(bool) & (face_idx >= 0)

    if v_world.ndim != 2 or v_world.shape[1] != 3:
        return None, None
    if f.ndim != 2 or f.shape[1] != 3:
        return None, None
    if not np.any(valid_pix):
        return None, None

    n_v = int(v_world.shape[0])
    valid_faces = (
        (f[:, 0] >= 0) & (f[:, 0] < n_v) &
        (f[:, 1] >= 0) & (f[:, 1] < n_v) &
        (f[:, 2] >= 0) & (f[:, 2] < n_v)
    )
    f = f[valid_faces]
    if f.shape[0] == 0:
        return None, None

    tri = v_world[f]
    edge_1 = tri[:, 1] - tri[:, 0]
    edge_2 = tri[:, 2] - tri[:, 0]
    face_normal_world = np.cross(edge_1, edge_2)
    face_norm = np.linalg.norm(face_normal_world, axis=1, keepdims=True)
    good_normal = face_norm.squeeze(-1) > 1e-12
    if not np.any(good_normal):
        return None, None
    face_normal_world = face_normal_world.copy()
    face_normal_world[good_normal] = face_normal_world[good_normal] / face_norm[good_normal]

    w2c_np = np.asarray(w2c, dtype=np.float32)
    rot = w2c_np[:3, :3]
    trans = w2c_np[:3, 3]
    tri_center_cam = (tri.mean(axis=1) @ rot.T) + trans[None, :]
    face_normal_cam = face_normal_world @ rot.T
    to_camera_cam = -tri_center_cam
    ray_norm = np.linalg.norm(to_camera_cam, axis=1, keepdims=True)
    good_ray = ray_norm.squeeze(-1) > 1e-12
    if not np.any(good_ray):
        return None, None
    to_camera_cam = to_camera_cam.copy()
    to_camera_cam[good_ray] = to_camera_cam[good_ray] / ray_norm[good_ray]

    facing_score = np.sum(face_normal_cam * to_camera_cam, axis=1)
    face_side_bit = (facing_score > 0.0).astype(np.uint8)

    side_bit = np.full(face_idx.shape, 255, dtype=np.uint8)
    side_bit[valid_pix] = face_side_bit[face_idx[valid_pix]]

    side_rgb = np.zeros((*face_idx.shape, 3), dtype=np.uint8)
    side_rgb[valid_pix & (side_bit == 1)] = np.array([48, 208, 96], dtype=np.uint8)
    side_rgb[valid_pix & (side_bit == 0)] = np.array([220, 76, 60], dtype=np.uint8)
    return side_bit, side_rgb


def _rasterize_face_index_and_barycentric_from_mesh(
    mask_np,
    vertices_local,
    init_pos_local,
    vertex_valid_mask,
    faces,
    K,
    w2c,
    env_origin,
    image_hw=None,
    projection_overlap_threshold=0.50,
):
    """
    Rasterize visible cloth mesh via PyTorch3D and output face/barycentric correspondence.

    Returns:
      - face_index: (H, W) int32, background=-1
      - face_vertex_ids: (H, W, 3) int32, background=-1
      - barycentric_weights: (H, W, 3) float32, background=0
      - nocs_visualization: (H, W, 3) uint8
    """
    if K is None or w2c is None:
        return None, None, None, None
    if vertices_local is None or init_pos_local is None or faces is None:
        return None, None, None, None

    try:
        import torch
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.structures import Meshes
        from pytorch3d.utils import cameras_from_opencv_projection
    except Exception as exc:
        raise RuntimeError(
            "PyTorch3D rasterization requires both 'torch' and 'pytorch3d'. "
            "Please install them in the current runtime environment."
        ) from exc

    has_seg_mask = mask_np is not None
    if has_seg_mask:
        m = _mask_to_bool(mask_np)
        h, w = m.shape[:2]
    else:
        if image_hw is None:
            return None, None, None, None
        h, w = int(image_hw[0]), int(image_hw[1])
        m = np.ones((h, w), dtype=bool)
    if h <= 0 or w <= 0:
        return None, None, None, None

    v_local = np.asarray(vertices_local, dtype=np.float32)
    if v_local.ndim != 2 or v_local.shape[1] != 3 or v_local.shape[0] == 0:
        return None, None, None, None

    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] == 0:
        return None, None, None, None
    n_v = int(v_local.shape[0])
    in_range_face = (
        (f[:, 0] >= 0) & (f[:, 0] < n_v) &
        (f[:, 1] >= 0) & (f[:, 1] < n_v) &
        (f[:, 2] >= 0) & (f[:, 2] < n_v)
    )
    f = f[in_range_face]
    if f.shape[0] == 0:
        return None, None, None, None

    proj = _resolve_projection_config(
        vertices_local=v_local,
        vertex_valid_mask=vertex_valid_mask,
        K=K,
        w2c=w2c,
        env_origin=env_origin,
        mask_bool=m,
    )
    if proj is None:
        return None, None, None, None

    # Consistency check against legacy projection to catch camera-frame mismatch.
    if has_seg_mask:
        old_uv, old_valid = _project_vertices_to_image(proj["v_world"], K, w2c, mask_np=m)
        old_overlap = _count_mask_overlap(m, old_uv) if old_uv is not None else 0
        new_overlap = _count_mask_overlap(m, proj["uv"][proj["visible"]])
        if old_overlap >= 16:
            ratio = float(new_overlap) / float(max(old_overlap, 1))
            if ratio < float(projection_overlap_threshold):
                raise RuntimeError(
                    f"PyTorch3D projection consistency check failed: overlap_ratio={ratio:.3f} "
                    f"new_overlap={new_overlap} old_overlap={old_overlap}"
                )

    v_world = np.asarray(proj["v_world"], dtype=np.float32)
    sx = float(proj["sx"])
    sy = float(proj["sy"])
    sz = float(proj["depth_sign"])
    axis_flip = np.eye(4, dtype=np.float32)
    axis_flip[0, 0] = sx
    axis_flip[1, 1] = sy
    axis_flip[2, 2] = sz
    w2c_adj = axis_flip @ np.asarray(w2c, dtype=np.float32)

    ones = np.ones((v_world.shape[0], 1), dtype=np.float32)
    v_h = np.concatenate([v_world, ones], axis=1)
    cam_xyz = (v_h @ w2c_adj.T)[:, :3]
    visible = cam_xyz[:, 2] > 1e-6
    # Keep triangles whenever they have at least one front-facing vertex.
    faces_visible = visible[f].any(axis=1)
    f_vis = f[faces_visible]
    if f_vis.shape[0] == 0:
        return None, None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verts_world_t = torch.from_numpy(v_world).to(device=device, dtype=torch.float32)
    faces_t = torch.from_numpy(f_vis).to(device=device, dtype=torch.int64)
    meshes = Meshes(verts=[verts_world_t], faces=[faces_t])

    R = torch.from_numpy(w2c_adj[:3, :3]).to(device=device, dtype=torch.float32).unsqueeze(0)
    T = torch.from_numpy(w2c_adj[:3, 3]).to(device=device, dtype=torch.float32).unsqueeze(0)
    K_t = torch.from_numpy(np.asarray(K, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
    image_size_t = torch.tensor([[float(h), float(w)]], device=device, dtype=torch.float32)
    cameras = cameras_from_opencv_projection(
        R=R,
        tvec=T,
        camera_matrix=K_t,
        image_size=image_size_t,
    )
    raster_settings = RasterizationSettings(
        image_size=(int(h), int(w)),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,
        clip_barycentric_coords=False,
    )
    fragments = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)(meshes)

    pix_to_face = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy().astype(np.int64)
    bary = fragments.bary_coords[0, ..., 0, :].detach().cpu().numpy().astype(np.float32)

    face_index = np.full((h, w), -1, dtype=np.int32)
    face_vertex_ids = np.full((h, w, 3), -1, dtype=np.int32)
    barycentric_weights = np.zeros((h, w, 3), dtype=np.float32)

    valid_pix = m & (pix_to_face >= 0)
    if not np.any(valid_pix):
        return None, None, None, None

    face_index[valid_pix] = pix_to_face[valid_pix].astype(np.int32)
    face_vertex_ids[valid_pix] = f_vis[pix_to_face[valid_pix]].astype(np.int32)

    bary_valid = bary[valid_pix]
    bary_valid = np.clip(bary_valid, 0.0, 1.0)
    bsum = bary_valid.sum(axis=1, keepdims=True)
    ok = bsum.squeeze(-1) > 1e-8
    bary_valid[ok] = bary_valid[ok] / bsum[ok]
    barycentric_weights[valid_pix] = bary_valid

    vertex_nocs = _compute_vertex_nocs(init_pos_local, vertex_valid_mask)
    nocs_u8 = _build_nocs_from_face_bary(
        face_vertex_ids=face_vertex_ids,
        barycentric_weights=barycentric_weights,
        vertex_nocs=vertex_nocs,
        valid_pixels=valid_pix,
    )
    # Keep visualization as raw rasterized NOCS. Hole filling can introduce artifacts.

    return face_index, face_vertex_ids, barycentric_weights.astype(np.float32), nocs_u8
