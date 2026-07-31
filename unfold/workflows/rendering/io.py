#!/usr/bin/env python3
"""I/O helpers for the rendering pipeline."""

import json
from pathlib import Path

import numpy as np
import torch

def to_uint8(img):
    if img.dtype in (torch.float32, torch.float64):
        return (img.clamp(0, 1) * 255.0).to(torch.uint8)
    return img.to(torch.uint8) if img.dtype != torch.uint8 else img

def _relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)

def _save_depth_png(depth_np: np.ndarray, out_path: Path):
    from PIL import Image
    depth_mm = np.clip(depth_np * 1000.0, 0.0, 65535.0).astype(np.uint16)
    Image.fromarray(depth_mm).save(out_path)

def _save_mask_png(mask_np: np.ndarray, out_path: Path):
    from PIL import Image
    m = mask_np.copy()
    while m.ndim > 2:
        m = m[..., 1]
    m = m.astype(np.int64)
    m[m < 0] = 0
    mx = m.max() if m.size else 0
    if mx <= 255:
        arr = m.astype(np.uint8)
    elif mx <= 65535:
        arr = m.astype(np.uint16)
    else:
        arr = m.astype(np.uint32)
    Image.fromarray(arr).save(out_path)

def _save_sample(sample_id, sample_dir, rgb_np, depth_np, mask_np,
                 meta_extra, manifest_base):
    """Save one sample triplet and return manifest entry (same as save_multiview_rgbd)."""
    from PIL import Image
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    H = W = None

    if rgb_np is not None:
        if rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[..., :3]
        Image.fromarray(rgb_np.astype(np.uint8)).save(sample_dir / 'rgb.png')
        paths['rgb'] = _relpath(sample_dir / 'rgb.png', manifest_base)
        H, W = rgb_np.shape[0], rgb_np.shape[1]
    # Save depth as .npy only (precise float32 data)
    if depth_np is not None:
        if depth_np.ndim == 3:
            depth_np = depth_np.squeeze(-1)
        np.save(sample_dir / 'depth.npy', depth_np.astype(np.float32))
        paths['depth'] = _relpath(sample_dir / 'depth.npy', manifest_base)
        if H is None:
            H, W = depth_np.shape[:2]

    # Save mask as .png only (space-efficient, easy to visualize)
    if mask_np is not None:
        _save_mask_png(mask_np, sample_dir / 'mask.png')
        paths['mask'] = _relpath(sample_dir / 'mask.png', manifest_base)

    # Prepare meta dict (exclude numpy arrays to avoid JSON serialization errors)
    meta = {
        'id': sample_id,
        'height': int(H) if H else None,
        'width': int(W) if W else None,
        'paths': paths,
    }

    # Add scalar/string metadata only (no numpy arrays)
    if meta_extra:
        for k, v in meta_extra.items():
            if k not in (
                'camera_intrinsics',
                'camera_extrinsics',
                'vertex_positions',
                'face_index',
                'face_vertex_ids',
                'barycentric_weights',
                'nocs_visualization',
                'side_bit',
                'side_visualization',
            ):
                meta[k] = v

    # Save camera intrinsics as JSON (small matrix, human-readable)
    if 'camera_intrinsics' in meta_extra and meta_extra['camera_intrinsics'] is not None:
        K = meta_extra['camera_intrinsics']
        camera_params = {
            'intrinsics': {
                'fx': float(K[0, 0]),
                'fy': float(K[1, 1]),
                'cx': float(K[0, 2]),
                'cy': float(K[1, 2]),
                'matrix': K.tolist()  # Full 3x3 matrix
            }
        }

        # Save camera extrinsics in the same file
        if 'camera_extrinsics' in meta_extra and meta_extra['camera_extrinsics'] is not None:
            E = meta_extra['camera_extrinsics']
            camera_params['extrinsics'] = {
                'matrix': E.tolist(),  # Full 4x4 matrix
                'rotation': E[:3, :3].tolist(),  # 3x3 rotation
                'translation': E[:3, 3].tolist()  # 3x1 translation
            }

        with open(sample_dir / 'camera.json', 'w') as f:
            json.dump(camera_params, f, ensure_ascii=False, indent=2)
        paths['camera'] = _relpath(sample_dir / 'camera.json', manifest_base)

    # Save masked vertex positions as .npy (variable size, many vertices)
    if 'vertex_positions' in meta_extra and meta_extra['vertex_positions'] is not None:
        vertices = meta_extra['vertex_positions']
        np.save(sample_dir / 'vertices.npy', vertices.astype(np.float32))
        paths['vertices'] = _relpath(sample_dir / 'vertices.npy', manifest_base)
        meta['num_vertices'] = int(len(vertices))

    if 'face_index' in meta_extra and meta_extra['face_index'] is not None:
        face_index = np.asarray(meta_extra['face_index'])
        np.save(sample_dir / 'face_index.npy', face_index.astype(np.int32))
        paths['face_index'] = _relpath(sample_dir / 'face_index.npy', manifest_base)
    if 'face_vertex_ids' in meta_extra and meta_extra['face_vertex_ids'] is not None:
        face_vertex_ids = np.asarray(meta_extra['face_vertex_ids'])
        np.save(sample_dir / 'face_vertex_ids.npy', face_vertex_ids.astype(np.int32))
        paths['face_vertex_ids'] = _relpath(sample_dir / 'face_vertex_ids.npy', manifest_base)
        if face_vertex_ids.size:
            visible_ids = np.unique(face_vertex_ids[face_vertex_ids >= 0])
            meta['num_visible_vertex_ids'] = int(len(visible_ids))
        else:
            meta['num_visible_vertex_ids'] = 0
    if 'barycentric_weights' in meta_extra and meta_extra['barycentric_weights'] is not None:
        barycentric_weights = np.asarray(meta_extra['barycentric_weights'])
        np.save(sample_dir / 'barycentric_weights.npy', barycentric_weights)
        paths['barycentric_weights'] = _relpath(sample_dir / 'barycentric_weights.npy', manifest_base)

    # Save NOCS visualization image from init_pos-based RGB features.
    if 'nocs_visualization' in meta_extra and meta_extra['nocs_visualization'] is not None:
        nocs_vis = meta_extra['nocs_visualization']
        from PIL import Image
        Image.fromarray(nocs_vis.astype(np.uint8)).save(sample_dir / 'nocs.png')
        paths['nocs'] = _relpath(sample_dir / 'nocs.png', manifest_base)

    if 'side_bit' in meta_extra and meta_extra['side_bit'] is not None:
        side_bit = np.asarray(meta_extra['side_bit'])
        np.save(sample_dir / 'side_bit.npy', side_bit.astype(np.uint8))
        paths['side_bit'] = _relpath(sample_dir / 'side_bit.npy', manifest_base)

    with open(sample_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    paths['meta'] = _relpath(sample_dir / 'meta.json', manifest_base)

    entry = {'id': sample_id, 'height': meta['height'], 'width': meta['width'], 'paths': paths}
    for k in (
        'asset_sample_id',
        'global_sample_id',
        'asset_dir',
        'episode',
        'env',
        'view',
        'renderer',
        'timestamp',
        'asset_path',
        'usd_path',
        'asset_id',
        'asset_name',
        'camera_path',
        'camera_distance_m',
        'num_vertices',
        'num_visible_vertex_ids',
        'texture_id',
        'patch_size_m',
        'size_ref_m',
        'tile_base',
        'texture_scale',
        'texture_rotate',
        'texture_translate',
        'external_texture_applied',
        'texture_apply_failure_reason',
    ):
        if k in meta:
            entry[k] = meta[k]
    return entry
