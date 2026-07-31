import numpy as np
import math
import torch
from typing import Optional, List, Tuple


def _compute_view_bounds(
    positions_tensor: torch.Tensor,
    view_bounds: Optional[tuple] = None,
) -> tuple[float, float, float, float, float, float]:
    if view_bounds is None:
        min_pos = positions_tensor.min(dim=0)[0]
        max_pos = positions_tensor.max(dim=0)[0]

        mx, my, mz = min_pos[0].item(), min_pos[1].item(), min_pos[2].item()
        Mx, My, Mz = max_pos[0].item(), max_pos[1].item(), max_pos[2].item()

        width = Mx - mx
        height = My - my
        pad_x = width * 0.1 + 0.05
        pad_y = height * 0.1 + 0.05

        min_x, min_y = mx - pad_x, my - pad_y
        max_x, max_y = Mx + pad_x, My + pad_y
        min_z, max_z = mz, Mz
    else:
        min_x, min_y = view_bounds[0]
        max_x, max_y = view_bounds[1]
        min_z = float(positions_tensor[:, 2].min().item())
        max_z = float(positions_tensor[:, 2].max().item())

    width = max(max_x - min_x, 1e-4)
    height = max(max_y - min_y, 1e-4)
    depth = max(max_z - min_z, 1e-4)
    return float(min_x), float(min_y), float(width), float(height), float(min_z), float(depth)


def compute_visible_vertices_cuda(
    positions_tensor: torch.Tensor,
    faces_tensor: torch.Tensor,
    resolution: int = 128,
    view_bounds: Optional[tuple] = None,
) -> tuple[torch.Tensor, tuple[float, float]]:
    """
    Compute visible vertices using PyTorch3D top-down mesh rasterization.

    Returns:
        grid_indices: (resolution, resolution) int64 tensor. 0 for background, (vertex_id + 1) for foreground.
        (scale_x, scale_y): Scale factors used for spatial action mapping.
    """
    try:
        from pytorch3d.renderer.mesh import rasterize_meshes
        from pytorch3d.structures import Meshes
    except Exception as exc:
        raise RuntimeError("PyTorch3D visibility rasterization requires pytorch3d.") from exc

    if positions_tensor.numel() == 0 or faces_tensor.numel() == 0:
        return torch.zeros((resolution, resolution), dtype=torch.int64, device=positions_tensor.device), (1.0, 1.0)

    min_x, min_y, width, height, min_z, depth = _compute_view_bounds(positions_tensor, view_bounds=view_bounds)
    scale_x = resolution / width
    scale_y = resolution / height

    verts = positions_tensor.to(dtype=torch.float32)
    faces = faces_tensor.long()
    num_vertices = int(verts.shape[0])
    valid_faces = (
        (faces[:, 0] >= 0) & (faces[:, 0] < num_vertices) &
        (faces[:, 1] >= 0) & (faces[:, 1] < num_vertices) &
        (faces[:, 2] >= 0) & (faces[:, 2] < num_vertices)
    )
    faces = faces[valid_faces]
    if faces.numel() == 0:
        return torch.zeros((resolution, resolution), dtype=torch.int64, device=positions_tensor.device), (scale_x, scale_y)

    # Map XY to NDC and flip Z so higher cloth points win the z-buffer in top-down view.
    x_ndc = ((verts[:, 0] - min_x) / width) * 2.0 - 1.0
    y_ndc = ((verts[:, 1] - min_y) / height) * 2.0 - 1.0
    z_ndc = 1.0 - ((verts[:, 2] - min_z) / depth)
    verts_ndc = torch.stack([x_ndc, y_ndc, z_ndc], dim=-1).unsqueeze(0)
    meshes = Meshes(verts=verts_ndc, faces=faces.unsqueeze(0))

    pix_to_face, _, bary, _ = rasterize_meshes(
        meshes,
        image_size=int(resolution),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=False,
        clip_barycentric_coords=False,
        cull_backfaces=False,
    )

    pix_to_face = pix_to_face[0, ..., 0]
    bary = bary[0, ..., 0, :]
    grid_indices = torch.zeros((resolution, resolution), dtype=torch.int64, device=positions_tensor.device)

    valid = pix_to_face >= 0
    if valid.any():
        face_ids = pix_to_face[valid]
        face_vids = faces[face_ids]
        dominant = torch.argmax(bary[valid], dim=-1)
        chosen_vid = face_vids.gather(1, dominant.unsqueeze(-1)).squeeze(-1)
        grid_indices[valid] = chosen_vid.long() + 1

    return grid_indices, (scale_x, scale_y)

def sample_visible_pair_fast(
    pos: torch.Tensor, 
    faces: torch.Tensor, 
    min_grasp_dist: float, 
    max_grasp_dist: float,
    resolution: int = 128,
    max_retries: int = 10,
) -> Optional[torch.Tensor]:
    """
    Optimized sampling: first randomly sample a (scale, angle), then find pairs.
    
    This is O(1) in terms of scale/angle combinations, compared to O(S*A) for
    the original sample_visible_pairs.
    
    Args:
        pos: (N, 3) vertex positions.
        faces: (F, 3) mesh faces.
        min_grasp_dist: Minimum distance for grasping (in world units).
        max_grasp_dist: Maximum distance for grasping (in world units).
        resolution: Resolution of the visibility grid.
        max_retries: Maximum number of retries if no valid pair found.
        
    Returns:
        (2,) tensor of vertex indices [id1, id2], or None if no valid pair found.
    """
    device = pos.device
    
    # 1. Compute visibility grid once
    grid, (sx, sy) = compute_visible_vertices_cuda(pos, faces, resolution=resolution)
    H, W = grid.shape
    
    if grid.max() == 0:
        return None  # No visible vertices
    
    for _ in range(max_retries):
        # 2. Randomly sample scale and angle
        rand_scale = torch.rand(1, device=device) * (max_grasp_dist - min_grasp_dist) + min_grasp_dist
        rand_angle = torch.rand(1, device=device) * 2 * np.pi
        
        # 3. Compute pixel offset
        dx_m = rand_scale * torch.cos(rand_angle)
        dy_m = rand_scale * torch.sin(rand_angle)
        
        if math.isnan(sx) or math.isnan(sy):
            continue
            
        dx_px = int((dx_m * sx).item())
        dy_px = int((dy_m * sy).item())
        
        if dx_px == 0 and dy_px == 0:
            continue
        
        # 4. Find valid pairs at this offset
        slice_y_start = max(0, -dy_px)
        slice_y_end = min(H, H - dy_px)
        slice_x_start = max(0, -dx_px)
        slice_x_end = min(W, W - dx_px)
        
        if slice_y_start >= slice_y_end or slice_x_start >= slice_x_end:
            continue
        
        grid_p1 = grid[slice_y_start:slice_y_end, slice_x_start:slice_x_end]
        
        target_y_start = slice_y_start + dy_px
        target_y_end = slice_y_end + dy_px
        target_x_start = slice_x_start + dx_px
        target_x_end = slice_x_end + dx_px
        
        grid_p2 = grid[target_y_start:target_y_end, target_x_start:target_x_end]
        
        # Valid mask: both non-background
        valid = (grid_p1 > 0) & (grid_p2 > 0)
        
        if valid.any():
            # 5. Randomly select one valid pair
            valid_indices = valid.nonzero()  # (K, 2)
            num_valid = valid_indices.shape[0]
            rand_idx = torch.randint(0, num_valid, (1,), device=device).item()
            
            py, px = valid_indices[rand_idx]
            v1 = grid_p1[py, px].item() - 1  # grid stores vid+1
            v2 = grid_p2[py, px].item() - 1
            
            return torch.tensor([v1, v2], dtype=torch.long, device=device)
    
    return None  # No valid pair found after retries


def sample_visible_pairs(
    pos: torch.Tensor, 
    faces: torch.Tensor, 
    min_grasp_dist: float, 
    max_grasp_dist: float,
    num_scales: int = 5, 
    num_angles: int = 8,
    resolution: int = 256
) -> List[torch.Tensor]:
    """
    Computes visible vertices and randomly samples vertex pairs using Spatial Action Mapping.
    
    Args:
        pos: (N, 3) vertex positions.
        faces: (F, 3) mesh faces.
        min_grasp_dist: Minimum distance for grasping (in world units).
        max_grasp_dist: Maximum distance for grasping (in world units).
        num_scales: Number of scales to sample.
        num_angles: Number of angles to sample.
        resolution: Resolution of the visibility grid.
        
    Returns:
        List of (N, 2) tensors containing pairs of vertex indices.
    """
    
    # 1. Compute Visibility Grid (CUDA)
    # grid: (H, W) tensor of vertex indices (1-based, 0 is background)
    grid, (sx, sy) = compute_visible_vertices_cuda(
        pos, faces, resolution=resolution
    )
    # 2. Spatial Action Mapping
    device = grid.device
    scales = torch.linspace(min_grasp_dist, max_grasp_dist, num_scales, device=device)
    angles = torch.linspace(0, 2 * np.pi, num_angles + 1, device=device)[:-1] 
    
    H, W = grid.shape
    valid_configs = [] # List of (N, 2) tensors
    
    for d in scales:
        for theta in angles:
            # Compute offset in pixels
            dx_m = d * torch.cos(theta)
            dy_m = d * torch.sin(theta)
            
            # Check for NaNs before conversion
            if torch.isnan(dx_m) or torch.isnan(dy_m) or math.isnan(sx) or math.isnan(sy):
                 # Only warn once to avoid spamming
                 if not hasattr(sample_visible_pairs, "_nan_warned"):
                     print(f"[WARN] NaN detected in sample_visible_pairs: dx_m={dx_m}, sx={sx}")
                     sample_visible_pairs._nan_warned = True
                 continue

            dx_px = int((dx_m * sx).item())
            dy_px = int((dy_m * sy).item())
            
            if dx_px == 0 and dy_px == 0:
                continue
                
            # Shift Strategy:
            # Find pairs (u, v) and (u + dx, v + dy)
            # Valid P1 source range:
            slice_y_start = max(0, -dy_px)
            slice_y_end = min(H, H - dy_px)
            slice_x_start = max(0, -dx_px)
            slice_x_end = min(W, W - dx_px)
            
            if slice_y_start >= slice_y_end or slice_x_start >= slice_x_end:
                continue
                
            # P1 patch
            grid_p1 = grid[slice_y_start:slice_y_end, slice_x_start:slice_x_end]
            
            # P2 patch (shifted target)
            target_y_start = slice_y_start + dy_px
            target_y_end = slice_y_end + dy_px
            target_x_start = slice_x_start + dx_px
            target_x_end = slice_x_end + dx_px
            
            grid_p2 = grid[target_y_start:target_y_end, target_x_start:target_x_end]
            
            # Check valid mask (both non-background)
            # Note: grid stores (vid + 1)
            valid = (grid_p1 > 0) & (grid_p2 > 0)
            
            if valid.any():
                    v1 = grid_p1[valid] - 1
                    v2 = grid_p2[valid] - 1
                    p = torch.stack([v1, v2], dim=1)
                    valid_configs.append(p)

    if not valid_configs:
        return []
    
    # Return ALL valid pairs grouped by configuration
    return valid_configs
