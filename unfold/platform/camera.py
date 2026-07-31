import math
import torch

def compute_centers_world(pos, mask, env_origins):
    """pos: (E,N,3) object points/vertices; mask: (E,N,1) valid points; env_origins: (E,3).
    Returns each environment target center (E,3), falling back to the origin when invalid.
    """
    valid = (mask.squeeze(-1) > 0.5).to(pos.dtype)
    counts = valid.sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # (E,1)
    center_local = (pos * valid.unsqueeze(-1)).sum(dim=1) / counts  # (E,3)
    return center_local + env_origins  # World coordinates.

def look_at_quat_world(cam, tgt):
    """
    Generate the same orientation quaternion as render_multiview().

    - Input cam/tgt: (...,3) world coordinates.
    - Output: (w,x,y,z), following the USD/OpenGL camera convention
      with right, up, and -forward as columns. This can be passed directly
      to `TiledCamera.set_world_poses(..., convention="opengl")`.

    The rotation matrix construction matches unfold/render/multiview.py:_quat_look_at:
    columns are right, up, and -forward in OpenGL/USD camera coordinates.
    """
    f = tgt - cam
    f = f / (f.norm(dim=-1, keepdim=True) + 1e-9)
    up = torch.tensor([0, 0, 1], dtype=f.dtype, device=f.device)
    # Avoid being parallel to up.
    parallel = (f @ up).abs() > 0.99
    up = torch.where(parallel.unsqueeze(-1), torch.tensor([0, 1, 0], device=f.device, dtype=f.dtype), up)

    # right = forward x up; up' = right x forward.
    right = torch.cross(f, up)
    right = right / (right.norm(dim=-1, keepdim=True) + 1e-9)
    up2 = torch.cross(right, f)

    # Columns are camera axes expressed in the world frame (x=right, y=up, z=-forward).
    R = torch.stack([right, up2, -f], dim=-1)  # (...,3,3)
    return rotmat_to_quat_wxyz(R)

def rotmat_to_quat_wxyz(R):
    """R (...,3,3) -> (...,4) (w,x,y,z)"""
    t = R[..., 0,0] + R[...,1,1] + R[...,2,2]
    w = torch.sqrt(torch.clamp(t + 1, min=0)) / 2
    x = torch.sqrt(torch.clamp(1 + R[...,0,0] - R[...,1,1] - R[...,2,2], min=0)) / 2
    y = torch.sqrt(torch.clamp(1 - R[...,0,0] + R[...,1,1] - R[...,2,2], min=0)) / 2
    z = torch.sqrt(torch.clamp(1 - R[...,0,0] - R[...,1,1] + R[...,2,2], min=0)) / 2
    # Choose the sign from the largest component.
    cond = (t >= torch.stack([R[...,0,0], R[...,1,1], R[...,2,2]], dim=-1).max(dim=-1).values)
    w = torch.where(cond, w,
         torch.where(R[...,0,0] >= R[...,1,1],
             torch.where(R[...,0,0] >= R[...,2,2],
                 x, z),
             torch.where(R[...,1,1] >= R[...,2,2], y, z)))
    x = torch.sign(R[...,2,1] - R[...,1,2]) * x
    y = torch.sign(R[...,0,2] - R[...,2,0]) * y
    z = torch.sign(R[...,1,0] - R[...,0,1]) * z
    q = torch.stack([w,x,y,z], dim=-1)
    return q / (q.norm(dim=-1, keepdim=True) + 1e-9)

def multiview_poses(centers_world, num_views=6, radius=1.5, elev_deg=35.0):
    """Return a list of length V; each item is (pos, quat) with shapes (E,3)/(E,4)."""
    device = centers_world.device
    E = centers_world.shape[0]
    elev = math.radians(elev_deg)
    ce, se = math.cos(elev), math.sin(elev)
    poses = []
    for v in range(num_views):
        yaw = 2.0 * math.pi * (v / num_views)
        cy, sy = math.cos(yaw), math.sin(yaw)
        offset = torch.tensor([radius*ce*cy, radius*ce*sy, radius*se], device=device, dtype=centers_world.dtype)
        cam_pos = centers_world + offset  # (E,3)
        quat = look_at_quat_world(cam_pos, centers_world)  # (E,4)
        poses.append((cam_pos, quat))
    return poses


# --------------------------------------------------------------------------- #
# Manual USD camera transform helper (fallback when TiledCamera set_world_poses_from_view is unreliable)
# --------------------------------------------------------------------------- #
def _quat_look_at_np(cam, tgt):
    """NumPy look-at implementation matching render_multiview; returns Gf-friendly (w,x,y,z)."""
    import numpy as np
    from pxr import Gf

    c = np.asarray(cam, dtype=np.float64)
    t = np.asarray(tgt, dtype=np.float64)
    f = t - c
    n = np.linalg.norm(f) + 1e-9
    f = f / n
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(f, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    s = np.cross(f, up); s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    # Columns: right, up, -forward.
    R = np.array([[s[0], u[0], -f[0]],
                  [s[1], u[1], -f[1]],
                  [s[2], u[2], -f[2]]], dtype=np.float64)
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = (tr + 1.0) ** 0.5 * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = (1.0 + R[0,0] - R[1,1] - R[2,2]) ** 0.5 * 2.0
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = (1.0 + R[1,1] - R[0,0] - R[2,2]) ** 0.5 * 2.0
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = (1.0 + R[2,2] - R[0,0] - R[1,1]) ** 0.5 * 2.0
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    n = (w*w + x*x + y*y + z*z) ** 0.5 + 1e-12
    return Gf.Quatd(float(w/n), Gf.Vec3d(float(x/n), float(y/n), float(z/n)))


def set_camera_prims_look_at(prim_paths, eyes, targets, stage=None):
    """
    Manually set world poses for USD camera prims when TiledCamera.set_world_poses_from_view is unreliable.

    Args:
        prim_paths (Sequence[str]): Camera prim paths with length N.
        eyes (Tensor/ndarray, shape [N,3]): Camera positions in world coordinates.
        targets (Tensor/ndarray, shape [N,3]): Look-at targets in world coordinates.
        stage (Usd.Stage, optional): Reuse this stage when provided; otherwise fetch the current stage via omni.usd.
    """
    import numpy as np
    from pxr import UsdGeom, Gf
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    eyes_np = np.asarray(eyes, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    for path, eye, tgt in zip(prim_paths, eyes_np, tgt_np):
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            continue
        q = _quat_look_at_np(eye, tgt)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*eye.tolist()))
        orient = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        orient.Set(q)
