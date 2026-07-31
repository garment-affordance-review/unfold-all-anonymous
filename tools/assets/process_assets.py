import sys
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

# -------------------------------------------------------------------------- #
# 1. Environment dependency checks
# -------------------------------------------------------------------------- #
try:
    from pxr import Usd, UsdGeom
except ImportError:
    print("[ERROR] pxr (USD) library not found.")
    print("Install usd-core (pip install usd-core) or run inside the Isaac Lab environment.")
    sys.exit(1)

try:
    from timm.models.vision_transformer import Block
except ImportError:
    print("[ERROR] timm library not found. Run: pip install timm")
    sys.exit(1)

# -------------------------------------------------------------------------- #
# 2. Point-MAE model definition with contiguous-layout fixes
# -------------------------------------------------------------------------- #

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

class PointNetPatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=384, patch_size=32, num_groups=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_groups = num_groups
        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, embed_dim, 1)
        )

    def forward(self, x):
        x = x.contiguous()
        B, M, C = x.shape
        center_idx = farthest_point_sample(x, self.num_groups)
        centers = index_points(x, center_idx) 
        dists = square_distance(x, centers).transpose(1, 2).contiguous()
        _, idx = dists.sort(dim=-1)
        idx = idx[:, :, :self.patch_size]
        batch_indices = torch.arange(B, dtype=torch.long).to(x.device).view(B, 1, 1)
        idx_base = idx + batch_indices * M
        neighborhood = x.view(B*M, 3)[idx_base.view(-1)].view(B, self.num_groups, self.patch_size, 3).contiguous()
        neighborhood = neighborhood - centers.unsqueeze(2) 
        feat = neighborhood.view(B * self.num_groups, self.patch_size, 3).transpose(1, 2).contiguous()
        feat = self.proj(feat)
        feat = torch.max(feat, 2)[0]
        feat = feat.view(B, self.num_groups, self.embed_dim)
        return feat, centers

class PointMAEEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, num_groups=64):
        super().__init__()
        self.num_groups = num_groups
        self.patch_embed = PointNetPatchEmbed(embed_dim=embed_dim, num_groups=num_groups)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_groups, embed_dim))
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=False)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        features, centers = self.patch_embed(x)
        features = features + self.pos_embed
        for block in self.blocks:
            features = block(features)
        features = self.norm(features)
        return features, centers

def three_nn_interpolation(orig_xyz, source_xyz, source_features):
    orig_xyz = orig_xyz.contiguous()
    source_xyz = source_xyz.contiguous()
    B, N, _ = orig_xyz.shape
    dists = square_distance(orig_xyz, source_xyz)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :3], idx[:, :, :3]
    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm 
    C = source_features.shape[-1]
    interpolated_feat = torch.zeros(B, N, C).to(orig_xyz.device)
    for b in range(B):
        feat_batch = source_features[b]
        idx_batch = idx[b]
        gathered_feat = feat_batch[idx_batch].contiguous()
        w = weight[b].unsqueeze(-1)
        interpolated_feat[b] = torch.sum(gathered_feat * w, dim=1)
    return interpolated_feat.transpose(1, 2).contiguous()

# -------------------------------------------------------------------------- #
# 3. Validation logic
# -------------------------------------------------------------------------- #


# -------------------------------------------------------------------------- #
# Category configuration with category-specific thresholds
# -------------------------------------------------------------------------- #

# -------------------------------------------------------------------------- #
# Category configuration with category-specific thresholds
# -------------------------------------------------------------------------- #
CATEGORY_CONFIG = {
    "Dress":    {"min_size": 0.3, "max_size": 1.6, "max_particles": 20000},
    "Tops":     {"min_size": 0.3, "max_size": 1.3, "max_particles": 15000},
    "Trousers": {"min_size": 0.3, "max_size": 1.3, "max_particles": 15000},
    # "Default": Removed. Only specific categories are processed.
}

def get_category_from_path(file_path):
    """Infer category from a path, e.g. 'Dress/Long_Gallus/...' -> 'Dress'."""
    parts = file_path.split(os.sep)
    if len(parts) > 0 and parts[0] in CATEGORY_CONFIG:
        return parts[0]
    return None  # Unclassified

def check_usd_validity(usd_path_str, scale, thresholds):
    """Validate that the USD is loadable and reasonably sized, and return particle count."""
    min_thresh = thresholds["min_size"]
    max_thresh = thresholds["max_size"]
    max_particles = thresholds["max_particles"]
    
    try:
        stage = Usd.Stage.Open(usd_path_str)
        if not stage: return False, "Open Failed", 0
        
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points_attr = mesh.GetPointsAttr().Get()
                
                if points_attr and len(points_attr) > 0:
                    points = np.array(points_attr)
                    particle_count = len(points)  # Number of particles (vertices).
                    
                    if particle_count > max_particles:
                        return False, f"Too Heavy ({particle_count} > {max_particles})", particle_count

                    b_min = points.min(axis=0)
                    b_max = points.max(axis=0)
                    # Compute the maximum side length after scaling.
                    extent = b_max - b_min
                    max_dim = np.max(extent * scale)
                    
                    if max_dim < min_thresh:
                        return False, f"Too Small ({max_dim:.3f}m < {min_thresh}m)", particle_count
                    elif max_dim > max_thresh:
                        return False, f"Too Large ({max_dim:.3f}m > {max_thresh}m)", particle_count
                    else:
                        return True, "Valid", particle_count
                        
        return False, "No Valid Mesh Found", 0
    except Exception as e:
        return False, f"Error: {str(e)}", 0

# -------------------------------------------------------------------------- #
# 4. Feature extraction logic (disabled)
# Offline training has been removed from this repository.
# -------------------------------------------------------------------------- #

# -------------------------------------------------------------------------- #
# 5. Main flow
# -------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Asset Validation Pipeline")
    parser.add_argument("--root", type=str, default="data/assets/cloth", help="Data root directory")
    parser.add_argument("--scale", type=float, default=0.0085, help="Global scale factor for validation")
    parser.add_argument("--fps", type=int, default=2048, help="Input points for Point-MAE (Unused)")
    parser.add_argument("--output-json", type=str, default="valid_assets.json", help="Filename of the valid list")
    
    # New Argument: Category Filter
    parser.add_argument("--categories", nargs="+", default=None, 
                        help="Specific categories to check (e.g. Dress Tops). Default: Check all defined in config.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root):
        print(f"Root path not found: {args.root}")
        sys.exit(1)

    json_save_path = os.path.join(args.root, args.output_json)

    # --- Phase 1: Validation ---
    print("="*60)
    print(f"PHASE 1: Validation")
    print(f"Root: {args.root} | Scale: {args.scale}")
    if args.categories:
        print(f"Target Categories: {args.categories}")
    else:
        print(f"Target Categories: All Defined {list(CATEGORY_CONFIG.keys())}")
    print("="*60)
    
    all_files = []
    # Walk relative to root
    for root, dirs, files in os.walk(args.root):
        for file in files:
            if file.endswith(".usd") and "border" not in file:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, args.root)
                
                # Filter by Category
                category = get_category_from_path(rel_path)
                
                # SKIP if Unclassified (None)
                if category is None:
                    continue
                
                # Filter by CLI Argument
                if args.categories and category not in args.categories:
                    continue
                    
                all_files.append((full_path, rel_path, category))
                
    valid_assets_with_counts = []  # (rel_path, particle_count)
    stats = {"valid": 0, "too_small": 0, "too_large": 0, "too_heavy": 0, "error": 0}
    
    for full_path, rel_path, category in tqdm(all_files, desc="Validating"):
        # Select Thresholds based on Category (Guaranteed to exist now)
        thresholds = CATEGORY_CONFIG[category]
        
        is_valid, reason, particle_count = check_usd_validity(full_path, args.scale, thresholds)
        
        if is_valid:
            valid_assets_with_counts.append((rel_path, particle_count))
            stats["valid"] += 1
        else:
            if "Too Small" in reason: stats["too_small"] += 1
            elif "Too Large" in reason: stats["too_large"] += 1
            elif "Too Heavy" in reason: stats["too_heavy"] += 1
            else: stats["error"] += 1

    # Sort by particle count
    valid_assets_with_counts.sort(key=lambda x: x[1])
    valid_assets = [path for path, _ in valid_assets_with_counts]
    
    # Save JSON
    with open(json_save_path, 'w') as f:
        json.dump(valid_assets, f, indent=4)
        
    print(f"\n[Validation Result]")
    print(f"  Valid: {stats['valid']}")
    print(f"    Too Small: {stats['too_small']}")
    print(f"    Too Large: {stats['too_large']}")
    print(f"    Too Heavy: {stats['too_heavy']}")
    print(f"    Errors:    {stats['error']}")
    if valid_assets_with_counts:
        print(f"  Particle count range: [{valid_assets_with_counts[0][1]}, {valid_assets_with_counts[-1][1]}]")
    print(f"  List saved to: {json_save_path}")

    # Feature extraction skipped...
    print("\n(Feature extraction skipped)")

if __name__ == "__main__":
    main()
