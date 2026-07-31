#!/usr/bin/env python3
from __future__ import annotations

"""
Collect initial states for USD garment assets using Batched Parallel Processing.
Simplified version using unfold_all.envs.Env.

This script:
1. Finds all *_obj.usd files in the assets directory.
2. Randomly shuffles them (Random Traversal).
3. Batches them into chunks of size `num_envs`.
4. Uses Env with `random_asset_assignment=False` to process each batch in parallel.
5. Saves 'init_pos.npy' (vertices) and 'init_pos_top_view.png' (RGB) to the asset directory.

Usage:
    python tools/assets/collect_initial_states.py --assets_root /path/to/assets --num_envs 8
"""

import sys
import os
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Update sys.path to ensure unfold_all is importable
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

if PROJECT_ROOT.exists() and str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Handle Isaac Lab path
DEFAULT_ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH")).resolve()
if DEFAULT_ISAACLAB_ROOT.exists():
    ISAACLAB_SOURCE = DEFAULT_ISAACLAB_ROOT / "source"
    if ISAACLAB_SOURCE.exists() and str(ISAACLAB_SOURCE) not in sys.path:
        sys.path.append(str(ISAACLAB_SOURCE))

from isaaclab.app import AppLauncher

def main():
    # -------------------------------------------------------------------------- #
    # 1. Argument Parsing & App Launch
    # -------------------------------------------------------------------------- #
    parser = argparse.ArgumentParser(description="Collect initial states for USD garment assets")
    
    parser.add_argument(
        "--assets_root", type=str, default="data/assets",
        help="Root directory containing garment USD files."
    )
    parser.add_argument(
        "--spawn_height", type=float, default=0.75,
        help="Spawn height in meters (default: 1.5)"
    )

    parser.add_argument(
        "--voxel_size", type=float, default=0.01,
        help="Voxel size (meters) for grid downsampling; target spacing (default: 0.01 m = 1 cm)."
    )

    parser.add_argument(
        "--num_envs", type=int, default=8,
        help="Number of parallel environments."
    )
    
    parser.add_argument(
        "--resume", action="store_true", 
        help="Skip processing for assets that already have init_pos.npy generated."
    )
    
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # Force enable cameras for rendering
    args.enable_cameras = True
    
    # Launch the Isaac Sim app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # -------------------------------------------------------------------------- #
    # 2. Lazy Imports (Must happen after App Launch)
    # -------------------------------------------------------------------------- #
    import unfold
    from unfold.simulation.env import Env
    from unfold.simulation.env import EnvCfg
    from unfold.platform.assets import resolve_assets_root
    from unfold.platform.config_utils import parse_yaml_config
    
    # -------------------------------------------------------------------------- #
    # 5. Configure Environment
    # -------------------------------------------------------------------------- #
    # Load base config from yaml
    yaml_path = PROJECT_ROOT / "configs" / "config.yaml"
    print(f"[INFO] Loading configuration from: {yaml_path}")
    cfg = parse_yaml_config(yaml_path, device=args.device if hasattr(args, 'device') and args.device else "cuda:0", env_cfg_class=EnvCfg)
    
    # Overrides from CLI
    if args.spawn_height is not None:
         cfg.spawn_cfg['center'] = (0.0, 0.0, args.spawn_height)
    
    cfg.spawn_cfg['rot_range_deg'] = [0.0, 0.0, 0.0]
    cfg.spawn_cfg['range'] = [0.0, 0.0, 0.0]
    
    if hasattr(cfg, "scene"):
        cfg.scene.num_envs = args.num_envs
    cfg.num_envs = args.num_envs # Keep both for safety
    
    # Request RGB + position observations
    cfg.obs_types = ["pos", "rgb"]
    cfg.vertex_control_enabled = True 
    cfg.randomize_on_reset = True

    if args.assets_root:
        cfg._resolved_assets_root = Path(args.assets_root).resolve()

    # -------------------------------------------------------------------------- #
    # 6. Initialize Environment
    # -------------------------------------------------------------------------- #
    print(f"\nInitializing Environment ({cfg.num_envs} parallel envs)...")
    
    try:
        env = Env(cfg=cfg, render_mode="rgb_array")
    except Exception as e:
        print(f"Failed to initialize Env: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)

    # -------------------------------------------------------------------------- #
    # 7. Access Asset Manager
    # -------------------------------------------------------------------------- #
    # Env initialization triggers _setup_dynamic_scene which creates _asset_pool
    if not hasattr(env, '_asset_pool') or env._asset_pool is None:
        # Fallback if init didn't create it (shouldn't happen with updated env)
        print("WARNING: AssetPool not initialized, forcing initialization.")
        from unfold.simulation.asset_pool import AssetPool
        env._asset_pool = AssetPool(cfg, cfg._resolved_assets_root if hasattr(cfg, '_resolved_assets_root') else None, "cpu")

    if not env._asset_pool or not env._asset_pool._pool:
        print("ERROR: Environment failed to load any assets.")
        simulation_app.close()
        sys.exit(1)

    all_usd_files = env._asset_pool._pool
    print(f"Environment loaded {len(all_usd_files)} assets from internal pool.")

    # -------------------------------------------------------------------------- #
    # 8. Process Batches - Use actual paths from env manager
    # -------------------------------------------------------------------------- #
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    # Disable shuffle for deterministic sequential traversal
    env._asset_pool.make_batches(cfg.num_envs, shuffle=False)
    num_batches = env._asset_pool.num_batches
    
    # Track processed assets to avoid duplicates from padding in last batch
    processed_paths = set()
    
    print(f"\nStarting Batch Processing: {num_batches} batches")
    
    for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
        
        # Resume Logic: Check if all assets in this batch already have init_pos.npy
        # We need to peek at what sample_indices would return
        peek_indices = env._asset_pool._batches[batch_idx]
        peek_paths = env._asset_pool.get_paths(peek_indices)
        
        if args.resume:
            all_exist = all(
                (Path(p).parent / "init_pos.npy").exists() 
                for p in peek_paths if p not in processed_paths
            )
            if all_exist:
                # Still need to advance the pool index
                env._asset_pool.sample_indices()
                for p in peek_paths:
                    processed_paths.add(p)
                stats['skipped'] += len([p for p in peek_paths if p not in processed_paths])
                continue

        try:
            # Reset environment - env will use sample_indices() internally
            obs, _ = env.reset(options={"switch_asset": True})
            
            # Get ACTUAL loaded USD paths from the manager
            actual_usd_paths = env._garment_manager._env_usd_paths
            
            # Iterate through all environments
            for i in range(cfg.num_envs):
                usd_path_str = actual_usd_paths[i]
                if usd_path_str is None:
                    continue
                    
                usd_path = Path(usd_path_str)
                
                # Skip if already processed (handles padding duplicates)
                if usd_path_str in processed_paths:
                    continue
                processed_paths.add(usd_path_str)
                
                # 1. Save Vertices + sampling mask
                if "pos" in obs and obs["pos"] is not None:
                    pos_data = obs["pos"]
                    pos_mask = obs["pos_mask"]
                    if i < pos_data.shape[0]:
                        # obs["pos"] is now in LOCAL frame (thanks to Env fix)
                        valid_count = int(pos_mask[i].squeeze(-1).sum().item())
                        np_pos = pos_data[i, :valid_count].cpu().numpy()

                        state_file = usd_path.parent / "init_pos.npy"
                        mask_file = usd_path.parent / "sample_mask.npy"

                        if not (args.resume and state_file.exists()):
                            np.save(state_file, np_pos)
                            stats['success'] += 1

                        # Generate sampling mask (voxel grid, keep vertex closest to voxel center)
                        if not (args.resume and mask_file.exists()):
                            voxel_indices = np.floor(np_pos / args.voxel_size).astype(np.int64)
                            chosen: dict[tuple[int, int, int], tuple[int, float]] = {}
                            for idx, (vx, vy, vz) in enumerate(voxel_indices):
                                key = (vx, vy, vz)
                                center = np.array([(vx + 0.5) * args.voxel_size,
                                                   (vy + 0.5) * args.voxel_size,
                                                   (vz + 0.5) * args.voxel_size], dtype=np.float32)
                                dist2 = float(np.sum((np_pos[idx] - center) ** 2))
                                best = chosen.get(key)
                                if best is None or dist2 < best[1]:
                                    chosen[key] = (idx, dist2)

                            sample_mask = np.zeros((np_pos.shape[0], 1), dtype=np.float32)
                            for idx, _ in chosen.values():
                                sample_mask[idx, 0] = 1.0

                            np.save(mask_file, sample_mask)
                    else:
                        print(f"  WARNING: Pos data index out of bounds for {usd_path.name}")
                        stats['failed'] += 1
                else:
                    print(f"  WARNING: 'pos' observation missing for {usd_path.name}")
                    stats['failed'] += 1
                
                # 2. Save Image
                if "rgb" in obs and obs["rgb"] is not None:
                    rgb_batch = obs["rgb"]
                    if i < rgb_batch.shape[0]:
                        rgb_data = rgb_batch[i]
                        
                        if isinstance(rgb_data, torch.Tensor):
                            rgb_data = rgb_data.cpu().numpy()
                            
                        # Ensure uint8
                        if rgb_data.dtype != np.uint8:
                            rgb_data = (np.clip(rgb_data, 0, 1) * 255).astype(np.uint8)
                        
                        # Remove alpha
                        if rgb_data.shape[-1] == 4:
                            rgb_data = rgb_data[..., :3]
                        
                        img_file = usd_path.parent / "init_pos_top_view.png"
                        if not (args.resume and img_file.exists()):
                            Image.fromarray(rgb_data).save(img_file)


        except Exception as e:
            print(f"  ERROR processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            stats['failed'] += cfg.num_envs

    # -------------------------------------------------------------------------- #
    # 8. Cleanup
    # -------------------------------------------------------------------------- #
    print(f"\nProcessing Complete!")
    print(f"Success: {stats['success']}")
    print(f"Failed:  {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    
    simulation_app.close()
    sys.exit(0)

if __name__ == "__main__":
    main()
