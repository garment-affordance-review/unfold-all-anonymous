
import argparse
import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import signal

# Import Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# Handle Ctrl+C for clean exit
def signal_handler(sig, frame):
    print("\n[INFO] Cancelled by user. Exiting...")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# 1. Parse Args (Must be before AppLauncher)
parser = argparse.ArgumentParser(description="Validate PhysX Cooking for Assets")
parser.add_argument("--assets_list", type=str, default="data/assets/cloth/valid_assets.json", help="Path to valid_assets.json")
parser.add_argument("--output_json", type=str, default="data/assets/cloth/valid_assets_cooked.json", help="Output validated list")
parser.add_argument("--start_idx", type=int, default=0, help="Start index")
parser.add_argument("--end_idx", type=int, default=None, help="End index")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to validate in parallel")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force headless if not set (though default is True above, safer to enforce)
if args.headless is None:
    args.headless = True

# 2. Launch App
print(f"[INFO] Launching SimulationApp (Headless: {args.headless})...")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 3. Imports
import omni.usd
import omni.timeline
import carb
from pxr import Usd, UsdGeom, Sdf, Gf, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.prims import SingleParticleSystem, SingleClothPrim
from isaacsim.core.api.materials import ParticleMaterial
from isaacsim.core.utils.prims import is_prim_path_valid, create_prim, delete_prim
from isaacsim.core.utils.stage import create_new_stage

# Suppress warnings
carb.settings.get_settings().set("/log/outputStreamLevel", "error")

def main():
    if not os.path.exists(args.assets_list):
        print(f"[ERROR] Asset list not found: {args.assets_list}")
        return

    with open(args.assets_list, "r") as f:
        all_assets = json.load(f)
    
    print(f"[INFO] Loaded {len(all_assets)} assets from {args.assets_list}")
    
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(all_assets)
    assets_to_validate = all_assets[start:end]
    
    print(f"[INFO] Validating assets: {start} to {end} (Total: {len(assets_to_validate)})")

    # Setup World
    sim_params = {
        "use_gpu": True,
        "worker_thread_count": 4,
        "enable_gpu_dynamics": True, # Critical for particles
        "broadphase_type": "MBP"
    }
    world = World(stage_units_in_meters=1.0, backend="torch", device="cuda:0", sim_params=sim_params)
    world.scene.add_default_ground_plane()
    stage = world.stage
    
    # Setup Shared Physics Logic
    # We create ONE particle system and reuse it
    sys_path = "/World/ParticleSystem"
    mat_path = "/World/ParticleMaterial"
    
    # Manual Override (Removed, relying on sim_params)
    physx_scene_path = world.get_physics_context().prim_path
    print(f"[INFO] Physics Scene Path: {physx_scene_path}")
    
    # Material
    ParticleMaterial(
        prim_path=mat_path,
        friction=0.5,
        particle_friction_scale=1.0,
    )
    
    # System
    particle_system = SingleParticleSystem(
        prim_path=sys_path,
        simulation_owner=world.get_physics_context().prim_path,
        particle_system_enabled=True,
        max_velocity=10.0,
        global_self_collision_enabled=False # Optimization
    )
    particle_system.apply_particle_material(ParticleMaterial(prim_path=mat_path))
    
    world.reset()
    
    valid_assets = []
    failed_assets = []
    
    pbar = tqdm(assets_to_validate, desc="Validating")
    
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()

    for asset_rel_path in pbar:
        # Construct full path
        # Assume assets_list is relative to repo root or somewhat absolute
        # The user provided valid_assets.json usually has relative paths to `data/assets/cloth`?
        # Let's try to resolve it.
        # Assuming script is run from repo root.
        
        # Try finding the file
        if os.path.exists(asset_rel_path):
             usd_path = os.path.abspath(asset_rel_path)
        elif os.path.exists(os.path.join("data/assets/cloth", asset_rel_path)):
             usd_path = os.path.abspath(os.path.join("data/assets/cloth", asset_rel_path))
        else:
             # Try assuming absolute if it starts with /
             usd_path = asset_rel_path
        
        if not os.path.exists(usd_path):
            print(f"[WARN] File not found: {usd_path}")
            failed_assets.append((asset_rel_path, "File Not Found"))
            continue

        prim_path = "/World/Cloth_Validation"
        
        # Clear previous
        if is_prim_path_valid(prim_path):
            delete_prim(prim_path)
        
        # 1. Create Prim
        create_prim(prim_path, usd_path=usd_path)
        
        # 2. Check if valid mesh
        mesh_path = f"{prim_path}/mesh"
        
        # 3. Apply Cloth API
        try:
            cloth_prim = SingleClothPrim(
                prim_path=mesh_path,
                particle_system=particle_system,
                particle_material=ParticleMaterial(prim_path=mat_path),
                particle_mass=0.02,
                stretch_stiffness=100.0,
                bend_stiffness=1.0,
                shear_stiffness=1.0,
                self_collision=False
            )
        except Exception as e:
            failed_assets.append((asset_rel_path, f"Cloth Init Failed: {str(e)}"))
            continue

        # 4. Step Simulation (Trigger Cooking)
        # We need to step a few times
        world.reset() # This allows physics to register the new prim
        
        # Step 1: Parse
        world.step(render=False) 
        # Step 2: Cook & Sim
        world.step(render=False)
        
        # 5. Check Validity
        # We check if the cloth has particles in the view
        # The view is automatically created by World? No, SingleClothPrim wraps it.
        # But SingleClothPrim doesn't expose a "view" directly to query particles easily in a batch way
        # We can use the underlying API or just create a view
        
        # Use PhysX Schema to check
        # Or better: check existing particles in the system?
        
        # Simpler check: If cooking failed, the particle cloth API usually reports issues or 
        # the mesh attributes for particles are empty.
        
        # Let's try to get the particle counts from the cloth prim wrapper if available,
        # or use PhysX Tensor API if initialized.
        
        # Correct way in Isaac Sim Core:
        # The SingleClothPrim doesn't provide a direct "get_num_particles" method easily.
        # But we can check if the prim is valid and configured.
        
        # Let's use omni.physx.get_physx_interface() to check for errors? No, hard to map to asset.
        
        # Robust Check:
        # Get the underlying Mesh prim
        prim = stage.GetPrimAtPath(mesh_path)
        if not prim.IsValid():
             failed_assets.append((asset_rel_path, "Prim Invalid"))
             continue
             
        # Check if PhysX Particle API exists
        if not prim.HasAPI(PhysxSchema.PhysxParticleClothAPI):
             failed_assets.append((asset_rel_path, "API Apply Failed"))
             continue
             
        # MOST ROBUST: Check if particles are actually simulated.
        # If cooking fails, usually no particles are created in the solver.
        # We can query the Cloth View.
        
        # Let's check for the "springIndices" warning equivalent...
        # "PhysxSchemaPhysxParticleClothAPI:springIndices has no elements."
        
        spring_indices = PhysxSchema.PhysxParticleClothAPI(prim).GetSpringIndicesAttr().Get()
        
        # Note: spring_indices are computed during cooking/setup. If empty or None, it likely failed or has no structure.
        if spring_indices is None or len(spring_indices) == 0:
            failed_assets.append((asset_rel_path, "No Springs/Cooking Failed"))
            # pbar.set_postfix({"fail": asset_rel_path.split("/")[-1]})
        else:
            valid_assets.append(asset_rel_path)

        # Cleanup
        delete_prim(prim_path)

    # Save Results
    print(f"\n[SUMMARY]")
    print(f"Total Checked: {len(assets_to_validate)}")
    print(f"Valid: {len(valid_assets)}")
    print(f"Failed: {len(failed_assets)}")
    
    if len(failed_assets) > 0:
        print("\n[FAILURES]")
        for p, reason in failed_assets:
            print(f"  {p} -> {reason}")
            
    # Write output
    with open(args.output_json, "w") as f:
        json.dump(valid_assets, f, indent=4)
    print(f"[INFO] Saved filtered list to {args.output_json}")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
