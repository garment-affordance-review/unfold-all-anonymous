# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import time
import sys
import math
import re
from pathlib import Path
from typing import Optional, Any, Sequence

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
from pxr import Gf, Sdf, UsdShade, UsdPhysics, UsdGeom, Usd
from isaacsim.core.utils import prims as prims_utils

# -------------------------------------------------------------------------- #
# Resolve project root and make sure the local cloth modules are importable.
# -------------------------------------------------------------------------- #
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CLOTH_ROOT = PROJECT_ROOT / "cloth"

for path in (PROJECT_ROOT, CLOTH_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))

# Import new separated modules
# from configs.env_config import EnvCfg # REMOVED
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
# ... imports
from .garment_pbd import GarmentManager, PBDGarmentManager
from unfold.platform.sim import _WorldProxy
from unfold.platform.assets import resolve_assets_root
from unfold.platform.rewards import compute_unfold_reward
from unfold.platform.profiling import profile
from unfold.platform.camera import compute_centers_world, set_camera_prims_look_at
from .asset_pool import AssetPool
from .control.unfold import Unfold
from .control.action import Action
from unfold.data.storage.replay_buffer import ReplayBuffer


@configclass
class EnvCfg(DirectRLEnvCfg):
    # Environment
    seed: int | None = 42
    decimation = 1
    episode_length_s: int = 100000
    
    # Physics params (PBD)
    # garment_physics: dict = {} 
    # ground_physics: dict = {}
    
    # Action / observation spaces used by IsaacLab validation.
    # These dimensions describe environment-level semantics; concrete meanings are used in Env.
    # Action: two vertex IDs [id1, id2]. A value of -1 asks the environment to auto-select.
    action_space = 2
    # Observation: keep 3 by default (cloth center x,y,z). Actual returns may append an XY-grid vertex-id map.
    observation_space = 3
    state_space = 0  # No additional state representation.


    
    # Camera settings (add "rgb" to obs_types to enable camera outputs)
    obs_types: Sequence[str] = ("pairs", "pos")
    camera_res: tuple[int, int] = (128, 128)
    camera_height: float = 3.0
    
    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60.0, render_interval=decimation)

    # Scene / cloning
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=2.5,
        replicate_physics=False,  # Deformable objects (particle cloth) do not support physics replication.
        clone_in_fabric=True,
    )

    clone_envs_from_source: bool = True
    
    steps_per_episode: int = 8
    episodes_per_asset_batch: int = 10  # Episodes per asset batch before switching
    
    # Random Policy Config
    random_policy: dict = {
        "min_grasp_dist": 0.3,
        "num_angles": 8,
        "num_scales": 5
    }


# -------------------------------------------------------------------------- #
# Resolve Assets directory

# -------------------------------------------------------------------------- #
def get_assets_root(cfg: Optional['EnvCfg'] = None) -> Path:
    """Return the Assets root directory path."""
    if cfg is not None and hasattr(cfg, '_resolved_assets_root'):
        return cfg._resolved_assets_root
    return resolve_assets_root(PROJECT_ROOT, None)

class Env(DirectRLEnv):
    cfg: EnvCfg

    def __init__(self, cfg: EnvCfg, render_mode: Optional[str] = None, **kwargs):
        self._world_proxy: Optional[_WorldProxy] = None
        self._reset_torch_gen: Optional[torch.Generator] = None
        self._reset_torch_seed: Optional[int] = None
        
        self._assets_root = get_assets_root(cfg)
        
        # Managers
        self._asset_pool: Optional[AssetPool] = None
        self._garment_manager: Optional[GarmentManager] = None
        self._unfold: Optional[Unfold] = None
        
        # Camera prim paths (populated in _setup_static_scene)
        self._cam_prim_paths: list[str] = []
        
        # Episode counters for internal asset switching
        self._batch_episode_counter = 0  # Episodes in current asset batch
        self._batch_counter = 0  # Total batch switches
        
        # Step stats buffer (saved before reset to preserve stats across Unfold recreation)
        self._last_step_stats = None
        
        super().__init__(cfg, render_mode, **kwargs)
        
        # Helper buffers
        self._centers_buf = torch.zeros((self.num_envs, 3), device=self.device)
        self._grid_obs: Optional[torch.Tensor] = None
        
        # AssetPool init moved to _setup_scene

    def _get_reset_torch_generator(self) -> torch.Generator:
        seed = getattr(self.cfg, "seed", None)
        seed_i = None if seed is None else int(seed) + 701
        if self._reset_torch_gen is None or self._reset_torch_seed != seed_i:
            if "cuda" in str(self.device):
                self._reset_torch_gen = torch.Generator(device=self.device)
            else:
                self._reset_torch_gen = torch.Generator()
            if seed_i is not None:
                self._reset_torch_gen.manual_seed(seed_i)
            self._reset_torch_seed = seed_i
        return self._reset_torch_gen

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the latest observation, preferring the XY grid."""

        if options and options.get("switch_asset", False):
            # Log asset switching with epoch/batch info if provided
            epoch_info = options.get("epoch_info", {})
            epoch_str = f"Epoch {epoch_info.get('epoch', '?')}" if epoch_info else ""
            batch_str = f"Batch {epoch_info.get('batch', '?')}/{epoch_info.get('total_batches', '?')}" if epoch_info else ""
            
            import omni.usd
            import omni.timeline
            from pxr import Sdf
            import omni.kit.app
            import gc

            stage = omni.usd.get_context().get_stage()
            omni.timeline.get_timeline_interface().stop()
            
            # Cleanup the full dynamic cloth subtree before respawning assets.
            stage.RemovePrim(Sdf.Path("/World/Cloth"))

            self.sim.step(render=False)
            self.scene.__init__(self.cfg.scene)
            
            # Re-setup dynamic scene (spawns new garments - logs asset names)
            self._setup_dynamic_scene()
            
            # Print structured log after spawning (garment names now known)
            asset_names = [Path(p).parent.name for p in self._garment_manager._env_usd_paths if p]
            names_preview = str(asset_names[:4]) + "..." if len(asset_names) > 4 else str(asset_names)
            print(f"[SWITCH] {epoch_str} | {batch_str} | Assets: {names_preview}", file=sys.stdout, flush=True)

            # Re-initialize scene handles (Critical for physics)
            self.sim.reset()
            # self.extras = {} 
            
        obs, info = super().reset(seed=seed, options=options)
             
        # Ensure switch info propagates to info (and thus to extras in step)
        if options and options.get("switch_asset", False):
            info["switch_asset"] = True
            if "epoch_info" in options:
                info["epoch_info"] = options["epoch_info"]
            
        if self._unfold:
            self._unfold.reset_buffers()

        return obs, info

    def _setup_scene(self):        
        self._setup_static_scene()
        # self._setup_dynamic_scene()

    def _setup_static_scene(self):
        """Set up static environment: ground and lighting."""
        stage = self.sim.stage

        # Create Environment root
        UsdGeom.Xform.Define(stage, "/World/Environment")
        
        # Ground
        ground_cfg = GroundPlaneCfg()
        spawn_ground_plane(prim_path="/World/Environment/Ground", cfg=ground_cfg)
        self._set_ground_size("/World/Environment/Ground")
        self._hide_default_ground_visual_meshes("/World/Environment/Ground")
        self._create_ground_visual_plane("/World/Environment/GroundVisual")
        
        # Remove SphereLight from default_environment (we use DomeLight)
        sphere_light = stage.GetPrimAtPath("/World/Environment/Ground/SphereLight")
        if sphere_light.IsValid():
            stage.RemovePrim("/World/Environment/Ground/SphereLight")
        
        # Ground physics material
        mtl_path = "/World/Environment/GroundMaterial"
        UsdShade.Material.Define(stage, mtl_path)
        material_prim = stage.GetPrimAtPath(mtl_path)
        
        physx_mat_api = UsdPhysics.MaterialAPI.Apply(material_prim)
        physx_mat_api.CreateStaticFrictionAttr(self.cfg.ground_physics['static_friction'])
        physx_mat_api.CreateDynamicFrictionAttr(self.cfg.ground_physics['dynamic_friction'])
        physx_mat_api.CreateRestitutionAttr(self.cfg.ground_physics['restitution'])

        # Bind material to ground
        ground_prim = stage.GetPrimAtPath("/World/Environment/Ground")
        if ground_prim and ground_prim.IsValid():
            binding_api = UsdShade.MaterialBindingAPI.Apply(ground_prim)
            binding_api.Bind(UsdShade.Material(material_prim), UsdShade.Tokens.strongerThanDescendants, "physics")

        if self._asset_pool is None:
             print("[ASSET_POOL] Initializing in _setup_scene...", file=sys.stdout, flush=True)
             self._asset_pool = AssetPool(
                cfg=self.cfg,
                assets_root=self._assets_root,
                device=self.device
            )

        # Cameras are now spawned exclusively by the downstream Replicator or rendering scripts

    def _hide_default_ground_visual_meshes(self, ground_root: str):
        """Hide default grid USD render meshes while keeping physics ground intact."""
        root = self.sim.stage.GetPrimAtPath(ground_root)
        if not root or not root.IsValid():
            return
        hidden = 0
        for prim in Usd.PrimRange(root):
            try:
                if prim and prim.IsValid() and prim.IsA(UsdGeom.Mesh):
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden += 1
            except Exception:
                continue
        print(f"[GROUND] hidden_default_visual_meshes={hidden}")

    def _create_ground_visual_plane(self, prim_path: str):
        """Create a single UV-mapped render plane for stable ground texturing."""
        stage = self.sim.stage
        plane_x = float(self._ground_size_m[0])
        plane_y = float(self._ground_size_m[1])
        half_x = 0.5 * plane_x
        half_y = 0.5 * plane_y
        z = 1e-3  # avoid z-fighting against the default ground geometry

        UsdGeom.Xform.Define(stage, prim_path)
        mesh_prim = UsdGeom.Mesh.Define(stage, f"{prim_path}/mesh")
        mesh = UsdGeom.Mesh(mesh_prim.GetPrim())
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(-half_x, -half_y, z),
                Gf.Vec3f(half_x, -half_y, z),
                Gf.Vec3f(half_x, half_y, z),
                Gf.Vec3f(-half_x, half_y, z),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
        mesh.SetNormalsInterpolation("vertex")
        primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)])
        mesh.CreateDoubleSidedAttr(True)
        print(f"[GROUND] created_visual_plane={prim_path}/mesh size=({plane_x:.3f},{plane_y:.3f})")

    def _set_ground_size(self, ground_root: str):
        """Resize ground to target metric size from config, then cache effective size."""
        from pxr import UsdGeom, Usd

        root = self.sim.stage.GetPrimAtPath(ground_root)
        if not root or not root.IsValid():
            raise RuntimeError(f"[GROUND] Ground root missing: {ground_root}")

        target = getattr(self.cfg, "ground_size_m", None)
        if not isinstance(target, (list, tuple)) or len(target) < 2:
            raise RuntimeError(f"[GROUND] Invalid ground_size_m: {target}")
        target_x = max(float(target[0]), 1e-6)
        target_y = max(float(target[1]), 1e-6)

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bound = bbox_cache.ComputeWorldBound(root)
        rng = bound.ComputeAlignedRange()
        cur_size = rng.GetSize()
        cur_x = max(float(cur_size[0]), 1e-6)
        cur_y = max(float(cur_size[1]), 1e-6)
        sx = target_x / cur_x
        sy = target_y / cur_y

        xform = UsdGeom.Xformable(root)
        scale_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        if scale_op is None:
            # GroundPlane often uses double-precision xform ops; match precision to avoid type mismatch.
            scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(sx, sy, 1.0))

        # Recompute effective size for downstream texture scaling.
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bound2 = bbox_cache.ComputeWorldBound(root)
        rng2 = bound2.ComputeAlignedRange()
        sz2 = rng2.GetSize()
        self._ground_size_m = (max(float(sz2[0]), 1e-6), max(float(sz2[1]), 1e-6))
        print(f"[GROUND] target_size=({target_x:.3f},{target_y:.3f}) effective_size=({self._ground_size_m[0]:.3f},{self._ground_size_m[1]:.3f})")

    def _setup_dynamic_scene(self):
        # 1. Select Assets
        asset_indices = self._asset_pool.sample_indices()
        self.current_asset_indices = np.array(asset_indices)   
        selected_paths = self._asset_pool.get_paths(asset_indices)
        
        # 2. Initialize Garment Manager (if not exists) & Spawn
        # Note: GarmentManager now holds the asset state.
        spawn_center = self.cfg.spawn_cfg['center']
        del self._garment_manager
        self._garment_manager = GarmentManager(
            cfg=None, # Use default
            env_cfg=self.cfg,
            num_envs=self.num_envs,
            device=self.device,
            usd_paths=selected_paths,
            sim=self.sim,
            scene=self.scene,
            spawn_center=spawn_center
        )
        # Spawn is now implicit in __init__
        self._world_proxy = _WorldProxy(self.scene, self.sim)
            
        # 3. Initialize Task Controller
        if self.cfg.vertex_control_enabled:
            # Task manager internally initializes ActionManager
            self._unfold = Unfold(
                cfg=self.cfg,
                num_envs=self.num_envs,
                device=self.device,
                garment_asset=self._garment_manager,  # PASS MANAGER AS ASSET
                physics_dt=self.physics_dt,
                env_origins=self.scene.env_origins if hasattr(self.scene, "env_origins") else None
            )

    def get_experience_data(
        self, 
        actions: torch.Tensor, 
        rewards: torch.Tensor,
        include_features: bool = True
    ) -> list:
        """
        Get experience data for external storage.
        
        Args:
            actions: Action tensor (num_envs, 2)
            rewards: Reward tensor (num_envs,)
            include_features: Whether to compute and return features
            
        Returns:
            List of (asset_path, id1, ft1, id2, ft2, reward) tuples for valid actions
        """
        experiences = []
        
        if not self.cfg.vertex_control_enabled:
            return experiences
        
        with torch.no_grad():
            # Filter valid actions
            valid_mask = (actions[:, 0] >= 0) & (actions[:, 1] >= 0)
            valid_envs = valid_mask.nonzero().squeeze(-1)
            
            for env_idx in valid_envs.cpu().tolist():
                id1 = actions[env_idx, 0].item()
                id2 = actions[env_idx, 1].item()
                
                ft1, ft2 = None, None
                
                if include_features:
                    features = self._garment_manager.get_feature_for_env(env_idx)
                    if features is not None and id1 < features.shape[0] and id2 < features.shape[0]:
                        ft1 = features[id1].cpu()
                        ft2 = features[id2].cpu()
                
                # We assume id1/id2 are valid if we skip features (policy responsibility)
                asset_idx = self.current_asset_indices[env_idx]
                asset_path = self._asset_pool.get_paths([asset_idx])[0]
                reward = rewards[env_idx].cpu().item()
                experiences.append((asset_path, id1, ft1, id2, ft2, reward))
        
        return experiences

    def step(self, actions: torch.Tensor):
        if actions.ndim != 2 or actions.shape != (self.num_envs, 2):
            raise ValueError(f"Actions shape must be (num_envs, 2), got {actions.shape}")
        
        # Actions are (id1, id2) now
        
        # Clear transient flags from previous step
        if hasattr(self, "extras") and "switch_asset" in self.extras:
            del self.extras["switch_asset"]
        
        # Capture detailed wall timing for the full step, not just the action rollout.
        wall_start_time = time.perf_counter()
        action_start_time = wall_start_time

        # Delegate to Task Controller
        if self.cfg.vertex_control_enabled and self._unfold:
            self._unfold.step(actions, self.sim, self.scene)
        else:
            for _ in range(10): 
                self.sim.step()
        action_duration = time.perf_counter() - action_start_time
        
        # Post-Processing
        self.episode_length += 1

        obs_start_time = time.perf_counter()
        self.obs = self._get_observations()
        obs_duration = time.perf_counter() - obs_start_time

        reward_start_time = time.perf_counter()
        self.reward = self._get_rewards()
        reward_duration = time.perf_counter() - reward_start_time
        reset_duration = 0.0
        
        if self.episode_length >= self.cfg.steps_per_episode:
            reset_start_time = time.perf_counter()
            # User requirement: Episode is counted globally (all envs sync).
            # We increment counters by 1 for the entire batch reset.
            self._batch_episode_counter += 1
            # Check for asset switching (if auto-switch enabled)
            if self._batch_episode_counter >= self.cfg.episodes_per_asset_batch:
                self._batch_counter += 1
                self._batch_episode_counter = 0  # Reset episode counter for new batch
                # Calculate epoch info (epoch = floor(batch / num_batches) + 1)
                num_batches = (self._asset_pool.size + self.num_envs - 1) // self.num_envs if self._asset_pool else 1
                
                prev_epoch = (self._batch_counter - 1) // num_batches + 1
                current_epoch = (self._batch_counter) // num_batches + 1
                current_batch = (self._batch_counter) % num_batches + 1
                
                # Check for Epoch Change (Asset Loop Completion)
                if current_epoch > prev_epoch:
                    if self._asset_pool:
                        self._asset_pool.shuffle()
                
                epoch_info = {
                    "epoch": current_epoch,
                    "total_epochs": "?",  # Unknown at env level
                    "batch": current_batch,
                    "total_batches": num_batches
                }
                options = {"switch_asset": True, "epoch_info": epoch_info}
            else:
                options = {"switch_asset": False}

            self.obs, reset_info = self.reset(options=options)
            
            # CRITICAL: Propagate reset info (e.g. switch_asset) to extras so it's available in step() return
            # AND restore the rewards extras we captured!
            if isinstance(reset_info, dict):
                self.extras.update(reset_info)
            reset_duration = time.perf_counter() - reset_start_time

        wall_duration = time.perf_counter() - wall_start_time

        # Save step stats (and log appropriately)
        if self._unfold:
            self._last_step_stats = self._unfold.step_stats
            
            # Internal Logging if progress info is set
            if hasattr(self, 'progress_info') and self.progress_info:
                p = self.progress_info
                stats = self._last_step_stats
                
                # Throughput (Samples per second = num_envs / full wall duration)
                throughput = self.num_envs / wall_duration if wall_duration > 0 else 0
                
                stats_str = (
                    f"stretch:{stats.stretch_steps}/{stats.stretch_time:.2f}s | "
                    f"stable:{stats.stable_steps}/{stats.stable_time:.2f}s | "
                    f"move:{stats.move_steps}/{stats.move_time:.2f}s"
                )
                
                print(
                    f"[STEP] Epoch {p.get('epoch', '?')}/{p.get('total_epochs', '?')} "
                    f"Step {p.get('step_in_epoch', '?')}/{p.get('steps_per_episode', '?')} | "
                    f"Global {p.get('step', '?')}/{p.get('total_steps', '?')} | "
                    f"Wall: {wall_duration:.2f}s | "
                    f"Rate: {throughput:.1f} sample/s | "
                    f"action:{action_duration:.2f}s obs:{obs_duration:.2f}s reward:{reward_duration:.2f}s "
                    f"reset:{reset_duration:.2f}s | {stats_str}",
                    flush=True
                )

        return self.obs, self.reward, self.reset_terminated, self.reset_time_outs, self.extras
    
    def _get_observations(self) -> dict:
        
        obs = {}
        
        # Get all positions as batched tensor: (num_envs, max_particles, 3)
        all_pos = self._garment_manager._get_particle_positions().clone()
        pos_mask = self._garment_manager.get_padding_mask()
        
        # CRITICAL: Convert to LOCAL FRAME (relative to env origin)
        # This simplifies all downstream tasks (rewards, data collection)
        all_pos = all_pos - self.scene.env_origins.unsqueeze(1)
        all_pos[pos_mask.squeeze(-1) == 0] = 0

        # Return batched tensor directly with padding mask
        obs["pos"] = all_pos  # (num_envs, max_particles, 3)
        obs["pos_mask"] = self._garment_manager.get_padding_mask()  # (num_envs, max_particles, 1) or None
        obs["pos_mask_sampled"] = self._garment_manager.get_sampling_mask()  # (num_envs, max_particles, 1) or None
        obs["init_pos"] = self._garment_manager.init_pos

        # Collect faces - need to pad to uniform size
        faces_list = []
        max_faces = 0
        
        for env_idx in range(self.num_envs):
            faces = self._unfold.action_manager.get_mesh_faces(env_idx)
            faces_list.append(faces)
            max_faces = max(max_faces, faces.shape[0])
        
        # Pad faces to uniform size
        padded_faces = torch.zeros((self.num_envs, max_faces, 3), dtype=torch.long, device=self.device)
        faces_mask = torch.zeros((self.num_envs, max_faces, 1), dtype=torch.float32, device=self.device)
        
        for env_idx, faces in enumerate(faces_list):
            num_faces = faces.shape[0]
            if num_faces > 0:
                padded_faces[env_idx, :num_faces] = faces
                faces_mask[env_idx, :num_faces] = 1.0
        
        obs["faces"] = padded_faces  # (num_envs, max_faces, 3)
        obs["faces_mask"] = faces_mask  # (num_envs, max_faces, 1)

        return obs

    def _get_rewards(self) -> torch.Tensor:
        # Pass debug configs from env cfg if present
        debug_cfg = getattr(self.cfg, "debug", None)
        rewards, rewards_extras = compute_unfold_reward(
            init_pos=self.obs["init_pos"],
            current_pos=self.obs["pos"],
            padding_mask=self.obs["pos_mask"],
            deformable_weight=self.cfg.deformable_weight,
            rigid_free_y=not getattr(debug_cfg, "rigid_fixed_y", False) if debug_cfg else True,
            sampling_mask=self.obs.get("pos_mask_sampled", self.obs["pos_mask"]),
        )
        self.extras['rewards_extras'] = rewards_extras
        return rewards

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        env_ids_list = env_ids.cpu().tolist() if isinstance(env_ids, torch.Tensor) else env_ids
        
        if self.cfg.vertex_control_enabled and self._unfold is not None:
            self._unfold.reset(env_ids_list)

        if not self.cfg.randomize_on_reset:
            super()._reset_idx(env_ids)
            return
        
        env_origins = self.scene.env_origins[env_ids]
        spawn_center_offset = torch.tensor(self.cfg.spawn_cfg['center'], device=self.device)
        spawn_range = torch.tensor(self.cfg.spawn_cfg['range'], device=self.device)
        rand_gen = self._get_reset_torch_generator()
        pos_noise = (torch.rand((len(env_ids), 3), device=self.device, generator=rand_gen) - 0.5) * 2 * spawn_range
        target_pos = env_origins + spawn_center_offset + pos_noise

        target_rot = self._sample_spawn_orientations(len(env_ids), rand_gen)
        
        self._garment_manager.reset_to_poses(env_ids, target_pos, target_rot)
        self.scene.write_data_to_sim()
        
        # Stabilization Loop (Velocity-based)
        self._unfold.stabilize(self.sim, self.scene)
        self._apply_predrop_relift(env_ids, rand_gen)
        self._unfold.action_manager.set_sequence_phase('idle')

        self.episode_length = 0

    def _sample_spawn_orientations(self, num_envs: int, rand_gen: torch.Generator) -> torch.Tensor:
        spawn_cfg = getattr(self.cfg, "spawn_cfg", {}) or {}
        mode = str(spawn_cfg.get("orientation_mode", "uniform_quat")).lower()
        if mode == "uniform_quat":
            u = torch.rand((num_envs, 3), device=self.device, generator=rand_gen)
            u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
            two_pi = 2.0 * torch.pi
            sqrt_1_u1 = torch.sqrt(torch.clamp(1.0 - u1, min=0.0))
            sqrt_u1 = torch.sqrt(torch.clamp(u1, min=0.0))
            x = sqrt_1_u1 * torch.sin(two_pi * u2)
            y = sqrt_1_u1 * torch.cos(two_pi * u2)
            z = sqrt_u1 * torch.sin(two_pi * u3)
            w = sqrt_u1 * torch.cos(two_pi * u3)
            return torch.stack((w, x, y, z), dim=-1)

        rot_range_deg = torch.tensor(spawn_cfg["rot_range_deg"], device=self.device)
        rot_range_rad = torch.deg2rad(rot_range_deg)
        rot_base_deg = torch.tensor(spawn_cfg["init_euler_deg"], device=self.device)
        rot_base_rad = torch.deg2rad(rot_base_deg)
        euler_angles = rot_base_rad + (
            torch.rand((num_envs, 3), device=self.device, generator=rand_gen) - 0.5
        ) * rot_range_rad
        return math_utils.quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

    def _apply_predrop_relift(self, env_ids: torch.Tensor, rand_gen: torch.Generator) -> None:
        spawn_cfg = getattr(self.cfg, "spawn_cfg", {}) or {}
        relift_cfg = spawn_cfg.get("predrop_relift", {}) or {}
        if not bool(relift_cfg.get("enabled", False)):
            return

        garment = self._garment_manager
        if garment is None:
            return
        all_pos = garment._get_particle_positions().clone()
        all_vel = garment._get_particle_velocities().clone()
        padding_mask = garment.get_padding_mask()

        lift_range = relift_cfg.get("height_range", [0.8, 1.2])
        if isinstance(lift_range, (int, float)):
            lift_min = lift_max = float(lift_range)
        else:
            lift_min = float(lift_range[0])
            lift_max = float(lift_range[1] if len(lift_range) > 1 else lift_range[0])
        xy_jitter = relift_cfg.get("xy_jitter", [0.0, 0.0])
        xy_jitter = [float(xy_jitter[0]), float(xy_jitter[1] if len(xy_jitter) > 1 else xy_jitter[0])]
        rerandomize_orientation = bool(relift_cfg.get("rerandomize_orientation", True))

        env_ids_long = env_ids.to(dtype=torch.long, device=self.device)
        env_ids_list = env_ids_long.cpu().tolist()
        for local_i, env_id in enumerate(env_ids_list):
            valid = padding_mask[env_id, :, 0] > 0.5 if padding_mask is not None else torch.ones(
                all_pos.shape[1], dtype=torch.bool, device=self.device
            )
            if int(valid.sum()) == 0:
                continue
            dz = lift_min + (lift_max - lift_min) * torch.rand((), device=self.device, generator=rand_gen)
            dx = (torch.rand((), device=self.device, generator=rand_gen) - 0.5) * 2.0 * xy_jitter[0]
            dy = (torch.rand((), device=self.device, generator=rand_gen) - 0.5) * 2.0 * xy_jitter[1]
            delta = torch.tensor([dx, dy, dz], device=self.device, dtype=all_pos.dtype)
            current_pos = all_pos[env_id, valid].clone()
            centroid = current_pos.mean(dim=0)
            if rerandomize_orientation:
                quat = self._sample_spawn_orientations(1, rand_gen)[0]
                centered = current_pos - centroid
                rotated = math_utils.transform_points(
                    centered.unsqueeze(0),
                    pos=torch.zeros((1, 3), device=self.device, dtype=current_pos.dtype),
                    quat=quat.unsqueeze(0),
                )[0]
                all_pos[env_id, valid] = rotated + centroid + delta
            else:
                all_pos[env_id, valid] = current_pos + delta
            all_vel[env_id, valid] = 0.0

        garment._set_particle_positions(all_pos, env_ids_long)
        garment._set_particle_velocities(all_vel, env_ids_long)
        self.scene.write_data_to_sim()
        self._unfold.stabilize(self.sim, self.scene)
