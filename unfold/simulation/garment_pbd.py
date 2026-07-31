"""
Unified Garment Manager
Merges high-level asset spawning (Scene Setup) and low-level physics interaction (Asset View).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence, Any, Dict, List
import numpy as np
import torch

import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.prims import SingleParticleSystem, SingleClothPrim
from isaacsim.core.api.materials import ParticleMaterial
from isaacsim.core.utils.prims import is_prim_path_valid
from pxr import Gf, Sdf, UsdGeom, UsdShade

# Isaac Lab Assets
from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.utils import configclass

import isaaclab.utils.math as math_utils
import omni.physics.tensors.impl.api as physx_api



@configclass
class PBDGarmentManagerCfg(AssetBaseCfg):
    """Configuration for PBDGarmentManager (PBD Particle Cloth)."""
    pass


# Backwards compatibility alias
GarmentManagerCfg = PBDGarmentManagerCfg


class PBDGarmentManager(AssetBase):
    """
    Unified manager for garment assets.
    Handles:
    1. Spawning USDs into the stage (Shared Physics).
    2. Managing Physics View for tensor API access.
    3. State initialization and resetting.
    """

    def __init__(self, cfg: GarmentManagerCfg, env_cfg, num_envs: int, device: str, 
                 usd_paths: List[str], sim, scene, spawn_center):
        self.env_cfg = env_cfg # Original full EnvCfg (contains physics params)
        self.num_envs = num_envs
        self._device = device
        
        # State
        self._view = None
        self._is_initialized = False
        
        # Shared physics resources
        self._sim = sim
        self._stage = sim.stage
        self._particle_system: Optional[SingleParticleSystem] = None
        self._particle_material: Optional[ParticleMaterial] = None
        
        # Per-Environment Data
        self._env_usd_paths = usd_paths
        self._prim_paths: List[str] = []
        
        # Caches
        self._init_pos_cache: Dict[int, torch.Tensor] = {} # Per-env init pos (template)
        self._features_cache: Dict[int, torch.Tensor] = {} # Per-env features
        self._num_particles_per_env_dict: Dict[int, int] = {}
        
        self._template_pos_per_env: Dict[int, torch.Tensor] = {} # Transformed template world pos
        self._initial_world_offset: Dict[int, torch.Tensor] = {} # Detected world offsets (from UsdGeom)
        self._omnipbr_cls = None
        self._omnipbr_cls_checked = False
        self._material_wrappers: Dict[str, Any] = {}
        base_seed = getattr(self.env_cfg, "seed", None)
        self._preview_rng = np.random.default_rng(None if base_seed is None else int(base_seed) + 401)

        # Batch Buffers
        self._initial_pos_stacked: Optional[torch.Tensor] = None
        self._padding_mask: Optional[torch.Tensor] = None
        
        # Store config for later initialization
        self._asset_cfg = cfg if cfg else PBDGarmentManagerCfg(prim_path="/World/Cloth/env_.*/garment/mesh")

        self._spawn_center = np.asarray(spawn_center, dtype=float)

        # =========================================================================
        # Spawning Logic (Integrated from former spawn_garments)
        # =========================================================================
        
        # Initialize Shared Physics
        self._init_shared_physics(self._stage)
        
        # Initial Spawn
        self._spawn_garments(usd_paths)
            
        super().__init__(self._asset_cfg)
            
        if getattr(self.env_cfg, 'enable_textures', False):
            self._apply_preview_materials()
        else:
            self._apply_debug_materials()

    def _spawn_garments(self, usd_paths: List[str]):
        """Spawns garments for the given paths."""
        self._env_usd_paths = usd_paths
        self._prim_paths = []
        
        env_origins = self.env_cfg.scene.env_origins.cpu().numpy() if hasattr(self.env_cfg.scene, "env_origins") else np.zeros((self.num_envs, 3))
        # Note: scene.env_origins might not be populated in __init__ if scene isn't built yet, 
        # but here we assume scene is passed in.
        # Fallback if scene doesn't have it yet (shouldn't happen if scene is initialized):
        if env_origins.shape[0] == 0:
             env_origins = np.zeros((self.num_envs, 3)) # Placeholder?
        
        cfg_phys = self.env_cfg.garment_physics
        scale = np.array(cfg_phys['scale'], dtype=float)

        for env_idx, usd_path in enumerate(usd_paths):
            if usd_path is None: continue
            
            # Layout: /World/Cloth/env_{i}/garment
            prim_path = f"/World/Cloth/env_{env_idx}/garment"
            self._prim_paths.append(prim_path)
            
            # Pos
            pos = env_origins[env_idx] + self._spawn_center
            self._initial_world_offset[env_idx] = torch.as_tensor(pos, device=self._device, dtype=torch.float32)
            
            # Add to Stage
            self._add_garment_prim(usd_path, prim_path, pos.tolist(), scale.tolist(), cfg_phys['particle_mass'])

            cloth_prim = prims_utils.get_prim_at_path(prim_path) 
            cloth_prim.SetInstanceable(False)
            self._apply_semantic_label(cloth_prim, semantic_type="class", semantic_data="cloth")

    @staticmethod
    def _apply_semantic_label(prim, semantic_type: str, semantic_data: str) -> None:
        """Apply semantic labels through USD APIs only.

        This avoids Replicator-dependent helper behavior in collection entrypoints.
        """
        try:
            from pxr import UsdSemantics

            schema_name = str(semantic_type)
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
            labels_attr.Set([str(semantic_data)])
        except Exception as e:
            print(f"[GARMENT_MGR] Warning: failed to apply semantic label to {prim.GetPath()}: {e}")

    def switch_assets(self, new_usd_paths: List[str]):
        """Switches assets in-place."""
        # 1. Pause (Safe modification)
        self._sim.pause()
        
        # 2. Cleanup Old Prims
        for prim_path in self._prim_paths:
            if is_prim_path_valid(prim_path):
                prims_utils.delete_prim(prim_path)
        
        # 3. Reset State
        self._view = None
        self._is_initialized = False
        self._prim_paths = []
        self._init_pos_cache.clear()
        self._features_cache.clear()
        # 4. Spawn New
        self._spawn_garments(new_usd_paths)
        # Apply materials per config
        mode = str(getattr(self.env_cfg, "material_mode", "")).lower()
        use_preview = (mode == "preview") or bool(getattr(self.env_cfg, "enable_textures", False))
        if use_preview:
            try:
                self._apply_preview_materials()
            except Exception as e:
                print(f"[GARMENT_MGR] preview materials failed: {e}")
                self._apply_debug_materials()
        else:
            self._apply_debug_materials()

        # 5. Play and Step (Flush)
        self._sim.play()
        self._sim.step(render=False)
        
        # 5. Play and Step (Flush)
        self._sim.play()
        self._sim.step(render=False)
        
        # 6. Re-Initialize View
        self._initialize_impl()

    def _init_shared_physics(self, stage, physics_scene_path: str = "/physicsScene"):
        """Shared particle system setup."""
        cfg = self.env_cfg.garment_physics
        
        cloth_root = "/World/Cloth"
        if not is_prim_path_valid(cloth_root):
            prims_utils.create_prim(cloth_root, prim_type="Xform")
            
        shared_root = f"{cloth_root}/SharedPhysics"
        if not is_prim_path_valid(shared_root):
            prims_utils.create_prim(shared_root, prim_type="Xform")
            
        # Material
        mat_path = f"{shared_root}/particleMaterial"
        self._particle_material = ParticleMaterial(
            prim_path=mat_path,
            friction=cfg['friction'],
            particle_friction_scale=cfg['particle_friction_scale'],
            damping=cfg['damping'],
            adhesion=cfg['adhesion'],
            particle_adhesion_scale=cfg['particle_adhesion_scale'],
            gravity_scale=cfg['gravity_scale'],
        )
        
        # System
        sys_path = f"{shared_root}/particleSystem"
        self._particle_system = SingleParticleSystem(
            prim_path=sys_path,
            simulation_owner=physics_scene_path,
            particle_system_enabled=True,
            solver_position_iteration_count=cfg['solver_position_iteration_count'],
            contact_offset=cfg['contact_offset'],
            rest_offset=cfg['rest_offset'],
            particle_contact_offset=cfg['particle_contact_offset'],
            solid_rest_offset=cfg['solid_rest_offset'],
            fluid_rest_offset=cfg['fluid_rest_offset'],
            max_depenetration_velocity=cfg['max_depenetration_velocity'],
            max_velocity=cfg['max_velocity'],
            global_self_collision_enabled=True,
        )
        self._particle_system.apply_particle_material(self._particle_material)

    def _add_garment_prim(self, usd_path: str, prim_path: str, pos, scale, particle_mass: float):
        cfg = self.env_cfg.garment_physics
        
        prims_utils.create_prim(prim_path, usd_path=usd_path)
        
        prim = self._stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddScaleOp().Set(Gf.Vec3d(*scale))
        
        mesh_path = f"{prim_path}/mesh"

        # Create Cloth Prim
        SingleClothPrim(
            prim_path=mesh_path,
            particle_system=self._particle_system,
            particle_material=self._particle_material,
            particle_mass=particle_mass,
            stretch_stiffness=cfg['stretch_stiffness'],
            bend_stiffness=cfg['bend_stiffness'],
            shear_stiffness=cfg['shear_stiffness'],
            spring_damping=cfg['spring_damping'],
            self_collision=True,
            self_collision_filter=True,
        )
        
        # Disable Welding to fix "Non-manifold after welding" errors
        # This forces PhysX to use the mesh as-is.
        from pxr import PhysxSchema
        prim = self._stage.GetPrimAtPath(mesh_path)
        if prim.IsValid():
            # Use AutoParticleClothAPI for auto-generated particle systems
            cloth_api = PhysxSchema.PhysxAutoParticleClothAPI.Apply(prim)
            try:
                cloth_api.CreateDisableMeshWeldingAttr(False)
            except Exception as e:
                print(f"[GARMENT_MGR] Warning: failed to set disableMeshWelding=False for {mesh_path}: {e}")
    
    def _apply_debug_materials(self):
        """Bind lightweight preview materials only when texture rendering is disabled."""
        from colorsys import hsv_to_rgb

        if getattr(self.env_cfg, "enable_textures", False):
            return
        color_mode = str(getattr(self.env_cfg, "debug_color_mode", "") or "").lower()
        grid_dim = max(1, int(np.ceil(np.sqrt(max(1, len(self._prim_paths))))))
        for prim_path in self._prim_paths:
            env_idx = int(prim_path.split("/env_")[1].split("/")[0])
            fixed_color = getattr(self.env_cfg, "debug_fixed_color", None)
            if fixed_color is not None:
                color = tuple(float(c) for c in fixed_color)
            elif color_mode == "x_gradient":
                # Env ordering in this scene is y-major, so the x column changes on the
                # slower axis. Use the grid column index to make color sweep left-to-right.
                col_idx = env_idx // grid_dim
                col_idx = max(0, min(grid_dim - 1, col_idx))
                t = col_idx / max(1, grid_dim - 1)
                palette = np.asarray(
                    [
                        hsv_to_rgb(0.06, 0.64, 0.90),  # orange
                        hsv_to_rgb(0.12, 0.62, 0.88),  # warm yellow
                        hsv_to_rgb(0.18, 0.60, 0.88),  # yellow-green
                        hsv_to_rgb(0.26, 0.58, 0.87),  # lime
                        hsv_to_rgb(0.36, 0.56, 0.87),  # green-cyan
                        hsv_to_rgb(0.48, 0.56, 0.88),  # cyan
                        hsv_to_rgb(0.58, 0.55, 0.88),  # sky blue
                        hsv_to_rgb(0.68, 0.54, 0.88),  # blue
                        hsv_to_rgb(0.78, 0.53, 0.88),  # violet
                        hsv_to_rgb(0.88, 0.56, 0.89),  # magenta-pink
                    ],
                    dtype=np.float32,
                )
                pos = t * float(len(palette) - 1)
                idx0 = int(np.floor(pos))
                idx1 = min(len(palette) - 1, idx0 + 1)
                alpha = float(pos - idx0)
                color = tuple((1.0 - alpha) * palette[idx0] + alpha * palette[idx1])
            else:
                color = tuple(float(c) for c in self._preview_rng.random(3))
            mat_path = f"{prim_path}/debug_material"
            mesh_path = f"{prim_path}/mesh"
            
            material = UsdShade.Material.Define(self._stage, mat_path)
            shader = UsdShade.Shader.Define(self._stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
            
            # Check for GeomSubsets
            mesh_prim = self._stage.GetPrimAtPath(mesh_path)
            if mesh_prim.IsValid():
                # Find all GeomSubset children
                subsets = [child for child in mesh_prim.GetChildren() if child.IsA(UsdGeom.Subset)]
                
                if subsets:
                    # Bind to each subset
                    for subset in subsets:
                        UsdShade.MaterialBindingAPI.Apply(subset).Bind(material)
                else:
                    # Bind to mesh directly
                    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)

    def _resolve_texture_paths(self, env_idx, external_texture_set=None):
        """Resolve texture paths for one environment.

        Returns dict with keys: 'basecolor', 'normal', 'roughness', 'metallic', 'mrpack', 'opacity', 'is_external'
        If external_texture_set provided (from Fabric), use it.
        Otherwise, scan asset's textures/ directory.
        """
        from pathlib import Path as _Path

        if external_texture_set:
            # Use external high-fidelity textures
            return {
                'basecolor': _Path(external_texture_set['basecolor']),
                'normal': _Path(external_texture_set['normal']),
                'roughness': _Path(external_texture_set['roughness']),
                'metallic': _Path(external_texture_set['metallic']),
                'mrpack': None,
                'opacity': None,
                'is_external': True  # Mark as external texture
            }

        # Use asset's built-in textures
        usd_abs = _Path(self._env_usd_paths[env_idx])
        tex_dir = usd_abs.parent / "textures"

        def _pick(tex_dir, keys):
            if not tex_dir.exists():
                return None
            for f in sorted(tex_dir.iterdir()):
                if f.suffix.lower() != ".png":
                    continue
                if any(k in f.name.lower() for k in keys):
                    return f
            return None

        base = _pick(tex_dir, ["_diffuse_"])
        normal = _pick(tex_dir, ["_normal_"])
        mrpack = _pick(tex_dir, ["_metallicroughness_", "_occlusionroughnessmetallic_", "_orm_"])
        if mrpack is None:
            rough = _pick(tex_dir, ["_roughness_"])
            metal = _pick(tex_dir, ["_metalness_"])
        else:
            rough = None
            metal = None
        opacity = _pick(tex_dir, ["_opacity_", "_alpha_", "_mask_"])

        # Reclassify if needed
        if rough and any(k in rough.name.lower() for k in ("metallicroughness", "occlusionroughnessmetallic", "orm")):
            mrpack = rough
            rough = None

        return {
            'basecolor': base,
            'normal': normal,
            'roughness': rough,
            'metallic': metal,
            'mrpack': mrpack,
            'opacity': opacity,
            'is_external': False  # Asset's own textures
        }

    def _try_enable_omnipbr_wrapper(self):
        """Require high-level OmniPBR wrapper (fail-fast if unavailable)."""
        if self._omnipbr_cls_checked:
            return self._omnipbr_cls
        self._omnipbr_cls_checked = True
        try:
            from isaacsim.core.experimental.materials import OmniPbrMaterial

            self._omnipbr_cls = OmniPbrMaterial
            print("[GARMENT_MGR] OmniPbrMaterial wrapper enabled.")
        except Exception as exc:
            raise RuntimeError(
                f"[GARMENT_MGR] OmniPbrMaterial is required but unavailable: {exc}"
            ) from exc
        return self._omnipbr_cls

    @staticmethod
    def _wrapper_value(value):
        if isinstance(value, Gf.Vec2f):
            return [[float(value[0]), float(value[1])]]
        if isinstance(value, Gf.Vec3f):
            return [[float(value[0]), float(value[1]), float(value[2])]]
        return [value]

    def _get_material_wrapper(self, material_path: str):
        self._try_enable_omnipbr_wrapper()
        wrapper = self._material_wrappers.get(material_path)
        if wrapper is not None:
            return wrapper
        if self._omnipbr_cls is None:
            raise RuntimeError("[GARMENT_MGR] OmniPbrMaterial class is not initialized.")
        try:
            wrapper = self._omnipbr_cls([str(material_path)])
            if getattr(wrapper, "valid", True):
                self._material_wrappers[material_path] = wrapper
                return wrapper
        except Exception as exc:
            raise RuntimeError(
                f"[GARMENT_MGR] Failed to initialize OmniPbrMaterial for {material_path}: {exc}"
            ) from exc
        raise RuntimeError(f"[GARMENT_MGR] OmniPbrMaterial wrapper is invalid for {material_path}.")

    def _set_omnipbr_input(self, material_path: str, name: str, value):
        wrapper = self._get_material_wrapper(material_path)
        try:
            wrapper.set_input_values(name=name, values=self._wrapper_value(value))
        except Exception as exc:
            raise RuntimeError(
                f"[GARMENT_MGR] OmniPbrMaterial.set_input_values failed for '{name}' at {material_path}: {exc}"
            ) from exc

    def _apply_preview_materials(self, texture_overrides=None):
        """Bind OmniPBR per garment mesh using UV mapping only.

        Args:
            texture_overrides: optional per-env external texture set.
        """
        from pxr import UsdGeom, Gf
        
        alb_rng = (1.0, 1.0)
        fix = getattr(self.env_cfg, "material_fix", {}) or {}
        rough_min = float(fix.get("roughness_min", 0.0))
        metal_fix = float(fix.get("metalness_value", 0.0))
        albedo_fix = float(fix.get("albedo_scale", 1.0))
        for i, prim_path in enumerate(self._prim_paths):
            mesh_path = f"{prim_path}/mesh"
            mesh_prim = self._stage.GetPrimAtPath(mesh_path)
            if not mesh_prim or not mesh_prim.IsValid():
                continue

            # Per-garment mild albedo jitter (kept minimal for consistency)
            alb_scale = float(self._preview_rng.uniform(*alb_rng)) * albedo_fix

            textures = self._resolve_texture_paths(
                i,
                external_texture_set=texture_overrides.get(i) if texture_overrides else None
            )

            base = textures['basecolor']
            normal = textures['normal']
            rough = textures['roughness']
            metal = textures['metallic']
            mrpack = textures['mrpack']
            opacity = textures['opacity']
            material_path = str(Sdf.Path(prim_path).AppendPath("Looks/OmniPBR"))
            # OmniPbrMaterial creates missing prims when path does not exist.
            self._get_material_wrapper(material_path)

            def _set(name, value):
                try:
                    self._set_omnipbr_input(material_path, name, value)
                except Exception as exc:
                    print(f"[OMNIPBR] set input failed ({name}): {exc}")
                    raise

            # Reset critical toggles / texture slots to avoid stale state across randomizations
            _set("diffuse_texture", "")
            _set("normalmap_texture", "")
            _set("reflectionroughness_texture", "")
            _set("metallic_texture", "")
            _set("ORM_texture", "")
            _set("enable_ORM_texture", False)
            _set("enable_opacity", False)
            _set("enable_opacity_texture", False)

            # Diffuse / basecolor
            if base:
                _set("diffuse_texture", str(base))
            else:
                base_val = 0.6 * alb_scale
                _set("diffuse_color_constant", Gf.Vec3f(base_val, base_val, base_val))

            # Normal
            if normal:
                _set("normalmap_texture", str(normal))

            # Roughness / ORM
            if rough:
                _set("reflectionroughness_texture", str(rough))
                _set("enable_ORM_texture", False)
            elif mrpack:
                _set("enable_ORM_texture", True)
                _set("ORM_texture", str(mrpack))
            else:
                _set("reflection_roughness_constant", max(0.5, rough_min))
                _set("enable_ORM_texture", False)

            # Metallic
            if metal:
                _set("metallic_texture", str(metal))
            elif mrpack:
                _set("enable_ORM_texture", True)
                _set("ORM_texture", str(mrpack))
            else:
                _set("metallic_constant", max(0.0, min(1.0, metal_fix)))

            # Opacity (optional)
            if opacity:
                _set("enable_opacity", True)
                _set("enable_opacity_texture", True)
                _set("opacity_texture", str(opacity))

            # Unified mapping: always use UVs, no project_uvw/triplanar branch.
            _set("project_uvw", False)
            _set("texture_rotate", 0.0)
            _set("texture_scale", Gf.Vec2f(1.0, 1.0))
            _set("texture_translate", Gf.Vec2f(0.0, 0.0))

            # Cloth spec tweak: low specular for fabric
            _set("specular_level", 0.08)

            UsdGeom.Mesh(mesh_prim).CreateDoubleSidedAttr(True).Set(True)
            material_prim = self._stage.GetPrimAtPath(material_path)
            if not material_prim.IsValid():
                raise RuntimeError(f"[GARMENT_MGR] OmniPBR material prim missing at {material_path}")
            shade_mat = UsdShade.Material(material_prim)
            if not shade_mat.GetPrim().IsValid():
                raise RuntimeError(f"[GARMENT_MGR] Invalid UsdShade.Material at {material_path}")
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).UnbindAllBindings()
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(shade_mat, UsdShade.Tokens.strongerThanDescendants)
    # =========================================================================
    # Part 2: Physics View (Initialization & Tensor API)
    # =========================================================================
    
    def _initialize_impl(self):
        """Called by AssetBase.initialize(). Creates the ParticleClothView."""
        from isaacsim.core.simulation_manager import SimulationManager
        
        # Create View
        physics_view = physx_api.create_simulation_view(SimulationManager.get_backend())
        
        # Our regex: /World/Cloth/env_.*/garment/mesh -> /World/Cloth/env_*/garment/mesh
        pattern = self.cfg.prim_path.replace(".*", "*")
        
        self._view = physics_view.create_particle_cloth_view(pattern)
        
        if self._view is None or (hasattr(self._view, '_backend') and self._view._backend is None):
            # Fallback for systems that might use different pathing
            print("[GARMENT_MGR] View creation failed, trying fallback...")
            alt = pattern.replace("/garment/mesh", "/particleSystem")
            self._view = physics_view.create_particle_cloth_view(alt)
            
        if self._view is None:
             raise RuntimeError("Failed to create ParticleClothView")
             
        # Initialize State
        self._refresh_particle_counts()
        self._populate_env_data()
            
        self._is_initialized = True

    def _refresh_particle_counts(self):
        """Infers particle counts from masses."""
        masses = self._view.get_masses()
        if masses is None: return
        
        masses_flat = masses.view(masses.shape[0], -1)
        counts = (masses_flat > 0).sum(dim=1)
        self._num_particles_per_env_dict = {i: int(c.item()) for i, c in enumerate(counts)}

    # =========================================================================
    # Part 3: Tensor API (Getters/Setters)
    # =========================================================================

    def _get_particle_positions(self) -> torch.Tensor:
        pos = self._view.get_positions()
        return pos.view(pos.shape[0], -1, 3)

    def _get_particle_velocities(self) -> torch.Tensor:
        vel = self._view.get_velocities()
        return vel.view(vel.shape[0], -1, 3)

    def _set_particle_positions(self, positions: torch.Tensor, indices: torch.Tensor):
        if self._view is None: return
        self._view.set_positions(positions.view(self.num_envs, -1), indices=indices)

    def _set_particle_velocities(self, velocities: torch.Tensor, indices: torch.Tensor):
        if self._view is None: return
        self._view.set_velocities(velocities.view(self.num_envs, -1), indices=indices)

    def _get_particle_masses(self):
        if self._view is None: return None
        return self._view.get_masses()
    
    def _set_particle_masses(self, masses, indices):
        if self._view is None: return
        self._view.set_masses(masses, indices=indices)

    def reset_to_poses(self, env_ids: torch.Tensor, root_pos: torch.Tensor, root_rot: torch.Tensor):
        """Resets particles to template pose transformed by root_pos/rot."""
        if not self._is_initialized: return
        
        env_ids_list = env_ids.long().cpu().tolist()
        all_pos = self._get_particle_positions()
        all_vel = self._get_particle_velocities()
        
        for i, env_id in enumerate(env_ids_list):
            if env_id not in self._template_pos_per_env: continue
            
            template = self._template_pos_per_env[env_id]
            r_pos = root_pos[i]
            r_rot = root_rot[i]
            
            # Apply Transform
            # If Identity
            if torch.allclose(r_rot, torch.tensor([1., 0., 0., 0.], device=self.device), atol=1e-6):
                target = template + r_pos
            else:
                target = math_utils.transform_points(template.unsqueeze(0), pos=r_pos.unsqueeze(0), quat=r_rot.unsqueeze(0))[0]
            
            all_pos[env_id, :target.shape[0]] = target
            all_vel[env_id] = 0.0
            
        indices = env_ids.to(dtype=torch.long, device=self.device)
        self._view.set_positions(all_pos.view(self.num_envs, -1), indices=indices)
        self._view.set_velocities(all_vel.view(self.num_envs, -1), indices=indices)

    def reset(self, env_ids: Sequence[int] | None = None):
        pass

    # =========================================================================
    # Part 4: Cache & Features
    # =========================================================================

    def _populate_env_data(self):
        """Loads cached init_pos, features and initializes templates for each environment."""
        all_pos = self._get_particle_positions()
        
        # Check if physics view has fewer cloths than expected (mesh cooking failures)
        actual_cloth_count = all_pos.shape[0]
        if actual_cloth_count < self.num_envs:
            print(f"[GARMENT_MGR] WARNING: Physics view has {actual_cloth_count} cloths but expected {self.num_envs}. Some meshes may have failed to cook.")
        
        self.max_particles = all_pos.shape[1] 
        self.init_pos = torch.zeros((self.num_envs, self.max_particles, 3), device=self.device)

        self._padding_mask = torch.zeros((self.num_envs, self.max_particles, 1), device=self.device)
        self._sampling_mask = torch.zeros((self.num_envs, self.max_particles, 1), device=self.device)

        for env_idx, path in enumerate(self._env_usd_paths):
            usd_path = Path(path)
            
            sample_mask_path = usd_path.parent / "sample_mask.npy"

            if (usd_path.parent / "init_pos.npy").exists():
                try:
                    npy_path = usd_path.parent / "init_pos.npy"
                    pos = torch.as_tensor(np.load(npy_path), device=self.device, dtype=torch.float32)
                    min_particles = min(pos.shape[0], self.max_particles)
                    self.init_pos[env_idx, :min_particles] = pos[:min_particles]
                except Exception as e:
                    print(f"[GARMENT_MGR] Warning: Failed to load init_pos.npy for {usd_path.name}: {e}")
        
            # Guard against mesh cooking failures
            if env_idx >= actual_cloth_count:
                print(f"[GARMENT_MGR] WARNING: Skipping env_{env_idx} - mesh cooking likely failed")
                continue
                
            template_world = all_pos[env_idx].clone().to(self.device)
            count = self._num_particles_per_env_dict.get(env_idx, 0)
            offset = self._initial_world_offset.get(env_idx)
            if count > 0 and offset is not None:
                self._template_pos_per_env[env_idx] = template_world[:count] - offset
                self._padding_mask[env_idx, :count] = 1.0

                # Load sampling mask if available; otherwise fall back to padding mask (all ones for real verts)
                if sample_mask_path.exists():
                    try:
                        sample_mask_np = np.load(sample_mask_path)
                        sample_mask_np = sample_mask_np.reshape(-1, 1)
                        sample_min = min(sample_mask_np.shape[0], count, self.max_particles)
                        self._sampling_mask[env_idx, :sample_min] = torch.as_tensor(sample_mask_np[:sample_min], device=self.device, dtype=torch.float32)
                        # Ensure masked-out padding is zero
                        if sample_min < self.max_particles:
                            self._sampling_mask[env_idx, sample_min:] = 0.0
                    except Exception as e:
                        print(f"[GARMENT_MGR] Warning: Failed to load sample_mask.npy for {usd_path.name}: {e}")
                        self._sampling_mask[env_idx, :count] = 1.0
                else:
                    self._sampling_mask[env_idx, :count] = 1.0
        return

    def get_feature_for_env(self, env_id: int) -> Optional[torch.Tensor]:
        return self._features_cache.get(env_id)

    def get_padding_mask(self):
        return self._padding_mask
    
    def get_sampling_mask(self):
        # Fall back to padding mask if sampling mask is not set
        return self._sampling_mask if hasattr(self, "_sampling_mask") else self.get_padding_mask()
        
    # =========================================================================
    # Required by AssetBase
    # =========================================================================
    @property
    def data(self): return None
    
    @property
    def num_instances(self): return self.num_envs
    
    def update(self, dt): pass
    
    def write_data_to_sim(self): pass


# Backwards compatibility alias
GarmentManager = PBDGarmentManager
