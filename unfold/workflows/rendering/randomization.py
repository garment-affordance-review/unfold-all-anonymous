#!/usr/bin/env python3
"""Facade for Replicator randomization registration and triggering."""


def _make_seeded_rng(env, offset: int):
    import numpy as np

    base_seed = getattr(getattr(env, "unwrapped", env), "cfg", None)
    base_seed = getattr(base_seed, "seed", None)
    seed = None if base_seed is None else int(base_seed) + int(offset)
    return np.random.default_rng(seed)


class SurfaceTextureCatalog:
    """Scan SurfaceMaterials texture folders and build OmniPBR-ready texture sets."""

    def __init__(self, root_path):
        from pathlib import Path

        self.root = Path(root_path)
        self.texture_sets = []
        self._scan()

    @staticmethod
    def _pick_file(folder, keys):
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts]
        for p in files:
            name = p.name.lower()
            if any(k in name for k in keys):
                return str(p.resolve())
        return None

    def _scan_folder(self, folder, set_id):
        base = self._pick_file(folder, ["basecolor", "_diff", "diffuse", "albedo"])
        normal = self._pick_file(folder, ["_normal", "_norm", "_n."])
        orm = self._pick_file(folder, ["_orm", "orm", "multi_r_rough_g_ao"])
        rough = self._pick_file(folder, ["roughness", "_rough"])
        metallic = self._pick_file(folder, ["metallic", "metalness", "_metal"])
        if base is None:
            return
        if orm is None and rough is None and metallic is None:
            return
        self.texture_sets.append(
            {
                "id": str(set_id),
                "basecolor": base,
                "normal": normal,
                "orm": orm,
                "roughness": rough,
                "metallic": metallic,
            }
        )

    def _scan(self):
        if not self.root.exists():
            print(f"[GROUND_MAT] SurfaceMaterials root missing: {self.root}")
            return
        for mat_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            # Prefer nested asset folder (e.g., M010/.../Concrete_Block/*.png), then textures/.
            nested_dirs = [p for p in sorted(mat_dir.iterdir()) if p.is_dir()]
            scanned = False
            for nd in nested_dirs:
                before = len(self.texture_sets)
                self._scan_folder(nd, f"{mat_dir.name}/{nd.name}")
                scanned = scanned or (len(self.texture_sets) > before)
            if not scanned:
                self._scan_folder(mat_dir, mat_dir.name)
        print(f"[GROUND_MAT] Loaded {len(self.texture_sets)} ground texture sets from {self.root}")


class GroundMaterialRandomizer:
    """Ground material randomization using OmniPbrMaterial only (no mdl authoring)."""

    def __init__(self, env, surface_material_root):
        import numpy as np
        from pxr import Sdf, UsdGeom, UsdShade, Usd
        from isaacsim.core.experimental.materials import OmniPbrMaterial

        self._np = np
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._UsdShade = UsdShade
        self._env = env
        self._stage = env.unwrapped.sim.stage
        self._catalog = SurfaceTextureCatalog(surface_material_root)
        if not self._catalog.texture_sets:
            raise RuntimeError("[GROUND_MAT] No valid texture set found in SurfaceMaterials.")
        self._rng = _make_seeded_rng(env, 101)
        patch = getattr(env.unwrapped.cfg, "ground_patch_size_m", None)
        if patch is None:
            raise RuntimeError("[GROUND_MAT] Missing required config: ground_patch_size_m")
        self._patch_size_m = float(patch)
        if not (self._patch_size_m > 0.0):
            raise RuntimeError(f"[GROUND_MAT] Invalid ground_patch_size_m: {patch}")
        self._material_path = str(Sdf.Path("/World/Environment/Ground/Looks/GroundOmniPBR"))
        self._wrapper = OmniPbrMaterial([self._material_path])
        if not getattr(self._wrapper, "valid", True):
            raise RuntimeError(f"[GROUND_MAT] OmniPbrMaterial wrapper invalid: {self._material_path}")
        self._mesh_paths = self._collect_ground_mesh_paths("/World/Environment/GroundVisual")
        self._bind_material_to_ground_meshes()

    @staticmethod
    def _wrapper_value(val):
        from pxr import Gf

        if isinstance(val, Gf.Vec2f):
            return [[float(val[0]), float(val[1])]]
        return [val]

    def _collect_ground_mesh_paths(self, root_path):
        from pxr import Usd

        root = self._stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"[GROUND_MAT] Ground root missing: {root_path}")
        mesh_paths = []
        for prim in Usd.PrimRange(root):
            try:
                if prim.GetPath() == root.GetPath():
                    continue
                if prim and prim.IsValid() and prim.IsA(self._UsdGeom.Mesh):
                    mesh_paths.append(str(prim.GetPath()))
            except Exception:
                continue
        if not mesh_paths:
            raise RuntimeError(f"[GROUND_MAT] No ground mesh found under {root_path}")
        return mesh_paths

    def _bind_material_to_ground_meshes(self):
        mat_prim = self._stage.GetPrimAtPath(self._material_path)
        if not mat_prim or not mat_prim.IsValid():
            raise RuntimeError(f"[GROUND_MAT] Material prim missing: {self._material_path}")
        shade_mat = self._UsdShade.Material(mat_prim)
        if not shade_mat.GetPrim().IsValid():
            raise RuntimeError(f"[GROUND_MAT] Invalid material at {self._material_path}")
        for mesh_path in self._mesh_paths:
            mesh_prim = self._stage.GetPrimAtPath(mesh_path)
            if not mesh_prim or not mesh_prim.IsValid():
                continue
            self._UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
                shade_mat, self._UsdShade.Tokens.strongerThanDescendants
            )

    def _set_input(self, name, value):
        self._wrapper.set_input_values(name=name, values=self._wrapper_value(value))

    def _ground_size_m(self):
        # Prefer explicitly computed size from Env._set_ground_size().
        cached = getattr(self._env.unwrapped, "_ground_size_m", None)
        if isinstance(cached, (list, tuple)) and len(cached) >= 2:
            return max(float(cached[0]), 1e-6), max(float(cached[1]), 1e-6)

        # Fallback: compute from mesh world bounds.
        bbox_cache = self._UsdGeom.BBoxCache(
            self._Usd.TimeCode.Default(), includedPurposes=[self._UsdGeom.Tokens.default_]
        )
        lo = [float("inf"), float("inf"), float("inf")]
        hi = [float("-inf"), float("-inf"), float("-inf")]
        for mesh_path in self._mesh_paths:
            prim = self._stage.GetPrimAtPath(mesh_path)
            if not prim or not prim.IsValid():
                continue
            rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            mn = rng.GetMin()
            mx = rng.GetMax()
            for i in range(3):
                lo[i] = min(lo[i], float(mn[i]))
                hi[i] = max(hi[i], float(mx[i]))
        if not (lo[0] < hi[0] and lo[1] < hi[1]):
            raise RuntimeError("[GROUND_MAT] Failed to compute ground world size for texture scaling.")
        return max(hi[0] - lo[0], 1e-6), max(hi[1] - lo[1], 1e-6)

    def randomize_once(self, capture_tag=None):
        from pxr import Gf

        tex = self._catalog.texture_sets[int(self._rng.integers(0, len(self._catalog.texture_sets)))]
        gx, gy = self._ground_size_m()
        tile_x = gx / self._patch_size_m
        tile_y = gy / self._patch_size_m
        sx = float(tile_x * self._rng.uniform(0.9, 1.1))
        sy = float(tile_y * self._rng.uniform(0.9, 1.1))
        tx = float(self._rng.uniform(0.0, 1.0))
        ty = float(self._rng.uniform(0.0, 1.0))
        rot = float(self._rng.choice([0, 90, 180, 270]))

        self._set_input("diffuse_texture", tex["basecolor"])
        self._set_input("normalmap_texture", tex["normal"] or "")
        if tex.get("orm"):
            self._set_input("enable_ORM_texture", True)
            self._set_input("ORM_texture", tex["orm"])
            self._set_input("reflectionroughness_texture", "")
            self._set_input("metallic_texture", "")
        else:
            self._set_input("enable_ORM_texture", False)
            self._set_input("ORM_texture", "")
            self._set_input("reflectionroughness_texture", tex.get("roughness") or "")
            self._set_input("metallic_texture", tex.get("metallic") or "")
        # Use UV-driven mapping on a dedicated plane mesh (no projected cubic seams).
        self._set_input("project_uvw", False)
        self._set_input("world_or_object", False)
        self._set_input("texture_scale", Gf.Vec2f(sx, sy))
        self._set_input("texture_translate", Gf.Vec2f(tx, ty))
        self._set_input("texture_rotate", rot)

        tag = capture_tag if capture_tag is not None else "-"
        print(
            f"[GROUND_MAT] tag={tag} texture={tex['id']} ground_size=({gx:.3f},{gy:.3f}) "
            f"patch={self._patch_size_m:.3f} scale=({sx:.3f},{sy:.3f}) "
            f"project_uvw=0 world_or_object=0 rot={int(rot)}"
        , flush=True)


class ClothMaterialRandomizer:
    """Cloth scalar-parameter randomization on the currently bound OmniPBR material."""

    def __init__(self, env, cloth_root: str, material_cfg):
        import numpy as np
        from isaacsim.core.experimental.materials import OmniPbrMaterial
        from pxr import UsdShade

        self._env = env
        self._stage = env.unwrapped.sim.stage
        self._num_envs = int(env.unwrapped.num_envs)
        self._cloth_root = str(cloth_root).rstrip("/") or "/World/Cloth"
        self._material_cfg = material_cfg or {}
        self._OmniPbrMaterial = OmniPbrMaterial
        self._UsdShade = UsdShade
        self._rng = _make_seeded_rng(env, 201)
        self._states = {}
        self.refresh_bindings()

    @staticmethod
    def _wrapper_value(value):
        from pxr import Gf

        if isinstance(value, Gf.Vec3f):
            return [[float(value[0]), float(value[1]), float(value[2])]]
        return [value]

    def _mesh_path(self, env_idx: int) -> str:
        return f"{self._cloth_root}/env_{env_idx}/garment/mesh"

    def refresh_bindings(self):
        self._states = {}
        valid = 0
        for env_idx in range(self._num_envs):
            mesh_prim = self._stage.GetPrimAtPath(self._mesh_path(env_idx))
            if not mesh_prim or not mesh_prim.IsValid():
                continue
            binding_api = self._UsdShade.MaterialBindingAPI(mesh_prim)
            bound = binding_api.ComputeBoundMaterial()
            mat = bound[0] if isinstance(bound, tuple) else bound
            if not mat or not mat.GetPrim().IsValid():
                continue
            material_path = str(mat.GetPath())
            try:
                wrapper = self._OmniPbrMaterial([material_path])
            except Exception:
                continue
            if not getattr(wrapper, "valid", True):
                continue
            self._states[env_idx] = {"material_path": material_path, "wrapper": wrapper}
            valid += 1
        print(f"[CLOTH_MAT] Bound OmniPBR wrappers for {valid}/{self._num_envs} cloth materials.")

    def randomize_once(self, capture_tag=None):
        from pxr import Gf

        albedo_jitter = DomainRandomizationFacade._range2(
            self._material_cfg.get("albedo_jitter") if isinstance(self._material_cfg, dict) else None,
            (1.0, 1.0),
        )
        roughness_jitter = DomainRandomizationFacade._range2(
            self._material_cfg.get("roughness_jitter") if isinstance(self._material_cfg, dict) else None,
            (0.6, 0.95),
        )
        metalness_range = DomainRandomizationFacade._range2(
            self._material_cfg.get("metalness_range") if isinstance(self._material_cfg, dict) else None,
            (0.0, 0.05),
        )

        if not self._states:
            self.refresh_bindings()

        tag = capture_tag if capture_tag is not None else "-"
        randomized = 0
        for state in self._states.values():
            wrapper = state["wrapper"]
            alb = float(self._rng.uniform(albedo_jitter[0], albedo_jitter[1]))
            rough = float(self._rng.uniform(roughness_jitter[0], roughness_jitter[1]))
            metal = float(self._rng.uniform(metalness_range[0], metalness_range[1]))
            try:
                wrapper.set_input_values(
                    name="diffuse_color_constant",
                    values=self._wrapper_value(Gf.Vec3f(alb, alb, alb)),
                )
                wrapper.set_input_values(name="reflection_roughness_constant", values=self._wrapper_value(rough))
                wrapper.set_input_values(name="metallic_constant", values=self._wrapper_value(metal))
                randomized += 1
            except Exception:
                continue
        print(
            f"[CLOTH_MAT] tag={tag} randomized={randomized}/{len(self._states)} "
            f"albedo=({albedo_jitter[0]:.2f},{albedo_jitter[1]:.2f}) "
            f"roughness=({roughness_jitter[0]:.2f},{roughness_jitter[1]:.2f}) "
            f"metalness=({metalness_range[0]:.2f},{metalness_range[1]:.2f})"
        )


class DomainRandomizationFacade:
    def __init__(self, event_name_map):
        self.event_name_map = event_name_map
        self._ground_randomizer = None
        self._cloth_randomizer = None

    @staticmethod
    def _cfg_section(cfg, *keys, default=None):
        cur = cfg
        for key in keys:
            if isinstance(cur, dict):
                cur = cur.get(key, None)
            else:
                cur = getattr(cur, key, None)
            if cur is None:
                return default
        return cur

    @staticmethod
    def _range2(value, default):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                lo = float(value[0])
                hi = float(value[1])
                return (lo, hi) if lo <= hi else (hi, lo)
            except Exception:
                pass
        return default

    @staticmethod
    def _choices(value, default):
        if isinstance(value, (list, tuple)):
            parsed = [item for item in value if item not in (None, "")]
            if parsed:
                return parsed
        return default

    def setup_dome_background_randomization(self, cfg, args):
        import omni.replicator.core as rep

        if args.no_dome_bg:
            return None

        light_cfg = self._cfg_section(cfg, "replicator", "light", default={}) or {}
        intensity_min, intensity_max = self._range2(
            self._cfg_section(
                light_cfg,
                "dome_intensity_range",
                default=self._cfg_section(light_cfg, "intensity_range", default=None),
            ),
            (150.0, 1200.0),
        )
        dome_intensity = int(round(0.5 * (float(intensity_min) + float(intensity_max))))
        dome_exposure_min, dome_exposure_max = self._range2(
            self._cfg_section(light_cfg, "dome_exposure_range", default=None),
            (-1.5, 0.0),
        )

        # Keep horizon upright: randomize yaw only to avoid floor/sky flips.
        yaw_min, yaw_max = self._range2(
            self._cfg_section(light_cfg, "dome_yaw_range_deg", default=None),
            (-180.0, 180.0),
        )

        default_hdri_urls = [
            "${PROJECT_ROOT}/data/assets/material/DomeLights/qwantani_4k.hdr",
            "${PROJECT_ROOT}/data/assets/material/DomeLights/champagne_castle_1_4k.hdr",
            "${PROJECT_ROOT}/data/assets/material/DomeLights/moonlit_golf_4k.hdr",
            "${PROJECT_ROOT}/data/assets/material/DomeLights/adams_place_bridge_4k.hdr",
            "${PROJECT_ROOT}/data/assets/material/DomeLights/photo_studio_01_4k.hdr",
            "${PROJECT_ROOT}/data/assets/material/DomeLights/approaching_storm_4k.hdr",
        ]
        hdri_urls = self._choices(
            self._cfg_section(light_cfg, "hdris", default=None),
            default_hdri_urls,
        )

        print("[SDG] Setting up Dome Background Randomizer...")
        dome = rep.create.light(
            light_type="dome",
            rotation=rep.distribution.uniform((0.0, 0.0, yaw_min), (0.0, 0.0, yaw_max)),
            texture=rep.distribution.choice(hdri_urls),
            intensity=dome_intensity,
            exposure=rep.distribution.uniform(dome_exposure_min, dome_exposure_max),
        )

        with rep.trigger.on_custom_event(event_name=self.event_name_map["dome_background"]):
            with dome:
                rep.modify.attribute("inputs:texture:file", rep.distribution.choice(hdri_urls))
                rep.modify.attribute(
                    "inputs:exposure",
                    rep.distribution.uniform(dome_exposure_min, dome_exposure_max),
                )
                rep.modify.pose(rotation=rep.distribution.uniform((0.0, 0.0, yaw_min), (0.0, 0.0, yaw_max)))

        return dome

    def setup_multi_light_randomization(self, cfg, args):
        import omni.replicator.core as rep
        light_cfg = self._cfg_section(cfg, "replicator", "light", default={}) or {}
        intensity_range = self._range2(
            self._cfg_section(
                light_cfg,
                "extra_light_intensity_range",
                default=self._cfg_section(light_cfg, "intensity_range", default=None),
            ),
            (100.0, 1800.0),
        )
        color_temp_range = self._range2(
            self._cfg_section(light_cfg, "color_temp_range", default=None),
            (3500.0, 7500.0),
        )
        height_range = self._range2(
            self._cfg_section(light_cfg, "height_range", default=None),
            (2.0, 6.0),
        )
        radius_range = self._range2(
            self._cfg_section(light_cfg, "radius_range", default=None),
            (0.1, 1.0),
        )
        position_jitter = float(self._cfg_section(light_cfg, "position_jitter", default=3.0) or 3.0)
        num_extra = int(self._cfg_section(light_cfg, "num_extra_lights", default=2) or 2)

        if args.no_extra_lights or num_extra <= 0:
            return

        print(f"[SDG] Setting up Multi-Light Randomizer ({num_extra} extra lights)...")
        lights = rep.create.light(
            light_type="sphere",
            count=num_extra,
            position=rep.distribution.uniform(
                (-position_jitter, -position_jitter, height_range[0]),
                (position_jitter, position_jitter, height_range[1]),
            ),
            intensity=rep.distribution.uniform(intensity_range[0], intensity_range[1]),
            temperature=rep.distribution.uniform(color_temp_range[0], color_temp_range[1]),
            scale=rep.distribution.uniform(
                (radius_range[0], radius_range[0], radius_range[0]),
                (radius_range[1], radius_range[1], radius_range[1]),
            ),
        )

        with rep.trigger.on_custom_event(event_name=self.event_name_map["lights"]):
            with lights:
                rep.modify.attribute(
                    "xformOp:translate",
                    rep.distribution.uniform(
                        (-position_jitter, -position_jitter, height_range[0]),
                        (position_jitter, position_jitter, height_range[1]),
                    ),
                )
                rep.modify.attribute("inputs:intensity", rep.distribution.uniform(intensity_range[0], intensity_range[1]))
                rep.modify.attribute("inputs:colorTemperature", rep.distribution.uniform(color_temp_range[0], color_temp_range[1]))
                rep.modify.attribute(
                    "xformOp:scale",
                    rep.distribution.uniform(
                        (radius_range[0], radius_range[0], radius_range[0]),
                        (radius_range[1], radius_range[1], radius_range[1]),
                    ),
                )

    def randomize_camera_intrinsics(self, cam_group, cfg, args):
        import omni.replicator.core as rep

        if args.no_cam_intrinsics:
            return

        rep_cam_cfg = getattr(cfg, "replicator", {}).get("camera", {})
        f_range = rep_cam_cfg.get("focal_length_range", [24.0, 35.0])
        if not isinstance(f_range, (list, tuple)) or len(f_range) < 2:
            f_range = [24.0, 35.0]
        f_min, f_max = float(f_range[0]), float(f_range[1])
        if f_max < f_min:
            f_min, f_max = f_max, f_min

        print(f"[SDG] Setting up Camera Intrinsics Randomizer (event mode, focalLength in [{f_min:.1f}, {f_max:.1f}]mm)...")
        with rep.trigger.on_custom_event(event_name=self.event_name_map["camera_intrinsics"]):
            with cam_group:
                rep.modify.attribute("focalLength", rep.distribution.uniform(f_min, f_max))

    def randomize_cloth_material(self, cfg, args):
        if args.no_material_rand:
            return

        material_cfg = self._cfg_section(cfg, "replicator", "material", default={}) or {}
        print("[SDG] Setting up Cloth Material Randomizer...")
        cloth_root = str(getattr(args, "cloth_root", "/World/Cloth")).rstrip("/") or "/World/Cloth"
        self._cloth_randomizer = ClothMaterialRandomizer(
            env=self._env,
            cloth_root=cloth_root,
            material_cfg=material_cfg,
        )

    def setup_native_randomization(self, env, cfg, cam_group, args):
        self._env = env
        # For rendering pipeline, disable env-layer main light so lighting is driven by
        # Replicator dome + optional extra lights only.
        try:
            stage = env.unwrapped.sim.stage
            env_light = stage.GetPrimAtPath("/World/Environment/Light")
            if env_light and env_light.IsValid():
                stage.RemovePrim("/World/Environment/Light")
                print("[SDG] Disabled env main light: /World/Environment/Light")
        except Exception as e:
            print(f"[SDG] Warning: failed to disable env main light ({e})")

        self.setup_dome_background_randomization(cfg, args)
        self.setup_multi_light_randomization(cfg, args)

        rep_cam_cfg = getattr(cfg, "replicator", {}).get("camera", {})
        intrinsics_mode = str(rep_cam_cfg.get("intrinsics_mode", "synchronized")).lower()
        if intrinsics_mode == "event":
            self.randomize_camera_intrinsics(cam_group, cfg, args)
        else:
            print(f"[SDG] Camera intrinsics randomization mode: {intrinsics_mode} (applied during pose randomization).")

        self.randomize_cloth_material(cfg, args)
        if not args.no_ground_color:
            root = "${PROJECT_ROOT}/data/assets/material/SurfaceMaterials"
            self._ground_randomizer = GroundMaterialRandomizer(env=env, surface_material_root=root)
            print("[SDG] Ground material randomizer enabled (OmniPbrMaterial).")

    def randomize_ground_material(self, capture_tag=None):
        if self._ground_randomizer is None:
            return
        self._ground_randomizer.randomize_once(capture_tag=capture_tag)

    def randomize_cloth_material_once(self, capture_tag=None):
        if self._cloth_randomizer is None:
            return
        self._cloth_randomizer.randomize_once(capture_tag=capture_tag)
