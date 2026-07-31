#!/usr/bin/env python3
"""Fabric texture catalog and cloth material switcher for replicator v2."""

import collections
import math
from pathlib import Path

import numpy as np


def _make_seeded_rng(env, offset: int):
    base_seed = getattr(getattr(env, "unwrapped", env), "cfg", None)
    base_seed = getattr(base_seed, "seed", None)
    seed = None if base_seed is None else int(base_seed) + int(offset)
    return np.random.default_rng(seed)


class FabricTextureCatalog:
    """Scan Fabric and cache valid external PBR texture sets."""

    def __init__(self, fabric_root: Path):
        self.fabric_root = Path(fabric_root)
        self.texture_sets = []
        self._scan_textures()

    def _scan_textures(self):
        if not self.fabric_root.exists():
            print(f"[FABRIC_CATALOG] Warning: {self.fabric_root} not found")
            return

        required = ("basecolor", "normal", "roughness", "metallic")
        for subdir in sorted(self.fabric_root.iterdir()):
            if not subdir.is_dir():
                continue
            paths = {name: subdir / f"{name}.png" for name in required}
            if all(p.exists() for p in paths.values()):
                self.texture_sets.append(
                    {"id": subdir.name, **{k: str(v.resolve()) for k, v in paths.items()}}
                )

        print(f"[FABRIC_CATALOG] Loaded {len(self.texture_sets)} texture sets from {self.fabric_root}")


class ClothMaterialSwitcher:
    """Per-capture cloth texture randomizer by updating currently bound OmniPBR inputs only."""

    _ASSET_PATH_INPUTS = {
        "diffuse_texture",
        "normalmap_texture",
        "reflectionroughness_texture",
        "metallic_texture",
        "ORM_texture",
        "opacity_texture",
    }

    def __init__(
        self,
        env,
        texture_catalog: FabricTextureCatalog,
        probability: float,
        cloth_root: str,
        patch_size_m: float,
        texture_scale_jitter=(0.85, 1.15),
        texture_rotate_choices=(0, 90, 180, 270),
        texture_translate_range=(0.0, 1.0),
    ):
        self._env = env
        self._stage = env.unwrapped.sim.stage
        self._num_envs = int(env.unwrapped.num_envs)
        self._texture_catalog = texture_catalog
        self._probability = float(np.clip(probability, 0.0, 1.0))
        self._cloth_root = str(cloth_root).rstrip("/") or "/World/Cloth"
        self._patch_size_m = max(float(patch_size_m), 1e-6)
        self._texture_scale_jitter = tuple(float(v) for v in texture_scale_jitter)
        self._texture_rotate_choices = tuple(int(v) for v in texture_rotate_choices)
        self._texture_translate_range = tuple(float(v) for v in texture_translate_range)
        self._rng = _make_seeded_rng(env, 301)
        self._env_states = {}
        self._asset_size_ref_cache = {}
        self._failure_stats = collections.Counter()
        self._omnipbr_cls = None
        self._omnipbr_cls_checked = False
        self.refresh_original_bindings()
        self._try_enable_omnipbr_wrapper()

    @staticmethod
    def _asset_to_str(val):
        if val is None:
            return None
        path_attr = getattr(val, "path", None)
        if path_attr is not None:
            return str(path_attr)
        return str(val)

    def _mesh_path(self, env_idx: int) -> str:
        return f"{self._cloth_root}/env_{env_idx}/garment/mesh"

    def _compute_bound_material_path(self, mesh_prim):
        from pxr import UsdShade

        binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
        bound = binding_api.ComputeBoundMaterial()
        mat = bound[0] if isinstance(bound, tuple) else bound
        if mat and mat.GetPrim().IsValid():
            return str(mat.GetPath())
        return None

    def _get_bound_shader(self, mesh_prim):
        import omni.usd
        from pxr import UsdShade

        material_path = self._compute_bound_material_path(mesh_prim)
        if not material_path:
            return None, None, None
        mat_prim = self._stage.GetPrimAtPath(material_path)
        if not mat_prim or not mat_prim.IsValid():
            return material_path, None, None

        shade_mat = UsdShade.Material(mat_prim)
        shader_prim = omni.usd.get_shader_from_material(shade_mat.GetPrim(), get_prim=True)
        if not shader_prim:
            return material_path, shade_mat, None
        return material_path, shade_mat, UsdShade.Shader(shader_prim)

    def _get_shader_input_value(self, shader, input_name):
        if not shader:
            return None
        inp = shader.GetInput(input_name)
        if not inp:
            return None
        return inp.Get()

    def _try_enable_omnipbr_wrapper(self):
        """Require high-level OmniPBR wrapper (fail-fast if unavailable)."""
        if self._omnipbr_cls_checked:
            return self._omnipbr_cls
        self._omnipbr_cls_checked = True
        try:
            from isaacsim.core.experimental.materials import OmniPbrMaterial

            self._omnipbr_cls = OmniPbrMaterial
            print("[FABRIC_SWITCH] OmniPbrMaterial wrapper enabled.")
        except Exception as exc:
            raise RuntimeError(
                f"[FABRIC_SWITCH] OmniPbrMaterial is required but unavailable: {exc}"
            ) from exc
        return self._omnipbr_cls

    def _state_wrapper(self, state):
        """Create/get cached OmniPBR wrapper for one bound material (fail-fast)."""
        wrapper = state.get("wrapper")
        if wrapper is not None:
            return wrapper
        mat_path = state.get("material_path")
        if not mat_path:
            raise RuntimeError("[FABRIC_SWITCH] material_path is missing for OmniPbrMaterial binding.")
        if self._omnipbr_cls is None:
            self._try_enable_omnipbr_wrapper()
        try:
            wrapper = self._omnipbr_cls([str(mat_path)])
            if getattr(wrapper, "valid", True):
                state["wrapper"] = wrapper
                return wrapper
        except Exception as exc:
            self._failure_stats["wrapper_init_failed"] += 1
            raise RuntimeError(
                f"[FABRIC_SWITCH] Failed to initialize OmniPbrMaterial for {mat_path}: {exc}"
            ) from exc
        raise RuntimeError(f"[FABRIC_SWITCH] OmniPbrMaterial wrapper is invalid for {mat_path}.")

    @staticmethod
    def _wrapper_value(value):
        """Normalize single-value payload for OmniPBR wrapper API."""
        from pxr import Gf

        if isinstance(value, Gf.Vec2f):
            return [[float(value[0]), float(value[1])]]
        if isinstance(value, Gf.Vec3f):
            return [[float(value[0]), float(value[1]), float(value[2])]]
        return [value]

    def _set_material_input(self, state, input_name, value, create_if_missing=False):
        """Set one material input via OmniPbrMaterial only (fail-fast)."""
        _ = create_if_missing
        if input_name in self._ASSET_PATH_INPUTS and value is None:
            value = ""
        wrapper = self._state_wrapper(state)
        try:
            wrapper.set_input_values(name=input_name, values=self._wrapper_value(value))
            return True
        except Exception as exc:
            self._failure_stats["wrapper_set_failed"] += 1
            raise RuntimeError(
                f"[FABRIC_SWITCH] OmniPbrMaterial.set_input_values failed for '{input_name}': {exc}"
            ) from exc

    @staticmethod
    def _compute_triangle_area_sum(points: np.ndarray, tris: np.ndarray) -> float:
        if points.size == 0 or tris.size == 0:
            return 0.0
        tri_pts = points[tris]
        ab = tri_pts[:, 1] - tri_pts[:, 0]
        ac = tri_pts[:, 2] - tri_pts[:, 0]
        cross = np.cross(ab, ac)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        return float(area.sum())

    def _compute_size_ref_m(self, mesh_prim, asset_key: str) -> float:
        from pxr import Gf, UsdGeom

        cached = self._asset_size_ref_cache.get(asset_key)
        if cached is not None:
            return cached

        mesh = UsdGeom.Mesh(mesh_prim)
        points_attr = mesh.GetPointsAttr().Get()
        counts_attr = mesh.GetFaceVertexCountsAttr().Get()
        indices_attr = mesh.GetFaceVertexIndicesAttr().Get()
        points = np.asarray(points_attr, dtype=np.float64) if points_attr is not None else np.empty((0, 3), dtype=np.float64)
        if points.size == 0:
            raise RuntimeError(f"[FABRIC_SWITCH] Cannot compute size_ref_m for {asset_key}: mesh points are empty.")

        xform = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(0)
        points_world = np.empty_like(points, dtype=np.float64)
        for i, p in enumerate(points):
            pw = xform.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            points_world[i] = (float(pw[0]), float(pw[1]), float(pw[2]))
        points = points_world

        counts = np.asarray(counts_attr, dtype=np.int64) if counts_attr is not None else np.empty((0,), dtype=np.int64)
        indices = np.asarray(indices_attr, dtype=np.int64) if indices_attr is not None else np.empty((0,), dtype=np.int64)
        tris = []
        off = 0
        for c in counts:
            c = int(c)
            if c < 3:
                off += max(c, 0)
                continue
            fvi = indices[off: off + c]
            off += c
            if fvi.size != c:
                raise RuntimeError(
                    f"[FABRIC_SWITCH] Cannot compute size_ref_m for {asset_key}: malformed face indices."
                )
            if c == 3:
                tris.append([int(fvi[0]), int(fvi[1]), int(fvi[2])])
            else:
                for j in range(1, c - 1):
                    tris.append([int(fvi[0]), int(fvi[j]), int(fvi[j + 1])])

        if not tris:
            raise RuntimeError(f"[FABRIC_SWITCH] Cannot compute size_ref_m for {asset_key}: no valid triangles.")
        tri_arr = np.asarray(tris, dtype=np.int64)
        a_rest = self._compute_triangle_area_sum(points, tri_arr)
        if not np.isfinite(a_rest) or a_rest <= 1e-12:
            raise RuntimeError(
                f"[FABRIC_SWITCH] Cannot compute size_ref_m for {asset_key}: non-positive world area ({a_rest})."
            )
        size_ref_m = float(math.sqrt(a_rest))

        self._asset_size_ref_cache[asset_key] = size_ref_m
        return size_ref_m

    def _extract_asset_key(self, env_idx: int) -> str:
        mgr = getattr(self._env.unwrapped, "_garment_manager", None)
        usd_paths = getattr(mgr, "_env_usd_paths", None)
        if isinstance(usd_paths, (list, tuple)) and env_idx < len(usd_paths):
            usd = usd_paths[env_idx]
            if usd:
                return str(usd)
        return self._mesh_path(env_idx)

    def _snapshot_original_inputs(self, shader):
        keys = (
            "diffuse_texture",
            "normalmap_texture",
            "reflectionroughness_texture",
            "metallic_texture",
            "ORM_texture",
            "opacity_texture",
            "enable_ORM_texture",
            "enable_opacity",
            "enable_opacity_texture",
            "project_uvw",
            "texture_rotate",
            "texture_scale",
            "texture_translate",
        )
        return {k: self._get_shader_input_value(shader, k) for k in keys}

    def refresh_original_bindings(self):
        self._env_states = {}
        valid_meshes = 0
        for env_idx in range(self._num_envs):
            mesh_path = self._mesh_path(env_idx)
            mesh_prim = self._stage.GetPrimAtPath(mesh_path)
            if not mesh_prim or not mesh_prim.IsValid():
                continue
            asset_key = self._extract_asset_key(env_idx)
            size_ref_m = self._compute_size_ref_m(mesh_prim, asset_key=asset_key)
            tile_base = float(size_ref_m / self._patch_size_m)
            mat_path, _shade_mat, shader = self._get_bound_shader(mesh_prim)
            if not mat_path or shader is None:
                self._failure_stats["material_not_found"] += 1
                self._env_states[env_idx] = {
                    "mesh_path": mesh_path,
                    "material_path": mat_path,
                    "shader": None,
                    "wrapper": None,
                    "asset_key": asset_key,
                    "size_ref_m": size_ref_m,
                    "tile_base": tile_base,
                    "original": {},
                }
                continue
            valid_meshes += 1
            self._env_states[env_idx] = {
                "mesh_path": mesh_path,
                "material_path": mat_path,
                "shader": shader,
                "wrapper": None,
                "asset_key": asset_key,
                "size_ref_m": size_ref_m,
                "tile_base": tile_base,
                "original": self._snapshot_original_inputs(shader),
            }
        print(
            f"[FABRIC_SWITCH] Recorded original cloth bindings for {valid_meshes}/{self._num_envs} env meshes."
        )

    def _restore_original(self, state):
        if state.get("shader") is None:
            return False
        ok = True
        for key, value in state.get("original", {}).items():
            write_ok = self._set_material_input(state, key, value, create_if_missing=False)
            ok = ok and write_ok
            if not write_ok:
                self._failure_stats["parameter_write_failed"] += 1
        return ok

    def _sample_scale(self, tile_base: float):
        j0, j1 = self._texture_scale_jitter if len(self._texture_scale_jitter) >= 2 else (0.85, 1.15)
        if j1 < j0:
            j0, j1 = j1, j0
        main = float(tile_base * self._rng.uniform(j0, j1))
        sx = float(main * (1.0 + self._rng.uniform(-0.05, 0.05)))
        sy = float(main * (1.0 + self._rng.uniform(-0.05, 0.05)))
        return sx, sy

    def _sample_translate(self):
        t0, t1 = self._texture_translate_range if len(self._texture_translate_range) >= 2 else (0.0, 1.0)
        if t1 < t0:
            t0, t1 = t1, t0
        return float(self._rng.uniform(t0, t1)), float(self._rng.uniform(t0, t1))

    def _apply_external(self, state, tex_set):
        from pxr import Gf

        if state.get("shader") is None:
            self._failure_stats["material_not_found"] += 1
            return False, "material_not_found", None, None, None

        for key in ("basecolor", "normal", "roughness", "metallic"):
            if not Path(str(tex_set.get(key, ""))).exists():
                self._failure_stats["texture_missing"] += 1
                return False, "texture_missing", None, None, None

        size_ref_m = float(state["size_ref_m"])
        tile_base = float(state["tile_base"])
        sx, sy = self._sample_scale(tile_base)
        tx, ty = self._sample_translate()
        rotate = int(self._rng.choice(self._texture_rotate_choices or (0, 90, 180, 270)))

        writes = [
            ("diffuse_texture", tex_set["basecolor"], True, True),
            ("normalmap_texture", tex_set["normal"], True, True),
            ("reflectionroughness_texture", tex_set["roughness"], True, True),
            ("metallic_texture", tex_set["metallic"], True, True),
            ("enable_ORM_texture", False, True, True),
            ("ORM_texture", "", False, False),
            ("enable_opacity", False, True, False),
            ("enable_opacity_texture", False, True, False),
            ("opacity_texture", "", False, False),
            ("project_uvw", False, True, True),
            ("texture_scale", Gf.Vec2f(sx, sy), True, True),
            ("texture_translate", Gf.Vec2f(tx, ty), True, True),
            ("texture_rotate", float(rotate), False, False),
        ]
        required_ok = True
        for key, value, create_if_missing, required in writes:
            write_ok = self._set_material_input(state, key, value, create_if_missing=create_if_missing)
            if required:
                required_ok = required_ok and write_ok
            if not write_ok:
                self._failure_stats["parameter_write_failed"] += 1
        if not required_ok:
            return False, "parameter_write_failed", tile_base, (sx, sy), (tx, ty, rotate)
        return True, None, tile_base, (sx, sy), (tx, ty, rotate)

    def apply_for_capture(self, probability=None, capture_tag=None):
        if not self._env_states:
            self.refresh_original_bindings()

        p = self._probability if probability is None else float(np.clip(probability, 0.0, 1.0))
        external_count = 0
        restore_count = 0
        skipped_count = 0
        per_env_meta = {}

        for env_idx in range(self._num_envs):
            state = self._env_states.get(env_idx)
            if not state:
                skipped_count += 1
                per_env_meta[env_idx] = {
                    "texture_id": None,
                    "patch_size_m": float(self._patch_size_m),
                    "size_ref_m": None,
                    "tile_base": None,
                    "texture_scale": None,
                    "texture_rotate": None,
                    "texture_translate": None,
                    "external_texture_applied": False,
                    "texture_apply_failure_reason": "material_not_found",
                }
                continue

            size_ref_m = float(state["size_ref_m"])
            tile_base_default = float(state["tile_base"])
            use_external = bool(self._texture_catalog and self._texture_catalog.texture_sets) and (float(self._rng.random()) < p)
            if use_external:
                tex_set = self._texture_catalog.texture_sets[int(self._rng.integers(0, len(self._texture_catalog.texture_sets)))]
                ok, reason, tile_base, scale, trans_rot = self._apply_external(state, tex_set)
                external_count += int(ok)
                skipped_count += int(not ok)
                per_env_meta[env_idx] = {
                    "texture_id": tex_set.get("id"),
                    "patch_size_m": float(self._patch_size_m),
                    "size_ref_m": size_ref_m,
                    "tile_base": float(tile_base if tile_base is not None else tile_base_default),
                    "texture_scale": [float(scale[0]), float(scale[1])] if scale is not None else None,
                    "texture_rotate": int(trans_rot[2]) if trans_rot is not None else None,
                    "texture_translate": [float(trans_rot[0]), float(trans_rot[1])] if trans_rot is not None else None,
                    "external_texture_applied": bool(ok),
                    "texture_apply_failure_reason": reason,
                }
            else:
                ok = self._restore_original(state)
                restore_count += int(ok)
                skipped_count += int(not ok)
                per_env_meta[env_idx] = {
                    "texture_id": None,
                    "patch_size_m": float(self._patch_size_m),
                    "size_ref_m": size_ref_m,
                    "tile_base": tile_base_default,
                    "texture_scale": None,
                    "texture_rotate": None,
                    "texture_translate": None,
                    "external_texture_applied": False,
                    "texture_apply_failure_reason": None if ok else "parameter_write_failed",
                }

        tag = capture_tag if capture_tag is not None else "-"
        print(
            f"[FABRIC_SWITCH] tag={tag} external={external_count} restore={restore_count} skip={skipped_count}"
        )
        return per_env_meta

    def get_failure_stats(self):
        return dict(self._failure_stats)
