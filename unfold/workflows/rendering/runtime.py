#!/usr/bin/env python3
"""Runtime facade for Isaac/Replicator stage, render and annotator access."""

import numpy as np


class RenderRuntimeFacade:
    """Thin facade that centralizes stage/render-product/annotator operations."""

    def __init__(
        self,
        cfg,
        args,
        *,
        extract_camera_parameters,
    ):
        self.cfg = cfg
        self.args = args
        self._extract_camera_parameters = extract_camera_parameters

    @staticmethod
    def get_stage():
        import omni.usd

        return omni.usd.get_context().get_stage()

    @staticmethod
    def set_render_products_updates(render_products, enabled: bool):
        for rp in render_products:
            rp.hydra_texture.set_updates_enabled(bool(enabled))

    @staticmethod
    def destroy_render_products_and_annotators(render_products, annotators):
        for annos in (annotators or {}).values():
            for annotator in (annos or {}).values():
                try:
                    annotator.detach()
                except Exception:
                    pass
        for rp in render_products or []:
            try:
                rp.destroy()
            except Exception:
                pass

    def render_subframes(self, env):
        rt_subframes = int(getattr(self.cfg, "replicator", {}).get("rt_subframes", self.args.spp))
        if rt_subframes < 1:
            rt_subframes = 1
        for _ in range(rt_subframes):
            env.unwrapped.sim.render()

    @staticmethod
    def warmup(env, num_frames: int = 3):
        for _ in range(int(num_frames)):
            env.unwrapped.sim.render()

    @staticmethod
    def send_event(rep, event_name: str):
        rep.utils.send_og_event(event_name=event_name)

    @staticmethod
    def extract_cloth_mask_from_semantic(semantic_data):
        mask_np = None
        error_msg = None
        if semantic_data and "data" in semantic_data and "info" in semantic_data:
            id_to_labels = semantic_data["info"].get("idToLabels", {})
            cloth_id = None
            for key, val in id_to_labels.items():
                if isinstance(val, dict):
                    v = val.get("class", "")
                else:
                    v = str(val)
                if "cloth" in v.lower():
                    cloth_id = int(key)
                    break
            sem_arr = np.array(semantic_data["data"])
            if cloth_id is not None:
                mask_np = (sem_arr == cloth_id).astype(np.uint8)
                if int(mask_np.sum()) <= 0:
                    error_msg = "semantic cloth label resolved but produced empty mask"
            else:
                semantic_labels = sorted(
                    {
                        str(val.get("class", "")) if isinstance(val, dict) else str(val)
                        for val in id_to_labels.values()
                    }
                )
                error_msg = (
                    "semantic cloth label missing; "
                    f"available_labels={semantic_labels}"
                )
        else:
            error_msg = "semantic annotator returned no usable data/info payload"
        return mask_np, error_msg

    def collect_camera_frame_data(self, cam_path, anns, stage, image_width, image_height):
        rgb = anns["rgb"].get_data()
        depth = anns["depth"].get_data()
        mask_np = None
        seg_error = None
        if "semantic" in anns:
            semantic_data = anns["semantic"].get_data()
            mask_np, seg_error = self.extract_cloth_mask_from_semantic(semantic_data)
        else:
            seg_error = "semantic annotator disabled or unavailable"

        K, w2c = self._extract_camera_parameters(cam_path, stage, image_width, image_height)
        return rgb, depth, mask_np, K, w2c, seg_error

    def create_camera_prims(self, rl_env, cloth_root: str):
        from pxr import Gf, UsdGeom

        stage = self.get_stage()
        cameras_all = []
        for i in range(rl_env.num_envs):
            for j in range(getattr(self.cfg, "mv_num_views", 1)):
                base_path = f"{cloth_root.rstrip('/')}/env_{i}/view_{j}/cam"
                cam_geom = UsdGeom.Camera.Define(stage, base_path)
                cam_geom.CreateFocalLengthAttr().Set(24.0)
                cam_geom.CreateFocusDistanceAttr().Set(400.0)
                cam_geom.CreateFStopAttr().Set(0.0)
                cam_geom.CreateHorizontalApertureAttr().Set(20.955)
                cam_geom.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))
                cameras_all.append(base_path)
        return cameras_all

    def create_render_products_and_annotators(
        self,
        cameras_all,
    ):
        import omni.replicator.core as rep

        cam_res = getattr(self.cfg, "camera_res", [1024, 1024])
        render_products = []
        all_annotators = {}

        print("[SDG] Setting up Replicator render products and annotators...")

        for cam_path in cameras_all:
            if isinstance(cam_path, list):
                cam_path = cam_path[0]
            rp = rep.create.render_product(cam_path, resolution=cam_res)
            render_products.append(rp)

            if self.args.disable_rp_between_captures:
                rp.hydra_texture.set_updates_enabled(False)

            annos = {
                "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
                "depth": rep.AnnotatorRegistry.get_annotator("distance_to_camera"),
            }
            if bool(getattr(self.args, "enable_semantic_seg", True)):
                annos["semantic"] = rep.AnnotatorRegistry.get_annotator(
                    "semantic_segmentation", init_params={"semanticTypes": ["class"]}
                )
            annos["rgb"].attach(rp)
            annos["depth"].attach(rp)
            if "semantic" in annos:
                annos["semantic"].attach(rp)

            all_annotators[cam_path] = annos

        return render_products, all_annotators
