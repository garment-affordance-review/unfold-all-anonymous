"""Application entry for the rendering pipeline."""

from __future__ import annotations

import os
import random
from pathlib import Path
import yaml

from isaaclab.app import AppLauncher

from .cli import build_parser
from .pipeline import ClothMaterialSwitcher, FabricTextureCatalog, _run_epochs, _setup_replicator

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _peek_seed_settings(config_path: str | os.PathLike[str]) -> tuple[int | None, bool]:
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg_dict = yaml.full_load(f) or {}
    except Exception as exc:
        print(f"[WARN] failed to read seed settings from {config_path}: {exc}")
        return None, True
    seed = cfg_dict.get("seed", None)
    torch_deterministic = bool(cfg_dict.get("torch_deterministic", True))
    return seed, torch_deterministic


def _apply_global_seed(seed: int | None, *, torch_deterministic: bool) -> int | None:
    if seed is None:
        return None
    seed_i = int(seed)
    try:
        from isaaclab.utils.seed import configure_seed

        seed_i = configure_seed(seed_i, torch_deterministic=torch_deterministic)
        print(
            f"[SEED] Official IsaacLab seed configured: {seed_i} "
            f"(torch_deterministic={int(torch_deterministic)})"
        )
        return int(seed_i)
    except Exception as exc:
        try:
            import numpy as np
            import torch

            random.seed(seed_i)
            np.random.seed(seed_i)
            torch.manual_seed(seed_i)
            os.environ["PYTHONHASHSEED"] = str(seed_i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed_i)
                torch.cuda.manual_seed_all(seed_i)
            try:
                import warp as wp

                wp.rand_init(seed_i)
            except Exception:
                pass
            if torch_deterministic:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
            else:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            print(
                f"[SEED] Fallback pre-launch seed configured: {seed_i} "
                f"(torch_deterministic={int(torch_deterministic)})"
            )
            print(f"[WARN] official IsaacLab configure_seed unavailable at this stage: {exc}")
            return seed_i
        except Exception as fallback_exc:
            print(f"[WARN] failed to apply any global seed {seed}: {fallback_exc}")
            return None


def _apply_renderer_settings(args):
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        renderer = args.pipeline_renderer or "PathTracing"
        lights_disabled_probe = bool(getattr(args, "no_dome_bg", False) and getattr(args, "no_extra_lights", False))
        pt_spp = args.pt_spp_per_frame if args.pt_spp_per_frame is not None else 64
        pt_total = args.pt_total_spp if args.pt_total_spp is not None else 256
        pt_bounces = args.pt_max_bounces if args.pt_max_bounces is not None else 8
        pt_denoise = args.pt_denoise if args.pt_denoise is not None else True
        settings.set("/rtx/rendermode", renderer)
        settings.set("/rtx/render/renderer", renderer)
        settings.set("/rtx/sceneDb/ambientLightIntensity", 0.0)
        if lights_disabled_probe:
            settings.set("/rtx/directLighting/enabled", False)
            settings.set("/rtx/indirectDiffuse/enabled", False)
            settings.set("/rtx/shadows/enabled", False)
            settings.set("/rtx/ambientOcclusion/enabled", False)
        if renderer == "PathTracing":
            if pt_denoise and not os.path.exists("/usr/share/nvidia/nvoptix.bin"):
                print("[WARN] nvoptix.bin not found, disabling denoiser.")
                pt_denoise = False
            settings.set("/rtx/pathtracing/spp", int(pt_spp))
            settings.set("/rtx/pathtracing/totalSpp", int(pt_total))
            settings.set("/rtx/pathtracing/maxBounces", int(pt_bounces))
            settings.set("/rtx/pathtracing/optixDenoiser/enabled", bool(pt_denoise))
            settings.set("/rtx/denoiser/enabled", bool(pt_denoise))
            settings.set("/rtx/denoiser/useDenoiser", bool(pt_denoise))
        rt_subframes = getattr(args, "rt_subframes", None)
        if rt_subframes is not None:
            settings.set("/omni/replicator/RTSubframes", int(rt_subframes))
        if getattr(args, "dlss_mode", None) is not None:
            settings.set("rtx/post/dlss/execMode", int(args.dlss_mode))
        cur = settings.get("/rtx/rendermode")
        ambient = settings.get("/rtx/sceneDb/ambientLightIntensity")
        cur_rt_subframes = settings.get("/omni/replicator/RTSubframes")
        print(
            f"[RENDER] rendermode set -> {cur} | spp/frame={pt_spp} | total={pt_total} "
            f"| ambient={ambient} | no_light_probe={int(lights_disabled_probe)} "
            f"| rt_subframes={cur_rt_subframes}"
        )
    except Exception as exc:
        print(f"[WARN] renderer settings failed: {exc}")


def _prepare_env_cfg(args):
    args.enable_cameras = True

    cfg_path = Path(args.config).resolve()
    seed, torch_deterministic = _peek_seed_settings(cfg_path)
    _apply_global_seed(seed, torch_deterministic=torch_deterministic)

    if os.path.exists("/usr/share/nvidia/nvoptix.bin"):
        os.environ["DISABLE_OPTIX_DENOISER"] = "0"
    else:
        os.environ.setdefault("DISABLE_OPTIX_DENOISER", "1")

    app = AppLauncher(args).app

    import omni.ext
    import isaaclab.sim as sim_utils
    import unfold  # noqa: F401
    from unfold.platform.config_utils import parse_yaml_config
    from unfold.simulation.env import EnvCfg

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.core.experimental.materials", True)

    env_cfg: EnvCfg = parse_yaml_config(
        cfg_path,
        device=(args.device if hasattr(args, "device") and args.device else "cuda:0"),
        env_cfg_class=EnvCfg,
    )
    _apply_global_seed(getattr(env_cfg, "seed", None), torch_deterministic=torch_deterministic)

    def pick(name, default):
        cli = getattr(args, name, None)
        if cli is not None:
            return cli
        cfg_val = getattr(env_cfg, name, None)
        return cfg_val if cfg_val is not None else default

    args.pipeline_renderer = (
        getattr(args, 'pipeline_renderer', None)
        if getattr(args, 'pipeline_renderer', None) is not None
        else getattr(env_cfg, 'renderer', 'PathTracing')
    )
    args.pt_spp_per_frame = pick('pt_spp_per_frame', 64)
    args.pt_total_spp = pick('pt_total_spp', 256)
    args.pt_max_bounces = pick('pt_max_bounces', 8)
    args.pt_denoise = pick('pt_denoise', False)
    args.render_mode = pick('render_mode', 'quality')
    args.aa = pick('aa', 'DLSS')
    args.dlss_mode = pick('dlss_mode', 2)
    args.spp = pick('spp', 64)
    args.denoise = pick('denoise', True)
    args.cloth_root = str(getattr(args, 'cloth_root', None) or pick('cloth_root', '/World/Cloth'))
    args.patch_size_m = float(
        getattr(args, 'patch_size_m', None)
        if getattr(args, 'patch_size_m', None) is not None
        else (
            getattr(args, 'fabric_physical_size', None)
            if getattr(args, 'fabric_physical_size', None) is not None
            else pick('patch_size_m', 0.2)
        )
    )
    args.samples_per_asset = int(pick('samples_per_asset', 3))

    env_cfg.sim.render = sim_utils.RenderCfg(
        rendering_mode=args.render_mode,
        antialiasing_mode=args.aa,
        dlss_mode=args.dlss_mode,
        enable_dl_denoiser=args.denoise,
        samples_per_pixel=int(args.spp),
        enable_reflections=True,
        enable_global_illumination=True,
    )
    env_cfg.renderer = args.pipeline_renderer
    env_cfg.pt_spp_per_frame = args.pt_spp_per_frame
    env_cfg.pt_total_spp = args.pt_total_spp
    env_cfg.pt_max_bounces = args.pt_max_bounces
    env_cfg.pt_denoise = args.pt_denoise
    env_cfg.cloth_root = args.cloth_root
    env_cfg.patch_size_m = args.patch_size_m
    args.rt_subframes = getattr(env_cfg, "replicator", {}).get("rt_subframes", None)
    # Rendering collection switches assets explicitly after enough saved samples.
    env_cfg.episodes_per_asset_batch = int(1_000_000_000)

    if args.num_envs is not None:
        env_cfg.scene.num_envs = int(args.num_envs)
        env_cfg.num_envs = int(args.num_envs)

    env_cfg.capture_rounds_per_step = (
        args.capture_rounds_per_step
        if getattr(args, 'capture_rounds_per_step', None) is not None
        else getattr(env_cfg, 'capture_rounds_per_step', 1)
    )

    return app, env_cfg


def run(args) -> None:
    app, env_cfg = _prepare_env_cfg(args)
    _apply_renderer_settings(args)

    import gymnasium as gym
    import omni.replicator.core as rep
    import unfold  # noqa: F401

    env = gym.make(args.task, cfg=env_cfg)
    if getattr(env_cfg, "seed", None) is not None:
        rep.set_global_seed(int(env_cfg.seed))
    asset_pool = getattr(env.unwrapped, "_asset_pool", None)
    if asset_pool is not None and hasattr(asset_pool, "make_batches"):
        asset_pool.make_batches(int(env_cfg.scene.num_envs), shuffle=False)
        print("[ASSET_POOL] Rendering mode uses valid_assets.json order (shuffle=0)")

    initial_epoch_info = {
        "epoch": 1,
        "total_epochs": int(getattr(args, "epochs", 1)),
        "batch": 1,
        "total_batches": getattr(getattr(env.unwrapped, "_asset_pool", None), "num_batches", "?"),
    }
    env.unwrapped.reset(seed=getattr(env_cfg, "seed", None), options={"switch_asset": True, "epoch_info": initial_epoch_info})

    cameras, render_products, annotators = _setup_replicator(env, env_cfg, args)
    cloth_material_switcher = None
    if not args.no_fabric_texture:
        fabric_root = PROJECT_ROOT / 'data' / 'assets' / 'material' / 'Fabric'
        catalog = FabricTextureCatalog(fabric_root)
        scale_jitter = getattr(env_cfg, 'texture_scale_jitter', [0.85, 1.15])
        rotate_choices = getattr(env_cfg, 'texture_rotate_choices', [0, 90, 180, 270])
        translate_range = getattr(env_cfg, 'texture_translate_range', [0.0, 1.0])
        if catalog.texture_sets:
            cloth_material_switcher = ClothMaterialSwitcher(
                env=env,
                texture_catalog=catalog,
                probability=args.fabric_texture_prob,
                cloth_root=args.cloth_root,
                patch_size_m=args.patch_size_m,
                texture_scale_jitter=scale_jitter,
                texture_rotate_choices=rotate_choices,
                texture_translate_range=translate_range,
            )
        else:
            print('[FABRIC_SWITCH] No external material available; skipping external texture randomization.')

    try:
        _run_epochs(env, env_cfg, args, cameras, render_products, annotators, cloth_material_switcher)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f'[ERROR] caught exception: {exc}')
    finally:
        print('[SDG] Closing environment and application...')
        env.close()
        app.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
