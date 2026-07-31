"""Environment loading helpers for offline collection pipeline."""

from __future__ import annotations

import math
from pathlib import Path

from .common import PROJECT_ROOT


def load_env_and_cfg(args):
    import gymnasium as gym
    from unfold.simulation.env import EnvCfg
    from unfold.platform.config_utils import parse_yaml_config

    if args.task != "UnfoldAll-Cloth-Direct-v0":
        raise ValueError(f"Unknown task: {args.task}")

    yaml_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    print(f"[INFO] Loading configuration from: {yaml_path}")
    env_cfg = parse_yaml_config(yaml_path, device=args.device if args.device else "cuda:0", env_cfg_class=EnvCfg)
    if getattr(args, "assets_manifest", None):
        env_cfg.assets_manifest = str(args.assets_manifest)

    if args.num_envs is not None:
        env_cfg.scene.num_envs = int(args.num_envs)
        env_cfg.num_envs = int(args.num_envs)
    if not isinstance(getattr(env_cfg, "ground_size_m", None), (list, tuple)) or len(getattr(env_cfg, "ground_size_m", [])) < 2:
        spacing = float(getattr(env_cfg.scene, "env_spacing", 2.0) or 2.0)
        grid_dim = max(1, math.ceil(math.sqrt(int(env_cfg.scene.num_envs))))
        default_size = max(4.0, (grid_dim + 1) * spacing)
        env_cfg.ground_size_m = [default_size, default_size]

    env = gym.make(args.task, cfg=env_cfg)
    return env, env_cfg


def calc_step_targets(env_cfg, pool, epochs):
    episodes_per_batch = env_cfg.episodes_per_asset_batch
    steps_per_episode = env_cfg.steps_per_episode
    num_batches = pool.num_batches
    steps_per_epoch = num_batches * episodes_per_batch * steps_per_episode
    total_steps = epochs * steps_per_epoch
    return num_batches, steps_per_epoch, total_steps
