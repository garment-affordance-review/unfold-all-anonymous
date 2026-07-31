# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from pathlib import Path
import yaml

from .assets import resolve_assets_root


def parse_yaml_config(yaml_path: str | Path, device: str = "cuda:0", env_cfg_class=None):
    """Parse YAML configuration file and convert to EnvCfg instance.
    
    Args:
        yaml_path: Path to the YAML configuration file.
        device: The device to run the simulation on. Defaults to "cuda:0".
        env_cfg_class: The configuration class to instantiate. If None, will try to import EnvCfg from env.env.
    
    Returns:
        The parsed configuration object.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    # Load YAML file
    with open(yaml_path, encoding="utf-8") as f:
        cfg_dict = yaml.full_load(f)
    
    # Extract the assets_root config when present for later resolution.
    # assets_root is not passed to EnvCfg; it is only used to resolve the actual ASSETS_ROOT path.
    assets_root_config = cfg_dict.pop("assets_root", None)
    

    
    # Auto-import EnvCfg if not provided (lazy import to avoid circular dependency)
    if env_cfg_class is None:
        from unfold.simulation.env import EnvCfg
        env_cfg_class = EnvCfg
    
    cfg = env_cfg_class()
    
    # Split config into known fields (for strict validation) and dynamic fields (for injection)
    # We check both the instance attributes and the class annotations to determine what is "known".
    known_keys = set(dir(cfg)) | set(getattr(cfg, "__annotations__", {}).keys())
    
    strict_dict = {}
    dynamic_dict = {}
    
    for key, value in cfg_dict.items():
        if key in known_keys:
            strict_dict[key] = value
        else:
            dynamic_dict[key] = value

    # Update known fields using the strict config-class method (handles nested configs like sim/scene)
    if strict_dict:
        cfg.from_dict(strict_dict)
    
    # Inject dynamic fields directly
    for key, value in dynamic_dict.items():
        setattr(cfg, key, value)
    
    # Set device
    cfg.sim.device = device
    
    # Resolve assets_root and store it on the config object for later ASSETS_ROOT resolution.
    # Compute the project root.
    # yaml_path: .../source/unfold_all/configs/config.yaml
    # config_dir: .../source/unfold_all/configs
    config_dir = yaml_path.parent
    # config.yaml is in .../configs/config.yaml
    # config_dir = .../configs
    # config_dir.parent = project_root
    project_root = config_dir.parent
    
    # Resolve the assets_root path. assets_root_config may be None, meaning use the default lookup logic.
    resolved_assets_root = resolve_assets_root(project_root, assets_root_config)
    # Store the resolved path on the config object.
    cfg._resolved_assets_root = resolved_assets_root
    
    return cfg
