# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional, Sequence

# -------------------------------------------------------------------------- #
# Resolve Assets directory
# -------------------------------------------------------------------------- #
def resolve_assets_root(project_root: Path, assets_root_config: str | Path | None = None) -> Path:
    """Resolve the Assets directory path using only the config value.
    
    Args:
        project_root: Project root path.
        assets_root_config: assets_root path read from the config file. It may be relative or absolute.
        
    Returns:
        Absolute path to the Assets directory.
        
    Raises:
        RuntimeError: If the Assets directory cannot be found.
    """
    _ASSET_CANDIDATES: list[Optional[Path]] = []
    
    # Priority 1: path from the config file when provided.
    if assets_root_config is not None:
        config_path = Path(assets_root_config).expanduser()
        if config_path.is_absolute():
            _ASSET_CANDIDATES.append(config_path)
        else:
            # Relative paths are resolved against the project root.
            _ASSET_CANDIDATES.append(project_root / config_path)
    
    for candidate in _ASSET_CANDIDATES:
        if candidate is None:
            continue
        candidate_resolved = candidate.resolve()
        if candidate_resolved.exists():
            return candidate_resolved

    raise RuntimeError(
        "Unable to locate Assets directory. "
        f"Set assets_root in config.yaml or ensure Assets exists in one of: {[str(c) for c in _ASSET_CANDIDATES if c]}"
    )


def load_assets_from_json(json_path: Path) -> list[str]:
    """Load valid asset paths from a validation JSON file.

    Args:
        json_path: Path to the validation JSON file.

    Returns:
        List of relative asset paths.
    """
    if not json_path.exists():
        return []
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
        
    except Exception as e:
        print(f"[UNFOLD][ASSETS] Error loading asset validation JSON: {e}", file=sys.stdout)
        return []


def collect_assets_from_categories(categories: Sequence[str], assets_root: Path) -> list[str]:
    """Collect all *_obj.usd files from the specified asset category directories.
    
    Args:
        categories: Category names corresponding to subdirectories under Assets/cloth,
            such as ["Dress", "Tops", "Trousers", "Glove", "Hat"].
        assets_root: Assets root directory path.
        
    Returns:
        Discovered asset paths relative to assets_root.
    """
    if assets_root is None:
        return []
    
    garment_root = assets_root / "cloth"
    if not garment_root.exists():
        print(f"[UNFOLD][BUILD_POOL]   ASSETS_ROOT: {assets_root}", file=sys.stdout, flush=True)
        return []
    
    candidates: list[str] = []
    for category in categories:
        category_path = garment_root / category
        if not category_path.exists():
            continue
        
        # Recursively search all *_obj.usd files under this category directory.
        for dirpath, _, filenames in os.walk(category_path):
            for fname in filenames:
                lower = fname.lower()
                if not lower.endswith(".usd"):
                    continue
                # Skip USD files that are clearly not garment body meshes.
                if "border" in lower:
                    continue
                # Only collect mesh files like *_obj.usd.
                if "obj" not in lower:
                    continue
                # Convert to a path relative to assets_root.
                full_path = Path(dirpath) / fname
                try:
                    rel_path = full_path.relative_to(assets_root)
                    candidates.append(str(rel_path))
                except ValueError:
                    # Fall back to an absolute path if a relative path cannot be computed.
                    candidates.append(str(full_path))
    
    return candidates


def absolute_asset_path(path_str: str | None, assets_root: Path) -> str | None:
    """Convert an asset path string to an absolute path.
    
    Args:
        path_str: Asset path string. It may be relative or absolute.
        assets_root: Assets root directory path.
        
    Returns:
        Absolute path string, or None when the input is None.
        
    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if path_str.startswith(("omniverse://", "http://", "https://")):
        return path_str
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (assets_root / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"[UNFOLD] Asset not found: {path}")
    return str(path)


def load_material_to_stage(material_usd_path: str, stage) -> str | None:
    """Load a material USD file into the stage and return the material prim path.
    
    Args:
        material_usd_path: Filesystem path to the material USD file.
        stage: USD stage object.
        
    Returns:
        Material prim path in the stage, or None if loading fails.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage
    import isaacsim.core.utils.prims as prims_utils
    from isaacsim.core.utils.string import find_unique_string_name
    
    # Create a unique material prim path.
    material_prim_path = find_unique_string_name(
        "/World/Materials/" + Path(material_usd_path).stem,
        is_unique_fn=lambda x: not stage.GetPrimAtPath(x).IsValid()
    )
    
    # Load the material file into the stage.
    add_reference_to_stage(usd_path=material_usd_path, prim_path=material_prim_path)
    material_prim = prims_utils.get_prim_at_path(material_prim_path)
    
    # Find the material prim, usually in the referenced root prim or one of its children.
    material_children = prims_utils.get_prim_children(material_prim)
    if material_children:
        # Use the first child prim as the material prim.
        material_prim_path = material_children[0].GetPath()
    else:
        # Use the root prim itself when there are no child prims.
        material_prim_path = material_prim.GetPath()
    
    return str(material_prim_path)
