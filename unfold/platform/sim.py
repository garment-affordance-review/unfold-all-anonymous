# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


class _SceneShim:
    def __init__(self, stage):
        self.stage = stage

    def object_exists(self, _name: str) -> bool:
        return False

    def add(self, _obj) -> None:
        pass

    def __getattr__(self, name):
        """Pass through attribute access to the underlying world object"""
        if hasattr(self._world, name):
            return getattr(self._world, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def check_cloth_stability(
    garment_asset,
    env_id: int,
    velocity_threshold: float = 0.01,
    return_max_speed: bool = False
):
    """Check if cloth is stable (velocity below threshold).
    
    Args:
        garment_asset: Cloth asset object
        env_id: Environment ID
        velocity_threshold: Speed threshold in m/s
        return_max_speed: Whether to return the max speed value
        
    Returns:
        bool or (bool, float): True if stable, optional max speed
    """
    import torch
    
    try:
        velocities = garment_asset._get_particle_velocities()
        if velocities is None or velocities.numel() == 0:
            return (False, float('inf')) if return_max_speed else False
        
        env_velocities = garment_asset._get_env_particle_data(velocities, env_id)
            
        if env_velocities is None:
            return (False, float('inf')) if return_max_speed else False
        
        # Calc speed
        speed = torch.linalg.vector_norm(env_velocities, dim=1)
        # Use P85 instead of Max to be robust against outliers
        p85_speed = float(torch.quantile(speed, 0.85).item())
        is_stable = p85_speed < velocity_threshold
        # Return P85 speed as the metric
        return (is_stable, p85_speed) if return_max_speed else is_stable
    except Exception:
        return (False, float('inf')) if return_max_speed else False


class _WorldProxy:
    def __init__(self, scene, sim):
        self.scene = _SceneShim(scene.stage)
        self.stage = scene.stage
        self._physics_context = sim._physics_context

    def get_physics_context(self):
        return self._physics_context
