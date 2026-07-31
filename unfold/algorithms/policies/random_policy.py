
from __future__ import annotations
from typing import Optional, Dict

import torch
import torch.nn as nn

class RandomPolicy(nn.Module):
    """
    Policy that samples random vertex pairs.
    Used for offline data collection.
    """
    
    def __init__(self, manager, cfg: Dict, device: str = "cuda"):
        super().__init__()
        self.manager = manager
        self.cfg = cfg
        self.device = device
        
    def forward(
        self, 
        obs: Dict[str, torch.Tensor], 
    ) -> torch.Tensor:
        """
        Generate random actions using optimized scale-fair sampling.
        
        Args:
            obs: Observation dict containing 'pos' and 'faces'.
            
        Returns:
            actions: (num_envs, 2) tensor of vertex indices
        """
        from unfold.platform.perception import sample_visible_pair_fast
        
        pos_list = obs.get("pos", [])
        faces_list = obs.get("faces", [])
        pos_mask_list = obs.get("pos_mask", [])
        faces_mask_list = obs.get("faces_mask", [])
        
        num_envs = len(pos_list)
        results = torch.full((num_envs, 2), -1, dtype=torch.long, device=self.device)
        
        # Parameters for spatial sampling from Config
        min_grasp_dist = self.cfg.get('min_grasp_dist', 0.3)
        resolution = self.cfg.get('resolution', 128)
        
        for i in range(num_envs):
            pos = pos_list[i][pos_mask_list[i].squeeze()>0]
            faces = faces_list[i][faces_mask_list[i].squeeze()>0]

            if pos is None or faces is None or pos.shape[0] == 0:
                continue
                
            # Determine dynamic bounds
            p_min = pos.min(dim=0)[0]
            p_max = pos.max(dim=0)[0]
            max_extent = torch.linalg.norm((p_max - p_min)[:2])
            current_max_dist = max(min_grasp_dist + 0.1, max_extent.item())
            
            # Use optimized fast sampling
            pair = sample_visible_pair_fast(
                pos, faces,
                min_grasp_dist=min_grasp_dist,
                max_grasp_dist=current_max_dist,
                resolution=resolution,
                max_retries=20
            )
            
            if pair is not None:
                results[i] = pair
            else:
                # Fallback to naive random if no valid spatial pairs found
                num_particles = pos.shape[0]
                if num_particles > 1:
                    results[i] = torch.randperm(num_particles, device=self.device)[:2]
                    
        return results

