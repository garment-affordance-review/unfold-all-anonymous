# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from typing import Optional, Union, Tuple

import numpy as np
import torch

from pxr import UsdGeom
from isaaclab.sim.utils import stage as stage_utils
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

class Action:
    """Action Manager (Mid-Level): Manages Vertex movements via Data-Oriented Programming (Tensor-based)."""
    
    # Maximum number of vertices we allow to be moved simultaneously per environment
    MAX_MOVING_VERTS = 4

    def __init__(self, cfg, garment_asset, num_envs, device, env_origins: Optional[torch.Tensor] = None, physics_dt: float = 1.0/60.0):
        """Initialize the vertex control manager (DOP version).
        
        Args:
            cfg: Environment configuration object.
            garment_asset: Cloth asset object.
            num_envs: Number of environments.
            device: Device (cuda/cpu).
            env_origins: Environment origins in world coordinates, shape (num_envs, 3).
        """
        self.cfg = cfg
        self.garment_asset = garment_asset
        self.num_envs = num_envs
        self.device = device
        self._env_origins: Optional[torch.Tensor] = env_origins.to(device) if env_origins is not None else None
        self.physics_dt = self.cfg.sim.dt

        # --- Vectorized State Tensors ---
        # Active mask: [num_envs] - Bool, is this env currently executing a move?
        self.move_active_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # Step counters: [num_envs] - Int, current step of the movement
        self.move_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # Duration: [num_envs] - Int, total steps for the movement
        self.move_duration = torch.ones(num_envs, dtype=torch.long, device=device) # Default 1 to avoid div/0
        
        # Interpolation Mode: 'linear' (default)
        # Could be extended to tensors if mixed modes needed, but usually global config.

        # Movement Data: [num_envs, MAX_MOVING_VERTS, 3]
        # We assume we move a limited number of vertices per env (e.g. 2 for grasp).
        self.move_vids = torch.full((num_envs, self.MAX_MOVING_VERTS), -1, dtype=torch.long, device=device)
        self.move_start_pos = torch.zeros((num_envs, self.MAX_MOVING_VERTS, 3), dtype=torch.float32, device=device)
        self.move_target_pos = torch.zeros((num_envs, self.MAX_MOVING_VERTS, 3), dtype=torch.float32, device=device)
        
        # Binding State (Mass modification)
        # We simply track which vertices are bound to know what to unbind.
        # [num_envs, MAX_MOVING_VERTS]
        self.bound_vids = torch.full((num_envs, self.MAX_MOVING_VERTS), -1, dtype=torch.long, device=device)
        
        # Store original masses to restore later: {(env_id, vid): mass}
        # Since this is sparse read/write, a dict is okay, or we could use a sparse tensor if strictly GPU.
        # Stick to dict for mass restoration logic for now as binding/unbinding is rare (once per phase).
        self._original_masses: dict[tuple[int, int], float] = {}

        # Global Phase
        self._sequence_phase = 'idle' 
        
        # Action Parameters (Per Env) - Tensorized
        # grasp_vertices: (num_envs, 2) - vertex IDs for v1, v2
        self.grasp_vertex_ids = torch.full((num_envs, 2), -1, dtype=torch.long, device=device)
        # grasp_distances: (num_envs,) - distance between grasp points
        self.grasp_distances = torch.zeros(num_envs, dtype=torch.float32, device=device)
        # grasp_active: (num_envs,) - whether this env has an active grasp
        self.grasp_active_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # Stretch State - Tensorized
        # stretch_max_distance: (num_envs,) - max stretch distance per env
        self.stretch_max_distances = torch.full((num_envs,), 0.7, dtype=torch.float32, device=device)
        # stretch_baseline_lengths: dict remains for now (stores pairs and rest_lengths tensors)
        self.stretch_baseline_lengths: dict[int, dict[str, torch.Tensor]] = {} 
        
        # Caches
        self._mesh_faces: dict[int, torch.Tensor] = {}
        self._last_stretch_debug_report: list[dict] = []


    @property
    def env_origins(self):
        if self._env_origins is None:
             # Fallback or lazy load if needed
             return torch.zeros((self.num_envs, 3), device=self.device)
        return self._env_origins

    # ========== Core: Vectorized Trajectory Update ==========

    def update_trajectory(self, env_ids: Union[list[int], torch.Tensor],
                         vertex_ids_list: list[list[int]], 
                         target_positions_list: list[list[torch.Tensor]],
                         duration_steps: Union[int, list[int]]) -> None:
        """
        Update trajectory commands for a batch of environments.
        Instead of calling 'move' on objects, we update the big tensors.
        
        Args:
            env_ids: list of env indices to update
            vertex_ids_list: list of [v1, v2...] for each env
            target_positions_list: list of [pos1, pos2...] for each env
            duration_steps: int (same for all) or list of ints
        """
        if len(env_ids) == 0:
            return

        # Prepare data for GPU
        env_indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        
        # Reset state for these envs
        self.move_active_mask[env_indices] = True
        self.move_step[env_indices] = 0
        
        # durations
        if isinstance(duration_steps, int):
            self.move_duration[env_indices] = duration_steps
        else:
            self.move_duration[env_indices] = torch.tensor(duration_steps, device=self.device, dtype=torch.long)

        # Retrieve current positions for start
        # Use batch retrieval if possible, but since we have ragged lists primarily from high level, we loop for setup.
        # Optimization: high level could pass packed tensors. For now, bridge the API.
        
        all_positions = self.garment_asset._get_particle_positions() # (N, V, 3) or (N*V, 3)
        # Ensure (N, V, 3)

        # We'll fill the tensors.
        # Reset current targets/starts to zero or sentinels
        self.move_vids[env_indices] = -1
        
        # This loop is Python-side but runs ONCE per Action Phase initiation, not per step.
        for i, env_id in enumerate(env_ids):
             vids = vertex_ids_list[i]
             targs = target_positions_list[i]
             
             if len(vids) > self.MAX_MOVING_VERTS:
                 print(f"[Action] Warning: Truncating movement vertices to {self.MAX_MOVING_VERTS}", file=sys.stderr)
                 vids = vids[:self.MAX_MOVING_VERTS]
                 targs = targs[:self.MAX_MOVING_VERTS]
             
             count = len(vids)
             if count == 0: continue
             
             # Store VIDs
             self.move_vids[env_id, :count] = torch.tensor(vids, device=self.device)
             
             # Get Start Pos
             # Direct access to global buffer is fastest
             # Need offset if flattened.
             # Assuming `garment_asset._get_particle_positions()` returns (NumEnvs, NumParticles, 3) for now
             # based on `vertex.py` logic which did `all_positions[self.env_id, vertex_id, :]`.
             start_p = all_positions[env_id, vids, :].clone()
             
             self.move_start_pos[env_id, :count] = start_p
             
             # Target Pos
             # targs is list of tensors. Stack them.
             if isinstance(targs[0], torch.Tensor):
                 t_stack = torch.stack(targs).to(self.device)
             else:
                 t_stack = torch.tensor(targs, device=self.device)
                 
             self.move_target_pos[env_id, :count] = t_stack
             

    # ========== Core: Vectorized Step Application ==========

    def apply_control(self):
        """
        Apply control logic for ALL environments in ONE go.
        Replaces `controller.update()` loop.
        """
        if self._sequence_phase not in ['lifting', 'stretching', 'holding']:
            return

        # 1. Update Step Counts
        # Only for active
        active_indices = self.move_active_mask.nonzero().squeeze(-1)
        if active_indices.numel() == 0:
            return

        # Increment step
        self.move_step[active_indices] += 1
        
        # Compute Progress: clamp(step / duration, 0, 1)
        progress = self.move_step[active_indices].float() / self.move_duration[active_indices].float()
        progress = torch.clamp(progress, 0.0, 1.0)
        
        # 2. Compute New Positions (Lerp)
        # shape: (N_active, MAX_MV, 3)
        # expand progress to (N_active, 1, 1)
        alpha = progress.view(-1, 1, 1)
        
        starts = self.move_start_pos[active_indices]
        targets = self.move_target_pos[active_indices]
        
        new_positions = starts + (targets - starts) * alpha
        
        # 3. Apply to Global Physics State
        all_positions = self.garment_asset._get_particle_positions()
        all_velocities = self.garment_asset._get_particle_velocities()
        
        # We need to scatter `new_positions` into `all_positions`.
        # Indices: move_vids[active_indices] -> (N_active, MAX_MV)
        
        # Flat indices for scatter:
        # Global_ID = Env_ID * NumParticlesPerEnv + Local_ID
        # Caution: Requires uniform particle count or offset tensor.
        # Assuming uniform for vectorized speedup. If not, we need `env_offsets`.
        
        # Let's check if we can write per-env safely.
        # Since we want to avoid for-loops inside apply_control, we assume we can write batched.
        
        # Simplest PyTorch Scatter:
        # Create coordinates (Batch, Index)
        bs = active_indices.shape[0]
        
        # Fetch VIDs: (N_active, MAX_MV)
        vids = self.move_vids[active_indices]
        
        # Mask out -1 (unused slots)
        valid_mask = (vids != -1) # (N_active, MAX_MV)
        
        if valid_mask.any():
            # Linearize indices
            # Global Index calculation
            # If garment_asset has uniform particles:
            # num_particles = self.garment_asset._num_particles_per_env
            # If not uniform, this breaks. Assuming uniform based on typical IsaacLab cloth usage.
            
            # flattened_env_ids = active_indices.view(-1, 1).expand(-1, self.MAX_MOVING_VERTS)
            # global_indices = flattened_env_ids * num_particles + vids
            
            # Actually, `all_positions` is usually (NumEnvs, NumParticles, 3).
            # So we can use scatter on first 2 dims.
            # But scatter usually works on flattened or same-dim.
            
            # Let's use advanced indexing.
            # active_indices (N,) -> match to vids
            
            # Broadcast env indices
            env_idx_expanded = active_indices.view(-1, 1).expand(-1, self.MAX_MOVING_VERTS) # (N_active, M)
            
            # Select valid ones
            valid_env_idx = env_idx_expanded[valid_mask] # (TotalSteps,)
            valid_vert_idx = vids[valid_mask]            # (TotalSteps,)
            
            valid_new_pos = new_positions[valid_mask]    # (TotalSteps, 3)
            
            # Update Position
            all_positions[valid_env_idx, valid_vert_idx, :] = valid_new_pos
            
            # Update Velocity (Zero out)
            if all_velocities is not None:
                all_velocities[valid_env_idx, valid_vert_idx, :] = 0.0
            
            # We updated `all_positions` locally.
            self.garment_asset._set_particle_positions(all_positions, active_indices)
            if all_velocities is not None:
                self.garment_asset._set_particle_velocities(all_velocities, active_indices)

    def _update_stretch_adaptive(self, all_pos: torch.Tensor):
        """Check strain and update targets incrementally.
        
        Args:
            all_pos: Pre-fetched particle positions (num_envs, max_particles, 3)
        """
        strain_threshold = self.cfg.stretch_strain_threshold
        stretch_velocity = self.cfg.stretch_motion_velocity 
        increment = stretch_velocity * self.physics_dt
        
        envs_to_update = []
        new_targets = []
        new_vids = []
        new_durations = []
        
        # Iterate over active grasps using tensor mask
        active_envs = self.grasp_active_mask.nonzero().squeeze(-1)
        
        for env_id in active_envs.cpu().tolist():
            if self._sequence_phase != 'stretching': continue
            
            # Strain Check
            baseline = self.stretch_baseline_lengths.get(env_id)
            if not baseline: continue
            
            # Get vertex IDs from tensor
            v1 = self.grasp_vertex_ids[env_id, 0].item()
            v2 = self.grasp_vertex_ids[env_id, 1].item()
            
            # Compute Strain (Vectorize this if bottleneck)
            pairs = baseline["pairs"]
            rest_lens = baseline["rest_lengths"]
            if pairs.numel() == 0: continue
            env_pos = all_pos[env_id]
            
            current_pairs = env_pos[pairs]
            current_lens = torch.norm(current_pairs[:,0] - current_pairs[:,1], dim=1)
            strains = current_lens / rest_lens
            max_strain = strains.max().item()
            
            if max_strain > strain_threshold:
                # Too tight, stop moving (hold current)
                continue
                
            # Else, extend valid
            current_dist = self.grasp_distances[env_id].item()
            max_dist = self.stretch_max_distances[env_id].item()
            
            if current_dist >= max_dist:
                continue
                
            self.grasp_distances[env_id] = current_dist + 2 * increment
            
            # Calc new target positions
            p1 = env_pos[v1]
            p2 = env_pos[v2]
            center = (p1 + p2) / 2.0
            env_origin = self.env_origins[env_id]
            
            mid_z = center[2]
            
            t1 = p1 + torch.tensor([-increment, 0, 0], device=self.device)
            t2 = p2 + torch.tensor([+increment, 0, 0], device=self.device)
            
            # We issue a "short move" command (5 steps)
            envs_to_update.append(env_id)
            new_vids.append([v1, v2])
            new_targets.append([t1, t2])
            new_durations.append(5)

        if envs_to_update:
            self.update_trajectory(envs_to_update, new_vids, new_targets, new_durations)

    def update_stretch_targets(self) -> bool:
        """Update stretch targets based on current strain.
        
        Should be called AFTER sim.step() to check strain correctly.
        
        Returns:
            bool: True if stretching is done
        """
        if self._sequence_phase != 'stretching':
            return True
        
        # Get positions AFTER simulation
        all_pos = self.garment_asset._get_particle_positions()
        
        # Check if done first
        is_done = self._is_stretching_done(all_pos)
        if is_done:
            return True
        
        # Update targets for next step
        self._update_stretch_adaptive(all_pos)
        return False

    def _is_stretching_done(self, all_pos: torch.Tensor) -> bool:
        """Check if stretching is complete.
        
        Args:
            all_pos: Particle positions AFTER simulation
            
        Returns:
            bool: True when all stretching envs are either tight (strain) or at max distance
        """
        strain_threshold = self.cfg.stretch_strain_threshold
        eps = 1e-4
        active_envs = self.grasp_active_mask.nonzero().squeeze(-1)
        if active_envs.numel() == 0:
            self._last_stretch_debug_report = []
            return True

        report: list[dict] = []
        for env_id in active_envs.cpu().tolist():
            max_dist = self.stretch_max_distances[env_id].item()
            current_dist = self.grasp_distances[env_id].item()
            env_report = {
                "env_id": int(env_id),
                "current_dist": float(current_dist),
                "max_dist": float(max_dist),
            }

            # If we still have room to extend, check strain
            if current_dist + eps < max_dist:
                baseline = self.stretch_baseline_lengths.get(env_id)
                # If no baseline data (empty edge_set), treat as done for this env
                if not baseline or baseline["pairs"].numel() == 0:
                    env_report["status"] = "no_baseline"
                    report.append(env_report)
                    continue  # Skip this env, consider it done
                env_pos = all_pos[env_id]
                pairs = baseline["pairs"]
                rest_lens = baseline["rest_lengths"]
                current_pairs = env_pos[pairs]
                current_lens = torch.norm(current_pairs[:, 0] - current_pairs[:, 1], dim=1)
                strains = current_lens / rest_lens
                max_strain = strains.max().item()
                env_report["max_strain"] = float(max_strain)
                if max_strain <= strain_threshold:
                    env_report["status"] = "needs_stretch"
                    report.append(env_report)
                    self._last_stretch_debug_report = report
                    return False  # This env still needs stretching
                env_report["status"] = "strain_limited"
                report.append(env_report)
                continue

            env_report["status"] = "max_distance_reached"
            report.append(env_report)

        self._last_stretch_debug_report = report
        return True

    # ========== High Level Actions ==========

    def start_grasp(self, env_id: int, vertex_id_1: int, vertex_id_2: int) -> None:
        """Bind and record grasp."""
        self._bind_vertices(env_id, [vertex_id_1, vertex_id_2])
        
        # Store in tensors
        self.grasp_vertex_ids[env_id, 0] = vertex_id_1
        self.grasp_vertex_ids[env_id, 1] = vertex_id_2
        self.grasp_active_mask[env_id] = True
        
        # Initial distance record
        all_pos = self.garment_asset._get_particle_positions()
        p1 = all_pos[env_id, vertex_id_1]
        p2 = all_pos[env_id, vertex_id_2]
        dist = torch.norm(p1[:2] - p2[:2]).item()
        self.grasp_distances[env_id] = dist
        self._sequence_phase = 'grasping'

    def lift_two_vertices_batch(self, env_ids: list[int], 
                               vertex_pairs: list[tuple[int, int]],
                               height_offset: Union[float, list[float]] = 1.0,
                               duration_steps: int = 120,
                               horizontal_distances: Optional[list[float]] = None):
        """Vectorized lift operation for multiple environments.
        
        Fetches particle positions ONCE for all environments, eliminating redundant GPU-CPU transfers.
        
        Args:
            env_ids: List of environment IDs to process
            vertex_pairs: List of (v1, v2) tuples, one per env_id
            height_offset: Height to lift vertices above current position
            duration_steps: Duration in simulation steps
            horizontal_distances: Optional list of horizontal distances, one per env_id
        """
        if not env_ids:
            return
            
        # Fetch positions ONCE for all environments
        all_pos = self.garment_asset._get_particle_positions()
        
        # Prepare batch data structures
        vids_list = []
        targets_list = []
        
        # Process all environments
        for i, env_id in enumerate(env_ids):
            v1, v2 = vertex_pairs[i]
            
            # Ensure grasp is active
            if not self.grasp_active_mask[env_id]:
                self.start_grasp(env_id, v1, v2)
            
            # Get positions for this env
            p1 = all_pos[env_id, v1]
            p2 = all_pos[env_id, v2]
            
            # Compute target z
            base_z = max(float(p1[2].item()), float(p2[2].item()))
            env_height_offset = height_offset[i] if isinstance(height_offset, list) else height_offset
            target_z = base_z + float(env_height_offset)
            
            # Pure vertical lift: preserve current x/y before later symmetric stretch positioning.
            t1 = p1.clone()
            t2 = p2.clone()
            t1[2] = target_z
            t2[2] = target_z
            
            vids_list.append([v1, v2])
            targets_list.append([t1, t2])
        
        # Single batched trajectory update
        self.update_trajectory(env_ids, vids_list, targets_list, duration_steps)
        self._sequence_phase = 'lifting'

    def stretch_two_vertices_batch(self, env_ids: list[int], 
                                   initial_distances: list[float],
                                   max_distances: Optional[list[float]] = None):
        """Vectorized stretch setup for multiple environments.
        
        Fetches particle positions ONCE for all environments, eliminating redundant GPU-CPU transfers.
        
        Args:
            env_ids: List of environment IDs to process
            initial_distances: List of initial distances, one per env_id
            max_distances: Optional list of max stretch distances, one per env_id
        """
        if not env_ids:
            return
        
        # Fetch positions ONCE for all environments
        all_pos = self.garment_asset._get_particle_positions()
        
        # Process all environments
        for i, env_id in enumerate(env_ids):
            # Setup stretch state
            self.stretch_max_distances[env_id] = max_distances[i]
            
            # Cache topology for strain
            v1 = self.grasp_vertex_ids[env_id, 0].item()
            v2 = self.grasp_vertex_ids[env_id, 1].item()
            
            # Optimized neighbor search
            neighbors_map = self.get_neighbors_for_vertices(env_id, [v1, v2])
            
            edge_set = set()
            for c in [v1, v2]:
                for n in neighbors_map.get(c, []):
                    edge_set.add(tuple(sorted((c, n))))
            
            if edge_set:
                pairs = torch.tensor(sorted(list(edge_set)), device=self.device, dtype=torch.long)
                # Use pre-fetched positions
                p = all_pos[env_id][pairs]
                rest_lens = torch.norm(p[:,0] - p[:,1], dim=1).clamp_min(1e-6)
                self.stretch_baseline_lengths[env_id] = {"pairs": pairs, "rest_lengths": rest_lens}
        
        self._sequence_phase = 'stretching'

    def lower_vertices_to_ground(self, ground_height_offset: float = 0.05, duration_steps: int = 30):
        # Batch collect requests
        env_ids = []
        vids_list = []
        targs_list = []
        
        all_pos = self.garment_asset._get_particle_positions()
        
        # Iterate over active grasps using tensor mask
        active_envs = self.grasp_active_mask.nonzero().squeeze(-1)
        
        for env_id in active_envs.cpu().tolist():
            v1 = self.grasp_vertex_ids[env_id, 0].item()
            v2 = self.grasp_vertex_ids[env_id, 1].item()
            
            p1 = all_pos[env_id, v1].clone()
            p2 = all_pos[env_id, v2].clone()
            
            env_origin = self.env_origins[env_id]
            target_z = env_origin[2] + ground_height_offset
            
            p1[2] = target_z
            p2[2] = target_z
            
            env_ids.append(env_id)
            vids_list.append([v1, v2])
            targs_list.append([p1, p2])
            
        self.update_trajectory(env_ids, vids_list, targs_list, duration_steps)
        self._sequence_phase = 'holding'

    def stop_all_control(self):
        # Unbind all active grasps
        active_envs = self.grasp_active_mask.nonzero().squeeze(-1)
        for env_id in active_envs.cpu().tolist():
            self._unbind_vertices(env_id)
            
        # Reset tensors
        self.grasp_vertex_ids.fill_(-1)
        self.grasp_distances.fill_(0.0)
        self.grasp_active_mask.fill_(False)
        self.move_active_mask.fill_(False)
        self._sequence_phase = 'releasing'

    # ========== Binding Logic ==========

    def _bind_vertices(self, env_id: int, vertex_ids: list[int]):
        """Set Kinematic Mass."""
        # Update bound state
        # In a real DOP system, we'd do this in batch too, but grasp is initiated sequentially usually.
        # We'll support single env call for now unless start_grasp is vectorized.
        
        all_masses = self.garment_asset._get_particle_masses()
        km_mult = self.cfg.kinematic_mass_multiplier
        
        for vid in vertex_ids:
            if (env_id, vid) not in self._original_masses:
                self._original_masses[(env_id, vid)] = float(all_masses[env_id, vid].item())
                
            original = self._original_masses[(env_id, vid)]
            new_mass = original * km_mult if original > 0 else 0.01
            all_masses[env_id, vid] = new_mass
            
            # Zero velocity
            # Handled in update loop usually, but good to reset on bind
            
        self.garment_asset._set_particle_masses(all_masses, torch.tensor([env_id], device=self.device))
        
        # update bound_vids tensor for tracking?
        # self.bound_vids[env_id, ...] = ...

    def _unbind_vertices(self, env_id: int):
        all_masses = self.garment_asset._get_particle_masses()
        
        # Restore
        restored = False
        keys_to_del = []
        for (e, v), m in self._original_masses.items():
            if e == env_id:
                all_masses[e, v] = m
                keys_to_del.append((e,v))
                restored = True
        
        for k in keys_to_del:
            del self._original_masses[k]
            
        if restored:
            self.garment_asset._set_particle_masses(all_masses, torch.tensor([env_id], device=self.device))


    # ========== Utils ==========
    
    def _get_vertex_world_positions(self, env_id: int, v1: int, v2: int):
        all_pos = self.garment_asset._get_particle_positions()
        return all_pos[env_id, v1], all_pos[env_id, v2]

    def _get_mesh_topology_data(self, env_id: int):
        stage = stage_utils.get_current_stage()
        mesh_path = f"/World/Cloth/env_{env_id}/garment/mesh"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        mesh = UsdGeom.Mesh(mesh_prim)
        return np.array(mesh.GetFaceVertexCountsAttr().Get()), np.array(mesh.GetFaceVertexIndicesAttr().Get())

    def get_mesh_faces(self, env_id: int) -> torch.Tensor:
        if env_id in self._mesh_faces: return self._mesh_faces[env_id]
        
        counts, indices = self._get_mesh_topology_data(env_id)
        stage = stage_utils.get_current_stage()
        mesh_path = f"/World/Cloth/env_{env_id}/garment/mesh"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        # Triangulate (Fan)
        triangles = []
        ptr = 0
        for c in counts:
            if c <= 0 or (ptr + c) > indices.shape[0]:
                ptr += max(int(c), 0)
                continue
            base = indices[ptr]
            for i in range(c - 2):
                triangles.append([base, indices[ptr+i+1], indices[ptr+i+2]])
            ptr += c
        
        # Build face candidates from mesh topology (legacy path).
        valid_limit = self.garment_asset._num_particles_per_env_dict[env_id]
        tris_mesh = np.asarray(triangles, dtype=np.int64).reshape(-1, 3) if len(triangles) > 0 else np.empty((0, 3), dtype=np.int64)

        # Prefer PhysX cooked welded topology when available: this is closest to solver-valid cloth faces.
        tris = tris_mesh
        if mesh_prim and mesh_prim.IsValid():
            try:
                wtri_attr = mesh_prim.GetAttribute("physxParticle:weldedTriangleIndices")
                wtri_val = wtri_attr.Get() if (wtri_attr and wtri_attr.IsValid()) else None
                if wtri_val is not None:
                    wtri_np = np.asarray(wtri_val, dtype=np.int64).reshape(-1)
                    if (wtri_np.size % 3) == 0 and wtri_np.size > 0:
                        wtri = wtri_np.reshape(-1, 3)
                        in_range = (wtri >= 0).all() and (wtri < int(valid_limit)).all()
                        if in_range:
                            tris = wtri
            except Exception:
                pass

        # Final range filter guard.
        mask = (tris < valid_limit).all(axis=1) & (tris >= 0).all(axis=1)
        t = torch.tensor(tris[mask], device=self.device, dtype=torch.long)
        self._mesh_faces[env_id] = t

        return t

    def get_neighbors_for_vertices(self, env_id: int, vertex_ids: list[int]) -> dict[int, list[int]]:
        """
        Efficiently find neighbors for specific vertices without building the full graph.
        """
        counts, indices = self._get_mesh_topology_data(env_id) # tensors or numpy arrays
        
        # Pre-calculate face boundaries if counts vary, otherwise assume uniform (e.g. triangles)
        # Using a cumulative sum to find face start/end indices
        face_ends = np.cumsum(counts)
        face_starts = np.empty_like(face_ends)
        face_starts[0] = 0
        face_starts[1:] = face_ends[:-1]

        neighbors_map = {}

        for vid in vertex_ids:
            # Vectorized search for all faces containing this vertex
            # np.where returns indices in the flattened 'indices' array
            locs = np.where(indices == vid)[0]
            
            # Find which face index each location belongs to
            face_indices = np.searchsorted(face_ends, locs, side='right')
            
            # Collect unique neighbors from these faces
            unique_neighbors = set()
            for f_idx in face_indices:
                start = face_starts[f_idx]
                end = face_ends[f_idx]
                face_verts = indices[start:end]
                for v in face_verts:
                    if v != vid and v < self.garment_asset._num_particles_per_env_dict[env_id]:
                        unique_neighbors.add(v)
            
            neighbors_map[vid] = list(unique_neighbors)
            
        return neighbors_map

    def check_all_envs_stable(self, all_pos: torch.Tensor) -> bool:
        """Check if all environments are stable (vectorized).
        
        Returns:
            bool: True if all environments have velocities below threshold
        """
        import torch
        
        # Get velocities ONCE for all environments
        velocities = self.garment_asset._get_particle_velocities()
        if velocities is None or velocities.numel() == 0:
            return False
        
        # Compute speeds for all environments (vectorized)
        # velocities shape: (num_envs, num_particles, 3)
        speeds = torch.linalg.vector_norm(velocities, dim=2)  # (num_envs, num_particles)
        
        # Compute P85 speed for each environment (vectorized)
        # Using quantile along particle dimension
        p85_speeds = torch.quantile(speeds, 0.85, dim=1)  # (num_envs,)
        
        # Check stability for all environments
        threshold = self.cfg.action_sequence["stabilizing"]["velocity_threshold"]
        stable_mask = p85_speeds < threshold  # (num_envs,)
        
        # Return True if all environments are stable
        return stable_mask.all().item()

    def reset(self, env_ids_list):
        for e in env_ids_list:
            self._unbind_vertices(e)

            # Clear tensor-based grasp state
            self.grasp_vertex_ids[e] = -1
            self.grasp_distances[e] = 0.0
            self.grasp_active_mask[e] = False

            # Clear stretch state
            if e in self.stretch_baseline_lengths: 
                del self.stretch_baseline_lengths[e]
            self.stretch_max_distances[e] = 0.7  # Reset to default
            
            self.move_active_mask[e] = False
            self.move_step[e] = 0

    @property
    def sequence_phase(self): return self._sequence_phase
    def set_sequence_phase(self, phase): self._sequence_phase = phase
