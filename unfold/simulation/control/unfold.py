import time
import torch
import carb
import isaaclab.sim as sim_utils
import sys
from dataclasses import dataclass, field
from .action import Action
from unfold.platform.profiling import profile

@dataclass
class StepStats:
    """Statistics for a single step, aggregated across all phases."""
    stretch_steps: int = 0
    stretch_time: float = 0.0
    stable_steps: int = 0
    stable_time: float = 0.0
    move_steps: int = 0
    move_time: float = 0.0
    
    def reset(self):
        self.stretch_steps = 0
        self.stretch_time = 0.0
        self.stable_steps = 0
        self.stable_time = 0.0
        self.move_steps = 0
        self.move_time = 0.0

class Unfold:
    """
    Manages the high-level unfold task sequence:
    Grasp -> Lift -> Stretch -> Hold -> Release -> Stabilize
    """
    def __init__(self, cfg, num_envs, device, garment_asset, physics_dt, env_origins=None):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        
        # Initialize Action Manager
        self.action_manager = Action(
            cfg=cfg,
            garment_asset=garment_asset,
            num_envs=num_envs,
            device=device,
            env_origins=env_origins,
            physics_dt=physics_dt
        )
        
        self.physics_dt = physics_dt
        
        # State tracking
        self.episode_length_buf = 0
        self.extras = {"invalid_reward": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}
        
        self._centers_buf = torch.zeros((self.num_envs, 3), device=self.device)
        self._grid_obs_buf = None
        self._action_pairs_buf = torch.full(
            (self.num_envs, 2), -1, device=self.device, dtype=torch.long
        )
        
        # Step statistics for logging
        self._step_stats = StepStats()
        self.init_mode = self._resolve_init_mode()
        self.loading_mode = self._resolve_loading_mode()
        self.frame_recorder = None
        self._last_stretch_debug_emitted = False

    def reset(self, env_ids_list: list[int]):
         """Reset controller state for the specified environments."""
         self.action_manager.reset(env_ids_list)
         
         for env_id in env_ids_list:
            # Ensure pairs buf is reset
            self._action_pairs_buf[env_id] = -1
            self.extras["invalid_reward"][env_id] = False

    def reset_buffers(self):
        """Resets buffers at start of episode."""
        self.extras["invalid_reward"].fill_(False)
        self._action_pairs_buf.fill_(-1)

    def step(self, actions: torch.Tensor, sim, scene) -> dict:
        """
        Executes one step of the task machine.
        Returns:
            dict with 'action_pairs' and updated 'extras'.
        """
        # Reset stats at beginning of each step
        self._step_stats.reset()

        # Phase 1: Prepare Actions (Grasp)
        with profile("unfold.prepare_actions"):
            self.action_manager.set_sequence_phase('lifting')
                
            for env_id in range(self.num_envs):
                # Select Vertex Pair - Actions are now always explicit indices (from policy or random)
                vertex_id_1, vertex_id_2, actual_distance = self._parse_action(env_id, actions)

                # Record Action
                self._action_pairs_buf[env_id, 0] = vertex_id_1
                self._action_pairs_buf[env_id, 1] = vertex_id_2

                # Store for Hold Phase
                self.action_manager.grasp_distances[env_id] = actual_distance
                
                # Start Grasp
                if not self.action_manager.grasp_active_mask[env_id]:
                    self.action_manager.start_grasp(env_id, vertex_id_1, vertex_id_2)
            
        # Phase 2: Execute Physics Steps (Lift, Stretch, Hold, Release, Stabilize)
        with profile("unfold.lifting_phase"):
            self._execute_lifting_phase(sim, scene)
        
        with profile("unfold.loading_phase"):
            self._execute_loading_phase(sim, scene)
        
        with profile("unfold.holding_phase"):
            self._execute_holding_phase(sim, scene)
        
        with profile("unfold.releasing_phase"):
            self._execute_releasing_phase(sim, scene)
        
        with profile("unfold.stabilizing_phase"):
            self._execute_stabilizing_phase(sim, scene)
         
        # Phase 3: Post-process
        self.action_manager.set_sequence_phase('idle')
        
        return {
            "action_pairs": self._action_pairs_buf,
            "extras": self.extras
        }
    
    @property
    def step_stats(self) -> StepStats:
        """Returns the statistics for the last step."""
        return self._step_stats
    
    def _parse_action(self, env_id, actions):
        vertex_id_1 = int(actions[env_id, 0].item())
        vertex_id_2 = int(actions[env_id, 1].item())
            
        positions = self.action_manager._get_vertex_world_positions(env_id, vertex_id_1, vertex_id_2)
        if positions is None:
             raise RuntimeError(f"Failed to get position for env {env_id}")
        pos1, pos2 = positions
        delta_xy = (pos1[:2] - pos2[:2]).to(device=self.device)
        actual_distance = float(torch.linalg.vector_norm(delta_xy).item())
        
        return vertex_id_1, vertex_id_2, actual_distance

    def _execute_lifting_phase(self, sim, scene):
        self._set_gravity((0.0, 0.0, -9.8))
        active_envs = self.action_manager.grasp_active_mask.nonzero().squeeze(-1).cpu().tolist()
        if not active_envs:
            return

        target_height = self._get_conditioned_target_height()
        pair_pose_velocity = self._get_pair_pose_velocity()

        if self.init_mode == "conditioned":
            self._move_grasps_to_symmetric_stretch_pose(
                sim=sim,
                scene=scene,
                env_ids=active_envs,
                velocity=pair_pose_velocity,
                target_height=target_height,
            )
            if self._debug_enabled("protocol_trace"):
                self._debug_log_env_state("conditioned.after_pose", active_envs, target_height=target_height)

        if self.loading_mode == "y_gravity":
            if self.init_mode != "conditioned":
                self._move_grasps_to_symmetric_stretch_pose(
                    sim=sim,
                    scene=scene,
                    env_ids=active_envs,
                    velocity=pair_pose_velocity,
                    target_height=target_height,
                )
                if self._debug_enabled("protocol_trace"):
                    self._debug_log_env_state("y_gravity.after_pose", active_envs, target_height=target_height)
            return

        self._adjust_grasp_height_closed_loop(sim, scene, env_ids=active_envs)

    def _execute_loading_phase(self, sim, scene):
        if self.loading_mode == "fling":
            self._execute_fling_loading_phase(sim, scene)
            return
        self._execute_y_gravity_loading_phase(sim, scene)

    def _execute_y_gravity_loading_phase(self, sim, scene):
        active_envs = self.action_manager.grasp_active_mask.nonzero().squeeze(-1).cpu().tolist()
        y_cfg = dict(self.cfg.action_sequence.get("y_gravity", {}) or {})
        if self._debug_enabled("protocol_trace"):
            self._debug_log_env_state("y_gravity.pre", active_envs)

        # 1. Switch Gravity
        self._set_gravity((0.0, 4.9, 0.0))
        self._run_physics_steps(sim, scene, self.cfg.action_sequence["holding"]["gravity_switch"]["duration_steps"], check_stability=True, phase_type="stable")
        pre_stretch_hold_steps = int(y_cfg.get("pre_stretch_hold_steps", 0))
        if self.init_mode == "conditioned":
            pre_stretch_hold_steps = int(
                y_cfg.get("conditioned_pre_stretch_hold_steps", pre_stretch_hold_steps)
            )
        if pre_stretch_hold_steps > 0:
            self._run_physics_steps(sim, scene, pre_stretch_hold_steps, check_stability=True, phase_type="stable")

        # Setup stretch for all active environments - Vectorized batch processing
        if active_envs:
            initial_dists = []
            max_dists = []
            stretch_max = self.cfg.stretch_max_distance
            
            for env_id in active_envs:
                initial_dists.append(self.action_manager.grasp_distances[env_id].item())
                max_dists.append(stretch_max)
            
            self.action_manager.stretch_two_vertices_batch(
                active_envs, initial_dists, max_distances=max_dists
            )
        
        self.action_manager.set_sequence_phase('stretching')
        
        # Calculate timeout from velocity and distance (each vertex moves half the total distance)
        stretch_duration_steps = self._calculate_steps_from_velocity(
            self.cfg.stretch_max_distance / 2,
            self.cfg.stretch_motion_velocity
        )
        # Use unified physics loop with stretch checking
        self._run_physics_steps(sim, scene, stretch_duration_steps, check_stretch=True, phase_type="stretch")

        post_stretch_hold_steps = int(y_cfg.get("post_stretch_hold_steps", 0))
        if post_stretch_hold_steps > 0:
            self._run_physics_steps(sim, scene, post_stretch_hold_steps, check_stability=True, phase_type="stable")

    def _execute_fling_loading_phase(self, sim, scene):
        self._set_gravity((0.0, 0.0, -9.8))

        active_envs = self.action_manager.grasp_active_mask.nonzero().squeeze(-1).cpu().tolist()
        if not active_envs:
            return

        fling_cfg = self._get_fling_cfg()
        if self._debug_enabled("protocol_trace"):
            self._debug_log_env_state("fling.pre_pose", active_envs)
        self._move_grasps_to_symmetric_stretch_pose(
            sim=sim,
            scene=scene,
            env_ids=active_envs,
            velocity=float(fling_cfg["stretch_pose_velocity"]),
            target_height=None,
        )
        if self._debug_enabled("protocol_trace"):
            self._debug_log_env_state("fling.after_pose", active_envs)

        pre_stretch_settle_steps = int(fling_cfg.get("pre_stretch_settle_steps", 0))
        if pre_stretch_settle_steps > 0:
            self._run_physics_steps(sim, scene, pre_stretch_settle_steps, check_stability=True, phase_type="stable")

        max_dists = [self.cfg.stretch_max_distance for _ in active_envs]
        initial_dists = [self.action_manager.grasp_distances[env_id].item() for env_id in active_envs]
        self.action_manager.stretch_two_vertices_batch(
            active_envs, initial_dists, max_distances=max_dists
        )
        self.action_manager.set_sequence_phase('stretching')
        stretch_duration_steps = self._calculate_steps_from_velocity(
            self.cfg.stretch_max_distance / 2,
            self.cfg.stretch_motion_velocity
        )
        self._run_physics_steps(sim, scene, stretch_duration_steps, check_stretch=True, phase_type="stretch")

        fling_height = self._get_current_grasp_height(active_envs)

        self._move_grasps_in_yz_plane(
            sim=sim,
            scene=scene,
            env_ids=active_envs,
            target_y=float(fling_cfg["forward_y"]),
            target_height=float(fling_height),
            velocity=float(fling_cfg["swing_velocity"]),
        )

        self._move_grasps_in_yz_plane(
            sim=sim,
            scene=scene,
            env_ids=active_envs,
            target_y=float(fling_cfg["backward_y"]),
            target_height=float(fling_cfg["backward_height"]),
            velocity=float(fling_cfg["swing_velocity"]),
        )

        hold_steps = int(fling_cfg["backward_hold_steps"])
        if hold_steps > 0:
            self._run_physics_steps(sim, scene, hold_steps, phase_type="move")

    def _execute_holding_phase(self, sim, scene):
        self.action_manager.set_sequence_phase('holding')
        active_envs = self.action_manager.grasp_active_mask.nonzero().squeeze(-1).cpu().tolist()
        use_y_gravity = self.loading_mode == "y_gravity"
        if use_y_gravity:
            # Keep the cloth under lateral +y gravity while the constrained vertices
            # are lowered. Restore normal z gravity only after the cloth is near ground.
            self._set_gravity((0.0, 4.9, 0.0))
        else:
            self._set_gravity((0.0, 0.0, -9.8))

        # 1. Lower constrained vertices toward the ground while preserving y-gravity.
        max_lower_distance = 0.0
        all_pos = self.action_manager.garment_asset._get_particle_positions()
        
        for env_id in range(self.num_envs):
            if self.action_manager.grasp_active_mask[env_id]:
                v1 = self.action_manager.grasp_vertex_ids[env_id, 0].item()
                v2 = self.action_manager.grasp_vertex_ids[env_id, 1].item()
                pos1 = all_pos[env_id, v1]
                pos2 = all_pos[env_id, v2]
                
                if pos1 is not None and pos2 is not None:
                    current_z = max(float(pos1[2].item()), float(pos2[2].item()))
                    env_origin = self.action_manager.env_origins[env_id]
                    target_z = float(env_origin[2]) + self.cfg.action_sequence["holding"]["lower_vertices"]["ground_height_offset"]
                    max_lower_distance = max(max_lower_distance, abs(current_z - target_z))
        
        normal_velocity = self.cfg.normal_motion_velocity
        lower_steps = self._calculate_steps_from_velocity(max_lower_distance, normal_velocity)
        
        self.action_manager.lower_vertices_to_ground(
            ground_height_offset=self.cfg.action_sequence["holding"]["lower_vertices"]["ground_height_offset"],
            duration_steps=lower_steps
        )
        self._run_physics_steps(sim, scene, lower_steps, phase_type="move")
        if self._debug_enabled("protocol_trace"):
            self._debug_log_env_state("holding.after_lower", active_envs)
        
        # 2. Restore normal gravity only after the constrained vertices are lowered.
        self._set_gravity((0.0, 0.0, -9.8))
        hold_restore_steps = self.cfg.action_sequence["holding"]["gravity_restore"]["duration_steps"] if use_y_gravity else 0
        if hold_restore_steps > 0:
            self._run_physics_steps(sim, scene, hold_restore_steps, check_stability=True, phase_type="stable")
        if self._debug_enabled("protocol_trace"):
            self._debug_log_env_state("holding.after_restore", active_envs)

    def _execute_releasing_phase(self, sim, scene):
        self.action_manager.set_sequence_phase('releasing')
        self.action_manager.stop_all_control()

    def _execute_stabilizing_phase(self, sim, scene):
        self.action_manager.set_sequence_phase('stabilizing')
        self._run_physics_steps(sim, scene, self.cfg.action_sequence["stabilizing"]["max_duration_steps"], check_stability=True, phase_type="stable")

    def _calculate_steps_from_velocity(self, distance: float, velocity: float) -> int:
        if velocity <= 0: raise ValueError(f"Velocity must be positive, got {velocity}")
        time_seconds = distance / velocity
        steps = int(time_seconds / self.physics_dt)
        return max(1, steps)

    def _set_gravity(self, direction):
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(*direction))

    def _run_physics_steps(self, sim, scene, num_steps, 
                           check_stability=False,
                           check_stretch=False,
                           phase_type="move"):
        """Unified physics stepping with optional checks.
        
        Args:
            num_steps: Maximum number of steps (can exit early)
            check_stability: Check cloth stability every N steps
            check_stretch: Update stretch targets and check completion
            phase_type: One of "stretch", "stable", "move" for stats tracking
            
        Returns:
            Tuple of (completed_early, actual_steps, elapsed_time)
        """
        recorder = self.frame_recorder
        check_interval = self.cfg.check_interval
        start_time = time.perf_counter()
        if check_stretch:
            self._last_stretch_debug_emitted = False
        
        all_pos = self.action_manager.garment_asset._get_particle_positions()
        actual_steps = 0

        for step_idx in range(num_steps):
            actual_steps = step_idx + 1
            with profile("sim.pre_step"):
                scene.write_data_to_sim()
                
                if check_stretch:
                    self.action_manager._update_stretch_adaptive(all_pos)

                # Apply control EVERY step (not conditionally)
                self.action_manager.apply_control()
            
            with profile("sim.step"):
                sim.step(render=False)
            
            with profile("sim.post_step"):
                recorder_needs_frame = False
                if recorder is not None:
                    try:
                        recorder_needs_frame = bool(
                            recorder.wants_capture(phase_type=phase_type, step_idx=step_idx)
                        )
                    except Exception:
                        recorder_needs_frame = True
                render_every_step = bool(getattr(recorder, "render_every_step", True)) if recorder is not None else False
                should_render = (
                    sim.has_gui()
                    or render_every_step
                    or recorder_needs_frame
                )
                if should_render and step_idx % self.cfg.sim.render_interval == 0:
                    sim.render()
                
                scene.update(dt=self.physics_dt)
                if recorder is not None:
                    try:
                        recorder.capture(phase_type=phase_type, step_idx=step_idx)
                    except Exception:
                        pass
                    if getattr(recorder, "should_stop", False):
                        elapsed = time.perf_counter() - start_time
                        self._accumulate_stats(phase_type, actual_steps, elapsed)
                        return True
                
                all_pos = self.action_manager.garment_asset._get_particle_positions()

                # Check AFTER simulation
                if step_idx % check_interval == 0:
                    # Stability check
                    if check_stability:
                        if self.action_manager.check_all_envs_stable(all_pos):
                            elapsed = time.perf_counter() - start_time
                            self._accumulate_stats(phase_type, actual_steps, elapsed)
                            return True
                    
                    # Stretch check
                    if check_stretch:
                        if self.action_manager._is_stretching_done(all_pos):
                            self._emit_stretch_debug(phase_type)
                            elapsed = time.perf_counter() - start_time
                            self._accumulate_stats(phase_type, actual_steps, elapsed)
                            return True
        
        if check_stretch:
            self._emit_stretch_debug(phase_type)
        elapsed = time.perf_counter() - start_time
        self._accumulate_stats(phase_type, actual_steps, elapsed)
        return False
    
    def _accumulate_stats(self, phase_type: str, steps: int, elapsed: float):
        """Accumulate stats for a phase into the step stats."""
        # DEBUG: Uncomment to diagnose stats issue
        # print(f"[DEBUG] _accumulate_stats: phase={phase_type}, steps={steps}, elapsed={elapsed:.3f}s")
        if phase_type == "stretch":
            self._step_stats.stretch_steps += steps
            self._step_stats.stretch_time += elapsed
        elif phase_type == "stable":
            self._step_stats.stable_steps += steps
            self._step_stats.stable_time += elapsed
        else:  # move
            self._step_stats.move_steps += steps
            self._step_stats.move_time += elapsed

    def stabilize(self, sim, scene):
        """Run physics steps until the target environments stabilize or time out."""
        self._run_physics_steps(sim, scene, self.cfg.num_stabilization_steps_on_reset, check_stability=False)
        self._run_physics_steps(sim, scene, self.cfg.action_sequence["stabilizing"]["max_duration_steps"], check_stability=True)

    def _resolve_loading_mode(self) -> str:
        action_cfg = getattr(self.cfg, "action_sequence", {}) or {}
        mode = str(action_cfg.get("loading_mode", "y_gravity")).strip().lower()
        if mode not in {"y_gravity", "fling"}:
            raise ValueError(f"Unsupported loading mode: {mode}")
        return mode

    def _resolve_init_mode(self) -> str:
        action_cfg = getattr(self.cfg, "action_sequence", {}) or {}
        mode = str(action_cfg.get("init_mode", "random")).strip().lower()
        if mode not in {"random", "conditioned"}:
            raise ValueError(f"Unsupported init mode: {mode}")
        return mode

    def _get_fling_cfg(self) -> dict:
        action_cfg = getattr(self.cfg, "action_sequence", {}) or {}
        fling_cfg = dict(action_cfg.get("fling", {}) or {})
        fling_cfg.setdefault("stretch_pose_velocity", 0.16)
        fling_cfg.setdefault("height_adjust_velocity", self.cfg.normal_motion_velocity)
        fling_cfg.setdefault("height_adjust_max_step", 0.05)
        fling_cfg.setdefault("height_adjust_target_clearance", 0.08)
        fling_cfg.setdefault("height_adjust_target_clearance_max", fling_cfg["height_adjust_target_clearance"])
        fling_cfg.setdefault("height_adjust_bulk_target_clearance", fling_cfg["height_adjust_target_clearance"])
        fling_cfg.setdefault(
            "height_adjust_bulk_target_clearance_max",
            fling_cfg["height_adjust_bulk_target_clearance"],
        )
        fling_cfg.setdefault("height_adjust_span_gain", 0.0)
        fling_cfg.setdefault("height_adjust_bulk_span_gain", fling_cfg["height_adjust_span_gain"])
        fling_cfg.setdefault("height_adjust_upper_margin", 0.10)
        fling_cfg.setdefault("height_adjust_bulk_upper_margin", fling_cfg["height_adjust_upper_margin"])
        fling_cfg.setdefault("height_adjust_tolerance", 0.01)
        fling_cfg.setdefault("height_adjust_max_rounds", 12)
        fling_cfg.setdefault("height_adjust_settle_steps", max(2, self.cfg.check_interval))
        fling_cfg.setdefault("forward_y", 0.20)
        fling_cfg.setdefault("backward_y", -0.20)
        fling_cfg.setdefault(
            "backward_height",
            float(self.cfg.action_sequence["holding"]["lower_vertices"]["ground_height_offset"]) + 0.04,
        )
        fling_cfg.setdefault("swing_velocity", 0.45)
        fling_cfg.setdefault("backward_hold_steps", 4)
        fling_cfg.setdefault("pre_stretch_settle_steps", max(2, self.cfg.check_interval))
        return fling_cfg

    def _debug_cfg(self):
        return getattr(self.cfg, "debug", None)

    def _debug_enabled(self, key: str) -> bool:
        debug_cfg = self._debug_cfg()
        if debug_cfg is None:
            return False
        if isinstance(debug_cfg, dict):
            return bool(debug_cfg.get(key, False))
        try:
            return bool(getattr(debug_cfg, key, False))
        except Exception:
            try:
                return bool(debug_cfg[key])
            except Exception:
                return False

    def _debug_log(self, message: str) -> None:
        print(f"[UNFOLD_DEBUG] {message}", file=sys.stdout, flush=True)

    def _debug_log_env_state(self, tag: str, env_ids: list[int], target_height: float | None = None) -> None:
        if not env_ids:
            return
        all_pos = self.action_manager.garment_asset._get_particle_positions()
        padding_mask = self.action_manager.garment_asset.get_padding_mask()
        parts = []
        for env_id in env_ids:
            env_origin_z = float(self.action_manager.env_origins[env_id][2])
            v1 = self.action_manager.grasp_vertex_ids[env_id, 0].item()
            v2 = self.action_manager.grasp_vertex_ids[env_id, 1].item()
            grasp_h = max(float(all_pos[env_id, v1, 2].item()), float(all_pos[env_id, v2, 2].item())) - env_origin_z
            valid = padding_mask[env_id, :, 0] > 0.5 if padding_mask is not None else torch.ones(
                all_pos.shape[1], dtype=torch.bool, device=self.device
            )
            cloth_min_z = float(all_pos[env_id, valid, 2].min().item()) if int(valid.sum()) > 0 else env_origin_z
            part = (
                f"env={env_id} grasp_h={grasp_h:.3f} "
                f"cloth_min_h={cloth_min_z - env_origin_z:.3f}"
            )
            if target_height is not None:
                part += f" target_h={float(target_height):.3f}"
            parts.append(part)
        self._debug_log(f"{tag}: " + " | ".join(parts))

    def _emit_stretch_debug(self, phase_type: str) -> None:
        if phase_type != "stretch" or self._last_stretch_debug_emitted or not self._debug_enabled("stretch_trace"):
            return
        report = getattr(self.action_manager, "_last_stretch_debug_report", []) or []
        if not report:
            return
        summary = []
        for item in report:
            piece = (
                f"env={item['env_id']} status={item['status']} "
                f"dist={item['current_dist']:.3f}/{item['max_dist']:.3f}"
            )
            if "max_strain" in item:
                piece += f" strain={item['max_strain']:.3f}"
            summary.append(piece)
        self._debug_log("stretch.done: " + " | ".join(summary))
        self._last_stretch_debug_emitted = True

    def _adjust_grasp_height_closed_loop(self, sim, scene, *, env_ids: list[int]) -> None:
        if not env_ids:
            return

        fling_cfg = self._get_fling_cfg()
        base_target_clearance = float(fling_cfg["height_adjust_target_clearance"])
        max_target_clearance = float(
            fling_cfg.get("height_adjust_target_clearance_max", base_target_clearance)
        )
        base_bulk_target_clearance = float(
            fling_cfg.get("height_adjust_bulk_target_clearance", base_target_clearance)
        )
        max_bulk_target_clearance = float(
            fling_cfg.get("height_adjust_bulk_target_clearance_max", base_bulk_target_clearance)
        )
        bulk_quantile = float(fling_cfg.get("height_adjust_bulk_quantile", 0.10))
        span_gain = float(fling_cfg.get("height_adjust_span_gain", 0.0))
        bulk_span_gain = float(fling_cfg.get("height_adjust_bulk_span_gain", span_gain))
        upper_margin = float(fling_cfg.get("height_adjust_upper_margin", 0.10))
        bulk_upper_margin = float(fling_cfg.get("height_adjust_bulk_upper_margin", upper_margin))
        tolerance = float(fling_cfg["height_adjust_tolerance"])
        max_rounds = int(fling_cfg.get("height_adjust_max_rounds", 200))
        max_step = float(fling_cfg["height_adjust_max_step"])
        velocity = float(fling_cfg["height_adjust_velocity"])
        settle_steps = int(fling_cfg["height_adjust_settle_steps"])
        padding_mask = self.action_manager.garment_asset.get_padding_mask()

        target_clearances: dict[int, float] = {}
        bulk_target_clearances: dict[int, float] = {}
        target_clearance_uppers: dict[int, float] = {}
        bulk_target_clearance_uppers: dict[int, float] = {}

        def _compute_target_band(all_positions: torch.Tensor, env_id: int) -> tuple[float, float, float, float]:
            env_origin = self.action_manager.env_origins[env_id]
            valid = padding_mask[env_id, :, 0] > 0.5 if padding_mask is not None else torch.ones(
                all_positions.shape[1], dtype=torch.bool, device=self.device
            )
            if int(valid.sum()) > 0:
                z_vals = all_positions[env_id, valid, 2]
                cloth_min_z = float(z_vals.min().item())
                cloth_max_z = float(z_vals.max().item())
            else:
                cloth_min_z = float(env_origin[2])
                cloth_max_z = float(env_origin[2])
            min_h = cloth_min_z - float(env_origin[2])
            max_h = cloth_max_z - float(env_origin[2])
            span_h = max(0.0, max_h - min_h)
            target_clearance = min(max_target_clearance, base_target_clearance + span_gain * span_h)
            bulk_target_clearance = min(
                max_bulk_target_clearance,
                base_bulk_target_clearance + bulk_span_gain * span_h,
            )
            return (
                target_clearance,
                bulk_target_clearance,
                target_clearance + upper_margin,
                bulk_target_clearance + bulk_upper_margin,
            )

        initial_positions = self.action_manager.garment_asset._get_particle_positions()
        for env_id in env_ids:
            target_clearance, bulk_target_clearance, target_clearance_upper, bulk_target_clearance_upper = (
                _compute_target_band(initial_positions, env_id)
            )
            target_clearances[env_id] = target_clearance
            bulk_target_clearances[env_id] = bulk_target_clearance
            target_clearance_uppers[env_id] = target_clearance_upper
            bulk_target_clearance_uppers[env_id] = bulk_target_clearance_upper

        def _measure_clearance() -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
            all_positions = self.action_manager.garment_asset._get_particle_positions()
            errors: dict[int, float] = {}
            mins: dict[int, float] = {}
            bulks: dict[int, float] = {}
            maxs: dict[int, float] = {}
            for env_id in env_ids:
                env_origin = self.action_manager.env_origins[env_id]
                valid = padding_mask[env_id, :, 0] > 0.5 if padding_mask is not None else torch.ones(
                    all_positions.shape[1], dtype=torch.bool, device=self.device
                )
                if int(valid.sum()) > 0:
                    z_vals = all_positions[env_id, valid, 2]
                    cloth_min_z = float(z_vals.min().item())
                    bulk_z = float(torch.quantile(z_vals, bulk_quantile).item())
                    cloth_max_z = float(z_vals.max().item())
                else:
                    cloth_min_z = float(env_origin[2])
                    bulk_z = float(env_origin[2])
                    cloth_max_z = float(env_origin[2])
                min_h = cloth_min_z - float(env_origin[2])
                bulk_h = bulk_z - float(env_origin[2])
                max_h = cloth_max_z - float(env_origin[2])
                target_clearance = target_clearances[env_id]
                bulk_target_clearance = bulk_target_clearances[env_id]
                mins[env_id] = min_h
                bulks[env_id] = bulk_h
                maxs[env_id] = max_h
                errors[env_id] = max(target_clearance - min_h, bulk_target_clearance - bulk_h)
            return errors, mins, bulks, maxs

        last_move_sign = {env_id: 0.0 for env_id in env_ids}
        completed_envs: set[int] = set()

        round_idx = 0
        while True:
            all_pos = self.action_manager.garment_asset._get_particle_positions()
            envs_to_move: list[int] = []
            vids_list: list[list[int]] = []
            targets_list: list[list[torch.Tensor]] = []
            deltas: list[float] = []
            debug_parts: list[str] = []
            complete_after_move: set[int] = set()

            for env_id in env_ids:
                if env_id in completed_envs:
                    continue
                env_origin = self.action_manager.env_origins[env_id]
                valid = padding_mask[env_id, :, 0] > 0.5 if padding_mask is not None else torch.ones(
                    all_pos.shape[1], dtype=torch.bool, device=self.device
                )
                if int(valid.sum()) > 0:
                    z_vals = all_pos[env_id, valid, 2]
                    cloth_min_z = float(z_vals.min().item())
                    bulk_z = float(torch.quantile(z_vals, bulk_quantile).item())
                    cloth_max_z = float(z_vals.max().item())
                else:
                    cloth_min_z = float(env_origin[2])
                    bulk_z = float(env_origin[2])
                    cloth_max_z = float(env_origin[2])
                min_h = cloth_min_z - float(env_origin[2])
                bulk_h = bulk_z - float(env_origin[2])
                max_h = cloth_max_z - float(env_origin[2])
                span_h = max(0.0, max_h - min_h)
                target_clearance = target_clearances[env_id]
                bulk_target_clearance = bulk_target_clearances[env_id]
                target_clearance_upper = target_clearance_uppers[env_id]
                bulk_target_clearance_upper = bulk_target_clearance_uppers[env_id]
                low_error = max(target_clearance - min_h, bulk_target_clearance - bulk_h)
                high_error = max(min_h - target_clearance_upper, bulk_h - bulk_target_clearance_upper)
                prev_sign = float(last_move_sign[env_id])
                error = 0.0
                delta_sign = 0.0
                status = "within_band"

                if low_error <= tolerance and high_error <= tolerance:
                    status = "within_band"
                    completed_envs.add(env_id)
                elif prev_sign > 0.0 and high_error > tolerance:
                    # We were moving upward and have now overshot the band.
                    # Stop immediately instead of reversing downward, which causes ping-pong.
                    status = "overshoot_after_up_stop"
                    completed_envs.add(env_id)
                elif prev_sign < 0.0 and low_error > tolerance:
                    # We were moving downward and have gone too low.
                    # Roll back by one upward correction and then stop.
                    error = low_error
                    delta_sign = 1.0
                    status = "rollback_after_down"
                elif low_error > tolerance:
                    error = low_error
                    delta_sign = 1.0
                    status = "move_up"
                elif high_error > tolerance:
                    error = high_error
                    delta_sign = -1.0
                    status = "move_down"

                if self._debug_enabled("protocol_trace"):
                    debug_parts.append(
                        f"env={env_id} min_h={min_h:.3f} bulk_h={bulk_h:.3f} "
                        f"span_h={span_h:.3f} "
                        f"targets=({target_clearance:.3f},{bulk_target_clearance:.3f}) "
                        f"upper=({target_clearance_upper:.3f},{bulk_target_clearance_upper:.3f}) "
                        f"error={error:.3f} sign={delta_sign:+.0f} prev_sign={prev_sign:+.0f} status={status}"
                    )
                if env_id in completed_envs:
                    continue

                v1 = self.action_manager.grasp_vertex_ids[env_id, 0].item()
                v2 = self.action_manager.grasp_vertex_ids[env_id, 1].item()
                p1 = all_pos[env_id, v1].clone()
                p2 = all_pos[env_id, v2].clone()
                delta_z = min(max_step, error)
                delta = torch.tensor([0.0, 0.0, delta_sign * delta_z], device=self.device, dtype=p1.dtype)
                envs_to_move.append(env_id)
                vids_list.append([v1, v2])
                targets_list.append([p1 + delta, p2 + delta])
                deltas.append(abs(float(delta_z)))
                last_move_sign[env_id] = delta_sign
                if status == "rollback_after_down":
                    complete_after_move.add(env_id)
                if self._debug_enabled("protocol_trace"):
                    debug_parts[-1] += f" delta_z={delta_sign * delta_z:.3f}"

            if self._debug_enabled("protocol_trace") and debug_parts:
                self._debug_log(f"fling.height_round{round_idx}: " + " | ".join(debug_parts))

            all_converged = len(completed_envs) == len(env_ids)
            if not envs_to_move and all_converged:
                if self._debug_enabled("protocol_trace"):
                    _, mins, bulks, maxs = _measure_clearance()
                    parts = [
                        f"env={env_id} min_h={mins[env_id]:.3f} bulk_h={bulks[env_id]:.3f} max_h={maxs[env_id]:.3f}"
                        for env_id in env_ids
                    ]
                    self._debug_log("fling.height_converged: " + " | ".join(parts))
                return
            if not envs_to_move:
                if settle_steps > 0:
                    self._run_physics_steps(sim, scene, settle_steps, check_stability=True, phase_type="stable")
                round_idx += 1
                if round_idx >= max_rounds:
                    raise RuntimeError(
                        f"Fling height adjustment did not converge after {max_rounds} rounds. "
                        f"Increase action_sequence.fling.height_adjust_max_rounds or inspect the grasp/cloth state."
                    )
                continue

            if round_idx >= max_rounds:
                raise RuntimeError(
                    f"Fling height adjustment did not converge after {max_rounds} rounds. "
                    f"Increase action_sequence.fling.height_adjust_max_rounds or inspect the grasp/cloth state."
                )

            duration_steps = self._calculate_steps_from_velocity(max(deltas), velocity)
            self.action_manager.update_trajectory(envs_to_move, vids_list, targets_list, duration_steps)
            self.action_manager.set_sequence_phase('lifting')
            self._run_physics_steps(sim, scene, duration_steps, phase_type="move")
            if settle_steps > 0:
                self._run_physics_steps(sim, scene, settle_steps, check_stability=True, phase_type="stable")
            completed_envs.update(complete_after_move)
            round_idx += 1

    def _get_current_grasp_height(self, env_ids: list[int]) -> float:
        all_pos = self.action_manager.garment_asset._get_particle_positions()
        max_height = 0.0
        for env_id in env_ids:
            v1 = self.action_manager.grasp_vertex_ids[env_id, 0].item()
            v2 = self.action_manager.grasp_vertex_ids[env_id, 1].item()
            max_height = max(max_height, float(all_pos[env_id, v1, 2].item()), float(all_pos[env_id, v2, 2].item()))
        env_origin_z = float(self.action_manager.env_origins[env_ids[0]][2]) if env_ids else 0.0
        return max(0.0, max_height - env_origin_z)

    def _get_conditioned_target_height(self) -> float:
        cfg_pc = getattr(self.cfg, "pair_conditioned_collection", None)
        if cfg_pc is None:
            return 1.5
        try:
            return float(cfg_pc["pair_target_height"])
        except Exception:
            return float(getattr(cfg_pc, "pair_target_height", 1.5))

    def _get_pair_pose_velocity(self) -> float:
        action_cfg = getattr(self.cfg, "action_sequence", {}) or {}
        return float(action_cfg.get("pair_pose_velocity", self.cfg.normal_motion_velocity))

    def _move_grasps_to_symmetric_stretch_pose(
        self,
        *,
        sim,
        scene,
        env_ids: list[int],
        velocity: float,
        target_height: float | None = None,
    ) -> None:
        if not env_ids:
            return

        all_pos = self.action_manager.garment_asset._get_particle_positions()
        vids_list = []
        targets_list = []
        max_distance = 0.0
        dtype = all_pos.dtype
        for env_id in env_ids:
            v1 = self.action_manager.grasp_vertex_ids[env_id, 0].item()
            v2 = self.action_manager.grasp_vertex_ids[env_id, 1].item()
            env_origin = self.action_manager.env_origins[env_id]
            current_height = max(float(all_pos[env_id, v1, 2].item()), float(all_pos[env_id, v2, 2].item()))
            target_world_height = current_height if target_height is None else float(env_origin[2]) + float(target_height)
            half_d = float(self.action_manager.grasp_distances[env_id].item()) / 2.0
            t1 = torch.tensor([float(env_origin[0]) - half_d, float(env_origin[1]), target_world_height], device=self.device, dtype=dtype)
            t2 = torch.tensor([float(env_origin[0]) + half_d, float(env_origin[1]), target_world_height], device=self.device, dtype=dtype)
            p1 = all_pos[env_id, v1].clone()
            p2 = all_pos[env_id, v2].clone()
            max_distance = max(
                max_distance,
                float(torch.linalg.vector_norm(t1 - p1).item()),
                float(torch.linalg.vector_norm(t2 - p2).item()),
            )
            vids_list.append([v1, v2])
            targets_list.append([t1, t2])

        duration_steps = self._calculate_steps_from_velocity(max_distance, velocity)
        self.action_manager.update_trajectory(env_ids, vids_list, targets_list, duration_steps)
        self.action_manager.set_sequence_phase('holding')
        self._run_physics_steps(sim, scene, duration_steps, phase_type="move")

    def _move_grasps_in_yz_plane(
        self,
        *,
        sim,
        scene,
        env_ids: list[int],
        target_y: float,
        target_height: float,
        velocity: float,
    ) -> None:
        if not env_ids:
            return

        all_pos = self.action_manager.garment_asset._get_particle_positions()
        vids_list = []
        targets_list = []
        max_distance = 0.0
        for env_id in env_ids:
            v1 = self.action_manager.grasp_vertex_ids[env_id, 0].item()
            v2 = self.action_manager.grasp_vertex_ids[env_id, 1].item()
            env_origin = self.action_manager.env_origins[env_id]
            world_target_y = float(env_origin[1]) + float(target_y)
            target_z = float(env_origin[2]) + float(target_height)

            p1 = all_pos[env_id, v1].clone()
            p2 = all_pos[env_id, v2].clone()
            t1 = p1.clone()
            t2 = p2.clone()
            t1[1] = world_target_y
            t2[1] = world_target_y
            t1[2] = target_z
            t2[2] = target_z
            max_distance = max(
                max_distance,
                float(torch.linalg.vector_norm(t1 - p1).item()),
                float(torch.linalg.vector_norm(t2 - p2).item()),
            )
            vids_list.append([v1, v2])
            targets_list.append([t1, t2])

        duration_steps = self._calculate_steps_from_velocity(max_distance, velocity)
        self.action_manager.update_trajectory(env_ids, vids_list, targets_list, duration_steps)
        self.action_manager.set_sequence_phase('holding')
        self._run_physics_steps(sim, scene, duration_steps, phase_type="move")
