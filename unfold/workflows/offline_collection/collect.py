#!/usr/bin/env python3
"""Offline data collection pipeline."""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

from .cli import build_collect_parser
from .common import configure_runtime_warnings, install_exit_signal_handlers
from .env import calc_step_targets, load_env_and_cfg


class OfflineCollector:
    def __init__(self, env, env_cfg, storage, args, num_batches, steps_per_epoch, total_steps):
        self.env = env
        self.env_cfg = env_cfg
        self.storage = storage
        self.args = args
        self.num_batches = num_batches
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_steps

        self.step_count = 0
        self.start_time = time.time()
        self.reward_buffer = deque(maxlen=10000)
        self.icp_buffer = deque(maxlen=10000)
        self.l2_buffer = deque(maxlen=10000)
        self.real_l2_buffer = deque(maxlen=10000)
        self.positions_buffer = defaultdict(list)
        self.policy = None
        self.vis_root = Path(self.args.vis_dir)
        self.vis_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _extract_distances(info):
        if not isinstance(info, dict):
            return None, None, None
        dist_dict = info.get("rewards_extras", info)
        return (
            dist_dict.get("l2_distance"),
            dist_dict.get("icp_distance"),
            dist_dict.get("real_l2_distance"),
        )

    @staticmethod
    def _to_list(val):
        if val is None:
            return []
        if torch.is_tensor(val):
            return val.detach().cpu().view(-1).tolist()
        if isinstance(val, np.ndarray):
            return val.reshape(-1).tolist()
        if isinstance(val, (list, tuple)):
            return list(val)
        return [float(val)]

    @staticmethod
    def _pick_env_val(val, idx):
        if val is None:
            return float("nan")
        if torch.is_tensor(val):
            if val.ndim == 0:
                return float(val.item())
            return float(val[idx].item())
        if isinstance(val, np.ndarray):
            return float(val.reshape(-1)[idx])
        if isinstance(val, (list, tuple)):
            return float(val[idx])
        return float(val)

    @staticmethod
    def _pick_vis_env_idx(rewards_extras) -> int:
        goal_xy = rewards_extras.get("goal_xy")
        if torch.is_tensor(goal_xy) and goal_xy.ndim > 0 and goal_xy.shape[0] > 0:
            return min(10, int(goal_xy.shape[0]) - 1)
        return 0

    def _refresh_policy(self):
        from unfold.algorithms.policies.random_policy import RandomPolicy

        manager = getattr(self.env.unwrapped, "_garment_manager", None)
        if manager is None:
            raise RuntimeError("Env missing _garment_manager")
        random_policy_cfg = getattr(self.env.unwrapped.cfg, "random_policy", {})
        self.policy = RandomPolicy(manager=manager, cfg=random_policy_cfg, device=self.env.unwrapped.device)

    def flush_buffers(self, reason: str):
        flushed_exp = self.storage.flush()
        total_pos_written = 0
        for asset_path, entries in list(self.positions_buffer.items()):
            if not entries:
                continue
            cropped_list = []
            for pos_np, mask_np in entries:
                if mask_np is None:
                    cropped = pos_np
                else:
                    count = int(mask_np.squeeze(-1).sum())
                    count = max(count, 0)
                    cropped = pos_np[:count] if count > 0 else pos_np[:1] * 0.0
                cropped_list.append(cropped.astype(np.float32, copy=False))

            self.storage.add_positions_batch(asset_path, cropped_list)
            total_pos_written += len(cropped_list)

        self.positions_buffer.clear()

        if flushed_exp or total_pos_written:
            elapsed = time.time() - self.start_time
            rate = self.storage.stats["total_stored"] / elapsed * 3600.0 if elapsed > 0 else 0.0
            print(
                f"[FLUSH] reason={reason} stored={self.storage.stats['total_stored']} "
                f"positions={total_pos_written} rate={rate:.0f} sam/hr",
                flush=True,
            )
            self.storage.update_metadata("last_step_count", self.step_count)

    def run(self, simulation_app):
        from unfold.platform.reward_vis import save_hist_png, save_scatter_png

        epoch_info = {"epoch": 1, "total_epochs": self.args.epochs, "batch": 1, "total_batches": self.num_batches}
        obs, info = self.env.unwrapped.reset(options={"switch_asset": True, "epoch_info": epoch_info})
        self._refresh_policy()

        print("\n[INFO] Starting offline data collection...")

        try:
            while simulation_app.is_running():
                with torch.no_grad():
                    actions = self.policy(obs)

                current_epoch = self.step_count // self.steps_per_epoch + 1
                step_in_epoch = self.step_count % self.env_cfg.steps_per_episode + 1
                self.env.unwrapped.progress_info = {
                    "epoch": current_epoch,
                    "total_epochs": self.args.epochs,
                    "step_in_epoch": step_in_epoch,
                    "steps_per_episode": self.env_cfg.steps_per_episode,
                    "step": self.step_count + 1,
                    "total_steps": self.total_steps,
                }

                obs, rewards, terminated, truncated, info = self.env.unwrapped.step(actions)
                self.step_count += 1

                if torch.is_tensor(rewards):
                    self.reward_buffer.extend(rewards.detach().cpu().view(-1).tolist())
                elif hasattr(rewards, "tolist"):
                    self.reward_buffer.extend(rewards.tolist())

                l2_val, icp_val, real_l2_val = self._extract_distances(info)
                self.l2_buffer.extend(self._to_list(l2_val))
                self.icp_buffer.extend(self._to_list(icp_val))
                self.real_l2_buffer.extend(self._to_list(real_l2_val))

                experiences = self.env.unwrapped.get_experience_data(actions, rewards, include_features=False)
                valid_mask = (actions[:, 0] >= 0) & (actions[:, 1] >= 0)
                valid_envs = valid_mask.nonzero().squeeze(-1).cpu().tolist()

                valid_position_entries: list[tuple[str, np.ndarray, np.ndarray | None]] = []
                for i, exp in enumerate(experiences):
                    asset_path, id1, _, id2, _, reward = exp
                    if math.isnan(reward):
                        continue
                    env_idx = valid_envs[i]
                    l2_env = self._pick_env_val(l2_val, env_idx)
                    icp_env = self._pick_env_val(icp_val, env_idx)
                    self.storage.add(asset_path, id1, id2, reward, l2_dist=l2_env, icp_dist=icp_env)
                    pos_env = obs["pos"][env_idx].detach().cpu().numpy()
                    mask_env = obs["pos_mask"][env_idx].detach().cpu().numpy() if obs.get("pos_mask") is not None else None
                    valid_position_entries.append((asset_path, pos_env, mask_env))

                for asset_path, pos_env, mask_env in valid_position_entries:
                    self.positions_buffer[asset_path].append((pos_env, mask_env))

                if self.step_count >= self.total_steps:
                    print(f"\n[INFO] Reached total steps ({self.total_steps}). Collection Complete.")
                    self.flush_buffers("total_steps")
                    break

                switched = info.get("switch_asset", False) if isinstance(info, dict) else False
                if switched:
                    self.flush_buffers("switch_asset")
                    self._refresh_policy()

                    if hasattr(self.env.unwrapped, "extras") and "switch_asset" in self.env.unwrapped.extras:
                        self.env.unwrapped.extras.pop("switch_asset", None)

                    all_rew, all_l2, all_icp = self.storage.get_all_distances()
                    if len(all_l2) > 0:
                        save_hist_png(all_l2, self.vis_root / "global_l2_dist.png", rmin=0, rmax=None, title=f"Global L2 Dist (Count {len(all_l2)})", xlabel="L2 Distance")
                    if len(all_icp) > 0:
                        save_hist_png(all_icp, self.vis_root / "global_icp_dist.png", rmin=0, rmax=None, title=f"Global ICP Dist (Count {len(all_icp)})", xlabel="ICP Distance")

                    rewards_extras = info["rewards_extras"]
                    vis_env_idx = self._pick_vis_env_idx(rewards_extras)
                    goal_xy = rewards_extras["goal_xy"][vis_env_idx].detach().cpu().numpy()
                    reverse_goal_xy = rewards_extras["reverse_goal_xy"][vis_env_idx].detach().cpu().numpy()
                    icp_verts = rewards_extras["icp_verts"][vis_env_idx].detach().cpu().numpy()
                    padding_mask = rewards_extras["padding_mask"][vis_env_idx].detach().cpu().numpy()

                    deformable_dist = rewards_extras.get("deformable_distance", rewards_extras.get("l2_distance"))
                    rigid_dist = rewards_extras.get("rigid_distance", rewards_extras.get("icp_distance"))
                    deformable_dist_val = deformable_dist[vis_env_idx].item() if torch.is_tensor(deformable_dist) else float(deformable_dist)
                    rigid_dist_val = rigid_dist[vis_env_idx].item() if torch.is_tensor(rigid_dist) else float(rigid_dist)

                    mask = padding_mask[:, 0] > 0.5
                    if mask.sum() > 0:
                        save_scatter_png(ref_points=goal_xy[mask], other_points=icp_verts[mask], out_path=self.vis_root / "deformable.png", title="Deformable (Goal vs ICP-Aligned Current)", ref_label="Goal", other_label="Aligned Current", distance=deformable_dist_val)
                        save_scatter_png(ref_points=goal_xy[mask], other_points=reverse_goal_xy[mask], out_path=self.vis_root / "rigid.png", title="Rigid (Goal vs Aligned Goal - Y Corrected)", ref_label="Goal", other_label="Aligned Goal", distance=rigid_dist_val)

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"[ERROR] {exc}")
        finally:
            print("\n[INFO] Closing and flushing storage...")
            try:
                self.flush_buffers("shutdown")
            except Exception as flush_err:
                print(f"[WARN] Flush during shutdown failed: {flush_err}")
            self.storage.close()
            self.env.close()
            simulation_app.close()

            print("\n========== Collection Stats ==========")
            print(f"Total Stored: {self.storage.stats['total_stored']}")
            for cat, count in self.storage.stats["by_category"].items():
                print(f"  {cat}: {count}")
            print("======================================\n")


def run(args) -> None:
    configure_runtime_warnings()
    install_exit_signal_handlers("Signal received, closing storage...")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import carb
    import unfold  # noqa: F401
    from unfold.data.storage.structured_hdf5 import StructuredHDF5Storage

    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    carb.log_warn = lambda *a, **k: None

    env, env_cfg = load_env_and_cfg(args)
    pool = env.unwrapped._asset_pool
    num_batches, steps_per_epoch, total_steps = calc_step_targets(env_cfg, pool, args.epochs)

    print("\n[INFO] Epoch Calculation:")
    print(f"  Assets: {pool.size}")
    print(f"  Batches: {num_batches} (Batch Size: {env_cfg.scene.num_envs})")
    print(f"  Episodes/Batch: {env_cfg.episodes_per_asset_batch}")
    print(f"  Steps/Episode: {env_cfg.steps_per_episode}")
    print("  --------------------------------")
    print(f"  Steps/Epoch: {steps_per_epoch}")
    print(f"  Target Epochs: {args.epochs}")
    print(f"  Total Steps: {total_steps}")

    storage = StructuredHDF5Storage(file_path=args.output, feature_dim=0, overwrite=args.overwrite, write_threshold=1024)

    print("\n[INFO] ========== Offline Collection Configuration ==========")
    print(f"  Task: {args.task}")
    print(f"  Num Envs: {env_cfg.scene.num_envs}")
    print(f"  Output: {args.output}")
    print(f"  Total Target Steps: {total_steps}")
    print("===========================================================\n")

    start_step = 0
    if not args.overwrite:
        stored_step = storage.get_metadata("last_step_count")
        if stored_step is not None:
            start_step = int(stored_step)
            print(f"[INFO] Resuming from step {start_step}...")

    collector = OfflineCollector(env, env_cfg, storage, args, num_batches, steps_per_epoch, total_steps)
    collector.step_count = start_step
    collector.run(simulation_app)


def main() -> None:
    parser = build_collect_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
