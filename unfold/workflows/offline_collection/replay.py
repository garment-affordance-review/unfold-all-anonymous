#!/usr/bin/env python3
"""Offline data replay pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

from .cli import build_replay_parser
from .common import configure_runtime_warnings, install_exit_signal_handlers
from .env import load_env_and_cfg


def run(args) -> None:
    configure_runtime_warnings()
    install_exit_signal_handlers("Signal received, shutting down replay...")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import carb
    import unfold  # noqa: F401
    from unfold.data.storage.structured_hdf5 import StructuredHDF5Storage
    from unfold.algorithms.policies.offline_replay_policy import OfflineReplayPolicy

    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    carb.log_warn = lambda *a, **k: None

    env, env_cfg = load_env_and_cfg(args)

    env_cfg.episodes_per_asset_batch = 16
    env_cfg.steps_per_episode = 1

    try:
        pool = env.unwrapped._asset_pool
        num_batches = pool.num_batches
        steps_per_epoch = num_batches * env_cfg.episodes_per_asset_batch * env_cfg.steps_per_episode
        epoch_target = max(int(args.epochs), 1)
        total_steps = epoch_target * steps_per_epoch
        print(f"\n[INFO] Step budget from epochs: {total_steps} (epochs={epoch_target}, batches={num_batches})")
    except Exception as exc:
        print(f"[WARNING] Could not calculate steps from pool info: {exc}")
        steps_per_epoch = env_cfg.steps_per_episode
        total_steps = None

    if args.max_steps is not None:
        total_steps = args.max_steps

    replay_writer = None
    replay_file = None
    if args.replay_save:
        replay_path = Path(args.replay_save)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_file = replay_path.open("w", newline="", buffering=1_048_576)
        import csv

        replay_writer = csv.writer(replay_file)
        replay_writer.writerow(["step", "env", "asset", "pair_a", "pair_b", "offline_reward", "actual_reward", "error", "huber", "abs_error"])

    storage = StructuredHDF5Storage(file_path=args.replay_h5, feature_dim=0, write_threshold=args.write_threshold, overwrite=args.overwrite_replay)
    policy = None

    try:
        step_count = 0
        start_time = time.time()
        mae_sum = 0.0
        huber_sum = 0.0
        err_count = 0

        epoch_info = {"epoch": 1, "total_epochs": args.epochs, "batch": 1, "total_batches": getattr(env.unwrapped._asset_pool, "num_batches", "?")}
        obs, info = env.unwrapped.reset(options={"switch_asset": True, "epoch_info": epoch_info})

        manager = getattr(env.unwrapped, "_garment_manager", None)
        if manager is None:
            raise RuntimeError("Env missing _garment_manager")

        policy = OfflineReplayPolicy(manager=manager, offline_file=args.offline_file, device=env.unwrapped.device, cache_assets=not args.no_cache, max_reward_abs=args.max_reward_abs, filter_zero=not args.keep_zero, seed=args.seed, visible_resolution=args.visible_resolution)

        while simulation_app.is_running():
            current_manager = getattr(env.unwrapped, "_garment_manager", None)
            if current_manager is not None and current_manager is not policy.manager:
                policy.manager = current_manager

            with torch.no_grad():
                actions = policy(obs)

            current_epoch = (step_count // steps_per_epoch) + 1 if total_steps else 1
            step_in_epoch = (step_count % env_cfg.steps_per_episode) + 1
            env.unwrapped.progress_info = {
                "epoch": current_epoch,
                "total_epochs": args.epochs,
                "step_in_epoch": step_in_epoch,
                "steps_per_episode": env_cfg.steps_per_episode,
                "step": step_count + 1,
                "total_steps": total_steps if total_steps is not None else "?",
            }

            obs, rewards, terminated, truncated, info = env.unwrapped.step(actions)
            step_count += 1
            offline_info = getattr(policy, "last_offline_info", None)
            if offline_info:
                if torch.is_tensor(rewards):
                    actual_rewards = rewards.detach().cpu().tolist()
                elif hasattr(rewards, "tolist"):
                    actual_rewards = rewards.tolist()
                else:
                    actual_rewards = [rewards]

                for env_idx, info_env in enumerate(offline_info):
                    offline_reward = info_env.get("reward")
                    pair = info_env.get("pair")
                    asset_path = info_env.get("asset")
                    actual_reward = actual_rewards[env_idx] if env_idx < len(actual_rewards) else None
                    error = None
                    huber = None
                    abs_err = None
                    if offline_reward is not None and actual_reward is not None:
                        error = float(actual_reward) - float(offline_reward)
                        abs_err = abs(error)
                        delta = args.max_reward_abs
                        huber = 0.5 * error * error if abs_err <= delta else delta * (abs_err - 0.5 * delta)
                        mae_sum += abs_err
                        huber_sum += huber
                        err_count += 1

                    if asset_path is not None and pair is not None and actual_reward is not None:
                        dist_dict = info.get("rewards_extras", info) if isinstance(info, dict) else {}
                        l2_val = dist_dict.get("l2_distance")
                        icp_val = dist_dict.get("icp_distance")
                        if torch.is_tensor(l2_val):
                            l2_use = float(l2_val[env_idx].item()) if l2_val.ndim > 0 else float(l2_val.item())
                        elif l2_val is None:
                            l2_use = 0.0
                        else:
                            l2_use = float(l2_val[env_idx] if isinstance(l2_val, (list, tuple)) else l2_val)

                        if torch.is_tensor(icp_val):
                            icp_use = float(icp_val[env_idx].item()) if icp_val.ndim > 0 else float(icp_val.item())
                        elif icp_val is None:
                            icp_use = 0.0
                        else:
                            icp_use = float(icp_val[env_idx] if isinstance(icp_val, (list, tuple)) else icp_val)

                        storage.add(asset_path=asset_path, id1=pair[0], id2=pair[1], reward=actual_reward, l2_dist=l2_use, icp_dist=icp_use)

                    if replay_writer is not None:
                        a0, a1 = pair if pair is not None else (None, None)
                        replay_writer.writerow([step_count, env_idx, asset_path, a0, a1, offline_reward, actual_reward, error, huber, abs_err])

            if total_steps is not None and step_count >= total_steps:
                print(f"\n[INFO] Reached target steps ({total_steps}). Replay complete.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[ERROR] {exc}")
    finally:
        print("\n[INFO] Closing replay...")
        if policy is not None:
            policy.close()
        storage.close()
        env.close()
        simulation_app.close()
        if replay_file is not None:
            replay_file.close()
        if err_count > 0:
            mae = mae_sum / err_count
            mean_huber = huber_sum / err_count
            print(f"[STATS] Count={err_count} | MAE={mae:.4f} | mean Huber (delta={args.max_reward_abs})={mean_huber:.4f}")


def main() -> None:
    parser = build_replay_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
