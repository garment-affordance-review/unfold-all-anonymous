"""CLI helpers for offline collection pipeline."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def build_collect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Data Collection for UnfoldAll.")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0", help="Gym task id.")
    parser.add_argument("--config", type=str, default="configs/offline_standard.yaml", help="YAML config path.")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of parallel environments.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of complete passes over the asset pool.")
    parser.add_argument("--output", type=str, default="./data/offline_data-42.h5", help="Output HDF5 path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing HDF5 file.")
    parser.add_argument("--vis-dir", type=str, default="logs/visuals", help="Root directory for debug visuals.")
    parser.add_argument("--assets-manifest", type=str, default=None, help="Optional asset manifest JSON overriding valid_assets.json.")
    parser.add_argument("--rigid-fixed-y", action="store_true", help="Disable global Y shift in rigid term.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def build_pair_conditioned_collect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pair-conditioned offline data collection for UnfoldAll.")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0", help="Gym task id.")
    parser.add_argument("--config", type=str, default="configs/offline_pair_conditioned.yaml", help="YAML config path.")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of parallel environments.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing asset directories if they already exist.")
    parser.add_argument("--vis-dir", type=str, default="logs/pair_conditioned_visuals", help="Root directory for debug visuals.")
    parser.add_argument("--assets-manifest", type=str, default=None, help="Optional asset manifest JSON overriding valid_assets.json.")
    parser.add_argument("--rigid-fixed-y", action="store_true", help="Disable global Y shift in rigid term.")
    parser.add_argument("--worker-id", type=int, default=None, help="Optional worker identifier for distributed collection metadata.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def build_replay_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay offline data for UnfoldAll (random + visibility).")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0", help="Gym task id.")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of parallel environments.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of passes over asset pool (for step budgeting).")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard cap on total steps.")
    parser.add_argument("--offline-file", type=str, default="./data/offline_data.h5", help="Path to offline HDF5 (read-only).")
    parser.add_argument("--max-reward-abs", type=float, default=2.0, help="Filter out samples with |reward| greater than this.")
    parser.add_argument("--keep-zero", action="store_true", help="Do not drop zero rewards (dropped by default).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--visible-resolution", type=int, default=128, help="Visibility grid resolution.")
    parser.add_argument("--replay-save", type=str, default="logs/offline_replay/replay_results.csv", help="CSV path to save replay (pairs, rewards, errors).")
    parser.add_argument("--replay-h5", type=str, default="logs/offline_replay/replay_data.h5", help="HDF5 path to save replayed samples (same schema as offline data).")
    parser.add_argument("--overwrite-replay", action="store_true", help="Overwrite existing replay HDF5.")
    parser.add_argument("--write-threshold", type=int, default=512, help="HDF5 write threshold; flush after buffering this many samples.")
    parser.add_argument("--no-cache", action="store_true", help="Disable per-asset caching of offline data.")
    AppLauncher.add_app_launcher_args(parser)
    return parser
