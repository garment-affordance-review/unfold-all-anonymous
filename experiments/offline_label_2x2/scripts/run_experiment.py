#!/usr/bin/env python3
"""Unified entrypoint for the offline-label 2x2 experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed experiment inputs and run the offline-label 2x2 repeatability experiment."
    )
    parser.add_argument("--mode", choices=["generate", "reuse"], default="generate")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable used to invoke the Isaac runner.")

    parser.add_argument("--valid-assets-json", type=str, default="data/assets/cloth/valid_assets.json")
    parser.add_argument("--clothes-root", type=str, default="data/clothes/assets")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-assets", type=int, default=100)
    parser.add_argument("--asset-bins", type=int, default=10)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--pair-distance-bins", type=int, default=4)
    parser.add_argument("--pairs-per-bin", type=int, default=8)

    parser.add_argument("--protocol", type=str, default="all", choices=["all", "random_fling", "random_y", "cond_fling", "cond_y"])
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--repeats-per-pair", type=int, default=8)
    parser.add_argument("--rot-noise-deg", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="UnfoldAll-Cloth-Direct-v0")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/offline_label_2x2/configs/offline_label_2x2.yaml",
    )
    parser.add_argument("--relift-height-min", type=float, default=0.8)
    parser.add_argument("--relift-height-max", type=float, default=1.2)
    parser.add_argument("--relift-xy-jitter", type=float, default=0.05)
    parser.add_argument(
        "--flush-every-batches",
        type=int,
        default=8,
        help="Flush records/progress/summary to disk every N simulation batches.",
    )

    parser.add_argument(
        "--assets-manifest",
        type=str,
        default=None,
        help="Optional prebuilt asset manifest. Required in reuse mode.",
    )
    parser.add_argument(
        "--pairs-manifest",
        type=str,
        default=None,
        help="Optional prebuilt pair manifest. Required in reuse mode.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="pilot",
        help="Run directory name under experiments/offline_label_2x2/runs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Pass through headless mode to the Isaac runner.")
    return parser


def _default_assets_manifest(run_name: str, seed: int, num_assets: int) -> Path:
    return Path(f"experiments/offline_label_2x2/manifests/{run_name}_assets_{num_assets}_seed{seed}.json")


def _default_pairs_manifest(run_name: str, seed: int, num_assets: int, anchor_count: int, pair_distance_bins: int, pairs_per_bin: int) -> Path:
    return Path(
        "experiments/offline_label_2x2/manifests/"
        f"{run_name}_pairs_{num_assets}assets_{anchor_count}anchors_{pair_distance_bins}x{pairs_per_bin}_seed{seed}.json"
    )


def _run_subprocess(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _normalize_assets_manifest_for_asset_pool(src_path: Path, dst_path: Path) -> Path:
    payload = json.loads(src_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        normalized = payload
    elif isinstance(payload, dict):
        assets = payload.get("assets", [])
        normalized = [
            {
                "asset_id": int(item.get("asset_id", item.get("asset_index", idx))),
                "asset_path": str(item.get("asset_path", item.get("path", item.get("usd", "")))),
            }
            for idx, item in enumerate(assets)
        ]
    else:
        raise TypeError(f"Unsupported assets manifest format: {type(payload).__name__}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return dst_path


def _generate_manifests(args, run_dir: Path) -> tuple[Path, Path]:
    assets_manifest = Path(args.assets_manifest) if args.assets_manifest else _default_assets_manifest(args.run_name, args.seed, args.num_assets)
    pairs_manifest = (
        Path(args.pairs_manifest)
        if args.pairs_manifest
        else _default_pairs_manifest(
            args.run_name,
            args.seed,
            args.num_assets,
            args.anchor_count,
            args.pair_distance_bins,
            args.pairs_per_bin,
        )
    )
    assets_manifest.parent.mkdir(parents=True, exist_ok=True)
    pairs_manifest.parent.mkdir(parents=True, exist_ok=True)

    _run_subprocess(
        [
            sys.executable,
            "experiments/offline_label_2x2/scripts/sample_assets.py",
            "--valid-assets-json",
            args.valid_assets_json,
            "--clothes-root",
            args.clothes_root,
            "--num-assets",
            str(args.num_assets),
            "--num-bins",
            str(args.asset_bins),
            "--seed",
            str(args.seed),
            "--output",
            str(assets_manifest),
        ]
    )
    _run_subprocess(
        [
            sys.executable,
            "experiments/offline_label_2x2/scripts/build_eval_pairs.py",
            "--assets-manifest",
            str(assets_manifest),
            "--clothes-root",
            args.clothes_root,
            "--anchor-count",
            str(args.anchor_count),
            "--pair-distance-bins",
            str(args.pair_distance_bins),
            "--pairs-per-bin",
            str(args.pairs_per_bin),
            "--seed",
            str(args.seed),
            "--output",
            str(pairs_manifest),
        ]
    )
    return assets_manifest, pairs_manifest


def _resolve_manifests(args, run_dir: Path) -> tuple[Path, Path]:
    if args.mode == "generate":
        return _generate_manifests(args, run_dir)

    if not args.assets_manifest or not args.pairs_manifest:
        raise ValueError("reuse mode requires both --assets-manifest and --pairs-manifest")
    assets_manifest = Path(args.assets_manifest)
    pairs_manifest = Path(args.pairs_manifest)
    if not assets_manifest.exists():
        raise FileNotFoundError(f"Missing assets manifest: {assets_manifest}")
    if not pairs_manifest.exists():
        raise FileNotFoundError(f"Missing pairs manifest: {pairs_manifest}")
    return assets_manifest, pairs_manifest


def _write_run_config(run_dir: Path, args, assets_manifest: Path, pairs_manifest: Path, runner_assets_manifest: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "python_bin": args.python_bin,
        "seed": args.seed,
        "valid_assets_json": args.valid_assets_json,
        "clothes_root": args.clothes_root,
        "num_assets": args.num_assets,
        "asset_bins": args.asset_bins,
        "anchor_count": args.anchor_count,
        "pair_distance_bins": args.pair_distance_bins,
        "pairs_per_bin": args.pairs_per_bin,
        "protocol": args.protocol,
        "num_envs": args.num_envs,
        "repeats_per_pair": args.repeats_per_pair,
        "rot_noise_deg": args.rot_noise_deg,
        "device": args.device,
        "task": args.task,
        "config": args.config,
        "relift_height_min": args.relift_height_min,
        "relift_height_max": args.relift_height_max,
        "relift_xy_jitter": args.relift_xy_jitter,
        "flush_every_batches": args.flush_every_batches,
        "assets_manifest": str(assets_manifest),
        "runner_assets_manifest": str(runner_assets_manifest),
        "pairs_manifest": str(pairs_manifest),
    }
    (run_dir / "run_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args) -> None:
    run_dir = Path("experiments/offline_label_2x2/runs") / args.run_name
    vis_dir = run_dir / "visuals"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}")

    assets_manifest, pairs_manifest = _resolve_manifests(args, run_dir)
    runner_assets_manifest = _normalize_assets_manifest_for_asset_pool(
        assets_manifest,
        run_dir / "asset_pool_manifest.json",
    )
    _write_run_config(run_dir, args, assets_manifest, pairs_manifest, runner_assets_manifest)

    cmd = [
        args.python_bin,
        "experiments/offline_label_2x2/scripts/run_protocol_repeatability.py",
        "--task",
        args.task,
        "--config",
        args.config,
        "--device",
        args.device,
        "--assets-manifest",
        str(runner_assets_manifest),
        "--pairs-manifest",
        str(pairs_manifest),
        "--protocol",
        args.protocol,
        "--num-envs",
        str(args.num_envs),
        "--repeats-per-pair",
        str(args.repeats_per_pair),
        "--rot-noise-deg",
        str(args.rot_noise_deg),
        "--output-dir",
        str(run_dir),
        "--vis-dir",
        str(vis_dir),
        "--relift-height-min",
        str(args.relift_height_min),
        "--relift-height-max",
        str(args.relift_height_max),
        "--relift-xy-jitter",
        str(args.relift_xy_jitter),
        "--flush-every-batches",
        str(args.flush_every_batches),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.headless:
        cmd.append("--headless")
    _run_subprocess(cmd)


if __name__ == "__main__":
    run(build_parser().parse_args())
