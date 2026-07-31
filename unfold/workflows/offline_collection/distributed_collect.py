#!/usr/bin/env python3
"""Lightweight distributed launcher for offline collection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from unfold.platform.assets import load_assets_from_json, resolve_assets_root


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed offline collection launcher.")
    parser.add_argument("--config", type=str, default="configs/offline_pair_conditioned_distributed.yaml", help="YAML config path.")
    parser.add_argument("--overwrite", action="store_true", help="Forward overwrite to child workers.")
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.full_load(f)


def _resolve_assets_and_list(cfg_dict: dict[str, Any], config_path: Path) -> tuple[Path, list[str]]:
    project_root = config_path.parent.parent
    assets_root = resolve_assets_root(project_root, cfg_dict.get("assets_root"))
    garment_root = assets_root / "cloth"
    valid_json = garment_root / "valid_assets.json"
    assets = load_assets_from_json(valid_json)
    categories = set(cfg_dict.get("garment_categories") or [])
    if categories:
        assets = [p for p in assets if str(p).split("/")[0] in categories]
    return garment_root, assets


def _interleaved_split(records: list[dict[str, Any]], num_workers: int) -> list[list[dict[str, Any]]]:
    shards = [[] for _ in range(num_workers)]
    for idx, record in enumerate(records):
        shards[idx % num_workers].append(record)
    return shards


def _write_worker_manifest(manifest_dir: Path, worker_id: int, records: list[dict[str, Any]]) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"worker_{worker_id}.json"
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _merge_pair_conditioned_metadata(output_root: Path, num_workers: int) -> None:
    index_records: dict[int, dict[str, Any]] = {}
    failed_records: list[dict[str, Any]] = []
    for worker_id in range(num_workers):
        index_path = output_root / f"asset_index.worker_{worker_id}.jsonl"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    asset_id = int(record["asset_id"])
                    if asset_id in index_records:
                        raise RuntimeError(f"Duplicate asset_id during merge: {asset_id}")
                    index_records[asset_id] = record
        fail_path = output_root / f"failed_assets.worker_{worker_id}.json"
        if fail_path.exists():
            failed_records.extend(json.loads(fail_path.read_text(encoding="utf-8")))

    merged_index_path = output_root / "asset_index.jsonl"
    with merged_index_path.open("w", encoding="utf-8") as f:
        for asset_id in sorted(index_records.keys()):
            f.write(json.dumps(index_records[asset_id], ensure_ascii=False) + "\n")

    merged_fail_path = output_root / "failed_assets.json"
    merged_fail_path.write_text(json.dumps(failed_records, ensure_ascii=False, indent=2), encoding="utf-8")


def _pair_output_root(cfg_dict: dict[str, Any]) -> Path:
    return (PROJECT_ROOT / cfg_dict["pair_conditioned_collection"]["output_root"]).resolve()


def _standard_output_root() -> Path:
    return (PROJECT_ROOT / "logs" / "offline_collection_distributed").resolve()


def _stream_worker_output(worker_id: int, proc: subprocess.Popen[str], log_path: Path) -> None:
    assert proc.stdout is not None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(
            f"\n===== worker={worker_id} pid={proc.pid} started_at={time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
        )
        log_f.flush()
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            stripped = line.rstrip()
            if not stripped:
                continue
            if stripped.startswith(("[SAVE]", "[FAIL]", "[MONITOR]", "[SKIP]", "[INFO]")):
                print(f"[W{worker_id}] {stripped}", flush=True)


def run(args: argparse.Namespace) -> None:
    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    cfg_dict = _load_yaml(config_path)
    dcfg = cfg_dict.get("distributed_collection") or {}
    gpu_ids = list(dcfg.get("gpu_ids") or [])
    if not gpu_ids:
        raise ValueError("distributed_collection.gpu_ids is empty")
    num_envs_per_gpu = int(dcfg.get("num_envs_per_gpu", 32))
    entrypoint = str(dcfg.get("entrypoint", "offline_pair_conditioned"))
    if entrypoint not in {"offline_standard", "offline_pair_conditioned"}:
        raise ValueError(f"Unsupported distributed_collection.entrypoint: {entrypoint}")

    garment_root, assets = _resolve_assets_and_list(cfg_dict, config_path)
    records = [{"asset_id": idx, "asset_path": path} for idx, path in enumerate(assets)]
    worker_records = _interleaved_split(records, len(gpu_ids))

    if entrypoint == "offline_pair_conditioned":
        output_root = _pair_output_root(cfg_dict)
        app_path = PROJECT_ROOT / "apps" / "collect_pair_conditioned_offline_dataset.py"
    else:
        output_root = _standard_output_root()
        app_path = PROJECT_ROOT / "apps" / "collect_offline_dataset.py"

    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[DIST] entrypoint={entrypoint} workers={len(gpu_ids)} assets={len(records)} "
        f"manifest_dir={manifest_dir}",
        flush=True,
    )

    procs: list[tuple[int, subprocess.Popen[str], threading.Thread]] = []
    start_time = time.time()
    for worker_id, gpu_id in enumerate(gpu_ids):
        manifest_path = _write_worker_manifest(manifest_dir, worker_id, worker_records[worker_id])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("DISPLAY", None)

        cmd = [
            sys.executable,
            str(app_path),
            "--headless",
            "--config",
            str(config_path),
            "--num-envs",
            str(num_envs_per_gpu),
            "--assets-manifest",
            str(manifest_path),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if entrypoint == "offline_pair_conditioned":
            cmd.extend(["--worker-id", str(worker_id)])
        else:
            worker_dir = output_root / f"worker_{worker_id}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(
                [
                    "--output",
                    str(worker_dir / "offline_data.h5"),
                    "--vis-dir",
                    str(worker_dir / "visuals"),
                ]
            )

        log_path = logs_dir / f"worker_{worker_id}.log"
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(
            target=_stream_worker_output,
            args=(worker_id, proc, log_path),
            daemon=True,
        )
        thread.start()
        print(
            f"[DIST] worker={worker_id} gpu={gpu_id} pid={proc.pid} assets={len(worker_records[worker_id])} "
            f"manifest={manifest_path} log={log_path}",
            flush=True,
        )
        procs.append((worker_id, proc, thread))

    failed_workers = 0
    for worker_id, proc, thread in procs:
        ret = proc.wait()
        thread.join(timeout=5.0)
        if ret != 0:
            failed_workers += 1
        print(f"[DIST] worker={worker_id} exit_code={ret}", flush=True)

    if entrypoint == "offline_pair_conditioned":
        _merge_pair_conditioned_metadata(output_root, len(gpu_ids))

    elapsed = time.time() - start_time
    print(
        f"[DIST] completed workers={len(gpu_ids)} failed_workers={failed_workers} "
        f"elapsed={elapsed / 3600.0:.2f}h",
        flush=True,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)
