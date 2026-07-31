from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _asset_ordered_split(asset_ids: list[str], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    ids = sorted(set(asset_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = max(1, int(round(len(ids) * float(val_ratio)))) if ids else 0
    val_ids = set(ids[:val_count])
    train_ids = set(ids[val_count:])
    return train_ids, val_ids


def _sample_asset_id(sample: dict[str, Any]) -> str:
    if sample.get("teacher_asset_id") is not None:
        return str(sample["teacher_asset_id"])
    if sample.get("asset_id") is not None:
        return str(sample["asset_id"])
    raise ValueError("pair-policy supervision index sample missing teacher_asset_id/asset_id")


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    # Handle numpy / torch scalar-like objects without importing those packages here.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _to_json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_pair_policy_index(
    *,
    supervision_index_path: str | Path,
    out_path: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    rebuild: bool = False,
    progress_every: int = 1000,
) -> dict[str, Any]:
    out_path = Path(out_path).resolve()
    if out_path.exists() and not rebuild:
        print(f"[INFO] Reusing pair-policy index cache: {out_path}", flush=True)
        return json.loads(out_path.read_text(encoding="utf-8"))

    sup_index = Path(supervision_index_path).resolve()
    obj = json.loads(sup_index.read_text(encoding="utf-8"))
    samples = obj.get("samples", [])
    print(
        f"[INFO] Building pair-policy index from supervision index: {sup_index} "
        f"(samples={len(samples)})",
        flush=True,
    )
    asset_ids = [_sample_asset_id(sample) for sample in samples]
    _, val_assets = _asset_ordered_split(asset_ids, val_ratio=val_ratio, seed=seed)

    rows: list[dict[str, Any]] = []
    every = max(1, int(progress_every))
    total = len(samples)
    for idx, sample in enumerate(samples, start=1):
        sup_meta = sample
        precomputed_a1_std = sup_meta.get("a1_std")
        precomputed_a1_top1_margin = sup_meta.get("a1_top1_margin")
        precomputed_reward_row_margin = sup_meta.get("reward_row_top1_margin")
        if (
            precomputed_a1_std is None
            or precomputed_a1_top1_margin is None
            or precomputed_reward_row_margin is None
        ):
            raise ValueError(
                "pair-policy supervision index must contain precomputed stats: "
                "a1_std, a1_top1_margin, reward_row_top1_margin"
            )
        if not sup_meta.get("shard_path"):
            raise ValueError(
                "pair-policy supervision index must contain shard_path/shard_index for direct training"
            )
        a1_std = float(precomputed_a1_std)
        a1_top1_margin = float(precomputed_a1_top1_margin)
        reward_row_top1_margin = float(precomputed_reward_row_margin)
        row = {
            "sample_id": str(sup_meta["sample_id"]),
            "asset_id": _sample_asset_id(sup_meta),
            "asset_path": str(sup_meta.get("asset_path", "")),
            "vertex_index_map": (
                str(Path(str(sup_meta["vertex_index_map"])).resolve())
                if sup_meta.get("vertex_index_map")
                else ""
            ),
            "rgb_path": (
                str(Path(str(sup_meta["source"]["rgb"])).resolve())
                if isinstance(sup_meta.get("source"), dict) and sup_meta["source"].get("rgb")
                else ""
            ),
            "mask_path": (
                str(Path(str(sup_meta["source"]["mask"])).resolve())
                if isinstance(sup_meta.get("source"), dict) and sup_meta["source"].get("mask")
                else ""
            ),
            "face_index_path": (
                str(Path(str(sup_meta["source"]["face_index"])).resolve())
                if isinstance(sup_meta.get("source"), dict) and sup_meta["source"].get("face_index")
                else ""
            ),
            "face_vertex_ids_path": (
                str(Path(str(sup_meta["source"]["face_vertex_ids"])).resolve())
                if isinstance(sup_meta.get("source"), dict) and sup_meta["source"].get("face_vertex_ids")
                else ""
            ),
            "barycentric_weights_path": (
                str(Path(str(sup_meta["source"]["barycentric_weights"])).resolve())
                if isinstance(sup_meta.get("source"), dict) and sup_meta["source"].get("barycentric_weights")
                else ""
            ),
            "num_candidates": int(sup_meta.get("num_candidates", -1)),
            "a1_std": a1_std,
            "a1_top1_margin": a1_top1_margin,
            "reward_row_top1_margin": reward_row_top1_margin,
        }
        row["shard_path"] = str(Path(sup_meta["shard_path"]).resolve())
        row["shard_index"] = int(sup_meta["shard_index"])
        precomputed_split = sup_meta.get("split")
        if precomputed_split in {"train", "val"}:
            row["split"] = str(precomputed_split)
        else:
            row["split"] = "val" if row["asset_id"] in val_assets else "train"
        rows.append(row)
        if idx % every == 0 or idx == total:
            print(
                f"[INFO] pair-policy index progress: {idx}/{total} "
                f"({100.0 * idx / max(total, 1):.1f}%)",
                flush=True,
            )

    payload = {
        "supervision_index": str(sup_index),
        "num_samples": len(rows),
        "num_train": sum(1 for row in rows if row["split"] == "train"),
        "num_val": sum(1 for row in rows if row["split"] == "val"),
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "rows": rows,
    }
    payload = _to_json_safe(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote pair-policy index cache: {out_path}", flush=True)
    return payload
