from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_asset_path(path: str) -> str:
    s = str(path).replace("\\", "/")
    marker = "/cloth/"
    if marker in s:
        s = s.split(marker, 1)[1]
    return s.lstrip("./")


def _backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{stamp}")
    path.rename(backup)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_records(
    *,
    asset_specs: list[dict[str, Any]],
    assets_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in asset_specs:
        asset_id = int(spec["asset_id"])
        asset_rel = str(spec["asset_path"])
        asset_dir = assets_root / f"asset_{asset_id:04d}"
        if not asset_dir.exists():
            raise FileNotFoundError(f"Missing asset directory for asset_id={asset_id}: {asset_dir}")
        records.append(
            {
                "asset_dir": asset_dir.name,
                "asset_id": asset_id,
                "asset_path": str((assets_root.parent.parent / "assets" / "cloth" / asset_rel).resolve()),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild cloth asset_index jsonl files from manifests.")
    parser.add_argument("--cloth-root", type=str, default="data/clothes")
    parser.add_argument("--valid-assets-json", type=str, default="data/assets/cloth/valid_assets.json")
    parser.add_argument("--backup", action="store_true", help="Backup existing jsonl files before overwriting.")
    args = parser.parse_args()

    cloth_root = Path(args.cloth_root).resolve()
    assets_root = cloth_root / "assets"
    manifest_dir = cloth_root / "manifests"
    valid_assets = [str(x) for x in _load_json(Path(args.valid_assets_json).resolve())]

    worker_paths = sorted(manifest_dir.glob("worker_*.json"))
    if not worker_paths:
        raise FileNotFoundError(f"No worker manifests found under {manifest_dir}")

    all_records_by_id: dict[int, dict[str, Any]] = {}
    worker_outputs: list[tuple[Path, list[dict[str, Any]]]] = []
    for worker_path in worker_paths:
        specs = _load_json(worker_path)
        if not isinstance(specs, list):
            raise ValueError(f"Worker manifest must be a list: {worker_path}")
        records = _build_records(asset_specs=specs, assets_root=assets_root)
        worker_outputs.append((cloth_root / f"asset_index.{worker_path.stem}.jsonl", records))
        for record in records:
            asset_id = int(record["asset_id"])
            prev = all_records_by_id.get(asset_id)
            if prev is not None and prev != record:
                raise ValueError(f"Duplicate asset_id with mismatched records: {asset_id}")
            all_records_by_id[asset_id] = record

    merged = [all_records_by_id[idx] for idx in sorted(all_records_by_id)]
    if len(merged) != len(valid_assets):
        raise ValueError(
            f"Merged asset count mismatch: merged={len(merged)} valid_assets={len(valid_assets)}"
        )

    merged_norm = [_normalize_asset_path(r["asset_path"]) for r in merged]
    if merged_norm != valid_assets:
        mismatches = [
            (i, valid_assets[i], merged_norm[i])
            for i in range(min(len(valid_assets), len(merged_norm)))
            if valid_assets[i] != merged_norm[i]
        ]
        preview = mismatches[:5]
        raise ValueError(f"Merged asset order does not match valid_assets.json; first mismatches: {preview}")

    output_paths = [cloth_root / "asset_index.jsonl", *[path for path, _ in worker_outputs]]
    if args.backup:
        for path in output_paths:
            _backup_if_exists(path)

    _write_jsonl(cloth_root / "asset_index.jsonl", merged)
    for path, records in worker_outputs:
        _write_jsonl(path, records)

    print(
        json.dumps(
            {
                "cloth_root": str(cloth_root),
                "merged_count": len(merged),
                "worker_counts": {path.name: len(records) for path, records in worker_outputs},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
