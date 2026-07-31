#!/usr/bin/env python3
"""
Scale all coord.npy files by a constant factor.

Usage (example):
    python tools/assets/preprocess_scale_coords.py \\
        --src data/clothes/assets \\
        --dst data/clothes_scaled/assets \\
        --scale 0.0085

By default writes to a new dst tree, mirroring src structure and copying
other *.npy files untouched. Use --inplace to overwrite coord.npy in src.
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Scale coord.npy for all assets.")
    parser.add_argument("--src", type=Path, required=True, help="Source assets root (contains asset_xxxx folders).")
    parser.add_argument("--dst", type=Path, help="Destination root. If omitted and --inplace is set, modifies src.")
    parser.add_argument("--scale", type=float, required=True, help="Scale factor to multiply coordinates.")
    parser.add_argument(
        "--center",
        action="store_true",
        help="After scaling, subtract each asset's bbox center ( (min+max)/2 ) to center coords.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite coord.npy in src. If set, --dst is ignored and backup can be made with --backup-suffix.",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".bak",
        help="When --inplace, save original coord.npy as coord.npy<suffix> before overwriting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_root: Path = args.src

    if not src_root.exists():
        raise FileNotFoundError(f"src path not found: {src_root}")

    if args.inplace:
        dst_root = src_root
    else:
        if args.dst is None:
            raise ValueError("Must provide --dst when not using --inplace.")
        dst_root: Path = args.dst
        dst_root.mkdir(parents=True, exist_ok=True)

    asset_dirs = sorted([p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("asset_")])
    if not asset_dirs:
        raise RuntimeError(f"No asset_* directories found under {src_root}")

    for asset_dir in tqdm(asset_dirs, desc="Scaling coords", unit="asset"):
        coord_path = asset_dir / "coord.npy"
        if not coord_path.exists():
            continue

        # Decide output paths
        if args.inplace:
            out_coord = coord_path
            if args.backup_suffix:
                backup = coord_path.with_name(coord_path.name + args.backup_suffix)
                if not backup.exists():
                    shutil.copy2(coord_path, backup)
        else:
            # replicate folder
            rel = asset_dir.relative_to(src_root)
            out_dir = dst_root / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            out_coord = out_dir / "coord.npy"
            # copy other files
            for f in asset_dir.iterdir():
                if f.name == "coord.npy":
                    continue
                dst_f = out_dir / f.name
                if not dst_f.exists():
                    if f.is_file():
                        shutil.copy2(f, dst_f)

        coord = np.load(coord_path)
        coord = coord.astype(np.float32) * args.scale
        if args.center:
            mn = coord.min(axis=0)
            mx = coord.max(axis=0)
            center = (mn + mx) / 2.0
            coord = coord - center
        np.save(out_coord, coord)


if __name__ == "__main__":
    main()
