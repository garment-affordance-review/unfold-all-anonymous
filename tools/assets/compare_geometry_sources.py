#!/usr/bin/env python3
"""Compare USD mesh vertices, init_pos.npy, and coord.npy for one asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from isaaclab.app import AppLauncher


def _load_usd_points(usd_path: Path) -> np.ndarray:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            if points and len(points) > 0:
                return np.asarray(points, dtype=np.float32)
    raise RuntimeError(f"No mesh points found in USD: {usd_path}")


def _stats(points: np.ndarray) -> dict[str, object]:
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    center = points.mean(axis=0)
    extent = pmax - pmin
    return {
        "count": int(points.shape[0]),
        "center": center.round(6).tolist(),
        "bbox_min": pmin.round(6).tolist(),
        "bbox_max": pmax.round(6).tolist(),
        "extent": extent.round(6).tolist(),
        "diag": float(np.linalg.norm(extent)),
    }


def _plot_views(ax_xy, ax_xz, points: np.ndarray, title: str, color: str) -> None:
    ax_xy.scatter(points[:, 0], points[:, 1], s=4, c=color, alpha=0.8, linewidths=0)
    ax_xy.set_title(f"{title} (XY)")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.2)

    ax_xz.scatter(points[:, 0], points[:, 2], s=4, c=color, alpha=0.8, linewidths=0)
    ax_xz.set_title(f"{title} (XZ)")
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.grid(True, alpha=0.2)


def _plot_overlay(out_path: Path, usd_points: np.ndarray, init_points: np.ndarray, coord_points: np.ndarray) -> None:
    usd_centered = usd_points - usd_points.mean(axis=0, keepdims=True)
    init_centered = init_points - init_points.mean(axis=0, keepdims=True)
    coord_centered = coord_points - coord_points.mean(axis=0, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(usd_centered[:, 0], usd_centered[:, 1], s=5, c="#1f77b4", alpha=0.45, label="USD")
    axes[0].scatter(init_centered[:, 0], init_centered[:, 1], s=5, c="#ff7f0e", alpha=0.45, label="init_pos")
    axes[0].scatter(coord_centered[:, 0], coord_centered[:, 1], s=7, c="#2ca02c", alpha=0.8, label="coord")
    axes[0].set_title("Centered Overlay (XY)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="best")

    axes[1].scatter(usd_centered[:, 0], usd_centered[:, 2], s=5, c="#1f77b4", alpha=0.45, label="USD")
    axes[1].scatter(init_centered[:, 0], init_centered[:, 2], s=5, c="#ff7f0e", alpha=0.45, label="init_pos")
    axes[1].scatter(coord_centered[:, 0], coord_centered[:, 2], s=7, c="#2ca02c", alpha=0.8, label="coord")
    axes[1].set_title("Centered Overlay (XZ)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="best")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _normalize_shape(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    extent = centered.max(axis=0) - centered.min(axis=0)
    diag = float(np.linalg.norm(extent))
    if diag <= 1e-8:
        return centered
    return centered / diag


def _plot_normalized_overlay(out_path: Path, usd_points: np.ndarray, init_points: np.ndarray, coord_points: np.ndarray) -> None:
    usd_norm = _normalize_shape(usd_points)
    init_norm = _normalize_shape(init_points)
    coord_norm = _normalize_shape(coord_points)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(usd_norm[:, 0], usd_norm[:, 1], s=5, c="#1f77b4", alpha=0.45, label="USD")
    axes[0].scatter(init_norm[:, 0], init_norm[:, 1], s=5, c="#ff7f0e", alpha=0.45, label="init_pos")
    axes[0].scatter(coord_norm[:, 0], coord_norm[:, 1], s=7, c="#2ca02c", alpha=0.8, label="coord")
    axes[0].set_title("Centered + Diag-Normalized Overlay (XY)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="best")

    axes[1].scatter(usd_norm[:, 0], usd_norm[:, 2], s=5, c="#1f77b4", alpha=0.45, label="USD")
    axes[1].scatter(init_norm[:, 0], init_norm[:, 2], s=5, c="#ff7f0e", alpha=0.45, label="init_pos")
    axes[1].scatter(coord_norm[:, 0], coord_norm[:, 2], s=7, c="#2ca02c", alpha=0.8, label="coord")
    axes[1].set_title("Centered + Diag-Normalized Overlay (XZ)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="best")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True, help="Path to *_obj.usd asset.")
    parser.add_argument("--pointcept-asset", type=Path, required=True, help="Path to output asset_xxxx directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to save comparison outputs.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    app = AppLauncher(headless=True).app
    usd_points = _load_usd_points(args.usd)

    init_path = args.usd.parent / "init_pos.npy"
    coord_path = args.pointcept_asset / "coord.npy"
    if not init_path.exists():
        raise FileNotFoundError(f"init_pos.npy not found: {init_path}")
    if not coord_path.exists():
        raise FileNotFoundError(f"coord.npy not found: {coord_path}")

    init_points = np.load(init_path).astype(np.float32)
    coord_points = np.load(coord_path).astype(np.float32)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)
    _plot_views(axes[0, 0], axes[0, 1], usd_points, "USD Mesh Points", "#1f77b4")
    _plot_views(axes[1, 0], axes[1, 1], init_points, "init_pos.npy", "#ff7f0e")
    _plot_views(axes[2, 0], axes[2, 1], coord_points, "coord.npy", "#2ca02c")
    fig.savefig(out_dir / "geometry_sources.png", dpi=180)
    plt.close(fig)

    _plot_overlay(out_dir / "geometry_sources_overlay.png", usd_points, init_points, coord_points)
    _plot_normalized_overlay(
        out_dir / "geometry_sources_overlay_normalized.png",
        usd_points,
        init_points,
        coord_points,
    )

    summary = {
        "usd": str(args.usd),
        "pointcept_asset": str(args.pointcept_asset),
        "stats": {
            "usd_mesh": _stats(usd_points),
            "init_pos": _stats(init_points),
            "coord": _stats(coord_points),
        },
    }
    with (out_dir / "geometry_sources_stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Wrote comparison to {out_dir}")
    app.close()


if __name__ == "__main__":
    main()
