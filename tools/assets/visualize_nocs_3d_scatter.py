#!/usr/bin/env python3
"""Visualize USD mesh vertices as a single-view 3D scatter with NOCS-like colors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_usd_points(usd_path: Path) -> np.ndarray:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    points_list: list[np.ndarray] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if points is None or len(points) == 0:
            continue
        points_list.append(np.asarray(points, dtype=np.float32))

    if not points_list:
        raise RuntimeError(f"No mesh points found in USD: {usd_path}")
    if len(points_list) == 1:
        return points_list[0]
    return np.concatenate(points_list, axis=0)


def _compute_vertex_nocs(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points.shape}")
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    denom = np.maximum(pmax - pmin, 1e-8)
    return np.clip((points - pmin[None, :]) / denom[None, :], 0.0, 1.0).astype(np.float32)


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    center = (pmin + pmax) * 0.5
    radius = float(np.max(pmax - pmin) * 0.5)
    if radius <= 1e-8:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _hide_axes(ax) -> None:
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.line.set_color((1, 1, 1, 0))
        except Exception:
            pass
        try:
            axis._axinfo["axisline"]["color"] = (1.0, 1.0, 1.0, 0.0)
            axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
            axis._axinfo["tick"]["color"] = (1.0, 1.0, 1.0, 0.0)
        except Exception:
            pass
    try:
        ax.set_frame_on(False)
    except Exception:
        pass


def _plot_single_view(ax, points: np.ndarray, colors: np.ndarray, elev: float, azim: float, point_size: float) -> None:
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=point_size,
        c=colors,
        alpha=0.98,
        linewidths=0,
        depthshade=False,
    )
    _set_equal_3d_axes(ax, points)
    ax.view_init(elev=elev, azim=azim)
    _hide_axes(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True, help="Path to *_obj.usd.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--point-size", type=float, default=10.0, help="Scatter point size.")
    parser.add_argument("--width", type=float, default=6.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=6.0, help="Figure height in inches.")
    parser.add_argument("--elev", type=float, default=90.0, help="Camera elevation.")
    parser.add_argument("--azim", type=float, default=-90.0, help="Camera azimuth. Default chooses +Z looking toward -Z.")
    args = parser.parse_args()

    points = _load_usd_points(args.usd)
    colors = _compute_vertex_nocs(points)

    fig = plt.figure(figsize=(args.width, args.height))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor((1, 1, 1, 0))
    _plot_single_view(
        ax,
        points=points,
        colors=colors,
        elev=args.elev,
        azim=args.azim,
        point_size=args.point_size,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, transparent=True)
    plt.close(fig)

    summary = {
        "usd": str(args.usd),
        "vertex_count": int(points.shape[0]),
        "camera": {
            "elev": float(args.elev),
            "azim": float(args.azim),
            "semantic": "camera at +Z looking toward -Z",
        },
        "output": str(args.output),
    }
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Wrote visualization to {args.output}")


if __name__ == "__main__":
    main()
