from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterable, Optional
import matplotlib.pyplot as plt
import matplotlib


def save_hist_png(values: Iterable[float], out_path: Path, bins: int = 80, rmin: float = -25, rmax: float = 0, title: Optional[str] = None, xlabel: str = "Reward"):
    # Simple histogram plotter using Matplotlib
    vals = np.fromiter(values, dtype=float)
    if vals.size == 0:
        return
    
    # Auto-range if not specified
    if rmin is None: rmin = vals.min()
    if rmax is None: rmax = vals.max()
    
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(8, 5.2))
    
    # Histogram
    n, bins, patches = plt.hist(vals, bins=bins, range=(rmin, rmax), color=(0.12, 0.47, 0.71) if rmin < 0 else (0.17, 0.63, 0.17), edgecolor='none')
    
    plt.title(title if title else "")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def save_scatter_png(
    ref_points: np.ndarray, 
    other_points: np.ndarray, 
    out_path: Path, 
    title: str = "Scatter Plot", 
    ref_label: str = "Ref", 
    other_label: str = "Other",
    distance: Optional[float] = None,
):
    """
    Save 2D scatter plot of two point sets using Matplotlib.
    Expects (N, 2) or (N, 3) arrays - if 3D, ignores Z.
    """
    if ref_points.ndim > 1 and ref_points.shape[1] >= 2:
        ref_xy = ref_points[:, :2]
    else:
        return
        
    if other_points.ndim > 1 and other_points.shape[1] >= 2:
        other_xy = other_points[:, :2]
    else:
        return

    matplotlib.use('Agg')
    fig = plt.figure(figsize=(7, 7))
    
    # Plot points with small marker size for clarity
    plt.scatter(other_xy[:, 0], other_xy[:, 1], c='blue', s=1, label=other_label, alpha=0.6)
    plt.scatter(ref_xy[:, 0], ref_xy[:, 1], c='red', s=1, label=ref_label, alpha=0.6)
    
    # Formatting
    title_text = title
    if distance is not None:
        title_text = f"{title} | Dist: {distance:.4f}"
    
    plt.title(title_text)
    plt.legend()
    plt.axis('equal')  # Ensure aspect ratio is correct
    plt.grid(True, alpha=0.2)
    
    # Remove axis ticks for cleaner look (optional, but requested style is "debug plot")
    # plt.xticks([])
    # plt.yticks([])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)

