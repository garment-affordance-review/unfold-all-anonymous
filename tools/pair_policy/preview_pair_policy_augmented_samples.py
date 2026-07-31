from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unfold.algorithms.supervision.projection import masked_softmax_heatmap
from unfold.workflows.pair_policy_train import _IMAGENET_MEAN, _IMAGENET_STD, _build_datasets, _load_yaml


def _argmax_xy(heat: np.ndarray, valid_mask: np.ndarray) -> np.ndarray | None:
    valid = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return None
    masked = np.where(valid, np.asarray(heat, dtype=np.float32), -np.inf)
    flat_idx = int(np.argmax(masked))
    y, x = np.unravel_index(flat_idx, masked.shape)
    return np.asarray([float(x), float(y)], dtype=np.float32)


def _display_heat(heat: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(heat, dtype=np.float32).copy()
    out[~np.asarray(valid_mask, dtype=bool)] = np.nan
    vmax = float(np.nanmax(out)) if np.any(np.isfinite(out)) else 0.0
    if vmax > 1e-12:
        out = out / vmax
    return out


def _draw_x(ax, xy: np.ndarray | None, *, color: str) -> None:
    if xy is None:
        return
    ax.scatter([xy[0]], [xy[1]], s=110, marker="x", c="black", linewidths=4, zorder=6)
    ax.scatter([xy[0]], [xy[1]], s=60, marker="x", c=color, linewidths=2.2, zorder=7)


def _draw_ring(ax, xy: np.ndarray | None, *, color: str) -> None:
    if xy is None:
        return
    ax.scatter([xy[0]], [xy[1]], s=130, facecolors="none", edgecolors="black", linewidths=3.5, zorder=6)
    ax.scatter([xy[0]], [xy[1]], s=82, facecolors="none", edgecolors=color, linewidths=2.0, zorder=7)


def _sample_preview_indices(length: int, num_samples: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    indices = list(range(length))
    rng.shuffle(indices)
    return indices[: min(num_samples, length)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/pair_policy/runs/debug/geometry_aug_preview"),
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config.resolve())
    ds_train, ds_val, _ = _build_datasets(cfg, args.out_dir.resolve())
    dataset = ds_train if args.split == "train" else ds_val

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = _sample_preview_indices(len(dataset), int(args.num_samples), int(args.seed))

    for preview_idx, ds_idx in enumerate(indices):
        item = dataset[ds_idx]

        image_np = item["image"].detach().cpu().permute(1, 2, 0).numpy()
        if getattr(dataset, "input_normalization", "none") == "imagenet":
            image_np = image_np * _IMAGENET_STD[None, None, :] + _IMAGENET_MEAN[None, None, :]
        image_np = np.clip(image_np, 0.0, 1.0)

        input_mask = item["input_mask"].detach().cpu().numpy()
        gt_mask = item["gt_mask"].detach().cpu().numpy()
        a1_valid = item["a1_target_mask"].detach().cpu().numpy() > 0.5
        a1_gt = item["a1_target"].detach().cpu().numpy()
        if dataset.target_name in {"masked_softmax", "topk_masked_softmax", "image_exp"}:
            a1_gt = masked_softmax_heatmap(np.log(np.clip(a1_gt, 1e-12, None)), a1_valid, tau=1.0)

        sampled_x1_xy = item["sampled_x1_xy"].detach().cpu().numpy()
        sampled_x1_valid = (sampled_x1_xy[:, 0] >= 0) & (sampled_x1_xy[:, 1] >= 0)
        query_idx = int(np.flatnonzero(sampled_x1_valid)[0]) if np.any(sampled_x1_valid) else 0
        x1_xy = sampled_x1_xy[query_idx] if np.any(sampled_x1_valid) else None
        a2_valid = item["a2_target_mask"][query_idx].detach().cpu().numpy() > 0.5
        a2_gt = item["a2_target"][query_idx].detach().cpu().numpy()
        if dataset.target_name in {"masked_softmax", "topk_masked_softmax", "image_exp"}:
            a2_gt = masked_softmax_heatmap(np.log(np.clip(a2_gt, 1e-12, None)), a2_valid, tau=1.0)

        a1_gt_xy = _argmax_xy(a1_gt, a1_valid)
        a2_gt_xy = _argmax_xy(a2_gt, a2_valid)
        a1_gt_vis = _display_heat(a1_gt, a1_valid)
        a2_gt_vis = _display_heat(a2_gt, a2_valid)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        value_label = "Exp Heatmap" if dataset.target_name == "image_exp" else "Heatmap"
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title("Input RGB")
        axes[0, 1].imshow(gt_mask, cmap="gray")
        axes[0, 1].set_title("GT Mask")
        axes[0, 2].imshow(input_mask, cmap="gray")
        axes[0, 2].set_title("Input Mask")
        axes[1, 0].imshow(a1_gt_vis, cmap="jet")
        axes[1, 0].set_title(f"A1 GT {value_label} (scaled)")
        axes[1, 1].imshow(a2_gt_vis, cmap="jet")
        axes[1, 1].set_title(f"A2 GT {value_label} (scaled)")
        axes[1, 2].imshow(image_np)
        axes[1, 2].imshow(input_mask, cmap="gray", alpha=0.25)
        axes[1, 2].set_title("RGB + Input Mask")
        for ax in axes.ravel():
            ax.axis("off")
        _draw_x(axes[1, 0], a1_gt_xy, color="magenta")
        _draw_ring(axes[1, 1], x1_xy, color="white")
        _draw_x(axes[1, 1], a2_gt_xy, color="yellow")
        fig.tight_layout()
        fig.savefig(out_dir / f"{preview_idx:02d}_idx{ds_idx}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"[INFO] saved {len(indices)} previews to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
