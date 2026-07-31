#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from unfold.algorithms.supervision.visualize import save_supervision_visuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize pair-policy supervision npz.")
    parser.add_argument("--supervision", type=str, required=True, help="Path to supervision.npz")
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB image")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for visualizations")
    args = parser.parse_args()

    sup = np.load(args.supervision, allow_pickle=False)
    candidate_xy = sup["candidate_xy"]
    a1_logits = sup["a1_logits"]
    reward_matrix = sup["reward_matrix"]
    mask_path = str(sup["mask_path"][0]) if "mask_path" in sup else args.rgb

    out_files = save_supervision_visuals(
        rgb_path=args.rgb,
        mask_path=mask_path,
        out_dir=args.out_dir,
        candidate_xy=candidate_xy,
        a1_logits=a1_logits,
        reward_matrix=reward_matrix,
    )
    print(f"[INFO] Wrote {len(out_files)} visualization files to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
