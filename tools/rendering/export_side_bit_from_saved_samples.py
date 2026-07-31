#!/usr/bin/env python3
"""Export side-bit visualizations from saved rendered samples."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _compute_side_from_sample(sample_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.load(sample_dir / "vertices.npy").astype(np.float32)
    face_vertex_ids = np.load(sample_dir / "face_vertex_ids.npy").astype(np.int64)
    face_index = np.load(sample_dir / "face_index.npy").astype(np.int64)
    cam = json.loads((sample_dir / "camera.json").read_text(encoding="utf-8"))
    w2c = np.asarray(cam["extrinsics"]["matrix"], dtype=np.float32)

    valid = face_index >= 0
    side_bit = np.full(face_index.shape, 255, dtype=np.uint8)
    side_rgb = np.zeros((*face_index.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return side_bit, side_rgb

    tri_vid = face_vertex_ids[valid]
    tri = vertices[tri_vid]
    edge_1 = tri[:, 1] - tri[:, 0]
    edge_2 = tri[:, 2] - tri[:, 0]
    normal_world = np.cross(edge_1, edge_2)
    normal_norm = np.linalg.norm(normal_world, axis=1, keepdims=True)
    good = normal_norm.squeeze(-1) > 1e-12
    normal_world[good] = normal_world[good] / normal_norm[good]

    rot = w2c[:3, :3]
    trans = w2c[:3, 3]
    center_cam = (tri.mean(axis=1) @ rot.T) + trans[None, :]
    normal_cam = normal_world @ rot.T
    to_camera_cam = -center_cam
    ray_norm = np.linalg.norm(to_camera_cam, axis=1, keepdims=True)
    good_ray = ray_norm.squeeze(-1) > 1e-12
    to_camera_cam[good_ray] = to_camera_cam[good_ray] / ray_norm[good_ray]

    facing = np.sum(normal_cam * to_camera_cam, axis=1)
    pix_side = (facing > 0.0).astype(np.uint8)
    side_bit[valid] = pix_side
    side_rgb[valid & (side_bit == 1)] = np.array([48, 208, 96], dtype=np.uint8)
    side_rgb[valid & (side_bit == 0)] = np.array([220, 76, 60], dtype=np.uint8)
    return side_bit, side_rgb


def _save_overlay(rgb: np.ndarray, side_rgb: np.ndarray, out_path: Path) -> None:
    overlay = rgb.astype(np.float32).copy()
    fg = np.any(side_rgb > 0, axis=-1)
    overlay[fg] = 0.55 * overlay[fg] + 0.45 * side_rgb[fg].astype(np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export side-bit visualizations from rendered sample folders.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--copy-rgb", action="store_true", default=True)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])[: int(args.max_samples)]
    index: list[dict] = []
    for sample_dir in sample_dirs:
        rgb_path = sample_dir / "rgb.png"
        if not rgb_path.exists():
            continue
        out_dir = output_root / sample_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        rgb = _load_rgb(rgb_path)
        side_bit, side_rgb = _compute_side_from_sample(sample_dir)
        np.save(out_dir / "side_bit.npy", side_bit.astype(np.uint8))
        Image.fromarray(side_rgb).save(out_dir / "side_bit.png")
        _save_overlay(rgb, side_rgb, out_dir / "side_overlay.png")
        if args.copy_rgb:
            shutil.copy2(rgb_path, out_dir / "rgb.png")
        meta = {"sample_id": sample_dir.name, "source_dir": str(sample_dir)}
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append(meta)

    (output_root / "index.json").write_text(
        json.dumps({"input_root": str(input_root), "num_samples": len(index), "samples": index}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
