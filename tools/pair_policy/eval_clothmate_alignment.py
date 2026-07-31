#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import imutils
import numpy as np
import skimage.transform as st
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOTHMATE_ROOT = Path("${WORKSPACE_ROOT}/clothmate")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CLOTHMATE_ROOT))

from clothmate.core.network import MaximumValuePolicy
from tools.pair_policy.artf_eval_common import (
    append_prediction_row,
    coco_segmentation_to_mask,
    crop_component_for_inference,
    draw_vis,
    iter_artf_samples,
    map_xy,
    summarize_rows,
    touches_border,
    write_rows_csv,
    write_split_plots,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ClothMate fling predictions against aRTF keypoints.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("${DATASET_ROOT}/aRTF-Clothes-dataset/extracted/aRTFClothes"),
    )
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--categories", type=str, default="shorts,tshirts,towels")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("${WORKSPACE_ROOT}/clothmate/models/latest_ckpt.pth"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "experiments/aRTFClothes/analysis/clothmate_keypoint_eval_full",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-rotations", type=int, default=17)
    parser.add_argument("--scales", type=str, default="1.0,1.3,1.5,2.0,2.5,3.0,3.5")
    parser.add_argument("--deformable-weight", type=float, default=0.7)
    parser.add_argument("--obs-dim", type=int, default=128)
    parser.add_argument("--pix-grasp-dist", type=int, default=16)
    parser.add_argument("--target-mask-area-ratio", type=float, default=0.145)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all.")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--vis-limit", type=int, default=12, help="Max number of visualization samples per split.")
    return parser.parse_args()


def parse_scales(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def get_center_affine(img_shape: tuple[int, int], rotation: float, scale: float) -> st.AffineTransform:
    offset = np.array(img_shape[::-1], dtype=np.float32) / 2.0
    pre = st.AffineTransform(translation=-offset).params
    aff = st.AffineTransform(rotation=rotation, scale=scale).params
    post = st.AffineTransform(translation=offset).params
    return st.AffineTransform(matrix=(post @ aff @ pre).astype(np.float32))


def transform_points(pts: np.ndarray, tx_new_old: np.ndarray) -> np.ndarray:
    dim = pts.shape[-1]
    return pts @ tx_new_old[:dim, :dim].T + tx_new_old[:dim, dim]


class OfflineImageStackTransformer:
    def __init__(self, img_shape: tuple[int, int], rotations: np.ndarray, scales: np.ndarray):
        self.shape = (len(rotations) * len(scales),) + tuple(img_shape)
        self.transform_tuples = [(rot, scale) for rot in rotations for scale in scales]
        self.transforms = [get_center_affine(img_shape=img_shape, rotation=rot, scale=scale) for rot, scale in self.transform_tuples]

    def forward_raw(self, raw: np.ndarray) -> np.ndarray:
        img_shape = self.shape[1:]
        if raw.ndim == 2:
            raw = raw[..., None]
        stack = np.empty((len(self.transforms),) + img_shape + raw.shape[2:], dtype=raw.dtype)
        order = 0 if raw.dtype == bool else 1
        for i, tf in enumerate(self.transforms):
            warped = st.warp(raw, tf.inverse, order=order, output_shape=img_shape, preserve_range=True)
            if raw.dtype == bool:
                stack[i] = warped > 0.5
            else:
                stack[i] = warped.astype(raw.dtype)
        return stack

    def get_inverse_coord_map(self) -> np.ndarray:
        identity_map = np.moveaxis(np.indices(self.shape[1:], dtype=np.float32)[::-1], 0, -1)
        maps = []
        for tf in self.transforms:
            tx = np.linalg.inv(tf.params)
            maps.append(transform_points(identity_map.reshape(-1, 2), tx).reshape(identity_map.shape))
        return np.stack(maps).astype(np.float32)


def generate_coordinate_map(dim: int, rotation_deg: float, scale: float) -> np.ndarray:
    """Match ClothMate's generate_coordinate_map() implementation."""
    max_scale = 5.0
    scale = 1.0
    coordinate_dim = int(dim * (max_scale / scale))
    x, y = np.meshgrid(
        max_scale * np.linspace(-1.0, 1.0, coordinate_dim, dtype=np.float32),
        max_scale * np.linspace(-1.0, 1.0, coordinate_dim, dtype=np.float32),
        indexing="ij",
    )
    x = x.reshape(coordinate_dim, coordinate_dim, 1)
    y = y.reshape(coordinate_dim, coordinate_dim, 1)
    xy = np.concatenate((x, y), axis=2)
    xy = imutils.rotate(xy, rotation_deg)
    center = coordinate_dim / 2.0
    new_dim = int(center + center * scale / max_scale) - int(center - center * scale / max_scale)
    offset = 0
    if int(new_dim) != dim:
        offset = int(dim - new_dim)
    start = int(center - center * scale / max_scale)
    end = int(center + center * scale / max_scale) + offset
    return xy[start:end, start:end, :].astype(np.float32)


def crop_keep_height_hcenter(
    rgb: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    out_h, out_w = rgb.shape[:2]
    if out_w == out_h:
        return rgb, mask, {"src_x0": 0, "src_y0": 0, "canvas_w": out_w, "canvas_h": out_h, "pad_left": 0, "pad_top": 0}

    target_w = out_h
    ys, xs = np.nonzero(mask > 0)
    center_x = out_w // 2 if xs.size == 0 else int(round(xs.mean()))

    if out_w >= target_w:
        left = max(0, min(center_x - target_w // 2, out_w - target_w))
        right = left + target_w
        return (
            rgb[:, left:right],
            mask[:, left:right],
            {"src_x0": left, "src_y0": 0, "canvas_w": target_w, "canvas_h": out_h, "pad_left": 0, "pad_top": 0},
        )

    canvas_rgb = np.zeros((out_h, target_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((out_h, target_w), dtype=mask.dtype)
    pad_left = max(0, min(target_w - out_w, target_w // 2 - center_x))
    canvas_rgb[:, pad_left:pad_left + out_w] = rgb
    canvas_mask[:, pad_left:pad_left + out_w] = mask
    return (
        canvas_rgb,
        canvas_mask,
        {"src_x0": 0, "src_y0": 0, "canvas_w": target_w, "canvas_h": out_h, "pad_left": pad_left, "pad_top": 0},
    )


def get_offset_stack(stack: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    value = np.nan
    if stack.dtype == np.dtype("bool"):
        value = False
    up_stack = np.full(stack.shape, value, dtype=stack.dtype)
    down_stack = np.full(stack.shape, value, dtype=stack.dtype)
    up_stack[:, offset:, ...] = stack[:, :-offset, ...]
    down_stack[:, :-offset, ...] = stack[:, offset:, ...]
    return up_stack, down_stack


def build_valid_transforms_mask(
    rotations: np.ndarray,
    base_scales: np.ndarray,
    obs_dim: int,
) -> np.ndarray:
    """Mirror ClothMate's transform-subset gating in get_pick_and_fling_input()."""
    rotation_indices = np.logical_and(rotations >= -np.pi / 4.0, rotations <= np.pi / 4.0)
    scale_indices = base_scales >= 1.4
    one_d_mask = np.logical_and(rotation_indices[:, None], scale_indices[None, :]).flatten()
    return np.tile(one_d_mask[:, None, None], (1, obs_dim, obs_dim))


def build_pick_on_obj_mask(
    transformer: OfflineImageStackTransformer,
    view_mask: np.ndarray,
    pix_grasp_dist: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fling validity directly from the transformed cloth mask, following ClothMate's mask logic."""
    mask_stack = transformer.forward_raw(view_mask.astype(bool))[..., 0]
    is_left_on_obj, is_right_on_obj = get_offset_stack(mask_stack, offset=pix_grasp_dist)
    is_action_on_obj = is_left_on_obj & is_right_on_obj
    return mask_stack, is_left_on_obj, is_right_on_obj


def build_clothmate_policy(args: argparse.Namespace) -> MaximumValuePolicy:
    gpu = 0
    if args.device.startswith("cuda:"):
        gpu = int(args.device.split(":", 1)[1])
    policy = MaximumValuePolicy(
        action_primitives=["fling"],
        num_rotations=args.num_rotations - 1,
        scale_factors=parse_scales(args.scales),
        obs_dim=args.obs_dim,
        pix_grasp_dist=args.pix_grasp_dist,
        pix_drag_dist=args.pix_grasp_dist,
        pix_place_dist=10,
        deformable_weight=args.deformable_weight,
        nocs_mode="collapsed",
        network_gpu=[gpu, gpu],
        action_expl_prob=0,
        action_expl_decay=0,
        value_expl_prob=0,
        value_expl_decay=0,
        dual_networks=None,
        input_channel_types="rgb_pos",
        deformable_pos=True,
        dump_network_inputs=True,
        gpu=gpu,
    )
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["net"], strict=True)
    policy.eval()
    return policy


def prepare_view(
    rgb: np.ndarray,
    mask: np.ndarray,
    target_mask_area_ratio: float,
    obs_dim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    base_meta = {
        "src_x0": 0,
        "src_y0": 0,
        "canvas_w": rgb.shape[1],
        "canvas_h": rgb.shape[0],
        "pad_left": 0,
        "pad_top": 0,
    }
    if touches_border(mask.astype(np.uint8)):
        crop_rgb = rgb
        crop_mask = mask.astype(np.uint8)
    else:
        crop_rgb, crop_mask, base_meta = crop_component_for_inference(rgb, mask.astype(np.uint8), target_mask_area_ratio)

    square_rgb, square_mask, square_meta = crop_keep_height_hcenter(crop_rgb, crop_mask.astype(np.uint8))
    view_rgb = np.asarray(Image.fromarray(square_rgb).resize((obs_dim, obs_dim), resample=Image.BILINEAR))
    view_mask = (
        np.asarray(Image.fromarray((square_mask.astype(np.uint8) * 255)).resize((obs_dim, obs_dim), resample=Image.NEAREST))
        > 127
    )
    meta = {
        "src_x0": int(base_meta["src_x0"]) + int(square_meta["src_x0"]) - int(base_meta["pad_left"]),
        "src_y0": int(base_meta["src_y0"]) + int(square_meta["src_y0"]) - int(base_meta["pad_top"]),
        "canvas_w": int(square_meta["canvas_w"]),
        "canvas_h": int(square_meta["canvas_h"]),
        "pad_left": int(square_meta["pad_left"]),
        "pad_top": int(square_meta["pad_top"]),
        "pre_square_w": int(crop_rgb.shape[1]),
        "pre_square_h": int(crop_rgb.shape[0]),
    }
    return view_rgb, view_mask, meta


def build_state(
    view_rgb: np.ndarray,
    view_mask: np.ndarray,
    rotations: np.ndarray,
    base_scales: np.ndarray,
    obs_dim: int,
    pix_grasp_dist: int,
) -> tuple[dict[str, Any], OfflineImageStackTransformer]:
    rows, cols = np.nonzero(view_mask)
    max_width = float(max(rows.max() - rows.min(), cols.max() - cols.min())) if rows.size > 0 else 0.5
    adaptive_scale_factor = max(max_width * 1.5 / float(obs_dim), 1.0 / 2.05 / float(np.min(base_scales)))
    actual_scales = 1.0 / (base_scales * adaptive_scale_factor)
    transformer = OfflineImageStackTransformer(img_shape=(obs_dim, obs_dim), rotations=rotations, scales=actual_scales)

    bgr_view = view_rgb[:, :, ::-1]
    masked_bgr = bgr_view.copy()
    masked_bgr[~view_mask] = 0
    obs_stack = transformer.forward_raw(masked_bgr).astype(np.float32)
    coord_stack = transformer.get_inverse_coord_map()
    mask_stack, _, _ = build_pick_on_obj_mask(transformer, view_mask, pix_grasp_dist)
    is_action_on_obj = mask_stack.copy()
    is_left_on_obj, is_right_on_obj = get_offset_stack(mask_stack, offset=pix_grasp_dist)
    is_action_on_obj = is_left_on_obj & is_right_on_obj

    valid_transforms_mask = build_valid_transforms_mask(rotations, base_scales, obs_dim)
    # Offline aRTF surrogate: without depth/reachability, keep ClothMate's object-on-mask constraint
    # and the same transform-subset gating used in real get_pick_and_fling_input().
    is_valid = is_action_on_obj & valid_transforms_mask

    left_coord, right_coord = get_offset_stack(coord_stack, offset=pix_grasp_dist)
    pos_enc = np.stack(
        [
            generate_coordinate_map(obs_dim, -rotation * (360.0 / (2.0 * np.pi)), 1.0 / scale)
            for rotation, scale in transformer.transform_tuples
        ]
    ).transpose(0, 3, 1, 2)
    extra = np.zeros((pos_enc.shape[0], 1, obs_dim, obs_dim), dtype=np.float32)
    transformed_rgb = obs_stack.transpose(0, 3, 1, 2) / 255.0
    transformed_obs = torch.from_numpy(np.concatenate([transformed_rgb, extra, pos_enc], axis=1)).float()

    state = {
        "transformed_obs": transformed_obs,
        "fling_mask": torch.from_numpy(is_valid).bool(),
        "fling_info": {
            "pick_on_obj": is_action_on_obj,
            "left_coord": left_coord,
            "right_coord": right_coord,
            "coords": coord_stack,
            "transformer": transformer,
            "scales": actual_scales,
            "adaptive_scale_factor": adaptive_scale_factor,
        },
    }
    return state, transformer


def clamp_point(point: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    return (int(np.clip(point[0], 0, width - 1)), int(np.clip(point[1], 0, height - 1)))


def draw_clothmate_debug(
    out_path: Path,
    rgb: np.ndarray,
    keypoints: list[tuple[str, float, float]],
    x1_full: tuple[int, int],
    x2_full: tuple[int, int],
    nearest_x1: str,
    nearest_x2: str,
    transformed_rgb: np.ndarray,
    value_map: np.ndarray,
) -> None:
    base = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(base)
    for name, px, py in keypoints:
        r = 4
        draw.ellipse((px - r, py - r, px + r, py + r), outline="lime", width=2)
        draw.text((px + 5, py - 5), name, fill="lime")
    draw.line((x1_full[0] - 6, x1_full[1] - 6, x1_full[0] + 6, x1_full[1] + 6), fill="white", width=2)
    draw.line((x1_full[0] - 6, x1_full[1] + 6, x1_full[0] + 6, x1_full[1] - 6), fill="white", width=2)
    draw.ellipse((x2_full[0] - 7, x2_full[1] - 7, x2_full[0] + 7, x2_full[1] + 7), outline="yellow", width=2)
    draw.text((x1_full[0] + 8, x1_full[1] + 8), f"x1->{nearest_x1}", fill="white")
    draw.text((x2_full[0] + 8, x2_full[1] + 8), f"x2->{nearest_x2}", fill="yellow")

    heat = value_map.astype(np.float32)
    finite = np.isfinite(heat)
    if finite.any():
        lo = float(np.min(heat[finite]))
        hi = float(np.max(heat[finite]))
        norm = np.zeros_like(heat, dtype=np.float32)
        if hi > lo:
            norm[finite] = (heat[finite] - lo) / (hi - lo)
        heat_rgb = np.stack([norm, np.zeros_like(norm), 1.0 - norm], axis=-1)
        heat_img = Image.fromarray(np.clip(heat_rgb * 255.0, 0, 255).astype(np.uint8)).resize(base.size, Image.NEAREST)
    else:
        heat_img = Image.new("RGB", base.size, color="black")

    transformed = Image.fromarray(transformed_rgb.astype(np.uint8)).resize(base.size, Image.NEAREST)
    canvas = Image.new("RGB", (base.width * 3, base.height))
    canvas.paste(base, (0, 0))
    canvas.paste(transformed, (base.width, 0))
    canvas.paste(heat_img, (base.width * 2, 0))
    canvas.save(out_path)


def run_split(
    split: str,
    samples: list[Any],
    *,
    args: argparse.Namespace,
    policy: MaximumValuePolicy,
    rotations: np.ndarray,
    base_scales: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vis_dir = args.out_dir / f"vis_{split}"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    vis_count = 0

    if args.max_images > 0:
        samples = samples[: args.max_images]

    for sample in samples:
        rgb = np.asarray(Image.open(sample.image_path).convert("RGB"))
        mask = coco_segmentation_to_mask(sample.annotation.get("segmentation", []), rgb.shape[:2]).astype(bool)
        if mask.sum() <= 0:
            print(f"[gt-mask-fail] {sample.category} {sample.file_name} err=empty COCO segmentation")
            continue
        view_rgb, view_mask, meta = prepare_view(rgb, mask, args.target_mask_area_ratio, args.obs_dim)
        state, _ = build_state(
            view_rgb=view_rgb,
            view_mask=view_mask,
            rotations=rotations,
            base_scales=base_scales,
            obs_dim=args.obs_dim,
            pix_grasp_dist=args.pix_grasp_dist,
        )
        state["transformed_obs"] = state["transformed_obs"].to(policy.device)
        state["fling_mask"] = state["fling_mask"].to(policy.device)

        with torch.inference_mode():
            action = policy.get_action_single(state, explore=False)
        if action is None or action.get("best_index") is None:
            print(f"[clothmate-fail] {sample.category} {sample.file_name} err=no valid action")
            continue

        best_index = tuple(int(v) for v in action["best_index"])
        info = state["fling_info"]
        left_coord = info["left_coord"][best_index]
        right_coord = info["right_coord"][best_index]
        canvas_w = int(meta["canvas_w"])
        canvas_h = int(meta["canvas_h"])
        src_x0 = int(meta["src_x0"])
        src_y0 = int(meta["src_y0"])
        pad_left = int(meta["pad_left"])
        pad_top = int(meta["pad_top"])

        x1_crop = map_xy((int(round(left_coord[0])), int(round(left_coord[1]))), args.obs_dim, args.obs_dim, canvas_w, canvas_h)
        x2_crop = map_xy((int(round(right_coord[0])), int(round(right_coord[1]))), args.obs_dim, args.obs_dim, canvas_w, canvas_h)
        x1_full = clamp_point((src_x0 + x1_crop[0] - pad_left, src_y0 + x1_crop[1] - pad_top), rgb.shape[1], rgb.shape[0])
        x2_full = clamp_point((src_x0 + x2_crop[0] - pad_left, src_y0 + x2_crop[1] - pad_top), rgb.shape[1], rgb.shape[0])

        score = float(action["best_value"].detach().cpu().item() if torch.is_tensor(action["best_value"]) else action["best_value"])
        row = append_prediction_row(
            rows,
            model="clothmate",
            sample=sample,
            score=score,
            x1_full=x1_full,
            x2_full=x2_full,
        )

        if args.save_vis and vis_count < int(args.vis_limit):
            value_maps = action["fling"]["all_value_maps"][best_index[0]].detach().cpu().numpy()
            transformed_rgb = state["transformed_obs"][best_index[0], :3].detach().cpu().numpy().transpose(1, 2, 0) * 255.0
            debug_path = vis_dir / f"{sample.category}_{sample.image_id}_debug.png"
            draw_clothmate_debug(
                debug_path,
                rgb,
                sample.keypoints,
                x1_full,
                x2_full,
                row["nearest_kp_x1"],
                row["nearest_kp_x2"],
                transformed_rgb[:, :, ::-1],
                value_maps,
            )
            overlay_path = vis_dir / f"{sample.category}_{sample.image_id}.png"
            draw_vis(overlay_path, rgb, sample.keypoints, x1_full, x2_full, row["nearest_kp_x1"], row["nearest_kp_x2"])
            vis_count += 1

    return rows


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    categories = [cat.strip().lower() for cat in args.categories.split(",") if cat.strip()]
    base_scales = np.asarray(parse_scales(args.scales), dtype=np.float32)
    rotations = np.linspace(-np.pi, np.pi, args.num_rotations, dtype=np.float32)
    policy = build_clothmate_policy(args)

    all_samples = iter_artf_samples(args.dataset_root, splits, categories)
    all_rows: list[dict[str, Any]] = []
    for split in splits:
        split_samples = [sample for sample in all_samples if sample.split == split]
        split_rows = run_split(split, split_samples, args=args, policy=policy, rotations=rotations, base_scales=base_scales)
        all_rows.extend(split_rows)
        summary = summarize_rows(split_rows, "clothmate", split)
        write_rows_csv(args.out_dir / f"predictions_{split}.csv", split_rows)
        write_summary(args.out_dir / f"summary_{split}.json", summary)
        write_split_plots(split_rows, summary, args.out_dir / f"plots_{split}")

    combined_summary = summarize_rows(all_rows, "clothmate", "combined")
    write_rows_csv(args.out_dir / "predictions_combined.csv", all_rows)
    write_summary(args.out_dir / "summary_combined.json", combined_summary)
    write_split_plots(all_rows, combined_summary, args.out_dir / "plots_combined")
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
