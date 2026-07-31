#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
SAM3_ROOT = Path("${WORKSPACE_ROOT}/sam3_test")

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SAM3_ROOT / "upstream_sam3"))

from unfold.algorithms.pair_policy.model import PairPolicyNet
from unfold.algorithms.supervision.projection import masked_softmax_heatmap

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot single-grasp proxy evaluation on the ICRA 2024 cloth competition benchmark."
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("${DATASET_ROOT}/cloth_competition/ICRA_2024_cloth_competition_dataset.zip"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "experiments/pair_policy/runs/train/segformer_mit_b4_exp_kl_tau01_sym_maxswap_nas/best.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples in the zip.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mask-source",
        choices=("full", "center_box", "sam3"),
        default="full",
        help="Simple valid-mask choice for the first benchmark iteration.",
    )
    parser.add_argument(
        "--center-box-ratio",
        type=float,
        default=0.72,
        help="Foreground width/height ratio when mask-source=center_box.",
    )
    parser.add_argument("--random-points", type=int, default=64, help="Random baseline samples per image.")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--vis-limit", type=int, default=24)
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        default=SAM3_ROOT / "sam3.pt",
    )
    parser.add_argument("--sam3-threshold", type=float, default=0.2)
    parser.add_argument("--sam3-prompt", type=str, default="cloth")
    parser.add_argument(
        "--target-mask-area-ratio",
        type=float,
        default=0.145,
        help="Target foreground occupancy after crop+pad when using SAM3 crops.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "experiments/cloth_competition_zero_shot/runs/pilot_zero_shot_single_grasp_v1",
    )
    return parser.parse_args()


def load_pair_policy(checkpoint_path: Path, device: str) -> tuple[PairPolicyNet, dict[str, Any]]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = PairPolicyNet(
        in_channels=int(cfg["model"].get("in_channels", 3)),
        feature_dim=int(cfg["model"].get("feature_dim", 64)),
        num_x1_samples=int(cfg["train"].get("num_x1_samples", 4)),
        backbone=cfg["model"].get("backbone"),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


def build_sam3_processor(device: str, checkpoint: Path, threshold: float) -> Any:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        compile=False,
    )
    return Sam3Processor(model, device=device, confidence_threshold=threshold)


def preprocess(rgb: np.ndarray, valid_mask: np.ndarray, cfg: dict[str, Any]) -> tuple[torch.Tensor, np.ndarray]:
    h = int(cfg["data"]["resize_height"])
    w = int(cfg["data"]["resize_width"])
    rgb_resized = np.asarray(Image.fromarray(rgb).resize((w, h), resample=Image.BILINEAR)).astype(np.float32) / 255.0
    mask_resized = np.asarray(
        Image.fromarray((valid_mask.astype(np.uint8) * 255)).resize((w, h), resample=Image.NEAREST)
    ).astype(np.float32)
    mask_resized = (mask_resized > 127).astype(np.float32)
    if str(cfg["data"].get("input_normalization", "none")) == "imagenet":
        rgb_resized = (rgb_resized - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    rgb_resized = rgb_resized * mask_resized[..., None]
    image_t = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0)
    return image_t, mask_resized


def decode_heatmap(logits: np.ndarray, valid_mask: np.ndarray, cfg: dict[str, Any], which: str) -> np.ndarray:
    target_name = str((cfg.get("target", {}) or {}).get("name", ""))
    if which == "a1":
        tau = float((cfg.get("train", {}) or {}).get("a1_target_tau", 1.0))
    else:
        tau = float((cfg.get("train", {}) or {}).get("a2_target_tau", 1.0))
    if target_name in {"masked_softmax", "topk_masked_softmax", "image_exp"}:
        pred = masked_softmax_heatmap(logits, valid_mask > 0.5, tau=tau).astype(np.float32, copy=False)
    else:
        pred = torch.sigmoid(torch.from_numpy(logits)).cpu().numpy().astype(np.float32, copy=False)
        pred[np.asarray(valid_mask) <= 0.5] = 0.0
    return pred


def argmax_xy(heat: np.ndarray, valid: np.ndarray) -> tuple[int, int]:
    masked = np.where(valid > 0.5, heat, -np.inf)
    flat = int(np.argmax(masked))
    h, w = masked.shape
    y, x = divmod(flat, w)
    return int(x), int(y)


def map_xy(xy: tuple[int, int], src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    x, y = xy
    if src_w <= 1 or src_h <= 1:
        return 0, 0
    x_dst = int(round(x * (dst_w - 1) / float(src_w - 1)))
    y_dst = int(round(y * (dst_h - 1) / float(src_h - 1)))
    return x_dst, y_dst


def build_valid_mask(image_h: int, image_w: int, source: str, ratio: float) -> np.ndarray:
    if source == "full":
        return np.ones((image_h, image_w), dtype=np.uint8)
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    box_w = max(1, int(round(image_w * ratio)))
    box_h = max(1, int(round(image_h * ratio)))
    x0 = max(0, (image_w - box_w) // 2)
    y0 = max(0, (image_h - box_h) // 2)
    mask[y0:y0 + box_h, x0:x0 + box_w] = 1
    return mask


def choose_mask(output: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray, float]:
    masks = output["masks"].squeeze(1).detach().float().cpu().numpy()
    boxes = output["boxes"].detach().float().cpu().numpy()
    scores = output["scores"].detach().float().cpu().numpy()
    if masks.size == 0:
        raise RuntimeError("SAM3 returned no masks.")
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    best_idx = int(np.argmax(scores + 1e-6 * areas))
    return masks[best_idx] > 0.5, boxes[best_idx], float(scores[best_idx])


def touches_border(mask: np.ndarray) -> bool:
    if mask.ndim != 2:
        return True
    if mask.shape[0] == 0 or mask.shape[1] == 0:
        return True
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def component_bbox(component_mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(component_mask)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, 1, 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return x0, y0, x1, y1


def crop_component_for_inference(
    rgb: np.ndarray,
    component_mask: np.ndarray,
    target_mask_area_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    frame_h, frame_w = component_mask.shape
    x0, y0, x1, y1 = component_bbox(component_mask)
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)
    component_area = float(component_mask.sum())
    bbox_area = float(bbox_w * bbox_h)
    target_ratio = float(max(target_mask_area_ratio, 1e-4))
    desired_area = max(component_area / target_ratio, bbox_area)
    aspect = bbox_w / float(bbox_h)
    desired_w = int(np.ceil(np.sqrt(desired_area * aspect)))
    desired_h = int(np.ceil(np.sqrt(desired_area / aspect)))
    desired_w = max(desired_w, bbox_w)
    desired_h = max(desired_h, bbox_h)

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    crop_x0 = int(np.floor(cx - desired_w / 2.0))
    crop_y0 = int(np.floor(cy - desired_h / 2.0))
    crop_x1 = crop_x0 + desired_w
    crop_y1 = crop_y0 + desired_h

    src_x0 = max(0, crop_x0)
    src_y0 = max(0, crop_y0)
    src_x1 = min(frame_w, crop_x1)
    src_y1 = min(frame_h, crop_y1)
    src_w = max(1, src_x1 - src_x0)
    src_h = max(1, src_y1 - src_y0)

    pad_left = max(0, -crop_x0)
    pad_top = max(0, -crop_y0)
    pad_right = max(0, crop_x1 - frame_w)
    pad_bottom = max(0, crop_y1 - frame_h)

    crop_rgb = rgb[src_y0:src_y1, src_x0:src_x1]
    crop_mask = component_mask[src_y0:src_y1, src_x0:src_x1]
    canvas_rgb = np.zeros((desired_h, desired_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((desired_h, desired_w), dtype=np.uint8)
    canvas_rgb[pad_top:pad_top + src_h, pad_left:pad_left + src_w] = crop_rgb
    canvas_mask[pad_top:pad_top + src_h, pad_left:pad_left + src_w] = crop_mask.astype(np.uint8)

    meta = {
        "src_x0": int(src_x0),
        "src_y0": int(src_y0),
        "src_x1": int(src_x1),
        "src_y1": int(src_y1),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "pad_right": int(pad_right),
        "pad_bottom": int(pad_bottom),
        "canvas_w": int(desired_w),
        "canvas_h": int(desired_h),
        "src_w": int(src_w),
        "src_h": int(src_h),
    }
    return canvas_rgb, canvas_mask, meta


def l2_distance(p: tuple[int, int], q: tuple[int, int]) -> float:
    return float(math.sqrt((float(p[0]) - float(q[0])) ** 2 + (float(p[1]) - float(q[1])) ** 2))


def image_diag(image_h: int, image_w: int) -> float:
    return float(math.sqrt(float(image_h) ** 2 + float(image_w) ** 2))


def draw_vis(
    out_path: Path,
    rgb: np.ndarray,
    gt_xy: tuple[int, int],
    a1_xy: tuple[int, int],
    a2_xy: tuple[int, int],
    best_xy: tuple[int, int],
) -> None:
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)

    def draw_cross(xy: tuple[int, int], color: str, size: int = 10, width: int = 3) -> None:
        x, y = xy
        draw.line((x - size, y - size, x + size, y + size), fill=color, width=width)
        draw.line((x - size, y + size, x + size, y - size), fill=color, width=width)

    def draw_ring(xy: tuple[int, int], color: str, r: int = 10, width: int = 3) -> None:
        x, y = xy
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=width)

    draw_ring(gt_xy, "lime", r=12, width=4)
    draw_cross(a1_xy, "white")
    draw_cross(a2_xy, "yellow")
    draw_ring(best_xy, "red", r=8, width=3)
    draw.text((20, 20), "GT", fill="lime")
    draw.text((20, 48), "A1", fill="white")
    draw.text((20, 76), "A2", fill="yellow")
    draw.text((20, 104), "Best(A1,A2)", fill="red")
    img.save(out_path)


def random_baselines(
    rng: random.Random,
    valid_mask: np.ndarray,
    gt_xy: tuple[int, int],
    count: int,
) -> tuple[float, float]:
    ys, xs = np.nonzero(valid_mask > 0)
    coords = list(zip(xs.tolist(), ys.tolist()))
    if not coords:
        return float("nan"), float("nan")
    samples = [coords[rng.randrange(len(coords))] for _ in range(max(1, count))]
    dists = [l2_distance(pt, gt_xy) for pt in samples]
    rand_single = float(sum(dists) / len(dists))
    rand_best2 = float(min(l2_distance(samples[i], gt_xy) for i in range(min(2, len(samples)))))
    return rand_single, rand_best2


def list_sample_ids(zf: zipfile.ZipFile) -> list[str]:
    sample_ids = sorted(
        {
            "/".join(name.split("/")[:2])
            for name in zf.namelist()
            if name.startswith("cloth_competition_dataset_") and "/sample_" in name
        }
    )
    return sample_ids


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.out_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_pair_policy(args.checkpoint, args.device)
    rng = random.Random(args.seed)
    processor = None
    if args.mask_source == "sam3":
        processor = build_sam3_processor(args.device, args.sam3_checkpoint, args.sam3_threshold)

    rows: list[dict[str, Any]] = []
    vis_count = 0
    skipped_sam3 = 0

    with zipfile.ZipFile(args.zip_path) as zf:
        sample_ids = list_sample_ids(zf)
        if args.max_samples > 0:
            sample_ids = sample_ids[: args.max_samples]

        for sample_id in sample_ids:
            prefix = f"{sample_id}/"
            rgb = np.asarray(Image.open(io.BytesIO(zf.read(prefix + "observation_start/image_left.png"))).convert("RGB"))
            ann = json.loads(zf.read(prefix + "grasp/grasp_annotation.json").decode("utf-8"))
            gt_xy = (int(ann["clicked_point_frontal"][0]), int(ann["clicked_point_frontal"][1]))

            crop_rgb = rgb
            valid_mask_full = None
            crop_meta = {
                "src_x0": 0,
                "src_y0": 0,
                "pad_left": 0,
                "pad_top": 0,
                "canvas_w": rgb.shape[1],
                "canvas_h": rgb.shape[0],
            }
            if args.mask_source == "sam3":
                try:
                    assert processor is not None
                    state = processor.set_image(Image.fromarray(rgb))
                    text_outputs = processor.model.backbone.forward_text([args.sam3_prompt], device=args.device)
                    state["backbone_out"].update(text_outputs)
                    state["geometric_prompt"] = processor.model._get_dummy_prompt()
                    sam_out = processor._forward_grounding(state)
                    valid_mask, _, _ = choose_mask(sam_out)
                except Exception as exc:
                    skipped_sam3 += 1
                    print(f"[sam3-fail] {sample_id} err={exc}")
                    continue
                valid_mask_full = valid_mask.astype(np.uint8).copy()
                if touches_border(valid_mask.astype(np.uint8)):
                    valid_mask = valid_mask.astype(np.uint8)
                else:
                    crop_rgb, valid_mask, crop_meta = crop_component_for_inference(
                        rgb, valid_mask.astype(np.uint8), target_mask_area_ratio=float(args.target_mask_area_ratio)
                    )
            else:
                valid_mask = build_valid_mask(rgb.shape[0], rgb.shape[1], args.mask_source, args.center_box_ratio)
                valid_mask_full = valid_mask.copy()

            image_t, mask_resized = preprocess(crop_rgb, valid_mask, cfg)
            image_t = image_t.to(args.device)
            sampled_x1_xy = torch.zeros((1, 1, 2), dtype=torch.float32, device=args.device)
            sampled_x1_valid = torch.zeros((1, 1), dtype=torch.bool, device=args.device)

            with torch.inference_mode():
                out_a1 = model(image_t, sampled_x1_xy, sampled_x1_valid)
                a1_logits = out_a1["a1_logits"][0].detach().cpu().numpy()
                a1_pred = decode_heatmap(a1_logits, mask_resized, cfg, which="a1")
                a1_small = argmax_xy(a1_pred, mask_resized)
                sampled_x1_xy[0, 0, 0] = float(a1_small[0])
                sampled_x1_xy[0, 0, 1] = float(a1_small[1])
                sampled_x1_valid[0, 0] = True

                out_a2 = model(image_t, sampled_x1_xy, sampled_x1_valid)
                a2_logits = out_a2["a2_logits"][0, 0].detach().cpu().numpy()
                a2_pred = decode_heatmap(a2_logits, mask_resized, cfg, which="a2")
                a2_small = argmax_xy(a2_pred, mask_resized)

            a1_crop = map_xy(a1_small, a1_pred.shape[1], a1_pred.shape[0], int(crop_meta["canvas_w"]), int(crop_meta["canvas_h"]))
            a2_crop = map_xy(a2_small, a2_pred.shape[1], a2_pred.shape[0], int(crop_meta["canvas_w"]), int(crop_meta["canvas_h"]))
            a1_xy = (
                int(crop_meta["src_x0"]) + a1_crop[0] - int(crop_meta["pad_left"]),
                int(crop_meta["src_y0"]) + a1_crop[1] - int(crop_meta["pad_top"]),
            )
            a2_xy = (
                int(crop_meta["src_x0"]) + a2_crop[0] - int(crop_meta["pad_left"]),
                int(crop_meta["src_y0"]) + a2_crop[1] - int(crop_meta["pad_top"]),
            )

            dist_a1 = l2_distance(a1_xy, gt_xy)
            dist_a2 = l2_distance(a2_xy, gt_xy)
            if dist_a1 <= dist_a2:
                best_xy = a1_xy
                best_map = "a1"
                dist_best = dist_a1
            else:
                best_xy = a2_xy
                best_map = "a2"
                dist_best = dist_a2

            center_xy = (rgb.shape[1] // 2, rgb.shape[0] // 2)
            dist_center = l2_distance(center_xy, gt_xy)
            assert valid_mask_full is not None
            rand_single, rand_best2 = random_baselines(rng, valid_mask_full, gt_xy, args.random_points)
            diag = image_diag(rgb.shape[0], rgb.shape[1])

            rows.append(
                {
                    "sample_id": sample_id,
                    "gt_x": gt_xy[0],
                    "gt_y": gt_xy[1],
                    "a1_x": a1_xy[0],
                    "a1_y": a1_xy[1],
                    "a2_x": a2_xy[0],
                    "a2_y": a2_xy[1],
                    "best_x": best_xy[0],
                    "best_y": best_xy[1],
                    "best_map": best_map,
                    "dist_a1_px": dist_a1,
                    "dist_a2_px": dist_a2,
                    "dist_best_px": dist_best,
                    "dist_center_px": dist_center,
                    "dist_random_single_px": rand_single,
                    "dist_random_best2_px": rand_best2,
                    "dist_a1_norm": dist_a1 / diag,
                    "dist_a2_norm": dist_a2 / diag,
                    "dist_best_norm": dist_best / diag,
                    "dist_center_norm": dist_center / diag,
                    "dist_random_single_norm": rand_single / diag,
                    "dist_random_best2_norm": rand_best2 / diag,
                }
            )

            if args.save_vis and vis_count < args.vis_limit:
                out_path = vis_dir / f"{sample_id.replace('/', '_')}.png"
                draw_vis(out_path, rgb, gt_xy, a1_xy, a2_xy, best_xy)
                vis_count += 1

    if not rows:
        raise RuntimeError("No benchmark samples were processed.")

    csv_path = args.out_dir / "single_grasp_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def mean_of(key: str) -> float:
        return float(np.mean([float(r[key]) for r in rows]))

    def median_of(key: str) -> float:
        return float(np.median([float(r[key]) for r in rows]))

    summary = {
        "zip_path": str(args.zip_path),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "mask_source": args.mask_source,
        "center_box_ratio": float(args.center_box_ratio),
        "sample_count": len(rows),
        "skipped_sam3": int(skipped_sam3),
        "dist_a1_px_mean": mean_of("dist_a1_px"),
        "dist_a1_px_median": median_of("dist_a1_px"),
        "dist_a2_px_mean": mean_of("dist_a2_px"),
        "dist_a2_px_median": median_of("dist_a2_px"),
        "dist_best_px_mean": mean_of("dist_best_px"),
        "dist_best_px_median": median_of("dist_best_px"),
        "dist_center_px_mean": mean_of("dist_center_px"),
        "dist_center_px_median": median_of("dist_center_px"),
        "dist_random_single_px_mean": mean_of("dist_random_single_px"),
        "dist_random_best2_px_mean": mean_of("dist_random_best2_px"),
        "dist_a1_norm_mean": mean_of("dist_a1_norm"),
        "dist_a2_norm_mean": mean_of("dist_a2_norm"),
        "dist_best_norm_mean": mean_of("dist_best_norm"),
        "dist_center_norm_mean": mean_of("dist_center_norm"),
        "dist_random_single_norm_mean": mean_of("dist_random_single_norm"),
        "dist_random_best2_norm_mean": mean_of("dist_random_best2_norm"),
        "best_map_counts": {
            "a1": int(sum(1 for r in rows if r["best_map"] == "a1")),
            "a2": int(sum(1 for r in rows if r["best_map"] == "a2")),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Cloth Competition Zero-Shot Single-Grasp Proxy",
        "",
        f"- samples: {summary['sample_count']}",
        f"- checkpoint: `{args.checkpoint}`",
        f"- mask_source: `{args.mask_source}`",
        "",
        "## Mean Pixel Distance To Benchmark Grasp",
        "",
        f"- `A1 top1`: {summary['dist_a1_px_mean']:.2f}",
        f"- `A2 top1`: {summary['dist_a2_px_mean']:.2f}",
        f"- `best(A1,A2)`: {summary['dist_best_px_mean']:.2f}",
        f"- `image center`: {summary['dist_center_px_mean']:.2f}",
        f"- `random single`: {summary['dist_random_single_px_mean']:.2f}",
        f"- `random best-of-two`: {summary['dist_random_best2_px_mean']:.2f}",
        "",
        "## Mean Normalized Distance",
        "",
        f"- `A1 top1`: {summary['dist_a1_norm_mean']:.4f}",
        f"- `A2 top1`: {summary['dist_a2_norm_mean']:.4f}",
        f"- `best(A1,A2)`: {summary['dist_best_norm_mean']:.4f}",
        f"- `image center`: {summary['dist_center_norm_mean']:.4f}",
        f"- `random single`: {summary['dist_random_single_norm_mean']:.4f}",
        f"- `random best-of-two`: {summary['dist_random_best2_norm_mean']:.4f}",
        "",
        "## Which Peak Was Closer",
        "",
        f"- `A1`: {summary['best_map_counts']['a1']}",
        f"- `A2`: {summary['best_map_counts']['a2']}",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
