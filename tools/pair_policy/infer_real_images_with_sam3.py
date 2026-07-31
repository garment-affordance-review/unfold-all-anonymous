#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
SAM3_ROOT = Path("${WORKSPACE_ROOT}/sam3_test")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SAM3_ROOT / "upstream_sam3"))

from unfold.algorithms.pair_policy.model import PairPolicyNet
from unfold.algorithms.supervision.projection import masked_softmax_heatmap


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
JET_CMAP = cm.get_cmap("jet")
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEFAULT_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pair-policy inference on real images using SAM3 masks.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("${WORKSPACE_ROOT}/clothmate_temp_2/outputs/real"),
        help="Directory containing real cloth RGB images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "experiments/pair_policy/runs/train/segformer_mit_b4_minmax_weighted_huber/best.pt",
        help="Pair-policy checkpoint.",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        default=SAM3_ROOT / "sam3.pt",
        help="Local SAM3 checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "experiments/pair_policy/runs/debug/real_sam3_inference",
        help="Output visualization directory.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-images", type=int, default=0, help="0 means all images.")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("${DATASET_ROOT}/aRTF-Clothes-dataset/extracted/aRTFClothes"),
        help="Dataset root containing train/test folders and COCO json files. Used with --mask-source coco_gt.",
    )
    parser.add_argument(
        "--mask-source",
        type=str,
        default="sam3",
        choices=("sam3", "coco_gt"),
        help="Use SAM3 or dataset COCO polygons as the object mask.",
    )
    parser.add_argument("--square-size", type=int, default=640, help="Square size used before SAM3 and pair-policy.")
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="square",
        choices=("square", "stretch"),
        help="How to prepare RGB before SAM3/pair-policy. square pads to square, stretch resizes to square directly.",
    )
    parser.add_argument(
        "--cropfit",
        action="store_true",
        help="Crop around the SAM3 mask and pad to match training object scale.",
    )
    parser.add_argument(
        "--target-mask-area-ratio",
        type=float,
        default=0.145,
        help="Target foreground occupancy after crop+pad (used with --cropfit).",
    )
    parser.add_argument(
        "--vis-mode",
        type=str,
        default="split",
        choices=("split", "dual", "paper"),
        help="Visualization mode: split (A1/A2 side-by-side), dual (A1 red + A2 blue), or paper (main image + zoomed heatmap insets).",
    )
    parser.add_argument(
        "--vis-crop-mode",
        type=str,
        default="none",
        choices=("none", "square_hcenter"),
        help="Visualization-only crop. square_hcenter keeps full height and crops width to a square centered on the mask.",
    )
    return parser.parse_args()


def _prompt_from_path(path: Path, image_root: Path) -> str:
    known = {"shorts", "tshirts", "towels", "boxershorts", "pants", "shirts", "tops", "dress", "skirt", "jumpsuit"}
    parent_token = path.parent.name.split("_")[0].lower()
    if parent_token in known:
        return parent_token
    rel_parts = path.relative_to(image_root).parts
    if rel_parts:
        # Prefer known garment categories if present in the path.
        for part in rel_parts:
            token = part.split("_")[0].lower()
            if token in known:
                return token
        return rel_parts[0].split("_")[0].lower()
    stem = path.stem.lower()
    return stem.split("_")[0]


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


def choose_mask(output: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray, float]:
    masks = output["masks"].squeeze(1).detach().float().cpu().numpy()
    boxes = output["boxes"].detach().float().cpu().numpy()
    scores = output["scores"].detach().float().cpu().numpy()
    if masks.size == 0:
        raise RuntimeError("SAM3 returned no masks.")
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    best_idx = int(np.argmax(scores + 1e-6 * areas))
    return masks[best_idx] > 0.5, boxes[best_idx], float(scores[best_idx])


def load_pair_policy(checkpoint_path: Path, device: str) -> tuple[PairPolicyNet, dict]:
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


def preprocess(rgb: np.ndarray, mask: np.ndarray, cfg: dict) -> tuple[torch.Tensor, np.ndarray]:
    h = int(cfg["data"]["resize_height"])
    w = int(cfg["data"]["resize_width"])
    rgb_resized = np.asarray(Image.fromarray(rgb).resize((w, h), resample=Image.BILINEAR)).astype(np.float32) / 255.0
    mask_resized = np.asarray(
        Image.fromarray((mask.astype(np.uint8) * 255)).resize((w, h), resample=Image.NEAREST)
    ).astype(np.float32)
    mask_resized = (mask_resized > 127).astype(np.float32)
    if str(cfg["data"].get("input_normalization", "none")) == "imagenet":
        rgb_resized = (rgb_resized - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    rgb_resized = rgb_resized * mask_resized[..., None]
    image_t = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0)
    return image_t, mask_resized


def pad_to_square(rgb: np.ndarray, square_size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = float(square_size) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    rgb_resized = np.asarray(Image.fromarray(rgb).resize((new_w, new_h), resample=Image.BILINEAR))
    canvas = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    top = (square_size - new_h) // 2
    left = (square_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = rgb_resized
    return canvas


def stretch_to_square(rgb: np.ndarray, square_size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize((square_size, square_size), resample=Image.BILINEAR))


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


def load_coco(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def coco_segmentation_to_mask(segmentation: Any, image_hw: tuple[int, int]) -> np.ndarray:
    image_h, image_w = image_hw
    mask_img = Image.new("L", (int(image_w), int(image_h)), 0)
    draw = ImageDraw.Draw(mask_img)
    if isinstance(segmentation, list):
        for poly in segmentation:
            if not poly or len(poly) < 6:
                continue
            points = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
            draw.polygon(points, outline=1, fill=1)
    return np.asarray(mask_img, dtype=np.uint8)


def build_coco_ann_lookup(coco: dict[str, Any]) -> dict[str, dict[str, Any]]:
    image_by_id = {int(img["id"]): img for img in coco.get("images", [])}
    lookup: dict[str, dict[str, Any]] = {}
    for ann in coco.get("annotations", []):
        img = image_by_id.get(int(ann["image_id"]))
        if img is None:
            continue
        lookup[str(img["file_name"])] = ann
    return lookup


def argmax_xy(heat: np.ndarray, valid: np.ndarray) -> tuple[int, int]:
    masked = np.where(valid > 0.5, heat, -np.inf)
    flat = int(np.argmax(masked))
    h, w = masked.shape
    y, x = divmod(flat, w)
    return int(x), int(y)


def decode_heatmap(logits: np.ndarray, valid_mask: np.ndarray, cfg: dict, which: str) -> np.ndarray:
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


def resize_heatmap(heat: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(heat.astype(np.float32), mode="F").resize((out_w, out_h), resample=Image.BILINEAR))


def map_xy(xy: tuple[int, int], src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    x, y = xy
    if src_w <= 1 or src_h <= 1:
        return 0, 0
    x_dst = int(round(x * (dst_w - 1) / float(src_w - 1)))
    y_dst = int(round(y * (dst_h - 1) / float(src_h - 1)))
    return x_dst, y_dst


def heat_to_rgb(heat: np.ndarray) -> np.ndarray:
    hmin = float(heat.min())
    hmax = float(heat.max())
    if hmax <= hmin + 1e-8:
        norm = np.zeros_like(heat, dtype=np.float32)
    else:
        norm = (heat - hmin) / (hmax - hmin)
    return (JET_CMAP(norm)[..., :3] * 255).astype(np.uint8)


def normalize_heat(heat: np.ndarray) -> np.ndarray:
    hmin = float(heat.min())
    hmax = float(heat.max())
    if hmax <= hmin + 1e-8:
        return np.zeros_like(heat, dtype=np.float32)
    return ((heat - hmin) / (hmax - hmin)).astype(np.float32, copy=False)


def overlay_component_heat(base_rgb: np.ndarray, component_mask: np.ndarray, heat: np.ndarray) -> np.ndarray:
    heat_rgb = heat_to_rgb(heat)
    overlay = base_rgb.copy()
    hmin = float(heat.min())
    hmax = float(heat.max())
    if hmax <= hmin + 1e-8:
        norm = np.zeros_like(heat, dtype=np.float32)
    else:
        norm = (heat - hmin) / (hmax - hmin)
    alpha = (0.18 + 0.52 * norm) * component_mask.astype(np.float32)
    alpha = alpha[..., None]
    overlay = (overlay.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return overlay


def overlay_standard_heat(base_rgb: np.ndarray, heat: np.ndarray, alpha_scale: float = 0.62) -> np.ndarray:
    heat_rgb = heat_to_rgb(heat)
    norm = normalize_heat(heat)
    alpha = (0.12 + alpha_scale * norm)[..., None]
    return (base_rgb.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)


def overlay_dual_heat(
    base_rgb: np.ndarray,
    component_mask: np.ndarray,
    a1_heat: np.ndarray,
    a2_heat: np.ndarray,
) -> np.ndarray:
    a1_norm = normalize_heat(a1_heat)
    a2_norm = normalize_heat(a2_heat)
    mask = component_mask.astype(np.float32)
    alpha = (0.18 + 0.52 * np.maximum(a1_norm, a2_norm)) * mask
    alpha = alpha[..., None]
    color = np.zeros_like(base_rgb, dtype=np.float32)
    color[..., 0] = 255.0 * a1_norm
    color[..., 2] = 255.0 * a2_norm
    overlay = (base_rgb.astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def draw_x(draw: ImageDraw.ImageDraw, xy: tuple[int, int], color: str, size: int = 8, width: int = 3) -> None:
    x, y = xy
    draw.line((x - size, y - size, x + size, y + size), fill=color, width=width)
    draw.line((x - size, y + size, x + size, y - size), fill=color, width=width)


def draw_o(draw: ImageDraw.ImageDraw, xy: tuple[int, int], color: str, radius: int = 9, width: int = 3) -> None:
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)


def draw_hit_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    color: str,
    label: str | None,
    image_size: tuple[int, int],
    font: ImageFont.ImageFont | None = None,
) -> None:
    x, y = xy
    arm = 42
    radius = 54
    draw.line((x - arm, y - arm, x + arm, y + arm), fill=color, width=9)
    draw.line((x - arm, y + arm, x + arm, y - arm), fill=color, width=9)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=7)
    img_w, img_h = image_size
    font = font or ImageFont.load_default()
    if label:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        cx = min(max(8 + text_w / 2.0, float(x)), img_w - 8 - text_w / 2.0)
        top_y = min(max(8, y + radius - 2), max(8, img_h - text_h - 8))
        draw.text((cx, top_y), label, fill=color, font=font, anchor="ma")


def crop_vis_square_hcenter(
    rgb: np.ndarray,
    mask: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    x1_xy: tuple[int, int],
    x2_xy: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    out_h, out_w = rgb.shape[:2]
    if out_w <= out_h:
        return rgb, mask, a1, a2, x1_xy, x2_xy

    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        center_x = out_w // 2
    else:
        center_x = int(round(xs.mean()))

    crop_w = out_h
    left = center_x - crop_w // 2
    left = max(0, min(left, out_w - crop_w))
    right = left + crop_w

    rgb_crop = rgb[:, left:right]
    mask_crop = mask[:, left:right]
    a1_crop = a1[:, left:right]
    a2_crop = a2[:, left:right]
    x1_crop = (int(x1_xy[0] - left), int(x1_xy[1]))
    x2_crop = (int(x2_xy[0] - left), int(x2_xy[1]))
    return rgb_crop, mask_crop, a1_crop, a2_crop, x1_crop, x2_crop


def crop_centered_window(
    rgb: np.ndarray,
    heat: np.ndarray,
    center_xy: tuple[int, int],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    h, w = rgb.shape[:2]
    cx, cy = center_xy
    crop_w = min(window_size, w)
    crop_h = min(window_size, h)
    left = max(0, min(cx - crop_w // 2, w - crop_w))
    top = max(0, min(cy - crop_h // 2, h - crop_h))
    right = left + crop_w
    bottom = top + crop_h
    rgb_crop = rgb[top:bottom, left:right]
    heat_crop = heat[top:bottom, left:right]
    local_xy = (int(cx - left), int(cy - top))
    return rgb_crop, heat_crop, local_xy


def build_paper_vis(
    rgb: np.ndarray,
    a1_heat: np.ndarray,
    a2_heat: np.ndarray,
    x1: tuple[int, int],
    x2: tuple[int, int],
    prompt: str,
    sam_score: float,
) -> Image.Image:
    base = Image.fromarray(rgb.copy())
    draw_main = ImageDraw.Draw(base)
    label_font = ImageFont.truetype(DEFAULT_FONT_BOLD, size=108)
    inset_font = ImageFont.truetype(DEFAULT_FONT_BOLD, size=96)
    draw_hit_marker(draw_main, x1, "white", None, base.size, font=label_font)
    draw_hit_marker(draw_main, x2, "yellow", None, base.size, font=label_font)

    inset_side = max(96, min(base.width // 4, base.height // 3))
    zoom_window = max(120, min(rgb.shape[0] // 2, 180))
    margin = 12

    def make_inset(label: str, heat: np.ndarray, center_xy: tuple[int, int], label_color: str) -> Image.Image:
        patch_rgb, patch_heat, local_xy = crop_centered_window(rgb, heat, center_xy, zoom_window)
        patch = overlay_standard_heat(patch_rgb, patch_heat, alpha_scale=0.70)
        patch_img = Image.fromarray(patch).resize((inset_side, inset_side), resample=Image.BILINEAR)
        scale_x = inset_side / float(max(1, patch_rgb.shape[1]))
        scale_y = inset_side / float(max(1, patch_rgb.shape[0]))
        px = int(round((local_xy[0] + 0.5) * scale_x - 0.5))
        py = int(round((local_xy[1] + 0.5) * scale_y - 0.5))
        draw = ImageDraw.Draw(patch_img)
        draw.rounded_rectangle((0, 0, inset_side - 1, inset_side - 1), radius=12, outline="white", width=3)
        draw.ellipse((px - 36, py - 36, px + 36, py + 36), outline=label_color, width=6)
        draw.line((px - 24, py - 24, px + 24, py + 24), fill=label_color, width=6)
        draw.line((px - 24, py + 24, px + 24, py - 24), fill=label_color, width=6)
        draw.text((inset_side // 2, inset_side - 10), label, fill=label_color, font=inset_font, anchor="ms")
        return patch_img

    x1_inset = make_inset("a₁", a1_heat, x1, "white")
    x2_inset = make_inset("a₂", a2_heat, x2, "yellow")
    base.paste(x1_inset, (margin, margin))
    base.paste(x2_inset, (base.width - inset_side - margin, margin))
    return base


def save_vis(
    out_path: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    x1_xy: tuple[int, int],
    x2_xy: tuple[int, int],
    prompt: str,
    sam_score: float,
    vis_mode: str,
    coords_in_full: bool = False,
    vis_crop_mode: str = "none",
) -> None:
    out_h, out_w = rgb.shape[:2]
    a1_up = resize_heatmap(a1, out_w, out_h)
    a2_up = resize_heatmap(a2, out_w, out_h)
    src_h, src_w = a1.shape
    if coords_in_full:
        x1 = (int(x1_xy[0]), int(x1_xy[1]))
        x2 = (int(x2_xy[0]), int(x2_xy[1]))
    else:
        x1 = map_xy(x1_xy, src_w, src_h, out_w, out_h)
        x2 = map_xy(x2_xy, src_w, src_h, out_w, out_h)

    if vis_crop_mode == "square_hcenter":
        rgb, mask, a1_up, a2_up, x1, x2 = crop_vis_square_hcenter(rgb, mask, a1_up, a2_up, x1, x2)
        out_h, out_w = rgb.shape[:2]

    if vis_mode == "paper":
        paper_img = build_paper_vis(rgb, a1_up, a2_up, x1, x2, prompt, sam_score)
        paper_img.save(out_path)
        return

    if vis_mode == "dual":
        dual_panel = overlay_dual_heat(rgb, mask, a1_up, a2_up)
        dual_img = Image.fromarray(dual_panel)
        draw_dual = ImageDraw.Draw(dual_img)
        draw_x(draw_dual, x1, "white")
        draw_o(draw_dual, x1, "white")
        draw_x(draw_dual, x2, "yellow")
        draw_dual.text((12, 12), f"A1(red)/A2(blue)  prompt={prompt}  sam={sam_score:.2f}", fill="white")
        dual_img.save(out_path)
        return

    a1_panel = overlay_component_heat(rgb, mask, a1_up)
    a2_panel = overlay_component_heat(rgb, mask, a2_up)
    a1_img = Image.fromarray(a1_panel)
    a2_img = Image.fromarray(a2_panel)
    draw_a1 = ImageDraw.Draw(a1_img)
    draw_a2 = ImageDraw.Draw(a2_img)
    draw_x(draw_a1, x1, "white")
    draw_o(draw_a2, x1, "white")
    draw_x(draw_a2, x2, "yellow")

    canvas = Image.new("RGB", (out_w * 2, out_h), color=(0, 0, 0))
    canvas.paste(a1_img, (0, 0))
    canvas.paste(a2_img, (out_w, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), f"A1 Overlay  prompt={prompt}  sam={sam_score:.2f}", fill="white")
    draw.text((out_w + 12, 12), "A2 Overlay", fill="white")
    canvas.save(out_path)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processor = None
    coco_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    if args.mask_source == "sam3":
        processor = build_sam3_processor(args.device, args.sam3_checkpoint, args.threshold)
    model, cfg = load_pair_policy(args.checkpoint, args.device)

    image_paths = sorted([p for p in args.image_dir.rglob("*.png")])
    if args.num_images > 0:
        image_paths = image_paths[: args.num_images]
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {args.image_dir}")

    for image_path in image_paths:
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        if args.cropfit:
            rgb_input = rgb
            pil_image = Image.fromarray(rgb_input)
        else:
            if args.resize_mode == "stretch":
                rgb_input = stretch_to_square(rgb, args.square_size)
            else:
                rgb_input = pad_to_square(rgb, args.square_size)
            pil_image = Image.fromarray(rgb_input)
        prompt = _prompt_from_path(image_path, args.image_dir)

        if args.mask_source == "coco_gt":
            rel_path = image_path.relative_to(args.image_dir)
            rel_posix = rel_path.as_posix()
            parts = rel_path.parts
            if len(parts) < 4:
                print(f"[skip] {image_path.name} prompt={prompt} reason=expected split/location/category/file layout")
                continue
            split = str(parts[0])
            category = str(parts[2])
            cache_key = (split, category)
            if cache_key not in coco_cache:
                coco_path = args.dataset_root / f"{category}-{split}.json"
                if not coco_path.exists():
                    print(f"[skip] {image_path.name} prompt={prompt} reason=missing coco json {coco_path}")
                    continue
                coco_cache[cache_key] = build_coco_ann_lookup(load_coco(coco_path))
            ann = coco_cache[cache_key].get(rel_posix)
            if ann is None:
                print(f"[skip] {image_path.name} prompt={prompt} reason=no coco ann for {rel_posix}")
                continue
            mask = coco_segmentation_to_mask(ann.get("segmentation", []), rgb.shape[:2])
            score = 1.0
            if mask.sum() <= 0:
                print(f"[skip] {image_path.name} prompt={prompt} reason=empty coco mask")
                continue
        else:
            with torch.inference_mode():
                assert processor is not None
                state = processor.set_image(pil_image)
                text_outputs = processor.model.backbone.forward_text([prompt], device=args.device)
                state["backbone_out"].update(text_outputs)
                state["geometric_prompt"] = processor.model._get_dummy_prompt()
                sam_out = processor._forward_grounding(state)
            try:
                mask, box, score = choose_mask(sam_out)
            except RuntimeError as exc:
                print(f"[skip] {image_path.name} prompt={prompt} reason={exc}")
                continue

        if args.cropfit and not touches_border(mask.astype(np.uint8)):
            crop_rgb, crop_mask, crop_meta = crop_component_for_inference(
                rgb_input,
                mask.astype(np.uint8),
                target_mask_area_ratio=float(args.target_mask_area_ratio),
            )
        else:
            h, w = rgb_input.shape[:2]
            crop_rgb = rgb_input
            crop_mask = mask.astype(np.uint8)
            crop_meta = {
                "src_x0": 0,
                "src_y0": 0,
                "src_x1": w,
                "src_y1": h,
                "pad_left": 0,
                "pad_top": 0,
                "pad_right": 0,
                "pad_bottom": 0,
                "canvas_w": w,
                "canvas_h": h,
                "src_w": w,
                "src_h": h,
            }

        image_t, mask_resized = preprocess(crop_rgb, crop_mask, cfg)
        image_t = image_t.to(args.device)
        sampled_x1_xy = torch.zeros((1, 1, 2), dtype=torch.float32, device=args.device)
        sampled_x1_valid = torch.zeros((1, 1), dtype=torch.bool, device=args.device)

        with torch.inference_mode():
            out_a1 = model(image_t, sampled_x1_xy, sampled_x1_valid)
            a1_logits = out_a1["a1_logits"][0].detach().cpu().numpy()
            a1_pred = decode_heatmap(a1_logits, mask_resized, cfg, which="a1")
            x1_xy = argmax_xy(a1_pred, mask_resized)
            sampled_x1_xy[0, 0, 0] = float(x1_xy[0])
            sampled_x1_xy[0, 0, 1] = float(x1_xy[1])
            sampled_x1_valid[0, 0] = True
            out_a2 = model(image_t, sampled_x1_xy, sampled_x1_valid)
            a2_logits = out_a2["a2_logits"][0, 0].detach().cpu().numpy()
            a2_pred = decode_heatmap(a2_logits, mask_resized, cfg, which="a2")
            x2_xy = argmax_xy(a2_pred, mask_resized)

        out_h, out_w = rgb_input.shape[:2]
        canvas_w = int(crop_meta["canvas_w"])
        canvas_h = int(crop_meta["canvas_h"])
        src_x0 = int(crop_meta["src_x0"])
        src_y0 = int(crop_meta["src_y0"])
        src_x1 = int(crop_meta["src_x1"])
        src_y1 = int(crop_meta["src_y1"])
        pad_left = int(crop_meta["pad_left"])
        pad_top = int(crop_meta["pad_top"])
        src_w = int(crop_meta["src_w"])
        src_h = int(crop_meta["src_h"])

        a1_canvas = resize_heatmap(a1_pred, canvas_w, canvas_h)
        a2_canvas = resize_heatmap(a2_pred, canvas_w, canvas_h)
        a1_full = np.zeros((out_h, out_w), dtype=np.float32)
        a2_full = np.zeros((out_h, out_w), dtype=np.float32)
        a1_full[src_y0:src_y1, src_x0:src_x1] = a1_canvas[pad_top:pad_top + src_h, pad_left:pad_left + src_w]
        a2_full[src_y0:src_y1, src_x0:src_x1] = a2_canvas[pad_top:pad_top + src_h, pad_left:pad_left + src_w]

        x1_crop = map_xy(x1_xy, a1_pred.shape[1], a1_pred.shape[0], canvas_w, canvas_h)
        x2_crop = map_xy(x2_xy, a2_pred.shape[1], a2_pred.shape[0], canvas_w, canvas_h)
        x1_full = (int(src_x0) + x1_crop[0] - int(pad_left), int(src_y0) + x1_crop[1] - int(pad_top))
        x2_full = (int(src_x0) + x2_crop[0] - int(pad_left), int(src_y0) + x2_crop[1] - int(pad_top))
        rel_path = image_path.relative_to(args.image_dir)
        out_path = (args.out_dir / rel_path).with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_vis(
            out_path.with_name(out_path.name + "_sam3_pair_policy.png"),
            rgb_input,
            mask.astype(np.uint8),
            a1_full,
            a2_full,
            x1_full,
            x2_full,
            prompt,
            score,
            args.vis_mode,
            coords_in_full=True,
            vis_crop_mode=args.vis_crop_mode,
        )
        print(f"[ok] {rel_path} prompt={prompt} sam_score={score:.3f} x1={x1_xy} x2={x2_xy}")

    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
