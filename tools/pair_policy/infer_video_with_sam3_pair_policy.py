#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
SAM3_ROOT = Path("${WORKSPACE_ROOT}/sam3_test")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SAM3_ROOT / "upstream_sam3"))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from unfold.algorithms.pair_policy.model import PairPolicyNet
from unfold.algorithms.supervision.projection import masked_softmax_heatmap


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
JET_CMAP = cm.get_cmap("jet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 + pair-policy on a real video with multiple cloth instances."
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "test.mp4",
        help="Input video path.",
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
        default=REPO_ROOT / "experiments/pair_policy/runs/debug/test_video_sam3_pair_policy",
        help="Output directory.",
    )
    parser.add_argument("--prompt", default="cloth", help="SAM3 text prompt.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.2, help="SAM3 confidence threshold.")
    parser.add_argument("--sam-score-min", type=float, default=0.4, help="Minimum SAM score to keep.")
    parser.add_argument("--infer-stride", type=int, default=3, help="Run SAM3/pair-policy every Nth frame and reuse the last result in between.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum processed frames, 0 means all.")
    parser.add_argument("--min-component-area", type=int, default=1800, help="Minimum component area in 640x480.")
    parser.add_argument("--max-components", type=int, default=8, help="Max kept components per frame.")
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument(
        "--vis-mode",
        type=str,
        default="split",
        choices=("split", "dual"),
        help="Visualization mode: split (A1/A2 side-by-side) or dual (A1 red + A2 blue).",
    )
    parser.add_argument(
        "--train-index",
        type=Path,
        default=Path("${DATA_ROOT}/pair_policy_train_v2/index.json"),
        help="Training index used to estimate typical foreground occupancy.",
    )
    parser.add_argument(
        "--target-mask-area-ratio",
        type=float,
        default=0.0,
        help="Override target foreground occupancy after crop+pad. <=0 means estimate from training data.",
    )
    parser.add_argument(
        "--target-mask-area-quantile",
        type=float,
        default=0.5,
        help="Quantile of training mask occupancy used when estimating target size.",
    )
    parser.add_argument(
        "--target-mask-area-samples",
        type=int,
        default=256,
        help="Number of training masks sampled to estimate target occupancy.",
    )
    return parser.parse_args()


def build_sam3_processor(device: str, checkpoint: Path, threshold: float) -> Sam3Processor:
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        compile=False,
    )
    return Sam3Processor(model, device=device, confidence_threshold=threshold)


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


def extract_processed_frames(
    video_path: Path,
    out_dir: Path,
    max_frames: int,
    target_width: int,
    target_height: int,
) -> tuple[list[Path], float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    helper = f"""
from pathlib import Path
import json
import cv2

video = Path(r'''{video_path}''')
out_dir = Path(r'''{out_dir}''')
meta_path = Path(r'''{meta_path}''')
target_w = {target_width}
target_h = {target_height}
max_frames = {max_frames}

cap = cv2.VideoCapture(str(video))
fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
saved = 0
idx = 0
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        break
    h, w = frame.shape[:2]
    scale = target_h / float(h)
    resized_w = max(target_w, int(round(w * scale)))
    resized = cv2.resize(frame, (resized_w, target_h), interpolation=cv2.INTER_AREA)
    left = max(0, (resized_w - target_w) // 2)
    cropped = resized[:, left:left + target_w]
    if cropped.shape[1] != target_w:
        pad = target_w - cropped.shape[1]
        cropped = cv2.copyMakeBorder(cropped, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    out_path = out_dir / f"frame_{{saved:06d}}.png"
    cv2.imwrite(str(out_path), cropped)
    saved += 1
    idx += 1
    if max_frames > 0 and saved >= max_frames:
        break
cap.release()
meta_path.write_text(json.dumps({{"fps": fps, "saved": saved}}), encoding="utf-8")
"""
    subprocess.run(
        ["/usr/bin/python3", "-c", helper],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frame_paths = sorted(out_dir.glob("frame_*.png"))
    return frame_paths, float(meta["fps"])


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


def estimate_target_mask_area_ratio(index_path: Path, quantile: float, max_samples: int) -> float:
    if not index_path.exists():
        return 0.145
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    rows = payload.get("samples") or payload.get("rows") or []
    if not rows:
        return 0.145
    rng = random.Random(42)
    sampled_rows = rows if len(rows) <= max_samples else rng.sample(rows, max_samples)
    ratios: list[float] = []
    for row in sampled_rows:
        source = row.get("source") or {}
        mask_path = source.get("mask")
        if not mask_path:
            continue
        mask_file = Path(mask_path)
        if not mask_file.exists():
            continue
        mask = np.asarray(Image.open(mask_file))
        if mask.ndim == 3:
            mask = mask[..., 0]
        ratios.append(float((mask > 0).mean()))
    if not ratios:
        return 0.145
    q = float(np.clip(quantile, 0.0, 1.0))
    return float(np.quantile(np.asarray(ratios, dtype=np.float64), q))


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


def run_sam3(processor: Sam3Processor, rgb: np.ndarray, prompt: str) -> dict[str, np.ndarray]:
    pil_image = Image.fromarray(rgb)
    with torch.inference_mode():
        state = processor.set_image(pil_image)
        text_outputs = processor.model.backbone.forward_text([prompt], device=processor.device)
        state["backbone_out"].update(text_outputs)
        state["geometric_prompt"] = processor.model._get_dummy_prompt()
        output = processor._forward_grounding(state)
    return {
        "masks": output["masks"].squeeze(1).detach().float().cpu().numpy(),
        "boxes": output["boxes"].detach().float().cpu().numpy(),
        "scores": output["scores"].detach().float().cpu().numpy(),
    }


def touches_border(component: np.ndarray) -> bool:
    return bool(
        component[0, :].any()
        or component[-1, :].any()
        or component[:, 0].any()
        or component[:, -1].any()
    )


def sam_union_components(
    sam_out: dict[str, np.ndarray],
    score_min: float,
    min_component_area: int,
    max_components: int,
) -> tuple[np.ndarray, list[np.ndarray], list[int], list[float]]:
    masks = sam_out["masks"]
    scores = sam_out["scores"]
    height, width = masks.shape[-2:]
    keep = scores >= score_min
    if keep.sum() == 0 and len(scores) > 0:
        keep[np.argmax(scores)] = True
    kept_masks = masks[keep] > 0.5
    kept_scores = scores[keep].tolist()
    if kept_masks.size == 0:
        return np.zeros((height, width), dtype=bool), [], [], []

    union_mask = np.any(kept_masks, axis=0)
    labels, num = ndimage.label(union_mask.astype(np.uint8))
    if num == 0:
        return union_mask, [], [], kept_scores

    components: list[np.ndarray] = []
    areas: list[int] = []
    for label in range(1, num + 1):
        component = labels == label
        area = int(component.sum())
        if area < min_component_area or touches_border(component):
            continue
        components.append(component)
        areas.append(area)

    order = np.argsort(areas)[::-1][:max_components]
    components = [components[i] for i in order]
    areas = [areas[i] for i in order]
    return union_mask, components, areas, kept_scores


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


def resize_heatmap(heat: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(heat.astype(np.float32), mode="F").resize((out_w, out_h), resample=Image.BILINEAR))


def map_xy(xy: tuple[int, int], src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    x, y = xy
    if src_w <= 1 or src_h <= 1:
        return 0, 0
    x_dst = int(round(x * (dst_w - 1) / float(src_w - 1)))
    y_dst = int(round(y * (dst_h - 1) / float(src_h - 1)))
    return x_dst, y_dst


def infer_component(
    model: PairPolicyNet,
    cfg: dict,
    rgb: np.ndarray,
    component_mask: np.ndarray,
    device: str,
    target_mask_area_ratio: float,
) -> dict[str, np.ndarray | tuple[int, int]]:
    crop_rgb, crop_mask, crop_meta = crop_component_for_inference(
        rgb,
        component_mask.astype(np.uint8),
        target_mask_area_ratio=target_mask_area_ratio,
    )
    image_t, mask_resized = preprocess(crop_rgb, crop_mask, cfg)
    image_t = image_t.to(device)
    sampled_x1_xy = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)
    sampled_x1_valid = torch.zeros((1, 1), dtype=torch.bool, device=device)

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

    return {
        "a1_pred": a1_pred,
        "a2_pred": a2_pred,
        "mask_resized": mask_resized,
        "x1_xy": x1_xy,
        "x2_xy": x2_xy,
        "crop_meta": crop_meta,
    }


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


def overlay_component_heat(
    base_rgb: np.ndarray,
    component_mask: np.ndarray,
    heat: np.ndarray,
) -> np.ndarray:
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


def save_overlay_frame(
    out_path: Path,
    rgb: np.ndarray,
    components: list[np.ndarray],
    component_results: list[dict[str, np.ndarray | tuple[int, int]]],
    vis_mode: str,
) -> None:
    a1_panel = rgb.copy()
    a2_panel = rgb.copy()
    dual_panel = rgb.copy()
    out_h, out_w = rgb.shape[:2]

    for component_mask, result in zip(components, component_results):
        crop_meta = result["crop_meta"]
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

        a1_canvas = resize_heatmap(result["a1_pred"], canvas_w, canvas_h)
        a2_canvas = resize_heatmap(result["a2_pred"], canvas_w, canvas_h)
        a1_frame = np.zeros((out_h, out_w), dtype=np.float32)
        a2_frame = np.zeros((out_h, out_w), dtype=np.float32)
        a1_frame[src_y0:src_y1, src_x0:src_x1] = a1_canvas[pad_top:pad_top + src_h, pad_left:pad_left + src_w]
        a2_frame[src_y0:src_y1, src_x0:src_x1] = a2_canvas[pad_top:pad_top + src_h, pad_left:pad_left + src_w]
        if vis_mode == "dual":
            dual_panel = overlay_dual_heat(dual_panel, component_mask, a1_frame, a2_frame)
        else:
            a1_panel = overlay_component_heat(a1_panel, component_mask, a1_frame)
            a2_panel = overlay_component_heat(a2_panel, component_mask, a2_frame)

    if vis_mode == "dual":
        dual_img = Image.fromarray(dual_panel)
        draw_dual = ImageDraw.Draw(dual_img)
        for component_mask, result in zip(components, component_results):
            src_h, src_w = result["a1_pred"].shape
            crop_meta = result["crop_meta"]
            x1_crop = map_xy(result["x1_xy"], src_w, src_h, int(crop_meta["canvas_w"]), int(crop_meta["canvas_h"]))
            x2_crop = map_xy(result["x2_xy"], src_w, src_h, int(crop_meta["canvas_w"]), int(crop_meta["canvas_h"]))
            x1 = (
                int(crop_meta["src_x0"]) + x1_crop[0] - int(crop_meta["pad_left"]),
                int(crop_meta["src_y0"]) + x1_crop[1] - int(crop_meta["pad_top"]),
            )
            x2 = (
                int(crop_meta["src_x0"]) + x2_crop[0] - int(crop_meta["pad_left"]),
                int(crop_meta["src_y0"]) + x2_crop[1] - int(crop_meta["pad_top"]),
            )
            draw_x(draw_dual, x1, "white")
            draw_o(draw_dual, x1, "white")
            draw_x(draw_dual, x2, "yellow")
        draw_dual.text((12, 12), "A1(red) / A2(blue)", fill="white")
        dual_img.save(out_path)
        return

    a1_img = Image.fromarray(a1_panel)
    a2_img = Image.fromarray(a2_panel)
    draw_a1 = ImageDraw.Draw(a1_img)
    draw_a2 = ImageDraw.Draw(a2_img)
    for component_mask, result in zip(components, component_results):
        src_h, src_w = result["a1_pred"].shape
        crop_meta = result["crop_meta"]
        x1_crop = map_xy(result["x1_xy"], src_w, src_h, int(crop_meta["canvas_w"]), int(crop_meta["canvas_h"]))
        x2_crop = map_xy(result["x2_xy"], src_w, src_h, int(crop_meta["canvas_w"]), int(crop_meta["canvas_h"]))
        x1 = (
            int(crop_meta["src_x0"]) + x1_crop[0] - int(crop_meta["pad_left"]),
            int(crop_meta["src_y0"]) + x1_crop[1] - int(crop_meta["pad_top"]),
        )
        x2 = (
            int(crop_meta["src_x0"]) + x2_crop[0] - int(crop_meta["pad_left"]),
            int(crop_meta["src_y0"]) + x2_crop[1] - int(crop_meta["pad_top"]),
        )
        draw_x(draw_a1, x1, "white")
        draw_o(draw_a2, x1, "white")
        draw_x(draw_a2, x2, "yellow")

    canvas = Image.new("RGB", (out_w * 2, out_h), color=(0, 0, 0))
    canvas.paste(a1_img, (0, 0))
    canvas.paste(a2_img, (out_w, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "A1 Overlay", fill="white")
    draw.text((out_w + 12, 12), "A2 Overlay", fill="white")
    canvas.save(out_path)


def make_summary_video(frame_dir: Path, out_path: Path, fps: float) -> None:
    helper = f"""
from pathlib import Path
import cv2
frame_dir = Path(r'''{frame_dir}''')
out_path = Path(r'''{out_path}''')
frames = sorted(frame_dir.glob('frame_*.png'))
if not frames:
    raise SystemExit(0)
first = cv2.imread(str(frames[0]))
h, w = first.shape[:2]
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), float({fps}), (w, h))
for frame_path in frames:
    frame = cv2.imread(str(frame_path))
    writer.write(frame)
writer.release()
"""
    subprocess.run(
        ["/usr/bin/python3", "-c", helper],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_vis_dir = args.out_dir / "frames"
    frame_vis_dir.mkdir(parents=True, exist_ok=True)

    processor = build_sam3_processor(args.device, args.sam3_checkpoint, args.threshold)
    model, cfg = load_pair_policy(args.checkpoint, args.device)
    target_mask_area_ratio = (
        float(args.target_mask_area_ratio)
        if float(args.target_mask_area_ratio) > 0.0
        else estimate_target_mask_area_ratio(
            index_path=args.train_index,
            quantile=float(args.target_mask_area_quantile),
            max_samples=int(args.target_mask_area_samples),
        )
    )
    timing = {
        "sam3_ms": [],
        "pair_policy_ms": [],
        "components_per_infer": [],
    }

    with tempfile.TemporaryDirectory(prefix="pair_video_frames_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        frame_paths, fps = extract_processed_frames(
            video_path=args.video_path,
            out_dir=tmp_dir,
            max_frames=args.max_frames,
            target_width=args.target_width,
            target_height=args.target_height,
        )
        if not frame_paths:
            raise RuntimeError("No frames were extracted from the input video.")

        last_components: list[np.ndarray] = []
        last_component_results: list[dict[str, np.ndarray | tuple[int, int]]] = []
        for out_idx, frame_path in enumerate(frame_paths):
            rgb = np.asarray(Image.open(frame_path).convert("RGB"))
            should_infer = (out_idx % max(args.infer_stride, 1) == 0) or (not last_component_results)
            if should_infer:
                t0 = time.perf_counter()
                sam_out = run_sam3(processor, rgb, args.prompt)
                t1 = time.perf_counter()
                _, components, areas, kept_scores = sam_union_components(
                    sam_out=sam_out,
                    score_min=args.sam_score_min,
                    min_component_area=args.min_component_area,
                    max_components=args.max_components,
                )
                t2 = time.perf_counter()
                component_results = [
                    infer_component(
                        model,
                        cfg,
                        rgb,
                        component_mask,
                        args.device,
                        target_mask_area_ratio=target_mask_area_ratio,
                    )
                    for component_mask in components
                ]
                t3 = time.perf_counter()
                timing["sam3_ms"].append((t1 - t0) * 1000.0)
                timing["pair_policy_ms"].append((t3 - t2) * 1000.0)
                timing["components_per_infer"].append(len(components))
                last_components = components
                last_component_results = component_results
            else:
                components = last_components
                component_results = last_component_results
                areas = [int(c.sum()) for c in components]
                kept_scores = []
            save_overlay_frame(
                out_path=frame_vis_dir / f"frame_{out_idx:04d}.png",
                rgb=rgb,
                components=components,
                component_results=component_results,
                vis_mode=args.vis_mode,
            )
            print(
                f"[ok] {frame_path.name} inferred={int(should_infer)} kept_masks={len(kept_scores)} "
                f"components={len(components)} component_areas={areas}"
            )

    summary_mp4 = args.out_dir / "summary.mp4"
    make_summary_video(frame_vis_dir, summary_mp4, fps=fps)
    stats = {
        "video_path": str(args.video_path),
        "fps_out": fps,
        "infer_stride": args.infer_stride,
        "frames_out": len(frame_paths),
        "inference_frames": len(timing["sam3_ms"]),
        "sam3_ms_mean": float(np.mean(timing["sam3_ms"])) if timing["sam3_ms"] else 0.0,
        "sam3_ms_median": float(np.median(timing["sam3_ms"])) if timing["sam3_ms"] else 0.0,
        "pair_policy_ms_mean": float(np.mean(timing["pair_policy_ms"])) if timing["pair_policy_ms"] else 0.0,
        "pair_policy_ms_median": float(np.median(timing["pair_policy_ms"])) if timing["pair_policy_ms"] else 0.0,
        "components_per_infer_mean": float(np.mean(timing["components_per_infer"])) if timing["components_per_infer"] else 0.0,
        "components_per_infer_median": float(np.median(timing["components_per_infer"])) if timing["components_per_infer"] else 0.0,
        "target_mask_area_ratio": float(target_mask_area_ratio),
    }
    (args.out_dir / "timing_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
