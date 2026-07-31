#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SAM3_ROOT = Path("${WORKSPACE_ROOT}/sam3_test")

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SAM3_ROOT / "upstream_sam3"))

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
from unfold.algorithms.pair_policy.model import PairPolicyNet
from unfold.algorithms.supervision.projection import masked_softmax_heatmap

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate x1/x2 alignment with COCO keypoints on aRTFClothes.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("${DATASET_ROOT}/aRTF-Clothes-dataset/extracted/aRTFClothes"),
        help="aRTFClothes root containing train/test folders and COCO json files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        choices=("", "train", "test"),
        help="Single split to evaluate. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="",
        help="Comma-separated splits to evaluate. Example: train,test",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="shorts,tshirts,towels",
        help="Comma-separated categories to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "experiments/pair_policy/runs/train/segformer_mit_b4_exp_kl_tau01_sym_maxswap_nas_1gpu_plateau_earlystop_amp/best.pt",
        help="Pair-policy checkpoint.",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        default=SAM3_ROOT / "sam3.pt",
        help="Local SAM3 checkpoint.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mask-source",
        type=str,
        default="coco_gt",
        choices=("sam3", "coco_gt"),
        help="Use SAM3 predictions or dataset COCO polygons as the object mask.",
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--target-mask-area-ratio",
        type=float,
        default=0.145,
        help="Target foreground occupancy after crop+pad.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "experiments/aRTFClothes/analysis/pair_policy_keypoint_eval_full",
        help="Output directory for CSV and summary JSON.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0 means all.")
    parser.add_argument("--save-vis", action="store_true", help="Save a small set of visualizations.")
    parser.add_argument("--vis-limit", type=int, default=12, help="Max number of visualization samples to save per split.")
    return parser.parse_args()


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


def choose_mask(output: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray, float]:
    masks = output["masks"].squeeze(1).detach().float().cpu().numpy()
    boxes = output["boxes"].detach().float().cpu().numpy()
    scores = output["scores"].detach().float().cpu().numpy()
    if masks.size == 0:
        raise RuntimeError("SAM3 returned no masks.")
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    best_idx = int(np.argmax(scores + 1e-6 * areas))
    return masks[best_idx] > 0.5, boxes[best_idx], float(scores[best_idx])


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


def argmax_xy(heat: np.ndarray, valid: np.ndarray) -> tuple[int, int]:
    masked = np.where(valid > 0.5, heat, -np.inf)
    flat = int(np.argmax(masked))
    _, w = masked.shape
    y, x = divmod(flat, w)
    return int(x), int(y)


def resolve_splits(args: argparse.Namespace) -> list[str]:
    if args.splits:
        return [split.strip() for split in args.splits.split(",") if split.strip()]
    if args.split:
        return [args.split]
    return ["test"]


def run_split(
    split: str,
    samples: list[Any],
    *,
    args: argparse.Namespace,
    model: PairPolicyNet,
    cfg: dict,
    processor: Any | None,
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
        if args.mask_source == "coco_gt":
            mask = coco_segmentation_to_mask(sample.annotation.get("segmentation", []), rgb.shape[:2])
            score = 1.0
            if mask.sum() <= 0:
                print(f"[gt-mask-fail] {sample.category} {sample.file_name} err=empty COCO segmentation")
                continue
        else:
            try:
                assert processor is not None
                state = processor.set_image(Image.fromarray(rgb))
                text_outputs = processor.model.backbone.forward_text([sample.category], device=args.device)
                state["backbone_out"].update(text_outputs)
                state["geometric_prompt"] = processor.model._get_dummy_prompt()
                sam_out = processor._forward_grounding(state)
                mask, _, score = choose_mask(sam_out)
            except Exception as exc:
                print(f"[sam3-fail] {sample.category} {sample.file_name} err={exc}")
                continue

        if touches_border(mask.astype(np.uint8)):
            crop_rgb = rgb
            crop_mask = mask.astype(np.uint8)
            crop_meta = {
                "src_x0": 0,
                "src_y0": 0,
                "canvas_w": rgb.shape[1],
                "canvas_h": rgb.shape[0],
                "pad_left": 0,
                "pad_top": 0,
            }
        else:
            crop_rgb, crop_mask, crop_meta = crop_component_for_inference(
                rgb, mask.astype(np.uint8), target_mask_area_ratio=float(args.target_mask_area_ratio)
            )

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

        canvas_w = int(crop_meta["canvas_w"])
        canvas_h = int(crop_meta["canvas_h"])
        src_x0 = int(crop_meta["src_x0"])
        src_y0 = int(crop_meta["src_y0"])
        pad_left = int(crop_meta["pad_left"])
        pad_top = int(crop_meta["pad_top"])
        x1_crop = map_xy(x1_xy, a1_pred.shape[1], a1_pred.shape[0], canvas_w, canvas_h)
        x2_crop = map_xy(x2_xy, a2_pred.shape[1], a2_pred.shape[0], canvas_w, canvas_h)
        x1_full = (int(src_x0) + x1_crop[0] - pad_left, int(src_y0) + x1_crop[1] - pad_top)
        x2_full = (int(src_x0) + x2_crop[0] - pad_left, int(src_y0) + x2_crop[1] - pad_top)

        row = append_prediction_row(
            rows,
            model="pair_policy",
            sample=sample,
            score=float(score),
            x1_full=x1_full,
            x2_full=x2_full,
        )

        if args.save_vis and vis_count < int(args.vis_limit):
            out_path = vis_dir / f"{sample.category}_{sample.image_id}.png"
            draw_vis(out_path, rgb, sample.keypoints, x1_full, x2_full, row["nearest_kp_x1"], row["nearest_kp_x2"])
            vis_count += 1

    return rows


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = resolve_splits(args)
    categories = [c.strip().lower() for c in args.categories.split(",") if c.strip()]
    processor = None
    if args.mask_source == "sam3":
        processor = build_sam3_processor(args.device, args.sam3_checkpoint, args.threshold)
    model, cfg = load_pair_policy(args.checkpoint, args.device)

    all_samples = iter_artf_samples(args.dataset_root, splits, categories)
    all_rows: list[dict[str, Any]] = []
    for split in splits:
        split_samples = [sample for sample in all_samples if sample.split == split]
        split_rows = run_split(split, split_samples, args=args, model=model, cfg=cfg, processor=processor)
        all_rows.extend(split_rows)
        summary = summarize_rows(split_rows, "pair_policy", split)
        write_rows_csv(args.out_dir / f"predictions_{split}.csv", split_rows)
        write_summary(args.out_dir / f"summary_{split}.json", summary)
        write_split_plots(split_rows, summary, args.out_dir / f"plots_{split}")

    combined_summary = summarize_rows(all_rows, "pair_policy", "combined")
    write_rows_csv(args.out_dir / "predictions_combined.csv", all_rows)
    write_summary(args.out_dir / "summary_combined.json", combined_summary)
    write_split_plots(all_rows, combined_summary, args.out_dir / "plots_combined")
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
