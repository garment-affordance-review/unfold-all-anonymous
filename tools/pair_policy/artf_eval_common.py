from __future__ import annotations

import collections
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class ArtfSample:
    split: str
    category: str
    image_id: int
    file_name: str
    image_path: Path
    annotation: dict[str, Any]
    keypoints: list[tuple[str, float, float]]


def load_coco(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def index_annotations(coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    anns = {}
    for ann in coco.get("annotations", []):
        anns[int(ann["image_id"])] = ann
    return anns


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


def category_keypoint_names(coco: dict[str, Any], category_name: str) -> list[str]:
    aliases = {
        "tshirts": "tshirt",
        "boxershorts": "boxershorts",
        "shorts": "shorts",
        "towels": "towel",
    }
    target = aliases.get(str(category_name).lower(), str(category_name).lower())
    for cat in coco.get("categories", []):
        if str(cat.get("name", "")).lower() == target:
            return [str(x) for x in cat.get("keypoints", [])]
    return []


def keypoints_from_ann(ann: dict[str, Any], keypoint_names: list[str]) -> list[tuple[str, float, float]]:
    kps = ann.get("keypoints", []) or []
    pts = []
    for i in range(0, len(kps), 3):
        x, y, v = kps[i : i + 3]
        if v > 0:
            idx = i // 3
            name = keypoint_names[idx] if idx < len(keypoint_names) else f"kp_{idx}"
            pts.append((name, float(x), float(y)))
    return pts


def iter_artf_samples(dataset_root: Path, splits: Iterable[str], categories: Iterable[str]) -> list[ArtfSample]:
    samples: list[ArtfSample] = []
    for split in splits:
        for cat in categories:
            coco_path = dataset_root / f"{cat}-{split}.json"
            if not coco_path.exists():
                continue
            coco = load_coco(coco_path)
            ann_index = index_annotations(coco)
            kp_names = category_keypoint_names(coco, cat)
            for img in coco.get("images", []):
                image_id = int(img["id"])
                ann = ann_index.get(image_id)
                if ann is None:
                    continue
                pts = keypoints_from_ann(ann, kp_names)
                if not pts:
                    continue
                samples.append(
                    ArtfSample(
                        split=split,
                        category=cat,
                        image_id=image_id,
                        file_name=str(img["file_name"]),
                        image_path=dataset_root / str(img["file_name"]),
                        annotation=ann,
                        keypoints=pts,
                    )
                )
    return samples


def nearest_keypoint(point: tuple[int, int], pts: list[tuple[str, float, float]]) -> tuple[str, float]:
    if not pts:
        return "none", float("nan")
    x, y = point
    best_name = "none"
    best_d2 = float("inf")
    for name, px, py in pts:
        d2 = float((x - px) ** 2 + (y - py) ** 2)
        if d2 < best_d2:
            best_d2 = d2
            best_name = name
    return best_name, float(np.sqrt(best_d2))


def bbox_diag(ann: dict[str, Any]) -> float:
    bbox = ann.get("bbox", None)
    if not bbox or len(bbox) < 4:
        return float("nan")
    _, _, w, h = bbox[:4]
    return float(math.sqrt(float(w) ** 2 + float(h) ** 2))


def draw_vis(
    out_path: Path,
    rgb: np.ndarray,
    pts: list[tuple[str, float, float]],
    x1: tuple[int, int],
    x2: tuple[int, int],
    x1_name: str,
    x2_name: str,
) -> None:
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for name, px, py in pts:
        r = 4
        draw.ellipse((px - r, py - r, px + r, py + r), outline="lime", width=2)
        draw.text((px + 5, py - 5), name, fill="lime")
    draw.line((x1[0] - 6, x1[1] - 6, x1[0] + 6, x1[1] + 6), fill="white", width=2)
    draw.line((x1[0] - 6, x1[1] + 6, x1[0] + 6, x1[1] - 6), fill="white", width=2)
    draw.ellipse((x2[0] - 7, x2[1] - 7, x2[0] + 7, x2[1] + 7), outline="yellow", width=2)
    draw.text((x1[0] + 8, x1[1] + 8), f"x1->{x1_name}", fill="white")
    draw.text((x2[0] + 8, x2[1] + 8), f"x2->{x2_name}", fill="yellow")
    img.save(out_path)


def save_bar_chart(counter: collections.Counter, title: str, out_path: Path, top_k: int = 10) -> None:
    items = counter.most_common(top_k)
    if not items:
        return
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_hist(values: list[float], title: str, out_path: Path) -> None:
    if not values:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


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


def map_xy(xy: tuple[int, int], src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    x, y = xy
    if src_w <= 1 or src_h <= 1:
        return 0, 0
    x_dst = int(round(x * (dst_w - 1) / float(src_w - 1)))
    y_dst = int(round(y * (dst_h - 1) / float(src_h - 1)))
    return x_dst, y_dst


def append_prediction_row(
    rows: list[dict[str, Any]],
    *,
    model: str,
    sample: ArtfSample,
    score: float,
    x1_full: tuple[int, int],
    x2_full: tuple[int, int],
) -> dict[str, Any]:
    x1_name, d1 = nearest_keypoint(x1_full, sample.keypoints)
    x2_name, d2 = nearest_keypoint(x2_full, sample.keypoints)
    diag = bbox_diag(sample.annotation)
    row = {
        "model": model,
        "split": sample.split,
        "category": sample.category,
        "image_id": sample.image_id,
        "file_name": sample.file_name,
        "score": float(score),
        "x1": int(x1_full[0]),
        "y1": int(x1_full[1]),
        "x2": int(x2_full[0]),
        "y2": int(x2_full[1]),
        "nearest_kp_x1": x1_name,
        "nearest_kp_x2": x2_name,
        "min_dist_x1": float(d1),
        "min_dist_x2": float(d2),
        "norm_dist_x1": float(d1 / diag) if np.isfinite(diag) and diag > 1e-6 else float("nan"),
        "norm_dist_x2": float(d2 / diag) if np.isfinite(diag) and diag > 1e-6 else float("nan"),
    }
    rows.append(row)
    return row


def summarize_rows(rows: list[dict[str, Any]], model: str, split_name: str) -> dict[str, Any]:
    def _summarize(sub_rows: list[dict[str, Any]]) -> dict[str, Any]:
        nearest_x1 = collections.Counter(row["nearest_kp_x1"] for row in sub_rows)
        nearest_x2 = collections.Counter(row["nearest_kp_x2"] for row in sub_rows)
        pair_counter = collections.Counter(f"{row['nearest_kp_x1']} -> {row['nearest_kp_x2']}" for row in sub_rows)
        d1 = [float(row["min_dist_x1"]) for row in sub_rows]
        d2 = [float(row["min_dist_x2"]) for row in sub_rows]
        nd1 = [float(row["norm_dist_x1"]) for row in sub_rows if np.isfinite(float(row["norm_dist_x1"]))]
        nd2 = [float(row["norm_dist_x2"]) for row in sub_rows if np.isfinite(float(row["norm_dist_x2"]))]
        return {
            "images_used": len(sub_rows),
            "min_dist_x1_mean": float(np.nanmean(d1)) if d1 else float("nan"),
            "min_dist_x1_median": float(np.nanmedian(d1)) if d1 else float("nan"),
            "min_dist_x2_mean": float(np.nanmean(d2)) if d2 else float("nan"),
            "min_dist_x2_median": float(np.nanmedian(d2)) if d2 else float("nan"),
            "norm_dist_x1_mean": float(np.nanmean(nd1)) if nd1 else float("nan"),
            "norm_dist_x1_median": float(np.nanmedian(nd1)) if nd1 else float("nan"),
            "norm_dist_x2_mean": float(np.nanmean(nd2)) if nd2 else float("nan"),
            "norm_dist_x2_median": float(np.nanmedian(nd2)) if nd2 else float("nan"),
            "nearest_kp_x1_counts": dict(nearest_x1.most_common()),
            "nearest_kp_x2_counts": dict(nearest_x2.most_common()),
            "nearest_pair_counts": dict(pair_counter.most_common()),
        }

    categories = sorted({str(row["category"]) for row in rows})
    return {
        "model": model,
        "split": split_name,
        "total": _summarize(rows),
        "categories": {cat: _summarize([row for row in rows if row["category"] == cat]) for cat in categories},
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_split_plots(rows: list[dict[str, Any]], summary: dict[str, Any], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for cat, stats in summary.get("categories", {}).items():
        save_bar_chart(
            collections.Counter(stats["nearest_kp_x1_counts"]),
            f"{cat} x1 nearest keypoint",
            plots_dir / f"{cat}_x1_counts.png",
        )
        save_bar_chart(
            collections.Counter(stats["nearest_kp_x2_counts"]),
            f"{cat} x2 nearest keypoint",
            plots_dir / f"{cat}_x2_counts.png",
        )
        save_bar_chart(
            collections.Counter(stats["nearest_pair_counts"]),
            f"{cat} x1->x2 pair counts",
            plots_dir / f"{cat}_pair_counts.png",
        )
        cat_rows = [row for row in rows if row["category"] == cat]
        save_hist(
            [float(row["norm_dist_x1"]) for row in cat_rows if np.isfinite(float(row["norm_dist_x1"]))],
            f"{cat} normalized x1 distance",
            plots_dir / f"{cat}_norm_dist_x1.png",
        )
        save_hist(
            [float(row["norm_dist_x2"]) for row in cat_rows if np.isfinite(float(row["norm_dist_x2"]))],
            f"{cat} normalized x2 distance",
            plots_dir / f"{cat}_norm_dist_x2.png",
        )

