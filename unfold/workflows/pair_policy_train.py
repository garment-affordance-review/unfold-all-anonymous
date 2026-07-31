from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from unfold.algorithms.pair_policy.augment import GeometricAugmentConfig, MaskAugmentConfig, RGBAugmentConfig
from unfold.algorithms.pair_policy.dataset import PairPolicyDataset, collate_pair_policy_batch
from unfold.algorithms.pair_policy.index import build_pair_policy_index
from unfold.algorithms.pair_policy.model import PairPolicyNet
from unfold.algorithms.pair_policy.losses import pair_policy_loss
from unfold.algorithms.supervision.projection import masked_softmax_heatmap

try:
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
except Exception:
    pass


_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    if _is_distributed():
        return int(dist.get_rank())
    return 0


def _get_world_size() -> int:
    if _is_distributed():
        return int(dist.get_world_size())
    return 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _log_info(message: str) -> None:
    if _is_main_process():
        print(message, flush=True)


def _init_distributed(train_cfg: dict[str, Any]) -> tuple[torch.device, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_name = str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if world_size <= 1:
        return torch.device(device_name), local_rank

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requested via WORLD_SIZE>1 but CUDA is unavailable")
    backend = str((train_cfg.get("distributed", {}) or {}).get("backend", "nccl"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return torch.device(f"cuda:{local_rank}"), local_rank


def _cleanup_distributed() -> None:
    if _is_distributed():
        _dist_barrier()
        dist.destroy_process_group()


def _dist_barrier() -> None:
    if not _is_distributed():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def _maybe_init_wandb(cfg: dict[str, Any], out_dir: Path):
    if not _is_main_process():
        return None
    wb_cfg = cfg.get("wandb", {}) or {}
    if not bool(wb_cfg.get("enabled", False)):
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] wandb requested but unavailable: {exc}")
        return None

    settings = dict(
        project=wb_cfg.get("project"),
        entity=wb_cfg.get("entity"),
        name=wb_cfg.get("name"),
        tags=wb_cfg.get("tags"),
        mode=wb_cfg.get("mode"),
        dir=str(out_dir),
        config=cfg,
        resume=wb_cfg.get("resume", False),
    )
    if not settings["mode"] and not os.environ.get("WANDB_API_KEY"):
        print("[WARN] wandb enabled but WANDB_API_KEY is not set; falling back to offline mode")
        settings["mode"] = "offline"
    run = wandb.init(**{k: v for k, v in settings.items() if v is not None})
    return run

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.full_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be mapping: {path}")
    return cfg


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch, device: torch.device):
    names = [
        "image",
        "input_mask",
        "gt_mask",
        "a1_target",
        "a1_target_mask",
        "a1_valid_mask",
        "sampled_x1_xy",
        "sampled_x1_valid",
        "a2_target",
        "a2_target_mask",
        "a2_valid_mask",
        "a2_target_valid",
        "mask_area_ratio",
    ]
    if hasattr(batch, "a1_negative_mask"):
        names.append("a1_negative_mask")
    if hasattr(batch, "a2_negative_mask"):
        names.append("a2_negative_mask")
    for name in names:
        setattr(batch, name, getattr(batch, name).to(device))
    return batch


def _rank_enabled(cfg: dict[str, Any]) -> bool:
    loss_cfg = cfg.get("loss", {}) or {}
    return float(loss_cfg.get("lambda_rank_a1", 0.0)) > 0.0 or float(loss_cfg.get("lambda_rank_a2", 0.0)) > 0.0


def _distribution_mode(cfg: dict[str, Any]) -> bool:
    return str((cfg.get("loss", {}) or {}).get("name", "masked_kl")) == "masked_kl"


def _build_datasets(cfg: dict[str, Any], out_dir: Path):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    index_path = Path(data_cfg.get("index_cache", out_dir / "pair_policy_index.json")).resolve()
    rebuild_index = bool(data_cfg.get("rebuild_index", False))

    if _is_distributed():
        if _is_main_process():
            if index_path.exists() and not rebuild_index:
                _log_info(f"[INFO] Using existing training index cache: {index_path}")
            else:
                _log_info(f"[INFO] Preparing training index cache: {index_path}")
            index_payload = build_pair_policy_index(
                supervision_index_path=data_cfg["supervision_index"],
                out_path=index_path,
                val_ratio=float(data_cfg.get("val_ratio", 0.2)),
                seed=int(data_cfg.get("split_seed", 42)),
                rebuild=rebuild_index,
                progress_every=int(data_cfg.get("index_progress_every", 1000)),
            )
            _dist_barrier()
        else:
            _dist_barrier()
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            print(f"[INFO] Rank {_get_rank()} loaded training index cache: {index_path}", flush=True)
    else:
        if index_path.exists() and not rebuild_index:
            print(f"[INFO] Using existing training index cache: {index_path}", flush=True)
        else:
            print(f"[INFO] Preparing training index cache: {index_path}", flush=True)
        index_payload = build_pair_policy_index(
            supervision_index_path=data_cfg["supervision_index"],
            out_path=index_path,
            val_ratio=float(data_cfg.get("val_ratio", 0.2)),
            seed=int(data_cfg.get("split_seed", 42)),
            rebuild=rebuild_index,
            progress_every=int(data_cfg.get("index_progress_every", 1000)),
        )
    geom_aug = GeometricAugmentConfig(**cfg.get("geometry_augment", {}))
    mask_aug = MaskAugmentConfig(**cfg.get("mask_augment", {}))
    rgb_aug = RGBAugmentConfig(**cfg.get("rgb_augment", {}))
    filter_cfg = data_cfg.get("filter", {}) or {}
    common = dict(
        index_payload=index_payload,
        num_x1_samples=int(train_cfg.get("num_x1_samples", 4)),
        target_name=str((cfg.get("target", {}) or {}).get("name", "masked_softmax")),
        a1_target_tau=float(train_cfg.get("a1_target_tau", 1.0)),
        a2_target_tau=float(train_cfg.get("a2_target_tau", 1.0)),
        top_ratio=float(((cfg.get("target", {}) or {}).get("top_ratio", 0.15))),
        bottom_ratio=float(((cfg.get("target", {}) or {}).get("bottom_ratio", 0.5))),
        resize_width=(int(data_cfg["resize_width"]) if data_cfg.get("resize_width") else None),
        resize_height=(int(data_cfg["resize_height"]) if data_cfg.get("resize_height") else None),
        input_normalization=str((data_cfg.get("input_normalization", "none"))),
        supervision_mask_mode=str((data_cfg.get("supervision_mask_mode", "heatmap_valid"))),
        loss_name=str((cfg.get("loss", {}) or {}).get("name", "masked_kl")),
        min_a1_std=float(filter_cfg.get("min_a1_std", 0.0)),
        min_a1_margin=float(filter_cfg.get("min_a1_top1_margin", 0.0)),
        min_reward_row_margin=float(filter_cfg.get("min_reward_row_top1_margin", 0.0)),
        seed=int(train_cfg.get("seed", 42)),
    )
    ds_train = PairPolicyDataset(
        split="train",
        train=True,
        geom_aug_cfg=geom_aug,
        mask_aug_cfg=mask_aug,
        rgb_aug_cfg=rgb_aug,
        **common,
    )
    ds_val = PairPolicyDataset(
        split="val",
        train=False,
        geom_aug_cfg=GeometricAugmentConfig(enabled=False),
        mask_aug_cfg=MaskAugmentConfig(enabled=False),
        rgb_aug_cfg=RGBAugmentConfig(enabled=False),
        **common,
    )
    return ds_train, ds_val, index_payload


def _make_loader(
    dataset: PairPolicyDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    *,
    sampler: Any | None = None,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    multiprocessing_context: str | None = None,
) -> DataLoader:
    kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(bool(shuffle) if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_pair_policy_batch,
        pin_memory=bool(pin_memory),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
        if multiprocessing_context:
            kwargs["multiprocessing_context"] = str(multiprocessing_context)
    return DataLoader(**kwargs)


def _save_visual_batch(*, out_dir: Path, batch, outputs: dict[str, torch.Tensor], dataset: PairPolicyDataset) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_np = batch.image[0].detach().cpu().permute(1, 2, 0).numpy()
    if getattr(dataset, "input_normalization", "none") == "imagenet":
        image_np = image_np * _IMAGENET_STD[None, None, :] + _IMAGENET_MEAN[None, None, :]
    image_np = np.clip(image_np, 0.0, 1.0)
    input_mask = batch.input_mask[0].detach().cpu().numpy()
    a1_valid = batch.a1_target_mask[0].detach().cpu().numpy() > 0.5
    a1_gt = batch.a1_target[0].detach().cpu().numpy()
    a1_logits = outputs["a1_logits"][0].detach().cpu().numpy()
    if dataset.target_name in {"masked_softmax", "topk_masked_softmax", "image_exp"}:
        a1_gt = masked_softmax_heatmap(np.log(np.clip(a1_gt, 1e-12, None)), a1_valid, tau=1.0)
    loss_name = getattr(dataset, "loss_name", "masked_kl")
    if dataset.target_name in {"masked_softmax", "topk_masked_softmax", "image_exp"}:
        a1_pred = masked_softmax_heatmap(a1_logits, a1_valid, tau=float(dataset.a1_target_tau))
    elif loss_name == "masked_huber_raw":
        a1_pred = np.clip(a1_logits, 0.0, 1.0)
        a1_pred[~a1_valid] = 0.0
    else:
        a1_pred = torch.sigmoid(torch.from_numpy(a1_logits)).cpu().numpy()
        a1_pred[~a1_valid] = 0.0
    valid_x1 = batch.sampled_x1_valid[0].detach().cpu().numpy().astype(bool)
    query_idx = int(np.flatnonzero(valid_x1)[0]) if np.any(valid_x1) else 0
    x1_xy = None
    a2_valid = batch.a2_target_mask[0, query_idx].detach().cpu().numpy() > 0.5
    a2_heat_gt = batch.a2_target[0, query_idx].detach().cpu().numpy()
    a2_logits = outputs["a2_logits"][0, query_idx].detach().cpu().numpy()
    if dataset.target_name in {"masked_softmax", "topk_masked_softmax", "image_exp"}:
        a2_heat_gt = masked_softmax_heatmap(np.log(np.clip(a2_heat_gt, 1e-12, None)), a2_valid, tau=1.0)
    if dataset.target_name in {"masked_softmax", "topk_masked_softmax", "image_exp"}:
        a2_heat_pred = masked_softmax_heatmap(a2_logits, a2_valid, tau=float(dataset.a2_target_tau))
    elif loss_name == "masked_huber_raw":
        a2_heat_pred = np.clip(a2_logits, 0.0, 1.0)
        a2_heat_pred[~a2_valid] = 0.0
    else:
        a2_heat_pred = torch.sigmoid(torch.from_numpy(a2_logits)).cpu().numpy()
        a2_heat_pred[~a2_valid] = 0.0
    if bool(batch.sampled_x1_valid[0, query_idx].detach().cpu().item()):
        x1_xy = batch.sampled_x1_xy[0, query_idx].detach().cpu().numpy()

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
        out[~valid_mask] = np.nan
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

    a1_gt_vis = _display_heat(a1_gt, a1_valid)
    a1_pred_vis = _display_heat(a1_pred, a1_valid)
    a2_gt_vis = _display_heat(a2_heat_gt, a2_valid)
    a2_pred_vis = _display_heat(a2_heat_pred, a2_valid)
    a1_gt_xy = _argmax_xy(a1_gt, a1_valid)
    a1_pred_xy = _argmax_xy(a1_pred, a1_valid)
    a2_gt_xy = _argmax_xy(a2_heat_gt, a2_valid)
    a2_pred_xy = _argmax_xy(a2_heat_pred, a2_valid)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    if dataset.target_name in {"masked_softmax", "topk_masked_softmax"}:
        value_label = "Dist"
    elif dataset.target_name == "image_exp":
        value_label = "Exp Heatmap"
    else:
        value_label = "Heatmap"
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Input RGB")
    axes[0, 1].imshow(input_mask, cmap="gray")
    axes[0, 1].set_title("Perturbed Mask")
    axes[0, 2].imshow(a1_gt_vis, cmap="jet")
    axes[0, 2].set_title(f"A1 GT {value_label} (scaled)")
    axes[1, 0].imshow(a1_pred_vis, cmap="jet")
    axes[1, 0].set_title(f"A1 Pred {value_label} (scaled)")
    axes[1, 1].imshow(a2_gt_vis, cmap="jet")
    axes[1, 1].set_title(f"A2 GT {value_label} (scaled)")
    axes[1, 2].imshow(a2_pred_vis, cmap="jet")
    axes[1, 2].set_title(f"A2 Pred {value_label} (scaled)")
    for ax in axes.ravel():
        ax.axis("off")
    _draw_x(axes[0, 2], a1_gt_xy, color="magenta")
    _draw_x(axes[1, 0], a1_pred_xy, color="white")
    _draw_ring(axes[1, 1], x1_xy, color="white")
    _draw_ring(axes[1, 2], x1_xy, color="white")
    _draw_x(axes[1, 1], a2_gt_xy, color="yellow")
    _draw_x(axes[1, 2], a2_pred_xy, color="yellow")
    fig.tight_layout()
    fig.savefig(out_dir / "qualitative.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _sample_visual_batch(
    *,
    dataset: PairPolicyDataset,
    device: torch.device,
    model,
    out_dir: Path,
    rng: random.Random,
) -> Path:
    idx = rng.randrange(len(dataset))
    batch = collate_pair_policy_batch([dataset[idx]])
    batch = _to_device(batch, device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(
            image=batch.image,
            sampled_x1_xy=batch.sampled_x1_xy,
            sampled_x1_valid=batch.sampled_x1_valid,
        )
    _save_visual_batch(out_dir=out_dir, batch=batch, outputs=outputs, dataset=dataset)
    if was_training:
        model.train(True)
    return out_dir / "qualitative.png"


def _log_wandb_epoch(
    run,
    *,
    cfg: dict[str, Any],
    epoch: int,
    global_step: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    train_vis_path: Path | None,
    val_vis_path: Path | None,
) -> None:
    if run is None:
        return
    payload = {
        "epoch": int(epoch),
        "train/loss": float(train_metrics["loss"]),
        "train/loss_a1": float(train_metrics["loss_a1"]),
        "train/loss_a2": float(train_metrics["loss_a2"]),
        "val/loss": float(val_metrics["loss"]),
        "val/loss_a1": float(val_metrics["loss_a1"]),
        "val/loss_a2": float(val_metrics["loss_a2"]),
    }
    if _distribution_mode(cfg):
        payload.update(
            {
                "train/mass_top_a1": float(train_metrics["mass_top_a1"]),
                "train/mass_top_a2": float(train_metrics["mass_top_a2"]),
                "train/argmax_top_a1": float(train_metrics["argmax_top_a1"]),
                "train/argmax_top_a2": float(train_metrics["argmax_top_a2"]),
                "train/pred_entropy_a1": float(train_metrics["pred_entropy_a1"]),
                "train/target_entropy_a1": float(train_metrics["target_entropy_a1"]),
                "train/pred_entropy_a2": float(train_metrics["pred_entropy_a2"]),
                "train/target_entropy_a2": float(train_metrics["target_entropy_a2"]),
                "val/mass_top_a1": float(val_metrics["mass_top_a1"]),
                "val/mass_top_a2": float(val_metrics["mass_top_a2"]),
                "val/argmax_top_a1": float(val_metrics["argmax_top_a1"]),
                "val/argmax_top_a2": float(val_metrics["argmax_top_a2"]),
                "val/pred_entropy_a1": float(val_metrics["pred_entropy_a1"]),
                "val/target_entropy_a1": float(val_metrics["target_entropy_a1"]),
                "val/pred_entropy_a2": float(val_metrics["pred_entropy_a2"]),
                "val/target_entropy_a2": float(val_metrics["target_entropy_a2"]),
            }
        )
    else:
        payload.update(
            {
                "train/mae_a1": float(train_metrics["mae_a1"]),
                "train/mae_a2": float(train_metrics["mae_a2"]),
                "val/mae_a1": float(val_metrics["mae_a1"]),
                "val/mae_a2": float(val_metrics["mae_a2"]),
            }
        )
    if float(train_metrics["loss_rank_a1"]) > 0.0 or float(train_metrics["loss_rank_a2"]) > 0.0:
        payload["train/loss_rank_a1"] = float(train_metrics["loss_rank_a1"])
        payload["train/loss_rank_a2"] = float(train_metrics["loss_rank_a2"])
        payload["val/loss_rank_a1"] = float(val_metrics["loss_rank_a1"])
        payload["val/loss_rank_a2"] = float(val_metrics["loss_rank_a2"])
    if train_vis_path is not None and train_vis_path.exists():
        try:
            import wandb  # type: ignore
            payload["visuals/train_qualitative"] = wandb.Image(str(train_vis_path))
        except Exception:
            pass
    if val_vis_path is not None and val_vis_path.exists():
        try:
            import wandb  # type: ignore
            payload["visuals/val_qualitative"] = wandb.Image(str(val_vis_path))
        except Exception:
            pass
    run.log(payload, step=int(global_step))


def _log_wandb_step(run, *, phase: str, global_step: int, metrics: dict[str, float]) -> None:
    if run is None:
        return
    payload = {
        f"{phase}/loss": float(metrics["loss"]),
        f"{phase}/loss_a1": float(metrics["loss_a1"]),
        f"{phase}/loss_a2": float(metrics["loss_a2"]),
    }
    if phase.startswith("train_step") and (
        float(metrics["pred_entropy_a1"]) > 0.0
        or float(metrics["pred_entropy_a2"]) > 0.0
        or float(metrics["target_entropy_a1"]) > 0.0
        or float(metrics["target_entropy_a2"]) > 0.0
    ):
        payload.update(
            {
                f"{phase}/mass_top_a1": float(metrics["mass_top_a1"]),
                f"{phase}/mass_top_a2": float(metrics["mass_top_a2"]),
                f"{phase}/argmax_top_a1": float(metrics["argmax_top_a1"]),
                f"{phase}/argmax_top_a2": float(metrics["argmax_top_a2"]),
                f"{phase}/pred_entropy_a1": float(metrics["pred_entropy_a1"]),
                f"{phase}/target_entropy_a1": float(metrics["target_entropy_a1"]),
                f"{phase}/pred_entropy_a2": float(metrics["pred_entropy_a2"]),
                f"{phase}/target_entropy_a2": float(metrics["target_entropy_a2"]),
            }
        )
    elif float(metrics["mae_a1"]) > 0.0 or float(metrics["mae_a2"]) > 0.0:
        payload.update(
            {
                f"{phase}/mae_a1": float(metrics["mae_a1"]),
                f"{phase}/mae_a2": float(metrics["mae_a2"]),
            }
        )
    if float(metrics["loss_rank_a1"]) > 0.0 or float(metrics["loss_rank_a2"]) > 0.0:
        payload[f"{phase}/loss_rank_a1"] = float(metrics["loss_rank_a1"])
        payload[f"{phase}/loss_rank_a2"] = float(metrics["loss_rank_a2"])
    run.log(payload, step=int(global_step))


def _build_scheduler(optimizer, cfg: dict[str, Any]):
    sched_cfg = (cfg.get("train", {}) or {}).get("lr_scheduler", {}) or {}
    if not bool(sched_cfg.get("enabled", False)):
        return None
    name = str(sched_cfg.get("name", "plateau")).lower()
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(sched_cfg.get("mode", "min")),
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 5)),
            threshold=float(sched_cfg.get("threshold", 1e-4)),
            threshold_mode=str(sched_cfg.get("threshold_mode", "rel")),
            cooldown=int(sched_cfg.get("cooldown", 0)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )
    raise ValueError(f"unsupported lr scheduler: {name}")


def _build_amp_context(cfg: dict[str, Any], device: torch.device):
    amp_cfg = (cfg.get("train", {}) or {}).get("amp", {}) or {}
    enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    dtype_name = str(amp_cfg.get("dtype", "bfloat16")).lower()
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"unsupported amp dtype: {dtype_name}")
    return enabled, dtype


def _reduce_epoch_totals(totals: dict[str, float], device: torch.device) -> dict[str, float]:
    if not _is_distributed():
        return totals
    packed = torch.tensor(
        [
            float(totals["loss"]),
            float(totals["loss_a1"]),
            float(totals["loss_a2"]),
            float(totals["loss_rank_a1"]),
            float(totals["loss_rank_a2"]),
            float(totals["mae_a1"]),
            float(totals["mae_a2"]),
            float(totals["pred_entropy_a1"]),
            float(totals["target_entropy_a1"]),
            float(totals["pred_entropy_a2"]),
            float(totals["target_entropy_a2"]),
            float(totals["mass_top_a1"]),
            float(totals["mass_top_a2"]),
            float(totals["argmax_top_a1"]),
            float(totals["argmax_top_a2"]),
            float(totals["steps"]),
            float(totals["data_sec"]),
            float(totals["compute_sec"]),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {
        "loss": float(packed[0].item()),
        "loss_a1": float(packed[1].item()),
        "loss_a2": float(packed[2].item()),
        "loss_rank_a1": float(packed[3].item()),
        "loss_rank_a2": float(packed[4].item()),
        "mae_a1": float(packed[5].item()),
        "mae_a2": float(packed[6].item()),
        "pred_entropy_a1": float(packed[7].item()),
        "target_entropy_a1": float(packed[8].item()),
        "pred_entropy_a2": float(packed[9].item()),
        "target_entropy_a2": float(packed[10].item()),
        "mass_top_a1": float(packed[11].item()),
        "mass_top_a2": float(packed[12].item()),
        "argmax_top_a1": float(packed[13].item()),
        "argmax_top_a2": float(packed[14].item()),
        "steps": float(packed[15].item()),
        "data_sec": float(packed[16].item()),
        "compute_sec": float(packed[17].item()),
    }


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    cfg,
    train: bool,
    phase: str,
    wandb_run=None,
    step_offset: int = 0,
) -> dict[str, float]:
    totals = {
        "loss": 0.0,
        "loss_a1": 0.0,
        "loss_a2": 0.0,
        "loss_rank_a1": 0.0,
        "loss_rank_a2": 0.0,
        "mae_a1": 0.0,
        "mae_a2": 0.0,
        "pred_entropy_a1": 0.0,
        "target_entropy_a1": 0.0,
        "pred_entropy_a2": 0.0,
        "target_entropy_a2": 0.0,
        "mass_top_a1": 0.0,
        "mass_top_a2": 0.0,
        "argmax_top_a1": 0.0,
        "argmax_top_a2": 0.0,
        "steps": 0,
        "data_sec": 0.0,
        "compute_sec": 0.0,
    }
    recent = {
        "loss": 0.0,
        "loss_a1": 0.0,
        "loss_a2": 0.0,
        "loss_rank_a1": 0.0,
        "loss_rank_a2": 0.0,
        "mae_a1": 0.0,
        "mae_a2": 0.0,
        "pred_entropy_a1": 0.0,
        "target_entropy_a1": 0.0,
        "pred_entropy_a2": 0.0,
        "target_entropy_a2": 0.0,
        "mass_top_a1": 0.0,
        "mass_top_a2": 0.0,
        "argmax_top_a1": 0.0,
        "argmax_top_a2": 0.0,
        "steps": 0,
        "data_sec": 0.0,
        "compute_sec": 0.0,
    }
    model.train(train)
    max_steps = int(cfg["train"].get("max_steps_per_epoch", 0))
    total_steps = min(len(loader), max_steps) if max_steps > 0 else len(loader)
    log_every = max(1, int(cfg["train"].get("log_every_steps", 200)))
    amp_enabled, amp_dtype = _build_amp_context(cfg, device)
    phase_t0 = time.perf_counter()
    batch_iter = iter(loader)
    while True:
        data_t0 = time.perf_counter()
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        data_sec = time.perf_counter() - data_t0

        compute_t0 = time.perf_counter()
        batch = _to_device(batch, device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    image=batch.image,
                    sampled_x1_xy=batch.sampled_x1_xy,
                    sampled_x1_valid=batch.sampled_x1_valid,
                )
                loss_dict = pair_policy_loss(
                    outputs,
                    batch,
                    lambda_a1=float(cfg["loss"].get("lambda_a1", 1.0)),
                    lambda_a2=float(cfg["loss"].get("lambda_a2", 1.0)),
                    lambda_rank_a1=float(cfg["loss"].get("lambda_rank_a1", 0.0)),
                    lambda_rank_a2=float(cfg["loss"].get("lambda_rank_a2", 0.0)),
                    rank_margin=float(cfg["loss"].get("rank_margin", 0.02)),
                    huber_delta=float(cfg["loss"].get("huber_delta", 0.1)),
                    weighted_huber_alpha=float(cfg["loss"].get("weighted_huber_alpha", 4.0)),
                    weighted_huber_gamma=float(cfg["loss"].get("weighted_huber_gamma", 1.0)),
                    diagnostic_top_ratio=float(cfg["loss"].get("diagnostic_top_ratio", 0.1)),
                    loss_name=str(cfg["loss"].get("name", "masked_kl")),
                )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss_dict["loss"].backward()
                optimizer.step()
        compute_sec = time.perf_counter() - compute_t0
        totals["loss"] += float(loss_dict["loss"].detach().cpu())
        totals["loss_a1"] += float(loss_dict["loss_a1"].cpu())
        totals["loss_a2"] += float(loss_dict["loss_a2"].cpu())
        totals["loss_rank_a1"] += float(loss_dict["loss_rank_a1"].cpu())
        totals["loss_rank_a2"] += float(loss_dict["loss_rank_a2"].cpu())
        totals["mae_a1"] += float(loss_dict["mae_a1"].cpu())
        totals["mae_a2"] += float(loss_dict["mae_a2"].cpu())
        totals["pred_entropy_a1"] += float(loss_dict["pred_entropy_a1"].cpu())
        totals["target_entropy_a1"] += float(loss_dict["target_entropy_a1"].cpu())
        totals["pred_entropy_a2"] += float(loss_dict["pred_entropy_a2"].cpu())
        totals["target_entropy_a2"] += float(loss_dict["target_entropy_a2"].cpu())
        totals["mass_top_a1"] += float(loss_dict["mass_top_a1"].cpu())
        totals["mass_top_a2"] += float(loss_dict["mass_top_a2"].cpu())
        totals["argmax_top_a1"] += float(loss_dict["argmax_top_a1"].cpu())
        totals["argmax_top_a2"] += float(loss_dict["argmax_top_a2"].cpu())
        totals["steps"] += 1
        totals["data_sec"] += float(data_sec)
        totals["compute_sec"] += float(compute_sec)
        recent["loss"] += float(loss_dict["loss"].detach().cpu())
        recent["loss_a1"] += float(loss_dict["loss_a1"].cpu())
        recent["loss_a2"] += float(loss_dict["loss_a2"].cpu())
        recent["loss_rank_a1"] += float(loss_dict["loss_rank_a1"].cpu())
        recent["loss_rank_a2"] += float(loss_dict["loss_rank_a2"].cpu())
        recent["mae_a1"] += float(loss_dict["mae_a1"].cpu())
        recent["mae_a2"] += float(loss_dict["mae_a2"].cpu())
        recent["pred_entropy_a1"] += float(loss_dict["pred_entropy_a1"].cpu())
        recent["target_entropy_a1"] += float(loss_dict["target_entropy_a1"].cpu())
        recent["pred_entropy_a2"] += float(loss_dict["pred_entropy_a2"].cpu())
        recent["target_entropy_a2"] += float(loss_dict["target_entropy_a2"].cpu())
        recent["mass_top_a1"] += float(loss_dict["mass_top_a1"].cpu())
        recent["mass_top_a2"] += float(loss_dict["mass_top_a2"].cpu())
        recent["argmax_top_a1"] += float(loss_dict["argmax_top_a1"].cpu())
        recent["argmax_top_a2"] += float(loss_dict["argmax_top_a2"].cpu())
        recent["steps"] += 1
        recent["data_sec"] += float(data_sec)
        recent["compute_sec"] += float(compute_sec)
        if totals["steps"] % log_every == 0 or totals["steps"] == total_steps:
            elapsed = time.perf_counter() - phase_t0
            steps_done = totals["steps"]
            steps_left = max(total_steps - steps_done, 0)
            sec_per_step = elapsed / max(steps_done, 1)
            eta_sec = steps_left * sec_per_step
            recent_steps = max(int(recent["steps"]), 1)
            avg_metrics = {
                "loss": recent["loss"] / recent_steps,
                "loss_a1": recent["loss_a1"] / recent_steps,
                "loss_a2": recent["loss_a2"] / recent_steps,
                "loss_rank_a1": recent["loss_rank_a1"] / recent_steps,
                "loss_rank_a2": recent["loss_rank_a2"] / recent_steps,
                "mae_a1": recent["mae_a1"] / recent_steps,
                "mae_a2": recent["mae_a2"] / recent_steps,
                    "pred_entropy_a1": recent["pred_entropy_a1"] / recent_steps,
                    "target_entropy_a1": recent["target_entropy_a1"] / recent_steps,
                    "pred_entropy_a2": recent["pred_entropy_a2"] / recent_steps,
                    "target_entropy_a2": recent["target_entropy_a2"] / recent_steps,
                    "mass_top_a1": recent["mass_top_a1"] / recent_steps,
                    "mass_top_a2": recent["mass_top_a2"] / recent_steps,
                    "argmax_top_a1": recent["argmax_top_a1"] / recent_steps,
                    "argmax_top_a2": recent["argmax_top_a2"] / recent_steps,
                    "data_sec": recent["data_sec"] / recent_steps,
                    "compute_sec": recent["compute_sec"] / recent_steps,
                }
            log_metrics = avg_metrics
            if _is_distributed():
                reduced = _reduce_epoch_totals(recent, device)
                reduced_steps = max(int(reduced["steps"]), 1)
                log_metrics = {
                    "loss": reduced["loss"] / reduced_steps,
                    "loss_a1": reduced["loss_a1"] / reduced_steps,
                    "loss_a2": reduced["loss_a2"] / reduced_steps,
                    "loss_rank_a1": reduced["loss_rank_a1"] / reduced_steps,
                    "loss_rank_a2": reduced["loss_rank_a2"] / reduced_steps,
                    "mae_a1": reduced["mae_a1"] / reduced_steps,
                    "mae_a2": reduced["mae_a2"] / reduced_steps,
                    "pred_entropy_a1": reduced["pred_entropy_a1"] / reduced_steps,
                    "target_entropy_a1": reduced["target_entropy_a1"] / reduced_steps,
                    "pred_entropy_a2": reduced["pred_entropy_a2"] / reduced_steps,
                    "target_entropy_a2": reduced["target_entropy_a2"] / reduced_steps,
                    "mass_top_a1": reduced["mass_top_a1"] / reduced_steps,
                    "mass_top_a2": reduced["mass_top_a2"] / reduced_steps,
                    "argmax_top_a1": reduced["argmax_top_a1"] / reduced_steps,
                    "argmax_top_a2": reduced["argmax_top_a2"] / reduced_steps,
                    "data_sec": reduced["data_sec"] / reduced_steps,
                    "compute_sec": reduced["compute_sec"] / reduced_steps,
                }
            rank_enabled = _rank_enabled(cfg)
            distribution_mode = _distribution_mode(cfg)
            metric_parts = [
                f"loss={log_metrics['loss']:.4f}",
                f"loss_a1={log_metrics['loss_a1']:.4f}",
                f"loss_a2={log_metrics['loss_a2']:.4f}",
            ]
            if rank_enabled:
                metric_parts.extend(
                    [
                        f"rank_a1={log_metrics['loss_rank_a1']:.4f}",
                        f"rank_a2={log_metrics['loss_rank_a2']:.4f}",
                    ]
                )
            if distribution_mode:
                metric_parts.extend(
                    [
                        f"pred_ent_a1={log_metrics['pred_entropy_a1']:.4f}",
                        f"pred_ent_a2={log_metrics['pred_entropy_a2']:.4f}",
                        f"target_ent_a1={log_metrics['target_entropy_a1']:.4f}",
                        f"target_ent_a2={log_metrics['target_entropy_a2']:.4f}",
                        f"mass_top_a1={log_metrics['mass_top_a1']:.4f}",
                        f"mass_top_a2={log_metrics['mass_top_a2']:.4f}",
                        f"argmax_top_a1={log_metrics['argmax_top_a1']:.4f}",
                        f"argmax_top_a2={log_metrics['argmax_top_a2']:.4f}",
                    ]
                )
            else:
                metric_parts.extend(
                    [
                        f"mae_a1={log_metrics['mae_a1']:.4f}",
                        f"mae_a2={log_metrics['mae_a2']:.4f}",
                    ]
                )
            _log_info(
                f"[INFO] {phase} step={steps_done}/{total_steps} "
                + " ".join(metric_parts)
                + f" data={log_metrics['data_sec'] * 1000.0:.1f}ms "
                + f"compute={log_metrics['compute_sec'] * 1000.0:.1f}ms "
                + f"elapsed={elapsed / 60.0:.1f}m eta={eta_sec / 60.0:.1f}m"
            )
            if train and _is_main_process():
                _log_wandb_step(
                    wandb_run,
                    phase="train_step",
                    global_step=step_offset + steps_done,
                    metrics=log_metrics,
                )
            for key in recent:
                recent[key] = 0.0 if key != "steps" else 0
        if max_steps > 0 and totals["steps"] >= max_steps:
            break
    local_steps = int(totals["steps"])
    reduced_totals = _reduce_epoch_totals(totals, device)
    denom = max(int(reduced_totals["steps"]), 1)
    out = {k: (v / denom if k != "steps" else int(v)) for k, v in reduced_totals.items()}
    out["local_steps"] = int(local_steps)
    out["global_steps"] = int(reduced_totals["steps"])
    return out


def run(config_path: str | Path) -> None:
    cfg_path = Path(config_path).resolve()
    cfg = _load_yaml(cfg_path)
    device, local_rank = _init_distributed(cfg["train"])
    try:
        out_dir = Path(cfg["train"]["output_dir"]).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if _is_main_process():
            (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        wandb_run = _maybe_init_wandb(cfg, out_dir)

        seed = int(cfg["train"].get("seed", 42))
        _seed_all(seed + _get_rank())
        ds_train, ds_val, index_payload = _build_datasets(cfg, out_dir)
        _log_info(f"[INFO] dataset train={len(ds_train)} val={len(ds_val)} index={index_payload['num_samples']}")

        train_sampler = None
        val_sampler = None
        if _is_distributed():
            train_sampler = DistributedSampler(
                ds_train,
                num_replicas=_get_world_size(),
                rank=_get_rank(),
                shuffle=bool(cfg["train"].get("shuffle", True)),
                seed=seed,
            )
            val_sampler = DistributedSampler(
                ds_val,
                num_replicas=_get_world_size(),
                rank=_get_rank(),
                shuffle=bool(cfg["train"].get("val_shuffle", False)),
                seed=seed,
            )

        loader_train = _make_loader(
            ds_train,
            int(cfg["train"].get("batch_size", 2)),
            bool(cfg["train"].get("shuffle", True)),
            int(cfg["train"].get("num_workers", 0)),
            sampler=train_sampler,
            pin_memory=bool(cfg["train"].get("pin_memory", False)),
            persistent_workers=bool(cfg["train"].get("persistent_workers", True)),
            prefetch_factor=int(cfg["train"].get("prefetch_factor", 4)),
            multiprocessing_context=str(cfg["train"].get("multiprocessing_context", "spawn") or "spawn"),
        )
        loader_val = _make_loader(
            ds_val,
            int(cfg["train"].get("val_batch_size", cfg["train"].get("batch_size", 2))),
            bool(cfg["train"].get("val_shuffle", False)),
            int(cfg["train"].get("num_workers", 0)),
            sampler=val_sampler,
            pin_memory=bool(cfg["train"].get("pin_memory", False)),
            persistent_workers=bool(cfg["train"].get("persistent_workers", True)),
            prefetch_factor=int(cfg["train"].get("prefetch_factor", 4)),
            multiprocessing_context=str(cfg["train"].get("multiprocessing_context", "spawn") or "spawn"),
        )

        model = PairPolicyNet(
            in_channels=int(cfg["model"].get("in_channels", 3)),
            feature_dim=int(cfg["model"].get("feature_dim", 64)),
            num_x1_samples=int(cfg["train"].get("num_x1_samples", 4)),
            backbone=cfg["model"].get("backbone"),
        ).to(device)
        if _is_distributed():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["train"].get("lr", 1e-3)),
            weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
        )
        scheduler = _build_scheduler(optimizer, cfg)

        history: list[dict[str, Any]] = []
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        early_stop_cfg = (cfg.get("train", {}) or {}).get("early_stop", {}) or {}
        early_stop_enabled = bool(early_stop_cfg.get("enabled", False))
        early_stop_patience = int(early_stop_cfg.get("patience", 0))
        early_stop_min_delta = float(early_stop_cfg.get("min_delta", 0.0))
        epochs = int(cfg["train"].get("epochs", 1))
        train_step_offset = 0
        for epoch in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            _log_info(f"[INFO] epoch={epoch} begin")
            train_metrics = _run_epoch(
                model,
                loader_train,
                optimizer,
                device,
                cfg,
                train=True,
                phase=f"train/epoch_{epoch}",
                wandb_run=wandb_run,
                step_offset=train_step_offset,
            )
            train_step_offset += int(train_metrics.get("local_steps", train_metrics["steps"]))
            val_metrics = _run_epoch(
                model,
                loader_val,
                optimizer,
                device,
                cfg,
                train=False,
                phase=f"val/epoch_{epoch}",
                wandb_run=wandb_run,
            )
            current_val_loss = float(val_metrics["loss"])
            improved = current_val_loss < (best_val_loss - early_stop_min_delta)
            if improved:
                best_val_loss = current_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if scheduler is not None:
                scheduler.step(current_val_loss)
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            if _is_main_process():
                history.append(record)
            _log_info(
                f"[INFO] epoch={epoch} "
                f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"train_a1={train_metrics['loss_a1']:.4f} train_a2={train_metrics['loss_a2']:.4f} "
                + (
                    f"train_rank_a1={train_metrics['loss_rank_a1']:.4f} train_rank_a2={train_metrics['loss_rank_a2']:.4f} "
                    if _rank_enabled(cfg)
                    else ""
                )
                + (
                    f"train_pred_ent_a1={train_metrics['pred_entropy_a1']:.4f} "
                    f"train_pred_ent_a2={train_metrics['pred_entropy_a2']:.4f} "
                    f"train_mass_top_a1={train_metrics['mass_top_a1']:.4f} "
                    f"train_mass_top_a2={train_metrics['mass_top_a2']:.4f} "
                    f"train_argmax_top_a1={train_metrics['argmax_top_a1']:.4f} "
                    f"train_argmax_top_a2={train_metrics['argmax_top_a2']:.4f}"
                    if _distribution_mode(cfg)
                    else f"train_mae_a1={train_metrics['mae_a1']:.4f} train_mae_a2={train_metrics['mae_a2']:.4f}"
                )
            )
            train_vis_path = None
            val_vis_path = None
            vis_every = int(cfg["train"].get("vis_every", 1))
            if _is_main_process() and vis_every > 0 and epoch % vis_every == 0:
                vis_dir = out_dir / "visuals" / f"epoch_{epoch:04d}"
                vis_rng = random.Random(seed + epoch)
                train_vis_path = _sample_visual_batch(
                    dataset=ds_train,
                    device=device,
                    model=model,
                    out_dir=vis_dir / "train",
                    rng=vis_rng,
                )
                val_vis_path = _sample_visual_batch(
                    dataset=ds_val,
                    device=device,
                    model=model,
                    out_dir=vis_dir / "val",
                    rng=vis_rng,
                )
            if _is_main_process():
                model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                ckpt = {
                    "epoch": epoch,
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                    "config": cfg,
                }
                torch.save(ckpt, out_dir / "last.pt")
                if improved:
                    torch.save(ckpt, out_dir / "best.pt")
                    _log_info(
                        f"[INFO] epoch={epoch} saved best checkpoint "
                        f"(val_loss={best_val_loss:.4f})"
                    )
                if scheduler is not None:
                    _log_info(
                        f"[INFO] epoch={epoch} lr={optimizer.param_groups[0]['lr']:.6g} "
                        f"plateau_bad_epochs={epochs_without_improvement}"
                    )
                _log_wandb_epoch(
                    wandb_run,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=train_step_offset,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    train_vis_path=train_vis_path,
                    val_vis_path=val_vis_path,
                )
            stop_now = (
                early_stop_enabled
                and early_stop_patience > 0
                and epochs_without_improvement >= early_stop_patience
            )
            if stop_now and _is_main_process():
                _log_info(
                    f"[INFO] early stopping at epoch={epoch} "
                    f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})"
                )
            if _is_distributed():
                stop_tensor = torch.tensor([1 if stop_now else 0], device=device, dtype=torch.int32)
                dist.broadcast(stop_tensor, src=0)
                stop_now = bool(int(stop_tensor.item()))
            if stop_now:
                break

        if _is_main_process():
            (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            if best_epoch > 0:
                (out_dir / "best.json").write_text(
                    json.dumps(
                        {
                            "epoch": best_epoch,
                            "val_loss": best_val_loss,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if wandb_run is not None:
                wandb_run.finish()
            print(f"[INFO] wrote {out_dir}")
    finally:
        _cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pair-policy from render supervision.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/pair_policy/configs/train/monai_unet.yaml",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
