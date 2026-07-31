from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class TeacherRewardInfer:
    """
    Lightweight Pointcept teacher wrapper for pair reward inference.
    """

    def __init__(
        self,
        teacher_cfg: str,
        teacher_ckpt: str,
        device: str = "cuda",
        pointcept_code_root: Optional[str] = None,
        grid_size: Optional[float] = None,
        align_min: bool = True,
    ):
        self.teacher_cfg = Path(teacher_cfg).resolve()
        self.teacher_ckpt = Path(teacher_ckpt).resolve()
        self.device = torch.device(device)
        self.align_min = bool(align_min)

        if not self.teacher_cfg.exists():
            raise FileNotFoundError(f"Teacher config not found: {self.teacher_cfg}")
        if not self.teacher_ckpt.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.teacher_ckpt}")

        if pointcept_code_root is None:
            # cfg is in .../exp/.../config.py, code root is .../exp/.../code
            pointcept_code_root = str(self.teacher_cfg.parent / "code")
        self.pointcept_code_root = Path(pointcept_code_root).resolve()
        if not self.pointcept_code_root.exists():
            raise FileNotFoundError(f"Pointcept code root not found: {self.pointcept_code_root}")

        self._bootstrap_imports()
        self.cfg = self.Config.fromfile(str(self.teacher_cfg))
        self._apply_inference_compat_overrides()
        self.grid_size = float(grid_size) if grid_size is not None else self._infer_grid_size_from_cfg(self.cfg)
        self.model = self._build_and_load_model()

    def _bootstrap_imports(self) -> None:
        code_root = str(self.pointcept_code_root)
        if code_root not in sys.path:
            sys.path.insert(0, code_root)

        from pointcept.utils.config import Config  # type: ignore
        from pointcept.models import build_model  # type: ignore

        self.Config = Config
        self.build_model = build_model

    def _apply_inference_compat_overrides(self) -> None:
        """
        Apply minimal runtime-only config overrides for environments that do not
        carry all training-time acceleration dependencies.
        """
        model_cfg = getattr(self.cfg, "model", None)
        backbone_cfg = getattr(model_cfg, "backbone", None) if model_cfg is not None else None
        if backbone_cfg is None:
            return
        if getattr(backbone_cfg, "enable_flash", False) and importlib.util.find_spec("flash_attn") is None:
            backbone_cfg.enable_flash = False

    @staticmethod
    def _infer_grid_size_from_cfg(cfg) -> float:
        # Best effort: parse val transform first, fallback to train transform, fallback to 0.02.
        for split in ("val", "train"):
            split_cfg = getattr(cfg.data, split, None)
            if split_cfg is None:
                continue
            transforms = getattr(split_cfg, "transform", None)
            if transforms is None:
                continue
            for t in transforms:
                t_type = t.get("type", None) if isinstance(t, dict) else getattr(t, "type", None)
                if t_type == "AddGridCoord":
                    g = t.get("grid_size", None) if isinstance(t, dict) else getattr(t, "grid_size", None)
                    if g is not None:
                        return float(g)
        return 0.02

    def _build_and_load_model(self) -> torch.nn.Module:
        model = self.build_model(self.cfg.model)
        checkpoint = torch.load(str(self.teacher_ckpt), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        cleaned = {}
        for k, v in state_dict.items():
            # Saved by DDP trainer; remove module prefix for single-process inference.
            cleaned[k[7:] if k.startswith("module.") else k] = v

        model.load_state_dict(cleaned, strict=True)
        model = model.to(self.device)
        model.eval()
        return model

    def _prepare_input(
        self,
        coord: np.ndarray,
        pairs: np.ndarray,
        normal: Optional[np.ndarray] = None,
    ) -> dict:
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f"coord must be (N,3), got {coord.shape}")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"pairs must be (M,2), got {pairs.shape}")

        coord = np.asarray(coord, dtype=np.float32)
        pairs = np.asarray(pairs, dtype=np.int64)

        n_points = coord.shape[0]
        if pairs.size > 0:
            if np.min(pairs) < 0 or np.max(pairs) >= n_points:
                raise ValueError(
                    f"pairs index out of range for coord size {n_points}: min={pairs.min()} max={pairs.max()}"
                )

        if normal is None:
            normal = np.zeros_like(coord, dtype=np.float32)
        else:
            normal = np.asarray(normal, dtype=np.float32)
            if normal.shape != coord.shape:
                raise ValueError(f"normal shape must match coord: {normal.shape} vs {coord.shape}")

        grid_coord = np.floor(coord / self.grid_size).astype(np.int32)
        if self.align_min and grid_coord.size > 0:
            grid_coord -= grid_coord.min(axis=0)

        d = {
            "coord": torch.from_numpy(coord).to(self.device),
            "grid_coord": torch.from_numpy(grid_coord).to(self.device),
            "feat": torch.from_numpy(normal).to(self.device),
            "offset": torch.tensor([coord.shape[0]], dtype=torch.int64, device=self.device),
            "pairs": torch.from_numpy(pairs).to(self.device),
            "pair_offset": torch.tensor([pairs.shape[0]], dtype=torch.int64, device=self.device),
        }
        return d

    @torch.no_grad()
    def encode_points(
        self,
        coord: np.ndarray,
        normal: Optional[np.ndarray] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode points once and reuse features for many pair chunks."""
        coord = np.asarray(coord, dtype=np.float32)
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f"coord must be (N,3), got {coord.shape}")

        if normal is None:
            normal = np.zeros_like(coord, dtype=np.float32)
        else:
            normal = np.asarray(normal, dtype=np.float32)
            if normal.shape != coord.shape:
                raise ValueError(f"normal shape must match coord: {normal.shape} vs {coord.shape}")

        grid_coord = np.floor(coord / self.grid_size).astype(np.int32)
        if self.align_min and grid_coord.size > 0:
            grid_coord -= grid_coord.min(axis=0)

        data_dict = {
            "coord": torch.from_numpy(coord).to(self.device),
            "grid_coord": torch.from_numpy(grid_coord).to(self.device),
            "feat": torch.from_numpy(normal).to(self.device),
            "offset": torch.tensor([coord.shape[0]], dtype=torch.int64, device=self.device),
        }
        point = self.model.backbone(data_dict)
        return point.feat, point.offset

    @torch.no_grad()
    def infer_pairs_from_features_torch(
        self,
        feat: torch.Tensor,
        point_offset: torch.Tensor,
        pairs: np.ndarray | torch.Tensor,
        max_pairs_per_forward: int = 65536,
    ) -> torch.Tensor:
        """Score pairs using cached point features and return GPU/CPU torch tensor on model device."""
        if isinstance(pairs, torch.Tensor):
            pair_tensor = pairs.to(device=self.device, dtype=torch.int64)
            n_pairs = int(pair_tensor.shape[0])
        else:
            pairs = np.asarray(pairs, dtype=np.int64)
            n_pairs = int(pairs.shape[0])
            pair_tensor = torch.from_numpy(pairs).to(self.device)
        if n_pairs == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)
        if pair_tensor.ndim != 2 or pair_tensor.shape[1] != 2:
            raise ValueError(f"pairs must be (M,2), got {tuple(pair_tensor.shape)}")

        n_points = int(point_offset[-1].item()) if point_offset.numel() > 0 else int(feat.shape[0])
        if int(torch.min(pair_tensor).item()) < 0 or int(torch.max(pair_tensor).item()) >= n_points:
            raise ValueError(
                f"pairs index out of range for encoded feature size {n_points}: "
                f"min={int(torch.min(pair_tensor).item())} max={int(torch.max(pair_tensor).item())}"
            )

        out_chunks: list[torch.Tensor] = []
        for start in range(0, n_pairs, max_pairs_per_forward):
            end = min(start + max_pairs_per_forward, n_pairs)
            pair_chunk = pair_tensor[start:end]
            f0 = feat[pair_chunk[:, 0]]
            f1 = feat[pair_chunk[:, 1]]
            if self.model.pair_mode == "concat_diff":
                pair_feat = torch.cat([f0, f1, f0 - f1], dim=1)
            elif self.model.pair_mode == "concat":
                pair_feat = torch.cat([f0, f1], dim=1)
            elif self.model.pair_mode == "diff":
                pair_feat = f0 - f1
            else:
                raise ValueError(f"Unknown pair_mode: {self.model.pair_mode}")
            pred = self.model.mlp(pair_feat).squeeze(-1)
            out_chunks.append(pred.float())
        return torch.cat(out_chunks, dim=0)

    @torch.no_grad()
    def infer_pairs_from_features(
        self,
        feat: torch.Tensor,
        point_offset: torch.Tensor,
        pairs: np.ndarray,
        max_pairs_per_forward: int = 65536,
    ) -> np.ndarray:
        """Score pairs using cached point features from a single backbone pass."""
        pred = self.infer_pairs_from_features_torch(
            feat=feat,
            point_offset=point_offset,
            pairs=pairs,
            max_pairs_per_forward=max_pairs_per_forward,
        )
        return pred.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def infer_pairs(
        self,
        coord: np.ndarray,
        pairs: np.ndarray,
        normal: Optional[np.ndarray] = None,
        max_pairs_per_forward: int = 65536,
    ) -> np.ndarray:
        """
        Returns predicted reward for each pair in input order.
        """
        feat, point_offset = self.encode_points(coord=coord, normal=normal)
        return self.infer_pairs_from_features(
            feat=feat,
            point_offset=point_offset,
            pairs=pairs,
            max_pairs_per_forward=max_pairs_per_forward,
        )
