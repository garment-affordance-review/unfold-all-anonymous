from __future__ import annotations

import numpy as np
import torch


def symmetrize_reward_matrix(
    reward_matrix: np.ndarray,
    mode: str = "none",
    diagonal_value: float = -np.inf,
) -> np.ndarray:
    """
    Optionally remove pair-order ambiguity by symmetrizing R[i, j].

    Modes:
      - "none": keep original ordered reward matrix
      - "max_swap": R_sym[i,j] = max(R[i,j], R[j,i])
    """
    r = np.asarray(reward_matrix, dtype=np.float32)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError(f"reward_matrix must be square [N,N], got {r.shape}")

    if mode == "none":
        out = r.copy()
    elif mode == "max_swap":
        out = np.maximum(r, r.T).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown reward matrix symmetrization mode: {mode}")

    if out.shape[0] > 0:
        np.fill_diagonal(out, np.float32(diagonal_value))
    return out.astype(np.float32, copy=False)


def symmetrize_reward_matrix_torch(
    reward_matrix: torch.Tensor,
    mode: str = "none",
    diagonal_value: float = -float("inf"),
) -> torch.Tensor:
    """
    Torch equivalent of symmetrize_reward_matrix.
    """
    r = reward_matrix.to(dtype=torch.float32)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError(f"reward_matrix must be square [N,N], got {tuple(r.shape)}")

    if mode == "none":
        out = r.clone()
    elif mode == "max_swap":
        out = torch.maximum(r, r.T)
    else:
        raise ValueError(f"Unknown reward matrix symmetrization mode: {mode}")

    if out.shape[0] > 0:
        diag_idx = torch.arange(out.shape[0], device=out.device)
        out[diag_idx, diag_idx] = float(diagonal_value)
    return out


def build_reward_matrix(
    num_candidates: int,
    local_pairs: np.ndarray,
    rewards: np.ndarray,
    fill_value: float = -np.inf,
    diagonal_value: float = -np.inf,
) -> np.ndarray:
    """
    Build dense reward matrix R[i,j] from sparse local pair list.
    local_pairs are candidate-local indices in [0, num_candidates).
    """
    rmat = np.full((num_candidates, num_candidates), fill_value, dtype=np.float32)
    if local_pairs.size > 0:
        ij = np.asarray(local_pairs, dtype=np.int64)
        rv = np.asarray(rewards, dtype=np.float32)
        # If duplicate (i,j) appears, keep larger reward.
        np.maximum.at(rmat, (ij[:, 0], ij[:, 1]), rv)
    if num_candidates > 0:
        np.fill_diagonal(rmat, np.float32(diagonal_value))
    return rmat


def build_reward_matrix_torch(
    num_candidates: int,
    local_pairs: torch.Tensor,
    rewards: torch.Tensor,
    fill_value: float = -float("inf"),
    diagonal_value: float = -float("inf"),
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Torch equivalent of build_reward_matrix.
    local_pairs are candidate-local indices in [0, num_candidates).
    """
    if device is None:
        if isinstance(rewards, torch.Tensor):
            device = rewards.device
        elif isinstance(local_pairs, torch.Tensor):
            device = local_pairs.device
        else:
            device = torch.device("cpu")

    rmat = torch.full((num_candidates, num_candidates), float(fill_value), dtype=torch.float32, device=device)
    if local_pairs.numel() > 0:
        ij = local_pairs.to(device=device, dtype=torch.int64)
        rv = rewards.to(device=device, dtype=torch.float32)
        flat_idx = ij[:, 0] * num_candidates + ij[:, 1]
        rmat.view(-1).scatter_reduce_(0, flat_idx, rv, reduce="amax", include_self=True)
    if num_candidates > 0:
        diag_idx = torch.arange(num_candidates, device=device)
        rmat[diag_idx, diag_idx] = float(diagonal_value)
    return rmat


def _logsumexp(v: np.ndarray) -> float:
    finite = np.isfinite(v)
    if not np.any(finite):
        return float("-inf")
    w = v[finite]
    m = float(np.max(w))
    return float(m + np.log(np.sum(np.exp(w - m), dtype=np.float64)))


def build_a1_from_reward_matrix(
    reward_matrix: np.ndarray,
    reduce: str = "max",
    topk: int = 8,
) -> np.ndarray:
    """
    Build first-point target logits A1 from reward matrix R.
    """
    r = np.asarray(reward_matrix, dtype=np.float32)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError(f"reward_matrix must be square [N,N], got {r.shape}")

    if reduce == "max":
        return np.max(r, axis=1).astype(np.float32)
    if reduce == "topk_mean":
        out = np.empty((r.shape[0],), dtype=np.float32)
        k = max(1, int(topk))
        for i in range(r.shape[0]):
            row = r[i]
            finite = np.isfinite(row)
            if not np.any(finite):
                out[i] = np.float32(-np.inf)
                continue
            vals = row[finite]
            kk = min(k, vals.shape[0])
            top_vals = np.partition(vals, -kk)[-kk:]
            out[i] = np.float32(np.mean(top_vals, dtype=np.float64))
        return out
    if reduce == "logsumexp":
        out = np.empty((r.shape[0],), dtype=np.float32)
        for i in range(r.shape[0]):
            out[i] = np.float32(_logsumexp(r[i]))
        return out
    raise ValueError(f"Unknown reduce mode: {reduce}")


def build_a1_from_reward_matrix_torch(
    reward_matrix: torch.Tensor,
    reduce: str = "max",
    topk: int = 8,
) -> torch.Tensor:
    """
    Torch equivalent of build_a1_from_reward_matrix.
    """
    r = reward_matrix.to(dtype=torch.float32)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError(f"reward_matrix must be square [N,N], got {tuple(r.shape)}")

    if reduce == "max":
        return torch.max(r, dim=1).values
    if reduce == "topk_mean":
        finite_mask = torch.isfinite(r)
        safe = torch.where(finite_mask, r, torch.full_like(r, -float("inf")))
        k = max(1, int(topk))
        kk = min(k, int(r.shape[1])) if int(r.shape[1]) > 0 else 1
        top_vals = torch.topk(safe, k=kk, dim=1).values
        valid_rows = finite_mask.any(dim=1)
        out = top_vals.mean(dim=1)
        out = torch.where(valid_rows, out, torch.full_like(out, -float("inf")))
        return out
    if reduce == "logsumexp":
        return torch.logsumexp(r, dim=1)
    raise ValueError(f"Unknown reduce mode: {reduce}")


def _row_softmax(logits: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64) / float(tau)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    if not np.any(finite):
        return out.astype(np.float32)
    xf = x[finite]
    m = np.max(xf)
    expv = np.exp(xf - m)
    z = np.sum(expv)
    if z <= 0:
        return out.astype(np.float32)
    out[finite] = expv / z
    return out.astype(np.float32)


def build_a2_conditional_topk(
    reward_matrix: np.ndarray,
    a1_logits: np.ndarray,
    topk: int = 8,
    tau: float = 1.0,
    exclude_self: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build conditional second-point targets for top-k first-point anchors.

    Returns:
      topk_x1_idx: (K,)
      a2_logits_topk: (K, N)
      a2_probs_topk: (K, N)
    """
    r = np.asarray(reward_matrix, dtype=np.float32)
    a1 = np.asarray(a1_logits, dtype=np.float32)
    n = r.shape[0]
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError(f"reward_matrix must be [N,N], got {r.shape}")
    if a1.shape != (n,):
        raise ValueError(f"a1_logits shape must be ({n},), got {a1.shape}")

    valid = np.isfinite(a1)
    valid_idx = np.nonzero(valid)[0]
    if valid_idx.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, n), dtype=np.float32),
            np.zeros((0, n), dtype=np.float32),
        )

    k = int(min(topk, valid_idx.size))
    order = valid_idx[np.argsort(a1[valid_idx])[::-1]]
    top_idx = order[:k].astype(np.int64)

    a2_logits = np.full((k, n), -np.inf, dtype=np.float32)
    a2_probs = np.zeros((k, n), dtype=np.float32)
    for rank, x1 in enumerate(top_idx.tolist()):
        row = r[x1].copy()
        if exclude_self and 0 <= x1 < n:
            row[x1] = -np.inf
        a2_logits[rank] = row
        a2_probs[rank] = _row_softmax(row, tau=tau)
    return top_idx, a2_logits, a2_probs
