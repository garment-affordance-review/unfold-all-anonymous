from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .dataset import PairPolicyBatch


def _masked_target_distribution(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (mask > 0.5).to(dtype=target.dtype)
    masked = target * valid
    denom = masked.sum(dim=(-2, -1), keepdim=True)
    safe = torch.where(denom > 0.0, denom, torch.ones_like(denom))
    return masked / safe


def _distribution_entropy(dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return dist.new_tensor(0.0)
    safe_dist = torch.clamp(dist, min=1e-12)
    ent = -(safe_dist * torch.log(safe_dist))
    ent = torch.where(valid, ent, torch.zeros_like(ent))
    denom = valid.to(dtype=ent.dtype).sum(dim=(-2, -1))
    has_valid = denom > 0
    if not torch.any(has_valid):
        return dist.new_tensor(0.0)
    per_sample = ent.sum(dim=(-2, -1))
    return per_sample[has_valid].mean()


def _top_region_mask_from_values(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_ratio: float,
) -> torch.Tensor:
    valid = valid_mask > 0.5
    out = torch.zeros_like(valid, dtype=torch.bool)
    flat_values = values.reshape(values.shape[0], -1)
    flat_valid = valid.reshape(valid.shape[0], -1)
    ratio = float(max(min(top_ratio, 1.0), 0.0))
    if ratio <= 0.0:
        return out
    flat_out = out.reshape(values.shape[0], -1)
    for idx in range(flat_values.shape[0]):
        valid_idx = torch.nonzero(flat_valid[idx], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        take = max(1, int(round(valid_idx.numel() * ratio)))
        take = min(take, int(valid_idx.numel()))
        valid_vals = flat_values[idx, valid_idx]
        _, order = torch.topk(valid_vals, k=take, largest=True, sorted=False)
        flat_out[idx, valid_idx[order]] = True
    return out


def _mean_distribution_mass_in_region(
    dist: torch.Tensor,
    region_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    region = (region_mask > 0.5) & (valid_mask > 0.5)
    if not torch.any(region):
        return dist.new_tensor(0.0)
    mass = torch.where(region, dist, torch.zeros_like(dist)).sum(dim=(-2, -1))
    has_region = region.flatten(start_dim=-2).any(dim=-1)
    if not torch.any(has_region):
        return dist.new_tensor(0.0)
    return mass[has_region].mean()


def _argmax_hit_rate(
    logits: torch.Tensor,
    region_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid = valid_mask > 0.5
    region = (region_mask > 0.5) & valid
    if not torch.any(valid):
        return logits.new_tensor(0.0)
    flat_logits = logits.reshape(logits.shape[0], -1)
    flat_valid = valid.reshape(valid.shape[0], -1)
    flat_region = region.reshape(region.shape[0], -1)
    hits = []
    for idx in range(flat_logits.shape[0]):
        valid_idx = torch.nonzero(flat_valid[idx], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        best_local = torch.argmax(flat_logits[idx, valid_idx])
        best_global = valid_idx[best_local]
        hits.append(flat_region[idx, best_global].to(dtype=logits.dtype))
    if not hits:
        return logits.new_tensor(0.0)
    return torch.stack(hits, dim=0).mean()


def _masked_log_softmax_2d(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = torch.where(mask > 0.5, logits, logits.new_full((), -1e9))
    return F.log_softmax(masked_logits.flatten(start_dim=-2), dim=-1).view_as(logits)


def _samplewise_ranking_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    pos = positive_mask > 0.5
    neg = negative_mask > 0.5
    flat_logits = logits.reshape(logits.shape[0], -1)
    flat_pos = pos.reshape(pos.shape[0], -1)
    flat_neg = neg.reshape(neg.shape[0], -1)
    losses = []
    for idx in range(flat_logits.shape[0]):
        pos_vals = flat_logits[idx][flat_pos[idx]]
        neg_vals = flat_logits[idx][flat_neg[idx]]
        if pos_vals.numel() == 0 or neg_vals.numel() == 0:
            continue
        loss = F.relu(float(margin) - pos_vals.unsqueeze(1) + neg_vals.unsqueeze(0))
        losses.append(loss.mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses, dim=0).mean()


def _masked_kl_on_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = (mask > 0.5)
    if not torch.any(valid):
        return logits.new_tensor(0.0)
    target_dist = _masked_target_distribution(target, mask)
    log_probs = _masked_log_softmax_2d(logits, mask)
    per_sample = -(target_dist * log_probs).sum(dim=(-2, -1))
    has_valid = valid.flatten(start_dim=-2).any(dim=-1)
    if not torch.any(has_valid):
        return logits.new_tensor(0.0)
    return per_sample[has_valid].mean()


def _masked_huber_on_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return logits.new_tensor(0.0)
    pred = torch.sigmoid(logits)
    diff = pred - target
    abs_diff = torch.abs(diff)
    delta_t = logits.new_tensor(float(delta))
    quadratic = torch.minimum(abs_diff, delta_t)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic * quadratic + delta_t * linear
    masked = torch.where(valid, loss, torch.zeros_like(loss))
    denom = valid.to(dtype=masked.dtype).sum(dim=(-2, -1))
    has_valid = denom > 0
    if not torch.any(has_valid):
        return logits.new_tensor(0.0)
    per_sample = masked.sum(dim=(-2, -1)) / torch.clamp(denom, min=1.0)
    return per_sample[has_valid].mean()


def _weighted_masked_huber_on_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return logits.new_tensor(0.0)
    pred = torch.sigmoid(logits)
    diff = pred - target
    abs_diff = torch.abs(diff)
    delta_t = logits.new_tensor(float(delta))
    quadratic = torch.minimum(abs_diff, delta_t)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic * quadratic + delta_t * linear
    weight = 1.0 + float(alpha) * torch.pow(torch.clamp(target, min=0.0), float(gamma))
    weighted = loss * weight
    masked = torch.where(valid, weighted, torch.zeros_like(weighted))
    denom = torch.where(valid, weight, torch.zeros_like(weight)).sum(dim=(-2, -1))
    has_valid = denom > 0
    if not torch.any(has_valid):
        return logits.new_tensor(0.0)
    per_sample = masked.sum(dim=(-2, -1)) / torch.clamp(denom, min=1.0)
    return per_sample[has_valid].mean()


def _masked_huber_on_raw(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return pred.new_tensor(0.0)
    diff = pred - target
    abs_diff = torch.abs(diff)
    delta_t = pred.new_tensor(float(delta))
    quadratic = torch.minimum(abs_diff, delta_t)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic * quadratic + delta_t * linear
    masked = torch.where(valid, loss, torch.zeros_like(loss))
    denom = valid.to(dtype=masked.dtype).sum(dim=(-2, -1))
    has_valid = denom > 0
    if not torch.any(has_valid):
        return pred.new_tensor(0.0)
    per_sample = masked.sum(dim=(-2, -1)) / torch.clamp(denom, min=1.0)
    return per_sample[has_valid].mean()


def _masked_bce_on_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return logits.new_tensor(0.0)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    masked = torch.where(valid, loss, torch.zeros_like(loss))
    denom = valid.to(dtype=masked.dtype).sum(dim=(-2, -1))
    has_valid = denom > 0
    if not torch.any(has_valid):
        return logits.new_tensor(0.0)
    per_sample = masked.sum(dim=(-2, -1)) / torch.clamp(denom, min=1.0)
    return per_sample[has_valid].mean()


def _masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return pred.new_tensor(0.0)
    err = torch.abs(pred - target)
    masked = torch.where(valid, err, torch.zeros_like(err))
    denom = valid.to(dtype=masked.dtype).sum(dim=(-2, -1))
    has_valid = denom > 0
    if not torch.any(has_valid):
        return pred.new_tensor(0.0)
    per_sample = masked.sum(dim=(-2, -1)) / torch.clamp(denom, min=1.0)
    return per_sample[has_valid].mean()


def pair_policy_loss(
    outputs: dict[str, torch.Tensor],
    batch: "PairPolicyBatch",
    lambda_a1: float = 1.0,
    lambda_a2: float = 1.0,
    lambda_rank_a1: float = 0.0,
    lambda_rank_a2: float = 0.0,
    rank_margin: float = 0.02,
    huber_delta: float = 0.1,
    weighted_huber_alpha: float = 4.0,
    weighted_huber_gamma: float = 1.0,
    diagnostic_top_ratio: float = 0.1,
    loss_name: str = "masked_kl",
) -> dict[str, torch.Tensor]:
    name = str(loss_name)
    if name == "masked_kl":
        a1_loss = _masked_kl_on_logits(outputs["a1_logits"], batch.a1_target, batch.a1_target_mask)
        a1_target_dist = _masked_target_distribution(batch.a1_target, batch.a1_target_mask)
        a1_pred_dist = _masked_target_distribution(
            torch.softmax(outputs["a1_logits"].flatten(start_dim=-2), dim=-1).view_as(outputs["a1_logits"]),
            batch.a1_target_mask,
        )
        a1_mae = outputs["a1_logits"].new_tensor(0.0)
        a1_pred_entropy = _distribution_entropy(a1_pred_dist, batch.a1_target_mask)
        a1_target_entropy = _distribution_entropy(a1_target_dist, batch.a1_target_mask)
        a1_diag_top = _top_region_mask_from_values(
            batch.a1_target,
            batch.a1_target_mask,
            top_ratio=diagnostic_top_ratio,
        )
        a1_mass_top = _mean_distribution_mass_in_region(a1_pred_dist, a1_diag_top, batch.a1_target_mask)
        a1_argmax_top = _argmax_hit_rate(outputs["a1_logits"], a1_diag_top, batch.a1_target_mask)
        if torch.any(batch.a2_target_valid):
            logits_flat = outputs["a2_logits"].reshape(-1, *outputs["a2_logits"].shape[-2:])
            target_flat = batch.a2_target.reshape(-1, *batch.a2_target.shape[-2:])
            mask_flat = batch.a2_target_mask.reshape(-1, *batch.a2_target_mask.shape[-2:])
            target_dist = _masked_target_distribution(target_flat, mask_flat)
            log_probs = _masked_log_softmax_2d(logits_flat, mask_flat)
            per_pair = -(target_dist * log_probs).sum(dim=(-2, -1))
            pred_dist = _masked_target_distribution(
                torch.softmax(logits_flat.flatten(start_dim=-2), dim=-1).view_as(logits_flat),
                mask_flat,
            )
            a2_loss = per_pair[batch.a2_target_valid.reshape(-1)].mean()
            a2_mae = outputs["a2_logits"].new_tensor(0.0)
            a2_pred_entropy = _distribution_entropy(
                pred_dist[batch.a2_target_valid.reshape(-1)],
                mask_flat[batch.a2_target_valid.reshape(-1)],
            )
            a2_target_entropy = _distribution_entropy(
                target_dist[batch.a2_target_valid.reshape(-1)],
                mask_flat[batch.a2_target_valid.reshape(-1)],
            )
            a2_diag_top = _top_region_mask_from_values(
                target_flat,
                mask_flat,
                top_ratio=diagnostic_top_ratio,
            )
            valid_pairs = batch.a2_target_valid.reshape(-1)
            a2_mass_top = _mean_distribution_mass_in_region(
                pred_dist[valid_pairs],
                a2_diag_top[valid_pairs],
                mask_flat[valid_pairs],
            )
            a2_argmax_top = _argmax_hit_rate(
                logits_flat[valid_pairs],
                a2_diag_top[valid_pairs],
                mask_flat[valid_pairs],
            )
        else:
            a2_loss = outputs["a2_logits"].new_tensor(0.0)
            a2_mae = outputs["a2_logits"].new_tensor(0.0)
            a2_pred_entropy = outputs["a2_logits"].new_tensor(0.0)
            a2_target_entropy = outputs["a2_logits"].new_tensor(0.0)
            a2_mass_top = outputs["a2_logits"].new_tensor(0.0)
            a2_argmax_top = outputs["a2_logits"].new_tensor(0.0)
        if float(lambda_rank_a1) > 0.0:
            rank_a1 = _samplewise_ranking_loss(
                outputs["a1_logits"],
                batch.a1_target_mask,
                batch.a1_negative_mask,
                margin=rank_margin,
            )
        else:
            rank_a1 = outputs["a1_logits"].new_tensor(0.0)
        if float(lambda_rank_a2) > 0.0 and torch.any(batch.a2_target_valid):
            logits_flat = outputs["a2_logits"].reshape(-1, *outputs["a2_logits"].shape[-2:])
            pos_flat = batch.a2_target_mask.reshape(-1, *batch.a2_target_mask.shape[-2:])
            neg_flat = batch.a2_negative_mask.reshape(-1, *batch.a2_negative_mask.shape[-2:])
            per_pair_rank = torch.stack(
                [
                    _samplewise_ranking_loss(
                        logits_flat[i : i + 1],
                        pos_flat[i : i + 1],
                        neg_flat[i : i + 1],
                        margin=rank_margin,
                    )
                    for i in range(logits_flat.shape[0])
                ],
                dim=0,
            )
            rank_a2 = per_pair_rank[batch.a2_target_valid.reshape(-1)].mean()
        else:
            rank_a2 = outputs["a2_logits"].new_tensor(0.0)
    elif name == "masked_huber":
        a1_loss = _masked_huber_on_logits(outputs["a1_logits"], batch.a1_target, batch.a1_valid_mask, huber_delta)
        a1_mae = _masked_mae(torch.sigmoid(outputs["a1_logits"]), batch.a1_target, batch.a1_valid_mask)
        if torch.any(batch.a2_target_valid):
            logits_flat = outputs["a2_logits"].reshape(-1, *outputs["a2_logits"].shape[-2:])
            target_flat = batch.a2_target.reshape(-1, *batch.a2_target.shape[-2:])
            mask_flat = batch.a2_valid_mask.reshape(-1, *batch.a2_valid_mask.shape[-2:])
            per_pair = torch.stack(
                [
                    _masked_huber_on_logits(logits_flat[i], target_flat[i], mask_flat[i], huber_delta)
                    for i in range(logits_flat.shape[0])
                ],
                dim=0,
            )
            pred_flat = torch.sigmoid(logits_flat)
            per_pair_mae = torch.stack(
                [_masked_mae(pred_flat[i], target_flat[i], mask_flat[i]) for i in range(logits_flat.shape[0])],
                dim=0,
            )
            a2_loss = per_pair[batch.a2_target_valid.reshape(-1)].mean()
            a2_mae = per_pair_mae[batch.a2_target_valid.reshape(-1)].mean()
        else:
            a2_loss = outputs["a2_logits"].new_tensor(0.0)
            a2_mae = outputs["a2_logits"].new_tensor(0.0)
        rank_a1 = outputs["a1_logits"].new_tensor(0.0)
        rank_a2 = outputs["a2_logits"].new_tensor(0.0)
        a1_pred_entropy = outputs["a1_logits"].new_tensor(0.0)
        a1_target_entropy = outputs["a1_logits"].new_tensor(0.0)
        a2_pred_entropy = outputs["a2_logits"].new_tensor(0.0)
        a2_target_entropy = outputs["a2_logits"].new_tensor(0.0)
        a1_mass_top = outputs["a1_logits"].new_tensor(0.0)
        a2_mass_top = outputs["a2_logits"].new_tensor(0.0)
        a1_argmax_top = outputs["a1_logits"].new_tensor(0.0)
        a2_argmax_top = outputs["a2_logits"].new_tensor(0.0)
    elif name == "weighted_masked_huber":
        a1_loss = _weighted_masked_huber_on_logits(
            outputs["a1_logits"],
            batch.a1_target,
            batch.a1_valid_mask,
            huber_delta,
            alpha=weighted_huber_alpha,
            gamma=weighted_huber_gamma,
        )
        a1_mae = _masked_mae(torch.sigmoid(outputs["a1_logits"]), batch.a1_target, batch.a1_valid_mask)
        if torch.any(batch.a2_target_valid):
            logits_flat = outputs["a2_logits"].reshape(-1, *outputs["a2_logits"].shape[-2:])
            target_flat = batch.a2_target.reshape(-1, *batch.a2_target.shape[-2:])
            mask_flat = batch.a2_valid_mask.reshape(-1, *batch.a2_valid_mask.shape[-2:])
            per_pair = torch.stack(
                [
                    _weighted_masked_huber_on_logits(
                        logits_flat[i],
                        target_flat[i],
                        mask_flat[i],
                        huber_delta,
                        alpha=weighted_huber_alpha,
                        gamma=weighted_huber_gamma,
                    )
                    for i in range(logits_flat.shape[0])
                ],
                dim=0,
            )
            pred_flat = torch.sigmoid(logits_flat)
            per_pair_mae = torch.stack(
                [_masked_mae(pred_flat[i], target_flat[i], mask_flat[i]) for i in range(logits_flat.shape[0])],
                dim=0,
            )
            a2_loss = per_pair[batch.a2_target_valid.reshape(-1)].mean()
            a2_mae = per_pair_mae[batch.a2_target_valid.reshape(-1)].mean()
        else:
            a2_loss = outputs["a2_logits"].new_tensor(0.0)
            a2_mae = outputs["a2_logits"].new_tensor(0.0)
        rank_a1 = outputs["a1_logits"].new_tensor(0.0)
        rank_a2 = outputs["a2_logits"].new_tensor(0.0)
        a1_pred_entropy = outputs["a1_logits"].new_tensor(0.0)
        a1_target_entropy = outputs["a1_logits"].new_tensor(0.0)
        a2_pred_entropy = outputs["a2_logits"].new_tensor(0.0)
        a2_target_entropy = outputs["a2_logits"].new_tensor(0.0)
        a1_mass_top = outputs["a1_logits"].new_tensor(0.0)
        a2_mass_top = outputs["a2_logits"].new_tensor(0.0)
        a1_argmax_top = outputs["a1_logits"].new_tensor(0.0)
        a2_argmax_top = outputs["a2_logits"].new_tensor(0.0)
    elif name == "masked_huber_raw":
        a1_loss = _masked_huber_on_raw(outputs["a1_logits"], batch.a1_target, batch.a1_valid_mask, huber_delta)
        a1_mae = _masked_mae(torch.clamp(outputs["a1_logits"], min=0.0, max=1.0), batch.a1_target, batch.a1_valid_mask)
        if torch.any(batch.a2_target_valid):
            logits_flat = outputs["a2_logits"].reshape(-1, *outputs["a2_logits"].shape[-2:])
            target_flat = batch.a2_target.reshape(-1, *batch.a2_target.shape[-2:])
            mask_flat = batch.a2_valid_mask.reshape(-1, *batch.a2_valid_mask.shape[-2:])
            per_pair = torch.stack(
                [
                    _masked_huber_on_raw(logits_flat[i], target_flat[i], mask_flat[i], huber_delta)
                    for i in range(logits_flat.shape[0])
                ],
                dim=0,
            )
            pred_flat = torch.clamp(logits_flat, min=0.0, max=1.0)
            per_pair_mae = torch.stack(
                [_masked_mae(pred_flat[i], target_flat[i], mask_flat[i]) for i in range(logits_flat.shape[0])],
                dim=0,
            )
            a2_loss = per_pair[batch.a2_target_valid.reshape(-1)].mean()
            a2_mae = per_pair_mae[batch.a2_target_valid.reshape(-1)].mean()
        else:
            a2_loss = outputs["a2_logits"].new_tensor(0.0)
            a2_mae = outputs["a2_logits"].new_tensor(0.0)
        rank_a1 = outputs["a1_logits"].new_tensor(0.0)
        rank_a2 = outputs["a2_logits"].new_tensor(0.0)
        a1_pred_entropy = outputs["a1_logits"].new_tensor(0.0)
        a1_target_entropy = outputs["a1_logits"].new_tensor(0.0)
        a2_pred_entropy = outputs["a2_logits"].new_tensor(0.0)
        a2_target_entropy = outputs["a2_logits"].new_tensor(0.0)
        a1_mass_top = outputs["a1_logits"].new_tensor(0.0)
        a2_mass_top = outputs["a2_logits"].new_tensor(0.0)
        a1_argmax_top = outputs["a1_logits"].new_tensor(0.0)
        a2_argmax_top = outputs["a2_logits"].new_tensor(0.0)
    elif name == "masked_bce":
        a1_loss = _masked_bce_on_logits(outputs["a1_logits"], batch.a1_target, batch.a1_valid_mask)
        a1_mae = _masked_mae(torch.sigmoid(outputs["a1_logits"]), batch.a1_target, batch.a1_valid_mask)
        if torch.any(batch.a2_target_valid):
            logits_flat = outputs["a2_logits"].reshape(-1, *outputs["a2_logits"].shape[-2:])
            target_flat = batch.a2_target.reshape(-1, *batch.a2_target.shape[-2:])
            mask_flat = batch.a2_valid_mask.reshape(-1, *batch.a2_valid_mask.shape[-2:])
            per_pair = torch.stack(
                [
                    _masked_bce_on_logits(logits_flat[i], target_flat[i], mask_flat[i])
                    for i in range(logits_flat.shape[0])
                ],
                dim=0,
            )
            pred_flat = torch.sigmoid(logits_flat)
            per_pair_mae = torch.stack(
                [_masked_mae(pred_flat[i], target_flat[i], mask_flat[i]) for i in range(logits_flat.shape[0])],
                dim=0,
            )
            a2_loss = per_pair[batch.a2_target_valid.reshape(-1)].mean()
            a2_mae = per_pair_mae[batch.a2_target_valid.reshape(-1)].mean()
        else:
            a2_loss = outputs["a2_logits"].new_tensor(0.0)
            a2_mae = outputs["a2_logits"].new_tensor(0.0)
        rank_a1 = outputs["a1_logits"].new_tensor(0.0)
        rank_a2 = outputs["a2_logits"].new_tensor(0.0)
        a1_pred_entropy = outputs["a1_logits"].new_tensor(0.0)
        a1_target_entropy = outputs["a1_logits"].new_tensor(0.0)
        a2_pred_entropy = outputs["a2_logits"].new_tensor(0.0)
        a2_target_entropy = outputs["a2_logits"].new_tensor(0.0)
        a1_mass_top = outputs["a1_logits"].new_tensor(0.0)
        a2_mass_top = outputs["a2_logits"].new_tensor(0.0)
        a1_argmax_top = outputs["a1_logits"].new_tensor(0.0)
        a2_argmax_top = outputs["a2_logits"].new_tensor(0.0)
    else:
        raise ValueError(f"unsupported pair-policy loss_name: {loss_name}")
    total = (
        float(lambda_a1) * a1_loss
        + float(lambda_a2) * a2_loss
        + float(lambda_rank_a1) * rank_a1
        + float(lambda_rank_a2) * rank_a2
    )
    return {
        "loss": total,
        "loss_a1": a1_loss.detach(),
        "loss_a2": a2_loss.detach(),
        "loss_rank_a1": rank_a1.detach(),
        "loss_rank_a2": rank_a2.detach(),
        "mae_a1": a1_mae.detach(),
        "mae_a2": a2_mae.detach(),
        "pred_entropy_a1": a1_pred_entropy.detach(),
        "target_entropy_a1": a1_target_entropy.detach(),
        "pred_entropy_a2": a2_pred_entropy.detach(),
        "target_entropy_a2": a2_target_entropy.detach(),
        "mass_top_a1": a1_mass_top.detach(),
        "mass_top_a2": a2_mass_top.detach(),
        "argmax_top_a1": a1_argmax_top.detach(),
        "argmax_top_a2": a2_argmax_top.detach(),
    }
