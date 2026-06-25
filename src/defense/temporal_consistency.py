"""Temporal consistency learning losses for future SHIELD-GNN integration."""

import torch
from torch import nn
from torch.nn import functional as F

from src.defense.consistency_utils import (
    sample_edge_indices,
    sample_temporal_pairs,
    validate_logits_shape,
    validate_mask,
)


def safe_softmax(logits):
    """Compute numerically stable probabilities from logits."""
    logits = validate_logits_shape(logits)
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    return torch.softmax(shifted, dim=-1)


def _zero_loss_like(logits):
    """Return a scalar zero loss on the same device/dtype as logits."""
    return logits.sum() * 0.0


def _distribution_loss(prob_a, prob_b, loss_type="mse"):
    """Compare probability distributions with the requested divergence."""
    eps = 1e-8
    if loss_type == "mse":
        return F.mse_loss(prob_a, prob_b)
    if loss_type == "kl":
        return F.kl_div((prob_a + eps).log(), prob_b, reduction="batchmean")
    if loss_type == "symmetric_kl":
        kl_ab = F.kl_div((prob_a + eps).log(), prob_b, reduction="batchmean")
        kl_ba = F.kl_div((prob_b + eps).log(), prob_a, reduction="batchmean")
        return 0.5 * (kl_ab + kl_ba)
    if loss_type == "js":
        midpoint = 0.5 * (prob_a + prob_b)
        kl_am = F.kl_div((prob_a + eps).log(), midpoint, reduction="batchmean")
        kl_bm = F.kl_div((prob_b + eps).log(), midpoint, reduction="batchmean")
        return 0.5 * (kl_am + kl_bm)
    raise ValueError("loss_type must be one of: mse, kl, symmetric_kl, js.")


def prediction_consistency_loss(logits_a, logits_b, mask=None, loss_type="mse", temperature=1.0):
    """Compare prediction distributions between two graph versions."""
    logits_a = validate_logits_shape(logits_a)
    logits_b = validate_logits_shape(logits_b)
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"logit shape mismatch: {tuple(logits_a.shape)} vs {tuple(logits_b.shape)}")
    mask = validate_mask(mask, logits_a.shape[0]).to(logits_a.device)
    if int(mask.sum()) == 0:
        return _zero_loss_like(logits_a)
    prob_a = safe_softmax(logits_a[mask] / max(float(temperature), 1e-8))
    prob_b = safe_softmax(logits_b[mask] / max(float(temperature), 1e-8))
    return _distribution_loss(prob_a, prob_b, loss_type=loss_type)


def temporal_smoothness_loss(
    logits,
    timesteps,
    mask=None,
    max_time_gap=1,
    loss_type="mse",
    sample_pairs=5000,
    seed=42,
):
    """Encourage prediction stability between nodes from nearby timesteps."""
    logits = validate_logits_shape(logits)
    if not torch.is_tensor(timesteps):
        timesteps = torch.as_tensor(timesteps, device=logits.device)
    timesteps = timesteps.to(logits.device)
    mask = validate_mask(mask, logits.shape[0]).to(logits.device)
    pairs = sample_temporal_pairs(timesteps, max_time_gap=max_time_gap, sample_pairs=sample_pairs, seed=seed)
    if pairs.shape[1] == 0:
        return _zero_loss_like(logits)
    pair_mask = mask[pairs[0]] & mask[pairs[1]]
    pairs = pairs[:, pair_mask]
    if pairs.shape[1] == 0:
        return _zero_loss_like(logits)
    probs = safe_softmax(logits)
    return _distribution_loss(probs[pairs[0]], probs[pairs[1]], loss_type=loss_type)


def edge_smoothness_loss(
    logits,
    edge_index,
    edge_weight=None,
    mask=None,
    sample_edges=20000,
    seed=42,
):
    """Encourage connected reliable nodes to have consistent predictions."""
    logits = validate_logits_shape(logits)
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, device=logits.device)
    edge_index = edge_index.long().to(logits.device)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges], found {tuple(edge_index.shape)}")
    if edge_index.shape[1] == 0:
        return _zero_loss_like(logits)
    selected_edges = sample_edge_indices(edge_index, sample_edges=sample_edges, seed=seed)
    sampled = edge_index[:, selected_edges]
    if mask is not None:
        mask = validate_mask(mask, logits.shape[0]).to(logits.device)
        edge_mask = mask[sampled[0]] & mask[sampled[1]]
        sampled = sampled[:, edge_mask]
        selected_edges = selected_edges[edge_mask]
    if sampled.shape[1] == 0:
        return _zero_loss_like(logits)
    probs = safe_softmax(logits)
    squared_diff = (probs[sampled[0]] - probs[sampled[1]]).pow(2).mean(dim=1)
    if edge_weight is not None:
        if not torch.is_tensor(edge_weight):
            edge_weight = torch.as_tensor(edge_weight, device=logits.device)
        weights = edge_weight.to(logits.device, dtype=squared_diff.dtype)[selected_edges]
        return (squared_diff * weights).sum() / weights.sum().clamp_min(1e-8)
    return squared_diff.mean()


def temporal_consistency_regularizer(
    clean_logits,
    defended_logits=None,
    attacked_logits=None,
    timesteps=None,
    edge_index=None,
    edge_weight=None,
    mask=None,
    alpha_clean_defended=1.0,
    beta_clean_attacked=0.5,
    eta_temporal=0.3,
    lambda_edge=0.2,
    loss_type="mse",
):
    """Combine prediction, temporal, and edge consistency losses."""
    clean_logits = validate_logits_shape(clean_logits)
    zero = _zero_loss_like(clean_logits)
    components = {
        "clean_defended_loss": zero,
        "clean_attacked_loss": zero,
        "temporal_smoothness_loss": zero,
        "edge_smoothness_loss": zero,
    }
    if defended_logits is not None:
        components["clean_defended_loss"] = prediction_consistency_loss(
            clean_logits, defended_logits, mask=mask, loss_type=loss_type
        )
    if attacked_logits is not None:
        components["clean_attacked_loss"] = prediction_consistency_loss(
            clean_logits, attacked_logits, mask=mask, loss_type=loss_type
        )
    if timesteps is not None:
        components["temporal_smoothness_loss"] = temporal_smoothness_loss(
            clean_logits, timesteps, mask=mask, loss_type=loss_type
        )
    if edge_index is not None:
        components["edge_smoothness_loss"] = edge_smoothness_loss(
            clean_logits, edge_index, edge_weight=edge_weight, mask=mask
        )
    total = (
        alpha_clean_defended * components["clean_defended_loss"]
        + beta_clean_attacked * components["clean_attacked_loss"]
        + eta_temporal * components["temporal_smoothness_loss"]
        + lambda_edge * components["edge_smoothness_loss"]
    )
    components["total_consistency_loss"] = total
    return total, components


class TemporalConsistencyLoss(nn.Module):
    """nn.Module wrapper for temporal consistency regularization."""

    def __init__(
        self,
        alpha_clean_defended=1.0,
        beta_clean_attacked=0.5,
        eta_temporal=0.3,
        lambda_edge=0.2,
        loss_type="mse",
    ):
        """Store temporal consistency weights."""
        super().__init__()
        self.alpha_clean_defended = alpha_clean_defended
        self.beta_clean_attacked = beta_clean_attacked
        self.eta_temporal = eta_temporal
        self.lambda_edge = lambda_edge
        self.loss_type = loss_type

    def forward(
        self,
        clean_logits,
        defended_logits=None,
        attacked_logits=None,
        timesteps=None,
        edge_index=None,
        edge_weight=None,
        mask=None,
    ):
        """Return total consistency loss and component dictionary."""
        return temporal_consistency_regularizer(
            clean_logits,
            defended_logits=defended_logits,
            attacked_logits=attacked_logits,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            mask=mask,
            alpha_clean_defended=self.alpha_clean_defended,
            beta_clean_attacked=self.beta_clean_attacked,
            eta_temporal=self.eta_temporal,
            lambda_edge=self.lambda_edge,
            loss_type=self.loss_type,
        )
