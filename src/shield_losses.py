"""Loss functions for final SHIELD-GNN integration."""

import torch
from torch.nn import functional as F

from src.defense.temporal_consistency import temporal_consistency_regularizer


def supervised_node_loss(logits, y, mask, class_weights=None):
    """Compute supervised node classification loss on labeled masked nodes."""
    selected = mask.bool() & (y != -1)
    if int(selected.sum()) == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[selected], y[selected], weight=class_weights)


def pruning_regularization(edge_scores=None, edge_weight=None, mode="l1"):
    """Compute a small pruning regularization term."""
    tensor = edge_weight if edge_weight is not None else edge_scores
    if tensor is None:
        return torch.tensor(0.0)
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor, dtype=torch.float32)
    if tensor.numel() == 0:
        return tensor.sum() * 0.0
    if mode == "l2":
        return tensor.float().pow(2).mean()
    if mode == "variance":
        return tensor.float().var(unbiased=False)
    return tensor.float().abs().mean()


def shield_total_loss(
    clean_logits,
    y,
    train_mask,
    defended_logits=None,
    attacked_logits=None,
    timesteps=None,
    edge_index=None,
    edge_weight=None,
    edge_scores=None,
    class_weights=None,
    alpha_defended=1.0,
    beta_consistency=0.5,
    gamma_pruning=0.1,
):
    """Compute the combined SHIELD-GNN objective without backward."""
    clean_loss = supervised_node_loss(clean_logits, y, train_mask, class_weights=class_weights)
    defended_loss = clean_logits.sum() * 0.0
    if defended_logits is not None:
        defended_loss = supervised_node_loss(defended_logits, y, train_mask, class_weights=class_weights)
    consistency_loss, consistency_components = temporal_consistency_regularizer(
        clean_logits,
        defended_logits=defended_logits,
        attacked_logits=attacked_logits,
        timesteps=timesteps,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mask=train_mask,
    )
    pruning_loss = pruning_regularization(edge_scores=edge_scores, edge_weight=edge_weight).to(clean_logits.device)
    total = clean_loss + alpha_defended * defended_loss + beta_consistency * consistency_loss + gamma_pruning * pruning_loss
    components = {
        "clean_supervised_loss": clean_loss,
        "defended_supervised_loss": defended_loss,
        "temporal_consistency_loss": consistency_loss,
        "pruning_regularization": pruning_loss,
        "total_loss": total,
        **{f"consistency_{key}": value for key, value in consistency_components.items()},
    }
    return total, components
