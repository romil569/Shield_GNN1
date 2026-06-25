"""Evaluation helpers for baseline node-classification experiments."""

import numpy as np

try:
    import torch
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError("Evaluation requires torch. Install requirements-gnn.txt first.") from exc


def _valid_mask(y_true, mask):
    """Return a mask that includes only requested labeled nodes."""
    return mask.bool() & (y_true != -1)


def accuracy_score_torch(y_true, y_pred, mask):
    """Compute accuracy over labeled nodes selected by mask."""
    selected = _valid_mask(y_true, mask)
    if int(selected.sum()) == 0:
        return 0.0
    return float((y_pred[selected] == y_true[selected]).float().mean().item())


def precision_recall_f1(y_true, y_pred, mask):
    """Compute binary precision, recall, and F1 for the illicit class."""
    selected = _valid_mask(y_true, mask)
    if int(selected.sum()) == 0:
        return 0.0, 0.0, 0.0

    true = y_true[selected]
    pred = y_pred[selected]
    true_positive = int(((pred == 1) & (true == 1)).sum())
    false_positive = int(((pred == 1) & (true == 0)).sum())
    false_negative = int(((pred == 0) & (true == 1)).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def compute_roc_auc_safe(y_true, positive_scores, mask):
    """Compute ROC-AUC, returning None when only one class is present."""
    selected = _valid_mask(y_true, mask)
    if int(selected.sum()) == 0:
        return None

    true_np = y_true[selected].detach().cpu().numpy()
    score_np = positive_scores[selected].detach().cpu().numpy()
    if len(np.unique(true_np)) < 2:
        return None

    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(true_np, score_np))
    except ValueError:
        return None


def evaluate_node_classifier(model, x, edge_index, y, mask, criterion=None):
    """Evaluate a node classifier on labeled nodes selected by mask."""
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        selected = _valid_mask(y, mask)
        if int(selected.sum()) == 0:
            return {
                "loss": None,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": None,
                "num_nodes": 0,
                "illicit_count": 0,
                "licit_count": 0,
            }

        loss = criterion(logits[selected], y[selected]) if criterion else F.cross_entropy(
            logits[selected], y[selected]
        )
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        precision, recall, f1 = precision_recall_f1(y, predictions, selected)

        return {
            "loss": float(loss.item()),
            "accuracy": accuracy_score_torch(y, predictions, selected),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": compute_roc_auc_safe(y, probabilities[:, 1], selected),
            "num_nodes": int(selected.sum().item()),
            "illicit_count": int(((y == 1) & selected).sum().item()),
            "licit_count": int(((y == 0) & selected).sum().item()),
        }
