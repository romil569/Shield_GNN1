"""Robust edge pruning defense for suspicious graph edges."""

import numpy as np

from src.defense.edge_scoring import compute_combined_edge_scores
from src.defense.pruning_utils import threshold_scores, validate_edge_index


def prune_edges_hard(edge_index, scores, prune_threshold=0.75):
    """Remove edges whose suspiciousness score exceeds the prune threshold."""
    edge_index = validate_edge_index(edge_index, int(edge_index.max()) + 1 if edge_index.size else 0)
    scores = np.asarray(scores, dtype=np.float32)
    pruned_mask = scores >= prune_threshold
    kept_mask = ~pruned_mask
    pruned_edge_index = edge_index[:, kept_mask]
    summary = {
        "mode": "hard",
        "original_num_edges": int(edge_index.shape[1]),
        "defended_num_edges": int(pruned_edge_index.shape[1]),
        "pruned_edges": int(pruned_mask.sum()),
        "kept_edges": int(kept_mask.sum()),
        "downweighted_edges": 0,
        "prune_threshold": float(prune_threshold),
    }
    return pruned_edge_index, kept_mask, pruned_mask, summary


def prune_edges_soft(edge_index, scores, min_weight=0.10):
    """Keep all edges and assign lower weights to suspicious edges."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    edge_weight = np.clip(1.0 - scores, min_weight, 1.0).astype(np.float32)
    summary = {
        "mode": "soft",
        "original_num_edges": int(edge_index.shape[1]),
        "defended_num_edges": int(edge_index.shape[1]),
        "pruned_edges": 0,
        "kept_edges": int(edge_index.shape[1]),
        "downweighted_edges": int((edge_weight < 1.0).sum()),
        "min_weight": float(min_weight),
    }
    return edge_index.copy(), edge_weight, summary


def prune_edges_hybrid(
    edge_index,
    scores,
    prune_threshold=0.80,
    downweight_threshold=0.50,
    min_weight=0.10,
):
    """Prune high-risk edges and down-weight medium-risk edges."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    kept_mask, downweighted_mask, pruned_mask = threshold_scores(
        scores, prune_threshold, downweight_threshold
    )
    defended_edge_index = edge_index[:, kept_mask]
    defended_scores = scores[kept_mask]
    defended_edge_weight = np.ones(defended_edge_index.shape[1], dtype=np.float32)
    downweighted_kept = defended_scores >= downweight_threshold
    defended_edge_weight[downweighted_kept] = np.clip(
        1.0 - defended_scores[downweighted_kept], min_weight, 1.0
    )
    summary = {
        "mode": "hybrid",
        "original_num_edges": int(edge_index.shape[1]),
        "defended_num_edges": int(defended_edge_index.shape[1]),
        "pruned_edges": int(pruned_mask.sum()),
        "downweighted_edges": int(downweighted_mask.sum()),
        "kept_edges": int(kept_mask.sum()),
        "prune_threshold": float(prune_threshold),
        "downweight_threshold": float(downweight_threshold),
        "min_weight": float(min_weight),
    }
    return defended_edge_index, defended_edge_weight, kept_mask, downweighted_mask, pruned_mask, summary


def resolve_prune_threshold(
    scores,
    threshold_mode="fixed",
    prune_threshold=0.80,
    score_percentile=95.0,
    prune_ratio=None,
    min_prune_edges=0,
):
    """Resolve a pruning threshold from fixed, percentile, or top-k settings."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return float(prune_threshold)
    threshold_mode = threshold_mode.lower()
    if threshold_mode == "percentile":
        threshold = float(np.percentile(scores, score_percentile))
    elif threshold_mode == "topk":
        requested = int(max(min_prune_edges, round(scores.size * float(prune_ratio or 0.0))))
        requested = max(1, min(requested, scores.size))
        threshold = float(np.partition(scores, scores.size - requested)[scores.size - requested])
    elif threshold_mode == "fixed":
        threshold = float(prune_threshold)
    else:
        raise ValueError("threshold_mode must be fixed, percentile, or topk.")

    requested_by_ratio = int(round(scores.size * float(prune_ratio or 0.0)))
    requested = max(int(min_prune_edges or 0), requested_by_ratio)
    if requested > 0:
        requested = max(1, min(requested, scores.size))
        topk_threshold = float(np.partition(scores, scores.size - requested)[scores.size - requested])
        threshold = min(threshold, topk_threshold)
    return threshold


def _component_summary(score_components):
    """Summarize means and maxima for score components."""
    summary = {}
    for name, values in score_components.items():
        values = np.asarray(values)
        summary[f"{name}_mean"] = float(values.mean()) if values.size else 0.0
        summary[f"{name}_max"] = float(values.max()) if values.size else 0.0
    return summary


def run_robust_edge_pruning(
    X,
    y,
    edge_index,
    timesteps=None,
    mode="hybrid",
    prune_threshold=0.80,
    downweight_threshold=0.50,
    min_weight=0.10,
    score_weights=None,
    threshold_mode="fixed",
    score_percentile=95.0,
    prune_ratio=None,
    min_prune_edges=0,
):
    """Score edges and apply hard, soft, or hybrid robust pruning."""
    X = np.asarray(X)
    y = np.asarray(y)
    edge_index = validate_edge_index(edge_index, X.shape[0])
    score_weights = score_weights or {}
    edge_scores, score_components = compute_combined_edge_scores(
        X,
        y,
        edge_index,
        timesteps=timesteps,
        feature_weight=score_weights.get("feature_weight", 0.40),
        degree_weight=score_weights.get("degree_weight", 0.25),
        label_weight=score_weights.get("label_weight", 0.20),
        temporal_weight=score_weights.get("temporal_weight", 0.15),
    )
    effective_prune_threshold = resolve_prune_threshold(
        edge_scores,
        threshold_mode=threshold_mode,
        prune_threshold=prune_threshold,
        score_percentile=score_percentile,
        prune_ratio=prune_ratio,
        min_prune_edges=min_prune_edges,
    )
    effective_downweight_threshold = min(float(downweight_threshold), effective_prune_threshold)

    if mode == "hard":
        defended_edge_index, kept_mask, pruned_mask, summary = prune_edges_hard(
            edge_index, edge_scores, effective_prune_threshold
        )
        defended_edge_weight = np.ones(defended_edge_index.shape[1], dtype=np.float32)
        downweighted_mask = np.zeros(edge_index.shape[1], dtype=bool)
    elif mode == "soft":
        defended_edge_index, defended_edge_weight, summary = prune_edges_soft(
            edge_index, edge_scores, min_weight
        )
        kept_mask = np.ones(edge_index.shape[1], dtype=bool)
        pruned_mask = np.zeros(edge_index.shape[1], dtype=bool)
        downweighted_mask = defended_edge_weight < 1.0
    elif mode == "hybrid":
        (
            defended_edge_index,
            defended_edge_weight,
            kept_mask,
            downweighted_mask,
            pruned_mask,
            summary,
        ) = prune_edges_hybrid(
            edge_index,
            edge_scores,
            prune_threshold=effective_prune_threshold,
            downweight_threshold=effective_downweight_threshold,
            min_weight=min_weight,
        )
    else:
        raise ValueError("mode must be one of: hard, soft, hybrid.")

    summary.update(
        {
            "prune_ratio": float(summary["pruned_edges"] / max(1, summary["original_num_edges"])),
            "threshold_mode": threshold_mode,
            "requested_prune_threshold": float(prune_threshold),
            "effective_prune_threshold": float(effective_prune_threshold),
            "score_percentile": float(score_percentile),
            "requested_prune_ratio": float(prune_ratio or 0.0),
            "min_prune_edges": int(min_prune_edges or 0),
            "score_min": float(edge_scores.min()) if edge_scores.size else 0.0,
            "score_mean": float(edge_scores.mean()) if edge_scores.size else 0.0,
            "score_max": float(edge_scores.max()) if edge_scores.size else 0.0,
            "score_p90": float(np.percentile(edge_scores, 90)) if edge_scores.size else 0.0,
            "score_p95": float(np.percentile(edge_scores, 95)) if edge_scores.size else 0.0,
            "score_p99": float(np.percentile(edge_scores, 99)) if edge_scores.size else 0.0,
            "downweight_threshold": float(effective_downweight_threshold),
            "min_weight": float(min_weight),
            **_component_summary(score_components),
        }
    )
    return defended_edge_index, defended_edge_weight, edge_scores, score_components, summary
