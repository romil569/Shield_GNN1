"""Utility helpers for robust edge pruning defense."""

import json
from pathlib import Path

import numpy as np


def validate_edge_index(edge_index, num_nodes):
    """Validate edge_index shape and node range."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges], found {edge_index.shape}")
    if edge_index.shape[1] and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("edge_index contains node indices outside the valid range.")
    return edge_index


def remove_self_loops_np(edge_index):
    """Remove edges where source equals target."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    return edge_index[:, edge_index[0] != edge_index[1]]


def remove_duplicate_edges_np(edge_index):
    """Remove duplicate directed edges."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.shape[1] == 0:
        return edge_index
    return np.unique(edge_index.T, axis=0).T.astype(np.int64, copy=False)


def make_undirected_np(edge_index):
    """Add reverse edges and remove duplicates."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.shape[1] == 0:
        return edge_index
    reverse_edges = edge_index[[1, 0], :]
    return remove_duplicate_edges_np(np.concatenate([edge_index, reverse_edges], axis=1))


def compute_degree_np(edge_index, num_nodes):
    """Compute total degree from a directed edge index."""
    edge_index = validate_edge_index(edge_index, num_nodes)
    degree = np.zeros(num_nodes, dtype=np.int64)
    if edge_index.shape[1]:
        np.add.at(degree, edge_index[0], 1)
        np.add.at(degree, edge_index[1], 1)
    return degree


def normalize_scores(scores):
    """Normalize scores to [0, 1] safely."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores.astype(np.float32)
    min_score = float(np.nanmin(scores))
    max_score = float(np.nanmax(scores))
    if not np.isfinite(min_score) or not np.isfinite(max_score) or max_score <= min_score:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - min_score) / (max_score - min_score)).astype(np.float32)


def threshold_scores(scores, prune_threshold, downweight_threshold):
    """Create keep, downweight, and prune masks from suspiciousness scores."""
    scores = np.asarray(scores)
    pruned_mask = scores >= prune_threshold
    downweighted_mask = (scores >= downweight_threshold) & ~pruned_mask
    kept_mask = ~pruned_mask
    return kept_mask, downweighted_mask, pruned_mask


def save_defense_summary(path, summary):
    """Save defense summary JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def safe_save_numpy(path, array):
    """Save a NumPy array after creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
