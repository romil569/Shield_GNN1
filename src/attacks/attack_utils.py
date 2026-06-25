"""Utility helpers for adversarial graph attack simulation."""

import json
import random
from pathlib import Path

import numpy as np


def set_attack_seed(seed):
    """Set deterministic seeds for attack simulations."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_edge_index_shape(edge_index):
    """Validate and return an int64 edge index with shape [2, num_edges]."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges], found {edge_index.shape}")
    return edge_index


def validate_graph_arrays(X, y, edge_index):
    """Validate feature, label, and edge arrays for graph attacks."""
    X = np.asarray(X)
    y = np.asarray(y)
    edge_index = ensure_edge_index_shape(edge_index)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D matrix, found shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D vector, found shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y node count mismatch: {X.shape[0]} vs {y.shape[0]}")
    if edge_index.size:
        if edge_index.min() < 0 or edge_index.max() >= X.shape[0]:
            raise ValueError("edge_index contains node IDs outside the valid node range.")
    bad_labels = set(np.unique(y).tolist()) - {-1, 0, 1}
    if bad_labels:
        raise ValueError(f"Unexpected labels found: {sorted(bad_labels)}")
    return X, y, edge_index


def remove_duplicate_edges(edge_index):
    """Remove duplicate directed edges while preserving [2, num_edges] format."""
    edge_index = ensure_edge_index_shape(edge_index)
    if edge_index.shape[1] == 0:
        return edge_index
    return np.unique(edge_index.T, axis=0).T.astype(np.int64, copy=False)


def make_undirected(edge_index):
    """Add reverse edges and remove duplicates."""
    edge_index = ensure_edge_index_shape(edge_index)
    if edge_index.shape[1] == 0:
        return edge_index
    reversed_edges = edge_index[[1, 0], :]
    return remove_duplicate_edges(np.concatenate([edge_index, reversed_edges], axis=1))


def compute_degree(edge_index, num_nodes):
    """Compute total degree for each node from a directed edge index."""
    edge_index = ensure_edge_index_shape(edge_index)
    degree = np.zeros(num_nodes, dtype=np.int64)
    if edge_index.shape[1]:
        np.add.at(degree, edge_index[0], 1)
        np.add.at(degree, edge_index[1], 1)
    return degree


def get_labeled_indices(y):
    """Return indices for labeled nodes."""
    return np.flatnonzero(np.asarray(y) != -1)


def get_illicit_indices(y):
    """Return indices for illicit labeled nodes."""
    return np.flatnonzero(np.asarray(y) == 1)


def get_licit_indices(y):
    """Return indices for licit labeled nodes."""
    return np.flatnonzero(np.asarray(y) == 0)


def select_target_nodes(y, mask=None, strategy="illicit", num_targets=None, seed=42):
    """Select target node indices for attacks."""
    rng = set_attack_seed(seed)
    y = np.asarray(y)
    if strategy == "illicit":
        candidates = get_illicit_indices(y)
    elif strategy == "licit":
        candidates = get_licit_indices(y)
    elif strategy == "labeled":
        candidates = get_labeled_indices(y)
    elif strategy in {"random", "unknown"}:
        candidates = np.arange(len(y)) if strategy == "random" else np.flatnonzero(y == -1)
    else:
        raise ValueError(f"Unsupported target selection strategy: {strategy}")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        candidates = candidates[mask[candidates]]
    if candidates.size == 0:
        candidates = np.arange(len(y))
    if num_targets is None or num_targets >= candidates.size:
        return candidates.astype(np.int64)
    return rng.choice(candidates, size=num_targets, replace=False).astype(np.int64)


def safe_numpy_save(path, array):
    """Create parent directories and save a NumPy array."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)


def save_attack_summary(path, summary):
    """Create parent directories and save attack summary JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def calculate_attack_stats(clean_X, clean_edge_index, attacked_X, attacked_edge_index):
    """Calculate basic clean-vs-attacked graph statistics."""
    clean_edge_index = ensure_edge_index_shape(clean_edge_index)
    attacked_edge_index = ensure_edge_index_shape(attacked_edge_index)
    return {
        "clean_num_nodes": int(clean_X.shape[0]),
        "clean_num_edges": int(clean_edge_index.shape[1]),
        "attacked_num_nodes": int(attacked_X.shape[0]),
        "attacked_num_edges": int(attacked_edge_index.shape[1]),
        "nodes_added": int(attacked_X.shape[0] - clean_X.shape[0]),
        "edges_added": int(attacked_edge_index.shape[1] - clean_edge_index.shape[1]),
    }
