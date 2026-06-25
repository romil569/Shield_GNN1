"""Suspicious edge scoring for robust edge pruning."""

import numpy as np

from src.defense.pruning_utils import compute_degree_np, normalize_scores, validate_edge_index


def compute_feature_distance_scores(X, edge_index):
    """Score edges by endpoint feature distance."""
    X = np.asarray(X, dtype=np.float32)
    edge_index = validate_edge_index(edge_index, X.shape[0])
    if edge_index.shape[1] == 0:
        return np.array([], dtype=np.float32)
    diffs = X[edge_index[0]] - X[edge_index[1]]
    distances = np.linalg.norm(diffs, axis=1)
    return normalize_scores(distances)


def compute_degree_abnormality_scores(edge_index, num_nodes):
    """Score edges attached to unusually high-degree endpoints."""
    edge_index = validate_edge_index(edge_index, num_nodes)
    if edge_index.shape[1] == 0:
        return np.array([], dtype=np.float32)
    degree = compute_degree_np(edge_index, num_nodes).astype(np.float32)
    endpoint_degree = np.maximum(degree[edge_index[0]], degree[edge_index[1]])
    return normalize_scores(endpoint_degree)


def compute_label_inconsistency_scores(y, edge_index):
    """Score label-inconsistent edges conservatively."""
    y = np.asarray(y)
    edge_index = validate_edge_index(edge_index, len(y))
    scores = np.zeros(edge_index.shape[1], dtype=np.float32)
    if edge_index.shape[1] == 0 or np.all(y == -1):
        return scores
    source_labels = y[edge_index[0]]
    target_labels = y[edge_index[1]]
    illicit_licit = ((source_labels == 1) & (target_labels == 0)) | (
        (source_labels == 0) & (target_labels == 1)
    )
    unknown_mix = (source_labels == -1) ^ (target_labels == -1)
    scores[unknown_mix] = 0.25
    scores[illicit_licit] = 1.0
    return scores


def compute_temporal_inconsistency_scores(timesteps, edge_index):
    """Score edges by absolute endpoint timestep difference."""
    if timesteps is None:
        return np.zeros(edge_index.shape[1], dtype=np.float32)
    timesteps = np.asarray(timesteps)
    edge_index = validate_edge_index(edge_index, len(timesteps))
    if edge_index.shape[1] == 0:
        return np.array([], dtype=np.float32)
    differences = np.abs(timesteps[edge_index[0]] - timesteps[edge_index[1]])
    return normalize_scores(differences)


def compute_combined_edge_scores(
    X,
    y,
    edge_index,
    timesteps=None,
    feature_weight=0.40,
    degree_weight=0.25,
    label_weight=0.20,
    temporal_weight=0.15,
):
    """Combine edge suspiciousness components into a final score."""
    edge_index = validate_edge_index(edge_index, X.shape[0])
    weights = np.array([feature_weight, degree_weight, label_weight, temporal_weight], dtype=np.float32)
    if weights.sum() <= 0:
        raise ValueError("At least one score weight must be positive.")
    weights = weights / weights.sum()
    components = {
        "feature_distance": compute_feature_distance_scores(X, edge_index),
        "degree_abnormality": compute_degree_abnormality_scores(edge_index, X.shape[0]),
        "label_inconsistency": compute_label_inconsistency_scores(y, edge_index),
        "temporal_inconsistency": compute_temporal_inconsistency_scores(timesteps, edge_index),
    }
    scores = (
        weights[0] * components["feature_distance"]
        + weights[1] * components["degree_abnormality"]
        + weights[2] * components["label_inconsistency"]
        + weights[3] * components["temporal_inconsistency"]
    )
    return normalize_scores(scores), components
