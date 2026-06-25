"""Fake node injection attacks for transaction graph simulation."""

import numpy as np

from src.attacks.attack_utils import (
    compute_degree,
    make_undirected,
    remove_duplicate_edges,
    set_attack_seed,
    validate_graph_arrays,
)


def _target_nodes(y, edge_index, connect_to, count, rng):
    """Select real nodes to connect fake nodes to."""
    if connect_to == "illicit":
        candidates = np.flatnonzero(y == 1)
    elif connect_to == "licit":
        candidates = np.flatnonzero(y == 0)
    elif connect_to == "high_degree":
        degree = compute_degree(edge_index, len(y))
        top_count = max(count, min(len(y), 1000))
        candidates = np.argsort(degree)[-top_count:]
    elif connect_to == "random":
        candidates = np.arange(len(y))
    else:
        raise ValueError(f"Unsupported connect_to strategy: {connect_to}")
    if candidates.size == 0:
        candidates = np.arange(len(y))
    return rng.choice(candidates, size=count, replace=True).astype(np.int64)


def _fake_features(X, target_indices, num_fake_nodes, feature_strategy, noise_std, rng):
    """Generate fake node features from real feature statistics."""
    if feature_strategy == "mean_noise":
        base = X[target_indices].mean(axis=0, keepdims=True)
        features = np.repeat(base, num_fake_nodes, axis=0)
        features += rng.normal(0.0, noise_std, size=features.shape)
    elif feature_strategy == "copy_perturb":
        copied = rng.choice(target_indices, size=num_fake_nodes, replace=True)
        features = X[copied].copy()
        features += rng.normal(0.0, noise_std, size=features.shape)
    elif feature_strategy == "random_normal":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        features = rng.normal(mean, std * max(noise_std, 1e-6), size=(num_fake_nodes, X.shape[1]))
    else:
        raise ValueError(f"Unsupported feature_strategy: {feature_strategy}")
    return features.astype(X.dtype, copy=False)


def inject_fake_nodes(
    X,
    y,
    edge_index,
    timesteps=None,
    injection_rate=0.05,
    connect_to="illicit",
    edges_per_fake_node=3,
    feature_strategy="mean_noise",
    noise_std=0.05,
    seed=42,
):
    """Inject fake unknown-label nodes and connect them to selected real nodes."""
    X, y, edge_index = validate_graph_arrays(X, y, edge_index)
    rng = set_attack_seed(seed)
    original_num_nodes = X.shape[0]
    original_num_edges = edge_index.shape[1]
    num_fake_nodes = max(1, int(round(original_num_nodes * injection_rate)))
    edges_per_fake_node = max(1, int(edges_per_fake_node))

    target_count = max(num_fake_nodes, min(original_num_nodes, num_fake_nodes * edges_per_fake_node))
    targets_for_features = _target_nodes(y, edge_index, connect_to, target_count, rng)
    fake_X = _fake_features(X, targets_for_features, num_fake_nodes, feature_strategy, noise_std, rng)
    fake_y = np.full(num_fake_nodes, -1, dtype=y.dtype)
    fake_node_indices = np.arange(original_num_nodes, original_num_nodes + num_fake_nodes, dtype=np.int64)

    fake_edges = []
    for fake_node in fake_node_indices:
        targets = _target_nodes(y, edge_index, connect_to, edges_per_fake_node, rng)
        for target in targets:
            fake_edges.append((int(fake_node), int(target)))
            fake_edges.append((int(target), int(fake_node)))
    fake_edges = np.asarray(fake_edges, dtype=np.int64).T if fake_edges else np.empty((2, 0), dtype=np.int64)

    attacked_X = np.vstack([X.copy(), fake_X])
    attacked_y = np.concatenate([y.copy(), fake_y])
    attacked_edge_index = remove_duplicate_edges(np.concatenate([edge_index.copy(), fake_edges], axis=1))

    attack_summary = {
        "attack_type": "fake_node_injection",
        "original_num_nodes": int(original_num_nodes),
        "fake_nodes_added": int(num_fake_nodes),
        "original_num_edges": int(original_num_edges),
        "fake_edges_added": int(attacked_edge_index.shape[1] - original_num_edges),
        "attacked_num_nodes": int(attacked_X.shape[0]),
        "attacked_num_edges": int(attacked_edge_index.shape[1]),
        "injection_rate": float(injection_rate),
        "feature_strategy": feature_strategy,
        "connect_to": connect_to,
        "edges_per_fake_node": int(edges_per_fake_node),
    }
    return attacked_X, attacked_y, attacked_edge_index, fake_node_indices, attack_summary
