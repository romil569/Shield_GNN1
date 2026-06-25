"""Fake edge injection attacks for transaction graph simulation."""

import numpy as np

from src.attacks.attack_utils import compute_degree, remove_duplicate_edges, set_attack_seed, validate_graph_arrays


def _sample_pairs(source_candidates, target_candidates, num_edges, rng):
    """Sample directed edge pairs while avoiding self-loops."""
    if source_candidates.size == 0:
        raise ValueError("No source candidates available for fake edge injection.")
    if target_candidates.size == 0:
        raise ValueError("No target candidates available for fake edge injection.")
    sources = rng.choice(source_candidates, size=num_edges, replace=True)
    targets = rng.choice(target_candidates, size=num_edges, replace=True)
    self_loop = sources == targets
    retries = 0
    while self_loop.any() and retries < 5:
        targets[self_loop] = rng.choice(target_candidates, size=int(self_loop.sum()), replace=True)
        self_loop = sources == targets
        retries += 1
    valid = sources != targets
    return np.vstack([sources[valid], targets[valid]]).astype(np.int64)


def inject_fake_edges(
    X,
    y,
    edge_index,
    injection_rate=0.05,
    strategy="illicit_to_licit",
    seed=42,
):
    """Inject adversarial edges without adding nodes."""
    X, y, edge_index = validate_graph_arrays(X, y, edge_index)
    rng = set_attack_seed(seed)
    original_num_edges = edge_index.shape[1]
    fake_edges_requested = max(1, int(round(original_num_edges * injection_rate)))

    if strategy == "illicit_to_licit":
        sources = np.flatnonzero(y == 1)
        targets = np.flatnonzero(y == 0)
    elif strategy == "illicit_to_unknown":
        sources = np.flatnonzero(y == 1)
        targets = np.flatnonzero(y == -1)
    elif strategy == "high_degree_bridge":
        degree = compute_degree(edge_index, X.shape[0])
        top_count = max(1, min(X.shape[0], fake_edges_requested))
        sources = np.argsort(degree)[-top_count:]
        targets = np.arange(X.shape[0])
    elif strategy == "random":
        sources = np.arange(X.shape[0])
        targets = np.arange(X.shape[0])
    else:
        raise ValueError(f"Unsupported fake edge strategy: {strategy}")

    if sources.size == 0:
        sources = np.arange(X.shape[0])
    if targets.size == 0:
        targets = np.arange(X.shape[0])

    fake_edges = _sample_pairs(sources, targets, fake_edges_requested, rng)
    attacked_edge_index = remove_duplicate_edges(np.concatenate([edge_index.copy(), fake_edges], axis=1))
    fake_edges_added = int(attacked_edge_index.shape[1] - original_num_edges)
    attack_summary = {
        "attack_type": "fake_edge_injection",
        "original_num_edges": int(original_num_edges),
        "fake_edges_requested": int(fake_edges_requested),
        "fake_edges_added": fake_edges_added,
        "attacked_num_edges": int(attacked_edge_index.shape[1]),
        "injection_rate": float(injection_rate),
        "strategy": strategy,
    }
    return attacked_edge_index, fake_edges, attack_summary
