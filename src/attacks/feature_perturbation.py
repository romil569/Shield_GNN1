"""Node feature perturbation attacks for transaction graph simulation."""

import numpy as np

from src.attacks.attack_utils import set_attack_seed


def _candidate_indices(X, y, target_mask, strategy):
    """Select candidate nodes for feature perturbation."""
    if target_mask is not None:
        return np.flatnonzero(np.asarray(target_mask, dtype=bool))
    if strategy == "illicit":
        return np.flatnonzero(y == 1)
    if strategy == "licit":
        return np.flatnonzero(y == 0)
    if strategy == "labeled":
        return np.flatnonzero(y != -1)
    if strategy == "random":
        return np.arange(len(y))
    if strategy == "high_magnitude":
        norms = np.linalg.norm(X, axis=1)
        cutoff = np.quantile(norms, 0.90)
        return np.flatnonzero(norms >= cutoff)
    raise ValueError(f"Unsupported feature perturbation strategy: {strategy}")


def perturb_node_features(
    X,
    y,
    target_mask=None,
    perturbation_rate=0.05,
    noise_std=0.05,
    strategy="illicit",
    clip=True,
    seed=42,
):
    """Perturb selected node features with controlled Gaussian noise."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be [num_nodes, num_features] and y must align with X.")
    rng = set_attack_seed(seed)
    candidates = _candidate_indices(X, y, target_mask, strategy)
    if candidates.size == 0:
        candidates = np.arange(X.shape[0])
    num_perturbed = max(1, int(round(candidates.size * perturbation_rate)))
    num_perturbed = min(num_perturbed, candidates.size)
    perturbed_node_indices = rng.choice(candidates, size=num_perturbed, replace=False).astype(np.int64)

    attacked_X = X.copy()
    feature_std = X.std(axis=0)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std)
    noise = rng.normal(0.0, noise_std, size=attacked_X[perturbed_node_indices].shape) * feature_std
    attacked_X[perturbed_node_indices] += noise.astype(attacked_X.dtype, copy=False)
    if clip:
        attacked_X[perturbed_node_indices] = np.clip(
            attacked_X[perturbed_node_indices],
            X.min(axis=0),
            X.max(axis=0),
        )

    attack_summary = {
        "attack_type": "feature_perturbation",
        "original_num_nodes": int(X.shape[0]),
        "perturbed_nodes": int(len(perturbed_node_indices)),
        "perturbation_rate": float(perturbation_rate),
        "noise_std": float(noise_std),
        "strategy": strategy,
        "clip": bool(clip),
    }
    return attacked_X, perturbed_node_indices, attack_summary
