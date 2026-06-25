"""Utility helpers for temporal consistency learning."""

import json
from pathlib import Path

import numpy as np
import torch


def validate_logits_shape(logits):
    """Validate logits with shape [num_nodes, num_classes]."""
    if not torch.is_tensor(logits):
        raise TypeError("logits must be a torch.Tensor.")
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [num_nodes, num_classes], found {tuple(logits.shape)}")
    return logits


def validate_mask(mask, num_nodes):
    """Validate or create a boolean mask with shape [num_nodes]."""
    if mask is None:
        return torch.ones(num_nodes, dtype=torch.bool)
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.bool()
    if mask.ndim != 1 or mask.shape[0] != num_nodes:
        raise ValueError(f"mask must have shape [{num_nodes}], found {tuple(mask.shape)}")
    return mask


def sample_temporal_pairs(timesteps, max_time_gap=1, sample_pairs=5000, seed=42):
    """Sample node pairs whose timestep difference is within max_time_gap."""
    if not torch.is_tensor(timesteps):
        timesteps = torch.as_tensor(timesteps)
    device = timesteps.device
    timesteps_cpu = timesteps.detach().cpu().long().numpy()
    rng = np.random.default_rng(seed)
    buckets = {}
    for idx, timestep in enumerate(timesteps_cpu.tolist()):
        buckets.setdefault(int(timestep), []).append(idx)
    unique_times = sorted(buckets)
    pairs = []
    attempts = 0
    max_attempts = max(sample_pairs * 20, 100)
    while len(pairs) < sample_pairs and attempts < max_attempts:
        t1 = int(rng.choice(unique_times))
        candidate_times = [t for t in unique_times if 0 < abs(t - t1) <= max_time_gap]
        if not candidate_times:
            attempts += 1
            continue
        t2 = int(rng.choice(candidate_times))
        i = int(rng.choice(buckets[t1]))
        j = int(rng.choice(buckets[t2]))
        if i != j:
            pairs.append((i, j))
        attempts += 1
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device).T


def sample_edge_indices(edge_index, sample_edges=20000, seed=42):
    """Sample edge column indices from an edge_index tensor."""
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index)
    num_edges = edge_index.shape[1]
    if num_edges == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)
    if sample_edges is None or sample_edges >= num_edges:
        return torch.arange(num_edges, dtype=torch.long, device=edge_index.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(num_edges, generator=generator)[:sample_edges]
    return indices.to(edge_index.device)


def build_reliable_edge_mask(edge_scores=None, threshold=0.5):
    """Build a mask for edges with suspiciousness below threshold."""
    if edge_scores is None:
        return None
    if not torch.is_tensor(edge_scores):
        edge_scores = torch.as_tensor(edge_scores)
    return edge_scores <= threshold


def convert_numpy_to_torch_safe(array, device=None, dtype=None):
    """Convert NumPy arrays or tensors into torch tensors safely."""
    if torch.is_tensor(array):
        tensor = array
    else:
        tensor = torch.as_tensor(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def save_consistency_report(path, report):
    """Save temporal consistency report JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
