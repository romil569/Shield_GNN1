"""Temporal graph utilities for SHIELD-GNN baseline experiments."""

import json
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "data" / "processed" / "graph_artifacts"


def load_temporal_graph_artifacts(artifact_dir=DEFAULT_ARTIFACT_DIR):
    """Load graph artifacts needed for temporal GNN baselines."""
    artifact_path = Path(artifact_dir)
    required = [
        "X.npy",
        "y.npy",
        "edge_index_directed.npy",
        "edge_index_undirected.npy",
        "timesteps.npy",
        "train_mask.npy",
        "val_mask.npy",
        "test_mask.npy",
        "labeled_mask.npy",
        "unknown_mask.npy",
        "temporal_snapshots.json",
        "graph_summary.json",
    ]
    missing = [name for name in required if not (artifact_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing temporal graph artifacts. Complete Step 2 first. Missing: "
            + ", ".join(missing)
        )

    return {
        "X": np.load(artifact_path / "X.npy"),
        "y": np.load(artifact_path / "y.npy"),
        "edge_index_directed": np.load(artifact_path / "edge_index_directed.npy"),
        "edge_index_undirected": np.load(artifact_path / "edge_index_undirected.npy"),
        "timesteps": np.load(artifact_path / "timesteps.npy"),
        "train_mask": np.load(artifact_path / "train_mask.npy"),
        "val_mask": np.load(artifact_path / "val_mask.npy"),
        "test_mask": np.load(artifact_path / "test_mask.npy"),
        "labeled_mask": np.load(artifact_path / "labeled_mask.npy"),
        "unknown_mask": np.load(artifact_path / "unknown_mask.npy"),
        "temporal_snapshots": load_temporal_snapshots(artifact_path),
        "graph_summary": json.loads((artifact_path / "graph_summary.json").read_text(encoding="utf-8")),
    }


def load_temporal_snapshots(artifact_dir=DEFAULT_ARTIFACT_DIR):
    """Load temporal snapshot metadata created during graph construction."""
    snapshot_path = Path(artifact_dir) / "temporal_snapshots.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"Temporal snapshots not found: {snapshot_path}. Complete Step 2 first."
        )
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def build_snapshot_edge_index(edge_index, node_indices):
    """Return global-index edges whose endpoints are both in ``node_indices``."""
    node_indices = np.asarray(node_indices, dtype=np.int64)
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges], found {edge_index.shape}")
    if node_indices.size == 0:
        return np.empty((2, 0), dtype=np.int64)

    source_mask = np.isin(edge_index[0], node_indices)
    target_mask = np.isin(edge_index[1], node_indices)
    return edge_index[:, source_mask & target_mask].astype(np.int64, copy=False)


def get_nodes_for_timestep(timesteps, timestep):
    """Return global node indices whose timestep equals ``timestep``."""
    timesteps = np.asarray(timesteps)
    return np.flatnonzero(timesteps == timestep).astype(np.int64)


def get_cumulative_nodes_until_timestep(timesteps, timestep):
    """Return global node indices whose timestep is at most ``timestep``."""
    timesteps = np.asarray(timesteps)
    return np.flatnonzero(timesteps <= timestep).astype(np.int64)


def create_snapshot_tensors(X, y, edge_index, timesteps, snapshot_metadata, mode="cumulative"):
    """Create lightweight temporal snapshot tensor metadata.

    This keeps global node indices by default and avoids duplicating the full
    feature matrix for each timestep. Use the global ``X`` and ``y`` arrays with
    each snapshot's node indices and edge index when materializing a snapshot.
    """
    if mode not in {"cumulative", "current"}:
        raise ValueError("mode must be either 'cumulative' or 'current'.")

    snapshots = []
    for snapshot in snapshot_metadata:
        timestep = int(snapshot["timestep"])
        if mode == "cumulative":
            node_indices = get_cumulative_nodes_until_timestep(timesteps, timestep)
        else:
            node_indices = get_nodes_for_timestep(timesteps, timestep)

        snapshot_edge_index = build_snapshot_edge_index(edge_index, node_indices)
        labels = y[node_indices] if len(node_indices) else np.array([], dtype=np.int64)
        snapshots.append(
            {
                "timestep": timestep,
                "node_indices": node_indices,
                "edge_index": snapshot_edge_index,
                "num_nodes": int(len(node_indices)),
                "num_edges": int(snapshot_edge_index.shape[1]),
                "num_labeled_nodes": int(((labels == 0) | (labels == 1)).sum()),
                "num_illicit": int((labels == 1).sum()),
                "num_licit": int((labels == 0).sum()),
                "num_unknown": int((labels == -1).sum()),
            }
        )
    return snapshots


def summarize_temporal_data(X, y, edge_index, timesteps, snapshots):
    """Summarize temporal graph data for reports and verification."""
    unique_timesteps = np.array(sorted(np.unique(timesteps).tolist()), dtype=np.int64)
    return {
        "num_nodes": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "num_edges": int(edge_index.shape[1]),
        "num_timesteps": int(len(unique_timesteps)),
        "timestep_min": int(unique_timesteps.min()),
        "timestep_max": int(unique_timesteps.max()),
        "num_snapshots": int(len(snapshots)),
        "labeled_nodes": int(((y == 0) | (y == 1)).sum()),
        "unknown_nodes": int((y == -1).sum()),
        "illicit_nodes": int((y == 1).sum()),
        "licit_nodes": int((y == 0).sum()),
    }


def make_small_temporal_subgraph(artifacts, max_nodes=2048):
    """Create a relabeled small temporal subgraph for safe dry-runs."""
    selected_nodes = np.arange(min(max_nodes, artifacts["X"].shape[0]), dtype=np.int64)
    edge_index = artifacts["edge_index_undirected"]
    edge_mask = np.isin(edge_index[0], selected_nodes) & np.isin(edge_index[1], selected_nodes)
    small_edges = edge_index[:, edge_mask]
    if small_edges.shape[1] == 0:
        raise ValueError("Small dry-run temporal subgraph has no edges.")

    local_index = {int(node_idx): int(idx) for idx, node_idx in enumerate(selected_nodes)}
    relabeled_edges = np.vectorize(local_index.get)(small_edges).astype(np.int64)
    return {
        "X": artifacts["X"][selected_nodes],
        "y": artifacts["y"][selected_nodes],
        "edge_index_undirected": relabeled_edges,
        "timesteps": artifacts["timesteps"][selected_nodes],
        "train_mask": artifacts["train_mask"][selected_nodes] & (artifacts["y"][selected_nodes] != -1),
    }


def temporal_numpy_to_torch(small_artifacts, device):
    """Convert a temporal artifact dictionary into torch tensors."""
    return {
        "x": torch.tensor(small_artifacts["X"], dtype=torch.float32, device=device),
        "y": torch.tensor(small_artifacts["y"], dtype=torch.long, device=device),
        "edge_index": torch.tensor(
            small_artifacts["edge_index_undirected"], dtype=torch.long, device=device
        ),
        "timesteps": torch.tensor(small_artifacts["timesteps"], dtype=torch.long, device=device),
        "train_mask": torch.tensor(small_artifacts["train_mask"], dtype=torch.bool, device=device),
    }
