"""Build graph artifacts for the full Elliptic Bitcoin Dataset."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_builder import (
    build_edge_index,
    build_feature_matrix,
    build_label_vector,
    create_temporal_masks,
    create_temporal_snapshots,
    save_graph_artifacts,
    summarize_graph_artifacts,
)


FULL_DIR = PROJECT_ROOT / "data" / "processed" / "full"
GRAPH_DIR = FULL_DIR / "graph_artifacts"


REQUIRED_FILES = {
    "features": "elliptic_full_features.csv",
    "classes": "elliptic_full_classes.csv",
    "edgelist": "elliptic_full_edgelist.csv",
    "mapping": "elliptic_full_node_mapping.csv",
    "indexed_edges": "elliptic_full_edgelist_indexed.csv",
    "summary": "elliptic_full_summary.json",
}


def load_full_processed():
    """Load full processed CSV files and align features by node_idx."""
    missing = [filename for filename in REQUIRED_FILES.values() if not (FULL_DIR / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing full processed files. Run scripts/10_prepare_full_elliptic_dataset.py first. "
            f"Missing: {', '.join(missing)}"
        )

    features = pd.read_csv(FULL_DIR / REQUIRED_FILES["features"], dtype={"txId": str})
    classes = pd.read_csv(FULL_DIR / REQUIRED_FILES["classes"], dtype={"txId": str})
    edges = pd.read_csv(FULL_DIR / REQUIRED_FILES["edgelist"], dtype={"txId1": str, "txId2": str})
    mapping = pd.read_csv(FULL_DIR / REQUIRED_FILES["mapping"], dtype={"txId": str})
    indexed_edges = pd.read_csv(
        FULL_DIR / REQUIRED_FILES["indexed_edges"],
        dtype={"source_txId": str, "target_txId": str},
    )
    summary = json.loads((FULL_DIR / REQUIRED_FILES["summary"]).read_text(encoding="utf-8"))

    features = features.merge(mapping[["txId", "node_idx"]], on="txId", how="left")
    if features["node_idx"].isna().any():
        raise ValueError("Some full feature rows are missing node_idx values.")
    features["node_idx"] = features["node_idx"].astype(int)
    features = features.sort_values("node_idx").reset_index(drop=True)
    mapping = mapping.sort_values("node_idx").reset_index(drop=True)
    expected = np.arange(len(mapping))
    if not np.array_equal(mapping["node_idx"].to_numpy(dtype=np.int64), expected):
        raise ValueError("Full node_idx values must be continuous from 0 to num_nodes - 1.")
    if not np.array_equal(features["node_idx"].to_numpy(dtype=np.int64), expected):
        raise ValueError("Full features are not aligned with node_idx.")

    return {
        "features": features,
        "classes": classes,
        "edges": edges,
        "mapping": mapping,
        "indexed_edges": indexed_edges,
        "summary": summary,
    }


def main():
    """Build and save full graph-ready NumPy artifacts."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading full processed Elliptic files...")
    full = load_full_processed()
    features = full["features"]
    mapping = full["mapping"]
    indexed_edges = full["indexed_edges"]

    X = build_feature_matrix(features)
    y = build_label_vector(mapping)
    num_nodes = len(mapping)
    edge_index_directed = build_edge_index(indexed_edges, num_nodes=num_nodes, undirected=False)
    edge_index_undirected = build_edge_index(indexed_edges, num_nodes=num_nodes, undirected=True)
    timesteps = mapping.sort_values("node_idx")["timestep"].to_numpy(dtype=np.int64)
    masks = create_temporal_masks(mapping)
    snapshots = create_temporal_snapshots(features, mapping, indexed_edges)

    artifacts = {
        "X": X,
        "y": y,
        "edge_index_directed": edge_index_directed,
        "edge_index_undirected": edge_index_undirected,
        "timesteps": timesteps,
        **masks,
        "temporal_snapshots": snapshots,
    }
    artifacts["graph_summary"] = summarize_graph_artifacts(artifacts)
    artifacts["graph_summary"]["dataset_variant"] = "full"
    artifacts["graph_summary"]["processed_summary"] = full["summary"]

    saved = save_graph_artifacts(GRAPH_DIR, artifacts)
    summary = artifacts["graph_summary"]
    print("\nFull graph artifact summary")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Directed edge index shape: {edge_index_directed.shape}")
    print(f"Undirected edge index shape: {edge_index_undirected.shape}")
    print(f"Timesteps: {summary['timestep_min']} to {summary['timestep_max']} ({summary['num_timesteps']})")
    print(
        "Train/val/test labeled nodes: "
        f"{summary['train_nodes']:,}/{summary['val_nodes']:,}/{summary['test_nodes']:,}"
    )
    print("\nSaved artifacts")
    for name, path in saved.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
