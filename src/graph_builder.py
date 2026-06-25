"""Graph construction utilities for the processed Elliptic subset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PROCESSED_FILES = {
    "features": "elliptic_100k_features.csv",
    "classes": "elliptic_100k_classes.csv",
    "edgelist": "elliptic_100k_edgelist.csv",
    "mapping": "elliptic_100k_node_mapping.csv",
    "indexed_edges": "elliptic_100k_edgelist_indexed.csv",
    "summary": "elliptic_100k_summary.json",
}


def validate_processed_files(processed_dir):
    """Validate that Step 1 processed subset files exist."""
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed directory not found: {processed_path}. Complete Step 1 first."
        )

    resolved_files = {}
    missing = []
    for key, filename in REQUIRED_PROCESSED_FILES.items():
        file_path = processed_path / filename
        if file_path.is_file():
            resolved_files[key] = file_path
            print(f"{filename}: FOUND")
        else:
            missing.append(filename)
            print(f"{filename}: MISSING")

    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            "Required processed subset files are missing. Complete Step 1 first. "
            f"Missing: {missing_list}"
        )

    return resolved_files


def load_processed_subset(processed_dir):
    """Load Step 1 processed subset files as pandas DataFrames."""
    files = validate_processed_files(processed_dir)
    print("Loading processed 100k subset files...")

    features = pd.read_csv(files["features"], dtype={"txId": str})
    classes = pd.read_csv(files["classes"], dtype={"txId": str})
    edges = pd.read_csv(files["edgelist"], dtype={"txId1": str, "txId2": str})
    mapping = pd.read_csv(files["mapping"], dtype={"txId": str})
    indexed_edges = pd.read_csv(
        files["indexed_edges"], dtype={"source_txId": str, "target_txId": str}
    )
    summary = json.loads(files["summary"].read_text(encoding="utf-8"))

    required_mapping_columns = {"txId", "node_idx", "timestep", "label"}
    if not required_mapping_columns.issubset(mapping.columns):
        missing = sorted(required_mapping_columns - set(mapping.columns))
        raise ValueError(f"Node mapping is malformed. Missing columns: {missing}")

    features = features.merge(mapping[["txId", "node_idx"]], on="txId", how="left")
    if features["node_idx"].isna().any():
        raise ValueError("Some feature rows are missing node_idx values after merge.")

    features["node_idx"] = features["node_idx"].astype(int)
    features = features.sort_values("node_idx").reset_index(drop=True)
    mapping = mapping.sort_values("node_idx").reset_index(drop=True)

    expected_indices = np.arange(len(mapping))
    if not np.array_equal(mapping["node_idx"].to_numpy(dtype=np.int64), expected_indices):
        raise ValueError("node_idx values must be continuous from 0 to num_nodes - 1.")

    if not np.array_equal(features["node_idx"].to_numpy(dtype=np.int64), expected_indices):
        raise ValueError("Feature rows are not aligned with node_idx after sorting.")

    print(
        "Loaded subset: "
        f"{len(features):,} nodes, {len(indexed_edges):,} indexed edges, "
        f"{mapping['timestep'].nunique():,} timesteps"
    )
    return {
        "features": features,
        "classes": classes,
        "edges": edges,
        "mapping": mapping,
        "indexed_edges": indexed_edges,
        "step1_summary": summary,
    }


def build_feature_matrix(features_df):
    """Build a float32 feature matrix aligned by node_idx."""
    if "node_idx" not in features_df.columns:
        raise ValueError("features_df must contain node_idx to align feature rows.")

    feature_columns = [column for column in features_df.columns if column.startswith("feature_")]
    if not feature_columns:
        raise ValueError("No feature_ columns found in features_df.")

    ordered = features_df.sort_values("node_idx")
    X = ordered[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    print(f"Feature matrix X shape: {X.shape}")
    return X


def build_label_vector(mapping_df):
    """Build an int64 label vector aligned by node_idx."""
    if "label" not in mapping_df.columns:
        raise ValueError("mapping_df must contain a label column.")

    ordered = mapping_df.sort_values("node_idx")
    y = ordered["label"].to_numpy(dtype=np.int64)
    allowed_labels = {-1, 0, 1}
    observed = set(np.unique(y).tolist())
    if not observed.issubset(allowed_labels):
        raise ValueError(f"Unexpected labels found: {sorted(observed - allowed_labels)}")

    print(f"Label vector y shape: {y.shape}")
    return y


def build_edge_index(indexed_edges_df, num_nodes=None, undirected=False):
    """Build a directed or deduplicated undirected edge index array."""
    required_columns = {"source_idx", "target_idx"}
    if not required_columns.issubset(indexed_edges_df.columns):
        missing = sorted(required_columns - set(indexed_edges_df.columns))
        raise ValueError(f"Indexed edge list is malformed. Missing columns: {missing}")

    directed_edges = indexed_edges_df[["source_idx", "target_idx"]].to_numpy(dtype=np.int64)
    if directed_edges.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        if num_nodes is None:
            num_nodes = int(directed_edges.max()) + 1
        min_index = int(directed_edges.min())
        max_index = int(directed_edges.max())
        if min_index < 0 or max_index >= num_nodes:
            raise ValueError(
                "Edge indices out of range: "
                f"min={min_index}, max={max_index}, expected [0, {num_nodes - 1}]"
            )

        if undirected:
            reversed_edges = directed_edges[:, [1, 0]]
            combined = np.vstack([directed_edges, reversed_edges])
            combined = np.unique(combined, axis=0)
            edge_index = combined.T.astype(np.int64, copy=False)
        else:
            edge_index = directed_edges.T.astype(np.int64, copy=False)

    graph_type = "undirected" if undirected else "directed"
    print(f"{graph_type.capitalize()} edge_index shape: {edge_index.shape}")
    return edge_index


def create_temporal_masks(mapping_df):
    """Create chronological train, validation, and test masks for labeled nodes."""
    ordered = mapping_df.sort_values("node_idx")
    num_nodes = len(ordered)
    labels = ordered["label"].to_numpy(dtype=np.int64)
    timesteps = ordered["timestep"].to_numpy(dtype=np.int64)
    unique_timesteps = np.array(sorted(ordered["timestep"].unique()), dtype=np.int64)

    num_timesteps = len(unique_timesteps)
    train_end = max(1, int(np.floor(num_timesteps * 0.70)))
    val_end = max(train_end + 1, int(np.floor(num_timesteps * 0.85)))
    val_end = min(val_end, num_timesteps)

    train_timesteps = set(unique_timesteps[:train_end].tolist())
    val_timesteps = set(unique_timesteps[train_end:val_end].tolist())
    test_timesteps = set(unique_timesteps[val_end:].tolist())

    labeled_mask = labels != -1
    unknown_mask = labels == -1
    train_mask = np.array([timestep in train_timesteps for timestep in timesteps]) & labeled_mask
    val_mask = np.array([timestep in val_timesteps for timestep in timesteps]) & labeled_mask
    test_mask = np.array([timestep in test_timesteps for timestep in timesteps]) & labeled_mask

    masks = {
        "train_mask": train_mask.astype(bool),
        "val_mask": val_mask.astype(bool),
        "test_mask": test_mask.astype(bool),
        "labeled_mask": labeled_mask.astype(bool),
        "unknown_mask": unknown_mask.astype(bool),
    }

    for name, mask in masks.items():
        if mask.shape != (num_nodes,):
            raise ValueError(f"{name} has invalid shape {mask.shape}, expected {(num_nodes,)}")

    print(
        "Temporal split labeled nodes: "
        f"train={int(train_mask.sum()):,}, "
        f"val={int(val_mask.sum()):,}, "
        f"test={int(test_mask.sum()):,}"
    )
    return masks


def create_temporal_snapshots(features_df, mapping_df, indexed_edges_df):
    """Create cumulative temporal snapshot metadata.

    The default snapshot at timestep t contains all nodes with timestep <= t and
    all directed edges whose endpoints are both available by timestep t.
    """
    ordered_mapping = mapping_df.sort_values("node_idx")
    labels = ordered_mapping["label"].to_numpy(dtype=np.int64)
    timesteps = ordered_mapping["timestep"].to_numpy(dtype=np.int64)
    unique_timesteps = sorted(ordered_mapping["timestep"].unique().tolist())
    edge_pairs = indexed_edges_df[["source_idx", "target_idx"]].to_numpy(dtype=np.int64)

    snapshots = []
    for timestep in unique_timesteps:
        available_mask = timesteps <= timestep
        available_nodes = np.flatnonzero(available_mask)
        available_node_set = set(available_nodes.tolist())

        if len(edge_pairs) == 0:
            num_edges = 0
        else:
            source_available = np.isin(edge_pairs[:, 0], available_nodes)
            target_available = np.isin(edge_pairs[:, 1], available_nodes)
            num_edges = int((source_available & target_available).sum())

        snapshot_labels = labels[available_nodes]
        num_illicit = int((snapshot_labels == 1).sum())
        num_licit = int((snapshot_labels == 0).sum())
        num_unknown = int((snapshot_labels == -1).sum())

        snapshots.append(
            {
                "timestep": int(timestep),
                "node_indices": [int(node_idx) for node_idx in available_nodes],
                "num_nodes": int(len(available_node_set)),
                "num_edges": num_edges,
                "num_labeled_nodes": int(num_illicit + num_licit),
                "num_illicit": num_illicit,
                "num_licit": num_licit,
                "num_unknown": num_unknown,
            }
        )

    print(f"Created {len(snapshots):,} cumulative temporal snapshots")
    return snapshots


def save_graph_artifacts(output_dir, artifacts):
    """Save graph-ready NumPy arrays and JSON metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    npy_names = [
        "X",
        "y",
        "edge_index_directed",
        "edge_index_undirected",
        "timesteps",
        "train_mask",
        "val_mask",
        "test_mask",
        "labeled_mask",
        "unknown_mask",
    ]
    saved_paths = {}
    for name in npy_names:
        path = output_path / f"{name}.npy"
        np.save(path, artifacts[name])
        saved_paths[name] = str(path)

    snapshots_path = output_path / "temporal_snapshots.json"
    snapshots_path.write_text(
        json.dumps(artifacts["temporal_snapshots"], indent=2), encoding="utf-8"
    )
    saved_paths["temporal_snapshots"] = str(snapshots_path)

    summary_path = output_path / "graph_summary.json"
    summary_path.write_text(json.dumps(artifacts["graph_summary"], indent=2), encoding="utf-8")
    saved_paths["graph_summary"] = str(summary_path)

    print(f"Saved graph artifacts to {output_path}")
    return saved_paths


def _split_class_counts(y, mask):
    """Return illicit and licit counts inside a boolean mask."""
    return int(((y == 1) & mask).sum()), int(((y == 0) & mask).sum())


def summarize_graph_artifacts(artifacts):
    """Create a graph summary dictionary from graph artifacts."""
    X = artifacts["X"]
    y = artifacts["y"]
    directed = artifacts["edge_index_directed"]
    undirected = artifacts["edge_index_undirected"]
    timesteps = artifacts["timesteps"]
    train_mask = artifacts["train_mask"]
    val_mask = artifacts["val_mask"]
    test_mask = artifacts["test_mask"]
    labeled_mask = artifacts["labeled_mask"]
    unknown_mask = artifacts["unknown_mask"]

    num_nodes = int(X.shape[0])
    num_directed_edges = int(directed.shape[1])
    num_undirected_edges = int(undirected.shape[1])
    possible_directed_edges = num_nodes * (num_nodes - 1)
    graph_density = (
        float(num_directed_edges / possible_directed_edges) if possible_directed_edges else 0.0
    )

    train_illicit, train_licit = _split_class_counts(y, train_mask)
    val_illicit, val_licit = _split_class_counts(y, val_mask)
    test_illicit, test_licit = _split_class_counts(y, test_mask)

    return {
        "num_nodes": num_nodes,
        "num_features": int(X.shape[1]),
        "num_directed_edges": num_directed_edges,
        "num_undirected_edges": num_undirected_edges,
        "num_timesteps": int(len(np.unique(timesteps))),
        "timestep_min": int(timesteps.min()),
        "timestep_max": int(timesteps.max()),
        "labeled_nodes": int(labeled_mask.sum()),
        "unknown_nodes": int(unknown_mask.sum()),
        "illicit_nodes": int((y == 1).sum()),
        "licit_nodes": int((y == 0).sum()),
        "train_nodes": int(train_mask.sum()),
        "val_nodes": int(val_mask.sum()),
        "test_nodes": int(test_mask.sum()),
        "train_illicit": train_illicit,
        "train_licit": train_licit,
        "val_illicit": val_illicit,
        "val_licit": val_licit,
        "test_illicit": test_illicit,
        "test_licit": test_licit,
        "graph_density": graph_density,
        "average_degree_directed": float(num_directed_edges / num_nodes) if num_nodes else 0.0,
        "average_degree_undirected": float(num_undirected_edges / num_nodes) if num_nodes else 0.0,
    }
