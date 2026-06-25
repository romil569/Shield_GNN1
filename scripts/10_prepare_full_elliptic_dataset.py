"""Prepare the full Elliptic Bitcoin Dataset without a node limit."""

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_classes, load_edges, load_features, verify_elliptic_files


RAW_DIR = PROJECT_ROOT / "data" / "raw"
FULL_DIR = PROJECT_ROOT / "data" / "processed" / "full"


def label_counts(classes):
    """Return label counts from normalized class data."""
    licit = int((classes["label"] == 0).sum())
    illicit = int((classes["label"] == 1).sum())
    unknown = int((classes["label"] == -1).sum())
    return {
        "labeled_nodes": licit + illicit,
        "unknown_nodes": unknown,
        "illicit_nodes": illicit,
        "licit_nodes": licit,
    }


def main():
    """Build full processed CSV files in data/processed/full."""
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    print("Verifying raw Elliptic dataset files...")
    verify_elliptic_files(RAW_DIR)

    features = load_features(RAW_DIR)
    classes = load_classes(RAW_DIR)
    edges = load_edges(RAW_DIR)

    features = features.sort_values(["timestep"], kind="mergesort").reset_index(drop=True)
    valid_txids = set(features["txId"])
    full_classes = classes[classes["txId"].isin(valid_txids)].copy().reset_index(drop=True)
    if len(full_classes) != len(features):
        raise ValueError(
            "Features/classes row count mismatch after filtering: "
            f"{len(features):,} features vs {len(full_classes):,} classes."
        )

    full_edges = edges[edges["txId1"].isin(valid_txids) & edges["txId2"].isin(valid_txids)]
    full_edges = full_edges.copy().reset_index(drop=True)

    node_mapping = features[["txId", "timestep"]].copy()
    node_mapping.insert(1, "node_idx", range(len(node_mapping)))
    node_mapping = node_mapping.merge(classes[["txId", "label"]], on="txId", how="left")
    if node_mapping["label"].isna().any():
        raise ValueError("Node mapping contains missing labels.")
    node_mapping["label"] = node_mapping["label"].astype(int)

    lookup = node_mapping[["txId", "node_idx"]]
    indexed_edges = full_edges.merge(
        lookup,
        left_on="txId1",
        right_on="txId",
        how="left",
    ).rename(columns={"node_idx": "source_idx"})
    indexed_edges = indexed_edges.drop(columns=["txId"])
    indexed_edges = indexed_edges.merge(
        lookup,
        left_on="txId2",
        right_on="txId",
        how="left",
    ).rename(columns={"node_idx": "target_idx"})
    indexed_edges = indexed_edges.drop(columns=["txId"])
    indexed_edges = indexed_edges.rename(columns={"txId1": "source_txId", "txId2": "target_txId"})
    indexed_edges = indexed_edges[["source_idx", "target_idx", "source_txId", "target_txId"]]
    if indexed_edges[["source_idx", "target_idx"]].isna().any().any():
        raise ValueError("Indexed edges contain missing node indices.")
    indexed_edges = indexed_edges.astype({"source_idx": int, "target_idx": int})

    feature_columns = [column for column in features.columns if column.startswith("feature_")]
    counts = label_counts(full_classes)
    summary = {
        "dataset_variant": "full",
        "raw_node_count": int(len(features)),
        "raw_edge_count": int(len(edges)),
        "full_node_count": int(len(features)),
        "full_edge_count": int(len(full_edges)),
        "filtered_invalid_edge_count": int(len(edges) - len(full_edges)),
        "total_timesteps": int(features["timestep"].nunique()),
        "min_timestep": int(features["timestep"].min()),
        "max_timestep": int(features["timestep"].max()),
        "feature_count": int(len(feature_columns)),
        **counts,
    }

    output_paths = {
        "features": FULL_DIR / "elliptic_full_features.csv",
        "classes": FULL_DIR / "elliptic_full_classes.csv",
        "edgelist": FULL_DIR / "elliptic_full_edgelist.csv",
        "node_mapping": FULL_DIR / "elliptic_full_node_mapping.csv",
        "indexed_edgelist": FULL_DIR / "elliptic_full_edgelist_indexed.csv",
        "summary_json": FULL_DIR / "elliptic_full_summary.json",
    }
    features.to_csv(output_paths["features"], index=False)
    full_classes.to_csv(output_paths["classes"], index=False)
    full_edges.to_csv(output_paths["edgelist"], index=False)
    node_mapping.to_csv(output_paths["node_mapping"], index=False)
    indexed_edges.to_csv(output_paths["indexed_edgelist"], index=False)
    output_paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFull Elliptic processed dataset summary")
    print(f"Total nodes: {summary['full_node_count']:,}")
    print(f"Total valid directed edges: {summary['full_edge_count']:,}")
    print(f"Total timesteps: {summary['total_timesteps']:,}")
    print(f"Feature count: {summary['feature_count']:,}")
    print(
        "Labels: "
        f"known={summary['labeled_nodes']:,}, unknown={summary['unknown_nodes']:,}, "
        f"illicit={summary['illicit_nodes']:,}, licit={summary['licit_nodes']:,}"
    )
    print("\nSaved files")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
