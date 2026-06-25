"""Build graph-ready NumPy artifacts from the processed Elliptic subset."""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_ROOT / "results" / "plots" / ".matplotlib-cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_builder import (
    build_edge_index,
    build_feature_matrix,
    build_label_vector,
    create_temporal_masks,
    create_temporal_snapshots,
    load_processed_subset,
    save_graph_artifacts,
    summarize_graph_artifacts,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_ARTIFACTS_DIR = PROCESSED_DIR / "graph_artifacts"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def save_summary_table(summary):
    """Save graph summary as a readable metric/value CSV."""
    table_path = TABLES_DIR / "graph_construction_summary.csv"
    rows = [{"metric": key, "value": value} for key, value in summary.items()]
    pd.DataFrame(rows).to_csv(table_path, index=False)
    return table_path


def save_snapshot_table(snapshots):
    """Save temporal snapshot metadata as a readable CSV."""
    table_path = TABLES_DIR / "temporal_snapshot_summary.csv"
    columns = [
        "timestep",
        "num_nodes",
        "num_edges",
        "num_labeled_nodes",
        "num_illicit",
        "num_licit",
        "num_unknown",
    ]
    pd.DataFrame(snapshots)[columns].to_csv(table_path, index=False)
    return table_path


def save_plots(summary, snapshots):
    """Generate basic graph construction plots."""
    split_plot_path = PLOTS_DIR / "graph_split_distribution.png"
    snapshot_plot_path = PLOTS_DIR / "temporal_nodes_edges.png"
    class_timeline_plot_path = PLOTS_DIR / "temporal_illicit_licit_distribution.png"

    split_df = pd.DataFrame(
        [
            {"split": "train", "count": summary["train_nodes"]},
            {"split": "validation", "count": summary["val_nodes"]},
            {"split": "test", "count": summary["test_nodes"]},
        ]
    )
    plt.figure(figsize=(7, 4))
    sns.barplot(data=split_df, x="split", y="count", hue="split", legend=False)
    plt.title("Chronological Labeled Node Split")
    plt.xlabel("Split")
    plt.ylabel("Labeled node count")
    plt.tight_layout()
    plt.savefig(split_plot_path, dpi=150)
    plt.close()

    snapshot_df = pd.DataFrame(snapshots)
    plt.figure(figsize=(9, 4))
    plt.plot(snapshot_df["timestep"], snapshot_df["num_nodes"], marker="o", label="nodes")
    plt.plot(snapshot_df["timestep"], snapshot_df["num_edges"], marker="o", label="edges")
    plt.title("Cumulative Temporal Nodes and Edges")
    plt.xlabel("Timestep")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snapshot_plot_path, dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(
        snapshot_df["timestep"],
        snapshot_df["num_illicit"],
        marker="o",
        color="#e76f51",
        label="illicit",
    )
    plt.plot(
        snapshot_df["timestep"],
        snapshot_df["num_licit"],
        marker="o",
        color="#2a9d8f",
        label="licit",
    )
    plt.title("Cumulative Labeled Class Distribution")
    plt.xlabel("Timestep")
    plt.ylabel("Node count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(class_timeline_plot_path, dpi=150)
    plt.close()

    return split_plot_path, snapshot_plot_path, class_timeline_plot_path


def main():
    """Build graph artifacts and write Step 2 outputs."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Validating and loading processed 100k subset...")
    subset = load_processed_subset(PROCESSED_DIR)
    features = subset["features"]
    mapping = subset["mapping"]
    indexed_edges = subset["indexed_edges"]

    X = build_feature_matrix(features)
    y = build_label_vector(mapping)
    num_nodes = len(mapping)
    edge_index_directed = build_edge_index(indexed_edges, num_nodes=num_nodes, undirected=False)
    edge_index_undirected = build_edge_index(indexed_edges, num_nodes=num_nodes, undirected=True)
    timesteps = mapping.sort_values("node_idx")["timestep"].to_numpy(dtype="int64")
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

    saved_artifacts = save_graph_artifacts(GRAPH_ARTIFACTS_DIR, artifacts)
    summary_table_path = save_summary_table(artifacts["graph_summary"])
    snapshot_table_path = save_snapshot_table(snapshots)
    split_plot_path, snapshot_plot_path, class_timeline_plot_path = save_plots(
        artifacts["graph_summary"], snapshots
    )

    summary = artifacts["graph_summary"]
    print("\nGraph construction summary")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    print(f"Directed edge index shape: {edge_index_directed.shape}")
    print(f"Undirected edge index shape: {edge_index_undirected.shape}")
    print(f"Number of timesteps: {summary['num_timesteps']}")
    print(
        "Train/val/test labeled nodes: "
        f"{summary['train_nodes']:,}/"
        f"{summary['val_nodes']:,}/"
        f"{summary['test_nodes']:,}"
    )
    print(
        "Train class distribution: "
        f"illicit={summary['train_illicit']:,}, licit={summary['train_licit']:,}"
    )
    print(
        "Validation class distribution: "
        f"illicit={summary['val_illicit']:,}, licit={summary['val_licit']:,}"
    )
    print(
        "Test class distribution: "
        f"illicit={summary['test_illicit']:,}, licit={summary['test_licit']:,}"
    )
    print(f"Temporal snapshots created: {len(snapshots):,}")

    print("\nSaved artifact paths")
    for name, path in saved_artifacts.items():
        print(f"{name}: {path}")
    print(f"graph_summary_table: {summary_table_path}")
    print(f"temporal_snapshot_table: {snapshot_table_path}")
    print(f"graph_split_distribution_plot: {split_plot_path}")
    print(f"temporal_nodes_edges_plot: {snapshot_plot_path}")
    print(f"temporal_illicit_licit_distribution_plot: {class_timeline_plot_path}")


if __name__ == "__main__":
    main()
