"""Prepare a temporal 100k-node subset of the Elliptic Bitcoin Dataset."""

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

from src.data_loader import load_classes, load_edges, load_features, verify_elliptic_files


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
MAX_NODES = 100_000


def _label_counts(classes):
    """Return licit, illicit, unknown, labeled, and unlabeled counts."""
    licit_count = int((classes["label"] == 0).sum())
    illicit_count = int((classes["label"] == 1).sum())
    unknown_count = int((classes["label"] == -1).sum())
    return {
        "licit_count": licit_count,
        "illicit_count": illicit_count,
        "unknown_count": unknown_count,
        "labeled_count": licit_count + illicit_count,
        "unlabeled_count": unknown_count,
    }


def select_complete_timesteps(features, max_nodes=MAX_NODES):
    """Select complete timesteps without exceeding ``max_nodes``."""
    timestep_counts = (
        features.groupby("timestep", sort=True)
        .size()
        .reset_index(name="node_count")
        .sort_values("timestep")
    )

    selected_timesteps = []
    selected_node_count = 0

    for row in timestep_counts.itertuples(index=False):
        next_count = selected_node_count + int(row.node_count)
        if next_count > max_nodes:
            break
        selected_timesteps.append(int(row.timestep))
        selected_node_count = next_count

    if not selected_timesteps:
        raise ValueError(
            "No complete timestep can be selected without exceeding "
            f"the max_nodes limit of {max_nodes:,}."
        )

    return selected_timesteps, selected_node_count


def save_summary_table(summary):
    """Save a readable metric/value summary CSV."""
    summary_rows = []
    for key, value in summary.items():
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        summary_rows.append({"metric": key, "value": value})

    table_path = TABLES_DIR / "elliptic_100k_dataset_summary.csv"
    pd.DataFrame(summary_rows).to_csv(table_path, index=False)
    return table_path


def save_plots(subset_classes, subset_features):
    """Save class and timestep distribution plots for the subset."""
    label_names = {0: "licit", 1: "illicit", -1: "unknown"}
    plot_classes = subset_classes.copy()
    plot_classes["class_name"] = plot_classes["label"].map(label_names)

    class_plot_path = PLOTS_DIR / "class_distribution_100k.png"
    plt.figure(figsize=(7, 4))
    sns.countplot(
        data=plot_classes,
        x="class_name",
        order=["licit", "illicit", "unknown"],
        hue="class_name",
        palette={"licit": "#2a9d8f", "illicit": "#e76f51", "unknown": "#6c757d"},
        legend=False,
    )
    plt.title("Class Distribution in Elliptic 100k Temporal Subset")
    plt.xlabel("Class")
    plt.ylabel("Transaction count")
    plt.tight_layout()
    plt.savefig(class_plot_path, dpi=150)
    plt.close()

    timestep_plot_path = PLOTS_DIR / "timestep_distribution_100k.png"
    plt.figure(figsize=(10, 4))
    sns.countplot(data=subset_features, x="timestep", color="#457b9d")
    plt.title("Timestep Distribution in Elliptic 100k Temporal Subset")
    plt.xlabel("Timestep")
    plt.ylabel("Transaction count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(timestep_plot_path, dpi=150)
    plt.close()

    return class_plot_path, timestep_plot_path


def main():
    """Create and save the temporal 100k-node Elliptic subset."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Verifying raw Elliptic dataset files...")
    verify_elliptic_files(RAW_DIR)

    features = load_features(RAW_DIR)
    classes = load_classes(RAW_DIR)
    edges = load_edges(RAW_DIR)

    raw_node_count = int(len(features))
    raw_edge_count = int(len(edges))
    total_timesteps = int(features["timestep"].nunique())
    feature_columns = [column for column in features.columns if column.startswith("feature_")]
    feature_count = int(len(feature_columns))
    raw_counts = _label_counts(classes)

    print("\nRaw dataset summary")
    print(f"Total transactions/nodes: {raw_node_count:,}")
    print(f"Total edges: {raw_edge_count:,}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Feature dimension: {feature_count:,}")
    print(
        "Class distribution: "
        f"licit={raw_counts['licit_count']:,}, "
        f"illicit={raw_counts['illicit_count']:,}, "
        f"unknown={raw_counts['unknown_count']:,}"
    )

    selected_timesteps, selected_node_count = select_complete_timesteps(features, MAX_NODES)
    selected_features = features[features["timestep"].isin(selected_timesteps)].copy()
    selected_features = selected_features.sort_values(["timestep"], kind="mergesort").reset_index(
        drop=True
    )
    selected_txids = set(selected_features["txId"])

    subset_classes = classes[classes["txId"].isin(selected_txids)].copy()
    if len(subset_classes) != len(selected_features):
        raise ValueError(
            "Selected features and classes do not align: "
            f"{len(selected_features):,} feature rows vs {len(subset_classes):,} class rows."
        )

    subset_edges = edges[edges["txId1"].isin(selected_txids) & edges["txId2"].isin(selected_txids)]
    subset_edges = subset_edges.copy().reset_index(drop=True)

    node_mapping = selected_features[["txId", "timestep"]].copy()
    node_mapping.insert(1, "node_idx", range(len(node_mapping)))
    node_mapping = node_mapping.merge(classes[["txId", "label"]], on="txId", how="left")

    if node_mapping["label"].isna().any():
        raise ValueError("Node mapping contains missing labels after merging classes.")

    indexed_edges = subset_edges.merge(
        node_mapping[["txId", "node_idx"]],
        left_on="txId1",
        right_on="txId",
        how="left",
    ).rename(columns={"node_idx": "source_idx"})
    indexed_edges = indexed_edges.drop(columns=["txId"])
    indexed_edges = indexed_edges.merge(
        node_mapping[["txId", "node_idx"]],
        left_on="txId2",
        right_on="txId",
        how="left",
    ).rename(columns={"node_idx": "target_idx"})
    indexed_edges = indexed_edges.drop(columns=["txId"])
    indexed_edges = indexed_edges.rename(columns={"txId1": "source_txId", "txId2": "target_txId"})
    indexed_edges = indexed_edges[
        ["source_idx", "target_idx", "source_txId", "target_txId"]
    ].astype({"source_idx": int, "target_idx": int})

    subset_counts = _label_counts(subset_classes)
    summary = {
        "raw_node_count": raw_node_count,
        "raw_edge_count": raw_edge_count,
        "subset_node_count": int(len(selected_features)),
        "subset_edge_count": int(len(subset_edges)),
        "selected_timesteps": selected_timesteps,
        "min_timestep": int(min(selected_timesteps)),
        "max_timestep": int(max(selected_timesteps)),
        "feature_count": feature_count,
        **subset_counts,
    }

    output_paths = {
        "features": PROCESSED_DIR / "elliptic_100k_features.csv",
        "classes": PROCESSED_DIR / "elliptic_100k_classes.csv",
        "edgelist": PROCESSED_DIR / "elliptic_100k_edgelist.csv",
        "node_mapping": PROCESSED_DIR / "elliptic_100k_node_mapping.csv",
        "indexed_edgelist": PROCESSED_DIR / "elliptic_100k_edgelist_indexed.csv",
        "summary_json": PROCESSED_DIR / "elliptic_100k_summary.json",
    }

    selected_features.to_csv(output_paths["features"], index=False)
    subset_classes.to_csv(output_paths["classes"], index=False)
    subset_edges.to_csv(output_paths["edgelist"], index=False)
    node_mapping.to_csv(output_paths["node_mapping"], index=False)
    indexed_edges.to_csv(output_paths["indexed_edgelist"], index=False)
    output_paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_table_path = save_summary_table(summary)
    class_plot_path, timestep_plot_path = save_plots(subset_classes, selected_features)

    print("\nTemporal subset summary")
    print(f"Selected timestep range: {summary['min_timestep']} to {summary['max_timestep']}")
    print(f"Selected timesteps: {selected_timesteps}")
    print(f"Subset node count: {summary['subset_node_count']:,}")
    print(f"Subset edge count: {summary['subset_edge_count']:,}")
    print(
        "Subset class distribution: "
        f"licit={summary['licit_count']:,}, "
        f"illicit={summary['illicit_count']:,}, "
        f"unknown={summary['unknown_count']:,}"
    )

    print("\nSaved files")
    for label, path in output_paths.items():
        print(f"{label}: {path}")
    print(f"summary_table: {summary_table_path}")
    print(f"class_distribution_plot: {class_plot_path}")
    print(f"timestep_distribution_plot: {timestep_plot_path}")


if __name__ == "__main__":
    main()
