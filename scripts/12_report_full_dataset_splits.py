"""Report full Elliptic dataset graph and supervised split statistics."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_GRAPH_DIR = PROJECT_ROOT / "data" / "processed" / "full" / "graph_artifacts"
SUBSET_GRAPH_DIR = PROJECT_ROOT / "data" / "processed" / "graph_artifacts"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def load_graph_summary(graph_dir):
    """Load graph summary JSON."""
    path = Path(graph_dir) / "graph_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing graph_summary.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_arrays(graph_dir):
    """Load arrays needed for independent split counts."""
    graph_path = Path(graph_dir)
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
    ]
    missing = [name for name in required if not (graph_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing graph artifacts: {', '.join(missing)}")
    return {name.removesuffix(".npy"): np.load(graph_path / name) for name in required}


def split_counts(y, mask):
    """Return labeled, illicit, and licit counts inside a split mask."""
    selected = mask & (y != -1)
    return {
        "labeled": int(selected.sum()),
        "illicit": int(((y == 1) & selected).sum()),
        "licit": int(((y == 0) & selected).sum()),
    }


def percentage(value, denominator):
    """Compute a stable percentage."""
    return float((value / denominator) * 100.0) if denominator else 0.0


def write_metric_csv(path, metrics):
    """Write metric/value rows."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


def write_comparison(full_summary):
    """Compare existing 100k subset graph summary with full graph summary."""
    subset_summary = load_graph_summary(SUBSET_GRAPH_DIR)
    metric_map = {
        "total_nodes": ("num_nodes", "num_nodes"),
        "total_edges_directed": ("num_directed_edges", "num_directed_edges"),
        "total_edges_undirected": ("num_undirected_edges", "num_undirected_edges"),
        "total_timesteps": ("num_timesteps", "num_timesteps"),
        "known_labeled_nodes": ("labeled_nodes", "labeled_nodes"),
        "unknown_nodes": ("unknown_nodes", "unknown_nodes"),
        "illicit_nodes": ("illicit_nodes", "illicit_nodes"),
        "licit_nodes": ("licit_nodes", "licit_nodes"),
        "train_labeled_nodes": ("train_nodes", "train_nodes"),
        "val_labeled_nodes": ("val_nodes", "val_nodes"),
        "test_labeled_nodes": ("test_nodes", "test_nodes"),
    }
    rows = []
    for metric, (subset_key, full_key) in metric_map.items():
        subset_value = int(subset_summary[subset_key])
        full_value = int(full_summary[full_key])
        rows.append(
            {
                "metric": metric,
                "subset_100k": subset_value,
                "full_dataset": full_value,
                "difference": full_value - subset_value,
            }
        )
    output_path = TABLES_DIR / "dataset_size_comparison_100k_vs_full.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main():
    """Print and save the full dataset summary reports."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(FULL_GRAPH_DIR)
    y = arrays["y"]
    train = split_counts(y, arrays["train_mask"])
    val = split_counts(y, arrays["val_mask"])
    test = split_counts(y, arrays["test_mask"])

    total_nodes = int(arrays["X"].shape[0])
    total_directed = int(arrays["edge_index_directed"].shape[1])
    total_undirected = int(arrays["edge_index_undirected"].shape[1])
    num_features = int(arrays["X"].shape[1])
    timesteps = arrays["timesteps"]
    total_timesteps = int(len(np.unique(timesteps)))
    min_timestep = int(timesteps.min())
    max_timestep = int(timesteps.max())
    labeled_nodes = int(arrays["labeled_mask"].sum())
    unknown_nodes = int(arrays["unknown_mask"].sum())
    illicit_nodes = int((y == 1).sum())
    licit_nodes = int((y == 0).sum())

    summary = {
        "total_nodes": total_nodes,
        "total_directed_edges": total_directed,
        "total_undirected_edges": total_undirected,
        "num_features": num_features,
        "total_timesteps": total_timesteps,
        "min_timestep": min_timestep,
        "max_timestep": max_timestep,
        "known_labeled_nodes": labeled_nodes,
        "unknown_nodes": unknown_nodes,
        "illicit_nodes": illicit_nodes,
        "licit_nodes": licit_nodes,
        "train_labeled_nodes": train["labeled"],
        "val_labeled_nodes": val["labeled"],
        "test_labeled_nodes": test["labeled"],
        "train_illicit": train["illicit"],
        "train_licit": train["licit"],
        "val_illicit": val["illicit"],
        "val_licit": val["licit"],
        "test_illicit": test["illicit"],
        "test_licit": test["licit"],
        "labeled_percentage_of_total": percentage(labeled_nodes, total_nodes),
        "unknown_percentage_of_total": percentage(unknown_nodes, total_nodes),
        "train_percentage_of_labeled": percentage(train["labeled"], labeled_nodes),
        "val_percentage_of_labeled": percentage(val["labeled"], labeled_nodes),
        "test_percentage_of_labeled": percentage(test["labeled"], labeled_nodes),
    }

    split_summary = {
        "train_labeled_nodes": train["labeled"],
        "validation_labeled_nodes": val["labeled"],
        "test_labeled_nodes": test["labeled"],
        "train_illicit_count": train["illicit"],
        "train_licit_count": train["licit"],
        "validation_illicit_count": val["illicit"],
        "validation_licit_count": val["licit"],
        "test_illicit_count": test["illicit"],
        "test_licit_count": test["licit"],
        "train_percentage_of_labeled": summary["train_percentage_of_labeled"],
        "validation_percentage_of_labeled": summary["val_percentage_of_labeled"],
        "test_percentage_of_labeled": summary["test_percentage_of_labeled"],
    }

    full_summary_for_comparison = {
        "num_nodes": total_nodes,
        "num_directed_edges": total_directed,
        "num_undirected_edges": total_undirected,
        "num_timesteps": total_timesteps,
        "labeled_nodes": labeled_nodes,
        "unknown_nodes": unknown_nodes,
        "illicit_nodes": illicit_nodes,
        "licit_nodes": licit_nodes,
        "train_nodes": train["labeled"],
        "val_nodes": val["labeled"],
        "test_nodes": test["labeled"],
    }

    summary_csv = TABLES_DIR / "full_dataset_summary.csv"
    split_csv = TABLES_DIR / "full_dataset_split_summary.csv"
    summary_json = TABLES_DIR / "full_dataset_summary.json"
    write_metric_csv(summary_csv, summary)
    write_metric_csv(split_csv, split_summary)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    comparison_csv = write_comparison(full_summary_for_comparison)

    print("\nFULL ELLIPTIC DATASET SUMMARY")
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total directed edges: {total_directed:,}")
    print(f"Total undirected edges: {total_undirected:,}")
    print(f"Number of features: {num_features:,}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Min timestep: {min_timestep}")
    print(f"Max timestep: {max_timestep}")

    print("\nLABEL SUMMARY")
    print(f"Known/labeled nodes: {labeled_nodes:,}")
    print(f"Unknown nodes: {unknown_nodes:,}")
    print(f"Illicit nodes: {illicit_nodes:,}")
    print(f"Licit nodes: {licit_nodes:,}")
    print(f"Labeled percentage out of total nodes: {summary['labeled_percentage_of_total']:.2f}%")
    print(f"Unknown percentage out of total nodes: {summary['unknown_percentage_of_total']:.2f}%")

    print("\nSUPERVISED SPLIT SUMMARY")
    print(f"Train labeled nodes: {train['labeled']:,}")
    print(f"Validation labeled nodes: {val['labeled']:,}")
    print(f"Test labeled nodes: {test['labeled']:,}")
    print(f"Train illicit count: {train['illicit']:,}")
    print(f"Train licit count: {train['licit']:,}")
    print(f"Validation illicit count: {val['illicit']:,}")
    print(f"Validation licit count: {val['licit']:,}")
    print(f"Test illicit count: {test['illicit']:,}")
    print(f"Test licit count: {test['licit']:,}")
    print(f"Train percentage out of labeled nodes: {summary['train_percentage_of_labeled']:.2f}%")
    print(f"Validation percentage out of labeled nodes: {summary['val_percentage_of_labeled']:.2f}%")
    print(f"Test percentage out of labeled nodes: {summary['test_percentage_of_labeled']:.2f}%")

    print("\nSaved reports")
    print(f"summary_csv: {summary_csv}")
    print(f"split_csv: {split_csv}")
    print(f"summary_json: {summary_json}")
    print(f"comparison_csv: {comparison_csv}")


if __name__ == "__main__":
    main()
