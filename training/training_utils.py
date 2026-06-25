"""Reusable training utilities for baseline GNN experiments."""

import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_ROOT / "results" / "plots" / ".matplotlib-cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError("Training utilities require torch. Install requirements-gnn.txt first.") from exc


GRAPH_ARTIFACTS_DIR = PROJECT_ROOT / "data" / "processed" / "graph_artifacts"


def set_seed(seed):
    """Set common random seeds for reproducible baseline runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    """Return a torch device, using CUDA automatically when requested and available."""
    def cuda_is_supported_by_build():
        if not torch.cuda.is_available():
            return False
        capability = torch.cuda.get_device_capability(0)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        supported_arches = set(torch.cuda.get_arch_list())
        if required_arch not in supported_arches:
            print(
                "CUDA is visible, but this PyTorch build does not support "
                f"GPU architecture {required_arch}. Falling back to CPU."
            )
            print(f"Supported CUDA architectures in this build: {sorted(supported_arches)}")
            return False
        return True

    if device_arg == "auto":
        return torch.device("cuda" if cuda_is_supported_by_build() else "cpu")
    if device_arg == "cuda" and not cuda_is_supported_by_build():
        print("CUDA was requested but is unavailable or unsupported. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_graph_artifacts(graph_dir=GRAPH_ARTIFACTS_DIR):
    """Load saved Step 2 graph artifacts from disk."""
    graph_path = Path(graph_dir)
    if not graph_path.is_absolute():
        graph_path = PROJECT_ROOT / graph_path
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
        "graph_summary.json",
    ]
    missing = [name for name in required if not (graph_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing graph artifacts. Complete Step 2 first. Missing: "
            + ", ".join(missing)
        )

    artifacts = {
        "X": np.load(graph_path / "X.npy"),
        "y": np.load(graph_path / "y.npy"),
        "edge_index_directed": np.load(graph_path / "edge_index_directed.npy"),
        "edge_index_undirected": np.load(graph_path / "edge_index_undirected.npy"),
        "timesteps": np.load(graph_path / "timesteps.npy"),
        "train_mask": np.load(graph_path / "train_mask.npy"),
        "val_mask": np.load(graph_path / "val_mask.npy"),
        "test_mask": np.load(graph_path / "test_mask.npy"),
        "labeled_mask": np.load(graph_path / "labeled_mask.npy"),
        "unknown_mask": np.load(graph_path / "unknown_mask.npy"),
        "graph_summary": json.loads((graph_path / "graph_summary.json").read_text(encoding="utf-8")),
        "artifact_dir": str(graph_path),
    }
    return artifacts


def compute_class_weights(y, train_mask):
    """Compute inverse-frequency class weights from train labels only."""
    train_labels = y[train_mask & (y != -1)]
    counts = torch.bincount(train_labels, minlength=2).float()
    if torch.any(counts == 0):
        return torch.ones(2, dtype=torch.float32, device=y.device)

    total = counts.sum()
    weights = total / (2.0 * counts)
    return weights.to(dtype=torch.float32, device=y.device)


def save_json(path, data):
    """Save a dictionary as pretty JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_training_curves(log_rows, loss_path, f1_path):
    """Save simple loss and F1 training curves."""
    if not log_rows:
        return

    epochs = [row["epoch"] for row in log_rows]
    train_loss = [row["train_loss"] for row in log_rows]
    val_loss = [row["val_loss"] for row in log_rows]
    train_f1 = [row["train_f1"] for row in log_rows]
    val_f1 = [row["val_f1"] for row in log_rows]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline GNN Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_f1, label="train F1")
    plt.plot(epochs, val_f1, label="validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Baseline GNN F1 Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f1_path, dpi=150)
    plt.close()


def count_labels(y, mask):
    """Count illicit and licit labels within a mask."""
    selected = mask & (y != -1)
    return {
        "num_nodes": int(selected.sum().item()),
        "illicit_count": int(((y == 1) & selected).sum().item()),
        "licit_count": int(((y == 0) & selected).sum().item()),
    }
