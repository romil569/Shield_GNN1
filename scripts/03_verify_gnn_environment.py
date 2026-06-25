"""Verify GNN dependencies and graph artifacts without training."""

import importlib.util
import platform
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = PROJECT_ROOT / "data" / "processed" / "graph_artifacts"
REQUIRED_GRAPH_FILES = [
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


def import_status(module_name):
    """Return whether a module is importable."""
    return importlib.util.find_spec(module_name) is not None


def main():
    """Print environment and graph artifact verification details."""
    print("SHIELD-GNN Step 3 environment verification")
    print(f"Python version: {platform.python_version()}")
    print(f"Project root: {PROJECT_ROOT}")

    print(f"NumPy importable: {import_status('numpy')}")
    print(f"Pandas importable: {import_status('pandas')}")

    torch_available = import_status("torch")
    print(f"Torch importable: {torch_available}")
    if torch_available:
        import torch

        print(f"Torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA device name: N/A")
    else:
        print("Torch version: NOT INSTALLED")
        print("CUDA available: False")
        print("CUDA device name: N/A")

    pyg_available = import_status("torch_geometric")
    print(f"torch_geometric importable: {pyg_available}")

    print("\nGraph artifact files:")
    missing = []
    for filename in REQUIRED_GRAPH_FILES:
        path = GRAPH_DIR / filename
        exists = path.is_file()
        print(f"{filename}: {'FOUND' if exists else 'MISSING'}")
        if not exists:
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            "Missing graph artifact files. Complete Step 2 first. Missing: "
            + ", ".join(missing)
        )

    X = np.load(GRAPH_DIR / "X.npy")
    y = np.load(GRAPH_DIR / "y.npy")
    edge_index = np.load(GRAPH_DIR / "edge_index_undirected.npy")
    train_mask = np.load(GRAPH_DIR / "train_mask.npy")
    val_mask = np.load(GRAPH_DIR / "val_mask.npy")
    test_mask = np.load(GRAPH_DIR / "test_mask.npy")

    print("\nGraph tensor shapes and masks:")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    print(f"Undirected edge index shape: {edge_index.shape}")
    print(f"Train mask count: {int(train_mask.sum())}")
    print(f"Validation mask count: {int(val_mask.sum())}")
    print(f"Test mask count: {int(test_mask.sum())}")

    if not torch_available or not pyg_available:
        print("\nGNN packages are not fully installed yet. Install requirements-gnn.txt before training.")


if __name__ == "__main__":
    main()
