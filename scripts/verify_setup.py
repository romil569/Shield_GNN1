"""Verify the initial SHIELD-GNN project setup."""

import importlib.util
import platform
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FOLDERS = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "src/models",
    "src/attacks",
    "src/defense",
    "results/tables",
    "results/plots",
    "results/model_checkpoints",
    "paper/figures",
    "paper/manuscript",
    "scripts",
]
REQUIRED_DATASET_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
]
MAIN_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "sklearn",
    "networkx": "networkx",
}


def check_folder(relative_path):
    """Return whether a required folder exists."""
    return (PROJECT_ROOT / relative_path).is_dir()


def check_dataset_file(filename):
    """Return whether a required dataset file exists."""
    return (PROJECT_ROOT / "data" / "raw" / filename).is_file()


def check_package(import_name):
    """Return whether a Python package can be imported."""
    return importlib.util.find_spec(import_name) is not None


def main():
    """Run setup checks and print a concise status report."""
    print("SHIELD-GNN setup verification")
    print(f"Python version: {platform.python_version()}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Project root: {PROJECT_ROOT}")

    print("\nRequired folders:")
    for folder in REQUIRED_FOLDERS:
        status = "FOUND" if check_folder(folder) else "MISSING"
        print(f"{folder}: {status}")

    print("\nRequired dataset files:")
    for filename in REQUIRED_DATASET_FILES:
        status = "FOUND" if check_dataset_file(filename) else "MISSING"
        print(f"{filename}: {status}")

    print("\nMain packages:")
    for label, import_name in MAIN_PACKAGES.items():
        status = "FOUND" if check_package(import_name) else "MISSING"
        print(f"{label}: {status}")


if __name__ == "__main__":
    main()
