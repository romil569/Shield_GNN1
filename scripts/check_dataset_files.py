"""Check whether the required Elliptic Bitcoin Dataset files exist."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
REQUIRED_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
]


def main():
    """Print FOUND or MISSING for each required raw dataset file."""
    for filename in REQUIRED_FILES:
        file_path = DATA_RAW_DIR / filename
        status = "FOUND" if file_path.exists() else "MISSING"
        print(f"{filename}: {status}")


if __name__ == "__main__":
    main()
