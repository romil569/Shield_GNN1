"""Data loading utilities for the Elliptic Bitcoin Dataset."""

from pathlib import Path

import pandas as pd


FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"


def _resolve_raw_dir(raw_dir):
    """Return a validated raw dataset directory path."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_path}")
    if not raw_path.is_dir():
        raise NotADirectoryError(f"Raw dataset path is not a directory: {raw_path}")
    return raw_path


def _has_named_header(file_path, expected_names):
    """Return True when the first row looks like a named CSV header."""
    first_row = pd.read_csv(file_path, header=None, nrows=1, dtype=str)
    values = [str(value).strip() for value in first_row.iloc[0].tolist()]
    return values[: len(expected_names)] == expected_names


def verify_elliptic_files(raw_dir):
    """Verify that required Elliptic dataset files exist in ``raw_dir``."""
    raw_path = _resolve_raw_dir(raw_dir)
    required_files = [FEATURES_FILE, CLASSES_FILE, EDGES_FILE]
    status = {}

    for filename in required_files:
        file_path = raw_path / filename
        exists = file_path.is_file()
        status[filename] = exists
        print(f"{filename}: {'FOUND' if exists else 'MISSING'}")

    missing = [filename for filename, exists in status.items() if not exists]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Missing required Elliptic dataset files: {missing_list}")

    return status


def load_features(raw_dir):
    """Load transaction features with columns txId, timestep, feature_1, ..."""
    raw_path = _resolve_raw_dir(raw_dir)
    file_path = raw_path / FEATURES_FILE
    if not file_path.is_file():
        raise FileNotFoundError(f"Features file not found: {file_path}")

    print(f"Loading features from {file_path}")
    has_header = _has_named_header(file_path, ["txId", "timestep"])
    features = pd.read_csv(file_path, header=0 if has_header else None)

    if features.shape[1] < 3:
        raise ValueError(
            f"Features file is malformed: expected at least 3 columns, found {features.shape[1]}"
        )

    feature_count = features.shape[1] - 2
    features.columns = ["txId", "timestep"] + [
        f"feature_{index}" for index in range(1, feature_count + 1)
    ]
    features["txId"] = features["txId"].astype(str)
    features["timestep"] = pd.to_numeric(features["timestep"], errors="raise").astype(int)

    print(
        "Loaded features: "
        f"{len(features):,} transactions, {feature_count:,} feature columns, "
        f"{features['timestep'].nunique():,} timesteps"
    )
    return features


def load_classes(raw_dir):
    """Load transaction classes and add numeric labels."""
    raw_path = _resolve_raw_dir(raw_dir)
    file_path = raw_path / CLASSES_FILE
    if not file_path.is_file():
        raise FileNotFoundError(f"Classes file not found: {file_path}")

    print(f"Loading classes from {file_path}")
    has_header = _has_named_header(file_path, ["txId", "class"])
    classes = pd.read_csv(file_path, header=0 if has_header else None, dtype=str)

    if classes.shape[1] != 2:
        raise ValueError(
            f"Classes file is malformed: expected 2 columns, found {classes.shape[1]}"
        )

    classes.columns = ["txId", "class"]
    classes["txId"] = classes["txId"].astype(str)
    normalized = classes["class"].astype(str).str.strip().str.lower()
    label_map = {"1": 1, "2": 0, "unknown": -1}
    classes["label"] = normalized.map(label_map)

    if classes["label"].isna().any():
        bad_values = sorted(classes.loc[classes["label"].isna(), "class"].unique().tolist())
        raise ValueError(f"Classes file contains unsupported class values: {bad_values}")

    classes["label"] = classes["label"].astype(int)
    print(
        "Loaded classes: "
        f"{len(classes):,} transactions "
        f"({(classes['label'] == 0).sum():,} licit, "
        f"{(classes['label'] == 1).sum():,} illicit, "
        f"{(classes['label'] == -1).sum():,} unknown)"
    )
    return classes


def load_edges(raw_dir):
    """Load transaction edges with columns txId1 and txId2."""
    raw_path = _resolve_raw_dir(raw_dir)
    file_path = raw_path / EDGES_FILE
    if not file_path.is_file():
        raise FileNotFoundError(f"Edges file not found: {file_path}")

    print(f"Loading edges from {file_path}")
    has_header = _has_named_header(file_path, ["txId1", "txId2"])
    edges = pd.read_csv(file_path, header=0 if has_header else None, dtype=str)

    if edges.shape[1] != 2:
        raise ValueError(f"Edges file is malformed: expected 2 columns, found {edges.shape[1]}")

    edges.columns = ["txId1", "txId2"]
    edges["txId1"] = edges["txId1"].astype(str)
    edges["txId2"] = edges["txId2"].astype(str)

    print(f"Loaded edges: {len(edges):,} directed edges")
    return edges
