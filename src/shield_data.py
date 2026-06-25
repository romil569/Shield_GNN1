"""Data loading and compatibility helpers for final SHIELD-GNN integration."""

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "data" / "processed" / "graph_artifacts"


def load_clean_graph(clean_dir=CLEAN_DIR, edge_type="undirected"):
    """Load mandatory clean graph artifacts."""
    clean_path = Path(clean_dir)
    if not clean_path.is_absolute():
        clean_path = PROJECT_ROOT / clean_path
    edge_name = f"edge_index_{edge_type}.npy"
    required = ["X.npy", "y.npy", edge_name, "timesteps.npy", "train_mask.npy", "val_mask.npy", "test_mask.npy"]
    missing = [name for name in required if not (clean_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing clean graph artifacts: {', '.join(missing)}")
    return {
        "X": np.load(clean_path / "X.npy"),
        "y": np.load(clean_path / "y.npy"),
        "edge_index": np.load(clean_path / edge_name),
        "timesteps": np.load(clean_path / "timesteps.npy"),
        "train_mask": np.load(clean_path / "train_mask.npy"),
        "val_mask": np.load(clean_path / "val_mask.npy"),
        "test_mask": np.load(clean_path / "test_mask.npy"),
        "edge_weight": None,
        "graph_summary": json.loads((clean_path / "graph_summary.json").read_text(encoding="utf-8")),
        "source": str(clean_path),
    }


def load_attack_graph(attack_dir):
    """Load optional attacked graph artifacts."""
    if attack_dir is None:
        return None
    attack_path = Path(attack_dir)
    if not attack_path.is_absolute():
        attack_path = PROJECT_ROOT / attack_path
    required = ["attacked_X.npy", "attacked_y.npy", "attacked_edge_index.npy"]
    missing = [name for name in required if not (attack_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing attack graph artifacts: {', '.join(missing)}")
    graph = {
        "X": np.load(attack_path / "attacked_X.npy"),
        "y": np.load(attack_path / "attacked_y.npy"),
        "edge_index": np.load(attack_path / "attacked_edge_index.npy"),
        "timesteps": np.load(attack_path / "attacked_timesteps.npy") if (attack_path / "attacked_timesteps.npy").is_file() else None,
        "edge_weight": None,
        "source": str(attack_path),
    }
    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        path = attack_path / f"attacked_{mask_name}.npy"
        if path.is_file():
            graph[mask_name] = np.load(path)
    return graph


def load_defense_graph(defense_dir, clean_graph=None):
    """Load optional defended edge artifacts."""
    if defense_dir is None:
        return None
    defense_path = Path(defense_dir)
    if not defense_path.is_absolute():
        defense_path = PROJECT_ROOT / defense_path
    required = ["defended_edge_index.npy", "defended_edge_weight.npy"]
    missing = [name for name in required if not (defense_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing defense graph artifacts: {', '.join(missing)}")
    graph = {
        "edge_index": np.load(defense_path / "defended_edge_index.npy"),
        "edge_weight": np.load(defense_path / "defended_edge_weight.npy"),
        "source": str(defense_path),
    }
    if (defense_path / "defended_X.npy").is_file() and (defense_path / "defended_y.npy").is_file():
        graph.update(
            {
                "X": np.load(defense_path / "defended_X.npy"),
                "y": np.load(defense_path / "defended_y.npy"),
                "timesteps": np.load(defense_path / "defended_timesteps.npy") if (defense_path / "defended_timesteps.npy").is_file() else None,
            }
        )
        for mask_name in ["train_mask", "val_mask", "test_mask"]:
            path = defense_path / f"defended_{mask_name}.npy"
            if path.is_file():
                graph[mask_name] = np.load(path)
    elif clean_graph is not None:
        graph.update(
            {
                "X": clean_graph["X"],
                "y": clean_graph["y"],
                "timesteps": clean_graph["timesteps"],
                "train_mask": clean_graph["train_mask"],
                "val_mask": clean_graph["val_mask"],
                "test_mask": clean_graph["test_mask"],
            }
        )
    return graph


def prepare_shield_inputs(clean_dir=CLEAN_DIR, attack_dir=None, defense_dir=None, edge_type="undirected"):
    """Load clean, optional attack, and optional defense inputs."""
    clean = load_clean_graph(clean_dir, edge_type=edge_type)
    attack = load_attack_graph(attack_dir) if attack_dir else None
    defense = load_defense_graph(defense_dir, clean_graph=clean) if defense_dir else None
    inputs = {"clean": clean, "attack": attack, "defense": defense}
    validate_shield_inputs(inputs)
    return inputs


def validate_shield_inputs(inputs):
    """Validate SHIELD input dictionaries."""
    clean = inputs.get("clean")
    if clean is None:
        raise ValueError("clean graph input is mandatory.")
    for name in ["clean", "attack", "defense"]:
        graph = inputs.get(name)
        if graph is None:
            continue
        X = graph.get("X")
        y = graph.get("y")
        edge_index = graph.get("edge_index")
        if X is not None and y is not None and X.shape[0] != y.shape[0]:
            raise ValueError(f"{name} X/y node count mismatch.")
        if edge_index is None or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"{name} edge_index must have shape [2, num_edges].")
        num_nodes = X.shape[0] if X is not None else clean["X"].shape[0]
        if edge_index.shape[1] and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
            raise ValueError(f"{name} edge_index contains invalid node IDs.")
    return True


def summarize_shield_inputs(inputs):
    """Summarize available clean, attack, and defense inputs."""
    summary = {}
    for name, graph in inputs.items():
        if graph is None:
            summary[f"{name}_available"] = False
            continue
        summary[f"{name}_available"] = True
        if graph.get("X") is not None:
            summary[f"{name}_num_nodes"] = int(graph["X"].shape[0])
            summary[f"{name}_num_features"] = int(graph["X"].shape[1])
        summary[f"{name}_num_edges"] = int(graph["edge_index"].shape[1])
        summary[f"{name}_has_edge_weight"] = graph.get("edge_weight") is not None
    return summary
