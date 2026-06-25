"""Verify adversarial attack simulation modules without training."""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks.attack_utils import remove_duplicate_edges, validate_graph_arrays
from src.attacks.combined_attack import run_combined_attack
from src.attacks.fake_edge_injection import inject_fake_edges
from src.attacks.fake_node_injection import inject_fake_nodes
from src.attacks.feature_perturbation import perturb_node_features
from src.training_utils import load_graph_artifacts


REPORT_PATH = PROJECT_ROOT / "results" / "tables" / "attack_simulation_readiness_report.json"


def save_report(report):
    """Save attack readiness report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def check_edge_valid(edge_index, num_nodes):
    """Return whether edge indices are within node range and deduplicated."""
    validate_graph_arrays(np.zeros((num_nodes, 1), dtype=np.float32), np.full(num_nodes, -1), edge_index)
    return remove_duplicate_edges(edge_index).shape[1] == edge_index.shape[1]


def main():
    """Run attack dry-runs and validate graph consistency."""
    artifacts = load_graph_artifacts()
    X = artifacts["X"]
    y = artifacts["y"]
    edge_index = artifacts["edge_index_undirected"]
    clean_X = X.copy()
    clean_y = y.copy()
    clean_edges = edge_index.copy()
    report = {
        "clean_num_nodes": int(X.shape[0]),
        "clean_num_edges": int(edge_index.shape[1]),
        "clean_num_features": int(X.shape[1]),
        "fake_node_attack_status": "not_run",
        "fake_edge_attack_status": "not_run",
        "feature_perturbation_status": "not_run",
        "combined_attack_status": "not_run",
        "readiness_status": "not_ready",
    }

    try:
        fn_X, fn_y, fn_edges, fake_nodes, _ = inject_fake_nodes(
            X, y, edge_index, injection_rate=0.001, seed=11
        )
        assert fn_X.shape[0] == fn_y.shape[0]
        assert fake_nodes.size > 0
        assert check_edge_valid(fn_edges, fn_X.shape[0])
        report["fake_node_attack_status"] = "passed"

        fe_edges, fake_edges, _ = inject_fake_edges(
            X, y, edge_index, injection_rate=0.001, seed=12
        )
        assert fake_edges.shape[0] == 2
        assert check_edge_valid(fe_edges, X.shape[0])
        report["fake_edge_attack_status"] = "passed"

        fp_X, perturbed_nodes, _ = perturb_node_features(
            X, y, perturbation_rate=0.001, seed=13
        )
        assert fp_X.shape == X.shape
        assert perturbed_nodes.size > 0
        report["feature_perturbation_status"] = "passed"

        combo_X, combo_y, combo_edges, _ = run_combined_attack(
            X,
            y,
            edge_index,
            timesteps=artifacts["timesteps"],
            fake_node_rate=0.001,
            fake_edge_rate=0.001,
            feature_perturb_rate=0.001,
            seed=14,
        )
        assert combo_X.shape[0] == combo_y.shape[0]
        assert check_edge_valid(combo_edges, combo_X.shape[0])
        report["combined_attack_status"] = "passed"

        assert np.array_equal(X, clean_X)
        assert np.array_equal(y, clean_y)
        assert np.array_equal(edge_index, clean_edges)
        report["readiness_status"] = "ready"
    except Exception as exc:
        report["readiness_status"] = f"not_ready: {exc}"

    save_report(report)
    print(json.dumps(report, indent=2))
    print(f"Saved attack readiness report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
