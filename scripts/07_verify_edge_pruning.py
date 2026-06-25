"""Verify robust edge pruning defense without training."""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.defense.edge_pruning import run_robust_edge_pruning
from src.defense.pruning_utils import validate_edge_index
from src.training_utils import load_graph_artifacts


REPORT_PATH = PROJECT_ROOT / "results" / "tables" / "edge_pruning_readiness_report.json"
ATTACK_ROOT = PROJECT_ROOT / "data" / "processed" / "attack_artifacts"


EXPECTED_SUMMARY_KEYS = {
    "original_num_edges",
    "defended_num_edges",
    "pruned_edges",
    "downweighted_edges",
    "kept_edges",
    "prune_ratio",
    "mode",
    "prune_threshold",
    "downweight_threshold",
    "min_weight",
}


def save_report(report):
    """Save readiness report JSON."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def latest_attack_dir():
    """Return the newest attack artifact directory if one exists."""
    if not ATTACK_ROOT.is_dir():
        return None
    dirs = [path for path in ATTACK_ROOT.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def validate_pruning_result(X, edge_index, defended_edge_index, defended_weight, scores, summary):
    """Validate shape, range, and summary consistency."""
    validate_edge_index(defended_edge_index, X.shape[0])
    if defended_weight.shape[0] != defended_edge_index.shape[1]:
        raise ValueError("edge weights do not match defended edge count.")
    if scores.shape[0] != edge_index.shape[1]:
        raise ValueError("edge scores do not match original edge count.")
    missing_keys = EXPECTED_SUMMARY_KEYS - set(summary)
    if missing_keys:
        raise ValueError(f"pruning summary missing keys: {sorted(missing_keys)}")


def main():
    """Run clean and optional attack pruning dry-runs."""
    artifacts = load_graph_artifacts()
    X = artifacts["X"]
    y = artifacts["y"]
    edge_index = artifacts["edge_index_undirected"]
    timesteps = artifacts["timesteps"]
    clean_edges_copy = edge_index.copy()
    report = {
        "clean_num_nodes": int(X.shape[0]),
        "clean_num_edges": int(edge_index.shape[1]),
        "mode_tested": "hybrid",
        "clean_pruning_status": "not_run",
        "attack_artifact_found": False,
        "attack_pruning_status": "not_run",
        "defended_num_edges": None,
        "pruned_edges": None,
        "downweighted_edges": None,
        "readiness_status": "not_ready",
    }

    try:
        defended_edge_index, defended_weight, scores, _, summary = run_robust_edge_pruning(
            X, y, edge_index, timesteps=timesteps, mode="hybrid"
        )
        validate_pruning_result(X, edge_index, defended_edge_index, defended_weight, scores, summary)
        if not np.array_equal(edge_index, clean_edges_copy):
            raise ValueError("clean edge_index was modified in-place.")
        report["clean_pruning_status"] = "passed"
        report["defended_num_edges"] = int(defended_edge_index.shape[1])
        report["pruned_edges"] = int(summary["pruned_edges"])
        report["downweighted_edges"] = int(summary["downweighted_edges"])

        attack_dir = latest_attack_dir()
        if attack_dir is not None:
            report["attack_artifact_found"] = True
            attack_X = np.load(attack_dir / "attacked_X.npy")
            attack_y = np.load(attack_dir / "attacked_y.npy")
            attack_edge_index = np.load(attack_dir / "attacked_edge_index.npy")
            attack_defended, attack_weight, attack_scores, _, attack_summary = run_robust_edge_pruning(
                attack_X, attack_y, attack_edge_index, timesteps=None, mode="hybrid"
            )
            validate_pruning_result(
                attack_X, attack_edge_index, attack_defended, attack_weight, attack_scores, attack_summary
            )
            report["attack_pruning_status"] = "passed"
        else:
            report["attack_pruning_status"] = "skipped_no_attack_artifacts"

        report["readiness_status"] = "ready"
    except Exception as exc:
        report["readiness_status"] = f"not_ready: {exc}"

    save_report(report)
    print(json.dumps(report, indent=2))
    print(f"Saved edge pruning readiness report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
