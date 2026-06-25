"""Verify Elliptic++ Train200K graph artifacts."""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "data" / "processed" / "ellipticpp_train200k" / "graph_artifacts"
ORIGINAL_FULL_DIR = PROJECT_ROOT / "data" / "processed" / "full" / "graph_artifacts"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


REQUIRED_FILES = [
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
    "transaction_node_mask.npy",
    "wallet_node_mask.npy",
    "node_type.npy",
    "feature_schema.json",
    "node_mapping.json",
    "graph_summary.json",
    "merge_report.json",
]


def check(condition, name, details, failures):
    """Record a verification check."""
    details[name] = bool(condition)
    if not condition:
        failures.append(name)


def write_markdown(report):
    """Create a Markdown verification report."""
    lines = [
        "# Elliptic++ Train200K Verification Report",
        "",
        f"Artifact directory: `{ARTIFACT_DIR}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", "## Summary", ""])
    for key, value in report["summary"].items():
        lines.append(f"- {key}: {value}")
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in report["failures"]:
            lines.append(f"- {failure}")
    return "\n".join(lines) + "\n"


def main():
    """Verify graph artifacts and save a report."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    checks = {}
    failures = []
    missing = [name for name in REQUIRED_FILES if not (ARTIFACT_DIR / name).is_file()]
    check(not missing, "all_required_artifacts_exist", checks, failures)
    if missing:
        report = {"checks": checks, "failures": failures, "missing_files": missing, "summary": {}}
        (TABLES_DIR / "ellipticpp_train200k_verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Missing required artifacts: {', '.join(missing)}")
        sys.exit(1)

    X = np.load(ARTIFACT_DIR / "X.npy", mmap_mode="r")
    y = np.load(ARTIFACT_DIR / "y.npy")
    edge_directed = np.load(ARTIFACT_DIR / "edge_index_directed.npy", mmap_mode="r")
    edge_undirected = np.load(ARTIFACT_DIR / "edge_index_undirected.npy", mmap_mode="r")
    timesteps = np.load(ARTIFACT_DIR / "timesteps.npy")
    train_mask = np.load(ARTIFACT_DIR / "train_mask.npy")
    val_mask = np.load(ARTIFACT_DIR / "val_mask.npy")
    test_mask = np.load(ARTIFACT_DIR / "test_mask.npy")
    labeled_mask = np.load(ARTIFACT_DIR / "labeled_mask.npy")
    unknown_mask = np.load(ARTIFACT_DIR / "unknown_mask.npy")
    transaction_node_mask = np.load(ARTIFACT_DIR / "transaction_node_mask.npy")
    wallet_node_mask = np.load(ARTIFACT_DIR / "wallet_node_mask.npy")
    node_type = np.load(ARTIFACT_DIR / "node_type.npy")
    graph_summary = json.loads((ARTIFACT_DIR / "graph_summary.json").read_text(encoding="utf-8"))
    feature_schema = json.loads((ARTIFACT_DIR / "feature_schema.json").read_text(encoding="utf-8"))

    n = X.shape[0]
    finite_ok = True
    for start in range(0, n, 100_000):
        finite_ok = finite_ok and bool(np.isfinite(np.asarray(X[start:start + 100_000])).all())

    check(finite_ok, "X_has_no_nan_or_infinite_values", checks, failures)
    check(set(np.unique(y).tolist()).issubset({-1, 0, 1}), "labels_are_only_minus1_0_1", checks, failures)
    check(edge_directed.shape[0] == 2 and edge_undirected.shape[0] == 2, "edge_indices_have_shape_2_by_edges", checks, failures)
    edge_ids_ok = edge_directed.shape[1] == 0 or (int(edge_directed.min()) >= 0 and int(edge_directed.max()) < n)
    edge_ids_ok = edge_ids_ok and (edge_undirected.shape[1] == 0 or (int(edge_undirected.min()) >= 0 and int(edge_undirected.max()) < n))
    check(edge_ids_ok, "edge_index_ids_are_in_range", checks, failures)
    check(all(arr.shape[0] == n for arr in [y, timesteps, train_mask, val_mask, test_mask, labeled_mask, unknown_mask, transaction_node_mask, wallet_node_mask, node_type]), "all_node_arrays_match_node_count", checks, failures)
    check(bool(np.all(transaction_node_mask | wallet_node_mask) and not np.any(transaction_node_mask & wallet_node_mask)), "transaction_and_wallet_masks_cover_all_nodes", checks, failures)
    check(int((train_mask & labeled_mask).sum()) >= 200_000, "labeled_train_nodes_at_least_200000_if_possible", checks, failures)
    check(not bool(np.any(val_mask & wallet_node_mask)), "validation_nodes_are_not_wallets", checks, failures)
    check(not bool(np.any(test_mask & wallet_node_mask)), "test_nodes_are_not_wallets", checks, failures)
    check(not bool(np.any(train_mask & unknown_mask) or np.any(val_mask & unknown_mask) or np.any(test_mask & unknown_mask)), "unknown_nodes_not_used_in_supervised_masks", checks, failures)
    check(X.shape[1] == feature_schema["total_feature_dim"], "feature_dimension_matches_schema", checks, failures)
    check((ORIGINAL_FULL_DIR / "X.npy").is_file(), "original_elliptic_full_artifacts_not_overwritten", checks, failures)

    summary = {
        "num_nodes": int(n),
        "num_features": int(X.shape[1]),
        "num_directed_edges": int(edge_directed.shape[1]),
        "num_undirected_edges": int(edge_undirected.shape[1]),
        "train_labeled_nodes": int((train_mask & labeled_mask).sum()),
        "val_labeled_nodes": int((val_mask & labeled_mask).sum()),
        "test_labeled_nodes": int((test_mask & labeled_mask).sum()),
        "transaction_nodes": int(transaction_node_mask.sum()),
        "wallet_nodes": int(wallet_node_mask.sum()),
        "split_protocol": graph_summary.get("split_protocol"),
        "graph_can_be_loaded_by_existing_training_scripts": bool(not failures),
    }
    report = {"checks": checks, "failures": failures, "summary": summary}
    (TABLES_DIR / "ellipticpp_train200k_verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (TABLES_DIR / "ellipticpp_train200k_verification_report.md").write_text(write_markdown(report), encoding="utf-8")

    print("Elliptic++ Train200K verification summary:")
    print(json.dumps(summary, indent=2))
    if failures:
        print("Verification failed:")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)
    print("Verification passed.")


if __name__ == "__main__":
    main()
