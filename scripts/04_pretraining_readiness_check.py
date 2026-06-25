"""Pre-training readiness checks for baseline GNN models without training."""

import json
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_PATH = PROJECT_ROOT / "results" / "tables" / "pretraining_readiness_report.json"


def _empty_report():
    """Create a report with stable keys."""
    return {
        "python_version": platform.python_version(),
        "torch_installed": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "cuda_architecture": None,
        "cuda_architecture_supported_by_torch": None,
        "torch_geometric_installed": False,
        "torch_geometric_version": None,
        "graph_artifacts_found": False,
        "X_shape": None,
        "y_shape": None,
        "edge_index_shape": None,
        "train_mask_count": None,
        "val_mask_count": None,
        "test_mask_count": None,
        "gcn_dry_run_status": "not_run",
        "gat_dry_run_status": "not_run",
        "graphsage_dry_run_status": "not_run",
        "sm_120_supported_status": "unknown",
        "gcn_cuda_dry_run_status": "not_run",
        "gat_cuda_dry_run_status": "not_run",
        "graphsage_cuda_dry_run_status": "not_run",
        "final_recommendation": "unknown",
        "readiness_status": "not_ready",
    }


def _save_report(report):
    """Save the readiness report to results/tables."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _small_graph_tensors(artifacts, torch, device, max_nodes=2048):
    """Create a small graph slice for safe forward/loss checks."""
    import numpy as np

    X = artifacts["X"]
    y = artifacts["y"]
    edge_index = artifacts["edge_index_undirected"]
    train_mask = artifacts["train_mask"]

    selected_nodes = np.arange(min(max_nodes, X.shape[0]))
    edge_mask = np.isin(edge_index[0], selected_nodes) & np.isin(edge_index[1], selected_nodes)
    small_edges = edge_index[:, edge_mask]

    if small_edges.shape[1] == 0:
        raise ValueError("Small graph slice has no edges; increase max_nodes.")

    local_index = {int(node_idx): index for index, node_idx in enumerate(selected_nodes)}
    relabeled_edges = np.vectorize(local_index.get)(small_edges).astype("int64")
    small_train = train_mask[selected_nodes] & (y[selected_nodes] != -1)

    if int(small_train.sum()) == 0:
        raise ValueError("Small graph slice has no labeled training nodes; increase max_nodes.")

    return {
        "x": torch.tensor(X[selected_nodes], dtype=torch.float32, device=device),
        "y": torch.tensor(y[selected_nodes], dtype=torch.long, device=device),
        "edge_index": torch.tensor(relabeled_edges, dtype=torch.long, device=device),
        "train_mask": torch.tensor(small_train, dtype=torch.bool, device=device),
    }


def _compute_class_weights(torch, y, train_mask):
    """Compute simple class weights for a small dry-run loss check."""
    train_labels = y[train_mask & (y != -1)]
    counts = torch.bincount(train_labels, minlength=2).float()
    if torch.any(counts == 0):
        return torch.ones(2, dtype=torch.float32)
    return counts.sum() / (2.0 * counts)


def _model_forward_status(model_name, create_model, torch, tensors):
    """Create one model, run one forward pass, and compute loss without training."""
    hidden_channels = 32 if model_name == "gat" else 64
    model = create_model(
        model_name,
        in_channels=tensors["x"].shape[1],
        hidden_channels=hidden_channels,
        out_channels=2,
        num_layers=2,
        dropout=0.5,
        heads=4,
    )
    model = model.to(tensors["x"].device)
    model.eval()
    with torch.no_grad():
        logits = model(tensors["x"], tensors["edge_index"])

    expected_shape = (tensors["x"].shape[0], 2)
    if tuple(logits.shape) != expected_shape:
        return f"failed: logits shape {tuple(logits.shape)} != {expected_shape}"

    weights = _compute_class_weights(torch, tensors["y"], tensors["train_mask"])
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    loss = criterion(logits[tensors["train_mask"]], tensors["y"][tensors["train_mask"]])
    if not torch.isfinite(loss):
        return "failed: dry-run loss is not finite"

    return f"passed: logits_shape={tuple(logits.shape)}, loss={float(loss.item()):.6f}"


def main():
    """Run readiness checks and save a JSON report."""
    report = _empty_report()

    try:
        import torch

        report["torch_installed"] = True
        report["torch_version"] = torch.__version__
        report["torch_cuda_version"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu_name"] = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            cuda_architecture = f"sm_{capability[0]}{capability[1]}"
            report["cuda_architecture"] = cuda_architecture
            report["cuda_architecture_supported_by_torch"] = (
                cuda_architecture in set(torch.cuda.get_arch_list())
            )
            report["sm_120_supported_status"] = (
                "supported"
                if cuda_architecture == "sm_120" and report["cuda_architecture_supported_by_torch"]
                else "not_supported"
            )
        else:
            report["gpu_name"] = None
            report["cuda_architecture_supported_by_torch"] = False
            report["sm_120_supported_status"] = "cuda_not_available"
    except ImportError as exc:
        report["readiness_status"] = f"not_ready: torch import failed: {exc}"
        _save_report(report)
        print(json.dumps(report, indent=2))
        return

    try:
        import torch_geometric

        report["torch_geometric_installed"] = True
        report["torch_geometric_version"] = torch_geometric.__version__
    except ImportError as exc:
        report["readiness_status"] = f"not_ready: torch_geometric import failed: {exc}"
        _save_report(report)
        print(json.dumps(report, indent=2))
        return

    from src.models.model_factory import create_model
    from src.training_utils import load_graph_artifacts

    try:
        artifacts = load_graph_artifacts()
        report["graph_artifacts_found"] = True
        report["X_shape"] = list(artifacts["X"].shape)
        report["y_shape"] = list(artifacts["y"].shape)
        report["edge_index_shape"] = list(artifacts["edge_index_undirected"].shape)
        report["train_mask_count"] = int(artifacts["train_mask"].sum())
        report["val_mask_count"] = int(artifacts["val_mask"].sum())
        report["test_mask_count"] = int(artifacts["test_mask"].sum())
    except Exception as exc:
        report["readiness_status"] = f"not_ready: graph artifact check failed: {exc}"
        _save_report(report)
        print(json.dumps(report, indent=2))
        return

    try:
        tensors = _small_graph_tensors(artifacts, torch, torch.device("cpu"))
        for model_name, report_key in [
            ("gcn", "gcn_dry_run_status"),
            ("gat", "gat_dry_run_status"),
            ("graphsage", "graphsage_dry_run_status"),
        ]:
            report[report_key] = _model_forward_status(model_name, create_model, torch, tensors)

        if report["cuda_available"] and report["cuda_architecture_supported_by_torch"]:
            cuda_tensors = _small_graph_tensors(artifacts, torch, torch.device("cuda"))
            for model_name, report_key in [
                ("gcn", "gcn_cuda_dry_run_status"),
                ("gat", "gat_cuda_dry_run_status"),
                ("graphsage", "graphsage_cuda_dry_run_status"),
            ]:
                report[report_key] = _model_forward_status(
                    model_name, create_model, torch, cuda_tensors
                )
        else:
            report["gcn_cuda_dry_run_status"] = "skipped: cuda unavailable or unsupported"
            report["gat_cuda_dry_run_status"] = "skipped: cuda unavailable or unsupported"
            report["graphsage_cuda_dry_run_status"] = "skipped: cuda unavailable or unsupported"
    except Exception as exc:
        report["readiness_status"] = f"not_ready: dry-run forward check failed: {exc}"
        report["final_recommendation"] = "cpu_fallback"
        _save_report(report)
        print(json.dumps(report, indent=2))
        return

    statuses = [
        report["gcn_dry_run_status"],
        report["gat_dry_run_status"],
        report["graphsage_dry_run_status"],
    ]
    cuda_statuses = [
        report["gcn_cuda_dry_run_status"],
        report["gat_cuda_dry_run_status"],
        report["graphsage_cuda_dry_run_status"],
    ]
    if all(status.startswith("passed") for status in statuses):
        if (
            report["cuda_available"]
            and report["cuda_architecture_supported_by_torch"]
            and all(status.startswith("passed") for status in cuda_statuses)
        ):
            report["readiness_status"] = "ready"
            report["final_recommendation"] = "gpu_ready"
        elif report["cuda_available"] and not report["cuda_architecture_supported_by_torch"]:
            report["readiness_status"] = "ready_cpu_fallback_cuda_arch_unsupported"
            report["final_recommendation"] = "cpu_fallback"
        else:
            report["readiness_status"] = "ready"
            report["final_recommendation"] = "cpu_fallback"
    else:
        report["readiness_status"] = "not_ready"
        report["final_recommendation"] = "cpu_fallback"
    _save_report(report)
    print(json.dumps(report, indent=2))
    print(f"Saved readiness report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
