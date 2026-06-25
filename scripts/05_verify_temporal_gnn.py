"""Verify temporal GNN readiness without training."""

import json
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_PATH = PROJECT_ROOT / "results" / "tables" / "temporal_gnn_readiness_report.json"


def save_report(report):
    """Save the temporal readiness report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def dry_run_status(model_name, create_model, torch, tensors):
    """Run one safe forward/loss validation for a temporal model."""
    hidden_channels = 64
    model = create_model(
        model_name,
        in_channels=tensors["x"].shape[1],
        hidden_channels=hidden_channels,
        out_channels=2,
        num_layers=2,
        dropout=0.5,
        max_timesteps=int(tensors["timesteps"].max().item()) + 1,
    ).to(tensors["x"].device)
    model.eval()
    with torch.no_grad():
        if model_name == "temporal_gcn_gru":
            logits = model.forward_full_sequence(
                tensors["x"],
                tensors["edge_index"],
                tensors["timesteps"],
                snapshot_mode="cumulative",
            )
        else:
            logits = model(tensors["x"], tensors["edge_index"], tensors["timesteps"])
        selected = tensors["train_mask"] & (tensors["y"] != -1)
        loss = torch.nn.functional.cross_entropy(logits[selected], tensors["y"][selected])

    if tuple(logits.shape) != (tensors["x"].shape[0], 2):
        return f"failed: logits shape {tuple(logits.shape)}"
    if not torch.isfinite(loss):
        return "failed: loss is not finite"
    return f"passed: logits_shape={tuple(logits.shape)}, loss={float(loss.item()):.6f}"


def main():
    """Run temporal GNN verification without optimizer steps or epochs."""
    report = {
        "python_version": platform.python_version(),
        "temporal_artifacts_found": False,
        "num_nodes": None,
        "num_features": None,
        "num_edges": None,
        "num_timesteps": None,
        "timestep_min": None,
        "timestep_max": None,
        "temporal_gcn_gru_dry_run_status": "not_run",
        "time_aware_gcn_dry_run_status": "not_run",
        "device_used": None,
        "cuda_available": False,
        "gpu_name": None,
        "readiness_status": "not_ready",
    }

    try:
        import torch
        import torch_geometric

        print(f"torch: {torch.__version__}")
        print(f"torch_geometric: {torch_geometric.__version__}")
    except ImportError as exc:
        report["readiness_status"] = f"not_ready: dependency import failed: {exc}"
        save_report(report)
        print(json.dumps(report, indent=2))
        return

    from src.models.model_factory import create_model
    from src.temporal_data import (
        load_temporal_graph_artifacts,
        make_small_temporal_subgraph,
        summarize_temporal_data,
        temporal_numpy_to_torch,
    )
    from src.training_utils import get_device

    device = get_device("auto")
    report["device_used"] = str(device)
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    try:
        artifacts = load_temporal_graph_artifacts()
        report["temporal_artifacts_found"] = True
        summary = summarize_temporal_data(
            artifacts["X"],
            artifacts["y"],
            artifacts["edge_index_undirected"],
            artifacts["timesteps"],
            artifacts["temporal_snapshots"],
        )
        report.update(
            {
                "num_nodes": summary["num_nodes"],
                "num_features": summary["num_features"],
                "num_edges": summary["num_edges"],
                "num_timesteps": summary["num_timesteps"],
                "timestep_min": summary["timestep_min"],
                "timestep_max": summary["timestep_max"],
            }
        )
        small = make_small_temporal_subgraph(artifacts)
        tensors = temporal_numpy_to_torch(small, device)
        report["temporal_gcn_gru_dry_run_status"] = dry_run_status(
            "temporal_gcn_gru", create_model, torch, tensors
        )
        report["time_aware_gcn_dry_run_status"] = dry_run_status(
            "time_aware_gcn", create_model, torch, tensors
        )
    except Exception as exc:
        report["readiness_status"] = f"not_ready: {exc}"
        save_report(report)
        print(json.dumps(report, indent=2))
        return

    if (
        report["temporal_gcn_gru_dry_run_status"].startswith("passed")
        and report["time_aware_gcn_dry_run_status"].startswith("passed")
    ):
        report["readiness_status"] = "ready"

    save_report(report)
    print(json.dumps(report, indent=2))
    print(f"Saved temporal readiness report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
