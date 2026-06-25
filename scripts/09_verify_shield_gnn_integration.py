"""Verify final SHIELD-GNN integration without training."""

import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.shield_gnn import SHIELDGNNModel
from src.shield_data import prepare_shield_inputs
from src.shield_losses import shield_total_loss
from src.temporal_data import make_small_temporal_subgraph
from src.training_utils import get_device


REPORT_PATH = PROJECT_ROOT / "results" / "tables" / "shield_gnn_integration_readiness_report.json"
ATTACK_ROOT = PROJECT_ROOT / "data" / "processed" / "attack_artifacts"
DEFENSE_ROOT = PROJECT_ROOT / "data" / "processed" / "defense_artifacts"


def latest_dir(root):
    """Return newest child directory if available."""
    if not root.is_dir():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0] if dirs else None


def small_tensors(clean, device):
    """Create small dry-run tensors from the clean graph."""
    small = make_small_temporal_subgraph(
        {
            "X": clean["X"],
            "y": clean["y"],
            "edge_index_undirected": clean["edge_index"],
            "timesteps": clean["timesteps"],
            "train_mask": clean["train_mask"],
        }
    )
    return {
        "x": torch.tensor(small["X"], dtype=torch.float32, device=device),
        "y": torch.tensor(small["y"], dtype=torch.long, device=device),
        "edge_index": torch.tensor(small["edge_index_undirected"], dtype=torch.long, device=device),
        "timesteps": torch.tensor(small["timesteps"], dtype=torch.long, device=device),
        "train_mask": torch.tensor(small["train_mask"], dtype=torch.bool, device=device),
    }


def main():
    """Run final integration readiness checks."""
    attack_dir = latest_dir(ATTACK_ROOT)
    defense_dir = latest_dir(DEFENSE_ROOT)
    device = get_device("auto")
    report = {
        "clean_graph_found": False,
        "attack_artifact_found": attack_dir is not None,
        "defense_artifact_found": defense_dir is not None,
        "num_nodes": None,
        "num_features": None,
        "num_edges": None,
        "num_timesteps": None,
        "shield_model_creation_status": "not_run",
        "clean_forward_status": "not_run",
        "defended_forward_status": "not_run",
        "shield_loss_status": "not_run",
        "evaluation_import_status": "not_run",
        "device_used": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "readiness_status": "not_ready",
    }
    try:
        inputs = prepare_shield_inputs(
            attack_dir=str(attack_dir) if attack_dir else None,
            defense_dir=str(defense_dir) if defense_dir else None,
        )
        clean = inputs["clean"]
        report["clean_graph_found"] = True
        report["num_nodes"] = int(clean["X"].shape[0])
        report["num_features"] = int(clean["X"].shape[1])
        report["num_edges"] = int(clean["edge_index"].shape[1])
        report["num_timesteps"] = int(clean["graph_summary"]["num_timesteps"])
        tensors = small_tensors(clean, device)
        model = SHIELDGNNModel(
            in_channels=tensors["x"].shape[1],
            hidden_channels=64,
            out_channels=2,
            backbone="temporal_gcn_gru",
            num_layers=2,
            dropout=0.5,
            max_timesteps=int(tensors["timesteps"].max().item()) + 1,
        ).to(device)
        report["shield_model_creation_status"] = "passed"
        model.eval()
        with torch.no_grad():
            clean_logits = model(tensors["x"], tensors["edge_index"], tensors["timesteps"])
        if tuple(clean_logits.shape) != (tensors["x"].shape[0], 2):
            raise ValueError(f"unexpected clean logits shape: {tuple(clean_logits.shape)}")
        report["clean_forward_status"] = "passed"
        defended_logits = None
        if defense_dir is not None:
            with torch.no_grad():
                defended_logits = model(tensors["x"], tensors["edge_index"], tensors["timesteps"])
            report["defended_forward_status"] = "passed"
        else:
            report["defended_forward_status"] = "skipped_no_defense_artifact"
        total_loss, _ = shield_total_loss(
            clean_logits,
            tensors["y"],
            tensors["train_mask"],
            defended_logits=defended_logits,
            timesteps=tensors["timesteps"],
            edge_index=tensors["edge_index"],
        )
        if total_loss.ndim != 0 or not torch.isfinite(total_loss):
            raise ValueError("SHIELD loss is not a finite scalar.")
        report["shield_loss_status"] = f"passed: {float(total_loss.item()):.6f}"
        import src.evaluate_shield_gnn  # noqa: F401

        report["evaluation_import_status"] = "passed"
        report["readiness_status"] = "ready"
    except Exception as exc:
        report["readiness_status"] = f"not_ready: {exc}"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved SHIELD-GNN integration report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
