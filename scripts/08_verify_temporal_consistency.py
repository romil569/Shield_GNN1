"""Verify temporal consistency losses without training."""

import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.defense.temporal_consistency import (
    TemporalConsistencyLoss,
    edge_smoothness_loss,
    prediction_consistency_loss,
    temporal_consistency_regularizer,
    temporal_smoothness_loss,
)
from src.training_utils import load_graph_artifacts


REPORT_PATH = PROJECT_ROOT / "results" / "tables" / "temporal_consistency_readiness_report.json"
DEFENSE_DIR = PROJECT_ROOT / "data" / "processed" / "defense_artifacts" / "clean_hybrid_pruned"


def _status_from_loss(loss):
    """Return passed status if a loss is scalar and finite."""
    if loss.ndim != 0:
        return f"failed: loss is not scalar, shape={tuple(loss.shape)}"
    if not torch.isfinite(loss):
        return "failed: loss is not finite"
    return f"passed: {float(loss.item()):.6f}"


def _dummy_logits(num_nodes, seed=42, device="cpu"):
    """Create reproducible dummy logits for verification."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    clean = torch.randn(num_nodes, 2, generator=generator, dtype=torch.float32).to(device)
    attacked = clean + 0.05 * torch.randn(num_nodes, 2, generator=generator, dtype=torch.float32).to(device)
    defended = clean + 0.02 * torch.randn(num_nodes, 2, generator=generator, dtype=torch.float32).to(device)
    return clean, attacked, defended


def main():
    """Run temporal consistency readiness checks."""
    artifacts = load_graph_artifacts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = int(artifacts["X"].shape[0])
    edge_index = torch.tensor(artifacts["edge_index_undirected"], dtype=torch.long, device=device)
    timesteps = torch.tensor(artifacts["timesteps"], dtype=torch.long, device=device)
    mask = torch.tensor(artifacts["labeled_mask"], dtype=torch.bool, device=device)
    clean_logits, attacked_logits, defended_logits = _dummy_logits(num_nodes, device=device)

    edge_weight = None
    defense_edge_weight_found = False
    if (DEFENSE_DIR / "defended_edge_weight.npy").is_file() and (
        DEFENSE_DIR / "defended_edge_index.npy"
    ).is_file():
        defense_edge_weight_found = True
        edge_weight = torch.tensor(
            __import__("numpy").load(DEFENSE_DIR / "defended_edge_weight.npy"),
            dtype=torch.float32,
            device=device,
        )
        edge_index_for_weight = torch.tensor(
            __import__("numpy").load(DEFENSE_DIR / "defended_edge_index.npy"),
            dtype=torch.long,
            device=device,
        )
    else:
        edge_index_for_weight = edge_index

    report = {
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "num_timesteps": int(artifacts["graph_summary"]["num_timesteps"]),
        "prediction_consistency_mse_status": "not_run",
        "prediction_consistency_kl_status": "not_run",
        "temporal_smoothness_status": "not_run",
        "edge_smoothness_status": "not_run",
        "combined_regularizer_status": "not_run",
        "temporal_consistency_class_status": "not_run",
        "defense_edge_weight_found": defense_edge_weight_found,
        "readiness_status": "not_ready",
    }

    try:
        empty_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        empty_loss = prediction_consistency_loss(clean_logits, attacked_logits, mask=empty_mask)
        if not torch.isfinite(empty_loss) or float(empty_loss.item()) != 0.0:
            raise ValueError("empty mask did not return zero finite loss")

        report["prediction_consistency_mse_status"] = _status_from_loss(
            prediction_consistency_loss(clean_logits, attacked_logits, mask=mask, loss_type="mse")
        )
        report["prediction_consistency_kl_status"] = _status_from_loss(
            prediction_consistency_loss(
                clean_logits, attacked_logits, mask=mask, loss_type="symmetric_kl"
            )
        )
        report["temporal_smoothness_status"] = _status_from_loss(
            temporal_smoothness_loss(clean_logits, timesteps, mask=mask, sample_pairs=5000)
        )
        report["edge_smoothness_status"] = _status_from_loss(
            edge_smoothness_loss(
                clean_logits,
                edge_index_for_weight,
                edge_weight=edge_weight,
                mask=mask,
                sample_edges=20000,
            )
        )
        total, components = temporal_consistency_regularizer(
            clean_logits,
            defended_logits=defended_logits,
            attacked_logits=attacked_logits,
            timesteps=timesteps,
            edge_index=edge_index,
            mask=mask,
        )
        report["combined_regularizer_status"] = _status_from_loss(total)
        module = TemporalConsistencyLoss()
        module_total, module_components = module(
            clean_logits,
            defended_logits=defended_logits,
            attacked_logits=attacked_logits,
            timesteps=timesteps,
            edge_index=edge_index,
            mask=mask,
        )
        report["temporal_consistency_class_status"] = _status_from_loss(module_total)
        expected_keys = {
            "clean_defended_loss",
            "clean_attacked_loss",
            "temporal_smoothness_loss",
            "edge_smoothness_loss",
            "total_consistency_loss",
        }
        if set(components) != expected_keys or set(module_components) != expected_keys:
            raise ValueError("component loss dictionary keys are incorrect")
        report["readiness_status"] = "ready"
    except Exception as exc:
        report["readiness_status"] = f"not_ready: {exc}"

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved temporal consistency readiness report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
