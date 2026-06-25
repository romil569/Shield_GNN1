"""Demonstrate temporal consistency losses without training."""

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.defense.temporal_consistency import temporal_consistency_regularizer
from src.training_utils import get_device, load_graph_artifacts


SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "temporal_consistency_demo_summary.json"


def parse_args():
    """Parse demo arguments."""
    parser = argparse.ArgumentParser(description="Run temporal consistency demo losses.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--loss-type", default="mse", choices=["mse", "kl", "symmetric_kl", "js"])
    parser.add_argument("--sample-pairs", type=int, default=5000)
    parser.add_argument("--sample-edges", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    """Run a no-training temporal consistency demo."""
    args = parse_args()
    device = get_device(args.device)
    artifacts = load_graph_artifacts()
    num_nodes = artifacts["X"].shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    clean_logits = torch.randn(num_nodes, 2, generator=generator).to(device)
    attacked_logits = clean_logits + 0.05 * torch.randn(num_nodes, 2, generator=generator).to(device)
    defended_logits = clean_logits + 0.02 * torch.randn(num_nodes, 2, generator=generator).to(device)
    timesteps = torch.tensor(artifacts["timesteps"], dtype=torch.long, device=device)
    edge_index = torch.tensor(artifacts["edge_index_undirected"], dtype=torch.long, device=device)
    mask = torch.tensor(artifacts["labeled_mask"], dtype=torch.bool, device=device)

    total, components = temporal_consistency_regularizer(
        clean_logits,
        defended_logits=defended_logits,
        attacked_logits=attacked_logits,
        timesteps=timesteps,
        edge_index=edge_index,
        mask=mask,
        loss_type=args.loss_type,
    )
    summary = {
        "dry_run": bool(args.dry_run),
        "device": str(device),
        "loss_type": args.loss_type,
        "num_nodes": int(num_nodes),
        "num_edges": int(edge_index.shape[1]),
        "component_losses": {key: float(value.item()) for key, value in components.items()},
        "note": "No backward pass, optimizer, epochs, or training were run.",
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Temporal consistency demo completed without training.")
    print(json.dumps(summary, indent=2))
    print(f"Saved demo summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
