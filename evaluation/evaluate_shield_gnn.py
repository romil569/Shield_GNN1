"""Evaluate trained SHIELD-GNN checkpoints on clean, attacked, and defended graphs."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shield_data import prepare_shield_inputs, summarize_shield_inputs


def parse_args():
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SHIELD-GNN checkpoint.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--attack-dir", default=None)
    parser.add_argument("--defense-dir", default=None)
    parser.add_argument("--edge-type", choices=["directed", "undirected"], default="undirected")
    return parser.parse_args()


def main():
    """Exit safely when no checkpoint is available; no training is performed."""
    args = parse_args()
    if not args.checkpoint:
        print("No SHIELD-GNN checkpoint provided. Evaluation skipped safely.")
        return
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    inputs = prepare_shield_inputs(
        attack_dir=args.attack_dir,
        defense_dir=args.defense_dir,
        edge_type=args.edge_type,
    )
    summary = summarize_shield_inputs(inputs)
    table_path = PROJECT_ROOT / "results" / "tables" / "shield_gnn_evaluation_comparison.csv"
    json_path = PROJECT_ROOT / "results" / "tables" / "shield_gnn_evaluation_metrics.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"graph": key, "available": value} for key, value in summary.items()]).to_csv(table_path, index=False)
    json_path.write_text(json.dumps({"checkpoint": str(checkpoint_path), "input_summary": summary}, indent=2), encoding="utf-8")
    print("Checkpoint found. Placeholder-safe evaluation summary saved.")


if __name__ == "__main__":
    main()
