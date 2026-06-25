"""Run robust edge pruning defense without training any model."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.defense.edge_pruning import run_robust_edge_pruning
from src.defense.pruning_utils import safe_save_numpy, save_defense_summary
from src.training_utils import load_graph_artifacts


DEFENSE_ROOT = PROJECT_ROOT / "data" / "processed" / "defense_artifacts"


def parse_args():
    """Parse edge pruning defense arguments."""
    parser = argparse.ArgumentParser(description="Run robust edge pruning defense.")
    parser.add_argument("--input-type", choices=["clean", "attack"], default="clean")
    parser.add_argument("--attack-dir", default=None)
    parser.add_argument("--edge-type", choices=["directed", "undirected"], default="undirected")
    parser.add_argument("--mode", choices=["hard", "soft", "hybrid"], default="hybrid")
    parser.add_argument("--prune-threshold", type=float, default=0.80)
    parser.add_argument("--threshold-mode", choices=["fixed", "percentile", "topk"], default="fixed")
    parser.add_argument("--score-percentile", type=float, default=95.0)
    parser.add_argument("--prune-ratio", type=float, default=None)
    parser.add_argument("--min-prune-edges", type=int, default=0)
    parser.add_argument("--preserve-original-test-mask", action="store_true")
    parser.add_argument("--downweight-threshold", type=float, default=0.50)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--feature-weight", type=float, default=0.40)
    parser.add_argument("--degree-weight", type=float, default=0.25)
    parser.add_argument("--label-weight", type=float, default=0.20)
    parser.add_argument("--temporal-weight", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "graph_artifacts"),
        help="Clean graph artifact directory used when --input-type clean.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFENSE_ROOT),
        help="Directory where defense artifacts are saved.",
    )
    return parser.parse_args()


def load_attack_artifacts(attack_dir):
    """Load attacked graph artifacts from a Step 5 attack output folder."""
    if attack_dir is None:
        raise ValueError("--attack-dir is required when --input-type attack.")
    attack_path = Path(attack_dir)
    if not attack_path.is_absolute():
        attack_path = PROJECT_ROOT / attack_path
    required = ["attacked_X.npy", "attacked_y.npy", "attacked_edge_index.npy"]
    missing = [name for name in required if not (attack_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing attack artifacts in {attack_path}: {', '.join(missing)}")
    return {
        "X": np.load(attack_path / "attacked_X.npy"),
        "y": np.load(attack_path / "attacked_y.npy"),
        "edge_index": np.load(attack_path / "attacked_edge_index.npy"),
        "timesteps": np.load(attack_path / "attacked_timesteps.npy") if (attack_path / "attacked_timesteps.npy").is_file() else None,
        "train_mask": np.load(attack_path / "attacked_train_mask.npy") if (attack_path / "attacked_train_mask.npy").is_file() else None,
        "val_mask": np.load(attack_path / "attacked_val_mask.npy") if (attack_path / "attacked_val_mask.npy").is_file() else None,
        "test_mask": np.load(attack_path / "attacked_test_mask.npy") if (attack_path / "attacked_test_mask.npy").is_file() else None,
        "source": str(attack_path),
    }


def load_input_graph(args):
    """Load clean or attacked graph arrays for pruning."""
    if args.input_type == "clean":
        artifacts = load_graph_artifacts(args.artifact_dir)
        return {
            "X": artifacts["X"],
            "y": artifacts["y"],
            "edge_index": artifacts[f"edge_index_{args.edge_type}"],
            "timesteps": artifacts["timesteps"],
            "train_mask": artifacts["train_mask"],
            "val_mask": artifacts["val_mask"],
            "test_mask": artifacts["test_mask"],
            "source": "clean_graph_artifacts",
        }
    return load_attack_artifacts(args.attack_dir)


def save_defense_outputs(args, result):
    """Save defended graph artifacts under a separate defense directory."""
    if args.output_name:
        output_name = args.output_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{args.input_type}_{args.mode}_pruned_{timestamp}"
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_dir = output_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_save_numpy(output_dir / "defended_X.npy", result["X"])
    safe_save_numpy(output_dir / "defended_y.npy", result["y"])
    safe_save_numpy(output_dir / "defended_edge_index.npy", result["defended_edge_index"])
    safe_save_numpy(output_dir / "defended_edge_weight.npy", result["defended_edge_weight"])
    if result.get("timesteps") is not None:
        safe_save_numpy(output_dir / "defended_timesteps.npy", result["timesteps"])
    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        if result.get(mask_name) is not None:
            safe_save_numpy(output_dir / f"defended_{mask_name}.npy", result[mask_name])
    safe_save_numpy(output_dir / "edge_scores.npy", result["edge_scores"])
    save_defense_summary(output_dir / "pruning_summary.json", result["pruning_summary"])
    np.savez(output_dir / "score_components.npz", **result["score_components"])
    return output_dir


def main():
    """Run robust edge pruning and optionally save defense artifacts."""
    args = parse_args()
    graph = load_input_graph(args)
    score_weights = {
        "feature_weight": args.feature_weight,
        "degree_weight": args.degree_weight,
        "label_weight": args.label_weight,
        "temporal_weight": args.temporal_weight,
    }
    defended_edge_index, defended_edge_weight, edge_scores, score_components, pruning_summary = (
        run_robust_edge_pruning(
            graph["X"],
            graph["y"],
            graph["edge_index"],
            timesteps=graph["timesteps"],
            mode=args.mode,
            prune_threshold=args.prune_threshold,
            downweight_threshold=args.downweight_threshold,
            min_weight=args.min_weight,
            score_weights=score_weights,
            threshold_mode=args.threshold_mode,
            score_percentile=args.score_percentile,
            prune_ratio=args.prune_ratio,
            min_prune_edges=args.min_prune_edges,
        )
    )
    result = {
        "X": graph["X"],
        "y": graph["y"],
        "timesteps": graph.get("timesteps"),
        "train_mask": graph.get("train_mask"),
        "val_mask": graph.get("val_mask"),
        "test_mask": graph.get("test_mask"),
        "defended_edge_index": defended_edge_index,
        "defended_edge_weight": defended_edge_weight,
        "edge_scores": edge_scores,
        "score_components": score_components,
        "pruning_summary": pruning_summary,
    }
    print("Robust edge pruning completed.")
    print(f"Input source: {graph['source']}")
    print(f"Mode: {args.mode}")
    print(f"Dry run: {args.dry_run}")
    print(f"X shape: {graph['X'].shape}")
    print(f"Original edge shape: {graph['edge_index'].shape}")
    print(f"Defended edge shape: {defended_edge_index.shape}")
    print(f"Defended edge weight shape: {defended_edge_weight.shape}")
    print("Pruning summary:", pruning_summary)
    print(f"Attacked/original edges: {graph['edge_index'].shape[1]}")
    print(f"Defended edges: {defended_edge_index.shape[1]}")
    print(f"Pruned edges: {pruning_summary['pruned_edges']}")
    print(f"Prune percentage: {pruning_summary['prune_ratio'] * 100.0:.4f}%")
    print(
        "Score min/mean/max: "
        f"{pruning_summary['score_min']:.6f}/"
        f"{pruning_summary['score_mean']:.6f}/"
        f"{pruning_summary['score_max']:.6f}"
    )
    print(
        "Score percentile thresholds: "
        f"p90={pruning_summary['score_p90']:.6f}, "
        f"p95={pruning_summary['score_p95']:.6f}, "
        f"p99={pruning_summary['score_p99']:.6f}"
    )
    top_count = min(5, edge_scores.shape[0])
    if top_count:
        top_indices = np.argsort(edge_scores)[-top_count:][::-1]
        examples = [
            {
                "edge_position": int(idx),
                "source": int(graph["edge_index"][0, idx]),
                "target": int(graph["edge_index"][1, idx]),
                "score": float(edge_scores[idx]),
            }
            for idx in top_indices
        ]
        print("Top suspicious edge examples:", examples)
    if args.save_artifacts and not args.dry_run:
        output_dir = save_defense_outputs(args, result)
        print(f"Saved defense artifacts: {output_dir}")
    elif args.save_artifacts and args.dry_run:
        print("Dry-run mode: defense artifacts were not saved.")


if __name__ == "__main__":
    main()
