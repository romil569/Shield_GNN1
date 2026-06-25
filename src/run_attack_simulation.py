"""Run adversarial attack simulations without training any model."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks.attack_utils import remove_duplicate_edges, safe_numpy_save, save_attack_summary
from src.attacks.combined_attack import run_combined_attack
from src.attacks.fake_edge_injection import inject_fake_edges
from src.attacks.fake_node_injection import inject_fake_nodes
from src.attacks.feature_perturbation import perturb_node_features
from src.training_utils import load_graph_artifacts


ATTACK_ROOT = PROJECT_ROOT / "data" / "processed" / "attack_artifacts"


def parse_args():
    """Parse attack simulation arguments."""
    parser = argparse.ArgumentParser(description="Simulate adversarial graph attacks.")
    parser.add_argument(
        "--attack",
        "--attack-type",
        dest="attack",
        choices=["fake_node", "fake_edge", "feature", "combined", "targeted_evasion"],
        required=True,
    )
    parser.add_argument("--edge-type", choices=["directed", "undirected"], default="undirected")
    parser.add_argument("--fake-node-rate", "--fake-node-ratio", dest="fake_node_rate", type=float, default=0.02)
    parser.add_argument("--fake-edge-rate", "--fake-edge-ratio", dest="fake_edge_rate", type=float, default=0.03)
    parser.add_argument(
        "--feature-perturb-rate",
        "--feature-perturbation-ratio",
        dest="feature_perturb_rate",
        type=float,
        default=0.05,
    )
    parser.add_argument("--edges-per-fake-node", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--target-strategy", default="illicit")
    parser.add_argument("--feature-strategy", default="mean_noise")
    parser.add_argument("--target-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--target-class", choices=["illicit", "licit", "labeled"], default="illicit")
    parser.add_argument("--perturbation-strategy", default="move_toward_licit_mean")
    parser.add_argument("--camouflage-strategy", default="connect_illicit_to_licit_hubs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "graph_artifacts"),
        help="Graph artifact directory to attack.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ATTACK_ROOT),
        help="Directory where attack artifacts are saved.",
    )
    parser.add_argument("--attack-name", default=None)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def _target_mask_for_split(args, y, train_mask, val_mask, test_mask):
    """Return target mask for a split/class pair."""
    split_masks = {"train": train_mask, "val": val_mask, "test": test_mask}
    mask = split_masks[args.target_split].copy()
    if args.target_class == "illicit":
        mask = mask & (y == 1)
    elif args.target_class == "licit":
        mask = mask & (y == 0)
    else:
        mask = mask & (y != -1)
    return mask


def _one_hop_neighbors(edge_index, nodes, num_nodes):
    """Return unique one-hop neighbors of selected nodes."""
    if len(nodes) == 0:
        return np.array([], dtype=np.int64)
    selected = np.zeros(num_nodes, dtype=bool)
    selected[nodes] = True
    incident = selected[edge_index[0]] | selected[edge_index[1]]
    neighbors = np.unique(edge_index[:, incident].reshape(-1))
    return neighbors[~selected[neighbors]].astype(np.int64)


def run_targeted_evasion_attack(args, artifacts, fake_node_rate, fake_edge_rate, feature_rate):
    """Attack original test fraud nodes and their neighborhood without changing labels."""
    X = artifacts["X"]
    y = artifacts["y"]
    edge_index = artifacts[f"edge_index_{args.edge_type}"]
    timesteps = artifacts["timesteps"]
    train_mask = artifacts["train_mask"]
    val_mask = artifacts["val_mask"]
    test_mask = artifacts["test_mask"]
    rng = np.random.default_rng(args.seed)

    target_mask = _target_mask_for_split(args, y, train_mask, val_mask, test_mask)
    target_nodes = np.flatnonzero(target_mask).astype(np.int64)
    if target_nodes.size == 0:
        raise ValueError("Targeted evasion attack found no target nodes.")

    neighbors = _one_hop_neighbors(edge_index, target_nodes, X.shape[0])
    licit_train = np.flatnonzero((y == 0) & train_mask)
    licit_all = np.flatnonzero(y == 0)
    licit_pool = licit_train if licit_train.size else licit_all
    if licit_pool.size == 0:
        raise ValueError("Targeted evasion attack requires licit nodes for camouflage.")

    degree = np.bincount(edge_index.reshape(-1), minlength=X.shape[0])
    hub_count = max(1, min(5000, licit_pool.size))
    licit_hubs = licit_pool[np.argsort(degree[licit_pool])[-hub_count:]]
    licit_mean = X[licit_pool].mean(axis=0)

    num_fake_nodes = max(1, int(round(X.shape[0] * fake_node_rate)))
    fake_node_indices = np.arange(X.shape[0], X.shape[0] + num_fake_nodes, dtype=np.int64)
    fake_anchor_nodes = rng.choice(target_nodes, size=num_fake_nodes, replace=True)
    fake_features = np.repeat(licit_mean.reshape(1, -1), num_fake_nodes, axis=0)
    fake_features += rng.normal(0.0, args.noise_std, size=fake_features.shape) * X[licit_pool].std(axis=0)
    fake_features = fake_features.astype(X.dtype, copy=False)
    fake_y = np.full(num_fake_nodes, -1, dtype=y.dtype)

    fake_edges = []
    for fake_node, anchor in zip(fake_node_indices, fake_anchor_nodes):
        fake_edges.append((int(fake_node), int(anchor)))
        fake_edges.append((int(anchor), int(fake_node)))

    fake_edge_requested = max(1, int(round(edge_index.shape[1] * fake_edge_rate)))
    sources = rng.choice(target_nodes, size=fake_edge_requested, replace=True)
    targets = rng.choice(licit_hubs, size=fake_edge_requested, replace=True)
    valid = sources != targets
    camouflage_edges = np.vstack([sources[valid], targets[valid]]).astype(np.int64)

    fake_node_edges = np.asarray(fake_edges, dtype=np.int64).T if fake_edges else np.empty((2, 0), dtype=np.int64)
    attacked_edge_index = remove_duplicate_edges(
        np.concatenate([edge_index.copy(), fake_node_edges, camouflage_edges], axis=1)
    )

    candidate_perturb = np.unique(np.concatenate([target_nodes, neighbors])).astype(np.int64)
    target_count = max(1, int(round(target_nodes.size * feature_rate)))
    neighbor_count = max(0, int(round(max(0, candidate_perturb.size - target_nodes.size) * feature_rate)))
    perturbed_targets = rng.choice(target_nodes, size=min(target_count, target_nodes.size), replace=False)
    neighbor_candidates = np.setdiff1d(candidate_perturb, target_nodes, assume_unique=False)
    perturbed_neighbors = (
        rng.choice(neighbor_candidates, size=min(neighbor_count, neighbor_candidates.size), replace=False)
        if neighbor_candidates.size
        else np.array([], dtype=np.int64)
    )
    perturbed_nodes = np.unique(np.concatenate([perturbed_targets, perturbed_neighbors])).astype(np.int64)

    attacked_X_original = X.copy()
    if args.perturbation_strategy == "move_toward_licit_mean":
        attacked_X_original[perturbed_nodes] = (
            0.35 * attacked_X_original[perturbed_nodes] + 0.65 * licit_mean
        ).astype(X.dtype, copy=False)
    else:
        attacked_X_original[perturbed_nodes] += rng.normal(
            0.0, args.noise_std, size=attacked_X_original[perturbed_nodes].shape
        ).astype(X.dtype, copy=False)

    attacked_X = np.vstack([attacked_X_original, fake_features])
    attacked_y = np.concatenate([y.copy(), fake_y])
    attacked_timesteps = np.concatenate(
        [timesteps.copy(), np.full(num_fake_nodes, int(timesteps.max()), dtype=timesteps.dtype)]
    )
    extra_mask = np.zeros(num_fake_nodes, dtype=bool)

    summary = {
        "attack_type": "targeted_evasion",
        "target_split": args.target_split,
        "target_class": args.target_class,
        "target_nodes": int(target_nodes.size),
        "target_test_illicit_nodes": int(((y == 1) & test_mask).sum()),
        "one_hop_neighbors": int(neighbors.size),
        "fake_nodes_added": int(num_fake_nodes),
        "fake_node_edges_added_before_dedup": int(fake_node_edges.shape[1]),
        "camouflage_edges_requested": int(fake_edge_requested),
        "camouflage_edges_added_before_dedup": int(camouflage_edges.shape[1]),
        "final_graph_stats": {
            "clean_num_nodes": int(X.shape[0]),
            "clean_num_edges": int(edge_index.shape[1]),
            "attacked_num_nodes": int(attacked_X.shape[0]),
            "attacked_num_edges": int(attacked_edge_index.shape[1]),
            "nodes_added": int(attacked_X.shape[0] - X.shape[0]),
            "edges_added": int(attacked_edge_index.shape[1] - edge_index.shape[1]),
        },
        "perturbed_original_nodes": int(perturbed_nodes.size),
        "perturbed_target_nodes": int(perturbed_targets.size),
        "perturbed_neighbor_nodes": int(perturbed_neighbors.size),
        "perturbation_strategy": args.perturbation_strategy,
        "camouflage_strategy": args.camouflage_strategy,
        "fake_node_ratio": float(fake_node_rate),
        "fake_edge_ratio": float(fake_edge_rate),
        "feature_perturbation_ratio": float(feature_rate),
        "fake_nodes_in_test_mask": 0,
        "preserved_original_test_mask": True,
    }
    return {
        "attacked_X": attacked_X,
        "attacked_y": attacked_y,
        "attacked_edge_index": attacked_edge_index,
        "fake_node_indices": fake_node_indices,
        "fake_edges": camouflage_edges,
        "perturbed_node_indices": perturbed_nodes,
        "summary": summary,
        "attacked_timesteps": attacked_timesteps,
        "attacked_train_mask": np.concatenate([train_mask.copy(), extra_mask]),
        "attacked_val_mask": np.concatenate([val_mask.copy(), extra_mask]),
        "attacked_test_mask": np.concatenate([test_mask.copy(), extra_mask]),
    }


def run_attack(args, artifacts):
    """Execute the selected attack and return outputs in a common dictionary."""
    X = artifacts["X"]
    y = artifacts["y"]
    edge_index = artifacts[f"edge_index_{args.edge_type}"]
    timesteps = artifacts["timesteps"]
    train_mask = artifacts["train_mask"]
    val_mask = artifacts["val_mask"]
    test_mask = artifacts["test_mask"]

    if args.dry_run:
        fake_node_rate = min(args.fake_node_rate, 0.001)
        fake_edge_rate = min(args.fake_edge_rate, 0.001)
        feature_rate = min(args.feature_perturb_rate, 0.001)
    else:
        fake_node_rate = args.fake_node_rate
        fake_edge_rate = args.fake_edge_rate
        feature_rate = args.feature_perturb_rate

    result = {
        "attacked_X": X.copy(),
        "attacked_y": y.copy(),
        "attacked_edge_index": edge_index.copy(),
        "fake_node_indices": None,
        "fake_edges": None,
        "perturbed_node_indices": None,
        "summary": None,
        "attacked_timesteps": timesteps.copy(),
        "attacked_train_mask": train_mask.copy(),
        "attacked_val_mask": val_mask.copy(),
        "attacked_test_mask": test_mask.copy(),
    }

    if args.attack == "targeted_evasion":
        return run_targeted_evasion_attack(args, artifacts, fake_node_rate, fake_edge_rate, feature_rate)
    if args.attack == "fake_node":
        attacked_X, attacked_y, attacked_edge_index, fake_node_indices, summary = inject_fake_nodes(
            X,
            y,
            edge_index,
            timesteps=timesteps,
            injection_rate=fake_node_rate,
            connect_to=args.target_strategy,
            edges_per_fake_node=args.edges_per_fake_node,
            feature_strategy=args.feature_strategy,
            noise_std=args.noise_std,
            seed=args.seed,
        )
        result.update(
            {
                "attacked_X": attacked_X,
                "attacked_y": attacked_y,
                "attacked_edge_index": attacked_edge_index,
                "fake_node_indices": fake_node_indices,
                "summary": summary,
                "attacked_timesteps": np.concatenate(
                    [timesteps.copy(), np.full(len(fake_node_indices), int(timesteps.max()), dtype=timesteps.dtype)]
                ),
                "attacked_train_mask": np.concatenate([train_mask.copy(), np.zeros(len(fake_node_indices), dtype=bool)]),
                "attacked_val_mask": np.concatenate([val_mask.copy(), np.zeros(len(fake_node_indices), dtype=bool)]),
                "attacked_test_mask": np.concatenate([test_mask.copy(), np.zeros(len(fake_node_indices), dtype=bool)]),
            }
        )
    elif args.attack == "fake_edge":
        edge_strategy = (
            "illicit_to_licit" if args.target_strategy == "illicit" else args.target_strategy
        )
        attacked_edge_index, fake_edges, summary = inject_fake_edges(
            X,
            y,
            edge_index,
            injection_rate=fake_edge_rate,
            strategy=edge_strategy,
            seed=args.seed,
        )
        result.update(
            {
                "attacked_edge_index": attacked_edge_index,
                "fake_edges": fake_edges,
                "summary": summary,
            }
        )
    elif args.attack == "feature":
        attacked_X, perturbed_node_indices, summary = perturb_node_features(
            X,
            y,
            perturbation_rate=feature_rate,
            noise_std=args.noise_std,
            strategy=args.target_strategy,
            seed=args.seed,
        )
        result.update(
            {
                "attacked_X": attacked_X,
                "perturbed_node_indices": perturbed_node_indices,
                "summary": summary,
            }
        )
    elif args.attack == "combined":
        attacked_X, attacked_y, attacked_edge_index, summary = run_combined_attack(
            X,
            y,
            edge_index,
            timesteps=timesteps,
            fake_node_rate=fake_node_rate,
            fake_edge_rate=fake_edge_rate,
            feature_perturb_rate=feature_rate,
            seed=args.seed,
        )
        result.update(
            {
                "attacked_X": attacked_X,
                "attacked_y": attacked_y,
                "attacked_edge_index": attacked_edge_index,
                "summary": summary,
            }
        )
        added_nodes = attacked_X.shape[0] - X.shape[0]
        if added_nodes:
            result.update(
                {
                    "attacked_timesteps": np.concatenate(
                        [timesteps.copy(), np.full(added_nodes, int(timesteps.max()), dtype=timesteps.dtype)]
                    ),
                    "attacked_train_mask": np.concatenate([train_mask.copy(), np.zeros(added_nodes, dtype=bool)]),
                    "attacked_val_mask": np.concatenate([val_mask.copy(), np.zeros(added_nodes, dtype=bool)]),
                    "attacked_test_mask": np.concatenate([test_mask.copy(), np.zeros(added_nodes, dtype=bool)]),
                }
            )
    return result


def save_outputs(args, result):
    """Save attacked graph artifacts in a versioned attack directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    attack_name = args.output_name or args.attack_name or (
        f"{args.attack}_fn{args.fake_node_rate}_fe{args.fake_edge_rate}_"
        f"fp{args.feature_perturb_rate}_{timestamp}"
    )
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_dir = output_root / attack_name
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_numpy_save(output_dir / "attacked_X.npy", result["attacked_X"])
    safe_numpy_save(output_dir / "attacked_y.npy", result["attacked_y"])
    safe_numpy_save(output_dir / "attacked_edge_index.npy", result["attacked_edge_index"])
    safe_numpy_save(output_dir / "attacked_timesteps.npy", result["attacked_timesteps"])
    safe_numpy_save(output_dir / "attacked_train_mask.npy", result["attacked_train_mask"])
    safe_numpy_save(output_dir / "attacked_val_mask.npy", result["attacked_val_mask"])
    safe_numpy_save(output_dir / "attacked_test_mask.npy", result["attacked_test_mask"])
    save_attack_summary(output_dir / "attack_summary.json", result["summary"])
    if result["fake_node_indices"] is not None:
        safe_numpy_save(output_dir / "fake_node_indices.npy", result["fake_node_indices"])
    if result["fake_edges"] is not None:
        safe_numpy_save(output_dir / "fake_edges.npy", result["fake_edges"])
    if result["perturbed_node_indices"] is not None:
        safe_numpy_save(output_dir / "perturbed_node_indices.npy", result["perturbed_node_indices"])
    return output_dir


def main():
    """Run attack simulation safely."""
    args = parse_args()
    artifacts = load_graph_artifacts(args.artifact_dir)
    result = run_attack(args, artifacts)
    print("Attack simulation completed.")
    print(f"Attack: {args.attack}")
    print(f"Dry run: {args.dry_run}")
    print(f"Clean X shape: {artifacts['X'].shape}")
    print(f"Attacked X shape: {result['attacked_X'].shape}")
    print(f"Clean edge shape: {artifacts[f'edge_index_{args.edge_type}'].shape}")
    print(f"Attacked edge shape: {result['attacked_edge_index'].shape}")
    print("Summary:", result["summary"])
    if args.save_artifacts and not args.dry_run:
        output_dir = save_outputs(args, result)
        print(f"Saved attack artifacts: {output_dir}")
    elif args.save_artifacts and args.dry_run:
        print("Dry-run mode: artifacts were not saved.")


if __name__ == "__main__":
    main()
