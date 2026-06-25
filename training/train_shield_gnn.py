"""Manual training entry point for final SHIELD-GNN integration."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import accuracy_score_torch, compute_roc_auc_safe, precision_recall_f1
from src.models.shield_gnn import SHIELDGNNModel
from src.shield_data import prepare_shield_inputs, summarize_shield_inputs
from src.shield_losses import shield_total_loss
from src.temporal_data import make_small_temporal_subgraph
from src.training_utils import compute_class_weights, get_device, save_json, save_training_curves, set_seed


CHECKPOINT_DIR = PROJECT_ROOT / "results" / "model_checkpoints"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def parse_args():
    """Parse SHIELD-GNN arguments."""
    parser = argparse.ArgumentParser(description="Train or dry-run final SHIELD-GNN.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--edge-type", choices=["directed", "undirected"], default="undirected")
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "graph_artifacts"),
        help="Clean graph artifact directory. Defaults to the 100k subset artifacts.",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backbone", choices=["temporal_gcn_gru", "time_aware_gcn"], default="temporal_gcn_gru")
    parser.add_argument("--snapshot-mode", choices=["current", "cumulative"], default="current")
    parser.add_argument("--temporal-grad-mode", choices=["full", "detach", "final_only"], default="detach")
    parser.add_argument("--detach-memory-each-step", action="store_true", default=True)
    parser.add_argument("--use-attack", action="store_true")
    parser.add_argument("--attack-dir", default=None)
    parser.add_argument("--use-defense", action="store_true")
    parser.add_argument("--defense-dir", default=None)
    parser.add_argument("--alpha-defended", type=float, default=1.0)
    parser.add_argument("--beta-consistency", type=float, default=0.5)
    parser.add_argument("--gamma-pruning", type=float, default=0.1)
    parser.add_argument("--disable-edge-pruning-loss", action="store_true")
    parser.add_argument("--disable-temporal-consistency", action="store_true")
    parser.add_argument("--disable-defense-branch", action="store_true")
    parser.add_argument("--disable-attack-branch", action="store_true")
    parser.add_argument("--disable-node-type-features", action="store_true")
    parser.add_argument("--disable-pruning-regularization", action="store_true")
    parser.add_argument("--ablation-name", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--eval-clean", action="store_true")
    parser.add_argument("--eval-attack", action="store_true")
    parser.add_argument("--eval-defense", action="store_true")
    parser.add_argument("--save-name", default="shield_gnn")
    return parser.parse_args()


def output_path(base_dir, args, filename):
    """Return an output path, optionally under a named subdirectory."""
    root = base_dir / args.output_subdir if args.output_subdir else base_dir
    root.mkdir(parents=True, exist_ok=True)
    return root / filename


def node_type_feature_slice(args):
    """Infer the node-type one-hot feature slice from an artifact schema."""
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir
    schema_path = artifact_dir / "feature_schema.json"
    if not schema_path.is_file():
        return None
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    start = int(schema.get("transaction_feature_dim", 0)) + int(schema.get("wallet_feature_dim", 0))
    return slice(start, start + 2)


def maybe_disable_node_type_features(graph, args):
    """Zero node-type one-hot columns for Elliptic++ ablations without editing artifacts."""
    if not args.disable_node_type_features:
        return graph
    feature_slice = node_type_feature_slice(args)
    if feature_slice is None:
        return graph
    graph = dict(graph)
    graph["X"] = graph["X"].copy()
    if graph["X"].shape[1] >= feature_slice.stop:
        graph["X"][:, feature_slice] = 0.0
    return graph


def graph_to_tensors(graph, device, args=None):
    """Convert a graph dictionary into tensors."""
    if args is not None:
        graph = maybe_disable_node_type_features(graph, args)
    tensors = {
        "x": torch.tensor(graph["X"], dtype=torch.float32, device=device),
        "y": torch.tensor(graph["y"], dtype=torch.long, device=device),
        "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long, device=device),
        "timesteps": torch.tensor(
            graph["timesteps"] if graph.get("timesteps") is not None else [0] * graph["X"].shape[0],
            dtype=torch.long,
            device=device,
        ),
    }
    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        if graph.get(mask_name) is not None:
            tensors[mask_name] = torch.tensor(graph[mask_name], dtype=torch.bool, device=device)
    if graph.get("edge_weight") is not None:
        tensors["edge_weight"] = torch.tensor(graph["edge_weight"], dtype=torch.float32, device=device)
    else:
        tensors["edge_weight"] = None
    return tensors


def small_clean_tensors(clean_graph, device, max_nodes=2048):
    """Build a small clean graph tensor batch for safe dry-runs."""
    small = make_small_temporal_subgraph(
        {
            "X": clean_graph["X"],
            "y": clean_graph["y"],
            "edge_index_undirected": clean_graph["edge_index"],
            "timesteps": clean_graph["timesteps"],
            "train_mask": clean_graph["train_mask"],
        },
        max_nodes=max_nodes,
    )
    graph = {
        "X": small["X"],
        "y": small["y"],
        "edge_index": small["edge_index_undirected"],
        "timesteps": small["timesteps"],
        "train_mask": small["train_mask"],
        "val_mask": small["train_mask"],
        "test_mask": small["train_mask"],
        "edge_weight": None,
    }
    return graph_to_tensors(graph, device)


def create_model(args, in_channels, max_timesteps, device):
    """Create SHIELD-GNN model."""
    return SHIELDGNNModel(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        backbone=args.backbone,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_temporal_edge_encoding=True,
        use_edge_weight=True,
        max_timesteps=max_timesteps,
        snapshot_mode=args.snapshot_mode,
        detach_memory_each_step=args.detach_memory_each_step,
        temporal_grad_mode=args.temporal_grad_mode,
    ).to(device)


def dry_run(args, inputs, device):
    """Run safe SHIELD-GNN forward/loss validation only."""
    clean = small_clean_tensors(inputs["clean"], device)
    model = create_model(args, clean["x"].shape[1], int(clean["timesteps"].max().item()) + 1, device)
    model.eval()
    with torch.no_grad():
        clean_logits = model(clean["x"], clean["edge_index"], clean["timesteps"])
        defended_logits = None
        total_loss, components = shield_total_loss(
            clean_logits,
            clean["y"],
            clean["train_mask"],
            defended_logits=defended_logits,
            timesteps=clean["timesteps"],
            edge_index=clean["edge_index"],
            class_weights=None,
            alpha_defended=args.alpha_defended,
            beta_consistency=args.beta_consistency,
            gamma_pruning=args.gamma_pruning,
        )
    print("SHIELD-GNN dry run only: no optimizer, backward pass, epochs, or checkpoint saving.")
    print(f"Device: {device}")
    print(f"Input summary: {summarize_shield_inputs(inputs)}")
    print(f"Clean logits shape: {tuple(clean_logits.shape)}")
    print(f"Dry-run total loss: {float(total_loss.item()):.6f}")
    print({key: float(value.item()) for key, value in components.items()})


def evaluate_shield(model, tensors, mask, criterion=None):
    """Evaluate SHIELD-GNN with temporal inputs and optional defended weights."""
    model.eval()
    with torch.no_grad():
        logits = model(
            tensors["x"],
            tensors["edge_index"],
            tensors["timesteps"],
            tensors["edge_weight"],
        )
        selected = mask.bool() & (tensors["y"] != -1)
        if int(selected.sum()) == 0:
            return {
                "loss": None,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": None,
                "num_nodes": 0,
                "illicit_count": 0,
                "licit_count": 0,
            }
        loss = criterion(logits[selected], tensors["y"][selected]) if criterion else torch.nn.functional.cross_entropy(
            logits[selected], tensors["y"][selected]
        )
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        precision, recall, f1 = precision_recall_f1(tensors["y"], preds, selected)
        return {
            "loss": float(loss.item()),
            "accuracy": accuracy_score_torch(tensors["y"], preds, selected),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": compute_roc_auc_safe(tensors["y"], probs[:, 1], selected),
            "num_nodes": int(selected.sum().item()),
            "illicit_count": int(((tensors["y"] == 1) & selected).sum().item()),
            "licit_count": int(((tensors["y"] == 0) & selected).sum().item()),
        }


def print_startup(args, device, inputs, tensors):
    """Print startup configuration for non-silent SHIELD runs."""
    print("SHIELD-GNN training started")
    print(f"Artifact dir: {args.artifact_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Num layers: {args.num_layers}")
    print(
        "Train/val/test counts: "
        f"{int(tensors['train_mask'].sum())}/"
        f"{int(tensors['val_mask'].sum())}/"
        f"{int(tensors['test_mask'].sum())}"
    )
    print(f"Use attack: {args.use_attack}")
    print(f"Use defense: {args.use_defense}")
    print(f"Save name: {args.save_name}")
    print(f"Input summary: {summarize_shield_inputs(inputs)}")


def train_one_epoch(model, tensors, optimizer, class_weights, args):
    """Run one manual-training epoch."""
    model.train()
    optimizer.zero_grad()
    clean = tensors["clean"]
    clean_logits = model(clean["x"], clean["edge_index"], clean["timesteps"], clean["edge_weight"])
    attacked_logits = None
    defended_logits = None
    attacked_loss = clean_logits.sum() * 0.0
    defended_loss = clean_logits.sum() * 0.0

    attacked = tensors.get("attack")
    if attacked is not None:
        attacked_logits = model(attacked["x"], attacked["edge_index"], attacked["timesteps"], attacked["edge_weight"])
        attacked_loss = torch.nn.functional.cross_entropy(
            attacked_logits[attacked["train_mask"] & (attacked["y"] != -1)],
            attacked["y"][attacked["train_mask"] & (attacked["y"] != -1)],
            weight=class_weights,
        )

    defended = tensors.get("defense")
    if defended is not None:
        defended_logits = model(defended["x"], defended["edge_index"], defended["timesteps"], defended["edge_weight"])
        defended_loss = torch.nn.functional.cross_entropy(
            defended_logits[defended["train_mask"] & (defended["y"] != -1)],
            defended["y"][defended["train_mask"] & (defended["y"] != -1)],
            weight=class_weights,
        )

    consistency_attacked = attacked_logits if attacked_logits is not None and attacked_logits.shape == clean_logits.shape else None
    consistency_defended = defended_logits if defended_logits is not None and defended_logits.shape == clean_logits.shape else None
    beta = 0.0 if args.disable_temporal_consistency else args.beta_consistency
    gamma = 0.0 if args.disable_pruning_regularization or args.disable_edge_pruning_loss else args.gamma_pruning
    total_loss, _ = shield_total_loss(
        clean_logits,
        clean["y"],
        clean["train_mask"],
        defended_logits=consistency_defended,
        attacked_logits=consistency_attacked,
        timesteps=clean["timesteps"],
        edge_index=clean["edge_index"],
        edge_weight=defended["edge_weight"] if defended is not None else clean["edge_weight"],
        class_weights=class_weights,
        alpha_defended=0.0,
        beta_consistency=beta,
        gamma_pruning=gamma,
    )
    total_loss = total_loss + attacked_loss + args.alpha_defended * defended_loss
    total_loss.backward()
    optimizer.step()
    return float(total_loss.item())


def main():
    """Run SHIELD-GNN dry-run or manual training."""
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    inputs = prepare_shield_inputs(
        clean_dir=args.artifact_dir,
        attack_dir=args.attack_dir if args.use_attack and not args.disable_attack_branch else None,
        defense_dir=args.defense_dir if args.use_defense and not args.disable_defense_branch else None,
        edge_type=args.edge_type,
    )
    if args.dry_run:
        print("SHIELD-GNN dry-run started")
        print(f"Artifact dir: {args.artifact_dir}")
        print(f"Device: {device}")
        print(f"Save name: {args.save_name}")
        dry_run(args, inputs, device)
        print("SHIELD-GNN dry-run success")
        return

    clean_tensors = graph_to_tensors(inputs["clean"], device, args=args)
    training_tensors = {"clean": clean_tensors}
    if inputs.get("attack") is not None:
        training_tensors["attack"] = graph_to_tensors(inputs["attack"], device, args=args)
    if inputs.get("defense") is not None:
        training_tensors["defense"] = graph_to_tensors(inputs["defense"], device, args=args)
    print_startup(args, device, inputs, clean_tensors)
    model = create_model(args, clean_tensors["x"].shape[1], int(clean_tensors["timesteps"].max().item()) + 1, device)
    class_weights = compute_class_weights(clean_tensors["y"], clean_tensors["train_mask"])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    best_epoch = 0
    wait = 0
    rows = []
    output_prefix = args.ablation_name if args.ablation_name else args.save_name
    checkpoint_path = output_path(CHECKPOINT_DIR, args, f"{output_prefix}_best.pt")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, training_tensors, optimizer, class_weights, args)
        train_metrics = evaluate_shield(model, clean_tensors, clean_tensors["train_mask"], criterion)
        val_metrics = evaluate_shield(model, clean_tensors, clean_tensors["val_mask"], criterion)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_recall": val_metrics["recall"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f}"
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                },
                checkpoint_path,
            )
        else:
            wait += 1
        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best validation F1: {best_val_f1:.4f}")
            break
    if not checkpoint_path.is_file():
        raise RuntimeError("Training finished without saving a checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_shield(model, clean_tensors, clean_tensors["test_mask"], criterion)
    log_path = output_path(TABLES_DIR, args, f"{output_prefix}_training_log.csv")
    metrics_path = output_path(TABLES_DIR, args, f"{output_prefix}_metrics.json")
    pd.DataFrame(rows).to_csv(log_path, index=False)
    metrics = {
        "model": "shield_gnn",
        "experiment_type": "attack_defense" if args.use_attack or args.use_defense else "clean",
        "artifact_scope": "full" if "full" in str(args.artifact_dir).lower() else "subset_100k",
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "training_log_path": str(log_path),
        "args": vars(args),
    }
    save_json(metrics_path, metrics)
    save_training_curves(
        rows,
        output_path(PLOTS_DIR, args, f"{output_prefix}_loss_curve.png"),
        output_path(PLOTS_DIR, args, f"{output_prefix}_f1_curve.png"),
    )
    print("\nFinal SHIELD-GNN test metrics")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved training log: {log_path}")
    print(f"Saved metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
