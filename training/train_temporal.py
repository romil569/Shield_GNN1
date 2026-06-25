"""Manual temporal GNN training entry point for temporal baselines."""

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "model_checkpoints"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def parse_args():
    """Parse command-line arguments for manual temporal training."""
    parser = argparse.ArgumentParser(description="Train temporal GNN baselines manually.")
    parser.add_argument("--model", choices=["temporal_gcn_gru", "time_aware_gcn"], required=True)
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
        help="Graph artifact directory. Defaults to the 100k subset artifacts.",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--snapshot-mode", choices=["cumulative", "current"], default="cumulative")
    parser.add_argument("--detach-memory-each-step", action="store_true")
    parser.add_argument(
        "--temporal-grad-mode",
        choices=["full", "detach", "final_only"],
        default="detach",
        help="Temporal gradient strategy for TemporalGCN-GRU. Use detach for full-dataset runs.",
    )
    parser.add_argument(
        "--current-snapshot-training",
        action="store_true",
        help="Use current timestep snapshots for TemporalGCN-GRU training.",
    )
    parser.add_argument(
        "--save-name",
        default=None,
        help="Optional output filename prefix for checkpoints, logs, metrics, and plots.",
    )
    return parser.parse_args()


def tensors_from_artifacts(artifacts, edge_type, device):
    """Convert temporal graph artifacts into torch tensors."""
    import torch

    edge_key = f"edge_index_{edge_type}"
    return {
        "x": torch.tensor(artifacts["X"], dtype=torch.float32, device=device),
        "y": torch.tensor(artifacts["y"], dtype=torch.long, device=device),
        "edge_index": torch.tensor(artifacts[edge_key], dtype=torch.long, device=device),
        "timesteps": torch.tensor(artifacts["timesteps"], dtype=torch.long, device=device),
        "train_mask": torch.tensor(artifacts["train_mask"], dtype=torch.bool, device=device),
        "val_mask": torch.tensor(artifacts["val_mask"], dtype=torch.bool, device=device),
        "test_mask": torch.tensor(artifacts["test_mask"], dtype=torch.bool, device=device),
    }


def small_tensors_from_artifacts(artifacts, edge_type, device, max_nodes=2048):
    """Create small relabeled tensors for safe temporal dry-runs."""
    import torch
    from src.temporal_data import make_small_temporal_subgraph

    small = make_small_temporal_subgraph(
        {
            **artifacts,
            "edge_index_undirected": artifacts[f"edge_index_{edge_type}"],
        },
        max_nodes=max_nodes,
    )
    return {
        "x": torch.tensor(small["X"], dtype=torch.float32, device=device),
        "y": torch.tensor(small["y"], dtype=torch.long, device=device),
        "edge_index": torch.tensor(small["edge_index_undirected"], dtype=torch.long, device=device),
        "timesteps": torch.tensor(small["timesteps"], dtype=torch.long, device=device),
        "train_mask": torch.tensor(small["train_mask"], dtype=torch.bool, device=device),
    }


def forward_temporal(
    model,
    model_name,
    tensors,
    max_timesteps=None,
    snapshot_mode="cumulative",
    detach_memory_each_step=False,
    temporal_grad_mode="detach",
):
    """Run the appropriate temporal forward pass."""
    if model_name == "temporal_gcn_gru":
        return model.forward_full_sequence(
            tensors["x"],
            tensors["edge_index"],
            tensors["timesteps"],
            max_timesteps=max_timesteps,
            snapshot_mode=snapshot_mode,
            detach_memory_each_step=detach_memory_each_step,
            temporal_grad_mode=temporal_grad_mode,
        )
    return model(tensors["x"], tensors["edge_index"], tensors["timesteps"])


def evaluate_logits(logits, y, mask, criterion=None):
    """Evaluate logits over labeled nodes selected by mask."""
    import torch
    from src.evaluate import accuracy_score_torch, compute_roc_auc_safe, precision_recall_f1

    selected = mask.bool() & (y != -1)
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

    loss = criterion(logits[selected], y[selected]) if criterion else torch.nn.functional.cross_entropy(
        logits[selected], y[selected]
    )
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)
    precision, recall, f1 = precision_recall_f1(y, predictions, selected)
    return {
        "loss": float(loss.item()),
        "accuracy": accuracy_score_torch(y, predictions, selected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": compute_roc_auc_safe(y, probabilities[:, 1], selected),
        "num_nodes": int(selected.sum().item()),
        "illicit_count": int(((y == 1) & selected).sum().item()),
        "licit_count": int(((y == 0) & selected).sum().item()),
    }


def run_dry_run(args, artifacts, device, create_model):
    """Run safe temporal dry-run validation without training."""
    import torch

    tensors = small_tensors_from_artifacts(artifacts, args.edge_type, device)
    model = create_model(
        args.model,
        in_channels=tensors["x"].shape[1],
        hidden_channels=args.hidden_channels,
        out_channels=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_timesteps=int(tensors["timesteps"].max().item()) + 1,
    ).to(device)
    model.eval()
    with torch.no_grad():
        logits = forward_temporal(
            model,
            args.model,
            tensors,
            max_timesteps=args.max_timesteps,
            snapshot_mode=args.snapshot_mode,
            detach_memory_each_step=args.detach_memory_each_step,
            temporal_grad_mode=args.temporal_grad_mode,
        )
        selected = tensors["train_mask"] & (tensors["y"] != -1)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits[selected], tensors["y"][selected])

    print("Temporal dry run only: no optimizer, epochs, checkpoint, or training loop will run.")
    print(f"Model: {args.model}")
    print(f"Save name: {args.save_name if args.save_name else f'temporal_{args.model}'}")
    print(f"Device: {device}")
    print(f"Feature tensor: {tuple(tensors['x'].shape)}")
    print(f"Edge index tensor: {tuple(tensors['edge_index'].shape)}")
    print(f"Artifact directory: {artifacts.get('artifact_dir', args.artifact_dir)}")
    print(f"Snapshot mode: {args.snapshot_mode}")
    print(f"Temporal grad mode: {args.temporal_grad_mode}")
    print(f"Detach memory each step: {args.detach_memory_each_step}")
    print(f"Timesteps: {int(tensors['timesteps'].min())} to {int(tensors['timesteps'].max())}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Dry-run loss: {float(loss.item()):.6f}")
    print(model)


def train_one_epoch(args, model, tensors, optimizer, criterion):
    """Run one temporal training epoch."""
    model.train()
    optimizer.zero_grad()
    logits = forward_temporal(
        model,
        args.model,
        tensors,
        max_timesteps=args.max_timesteps,
        snapshot_mode=args.snapshot_mode,
        detach_memory_each_step=args.detach_memory_each_step,
        temporal_grad_mode=args.temporal_grad_mode,
    )
    train_mask = tensors["train_mask"] & (tensors["y"] != -1)
    loss = criterion(logits[train_mask], tensors["y"][train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item()), logits.detach()


def main():
    """Run manual temporal training or a safe dry run."""
    args = parse_args()
    if args.current_snapshot_training:
        args.snapshot_mode = "current"

    try:
        import torch
        from torch import nn

        from src.models.model_factory import create_model
        from src.training_utils import (
            compute_class_weights,
            get_device,
            load_graph_artifacts,
            save_json,
            save_training_curves,
            set_seed,
        )
    except ImportError as exc:
        print("Temporal GNN dependencies are not ready.")
        print("Dependency detail:", exc)
        sys.exit(1)

    set_seed(args.seed)
    device = get_device(args.device)
    artifacts = load_graph_artifacts(args.artifact_dir)

    if args.dry_run:
        run_dry_run(args, artifacts, device, create_model)
        return

    tensors = tensors_from_artifacts(artifacts, args.edge_type, device)
    model = create_model(
        args.model,
        in_channels=tensors["x"].shape[1],
        hidden_channels=args.hidden_channels,
        out_channels=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_timesteps=int(tensors["timesteps"].max().item()) + 1,
    ).to(device)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = args.save_name if args.save_name else f"temporal_{args.model}"

    class_weights = compute_class_weights(tensors["y"], tensors["train_mask"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_f1 = -1.0
    best_epoch = 0
    wait = 0
    log_rows = []
    checkpoint_path = CHECKPOINT_DIR / f"{output_prefix}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_logits = train_one_epoch(args, model, tensors, optimizer, criterion)
        model.eval()
        with torch.no_grad():
            logits = forward_temporal(
                model,
                args.model,
                tensors,
                max_timesteps=args.max_timesteps,
                snapshot_mode=args.snapshot_mode,
                detach_memory_each_step=args.detach_memory_each_step,
                temporal_grad_mode=args.temporal_grad_mode,
            )
        train_metrics = evaluate_logits(train_logits, tensors["y"], tensors["train_mask"], criterion)
        val_metrics = evaluate_logits(logits, tensors["y"], tensors["val_mask"], criterion)
        log_rows.append(
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = forward_temporal(
            model,
            args.model,
        tensors,
        max_timesteps=args.max_timesteps,
        snapshot_mode=args.snapshot_mode,
        detach_memory_each_step=args.detach_memory_each_step,
        temporal_grad_mode=args.temporal_grad_mode,
    )
    test_metrics = evaluate_logits(logits, tensors["y"], tensors["test_mask"], criterion)

    log_path = TABLES_DIR / f"{output_prefix}_training_log.csv"
    metrics_path = TABLES_DIR / f"{output_prefix}_metrics.json"
    loss_plot_path = PLOTS_DIR / f"{output_prefix}_loss_curve.png"
    f1_plot_path = PLOTS_DIR / f"{output_prefix}_f1_curve.png"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    save_json(
        metrics_path,
        {
            "model": args.model,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "test_metrics": test_metrics,
            "checkpoint_path": str(checkpoint_path),
        },
    )
    save_training_curves(log_rows, loss_plot_path, f1_plot_path)

    print("\nFinal temporal test metrics")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
