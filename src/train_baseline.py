"""Manual baseline GNN training entry point for GCN, GAT, and GraphSAGE."""

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
torch = None
nn = None
evaluate_node_classifier = None
create_model = None
compute_class_weights = None
get_device = None
load_graph_artifacts = None
save_json = None
save_training_curves = None
set_seed = None


def parse_args():
    """Parse command-line arguments for manual baseline training."""
    parser = argparse.ArgumentParser(description="Train baseline GNN models manually.")
    parser.add_argument("--model", choices=["gcn", "gat", "graphsage"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--edge-type", choices=["directed", "undirected"], default="undirected")
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "graph_artifacts"),
        help="Graph artifact directory. Defaults to the 100k subset artifacts.",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heads", type=int, default=4, help="Attention heads for GAT only.")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training.")
    parser.add_argument(
        "--save-name",
        default=None,
        help="Optional output filename prefix for checkpoints, logs, metrics, and plots.",
    )
    return parser.parse_args()


def tensors_from_artifacts(artifacts, edge_type, device):
    """Convert NumPy graph artifacts into torch tensors."""
    edge_key = f"edge_index_{edge_type}"
    tensors = {
        "x": torch.tensor(artifacts["X"], dtype=torch.float32, device=device),
        "y": torch.tensor(artifacts["y"], dtype=torch.long, device=device),
        "edge_index": torch.tensor(artifacts[edge_key], dtype=torch.long, device=device),
        "train_mask": torch.tensor(artifacts["train_mask"], dtype=torch.bool, device=device),
        "val_mask": torch.tensor(artifacts["val_mask"], dtype=torch.bool, device=device),
        "test_mask": torch.tensor(artifacts["test_mask"], dtype=torch.bool, device=device),
    }
    return tensors


def print_dry_run_report(args, artifacts, tensors, model):
    """Print dry-run information and exit without training."""
    print("Dry run only: no optimizer, epochs, checkpoint, or training loop will run.")
    print(f"Model: {args.model}")
    print(f"Save name: {args.save_name if args.save_name else f'baseline_{args.model}'}")
    print(f"Device: {tensors['x'].device}")
    print(f"Feature tensor: {tuple(tensors['x'].shape)}")
    print(f"Label tensor: {tuple(tensors['y'].shape)}")
    print(f"Edge index tensor: {tuple(tensors['edge_index'].shape)} ({args.edge_type})")
    print(f"Artifact directory: {artifacts.get('artifact_dir', args.artifact_dir)}")
    print(f"Train/val/test masks: {int(tensors['train_mask'].sum())}/"
          f"{int(tensors['val_mask'].sum())}/{int(tensors['test_mask'].sum())}")
    print(model)
    print("Graph summary:", artifacts["graph_summary"])


def train_one_epoch(model, tensors, optimizer, criterion):
    """Run one training epoch."""
    model.train()
    optimizer.zero_grad()
    logits = model(tensors["x"], tensors["edge_index"])
    train_mask = tensors["train_mask"] & (tensors["y"] != -1)
    loss = criterion(logits[train_mask], tensors["y"][train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


def main():
    """Run manual training or a safe dry run."""
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")

    global torch
    global nn
    global evaluate_node_classifier
    global create_model
    global compute_class_weights
    global get_device
    global load_graph_artifacts
    global save_json
    global save_training_curves
    global set_seed

    try:
        import torch as torch_module
        from torch import nn as nn_module

        from src.evaluate import evaluate_node_classifier as evaluate_node_classifier_func
        from src.models.model_factory import create_model as create_model_func
        from src.training_utils import (
            compute_class_weights as compute_class_weights_func,
            get_device as get_device_func,
            load_graph_artifacts as load_graph_artifacts_func,
            save_json as save_json_func,
            save_training_curves as save_training_curves_func,
            set_seed as set_seed_func,
        )
    except ImportError as exc:
        print("Baseline GNN dependencies are not ready.")
        print("Install compatible PyTorch and PyTorch Geometric packages first.")
        print("Dependency detail:", exc)
        if args.dry_run:
            print("Dry run stopped safely before model creation. No training was started.")
        sys.exit(1)

    torch = torch_module
    nn = nn_module
    evaluate_node_classifier = evaluate_node_classifier_func
    create_model = create_model_func
    compute_class_weights = compute_class_weights_func
    get_device = get_device_func
    load_graph_artifacts = load_graph_artifacts_func
    save_json = save_json_func
    save_training_curves = save_training_curves_func
    set_seed = set_seed_func

    set_seed(args.seed)
    device = get_device(args.device)
    artifacts = load_graph_artifacts(args.artifact_dir)
    tensors = tensors_from_artifacts(artifacts, args.edge_type, device)

    model = create_model(
        args.model,
        in_channels=tensors["x"].shape[1],
        hidden_channels=args.hidden_channels,
        out_channels=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
        heads=args.heads,
    ).to(device)

    if args.dry_run:
        print_dry_run_report(args, artifacts, tensors, model)
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = args.save_name if args.save_name else f"baseline_{args.model}"

    class_weights = compute_class_weights(tensors["y"], tensors["train_mask"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    log_rows = []
    checkpoint_path = CHECKPOINT_DIR / f"{output_prefix}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, tensors, optimizer, criterion)
        train_metrics = evaluate_node_classifier(
            model, tensors["x"], tensors["edge_index"], tensors["y"], tensors["train_mask"], criterion
        )
        val_metrics = evaluate_node_classifier(
            model, tensors["x"], tensors["edge_index"], tensors["y"], tensors["val_mask"], criterion
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        log_rows.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
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
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best validation F1: {best_val_f1:.4f}")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_node_classifier(
        model, tensors["x"], tensors["edge_index"], tensors["y"], tensors["test_mask"], criterion
    )

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

    print("\nFinal test metrics")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved training log: {log_path}")
    print(f"Saved metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
