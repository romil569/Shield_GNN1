"""Dry-run baseline model setup without training."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Create each baseline model and validate graph tensor shapes without training."""
    try:
        import torch
    except ImportError:
        print("Torch is not installed. Install requirements-gnn.txt before dry-running models.")
        return

    try:
        from src.models.model_factory import create_model
        from src.training_utils import get_device, load_graph_artifacts
    except ImportError as exc:
        print(f"Could not import GNN model dependencies: {exc}")
        return

    device = get_device("auto")
    artifacts = load_graph_artifacts()
    x = torch.tensor(artifacts["X"], dtype=torch.float32, device=device)
    edge_index = torch.tensor(
        artifacts["edge_index_undirected"], dtype=torch.long, device=device
    )

    print("Dry run only: no optimizer, epochs, or training loop will run.")
    print(f"Device: {device}")
    print(f"Feature tensor shape: {tuple(x.shape)}")
    print(f"Edge index shape: {tuple(edge_index.shape)}")

    for model_name in ["gcn", "gat", "graphsage"]:
        model = create_model(
            model_name,
            in_channels=x.shape[1],
            hidden_channels=32 if model_name == "gat" else 64,
            out_channels=2,
            num_layers=2,
            dropout=0.5,
            heads=4,
        ).to(device)
        print(f"{model_name}: CREATED")
        print(model)


if __name__ == "__main__":
    main()
