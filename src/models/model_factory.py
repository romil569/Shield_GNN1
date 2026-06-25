"""Factory for creating supported baseline GNN models."""

from src.models.gat import GATModel
from src.models.gcn import GCNModel
from src.models.graphsage import GraphSAGEModel
from src.models.temporal_gnn import TemporalGCNGRUModel, TimeAwareGCNModel


def create_model(
    model_name,
    in_channels,
    hidden_channels,
    out_channels,
    num_layers,
    dropout,
    heads=None,
    max_timesteps=None,
):
    """Create a baseline model by name."""
    normalized_name = model_name.lower().strip()

    if normalized_name == "gcn":
        return GCNModel(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if normalized_name == "gat":
        return GATModel(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            heads=heads if heads is not None else 4,
            dropout=dropout,
        )
    if normalized_name == "graphsage":
        return GraphSAGEModel(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if normalized_name == "temporal_gcn_gru":
        return TemporalGCNGRUModel(
            in_channels,
            hidden_channels,
            out_channels,
            num_gcn_layers=num_layers,
            dropout=dropout,
        )
    if normalized_name == "time_aware_gcn":
        return TimeAwareGCNModel(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers=num_layers,
            dropout=dropout,
            max_timesteps=max_timesteps if max_timesteps is not None else 64,
        )

    supported = "gcn, gat, graphsage, temporal_gcn_gru, time_aware_gcn"
    raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}.")
