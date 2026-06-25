"""Graph Attention Network baseline model for node classification."""

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch_geometric.nn import GATConv
except ImportError as exc:
    raise ImportError(
        "GATModel requires torch-geometric. Install requirements-gnn.txt before training."
    ) from exc


class GATModel(nn.Module):
    """A multi-layer GAT with careful hidden-dimension handling for attention heads."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        heads=4,
        dropout=0.5,
    ):
        """Initialize the GAT baseline."""
        super().__init__()
        self._validate_parameters(
            in_channels, hidden_channels, out_channels, num_layers, heads, dropout
        )
        self.dropout = float(dropout)

        if num_layers == 1:
            self.convs = nn.ModuleList(
                [GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout)]
            )
            return

        convs = [
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                concat=True,
                dropout=dropout,
            )
        ]
        current_channels = hidden_channels * heads
        for _ in range(num_layers - 2):
            convs.append(
                GATConv(
                    current_channels,
                    hidden_channels,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            current_channels = hidden_channels * heads
        convs.append(
            GATConv(
                current_channels,
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        )
        self.convs = nn.ModuleList(convs)

    @staticmethod
    def _validate_parameters(in_channels, hidden_channels, out_channels, num_layers, heads, dropout):
        """Validate model hyperparameters before layers are created."""
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if heads is None or heads < 1:
            raise ValueError("heads must be at least 1 for GAT.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

    def forward(self, x, edge_index):
        """Run a forward pass and return logits."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)
