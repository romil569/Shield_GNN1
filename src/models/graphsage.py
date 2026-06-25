"""GraphSAGE baseline model for node classification."""

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch_geometric.nn import SAGEConv
except ImportError as exc:
    raise ImportError(
        "GraphSAGEModel requires torch-geometric. Install requirements-gnn.txt before training."
    ) from exc


class GraphSAGEModel(nn.Module):
    """A beginner-readable multi-layer GraphSAGE model that returns class logits."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        """Initialize the GraphSAGE baseline."""
        super().__init__()
        self._validate_parameters(in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.dropout = float(dropout)

        if num_layers == 1:
            self.convs = nn.ModuleList([SAGEConv(in_channels, out_channels)])
        else:
            convs = [SAGEConv(in_channels, hidden_channels)]
            for _ in range(num_layers - 2):
                convs.append(SAGEConv(hidden_channels, hidden_channels))
            convs.append(SAGEConv(hidden_channels, out_channels))
            self.convs = nn.ModuleList(convs)

    @staticmethod
    def _validate_parameters(in_channels, hidden_channels, out_channels, num_layers, dropout):
        """Validate model hyperparameters before layers are created."""
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

    def forward(self, x, edge_index):
        """Run a forward pass and return logits."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)
