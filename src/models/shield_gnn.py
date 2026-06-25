"""Final integrated SHIELD-GNN model.

SHIELD-GNN combines temporal graph learning, optional temporal edge encoding,
robust edge-pruning weights, and node-level fraud classification. This module
defines the model only; it does not train or run experiments by itself.
"""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError as exc:
    raise ImportError("SHIELDGNNModel requires torch-geometric.") from exc


class TemporalEdgeEncoder(nn.Module):
    """Lightweight encoder for temporal edge reliability information."""

    def __init__(self, hidden_channels, use_edge_weight=True):
        """Initialize the temporal edge encoder."""
        super().__init__()
        self.use_edge_weight = use_edge_weight
        self.edge_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, edge_index, timesteps=None, edge_weight=None):
        """Return edge weights adjusted by temporal gap and optional defense weight."""
        num_edges = edge_index.shape[1]
        device = edge_index.device
        if num_edges == 0:
            return torch.empty(0, dtype=torch.float32, device=device)
        if timesteps is None:
            temporal_gap = torch.zeros(num_edges, dtype=torch.float32, device=device)
        else:
            temporal_gap = (timesteps[edge_index[0]] - timesteps[edge_index[1]]).abs().float()
            temporal_gap = temporal_gap / temporal_gap.max().clamp_min(1.0)
        if edge_weight is None or not self.use_edge_weight:
            base_weight = torch.ones(num_edges, dtype=torch.float32, device=device)
        else:
            base_weight = edge_weight.float().to(device)
        encoded = self.edge_mlp(torch.stack([temporal_gap, base_weight], dim=1)).view(-1)
        return (base_weight * encoded).clamp_min(1e-6)


class SHIELDGNNModel(nn.Module):
    """Final SHIELD-GNN model for integrated robust temporal graph learning."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        backbone="temporal_gcn_gru",
        num_layers=2,
        dropout=0.5,
        use_temporal_edge_encoding=True,
        use_edge_weight=True,
        max_timesteps=None,
        snapshot_mode="current",
        detach_memory_each_step=True,
        temporal_grad_mode="detach",
    ):
        """Initialize SHIELD-GNN."""
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels, hidden_channels, and out_channels must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")
        if backbone not in {"temporal_gcn_gru", "time_aware_gcn"}:
            raise ValueError("backbone must be temporal_gcn_gru or time_aware_gcn.")
        if snapshot_mode not in {"current", "cumulative"}:
            raise ValueError("snapshot_mode must be current or cumulative.")
        if temporal_grad_mode not in {"full", "detach", "final_only"}:
            raise ValueError("temporal_grad_mode must be full, detach, or final_only.")

        self.backbone = backbone
        self.dropout = float(dropout)
        self.use_temporal_edge_encoding = use_temporal_edge_encoding
        self.use_edge_weight = use_edge_weight
        self.max_timesteps = max_timesteps or 64
        self.snapshot_mode = snapshot_mode
        self.detach_memory_each_step = bool(detach_memory_each_step)
        self.temporal_grad_mode = temporal_grad_mode
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.time_embedding = nn.Embedding(self.max_timesteps + 1, hidden_channels)
        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.gru_cell = nn.GRUCell(hidden_channels, hidden_channels)
        self.edge_encoder = TemporalEdgeEncoder(hidden_channels, use_edge_weight=use_edge_weight)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def _encoded_edge_weight(self, edge_index, timesteps, edge_weight):
        """Create edge weights for GCN layers."""
        if self.use_temporal_edge_encoding:
            return self.edge_encoder(edge_index, timesteps=timesteps, edge_weight=edge_weight)
        if edge_weight is not None and self.use_edge_weight:
            return edge_weight.float().to(edge_index.device)
        return None

    def _gcn_stack(self, h, edge_index, encoded_edge_weight):
        """Run GCN layers with optional edge weights."""
        for conv in self.gcn_layers:
            h = conv(h, edge_index, edge_weight=encoded_edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def _snapshot_subgraph(self, edge_index, edge_weight, active, num_nodes):
        """Return active nodes, local edges, and matching edge weights."""
        active_nodes = torch.nonzero(active, as_tuple=False).view(-1)
        if active_nodes.numel() == 0:
            return active_nodes, edge_index.new_empty((2, 0)), None
        edge_mask = active[edge_index[0]] & active[edge_index[1]]
        snapshot_edges = edge_index[:, edge_mask]
        if snapshot_edges.numel() == 0:
            return active_nodes, edge_index.new_empty((2, 0)), None
        lookup = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
        lookup[active_nodes] = torch.arange(active_nodes.numel(), dtype=torch.long, device=edge_index.device)
        local_edges = lookup[snapshot_edges]
        valid = (local_edges[0] >= 0) & (local_edges[1] >= 0)
        local_weight = edge_weight[edge_mask][valid] if edge_weight is not None else None
        return active_nodes, local_edges[:, valid], local_weight

    def forward(
        self,
        x,
        edge_index,
        timesteps=None,
        edge_weight=None,
        mode="clean",
        return_embeddings=False,
    ):
        """Run SHIELD-GNN and return logits, optionally with embeddings."""
        if timesteps is None:
            timesteps = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            timesteps = timesteps.long().to(x.device)
        edge_index = edge_index.long().to(x.device)
        edge_weight = edge_weight.to(x.device) if edge_weight is not None else None
        encoded_edge_weight = self._encoded_edge_weight(edge_index, timesteps, edge_weight)
        h = self.input_projection(x)

        if self.backbone == "time_aware_gcn":
            clipped = torch.clamp(timesteps, min=0, max=self.max_timesteps)
            h = h + self.time_embedding(clipped)
            embeddings = self._gcn_stack(h, edge_index, encoded_edge_weight)
        else:
            memory = torch.zeros_like(h)
            unique_timesteps = sorted(int(t) for t in torch.unique(timesteps).detach().cpu().tolist())
            last_step_index = len(unique_timesteps) - 1
            for step_index, timestep in enumerate(unique_timesteps):
                active = timesteps <= timestep if self.snapshot_mode == "cumulative" else timesteps == timestep
                if int(active.sum()) == 0:
                    continue
                active_nodes, sub_edges, sub_weight = self._snapshot_subgraph(
                    edge_index, encoded_edge_weight, active, x.shape[0]
                )
                if sub_edges.numel() == 0:
                    continue
                use_no_grad = self.temporal_grad_mode == "final_only" and step_index < last_step_index
                grad_context = torch.no_grad() if use_no_grad else nullcontext()
                with grad_context:
                    encoded = self._gcn_stack(h[active_nodes], sub_edges, sub_weight)
                    updated = self.gru_cell(encoded, memory[active_nodes])
                    memory = memory.index_copy(0, active_nodes, updated)
                if self.detach_memory_each_step or self.temporal_grad_mode in {"detach", "final_only"}:
                    memory = memory.detach()
            embeddings = memory

        logits = self.classifier(embeddings)
        if return_embeddings:
            return logits, embeddings
        return logits


# Backward-compatible alias for early project placeholders.
SHIELDGNN = SHIELDGNNModel
