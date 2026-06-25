"""Temporal GNN baselines for SHIELD-GNN research.

These models are comparison baselines for temporal graph learning. They are not
the final SHIELD-GNN defense, attack simulation, edge pruning, or temporal
consistency defense implementation.
"""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError as exc:
    raise ImportError(
        "Temporal GNN models require torch-geometric. Install GNN dependencies first."
    ) from exc


class TemporalGCNGRUModel(nn.Module):
    """Temporal baseline that combines GCN graph encoding with GRU node memory."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_gcn_layers=2, dropout=0.5):
        """Initialize a TemporalGCN-GRU model."""
        super().__init__()
        self._validate_parameters(in_channels, hidden_channels, out_channels, num_gcn_layers, dropout)
        self.hidden_channels = int(hidden_channels)
        self.dropout = float(dropout)
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_gcn_layers)]
        )
        self.gru_cell = nn.GRUCell(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    @staticmethod
    def _validate_parameters(in_channels, hidden_channels, out_channels, num_gcn_layers, dropout):
        """Validate constructor parameters."""
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if num_gcn_layers < 1:
            raise ValueError("num_gcn_layers must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

    def _snapshot_edge_index(self, edge_index, active_mask):
        """Filter edges so both endpoints are active in the current snapshot."""
        edge_mask = active_mask[edge_index[0]] & active_mask[edge_index[1]]
        return edge_index[:, edge_mask]

    def _snapshot_subgraph(self, edge_index, active_mask, num_nodes):
        """Return active nodes and relabeled snapshot edges for memory-safe GCNConv."""
        active_nodes = torch.nonzero(active_mask, as_tuple=False).view(-1)
        if active_nodes.numel() == 0:
            return active_nodes, edge_index.new_empty((2, 0))

        snapshot_edges = self._snapshot_edge_index(edge_index, active_mask)
        if snapshot_edges.numel() == 0:
            return active_nodes, edge_index.new_empty((2, 0))

        node_lookup = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
        node_lookup[active_nodes] = torch.arange(
            active_nodes.numel(), dtype=torch.long, device=edge_index.device
        )
        local_edges = node_lookup[snapshot_edges]
        valid_edges = (local_edges[0] >= 0) & (local_edges[1] >= 0)
        return active_nodes, local_edges[:, valid_edges]

    def forward_single_graph(self, x, edge_index):
        """Run a non-temporal graph pass for quick validation or ablation."""
        h = self.input_projection(x)
        for conv in self.gcn_layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)

    def forward_full_sequence(
        self,
        x,
        edge_index,
        timesteps,
        selected_timesteps=None,
        max_timesteps=None,
        dry_run_nodes=None,
        snapshot_mode="cumulative",
        detach_memory_each_step=False,
        temporal_grad_mode="full",
        return_all_logits=False,
    ):
        """Run temporal message passing and return logits for all nodes.

        The model keeps global node memory with shape [num_nodes, hidden_channels].
        Cumulative snapshots update nodes with timestep <= t; current snapshots
        update nodes with timestep == t.
        """
        if snapshot_mode not in {"cumulative", "current"}:
            raise ValueError("snapshot_mode must be 'cumulative' or 'current'.")
        if temporal_grad_mode not in {"full", "detach", "final_only"}:
            raise ValueError("temporal_grad_mode must be 'full', 'detach', or 'final_only'.")

        num_nodes = x.shape[0]
        device = x.device
        projected = self.input_projection(x)
        memory = torch.zeros(num_nodes, self.hidden_channels, device=device, dtype=projected.dtype)
        unique_timesteps = torch.unique(timesteps).detach().cpu().tolist()
        unique_timesteps = sorted(int(t) for t in unique_timesteps)
        if selected_timesteps is not None:
            selected = {int(t) for t in selected_timesteps}
            unique_timesteps = [t for t in unique_timesteps if t in selected]
        if max_timesteps is not None:
            unique_timesteps = unique_timesteps[: int(max_timesteps)]

        logits_by_timestep = [] if return_all_logits else None
        last_step_index = len(unique_timesteps) - 1

        for step_index, timestep in enumerate(unique_timesteps):
            if snapshot_mode == "cumulative":
                active_mask = timesteps <= timestep
            else:
                active_mask = timesteps == timestep
            if dry_run_nodes is not None:
                dry_mask = torch.zeros_like(active_mask, dtype=torch.bool)
                dry_mask[: min(int(dry_run_nodes), num_nodes)] = True
                active_mask = active_mask & dry_mask
            if int(active_mask.sum()) == 0:
                continue

            active_nodes, snapshot_edges = self._snapshot_subgraph(edge_index, active_mask, num_nodes)
            if snapshot_edges.numel() == 0:
                continue

            use_no_grad = temporal_grad_mode == "final_only" and step_index < last_step_index
            grad_context = torch.no_grad() if use_no_grad else nullcontext()
            with grad_context:
                encoded = projected[active_nodes]
                for conv in self.gcn_layers:
                    encoded = conv(encoded, snapshot_edges)
                    encoded = F.relu(encoded)
                    encoded = F.dropout(encoded, p=self.dropout, training=self.training)

                updated_active_memory = self.gru_cell(encoded, memory[active_nodes])
                memory = memory.index_copy(0, active_nodes, updated_active_memory)

            if detach_memory_each_step or temporal_grad_mode in {"detach", "final_only"}:
                memory = memory.detach()

            if return_all_logits:
                logits_by_timestep.append(self.classifier(memory))

        if return_all_logits:
            return logits_by_timestep
        return self.classifier(memory)

    def forward(self, x, edge_index, timesteps):
        """Run the full cumulative temporal sequence."""
        return self.forward_full_sequence(x, edge_index, timesteps)


class TimeAwareGCNModel(nn.Module):
    """Lightweight temporal baseline using timestep embeddings with GCN layers."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        max_timesteps=64,
    ):
        """Initialize a time-aware GCN baseline."""
        super().__init__()
        self._validate_parameters(in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.dropout = float(dropout)
        self.max_timesteps = int(max_timesteps)
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.time_embedding = nn.Embedding(self.max_timesteps + 1, hidden_channels)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)

    @staticmethod
    def _validate_parameters(in_channels, hidden_channels, out_channels, num_layers, dropout):
        """Validate constructor parameters."""
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

    def forward(self, x, edge_index, timesteps):
        """Return logits after adding learned timestep embeddings."""
        clipped_timesteps = torch.clamp(timesteps.long(), min=0, max=self.max_timesteps)
        h = self.input_projection(x) + self.time_embedding(clipped_timesteps)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)
