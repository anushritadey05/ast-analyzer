from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphSAGE, global_mean_pool


class BaselineGNN(nn.Module):
    def __init__(self, num_node_types: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(num_node_types, hidden_dim)

        # Choose GraphSAGE-style message passing
        self.conv1 = GraphSAGE(hidden_dim, hidden_dim)
        self.conv2 = GraphSAGE(hidden_dim, hidden_dim)

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.emb(x.squeeze(-1))
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.lin1(x).relu()
        return self.lin2(x)
