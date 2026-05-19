from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import networkx as nx


EDGE_TYPE_MAP = {
    "ast": 0,
    "control": 1,
    "data": 2,
}


@dataclass
class PyGConverter:
    """Converts a NetworkX AST graph to a PyTorch Geometric Data object.

    - Node features: single integer type id (dtype long)
    - Edge features: integer edge type id (dtype long)
    """

    node_type_vocab: Dict[str, int] = field(default_factory=dict)

    def build_vocab(self, graph: nx.MultiDiGraph) -> None:
        for _, data in graph.nodes(data=True):
            ntype = data.get("type", "unknown")
            if ntype not in self.node_type_vocab:
                self.node_type_vocab[ntype] = len(self.node_type_vocab)

    def to_pyg(self, graph: nx.MultiDiGraph):
        try:
            import torch
            from torch_geometric.data import Data
        except Exception as exc:  # pragma: no cover - import guarded
            raise RuntimeError(
                "PyG conversion requires extra dependencies. "
                "Install with: pip install -e '.[pyg]'"
            ) from exc

        if not self.node_type_vocab:
            self.build_vocab(graph)

        # Nodes
        num_nodes = graph.number_of_nodes()
        x = torch.zeros((num_nodes, 1), dtype=torch.long)
        for nid, data in graph.nodes(data=True):
            ntype = data.get("type", "unknown")
            x[nid, 0] = self.node_type_vocab.get(ntype, 0)

        # Edges
        edges: List[Tuple[int, int]] = []
        edge_types: List[int] = []
        for u, v, data in graph.edges(data=True):
            edges.append((u, v))
            edge_types.append(EDGE_TYPE_MAP.get(data.get("type"), 0))

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_types, dtype=torch.long).view(-1, 1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
