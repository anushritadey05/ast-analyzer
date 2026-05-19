from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset

from .dataset import BIG_O_CLASSES, Sample
from .graph_builder import ASTGraphBuilder
from .parser import CodeParser
from .pyg_converter import PyGConverter


@dataclass
class GraphSample:
    data: "torch_geometric.data.Data"
    label: int


class BigODataset(Dataset):
    """Turns code samples into PyG graphs with labels."""

    def __init__(self, samples: Iterable[Sample]) -> None:
        self.samples = list(samples)
        self.label_to_idx: Dict[str, int] = {c: i for i, c in enumerate(BIG_O_CLASSES)}

        self.parser = CodeParser("python")
        self.builder = ASTGraphBuilder()
        self.converter = PyGConverter()

        self._graphs: List[GraphSample] = []
        self._build_graphs()

    def _build_graphs(self) -> None:
        for sample in self.samples:
            parsed = self.parser.parse(sample.code)
            graph = self.builder.build(parsed).graph
            pyg = self.converter.to_pyg(graph)
            label = self.label_to_idx[sample.label]
            self._graphs.append(GraphSample(data=pyg, label=label))

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> GraphSample:
        return self._graphs[idx]
