# AST Algorithmic Analyzer (MVP)

This repository contains the initial **parser + graph builder** slice for the AST Algorithmic Analyzer.

## What's included
- **Tree-sitter parsing** for Python 3.12
- **AST → Graph** conversion (parent/child edges + minimal control/data flow heuristics)
- Optional **PyTorch Geometric conversion** helper
- **Synthetic dataset generator** (JSONL)
- **Baseline GNN training loop** (GraphSAGE + global pooling)
- A small **demo script** to show graph construction

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m ast_analyzer.demo
```

## Generate synthetic dataset

```bash
ast-analyzer-generate-dataset --count 200 --out data.jsonl
```

## Train baseline GNN

```bash
pip install -e ".[pyg]"
ast-analyzer-train-gnn --count 500 --epochs 10 --batch-size 16
```

## Optional: PyG conversion

```bash
pip install -e ".[pyg]"
```

Then in Python:

```python
from ast_analyzer.pyg_converter import PyGConverter

converter = PyGConverter()
pyg_data = converter.to_pyg(graph)
```

## Notes
This is intentionally minimal and designed to be extended into:
- richer control/data-flow extraction
- PyTorch Geometric conversion
- model training + inference

