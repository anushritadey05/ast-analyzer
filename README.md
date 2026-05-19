# AST Algorithmic Analyzer (MVP)

This repository contains the initial **parser + graph builder** slice for the AST Algorithmic Analyzer.

## What's included
- **Tree-sitter parsing** for Python 3.12
- **AST → Graph** conversion (parent/child edges + minimal control/data flow heuristics)
- A small **demo script** to show graph construction

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m ast_analyzer.demo
```

## Notes
This is intentionally minimal and designed to be extended into:
- richer control/data-flow extraction
- PyTorch Geometric conversion
- model training + inference

