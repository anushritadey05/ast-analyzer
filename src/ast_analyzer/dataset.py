from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Iterable, List


BIG_O_CLASSES = ["O(1)", "O(log N)", "O(N)", "O(N log N)", "O(N^2)"]


@dataclass
class Sample:
    code: str
    label: str
    lang: str = "python"


class SyntheticDatasetGenerator:
    """Generate simple synthetic functions with known Big-O labels.

    This is intentionally small and heuristic-based. We'll expand it later.
    """

    def __init__(self, seed: int = 1337) -> None:
        self._rng = random.Random(seed)

    def generate(self, n: int) -> List[Sample]:
        samples: List[Sample] = []
        for _ in range(n):
            label = self._rng.choice(BIG_O_CLASSES)
            code = self._code_for_label(label)
            samples.append(Sample(code=code, label=label))
        return samples

    def _code_for_label(self, label: str) -> str:
        if label == "O(1)":
            return """

def f(n):
    x = 42
    y = x + 1
    return y
""".strip()

        if label == "O(log N)":
            return """

def f(n):
    x = n
    while x > 1:
        x = x // 2
    return x
""".strip()

        if label == "O(N)":
            return """

def f(n):
    s = 0
    for i in range(n):
        s += i
    return s
""".strip()

        if label == "O(N log N)":
            return """

def f(n):
    s = 0
    for i in range(n):
        x = i
        while x > 1:
            x = x // 2
            s += x
    return s
""".strip()

        # O(N^2)
        return """

def f(n):
    s = 0
    for i in range(n):
        for j in range(n):
            s += i * j
    return s
""".strip()


def write_jsonl(samples: Iterable[Sample], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.__dict__) + "\n")
