from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tree_sitter import Node, Tree
from tree_sitter_languages import get_parser


@dataclass
class ParsedCode:
    language: str
    tree: Tree
    root: Node
    source_bytes: bytes


class CodeParser:
    """Thin wrapper around Tree-sitter for parsing source code."""

    def __init__(self, language: str = "python") -> None:
        self.language = language
        self._parser = get_parser(language)

    def parse(self, source_code: str) -> ParsedCode:
        source_bytes = source_code.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        return ParsedCode(
            language=self.language,
            tree=tree,
            root=tree.root_node,
            source_bytes=source_bytes,
        )


def walk(node: Node) -> Iterable[Node]:
    """Pre-order traversal of a Tree-sitter node tree."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        # reversed for natural left-to-right traversal
        for child in reversed(current.children):
            stack.append(child)
