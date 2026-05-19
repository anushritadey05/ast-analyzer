from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import networkx as nx
from tree_sitter import Node

from .parser import ParsedCode, walk


@dataclass
class GraphBuildResult:
    graph: nx.MultiDiGraph
    node_map: Dict[int, int]  # maps Tree-sitter Node.id to graph node id


CONTROL_FLOW_NODE_TYPES = {
    "if_statement",
    "for_statement",
    "while_statement",
    "function_definition",
    "return_statement",
    "break_statement",
    "continue_statement",
}

ASSIGNMENT_NODE_TYPES = {
    "assignment",
    "augmented_assignment",
}

IDENTIFIER_NODE_TYPES = {"identifier"}


class ASTGraphBuilder:
    """Builds a rich graph from a Tree-sitter AST.

    - Parent/child edges
    - Minimal control-flow edges (heuristics)
    - Minimal data-flow edges (last assignment → usage)
    """

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()
        self._node_map: Dict[int, int] = {}
        self._next_id = 0

    def build(self, parsed: ParsedCode) -> GraphBuildResult:
        self._graph.clear()
        self._node_map.clear()
        self._next_id = 0

        # Build nodes + parent/child edges
        for node in walk(parsed.root):
            g_id = self._add_node(node, parsed)
            if node.parent is not None:
                parent_id = self._get_node_id(node.parent, parsed)
                self._graph.add_edge(parent_id, g_id, type="ast")

        # Add control-flow edges (heuristic)
        self._add_control_flow_edges(parsed)

        # Add data-flow edges (heuristic: last assignment → usage)
        self._add_data_flow_edges(parsed)

        return GraphBuildResult(graph=self._graph, node_map=self._node_map)

    def _add_node(self, node: Node, parsed: ParsedCode) -> int:
        if node.id in self._node_map:
            return self._node_map[node.id]

        g_id = self._next_id
        self._next_id += 1
        self._node_map[node.id] = g_id

        text = parsed.source_bytes[node.start_byte : node.end_byte].decode("utf-8", "ignore")

        self._graph.add_node(
            g_id,
            type=node.type,
            text=text,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_point=node.start_point,
            end_point=node.end_point,
        )
        return g_id

    def _get_node_id(self, node: Node, parsed: ParsedCode) -> int:
        return self._node_map.get(node.id) or self._add_node(node, parsed)

    def _add_control_flow_edges(self, parsed: ParsedCode) -> None:
        for node in walk(parsed.root):
            if node.type not in CONTROL_FLOW_NODE_TYPES:
                continue

            node_id = self._get_node_id(node, parsed)

            # Heuristic: connect condition → body/alternative for if/while/for
            if node.type == "if_statement":
                condition = node.child_by_field_name("condition")
                consequence = node.child_by_field_name("consequence")
                alternative = node.child_by_field_name("alternative")
                self._connect_if_present(node_id, condition, parsed, "control")
                self._connect_if_present(node_id, consequence, parsed, "control")
                self._connect_if_present(node_id, alternative, parsed, "control")

            elif node.type in {"for_statement", "while_statement"}:
                body = node.child_by_field_name("body")
                condition = node.child_by_field_name("condition")
                self._connect_if_present(node_id, condition, parsed, "control")
                self._connect_if_present(node_id, body, parsed, "control")

            elif node.type == "function_definition":
                body = node.child_by_field_name("body")
                self._connect_if_present(node_id, body, parsed, "control")

    def _add_data_flow_edges(self, parsed: ParsedCode) -> None:
        # Track last assignment per identifier within the current function scope
        last_assignment: Dict[str, int] = {}
        current_function_id: Optional[int] = None

        for node in walk(parsed.root):
            if node.type == "function_definition":
                current_function_id = self._get_node_id(node, parsed)
                last_assignment = {}

            if node.type in ASSIGNMENT_NODE_TYPES:
                # Heuristic: left side is the variable being assigned
                left = node.child_by_field_name("left")
                if left is not None and left.type in IDENTIFIER_NODE_TYPES:
                    var_name = parsed.source_bytes[left.start_byte : left.end_byte].decode(
                        "utf-8", "ignore"
                    )
                    last_assignment[var_name] = self._get_node_id(node, parsed)

            if node.type in IDENTIFIER_NODE_TYPES:
                var_name = parsed.source_bytes[node.start_byte : node.end_byte].decode(
                    "utf-8", "ignore"
                )
                if var_name in last_assignment:
                    src_id = last_assignment[var_name]
                    dst_id = self._get_node_id(node, parsed)
                    if src_id != dst_id:
                        self._graph.add_edge(src_id, dst_id, type="data")

            # reset on leaving function (best-effort)
            if current_function_id is not None and node.type == "module":
                current_function_id = None
                last_assignment = {}

    def _connect_if_present(
        self,
        src_id: int,
        node: Optional[Node],
        parsed: ParsedCode,
        edge_type: str,
    ) -> None:
        if node is None:
            return
        dst_id = self._get_node_id(node, parsed)
        self._graph.add_edge(src_id, dst_id, type=edge_type)
