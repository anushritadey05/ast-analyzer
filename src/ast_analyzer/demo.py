from __future__ import annotations

from .graph_builder import ASTGraphBuilder
from .parser import CodeParser


SAMPLE_CODE = """

def foo(n):
    s = 0
    for i in range(n):
        s += i
    return s

"""


def main() -> None:
    parser = CodeParser("python")
    parsed = parser.parse(SAMPLE_CODE)

    builder = ASTGraphBuilder()
    result = builder.build(parsed)

    print("Nodes:", result.graph.number_of_nodes())
    print("Edges:", result.graph.number_of_edges())

    # Print a few example edges
    for i, (u, v, data) in enumerate(result.graph.edges(data=True)):
        if i > 10:
            break
        print(u, "->", v, data)


if __name__ == "__main__":
    main()
