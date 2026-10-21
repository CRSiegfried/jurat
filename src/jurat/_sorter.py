"""The turn's sorter: one ``graphlib.TopologicalSorter`` per turn, inside interpret.

Built from the loaded design's node-to-predecessors mapping less the carried providers, each
removed as a key and from every predecessor set, since the constructor would add a named
predecessor as a node and return it as a root. Prepared once, here, and never
again: a sorter serves one walk.

A wavefront is returned as an unordered set; interpret sorts it. Which
returned nodes are ready, holding a source to the next round, and the drain are interpret's.
``done()`` on a node not yet returned raises ``graphlib``'s ``ValueError``; while a returned
node is not done, ``is_active()`` stays true even when ``get_ready()`` is empty.
"""

from collections.abc import Mapping
from graphlib import TopologicalSorter

from jurat._refs import code_point_sorted


class TurnSorter:
    """One turn's walk over the dependency graph."""

    __slots__ = ("_sorter",)

    def __init__(self, graph: Mapping[str, frozenset[str]], carried: frozenset[str]) -> None:
        # Keys and predecessors are added in code-point order, so the walk graphlib keeps inside
        # is alike under every hash seed; a wavefront still leaves as a frozenset, which keeps
        # no order, so interpret sorts it.
        sorter: TopologicalSorter[str] = TopologicalSorter()
        for node in code_point_sorted(graph):
            if node not in carried:
                sorter.add(node, *code_point_sorted(graph[node] - carried))
        sorter.prepare()  # a subgraph of load's acyclic graph raises no CycleError
        self._sorter = sorter

    def get_ready(self) -> frozenset[str]:
        """The next wavefront: the nodes whose predecessors are all done; empty when none."""
        return frozenset(self._sorter.get_ready())

    def done(self, node: str) -> None:
        """Mark a returned node done; ``ValueError`` for any other."""
        self._sorter.done(node)

    def is_active(self) -> bool:
        """Whether any node is left, returned or not; false at the fixpoint."""
        return self._sorter.is_active()
