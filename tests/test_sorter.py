# pyright: reportPrivateUsage=false
"""The turn's sorter: wavefronts, `done()`, `is_active()`, and the carried
providers left out; the `graphlib` facts it leans on, L1 to L4 below."""

import os
import random
import subprocess
import sys
from collections.abc import Callable, Mapping
from graphlib import TopologicalSorter
from pathlib import Path
from types import MappingProxyType

import pytest

from designs import full, minimal
from jurat import load
from jurat._sorter import TurnSorter

TESTS = Path(__file__).parent
NONE: frozenset[str] = frozenset()
# Id heads whose code-point order is not their UTF-16 order: an astral letter sorts last.
HEADS = "aZ\u00e9\u5b57\U0001d518"

# load(full())._graph: intent <- lookup; score <- intent; route <- intent, lookup, score.
FULL = load(full())._graph


def walk(
    graph: Mapping[str, frozenset[str]],
    carried: frozenset[str] = NONE,
    *,
    order: Callable[[list[str]], list[str]] = lambda nodes: nodes,
    held: Callable[[str], bool] = lambda node: False,
) -> list[frozenset[str]]:
    """Every wavefront to the end. A held node is done one call later, as interpret holds a
    source to the next round; `order` sets the `done()` order of each batch."""
    sorter = TurnSorter(graph, carried)
    out: list[frozenset[str]] = []
    pending: list[str] = []
    while sorter.is_active():
        ready = sorter.get_ready()
        out.append(ready)
        now = [n for n in sorted(ready) if not held(n)]
        for node in order(now + pending):
            sorter.done(node)
        pending = [n for n in sorted(ready) if held(n)]
    return out


def random_dag(rng: random.Random, size: int) -> dict[str, frozenset[str]]:
    ids = [rng.choice(HEADS) + str(i) for i in range(size)]
    return {node: frozenset(p for p in ids[:i] if rng.random() < 0.3) for i, node in enumerate(ids)}


# ---------------------------------------------------------------------------------------------
# The wavefronts


def test_the_loaded_graph_walks_in_wavefronts() -> None:
    assert walk(FULL) == [
        frozenset({"lookup"}),
        frozenset({"intent"}),
        frozenset({"score"}),
        frozenset({"route"}),
    ]


def test_a_wavefront_is_a_frozenset_and_interpret_sorts_it() -> None:
    sorter = TurnSorter({"b": NONE, "a": NONE, "c": frozenset({"a"})}, NONE)
    ready = sorter.get_ready()
    assert type(ready) is frozenset
    assert ready == {"a", "b"}


def test_an_empty_graph_is_not_active() -> None:
    sorter = TurnSorter(load(minimal())._graph, NONE)
    assert not sorter.is_active()
    assert sorter.get_ready() == NONE


def test_a_node_is_returned_once_only_when_its_predecessors_are_done() -> None:
    rng = random.Random(7)
    for _ in range(50):
        graph = random_dag(rng, 15)
        seen: set[str] = set()
        for ready in walk(graph):
            assert not ready & seen
            assert all(graph[n] <= seen for n in ready)
            seen |= ready
        assert seen == set(graph)


# ---------------------------------------------------------------------------------------------
# Wavefront membership does not depend on done() order


def test_the_same_done_calls_in_any_order_give_the_same_sets() -> None:
    rng = random.Random(11)
    for _ in range(100):
        graph = random_dag(rng, 20)
        held = {n for n in graph if rng.random() < 0.3}
        expected = walk(graph, held=held.__contains__)
        for seed in range(10):
            shuffle = random.Random(seed)
            permuted = walk(
                graph,
                order=lambda nodes, s=shuffle: s.sample(nodes, len(nodes)),
                held=held.__contains__,
            )
            assert permuted == expected


_SEED_PROBE = (
    "import random\n"
    "from test_sorter import FULL, random_dag, walk\n"
    "from jurat._refs import code_point_sorted\n"
    "rng = random.Random(3)\n"
    "for g in [FULL] + [random_dag(rng, 20) for _ in range(30)]:\n"
    "    held = {n for n in sorted(g) if rng.random() < 0.3}\n"
    # The done() order follows the hash seed; the holds do not.
    "    walked = walk(g, order=lambda ns: sorted(ns, key=hash), held=held.__contains__)\n"
    "    print([code_point_sorted(w) for w in walked])\n"
)


def test_the_sets_are_alike_under_every_hash_seed() -> None:
    outputs: set[str] = set()
    for seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", _SEED_PROBE],
            capture_output=True,
            text=True,
            check=True,
            cwd=TESTS,
            env=env,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1
    lines = next(iter(outputs)).splitlines()
    assert len(lines) == 31
    assert lines[0] == str([["lookup"], ["intent"], ["score"], ["route"]])


# ---------------------------------------------------------------------------------------------
# L1: one walk, prepared once


def test_the_sorter_exposes_only_the_walk() -> None:
    sorter = TurnSorter(FULL, NONE)
    assert {n for n in dir(sorter) if not n.startswith("_")} == {"done", "get_ready", "is_active"}


def test_a_finished_walk_has_no_node_left() -> None:
    sorter = TurnSorter(FULL, NONE)
    while sorter.is_active():
        for node in sorter.get_ready():
            sorter.done(node)
    assert sorter.get_ready() == NONE
    assert not sorter.is_active()


def test_graphlib_needs_prepare_before_get_ready() -> None:
    with pytest.raises(ValueError, match="prepare"):
        TopologicalSorter({"a": ()}).get_ready()


def test_graphlib_refuses_prepare_after_get_ready_and_offers_no_reset() -> None:
    # A second prepare() before get_ready() raises on 3.11 to 3.13 and not on 3.14, so only
    # the case alike on every version is asserted.
    sorter = TopologicalSorter({"a": ()})
    sorter.prepare()
    sorter.get_ready()
    with pytest.raises(ValueError, match="prepare"):
        sorter.prepare()
    assert not any(hasattr(sorter, n) for n in ("reset", "copy", "restart"))


# ---------------------------------------------------------------------------------------------
# L2: done() only on a returned node


def test_done_on_a_node_not_yet_returned_raises() -> None:
    sorter = TurnSorter(FULL, NONE)
    with pytest.raises(ValueError, match="intent"):
        sorter.done("intent")
    assert sorter.get_ready() == {"lookup"}  # the refusal left the walk as it was


def test_done_twice_raises() -> None:
    sorter = TurnSorter(FULL, NONE)
    sorter.get_ready()
    sorter.done("lookup")
    with pytest.raises(ValueError, match="lookup"):
        sorter.done("lookup")


def test_done_on_an_unknown_node_raises() -> None:
    sorter = TurnSorter(FULL, NONE)
    sorter.get_ready()
    with pytest.raises(ValueError, match="nowhere"):
        sorter.done("nowhere")


def test_done_on_a_carried_provider_raises() -> None:
    sorter = TurnSorter(FULL, frozenset({"lookup"}))
    with pytest.raises(ValueError, match="lookup"):
        sorter.done("lookup")


# ---------------------------------------------------------------------------------------------
# L3: the carried providers, left out as keys and as predecessors


def test_a_carried_provider_is_neither_a_node_nor_a_predecessor() -> None:
    assert walk(FULL, frozenset({"lookup"})) == [
        frozenset({"intent"}),
        frozenset({"score"}),
        frozenset({"route"}),
    ]


def test_carried_providers_never_come_back_as_roots() -> None:
    rng = random.Random(5)
    for _ in range(50):
        graph = random_dag(rng, 15)
        carried = frozenset(n for n in graph if rng.random() < 0.3)
        returned = [n for ready in walk(graph, carried) for n in ready]
        assert sorted(returned) == sorted(set(graph) - carried)


def test_the_loaded_graph_is_left_as_it_was() -> None:
    before = dict(FULL)
    walk(FULL, frozenset({"lookup", "intent"}))
    assert isinstance(FULL, MappingProxyType)
    assert dict(FULL) == before


# ---------------------------------------------------------------------------------------------
# L4: active while a returned node is not done


def test_is_active_while_a_returned_node_is_not_done() -> None:
    sorter = TurnSorter(FULL, NONE)
    assert sorter.get_ready() == {"lookup"}
    assert sorter.is_active()
    assert sorter.get_ready() == NONE  # nothing ready, yet active: a drain would spin here
    assert sorter.is_active()
    sorter.done("lookup")
    assert sorter.get_ready() == {"intent"}
