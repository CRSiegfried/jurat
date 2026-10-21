# pyright: reportPrivateUsage=false
"""Load: the loaded design of each slice feature, one invalid design per
check clause, check 12's refused forms, the graph and check 5. The boundaries fields and the
grid are in test_boundaries.py, and loading from a document in test_load_document.py."""

import copy
import dataclasses
import inspect
import os
import pickle
import subprocess
import sys
import time
from collections.abc import Callable
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

import designs
from counting import SmallBounds, bound, counted
from designs import full, minimal, question, rule, slot, with_rule, with_source
from jurat import (
    Design,
    DesignFailure,
    Group,
    InvalidDesign,
    LoadedDesign,
    Parameter,
    PlanDeclaration,
    Policy,
    PolicyTable,
    Question,
    Requirement,
    Source,
    SourceRequirement,
    Table,
    Text,
    When,
    load,
)
from jurat import types as t
from jurat._canonical import decode, encode, sha256_ref
from jurat._design import _BUILT_BY_LOAD, _rebuild_invalid_design
from jurat._types import Interval, Label, ListOf, MappingOf, Nullable, RecordOf, Scalar, parse_type
from jurat._values import MAX_DEPTH

TESTS = Path(__file__).parent


def refused(design: Design) -> list[DesignFailure]:
    with pytest.raises(InvalidDesign) as info:
        load(design)
    return list(info.value.failures)


def checks(design: Design) -> set[int]:
    return {f.check for f in refused(design)}


def only(design: Design, check: int) -> list[DesignFailure]:
    found = refused(design)
    assert {f.check for f in found} == {check}, found
    return found


def src_hash(fn: Callable[..., Any]) -> str:
    return sha256_ref(inspect.getsource(fn).encode("utf-8"))


# ---------------------------------------------------------------------------------------------
# Valid designs: the document, the hash, the graph, the providers, the classes


def test_a_minimal_design_writes_every_member_with_its_default() -> None:
    loaded = load(minimal())
    assert decode(loaded.document) == {
        "format": 1,
        "name": None,
        "state": "record{}",
        "sources": {},
        "rules": {},
        "policy": {
            "form": "function",
            "reads": [],
            "params": [],
            "boundaries": None,
            "module": "designs",
            "qualname": "policy",
            "version": "p1",
            "source": src_hash(designs.policy),
        },
        "requirements": [],
        "default_deny": True,
        "tools": {},
        "parameters": {},
        "types": {},
        "plans": {},
        "groups": {},
        "classes": {},
    }
    assert loaded.design_hash == sha256_ref(loaded.document)
    assert dict(loaded._graph) == {}
    assert dict(loaded._providers) == {}
    assert loaded._classes == {"t:state": {}}


def test_the_document_is_canonical() -> None:
    loaded = load(full())
    assert encode(decode(loaded.document), 128) == loaded.document


def test_a_source_and_its_questions() -> None:
    doc = decode(load(full()).document)
    assert doc["sources"]["intent"] == {
        "provides": {
            "q:intent/email": {
                "kind": "noul",
                "wording": {
                    "template": "Is {subject} mail for {who}? {{bar}} is {bar}",
                    "slots": [
                        {"name": "subject", "origin": {"ref": "t:state.subject"}},
                        {"name": "who", "origin": {"ref": "t:lookup"}},
                        {"name": "bar", "origin": {"param": "intent/bar"}},
                    ],
                },
                "labels": None,
            },
            "q:intent/tone": {
                "kind": "choice",
                "wording": {"template": "Which tone?", "slots": []},
                "labels": ["warm", "cold"],
            },
        },
        "needs": ["t:state", "t:state.sender"],
        "when": None,
        "requirement": None,
        "oracle": "mail",
        "timeout": 30,
        "payload": {"model": "m", "tags": ["a", "b"], "mood": "calm"},
        "cost": 2,
        "feedback": True,
        "grounded": None,
    }
    assert doc["sources"]["lookup"] == {
        "provides": {
            "t:lookup": {"type": "string", "candidates": False},
            "t:profile": {"type": "record{age: integer, name: string}", "candidates": False},
        },
        "needs": ["t:state"],
        "when": None,
        "requirement": None,
        "oracle": None,
        "timeout": None,
        "payload": None,
        "cost": 0,
        "feedback": False,
        "grounded": None,
    }


def test_function_rules_and_the_policy() -> None:
    doc = decode(load(full()).document)
    assert doc["rules"]["route"] == {
        "form": "function",
        "provides": "d:route",
        "outputs": "verdict",
        "reads": ["d:score", "q:intent/email", "t:profile.age"],
        "params": ["intent/bar"],
        "boundaries": None,
        "candidates": False,
        "grounded": None,
        "module": "designs",
        "qualname": "route",
        "version": "r2",
        "source": src_hash(designs.route),
    }
    assert doc["policy"] == {
        "form": "function",
        "reads": ["d:route", "t:state.subject"],
        "params": ["list/cap"],
        "boundaries": None,
        "module": "designs",
        "qualname": "policy",
        "version": "p3",
        "source": src_hash(designs.policy),
    }


def test_a_function_whose_source_cannot_be_read_has_a_null_source() -> None:
    design = with_rule(minimal(), "size", rule("size", outputs=t.integer(), function=len))
    rule_doc = decode(load(design).document)["rules"]["size"]
    assert (rule_doc["module"], rule_doc["qualname"], rule_doc["source"]) == (
        "builtins",
        "len",
        None,
    )


def test_requirements_default_deny_tools_parameters_types_and_groups() -> None:
    doc = decode(load(full()).document)
    assert doc["requirements"] == [{"ref": "d:route", "value": "yes"}]
    assert doc["default_deny"] is False
    assert doc["tools"] == {"archive": {}, "send": {}}
    assert doc["parameters"] == {
        "intent/bar": {"type": "number [0, 1]", "value": 0.9},
        "list/cap": {"type": "list<string>", "value": ["a", "b"]},
        "blob": {"type": "bytes", "value": "AAE"},
        "mood": {"type": 'label{"calm", "wild"}', "value": "calm"},
        "weights": {"type": "mapping<string, number>", "value": {"a": 1}},
    }
    assert doc["types"] == {"verdict": 'label{"no", "yes"}'}
    assert doc["groups"] == {"intents": ["q:intent/email", "q:intent/tone"]}
    assert doc["state"] == "record{sender: record{address: string}, subject: string}"
    assert doc["name"] == "mail"


def test_requirements_keep_declaration_order() -> None:
    design = full(requirements=[Requirement("d:score", 0.5), Requirement("d:route", "no")])
    assert decode(load(design).document)["requirements"] == [
        {"ref": "d:score", "value": 0.5},
        {"ref": "d:route", "value": "no"},
    ]


def test_unordered_lists_sort_by_canonical_bytes() -> None:
    design = with_rule(
        full(), "x", rule("x", reads=("t:state", "d:score", "q:intent/tone", "d:route"))
    )
    reads = decode(load(design).document)["rules"]["x"]["reads"]
    assert reads == sorted(reads, key=encode) == ["d:route", "d:score", "q:intent/tone", "t:state"]


def test_the_graph_maps_every_node_to_its_providers_in_code_point_order() -> None:
    loaded = load(full())
    assert isinstance(loaded._graph, MappingProxyType)
    assert list(loaded._graph) == ["intent", "lookup", "route", "score"]
    assert dict(loaded._graph) == {
        "intent": frozenset({"lookup"}),  # the slot t:lookup; t:state gives no edge
        "lookup": frozenset(),
        "route": frozenset({"intent", "lookup", "score"}),
        "score": frozenset({"intent"}),
    }
    assert all(type(v) is frozenset for v in loaded._graph.values())


def test_the_policy_and_requirements_add_no_edge() -> None:
    bare = full(policy=Policy(designs.policy, version="p3"), requirements=[])
    assert dict(load(full())._graph) == dict(load(bare)._graph)


def test_every_predecessor_is_a_key() -> None:
    graph = load(full())._graph
    assert all(p in graph for preds in graph.values() for p in preds)


def test_the_provider_map_has_no_state_key() -> None:
    loaded = load(full())
    assert dict(loaded._providers) == {
        "d:route": "route",
        "d:score": "score",
        "q:intent/email": "intent",
        "q:intent/tone": "intent",
        "t:lookup": "lookup",
        "t:profile": "lookup",
    }
    assert list(loaded._providers) == sorted(loaded._providers)


def test_class_trees_are_computed_in_full_for_the_state_and_every_provided_ref() -> None:
    assert load(full())._classes == {
        "t:state": {"sender": {"address": "content"}, "subject": "content"},
        "q:intent/email": "open",
        "q:intent/tone": "open",
        "t:lookup": "content",
        "t:profile": {"age": "open", "name": "content"},
        "d:route": "open",
        "d:score": "open",
    }


def test_class_trees_through_lists_mappings_nullables_and_definitions() -> None:
    kind = t.record(text=t.string, blob=t.bytes_, n=t.nullable(t.integer()))
    design = with_source(
        minimal(types={"item": kind}),
        "s",
        Source(
            provides={
                "t:list": t.list_("item"),
                "t:map": t.mapping(t.list_(t.label("a"))),
                "t:maybe": t.nullable(t.string),
            }
        ),
    )
    classes = load(design)._classes
    assert classes["t:list"] == {"blob": "content", "n": "open", "text": "content"}
    assert classes["t:map"] == ["open"]
    assert classes["t:maybe"] == "content"


def test_question_slots_become_needs_but_not_document_needs() -> None:
    loaded = load(full())
    assert loaded._needs["intent"] == frozenset(
        {"t:state", "t:state.sender", "t:state.subject", "t:lookup"}
    )
    assert decode(loaded.document)["sources"]["intent"]["needs"] == ["t:state", "t:state.sender"]


def test_the_loaded_design_holds_callables_types_leaves_and_groups() -> None:
    loaded = load(full())
    assert dict(loaded._callables) == {
        "route": designs.route,
        "score": designs.score,
        "policy": designs.policy,
    }
    assert loaded._leaf_types["mood"] == Label(("calm", "wild"))
    assert loaded._leaf_types["list/cap"] == ListOf(Scalar("string"))
    assert dict(loaded._groups) == {"intents": ("q:intent/email", "q:intent/tone")}
    assert dict(loaded._grid) == {}


def test_a_leaf_type_naming_a_definition_is_held_resolved() -> None:
    design = minimal(
        types={"tone": t.label("cold", "warm")},
        parameters={"tones": Parameter(t.list_("tone"), ["warm"])},
    )
    assert load(design)._leaf_types["tones"] == ListOf(Label(("cold", "warm")))


def test_each_t_and_d_refs_declared_type_is_held_parsed() -> None:
    loaded = load(full())
    assert dict(loaded._types) == {
        "d:route": Label(("no", "yes")),  # the definition `verdict`, resolved
        "d:score": Interval("number", 0, 1),
        "t:lookup": Scalar("string"),
        "t:profile": RecordOf((("age", Scalar("integer")), ("name", Scalar("string")))),
        "t:state": RecordOf(
            (("sender", RecordOf((("address", Scalar("string")),))), ("subject", Scalar("string")))
        ),
    }
    assert list(loaded._types) == sorted(loaded._types)


def test_a_refs_type_naming_a_definition_is_held_resolved_at_every_depth() -> None:
    design = with_source(
        minimal(types={"tone": t.label("cold", "warm")}),
        "s",
        Source(provides={"t:tones": t.nullable(t.list_("tone"))}),
    )
    held = load(design)._types["t:tones"]
    assert held == Nullable(ListOf(Label(("cold", "warm"))))
    assert held == parse_type(t.nullable(t.list_(t.label("cold", "warm"))), nullable=True)


def test_a_refs_type_naming_a_definition_under_a_mapping_and_a_field_is_held_resolved() -> None:
    design = with_source(
        minimal(types={"tone": t.label("cold", "warm"), "note": t.record(tone="tone")}),
        "s",
        Source(provides={"t:notes": t.mapping("note")}),
    )
    held = load(design)._types["t:notes"]
    assert held == MappingOf(RecordOf((("tone", Label(("cold", "warm"))),)))
    assert held == parse_type(t.mapping(t.record(tone=t.label("cold", "warm"))), nullable=True)


def test_the_state_may_be_a_definitions_name() -> None:
    design = minimal(state="envelope", types={"envelope": t.record(subject=t.string)})
    assert load(design)._classes["t:state"] == {"subject": "content"}


def test_payloads_and_leaf_values_are_held_as_decoded() -> None:
    loaded = load(full())
    payload = loaded._payload("intent")
    assert type(payload["tags"]) is list
    assert type(payload["mood"]) is str
    assert type(loaded._leaf_value("list/cap")) is list
    assert type(loaded._leaf_value("mood")) is str
    assert loaded._leaf_value("blob") == b"\x00\x01"
    # A fresh value at every read: a caller's change reaches no later read.
    loaded._leaf_value("list/cap").append("z")
    assert loaded._leaf_value("list/cap") == ["a", "b"]


def test_bytes_leaves_in_lists_and_mappings() -> None:
    design = minimal(
        parameters={
            "l": Parameter(t.list_(t.bytes_), [b"\xff"]),
            "m": Parameter(t.mapping(t.bytes_), {"k": b""}),
        }
    )
    loaded = load(design)
    assert decode(loaded.document)["parameters"]["l"]["value"] == ["_w"]
    assert loaded._leaf_value("l") == [b"\xff"]
    assert loaded._leaf_value("m") == {"k": b""}


def test_a_later_change_to_the_callers_objects_changes_no_loaded_design() -> None:
    sources = {"lookup": designs.lookup_source()}
    design = minimal(sources=sources)
    before = load(design).document
    sources["other"] = designs.lookup_source()
    assert load(design).document == before


def test_the_loaded_design_is_read_only_and_built_only_by_load() -> None:
    loaded = load(minimal())
    with pytest.raises(AttributeError):
        loaded.design_hash = "x"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        loaded._hash = "x"
    with pytest.raises(AttributeError, match="read-only"):
        del loaded._hash
    assert loaded.design_hash == load(minimal()).design_hash
    with pytest.raises(TypeError, match="only by load"):
        LoadedDesign(object())
    assert isinstance(LoadedDesign(_BUILT_BY_LOAD), LoadedDesign)


def test_a_changed_word_changes_the_hash() -> None:
    provides = {**designs.intent_source().provides, "q:intent/tone": question("Tone?")}
    other = with_source(full(), "intent", designs.intent_source(provides=provides))
    assert load(other).design_hash != load(full()).design_hash


_HASH_PROBE = (
    "import designs; from jurat import load, InvalidDesign\n"
    "print(load(designs.full()).design_hash)\n"
    "print(load(designs.gated()).design_hash, sorted(load(designs.gated())._classes.items()))\n"
    "print(load(designs.bounded()).design_hash, list(load(designs.bounded())._grid.items()))\n"
    "try:\n"
    "    load(designs.ungated())\n"
    "except InvalidDesign as e:\n"
    "    print(repr(str(e)))\n"
    "try:\n"
    "    load(designs.held_back())\n"
    "except InvalidDesign as e:\n"
    "    print(repr(str(e)), sorted({f.check for f in e.failures}))\n"
    "try:\n"
    "    load(designs.with_rule(designs.with_rule(designs.full(), 'a', "
    "designs.rule('a', reads=('d:b',))), 'b', designs.rule('b', reads=('d:a', 'q:x'))))\n"
    "except InvalidDesign as e:\n"
    "    print(repr(str(e)))\n"
)


def test_the_hash_and_the_failures_are_alike_under_every_hash_seed() -> None:
    outputs: set[str] = set()
    for seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, "-c", _HASH_PROBE],
            capture_output=True,
            encoding="utf-8",
            check=True,
            cwd=TESTS,
            env=env,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1
    assert next(iter(outputs)).splitlines()[0] == load(full()).design_hash


# ---------------------------------------------------------------------------------------------
# Thin forms, and the document path


def test_functions_beside_a_design_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="functions"):
        load(minimal(), functions={})


def test_functions_beside_a_document_is_a_mapping() -> None:
    with pytest.raises(TypeError, match="functions"):
        load(load(minimal()).document, functions=[])  # type: ignore[arg-type]


def test_load_takes_a_design_or_a_document() -> None:
    with pytest.raises(TypeError):
        load(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# Check 1


@pytest.mark.parametrize(
    "design",
    [
        with_source(minimal(), "Bad", Source({"t:x": t.string})),
        with_rule(minimal(), "policy", rule("policy")),
        with_rule(minimal(), "codec", rule("codec")),
        minimal(types={"number": t.string}),
        minimal(types={"Upper": t.string}),
        minimal(groups={"bad group": Group([])}),
        minimal(tools=["send", "send"]),
        minimal(tools=["Send"]),
        minimal(parameters={"a": Parameter(t.number(), 1), "a/b": Parameter(t.number(), 2)}),
        minimal(parameters={"a/": Parameter(t.number(), 1)}),
        with_source(minimal(), "s", Source({"t:x": t.string}, oracle="default")),
        with_rule(
            with_source(minimal(), "x", Source({"t:y": t.string})), "x", rule("x")
        ),  # a source and a rule share an id
        minimal(tools=["x"], parameters={"x": Parameter(t.number(), 1)}),
        with_source(minimal(), "s", Source({"t:x": t.string}, needs=["t:state", "t:state"])),
        with_rule(minimal(), "r", rule("r", reads=("t:state", "t:state"))),
        with_rule(
            minimal(parameters={"p": Parameter(t.number(), 1)}),
            "r",
            rule("r", params=("p", "p")),
        ),
        minimal(policy=Policy(designs.policy, version="1", reads=["t:state", "t:state"])),
        minimal(
            policy=Policy(designs.policy, version="1", params=["p", "p"]),
            parameters={"p": Parameter(t.number(), 1)},
        ),
    ],
)
def test_check_1(design: Design) -> None:
    only(design, 1)


def test_a_repeated_malformed_ref_is_check_7s_alone() -> None:
    design = with_source(minimal(), "s", Source({"t:x": t.string}, needs=["bad", "bad"]))
    assert [(f.check, f.names) for f in refused(design)] == [(7, ("bad", "s"))]


# ---------------------------------------------------------------------------------------------
# Check 2


def test_check_2_two_providers() -> None:
    design = with_source(full(), "other", Source({"t:lookup": t.string}))
    (failure,) = only(design, 2)
    assert failure.names == ("lookup", "other", "t:lookup")


def test_check_2_a_source_providing_the_state() -> None:
    design = with_source(minimal(), "s", Source({"t:state": t.record()}))
    (failure,) = only(design, 2)
    assert "the state declaration" in failure.message


# ---------------------------------------------------------------------------------------------
# Check 3


def nowhere(design: Design, *, check: int = 3) -> None:
    only(design, check)


@pytest.mark.parametrize(
    "design",
    [
        with_source(minimal(), "s", Source({"t:x": t.string}, needs=["d:none"])),
        with_rule(minimal(), "r", rule("r", reads=("q:none",))),
        minimal(policy=Policy(designs.policy, version="1", reads=["d:none"])),
        minimal(requirements=[Requirement("d:none", True)]),
        minimal(groups={"g": Group(["q:none"])}),
        full(groups={"g": Group(["q:intent/tone", "q:intent/tone"])}),
        with_source(minimal(), "s", Source({"q:x": question("Is {a}?", slot("a", ref="t:none"))})),
        with_rule(full(), "r", rule("r", reads=('q:intent/tone["a"]',))),
    ],
)
def test_check_3(design: Design) -> None:
    nowhere(design)


def test_a_field_path_under_the_state_has_its_provider() -> None:
    design = with_rule(minimal(), "r", rule("r", reads=("t:state.any",)))
    assert checks(design) == {9}  # provided, and typed at check 9


def test_a_candidate_slot_on_an_ungrounded_source_fails_checks_3_and_4() -> None:
    design = with_source(
        minimal(), "s", Source({"q:x": question("Is {c}?", slot("c", candidate="t:set"))})
    )
    assert checks(design) == {3, 4}


# ---------------------------------------------------------------------------------------------
# Check 4


@pytest.mark.parametrize(
    "design",
    [
        with_source(minimal(), "s", Source({"d:x": t.string})),
        with_rule(minimal(), "r", rule("r", provides="d:other")),
        full(policy=Policy(designs.policy, version="1", reads=["q:intent/tone"])),
        full(requirements=[Requirement("t:lookup", "a")]),
        with_source(
            full(), "s", Source({"q:x": question("Is {a}?", slot("a", ref="q:intent/tone"))})
        ),
    ],
)
def test_check_4(design: Design) -> None:
    only(design, 4)


def test_check_4_a_source_provides_refs_of_one_level() -> None:
    mixed = Source({"q:x": question(), "t:y": t.string})
    (failure,) = only(with_source(minimal(), "s", mixed), 4)
    assert failure.names == ("s",)
    assert (
        failure.message == 'source "s" provides q: and t: refs; a source provides refs of one level'
    )


def test_check_1_names_the_oracle_default() -> None:
    (failure,) = only(with_source(minimal(), "s", Source({"t:x": t.string}, oracle="default")), 1)
    assert failure.message == 'source "s" has the oracle name "default"'


# ---------------------------------------------------------------------------------------------
# Check 5


def test_check_5_a_cycle_between_rules_names_its_nodes() -> None:
    design = with_rule(
        with_rule(minimal(), "a", rule("a", reads=("d:b",))), "b", rule("b", reads=("d:a",))
    )
    (failure,) = only(design, 5)
    assert failure.names == ("a", "b")
    assert failure.message.count('"a"') + failure.message.count('"b"') == 3  # closed cycle


def test_check_5_a_rule_reading_its_own_ref_is_a_self_loop() -> None:
    (failure,) = only(with_rule(minimal(), "a", rule("a", reads=("d:a",))), 5)
    assert failure.names == ("a",)


def test_check_5_a_cycle_through_a_source() -> None:
    design = with_rule(
        with_source(minimal(), "s", Source({"q:x": question()}, needs=["d:r"])),
        "r",
        rule("r", reads=("q:x",)),
    )
    assert only(design, 5)[0].names == ("r", "s")


def test_check_5_a_source_needing_its_own_answer() -> None:
    design = with_source(minimal(), "s", Source({"q:x": question()}, needs=["q:x"]))
    assert only(design, 5)[0].names == ("s",)


def test_check_5_does_not_run_while_an_earlier_check_fails() -> None:
    design = with_rule(minimal(), "a", rule("a", reads=("d:a", "d:none")))
    assert checks(design) == {3}


def test_cycle_error_names_a_closed_cycle() -> None:
    with pytest.raises(CycleError) as info:
        TopologicalSorter({"a": {"b"}, "b": {"a"}}).prepare()
    cycle = info.value.args[1]
    assert cycle[0] == cycle[-1]
    assert set(cycle) == {"a", "b"}
    assert len(cycle) == 3
    with pytest.raises(CycleError) as info:
        TopologicalSorter({"a": {"a"}}).prepare()
    assert info.value.args[1] == ["a", "a"]


def test_the_sorter_adds_an_unkeyed_predecessor_as_a_root() -> None:
    sorter = TopologicalSorter({"a": {"b"}})
    sorter.prepare()
    assert sorter.get_ready() == ("b",)  # why every predecessor must be a key


def test_the_loaded_graph_drives_a_sorter() -> None:
    sorter = TopologicalSorter(load(full())._graph)
    assert list(sorter.static_order()) == ["lookup", "intent", "score", "route"]


# ---------------------------------------------------------------------------------------------
# Check 7


@pytest.mark.parametrize(
    "design",
    [
        with_source(minimal(), "s", Source({"t:x": t.string}, needs=["x:y"])),
        with_source(minimal(), "s", Source({"t:x": t.string}, needs=["t:"])),
        with_source(minimal(), "s", Source({"t:x.f": t.string})),
        with_source(minimal(), "s", Source({'t:x["a"]': t.string})),
        with_source(minimal(), "s", Source({"t:X": t.string})),
        with_source(minimal(), "s", Source({"q:X": question()})),
        with_rule(minimal(), "r", rule("r", reads=("t:state.",))),
        with_rule(minimal(), "r", rule("r", provides="d:r.f")),
        minimal(policy=Policy(designs.policy, version="1", reads=["t:state["])),
        minimal(requirements=[Requirement("rule:1", True)]),
        minimal(groups={"g": Group(["q:"])}),
        with_source(minimal(), "s", Source({"q:x": question("Is {a}?", slot("a", ref="nope"))})),
    ],
)
def test_check_7(design: Design) -> None:
    only(design, 7)


def test_a_provided_ref_check_7_refuses_leaves_its_declaration_checked() -> None:
    # Only the ref is left out of the other checks; its type, question and outputs
    # are still checked.
    bad_type = with_source(minimal(), "s", Source({"t:X": "nope<"}))
    assert checks(bad_type) == {7, 10}
    bad_question = with_source(
        minimal(), "s", Source({"q:X": question("Is {a}?", slot("a", ref="t:none"))})
    )
    assert checks(bad_question) == {3, 7}
    bad_outputs = with_rule(minimal(), "r", rule("r", provides="d:r.f", outputs="nope<"))
    assert checks(bad_outputs) == {7, 10}


def test_a_candidate_slots_clause_stands_when_its_ref_fails_check_7() -> None:
    design = with_source(
        minimal(), "s", Source({"q:x": question("Is {c}?", slot("c", candidate="bad"))})
    )
    assert checks(design) == {4, 7}


def test_a_ref_check_7_refuses_is_left_out_of_checks_3_and_4() -> None:
    design = minimal(policy=Policy(designs.policy, version="1", reads=["q:bad["]))
    assert checks(design) == {7}


# ---------------------------------------------------------------------------------------------
# Check 8


@pytest.mark.parametrize(
    "design",
    [
        with_rule(minimal(), "r", rule("r", params=("none",))),
        minimal(policy=Policy(designs.policy, version="1", params=["none"])),
        with_source(minimal(), "s", Source({"q:x": question("Is {a}?", slot("a", param="none"))})),
        minimal(parameters={"p": Parameter(t.number(0, 1), 70)}),
        minimal(parameters={"p": Parameter(t.mapping(t.number(0, 1)), {"yes": 0.9, "no": 70})}),
        minimal(parameters={"p": Parameter(t.integer(), 1.0)}),
    ],
)
def test_check_8(design: Design) -> None:
    only(design, 8)


def test_a_leaf_whose_type_fails_check_10_has_no_value_test() -> None:
    assert checks(minimal(parameters={"p": Parameter("nope<", 1)})) == {10}
    assert checks(minimal(types={"bad": "answer"}, parameters={"p": Parameter("bad", 1)})) == {10}


# ---------------------------------------------------------------------------------------------
# Check 9


def templated(template: str, *slots: dict[str, Any]) -> Design:
    return with_source(full(), "s", Source({"q:x": question(template, *slots)}))


@pytest.mark.parametrize(
    "design",
    [
        with_rule(full(), "r", rule("r", reads=("t:profile.none",))),
        with_rule(full(), "r", rule("r", reads=("t:profile.age.more",))),
        with_rule(full(), "r", rule("r", reads=("q:intent/tone.f",))),
        full(policy=Policy(designs.policy, version="1", reads=["t:state.none"])),
        templated("Is { name }?"),
        templated("Is {}?"),
        templated("Is } it?"),
        templated("Is {open"),
        templated("Is {a/b}?", slot("a/b", ref="t:lookup")),
        templated("Is {a}?", slot("a", ref="t:lookup"), slot("a", ref="t:lookup")),
        templated("Is {a}?"),
        templated("Is it?", slot("a", ref="t:lookup")),
        templated("  {a}  ", slot("a", ref="t:lookup")),
        full(requirements=[Requirement("d:route", "maybe")]),
    ],
)
def test_check_9(design: Design) -> None:
    only(design, 9)


def required(outputs: str, value: object) -> Design:
    """``full`` with a rule ``d:x`` of type ``outputs`` and a plan-level requirement on it."""
    design = with_rule(full(), "x", rule("x", outputs=outputs))
    return dataclasses.replace(design, requirements=[Requirement("d:x", value)])


@pytest.mark.parametrize(
    "design",
    [
        required(t.bytes_, "AAE"),
        required(t.nullable(t.bytes_), None),
        required(t.record(a=t.string, b=t.list_(t.bytes_)), {"a": "x", "b": []}),
        required(t.record(a=t.string), {"a": "x", "extra": 1}),
        required(t.list_(t.record(a=t.string)), [{"a": "x"}, {"a": "y", "z": None}]),
        required(t.mapping(t.record(a=t.string)), {"k": {"a": "x", "z": 1}}),
        required(t.record(a=t.nullable(t.record(b=t.string))), {"a": {"b": "x", "c": "y"}}),
    ],
)
def test_check_9_refuses_a_requirement_that_can_never_hold(design: Design) -> None:
    # A bytes entry has no encoding to compare, and an undeclared field is dropped at the
    # write, so neither requirement could hold.
    only(design, 9)


@pytest.mark.parametrize(
    ("design", "reason"),
    [
        (required(t.bytes_, "AAE"), "its ref's type holds bytes"),
        (required(t.record(a=t.string), {"a": 1}), "its value does not conform to its ref's type"),
        (
            required(t.record(a=t.string), {"a": "x", "b": 1}),
            "its value holds a field its ref's type does not declare",
        ),
    ],
)
def test_check_9s_message_names_why_a_requirement_cannot_hold(design: Design, reason: str) -> None:
    (failure,) = only(design, 9)
    assert failure.message == f'a plan-level requirement on "d:x" can never hold: {reason}'


@pytest.mark.parametrize(
    "design",
    [
        required(t.record(a=t.string), {"a": "x"}),
        required(t.list_(t.record(a=t.nullable(t.string))), [{"a": None}, {"a": "y"}]),
        required(t.mapping(t.integer()), {"k": 1}),
        required(t.nullable(t.record(a=t.string)), None),
    ],
)
def test_a_requirement_that_can_hold_loads(design: Design) -> None:
    load(design)


@pytest.mark.parametrize(
    "design",
    [
        full(requirements=[Requirement("d:route.f", "yes")]),
        minimal(requirements=[Requirement("d:none.f", 1)]),
        minimal(requirements=[Requirement("t:state.f", 1)]),
    ],
)
def test_a_requirement_on_a_field_path_fails_check_7_alone(design: Design) -> None:
    # A requirement names a whole estimate; check 7 refuses the ref, so no
    # other check reads it.
    only(design, 7)


def test_escapes_and_placeholders_that_are_well_formed() -> None:
    load(templated("{{}} {{a}} }}{{ {a}{a}", slot("a", ref="t:lookup")))
    load(templated("{{"))


def test_check_9_does_not_run_while_an_earlier_check_fails() -> None:
    design = with_source(
        minimal(), "s", Source({"q:x": question("{ bad", slot("a", ref="t:none"))})
    )
    assert checks(design) == {3}


# ---------------------------------------------------------------------------------------------
# Check 10


@pytest.mark.parametrize(
    "design",
    [
        minimal(state="record{a: string,b: string}"),
        minimal(state="list<string>"),
        minimal(state="record{} or null"),
        minimal(state="nope<"),
        minimal(state=None),  # type: ignore[arg-type]
        with_source(minimal(), "s", Source({"t:x": "list< string >"})),
        with_source(minimal(), "s", Source({"t:x": "answer"})),
        with_source(minimal(), "s", Source({"t:x": "undefined"})),
        with_source(minimal(), "s", Source({"t:x": t.string}, payload={1, 2})),
        with_source(minimal(), "s", Source({"t:x": t.string}, payload=b"bytes")),
        with_rule(minimal(), "r", rule("r", outputs="list<string or null>")),
        with_rule(minimal(), "r", rule("r", version=None)),
        with_rule(minimal(), "r", rule("r", version=3)),
        minimal(policy=Policy(designs.policy)),
        minimal(parameters={"p": Parameter(t.nullable(t.number()), 1)}),
        minimal(parameters={"p": Parameter(t.record(a=t.number()), {"a": 1})}),
        minimal(parameters={"p": Parameter(t.list_(t.list_(t.number())), [[1]])}),
        minimal(parameters={"p": Parameter(t.record(a=t.nullable(t.number())), {"a": 1})}),
        minimal(types={"a": "b", "b": "a"}),
        minimal(types={"a": "list<missing>"}),
        minimal(types={"a": "nope<"}),
        minimal(types={"a": "answer"}),
        minimal(types={"maybe": t.nullable(t.number())}, parameters={"p": Parameter("maybe", 1)}),
    ],
)
def test_check_10(design: Design) -> None:
    only(design, 10)


def src(**changes: Any) -> Design:
    fields: dict[str, Any] = {"provides": {"t:x": t.string}}
    fields.update(changes)
    return with_source(minimal(), "s", Source(**fields))


def asks(*args: Any, **changes: Any) -> Design:
    return with_source(minimal(), "s", Source({"q:x": Question(*args, **changes)}))


NAN = float("nan")

FORMS: list[Design] = [
    # The design's own members, a None block among them
    minimal(sources=None),
    minimal(rules=None),
    minimal(parameters=None),
    minimal(types=None),
    minimal(plans=None),
    minimal(groups=None),
    minimal(classes=None),
    minimal(requirements=None),
    minimal(requirements=[1]),
    minimal(tools="send"),
    minimal(tools=[1]),
    minimal(default_deny=1),
    minimal(name=1),
    minimal(policy=1),
    minimal(sources={"s": 1}),
    minimal(sources={1: Source({})}),
    minimal(rules={"r": 1}),
    minimal(parameters={"p": 1}),
    minimal(groups={"g": 1}),
    minimal(groups={"g": Group("q:x")}),
    minimal(plans={"p": 1}),
    full(requirements=[Requirement("d:score", {0.5})]),
    minimal(name=b"x"),
    # A source
    src(provides=None),
    src(provides={"q:x": t.string}),
    src(provides={"t:x": question()}),
    src(provides={"t:x": {"kind": t.string}}),
    src(provides={"t:x": {"type": t.string, "candidates": 1}}),
    src(provides={"t:x": 1}),
    src(needs="t:x"),
    src(needs=[1]),
    src(oracle=1),
    src(timeout=0),
    src(timeout=-1),
    src(timeout=float("inf")),
    src(timeout="1"),
    src(timeout=True),
    src(cost=-1),
    src(cost=True),
    src(cost=NAN),
    src(cost=2**60),
    src(feedback=1),
    src(grounded=1),
    # A question and its text
    asks("maybe", Text("Is it?")),
    asks("noul", "Is it?"),
    asks("noul", Text("Is it?"), labels=["yes", "no"]),
    asks("choice", Text("Which?")),
    asks("choice", Text("Which?"), labels=[]),
    asks("choice", Text("Which?"), labels=["a", "a"]),
    asks("choice", Text("Which?"), labels=[1]),
    asks("choice", Text("Which?"), labels="ab"),
    asks("score", Text("Rate"), labels=[]),
    asks("score", Text("Rate"), labels=["lo", "lo"]),
    asks("score", Text("Rate"), labels={"fixed": [], "candidates": "t:s"}),
    asks("choice", Text("Which?"), labels={"other": 1}),
    asks("choice", Text("Which?"), labels={"fixed": ["a", "a"], "candidates": "t:s"}),
    asks("choice", Text("Which?"), labels={"fixed": [], "candidates": 1}),
    asks("choice", Text("Which?"), labels={"fixed": [], "candidates": None}),
    asks("noul", Text(1)),  # type: ignore[arg-type]
    asks("noul", Text(None)),  # type: ignore[arg-type]
    asks("noul", Text("Is it?", "ab")),  # type: ignore[arg-type]
    asks("noul", Text("Is {a}?", [1])),  # type: ignore[arg-type]
    asks("noul", Text("Is {a}?", [{"name": "a"}])),
    asks("noul", Text("Is {a}?", [{"name": "a", "origin": {"ref": "t:x", "param": "p"}}])),
    asks("noul", Text("Is {a}?", [{"name": "a", "origin": {"other": "t:x"}}])),
    asks("noul", Text("Is {a}?", [{"name": "a", "origin": {"ref": 1}}])),
    asks("noul", Text("Is {a}?", [{"name": 1, "origin": {"ref": "t:x"}}])),
    # A rule and the policy
    with_rule(minimal(), "r", rule("r", function=1)),
    with_rule(minimal(), "r", rule("r", provides=1)),
    with_rule(minimal(), "r", rule("r", reads="t:state")),  # type: ignore[arg-type]
    with_rule(minimal(), "r", rule("r", reads=[1])),  # type: ignore[arg-type]
    with_rule(minimal(), "r", rule("r", params=[1])),
    with_rule(minimal(), "r", rule("r", candidates=1)),
    with_rule(minimal(), "r", rule("r", grounded=1)),
    minimal(policy=Policy(1, version="1")),  # type: ignore[arg-type]
    minimal(policy=Policy(designs.policy, version="1", reads="d:x")),
]


@pytest.mark.parametrize("design", FORMS)
def test_check_10_tests_every_members_form(design: Design) -> None:
    only(design, 10)


def test_two_keys_sharing_string_data_fail_check_10() -> None:
    class Odd(str):
        __slots__ = ()

        def __eq__(self, other: object) -> bool:
            return False

        def __hash__(self) -> int:
            return 1

    only(minimal(types={Odd("a"): t.number(), "a": t.number()}), 10)


def test_a_leaf_value_outside_the_domain_fails_checks_8_and_10() -> None:
    assert checks(minimal(parameters={"p": Parameter(t.string, {"a", "b"})})) == {8, 10}


def test_a_function_without_a_qualified_name_fails_check_10() -> None:
    class NoName:
        __module__ = "m"

        def __call__(self) -> None: ...

    only(minimal(policy=Policy(NoName(), version="1")), 10)


def test_a_choice_label_sets_fixed_part_may_be_empty() -> None:
    design = asks("choice", Text("Which?"), labels={"fixed": [], "candidates": "t:s"})
    assert checks(design) == {12}


def test_a_non_canonical_type_names_its_spelling() -> None:
    (failure,) = only(with_source(minimal(), "s", Source({"t:x": "list< string >"})), 10)
    assert "list<string>" in failure.message


def test_a_definition_failing_is_reported_once_not_at_each_use() -> None:
    design = minimal(
        types={"a": "nope<", "b": "list<a>"},
        state="record{x: b}",
        parameters={"p": Parameter("a", 1)},
    )
    (failure,) = only(design, 10)
    assert failure.names == ("a",)


def test_a_definition_naming_a_failing_one_is_not_reported_again() -> None:
    (failure,) = only(minimal(types={"a": "answer", "b": "list<a>", "c": "b"}), 10)
    assert failure.names == ("a",)


def test_a_refused_forms_malformed_ref_provides_nothing() -> None:
    reader = rule("r", reads=("d:tab",))
    design = with_rule(with_rule(minimal(), "tab", table("d:tab.f")), "r", reader)
    assert {(f.check, f.names) for f in refused(design)} == {
        (3, ("d:tab", "r")),  # nothing provides d:tab
        (12, ("tab",)),
    }
    assert checks(with_rule(minimal(), "tab", table("bad"))) == {12}


def test_a_refused_source_sharing_an_id_with_a_rule_is_not_reported_at_check_1() -> None:
    # No other check's clause runs on a refused form (check 12).
    design = with_rule(
        with_source(minimal(), "x", Source({"q:y": question()}, grounded="t:set")),
        "x",
        rule("x"),
    )
    assert checks(design) == {12}


def test_a_cycle_of_definitions_is_one_failure() -> None:
    (failure,) = only(minimal(types={"a": "list<b>", "b": "list<a>"}), 10)
    assert failure.names == ("a", "b")


def test_a_payload_outside_the_domain_is_not_encoded() -> None:
    design = with_source(full(), "s", Source({"t:x": t.string}, payload=float("nan")))
    assert checks(design) == {10}


# ---------------------------------------------------------------------------------------------
# Check 12


def table(provides: str, **changes: Any) -> Table:
    return Table(provides, t.boolean, "U", [], [], **changes)


REFUSED: list[tuple[Design, str]] = [
    (with_rule(minimal(), "tab", table("d:tab")), "decision table"),
    # A data rule is a one-input table; until the decision tables' step defines a table's
    # members, check 12 names it as a table.
    (with_rule(minimal(), "data", table("d:data", order=None)), "decision table"),
    (minimal(policy=PolicyTable([], [])), "policy table"),
    (minimal(plans={"ask": PlanDeclaration("clarify")}), "plan declaration"),
    (
        with_source(
            minimal(), "s", Source({"t:set": {"type": t.list_(t.string), "candidates": True}})
        ),
        "candidate set",
    ),
    (
        with_rule(minimal(), "r", rule("r", outputs=t.list_(t.string), candidates=True)),
        "candidate set",
    ),
    (
        with_source(minimal(), "s", Source({"q:x": question()}, grounded="t:set")),
        "grounded declaration",
    ),
    (with_rule(minimal(), "r", rule("r", grounded="t:set")), "grounded declaration"),
    (
        with_source(
            minimal(),
            "s",
            Source(
                {"q:x": question(kind="choice", labels={"fixed": ["none"], "candidates": "t:set"})}
            ),
        ),
        "candidate-valued label set",
    ),
]


@pytest.mark.parametrize(("design", "form"), REFUSED)
def test_check_12_names_each_refused_form(design: Design, form: str) -> None:
    (failure,) = only(design, 12)
    assert form in failure.message


@pytest.mark.parametrize(
    "design",
    [
        with_rule(minimal(), "tab", Table("d:tab", t.boolean, "X", [], [])),
        with_rule(minimal(), "tab", Table(1, t.boolean, "U", [], [])),  # type: ignore[arg-type]
        with_rule(minimal(), "tab", Table("d:tab", None, "U", [], [])),  # type: ignore[arg-type]
        with_rule(minimal(), "tab", table("d:tab", order="first")),
        with_rule(minimal(), "tab", table("d:tab", candidates=1)),
        with_rule(minimal(), "tab", table("d:tab", grounded=1)),
        with_rule(minimal(), "tab", Table("d:tab", t.boolean, "U", {1, 2}, [])),
        minimal(policy=PolicyTable([], [float("nan")])),
        minimal(plans={"ask": PlanDeclaration(1)}),  # type: ignore[arg-type]
        minimal(plans={"ask": PlanDeclaration("clarify", steps="send")}),  # type: ignore[arg-type]
        minimal(plans={"ask": PlanDeclaration("clarify", steps=[{1, 2}])}),
        minimal(plans={"ask": PlanDeclaration("clarify", reason="why")}),  # type: ignore[arg-type]
        minimal(plans={"ask": PlanDeclaration("clarify", grounded=1)}),  # type: ignore[arg-type]
    ],
)
def test_check_10s_member_forms_test_a_refused_form(design: Design) -> None:
    # Check 10's member forms apply to a refused form, so check 12 can name it.
    assert checks(design) == {10, 12}


def test_a_refused_form_still_provides_for_checks_2_and_3() -> None:
    design = with_rule(
        with_rule(minimal(), "tab", table("d:tab")), "r", rule("r", reads=("d:tab",))
    )
    assert checks(design) == {12}
    clash = with_rule(
        with_source(minimal(), "s", Source({"t:x": t.string})),
        "tab",
        Table("t:x", t.boolean, "U", [], []),
    )
    assert checks(clash) == {2, 12}


def test_no_other_clause_runs_on_a_refused_form() -> None:
    design = with_rule(minimal(), "Bad", Table("d:Bad", "nope<", "U", [], []))
    assert checks(design) == {12}


def test_a_grounded_provider_provides_its_bracketed_refs() -> None:
    design = with_rule(
        with_source(minimal(), "s", Source({"q:x": question()}, grounded="t:set")),
        "r",
        rule("r", reads=('q:x["a"]',)),
    )
    assert checks(design) == {12}


def test_a_refused_form_gates_check_5() -> None:
    design = with_rule(with_rule(minimal(), "a", rule("a", reads=("d:a",))), "tab", table("d:tab"))
    assert checks(design) == {12}


# ---------------------------------------------------------------------------------------------
# InvalidDesign


def test_failures_are_ordered_by_check_then_names_then_message() -> None:
    design = full(
        tools=["Z", "A"],
        policy=Policy(designs.policy, reads=["d:none"]),
        types={"verdict": VERDICT, "number": t.string},
    )
    found = refused(design)
    assert found == sorted(found, key=lambda f: (f.check, list(f.names), f.message))
    assert [f.check for f in found] == [1, 1, 1, 3, 10]


VERDICT = designs.VERDICT


def test_alike_failures_are_one() -> None:
    design = with_rule(
        with_rule(minimal(), "r", rule("r", params=("none",))),
        "q",
        rule("q", params=("none",)),
    )
    assert len(refused(design)) == 2
    built = DesignFailure(1, ("a",), "m", _BUILT_BY_LOAD)
    again = DesignFailure(1, ("a",), "m", _BUILT_BY_LOAD)
    assert InvalidDesign([built, again], _BUILT_BY_LOAD).failures == (built,)


def test_design_failure_and_invalid_design_are_built_only_by_load() -> None:
    with pytest.raises(TypeError, match="only by load"):
        DesignFailure(1, ("a",), "m")
    with pytest.raises(TypeError, match="only by load"):
        InvalidDesign([])
    with pytest.raises(ValueError, match="at least one"):
        InvalidDesign([], _BUILT_BY_LOAD)


def test_failures_are_a_tuple_of_frozen_design_failures() -> None:
    with pytest.raises(InvalidDesign) as info:
        load(minimal(tools=["Bad"]))
    failures = info.value.failures
    assert type(failures) is tuple
    (failure,) = failures
    assert (failure.check, failure.names) == (1, ("Bad",))
    assert [f.name for f in dataclasses.fields(failure)] == ["check", "names", "message"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        failure.check = 2  # type: ignore[misc]


def test_the_string_is_the_messages_one_per_line() -> None:
    design = full(tools=["Z", "A"])
    with pytest.raises(InvalidDesign) as info:
        load(design)
    assert str(info.value) == "\n".join(f.message for f in info.value.failures)
    assert all("\n" not in f.message for f in info.value.failures)


@pytest.mark.parametrize("failure", [object(), ("1", (), "m"), None])
def test_the_rebuild_takes_only_design_failures(failure: object) -> None:
    built = DesignFailure(1, ("a",), "m", _BUILT_BY_LOAD)
    with pytest.raises(TypeError, match="only DesignFailure"):
        _rebuild_invalid_design([built, failure])  # type: ignore[list-item]


def test_the_rebuild_refuses_a_design_failure_subclass() -> None:
    class Forged(DesignFailure):
        pass

    with pytest.raises(TypeError, match="only DesignFailure"):
        _rebuild_invalid_design([Forged(1, ("a",), "m", _BUILT_BY_LOAD)])


def test_a_parameter_holding_an_integer_past_the_digit_limit_is_outside_the_domain() -> None:
    design = dataclasses.replace(minimal(), parameters={"bar": Parameter(t.number(), 10**5000)})
    messages = [f.message for f in refused(design) if f.check == 10]
    assert messages == ['parameter "bar"\'s value is outside the value domain']


def _pickled(value: Any) -> Any:
    return pickle.loads(pickle.dumps(value))


@pytest.mark.parametrize(
    "clone",
    [_pickled, copy.copy, copy.deepcopy],
    ids=["pickle", "copy", "deepcopy"],
)
def test_an_invalid_design_survives_pickle_and_copy(clone: Callable[[Any], Any]) -> None:
    with pytest.raises(InvalidDesign) as info:
        load(full(tools=["Z", "A"]))
    original = info.value
    original.add_note("seen")
    again = clone(original)
    assert type(again) is InvalidDesign
    assert again is not original
    assert again.failures == original.failures
    assert all(type(f) is DesignFailure for f in again.failures)
    assert str(again) == str(original)
    assert again.args == original.args
    assert again.__notes__ == ["seen"]
    for failure in original.failures:
        assert clone(failure) == failure


def test_a_message_quotes_names_as_json_strings() -> None:
    (failure,) = only(with_source(minimal(), 'b"ad', Source({"t:x": t.string})), 1)
    assert failure.message == 'source "b\\"ad" is not in the segment grammar'


def test_names_are_in_code_point_order() -> None:
    (failure,) = only(full(policy=Policy(designs.policy, version="1", reads=["d:zz"])), 3)
    assert failure.names == ("d:zz", "policy")


def test_invalid_design_is_a_value_error() -> None:
    assert issubclass(InvalidDesign, ValueError)


def test_a_score_question_keeps_its_levels_in_order() -> None:
    design = with_source(
        minimal(), "s", Source({"q:x": Question("score", Text("Rate"), labels=["2", "1"])})
    )
    assert decode(load(design).document)["sources"]["s"]["provides"]["q:x"]["labels"] == ["2", "1"]


# ---------------------------------------------------------------------------------------------
# Check 5's report under every hash seed


def three_cycle() -> Design:
    design = with_rule(minimal(), "a", rule("a", reads=("d:b", "d:c")))
    design = with_rule(design, "c", rule("c", reads=("d:b",)))
    return with_rule(design, "b", rule("b", reads=("d:c",)))


def two_cycles() -> Design:
    design = minimal()
    spec = {"a": ("d:p", "d:q"), "p": ("d:p2",), "p2": ("d:p",), "q": ("d:q2",), "q2": ("d:q",)}
    for n, reads in spec.items():
        design = with_rule(design, n, rule(n, reads=reads))
    return design


_CYCLE_PROBE = (
    "import test_load\n"
    "from jurat import load, InvalidDesign\n"
    "for build in (test_load.three_cycle, test_load.two_cycles):\n"
    "    try:\n"
    "        load(build())\n"
    "    except InvalidDesign as e:\n"
    "        print([(f.check, f.names, f.message) for f in e.failures])\n"
)


def test_check_5s_failure_is_alike_under_every_hash_seed() -> None:
    outputs: set[str] = set()
    for seed in ("0", "1", "2", "3", "77"):
        result = subprocess.run(
            [sys.executable, "-c", _CYCLE_PROBE],
            capture_output=True,
            encoding="utf-8",
            check=True,
            cwd=TESTS,
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"},
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1, outputs
    # The probe printed each design's failures, as this process finds them.
    expected = [
        repr([(f.check, f.names, f.message) for f in refused(build())])
        for build in (three_cycle, two_cycles)
    ]
    assert next(iter(outputs)).splitlines() == expected


def test_a_reported_cycle_starts_at_its_least_id() -> None:
    (failure,) = only(three_cycle(), 5)
    assert failure.names == ("b", "c")
    assert failure.message == 'the dependency graph has a cycle: "b" -> "c" -> "b"'
    (failure,) = only(two_cycles(), 5)  # graphlib reports one cycle per prepare()
    assert failure.message.startswith('the dependency graph has a cycle: "p"') or (
        failure.message.startswith('the dependency graph has a cycle: "q"')
    )


def test_a_cycle_through_a_question_slot() -> None:
    design = with_rule(
        with_source(minimal(), "s", Source({"q:x": question("Is {a}?", slot("a", ref="d:r"))})),
        "r",
        rule("r", reads=("q:x",)),
    )
    assert only(design, 5)[0].names == ("r", "s")


# ---------------------------------------------------------------------------------------------
# What once escaped as an encoder error is InvalidDesign


@pytest.mark.parametrize(
    "design",
    [
        minimal(requirements=[Requirement("d:\ud800", True)]),
        with_source(
            minimal(),
            "s",
            Source({"q:x": question("Is {a}?", {"name": "\ud800", "origin": {"ref": "t:state"}})}),
        ),
        with_rule(minimal(), "r", rule("r", provides="d:r.f", outputs=object())),
        with_source(minimal(), "s", Source({"t:X": {"type": object()}})),
        with_rule(minimal(), "r", rule("r", provides="d:X", outputs="\ud800")),
    ],
)
def test_a_string_or_value_outside_the_domain_is_invalid_design(design: Design) -> None:
    assert 10 in checks(design)


# ---------------------------------------------------------------------------------------------
# A member check 10's form test refuses still provides and declares


def test_a_refused_question_still_provides_its_ref() -> None:
    design = with_rule(
        with_source(minimal(), "s", Source({"q:x": Question("choice", Text("Q?"), labels=())})),
        "r",
        rule("r", reads=("q:x",)),
    )
    assert checks(design) == {10}


def test_a_refused_t_declaration_still_provides_its_ref() -> None:
    design = with_rule(
        with_source(minimal(), "s", Source({"t:x": 5})), "r", rule("r", reads=("t:x",))
    )
    assert checks(design) == {10}


def test_a_refused_leaf_is_still_a_declared_leaf_and_name() -> None:
    design = with_rule(minimal(parameters={"p": 5}), "r", rule("r", params=("p",)))
    assert checks(design) == {10}
    assert checks(minimal(parameters={"policy": 5})) == {1, 10}
    clash = minimal(parameters={"a": 5, "a/b": Parameter(t.number(), 1)})
    assert checks(clash) == {1, 10}


def test_a_refused_question_still_provides_for_check_2() -> None:
    design = with_source(
        with_source(minimal(), "s", Source({"q:x": Question("noul", 1)})),  # type: ignore[arg-type]
        "u",
        Source({"q:x": question()}),
    )
    assert checks(design) == {2, 10}


# ---------------------------------------------------------------------------------------------
# More invalid cases


@pytest.mark.parametrize(
    "design",
    [
        minimal(parameters={"policy": Parameter(t.number(), 1)}),
        minimal(tools=["codec"]),
    ],
)
def test_check_1_a_parameter_or_tool_with_a_role_name(design: Design) -> None:
    only(design, 1)


def test_check_10_a_type_past_the_depth_bound() -> None:
    deep = t.string
    for _ in range(65):
        deep = f"list<{deep}>"
    (failure,) = only(with_source(minimal(), "s", Source({"t:x": deep})), 10)
    assert "64 deep" in failure.message
    nested = minimal(types={"d": "list<" * 40 + "string" + ">" * 40})
    over = with_source(nested, "s", Source({"t:x": "list<" * 30 + "d" + ">" * 30}))
    assert checks(over) == {10}


def test_the_sources_costs_summed_exactly_are_at_most_2_53_minus_1() -> None:
    # Each tag is within the domain; their sum is not.
    big = 2**53 - 1
    sources = {
        "b": Source({"t:b": t.string}, cost=big),
        "a": Source({"t:a": t.string}, cost=1),
        "z": Source({"t:z": t.string}, cost=0),
    }
    (failure,) = only(minimal(sources=sources), 10)
    assert failure.names == ("a", "b")  # every source with a positive cost
    sources["a"] = Source({"t:a": t.string}, cost=0)
    load(minimal(sources=sources))


def test_the_cost_sum_is_exact_not_a_float_sum() -> None:
    # As floats, 2**53 - 2 plus 0.5 plus 0.5 rounds to 2**53 - 2; exactly it is 2**53 - 1.
    sources = {
        "a": Source({"t:a": t.string}, cost=float(2**53 - 2)),
        "b": Source({"t:b": t.string}, cost=0.5),
        "c": Source({"t:c": t.string}, cost=0.5),
    }
    load(minimal(sources=sources))
    sources["d"] = Source({"t:d": t.string}, cost=0.5)
    (failure,) = only(minimal(sources=sources), 10)
    assert failure.names == ("a", "b", "c", "d")


# ---------------------------------------------------------------------------------------------
# A function when, a source requirement and a declared classes block


def gate(**changes: Any) -> Design:
    """``full`` with a source ``s`` providing ``t:x``, with ``changes`` to it."""
    return with_source(full(), "s", Source({"t:x": t.string}, **changes))


def needs_route(*values: object) -> SourceRequirement:
    return SourceRequirement("d:route", list(values))


def gated_on(outputs: str, *values: object) -> Design:
    """``full`` with a rule ``d:x`` of type ``outputs`` and a source requiring it."""
    design = with_rule(full(), "x", rule("x", outputs=outputs))
    requirement = SourceRequirement("d:x", list(values))
    return with_source(design, "s", Source({"t:out": t.string}, requirement=requirement))


def test_a_function_when_is_written_as_a_function_object() -> None:
    doc = decode(load(gate(when=When(designs.route, version="w1"))).document)["sources"]["s"]
    assert doc["when"] == {
        "module": "designs",
        "qualname": "route",
        "version": "w1",
        "source": src_hash(designs.route),
    }
    assert doc["requirement"] is None


def test_a_whens_callable_is_keyed_by_its_source_id_a_dot_and_when() -> None:
    # The key interpret reads, spelt as load spells a field function's.
    loaded = load(gate(when=When(designs.score, version="w1")))
    assert loaded._callables["s.when"] is designs.score
    assert set(loaded._callables) == {"policy", "route", "score", "s.when"}


def test_a_whens_version_changes_the_hash() -> None:
    first = load(gate(when=When(designs.route, version="w1"))).design_hash
    second = load(gate(when=When(designs.route, version="w2"))).design_hash
    assert len({first, second, load(gate()).design_hash}) == 3


def test_a_when_whose_source_cannot_be_read_has_a_null_source() -> None:
    doc = decode(load(gate(when=When(len, version="1"))).document)
    assert doc["sources"]["s"]["when"] == {
        "module": "builtins",
        "qualname": "len",
        "version": "1",
        "source": None,
    }


def test_a_source_requirement_is_written_as_data_its_values_by_canonical_bytes() -> None:
    doc = decode(load(gate(requirement=needs_route("yes", "no"))).document)["sources"]["s"]
    assert doc["requirement"] == {"ref": "d:route", "values": ["no", "yes"]}
    assert doc["when"] is None


def test_reordered_values_give_one_hash_and_another_value_another() -> None:
    first = load(gate(requirement=needs_route("yes", "no"))).design_hash
    assert load(gate(requirement=needs_route("no", "yes"))).design_hash == first
    assert load(gate(requirement=needs_route("yes"))).design_hash != first


def test_a_source_may_declare_both_forms() -> None:
    design = gate(when=When(designs.route, version="w1"), requirement=needs_route("yes"))
    doc = decode(load(design).document)["sources"]["s"]
    assert (doc["when"]["version"], doc["requirement"]) == (
        "w1",
        {"ref": "d:route", "values": ["yes"]},
    )


def test_a_requirements_ref_is_an_edge_and_not_a_need() -> None:
    # The binding receives the needs its source declares and the slots load adds, not a
    # requirement ref the source left out.
    loaded = load(gate(requirement=needs_route("yes")))
    assert loaded._graph["s"] == frozenset({"route"})
    assert loaded._needs["s"] == frozenset()
    assert decode(loaded.document)["sources"]["s"]["needs"] == []


def test_a_requirements_ref_the_source_declares_a_need_stays_one() -> None:
    loaded = load(gate(needs=["d:route"], requirement=needs_route("yes")))
    assert loaded._needs["s"] == frozenset({"d:route"})
    assert loaded._graph["s"] == frozenset({"route"})


def test_a_function_when_adds_no_edge() -> None:
    assert load(gate(when=When(designs.route, version="1")))._graph["s"] == frozenset()


@pytest.mark.parametrize(
    "design",
    [
        gated_on(t.record(a=t.string), {"a": "x"}, {"a": "y"}),
        gated_on(t.list_(t.record(a=t.nullable(t.string))), [{"a": None}], []),
        gated_on(t.nullable(t.integer()), None, 3),
        gated_on(t.number(0, 1), 1, 0.5),
    ],
)
def test_a_source_requirement_each_of_whose_values_can_hold_loads(design: Design) -> None:
    load(design)


def test_a_declared_object_overrides_its_fields_and_leaves_the_rest_their_default() -> None:
    loaded = load(full(classes={"t:state": {"subject": "identifier"}}))
    assert loaded._classes["t:state"] == {
        "sender": {"address": "content"},
        "subject": "identifier",
    }
    assert decode(loaded.document)["classes"] == {"t:state": {"subject": "identifier"}}


def test_a_class_name_over_a_record_covers_every_field_beneath_it() -> None:
    loaded = load(full(classes={"t:profile": "identifier", "t:state": {"sender": "open"}}))
    assert loaded._classes["t:profile"] == {"age": "identifier", "name": "identifier"}
    assert loaded._classes["t:state"] == {"sender": {"address": "open"}, "subject": "content"}


def test_a_d_key_and_an_empty_object_override_the_default() -> None:
    loaded = load(full(classes={"d:route": "content", "t:state": {}}))
    assert loaded._classes["d:route"] == "content"
    assert loaded._classes["d:score"] == "open"
    assert loaded._classes["t:state"] == load(full())._classes["t:state"]


def test_a_declared_tree_through_list_mapping_and_null_layers_and_a_definition() -> None:
    kind = t.record(text=t.string, n=t.nullable(t.integer()))
    provides = {
        "t:list": t.list_("item"),
        "t:map": t.mapping("item"),
        "t:maybe": t.nullable("item"),
        "t:deep": t.mapping(t.list_(t.mapping(t.integer()))),
    }
    design = with_source(minimal(types={"item": kind}), "s", Source(provides))
    classes = {
        "t:list": {"text": "identifier"},
        "t:map": {"n": "content"},
        "t:maybe": "identifier",
        "t:deep": "content",
    }
    loaded = load(dataclasses.replace(design, classes=classes))
    assert loaded._classes["t:list"] == {"n": "open", "text": "identifier"}
    assert loaded._classes["t:map"] == [{"n": "content", "text": "content"}]
    assert loaded._classes["t:maybe"] == {"n": "identifier", "text": "identifier"}
    assert loaded._classes["t:deep"] == [["content"]]


def test_a_declared_tree_is_written_as_plain_data_and_hashed() -> None:
    nested = MappingProxyType({"sender": MappingProxyType({"address": "identifier"})})
    loaded = load(full(classes={"t:state": nested}))
    assert decode(loaded.document)["classes"] == {"t:state": {"sender": {"address": "identifier"}}}
    plain = load(full(classes={"t:state": {"sender": {"address": "identifier"}}}))
    assert plain.design_hash == loaded.design_hash != load(full()).design_hash


def test_a_later_change_to_a_declared_tree_changes_no_loaded_design() -> None:
    tree: dict[str, Any] = {"subject": "identifier"}
    loaded = load(full(classes={"t:state": tree}))
    tree["subject"] = "open"
    assert decode(loaded.document)["classes"] == {"t:state": {"subject": "identifier"}}
    assert loaded._classes["t:state"]["subject"] == "identifier"


@pytest.mark.parametrize(
    "design",
    [
        gate(requirement=needs_route("yes", "yes")),
        gated_on(t.number(0, 1), 1, 1.0),  # one canonical spelling, 1
        gated_on(t.record(a=t.integer()), {"a": 1}, {"a": 1.0}),
    ],
)
def test_check_1_a_repeated_value_by_canonical_bytes(design: Design) -> None:
    (failure,) = only(design, 1)
    assert failure.names in (("d:route", "s"), ("d:x", "s"))


def test_check_1s_message_shows_the_repeated_value_as_canonical_json() -> None:
    (failure,) = only(gate(requirement=needs_route("yes", "yes")), 1)
    assert failure.message == 'source "s"\'s requirement lists "yes" twice in its values'


def test_true_and_1_are_not_a_repeat() -> None:
    assert checks(gated_on(t.boolean, True, 1)) == {9}  # 1 is not a boolean, but no repeat


@pytest.mark.parametrize(
    "design",
    [
        gate(requirement=SourceRequirement("d:none", ["a"])),
        full(classes={"q:intent/tone": "open"}),
        full(classes={'t:lookup["a"]': "open"}),
        full(classes={"t:state.subject": "open"}),
        full(classes={"d:none": "open"}),
        full(classes={"t:none": "open"}),
    ],
)
def test_check_3_a_requirement_ref_and_a_classes_key(design: Design) -> None:
    only(design, 3)


@pytest.mark.parametrize(
    "design",
    [
        gate(requirement=SourceRequirement("t:lookup", ["a"])),
        gate(requirement=SourceRequirement("q:intent/tone", ["warm"])),
    ],
)
def test_check_4_a_source_requirements_ref_is_at_level_d(design: Design) -> None:
    (failure,) = only(design, 4)
    assert failure.message.startswith('source "s"\'s requirement reads ')


def test_check_5_a_cycle_through_a_source_requirements_edge() -> None:
    requirement = SourceRequirement("d:r", [True])
    design = with_rule(
        with_source(minimal(), "s", Source({"q:x": question()}, requirement=requirement)),
        "r",
        rule("r", reads=("q:x",)),
    )
    assert only(design, 5)[0].names == ("r", "s")


@pytest.mark.parametrize(
    "design",
    [
        gate(requirement=SourceRequirement("d:Route", ["yes"])),
        gate(requirement=SourceRequirement("d:route.f", ["yes"])),
        gate(requirement=SourceRequirement("d:none.f", ["yes"])),
        full(classes={"T:x": "open"}),
        full(classes={"d:": "open"}),
    ],
)
def test_check_7_a_requirement_ref_and_a_classes_key(design: Design) -> None:
    only(design, 7)


def test_check_7_a_source_requirement_names_no_field_path() -> None:
    (failure,) = only(gate(requirement=SourceRequirement("d:route.f", ["yes"])), 7)
    assert failure.names == ("d:route.f", "s")
    assert failure.message == (
        'source "s"\'s requirement names "d:route.f", which has a field path'
    )


@pytest.mark.parametrize(
    "design",
    [
        gated_on(t.record(a=t.string), {"a": 1}),
        gated_on(t.record(a=t.string), {"a": "x", "b": 1}),
        gated_on(t.bytes_, "AAE"),
        gated_on(t.record(a=t.list_(t.bytes_)), {"a": []}),
        gated_on(t.label("a", "b"), "a", "c"),
        gate(requirement=needs_route("yes", "maybe")),
    ],
)
def test_check_9_a_source_requirement_any_of_whose_values_cannot_hold(design: Design) -> None:
    only(design, 9)


def test_check_9_names_each_value_that_cannot_hold() -> None:
    failures = only(gate(requirement=needs_route("perhaps", "yes", "maybe")), 9)
    reason = "its value does not conform to its ref's type"
    assert [f.message for f in failures] == [
        f'source "s"\'s requirement on "d:route" can never hold for "maybe": {reason}',
        f'source "s"\'s requirement on "d:route" can never hold for "perhaps": {reason}',
    ]
    assert {f.names for f in failures} == {("d:route", "s")}


def too_deep(depth: int) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for _ in range(depth):
        tree = {"a": tree}
    return tree


@pytest.mark.parametrize(
    "design",
    [
        gate(when=designs.route),  # a bare callable is not a When
        gate(when=When(1, version="1")),  # type: ignore[arg-type]
        gate(when=When(designs.route)),  # no version
        gate(when=When(designs.route, version=1)),  # type: ignore[arg-type]
        gate(requirement=Requirement("d:route", "yes")),
        gate(requirement=SourceRequirement("d:route", [])),
        gate(requirement=SourceRequirement("d:route", "yes")),  # type: ignore[arg-type]
        gate(requirement=SourceRequirement("d:route", {"yes"})),  # type: ignore[arg-type]
        gate(requirement=SourceRequirement(1, ["yes"])),  # type: ignore[arg-type]
        gate(requirement=needs_route(float("nan"))),
        gate(requirement=needs_route({"a", "b"}, "yes")),
        full(classes={"t:state": "secret"}),
        full(classes={"t:state": ["open"]}),
        full(classes={"t:state": {"subject": ["open"]}}),
        full(classes={"t:state": {1: "open"}}),
        full(classes={"t:state": None}),
        full(classes=["t:state"]),
        full(classes={"t:state": too_deep(70)}),
        full(classes={"t:lookup": {"x": "open"}}),
        full(classes={"t:state": {"subject": {}}}),
        full(classes={"t:state": {"sender": {"address": {"x": "open"}}}}),
        full(classes={"t:state": {"nofield": "open"}}),
        full(classes={"d:route": {}}),
    ],
)
def test_check_10_a_when_a_source_requirement_and_a_class_tree(design: Design) -> None:
    only(design, 10)


def test_check_10_names_where_a_tree_does_not_conform() -> None:
    design = full(classes={"t:state": {"sender": {"address": {}, "x": "open"}}})
    assert [f.message for f in only(design, 10)] == [
        'the class tree of "t:state" holds an object at "sender.address", where its type is '
        "not a record",
        'the class tree of "t:state" names "x" at "sender", which is not a field of the record',
    ]


def test_check_10s_tree_conformance_does_not_run_while_check_3_fails() -> None:
    assert checks(full(classes={"t:state": {"nofield": "open"}, "d:none": "open"})) == {3}


def test_a_refused_when_or_requirement_leaves_its_source_standing() -> None:
    reader = rule("r", reads=("t:x",))
    assert checks(with_rule(gate(when=designs.route), "r", reader)) == {10}
    assert checks(gate(when=designs.route, needs=["d:none"])) == {3, 10}
    # A refused requirement's ref is left out of every other check.
    assert checks(gate(requirement=SourceRequirement("d:none", "x"))) == {10}  # type: ignore[arg-type]


def test_the_held_back_probe_fails_only_check_9_and_check_10s_tree_test() -> None:
    found = refused(designs.held_back())
    assert {f.check for f in found} == {9, 10}
    assert all(f.message.startswith('the class tree of "t:state"') for f in found if f.check == 10)
    assert len([f for f in found if f.check == 9]) == 2


def test_an_invalid_designs_pickled_state_leaves_out_its_failures() -> None:
    with pytest.raises(InvalidDesign) as info:
        load(full(tools=["Z", "A"]))
    original = info.value
    _, args, state = original.__reduce__()
    assert args == (original.failures,)
    assert "failures" not in (state or {})
    again = _rebuild_invalid_design(*args)
    forged = DesignFailure(99, (), "forged", _BUILT_BY_LOAD)
    again.__setstate__({"failures": (forged,), "__notes__": ["n"]})
    assert again.failures == original.failures
    assert again.__notes__ == ["n"]
    with pytest.raises(TypeError, match="state is a dict"):
        again.__setstate__(None)


# ---------------------------------------------------------------------------------------------
# Check 10: a type's size with its definitions resolved, and the class trees' sum

_BOUND = 2**20
_OVER = "the type spells more than 2^20 code points with its definitions resolved"


def _chain(n: int) -> dict[str, str]:
    """``n`` definitions, each a record of two fields of the one before: 2^n nodes resolved."""
    types = {"t0": "record{a: string, b: string}"}
    for k in range(1, n):
        types[f"t{k}"] = f"record{{a: t{k - 1}, b: t{k - 1}}}"
    return types


def _chain_sizes(n: int) -> list[int]:
    sizes = [len("record{a: string, b: string}")]
    for _ in range(1, n):
        sizes.append(len("record{a: , b: }") + 2 * sizes[-1])
    return sizes


def test_check_10_refuses_an_exponential_chain_fast_naming_the_first_definition_over() -> None:
    first = next(k for k, size in enumerate(_chain_sizes(40)) if size > _BOUND)
    design = minimal(types=_chain(40), state="record{x: t39}")
    started = time.perf_counter()
    (failure,) = only(design, 10)  # the definitions after it, and the state, are not named
    assert time.perf_counter() - started < 1
    assert failure.names == (f"t{first}",)
    assert failure.message == f'type definition "t{first}": {_OVER}'


def test_a_modest_chain_loads_and_resolves_each_definition_once() -> None:
    loaded = load(minimal(types=_chain(12), state="record{x: t11, y: t11}"))
    state = loaded._types["t:state"]
    assert isinstance(state, RecordOf)
    x, y = (ft for _, ft in state.fields)
    assert x is y  # one resolution of t11, standing at each use
    tree: Any = loaded._classes["t:state"]
    assert tree["x"] == tree["y"]
    tree = tree["x"]
    for _ in range(11):
        assert tree["a"] == tree["b"]
        tree = tree["a"]
    assert tree == {"a": "content", "b": "content"}


def _long(size: int) -> str:
    """A record type of exactly ``size`` code points, by its field name's length."""
    return "record{" + "f" * (size - len("record{: string}")) + ": string}"


def test_check_10_refuses_a_declared_type_over_the_bound_through_its_definitions() -> None:
    body = _long(_BOUND - 30)
    state = "record{a: big, bc: big}"
    size = len(state) - 2 * len("big") + 2 * len(body)
    assert size > _BOUND
    (failure,) = only(minimal(types={"big": body}, state=state), 10)
    assert failure.names == ("t:state",)
    assert failure.message == f"the state declaration: {_OVER}"
    # At the bound it passes: the state is its own spelling with big's in place of the name.
    load(minimal(types={"big": _long(_BOUND - len("record{a: }"))}, state="record{a: big}"))


@pytest.mark.parametrize("where", ["source", "rule", "parameter", "definition"])
def test_check_10_names_each_declaration_whose_type_is_over_the_bound(where: str) -> None:
    over = _long(_BOUND + 1)
    if where == "source":
        design = with_source(minimal(), "s", Source({"t:x": over}))
        names, says = ("s", "t:x"), f'source "s"\'s "t:x": {_OVER}'
    elif where == "rule":
        design = with_rule(minimal(), "r", rule("r", outputs=over))
        names, says = ("r",), f'rule "r"\'s outputs: {_OVER}'
    elif where == "parameter":
        design = minimal(parameters={"p": Parameter(f"mapping<string, {over[:-1]}}}>", {})})
        names, says = ("p",), f'parameter "p": {_OVER}'
    else:
        design = minimal(types={"big": over}, state="record{x: big}")
        names, says = ("big",), f'type definition "big": {_OVER}'
    (failure,) = only(design, 10)
    assert (failure.names, failure.message) == (names, says)


def _summed(state_field: str) -> Design:
    """A design whose class trees' types sum to 2^20 code points with ``state_field`` ``ab``:
    the state, ``t:x`` and the answer's six."""
    big = _long((_BOUND - len("record{ab: }") - len("answer")) // 2)
    design = minimal(types={"big": big}, state=f"record{{{state_field}: big}}")
    return with_source(
        with_source(design, "s", Source({"t:x": "big"})), "a", Source({"q:ask": question()})
    )


def test_check_10_bounds_the_class_trees_types_summed_naming_every_ref() -> None:
    def total(design: Design) -> int:
        big = design.types["big"]
        return len(design.state) - 3 + len(big) + len(big) + len("answer")

    assert total(_summed("ab")) == _BOUND
    load(_summed("ab"))
    assert total(_summed("abc")) == _BOUND + 1
    (failure,) = only(_summed("abc"), 10)
    assert failure.names == ("q:ask", "t:state", "t:x")
    assert failure.message == (
        "the types of t:state and the provided refs sum to more than 2^20 code points with "
        "their definitions resolved"
    )


def test_the_class_trees_sum_leaves_out_a_type_check_10_already_refused() -> None:
    design = with_source(
        minimal(state="record{a: string}"), "s", Source({"t:x": _long(_BOUND + 1)})
    )
    (failure,) = only(design, 10)
    assert failure.names == ("s", "t:x")


# ---------------------------------------------------------------------------------------------
# Check 10: each value of a Design within the value domain's size bounds


def _shared_tree(n: int) -> Any:
    tree: Any = "open"
    for _ in range(n):
        tree = {"a": tree, "b": tree}
    return tree


def test_check_10_size_tests_every_value_of_a_design_where_it_stands(
    small_bounds: SmallBounds,
) -> None:
    rows = ["x" * 16] * 17  # one string at 17 places: past the 2^8 code points set
    sized = Source({"t:x": t.string}, payload=rows)
    (failure,) = only(with_source(minimal(), "s", sized), 10)
    assert failure.message == 'source "s"\'s payload is outside the value domain'
    leaf = Parameter("list<string>", rows)
    (failure,) = only(minimal(parameters={"p": leaf}), 10)
    assert failure.message == 'parameter "p"\'s value is outside the value domain'
    (failure,) = only(full(requirements=[Requirement("d:route", rows)]), 10)
    assert failure.names == ("d:route",)


def test_check_10_refuses_a_shared_class_tree_fast(small_bounds: SmallBounds) -> None:
    design = minimal(state="record{a: string, b: string}", classes={"t:state": _shared_tree(60)})
    with counted() as n:
        (failure,) = only(design, 10)
    assert n[0] < 4 * bound()  # the copy stops past the bound
    assert failure.message == 'the class tree of "t:state" is outside the value domain'
    # A shared tree within the bounds is read once and stands at each place.
    small = minimal(
        state="record{a: record{a: string, b: string}, b: record{a: string, b: string}}"
    )
    loaded = load(dataclasses.replace(small, classes={"t:state": _shared_tree(2)}))
    assert loaded._classes["t:state"] == _shared_tree(2)


def test_check_10_refuses_a_class_tree_too_deep_where_a_shared_member_stands() -> None:
    shared = {"a": "open"}
    deep: Any = shared
    for _ in range(MAX_DEPTH - 1):
        deep = {"a": deep}
    (failure,) = only(minimal(classes={"t:state": {"a": shared, "b": deep}}), 10)
    assert "is not a class name or an object of trees" in failure.message
    looped: dict[str, Any] = {}
    looped["a"] = looped
    (failure,) = only(minimal(classes={"t:state": looped}), 10)
    assert "is not a class name or an object of trees" in failure.message


def test_check_10_counts_a_class_trees_keys_as_values(small_bounds: SmallBounds) -> None:
    (failure,) = only(minimal(classes={"t:state": {"k" * 257: "open"}}), 10)
    assert failure.message == 'the class tree of "t:state" is outside the value domain'


# ---------------------------------------------------------------------------------------------
# Check 10: a question's labels within their count and code-point bounds


def _asking(labels: Any, kind: str = "choice") -> Design:
    return with_source(minimal(), "s", Source({"q:pick": question(kind=kind, labels=labels)}))


def test_check_10_bounds_a_questions_labels_at_the_real_bounds() -> None:
    (failure,) = only(_asking([str(i) for i in range(2**18 + 1)]), 10)
    assert failure.names == ("q:pick", "s")
    assert failure.message == 'source "s"\'s "q:pick"\'s labels number more than 2^18'
    (failure,) = only(_asking(["x" * (2**24 + 1)], kind="score"), 10)
    assert failure.message == 'source "s"\'s "q:pick"\'s labels hold more than 2^24 code points'


def test_check_10_bounds_a_questions_labels_counted_before_they_are_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jurat._form as form

    monkeypatch.setattr(form, "MAX_LABELS", 4)
    monkeypatch.setattr(form, "MAX_LABEL_CODE_POINTS", 8)
    load(_asking(["a", "b", "c", "d"]))
    (failure,) = only(_asking(["a", "b", "c", "d", "e"]), 10)
    assert failure.message.endswith("labels number more than 2^2")
    load(_asking(["abcd", "efgh"]))
    (failure,) = only(_asking(["abcd", "efgh", "\ud800"]), 10)  # counted before it is read
    assert failure.message.endswith("labels hold more than 2^3 code points")
    (failure,) = only(_asking(["abcd", "\ud800"]), 10)  # within the bounds, then read
    assert failure.message.endswith("labels are not a list of strings")
    fixed = {"fixed": ["a", "b", "c", "d", "e"], "candidates": "t:pool"}
    assert any(f.message.endswith("number more than 2^2") for f in refused(_asking(fixed)))


def test_a_long_chain_of_aliases_loads_without_recursion() -> None:
    types = {"a0": "string", **{f"a{i}": f"a{i - 1}" for i in range(1, 3001)}}
    started = time.perf_counter()
    loaded = load(minimal(types=types, state="record{x: a3000, y: list<a2999>}"))
    assert time.perf_counter() - started < 5  # each definition's checks are taken once
    assert loaded._types["t:state"] == RecordOf(
        (("x", Scalar("string")), ("y", ListOf(Scalar("string"))))
    )


def test_many_definitions_are_checked_in_time_linear_in_them() -> None:
    types = {f"d{i}": "record{a: string}" for i in range(5000)}
    types["bad"] = "answer"
    types.update({f"u{i}": "list<bad>" for i in range(5000)})
    started = time.perf_counter()
    (failure,) = only(minimal(types=types), 10)  # the uses of a failing definition: once
    assert time.perf_counter() - started < 5
    assert failure.names == ("bad",)


def test_a_declaration_over_the_bound_only_through_two_passing_definitions_is_named(
    small_bounds: SmallBounds,
) -> None:
    small_bounds(type_size=2**8)
    a = _long(120)
    b = _long(121)
    design = minimal(types={"a": a, "b": b}, state="record{x: a, y: b}")
    assert len("record{x: , y: }") + 120 + 121 > 2**8
    (failure,) = only(design, 10)
    assert failure.names == ("t:state",)
    assert failure.message == (
        "the state declaration: the type spells more than 2^8 code points with its definitions "
        "resolved"
    )


def test_a_definition_naming_one_that_names_an_undefined_name_is_not_reported() -> None:
    (failure,) = only(minimal(types={"a": "list<nope>", "b": "list<a>", "c": "b"}), 10)
    assert failure.names == ("a", "nope")


def test_each_parameter_that_is_a_node_of_another_is_named_with_it() -> None:
    n = Parameter(t.number(), 1)
    design = minimal(parameters={"a/b/c": n, "x/y": n, "a": n, "a/b": n, "ab/c": n})
    found = only(design, 1)
    assert [(f.names, f.message) for f in found] == [
        (("a", "a/b"), 'parameter "a" is a node of "a/b"'),
        (("a", "a/b/c"), 'parameter "a" is a node of "a/b/c"'),
        (("a/b", "a/b/c"), 'parameter "a/b" is a node of "a/b/c"'),
    ]


def test_a_provides_mappings_type_is_loaded_as_its_string_data() -> None:
    class Name(str):
        __slots__ = ()

    source = Source({"t:x": t.string})
    object.__setattr__(source, "provides", {"t:x": {"type": Name("string"), "candidates": False}})
    loaded = load(with_source(minimal(), "s", source))
    assert type(loaded._design.sources["s"].provides["t:x"]["type"]) is str
