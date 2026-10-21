# pyright: reportPrivateUsage=false
"""The design types: their fields, their snapshots and their type checks."""

import dataclasses
from types import MappingProxyType
from typing import Any

import pytest

import designs
from jurat import (
    AnswerKind,
    Design,
    Group,
    Parameter,
    PlanDeclaration,
    Policy,
    PolicyTable,
    Question,
    Requirement,
    Rule,
    Source,
    SourceRequirement,
    Table,
    Text,
    When,
)
from jurat import types as t


class Name(str):
    __slots__ = ()

    def __str__(self) -> str:
        return "overridden"


def test_form_is_a_member_no_constructor_sets() -> None:
    assert designs.rule("r").form == "function"
    assert Table("d:t", t.boolean, "U", [], []).form == "table"
    assert Policy(designs.policy).form == "function"
    assert PolicyTable([], []).form == "table"
    with pytest.raises(TypeError):
        Rule("d:r", t.boolean, designs.route, form="table")  # type: ignore[call-arg]


def test_a_design_snapshots_its_mappings_and_sequences() -> None:
    sources = {"s": Source({"t:x": t.string})}
    tools = ["a"]
    design = Design(state=t.record(), policy=Policy(designs.policy), sources=sources, tools=tools)
    sources["other"] = Source({"t:y": t.string})
    tools.append("b")
    assert list(design.sources) == ["s"]
    assert design.tools == ("a",)
    assert isinstance(design.sources, MappingProxyType)


def test_keys_are_read_by_their_string_data() -> None:
    design = designs.minimal(tools=[Name("send")], parameters={Name("p"): Parameter("number", 1)})
    assert type(next(iter(design.parameters))) is str
    assert design.tools == ("send",)
    assert type(design.tools[0]) is str


def test_keys_that_would_share_string_data_are_kept_as_given() -> None:
    class Odd(str):
        __slots__ = ()

        def __eq__(self, other: object) -> bool:
            return False

        def __hash__(self) -> int:
            return 1

    design = designs.minimal(types={Odd("a"): "number", "a": "number"})
    assert len(design.types) == 2


def test_the_design_defaults() -> None:
    design = Design(state=t.record(), policy=Policy(designs.policy))
    assert design.requirements == ()
    assert design.default_deny is True
    assert design.name is None
    for name in ("sources", "rules", "parameters", "types", "plans", "groups", "classes"):
        assert dict(getattr(design, name)) == {}


def test_source_defaults_and_its_provides() -> None:
    source = Source({"t:a": Name("string"), "t:b": {"type": t.string, "candidates": True}})
    assert type(source.provides["t:a"]) is str
    assert dict(source.provides["t:b"]) == {"type": "string", "candidates": True}
    assert (source.cost, source.feedback, source.payload, source.oracle) == (0, False, None, None)
    assert (source.timeout, source.when, source.requirement, source.grounded) == (None,) * 4


def test_a_provides_mappings_type_is_held_as_its_string_data() -> None:
    source = Source({"t:b": {"type": Name("string"), "candidates": False}})
    assert type(source.provides["t:b"]["type"]) is str
    odd = Source({"t:b": {1: "x"}})
    assert dict(odd.provides["t:b"]) == {1: "x"}


def test_a_question_takes_its_kind_as_a_string() -> None:
    q = Question("choice", Text("Which?"), labels=["a", Name("b")])
    assert q.kind is AnswerKind.choice
    assert q.labels == ("a", "b")
    label_set = Question("choice", Text("Which?"), labels={"fixed": ["x"], "candidates": "t:s"})
    assert dict(label_set.labels) == {"fixed": ("x",), "candidates": "t:s"}  # type: ignore[arg-type]


def test_a_texts_slots_are_snapshotted() -> None:
    text = Text("{a}", [{"name": "a", "origin": {"ref": "t:x"}}])
    assert dict(text.slots[0]["origin"]) == {"ref": "t:x"}
    assert isinstance(text.slots[0], MappingProxyType)


@pytest.mark.parametrize(
    "build",
    [
        lambda: Text(1),  # type: ignore[arg-type]
        lambda: Text("\ud800"),
        lambda: Text("x", "ab"),  # type: ignore[arg-type]
        lambda: Text("x", [{"name": "a"}]),
        lambda: Text("x", [{"name": "a", "origin": 1}]),
        lambda: Text("x", [1]),  # type: ignore[list-item]
        lambda: Question("maybe", Text("x")),
        lambda: Question("noul", "wording"),  # type: ignore[arg-type]
        lambda: Question("noul", Text("x"), labels="ab"),
        lambda: Question("choice", Text("x"), labels={"other": 1}),
        lambda: Source({1: t.string}),  # type: ignore[dict-item]
        lambda: Source([]),  # type: ignore[arg-type]
        lambda: Source({"q:x": t.string}),
        lambda: Source({}, needs="t:x"),
        lambda: Source({}, needs=[1]),  # type: ignore[list-item]
        lambda: Source({}, when=1),  # type: ignore[arg-type]
        lambda: Source({}, cost=True, timeout=-1, feedback=1),  # type: ignore[arg-type]
        lambda: Rule("d:r", t.boolean, 1),  # type: ignore[arg-type]
        lambda: Rule(1, None, designs.route, boundaries=[]),  # type: ignore[arg-type]
        lambda: Policy(designs.policy, reads="d:x"),
        lambda: Requirement(1, {1}),  # type: ignore[arg-type]
        lambda: SourceRequirement("d:x", "ab"),
        lambda: When(1, version=2),  # type: ignore[arg-type]
        lambda: Source({}, when=designs.route),  # type: ignore[arg-type]
        lambda: Group("q:x"),
        lambda: Table(1, t.boolean, "U", [], []),  # type: ignore[arg-type]
        lambda: Design(state=None, policy=1),  # type: ignore[arg-type]
        lambda: designs.minimal(sources=None, rules={"r": Source({})}, requirements=[1]),
        lambda: designs.minimal(default_deny=1, name=1, tools="send", types=None),
    ],
)
def test_a_design_type_validates_nothing_when_built(build: Any) -> None:
    build()  # check 10 tests every member's form at load


def test_a_question_kind_that_is_no_answer_kind_is_kept() -> None:
    assert Question("maybe", Text("x")).kind == "maybe"


def test_design_types_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        designs.minimal().name = "x"  # type: ignore[misc]


def test_refused_forms_build() -> None:
    plan = PlanDeclaration("clarify", reply=Text("Which?"))
    assert plan.kind == "clarify"
    assert plan.steps == ()


def test_a_when_holds_its_function_and_its_version_as_string_data() -> None:
    w = When(designs.route, version=Name("w1"))
    assert w.function is designs.route
    assert type(w.version) is str
    assert w.version == "w1"
    assert When(designs.route).version is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.version = "w2"  # type: ignore[misc]


def test_a_source_requirement_snapshots_its_values() -> None:
    values = ["a", "b"]
    requirement = SourceRequirement(Name("d:x"), values)
    values.append("c")
    assert requirement.values == ("a", "b")
    assert type(requirement.ref) is str
