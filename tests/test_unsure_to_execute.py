# pyright: reportPrivateUsage=false
"""The ``Unsure``-to-execute lint, ``unsure_to_execute``: each ``d:``
base ref set to ``Unsure`` in turn while the other reads range over their points, the
policy called alone on a fresh board, what counts as ``execute``, the ``ValueError``s before
any call, and a report of plain data. The points are tested in ``test_lint_points.py``."""

import contextlib
import copy
import json
from typing import Any

import pytest

from jurat import (
    Design,
    LoadedDesign,
    Parameter,
    Plan,
    Policy,
    Question,
    Requirement,
    Rule,
    Source,
    Step,
    Text,
    Undetermined,
    load,
)
from jurat import types as t
from jurat._canonical import encode
from jurat.helpers import unsure_to_execute

CALLS: list[dict[str, object]] = []


def yes(r: Any) -> bool:
    return True


def half(r: Any) -> float:
    return 0.5


def lean(r: Any) -> Plan:
    """Execute when ``d:ok`` is true, else when ``d:level`` clears the cut."""
    ok, level = r.read("d:ok"), r.read("d:level")
    CALLS.append({"ok": ok, "level": level, "count": r.read("t:count")})
    if ok is True:
        return Plan("execute")
    if isinstance(level, Undetermined):
        return Plan("escalate", reason="unsure")
    return Plan("execute") if level >= r.param("level/cut") else Plan("escalate", reason="low")


SOURCES = {
    "ask": Source({"q:go": Question("noul", Text("Go?"))}),
    "look": Source({"t:count": t.integer(0, 10), "t:name": t.string}),
}
RULES = {
    "ok": Rule("d:ok", t.boolean, yes, version="1", reads=["t:count", "q:go"]),
    "level": Rule("d:level", t.number(0, 1), half, version="1", reads=["q:go"]),
    "info": Rule(
        "d:info",
        t.record(a=t.boolean, b=t.label("x", "y")),
        yes,
        version="1",
        reads=["q:go"],
    ),
}
PARAMETERS = {
    "level/cut": Parameter(t.number(0, 1), 0.5),
    "count/max": Parameter(t.integer(), 3),
}
BOUNDARIES = {"d:level": ["level/cut"], "t:count": ["count/max"]}


def design(
    policy: Any = lean, reads: tuple[str, ...] = ("d:level", "d:ok", "t:count"), **changes: Any
) -> Design:
    boundaries = {k: v for k, v in BOUNDARIES.items() if k in reads}
    fields: dict[str, Any] = {
        "state": t.record(flag=t.boolean),
        "policy": Policy(
            policy, version="p", reads=list(reads), params=["level/cut"], boundaries=boundaries
        ),
        "sources": SOURCES,
        "rules": RULES,
        "parameters": PARAMETERS,
        "tools": ["send"],
    }
    fields.update(changes)
    return Design(**fields)


LOADED = load(design())


def unsure(*refs: str) -> dict[str, Any]:
    return {"kind": "Unsure", "cause": {"code": "abstained", "refs": list(refs)}}


def lint(policy: Any, reads: tuple[str, ...] = ("d:ok",), **keywords: Any) -> dict[str, Any]:
    return unsure_to_execute(load(design(policy, reads)), **keywords)


def execute(r: Any) -> Plan:
    return Plan("execute")


def escalate(r: Any) -> Plan:
    return Plan("escalate", reason="r")


# ---------------------------------------------------------------------------------------------
# The search and the report


def test_each_execute_on_unsure_is_reported_with_its_read_and_point_in_order() -> None:
    report = unsure_to_execute(LOADED)
    level = unsure("q:go")
    ok = unsure("q:go", "t:count")  # the writing rule's declared reads, in code-point order
    assert report == {
        "points": 2 * 3 + 3 * 3,
        "execute": [
            {"read": "d:level", "point": {"d:level": level, "d:ok": True, "t:count": 0}},
            {"read": "d:level", "point": {"d:level": level, "d:ok": True, "t:count": 3}},
            {"read": "d:level", "point": {"d:level": level, "d:ok": True, "t:count": 10}},
            {"read": "d:ok", "point": {"d:level": 0.5, "d:ok": ok, "t:count": 0}},
            {"read": "d:ok", "point": {"d:level": 0.5, "d:ok": ok, "t:count": 3}},
            {"read": "d:ok", "point": {"d:level": 0.5, "d:ok": ok, "t:count": 10}},
            {"read": "d:ok", "point": {"d:level": 1.0, "d:ok": ok, "t:count": 0}},
            {"read": "d:ok", "point": {"d:level": 1.0, "d:ok": ok, "t:count": 3}},
            {"read": "d:ok", "point": {"d:level": 1.0, "d:ok": ok, "t:count": 10}},
        ],
    }


def test_the_policy_reads_the_marker_and_the_other_reads_live() -> None:
    CALLS.clear()
    unsure_to_execute(LOADED)
    assert len(CALLS) == 15
    first = CALLS[0]
    assert type(first["level"]).__name__ == "Unsure"
    assert first["level"].cause.refs == ("q:go",)  # type: ignore[attr-defined]
    assert (first["ok"], first["count"]) == (False, 0)
    assert all(type(c["count"]) is int for c in CALLS)


def test_no_t_read_is_set_and_with_no_d_read_nothing_is_called() -> None:
    called: list[object] = []

    def counting(r: Any) -> Plan:
        called.append(r.read("t:count"))
        return Plan("execute")

    assert lint(counting, ("t:count",)) == {"points": 0, "execute": []}
    assert called == []


def test_an_escalate_on_unsure_is_not_reported() -> None:
    assert lint(escalate)["execute"] == []


def test_a_d_field_path_read_sets_its_base_ref_and_every_read_under_it_shows_the_marker() -> None:
    seen: list[tuple[object, object]] = []

    def paths(r: Any) -> Plan:
        seen.append((r.read("d:info.a"), r.read("d:info.b")))
        return Plan("execute")

    report = lint(paths, ("d:info.a", "d:info.b"))
    assert report == {
        "points": 1,
        "execute": [
            {"read": "d:info", "point": {"d:info.a": unsure("q:go"), "d:info.b": unsure("q:go")}}
        ],
    }
    a, b = seen[0]
    assert type(a).__name__ == "Unsure"
    assert a is b


def test_a_read_under_another_is_filled_as_its_field_and_shown_through_it() -> None:
    report = lint(execute, ("d:ok", "t:state", "t:state.flag"))
    assert report["points"] == 2
    assert [item["point"] for item in report["execute"]] == [
        {"d:ok": unsure("q:go", "t:count"), "t:state": {"flag": False}, "t:state.flag": False},
        {"d:ok": unsure("q:go", "t:count"), "t:state": {"flag": True}, "t:state.flag": True},
    ]


def test_a_values_list_fills_a_read_its_type_does_not() -> None:
    report = lint(execute, ("d:ok", "t:name"), values={"t:name": ["ann", "bo"]})
    assert [item["point"]["t:name"] for item in report["execute"]] == ["ann", "bo"]


# ---------------------------------------------------------------------------------------------
# What is not execute


class Sub(Plan):
    __slots__ = ()


def caught_require(r: Any) -> Plan:
    with contextlib.suppress(BaseException):  # the stop is recorded though caught
        r.require("d:ok")
    return Plan("execute")


def caught_undeclared(r: Any) -> Plan:
    with contextlib.suppress(BaseException):
        r.read("t:name")
    return Plan("execute")


def raising(r: Any) -> Plan:
    raise RuntimeError("no")


def not_a_plan(r: Any) -> object:
    return "execute"


def a_subclass(r: Any) -> Plan:
    return Sub("execute")


@pytest.mark.parametrize(
    "policy", [caught_require, caught_undeclared, raising, not_a_plan, a_subclass]
)
def test_a_recorded_stop_a_raise_and_a_return_not_exactly_a_plan_are_not_reported(
    policy: Any,
) -> None:
    assert lint(policy) == {"points": 1, "execute": []}


def test_the_plan_rules_and_the_guard_are_not_applied() -> None:
    def breaking(r: Any) -> Plan:
        return Plan("execute", [Step("undeclared_tool", {})], reply="rule 5 forbids this")

    loaded = load(design(breaking, ("d:ok",), requirements=[Requirement("d:ok", True)]))
    assert unsure_to_execute(loaded)["execute"] == [
        {"read": "d:ok", "point": {"d:ok": unsure("q:go", "t:count")}}
    ]


def test_each_call_gets_a_fresh_board() -> None:
    seen: list[object] = []

    def mutating(r: Any) -> Plan:
        state = r.read("t:state")
        seen.append(dict(state))
        state["flag"] = "spoiled"
        return Plan("execute")

    report = lint(mutating, ("d:ok", "t:state"))
    assert seen == [{"flag": False}, {"flag": True}]
    assert [item["point"]["t:state"] for item in report["execute"]] == [
        {"flag": False},
        {"flag": True},
    ]


# ---------------------------------------------------------------------------------------------
# ValueErrors before any call


@pytest.mark.parametrize(
    ("reads", "values"),
    [
        (("d:ok", "t:name"), None),  # t:name is unfilled
        (("t:name",), None),  # unfilled though no d: read is set
        (("d:ok", "t:name"), {"t:name": ["a"], "q:go": [1]}),  # not a read this lint fills
        (("d:ok", "t:name"), {"t:name": ["a"], "d:ok": [True]}),  # its type fills it
        (("d:ok", "t:name"), {"t:name": [1]}),
        (("d:ok", "t:name"), {"t:name": ["a", "a"]}),
    ],
)
def test_a_value_error_is_raised_before_any_call(reads: tuple[str, ...], values: Any) -> None:
    called: list[object] = []

    def counting(r: Any) -> Plan:
        called.append(1)
        return Plan("execute")

    with pytest.raises(ValueError):  # noqa: PT011
        lint(counting, reads, values=values)
    assert called == []


def test_loaded_not_exactly_a_loaded_design_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        unsure_to_execute(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# The report


def test_the_report_is_plain_data_json_and_deepcopy_take() -> None:
    report = unsure_to_execute(LOADED)
    assert json.loads(json.dumps(report)) == report
    assert copy.deepcopy(report) == report


def test_two_runs_give_the_same_report_bytes() -> None:
    assert encode(unsure_to_execute(LOADED)) == encode(unsure_to_execute(LOADED))


def test_the_lint_raises_nothing_on_a_finding_and_changes_no_loaded_design() -> None:
    before = (LOADED.document, LOADED.design_hash, repr(LOADED._grid))
    assert unsure_to_execute(LOADED)["execute"]
    assert (LOADED.document, LOADED.design_hash, repr(LOADED._grid)) == before
    assert type(LOADED) is LoadedDesign


def test_a_read_at_a_path_with_no_read_above_it_is_written_at_its_path() -> None:
    seen: list[object] = []

    def flag(r: Any) -> Plan:
        seen.append(r.read("t:state.flag"))
        return Plan("execute")

    report = lint(flag, ("d:ok", "t:state.flag"))
    assert seen == [False, True]
    assert [item["point"]["t:state.flag"] for item in report["execute"]] == [False, True]


def test_a_read_more_than_one_field_deep_is_written_at_its_path() -> None:
    seen: list[object] = []

    def deep(r: Any) -> Plan:
        seen.append(r.read("t:state.inner.flag"))
        return Plan("execute")

    loaded = load(
        design(deep, ("d:ok", "t:state.inner.flag"), state=t.record(inner=t.record(flag=t.boolean)))
    )
    report = unsure_to_execute(loaded)
    assert seen == [False, True]
    assert [item["point"]["t:state.inner.flag"] for item in report["execute"]] == [False, True]


async def async_policy(r: Any) -> Plan:
    return Plan("execute")


def returning_a_coroutine(r: Any) -> object:
    return async_policy(r)


@pytest.mark.parametrize("policy", [async_policy, returning_a_coroutine])
def test_an_async_policy_or_one_returning_a_coroutine_is_not_execute(policy: Any) -> None:
    assert lint(policy) == {"points": 1, "execute": []}
