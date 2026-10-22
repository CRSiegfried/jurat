# pyright: reportPrivateUsage=false
"""Compile: the policy's one call over its read view, the
order of outcomes, what only the engine writes, the seven plan rules in their fixed order, the
``plan`` and ``verdict`` items and what ``unhashed`` keeps of the policy, run after interpret on
an open ledger as decide will. Probes a, e and f2 are ported here."""

import asyncio
import contextlib
import copy
import enum
import gc
import inspect
import warnings
from collections.abc import Callable, ItemsView, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, cast

import pytest

from counting import SmallBounds, bound, counted
from designs import minimal
from jurat import (
    Binding,
    Cause,
    CauseCode,
    GuardFailure,
    LoadedDesign,
    Origin,
    Parameter,
    Plan,
    PlanKind,
    Policy,
    Record,
    Requirement,
    Rule,
    Step,
    Unasked,
    Unavailable,
    Verdict,
    VerdictKind,
    _compile,
    load,
    read_record,
)
from jurat import types as t
from jurat._canonical import encode
from jurat._compile import compile_turn
from jurat._component import at, exception_record
from jurat._plan import candidate_literal, engine_plan
from jurat._read_view import Access, UndeclaredRead, ViewClosed, _UndeclaredStop, _ViewError
from jurat._values import UNREPRESENTABLE, string_data
from jurat.helpers import pick_with_margin
from scripted import (
    Claims,
    InheritsAwait,
    MetaRaising,
    NotACoroutine,
    clashing,
    compiled,
    generator_coroutine,
    interpret_on,
    open_ledger,
    settings,
    started_raising_on_close,
    stop,
)

STATE_T = t.record(subject=t.string, to=t.string)
STATE = {"subject": "hello", "to": "dana"}
ORDER_T = t.record(status=t.string, n=t.integer())
ORDER = {"status": "open", "n": 2}


def const(value: object) -> Callable[[Any], object]:
    def fn(r: Any) -> object:
        return value

    return fn


def _rule(ref: str, outputs: str, value: object) -> Rule:
    return Rule(ref, outputs, const(value), version="1")


RULES = {
    "route": _rule("d:route", t.label("archive", "keep"), "archive"),
    "order": _rule("d:order", ORDER_T, ORDER),
    "gone": _rule("d:gone", t.boolean, Unavailable(Cause(CauseCode.raised, ["gone"]))),
    "lost": _rule("d:lost", ORDER_T, Unasked(Cause(CauseCode.round_bound))),
    "maybe": _rule("d:maybe", t.nullable(ORDER_T), None),
    "blob": _rule("d:blob", t.bytes_, b"\x00\x01"),
    "info": _rule("d:info", t.string, "not read by the policy"),
}
READS = (
    "d:route",
    "d:order",
    "d:order.status",
    "d:gone",
    "d:lost",
    "d:lost.status",
    "d:maybe.status",
    "d:blob",
    "t:state.subject",
)
PARAMETERS = {
    "folder": Parameter(t.string, "spam"),
    "other": Parameter(t.integer(), 3),
}

Policy_ = Callable[[Any], object]


def make(
    policy: Policy_,
    *,
    reads: Sequence[str] = READS,
    params: Sequence[str] = ("folder",),
    requirements: Sequence[Requirement] = (),
    default_deny: bool = True,
) -> LoadedDesign:
    return load(
        minimal(
            state=STATE_T,
            rules=RULES,
            policy=Policy(policy, version="1", reads=reads, params=params),
            requirements=requirements,
            default_deny=default_deny,
            tools=["archive", "send"],
            parameters=PARAMETERS,
        )
    )


def run(policy: Policy_, **design: Any) -> tuple[Record, Plan, Verdict]:
    return compiled(make(policy, **design), {}, STATE)


def returning(plan: object, **design: Any) -> tuple[Record, Plan, Verdict]:
    return run(const(plan), **design)


def plan_item(record: Record) -> dict[str, Any]:
    item = record.log[-2]
    assert item["item"] == "plan"
    return item


def cause_of(record: Record) -> dict[str, Any] | None:
    return plan_item(record)["cause"]


def broken(code: str, *refs: str) -> dict[str, Any]:
    return {"code": code, "refs": list(refs)}


def invalid(*refs: str) -> dict[str, Any]:
    return broken("invalid-return", *refs)


def rule_break(plan: object) -> list[str]:
    record, returned, verdict = returning(plan)
    cause = cause_of(record)
    assert cause is not None
    assert cause["code"] == "invalid-return", cause
    assert returned.kind is PlanKind.broken
    assert verdict == Verdict(VerdictKind.passed)
    return cause["refs"]


def plan_index(record: Record) -> str:
    return str(len(record.log) - 2)


R = Requirement("d:route", "archive")


def archive(**arguments: Origin | Binding) -> Step:
    return Step("archive", dict(arguments), [R])


def execute(*steps: Step) -> Plan:
    return Plan("execute", list(steps))


GOOD = execute(
    Step(
        "archive",
        {"folder": Origin(literal="spam"), "what": Origin(ref="d:route")},
        [Requirement("d:route", "archive")],
    ),
    Step(
        "send",
        {"n": Binding(1), "to": Origin(ref="t:state.subject"), "at": Binding(1, "items.name")},
        [Requirement("d:order", {"status": "open", "n": 2})],
    ),
)


# ---------------------------------------------------------------------------------------------
# The defect probes


def test_probe_a_an_escalate_plan_with_a_step_is_broken_under_rule_1() -> None:
    # The reference let an escalate plan's irreversible step past its guard.
    plan = Plan("escalate", [Step("archive", {}, [R])], reason="hand on")
    record, returned, verdict = returning(plan)
    assert cause_of(record) == invalid("rule:1")
    assert (returned.kind, returned.steps) == (PlanKind.broken, ())
    assert plan_item(record)["steps"] == []
    assert verdict.kind is VerdictKind.passed


def test_probe_e_a_binding_to_a_later_step_is_broken_under_rule_2() -> None:
    assert rule_break(execute(archive(x=Binding(5)))) == ["rule:2", "step:1"]


def test_probe_f2_a_policy_returning_none_is_invalid_return_naming_the_policy() -> None:
    record, returned, _ = returning(None)
    assert cause_of(record) == invalid("policy")
    assert returned.cause == Cause(CauseCode.invalid_return, ["policy"])


# ---------------------------------------------------------------------------------------------
# The plan item's bytes


def test_the_plan_and_verdict_items_bytes() -> None:
    record, _, _ = returning(None)
    assert encode(plan_item(record)) == (
        b'{"cause":{"code":"invalid-return","refs":["policy"]},"derivation":[],"item":"plan",'
        b'"kind":"broken","reason":null,"reply":null,"steps":[]}'
    )
    assert encode(record.log[-1]) == b'{"failures":[],"item":"verdict","kind":"passed"}'
    record, _, _ = returning(Plan("escalate", [archive()], reason="x"))
    assert encode(plan_item(record)) == (
        b'{"cause":{"code":"invalid-return","refs":["rule:1"]},"derivation":[],"item":"plan",'
        b'"kind":"broken","reason":null,"reply":null,"steps":[]}'
    )
    record, _, _ = returning(execute(archive(x=Binding(5))))
    assert b'"cause":{"code":"invalid-return","refs":["rule:2","step:1"]}' in encode(
        plan_item(record)
    )
    record, _, _ = run(lambda r: r.require("d:lost.status"))
    assert encode(plan_item(record)) == (
        b'{"cause":{"code":"required","refs":["d:lost.status"]},"derivation":[],"item":"plan",'
        b'"kind":"escalate","reason":null,"reply":null,"steps":[]}'
    )
    record, _, verdict = returning(GOOD)
    assert encode(plan_item(record)) == (
        b'{"cause":null,"derivation":[],"item":"plan","kind":"execute","reason":null,'
        b'"reply":null,"steps":[{"arguments":{"folder":{"literal":"spam"},'
        b'"what":{"ref":"d:route"}},"requirements":[{"ref":"d:route","value":"archive"}],'
        b'"tool":"archive"},{"arguments":{"at":{"path":"items.name","step":1},'
        b'"n":{"path":null,"step":1},"to":{"ref":"t:state.subject"}},'
        b'"requirements":[{"ref":"d:order","value":{"n":2,"status":"open"}}],"tool":"send"}]}'
    )
    assert verdict == Verdict(VerdictKind.passed)
    assert encode(record.log[-1]) == b'{"failures":[],"item":"verdict","kind":"passed"}'


def test_a_valid_plan_is_returned_and_recorded_as_the_policy_wrote_it() -> None:
    record, returned, verdict = returning(GOOD)
    assert returned == GOOD
    assert returned is not GOOD
    assert cause_of(record) is None
    assert verdict.kind is VerdictKind.passed
    assert [i["item"] for i in record.log[-2:]] == ["plan", "verdict"]


class Mood(enum.StrEnum):
    calm = "calm"


def test_the_returned_plan_holds_the_ledgers_stored_copy_not_the_policys_values() -> None:
    literal: list[Any] = [1, ("a", Mood.calm)]
    value: dict[str, Any] = {"status": "open", "n": 2}
    plan = execute(
        Step("archive", {"x": Origin(literal=literal)}, [R]),
        Step("send", {}, [Requirement("d:order", value)]),
    )
    record, returned, _ = returning(plan)
    literal.append("changed")
    value["n"] = 3
    first = returned.steps[0].arguments["x"]
    assert isinstance(first, Origin)
    assert first.literal == [1, ["a", "calm"]]
    assert type(first.literal[1][1]) is str
    assert returned.steps[1].requirements[0].value == {"status": "open", "n": 2}
    assert plan_item(record)["steps"][0]["arguments"]["x"] == {"literal": [1, ["a", "calm"]]}


# ---------------------------------------------------------------------------------------------
# What the policy returned, before the plan rules


class Subplan(Plan):
    __slots__ = ()


@pytest.mark.parametrize(
    "value",
    [None, "execute", {"kind": "execute"}, Step("archive"), Subplan("inform", reply="x"), [GOOD]],
)
def test_a_value_whose_type_is_not_exactly_plan_is_invalid_return_naming_the_policy(
    value: object,
) -> None:
    assert rule_break(value) == ["policy"]


def with_cause(plan: Plan) -> Plan:
    object.__setattr__(plan, "cause", Cause(CauseCode.required, ["d:gone"]))
    return plan


def with_slots(plan: Plan) -> Plan:
    object.__setattr__(plan, "slots", {"name": {"ref": "d:route"}})
    return plan


@pytest.mark.parametrize(
    "plan",
    [
        engine_plan(PlanKind.broken, Cause(CauseCode.raised, ["policy"])),
        engine_plan(PlanKind.escalate, Cause(CauseCode.required, ["d:gone"])),
        with_slots(Plan("inform", reply="x")),
        execute(archive(x=candidate_literal("Dana", "d:people"))),
        # Tested before rule 1: an escalate plan with a step and slots, or a cause, names the
        # policy, not rule 1.
        with_slots(Plan("escalate", [archive()], reason="x")),
        with_cause(Plan("escalate", [archive()], reason="x")),
    ],
)
def test_a_plan_carrying_what_only_the_engine_writes_is_invalid_return_naming_the_policy(
    plan: Plan,
) -> None:
    assert rule_break(plan) == ["policy"]


# ---------------------------------------------------------------------------------------------
# The plan-level rules: 1, 5, then 7 over reason and reply


@pytest.mark.parametrize(
    "plan",
    [
        Plan("clarify", [archive()], reply="which?"),
        Plan("inform", [archive()], reply="done"),
        Plan("broken", [archive()]),
    ],
)
def test_rule_1_a_plan_that_is_not_execute_has_no_steps(plan: Plan) -> None:
    assert rule_break(plan) == ["rule:1"]


@pytest.mark.parametrize(
    "plan",
    [
        Plan("inform"),
        Plan("clarify"),
        Plan("clarify", reply=" \t\n"),
        Plan("inform", reply=""),
        Plan("execute", reply="done"),
        Plan("escalate", reason="x", reply="y"),
        Plan("escalate"),
        Plan("escalate", reason="\u2003"),  # an em space
        Plan("broken"),
        Plan("broken", reason="x"),
        Plan("execute", reason="  "),  # carried text holds a character that is not whitespace
        Plan("inform", reply="ok", reason="\n"),
    ],
)
def test_rule_5_reply_and_reason_as_the_kind_requires(plan: Plan) -> None:
    assert rule_break(plan) == ["rule:5"]


@pytest.mark.parametrize(
    "plan",
    [
        Plan("execute"),
        Plan("execute", reason="why"),
        Plan("clarify", reply="which?"),
        Plan("inform", reply="done", reason="asked"),
        Plan("escalate", reason="hand on"),
    ],
)
def test_plans_that_keep_rule_5(plan: Plan) -> None:
    record, returned, _ = returning(plan)
    assert cause_of(record) is None
    assert returned == plan


@pytest.mark.parametrize(
    "plan",
    [Plan("escalate", reason="a \ud800"), Plan("inform", reply="\udfff!")],
)
def test_rule_7_reason_and_reply_are_in_the_value_domain(plan: Plan) -> None:
    assert rule_break(plan) == ["rule:7"]


# ---------------------------------------------------------------------------------------------
# The step rules: 2, 3, 4, 6, then 7, each step by position


@pytest.mark.parametrize(
    "plan",
    [
        execute(archive(x=Binding(1))),
        execute(archive(), archive(x=Binding(2))),
        execute(archive(), archive(x=Binding(0))),
        execute(archive(), archive(x=Binding(-1))),
        execute(archive(), archive(x=Binding(1, ""))),
        execute(archive(), archive(x=Binding(1, ".a"))),
        execute(archive(), archive(x=Binding(1, "a."))),
        execute(archive(), archive(x=Binding(1, "a..b"))),
        execute(archive(), archive(x=Binding(1, "A"))),
        execute(archive(), archive(x=Binding(1, "a/b"))),
        execute(archive(), archive(x=Binding(1, '["x"]'))),
    ],
)
def test_rule_2_a_step_binds_only_earlier_results_by_a_path_in_the_grammar(plan: Plan) -> None:
    assert rule_break(plan)[0] == "rule:2"


def test_a_binding_to_an_earlier_step_by_a_dotted_path_keeps_rule_2() -> None:
    plan = execute(archive(), archive(), archive(a=Binding(2, "items.first_name"), b=Binding(1)))
    record, _, _ = returning(plan)
    assert cause_of(record) is None


class Arg(enum.StrEnum):
    key = "idempotency_key"


@pytest.mark.parametrize(
    "plan",
    [
        execute(Step("delete", {}, [R])),
        execute(Step("Archive", {}, [R])),
        execute(archive(idempotency_key=Origin(literal="k"))),
        execute(Step("archive", {Arg.key: Origin(literal="k")}, [R])),
    ],
)
def test_rule_3_a_declared_tool_and_no_argument_idempotency_key(plan: Plan) -> None:
    assert rule_break(plan) == ["rule:3", "step:1"]


@pytest.mark.parametrize(
    "ref",
    [
        "d:info",  # provided, but not a read of the policy
        "d:none",
        "t:state",  # the policy declares t:state.subject, not t:state
        "d:route.x",
        "d:gone",  # a marker
        "d:lost.status",  # a marker at the path's base
    ],
)
def test_rule_4_a_board_ref_origin_is_a_declared_read_that_holds_a_value(ref: str) -> None:
    assert rule_break(execute(archive(x=Origin(ref=ref)))) == ["rule:4", "step:1"]


def test_a_null_on_a_board_ref_origins_path_is_a_value() -> None:
    record, _, _ = returning(execute(archive(x=Origin(ref="d:maybe.status"))))
    assert cause_of(record) is None


def cyclic() -> dict[str, Any]:
    value: dict[str, Any] = {"status": "open", "n": 2}
    value["self"] = value
    return value


@pytest.mark.parametrize(
    "requirement",
    [
        Requirement("t:state", STATE),  # not a d: ref
        Requirement("d:none", True),  # no provider
        Requirement("d:order.status", "open"),  # a field path
        Requirement('d:route["x"]', "archive"),  # a grounded bracket, with no provider
        Requirement(1, "archive"),  # type: ignore[arg-type]
        Requirement("route", "archive"),
        Requirement("d:route", "maybe"),  # does not conform
        Requirement("d:order", {"status": "open"}),  # a field missing
        Requirement("d:order", {"status": "open", "n": 2, "x": 1}),  # an undeclared field
        Requirement("d:blob", "AAE"),  # a bytes type can never hold
    ],
)
def test_rule_6_a_requirement_on_a_provided_d_ref_that_can_hold(requirement: Requirement) -> None:
    assert rule_break(execute(Step("archive", {}, [requirement]))) == ["rule:6", "step:1"]


def deep(n: int) -> object:
    value: object = 0
    for _ in range(n):
        value = [value]
    return value


@pytest.mark.parametrize(
    "plan",
    [
        execute(archive(x=Origin(literal=b"bytes"))),
        execute(archive(x=Origin(literal={1, 2}))),
        execute(archive(x=Origin(literal=float("nan")))),
        execute(archive(x=Origin(literal=2**53))),
        execute(archive(x=Origin(literal=deep(65)))),
        execute(archive(x=Origin(literal="\ud800"))),
        execute(Step("archive", {"\ud800": Origin(literal=1)}, [R])),
    ],
)
def test_rule_7_literals_and_argument_names_are_in_the_value_domain(plan: Plan) -> None:
    assert rule_break(plan) == ["rule:7", "step:1"]


def test_a_literal_64_deep_keeps_rule_7() -> None:
    record, _, _ = returning(execute(archive(x=Origin(literal=deep(64)))))
    assert cause_of(record) is None


# ---------------------------------------------------------------------------------------------
# The fixed order


@pytest.mark.parametrize(
    ("plan", "first"),
    [
        # Plan-level before any step.
        (Plan("clarify", [Step("delete")], reply=" "), ["rule:1"]),
        (Plan("execute", [Step("delete")], reply="x"), ["rule:5"]),
        (Plan("execute", [Step("delete")], reason="\ud800"), ["rule:7"]),
        (Plan("escalate", reason="\ud800 "), ["rule:7"]),
        (Plan("inform", reply="  \ud800"), ["rule:7"]),  # a surrogate is not whitespace
        # Within a step: 2, 3, 4, 6, then 7.
        (execute(Step("delete", {"x": Binding(1)}, [R])), ["rule:2", "step:1"]),
        (execute(Step("delete", {"x": Origin(ref="d:none")}, [R])), ["rule:3", "step:1"]),
        (
            execute(Step("archive", {"x": Origin(ref="d:none")}, [Requirement("d:none", 1)])),
            ["rule:4", "step:1"],
        ),
        (
            execute(Step("archive", {"x": Origin(literal=1)}, [Requirement("d:none", 1)])),
            ["rule:6", "step:1"],
        ),
        # Within a step, rule 7 before rule 6.
        (
            execute(Step("archive", {"x": Origin(literal=b"")}, [Requirement("d:none", 1)])),
            ["rule:7", "step:1"],
        ),
        # Each step by position: step 1's later rule before step 2's earlier one.
        (execute(archive(x=Origin(literal=b"")), archive(x=Binding(9))), ["rule:7", "step:1"]),
        (execute(archive(x=Binding(1)), Step("delete", {}, [R])), ["rule:2", "step:1"]),
        (execute(archive(), Step("delete", {}, [R]), archive(x=Binding(9))), ["rule:3", "step:2"]),
    ],
)
def test_the_cause_names_the_first_break_in_the_fixed_order(plan: Plan, first: list[str]) -> None:
    assert rule_break(plan) == first


# ---------------------------------------------------------------------------------------------
# The order of outcomes: a stop, then a raise, then the return


def caught(*reads: str, then: Callable[[Any], object]) -> Policy_:
    def policy(r: Any) -> object:
        for ref in reads:
            with contextlib.suppress(BaseException):
                r.require(ref)
        return then(r)

    return policy


def test_a_caught_undeclared_read_wins_over_a_valid_plan() -> None:
    record, returned, verdict = run(caught("d:info", then=const(GOOD)))
    assert cause_of(record) == broken("undeclared-read", "policy", "d:info")
    assert (returned.kind, returned.steps) == (PlanKind.broken, ())
    assert verdict.kind is VerdictKind.passed


def test_a_caught_require_stop_wins_over_a_valid_plan() -> None:
    record, returned, _ = run(caught("d:gone", then=const(GOOD)))
    assert cause_of(record) == broken("required", "d:gone")
    assert returned.kind is PlanKind.escalate
    assert (returned.reason, returned.reply, returned.steps) == (None, None, ())


def test_a_raise_after_a_caught_stop_loses_to_it_and_its_text_is_kept() -> None:
    def then(r: Any) -> object:
        raise ValueError("after the stop")

    record, _, _ = run(caught("d:gone", then=then))
    assert cause_of(record) == broken("required", "d:gone")
    kept = record.unhashed["exceptions"][plan_index(record)]
    assert kept["type"] == "ValueError"
    assert kept["message"] == "after the stop"
    assert "ValueError: after the stop" in kept["traceback"]
    assert "reads" not in record.unhashed


# ---------------------------------------------------------------------------------------------
# An awaitable returned


async def async_policy(r: Any) -> object:
    return GOOD


def coroutine_policy(r: Any) -> object:
    return async_policy(r)


@pytest.fixture
def unruled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan rules fail if reached: an awaitable is refused before them."""

    def never(self: object, returned: object) -> list[str] | None:
        raise AssertionError("the return reached the plan rules")

    monkeypatch.setattr(_compile._Rules, "first_break", never)


@pytest.mark.usefixtures("unruled")
@pytest.mark.parametrize("policy", [async_policy, coroutine_policy], ids=["async", "sync"])
def test_a_policy_returning_a_coroutine_is_invalid_return_and_no_warning_leaks(
    policy: Policy_,
) -> None:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        record, returned, verdict = run(policy)
        gc.collect()
    assert cause_of(record) == invalid("policy")
    assert returned.kind is PlanKind.broken
    assert verdict.kind is VerdictKind.passed
    assert "exceptions" not in record.unhashed
    assert [str(w.message) for w in caught_warnings] == []


@pytest.mark.usefixtures("unruled")
@pytest.mark.parametrize("make_", [NotACoroutine, InheritsAwait], ids=["own", "inherited"])
def test_a_policy_returning_an_awaitable_by_its_type_is_invalid_return_and_not_closed(
    make_: Callable[[], NotACoroutine],
) -> None:
    awaitable = make_()
    record, _, _ = returning(awaitable)
    assert cause_of(record) == invalid("policy")
    assert "exceptions" not in record.unhashed
    assert not awaitable.closed


@pytest.mark.usefixtures("unruled")
def test_a_policys_generator_coroutine_is_invalid_return_and_closed_at_the_close() -> None:
    ran: list[str] = []
    generator = generator_coroutine(ran, started=True)
    record, _, _ = returning(generator)
    assert cause_of(record) == invalid("policy")
    assert inspect.getgeneratorstate(generator) == inspect.GEN_CLOSED
    assert ran == ["finally"]


@pytest.mark.parametrize("make_", [Claims, MetaRaising], ids=["claims", "metaclass"])
def test_a_policy_return_that_is_no_awaitable_by_its_type_meets_the_plan_rules(
    make_: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []
    first_break = _compile._Rules.first_break

    def spy(self: Any, returned: object) -> list[str] | None:
        seen.append(returned)
        return first_break(self, returned)

    monkeypatch.setattr(_compile._Rules, "first_break", spy)
    returned = make_()
    record, _, _ = returning(returned)
    assert cause_of(record) == invalid("policy")
    assert seen == [returned]
    assert "exceptions" not in record.unhashed


def test_a_policys_coroutine_re_raising_a_kept_stop_on_close_is_invalid_return() -> None:
    record, _, _ = run(lambda r: started_raising_on_close(stop()))
    assert cause_of(record) == invalid("policy")
    assert "exceptions" not in record.unhashed


def test_a_policy_return_whose_awaitable_test_raises_a_kept_stop_meets_the_rules() -> None:
    record, _, _ = run(lambda r: clashing(stop()))
    assert cause_of(record) == invalid("policy")
    assert "exceptions" not in record.unhashed


def test_a_policy_return_whose_awaitable_test_raises_is_the_policys_raise() -> None:
    record, returned, _ = run(lambda r: clashing())
    assert cause_of(record) == broken("raised", "policy")
    assert returned.kind is PlanKind.broken
    kept = record.unhashed["exceptions"][plan_index(record)]
    assert (kept["type"], kept["message"]) == ("ValueError", "the type test raised")


def test_a_policys_coroutine_is_closed_when_an_undeclared_read_wins() -> None:
    kept: list[Any] = []

    def then(r: Any) -> object:
        kept.append(async_policy(r))
        return kept[0]

    record, _, _ = run(caught("d:info", then=then))
    assert cause_of(record) == broken("undeclared-read", "policy", "d:info")
    assert inspect.getcoroutinestate(kept[0]) == inspect.CORO_CLOSED


def test_a_policys_coroutine_raising_on_close_keeps_the_outcome_and_its_text() -> None:
    record, _, _ = run(lambda r: started_raising_on_close("on close"))
    assert cause_of(record) == invalid("policy")
    kept = record.unhashed["exceptions"][plan_index(record)]
    assert (kept["type"], kept["message"]) == ("ValueError", "on close")


def test_an_undeclared_read_outranks_an_earlier_require_stop() -> None:
    record, _, _ = run(caught("d:gone", "d:info", then=const(GOOD)))
    assert cause_of(record) == broken("undeclared-read", "policy", "d:info")


def test_only_the_first_undeclared_read_counts() -> None:
    record, _, _ = run(caught("d:nothing", "d:info", then=const(GOOD)))
    assert cause_of(record) == broken("undeclared-read", "policy")
    assert record.unhashed["reads"] == {plan_index(record): "d:nothing"}


@pytest.mark.parametrize(
    ("read", "named"),
    [
        ("d:info", True),
        ("t:state", True),
        ("t:state.to", True),
        ("d:order.n", True),
        ("d:nothing", False),
        ("t:state.none", False),
        ('d:info["x"]', False),
        ("q:x", False),
        ("not a ref", False),
    ],
)
def test_an_undeclared_read_names_the_read_only_when_the_design_holds_it(
    read: str, named: bool
) -> None:
    record, _, _ = run(lambda r: r.read(read))
    if named:
        assert cause_of(record) == broken("undeclared-read", "policy", read)
        assert "reads" not in record.unhashed
    else:
        assert cause_of(record) == broken("undeclared-read", "policy")
        assert record.unhashed["reads"] == {plan_index(record): read}


def test_an_undeclared_read_of_a_non_string_puts_its_repr_in_reads() -> None:
    record, _, _ = run(lambda r: r.read(5))
    assert cause_of(record) == broken("undeclared-read", "policy")
    assert record.unhashed["reads"] == {plan_index(record): "5"}


@pytest.mark.parametrize(("path", "named"), [("other", True), ("nope", False)])
def test_an_undeclared_parameter_is_named_only_when_it_is_a_declared_leaf(
    path: str, named: bool
) -> None:
    record, _, _ = run(lambda r: r.param(path))
    refs = ["policy", path] if named else ["policy"]
    assert cause_of(record) == broken("undeclared-read", *refs)


def test_the_policy_reads_its_declared_reads_and_parameters() -> None:
    seen: list[object] = []

    def policy(r: Any) -> object:
        seen.extend([r.read("d:order.status"), r.read("t:state.subject"), r.param("folder")])
        seen.append(r.read("d:blob"))
        return Plan("execute")

    record, _, _ = run(policy)
    assert seen == ["open", "hello", "spam", b"\x00\x01"]
    assert cause_of(record) is None


def test_require_on_a_field_path_names_the_read_as_given() -> None:
    record, returned, _ = run(lambda r: r.require("d:lost.status"))
    assert cause_of(record) == broken("required", "d:lost.status")
    assert returned.cause == Cause(CauseCode.required, ["d:lost.status"])


def test_a_policy_that_raises_is_broken_raised_with_its_text_kept() -> None:
    def policy(r: Any) -> object:
        raise KeyError("lost")

    record, returned, _ = run(policy)
    assert cause_of(record) == broken("raised", "policy")
    assert returned.cause == Cause(CauseCode.raised, ["policy"])
    assert record.unhashed["exceptions"][plan_index(record)]["type"] == "KeyError"


class _RaisingArguments(Mapping[str, Origin | Binding]):
    def __getitem__(self, name: str) -> Origin | Binding:
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> ItemsView[str, Origin | Binding]:
        raise LookupError("no items")


def test_a_step_whose_arguments_raise_in_the_policy_is_broken_raised() -> None:
    record, returned, _ = run(lambda r: execute(Step("archive", _RaisingArguments())))
    assert cause_of(record) == broken("raised", "policy")
    assert returned.kind is PlanKind.broken
    assert returned.cause == Cause(CauseCode.raised, ["policy"])


def test_a_steps_arguments_from_any_mapping_record_the_same_plan_item() -> None:
    step = GOOD.steps[0]
    args = dict(step.arguments)
    first, _, _ = returning(execute(Step(step.tool, args, step.requirements)))
    assert cause_of(first) is None
    for given in (MappingProxyType(args), step.arguments):
        record, _, _ = returning(execute(Step(step.tool, given, step.requirements)))
        assert encode(plan_item(record)) == encode(plan_item(first))


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(1), asyncio.CancelledError()])
def test_a_base_exception_from_the_policy_propagates_with_no_plan_item(
    exc: BaseException,
) -> None:
    def policy(r: Any) -> object:
        raise exc

    loaded = make(policy)
    run_settings = settings()
    ledger = open_ledger(loaded, STATE, run_settings)
    board = asyncio.run(interpret_on(ledger, loaded, STATE, {}, run_settings))
    with pytest.raises(type(exc)):
        compile_turn(ledger, loaded, board)
    log = ledger.close().log
    assert [i["item"] for i in log if i["item"] in ("plan", "verdict")] == []


# ---------------------------------------------------------------------------------------------
# derivation on every plan item


def test_derivation_lists_the_policys_helper_calls_on_an_escalate_plan() -> None:
    def policy(r: Any) -> object:
        pick_with_margin(r, "d:gone", 0.5, 0.1)  # a marker read is returned unchanged
        return r.require("d:gone")

    record, _, _ = run(policy)
    assert cause_of(record) == broken("required", "d:gone")
    assert plan_item(record)["derivation"] == [
        {"helper": "pick_with_margin", "refs": ["d:gone"], "rested_on": ["d:gone"]}
    ]


def test_derivation_lists_a_helper_that_raised_on_a_broken_plan() -> None:
    record, _, _ = run(lambda r: pick_with_margin(r, "d:route", 0.5, 0.1))
    assert cause_of(record) == broken("raised", "policy")
    assert plan_item(record)["derivation"] == [
        {"helper": "pick_with_margin", "refs": ["d:route"], "rested_on": []}
    ]


def test_derivation_is_kept_on_an_invalid_return_and_a_valid_plan() -> None:
    def policy(then: object) -> Policy_:
        def fn(r: Any) -> object:
            pick_with_margin(r, "d:lost", 0.5, 0.1)
            return then

        return fn

    item = {"helper": "pick_with_margin", "refs": ["d:lost"], "rested_on": ["d:lost"]}
    record, _, _ = run(policy(None))
    assert (cause_of(record), plan_item(record)["derivation"]) == (invalid("policy"), [item])
    record, _, _ = run(policy(GOOD))
    assert (cause_of(record), plan_item(record)["derivation"]) == (None, [item])


def test_a_helper_stopped_at_an_undeclared_read_lists_no_item() -> None:
    record, _, _ = run(lambda r: pick_with_margin(r, "q:spam", 0.5, 0.1))
    assert cause_of(record) == broken("undeclared-read", "policy")
    assert plan_item(record)["derivation"] == []


def test_no_plan_item_holds_rows_or_slots() -> None:
    def require(r: Any) -> object:
        return r.require("d:gone")

    for policy in (const(GOOD), const(None), require):
        item = plan_item(run(policy)[0])
        assert "rows" not in item
        assert "slots" not in item


# ---------------------------------------------------------------------------------------------
# The guard after compile


def test_a_refused_verdict_keeps_the_policys_plan_and_reads_the_designs_guards() -> None:
    keep = Requirement("d:route", "keep")
    plan = execute(Step("archive", {}, [keep]), Step("send"))
    record, returned, verdict = returning(plan, requirements=[Requirement("d:order", ORDER)])
    assert returned == plan
    assert verdict.failures == (GuardFailure(1, keep, "d:route"), GuardFailure(2, None, None))
    assert record.log[-1] == {
        "item": "verdict",
        "kind": "refused",
        "failures": [
            {"step": 1, "requirement": {"ref": "d:route", "value": "keep"}, "ref": "d:route"},
            {"step": 2, "requirement": None, "ref": None},
        ],
    }
    _, _, verdict = returning(plan, requirements=[Requirement("d:gone", True)], default_deny=False)
    assert verdict.failures == (
        GuardFailure("plan", Requirement("d:gone", True), "d:gone"),
        GuardFailure(1, keep, "d:route"),
    )


def test_a_compile_built_plan_passes_the_guard_whatever_it_requires() -> None:
    record, _, verdict = run(const(None), requirements=[Requirement("d:gone", True)])
    assert verdict == Verdict(VerdictKind.passed)
    assert record.log[-1] == {"item": "verdict", "kind": "passed", "failures": []}


# ---------------------------------------------------------------------------------------------
# What compile writes


def test_compile_appends_the_plan_then_the_verdict_and_writes_no_board_entry() -> None:
    seen: list[tuple[int, str]] = []
    loaded = make(const(GOOD))
    run_settings = settings()
    ledger = open_ledger(
        loaded, STATE, run_settings, lambda i, item: seen.append((i, item["item"]))
    )
    board = asyncio.run(interpret_on(ledger, loaded, STATE, {}, run_settings))
    before = copy.deepcopy(board)
    del seen[:]
    compile_turn(ledger, loaded, board)
    assert [kind for _, kind in seen] == ["plan", "verdict"]
    assert board == before


def test_the_returned_plan_keeps_live_types_where_the_records_bytes_do_not() -> None:
    # A float stays a float in the stored copy; the bytes write it as 1, which reads back as
    # an int.
    record, returned, _ = returning(execute(archive(x=Origin(literal=1.0))))
    literal = returned.steps[0].arguments["x"]
    assert isinstance(literal, Origin)
    assert type(literal.literal) is float
    assert type(plan_item(record)["steps"][0]["arguments"]["x"]["literal"]) is int


# ---------------------------------------------------------------------------------------------
# Caller code compile meets: no stop and no raise escapes it (probe6)


class Flip:
    """A requirement ref whose ``__class__`` raises once it has been read once."""

    reads = 0

    @property
    def __class__(self) -> type:  # type: ignore[override]
        Flip.reads += 1
        if Flip.reads > 1:
            raise RuntimeError("caller code in compile")
        return Flip


def test_rule_6_tests_a_requirement_ref_by_its_type_alone() -> None:
    Flip.reads = 0

    def policy(r: Any) -> object:
        return execute(Step("archive", {}, [Requirement(cast(Any, Flip()), 1)]))

    record, _, _ = run(policy)
    assert cause_of(record) == invalid("rule:6", "step:1")


class Twin(str):
    __slots__ = ()

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return self is other


def test_two_argument_names_of_one_string_data_break_rule_7() -> None:
    step = Step("archive", {"a": Origin(literal=1), Twin("a"): Origin(literal=2)}, [R])
    assert rule_break(execute(step)) == ["rule:7", "step:1"]


class Evil(str):
    """A name whose hash is ``idempotency_key``'s and whose hash and comparison raise once
    armed: compile reads it by its type and string data alone."""

    __slots__ = ()
    armed = False

    def __hash__(self) -> int:
        if Evil.armed:
            raise RuntimeError("hashed")
        return hash("idempotency_key")

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("compared")


def test_a_name_whose_hash_and_comparison_raise_runs_neither_in_compile() -> None:
    given: dict[str, Any] = {"a": Origin(literal=1), Evil("a"): Origin(literal=2)}
    keyed: dict[str, Any] = {Evil("idempotency_key"): Origin(literal=1)}
    Evil.armed = True
    try:
        step = Step("archive", given, [R])
        assert rule_break(execute(step)) == ["rule:7", "step:1"]
        assert [type(k) for k in step.arguments] == [str, Evil]
        alone = Step("archive", keyed, [R])
        assert rule_break(execute(alone)) == ["rule:3", "step:1"]
        assert [type(k) for k in alone.arguments] == [str]
    finally:
        Evil.armed = False


def test_twinned_arguments_are_read_by_position_and_string_data() -> None:
    one, two = Origin(literal=1), Origin(literal=2)
    arguments = Step("archive", {"a": one, Twin("a"): two}).arguments
    assert list(arguments.values()) == [one, two]
    assert [(string_data(n), o) for n, o in arguments.items()] == [("a", one), ("a", two)]
    assert arguments["a"] is one
    assert "a" in arguments
    assert "b" not in arguments
    assert 1 not in arguments
    assert two in arguments.values()
    assert Origin(literal=3) not in arguments.values()
    assert arguments == arguments
    assert arguments != {"a": one}
    with pytest.raises(KeyError):
        arguments["b"]
    with pytest.raises(TypeError):
        hash(arguments)
    with pytest.raises(TypeError):
        copy.deepcopy(arguments)


def test_an_exception_whose_text_reads_the_kept_view_gives_no_stop() -> None:
    held: dict[str, Any] = {}

    class Sneaky(Exception):
        def __str__(self) -> str:
            held["r"].read("d:info")
            return "x"

    def policy(r: Any) -> object:
        held["r"] = r
        raise Sneaky()

    record, _, _ = run(policy)
    assert cause_of(record) == broken("raised", "policy")
    kept = record.unhashed["exceptions"][plan_index(record)]
    assert (
        kept["type"]
        == "test_an_exception_whose_text_reads_the_kept_view_gives_no_stop.<locals>.Sneaky"
    )
    assert kept["message"] == UNREPRESENTABLE
    assert "reads" not in record.unhashed


def test_exception_text_that_raises_a_stop_is_unrepresentable() -> None:
    class Stopping(Exception):
        def __str__(self) -> str:
            raise _UndeclaredStop(UndeclaredRead("d:x", True, Access.read))

    assert exception_record(Stopping())["message"] == UNREPRESENTABLE


def test_a_view_kept_past_its_call_raises_an_ordinary_exception() -> None:
    held: dict[str, Any] = {}

    def keep(r: Any) -> object:
        held["r"] = r
        return Plan("execute")

    run(keep)
    with pytest.raises(ViewClosed):
        held["r"].read("d:route")
    with pytest.raises(ViewClosed):
        held["r"].require("d:gone")
    with pytest.raises(ViewClosed):
        held["r"].param("folder")
    assert held["r"].stops == ()


def test_a_rules_view_read_inside_the_policys_call_is_the_policys_raise() -> None:
    held: dict[str, Any] = {}

    def keep(r: Any) -> object:
        held["r"] = r
        return "archive"

    def policy(r: Any) -> object:
        return held["r"].read("d:route")

    rules = {**RULES, "route": Rule("d:route", t.label("archive", "keep"), keep, version="1")}
    loaded = load(
        minimal(
            state=STATE_T,
            rules=rules,
            policy=Policy(policy, version="1", reads=READS),
            tools=["archive"],
        )
    )
    record, _, _ = compiled(loaded, {}, STATE)
    assert cause_of(record) == broken("raised", "policy")
    assert record.unhashed["exceptions"][plan_index(record)]["type"] == "ViewClosed"
    assert plan_item(record)["derivation"] == []


def test_an_entry_that_does_not_hold_its_path_is_the_engines_defect() -> None:
    with pytest.raises(_ViewError, match="does not hold its path"):
        at({"d:x": {"a": 1}}, "d:x.b")
    with pytest.raises(_ViewError, match="does not hold its path"):
        at({"d:x": 5}, "d:x.b")
    assert at({"d:x": None}, "d:x.b") is None


# ---------------------------------------------------------------------------------------------
# The observer raising at the plan item's append


class Seen(Exception):
    pass


def observed_at_plan(policy: Policy_) -> tuple[Record, BaseException | None]:
    def observer(index: int, item: dict[str, Any]) -> None:
        if item["item"] == "plan":
            raise Seen("at the plan")

    loaded = make(policy)
    run_settings = settings()
    ledger = open_ledger(loaded, STATE, run_settings, observer)
    board = asyncio.run(interpret_on(ledger, loaded, STATE, {}, run_settings))
    with pytest.raises(Seen):
        compile_turn(ledger, loaded, board)
    return ledger.close(), ledger.observer_error


def test_the_observer_raising_at_the_plan_append_writes_no_text_and_no_verdict() -> None:
    def policy(r: Any) -> object:
        raise ValueError("lost")

    record, mark = observed_at_plan(policy)
    assert isinstance(mark, Seen)
    assert record.log[-1]["item"] == "plan"
    assert "exceptions" not in record.unhashed


def test_the_observer_raising_at_the_plan_append_writes_no_unnamed_read() -> None:
    record, _ = observed_at_plan(lambda r: r.read("d:nothing"))
    assert cause_of_last(record) == broken("undeclared-read", "policy")
    assert "reads" not in record.unhashed


def cause_of_last(record: Record) -> dict[str, Any] | None:
    assert record.log[-1]["item"] == "plan"
    return record.log[-1]["cause"]


# ---------------------------------------------------------------------------------------------
# Each item's unhashed members come after its append and before the next


def test_compile_writes_the_plan_items_text_before_the_verdict_append() -> None:
    def policy(r: Any) -> object:
        with contextlib.suppress(BaseException):
            r.read("d:nothing")
        raise ValueError("lost")

    seen: dict[int, set[str]] = {}
    box: list[Any] = []

    def observer(index: int, item: dict[str, Any]) -> None:
        if not box:
            return  # open_ledger's own appends
        unhashed = box[0]._unhashed
        seen[index] = {f"{m}/{k}" for m in ("exceptions", "reads") for k in unhashed.get(m, {})}

    loaded = make(policy)
    run_settings = settings()
    ledger = open_ledger(loaded, STATE, run_settings, observer)
    box.append(ledger)
    board = asyncio.run(interpret_on(ledger, loaded, STATE, {}, run_settings))
    compile_turn(ledger, loaded, board)
    plan = len(ledger.close().log) - 2
    assert seen[plan] == set()
    assert seen[plan + 1] == {f"exceptions/{plan}", f"reads/{plan}"}


def test_the_guard_reads_loads_copy_of_a_plan_level_requirement_not_the_callers() -> None:
    # A caller changing a nested value after load changes neither the hash nor the guard.
    value: dict[str, Any] = {"status": "open", "n": 2}
    loaded = make(const(execute()), requirements=[Requirement("d:order", value)])
    before = loaded.design_hash
    value["status"] = "shut"
    record, _, verdict = compiled(loaded, {}, STATE)
    assert loaded.design_hash == before
    assert verdict.kind is VerdictKind.passed
    assert record.log[-1]["failures"] == []
    value["n"] = {1, 2}  # outside the domain: the guard never sees it
    assert compiled(loaded, {}, STATE)[2].kind is VerdictKind.passed


# ---------------------------------------------------------------------------------------------
# Rule 7 holds each text, literal, the argument names and requirement value to the value
# domain's size bounds


def _doubled(n: int) -> Any:
    v: Any = 0
    for _ in range(n):
        v = [v, v]
    return v


# Under small bounds (the small_bounds fixture): 2^6 values, 2^8 code points.
_BIG = "x" * 257


@pytest.mark.parametrize(
    ("plan", "first"),
    [
        (Plan("escalate", reason=_BIG), ["rule:7"]),
        (Plan("inform", reply=_BIG), ["rule:7"]),
        (execute(archive(x=Origin(literal=_doubled(60)))), ["rule:7", "step:1"]),
        (execute(archive(x=Origin(literal=["x" * 16] * 17))), ["rule:7", "step:1"]),
        (execute(Step("archive", {_BIG: Origin(literal=1)}, [R])), ["rule:7", "step:1"]),
    ],
    ids=["reason", "reply", "shared-literal", "rows", "argument-name"],
)
def test_rule_7_refuses_a_plan_over_the_size_bounds_fast(
    plan: Plan, first: list[str], small_bounds: SmallBounds
) -> None:
    with counted() as n:
        assert rule_break(plan) == first
    assert n[0] < 4 * bound()  # each walk stops past the bound


def test_each_argument_name_is_its_own_value(small_bounds: SmallBounds) -> None:
    # 100 names: each within the bounds; as one object of names they would hold 2^6 and more.
    many = execute(Step("archive", {f"a{i}": Origin(literal=i) for i in range(100)}, [R]))
    record, _, _ = returning(many)
    assert cause_of(record) is None
    assert len(plan_item(record)["steps"][0]["arguments"]) == 100
    assert read_record(record.document).document == record.document


@pytest.mark.parametrize(
    "requirement",
    [
        Requirement("d:route", b"archive"),  # can never hold, and outside the domain
        Requirement("d:order", cyclic()),  # an undeclared field, and outside the domain
    ],
)
def test_a_requirement_breaking_rules_6_and_7_is_rule_7(requirement: Requirement) -> None:
    assert rule_break(execute(Step("archive", {}, [requirement]))) == ["rule:7", "step:1"]


def test_a_shared_oversized_requirement_value_returns_promptly(small_bounds: SmallBounds) -> None:
    # Rule 6 would walk it against its type at every place; rule 7's size test comes first.
    shared = _doubled(60)
    with counted() as n:
        assert rule_break(execute(Step("archive", {}, [Requirement("d:route", shared)]))) == [
            "rule:7",
            "step:1",
        ]
    assert n[0] < 4 * bound()


def test_each_argument_name_is_size_bounded_as_its_own_value(small_bounds: SmallBounds) -> None:
    names: dict[str, Any] = {f"{i:016d}": Origin(literal=i) for i in range(17)}  # 272 in all
    record, _, _ = returning(execute(Step("archive", names, [R])))
    assert cause_of(record) is None


class _ArmedTwin(str):
    """A name of ``idempotency_key``'s string data whose comparison raises once armed."""

    __slots__ = ()
    armed = False

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        if _ArmedTwin.armed:
            raise RuntimeError("compared")
        return self is other


def test_twinned_idempotency_key_names_break_rule_3_first() -> None:
    given: dict[str, Any] = {
        "idempotency_key": Origin(literal=1),
        _ArmedTwin("idempotency_key"): Origin(literal=2),
    }
    _ArmedTwin.armed = True
    try:
        step = Step("archive", given, [R])
        assert rule_break(execute(step)) == ["rule:3", "step:1"]
    finally:
        _ArmedTwin.armed = False


def test_twinned_items_hold_a_pair_by_position() -> None:
    one, two = Origin(literal=1), Origin(literal=2)
    items: Any = Step("archive", {"a": one, Twin("a"): two}).arguments.items()
    assert ("a", one) in items
    assert ("a", two) in items
    assert ("a", Origin(literal=1)) not in items  # an origin by identity, never by ==
    assert ("b", one) not in items
    assert (1, one) not in items
    assert ("a",) not in items
    assert ["a", one] not in items
