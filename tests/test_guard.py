# pyright: reportPrivateUsage=false
"""The guard and its types, ``Verdict`` and ``GuardFailure``, which check themselves."""

import copy
from collections.abc import Callable
from typing import Any

import pytest

from jurat import (
    Cause,
    CauseCode,
    GuardFailure,
    Plan,
    PlanKind,
    Requirement,
    Step,
    Unasked,
    Unavailable,
    Unsure,
    Verdict,
    VerdictKind,
)
from jurat._canonical import decode, encode
from jurat._guard import guard, verdict_item
from jurat._ledger import Ledger
from ledger_logs import NORMAL, PLAN, fresh


def bad(value: object) -> Any:
    """A value of the wrong type, passed on purpose."""
    return value


def step(*requirements: Requirement, tool: str = "archive") -> Step:
    return Step(tool, {}, list(requirements))


def execute(*steps: Step) -> Plan:
    return Plan("execute", list(steps))


def refs(verdict: Verdict) -> list[str]:
    cause = verdict.cause
    assert cause is not None
    assert cause.code is CauseCode.refused
    return list(cause.refs)


ROUTE = Requirement("d:route", "archive")
OK = Requirement("d:ok", True)
BOARD: dict[str, object] = {"d:route": "archive", "d:ok": True, "d:eligible": "no"}


# ---------------------------------------------------------------------------------------------
# What passes and what fails, in order


def test_an_execute_plan_whose_requirements_hold_passes() -> None:
    verdict = guard(execute(step(ROUTE), step(OK, ROUTE)), BOARD, [OK], True)
    assert verdict == Verdict(VerdictKind.passed)
    assert verdict.cause is None


def test_every_failure_is_listed_plan_level_first_then_each_step_in_order() -> None:
    eligible = Requirement("d:eligible", "yes")
    missing = Requirement("d:none", 1)
    plan = execute(step(ROUTE), step(eligible, missing, ROUTE), step())
    verdict = guard(plan, BOARD, [Requirement("d:ok", False), missing], True)
    assert verdict.kind is VerdictKind.refused
    assert verdict.failures == (
        GuardFailure("plan", Requirement("d:ok", False), "d:ok"),
        GuardFailure("plan", missing, "d:none"),
        GuardFailure(2, eligible, "d:eligible"),
        GuardFailure(2, missing, "d:none"),
        GuardFailure(3, None, None),
    )
    assert refs(verdict) == ["d:ok", "d:none", "d:eligible", "d:none", "step:3"]


def test_a_requirement_with_no_entry_fails_naming_its_ref_and_never_raises() -> None:
    verdict = guard(execute(step(Requirement("d:absent", "x"))), {}, [], True)
    assert verdict.failures == (GuardFailure(1, Requirement("d:absent", "x"), "d:absent"),)


@pytest.mark.parametrize("marker", [Unsure, Unavailable, Unasked])
def test_a_marker_never_satisfies_a_requirement(marker: type[Unsure]) -> None:
    # The backstop: a required estimate that is undetermined refuses.
    board: dict[str, object] = {"d:route": marker(Cause(CauseCode.raised, ["route"]))}
    verdict = guard(execute(step(ROUTE)), board, [ROUTE], True)
    assert refs(verdict) == ["d:route", "d:route"]


@pytest.mark.parametrize(
    ("entry", "declared", "holds"),
    [
        (True, 1, False),
        (1, True, False),
        (0, False, False),
        (1, 1.0, True),
        (1.0, 1, True),
        ({"a": 1, "b": [2]}, {"b": [2], "a": 1}, True),
        ([1, "x"], (1, "x"), True),
        ((1, "x"), [1, "x"], True),
        (None, None, True),
        ("1", 1, False),
        ({"a": None}, {}, False),
        (b"\x00", "AA", False),  # a bytes entry has no encoding to compare
    ],
)
def test_a_requirement_holds_by_canonical_equality(entry: Any, declared: Any, holds: bool) -> None:
    requirement = Requirement("d:x", declared)
    verdict = guard(execute(step(requirement)), {"d:x": entry}, [], True)
    assert (verdict.kind is VerdictKind.passed) is holds


# ---------------------------------------------------------------------------------------------
# Default-deny and the plans that are not execute


def test_default_deny_refuses_a_step_with_no_requirement_of_its_own() -> None:
    # A plan-level requirement does not count, and the failure stands in the step's place.
    verdict = guard(execute(step(ROUTE), step(), step(OK)), BOARD, [OK], True)
    assert verdict.failures == (GuardFailure(2, None, None),)
    assert refs(verdict) == ["step:2"]


def test_a_step_whose_requirements_fail_has_no_default_deny_failure_too() -> None:
    eligible = Requirement("d:eligible", "yes")
    verdict = guard(execute(step(eligible)), BOARD, [], True)
    assert verdict.failures == (GuardFailure(1, eligible, "d:eligible"),)


def test_default_deny_off_lets_a_step_with_no_requirement_pass() -> None:
    assert guard(execute(step(), step(ROUTE)), BOARD, [], False).kind is VerdictKind.passed
    assert guard(execute(step()), {}, [OK], False).failures == (GuardFailure("plan", OK, "d:ok"),)


def test_an_execute_plan_with_no_steps_is_checked_against_the_plan_level_requirements() -> None:
    assert guard(execute(), BOARD, [OK], True).kind is VerdictKind.passed
    assert refs(guard(execute(), {}, [OK], True)) == ["d:ok"]


@pytest.mark.parametrize(
    "plan",
    [
        Plan("broken"),
        Plan("escalate", reason="hand on"),
        Plan("clarify", reply="which?"),
        Plan("inform", reply="done"),
        # Rule 1 keeps steps off these kinds; the guard checks only execute plans anyway.
        Plan("escalate", [Step("delete_all")], reason="x"),
    ],
)
def test_a_plan_that_is_not_execute_passes_whatever_the_board(plan: Plan) -> None:
    # A failing plan-level requirement, default-deny on and the empty board of a broken entry.
    failing = [Requirement("d:none", True)]
    assert guard(plan, {}, failing, True) == Verdict(VerdictKind.passed)


def test_the_guard_changes_neither_the_plan_nor_the_board() -> None:
    board: dict[str, object] = {"d:route": {"a": [1, 2]}, "d:ok": True}

    def build() -> Plan:
        return execute(step(Requirement("d:route", {"a": [1, 2]})), step())

    plan = build()
    before = copy.deepcopy(board)
    guard(plan, board, [OK], True)
    assert (board, plan) == (before, build())


# ---------------------------------------------------------------------------------------------
# The verdict item and the round trip


def test_the_verdict_items_bytes() -> None:
    plan = execute(step(Requirement("d:eligible", "yes")), step())
    refused = guard(plan, {"d:eligible": "no"}, [OK], True)
    assert encode(verdict_item(refused)) == (
        b'{"failures":[{"ref":"d:ok","requirement":{"ref":"d:ok","value":true},"step":"plan"},'
        b'{"ref":"d:eligible","requirement":{"ref":"d:eligible","value":"yes"},"step":1},'
        b'{"ref":null,"requirement":null,"step":2}],"item":"verdict","kind":"refused"}'
    )
    assert encode(verdict_item(Verdict("passed"))) == (
        b'{"failures":[],"item":"verdict","kind":"passed"}'
    )


def test_the_ledger_takes_the_verdict_item_and_a_verdict_built_from_it_has_its_cause() -> None:
    plan = execute(step(Requirement("d:eligible", "yes")), step(), step(ROUTE))
    live = guard(plan, BOARD, [OK, Requirement("d:none", [1, {"a": None}])], True)
    ledger = Ledger()
    for item in fresh([*NORMAL[:12], PLAN]):
        ledger.append(item)
    ledger.append(verdict_item(live))
    recorded = decode(ledger.close().document)["log"][-1]
    rebuilt = Verdict(
        recorded["kind"],
        [
            GuardFailure(
                f["step"],
                None if f["requirement"] is None else Requirement(**f["requirement"]),
                f["ref"],
            )
            for f in recorded["failures"]
        ],
    )
    assert rebuilt == live
    assert rebuilt.cause == live.cause
    assert refs(rebuilt) == ["d:none", "d:eligible", "step:2"]


# ---------------------------------------------------------------------------------------------
# Verdict and GuardFailure check themselves


def test_a_verdicts_kind_is_a_member_or_its_string() -> None:
    assert Verdict("passed").kind is VerdictKind.passed
    assert Verdict(Name("passed")).kind is VerdictKind.passed
    failure = GuardFailure(1, None, None)
    assert Verdict(VerdictKind.refused, [failure]).failures == (failure,)


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (lambda: Verdict("maybe"), ValueError, "not passed or refused"),
        (lambda: Verdict(bad(1)), ValueError, "not passed or refused"),
        (lambda: Verdict("refused"), ValueError, "refused exactly when it has failures"),
        (
            lambda: Verdict("passed", [GuardFailure(1, None, None)]),
            ValueError,
            "refused exactly when",
        ),
        (lambda: Verdict("refused", bad("x")), TypeError, "not a list or tuple"),
        (lambda: Verdict("refused", bad([{"step": 1}])), TypeError, "not a GuardFailure"),
        (lambda: GuardFailure(0, None, None), ValueError, "below 1"),
        (lambda: GuardFailure(-1, OK, "d:ok"), ValueError, "below 1"),
        (lambda: GuardFailure(bad("step"), OK, "d:ok"), ValueError, "neither a position"),
        (lambda: GuardFailure(True, None, None), TypeError, "not an int"),
        (lambda: GuardFailure(bad(1.0), None, None), TypeError, "not an int"),
        (lambda: GuardFailure(1, bad({"ref": "d:ok"}), "d:ok"), TypeError, "Requirement"),
        (lambda: GuardFailure(1, OK, bad(1)), TypeError, "not a string or None"),
        (lambda: GuardFailure(1, OK, None), ValueError, "None together"),
        (lambda: GuardFailure(1, None, "d:ok"), ValueError, "None together"),
        (lambda: GuardFailure("plan", None, None), ValueError, "names a step, not the plan"),
        (lambda: GuardFailure(1, OK, "d:other"), ValueError, "not its requirement's"),
    ],
)
def test_a_verdict_or_failure_that_does_not_pair_raises(
    build: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        build()


class Name(str):
    __slots__ = ()


def test_a_failures_string_members_are_stored_as_their_string_data() -> None:
    failure = GuardFailure(Name("plan"), OK, Name("d:ok"))  # type: ignore[arg-type]
    assert (failure.step, failure.ref) == ("plan", "d:ok")
    assert (type(failure.step), type(failure.ref)) == (str, str)


def test_a_passed_plan_kind_is_not_the_verdicts() -> None:
    # The kinds are closed sets of their own.
    with pytest.raises(ValueError, match="not passed or refused"):
        Verdict(PlanKind.execute)


class Unprintable(str):
    """A value whose own repr raises; a str subclass so it reaches the string paths too."""

    __slots__ = ()

    def __repr__(self) -> str:
        raise RuntimeError("repr was called")


class Opaque:
    def __repr__(self) -> str:
        raise RuntimeError("repr was called")


@pytest.mark.parametrize(
    ("build", "error", "message"),
    [
        (
            lambda: GuardFailure(bad(Unprintable("x")), ROUTE, "d:route"),
            ValueError,
            "a failure's step 'x' is neither a position nor \"plan\"",
        ),
        (
            lambda: GuardFailure(bad(Opaque()), ROUTE, "d:route"),
            TypeError,
            'a failure\'s step <Opaque> is not an int or "plan"',
        ),
        (
            lambda: GuardFailure(-(10**5000), ROUTE, "d:route"),
            ValueError,
            "a failure's step integer of 16610 bits is below 1",
        ),
        (
            lambda: GuardFailure(1, bad(Opaque()), "d:route"),
            TypeError,
            "a failure's requirement <Opaque> is not a Requirement",
        ),
        (
            lambda: GuardFailure(1, ROUTE, bad(Opaque())),
            TypeError,
            "a failure's ref <Opaque> is not a string or None",
        ),
        (
            lambda: GuardFailure(1, ROUTE, Unprintable("d:other")),
            ValueError,
            "a failure's ref 'd:other' is not its requirement's",
        ),
        (
            lambda: Verdict(bad(Opaque())),
            ValueError,
            "verdict kind <Opaque> is not passed or refused",
        ),
        (
            lambda: Verdict(Unprintable("maybe")),
            ValueError,
            "verdict kind 'maybe' is not passed or refused",
        ),
        (
            lambda: Verdict("refused", bad(Opaque())),
            TypeError,
            "a verdict's failures <Opaque> are not a list or tuple",
        ),
    ],
    ids=[
        "step-str",
        "step",
        "step-huge",
        "requirement",
        "ref",
        "ref-other",
        "kind",
        "kind-str",
        "failures",
    ],
)
def test_a_bad_value_whose_repr_raises_is_the_usual_error(
    build: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error) as info:
        build()
    assert str(info.value) == message


def test_a_step_past_the_digit_limit_is_named_by_its_bit_length() -> None:
    with pytest.raises(ValueError, match=r"^a failure's step integer of 16610 bits is below 1$"):
        GuardFailure(-(10**5000), ROUTE, "d:route")


class Liar:
    """Claims to be a str through __class__."""

    @property
    def __class__(self) -> type:  # pyright: ignore[reportIncompatibleMethodOverride]
        return str


class ClassRaises:
    @property
    def __class__(self) -> type:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise RuntimeError("__class__ was read")


CLASS_BUILDS: dict[str, Callable[[Any], object]] = {
    "step": lambda v: GuardFailure(v, ROUTE, "d:route"),
    "ref": lambda v: GuardFailure(1, ROUTE, v),
    "cause-ref": lambda v: Cause(CauseCode.raised, [v]),
    "carrier": lambda v: Unsure(v),
}


@pytest.mark.parametrize("make", [Liar, ClassRaises], ids=["liar", "raises"])
@pytest.mark.parametrize("build", CLASS_BUILDS.values(), ids=CLASS_BUILDS.keys())
def test_a_value_is_judged_by_its_type_not_its_class_attribute(
    build: Any, make: Callable[[], object]
) -> None:
    with pytest.raises(TypeError):
        build(make())


def test_each_entry_is_encoded_once_per_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    import jurat._guard as guard_module

    encoded: list[object] = []

    def counting(value: object) -> bytes:
        encoded.append(value)
        return encode(value)

    monkeypatch.setattr(guard_module, "encode", counting)
    plan = execute(step(ROUTE, OK), step(ROUTE), step(Requirement("d:route", "keep")))
    verdict = guard(plan, BOARD, [ROUTE], False)
    assert refs(verdict) == ["d:route"]
    assert len(encoded) == 5 + 2  # each of the five declared values, and each entry once


def test_an_entry_outside_the_domain_fails_every_requirement_on_it() -> None:
    board: dict[str, object] = {"d:route": {1, 2}}
    verdict = guard(execute(step(ROUTE), step(ROUTE)), board, [], False)
    assert refs(verdict) == ["d:route", "d:route"]


def test_a_declared_value_outside_the_domain_fails() -> None:
    verdict = guard(execute(step(Requirement("d:route", bad({1})))), BOARD, [], False)
    assert refs(verdict) == ["d:route"]
