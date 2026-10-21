# pyright: reportPrivateUsage=false
"""The helper library: its version, the log item each call writes on the read view, the
raise on a `when`'s view, and `pick_with_margin`.

The mechanics tests drive `called` through test helpers built on it, as a library helper is.
The other choice helpers are tested in `test_helpers_choice.py`, the combinators in
`test_helpers_combinators.py`.
"""

import inspect
import math
from fractions import Fraction
from typing import Any

import pytest

import jurat
import jurat.helpers
from jurat import Answer, Cause, CauseCode, Unavailable, Unsure
from jurat._read_view import Access, ComponentKind, ReadView, Role, UndeclaredRead, _Stop
from jurat._types import RecordOf
from jurat.helpers import pick_with_margin
from jurat.helpers._call import HelperOnWhen, called

INTENT = Answer("noul", ["yes", "no"], {"yes": 0.7, "no": 0.3})
GONE = Unavailable(Cause(CauseCode.raised, ["d:gone"]))
BOARD: dict[str, object] = {"t:state": {}, "q:intent": INTENT, "d:gone": GONE}


def view(
    kind: ComponentKind = ComponentKind.rule, reads: tuple[str, ...] = ("q:intent", "d:gone")
) -> ReadView:
    name = {"rule": "route", "policy": "policy", "when": "mail"}[kind]
    state = RecordOf(()) if kind is ComponentKind.when else None
    return ReadView(Role(kind, name), reads, (), BOARD, {}, state=state)


def echo(r: ReadView, ref: str) -> object:
    """A test helper over one read: returns it and rests on its ref."""
    with called(r, "echo", [ref]) as call:
        (value,) = call.values
        call.rested_on([ref])
        return value


def boom(r: ReadView, ref: str) -> object:
    """A test helper that reads, then raises."""
    with called(r, "boom", [ref]):
        raise ValueError("a label the answer lacks")


def late(r: ReadView, ref: str) -> object:
    """A test helper that sets ``rested_on``, then raises."""
    with called(r, "late", [ref]) as call:
        call.rested_on([ref])
        raise ValueError("raised after rested_on")


def stopped_inside(r: ReadView, ref: str, inner: str) -> object:
    """A test helper that sets ``rested_on``, then calls a helper whose read stops."""
    with called(r, "stopped_inside", [ref]) as call:
        call.rested_on([ref])
        return echo(r, inner)


def both(r: ReadView, *refs: str) -> tuple[object, ...]:
    """A test helper over several reads, which sees its own item listed."""
    with called(r, "both", list(refs)) as call:
        assert [i.helper for i in r.helper_log()][-1] == "both"
        call.rested_on(list(refs))
        return call.values


def outer(r: ReadView, *refs: str) -> object:
    """A test helper that calls another helper inside itself."""
    with called(r, "outer", list(refs)) as call:
        values = [echo(r, ref) for ref in refs]
        call.rested_on(list(refs[:1]))
        return values[0]


def items(r: ReadView) -> list[tuple[str, tuple[object, ...], tuple[object, ...]]]:
    return [(i.helper, i.refs, i.rested_on) for i in r.helper_log()]


# ---------------------------------------------------------------------------------------------
# The helper version and the package's place


def test_the_helper_version_is_1() -> None:
    assert jurat.helpers.HELPERS == 1
    assert type(jurat.helpers.HELPERS) is int


def test_helpers_is_not_in_the_engines_api() -> None:
    assert "helpers" not in jurat.__all__
    names = {
        "pick_with_margin",
        "pick_with_thresholds",
        "size",
        "mass",
        "unique",
        "agree",
        "first",
        "priority",
        "collect",
        "run_plan",
        "RunRefusal",
        "BindingNotFound",
        "dead_questions",
        "unsure_to_execute",
    }
    # HELPERS is a module attribute the record's design item reads, not an exported name.
    assert {n for n in vars(jurat.helpers) if not n.startswith("_")} == {*names, "HELPERS"}
    assert set(jurat.helpers.__all__) == names
    for name in {*names, "HELPERS"}:
        assert not hasattr(jurat, name)


# ---------------------------------------------------------------------------------------------
# The log item


def test_a_call_is_listed_with_its_name_refs_and_rested_on() -> None:
    r = view()
    assert echo(r, "q:intent") is INTENT
    assert items(r) == [("echo", ("q:intent",), ("q:intent",))]


def test_a_marker_read_comes_back_as_the_same_object() -> None:
    r = view()
    assert echo(r, "d:gone") is GONE
    assert GONE.cause == Cause(CauseCode.raised, ["d:gone"])


def test_a_rule_that_calls_no_helper_has_an_empty_log() -> None:
    def rule(r: Any) -> object:
        return r.read("q:intent")

    r = view()
    rule(r)
    assert r.helper_log() == ()


def test_calls_are_listed_in_call_order() -> None:
    r = view()
    echo(r, "d:gone")
    echo(r, "q:intent")
    assert [i.refs for i in r.helper_log()] == [("d:gone",), ("q:intent",)]


def test_a_helper_that_raised_is_listed_with_rested_on_empty() -> None:
    r = view()
    with pytest.raises(ValueError, match="label"):
        boom(r, "q:intent")
    assert items(r) == [("boom", ("q:intent",), ())]
    echo(r, "q:intent")  # the log goes on after the raise, at the top level
    assert [i.helper for i in r.helper_log()] == ["boom", "echo"]


def test_a_helper_that_raised_after_setting_rested_on_is_listed_with_it_empty() -> None:
    r = view()
    with pytest.raises(ValueError, match="after rested_on"):
        late(r, "q:intent")
    assert items(r) == [("late", ("q:intent",), ())]


def test_a_stop_passing_through_a_helper_that_set_rested_on_leaves_it_empty() -> None:
    r = view()
    with pytest.raises(_Stop):
        stopped_inside(r, "q:intent", "d:other")
    assert r.stops == (UndeclaredRead.of("d:other", Access.read),)
    assert items(r) == [("stopped_inside", ("q:intent",), ())]
    echo(r, "d:gone")  # the nesting depth is restored
    assert [i.helper for i in r.helper_log()] == ["stopped_inside", "echo"]


def test_a_helper_reads_each_ref_in_argument_order_then_lists_its_item() -> None:
    r = view()
    assert both(r, "d:gone", "q:intent") == (GONE, INTENT)
    assert items(r) == [("both", ("d:gone", "q:intent"), ("d:gone", "q:intent"))]


def test_an_undeclared_ref_stops_the_helper_at_its_read_before_any_item() -> None:
    r = view()
    with pytest.raises(_Stop):
        echo(r, "d:other")
    assert r.stops == (UndeclaredRead.of("d:other", Access.read),)
    assert r.helper_log() == ()


def test_an_undeclared_ref_after_a_declared_one_lists_no_item() -> None:
    r = view()
    with pytest.raises(_Stop):
        both(r, "q:intent", "d:other", "d:else")
    assert r.stops == (UndeclaredRead.of("d:other", Access.read),)  # the reads stop at the first
    assert r.helper_log() == ()


def test_refs_are_listed_by_their_string_data() -> None:
    class Sly(str):
        pass

    r = view()
    echo(r, Sly("q:intent"))
    (item,) = r.helper_log()
    assert [type(x) for x in (*item.refs, *item.rested_on)] == [str, str]


def test_only_the_components_own_calls_are_listed() -> None:
    r = view()
    assert outer(r, "q:intent", "d:gone") is INTENT
    assert items(r) == [("outer", ("q:intent", "d:gone"), ("q:intent",))]
    echo(r, "d:gone")
    assert [i.helper for i in r.helper_log()] == ["outer", "echo"]


# ---------------------------------------------------------------------------------------------
# A `when`'s view


def test_a_helper_on_a_whens_view_raises_before_any_read() -> None:
    r = view(ComponentKind.when, reads=())
    with pytest.raises(HelperOnWhen, match="mail") as info:
        echo(r, "d:other")  # an undeclared read, had the helper read it
    assert isinstance(info.value, Exception)
    assert r.stops == ()
    assert r.helper_log() == ()


def test_a_when_that_catches_the_raise_goes_on() -> None:
    r = view(ComponentKind.when, reads=("q:intent",))

    def when(r: ReadView) -> bool:
        try:
            echo(r, "q:intent")
        except Exception:
            return False
        return True

    assert when(r) is False
    assert r.stops == ()


@pytest.mark.parametrize("kind", [ComponentKind.rule, ComponentKind.policy])
def test_a_helper_on_a_rule_or_the_policy_reads(kind: ComponentKind) -> None:
    r = view(kind)
    assert echo(r, "q:intent") is INTENT


# ---------------------------------------------------------------------------------------------
# `pick_with_margin`

PICKS: dict[str, object] = {
    "t:state": {},
    "q:sixty": Answer("noul", ["yes", "no"], {"yes": 0.6, "no": 0.4}),
    "q:nay": Answer("noul", ["yes", "no"], {"yes": 0.1, "no": 0.9}),
    "q:tie": Answer("choice", ["a", "b", "c"], {"a": 0.4, "b": 0.2, "c": 0.4}),
    "q:even": Answer("noul", ["yes", "no"], {"yes": 0.5, "no": 0.5}),
    "q:one": Answer("choice", ["only"], {"only": 1}),
    "q:slip": Answer("choice", ["a", "b", "c"], {"a": 0.8, "b": 0.1, "c": 0.1}),
    "q:second": Answer("choice", ["a", "b", "c"], {"a": 0.5, "b": 0.25, "c": 0.25}),
    "q:levels": Answer("score", ["low", "mid", "high"], {"low": 0.1, "mid": 0.2, "high": 0.7}),
    "d:gone": GONE,
    "t:lookup": "not an answer",
}


def picks() -> ReadView:
    return ReadView(Role(ComponentKind.rule, "route"), tuple(PICKS), (), PICKS, {})


def abstained(ref: str) -> Unsure:
    return Unsure(Cause(CauseCode.abstained, [ref]))


def test_0_6_leads_0_4_by_a_margin_of_0_2() -> None:
    assert 0.6 - 0.4 < 0.2  # in floats it does not
    assert Fraction(0.6) - Fraction(0.4) < Fraction(0.2)  # nor over their binary values
    assert pick_with_margin(picks(), "q:sixty", 0.6, 0.2) == "yes"


def test_a_lead_float_subtraction_passes_but_the_exact_test_fails_abstains() -> None:
    assert 0.8 - 0.1 >= 0.7000000000000001  # in floats it would pass
    assert pick_with_margin(picks(), "q:slip", 0.8, 0.7000000000000001) == abstained("q:slip")
    assert pick_with_margin(picks(), "q:slip", 0.8, 0.7) == "a"


def test_a_tie_for_second_below_the_top_does_not_abstain() -> None:
    assert pick_with_margin(picks(), "q:second", 0.5, 0.25) == "a"
    assert pick_with_margin(picks(), "q:second", 0.5, 0.26) == abstained("q:second")


@pytest.mark.parametrize(
    ("ref", "bar", "margin", "picked"),
    [
        ("q:sixty", 0.61, 0.2, None),  # below the bar
        ("q:sixty", 0.6, 0.21, None),  # the lead too small
        ("q:sixty", 0, 0, "yes"),
        ("q:nay", 0.9, 0.8, "no"),  # a noul answer picks either label
        ("q:nay", 0.9, 0.81, None),
        ("q:levels", 0.7, 0.5, "high"),
        ("q:one", 1, 1, "only"),  # one label, against a runner-up of 0
        ("q:one", 1.0, 1.0, "only"),
    ],
)
def test_the_top_label_at_or_above_bar_and_leading_by_margin(
    ref: str, bar: float, margin: float, picked: str | None
) -> None:
    got = pick_with_margin(picks(), ref, bar, margin)
    if picked is None:
        assert got == abstained(ref)
    else:
        assert got == picked
        assert type(got) is str


@pytest.mark.parametrize("ref", ["q:tie", "q:even"])
def test_a_tie_at_the_top_abstains_even_at_margin_0(ref: str) -> None:
    assert pick_with_margin(picks(), ref, 0, 0) == abstained(ref)


def test_an_abstention_names_the_ref_and_is_logged_as_resting_on_it() -> None:
    r = picks()
    got = pick_with_margin(r, "q:even", 0, 0)
    assert isinstance(got, Unsure)
    assert got.cause == Cause(CauseCode.abstained, ["q:even"])
    assert items(r) == [("pick_with_margin", ("q:even",), ("q:even",))]


def test_a_marker_read_is_returned_unchanged() -> None:
    r = picks()
    assert pick_with_margin(r, "d:gone", 0.5, 0.2) is GONE
    assert GONE.cause == Cause(CauseCode.raised, ["d:gone"])
    assert items(r) == [("pick_with_margin", ("d:gone",), ("d:gone",))]


def test_a_pick_is_logged_with_its_ref_and_no_number() -> None:
    r = picks()
    pick_with_margin(r, "q:sixty", 0.6, 0.2)
    pick_with_margin(r, "q:tie", 0.5, 0)
    assert items(r) == [
        ("pick_with_margin", ("q:sixty",), ("q:sixty",)),
        ("pick_with_margin", ("q:tie",), ("q:tie",)),
    ]
    for item in r.helper_log():
        assert all(type(x) is str for x in (item.helper, *item.refs, *item.rested_on))


BAD = [
    -0.1,
    1.5,
    -1,
    2,
    10**5000,  # no float holds it, and its decimal is past Python's default digit limit
    math.nan,
    math.inf,
    -math.inf,
    True,
    False,
    "0.5",
    None,
    Fraction(1, 2),
    [0.5],
]


@pytest.mark.parametrize("bad", BAD, ids=[type(b).__name__ for b in BAD])
@pytest.mark.parametrize("ref", ["q:sixty", "d:gone"])
@pytest.mark.parametrize("which", ["bar", "margin"])
def test_a_bad_argument_raises_after_the_item_is_logged(which: str, ref: str, bad: Any) -> None:
    r = picks()
    args: dict[str, Any] = {"bar": 0.5, "margin": 0.1, which: bad}
    with pytest.raises((TypeError, ValueError), match=f"'s {which} "):  # before the marker returns
        pick_with_margin(r, ref, **args)
    assert items(r) == [("pick_with_margin", (ref,), ())]
    assert r.stops == ()


def test_bar_and_margin_both_bad_raises_on_bar() -> None:
    r = picks()
    with pytest.raises(ValueError, match="'s bar "):
        pick_with_margin(r, "q:sixty", 2, -1)


def test_a_read_that_is_not_an_answer_raises_after_the_item_is_logged() -> None:
    r = picks()
    with pytest.raises(TypeError, match="no answer"):
        pick_with_margin(r, "t:lookup", 0.5, 0.1)
    assert items(r) == [("pick_with_margin", ("t:lookup",), ())]


def test_the_arguments_are_tested_before_the_value_read() -> None:
    r = picks()
    with pytest.raises(ValueError, match="bar"):
        pick_with_margin(r, "t:lookup", 2, 0.1)


def test_an_undeclared_ref_stops_the_pick_before_any_item() -> None:
    r = picks()
    with pytest.raises(_Stop):
        pick_with_margin(r, "q:other", 2, 0.1)  # the stop comes before the argument test
    assert r.stops == (UndeclaredRead.of("q:other", Access.read),)
    assert r.helper_log() == ()


def test_a_pick_on_a_whens_view_raises_before_its_read() -> None:
    r = ReadView(Role(ComponentKind.when, "mail"), ("q:sixty",), (), PICKS, {}, state=RecordOf(()))
    with pytest.raises(HelperOnWhen):
        pick_with_margin(r, "q:other", 0.5, 0.1)
    assert r.stops == ()
    assert r.helper_log() == ()


def test_every_argument_is_required() -> None:
    params = inspect.signature(pick_with_margin).parameters
    assert list(params) == ["r", "ref", "bar", "margin"]
    assert all(p.default is inspect.Parameter.empty for p in params.values())
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params.values())


def test_a_pick_leaves_the_answer_unchanged() -> None:
    answer = PICKS["q:tie"]
    assert isinstance(answer, Answer)
    before = dict(answer.probabilities)
    pick_with_margin(picks(), "q:tie", 0, 0)
    assert answer.probabilities == before
