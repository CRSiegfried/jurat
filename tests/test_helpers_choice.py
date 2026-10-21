# pyright: reportPrivateUsage=false
"""`pick_with_thresholds`, `size` and `mass`."""

import enum
import inspect
import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest

from jurat import Answer, Cause, CauseCode, Unasked, Unavailable, Unsure
from jurat._read_view import Access, ComponentKind, ReadView, Role, UndeclaredRead, _Stop
from jurat._types import RecordOf
from jurat.helpers import mass, pick_with_margin, pick_with_thresholds, size
from jurat.helpers._call import HelperOnWhen

GONE = Unavailable(Cause(CauseCode.raised, ["d:gone"]))
SKIPPED = Unasked(Cause(CauseCode.when_false, ["q:skipped"]))

BOARD: dict[str, object] = {
    "t:state": {},
    "q:route": Answer(
        "choice", ["refund", "other", "chat"], {"refund": 0.5, "other": 0.3, "chat": 0.2}
    ),
    "q:sixty": Answer("noul", ["yes", "no"], {"yes": 0.6, "no": 0.4}),
    "q:tie": Answer("choice", ["a", "b", "c"], {"a": 0.4, "b": 0.4, "c": 0.2}),
    "q:one": Answer("choice", ["only"], {"only": 1}),
    "q:slip": Answer("choice", ["a", "b", "c"], {"a": 0.8, "b": 0.1, "c": 0.1}),
    "q:quarters": Answer("choice", ["a", "b", "c", "d"], {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}),
    "q:over": Answer("choice", ["a", "b"], {"a": 0.5000001, "b": 0.5}),
    "q:levels": Answer("score", ["low", "mid", "high"], {"low": 0.1, "mid": 0.2, "high": 0.7}),
    # Answers off 1, which the helpers read as held.
    "q:doubled": Answer("choice", ["a", "b"], {"a": 0.9, "b": 0.9}),
    "q:short": Answer("choice", ["a", "b", "c"], {"a": 0.5, "b": 0.2, "c": 0.0}),
    "q:none": Answer("choice", ["only"], {"only": 0}),
    "q:zeros": Answer("noul", ["yes", "no"], {"yes": 0, "no": 0.0}),
    "d:gone": GONE,
    "q:skipped": SKIPPED,
    "t:items": ["x", "y", "z"],
    "t:empty": [],
    "t:pair": ("x", "y"),
    "t:mapping": {"x": 1},
    "t:text": "xyz",
    "t:order": {"lines": ["a", "b"], "status": "open"},
}


def view(kind: ComponentKind = ComponentKind.rule) -> ReadView:
    name = {"rule": "route", "policy": "policy", "when": "mail"}[kind]
    reads = (*BOARD, "t:order.lines")
    return ReadView(Role(kind, name), reads, (), BOARD, {})


def items(r: ReadView) -> list[tuple[str, tuple[object, ...], tuple[object, ...]]]:
    return [(i.helper, i.refs, i.rested_on) for i in r.helper_log()]


def abstained(ref: str) -> Unsure:
    return Unsure(Cause(CauseCode.abstained, [ref]))


class Key(str):
    pass


class Word(enum.StrEnum):
    refund = "refund"


class Mapping(dict[str, float]):
    pass


class Labels(list[str]):
    pass


# ---------------------------------------------------------------------------------------------
# `pick_with_thresholds`


def test_a_bar_per_label_picks_the_most_probable_qualifier() -> None:
    r = view()
    assert pick_with_thresholds(r, "q:route", {"refund": 0.5, "other": 0.3}, 0.2) == "refund"
    assert items(r) == [("pick_with_thresholds", ("q:route",), ("q:route",))]


def test_a_probability_exactly_at_its_threshold_qualifies() -> None:
    assert pick_with_thresholds(view(), "q:sixty", {"yes": 0.6}, 0.2) == "yes"
    assert pick_with_thresholds(view(), "q:sixty", {"yes": 0.6000000000000001}, 0) == abstained(
        "q:sixty"
    )


def test_a_lead_exactly_at_the_margin_passes_where_floats_fail() -> None:
    assert 0.6 - 0.4 < 0.2
    assert pick_with_thresholds(view(), "q:sixty", {"yes": 0.6, "no": 0.9}, 0.2) == "yes"
    assert pick_with_thresholds(view(), "q:sixty", {"yes": 0.6}, 0.21) == abstained("q:sixty")


def test_a_lead_float_subtraction_passes_but_the_exact_test_fails_abstains() -> None:
    assert 0.8 - 0.1 >= 0.7000000000000001
    bars = {"a": 0.8}
    assert pick_with_thresholds(view(), "q:slip", bars, 0.7000000000000001) == abstained("q:slip")
    assert pick_with_thresholds(view(), "q:slip", bars, 0.7) == "a"


def test_the_second_is_the_highest_other_label_qualifying_or_not() -> None:
    # refund 0.5 is below its bar, so other 0.3 is the best qualifier and refund outranks it.
    got = pick_with_thresholds(view(), "q:route", {"refund": 0.9, "other": 0.3}, 0)
    assert got == abstained("q:route")


def test_a_qualifier_tied_with_a_non_qualifier_abstains_at_margin_0() -> None:
    assert pick_with_thresholds(view(), "q:tie", {"a": 0.4, "b": 0.9}, 0) == abstained("q:tie")


def test_a_tie_between_two_qualifiers_abstains_at_margin_0() -> None:
    assert pick_with_thresholds(view(), "q:tie", {"a": 0.4, "b": 0.4}, 0) == abstained("q:tie")


def test_a_label_absent_from_thresholds_cannot_be_chosen() -> None:
    # high leads, but only low and mid have bars, and neither leads high.
    got = pick_with_thresholds(view(), "q:levels", {"low": 0, "mid": 0}, 0)
    assert got == abstained("q:levels")
    assert pick_with_thresholds(view(), "q:levels", {"high": 0.7}, 0.5) == "high"


def test_no_qualifying_label_abstains() -> None:
    r = view()
    assert pick_with_thresholds(r, "q:route", {"refund": 0.6, "other": 0.6}, 0) == abstained(
        "q:route"
    )
    assert items(r) == [("pick_with_thresholds", ("q:route",), ("q:route",))]


def test_empty_thresholds_abstains() -> None:
    assert pick_with_thresholds(view(), "q:route", {}, 0) == abstained("q:route")


@pytest.mark.parametrize(("margin", "picked"), [(1, "only"), (1.0, "only"), (0, "only")])
def test_a_one_label_answer_is_measured_against_0(margin: float, picked: str) -> None:
    got = pick_with_thresholds(view(), "q:one", {"only": 1}, margin)
    assert got == picked
    assert type(got) is str


@pytest.mark.parametrize("marker", ["d:gone", "q:skipped"])
def test_a_marker_read_is_returned_unchanged(marker: str) -> None:
    r = view()
    got = pick_with_thresholds(r, marker, {"zzz": 0.5}, 0.2)  # no answer to lack the label
    assert got is BOARD[marker]
    assert items(r) == [("pick_with_thresholds", (marker,), (marker,))]


BAD_NUMBERS = [-0.1, 1.5, 2, 10**5000, math.nan, math.inf, -math.inf, True, "0.5", None]


@pytest.mark.parametrize("bad", BAD_NUMBERS, ids=[type(b).__name__ for b in BAD_NUMBERS])
@pytest.mark.parametrize("ref", ["q:route", "d:gone"])
def test_a_bad_threshold_raises_even_over_a_marker(ref: str, bad: Any) -> None:
    r = view()
    with pytest.raises((TypeError, ValueError), match="'s thresholds "):
        pick_with_thresholds(r, ref, {"refund": bad}, 0.1)
    assert items(r) == [("pick_with_thresholds", (ref,), ())]


@pytest.mark.parametrize("bad", BAD_NUMBERS, ids=[type(b).__name__ for b in BAD_NUMBERS])
@pytest.mark.parametrize("ref", ["q:route", "d:gone"])
def test_a_bad_margin_raises_even_over_a_marker(ref: str, bad: Any) -> None:
    r = view()
    with pytest.raises((TypeError, ValueError), match="'s margin "):
        pick_with_thresholds(r, ref, {"refund": 0.5}, bad)
    assert items(r) == [("pick_with_thresholds", (ref,), ())]


BAD_THRESHOLDS: list[object] = [
    Mapping(refund=0.5),
    [("refund", 0.5)],
    None,
    {1: 0.5},
    {Key("refund"): 0.5},
    {Word.refund: 0.5},
]


@pytest.mark.parametrize("bad", BAD_THRESHOLDS, ids=[repr(b) for b in BAD_THRESHOLDS])
@pytest.mark.parametrize("ref", ["q:route", "d:gone"])
def test_thresholds_not_a_dict_of_str_raises_even_over_a_marker(ref: str, bad: Any) -> None:
    r = view()
    with pytest.raises(TypeError, match="'s thresholds "):
        pick_with_thresholds(r, ref, bad, 0.1)
    assert items(r) == [("pick_with_thresholds", (ref,), ())]


def test_a_threshold_label_the_answer_lacks_raises_after_the_type_test() -> None:
    r = view()
    with pytest.raises(ValueError, match="'nope', which the answer lacks"):
        pick_with_thresholds(r, "q:route", {"refund": 0.1, "nope": 0.5}, 0.1)
    assert items(r) == [("pick_with_thresholds", ("q:route",), ())]
    with pytest.raises(TypeError, match="no answer"):
        pick_with_thresholds(view(), "t:items", {"nope": 0.5}, 0.1)


def test_thresholds_are_tested_before_margin() -> None:
    with pytest.raises(ValueError, match="'s thresholds "):
        pick_with_thresholds(view(), "q:route", {"refund": 2}, -1)


def test_a_pick_with_thresholds_on_a_whens_view_raises_before_its_read() -> None:
    r = ReadView(Role(ComponentKind.when, "mail"), ("q:route",), (), BOARD, {}, state=RecordOf(()))
    with pytest.raises(HelperOnWhen):
        pick_with_thresholds(r, "q:other", {"refund": 0.5}, 0.1)
    assert r.stops == ()
    assert r.helper_log() == ()


def test_an_undeclared_ref_stops_pick_with_thresholds_before_any_item() -> None:
    r = view()
    with pytest.raises(_Stop):
        pick_with_thresholds(r, "q:other", {"refund": 2}, 0.1)
    assert r.stops == (UndeclaredRead.of("q:other", Access.read),)
    assert r.helper_log() == ()


# ---------------------------------------------------------------------------------------------
# `size`


@pytest.mark.parametrize(
    ("ref", "n"), [("t:items", 3), ("t:empty", 0), ("t:pair", 2), ("t:order.lines", 2)]
)
def test_size_counts_a_lists_members(ref: str, n: int) -> None:
    r = view()
    got = size(r, ref)
    assert got == n
    assert type(got) is int
    assert items(r) == [("size", (ref,), (ref,))]


@pytest.mark.parametrize("marker", ["d:gone", "q:skipped"])
def test_size_returns_a_marker_unchanged(marker: str) -> None:
    r = view()
    assert size(r, marker) is BOARD[marker]
    assert items(r) == [("size", (marker,), (marker,))]


@pytest.mark.parametrize("ref", ["t:mapping", "q:route", "t:text", "t:order"])
def test_size_over_a_value_that_is_not_a_list_raises(ref: str) -> None:
    r = view()
    with pytest.raises(TypeError, match="no list"):
        size(r, ref)
    assert items(r) == [("size", (ref,), ())]


def test_size_on_a_whens_view_or_an_undeclared_ref() -> None:
    w = ReadView(Role(ComponentKind.when, "mail"), ("t:items",), (), BOARD, {}, state=RecordOf(()))
    with pytest.raises(HelperOnWhen):
        size(w, "t:items")
    assert w.helper_log() == ()
    r = view()
    with pytest.raises(_Stop):
        size(r, "t:other")
    assert r.helper_log() == ()


# ---------------------------------------------------------------------------------------------
# `mass`


def test_mass_sums_the_labels_exactly() -> None:
    r = view()
    got = mass(r, "q:levels", ["mid", "high"])
    assert got == 0.9
    assert type(got) is float
    assert items(r) == [("mass", ("q:levels",), ("q:levels",))]


def test_mass_does_not_depend_on_the_order_of_labels() -> None:
    assert 0.1 + 0.2 + 0.3 != 0.3 + 0.2 + 0.1  # in floats it does
    forward = mass(view(), "q:quarters", ["a", "b", "c"])
    backward = mass(view(), "q:quarters", ("c", "b", "a"))
    assert forward == backward == 0.6
    assert float(Fraction("0.1") + Fraction("0.2") + Fraction("0.3")) == 0.6


def test_mass_is_capped_at_1() -> None:
    assert mass(view(), "q:over", ["a", "b"]) == 1.0
    assert mass(view(), "q:over", ["a"]) == 0.5000001
    full = mass(view(), "q:quarters", ["a", "b", "c", "d"])
    assert full == 1.0
    assert type(full) is float


def test_mass_over_no_label_is_0() -> None:
    got = mass(view(), "q:route", [])
    assert got == 0
    assert type(got) is float


def test_mass_over_a_one_label_answer() -> None:
    assert mass(view(), "q:one", ["only"]) == 1.0


@pytest.mark.parametrize("marker", ["d:gone", "q:skipped"])
def test_mass_returns_a_marker_unchanged(marker: str) -> None:
    r = view()
    assert mass(r, marker, ["zzz"]) is BOARD[marker]
    assert items(r) == [("mass", (marker,), (marker,))]


BAD_LABELS: list[object] = [
    {"a"},
    frozenset({"a"}),
    "ab",
    {"a": 1},
    Labels(["a"]),
    None,
    ["a", "a"],
    ("a", 1),
    [Key("a")],
    [Word.refund],
]


@pytest.mark.parametrize("bad", BAD_LABELS, ids=[repr(b) for b in BAD_LABELS])
@pytest.mark.parametrize("ref", ["q:quarters", "d:gone"])
def test_bad_labels_raise_even_over_a_marker(ref: str, bad: Any) -> None:
    r = view()
    with pytest.raises((TypeError, ValueError), match="mass's labels "):
        mass(r, ref, bad)
    assert items(r) == [("mass", (ref,), ())]


def test_a_mass_label_the_answer_lacks_raises() -> None:
    r = view()
    with pytest.raises(ValueError, match="'z', which the answer lacks"):
        mass(r, "q:quarters", ["a", "z"])
    assert items(r) == [("mass", ("q:quarters",), ())]


def test_mass_over_a_value_that_is_not_an_answer_raises() -> None:
    with pytest.raises(TypeError, match="no answer"):
        mass(view(), "t:items", ["x"])


def test_mass_on_a_whens_view_or_an_undeclared_ref() -> None:
    w = ReadView(Role(ComponentKind.when, "mail"), ("q:route",), (), BOARD, {}, state=RecordOf(()))
    with pytest.raises(HelperOnWhen):
        mass(w, "q:route", ["refund"])
    assert w.helper_log() == ()
    r = view()
    with pytest.raises(_Stop):
        mass(r, "q:other", {"bad"})  # pyright: ignore[reportArgumentType]
    assert r.helper_log() == ()


# ---------------------------------------------------------------------------------------------
# Signatures


@pytest.mark.parametrize(
    ("helper", "names"),
    [
        (pick_with_thresholds, ["r", "ref", "thresholds", "margin"]),
        (size, ["r", "ref"]),
        (mass, ["r", "ref", "labels"]),
    ],
)
def test_every_argument_is_required(helper: Any, names: list[str]) -> None:
    params = inspect.signature(helper).parameters
    assert list(params) == names
    assert all(p.default is inspect.Parameter.empty for p in params.values())
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params.values())


class Loud(str):
    def __repr__(self) -> str:
        raise RuntimeError("caller code ran")


def loud_calls() -> dict[str, Callable[[ReadView], object]]:
    return {
        "size": lambda r: size(r, Loud("t:mapping")),
        "mass": lambda r: mass(r, Loud("t:items"), ["x"]),
        "pick_with_thresholds": lambda r: pick_with_thresholds(r, Loud("t:items"), {}, 0),
        "pick_with_margin": lambda r: pick_with_margin(r, Loud("t:items"), 0, 0),
    }


@pytest.mark.parametrize("helper", list(loud_calls()))
def test_a_raise_names_a_ref_by_its_string_data_not_its_repr(helper: str) -> None:
    r = view()
    call = loud_calls()[helper]
    with pytest.raises(TypeError, match='read "t:') as info:
        call(r)
    assert not isinstance(info.value, RuntimeError)
    assert len(r.helper_log()) == 1


# ---------------------------------------------------------------------------------------------
# Answers off 1: read as held, the cap and the margins over the values themselves


def test_a_two_label_tie_at_0_9_abstains() -> None:
    assert pick_with_margin(view(), "q:doubled", 0.5, 0) == abstained("q:doubled")
    assert pick_with_thresholds(view(), "q:doubled", {"a": 0.5}, 0) == abstained("q:doubled")


def test_mass_of_an_answer_summing_to_1_8_is_capped_at_1() -> None:
    assert mass(view(), "q:doubled", ["a", "b"]) == 1.0
    assert mass(view(), "q:doubled", ["a"]) == 0.9


def test_an_answer_summing_under_1_is_picked_on_its_own_values() -> None:
    # 0.5 leads 0.2 by 0.3 whatever the missing 0.3 of mass would have been.
    assert pick_with_margin(view(), "q:short", 0.5, 0.3) == "a"
    assert pick_with_margin(view(), "q:short", 0.5, 0.31) == abstained("q:short")
    assert pick_with_margin(view(), "q:short", 0.51, 0) == abstained("q:short")
    assert pick_with_thresholds(view(), "q:short", {"a": 0.5, "b": 0.1}, 0.3) == "a"
    assert mass(view(), "q:short", ["a", "b", "c"]) == 0.7


def test_a_one_label_answer_at_0_ties_its_runner_up_of_0_and_abstains() -> None:
    assert pick_with_margin(view(), "q:none", 0, 0) == abstained("q:none")
    assert pick_with_thresholds(view(), "q:none", {"only": 0}, 0) == abstained("q:none")
    assert mass(view(), "q:none", ["only"]) == 0.0


def test_an_answer_of_all_zeros_abstains_and_has_no_mass() -> None:
    assert pick_with_margin(view(), "q:zeros", 0, 0) == abstained("q:zeros")
    assert mass(view(), "q:zeros", ["yes", "no"]) == 0.0
