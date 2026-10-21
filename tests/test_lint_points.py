# pyright: reportPrivateUsage=false
"""The lints' points: an axis, the answer points by kind, the exact arithmetic
and its landing through the helpers, the fill order and the ``values`` slips. Both lints
call the one ``_points`` module; the lints' own tests are in ``test_dead_questions.py`` and
``test_unsure_to_execute.py``."""

import math
import sys
from fractions import Fraction
from typing import Any, cast

import pytest

from designs import roles
from jurat import (
    Answer,
    Design,
    LoadedDesign,
    Parameter,
    Plan,
    Policy,
    Question,
    Rule,
    Source,
    Text,
    load,
)
from jurat import types as t
from jurat._read_view import ComponentKind, ReadView, Role
from jurat.helpers import mass, pick_with_margin, pick_with_thresholds
from jurat.helpers._numbers import exact
from jurat.helpers._points import axis, fill

F = Fraction


def plan(r: Any) -> Plan:
    return Plan("escalate", reason="r")


def echo(r: Any) -> bool:
    return True


def design(
    provides: dict[str, Any],
    boundaries: dict[str, Any] | None = None,
    parameters: dict[str, Parameter] | None = None,
    state: str = t.record(),
) -> LoadedDesign:
    """A source providing ``provides``, one rule over every read with ``boundaries``."""
    reads = sorted(set(provides) | set(boundaries or {}))
    rule = Rule("d:echo", t.boolean, echo, version="1", reads=reads, boundaries=boundaries)
    return load(
        Design(
            state=state,
            policy=Policy(plan, version="p"),
            sources={"ask": Source(provides)},
            rules={"echo": rule},
            parameters=parameters or {},
        )
    )


def unit(value: float) -> Parameter:
    return Parameter(t.number(0, 1), value)


def noul(**given: Any) -> LoadedDesign:
    params = {f"p/{name}": unit(v) for name, v in given.items() if name != "mass"}
    mass_ = [{"labels": labels, "bounds": ["p/m"]} for labels in given.get("mass", [])]
    if mass_:
        params["p/m"] = unit(0.3)
    entry = roles(
        bar=["p/bar"] if "bar" in given else [],
        margin=["p/margin"] if "margin" in given else [],
        mass=mass_,
    )
    return design({"q:go": Question("noul", Text("Go?"))}, {"q:go": entry}, params)


def choice(
    labels: tuple[str, ...], entry: dict[str, Any], params: dict[str, Parameter]
) -> LoadedDesign:
    q = Question("choice", Text("Which?"), labels=labels)
    return design({"q:pick": q}, {"q:pick": entry}, params)


def probabilities(loaded: LoadedDesign, ref: str) -> list[dict[str, float] | None]:
    """A ``q:`` read's points as probabilities, the marker as ``None``."""
    points = fill(loaded, [ref], None)[ref]
    out: list[dict[str, float] | None] = []
    for p in points:
        rec = p.record
        assert isinstance(rec, dict)
        out.append(rec.get("probabilities"))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return out


def on(ref: str, answer: Answer) -> ReadView:
    return ReadView(Role(ComponentKind.rule, "echo"), [ref], [], {ref: answer}, {})


# ---------------------------------------------------------------------------------------------
# An axis


def test_an_axis_holds_each_boundary_each_midpoint_and_one_past_each_end() -> None:
    assert axis([F(1), F(3)], None, None, integer=False) == [0, 1, 2, 3, 4]


def test_an_axis_ends_at_the_types_interval_bounds_and_drops_duplicates() -> None:
    assert axis([F(1, 2), F(1, 2), F(0)], F(0), F(1), integer=False) == [0, F(1, 4), F(1, 2), 1]


def test_an_integer_axis_takes_the_least_integer_above_the_lower_boundary() -> None:
    assert axis([F(1), F(5)], None, None, integer=True) == [0, 1, 2, 5, 6]
    assert axis([F(3), F(4)], None, None, integer=True) == [2, 3, 4, 5]  # no integer between


def test_an_integer_axis_ends_at_the_nearest_integer_beyond_each_end() -> None:
    assert axis([F(4)], None, None, integer=True) == [3, 4, 5]


def test_an_integer_axis_drops_a_boundary_that_is_not_integral() -> None:
    assert axis([F(7, 2), F(5)], None, None, integer=True) == [4, 5, 6]
    assert axis([F(7, 2)], F(0), F(10), integer=True) == [0, 10]


def test_a_boundary_outside_the_interval_is_not_on_the_axis() -> None:
    assert axis([F(20), F(3)], F(0), F(10), integer=True) == [0, 3, 10]
    assert axis([F(-5)], F(0), F(10), integer=True) == [0, 10]
    assert axis([F(5)], F(0), F(1), integer=False) == [0, 1]


def test_an_axis_with_no_boundary_has_no_points() -> None:
    assert axis([], F(0), F(1), integer=False) == []


def test_a_midpoint_is_exact_over_the_decimals_then_rounded_once() -> None:
    loaded = design(
        {"t:n": t.number()},
        {"t:n": ["n/a", "n/b"]},
        {"n/a": Parameter(t.number(), 0.1), "n/b": Parameter(t.number(), 0.2)},
    )
    assert [p.record for p in fill(loaded, ["t:n"], None)["t:n"]] == [-0.9, 0.1, 0.15, 0.2, 1.2]
    assert (0.1 + 0.2) / 2 != 0.15  # the float sum would miss


def test_a_numeric_reads_points_conform_to_its_type() -> None:
    loaded = design(
        {"t:n": t.integer(0, 10)}, {"t:n": ["n/cut"]}, {"n/cut": Parameter(t.integer(), 3)}
    )
    points = fill(loaded, ["t:n"], None)["t:n"]
    assert [p.record for p in points] == [0, 3, 10]
    assert all(type(p.record) is int for p in points)


def test_an_integer_reads_integral_number_boundary_is_its_integer() -> None:
    loaded = design({"t:n": t.integer()}, {"t:n": ["n/cut"]}, {"n/cut": Parameter(t.number(), 3.0)})
    points = fill(loaded, ["t:n"], None)["t:n"]
    assert [p.record for p in points] == [2, 3, 4]
    assert all(type(p.record) is int for p in points)


def test_an_integer_reads_points_lie_within_its_interval() -> None:
    loaded = design(
        {"t:n": t.integer(0, 10)},
        {"t:n": ["n/a", "n/b", "n/c"]},
        {
            "n/a": Parameter(t.number(), 2.5),
            "n/b": Parameter(t.integer(), 20),
            "n/c": Parameter(t.integer(), -5),
        },
    )
    assert [p.record for p in fill(loaded, ["t:n"], None)["t:n"]] == [0, 10]


MAX = 2**53 - 1


def numbers(read_type: str, *leaves: Parameter) -> list[object]:
    names = [f"n/l{i}" for i in range(len(leaves))]
    loaded = design({"t:n": read_type}, {"t:n": names}, dict(zip(names, leaves, strict=True)))
    return [p.record for p in fill(loaded, ["t:n"], None)["t:n"]]


def test_a_number_end_that_x_plus_or_minus_1_rounds_onto_is_the_next_float_outward() -> None:
    assert numbers(t.number(), Parameter(t.number(), 1e308)) == [
        math.nextafter(1e308, -math.inf),
        1e308,
        math.nextafter(1e308, math.inf),
    ]


def test_a_number_end_at_the_largest_float_is_the_boundary_itself() -> None:
    top = sys.float_info.max
    below = math.nextafter(top, -math.inf)
    assert numbers(t.number(), Parameter(t.number(), top)) == [below, top]
    assert numbers(t.number(), Parameter(t.number(), -top)) == [-top, -below]


def test_a_number_midpoint_that_rounds_onto_a_boundary_is_no_point() -> None:
    a, b = 1e16, 1e16 + 2  # the exact midpoint 1e16 + 1 rounds to 1e16
    assert float((exact(a) + exact(b)) / 2) == a
    assert numbers(t.number(), Parameter(t.number(), a), Parameter(t.number(), b)) == [
        a - 2,
        a,
        b,
        b + 2,
    ]


def test_an_integer_boundary_past_the_value_domain_is_off_the_axis() -> None:
    assert numbers(t.integer(), Parameter(t.number(), 1e20)) == [-MAX, MAX]
    assert numbers(t.integer(), Parameter(t.number(), 2.0**60), Parameter(t.integer(), 3)) == [
        2,
        3,
        4,
    ]


def test_an_integer_end_past_the_value_domain_is_its_bound() -> None:
    assert numbers(t.integer(), Parameter(t.integer(), MAX)) == [MAX - 1, MAX]
    assert numbers(t.integer(), Parameter(t.integer(), -MAX)) == [-MAX, -MAX + 1]


# ---------------------------------------------------------------------------------------------
# Answer points


def test_a_noul_reads_bar_and_margin_share_one_p_yes_axis() -> None:
    loaded = noul(bar=0.8, margin=0.2)
    yes = [d["yes"] if d else None for d in probabilities(loaded, "q:go")]
    assert yes == [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, None]


def test_a_noul_point_is_yes_at_p_and_no_at_one_minus_p_exactly() -> None:
    loaded = noul(bar=0.001)
    points = probabilities(loaded, "q:go")
    assert {"yes": 0.999, "no": 0.001} in points  # 1 - (1 - 0.001) in floats would miss
    assert 1 - (1 - 0.001) != 0.001


def test_a_noul_mass_bound_over_yes_is_at_x_over_no_at_one_minus_x() -> None:
    assert [d and d["yes"] for d in probabilities(noul(mass=[["yes"]]), "q:go")] == [
        0.0,
        0.3,
        1.0,
        None,
    ]
    assert [d and d["yes"] for d in probabilities(noul(mass=[["no"]]), "q:go")] == [
        0.0,
        0.7,
        1.0,
        None,
    ]


def test_a_q_read_whose_entry_yields_no_answer_point_is_filled_as_off_the_grid() -> None:
    assert probabilities(noul(mass=[["yes", "no"]]), "q:go") == [
        {"yes": 1.0, "no": 0.0},
        {"yes": 0.0, "no": 1.0},
        None,
    ]


def test_choice_points_are_bar_and_thresholds_then_margins_label_by_label() -> None:
    loaded = choice(
        ("warm", "cold"),
        roles(bar=["p/bar"], thresholds=["p/th"], margin=["p/margin"]),
        {
            "p/bar": unit(0.5),
            "p/th": Parameter(t.mapping(t.number(0, 1)), {"cold": 0.6}),
            "p/margin": unit(0.2),
        },
    )
    assert probabilities(loaded, "q:pick") == [
        {"warm": 0.0, "cold": 1.0},
        {"warm": 0.5, "cold": 0.5},
        {"warm": 1.0, "cold": 0.0},
        {"warm": 0.45, "cold": 0.55},
        {"warm": 0.4, "cold": 0.6},
        {"warm": 0.6, "cold": 0.4},
        None,  # every distribution already listed is dropped, the first kept
    ]


def test_a_choice_point_spreads_the_rest_evenly() -> None:
    loaded = choice(("a", "b", "c", "d"), roles(bar=["p/bar"]), {"p/bar": unit(0.6)})
    assert probabilities(loaded, "q:pick")[1] == {
        "a": 0.6,
        "b": float(F(2, 15)),
        "c": float(F(2, 15)),
        "d": float(F(2, 15)),
    }


def test_a_margin_points_next_label_wraps_after_the_last() -> None:
    loaded = choice(("a", "b", "c"), roles(margin=["p/margin"]), {"p/margin": unit(0.5)})
    points = probabilities(loaded, "q:pick")
    assert points[7] == {"a": 0.25, "b": 0.0, "c": 0.75}  # c on top, then a


def test_a_mass_point_holds_x_on_the_sets_first_label_in_declared_order() -> None:
    loaded = choice(
        ("a", "b", "c"),
        roles(mass=[{"labels": ["c", "b"], "bounds": ["p/m"]}]),
        {"p/m": unit(0.25)},
    )
    assert probabilities(loaded, "q:pick") == [
        {"a": 1.0, "b": 0.0, "c": 0.0},
        {"a": 0.75, "b": 0.25, "c": 0.0},
        {"a": 0.0, "b": 1.0, "c": 0.0},
        None,
    ]


def test_a_mass_point_no_distribution_realises_is_dropped() -> None:
    loaded = choice(
        ("x", "y"),
        roles(mass=[{"labels": ["x", "y"], "bounds": ["p/m"]}]),
        {"p/m": unit(0.5)},
    )
    assert probabilities(loaded, "q:pick") == [{"x": 1.0, "y": 0.0}, None]


def test_a_mass_distribution_already_listed_is_dropped() -> None:
    loaded = choice(
        ("x", "y"),
        roles(bar=["p/bar"], mass=[{"labels": ["x"], "bounds": ["p/bar"]}]),
        {"p/bar": unit(0.5)},
    )
    assert probabilities(loaded, "q:pick") == [
        {"x": 0.0, "y": 1.0},
        {"x": 0.5, "y": 0.5},
        {"x": 1.0, "y": 0.0},
        None,
    ]


def test_a_choice_mass_over_the_empty_set_keeps_only_its_point_at_0() -> None:
    loaded = choice(
        ("a", "b", "c"),
        roles(mass=[{"labels": [], "bounds": ["p/m"]}]),
        {"p/m": unit(0.3)},
    )
    third = float(F(1, 3))
    assert probabilities(loaded, "q:pick") == [{"a": third, "b": third, "c": third}, None]


def test_a_margin_point_past_14_decimals_is_kept_though_it_misses() -> None:
    margin = 0.3333333333333333
    loaded = choice(("a", "b"), roles(margin=["p/margin"]), {"p/margin": unit(margin)})
    points = probabilities(loaded, "q:pick")
    hi, lo = float((1 + exact(margin)) / 2), float((1 - exact(margin)) / 2)
    assert {"a": hi, "b": lo} in points
    assert {"a": lo, "b": hi} in points
    assert exact(hi) - exact(lo) != exact(margin)


def test_a_question_with_one_label_has_one_answer_point() -> None:
    loaded = choice(("only",), roles(bar=["p/bar"], margin=["p/bar"]), {"p/bar": unit(0.5)})
    assert probabilities(loaded, "q:pick") == [{"only": 1.0}, None]


def test_a_score_reads_levels_are_its_labels() -> None:
    q = Question("score", Text("How much?"), labels=("low", "high"))
    loaded = design(
        {"q:s": q},
        {"q:s": roles(thresholds=["p/th"])},
        {"p/th": Parameter(t.mapping(t.number(0, 1)), {"high": 0.7})},
    )
    assert probabilities(loaded, "q:s") == [
        {"low": 1.0, "high": 0.0},
        {"low": float(F(3, 10)), "high": 0.7},
        {"low": 0.0, "high": 1.0},
        None,
    ]


def test_a_q_read_ends_with_unavailable_refused_in_record_form() -> None:
    points = fill(noul(bar=0.5), ["q:go"], None)["q:go"]
    assert points[-1].record == {
        "kind": "Unavailable",
        "cause": {"code": "refused", "refs": ["q:go"]},
    }
    assert points[0].record == {
        "kind": "noul",
        "labels": ["yes", "no"],
        "probabilities": {"yes": 0.0, "no": 1.0},
    }


# ---------------------------------------------------------------------------------------------
# Landing exactly, through the helpers


@pytest.mark.parametrize("bar", [0.55, 0.7, 0.9, 0.999])
def test_a_bar_point_lands_exactly_on_the_bar(bar: float) -> None:
    landed = 0
    for d in probabilities(noul(bar=bar), "q:go")[:-1]:
        assert d is not None
        top = max(d.values())
        if top == bar:  # yes or no sits on the bar, and clears it
            landed += 1
            picked = pick_with_margin(on("q:go", Answer("noul", ("yes", "no"), d)), "q:go", bar, 0)
            assert picked in ("yes", "no")
    assert landed == 2


@pytest.mark.parametrize("margin", [0.059, 0.2, 0.333, 0.001])
def test_a_margin_point_leads_by_exactly_the_margin(margin: float) -> None:
    loaded = choice(("a", "b"), roles(margin=["p/margin"]), {"p/margin": unit(margin)})
    lead = (1 + exact(margin)) / 2
    landed = 0
    for d in probabilities(loaded, "q:pick")[:-1]:
        assert d is not None
        if exact(max(d.values())) == lead:
            landed += 1
            assert exact(max(d.values())) - exact(min(d.values())) == exact(margin)
            r = on("q:pick", Answer("choice", ("a", "b"), d))
            assert pick_with_margin(r, "q:pick", 0, margin) in ("a", "b")
    assert landed == 2  # each label on top


def test_a_threshold_point_meets_the_threshold() -> None:
    loaded = choice(
        ("a", "b", "c"),
        roles(thresholds=["p/th"]),
        {"p/th": Parameter(t.mapping(t.number(0, 1)), {"b": 0.45})},
    )
    d = probabilities(loaded, "q:pick")[1]
    assert d is not None
    assert d["b"] == 0.45
    r = on("q:pick", Answer("choice", ("a", "b", "c"), d))
    assert pick_with_thresholds(r, "q:pick", {"b": 0.45}, 0) == "b"


@pytest.mark.parametrize("x", [0.1, 0.3, 0.7, 0.123456789])
def test_a_mass_point_gives_mass_exactly_x(x: float) -> None:
    loaded = choice(
        ("a", "b", "c"),
        roles(mass=[{"labels": ["b", "c"], "bounds": ["p/m"]}]),
        {"p/m": unit(x)},
    )
    d = probabilities(loaded, "q:pick")[1]
    assert d is not None
    assert mass(on("q:pick", Answer("choice", ("a", "b", "c"), d)), "q:pick", ["b", "c"]) == x


# ---------------------------------------------------------------------------------------------
# The fill


def test_type_fills_labels_in_order_booleans_and_null_last() -> None:
    loaded = design({"t:l": t.nullable(t.label("b", "a")), "t:f": t.boolean})
    fills = fill(loaded, ["t:f", "t:l"], None)
    assert [p.record for p in fills["t:l"]] == ["a", "b", None]
    assert [p.record for p in fills["t:f"]] == [False, True]


def test_a_record_fills_its_fields_in_type_order_the_last_varying_fastest() -> None:
    loaded = design({"t:r": t.record(b=t.boolean, a=t.label("x", "y"))})
    points = [p.record for p in fill(loaded, ["t:r"], None)["t:r"]]
    assert points == [
        {"a": "x", "b": False},
        {"a": "x", "b": True},
        {"a": "y", "b": False},
        {"a": "y", "b": True},
    ]
    assert [list(cast("dict[str, object]", p)) for p in points] == [["a", "b"]] * 4


def test_an_empty_record_fills_to_one_empty_object() -> None:
    assert [p.record for p in fill(design({}), ["t:state"], None)["t:state"]] == [{}]


def test_a_q_read_off_the_grid_fills_each_label_at_one_then_the_marker() -> None:
    q = Question("choice", Text("Which?"), labels=("warm", "cold"))
    points = probabilities(design({"q:tone": q}), "q:tone")
    assert points == [{"warm": 1.0, "cold": 0.0}, {"warm": 0.0, "cold": 1.0}, None]


def test_a_record_field_is_filled_under_its_field_path_from_values_or_the_grid() -> None:
    loaded = design(
        {"t:p": t.record(age=t.integer(), name=t.string)},
        {"t:p.age": ["p/age"]},
        {"p/age": Parameter(t.integer(), 18)},
    )
    points = fill(loaded, ["t:p"], {"t:p.name": ["ann"]})["t:p"]
    assert [p.record for p in points] == [
        {"age": 17, "name": "ann"},
        {"age": 18, "name": "ann"},
        {"age": 19, "name": "ann"},
    ]


def test_a_values_list_fills_a_read_and_a_bytes_value_is_live_bytes_and_base64url() -> None:
    loaded = design({"t:b": t.bytes_})
    points = fill(loaded, ["t:b"], {"t:b": [b"\x00\xff"]})["t:b"]
    assert (points[0].live, points[0].record) == (b"\x00\xff", "AP8")


def test_a_record_values_fill_whole_when_its_fields_do_not_all_fill() -> None:
    loaded = design({}, state=t.record(subject=t.string, flag=t.boolean))
    state = {"subject": "s", "flag": True}
    points = fill(loaded, ["t:state"], {"t:state": [state]})["t:state"]
    assert [p.record for p in points] == [state]


def test_a_record_filled_whole_drops_its_fields_slips() -> None:
    loaded = design({}, state=t.record(subject=t.string, flag=t.string))
    state = {"subject": "s", "flag": "f"}
    says = r'^values names "t:state\.subject", a read the lint does not fill$'
    with pytest.raises(ValueError, match=says):
        fill(loaded, ["t:state"], {"t:state": [state], "t:state.subject": [1]})


def test_every_unfilled_read_is_named_in_one_value_error() -> None:
    loaded = design({"t:s": t.string}, state=t.record(subject=t.string, sender=t.string))
    with pytest.raises(ValueError, match="unfilled") as info:
        fill(loaded, ["t:s", "t:state"], None)
    for name in ('"t:s"', '"t:state.sender"', '"t:state.subject"'):
        assert name in str(info.value)


class _NoRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr")


@pytest.mark.parametrize(
    ("values", "says"),
    [
        ({"t:x": ["a"], "t:other": ["a"]}, "a read the lint does not fill"),
        ({"t:x": ["a"], "t:f": [True]}, "its grid points or its type fill"),
        ({"t:x": ["a"], "q:go": [{"yes": 1, "no": 0}]}, "its grid points or its type fill"),
        ({"t:x": ["a"], "t:n": [1]}, "its grid points or its type fill"),
        ({"t:x": [1]}, "does not conform"),
        ({"t:x": ["a", "a"]}, "one value twice"),
        ({"t:x": []}, "is empty"),
        ({"t:x": ["a"], 3: ["a"]}, "the values key 3 is not a string"),
        ({"t:x": ["a"], _NoRepr(): ["a"]}, "the values key <_NoRepr> is not a string"),
    ],
)
def test_a_values_slip_is_a_value_error(values: dict[Any, Any], says: str) -> None:
    loaded = design(
        {"t:x": t.string, "t:f": t.boolean, "t:n": t.number(), "q:go": Question("noul", Text("?"))},
        {"t:n": ["n/cut"]},
        {"n/cut": Parameter(t.number(), 1)},
    )
    with pytest.raises(ValueError, match=says):
        fill(loaded, ["q:go", "t:f", "t:n", "t:state", "t:x"], values)


def test_two_values_with_the_same_canonical_bytes_are_one_value_twice() -> None:
    loaded = design({"t:n": t.number()})
    with pytest.raises(ValueError, match="one value twice"):
        fill(loaded, ["t:n"], {"t:n": [1, 1.0]})


def test_values_that_is_not_a_mapping_of_lists_is_a_type_error() -> None:
    loaded = design({"t:x": t.string})
    with pytest.raises(TypeError):
        fill(loaded, ["t:x"], [("t:x", ["a"])])
    with pytest.raises(TypeError):
        fill(loaded, ["t:x"], {"t:x": "a"})


def test_an_unbounded_integer_axis_left_with_no_boundary_takes_the_domains_bounds() -> None:
    assert axis([F(7, 2)], None, None, integer=True) == [-MAX, MAX]  # none integral
    assert numbers(t.integer(), Parameter(t.number(), 2.5)) == [-MAX, MAX]
    assert numbers(t.integer(0, 10), Parameter(t.number(), 2.5)) == [0, 10]  # its interval's
