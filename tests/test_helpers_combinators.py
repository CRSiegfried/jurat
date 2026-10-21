# pyright: reportPrivateUsage=false
"""The combinators `unique`, `agree`, `first`, `priority` and `collect`."""

import enum
import inspect
import math
from typing import Any

import pytest

from jurat import Answer, Cause, CauseCode, Unasked, Unavailable, Unsure
from jurat._read_view import Access, ComponentKind, ReadView, Role, UndeclaredRead, _Stop
from jurat._types import RecordOf
from jurat.helpers import agree, collect, first, priority, unique
from jurat.helpers._call import HelperOnWhen

YES = Answer("noul", ["yes", "no"], {"yes": 0.7, "no": 0.3})
YES_AGAIN = Answer("noul", ["yes", "no"], {"yes": 0.7, "no": 0.3})
NO = Answer("noul", ["yes", "no"], {"yes": 0.2, "no": 0.8})

BOARD: dict[str, object] = {
    "t:state": {},
    "d:a": "refund",
    "d:a2": "refund",
    "d:b": "chat",
    "d:c": "other",
    "d:none": None,
    "d:one": 1,
    "d:onef": 1.0,
    "d:true": True,
    "d:two": 2,
    "d:tenth": 0.1,
    "d:fifth": 0.2,
    "d:big": 1.7976931348623157e308,
    "d:max": 2**53 - 1,
    "d:min": -(2**53 - 1),
    "d:negone": -1,
    "q:yesdoc": {"kind": "noul", "labels": ["yes", "no"], "probabilities": {"yes": 0.7, "no": 0.3}},
    "q:sure": Answer("noul", ["yes", "no"], {"yes": 1, "no": 0}),
    "q:suref": Answer("noul", ["yes", "no"], {"yes": 1.0, "no": 0.0}),
    "d:list": [1, 2],
    "d:obj": {"x": [1]},
    "q:yes": YES,
    "q:yes2": YES_AGAIN,
    "q:no": NO,
    "d:gone": Unavailable(Cause(CauseCode.raised, ["d:gone"])),
    "d:gone2": Unavailable(Cause(CauseCode.timed_out, ["d:gone2"])),
    "d:unsure": Unsure(Cause(CauseCode.abstained, ["q:yes"])),
    "d:unsure2": Unsure(Cause(CauseCode.conflict, ["d:a", "d:b"])),
    "d:unasked": Unasked(Cause(CauseCode.when_false, ["d:unasked"])),
    "t:order": {"status": "open", "tier": "gold"},
}


def view(kind: ComponentKind = ComponentKind.rule) -> ReadView:
    name = {"rule": "route", "policy": "policy", "when": "mail"}[kind]
    reads = (*BOARD, "t:order.status")
    return ReadView(Role(kind, name), reads, (), BOARD, {})


def items(r: ReadView) -> list[tuple[str, tuple[object, ...], tuple[object, ...]]]:
    return [(i.helper, i.refs, i.rested_on) for i in r.helper_log()]


def required(kind: type, ref: str) -> object:
    return kind(Cause(CauseCode.required, [ref]))


def conflict(*refs: str) -> Unsure:
    return Unsure(Cause(CauseCode.conflict, list(refs)))


def run(combinator: str, *refs: str, **kw: Any) -> tuple[object, list[Any]]:
    """Call a combinator on a fresh view; its result and its one log item."""
    r = view()
    fn = {"unique": unique, "agree": agree, "first": first, "collect": collect}[combinator]
    got = fn(r, *refs, **kw)
    return got, items(r)


def run_priority(order: Any, *refs: str) -> tuple[object, list[Any]]:
    r = view()
    got = priority(r, order, *refs)
    return got, items(r)


# ---------------------------------------------------------------------------------------------
# Nothing determined: every combinator

COMBINATORS = ["unique", "agree", "first", "collect", "priority"]


def call_any(name: str, r: ReadView, *refs: str, **kw: Any) -> object:
    if name == "priority":
        return priority(r, ["refund", "chat"], *refs)
    fn = {"unique": unique, "agree": agree, "first": first, "collect": collect}[name]
    return fn(r, *refs, **kw)


@pytest.mark.parametrize("name", COMBINATORS)
@pytest.mark.parametrize(
    ("refs", "kind", "ref"),
    [
        (("d:unasked", "d:unsure", "d:gone", "d:gone2"), Unavailable, "d:gone"),
        (("d:unasked", "d:unsure", "d:unsure2"), Unsure, "d:unsure"),
        (("d:unasked",), Unasked, "d:unasked"),
        (("d:unsure2",), Unsure, "d:unsure2"),
    ],
)
def test_nothing_determined_gives_a_new_marker_of_highest_precedence(
    name: str, refs: tuple[str, ...], kind: type, ref: str
) -> None:
    r = view()
    got = call_any(name, r, *refs)
    assert got == required(kind, ref)
    assert type(got) is kind
    assert got is not BOARD[ref]  # a new marker, not the input's own
    assert items(r) == [(name, refs, (ref,))]


@pytest.mark.parametrize("agg", [None, "sum", "min", "max", "count"])
def test_collect_with_nothing_determined_is_a_marker_not_empty_or_0(agg: str | None) -> None:
    got, _ = run("collect", "d:unsure", "d:unasked", agg=agg)
    assert got == required(Unsure, "d:unsure")


# ---------------------------------------------------------------------------------------------
# `unique`


def test_unique_returns_the_one_determined_value() -> None:
    got, log = run("unique", "d:unsure", "d:a", "d:unasked")
    assert got == "refund"
    assert log == [("unique", ("d:unsure", "d:a", "d:unasked"), ("d:a",))]


def test_unique_takes_none_as_a_value() -> None:
    got, log = run("unique", "d:unsure", "d:none")
    assert got is None
    assert log == [("unique", ("d:unsure", "d:none"), ("d:none",))]


@pytest.mark.parametrize(
    "refs", [("d:a", "d:a2"), ("d:a", "d:b"), ("d:a", "d:unsure", "d:b", "d:c")]
)
def test_two_determined_inputs_overlap_equal_or_not(refs: tuple[str, ...]) -> None:
    got, log = run("unique", *refs)
    determined = tuple(ref for ref in refs if ref != "d:unsure")
    assert got == conflict(*determined)
    assert log == [("unique", refs, determined)]


def test_unique_overlap_is_conflict_whatever_is_unavailable() -> None:
    got, log = run("unique", "d:gone", "d:a", "d:a2")
    assert got == conflict("d:a", "d:a2")
    assert log[0][2] == ("d:a", "d:a2")


def test_unique_with_an_unavailable_input_is_unavailable() -> None:
    got, log = run("unique", "d:a", "d:unsure", "d:gone2", "d:gone")
    assert got == required(Unavailable, "d:gone2")
    assert log == [("unique", ("d:a", "d:unsure", "d:gone2", "d:gone"), ("d:gone2",))]


# ---------------------------------------------------------------------------------------------
# `agree`


def test_agree_returns_the_shared_value() -> None:
    got, log = run("agree", "d:a", "d:unsure", "d:a2", "d:unasked")
    assert got == "refund"
    assert log == [("agree", ("d:a", "d:unsure", "d:a2", "d:unasked"), ("d:a", "d:a2"))]


def test_agree_over_one_determined_value() -> None:
    got, log = run("agree", "d:unsure", "d:b")
    assert got == "chat"
    assert log[0][2] == ("d:b",)


def test_agree_equality_is_canonical_bytes() -> None:
    got, _ = run("agree", "d:one", "d:onef")  # 1 and 1.0 are one canonical number
    assert got == 1
    assert type(got) is int  # the first determined input's value
    got, _ = run("agree", "d:onef", "d:one")
    assert type(got) is float
    got, _ = run("agree", "d:one", "d:true")  # true is not 1
    assert got == conflict("d:one", "d:true")
    got, _ = run("agree", "q:yes", "q:yes2")  # answers compare by their recorded objects
    assert got is YES
    got, _ = run("agree", "q:yes", "q:no")
    assert got == conflict("q:yes", "q:no")


def test_a_failed_majority_names_every_determined_input() -> None:
    got, log = run("agree", "d:a", "d:unasked", "d:a2", "d:b")
    assert got == conflict("d:a", "d:a2", "d:b")
    assert log[0][2] == ("d:a", "d:a2", "d:b")


def test_agree_disagreement_is_conflict_whatever_is_unavailable() -> None:
    got, _ = run("agree", "d:gone", "d:a", "d:b")
    assert got == conflict("d:a", "d:b")


def test_agree_with_an_unavailable_input_is_unavailable() -> None:
    got, log = run("agree", "d:a", "d:gone", "d:a2")
    assert got == required(Unavailable, "d:gone")
    assert log[0][2] == ("d:gone",)


# ---------------------------------------------------------------------------------------------
# `first`


def test_first_falls_back_past_unsure_and_unasked() -> None:
    got, log = run("first", "d:unsure", "d:unasked", "d:b", "d:a")
    assert got == "chat"
    assert log == [("first", ("d:unsure", "d:unasked", "d:b", "d:a"), ("d:b",))]


def test_first_with_unavailable_before_the_winner_is_unavailable() -> None:
    got, log = run("first", "d:unsure", "d:gone2", "d:gone", "d:a")
    assert got == required(Unavailable, "d:gone2")
    assert log[0][2] == ("d:gone2",)


def test_first_with_unavailable_after_the_winner_is_the_winner() -> None:
    got, log = run("first", "d:a", "d:gone")
    assert got == "refund"
    assert log[0][2] == ("d:a",)


def test_first_reads_a_field_path() -> None:
    got, log = run("first", "d:unsure", "t:order.status")
    assert got == "open"
    assert log == [("first", ("d:unsure", "t:order.status"), ("t:order.status",))]


# ---------------------------------------------------------------------------------------------
# `priority`


def test_priority_returns_the_highest_ranked_value() -> None:
    got, log = run_priority(["refund", "chat", "other"], "d:c", "d:unsure", "d:b")
    assert got == "chat"
    assert log == [("priority", ("d:c", "d:unsure", "d:b"), ("d:b",))]


def test_priority_tie_goes_to_the_first_in_argument_order() -> None:
    got, log = run_priority(["refund", "chat", "other"], "d:c", "d:a2", "d:a")
    assert got == "refund"
    assert log[0][2] == ("d:a2",)


def test_priority_with_the_top_value_ignores_unavailable() -> None:
    got, log = run_priority(["refund", "chat"], "d:gone", "d:b", "d:a")
    assert got == "refund"
    assert log[0][2] == ("d:a",)


def test_priority_with_a_lower_winner_and_unavailable_is_unavailable() -> None:
    got, log = run_priority(["refund", "chat"], "d:b", "d:gone2", "d:gone")
    assert got == required(Unavailable, "d:gone2")
    assert log[0][2] == ("d:gone2",)


def test_priority_compares_by_canonical_bytes() -> None:
    got, _ = run_priority([1.0, 2], "d:two", "d:one")
    assert got == 1
    assert type(got) is int  # the input's value, not order's member
    with pytest.raises(ValueError, match="does not rank"):
        run_priority([1, 2], "d:true")
    got, _ = run_priority([{"x": [1]}, [1, 2]], "d:list", "d:obj")
    assert got == {"x": [1]}


def test_priority_over_a_value_order_lacks_raises_whatever_decides() -> None:
    r = view()
    with pytest.raises(ValueError, match='"d:c", whose value order does not rank'):
        priority(r, ["refund"], "d:a", "d:c")  # refund is the top value, yet other raises
    assert items(r) == [("priority", ("d:a", "d:c"), ())]
    with pytest.raises(ValueError, match="does not rank"):
        run_priority(["refund"], "d:gone", "q:yes")


BAD_ORDERS: list[object] = [
    [],
    (),
    {"refund"},
    "refund",
    {"refund": 1},
    None,
    ["refund", "refund"],
    [1, 1.0],
    [math.nan],
    [math.inf],
    [10**20],
    [YES],
    [b"refund"],
    [{1: "x"}],
]


class Rows(list[object]):
    pass


BAD_ORDERS.append(Rows(["refund"]))
BAD_ORDERS.append([Rows(["refund"])])


@pytest.mark.parametrize("bad", BAD_ORDERS, ids=[repr(b) for b in BAD_ORDERS])
@pytest.mark.parametrize("refs", [("d:a",), ("d:gone",)])
def test_a_bad_order_raises_even_when_the_result_is_a_marker(
    refs: tuple[str, ...], bad: Any
) -> None:
    r = view()
    with pytest.raises((TypeError, ValueError), match="priority's order "):
        priority(r, bad, *refs)
    assert items(r) == [("priority", refs, ())]


# ---------------------------------------------------------------------------------------------
# `collect`


def test_collect_returns_the_determined_values_in_argument_order() -> None:
    got, log = run("collect", "d:b", "d:unsure", "d:none", "d:a", "d:unasked")
    assert got == ["chat", None, "refund"]
    assert type(got) is list
    assert log == [
        ("collect", ("d:b", "d:unsure", "d:none", "d:a", "d:unasked"), ("d:b", "d:none", "d:a"))
    ]


def test_collect_count() -> None:
    got, log = run("collect", "d:b", "d:unsure", "q:yes", agg="count")
    assert got == 2
    assert type(got) is int
    assert log[0][2] == ("d:b", "q:yes")


@pytest.mark.parametrize("agg", [None, "sum", "min", "max", "count"])
def test_collect_with_an_unavailable_input_is_unavailable(agg: str | None) -> None:
    got, log = run("collect", "d:one", "d:unsure", "d:gone", "d:two", agg=agg)
    assert got == required(Unavailable, "d:gone")
    assert log[0][2] == ("d:gone",)


def test_an_int_sum_is_an_int() -> None:
    got, _ = run("collect", "d:one", "d:two", "d:unsure", agg="sum")
    assert got == 3
    assert type(got) is int


def test_a_sum_with_a_float_is_exact_then_rounded() -> None:
    assert 0.1 + 0.2 != 0.3
    got, log = run("collect", "d:tenth", "d:fifth", agg="sum")
    assert got == 0.3
    assert type(got) is float
    assert log[0][2] == ("d:tenth", "d:fifth")
    got, _ = run("collect", "d:one", "d:two", "d:onef", agg="sum")
    assert got == 4.0
    assert type(got) is float


def test_a_sum_is_the_same_in_every_argument_order() -> None:
    forward, _ = run("collect", "d:tenth", "d:fifth", "d:onef", agg="sum")
    backward, _ = run("collect", "d:onef", "d:fifth", "d:tenth", agg="sum")
    assert forward == backward == 1.3


def test_a_sum_no_finite_float_holds_raises() -> None:
    got, _ = run("collect", "d:big", agg="sum")  # a float sum alone
    assert got == 1.7976931348623157e308
    r = ReadView(
        Role(ComponentKind.rule, "route"),
        ("d:x", "d:y"),
        (),
        {"t:state": {}, "d:x": 1.7976931348623157e308, "d:y": 1.7976931348623157e308},
        {},
    )
    with pytest.raises(ValueError, match="no finite float holds it"):
        collect(r, "d:x", "d:y", agg="sum")
    assert items(r) == [("collect", ("d:x", "d:y"), ())]


@pytest.mark.parametrize(
    ("agg", "refs", "value", "rested"),
    [
        ("min", ("d:two", "d:one", "d:onef"), 1, ("d:two", "d:one", "d:onef")),
        ("min", ("d:two", "d:onef", "d:one"), 1.0, ("d:two", "d:onef", "d:one")),
        ("max", ("d:onef", "d:two", "d:tenth"), 2, ("d:onef", "d:two", "d:tenth")),
        ("max", ("d:unsure", "d:tenth"), 0.1, ("d:tenth",)),
    ],
)
def test_min_and_max_return_the_first_input_holding_the_extreme(
    agg: str, refs: tuple[str, ...], value: object, rested: tuple[str, ...]
) -> None:
    got, log = run("collect", *refs, agg=agg)
    assert got == value
    assert type(got) is type(value)
    assert log[0][2] == rested


@pytest.mark.parametrize("agg", ["sum", "min", "max"])
@pytest.mark.parametrize("bad", ["d:true", "q:yes", "d:a", "d:none", "d:list"])
def test_a_non_number_under_an_aggregate_raises_even_beside_unavailable(agg: str, bad: str) -> None:
    r = view()
    with pytest.raises(TypeError, match=f'collect read "{bad}", which holds no number to {agg}'):
        collect(r, "d:gone", "d:one", bad, agg=agg)
    assert items(r) == [("collect", ("d:gone", "d:one", bad), ())]


def test_count_and_no_aggregate_take_any_value() -> None:
    got, _ = run("collect", "d:true", "q:yes", "d:list", agg="count")
    assert got == 3
    got, _ = run("collect", "d:true", "q:yes")
    assert got == [True, YES]


class Agg(enum.StrEnum):
    sum = "sum"


BAD_AGGS: list[object] = ["+", "C+", "#", "SUM", "", "mean", 1, True, ["sum"], Agg.sum]


@pytest.mark.parametrize("bad", BAD_AGGS, ids=[repr(b) for b in BAD_AGGS])
@pytest.mark.parametrize("refs", [("d:one",), ("d:gone",)])
def test_a_bad_agg_raises_even_when_the_result_is_a_marker(refs: tuple[str, ...], bad: Any) -> None:
    r = view()
    kind = ValueError if type(bad) is str else TypeError
    with pytest.raises(kind, match="collect's agg "):
        collect(r, *refs, agg=bad)
    assert items(r) == [("collect", refs, ())]


# ---------------------------------------------------------------------------------------------
# Refs, a `when`'s view, an undeclared ref and the signatures


def call_named(name: str, r: ReadView, *refs: str) -> object:
    if name == "priority":
        return priority(r, ["refund"], *refs)
    return {"unique": unique, "agree": agree, "first": first, "collect": collect}[name](r, *refs)


@pytest.mark.parametrize("name", COMBINATORS)
def test_no_ref_raises_after_the_item_is_logged(name: str) -> None:
    r = view()
    with pytest.raises(ValueError, match="no ref"):
        call_named(name, r)
    assert items(r) == [(name, (), ())]


@pytest.mark.parametrize("name", COMBINATORS)
@pytest.mark.parametrize("refs", [("d:a", "d:a"), ("d:gone", "d:b", "d:gone")])
def test_a_repeated_ref_raises_even_over_a_marker(name: str, refs: tuple[str, ...]) -> None:
    r = view()
    with pytest.raises(ValueError, match="a ref twice"):
        call_named(name, r, *refs)
    assert items(r) == [(name, refs, ())]


@pytest.mark.parametrize("name", COMBINATORS)
def test_a_combinator_on_a_whens_view_raises_before_any_read(name: str) -> None:
    r = ReadView(Role(ComponentKind.when, "mail"), ("d:a",), (), BOARD, {}, state=RecordOf(()))
    with pytest.raises(HelperOnWhen, match="mail"):
        call_named(name, r, "d:a", "d:other")
    assert r.stops == ()
    assert r.helper_log() == ()


@pytest.mark.parametrize("name", COMBINATORS)
def test_an_undeclared_ref_stops_the_combinator_before_any_item(name: str) -> None:
    r = view()
    with pytest.raises(_Stop):
        call_named(name, r, "d:a", "d:other", "d:a")  # the stop comes before the ref test
    assert r.stops == (UndeclaredRead.of("d:other", Access.read),)
    assert r.helper_log() == ()


@pytest.mark.parametrize("name", COMBINATORS)
def test_calls_are_listed_in_call_order_with_names_and_refs_only(name: str) -> None:
    r = view()
    call_named(name, r, "d:a")
    call_named(name, r, "d:gone", "d:a")
    assert [i.refs for i in r.helper_log()] == [("d:a",), ("d:gone", "d:a")]
    for item in r.helper_log():
        assert all(type(x) is str for x in (item.helper, *item.refs, *item.rested_on))


def test_a_raise_leaves_the_item_with_rested_on_empty_and_the_log_goes_on() -> None:
    r = view()
    with pytest.raises(TypeError):
        collect(r, "d:a", agg="sum")
    first(r, "d:a")
    assert items(r) == [("collect", ("d:a",), ()), ("first", ("d:a",), ("d:a",))]


def test_the_signatures() -> None:
    for fn in (unique, agree, first):
        params = list(inspect.signature(fn).parameters.values())
        assert [p.name for p in params] == ["r", "refs"]
        assert params[1].kind is inspect.Parameter.VAR_POSITIONAL
    params = list(inspect.signature(priority).parameters.values())
    assert [p.name for p in params] == ["r", "order", "refs"]
    assert params[1].default is inspect.Parameter.empty
    params = list(inspect.signature(collect).parameters.values())
    assert [p.name for p in params] == ["r", "refs", "agg"]
    assert params[2].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[2].default is None


def test_an_answer_equals_a_mapping_with_its_record_bytes() -> None:
    got, _ = run("agree", "q:yes", "q:yesdoc")
    assert got is YES
    got, _ = run_priority([BOARD["q:yesdoc"]], "q:yes")
    assert got is YES


def test_answers_whose_probabilities_differ_as_1_and_1_0_are_equal() -> None:
    sure, suref = BOARD["q:sure"], BOARD["q:suref"]
    assert isinstance(sure, Answer)
    assert isinstance(suref, Answer)
    assert type(sure.probabilities["yes"]) is int
    assert type(suref.probabilities["yes"]) is float
    got, log = run("agree", "q:sure", "q:suref")
    assert got is sure
    assert log[0][2] == ("q:sure", "q:suref")
    got, _ = run("unique", "q:sure", "q:suref")
    assert got == conflict("q:sure", "q:suref")  # equal, yet two overlap


def test_an_int_sum_at_the_domain_bound_is_kept() -> None:
    got, _ = run("collect", "d:max", agg="sum")
    assert got == 2**53 - 1
    got, _ = run("collect", "d:max", "d:negone", "d:one", agg="sum")
    assert got == 2**53 - 1


@pytest.mark.parametrize("refs", [("d:max", "d:one"), ("d:min", "d:negone"), ("d:max", "d:max2")])
def test_an_int_sum_outside_the_records_domain_raises(refs: tuple[str, ...]) -> None:
    board = {**BOARD, "d:max2": 2**53 - 1}
    r = ReadView(Role(ComponentKind.rule, "route"), refs, (), board, {})
    with pytest.raises(ValueError, match="outside the record's value domain"):
        collect(r, *refs, agg="sum")
    assert items(r) == [("collect", refs, ())]


class Loud(str):
    def __repr__(self) -> str:
        raise RuntimeError("caller code ran")


@pytest.mark.parametrize("name", ["priority", "collect"])
def test_a_raise_names_a_ref_by_its_string_data_not_its_repr(name: str) -> None:
    r = view()

    def call() -> object:
        if name == "priority":
            return priority(r, ["chat"], Loud("d:a"))
        return collect(r, Loud("d:a"), agg="sum")

    with pytest.raises(Exception, match='"d:a"') as info:
        call()
    assert not isinstance(info.value, RuntimeError)
    assert items(r) == [(name, ("d:a",), ())]
