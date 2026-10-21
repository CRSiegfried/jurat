# pyright: reportPrivateUsage=false
"""The fixture binding: what it returns for each kind of
recorded call and entry, matched per source from the call item that lists it and per ref from
an entry naming that call; and the typed decoder it shares with replay's state entry."""

import asyncio
import json
from typing import Any

import pytest

from jurat import (
    AnswerKind,
    CallResult,
    Cause,
    CauseCode,
    Failed,
    InvalidRecord,
    LoadedDesign,
    Record,
    RenderedQuestion,
    Source,
    SourceCall,
    Unavailable,
    read_record,
)
from jurat import types as t
from jurat._fixture import FixtureBinding, RecordedCalls
from jurat._typed import decoded
from jurat._types import AnswerType, Interval, ListOf, MappingOf, Nullable, RecordOf, Scalar
from scripted import Scripted, raising, returns
from test_decide import ASK, BODY, SPAM, bound, go, slice_design

NOUL = RenderedQuestion(AnswerKind.noul, "Is it spam?", ("yes", "no"))
BLOBS = Source({"t:blob": t.bytes_, "t:weight": t.number(), "t:size": t.integer()})
LOADED = slice_design(sources={"ask": ASK, "body": BODY, "look": BLOBS})


def ask(*, refs: tuple[str, ...] = (), questions: Any = None) -> SourceCall:
    return SourceCall("ask", None, {}, refs, {"q:spam": NOUL} if questions is None else questions)


def body(refs: tuple[str, ...] = ("t:body",)) -> SourceCall:
    return SourceCall("body", None, {}, refs, {})


def look(refs: tuple[str, ...] = ("t:blob", "t:size", "t:weight")) -> SourceCall:
    return SourceCall("look", None, {}, refs, {})


def recorded(loaded: LoadedDesign = LOADED, **replies: object) -> Record:
    defaults: dict[str, object] = {
        "ask": returns(SPAM),
        "look": returns({"t:blob": b"\x00\xff", "t:weight": 3, "t:size": 4}),
    }
    return go(loaded, binds=bound(Scripted({**defaults, **replies}))).record


def fixture(record: Record, loaded: LoadedDesign = LOADED) -> FixtureBinding:
    return FixtureBinding(RecordedCalls(record.log), loaded)


def answer(binding: FixtureBinding, *batch: SourceCall) -> object:
    return asyncio.run(binding(batch))


def values(result: object) -> dict[str, Any]:
    assert type(result) is CallResult
    for name in ("snapshot", "elicitation", "prompt", "usage", "attempts"):
        assert getattr(result, name) is None
    return result.values


def edited(record: Record, change: Any) -> Record:
    doc = json.loads(record.document)
    change(doc["log"])
    return read_record(doc)


# ---------------------------------------------------------------------------------------------
# Values, typed by the given design


def test_each_asked_ref_gets_its_recorded_value_typed_and_provenance_is_null() -> None:
    got = values(answer(fixture(recorded()), ask(), body(), look()))
    assert got == {
        "q:spam": {"yes": 0.9, "no": 0.1},
        "t:body": "text of the mail",
        "t:blob": b"\x00\xff",
        "t:size": 4,
        "t:weight": 3.0,
    }
    assert type(got["t:weight"]) is float
    assert type(got["t:size"]) is int


def test_a_q_value_is_its_probabilities_with_an_integral_one_a_float() -> None:
    record = recorded(ask=returns({**SPAM, "q:spam": {"yes": 1, "no": 0}}))
    got = values(answer(fixture(record), ask()))
    assert got["q:spam"] == {"yes": 1.0, "no": 0.0}
    assert all(type(p) is float for p in got["q:spam"].values())


def test_a_q_value_that_is_not_an_answers_object_is_refused_before_the_fixture() -> None:
    # The ledger refuses it at the read, so the fixture reads probabilities unchecked.
    def change(log: list[Any]) -> None:
        for item in log:
            if item.get("ref") == "q:spam":
                item["value"] = "not an answer"

    with pytest.raises(InvalidRecord, match="'value' is not an object"):
        edited(recorded(), change)


@pytest.mark.parametrize("code", ["refused", "invalid-answer", "invalid-return"])
def test_a_ref_recorded_with_a_binding_marker_comes_back_as_that_marker(code: str) -> None:
    reply = {**SPAM, "q:spam": Unavailable(Cause(code, ["x"]))}
    if code == "invalid-answer":
        reply["q:spam"] = {"yes": 2}
    if code == "invalid-return":
        reply["q:spam"] = Unavailable(Cause("not-returned", ["x"]))  # not one a binding may return
    got = values(answer(fixture(recorded(ask=returns(reply))), ask(), body()))
    assert got["q:spam"] == Unavailable(Cause(code, ["q:spam"]))
    assert got["t:body"] == "text of the mail"


def test_a_ref_recorded_with_any_other_marker_gets_no_key() -> None:
    got = values(answer(fixture(recorded(ask=returns({"t:body": "b"}))), ask(), body()))
    assert got == {"t:body": "b"}  # q:spam was not-returned


def test_a_value_that_no_longer_decodes_is_passed_on_undecoded() -> None:
    def change(log: list[Any]) -> None:
        for item in log:
            if item.get("ref") == "t:blob":
                item["value"] = "not base64url!"

    got = values(answer(fixture(edited(recorded(), change)), look()))
    assert got["t:blob"] == "not base64url!"


# ---------------------------------------------------------------------------------------------
# Matching per source and per ref


def test_a_source_not_in_the_record_gets_no_key_and_its_batchmates_are_answered() -> None:
    extra = SourceCall("extra", None, {}, ("t:extra",), {})
    got = values(answer(fixture(recorded()), extra, look()))
    assert set(got) == {"t:blob", "t:size", "t:weight"}


def test_a_ref_with_no_entry_under_its_sources_call_gets_no_key() -> None:
    # t:body's entry names ask's call, so asked under look it is not answered.
    got = values(answer(fixture(recorded()), look(("t:blob", "t:body"))))
    assert got == {"t:blob": b"\x00\xff"}


def test_only_the_asked_refs_are_returned() -> None:
    got = values(answer(fixture(recorded()), ask(questions={}), body()))
    assert got == {"t:body": "text of the mail"}


# ---------------------------------------------------------------------------------------------
# Failed calls


def test_a_failed_call_comes_back_as_its_outcome_when_it_holds_every_asked_source() -> None:
    record = recorded(ask=raising(ValueError("down")))
    got = answer(fixture(record), ask())
    assert got == Failed(Cause(CauseCode.raised, ["q:spam", "t:body"]))
    assert type(got) is Failed


def test_a_timed_out_call_comes_back_at_once() -> None:
    record = recorded(ask=Failed(Cause("timed-out", [])))
    assert answer(fixture(record), ask()) == Failed(Cause("timed-out", ["q:spam", "t:body"]))


def test_a_source_in_a_failed_call_beside_others_gets_no_key() -> None:
    record = recorded(ask=raising(ValueError("down")))
    got = values(answer(fixture(record), ask(), look()))
    assert set(got) == {"t:blob", "t:size", "t:weight"}


def test_two_sources_in_two_failed_calls_get_no_keys() -> None:
    record = recorded(ask=raising(ValueError("down")), look=raising(ValueError("down")))
    assert values(answer(fixture(record), ask(), look())) == {}


# ---------------------------------------------------------------------------------------------
# No state between calls, fresh containers


def test_each_call_is_answered_alike_and_shares_no_container() -> None:
    nested = slice_design(
        sources={
            "ask": ASK,
            "body": BODY,
            "look": Source({"t:size": t.list_(t.record(n=t.number()))}),
        }
    )
    record = recorded(nested, look=returns({"t:size": [{"n": 1}]}))
    binding = fixture(record, nested)
    first = values(answer(binding, look(("t:size",))))
    first["t:size"][0]["n"] = 99
    second = values(answer(binding, look(("t:size",))))
    assert second == {"t:size": [{"n": 1.0}]}
    assert second["t:size"] is not first["t:size"]


def test_the_record_is_left_as_it_was() -> None:
    record = recorded()
    before = record.document
    binding = fixture(record)
    got = values(answer(binding, ask(), look()))
    got["q:spam"]["yes"] = 0
    assert record.document == before
    assert values(answer(binding, ask()))["q:spam"] == {"yes": 0.9, "no": 0.1}


# ---------------------------------------------------------------------------------------------
# The typed decoder


NUMBER = Scalar("number")
BYTES = Scalar("bytes")


@pytest.mark.parametrize(
    ("value", "type_", "expected"),
    [
        ("AAE", BYTES, b"\x00\x01"),
        ("", BYTES, b""),
        ("AA=", BYTES, "AA="),  # padded: not unpadded base64url, left as it is
        ("A", BYTES, "A"),
        (7, BYTES, 7),
        (3, NUMBER, 3.0),
        (1, Interval("number", 0, 1), 1.0),
        (3, Interval("integer", 0, 5), 3),
        (3, Scalar("integer"), 3),
        (True, NUMBER, True),  # a boolean is never a number
        (0.5, NUMBER, 0.5),
        (None, Nullable(NUMBER), None),
        (2, Nullable(NUMBER), 2.0),
        ([1, "AAE"], ListOf(NUMBER), [1.0, "AAE"]),
        ({"a": "AAE"}, MappingOf(BYTES), {"a": b"\x00\x01"}),
        (
            {"n": 1, "b": "AAE", "extra": {"k": [1]}},
            RecordOf((("b", BYTES), ("n", NUMBER))),
            {"n": 1.0, "b": b"\x00\x01", "extra": {"k": [1]}},
        ),
        ("x", RecordOf((("n", NUMBER),)), "x"),
        ([1], MappingOf(NUMBER), [1]),
        ({"a": 1}, ListOf(NUMBER), {"a": 1}),
        ({"a": [1]}, AnswerType(), {"a": [1]}),
    ],
)
def test_decoded_types_a_canonical_value_and_never_refuses(
    value: object, type_: Any, expected: object
) -> None:
    got = decoded(value, type_)
    assert got == expected
    assert type(got) is type(expected)


def test_decoded_builds_every_container_afresh() -> None:
    value: dict[str, Any] = {"extra": {"k": [1]}, "n": [1]}
    got: Any = decoded(value, RecordOf((("n", ListOf(NUMBER)),)))
    assert got["extra"] is not value["extra"]
    assert got["extra"]["k"] is not value["extra"]["k"]
    assert got["n"] is not value["n"]


def test_a_recorded_null_is_a_value_not_a_missing_key() -> None:
    nullable = slice_design(
        sources={
            "ask": ASK,
            "body": BODY,
            "look": Source({"t:size": t.nullable(t.integer())}),
        }
    )
    record = recorded(nullable, look=returns({"t:size": None}))
    assert values(answer(fixture(record, nullable), look(("t:size",)))) == {"t:size": None}


# ---------------------------------------------------------------------------------------------
# Regrouped batches: a recorded call of several sources, asked in other groupings


def merged(record: Record) -> Record:
    """The record with look's call item folded into ask's, as one call of every source, its
    outcome ``Failed`` naming every ref when either was, else returned."""
    doc = json.loads(record.document)
    log: list[Any] = doc["log"]
    first, second = 3, 4
    assert log[first]["sources"] == ["ask", "body"]
    assert log[second]["sources"] == ["look"]
    a, b = log[first], log.pop(second)
    a["sources"] = ["ask", "body", "look"]
    a["questions"] = {**a["questions"], **b["questions"]}
    if a["outcome"] != "returned" or b["outcome"] != "returned":
        code = (a["outcome"] if a["outcome"] != "returned" else b["outcome"])["failed"]["code"]
        a["outcome"] = {"failed": {"code": code, "refs": ["q:spam", "t:blob", "t:body"]}}
    for item in log[second:]:
        if item["item"] == "entry" and item["call"] is not None:
            item["call"] = first
    doc["unhashed"] = {}
    return read_record(doc)


def test_a_subset_of_a_recorded_failed_call_of_several_sources_comes_back_as_that_failed() -> None:
    record = merged(recorded(ask=raising(ValueError("down"))))
    binding = fixture(record)
    expected = Failed(Cause(CauseCode.raised, ["q:spam", "t:blob", "t:body"]))
    assert answer(binding, ask()) == expected
    assert answer(binding, look()) == expected
    assert answer(binding, ask(), look()) == expected


def test_a_recorded_returned_call_split_across_two_batches_answers_each_half() -> None:
    binding = fixture(merged(recorded()))
    assert values(answer(binding, ask(), body())) == {
        "q:spam": {"yes": 0.9, "no": 0.1},
        "t:body": "text of the mail",
    }
    assert values(answer(binding, look())) == {"t:blob": b"\x00\xff", "t:size": 4, "t:weight": 3.0}


def test_a_copy_holding_a_value_where_a_marker_was_answers_the_value() -> None:
    reply = {**SPAM, "q:spam": Unavailable(Cause("refused", ["q:spam"]))}

    def change(log: list[Any]) -> None:
        item = next(i for i in log if i.get("ref") == "q:spam")
        del item["marker"]
        item["value"] = {
            "kind": "noul",
            "labels": ["yes", "no"],
            "probabilities": {"yes": 1, "no": 0},
        }

    got = values(answer(fixture(edited(recorded(ask=returns(reply)), change)), ask()))
    assert got["q:spam"] == {"yes": 1.0, "no": 0.0}


def test_a_t_ref_the_given_design_dropped_is_not_asked_and_so_left_out() -> None:
    got = values(answer(fixture(recorded()), ask(refs=())))
    assert got == {"q:spam": {"yes": 0.9, "no": 0.1}}


def test_the_fixture_logs_the_refs_each_source_was_asked() -> None:
    binding = fixture(recorded())
    answer(binding, ask(refs=()), look(("t:size",)))
    assert binding.asked == {"ask": ("q:spam",), "look": ("t:size",)}
