# pyright: reportPrivateUsage=false
"""The extraction kind through a turn: what the binding is passed, each value fault and each
temporal grammar at the write, ``truncated``, the ledger's shape and kind checks and its read of
formats 1 and 2, a golden format-2 record, replay and the fixture's form, a format-1 record and
design, the lint's points over an extraction read, and redaction field by field."""

import copy
import hashlib
import hmac
import math
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import rfc8785

import extraction as x
from counting import SmallBounds
from jurat import (
    CallResult,
    Cause,
    CauseCode,
    ExtractionAnswer,
    FindingKind,
    InvalidRecord,
    LoadedDesign,
    Policy,
    Question,
    Record,
    RenderedQuestion,
    Source,
    SourceCall,
    Text,
    Unavailable,
    ValueType,
    decide_sync,
    load,
    read_record,
    redact,
    replay_sync,
)
from jurat import types as t
from jurat._canonical import decode
from jurat._fixture import _binding_form
from jurat._ledger import Ledger, _LedgerError
from jurat.helpers import HELPERS, dead_questions
from jurat.helpers._points import filled

Values = dict[str, Any]
FORMAT_1 = Path(__file__).parent / "format_1"


def binding(values: Values | None = None, **result: Any) -> Callable[..., Any]:
    """A test's own fixture: both answers, changed by ``values``, with ``result``'s members."""

    async def answer(batch: tuple[SourceCall, ...]) -> object:
        return CallResult({"q:amount": x.AMOUNT, "q:dates": x.DATES, **(values or {})}, **result)

    return answer


def ran(
    values: Values | None = None,
    loaded: LoadedDesign | None = None,
    state: Values | None = None,
    **result: Any,
) -> Record:
    """One turn; its record round-trips through ``read_record``."""
    record = decide_sync(
        x.loaded() if loaded is None else loaded,
        x.STATE if state is None else state,
        {"default": binding(values, **result)},
    ).record
    assert read_record(record.document).document == record.document
    return record


def entry(record: Record, ref: str) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "entry" and i["ref"] == ref)


def call_of(record: Record) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "call")


def calls(record: Record) -> dict[str, Any]:
    return cast("dict[str, Any]", next(iter(record.unhashed["calls"].values())))


def invalid_answer(ref: str) -> dict[str, Any]:
    return {"kind": "Unavailable", "cause": {"code": "invalid-answer", "refs": [ref]}}


# ---------------------------------------------------------------------------------------------
# What the binding is passed, and the answer written


def test_the_binding_is_passed_the_extraction_members() -> None:
    seen: list[tuple[SourceCall, ...]] = []

    async def answer(batch: tuple[SourceCall, ...]) -> object:
        seen.append(batch)
        return CallResult({"q:amount": x.AMOUNT, "q:dates": x.DATES})

    decide_sync(x.loaded(), x.STATE, {"default": answer})
    (call,) = seen[0]
    assert call.questions["q:amount"] == RenderedQuestion(
        cast(Any, "extraction"),
        "What is owed in Owe 120 by 2024-02-29?",
        (),
        value_type=ValueType.number,
        is_list=False,
        text="t:state.body",
        reference_date="2024-01-01",
    )
    dates = call.questions["q:dates"]
    assert (dates.value_type, dates.is_list, dates.labels) == (ValueType.date, True, ())
    assert type(dates.value_type) is ValueType
    # The text and the reference date are needs, keyed as declared.
    assert call.needs == {"t:state.body": x.BODY, "t:state.today": "2024-01-01"}


def test_an_extraction_answer_is_written_with_its_eight_members() -> None:
    record = ran()
    assert entry(record, "q:amount")["value"] == {
        "kind": "extraction",
        "value_type": "number",
        "is_list": False,
        "value": 120.0,
        "guess": 120.0,
        "p_null": 0.1,
        "confidence": 0.9,
        "span": [4, 7],
    }
    assert entry(record, "d:amount")["value"] == "big"
    assert entry(record, "d:dates")["value"] == ["2024-02-29"]
    assert call_of(record)["questions"]["q:dates"] == {
        "identity": call_of(record)["questions"]["q:dates"]["identity"],
        "kind": "extraction",
        "labels": None,
        "value_type": "date",
        "is_list": True,
    }
    # No sum is flagged for an extraction answer.
    assert "sums" not in calls(record)


def test_a_q_need_is_the_extraction_answer_itself() -> None:
    seen: list[object] = []

    async def answer(batch: tuple[SourceCall, ...]) -> object:
        for call in batch:
            if call.id == "echo":
                seen.append(call.needs["q:amount"])
                return CallResult({"t:echo": "ok"})
        return CallResult({"q:amount": {**x.AMOUNT, "value": None}, "q:dates": x.DATES})

    d = x.design(
        sources={"read": x.reader(), "echo": Source({"t:echo": t.string}, needs=["q:amount"])}
    )
    decide_sync(load(d), x.STATE, {"default": answer})
    (need,) = seen
    assert type(need) is ExtractionAnswer
    assert need.value is None  # a model's none is a value, never a marker


def test_a_models_none_is_a_value_and_extract_abstains_on_it() -> None:
    none = {"value": None, "guess": None, "p_null": 0.95, "confidence": None, "span": None}
    record = ran({"q:amount": none})
    assert entry(record, "q:amount")["value"]["value"] is None
    assert entry(record, "d:amount")["marker"] == {
        "kind": "Unsure",
        "cause": {"code": "abstained", "refs": ["q:amount"]},
    }


def test_a_refusal_is_the_bindings_marker() -> None:
    record = ran({"q:amount": Unavailable(Cause(CauseCode.refused, ["q:amount"]))})
    assert entry(record, "q:amount")["marker"]["cause"]["code"] == "refused"


AMOUNT_FAULTS: list[Any] = [
    [1, 2],
    "120",
    None,
    {k: v for k, v in x.AMOUNT.items() if k != "span"},
    {**x.AMOUNT, "unit": "EUR"},
    {**x.AMOUNT, "value": "120"},
    {**x.AMOUNT, "value": True},
    {**x.AMOUNT, "value": 2**53},
    {**x.AMOUNT, "value": math.inf},
    {**x.AMOUNT, "guess": "many"},
    {**x.AMOUNT, "p_null": True},
    {**x.AMOUNT, "p_null": 1.5},
    {**x.AMOUNT, "confidence": math.nan},
    {**x.AMOUNT, "span": [4]},
    {**x.AMOUNT, "span": [4, 7.0]},
    {**x.AMOUNT, "span": (4, 7)},
    {**x.AMOUNT, "span": [7, 4]},
    {**x.AMOUNT, "span": [True, 7]},
    {**x.AMOUNT, "span": [-1, 7]},
    {**x.AMOUNT, "span": [4, len(x.BODY) + 1]},
    {**x.AMOUNT, "value": None, "guess": None},
]
DATES_FAULTS: list[Any] = [
    {**x.DATES, "value": ["2023-02-29"]},
    {**x.DATES, "value": ("2024-02-29",)},
    {**x.DATES, "value": ["2024-02-29", None], "span": None},
    {**x.DATES, "guess": ["2024-02-29", None]},
    {**x.DATES, "value": "2024-02-29"},
    {**x.DATES, "span": [[11, 21], [0, 1]]},
    {**x.DATES, "span": [[11, 22]]},
    {**x.DATES, "guess": ["2024-02-30"]},
]


@pytest.mark.parametrize(
    ("ref", "value"),
    [("q:amount", v) for v in AMOUNT_FAULTS] + [("q:dates", v) for v in DATES_FAULTS],
)
def test_each_value_fault_is_invalid_answer_on_its_ref_alone(ref: str, value: Any) -> None:
    record = ran({ref: value})
    assert entry(record, ref)["marker"] == invalid_answer(ref)
    other = "q:dates" if ref == "q:amount" else "q:amount"
    assert "value" in entry(record, other)
    assert call_of(record)["outcome"] == "returned"


def test_an_empty_list_value_is_a_value_apart_from_none() -> None:
    empty: dict[str, Any] = {**x.DATES, "value": [], "guess": None, "span": []}
    written = entry(ran({"q:dates": empty}), "q:dates")["value"]
    assert (written["value"], written["span"]) == ([], [])
    none = {**x.DATES, "value": None, "guess": None, "span": None}
    assert entry(ran({"q:dates": none}), "q:dates")["value"]["value"] is None


@pytest.mark.parametrize("p", [0, 1, -0.0, 0.0, 1.0])
def test_p_null_and_confidence_at_their_ends_are_accepted(p: float) -> None:
    written = entry(ran({"q:amount": {**x.AMOUNT, "p_null": p, "confidence": p}}), "q:amount")
    assert (written["value"]["p_null"], written["value"]["confidence"]) == (p, p)


def test_a_number_list_holds_its_int_items_as_floats() -> None:
    built = ExtractionAnswer(ValueType.number, True, [1, 2.5], [3], None, None, None)
    assert (built.value, built.guess) == ((1.0, 2.5), (3.0,))
    assert all(type(v) is float for v in (*built.value, *built.guess))
    seen: list[object] = []
    listed = {"value": [1, 2.5], "guess": [3], "p_null": None, "confidence": None, "span": None}

    async def answer(batch: tuple[SourceCall, ...]) -> object:
        for call in batch:
            if call.id == "echo":
                seen.append(call.needs["q:dates"])
                return CallResult({"t:echo": "ok"})
        return CallResult({"q:amount": x.AMOUNT, "q:dates": listed})

    numbers = x.question("number", True, wording=Text("How much?"))
    echo = Source({"t:echo": t.string}, needs=["q:dates"])
    d = x.design(sources={"read": x.reader(**{"q:dates": numbers}), "echo": echo})
    decide_sync(load(d), x.STATE, {"default": answer})
    (need,) = seen
    assert type(need) is ExtractionAnswer
    assert (need.value, need.guess) == ((1.0, 2.5), (3.0,))
    assert all(type(v) is float for v in (*need.value, *need.guess))


def test_a_key_of_a_str_subclass_is_read_by_its_string_data() -> None:
    class Key(str):
        __slots__ = ()

    record = ran({"q:amount": {Key(k): v for k, v in x.AMOUNT.items()}})
    assert entry(record, "q:amount")["value"]["value"] == 120.0

    class Apart(str):
        """A key a dict holds apart from its string data's own."""

        __slots__ = ()

        def __hash__(self) -> int:
            return 0

        def __eq__(self, other: object) -> bool:
            return other is self

    twice = {**x.AMOUNT, Apart("value"): 3}
    assert len(twice) == 6
    assert entry(ran({"q:amount": twice}), "q:amount")["marker"] == invalid_answer("q:amount")


def test_a_span_ends_at_most_at_the_texts_length_in_code_points() -> None:
    body = "\U0001f600 owes 5"  # one code point, two UTF-16 units, four UTF-8 bytes
    state = {**x.STATE, "body": body}
    whole = {**x.AMOUNT, "value": 5, "guess": 5, "span": [0, len(body)]}
    written = entry(ran({"q:amount": whole}, state=state), "q:amount")
    assert written["value"]["span"] == [0, 8]
    past = {**whole, "span": [0, len(body) + 1]}
    assert "marker" in entry(ran({"q:amount": past}, state=state), "q:amount")


def test_an_answer_over_the_value_domains_size_bounds_is_invalid_answer(
    small_bounds: SmallBounds,
) -> None:
    many = {**x.DATES, "value": ["2024-02-29"] * 40, "guess": None, "span": None}
    record = ran({"q:dates": many})
    assert entry(record, "q:dates")["marker"] == invalid_answer("q:dates")
    assert "value" in entry(record, "q:amount")


@pytest.mark.parametrize(
    ("value_type", "good", "bad"),
    [
        ("text", "anything", 3),
        ("number", 1.5, "1.5"),
        ("date", "2024-02-29", "2023-02-29"),
        ("time", "23:59:59.5+01:00", "24:00:00"),
        ("datetime", "2024-02-29T00:00:00Z", "2024-02-29T00:00:00z"),
        ("duration", "P1DT2H", "P1W"),
    ],
)
def test_each_temporal_grammar_and_value_type_at_the_write(
    value_type: str, good: object, bad: object
) -> None:
    loaded = load(x.plain(**{"q:amount": x.question(value_type)}))
    scalar = {"guess": None, "p_null": None, "confidence": None, "span": None}
    written = entry(ran({"q:amount": {**scalar, "value": good}}, loaded), "q:amount")
    assert written["value"]["value"] == good
    refused = entry(ran({"q:amount": {**scalar, "value": bad}}, loaded), "q:amount")
    assert refused["marker"] == invalid_answer("q:amount")


# ---------------------------------------------------------------------------------------------
# truncated and passes


def test_truncated_is_written_under_its_call_in_code_point_order() -> None:
    record = ran(truncated=("q:dates", "q:amount"), usage={"passes": 3, "in": 10})
    assert calls(record)["truncated"] == ["q:amount", "q:dates"]
    assert calls(record)["usage"] == {"passes": 3, "in": 10}
    # It changes no outcome: the log is the log of a turn without it.
    assert ran().record_hash == record.record_hash


def test_a_call_that_fails_structurally_keeps_its_truncated() -> None:
    record = ran({"q:nowhere": 1}, truncated=("q:dates",))
    assert call_of(record)["outcome"]["failed"]["code"] == "unexpected-ref"
    assert calls(record)["truncated"] == ["q:dates"]
    record = ran({"q:nowhere": 1}, truncated=("q:nowhere",))
    assert "truncated" not in calls(record)
    assert set(calls(record)["invalid"]) == {"truncated"}


def test_a_question_with_no_reference_date_is_passed_none_and_needs_no_date() -> None:
    seen: list[tuple[SourceCall, ...]] = []

    async def answer(batch: tuple[SourceCall, ...]) -> object:
        seen.append(batch)
        return CallResult({"q:amount": x.AMOUNT, "q:dates": x.DATES})

    undated = x.question(reference_date=None)
    dates = x.question("date", True, wording=Text("When is it due?"), reference_date=None)
    d = x.design(sources={"read": x.reader(**{"q:amount": undated, "q:dates": dates})})
    decide_sync(load(d), x.STATE, {"default": answer})
    (call,) = seen[0]
    assert all(q.reference_date is None for q in call.questions.values())
    assert call.needs == {"t:state.body": x.BODY}


def test_an_empty_truncated_is_not_written() -> None:
    assert "truncated" not in calls(ran())
    assert "invalid" not in calls(ran())


class _Ref(str):
    __slots__ = ()


@pytest.mark.parametrize(
    "value",
    [
        ["q:amount"],
        ("q:amount", "q:amount"),
        ("q:nowhere",),
        ("not a ref",),
        ('q:amount["x"]',),
        ("q:amount", 1),
        (_Ref("q:amount"),),
        "q:amount",
        None,
    ],
)
def test_a_truncated_that_does_not_conform_is_kept_in_invalid_alone(value: Any) -> None:
    record = ran(truncated=value)
    assert "truncated" not in calls(record)
    assert set(calls(record)["invalid"]) == {"truncated"}
    assert ran().record_hash == record.record_hash


# ---------------------------------------------------------------------------------------------
# The ledger


def stored(record: Record) -> dict[str, Any]:
    return {"log": record.log, "unhashed": record.unhashed}


def edited(change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    doc = stored(ran(truncated=("q:amount",)))
    change(doc)
    return doc


def at(doc: dict[str, Any], ref: str) -> dict[str, Any]:
    return next(i for i in doc["log"] if i.get("ref") == ref)


def call_item(doc: dict[str, Any]) -> dict[str, Any]:
    return next(i for i in doc["log"] if i["item"] == "call")


def calls_entry(doc: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", next(iter(doc["unhashed"]["calls"].values())))


def _set(path: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def change(doc: dict[str, Any]) -> None:
        member = at(doc, path[0])["value"]
        if value is _DROP:
            del member[path[1]]
        else:
            member[path[1]] = value

    return change


_DROP = object()


@pytest.mark.parametrize(
    ("change", "says"),
    [
        (_set(("q:amount", "span"), _DROP), "no 'span' member"),
        (_set(("q:amount", "unit"), "EUR"), "member 'unit'"),
        (_set(("q:amount", "value_type"), "money"), "value_type 'money' is not one of the six"),
        (_set(("q:amount", "is_list"), 0), "'is_list' is not a boolean"),
        (_set(("q:amount", "value"), {"a": 1}), "'value' is not null, a string"),
        (_set(("q:amount", "guess"), [[1]]), "'guess' is not null, a string"),
        (_set(("q:amount", "p_null"), "low"), "'p_null' is not a number or null"),
        (_set(("q:amount", "span"), [1, 2, 3]), "'span' is not null"),
        (_set(("q:amount", "span"), [1.0, 2]), "'span' is not null"),
        # Read by its kind, an object of the extraction members is no other kind's answer.
        (_set(("q:amount", "kind"), "noul"), "has the member 'confidence'"),
        (_set(("q:amount", "kind"), "estimate"), "has the member 'confidence'"),
    ],
)
def test_the_ledger_refuses_an_extraction_value_off_its_shape(
    change: Callable[[dict[str, Any]], None], says: str
) -> None:
    with pytest.raises(InvalidRecord, match=says):
        read_record(edited(change))


def test_the_ledger_tests_no_range_grammar_or_span_bound() -> None:
    def change(doc: dict[str, Any]) -> None:
        value = at(doc, "q:amount")["value"]
        value.update(p_null=7, value="not a number", span=[0, 10**6])

    read_record(edited(change))


def test_a_q_value_is_of_its_call_items_question_and_kind() -> None:
    def other_kind(doc: dict[str, Any]) -> None:
        at(doc, "q:amount")["value"] = {
            "kind": "noul",
            "labels": ["yes", "no"],
            "probabilities": {"yes": 1, "no": 0},
        }

    with pytest.raises(InvalidRecord, match="kind 'noul' is not its question's, 'extraction'"):
        read_record(edited(other_kind))

    def unasked(doc: dict[str, Any]) -> None:
        del call_item(doc)["questions"]["q:amount"]
        del calls_entry(doc)["truncated"]

    with pytest.raises(InvalidRecord, match="'q:amount' is not a question of its call item"):
        read_record(edited(unasked))


def test_a_q_marker_is_of_its_call_items_question() -> None:
    def unasked(doc: dict[str, Any]) -> None:
        entry = at(doc, "q:amount")
        del entry["value"]
        entry["marker"] = invalid_answer("q:amount")
        del call_item(doc)["questions"]["q:amount"]
        del calls_entry(doc)["truncated"]

    with pytest.raises(InvalidRecord, match="'q:amount' is not a question of its call item"):
        read_record(edited(unasked))


def spam_record() -> dict[str, Any]:
    """A stored record of a turn that also asks the noul question ``q:spam``."""
    loaded = load(x.with_question("q:spam", Question("noul", Text("Is it?"))))
    return stored(ran({"q:spam": {"yes": 0.9, "no": 0.1}}, loaded))


def test_a_choice_answer_under_a_noul_question_is_refused() -> None:
    doc = spam_record()
    read_record(doc)
    at(doc, "q:spam")["value"]["kind"] = "choice"
    with pytest.raises(InvalidRecord, match="kind 'choice' is not its question's, 'noul'"):
        read_record(doc)


def test_a_probability_question_carrying_value_type_is_refused() -> None:
    doc = spam_record()
    call_item(doc)["questions"]["q:spam"]["value_type"] = "text"
    with pytest.raises(InvalidRecord, match="member 'value_type', which its kind does not name"):
        read_record(doc)


@pytest.mark.parametrize(
    ("member", "value", "says"),
    [
        ("value_type", _DROP, "no 'value_type' member"),
        ("is_list", _DROP, "no 'is_list' member"),
        ("value_type", "money", "value_type 'money' is not one of the six"),
        ("is_list", "no", "'is_list' is not a boolean"),
        ("labels", ["a"], "labels are not null, as an extraction's are"),
        ("unit", "EUR", "member 'unit'"),
    ],
)
def test_the_ledger_refuses_an_extraction_question_off_its_shape(
    member: str, value: Any, says: str
) -> None:
    def change(doc: dict[str, Any]) -> None:
        question = call_item(doc)["questions"]["q:dates"]
        if value is _DROP:
            del question[member]
        else:
            question[member] = value

    with pytest.raises(InvalidRecord, match=says):
        read_record(edited(change))


@pytest.mark.parametrize(
    ("truncated", "says"),
    [
        ([], "empty"),
        (["q:dates", "q:amount"], "distinct and in code-point order"),
        (["q:amount", "q:amount"], "distinct and in code-point order"),
        (["q:other"], "'truncated' has the ref 'q:other', which names no key"),
        (['q:amount["x"]'], "is a grounded ref"),
        ("q:amount", "a list of strings"),
    ],
)
def test_the_ledger_refuses_a_truncated_off_its_shape(truncated: Any, says: str) -> None:
    def change(doc: dict[str, Any]) -> None:
        calls_entry(doc)["truncated"] = truncated

    with pytest.raises(InvalidRecord, match=says):
        read_record(edited(change))


def format_1(doc: dict[str, Any]) -> None:
    doc["log"][0]["format"] = 1


def test_a_format_1_record_holds_only_format_1s_rules() -> None:
    with pytest.raises(InvalidRecord, match="extraction question, which a format-1 record"):
        read_record(edited(format_1))

    # A format-1 log of the probability kinds, with a truncated member or an invalid one.
    record = read_record((FORMAT_1 / "record.json").read_bytes())
    doc = stored(record)
    key = next(iter(doc["unhashed"]["calls"]))
    doc["unhashed"]["calls"][key]["truncated"] = ["q:spam"]
    with pytest.raises(InvalidRecord, match="holds truncated, which a format-1 record does not"):
        read_record(doc)
    doc = stored(record)
    doc["unhashed"]["calls"][key]["invalid"] = {"truncated": "[1]"}
    with pytest.raises(InvalidRecord, match="holds truncated, which a format-1 record does not"):
        read_record(doc)
    doc = stored(record)
    doc["unhashed"]["calls"][key]["invalid"] = {"truncated": "[1]"}
    format_2 = copy.deepcopy(doc)
    format_2["log"][0]["format"] = 2
    read_record(format_2)


def test_an_extraction_answer_in_a_format_1_log_is_refused_without_a_call() -> None:
    def change(doc: dict[str, Any]) -> None:
        format_1(doc)
        doc["log"] = [i for i in doc["log"] if i["item"] != "call"]
        for item in doc["log"]:
            if item.get("call") is not None:
                item["call"] = None
        doc["unhashed"] = {}

    with pytest.raises(InvalidRecord, match="an extraction answer, which a format-1 record"):
        read_record(edited(change))


def test_a_format_1_record_reads_back_to_its_own_hash() -> None:
    data = (FORMAT_1 / "record.json").read_bytes()
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == (
        "sha256:801b6def62ecff453cb884b8b71aca68590c3b1639d27ca4b0d3f4f63893fbe2"
    )
    assert record.log[0] == {
        "item": "design",
        "hash": "sha256:c2760531ea781e26dac6f13edfbfa8e65fe9cd31c6b921580d5dd099dd345791",
        "format": 1,
        "helpers": 1,
    }


def test_the_append_refuses_a_design_item_of_an_unknown_format() -> None:
    ledger = Ledger()
    with pytest.raises(_LedgerError, match="the format 3 is not one this ledger knows"):
        ledger.append({"item": "design", "hash": GOLDEN_X_DESIGN, "format": 3, "helpers": 2})


def test_a_record_of_an_unknown_format_is_refused() -> None:
    def change(doc: dict[str, Any]) -> None:
        doc["log"][0]["format"] = 3

    with pytest.raises(InvalidRecord, match="format 3 is not one this reader knows: 1, 2"):
        read_record(edited(change))


# ---------------------------------------------------------------------------------------------
# The golden format-2 record

GOLDEN_X_DESIGN = "sha256:7207754c8732c54aa2a238eb7617668ae313a6839d25e62bcd95135b9ff6f2f7"
GOLDEN_X_RECORD = "sha256:ecb39d88077d80c5579764f3e2173f308926e088f1a643aec30d2d8ba512344f"


def test_the_golden_extraction_record() -> None:
    record = ran()
    assert record.log[0] == {
        "item": "design",
        "hash": GOLDEN_X_DESIGN,
        "format": 2,
        "helpers": HELPERS,
    }
    assert (record.design_hash, record.record_hash) == (GOLDEN_X_DESIGN, GOLDEN_X_RECORD)


_GOLDEN_PROBE = (
    "from test_extraction_turn import ran\n"
    "record = ran()\n"
    "print(record.design_hash, record.record_hash)\n"
)


@pytest.mark.parametrize("seed", ["0", "12345"])
def test_the_golden_extraction_record_holds_under_every_hash_seed(seed: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _GOLDEN_PROBE],
        capture_output=True,
        encoding="utf-8",
        check=True,
        cwd=Path(__file__).parent,
        env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.stdout.split() == [GOLDEN_X_DESIGN, GOLDEN_X_RECORD]


# ---------------------------------------------------------------------------------------------
# Replay and the fixture binding


def test_an_extraction_turn_replays_with_no_finding() -> None:
    record = ran(truncated=("q:amount",))
    again, findings = replay_sync(read_record(record.document), x.loaded())
    assert findings == []
    assert again is not None
    assert "truncated" not in calls(again)  # the fixture returns truncated empty


def test_the_fixture_returns_the_recorded_answer_in_the_bindings_form() -> None:
    value = decode(ran().document)["log"]
    recorded = next(i for i in value if i.get("ref") == "q:dates")["value"]
    assert recorded["p_null"] == 0
    assert type(recorded["p_null"]) is int
    form = _binding_form(recorded)
    assert form == {
        "value": ["2024-02-29"],
        "guess": ["2024-02-29"],
        "p_null": 0.0,
        "confidence": 1.0,
        "span": [[11, 21]],
    }
    assert type(form["p_null"]) is float
    assert type(form["confidence"]) is float
    amount = next(i for i in value if i.get("ref") == "q:amount")["value"]
    whole = {**amount, "value": 120, "guess": [1, 2], "value_type": "number"}
    assert _binding_form(whole)["value"] == 120.0
    assert _binding_form(whole)["guess"] == [1.0, 2.0]
    # Under another value type a whole number stays as recorded.
    assert _binding_form({**whole, "value_type": "text"})["value"] == 120
    again = _binding_form(recorded)
    # Each call gets fresh containers.
    assert again["span"] is not form["span"]
    assert again["value"] is not form["value"]
    assert again["guess"] is not form["guess"]
    assert form["value"] is not recorded["value"]
    assert form["guess"] is not recorded["guess"]


def changed_question(q: Question) -> list[dict[str, Any]]:
    """Replay a turn of ``x.plain()`` under a copy asking ``q:amount`` as ``q``."""
    record = ran(loaded=load(x.plain()))
    _, findings = replay_sync(read_record(record.document), load(x.plain(**{"q:amount": q})))
    return findings


@pytest.mark.parametrize(
    "q",
    [
        x.question("text"),
        x.question("number", True),
        Question("noul", Text("Is anything owed?")),
    ],
)
def test_a_changed_value_type_list_flag_or_kind_is_question_changed_then_invalid_answer(
    q: Question,
) -> None:
    findings = changed_question(q)
    kinds = [f["kind"] for f in findings]
    assert kinds == [
        FindingKind.design_mismatch,
        FindingKind.question_changed,
        FindingKind.entry,
        FindingKind.entry,
        FindingKind.plan,
    ]
    assert findings[1]["refs"] == ["q:amount"]
    (found,) = [f for f in findings if f["kind"] == FindingKind.entry and f["ref"] == "q:amount"]
    assert found["replayed"]["marker"] == invalid_answer("q:amount")


def test_a_recorded_probability_answer_asked_now_as_an_extraction_is_invalid_answer() -> None:
    noul = Question("noul", Text("Is anything owed?"))
    record = ran({"q:amount": {"yes": 0.9, "no": 0.1}}, loaded=load(x.plain(**{"q:amount": noul})))
    _, findings = replay_sync(read_record(record.document), load(x.plain()))
    (found,) = [f for f in findings if f["kind"] == FindingKind.entry and f["ref"] == "q:amount"]
    assert found["replayed"]["marker"] == invalid_answer("q:amount")


def test_a_format_1_record_replays_under_its_format_1_design_with_helpers_mismatch_alone() -> None:
    loaded = load((FORMAT_1 / "design.json").read_bytes())
    record = read_record((FORMAT_1 / "record.json").read_bytes())
    again, findings = replay_sync(record, loaded)
    assert findings == [{"kind": FindingKind.helpers_mismatch, "recorded": 1, "replayed": HELPERS}]
    assert again is not None
    assert again.log[0]["format"] == 2  # the engine writes its own format


def test_a_format_1_design_loads_to_its_own_hash_and_keeps_its_format() -> None:
    data = (FORMAT_1 / "design.json").read_bytes()
    loaded = load(data)
    assert loaded.design_hash == (
        "sha256:c2760531ea781e26dac6f13edfbfa8e65fe9cd31c6b921580d5dd099dd345791"
    )
    assert loaded.document == data
    assert decode(loaded.document)["format"] == 1
    stored_state = read_record((FORMAT_1 / "record.json").read_bytes()).log[2]["value"]

    async def nothing(batch: tuple[SourceCall, ...]) -> object:
        return None  # each ref Unavailable: the turn still writes its design item

    record = decide_sync(loaded, stored_state, {"mail": nothing, "default": nothing}).record
    assert record.log[0]["item"] == "design"
    assert record.log[0]["format"] == 2  # the engine writes its own format


# ---------------------------------------------------------------------------------------------
# The lint's points


def test_an_extraction_reads_points_cross_value_null_and_confidence() -> None:
    fills, on_grid = filled(x.loaded(), ["q:amount"], None)
    points = [cast("dict[str, Any]", p.record) for p in fills["q:amount"]]
    assert "q:amount" in on_grid
    crossed = [(p["value"], p["p_null"], p["confidence"]) for p in points[:-1]]
    # Value from the grid's value axis (100), p_null from the null axis (0.5), confidence 1.
    assert crossed == [
        *[
            (v, n, c)
            for v in (99.0, 100.0, 101.0)
            for n in (0.0, 0.5, 1.0, None)
            for c in (1.0, None)
        ],
        (None, 1.0, None),
        (None, None, None),
    ]
    assert all(p["guess"] is None and p["span"] is None for p in points[:-1])
    assert points[-1] == {"kind": "Unavailable", "cause": {"code": "refused", "refs": ["q:amount"]}}
    live = fills["q:amount"][0].live
    assert live == {"value": 99.0, "guess": None, "p_null": 0.0, "confidence": 1.0, "span": None}


def test_an_extraction_read_off_the_grid_is_filled_from_values() -> None:
    fills, on_grid = filled(x.loaded(), ["q:dates"], {"q:dates": [["2024-02-29"], []]})
    assert "q:dates" not in on_grid
    points = [cast("dict[str, Any]", p.record) for p in fills["q:dates"]][:-1]
    assert [(p["value"], p["p_null"], p["confidence"]) for p in points] == [
        (["2024-02-29"], 0.0, 1.0),
        (["2024-02-29"], 0.0, None),
        (["2024-02-29"], None, 1.0),
        (["2024-02-29"], None, None),
        ([], 0.0, 1.0),
        ([], 0.0, None),
        ([], None, 1.0),
        ([], None, None),
        (None, 1.0, None),
        (None, None, None),
    ]


@pytest.mark.parametrize(
    ("values", "says"),
    cast(
        "list[tuple[Any, str]]",
        [
            (None, 'unfilled: "q:dates"'),
            ({"q:dates": [["2023-02-29"]]}, "does not conform to its type"),
            ({"q:dates": ["2024-02-29"]}, "does not conform to its type"),
            ({"q:dates": [None]}, "does not conform to its type"),
            ({"q:dates": [[], []]}, "lists one value twice"),
            ({"q:dates": []}, "is empty"),
            ({"q:dates": [[]], "q:amount": [3]}, "its grid points or its type fill"),
        ],
    ),
)
def test_an_extraction_reads_fill_raises_before_any_run(values: Any, says: str) -> None:
    with pytest.raises(ValueError, match=says):
        dead_questions(x.loaded(), values=values)


def test_an_extraction_value_outside_the_value_domain_is_refused_before_any_run(
    small_bounds: SmallBounds,
) -> None:
    with pytest.raises(ValueError, match='a value for "q:dates" is outside the value domain'):
        dead_questions(x.loaded(), values={"q:dates": [["2024-02-29"] * 40]})


@pytest.mark.parametrize(
    ("listed", "says"),
    cast(
        "list[tuple[list[Any], str]]",
        [
            ([True], "does not conform to its type"),
            ([2**53], "does not conform to its type"),
            ([math.nan], "does not conform to its type"),
            ([math.inf], "does not conform to its type"),
            ([5, 5.0], "lists one value twice"),
        ],
    ),
)
def test_an_extraction_number_value_in_values_is_of_its_domain(
    listed: list[Any], says: str
) -> None:
    with pytest.raises(ValueError, match=says):
        dead_questions(load(x.plain()), values={"q:amount": listed})


def gridded(entry: dict[str, list[str]]) -> LoadedDesign:
    """The design with ``q:amount``'s grid entry replaced."""
    d = x.design()
    rule = replace(d.rules["amount"], boundaries={"q:amount": entry})
    return load(x.replaced(d, rules={**d.rules, "amount": rule}))


def test_an_extraction_grid_entry_naming_confidence_gives_its_leaves() -> None:
    loaded = gridded({"null": [], "confidence": ["amount/null"], "value": ["amount/big"]})
    fills, on_grid = filled(loaded, ["q:amount"], None)
    assert "q:amount" in on_grid
    points = [cast("dict[str, Any]", p.record) for p in fills["q:amount"]][:-3]
    assert {p["confidence"] for p in points} == {0.0, 0.5, 1.0, None}
    assert {p["p_null"] for p in points} == {0.0, None}


def test_an_extraction_grid_entry_with_no_value_leaf_takes_its_values_from_values() -> None:
    loaded = gridded({"null": ["amount/null"], "confidence": [], "value": []})
    fills, on_grid = filled(loaded, ["q:amount"], {"q:amount": [5]})
    assert "q:amount" in on_grid
    points = [cast("dict[str, Any]", p.record) for p in fills["q:amount"]][:-3]
    assert {p["value"] for p in points} == {5.0}
    assert {p["p_null"] for p in points} == {0.0, 0.5, 1.0, None}


STATE_VALUES: dict[str, list[Any]] = {"t:state.body": [x.BODY], "t:state.today": ["2024-01-01"]}


def test_the_dead_question_lint_runs_extraction_points() -> None:
    values = {**STATE_VALUES, "q:dates": [["2024-02-29"], []]}
    report = dead_questions(x.loaded(), values=values)
    assert report["questions"]["q:amount"]["status"] == "live"
    assert report["questions"]["q:dates"]["status"] == "live"
    witness = report["questions"]["q:amount"]["witness"]
    # The first pair in grid order: 99 is small at p_null 0 and abstains at the null bar.
    assert [(w["q:amount"]["value"], w["q:amount"]["p_null"]) for w in witness] == [
        (99.0, 0.0),
        (99.0, 0.5),
    ]

    ignored = x.loaded(policy=Policy(x.dates_only, version="1", reads=["d:dates"]))
    report = dead_questions(ignored, values={**STATE_VALUES, "q:dates": [["2024-02-29"]]})
    assert report["questions"]["q:amount"] == {"status": "no-change-found", "witness": None}


def test_an_extraction_read_with_no_grid_entry_that_changes_nothing_is_not_on_grid() -> None:
    d = x.plain()
    loaded = load(x.replaced(d, policy=Policy(x.dates_only, version="1", reads=["d:dates"])))
    values = {**STATE_VALUES, "q:amount": [5], "q:dates": [["2024-02-29"]]}
    report = dead_questions(loaded, values=values)
    assert report["questions"]["q:amount"] == {"status": "not-on-grid", "witness": None}


# ---------------------------------------------------------------------------------------------
# Redaction


ROOT = bytes(range(32))


def digest(value: Any, record: Record) -> str:
    """A keyed digest computed apart from the code under test."""
    key = hmac.digest(ROOT, record.record_hash.encode("ascii") + b"\x01", "sha256")
    return "hmac-sha256:" + hmac.new(key, rfc8785.dumps(value), hashlib.sha256).hexdigest()


def redacted(record: Record) -> dict[str, Any]:
    return cast("dict[str, Any]", decode(redact(record, ROOT)))


def test_an_invalid_truncated_redacts_to_a_digest() -> None:
    record = ran(truncated=["q:amount"])
    text = calls(record)["invalid"]["truncated"]
    (keyed,) = redacted(record)["unhashed"]["calls"].values()
    assert keyed["invalid"] == {"truncated": digest(text, record)}


def test_an_extraction_answer_redacts_field_by_field() -> None:
    record = ran()
    view = redacted(record)
    amount = next(i for i in view["log"] if i.get("ref") == "q:amount")["value"]
    assert amount == {
        "kind": "extraction",
        "value_type": "number",
        "is_list": False,
        "value": digest(120.0, record),
        "guess": digest(120.0, record),
        "p_null": 0.1,
        "confidence": 0.9,
        "span": digest([4, 7], record),
    }
    dates = next(i for i in view["log"] if i.get("ref") == "q:dates")["value"]
    assert dates["value"] == digest(["2024-02-29"], record)  # a list of content: one digest
    assert dates["span"] == digest([[11, 21]], record)


def test_a_declared_tree_redacts_an_extraction_answer_by_it() -> None:
    record = ran(
        {"q:amount": {**x.AMOUNT, "guess": None}},
        loaded=x.loaded(classes={"q:amount": "open", "q:dates": {"p_null": "identifier"}}),
    )
    view = redacted(record)
    amount = next(i for i in view["log"] if i.get("ref") == "q:amount")["value"]
    assert (amount["value"], amount["guess"], amount["span"]) == (120.0, None, [4, 7])
    dates = next(i for i in view["log"] if i.get("ref") == "q:dates")["value"]
    assert dates["p_null"] == digest(0, record)
    assert dates["value"] == digest(["2024-02-29"], record)


def test_a_truncated_ref_passes_redaction_open() -> None:
    record = ran(truncated=("q:amount",))
    (entry_,) = redacted(record)["unhashed"]["calls"].values()
    assert entry_["truncated"] == ["q:amount"]


# ---------------------------------------------------------------------------------------------
# The observer


def test_an_extraction_turn_with_and_without_an_observer_has_the_same_record() -> None:
    seen: list[dict[str, Any]] = []
    answers = binding(truncated=("q:amount",))
    watched = decide_sync(
        x.loaded(), x.STATE, {"default": answers}, observer=lambda i, item: seen.append(item)
    ).record
    plain = decide_sync(x.loaded(), x.STATE, {"default": answers}).record
    assert watched.record_hash == plain.record_hash
    assert seen == watched.log
    assert calls(watched)["truncated"] == calls(plain)["truncated"] == ["q:amount"]
