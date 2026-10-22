# pyright: reportPrivateUsage=false
"""Format 3: a stored format-2 design and record read under format 2's rules to their own hash;
label descriptions at load, in the document, the call and the record; a call's warnings and
precision; the binding's ``unsupported`` refusal; the record document's ``hash``; the edited
tracebacks; and the ledger's refusal of each format-3 form under formats 1 and 2."""

import copy
import dataclasses
import enum
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest

from counting import SmallBounds
from designs import bounded, minimal, slot, with_source
from jurat import (
    CallResult,
    Cause,
    CauseCode,
    Design,
    Failed,
    FindingKind,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Plan,
    Policy,
    Question,
    Record,
    RenderedQuestion,
    Rule,
    RunSettings,
    Source,
    SourceCall,
    Text,
    Unasked,
    Unavailable,
    Unsure,
    When,
    decide_sync,
    load,
    read_record,
    redact,
    replay_sync,
)
from jurat import types as t
from jurat._canonical import decode, encode, hash_value
from jurat._component import exception_record
from jurat._values import MAX_DOCUMENT_DEPTH
from jurat.helpers import pick_with_margin
from jurat.helpers._points import filled
from ledger_logs import DESCRIBED, UNHASHED, fresh, rehashed
from scripted import sleeping, unsupported

FORMAT_2 = Path(__file__).parent / "format_2"
FORMAT_2_DESIGN = "sha256:56f74e4909ca929686a9ec747af56e98ced7a47774c2a43463113a6a61e0a9f4"
FORMAT_2_RECORD = "sha256:b0a8c2aa57a346234e343b9e0945141b4670b4a354686e34dae874c9c10aedc7"

Batch = tuple[SourceCall, ...]


# ---------------------------------------------------------------------------------------------
# The stored format-2 design and record


def format_2_design() -> LoadedDesign:
    return load((FORMAT_2 / "design.json").read_bytes())


def format_2_record() -> Record:
    return read_record((FORMAT_2 / "record.json").read_bytes())


def test_a_format_2_design_loads_to_its_own_hash_and_keeps_its_format() -> None:
    data = (FORMAT_2 / "design.json").read_bytes()
    loaded = load(data)
    assert loaded.design_hash == FORMAT_2_DESIGN
    assert loaded.document == data
    assert decode(loaded.document)["format"] == 2


def test_a_format_2_record_reads_back_to_its_own_hash_with_no_hash_member() -> None:
    data = (FORMAT_2 / "record.json").read_bytes()
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == FORMAT_2_RECORD
    assert record.log[0] == {"item": "design", "hash": FORMAT_2_DESIGN, "format": 2, "helpers": 2}
    assert set(decode(data)) == {"log", "unhashed"}
    call = next(i for i in record.log if i["item"] == "call")
    assert "precision" not in call


def test_a_format_2_record_replays_under_its_format_2_design_with_no_finding() -> None:
    again, findings = replay_sync(format_2_record(), format_2_design())
    assert findings == []
    assert again is not None
    assert again.log[0]["format"] == 3  # the engine writes its own format
    assert decode(again.document)["hash"] == again.record_hash


def test_a_format_2_record_under_the_format_3_design_is_design_mismatch_alone() -> None:
    from test_decide import LOADED

    _, findings = replay_sync(format_2_record(), LOADED)
    assert findings == [
        {
            "kind": FindingKind.design_mismatch,
            "recorded": FORMAT_2_DESIGN,
            "replayed": LOADED.design_hash,
        }
    ]


def test_an_undescribed_question_keeps_its_identity_across_formats() -> None:
    from test_decide import go

    def identity(record: Record) -> str:
        call = next(i for i in record.log if i["item"] == "call")
        return cast(str, call["questions"]["q:spam"]["identity"])

    assert identity(go().record) == identity(format_2_record())


def test_a_format_2_design_decides_a_format_3_turn() -> None:
    stored_state = format_2_record().log[2]["value"]

    async def nothing(batch: Batch) -> object:
        return None

    record = decide_sync(
        format_2_design(), stored_state, {"mail": nothing, "default": nothing}
    ).record
    assert record.log[0] == {"item": "design", "hash": FORMAT_2_DESIGN, "format": 3, "helpers": 2}


# ---------------------------------------------------------------------------------------------
# A described design

WARM = {"name": "warm", "description": "Friendly or grateful"}
STATE_T = t.record(body=t.string)
STATE = {"body": "Thanks so much"}


def tone(r: Any) -> Any:
    return pick_with_margin(r, "q:tone", 0.5, 0.1)


def other(r: Any) -> Any:
    r.read("q:other")
    return True


def inform(r: Any) -> Plan:
    return Plan("inform", reply="ok")


def tone_question(labels: Any = (WARM, "cold"), kind: str = "choice") -> Question:
    return Question(kind, Text("Which tone is {body}?", [slot("body", ref="t:state.body")]), labels)


def described(question: Question | None = None, **changes: Any) -> Design:
    fields: dict[str, Any] = {
        "state": STATE_T,
        "sources": {
            "ask": Source(
                {
                    "q:tone": tone_question() if question is None else question,
                    "q:other": Question("noul", Text("Is it spam?")),
                },
                needs=["t:state.body"],
                cost=1,
            )
        },
        "rules": {
            "tone": Rule("d:tone", t.label("cold", "warm"), tone, version="1", reads=["q:tone"]),
            "other": Rule("d:other", t.boolean, other, version="1", reads=["q:other"]),
        },
        "policy": Policy(inform, version="1"),
    }
    fields.update(changes)
    return minimal(**fields)


TONE = {"q:tone": {"warm": 0.8, "cold": 0.2}, "q:other": {"yes": 0.1, "no": 0.9}}


def answering(values: dict[str, Any] = TONE, **result: Any) -> Callable[[Batch], Any]:
    async def answer(batch: Batch) -> object:
        return CallResult(dict(values), **result)

    return answer


def ran(
    binding: Callable[[Batch], Any] | None = None,
    loaded: LoadedDesign | None = None,
    settings: RunSettings | None = None,
) -> Record:
    """One turn of the described design; its record round-trips through ``read_record``."""
    record = decide_sync(
        load(described()) if loaded is None else loaded,
        STATE,
        {"default": answering() if binding is None else binding},
        settings,
    ).record
    assert read_record(record.document).document == record.document
    return record


def call_of(record: Record) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "call")


def calls_entry(record: Record) -> dict[str, Any]:
    return cast("dict[str, Any]", next(iter(record.unhashed["calls"].values())))


def entry(record: Record, ref: str) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "entry" and i["ref"] == ref)


def refused(design: Design) -> list[Any]:
    with pytest.raises(InvalidDesign) as info:
        load(design)
    return list(info.value.failures)


def only(design: Design, check: int) -> list[Any]:
    found = refused(design)
    assert {f.check for f in found} == {check}, found
    return found


# ---------------------------------------------------------------------------------------------
# Labels at load


def test_a_question_holds_its_labels_as_a_tuple_of_names_and_read_only_copies() -> None:
    given: dict[str, str] = {"name": "warm", "description": "Friendly"}
    q = tone_question([given, "cold"])
    labels: object = q.labels
    assert type(labels) is tuple
    first = cast("tuple[Any, ...]", labels)[0]
    assert type(first) is MappingProxyType
    assert first == given
    given["description"] = "changed"  # the caller's dict reaches no design
    assert first["description"] == "Friendly"


def test_a_described_design_loads_and_writes_its_labels_as_declared() -> None:
    loaded = load(described())
    asked = decode(loaded.document)["sources"]["ask"]["provides"]["q:tone"]
    assert asked["labels"] == [WARM, "cold"]
    assert decode(loaded.document)["format"] == 3
    assert loaded._labels["q:tone"] == (("warm", "cold"), ("Friendly or grateful", None))
    assert loaded._labels["q:other"] == (("yes", "no"), (None, None))


def test_a_described_document_loads_back_byte_equal_to_its_own_hash() -> None:
    loaded = load(described())
    again = load(loaded.document)
    assert again.document == loaded.document
    assert again.design_hash == loaded.design_hash
    assert again._labels == loaded._labels


LEVELS = ("low", {"name": "high", "description": "Needs an answer today"})


def urgency(r: Any) -> Any:
    return pick_with_margin(r, "q:tone", 0.5, 0.1)


def scored() -> Design:
    """``described`` asking a described ``score`` question in place of the choice."""
    rules = {
        **described().rules,
        "tone": Rule("d:tone", t.label("high", "low"), urgency, version="1", reads=["q:tone"]),
    }
    return described(tone_question(LEVELS, kind="score"), rules=rules)


def test_a_described_score_question_loads_round_trips_and_is_asked_as_declared() -> None:
    loaded = load(scored())
    assert loaded._labels["q:tone"] == (("low", "high"), (None, "Needs an answer today"))
    again = load(loaded.document)
    assert (again.document, again.design_hash) == (loaded.document, loaded.design_hash)
    levels = [{"name": "high", "description": "Needs an answer today"}]
    asked = decode(loaded.document)["sources"]["ask"]["provides"]["q:tone"]["labels"]
    assert asked == ["low", *levels]
    seen: list[Batch] = []

    async def answer(batch: Batch) -> object:
        seen.append(batch)
        return CallResult({"q:tone": {"low": 0.1, "high": 0.9}, "q:other": {"yes": 0, "no": 1}})

    record = ran(answer, loaded=loaded)
    assert seen[0][0].questions["q:tone"].descriptions == (None, "Needs an answer today")
    assert call_of(record)["questions"]["q:tone"]["labels"] == ["low", *levels]
    assert entry(record, "q:tone")["value"]["labels"] == ["low", "high"]
    assert entry(record, "d:tone")["value"] == "high"


def test_descriptions_may_repeat() -> None:
    same = {"name": "cold", "description": "Friendly or grateful"}
    load(described(tone_question([WARM, same])))


@pytest.mark.parametrize(
    ("labels", "says"),
    [
        ([{"name": "warm"}, "cold"], "not exactly a name and a description"),
        ([{**WARM, "colour": "red"}, "cold"], "not exactly a name and a description"),
        ([{"name": "warm", "description": None}, "cold"], "name or description is not a string"),
        ([{"name": 1, "description": "x"}, "cold"], "name or description is not a string"),
        ([{"name": "warm", "description": ""}, "cold"], "no character that is not whitespace"),
        ([{"name": "warm", "description": " \t\n"}, "cold"], "no character that is not whitespace"),
        ([{"name": "warm", "description": "\ud800"}, "cold"], "not of Unicode scalar values"),
        ([WARM, {"name": "warm", "description": "again"}], "repeat a label"),
        ([WARM, "warm"], "repeat a label"),
        ([WARM, 3], "neither a string nor an object"),
        ("warm", "not a list of labels"),
    ],
)
def test_check_10_refuses_a_label_off_its_form(labels: Any, says: str) -> None:
    (failure,) = only(described(tone_question(labels)), 10)
    assert says in failure.message, failure.message


def test_check_10_bounds_names_and_descriptions_together(monkeypatch: pytest.MonkeyPatch) -> None:
    import jurat._form as form

    monkeypatch.setattr(form, "MAX_LABEL_CODE_POINTS", 8)
    load(described(tone_question([{"name": "w", "description": "abc"}, "cold"])))
    (failure,) = only(described(tone_question([{"name": "w", "description": "abcd"}, "cold"])), 10)
    assert failure.message.endswith("labels hold more than 2^3 code points")
    # Counted before any is read: a surrogate past the bound is the bound's.
    big = {"name": "w", "description": "abcd\ud800"}
    (failure,) = only(described(tone_question([big, "cold"])), 10)
    assert failure.message.endswith("labels hold more than 2^3 code points")


@pytest.mark.parametrize("kind", ["noul", "extraction"])
def test_a_noul_or_extraction_question_with_object_labels_still_declares_labels(kind: str) -> None:
    q = Question(kind, Text("Is it?"), [WARM])
    found = refused(with_source(minimal(), "s", Source({"q:x": q})))
    assert any(
        f"is a{'n' if kind == 'extraction' else ''} {kind} question and declares labels"
        in f.message
        for f in found
    )


def test_a_candidate_valued_sets_fixed_part_takes_the_described_form() -> None:
    labels = {"fixed": [WARM], "candidates": "t:pool"}
    found = refused(described(tone_question(labels)))
    assert not any(f.check == 10 for f in found)
    assert any("candidate-valued label set" in f.message for f in found)


def described_bounded(thresholds: dict[str, float]) -> Design:
    """``bounded`` with its tone question described and a thresholds leaf keyed as given."""
    design = bounded()
    source = design.sources["intent"]
    q = cast(Question, source.provides["q:intent/tone"])
    q = dataclasses.replace(q, labels=({"name": "warm", "description": "Kind"}, "cold"))
    source = dataclasses.replace(source, provides={**source.provides, "q:intent/tone": q})
    from jurat import Parameter

    score = design.rules["score"]
    score = dataclasses.replace(
        score,
        boundaries={
            "q:intent/tone": {"bar": [], "thresholds": ["tone/keyed"], "margin": [], "mass": []}
        },
    )
    return dataclasses.replace(
        design,
        sources={**design.sources, "intent": source},
        rules={**design.rules, "score": score},
        parameters={
            **design.parameters,
            "tone/keyed": Parameter(t.mapping(t.number(0, 1)), thresholds),
        },
    )


def test_check_9_reads_the_label_names_alone() -> None:
    load(described_bounded({"warm": 0.6}))
    found = only(described_bounded({"Kind": 0.6}), 9)
    assert any('the key "Kind"' in f.message for f in found)


def test_the_lint_points_take_the_label_names() -> None:
    loaded = load(described())
    fills, _ = filled(loaded, ["q:tone"], None)
    records: list[object] = [p.record for p in fills["q:tone"]]
    labels = {
        tuple(cast("dict[str, Any]", r)["labels"])
        for r in records
        if type(r) is dict and "labels" in r
    }
    assert labels == {("warm", "cold")}


# ---------------------------------------------------------------------------------------------
# Labels in the design document under check 11


def document_of(design: Design) -> dict[str, Any]:
    return cast("dict[str, Any]", decode(load(design).document))


def tone_labels(doc: dict[str, Any]) -> list[Any]:
    return cast("list[Any]", doc["sources"]["ask"]["provides"]["q:tone"]["labels"])


def check_11(doc: dict[str, Any]) -> str:
    with pytest.raises(InvalidDesign) as info:
        load(doc)
    (failure,) = info.value.failures
    assert failure.check == 11
    return failure.message


@pytest.mark.parametrize("fmt", [1, 2])
def test_check_11_refuses_a_described_label_in_an_earlier_format(fmt: int) -> None:
    doc = document_of(described())
    doc["format"] = fmt
    assert f"has a described label, which format {fmt} does not have" in check_11(doc)


def test_check_11_refuses_a_described_label_with_a_member_missing_or_extra() -> None:
    doc = document_of(described())
    del tone_labels(doc)[0]["description"]
    assert 'described label has no member "description"' in check_11(doc)
    doc = document_of(described())
    tone_labels(doc)[0]["colour"] = "red"
    assert 'has the member "colour", which its form does not have' in check_11(doc)


def test_a_format_2_document_with_undescribed_labels_loads() -> None:
    doc = document_of(described(tone_question(["warm", "cold"])))
    doc["format"] = 2
    loaded = load(doc)
    assert decode(loaded.document)["format"] == 2


# ---------------------------------------------------------------------------------------------
# Labels in the call and the record


def test_the_binding_is_passed_names_and_aligned_descriptions() -> None:
    seen: list[Batch] = []

    async def answer(batch: Batch) -> object:
        seen.append(batch)
        return CallResult(dict(TONE))

    ran(answer)
    (call,) = seen[0]
    assert call.questions["q:tone"] == RenderedQuestion(
        cast(Any, "choice"),
        "Which tone is Thanks so much?",
        ("warm", "cold"),
        descriptions=("Friendly or grateful", None),
    )
    assert call.questions["q:other"].descriptions == (None, None)


def test_an_extraction_question_is_passed_no_descriptions() -> None:
    import extraction as x

    seen: list[Batch] = []

    async def answer(batch: Batch) -> object:
        seen.append(batch)
        return CallResult({"q:amount": x.AMOUNT, "q:dates": x.DATES})

    decide_sync(x.loaded(), x.STATE, {"default": answer})
    assert seen[0][0].questions["q:amount"].descriptions == ()


def test_the_call_item_holds_the_labels_as_declared_and_the_answer_their_names() -> None:
    record = ran()
    assert call_of(record)["questions"]["q:tone"]["labels"] == [WARM, "cold"]
    value = entry(record, "q:tone")["value"]
    assert value["labels"] == ["warm", "cold"]
    assert entry(record, "d:tone")["value"] == "warm"


def test_a_probability_keyed_by_a_description_is_invalid_answer() -> None:
    record = ran(answering({**TONE, "q:tone": {"Friendly or grateful": 0.8, "cold": 0.2}}))
    assert entry(record, "q:tone")["marker"]["cause"] == {
        "code": "invalid-answer",
        "refs": ["q:tone"],
    }


def test_a_description_change_is_question_changed_and_design_mismatch() -> None:
    undescribed = load(described(tone_question(["warm", "cold"])))
    record = ran(loaded=undescribed)
    _, findings = replay_sync(read_record(record.document), load(described()))
    kinds = [f["kind"] for f in findings]
    assert kinds[:2] == [FindingKind.design_mismatch, FindingKind.question_changed]
    assert findings[1]["refs"] == ["q:tone"]
    again = load(described(tone_question([{**WARM, "description": "Kind"}, "cold"])))
    _, findings = replay_sync(read_record(ran().document), again)
    assert [f["kind"] for f in findings][:2] == kinds[:2]


def test_a_described_turn_replays_with_no_finding() -> None:
    record = ran(answering(precision=2, warnings=("rounded to 2 places",)))
    again, findings = replay_sync(read_record(record.document), load(described()))
    assert findings == []
    assert again is not None
    # The fixture returns null precision and no warnings.
    assert call_of(again)["precision"] is None
    assert "warnings" not in calls_entry(again)


# ---------------------------------------------------------------------------------------------
# Warnings


def test_warnings_are_written_under_the_call_in_the_bindings_order() -> None:
    lines = ("second", "first", "second")
    record = ran(answering(warnings=lines))
    assert calls_entry(record)["warnings"] == list(lines)
    assert call_of(record)["outcome"] == "returned"


def test_no_warnings_write_no_member() -> None:
    assert "warnings" not in calls_entry(ran())


class Line(str):
    __slots__ = ()


@pytest.mark.parametrize(
    "warnings",
    [
        ["a list"],
        ("",),
        ("ok", " \t\n"),
        ("ok", 3),
        (Line("a subclass"),),
        ("\ud800",),
        ("x",) * (2**8 + 1),
        ("x" * 2**15, "x" * 2**15, "y"),
        "a string",
    ],
    ids=[
        "list",
        "empty",
        "whitespace",
        "int",
        "subclass",
        "surrogate",
        "too-many",
        "too-long",
        "string",
    ],
)
def test_a_warnings_off_its_form_is_kept_in_invalid_and_is_no_fault(warnings: Any) -> None:
    record = ran(answering(warnings=warnings))
    keyed = calls_entry(record)
    assert "warnings" not in keyed
    assert keyed["invalid"]["warnings"]
    assert call_of(record)["outcome"] == "returned"
    assert entry(record, "d:tone")["value"] == "warm"


def test_warnings_at_their_bounds_are_written() -> None:
    record = ran(answering(warnings=("x",) * 2**8))
    assert len(calls_entry(record)["warnings"]) == 2**8
    record = ran(answering(warnings=("x" * 2**15, "y" * 2**15)))
    assert len(calls_entry(record)["warnings"]) == 2


def test_warnings_beside_a_structural_fault_are_still_written() -> None:
    record = ran(answering({**TONE, "q:nowhere": 1}, warnings=("odd",)))
    assert call_of(record)["outcome"] == {
        "failed": {"code": "unexpected-ref", "refs": ["q:other", "q:tone"]}
    }
    assert calls_entry(record)["warnings"] == ["odd"]


def test_a_late_result_keeps_no_warnings() -> None:
    late = sleeping(1, CallResult(dict(TONE), warnings=("too late",)))
    record = ran(late, settings=RunSettings(timeout=0.01))
    assert call_of(record)["outcome"] == {
        "failed": {"code": "timed-out", "refs": ["q:other", "q:tone"]}
    }
    assert "warnings" not in calls_entry(record)


def test_redact_digests_the_warnings_as_one_value() -> None:
    record = ran(answering(warnings=("one", "two")))
    view = decode(redact(record, bytes(range(32))))
    from test_views import digest

    assert view["unhashed"]["calls"]["3"]["warnings"] == digest(["one", "two"], record)
    assert view["hash"] == record.record_hash


# ---------------------------------------------------------------------------------------------
# Precision


@pytest.mark.parametrize("precision", [None, 0, 2, 17])
def test_a_conforming_precision_is_hashed_on_the_call_item(precision: int | None) -> None:
    record = ran(answering(precision=precision))
    assert call_of(record)["precision"] == precision
    assert call_of(record)["outcome"] == "returned"


class Places(enum.IntEnum):
    two = 2


@pytest.mark.parametrize("precision", [-1, 18, True, 2.0, "2", 2**64, Places.two])
def test_a_precision_off_its_form_is_invalid_return_for_the_whole_call(precision: Any) -> None:
    record = ran(answering(precision=precision))
    call = call_of(record)
    assert call["outcome"] == {"failed": {"code": "invalid-return", "refs": ["q:other", "q:tone"]}}
    assert call["precision"] is None
    assert calls_entry(record)["invalid"]["precision"]


def test_a_failed_call_item_holds_a_null_precision() -> None:
    async def failed(batch: Batch) -> object:
        return Failed(Cause(CauseCode.raised, ["q:other", "q:tone"]))

    assert call_of(ran(failed))["precision"] is None


# ---------------------------------------------------------------------------------------------
# The binding's refusal before any call


def test_an_unsupported_refusal_marks_each_ref_and_keeps_its_reason_as_a_warning() -> None:
    record = ran(unsupported("no tone questions here"))
    call = call_of(record)
    assert call["outcome"] == "returned"
    assert (call["snapshot"], call["elicitation"], call["prompt"]) == (None, None, None)
    for ref in ("q:other", "q:tone"):
        assert entry(record, ref)["marker"] == {
            "kind": "Unavailable",
            "cause": {"code": "unsupported", "refs": [ref]},
        }
    assert calls_entry(record)["warnings"] == ["no tone questions here"]
    rounds = [i for i in record.log if i["item"] == "round"]
    assert rounds[0]["cost"] == 1  # the call is charged


def test_one_ref_refused_keeps_the_rest_of_the_batch() -> None:
    marker = Unavailable(Cause(CauseCode.unsupported))
    record = ran(answering({**TONE, "q:other": marker}, warnings=("no noul",)))
    assert entry(record, "q:other")["marker"]["cause"] == {
        "code": "unsupported",
        "refs": ["q:other"],
    }
    assert entry(record, "q:tone")["value"]["labels"] == ["warm", "cold"]


def test_unsupported_is_no_failed_outcome() -> None:
    async def failed(batch: Batch) -> object:
        return Failed(Cause(CauseCode.unsupported, ["q:other", "q:tone"]))

    record = ran(failed)
    assert call_of(record)["outcome"]["failed"]["code"] == "invalid-return"


MARKERS = (Unavailable, Unasked, Unsure)


def passing_tone(r: Any) -> Any:
    a = r.read("q:tone")
    if isinstance(a, MARKERS):
        return a
    return tone(r)


def requiring_tone(r: Any) -> Plan:
    r.require("d:tone")
    return Plan("inform", reply="ok")


def test_a_rule_passing_an_unsupported_marker_through_writes_it_and_a_require_escalates() -> None:
    rules = {
        "tone": Rule(
            "d:tone", t.label("cold", "warm"), passing_tone, version="1", reads=["q:tone"]
        ),
        "other": Rule("d:other", t.boolean, other, version="1", reads=["q:other"]),
    }
    policy = Policy(requiring_tone, version="1", reads=["d:tone"])
    record = ran(unsupported(), load(described(rules=rules, policy=policy)))
    assert entry(record, "d:tone")["marker"] == {
        "kind": "Unavailable",
        "cause": {"code": "unsupported", "refs": ["q:tone"]},
    }
    plan = next(i for i in record.log if i["item"] == "plan")
    assert plan["kind"] == "escalate"
    assert plan["cause"]["code"] == "required"


def test_an_unsupported_turn_replays_with_no_finding() -> None:
    record = ran(unsupported())
    _, findings = replay_sync(read_record(record.document), load(described()))
    assert findings == []


# ---------------------------------------------------------------------------------------------
# The record document's hash


def test_a_format_3_document_holds_its_record_hash() -> None:
    record = ran()
    doc = json.loads(record.document)
    assert set(doc) == {"hash", "log", "unhashed"}
    assert doc["hash"] == record.record_hash == hash_value(doc["log"], MAX_DOCUMENT_DEPTH)
    assert record.document.startswith(b'{"hash":"sha256:')
    assert record.document == encode(decode(record.document), MAX_DOCUMENT_DEPTH)


def test_the_hash_is_recomputed_at_the_read_and_a_mismatch_refused() -> None:
    record = ran()
    doc = json.loads(record.document)
    doc["log"][-1]["kind"] = "refused"  # an edit made without recomputing the hash
    doc["log"][-1]["failures"] = [{"step": "plan", "requirement": None, "ref": None}]
    with pytest.raises(InvalidRecord):
        read_record(doc)
    doc = json.loads(record.document)
    doc["log"][2]["value"]["body"] = "edited"
    with pytest.raises(InvalidRecord, match="is not its log's record hash"):
        read_record(doc)
    read_record(rehashed(doc))  # recomputed, it reads
    doc = json.loads(record.document)
    tone = next(i for i in doc["log"] if i.get("ref") == "q:tone")
    tone["value"]["probabilities"] = {"warm": 0.1, "cold": 0.9}
    with pytest.raises(InvalidRecord, match="is not its log's record hash"):
        read_record(doc)


def test_a_missing_hash_is_refused_before_any_item_is_read() -> None:
    doc = json.loads(ran().document)
    del doc["hash"]
    doc["log"][1] = {"item": "bogus"}  # a bad log, never reached
    with pytest.raises(InvalidRecord, match="has no hash member"):
        read_record(doc)


_DROP = object()


@pytest.mark.parametrize(
    ("member", "value", "says"),
    [
        ("hash", _DROP, "has no hash member"),
        ("hash", None, "hash that is not a string"),
        ("hash", "sha256:" + "0" * 64, "is not its log's record hash"),
        ("extra", 1, "other than log, unhashed and hash"),
    ],
)
def test_a_format_3_document_off_its_hash_is_refused(member: str, value: Any, says: str) -> None:
    doc = json.loads(ran().document)
    if value is _DROP:
        del doc[member]
    else:
        doc[member] = value
    with pytest.raises(InvalidRecord, match=says):
        read_record(doc)


def test_a_format_2_document_holding_a_hash_is_refused() -> None:
    doc = json.loads((FORMAT_2 / "record.json").read_bytes())
    doc["hash"] = FORMAT_2_RECORD
    with pytest.raises(InvalidRecord, match="holds hash, which a format-2 record does not"):
        read_record(doc)


# ---------------------------------------------------------------------------------------------
# The ledger under formats 1 and 2


def described_doc(fmt: int) -> dict[str, Any]:
    """The format-3 test log, made format ``fmt``."""
    log = fresh(DESCRIBED)
    log[0]["format"] = fmt
    return {"log": log, "unhashed": copy.deepcopy(UNHASHED["described"])}


def test_the_format_3_test_log_reads() -> None:
    read_record(rehashed(described_doc(3)))


def strip(
    doc: dict[str, Any],
    *,
    precision: bool = True,
    label: bool = True,
    marker: bool = True,
    warnings: bool = True,
) -> dict[str, Any]:
    """The format-3 forms of the test log dropped, each unless asked to stay."""
    call = doc["log"][3]
    if precision:
        del call["precision"]
    if label:
        call["questions"]["q:tone"]["labels"] = ["warm", "cold"]
    if marker:
        doc["log"][4]["marker"]["cause"]["code"] = "refused"
    if warnings:
        del doc["unhashed"]["calls"]["3"]["warnings"]
    return doc


@pytest.mark.parametrize("fmt", [1, 2])
def test_a_format_1_or_2_record_without_format_3s_forms_reads(fmt: int) -> None:
    read_record(strip(described_doc(fmt)))


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize(
    ("kept", "says"),
    [
        ("precision", "holds precision, which a format-{fmt} record does not"),
        ("label", "declare a described label, which a format-{fmt} record does not"),
        ("marker", "code 'unsupported' is not in format {fmt}'s cause set"),
        ("warnings", "holds warnings, which a format-{fmt} record does not"),
    ],
)
def test_a_format_3_form_in_an_earlier_format_is_refused(fmt: int, kept: str, says: str) -> None:
    doc = strip(described_doc(fmt), **{kept: False})
    with pytest.raises(InvalidRecord, match=re.escape(says.format(fmt=fmt))):
        read_record(doc)


@pytest.mark.parametrize("name", ["warnings", "precision"])
def test_a_format_3_invalid_key_in_an_earlier_format_is_refused(name: str) -> None:
    doc = strip(described_doc(2))
    doc["unhashed"]["calls"]["3"]["invalid"] = {name: "[1]"}
    with pytest.raises(InvalidRecord, match=f"holds {name}, which a format-2 record does not"):
        read_record(doc)
    doc = rehashed(described_doc(3))
    doc["unhashed"]["calls"]["3"]["invalid"] = {name: "[1]"}
    read_record(doc)


@pytest.mark.parametrize("code", ["unsupported"])
def test_an_unsupported_cause_elsewhere_in_a_format_2_log_is_refused(code: str) -> None:
    doc = strip(described_doc(2))
    doc["log"][3]["outcome"] = {"failed": {"code": code, "refs": ["q:other", "q:tone"]}}
    doc["log"][4] = {**doc["log"][4], "call": 3}
    with pytest.raises(InvalidRecord, match="its outcome's cause's code 'unsupported'"):
        read_record(doc)
    doc = strip(described_doc(2))
    doc["log"][7]["cause"] = {"code": code, "refs": []}
    with pytest.raises(InvalidRecord, match="its cause's code 'unsupported'"):
        read_record(doc)


@pytest.mark.parametrize(
    ("value", "says"),
    [
        (_DROP, "has no 'precision' member"),
        ("2", "'precision' is not an integer or null"),
        (True, "'precision' is not an integer or null"),
        (2.5, "'precision' is not an integer or null"),
    ],
)
def test_a_format_3_call_items_precision_off_its_kind_is_refused(value: Any, says: str) -> None:
    doc = described_doc(3)
    if value is _DROP:
        del doc["log"][3]["precision"]
    else:
        doc["log"][3]["precision"] = value
    with pytest.raises(InvalidRecord, match=re.escape(says)):
        read_record(rehashed(doc))


@pytest.mark.parametrize("label", [{"name": "x"}, {"name": "x", "description": 1}, None])
def test_a_format_3_call_items_label_off_its_shape_is_refused(label: Any) -> None:
    doc = described_doc(3)
    doc["log"][3]["questions"]["q:tone"]["labels"].append(label)
    with pytest.raises(InvalidRecord, match="is not a list of labels"):
        read_record(rehashed(doc))


@pytest.mark.parametrize(
    ("warnings", "says"),
    [
        ([], "is empty; an empty member is not written"),
        (["a", 1], "is not a list of strings"),
        ("a", "is not a list of strings"),
    ],
)
def test_a_warnings_off_its_shape_is_refused_at_the_read(warnings: Any, says: str) -> None:
    doc = rehashed(described_doc(3))
    doc["unhashed"]["calls"]["3"]["warnings"] = warnings
    with pytest.raises(InvalidRecord, match=says):
        read_record(doc)


def test_the_warnings_are_size_tested_as_one_value_at_the_read(small_bounds: SmallBounds) -> None:
    doc = rehashed(described_doc(3))
    doc["unhashed"]["calls"]["3"]["warnings"] = ["x" * 300]
    with pytest.raises(InvalidRecord, match="entry 0's 'warnings' is outside the value domain"):
        read_record(doc)


def test_a_described_fixed_part_reads_in_a_format_3_call_item() -> None:
    doc = described_doc(3)
    doc["log"][3]["questions"]["q:tone"]["labels"] = {"fixed": [WARM], "candidates": "t:pool"}
    read_record(rehashed(doc))
    doc = strip(described_doc(2))
    doc["log"][3]["questions"]["q:tone"]["labels"] = {"fixed": [WARM], "candidates": "t:pool"}
    with pytest.raises(InvalidRecord, match="described label"):
        read_record(doc)


# ---------------------------------------------------------------------------------------------
# Tracebacks


def frames(text: str) -> list[str]:
    """The ``File`` names of a traceback's frames, in order."""
    return re.findall(r'File "([^"]*)"', text)


def broken_rule(r: Any) -> Any:
    raise ValueError("the rule broke")


def broken_policy(r: Any) -> Plan:
    raise ValueError("the policy broke")


def helper_misused(r: Any) -> Any:
    return pick_with_margin(r, "q:tone", 2.0, 0.1)


def exceptions(record: Record) -> list[dict[str, str]]:
    return list(record.unhashed["exceptions"].values())


def text_of(record: Record) -> str:
    (text,) = exceptions(record)
    return text["traceback"]


def no_paths(names: list[str]) -> bool:
    return all("/" not in n and "\\" not in n for n in names)


def test_a_raising_rules_traceback_starts_at_the_rule_and_names_modules() -> None:
    rules = {
        "tone": Rule("d:tone", t.label("cold", "warm"), broken_rule, version="1", reads=["q:tone"])
    }
    text = text_of(ran(loaded=load(described(rules={**described().rules, **rules}))))
    names = frames(text)
    assert names[0] == "test_format_3"
    assert not any(n.startswith("jurat") for n in names)
    assert no_paths(names)
    assert text.endswith("ValueError: the rule broke\n")


def test_a_helpers_frames_inside_a_rule_are_kept() -> None:
    rules = {
        "tone": Rule(
            "d:tone", t.label("cold", "warm"), helper_misused, version="1", reads=["q:tone"]
        )
    }
    names = frames(text_of(ran(loaded=load(described(rules={**described().rules, **rules})))))
    assert names[0] == "test_format_3"
    assert any(n.startswith("jurat.helpers") for n in names[1:])
    assert no_paths(names)


def test_a_raising_policys_traceback_starts_at_the_policy() -> None:
    text = text_of(ran(loaded=load(described(policy=Policy(broken_policy, version="1")))))
    names = frames(text)
    assert names[0] == "test_format_3"
    assert not any(n.startswith("jurat") for n in names)
    assert no_paths(names)
    assert text.endswith("ValueError: the policy broke\n")


def test_a_raising_codecs_traceback_starts_at_the_codec() -> None:
    def codec(state: object) -> object:
        raise ValueError("bad state")

    record = decide_sync(load(described()), STATE, {"default": answering()}, codec=codec).record
    names = frames(text_of(record))
    assert names[0] == "test_format_3"
    assert no_paths(names)


def test_a_raising_bindings_traceback_starts_at_the_binding() -> None:
    async def binding(batch: Batch) -> object:
        raise RuntimeError("down")

    names = frames(text_of(ran(binding)))
    assert names == ["test_format_3"]


def test_a_chain_and_a_group_name_every_frame_by_its_module() -> None:
    def raise_value() -> None:
        raise ValueError("inner")

    def chained(r: Any) -> Any:
        try:
            raise_value()
        except ValueError as exc:
            raise KeyError("outer") from exc

    def grouped(r: Any) -> Any:
        errors: list[Exception] = []
        for _ in range(2):
            try:
                raise_value()
            except ValueError as exc:
                errors.append(exc)
        raise ExceptionGroup(f"two {len(errors)}", errors)

    for fn in (chained, grouped):
        rules = {"tone": Rule("d:tone", t.label("cold", "warm"), fn, version="1", reads=["q:tone"])}
        text = text_of(ran(loaded=load(described(rules={**described().rules, **rules}))))
        names = frames(text)
        assert len(names) >= 3, text
        assert set(names) == {"test_format_3"}, text


def test_a_frame_with_no_module_name_is_named_by_its_basename_and_a_pseudo_file_kept() -> None:
    namespace: dict[str, Any] = {}
    code = compile(
        "def f():\n    raise ValueError('nameless')\n", "/some/where/nameless.py", "exec"
    )
    exec(code, namespace)  # no __name__ global
    pseudo: dict[str, Any] = {"__name__": "pseudo"}
    exec(compile("def g():\n    raise ValueError('pseudo')\n", "<string>", "exec"), pseudo)
    win: dict[str, Any] = {"__name__": 7}
    exec(compile("def h():\n    raise ValueError('win')\n", "C:\\code\\win.py", "exec"), win)
    for fn, last in (
        (namespace["f"], "nameless.py"),
        (pseudo["g"], "<string>"),
        (win["h"], "win.py"),
    ):
        with pytest.raises(ValueError, match=r"nameless|pseudo|win") as info:
            fn()
        assert frames(exception_record(info.value)["traceback"])[-1] == last


def test_asyncio_frames_between_the_engine_and_the_component_are_dropped() -> None:
    def component() -> None:
        raise ValueError("deep")

    engine: dict[str, Any] = {"__name__": "jurat._probe"}
    loop: dict[str, Any] = {"__name__": "asyncio.probe"}
    exec(compile("def hop(fn):\n    fn()\n", "<asyncio probe>", "exec"), loop)
    source = (
        "def run(fn):\n    try:\n        hop(fn)\n    except ValueError as e:\n        return e\n"
    )
    scope = {**engine, "hop": loop["hop"]}
    exec(compile(source, "<engine probe>", "exec"), scope)
    exc = scope["run"](component)
    assert frames(exception_record(exc)["traceback"]) == ["test_format_3"]
    # A stack that opens with asyncio frames, as a component's own task group's does, drops
    # them too.
    runner: dict[str, Any] = {"__name__": "asyncio.runner", "hop": loop["hop"]}
    exec(compile(source, "<asyncio runner>", "exec"), runner)
    exc = runner["run"](component)
    assert frames(exception_record(exc)["traceback"]) == ["test_format_3"]
    # A stack that opens with another frame keeps every frame after it.
    outer: dict[str, Any] = {"__name__": "caller", "hop": loop["hop"]}
    exec(compile(source, "/py/caller.py", "exec"), outer)
    exc = outer["run"](component)
    assert frames(exception_record(exc)["traceback"]) == [
        "caller",
        "<asyncio probe>",
        "test_format_3",
    ]


def test_a_builtin_components_frames_are_all_kept() -> None:
    # A builtin as a function when: it has no frame of its own, so the engine's are kept.
    source = described().sources["ask"]
    gated = dataclasses.replace(source, when=When(cast(Any, int), version="1"))
    record = ran(loaded=load(described(sources={"ask": gated})))
    text = text_of(record)
    names = frames(text)
    assert names, text
    assert all(n.startswith("jurat") for n in names), text


# ---------------------------------------------------------------------------------------------
# The golden format-3 record

GOLDEN_DESIGN = "sha256:058a91ad6eda0b83ada368bb50a8e15650dcc004335532010c7e5ee9661417a2"
GOLDEN_RECORD = "sha256:b25ee036cef42cdfe37fbd3cc3d0adc6b110ffa14dfd1f16e5babb7d9673c741"


def golden() -> Record:
    return ran(answering(precision=2, warnings=("rounded to 2 places",)))


def test_the_golden_described_record() -> None:
    record = golden()
    assert (record.design_hash, record.record_hash) == (GOLDEN_DESIGN, GOLDEN_RECORD)
    assert decode(record.document)["hash"] == GOLDEN_RECORD


_GOLDEN_PROBE = (
    "from test_format_3 import golden\n"
    "record = golden()\n"
    "print(record.design_hash, record.record_hash)\n"
)


@pytest.mark.parametrize("seed", ["0", "12345"])
def test_the_golden_described_record_holds_under_every_hash_seed(seed: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _GOLDEN_PROBE],
        capture_output=True,
        encoding="utf-8",
        check=True,
        cwd=Path(__file__).parent,
        env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.stdout.split() == [GOLDEN_DESIGN, GOLDEN_RECORD]
