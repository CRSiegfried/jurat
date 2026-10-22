# pyright: reportPrivateUsage=false
"""``from_record``: a caller's live turn answered from a stored record under the oracle names
it is bound to, each of lap 2's stored-binding traps refused or kept as the record holds it,
over real decide runs."""

import asyncio
import dataclasses
import json
from typing import Any, cast

import pytest

import extraction
from jurat import (
    AnswerKind,
    Cause,
    CauseCode,
    EntryRefusal,
    ExtractionAnswer,
    InvalidRecord,
    LoadedDesign,
    Question,
    Record,
    RenderedQuestion,
    Rule,
    Source,
    SourceCall,
    Text,
    Unavailable,
    ValueType,
    from_record,
    load,
    replay_sync,
)
from jurat import types as t
from jurat._fixture import _bounded, _differing, _FromRecord
from jurat._interpret import MAX_WARNING_CODE_POINTS, MAX_WARNINGS, _record_form
from scripted import Scripted, raising, returns
from test_decide import (
    ASK,
    BODY,
    LOADED,
    LOOK,
    REPLIES,
    RULES,
    SPAM,
    STATE,
    bound,
    go,
    slice_design,
)

U = CauseCode.unsupported


def stored(loaded: LoadedDesign = LOADED, state: Any = STATE, **replies: object) -> Record:
    """A live turn's record under ``loaded``, every oracle scripted."""
    return go(loaded, state, binds=bound(Scripted({**REPLIES, **replies}))).record


def mixed(
    record: object, loaded: LoadedDesign = LOADED, state: Any = STATE, **replies: object
) -> tuple[Record, Scripted]:
    """A live turn with ``mail`` answered from ``record`` and ``default`` scripted live."""
    live = Scripted({**REPLIES, **replies})
    binds = {"mail": from_record(record, loaded), "default": live}
    return go(loaded, state, binds=binds).record, live


def entries(record: Record) -> dict[str, dict[str, Any]]:
    return {i["ref"]: i for i in record.log if i["item"] == "entry"}


def markers(record: Record) -> dict[str, str]:
    return {r: e["marker"]["cause"]["code"] for r, e in entries(record).items() if "marker" in e}


def calls(record: Record) -> list[dict[str, Any]]:
    return [i for i in record.log if i["item"] == "call"]


def warned(record: Record, sid: str) -> list[str]:
    """The warnings of the call item listing ``sid``."""
    log = record.log
    index = next(n for n, i in enumerate(log) if i["item"] == "call" and sid in i["sources"])
    return record.unhashed["calls"][str(index)].get("warnings", [])


def plan(record: Record) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "plan")


# ---------------------------------------------------------------------------------------------
# Whole turns


def test_a_full_turn_answered_from_its_record_reproduces_the_plan_and_replays_clean() -> None:
    record = stored()
    binding = from_record(record, LOADED)
    again = go(binds=bound(binding)).record
    assert plan(again) == plan(record)
    assert {r: e.get("value") for r, e in entries(again).items()} == {
        r: e.get("value") for r, e in entries(record).items()
    }
    assert replay_sync(again, LOADED)[1] == []


def test_a_mixed_turn_answers_one_oracle_from_the_record_and_the_other_live() -> None:
    record = stored()
    again, live = mixed(record, look=returns({"t:size": 9}))
    assert [b[0].id for b in live.batches] == ["look"]
    assert entries(again)["t:size"]["value"] == 9
    assert entries(again)["q:spam"] == entries(record)["q:spam"]
    assert entries(again)["t:body"] == entries(record)["t:body"]
    assert plan(again) == plan(record)


def test_a_stored_document_bytes_or_parsed_answers_as_the_record_does() -> None:
    record = stored()
    for given in (record.document, json.loads(record.document)):
        assert plan(mixed(given)[0]) == plan(record)


def test_an_extraction_turn_answered_from_its_record_reproduces_the_plan() -> None:
    loaded = extraction.loaded()
    record = go(loaded, extraction.STATE, binds={"default": Scripted(_extraction_replies())}).record
    again = go(loaded, extraction.STATE, binds={"default": from_record(record, loaded)}).record
    assert plan(again) == plan(record)
    assert replay_sync(again, loaded)[1] == []


def _extraction_replies() -> dict[str, object]:
    return {"read": returns({"q:amount": extraction.AMOUNT, "q:dates": extraction.DATES})}


# ---------------------------------------------------------------------------------------------
# Lap 2's traps: (1) described labels


def _toned(description: str) -> LoadedDesign:
    labels = ({"name": "warm", "description": description}, "cold")
    tone = Source({"q:tone": Question("choice", Text("Which tone?"), labels=labels)}, oracle="mail")
    return slice_design(sources={"ask": ASK, "body": BODY, "look": LOOK, "tone": tone})


TONE = {"q:tone": {"warm": 0.7, "cold": 0.3}}


def test_a_described_label_answers_when_the_live_question_is_described_alike() -> None:
    loaded = _toned("friendly")
    record = stored(loaded, ask=returns({**SPAM, **TONE}))
    again = mixed(record, loaded)[0]
    assert entries(again)["q:tone"]["value"] == entries(record)["q:tone"]["value"]
    assert "q:tone" not in markers(again)


def test_a_described_label_is_refused_when_the_description_differs() -> None:
    record = stored(_toned("friendly"), ask=returns({**SPAM, **TONE}))
    again = mixed(record, _toned("kindly"))[0]
    assert markers(again) == {"q:tone": "unsupported"}
    assert "'q:tone': its labels differ from the stored question's" in warned(again, "ask")


# ---------------------------------------------------------------------------------------------
# (2) recorded markers


def test_a_recorded_invalid_answer_is_kept_as_recorded() -> None:
    record = stored(ask=returns({"q:spam": {"maybe": 1.0}, "t:body": "b"}))
    assert markers(record)["q:spam"] == "invalid-answer"
    again = mixed(record)[0]
    assert markers(again)["q:spam"] == "invalid-answer"
    assert warned(again, "ask") == [f"answered from record {record.record_hash}"]


def test_a_recorded_refused_comes_back_unsupported_with_a_line_naming_it() -> None:
    refused = Unavailable(Cause(CauseCode.refused))
    record = stored(ask=returns({"q:spam": refused, "t:body": "b"}))
    again = mixed(record)[0]
    assert markers(again)["q:spam"] == "unsupported"
    assert "'q:spam': recorded refused, no answer to give" in warned(again, "ask")


def test_a_recorded_not_returned_comes_back_unsupported_with_a_line_naming_it() -> None:
    record = stored(ask=returns({"q:spam": {"yes": 0.9, "no": 0.1}}))
    assert markers(record)["t:body"] == "not-returned"
    again = mixed(record)[0]
    assert markers(again)["t:body"] == "unsupported"
    assert "'t:body': recorded not-returned, no answer to give" in warned(again, "body")


# ---------------------------------------------------------------------------------------------
# (3) answers for another request


def test_needs_that_differ_from_the_stored_turn_refuse_every_ref_of_the_source() -> None:
    record = stored()
    again = mixed(record, state={**STATE, "subject": "another"})[0]
    assert markers(again)["q:spam"] == markers(again)["t:body"] == "unsupported"
    lines = warned(again, "ask")
    assert "source 'ask': needs differ from the stored turn's: 't:state.subject'" in lines
    assert "source 'body': needs differ from the stored turn's: 't:state.subject'" in lines


def test_a_state_change_outside_the_needs_still_answers() -> None:
    record = stored()
    again = mixed(record, state={**STATE, "to": "eve@example.com"})[0]
    assert "q:spam" not in markers(again)


def test_a_whole_state_need_holding_bytes_is_compared_in_record_form() -> None:
    record = stored()
    same = go(binds=bound(from_record(record, LOADED))).record
    assert entries(same)["t:size"] == entries(record)["t:size"]
    other = go(state={**STATE, "blob": b"\x02"}, binds=bound(from_record(record, LOADED)))
    assert markers(other.record)["t:size"] == "unsupported"


def test_an_answer_or_a_list_need_is_compared_by_its_document() -> None:
    later = Source({"t:later": t.string}, needs=["q:spam"], oracle="mail")
    loaded = slice_design(sources={"ask": ASK, "body": BODY, "look": LOOK, "later": later})
    record = stored(loaded, later=returns({"t:later": "x"}))
    again = mixed(record, loaded)[0]
    assert entries(again)["t:later"]["value"] == "x"
    d = extraction.design()
    by = Source({"t:by": t.string}, needs=["d:dates", "q:amount"])
    loaded = load(dataclasses.replace(d, sources={**d.sources, "by": by}))
    replies = {**_extraction_replies(), "by": returns({"t:by": "y"})}
    record = go(loaded, extraction.STATE, binds={"default": Scripted(replies)}).record
    again = go(loaded, extraction.STATE, binds={"default": from_record(record, loaded)}).record
    assert entries(again)["t:by"]["value"] == "y"


# ---------------------------------------------------------------------------------------------
# (4) copied provenance


def test_provenance_is_null_and_the_first_line_names_the_stored_record() -> None:
    record = stored(ask=returns(SPAM, snapshot="m-1", prompt="p", precision=2))
    item = next(c for c in calls(record) if "ask" in c["sources"])
    assert (item["snapshot"], item["prompt"], item["precision"]) == ("m-1", "p", 2)
    again = mixed(record)[0]
    item = next(c for c in calls(again) if "ask" in c["sources"])
    for name in ("snapshot", "elicitation", "prompt", "precision"):
        assert item[name] is None
    assert warned(again, "ask") == [f"answered from record {record.record_hash}"]
    unhashed = again.unhashed["calls"]
    index = str(again.log.index(item))
    assert unhashed[index].get("usage") is None
    assert unhashed[index].get("attempts") is None
    assert unhashed[index].get("truncated") is None


# ---------------------------------------------------------------------------------------------
# Sources the record cannot answer


def test_a_source_with_no_call_in_the_record_is_refused() -> None:
    alone = {"route": RULES["route"]}
    without = slice_design(sources={"ask": ASK, "look": LOOK}, rules=alone)
    record = stored(without, ask=returns({"q:spam": SPAM["q:spam"]}))
    again = mixed(record)[0]
    assert markers(again)["t:body"] == "unsupported"
    assert "q:spam" not in markers(again)
    assert "source 'body': the record holds no call for it" in warned(again, "ask")


def test_a_failed_stored_call_is_refused() -> None:
    record = stored(ask=raising(RuntimeError("down")))
    again = mixed(record)[0]
    assert markers(again)["q:spam"] == markers(again)["t:body"] == "unsupported"
    lines = warned(again, "ask")
    assert "source 'ask': its recorded call failed with raised" in lines


def test_a_ref_the_recorded_call_wrote_no_entry_for_is_refused() -> None:
    record = stored()
    wider = Source({"t:body": t.string, "t:extra": t.string}, needs=["t:state.subject"])
    loaded = slice_design(
        sources={"ask": ASK, "body": dataclasses.replace(wider, oracle="mail"), "look": LOOK}
    )
    again = mixed(record, loaded)[0]
    assert markers(again)["t:extra"] == "unsupported"
    assert "'t:extra': the recorded call wrote no entry for it" in warned(again, "ask")


# ---------------------------------------------------------------------------------------------
# A question that differs from the stored one


def test_a_question_the_stored_call_did_not_ask_is_refused() -> None:
    record = stored()
    asks = Source(
        {"q:spam": ASK.provides["q:spam"], "q:new": Question("noul", Text("New?"))},
        needs=["t:state.subject"],
        oracle="mail",
    )
    loaded = slice_design(sources={"ask": asks, "body": BODY, "look": LOOK})
    again = mixed(record, loaded)[0]
    assert markers(again) == {"q:new": "unsupported"}
    line = "'q:new': its kind, labels, value type and list flag differ from the stored question's"
    assert line in warned(again, "ask")


def test_a_question_of_another_kind_is_refused() -> None:
    record = stored()
    labels = ("yes", "no")
    asks = dataclasses.replace(
        ASK, provides={"q:spam": Question("choice", Text("Is it spam?"), labels=labels)}
    )
    rules = {
        **RULES,
        "route": Rule(
            "d:route", t.label("no", "yes"), lambda r: "no", version="2", reads=["q:spam"]
        ),
    }
    loaded = slice_design(sources={"ask": asks, "body": BODY, "look": LOOK}, rules=rules)
    again = mixed(record, loaded)[0]
    assert markers(again)["q:spam"] == "unsupported"
    assert "'q:spam': its kind, labels differ from the stored question's" in warned(again, "ask")


@pytest.mark.parametrize(
    ("question", "member"),
    [
        (extraction.question("text"), "value type"),
        (extraction.question(is_list=True), "list flag"),
    ],
)
def test_an_extraction_of_another_value_type_or_list_flag_is_refused(
    question: Question, member: str
) -> None:
    loaded = load(extraction.plain())
    record = go(loaded, extraction.STATE, binds={"default": Scripted(_extraction_replies())}).record
    changed = load(extraction.plain(**{"q:amount": question}))
    again = go(changed, extraction.STATE, binds={"default": from_record(record, changed)}).record
    assert markers(again)["q:amount"] == "unsupported"
    assert f"'q:amount': its {member} differ from the stored question's" in warned(again, "read")
    assert "q:dates" not in markers(again)


def test_an_extraction_asked_as_stored_differs_in_no_member() -> None:
    extract = RenderedQuestion(
        AnswerKind.extraction, "What?", (), value_type=ValueType.number, text="t:state.body"
    )
    stored_e = {"kind": "extraction", "labels": None, "value_type": "number", "is_list": False}
    assert _differing(extract, stored_e) == ""


# ---------------------------------------------------------------------------------------------
# The entry, the needs compared, the record form and the warnings bounds


def test_a_loaded_that_is_not_exactly_a_loaded_design_is_an_entry_refusal() -> None:
    with pytest.raises(EntryRefusal, match="not exactly a LoadedDesign"):
        from_record(stored(), cast(Any, object()))


def test_a_document_that_does_not_read_is_an_invalid_record() -> None:
    with pytest.raises(InvalidRecord):
        from_record(b"{}", LOADED)
    with pytest.raises(InvalidRecord):
        from_record(cast(Any, 3), LOADED)


def test_a_need_on_a_missing_entry_a_marker_or_a_path_the_entry_lacks_differs() -> None:
    record = stored(ask=returns({"q:spam": {"maybe": 1.0}, "t:body": "b"}))
    binding = _FromRecord(record, LOADED)
    assert binding._differs("t:state.subject", "hello") is False
    assert binding._differs("t:missing", "x") is True
    assert binding._differs("q:spam", None) is True  # its entry is a marker
    assert binding._differs("t:state.subject.deeper", "x") is True
    assert binding._differs("t:state.nope", "x") is True


def test_the_record_form_of_a_need_value() -> None:
    answer = ExtractionAnswer(ValueType.text, False, "a", "a", 0.0, 1.0, [0, 1])
    got = _record_form({"b": b"\x00", "l": (1, [b"\x01"]), "x": answer})
    assert got == {
        "b": "AA",
        "l": [1, ["AQ"]],
        "x": {
            "kind": "extraction",
            "value_type": "text",
            "is_list": False,
            "value": "a",
            "guess": "a",
            "p_null": 0.0,
            "confidence": 1.0,
            "span": [0, 1],
        },
    }


def test_the_warnings_stay_within_interpret_bounds() -> None:
    many = [f"line {n}" for n in range(MAX_WARNINGS + 10)]
    kept = _bounded(many)
    assert len(kept) == MAX_WARNINGS
    assert kept[-1] == "11 more lines left out"
    exact = [f"line {n}" for n in range(MAX_WARNINGS)]
    assert _bounded(exact) == tuple(exact)
    long = ["x" * (MAX_WARNING_CODE_POINTS // 2)] * 3
    assert _bounded(long) == (long[0], "2 more lines left out")
    near = ["x" * (MAX_WARNING_CODE_POINTS - 100), "y" * 80]
    assert _bounded(near) == tuple(near)
    over = ["x" * (MAX_WARNING_CODE_POINTS - 100), "y" * 150]
    assert _bounded(over) == (over[0], "1 more line left out")


def test_a_source_refused_whole_gives_one_line_beside_the_record_hash() -> None:
    record = stored()
    binding = from_record(record, LOADED)
    refs = tuple(f"t:r{n}" for n in range(MAX_WARNINGS + 5))

    async def call() -> object:
        return await binding((SourceCall("nowhere", None, {}, refs, {}),))

    result = asyncio.run(call())
    values = cast(Any, result).values
    assert all(v == Unavailable(Cause(U)) for v in values.values())
    assert cast(Any, result).truncated == ()
    for name in ("snapshot", "elicitation", "prompt", "precision", "usage", "attempts"):
        assert getattr(result, name) is None
    warnings = cast(Any, result).warnings
    assert warnings == (
        f"answered from record {record.record_hash}",
        "source 'nowhere': the record holds no call for it",
    )


def test_a_changed_wording_answers_and_the_caller_finds_it_by_identity_after_the_turn() -> None:
    record = stored()
    reworded = dataclasses.replace(
        ASK, provides={"q:spam": Question("noul", Text("Is it junk mail?"))}
    )
    loaded = slice_design(sources={"ask": reworded, "body": BODY, "look": LOOK})
    again = mixed(record, loaded)[0]
    assert "q:spam" not in markers(again)
    before, after = (
        next(c for c in calls(r) if "ask" in c["sources"])["questions"]["q:spam"]["identity"]
        for r in (record, again)
    )
    assert before != after
