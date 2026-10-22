# pyright: reportPrivateUsage=false
"""Do-and-diff's fifth form, another record's answers: two real records of one turn, decided
under two scripted bindings, each rerun with the other's calls answering; the findings, the
cut, the refusals in their order, and both records left as they were."""

import asyncio
import copy
import json
from typing import Any

import pytest

from jurat import (
    Cause,
    EntryRefusal,
    Failed,
    FindingKind,
    InvalidRecord,
    LoadedDesign,
    Record,
    Requirement,
    RunSettings,
    read_record,
)
from jurat.evaluation import (
    EvaluationRefusal,
    RefusalReason,
    _diff,
    use_answers,
    use_answers_sync,
)
from scripted import Scripted, returns, unsupported
from test_decide import (
    LOADED,
    REPLIES,
    SPAM,
    STATE,
    Seen,
    bound,
    caught,
    deciding,
    go,
    record_of,
    slice_design,
)
from test_replay import CHAINED, Slowly, edited, entry, rule_of

K = FindingKind
R = RefusalReason
HAM = {**SPAM, "q:spam": {"yes": 0.1, "no": 0.9}}
REFUSING = slice_design(requirements=[Requirement("d:route", "no")])


def live(
    loaded: LoadedDesign = LOADED, settings: Any = None, state: Any = STATE, **replies: object
) -> Record:
    binds = bound(Scripted({**REPLIES, **replies}))
    return go(loaded, state, binds=binds, settings=settings).record


def found(record: object, other: object, loaded: LoadedDesign = LOADED) -> list[Any]:
    result = use_answers_sync(record, loaded, other)
    assert result["other"] == (other if type(other) is Record else read_record(other)).record_hash
    return result["findings"]


def diff(record: Record, other: Record, loaded: LoadedDesign = LOADED) -> list[Any]:
    """The findings a rerun reproducing ``other``'s turn gives against ``record``."""
    mine = {i["ref"]: i for i in record.log if i["item"] == "entry"}
    theirs = {i["ref"]: i for i in other.log if i["item"] == "entry"}
    rules = {"d:route": rule_of(loaded, "route"), "d:echo": rule_of(loaded, "echo")}
    out: list[Any] = [
        {
            "kind": K.entry,
            "ref": ref,
            "recorded": mine.get(ref),
            "replayed": theirs.get(ref),
            "rule": rules.get(ref),
        }
        for ref in sorted({*mine, *theirs})
        if mine.get(ref) != theirs.get(ref)
    ]
    for kind in (K.plan, K.verdict):
        a, b = (next(i for i in r.log if i["item"] == kind.value) for r in (record, other))
        if a != b:
            out.append({"kind": kind, "recorded": a, "replayed": b})
    return out


@pytest.fixture
def no_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any replay fails the test: the refusal came before it."""

    async def replayed(*args: object, **kwargs: object) -> object:
        raise AssertionError("replayed")

    monkeypatch.setattr(_diff, "_replay_with", replayed)


def refusal(record: object, other: object, loaded: LoadedDesign = LOADED) -> EvaluationRefusal:
    with pytest.raises(EvaluationRefusal) as raised:
        use_answers_sync(record, loaded, other)
    return raised.value


# ---------------------------------------------------------------------------------------------
# The findings


def test_a_record_as_its_own_other_gives_no_finding_in_every_form() -> None:
    record = go().record
    for given in (record, record.document, json.loads(record.document)):
        assert use_answers_sync(given, LOADED, given) == {
            "other": record.record_hash,
            "findings": [],
        }


def test_two_records_of_the_same_answers_give_no_finding() -> None:
    assert found(live(), live()) == []


def test_one_differing_answer_gives_exactly_the_findings_downstream_of_it() -> None:
    record, other = live(REFUSING), live(REFUSING, ask=returns(HAM))
    result = found(record, other, REFUSING)
    assert result == diff(record, other, REFUSING)
    assert [(f["kind"], f.get("ref")) for f in result] == [
        (K.entry, "d:route"),
        (K.entry, "q:spam"),
        (K.plan, None),
        (K.verdict, None),
    ]
    assert result[1]["replayed"] == entry(other.log, "q:spam")
    assert result[0]["replayed"]["value"] == "no"
    assert result[3]["recorded"]["kind"] == "refused"
    assert result[3]["replayed"]["kind"] == "passed"


def test_a_q_and_a_t_answer_are_both_taken_from_other() -> None:
    other = live(ask=returns(HAM), look=returns({"t:size": 5}))
    result = found(live(), other)
    assert result == diff(live(), other)
    assert [f.get("ref") for f in result] == ["d:route", "q:spam", "t:size", None]


def test_the_findings_are_plain_data() -> None:
    result = use_answers_sync(live(REFUSING), REFUSING, live(REFUSING, ask=returns(HAM)))
    assert json.loads(json.dumps(result)) == result
    assert copy.deepcopy(result) == result


def test_a_failed_call_in_other_comes_back_as_its_outcome() -> None:
    other = live(ask=Failed(Cause("raised", ["q:spam"])))
    result = found(live(), other)
    assert result == diff(live(), other)
    spam = next(f for f in result if f.get("ref") == "q:spam")
    cause = {"code": "raised", "refs": ["q:spam", "t:body"]}
    assert spam["replayed"]["marker"] == {"kind": "Unavailable", "cause": cause}
    assert entry(other.log, "t:body")["marker"]["cause"] == cause


def test_an_unsupported_marker_in_other_comes_back_as_recorded() -> None:
    other = live(ask=unsupported())
    result = found(live(), other)
    assert result == diff(live(), other)
    spam = next(f for f in result if f.get("ref") == "q:spam")
    assert spam["replayed"]["marker"] == {
        "kind": "Unavailable",
        "cause": {"code": "unsupported", "refs": ["q:spam"]},
    }


def test_others_unhashed_is_never_read() -> None:
    other = live(ask=returns(HAM, usage={"tokens": 5}))
    blanked = edited(other, lambda d: d["unhashed"].update(calls={}))
    assert found(live(), blanked) == found(live(), other)


# ---------------------------------------------------------------------------------------------
# The cut is record's


CUT = RunSettings(deadline=0.2)


def cut_of(record: Record) -> list[tuple[int, bool]]:
    return [(i["round"], i["deadline"]) for i in record.log if i["item"] == "round"]


def test_records_cut_holds_when_other_ran_more_rounds() -> None:
    record = live(CHAINED, CUT, look=Slowly(10, returns({"t:size": 4})))
    other = live(CHAINED, CUT, late=returns({"t:late": "x"}))
    assert cut_of(record) == [(1, True)]
    assert entry(other.log, "t:late")["value"] == "x"
    assert all(not cut for _, cut in cut_of(other))
    result = found(record, other, CHAINED)
    assert K.not_in_record not in [f["kind"] for f in result]
    # Only look's answer comes back; late, past the cut, is cut again.
    assert [f.get("ref") for f in result] == ["t:size"]
    assert result[0]["replayed"] == entry(other.log, "t:size")


def test_a_source_other_never_called_is_not_in_record() -> None:
    record = live(CHAINED, CUT, late=returns({"t:late": "x"}))
    other = live(CHAINED, CUT, look=Slowly(10, returns({"t:size": 4})))
    assert cut_of(other) == [(1, True)]
    result = found(record, other, CHAINED)
    assert result[0] == {"kind": K.not_in_record, "sources": ["late"]}
    assert K.ref_not_in_record not in [f["kind"] for f in result]
    late = next(f for f in result if f.get("ref") == "t:late")
    assert late["replayed"]["marker"]["cause"]["code"] == "not-returned"


# ---------------------------------------------------------------------------------------------
# The refusals, before any replay


def unfinished() -> Record:
    return record_of(caught(deciding(observer=Seen(5))))


def broken() -> Record:
    return go(state={"subject": "no other field"}).record


def test_an_unfinished_record_or_other_is_refused_record_first() -> None:
    record, other = unfinished(), unfinished()
    for given, got, name in ((record, other, record), (go().record, other, other)):
        exc = refusal(given, got)
        assert exc.reason is R.unfinished
        assert exc.turn == name.record_hash


def test_nothing_to_replay_is_refused_after_unfinished() -> None:
    exc = refusal(broken(), unfinished())
    assert (exc.reason, exc.turn) == (R.unfinished, unfinished().record_hash)
    for record, other, about in ((broken(), broken(), "record"), (go().record, broken(), "other")):
        exc = refusal(record, other)
        assert exc.reason is R.nothing_to_replay
        assert str(exc).startswith(about)
        assert exc.turn == broken().record_hash


@pytest.mark.usefixtures("no_replay")
def test_a_record_of_another_design_is_wrong_design_naming_its_turn_and_hash() -> None:
    changed = slice_design(tools=["archive", "send"])
    theirs = live(changed)
    exc = refusal(go().record, theirs)
    assert exc.reason is R.wrong_design
    assert exc.turn == theirs.record_hash
    assert str(exc) == (
        f"other's design hash is {changed.design_hash}, not the loaded design's "
        f"{LOADED.design_hash}"
    )
    exc = refusal(theirs, theirs)
    assert str(exc).startswith("record's")


@pytest.mark.usefixtures("no_replay")
def test_wrong_design_is_refused_after_nothing_to_replay() -> None:
    changed = live(slice_design(tools=["archive", "send"]))
    assert refusal(changed, broken()).reason is R.nothing_to_replay


@pytest.mark.parametrize(
    ("other", "what"),
    [
        (lambda: live(state={**STATE, "subject": "bye"}), "state entry"),
        (lambda: live(settings=RunSettings(timeout=7)), "settings item"),
    ],
    ids=["another state", "other settings"],
)
@pytest.mark.usefixtures("no_replay")
def test_another_state_or_settings_is_different_turn(other: Any, what: str) -> None:
    theirs = other()
    exc = refusal(go().record, theirs)
    assert exc.reason is R.different_turn
    assert exc.turn == theirs.record_hash
    assert what in str(exc)


@pytest.mark.usefixtures("no_replay")
def test_different_turn_is_refused_after_wrong_design() -> None:
    changed = slice_design(tools=["archive", "send"])
    exc = refusal(go().record, live(changed, settings=RunSettings(timeout=7)))
    assert exc.reason is R.wrong_design


@pytest.mark.parametrize("which", ["record", "other"])
def test_a_document_that_does_not_read_is_invalid_record_carrying_the_error(which: str) -> None:
    good = go().record.document
    doc = json.loads(good)
    doc["log"][2]["value"]["subject"] = "edited"  # its hash not recomputed
    given = (doc, good) if which == "record" else (good, doc)
    exc = refusal(*given)
    assert exc.reason is R.invalid_record
    assert type(exc.error) is InvalidRecord
    assert exc.__cause__ is exc.error
    assert exc.turn is None
    assert str(exc).startswith(f"{which} is not a valid record")


def test_invalid_record_is_refused_before_unfinished() -> None:
    assert refusal(unfinished(), b"not json").reason is R.invalid_record
    assert refusal(unfinished(), go().record).reason is R.unfinished


def test_a_refusal_names_only_what_its_reason_names() -> None:
    exc = refusal(unfinished(), go().record)
    named = ("key", "ref", "source", "point", "path", "group", "split", "error")
    assert all(getattr(exc, name) is None for name in named)
    assert type(exc.reason) is RefusalReason
    assert {r.value for r in RefusalReason} == {
        "unfinished",
        "nothing-to-replay",
        "wrong-design",
        "different-turn",
        "invalid-record",
    }


def test_a_loaded_not_exactly_a_loaded_design_is_refused_before_the_records_are_read() -> None:
    with pytest.raises(EntryRefusal, match="exactly a LoadedDesign"):
        use_answers_sync(b"never read", object(), b"never read")  # type: ignore[arg-type]


def test_neither_records_document_changes() -> None:
    record, other = live(REFUSING), live(REFUSING, ask=returns(HAM))
    before = (record.document, other.document)
    parsed = json.loads(other.document)
    kept = copy.deepcopy(parsed)
    use_answers_sync(record, REFUSING, other)
    use_answers_sync(record.document, REFUSING, parsed)
    assert (record.document, other.document) == before
    assert parsed == kept


# ---------------------------------------------------------------------------------------------
# The sync twin


def test_use_answers_is_awaitable_and_equals_the_sync_twin() -> None:
    record, other = live(), live(ask=returns(HAM))
    assert asyncio.run(use_answers(record, LOADED, other)) == use_answers_sync(
        record, LOADED, other
    )


def test_use_answers_sync_inside_a_running_loop_raises_before_doing_anything() -> None:
    async def main() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            use_answers_sync(object(), object(), object())  # type: ignore[arg-type]

    asyncio.run(main())
