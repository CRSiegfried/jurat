# pyright: reportPrivateUsage=false
"""Replay: round trips asserting no finding,
an edited copy asserting exactly its entry's finding, each finding kind in its
fixed order, the entry refusals, the cut and the budget, ``replay_sync``, and the observer's
exception and a cancellation carrying the replay record."""

import asyncio
import copy
import json
import time
from collections.abc import Callable
from typing import Any

import pytest

from designs import roles
from jurat import (
    Cause,
    EntryRefusal,
    Failed,
    FindingKind,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Parameter,
    Policy,
    Question,
    Record,
    Requirement,
    Rule,
    RunSettings,
    Source,
    SourceRequirement,
    Text,
    Unavailable,
    When,
    load,
    read_record,
    replay,
    replay_sync,
)
from jurat import types as t
from jurat._canonical import decode
from jurat.helpers import HELPERS, pick_with_margin
from scripted import Scripted, raising, returns
from test_decide import (
    ASK,
    GOLDEN_BROKEN,
    GOLDEN_RECORD,
    LOADED,
    LOOK,
    REPLIES,
    RULES,
    SPAM,
    Counting,
    Seen,
    archive_policy,
    caught,
    deciding,
    echo,
    escalate_policy,
    go,
    golden_escalate,
    golden_refused,
    ok_policy,
    record_of,
    route,
    slice_design,
)

K = FindingKind


def run(record: Record, loaded: LoadedDesign = LOADED) -> tuple[Record | None, list[Any]]:
    return replay_sync(record, loaded)


def findings(record: Record, loaded: LoadedDesign = LOADED) -> list[Any]:
    return run(record, loaded)[1]


def live(loaded: LoadedDesign = LOADED, settings: Any = None, **replies: object) -> Record:
    return go(loaded, binds={"default": Scripted({**REPLIES, **replies})}, settings=settings).record


def edited(record: Record, change: Callable[[dict[str, Any]], object]) -> Record:
    doc = json.loads(record.document)
    change(doc)
    return read_record(doc)


def entry(log: list[dict[str, Any]], ref: str) -> dict[str, Any]:
    return next(i for i in log if i["item"] == "entry" and i["ref"] == ref)


def entries(record: Record) -> list[dict[str, Any]]:
    return [i for i in record.log if i["item"] in ("entry", "plan", "verdict")]


def rule_of(loaded: LoadedDesign, rid: str) -> dict[str, Any]:
    declared = decode(loaded.document)["rules"][rid]  # type: ignore[index]
    return {k: declared[k] for k in ("module", "qualname", "version", "source")}


# ---------------------------------------------------------------------------------------------
# Round trips with no finding


def test_the_golden_record_replays_with_no_finding() -> None:
    record = go().record
    assert record.record_hash == GOLDEN_RECORD
    rerun, found = run(record)
    assert found == []
    assert type(rerun) is Record
    assert entries(rerun) == entries(record)
    assert rerun.log[:2] == record.log[:2]


def test_a_record_read_back_replays_with_no_finding() -> None:
    assert findings(read_record(go().record.document)) == []


def test_the_replay_record_is_a_record_of_the_given_design_with_the_records_settings() -> None:
    changed = slice_design(tools=["archive", "send"])
    record = live(settings=RunSettings(round_bound=3, timeout=7))
    rerun, _ = run(record, changed)
    assert rerun is not None
    assert rerun.log[0]["hash"] == changed.design_hash
    assert rerun.log[1]["settings"] == record.log[1]["settings"]
    assert rerun.log[1]["providers"] == dict(changed._providers)


BLOBS = Source(
    {
        "t:blob": t.bytes_,
        "t:weight": t.number(),
        "t:size": t.integer(),
        "t:maybe": t.nullable(t.number()),
    }
)
LATE = Source({"t:late": t.string}, needs=["d:route"])
CHAINED = slice_design(sources={"ask": ASK, "look": LOOK, "late": LATE})
TYPED = slice_design(sources={"ask": ASK, "look": BLOBS})
SURE = {**SPAM, "q:spam": {"yes": 1, "no": 0}}

SCENARIOS: dict[str, Callable[[], tuple[Record, LoadedDesign]]] = {
    "bytes, integral number, null": lambda: (
        live(
            TYPED,
            look=returns({"t:blob": b"\x00\xff", "t:weight": 3, "t:size": 4, "t:maybe": None}),
        ),
        TYPED,
    ),
    "probability exactly 1": lambda: (live(ask=returns(SURE)), LOADED),
    "refused": lambda: (
        live(ask=returns({**SPAM, "q:spam": Unavailable(Cause("refused", ["q:spam"]))})),
        LOADED,
    ),
    "invalid answer": lambda: (live(ask=returns({**SPAM, "q:spam": {"yes": 2}})), LOADED),
    "invalid return": lambda: (live(look=returns({"t:size": "four"})), LOADED),
    "binding marker invalid-return": lambda: (
        live(look=returns({"t:size": Unavailable(Cause("invalid-return", ["x"]))})),
        LOADED,
    ),
    "not returned": lambda: (live(ask=returns({"t:body": "b"})), LOADED),
    "raised": lambda: (live(ask=raising(ValueError("down"))), LOADED),
    "failed as returned": lambda: (live(ask=Failed(Cause("raised", ["q:spam"]))), LOADED),
    "unexpected ref": lambda: (live(look=returns({"t:size": 4, "t:other": 1})), LOADED),
    "timed out": lambda: (live(ask=Failed(Cause("timed-out", []))), LOADED),
    "a second round": lambda: (live(CHAINED, late=returns({"t:late": "x"})), CHAINED),
    "need unavailable": lambda: (
        live(CHAINED, ask=raising(ValueError("down")), late=returns({"t:late": "x"})),
        CHAINED,
    ),
    "round bound": lambda: (live(CHAINED, RunSettings(round_bound=1)), CHAINED),
    "refused plan": lambda: (
        live(slice_design(requirements=[Requirement("d:route", "no")])),
        slice_design(requirements=[Requirement("d:route", "no")]),
    ),
    "a deadline with no cut": lambda: (
        edited(go().record, lambda d: d["log"][1]["settings"].update(deadline=5)),
        LOADED,
    ),
}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_round_trip_has_no_finding(scenario: str) -> None:
    record, loaded = SCENARIOS[scenario]()
    rerun, found = run(record, loaded)
    assert found == []
    assert rerun is not None
    assert entries(rerun) == entries(record)


def test_a_bytes_and_a_number_reach_the_rules_typed_in_replay_as_live() -> None:
    seen: list[object] = []

    def keep(r: Any) -> Any:
        seen.append((r.read("t:blob"), r.read("t:weight"), r.read("t:state.weight")))
        return "kept"

    rules = {
        **RULES,
        "keep": Rule(
            "d:keep", t.string, keep, version="1", reads=["t:blob", "t:weight", "t:state.weight"]
        ),
    }
    loaded = slice_design(sources={"ask": ASK, "look": BLOBS}, rules=rules)
    record = live(
        loaded, look=returns({"t:blob": b"\x01", "t:weight": 2, "t:size": 1, "t:maybe": 1})
    )
    assert findings(record, loaded) == []
    assert seen == [(b"\x01", 2.0, 3.0)] * 2
    assert all(type(w) is float for _, w, _ in seen)  # type: ignore[misc]


def test_a_copy_setting_a_value_where_a_value_fault_was_clears_it() -> None:
    record = live(look=returns({"t:size": "four"}))

    def change(doc: dict[str, Any]) -> None:
        item = entry(doc["log"], "t:size")
        del item["marker"]
        item["value"] = 4

    copy_ = edited(record, change)
    rerun, found = run(copy_)
    assert found == []
    assert rerun is not None
    assert entry(rerun.log, "t:size")["value"] == 4


def test_the_fixture_waits_for_nothing() -> None:
    record = live(settings=RunSettings(timeout=0.05))
    assert findings(record) == []


def hashed(record: Record | None) -> list[dict[str, Any]]:
    assert record is not None
    return record.log


def test_replay_reads_nothing_from_unhashed() -> None:
    record = live(ask=returns(SPAM, usage={"tokens": 5}, attempts=2, snapshot="m-1"))

    def change(doc: dict[str, Any]) -> None:
        calls = doc["unhashed"]["calls"]
        calls["3"].update(
            usage={"tokens": 99},
            attempts=7,
            unexpected=["t:x"],
            invalid={"prompt": "'p'"},
        )
        calls["4"].update(usage=None, attempts=None)
        del doc["unhashed"]["engine"]

    copy_ = edited(record, change)
    assert copy_.unhashed != record.unhashed
    assert hashed(copy_) == hashed(record)
    rerun, found = run(record)
    rerun_copy, found_copy = run(copy_)
    assert found == found_copy == []  # call items are not compared
    assert hashed(rerun_copy) == hashed(rerun)
    assert all(item.get("snapshot") is None for item in hashed(rerun))
    assert rerun is not None
    assert all(
        c["usage"] is None and c["attempts"] is None for c in rerun.unhashed["calls"].values()
    )


def test_an_answer_off_one_replays_to_its_entry_and_the_same_sums() -> None:
    record = live(ask=returns({**SPAM, "q:spam": {"yes": 0.9, "no": 0.9}}))
    (key,) = [k for k, c in record.unhashed["calls"].items() if "sums" in c]
    assert record.unhashed["calls"][key]["sums"] == {"q:spam": 1.8}
    back = read_record(record.document)
    rerun, found = run(back)
    assert found == []
    assert rerun is not None
    assert hashed(rerun) == hashed(record)
    assert rerun.unhashed["calls"][key]["sums"] == {"q:spam": 1.8}


def test_replay_is_awaitable_and_equals_replay_sync() -> None:
    record = go().record
    rerun, found = asyncio.run(replay(record, LOADED))
    assert found == []
    assert rerun is not None
    assert entries(rerun) == entries(record)


# ---------------------------------------------------------------------------------------------
# The edited copy


def test_a_copy_with_one_d_entry_changed_gives_exactly_that_entrys_finding() -> None:
    record = go().record
    copy_ = edited(record, lambda d: entry(d["log"], "d:echo").update(value="changed"))
    found = findings(copy_)
    assert found == [
        {
            "kind": K.entry,
            "ref": "d:echo",
            "recorded": entry(copy_.log, "d:echo"),
            "replayed": entry(record.log, "d:echo"),
            "rule": rule_of(LOADED, "echo"),
        }
    ]
    assert found[0]["recorded"]["value"] == "changed"
    assert found[0]["replayed"]["value"] == "hello"


def test_a_finding_is_plain_data_holding_no_record() -> None:
    record = go().record
    copy_ = edited(record, lambda d: entry(d["log"], "d:echo").update(value="changed"))
    found = findings(copy_, slice_design(tools=["archive", "send"]))
    assert json.loads(json.dumps(found)) == found
    assert copy.deepcopy(found) == found
    assert all(type(f) is dict and type(f["kind"]) is FindingKind for f in found)


def test_an_entry_finding_names_no_rule_for_a_d_ref_the_given_design_no_longer_provides() -> None:
    record = go().record
    changed = slice_design(rules={"route": RULES["route"]})
    found = findings(record, changed)
    assert found == [
        {
            "kind": K.design_mismatch,
            "recorded": LOADED.design_hash,
            "replayed": changed.design_hash,
        },
        {
            "kind": K.entry,
            "ref": "d:echo",
            "recorded": entry(record.log, "d:echo"),
            "replayed": None,
            "rule": None,
        },
    ]


# ---------------------------------------------------------------------------------------------
# The early returns


def test_a_record_with_no_verdict_is_unfinished_and_runs_nothing() -> None:
    record = record_of(caught(deciding(observer=Seen(5))))
    seen = Seen()
    assert asyncio.run(replay(record, LOADED, observer=seen)) == (None, [{"kind": K.unfinished}])
    assert seen.items == []


def test_a_cancelled_record_is_unfinished_whatever_the_design() -> None:
    record = record_of(caught(deciding(observer=Seen(0))))
    assert run(record, slice_design(tools=["send"])) == (None, [{"kind": K.unfinished}])


def test_a_broken_entry_has_nothing_to_replay_and_no_design_mismatch() -> None:
    broken = go(state={"subject": "no other field"}).record
    assert broken.record_hash == GOLDEN_BROKEN
    assert run(broken, slice_design(tools=["send"])) == (None, [{"kind": K.nothing_to_replay}])
    raised = go(codec=Counting(raises=ValueError("x"))).record
    assert run(raised) == (None, [{"kind": K.nothing_to_replay}])


# ---------------------------------------------------------------------------------------------
# Findings under another design, in the fixed order


def test_a_changed_design_is_design_mismatch_and_replay_proceeds() -> None:
    changed = slice_design(tools=["archive", "send"])
    rerun, found = run(go().record, changed)
    assert found == [
        {"kind": K.design_mismatch, "recorded": LOADED.design_hash, "replayed": changed.design_hash}
    ]
    assert rerun is not None


def test_an_edited_helper_version_is_helpers_mismatch_after_design_mismatch() -> None:
    copy_ = edited(go().record, lambda d: d["log"][0].update(helpers=HELPERS + 1))
    assert findings(copy_) == [
        {"kind": K.helpers_mismatch, "recorded": HELPERS + 1, "replayed": HELPERS}
    ]
    changed = slice_design(tools=["send", "archive", "x"])
    assert [f["kind"] for f in findings(copy_, changed)] == [K.design_mismatch, K.helpers_mismatch]


def other(r: Any) -> Any:
    return "other"


def test_a_changed_function_rule_is_an_entry_finding_naming_the_given_designs_rule() -> None:
    rules = {**RULES, "echo": Rule("d:echo", t.string, other, version="2", reads=["t:body"])}
    changed = slice_design(rules=rules)
    found = findings(go().record, changed)
    assert [f["kind"] for f in found] == [K.design_mismatch, K.entry]
    assert found[1]["ref"] == "d:echo"
    assert found[1]["rule"] == rule_of(changed, "echo")
    assert found[1]["rule"]["qualname"] == "other"
    assert found[1]["replayed"]["value"] == "other"


def test_an_added_source_is_not_in_record_and_its_refs_entry_findings() -> None:
    # Both ids sort after the kept sources, so the kept calls keep their indices.
    extra = Source({"t:extra": t.string})
    zed = Source({"t:zed": t.string})
    changed = slice_design(sources={"ask": ASK, "look": LOOK, "zed": zed, "xtra": extra})
    rerun, found = run(go().record, changed)
    assert rerun is not None
    assert found == [
        {
            "kind": K.design_mismatch,
            "recorded": LOADED.design_hash,
            "replayed": changed.design_hash,
        },
        {"kind": K.not_in_record, "sources": ["xtra"]},
        {"kind": K.not_in_record, "sources": ["zed"]},
        {
            "kind": K.entry,
            "ref": "t:extra",
            "recorded": None,
            "replayed": entry(rerun.log, "t:extra"),
            "rule": None,
        },
        {
            "kind": K.entry,
            "ref": "t:zed",
            "recorded": None,
            "replayed": entry(rerun.log, "t:zed"),
            "rule": None,
        },
    ]
    assert entry(rerun.log, "t:extra")["marker"]["cause"] == {
        "code": "not-returned",
        "refs": ["t:extra"],
    }


def test_an_added_ref_is_ref_not_in_record_once_for_the_run() -> None:
    ask = Source({**ASK.provides, "t:more": t.string, "t:also": t.string}, needs=ASK.needs, cost=1)
    look = Source({**LOOK.provides, "t:bigger": t.integer()}, needs=LOOK.needs)
    changed = slice_design(sources={"ask": ask, "look": look})
    found = findings(go().record, changed)
    assert found[:2] == [
        {
            "kind": K.design_mismatch,
            "recorded": LOADED.design_hash,
            "replayed": changed.design_hash,
        },
        {"kind": K.ref_not_in_record, "refs": ["t:also", "t:bigger", "t:more"]},
    ]
    assert [f.get("ref") for f in found[2:]] == ["t:also", "t:bigger", "t:more"]


def test_a_ref_moved_to_another_source_finds_no_entry_under_that_sources_call() -> None:
    ask = Source({"q:spam": ASK.provides["q:spam"]}, needs=ASK.needs, cost=1)
    look = Source({**LOOK.provides, "t:body": t.string}, needs=LOOK.needs)
    changed = slice_design(sources={"ask": ask, "look": look})
    found = findings(go().record, changed)
    assert found[1] == {"kind": K.ref_not_in_record, "refs": ["t:body"]}
    assert [f.get("ref") for f in found[2:]] == ["t:body"]


def test_a_ref_asked_under_a_failed_call_with_no_entry_is_ref_not_in_record() -> None:
    record = live(look=raising(ValueError("down")))
    look = Source({**LOOK.provides, "t:bigger": t.integer()}, needs=LOOK.needs)
    changed = slice_design(sources={"ask": ASK, "look": look})
    found = findings(record, changed)
    assert found[1] == {"kind": K.ref_not_in_record, "refs": ["t:bigger"]}


def test_a_changed_wording_is_question_changed_and_still_answered() -> None:
    q = Question("noul", Text("Is it spam, really?"))
    ask = Source({"q:spam": q, "t:body": t.string}, needs=ASK.needs, cost=1)
    changed = slice_design(sources={"ask": ask, "look": LOOK})
    rerun, found = run(go().record, changed)
    assert found == [
        {
            "kind": K.design_mismatch,
            "recorded": LOADED.design_hash,
            "replayed": changed.design_hash,
        },
        {"kind": K.question_changed, "refs": ["q:spam"]},
    ]
    assert rerun is not None
    assert entry(rerun.log, "q:spam")["value"]["probabilities"] == {"yes": 0.9, "no": 0.1}


def test_changed_labels_are_question_changed_and_the_answer_invalid() -> None:
    q = Question("choice", Text("Is it spam?"), labels=("ham", "spam"))
    ask = Source({"q:spam": q, "t:body": t.string}, needs=ASK.needs, cost=1)
    rules = {
        **RULES,
        "route": Rule("d:route", t.label("no", "yes"), other_route, version="1", reads=["q:spam"]),
    }
    changed = slice_design(sources={"ask": ask, "look": LOOK}, rules=rules)
    rerun, found = run(go().record, changed)
    assert [f["kind"] for f in found[:2]] == [K.design_mismatch, K.question_changed]
    assert rerun is not None
    assert entry(rerun.log, "q:spam")["marker"]["cause"]["code"] == "invalid-answer"


def other_route(r: Any) -> Any:
    return "no"


def test_every_kind_together_is_in_the_fixed_order() -> None:
    q = Question("noul", Text("Changed?"))
    ask = Source({"q:spam": q, "t:body": t.string, "t:more": t.string}, needs=ASK.needs, cost=1)
    changed = slice_design(sources={"ask": ask, "look": LOOK, "new": Source({"t:new": t.string})})
    copy_ = edited(
        go().record,
        lambda d: (
            d["log"][0].update(helpers=HELPERS + 1),
            entry(d["log"], "d:echo").update(value="x"),
        ),
    )
    found = findings(copy_, changed)
    assert [(f["kind"], f.get("ref")) for f in found] == [
        (K.design_mismatch, None),
        (K.helpers_mismatch, None),
        (K.not_in_record, None),
        (K.ref_not_in_record, None),
        (K.question_changed, None),
        (K.entry, "d:echo"),
        (K.entry, "t:more"),
        (K.entry, "t:new"),
    ]


def test_a_changed_plan_and_verdict_are_their_findings_last() -> None:
    record = go().record
    changed = slice_design(requirements=[Requirement("d:route", "no")])
    rerun, found = run(record, changed)
    assert rerun is not None
    kinds = [f["kind"] for f in found]
    assert kinds == [K.design_mismatch, K.verdict]
    assert found[-1]["recorded"] == record.log[-1]
    assert found[-1]["replayed"] == rerun.log[-1]
    copy_ = edited(record, lambda d: d["log"][-2].update(reason="edited"))
    assert [f["kind"] for f in findings(copy_)] == [K.plan]


def test_a_state_that_no_longer_conforms_replays_as_a_broken_entry() -> None:
    changed = slice_design(state=t.record(subject=t.integer()))
    rerun, found = run(go().record, changed)
    assert rerun is not None
    assert rerun.cause == {"code": "invalid-return", "refs": ["t:state"]}
    assert found[1] == {
        "kind": K.entry,
        "ref": "d:echo",
        "recorded": entry(go().record.log, "d:echo"),
        "replayed": None,
        "rule": rule_of(changed, "echo"),
    }
    assert found[-1]["kind"] is K.plan  # both verdicts passed: no verdict finding


def marked_state(doc: dict[str, Any]) -> None:
    state = entry(doc["log"], "t:state")
    del state["value"]
    state["marker"] = {"kind": "Unavailable", "cause": {"code": "raised", "refs": ["t:state"]}}


def look_also_lists_ask(doc: dict[str, Any]) -> None:
    calls = [i for i in doc["log"] if i["item"] == "call"]
    assert [c["sources"] for c in calls] == [["ask"], ["look"]]
    calls[1]["sources"] = ["ask", "look"]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (marked_state, "a t:state entry holding a marker"),
        (look_also_lists_ask, "its sources list 'ask', which call item 3 lists"),
    ],
    ids=["a marker as the state", "a source in two call items"],
)
def test_a_copy_no_turn_could_write_is_refused_at_the_read_so_never_replayed(
    change: Callable[[dict[str, Any]], object], match: str
) -> None:
    with pytest.raises(InvalidRecord, match=match):
        edited(go().record, change)


def test_a_bytes_state_field_that_is_not_base64url_is_a_broken_entry() -> None:
    copy_ = edited(go().record, lambda d: entry(d["log"], "t:state")["value"].update(blob="A"))
    rerun, _ = run(copy_)
    assert rerun is not None
    assert rerun.cause == {"code": "invalid-return", "refs": ["t:state"]}


# ---------------------------------------------------------------------------------------------
# Refusals


class LoadedSub(LoadedDesign):
    __slots__ = ()


class RecordSub(Record):
    __slots__ = ()

    @property
    def log(self) -> list[dict[str, Any]]:  # type: ignore[override]
        raise AssertionError("the record was read")


def test_a_loaded_or_record_not_exactly_its_type_is_refused_before_the_record_is_read() -> None:
    sub_loaded = object.__new__(LoadedSub)
    sub_record = object.__new__(RecordSub)
    with pytest.raises(EntryRefusal, match="exactly a LoadedDesign"):
        replay_sync(sub_record, sub_loaded)
    with pytest.raises(EntryRefusal, match="exactly a Record"):
        replay_sync(sub_record, LOADED)
    with pytest.raises(EntryRefusal, match="exactly a Record"):
        replay_sync(go().record.document, LOADED)  # type: ignore[arg-type]


def test_an_uncallable_observer_is_refused_before_the_record_is_read() -> None:
    unfinished = record_of(caught(deciding(observer=Seen(5))))
    with pytest.raises(EntryRefusal, match="observer"):
        replay_sync(unfinished, LOADED, observer=42)  # type: ignore[arg-type]


@pytest.mark.parametrize("settings", [{"certificate_cap": 1}, {"timeout": 0}, {"deadline": -1}])
def test_a_recorded_setting_outside_its_bounds_is_refused_with_no_findings(
    settings: dict[str, Any],
) -> None:
    copy_ = edited(go().record, lambda d: d["log"][1]["settings"].update(settings))
    seen = Seen()
    with pytest.raises(EntryRefusal):
        replay_sync(copy_, LOADED, observer=seen)
    assert seen.items == []


def test_a_recorded_round_bound_of_0_is_refused_at_the_read_or_with_no_findings() -> None:
    # A log holding round 1 under a bound of 0 is no turn's, refused at the read;
    # one with no round reads, and replay refuses its setting.
    with pytest.raises(InvalidRecord, match="above the round bound"):
        edited(go().record, lambda d: d["log"][1]["settings"].update(round_bound=0))
    roundless = slice_design(sources={}, rules={}, policy=Policy(ok_policy, version="1"))
    record = go(roundless, binds={}).record
    copy_ = edited(record, lambda d: d["log"][1]["settings"].update(round_bound=0))
    seen = Seen()
    with pytest.raises(EntryRefusal):
        replay_sync(copy_, roundless, observer=seen)
    assert seen.items == []


def test_replay_sync_inside_a_running_loop_raises_before_doing_anything() -> None:
    seen = Seen()

    async def main() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            replay_sync(object(), object(), observer=seen)  # type: ignore[arg-type]

    asyncio.run(main())
    assert seen.items == []


# ---------------------------------------------------------------------------------------------
# The observer and a cancellation


def test_the_observer_sees_the_replay_records_items() -> None:
    seen = Seen()
    rerun, found = replay_sync(go().record, LOADED, observer=seen)
    assert found == []
    assert rerun is not None
    assert [item for _, item in seen.items] == rerun.log


@pytest.mark.parametrize("at", [0, 2, 5, 12])
def test_the_observers_exception_propagates_with_the_replay_record_so_far(at: int) -> None:
    record = go().record
    seen = Seen(at)
    exc = caught(replay(record, LOADED, observer=seen))
    assert exc is seen.raises
    rerun = record_of(exc)
    assert rerun.log == record.log[: at + 1]


def cancelling(r: Any) -> Any:
    raise asyncio.CancelledError


def test_a_cancellation_propagates_with_the_replay_record_ending_in_its_cause() -> None:
    rules = {**RULES, "echo": Rule("d:echo", t.string, cancelling, version="1", reads=["t:body"])}
    exc = caught(replay(go().record, slice_design(rules=rules)))
    assert type(exc) is asyncio.CancelledError
    rerun = record_of(exc)
    assert rerun.log[-1] == {"item": "cause", "cause": {"code": "cancelled", "refs": []}}


def test_a_copy_clearing_a_refused_answer_replays_the_value() -> None:
    record = live(ask=returns({**SPAM, "q:spam": Unavailable(Cause("refused", ["q:spam"]))}))
    value = {"kind": "noul", "labels": ["yes", "no"], "probabilities": {"no": 0.2, "yes": 0.8}}

    def change(doc: dict[str, Any]) -> None:
        item = entry(doc["log"], "q:spam")
        del item["marker"]
        item["value"] = value

    copy_ = edited(record, change)
    rerun, found = run(copy_)
    assert rerun is not None
    assert entry(rerun.log, "q:spam")["value"] == value
    # The cleared fault holds: q:spam matches, and only what reads it changes downstream.
    assert "q:spam" not in [f.get("ref") for f in found]
    assert (found[0]["kind"], found[0].get("ref")) == (K.entry, "d:route")


def test_a_dropped_t_ref_is_never_asked_so_it_is_no_ref_not_in_record() -> None:
    ask = Source({"q:spam": ASK.provides["q:spam"]}, needs=ASK.needs, cost=1)
    echo = Rule("d:echo", t.string, RULES["echo"].function, version="1", reads=["t:state.subject"])
    changed = slice_design(sources={"ask": ask, "look": LOOK}, rules={**RULES, "echo": echo})
    record = go().record
    found = findings(record, changed)
    assert K.ref_not_in_record not in [f["kind"] for f in found]
    (body,) = [f for f in found if f.get("ref") == "t:body"]
    assert body["replayed"] is None


def test_a_candidate_valued_label_set_is_refused_at_load_so_replay_never_meets_one() -> None:
    q = Question("choice", Text("Which?"), labels={"fixed": ["none"], "candidates": "t:set"})
    with pytest.raises(InvalidDesign) as raised:
        slice_design(sources={"ask": Source({"q:which": q}), "look": LOOK})
    assert 12 in {f.check for f in raised.value.failures}


def test_a_when_reading_a_state_path_gates_its_source_and_replay_agrees() -> None:
    def to_dana(r: Any) -> bool:
        return r.read("t:state.sender.address") == "a@example.com"

    def to_nobody(r: Any) -> bool:
        return r.read("t:state.to") == "nobody"

    sources = {
        "ask": ASK,
        "look": LOOK,
        "mail": Source({"t:mail": t.string}, needs=["d:route"], when=When(to_dana, version="1")),
        "skip": Source({"t:skip": t.string}, when=When(to_nobody, version="1")),
        "gate": Source({"t:gate": t.string}, requirement=SourceRequirement("d:route", ["yes"])),
    }
    loaded = slice_design(sources=sources)
    replies = {"mail": returns({"t:mail": "m"}), "gate": returns({"t:gate": "g"})}
    record = live(loaded, **replies)
    assert entry(record.log, "t:mail")["call"] is not None
    assert entry(record.log, "t:gate")["call"] is not None
    assert entry(record.log, "t:skip")["marker"]["cause"] == {
        "code": "when-false",
        "refs": ["skip"],
    }
    assert findings(record, loaded) == []


# ---------------------------------------------------------------------------------------------
# The cut and the budget


def cut_of(record: Record) -> list[tuple[int, bool]]:
    return [(i["round"], i["deadline"]) for i in record.log if i["item"] == "round"]


def test_a_record_cut_before_asking_replays_with_no_finding() -> None:
    def slow(state: object) -> object:
        time.sleep(0.05)
        return state

    record = go(settings=RunSettings(deadline=0.01), codec=slow).record
    assert cut_of(record) == [(1, True)]
    rerun, found = run(record)
    assert found == []
    assert rerun is not None
    assert cut_of(rerun) == [(1, True)]
    assert entries(rerun) == entries(record)


def test_a_record_cut_after_asking_replays_with_no_finding_and_waits_for_nothing() -> None:
    record = live(
        CHAINED, settings=RunSettings(deadline=0.1), look=Slowly(10, returns({"t:size": 4}))
    )
    assert cut_of(record) == [(1, True)]
    (look,) = [i for i in record.log if i["item"] == "call" and i["sources"] == ["look"]]
    assert look["outcome"] == {"failed": {"code": "timed-out", "refs": ["t:size"]}}
    assert entry(record.log, "t:late")["marker"]["cause"]["code"] == "deadline"
    began = time.monotonic()
    rerun, found = run(record, CHAINED)
    assert time.monotonic() - began < 2  # the recorded call took 0.1; the fixture none
    assert found == []
    assert rerun is not None
    assert cut_of(rerun) == [(1, True)]
    # The cut asks round 1 only: the late source, past it, is cut again.
    assert entry(rerun.log, "t:late")["marker"]["cause"]["code"] == "deadline"


def test_a_cut_holds_under_a_design_that_moves_a_source_past_it() -> None:
    record = live(settings=RunSettings(deadline=0.1), ask=Slowly(10, returns(SPAM)))
    assert cut_of(record) == [(1, True)]
    assert entry(record.log, "t:size")["value"] == 4  # look was asked in round 1
    moved = Source(LOOK.provides, needs=["t:state", "d:route"])
    rerun, _ = run(record, slice_design(sources={"ask": ASK, "look": moved}))
    assert rerun is not None
    assert cut_of(rerun) == [(1, True)]
    assert all("look" not in i["sources"] for i in rerun.log if i["item"] == "call")
    marker = entry(rerun.log, "t:size")
    assert marker["marker"] == {"kind": "Unasked", "cause": {"code": "deadline", "refs": []}}
    assert (marker["round"], marker["call"]) == (1, None)


def test_a_recorded_budget_cut_replays_with_no_finding() -> None:
    late = Source({"t:late": t.string}, needs=["d:route"], cost=1)
    costly = slice_design(sources={"ask": ASK, "look": LOOK, "late": late})
    record = live(costly, settings=RunSettings(budget=1.5))
    assert cut_of(record) == [(1, False), (2, False)]
    assert entry(record.log, "t:late")["marker"]["cause"]["code"] == "over-budget"
    assert findings(record, costly) == []


def test_a_changed_parameter_a_slot_reads_is_question_changed() -> None:
    def design(bar: float) -> LoadedDesign:
        q = Question(
            "noul", Text("Is it spam above {bar}?", [{"name": "bar", "origin": {"param": "bar"}}])
        )
        ask = Source({"q:spam": q, "t:body": t.string}, needs=ASK.needs, cost=1)
        return slice_design(
            sources={"ask": ask, "look": LOOK}, parameters={"bar": Parameter(t.number(), bar)}
        )

    before = design(0.5)
    record = live(before)
    assert findings(record, before) == []
    after = design(0.6)
    found = findings(record, after)
    assert [f["kind"] for f in found] == [K.design_mismatch, K.question_changed]
    assert found[1]["refs"] == ["q:spam"]


class Slowly:
    """A reply that sleeps ``seconds`` before returning ``then``."""

    def __init__(self, seconds: float, then: object) -> None:
        self.seconds, self.then = seconds, then

    async def __call__(self, batch: Any) -> object:
        await asyncio.sleep(self.seconds)
        return self.then


# ---------------------------------------------------------------------------------------------
# Round trips over each feature. The budget and deadline cuts are the
# tests above.


def bounded_route(r: Any) -> Any:
    return pick_with_margin(r, "q:spam", r.param("spam/bar"), r.param("spam/margin"))


def to_dana(r: Any) -> bool:
    return r.read("t:state.sender.address") == "a@example.com"


def to_nobody(r: Any) -> bool:
    return r.read("t:state.to") == "nobody"


def routed_to_dana(r: Any) -> bool:
    return r.read("d:route") == "yes" and r.read("t:state.sender.address") == "a@example.com"


SPAM_BAR = {
    "spam/bar": Parameter(t.number(0, 1), 0.5),
    "spam/margin": Parameter(t.number(0, 1), 0.1),
}
MAILED = slice_design(
    sources={
        "ask": Source(ASK.provides, needs=ASK.needs, cost=1, batch="mail"),
        "look": Source(LOOK.provides, needs=LOOK.needs, batch="mail"),
    }
)
SLOTTED = slice_design(
    sources={
        "ask": Source(
            {
                "q:spam": Question(
                    "noul",
                    Text(
                        "Is {subject}, {size} long, spam above {bar}?",
                        [
                            {"name": "subject", "origin": {"ref": "t:state.subject"}},
                            {"name": "size", "origin": {"ref": "t:size"}},
                            {"name": "bar", "origin": {"param": "spam/bar"}},
                        ],
                    ),
                ),
                "t:body": t.string,
            },
            needs=ASK.needs,
            cost=1,
        ),
        "look": LOOK,
    },
    parameters=SPAM_BAR,
)
GATED = slice_design(
    sources={
        "ask": ASK,
        "look": LOOK,
        "mail": Source({"t:mail": t.string}, needs=["t:state"], when=When(to_dana, version="1")),
        "skip": Source({"t:skip": t.string}, when=When(to_nobody, version="1")),
        "held": Source({"t:held": t.string}, requirement=SourceRequirement("d:route", ["yes"])),
        "unheld": Source({"t:unheld": t.string}, requirement=SourceRequirement("d:route", ["no"])),
    }
)
GATED_REPLIES = {
    "mail": returns({"t:mail": "m"}),
    "held": returns({"t:held": "h"}),
    "skip": raising(AssertionError("skip is never asked")),
    "unheld": raising(AssertionError("unheld is never asked")),
}
CLASSED = slice_design(
    classes={"t:state": {"subject": "identifier"}, "t:body": "content", "d:route": "open"}
)
BOUNDED = slice_design(
    rules={
        **RULES,
        "route": Rule(
            "d:route",
            t.label("no", "yes"),
            bounded_route,
            version="1",
            reads=["q:spam"],
            params=["spam/bar", "spam/margin"],
            boundaries={"q:spam": roles(bar=["spam/bar"], margin=["spam/margin"])},
        ),
    },
    parameters=SPAM_BAR,
)
FROM_DOCUMENT = load(
    LOADED.document, functions={"route": route, "echo": echo, "policy": archive_policy}
)


def mailed() -> Record:
    reply = returns({**SPAM, "t:size": 4})
    return go(MAILED, binds={"mail": Scripted({"ask": reply})}).record


def calls_of(record: Record) -> list[list[str]]:
    return [i["sources"] for i in record.log if i["item"] == "call"]


FEATURES: dict[str, Callable[[], tuple[Record, LoadedDesign]]] = {
    "a batch key with several sources": lambda: (mailed(), MAILED),
    "board and parameter wording slots": lambda: (live(SLOTTED), SLOTTED),
    "a function when and a source requirement, each true and false": lambda: (
        live(GATED, **GATED_REPLIES),
        GATED,
    ),
    "declared classes": lambda: (live(CLASSED), CLASSED),
    "boundaries": lambda: (live(BOUNDED), BOUNDED),
    "a design loaded from its document": lambda: (go().record, FROM_DOCUMENT),
    "a turn under a design loaded from its document": lambda: (live(FROM_DOCUMENT), LOADED),
    "a refused verdict": lambda: (
        golden_refused(),
        slice_design(requirements=[Requirement("d:route", "no")]),
    ),
    "an escalate plan": lambda: (
        golden_escalate(),
        slice_design(policy=Policy(escalate_policy, version="1", reads=["d:route"])),
    ),
}


@pytest.mark.parametrize("feature", FEATURES)
def test_a_round_trip_over_each_feature_has_no_finding(feature: str) -> None:
    record, loaded = FEATURES[feature]()
    assert record.log[0]["hash"] == loaded.design_hash
    rerun, found = run(record, loaded)
    assert found == []
    assert rerun is not None
    assert entries(rerun) == entries(record)


def test_the_features_round_trips_exercise_what_they_name() -> None:
    assert calls_of(mailed()) == [["ask", "look"]]
    slotted = live(SLOTTED)
    assert entry(slotted.log, "q:spam")["round"] == 2  # the board slot is a need
    gated = live(GATED, **GATED_REPLIES)
    assert entry(gated.log, "t:mail")["value"] == "m"
    assert entry(gated.log, "t:held")["value"] == "h"
    for ref, refs in (("t:skip", ["skip"]), ("t:unheld", ["d:route"])):
        assert entry(gated.log, ref)["marker"]["cause"] == {"code": "when-false", "refs": refs}
    assert live(CLASSED).log[1]["classes"]["d:route"] == "open"
    assert dict(BOUNDED._grid)
    assert FROM_DOCUMENT.design_hash == LOADED.design_hash
    assert FROM_DOCUMENT.document == LOADED.document


# ---------------------------------------------------------------------------------------------
# Under another design: answered by source, not by round or batch


def test_a_when_that_moves_a_source_to_another_round_still_finds_its_call() -> None:
    before = slice_design(
        sources={
            "ask": ASK,
            "look": LOOK,
            "mail": Source(
                {"t:mail": t.string}, needs=["t:state"], when=When(to_dana, version="1")
            ),
        }
    )
    after = slice_design(
        sources={
            "ask": ASK,
            "look": LOOK,
            "mail": Source(
                {"t:mail": t.string},
                needs=["t:state", "d:route"],
                when=When(routed_to_dana, version="2"),
            ),
        }
    )
    record = live(before, mail=returns({"t:mail": "m"}))
    assert entry(record.log, "t:mail")["round"] == 1
    rerun, found = run(record, after)
    assert rerun is not None
    moved = entry(rerun.log, "t:mail")
    assert (moved["round"], moved["value"]) == (2, "m")
    assert found == [
        {"kind": K.design_mismatch, "recorded": before.design_hash, "replayed": after.design_hash},
        {
            "kind": K.entry,
            "ref": "t:mail",
            "recorded": entry(record.log, "t:mail"),
            "replayed": moved,
            "rule": None,
        },
    ]


def answered(record: Record) -> dict[str, Any]:
    return {
        i["ref"]: i.get("value", i.get("marker"))
        for i in record.log
        if i["item"] == "entry" and i["ref"] != "t:state"
    }


@pytest.mark.parametrize(
    ("recorded", "given"),
    [("batched", "split"), ("split", "batched")],
    ids=["a batch split", "two calls batched"],
)
def test_a_regrouped_batch_is_answered_by_source_in_its_new_batch(
    recorded: str, given: str
) -> None:
    designs = {"batched": MAILED, "split": LOADED}
    record = mailed() if recorded == "batched" else go().record
    loaded = designs[given]
    rerun, found = run(record, loaded)
    assert rerun is not None
    assert calls_of(rerun) == ([["ask", "look"]] if given == "batched" else [["ask"], ["look"]])
    # Every value is the recorded one; only the call index t:size names moved.
    assert answered(rerun) == answered(record)
    moved = (entry(record.log, "t:size")["call"], entry(rerun.log, "t:size")["call"])
    assert moved == ((3, 4) if recorded == "batched" else (4, 3))
    assert found == [
        {
            "kind": K.design_mismatch,
            "recorded": designs[recorded].design_hash,
            "replayed": loaded.design_hash,
        },
        {
            "kind": K.entry,
            "ref": "t:size",
            "recorded": entry(record.log, "t:size"),
            "replayed": entry(rerun.log, "t:size"),
            "rule": None,
        },
    ]
