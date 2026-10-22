# pyright: reportPrivateUsage=false
"""Interpret: rounds over the turn's sorter, the oracle calls, the write
path and every component-failure case the v0.0.1 slice reaches, against a scripted fixture
binding."""

import asyncio
import contextlib
import enum
import gc
import inspect
import random
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any, cast

import pytest

from counting import SmallBounds, bound, counted
from designs import minimal, slot
from jurat import (
    Answer,
    AnswerKind,
    CallResult,
    Cause,
    CauseCode,
    Failed,
    LoadedDesign,
    Parameter,
    Question,
    Record,
    RenderedQuestion,
    Rule,
    Source,
    SourceCall,
    SourceRequirement,
    Text,
    Unasked,
    Unavailable,
    Unsure,
    When,
    _interpret,
    load,
    read_record,
)
from jurat import types as t
from jurat._canonical import hash_value
from jurat._read_view import _ViewError
from jurat.helpers import pick_with_margin
from scripted import (
    Batch,
    Claims,
    InheritsAwait,
    MetaRaising,
    NotACoroutine,
    Scripted,
    clashing,
    generator_coroutine,
    ignoring_cancel,
    interpret_on,
    open_ledger,
    raising,
    raising_on_cancel,
    returns,
    settings,
    sleeping,
    started_raising_on_close,
    stop,
    turn,
)

STATE_T = t.record(subject=t.string)
STATE = {"subject": "hello"}
NOUL = {"yes": 0.9, "no": 0.1}
TONE = {"warm": 0.75, "cold": 0.25}
OK = {"q:spam": NOUL, "q:tone": TONE, "t:body": "text"}
ASKED = ["q:spam", "q:tone", "t:body"]
PROFILE = t.record(name=t.string, age=t.integer(), home=t.record(city=t.string))


def ok(**provenance: Any) -> CallResult:
    return returns(dict(OK), **provenance)


def with_values(**changes: Any) -> CallResult:
    """``OK`` with the refs named by their name, ``q_spam`` for ``q:spam``, changed; a value
    of ``DROP`` leaves the ref out."""
    values: dict[str, Any] = dict(OK)
    for name, value in changes.items():
        ref = name.replace("_", ":", 1)
        if value is DROP:
            del values[ref]
        else:
            values[ref] = value
    return returns(values)


DROP = object()


def ask(**changes: Any) -> Source:
    """The questions of the ``mail`` oracle; ``told`` is its ``t:`` half, so a design with
    both makes one call."""
    fields: dict[str, Any] = {
        "provides": {
            "q:spam": Question("noul", Text("Is it spam?")),
            "q:tone": Question("choice", Text("Which tone?"), labels=("warm", "cold")),
        },
        "needs": ["t:state.subject"],
        "oracle": "mail",
        "payload": {"model": "m", "tags": ("a", "b")},
        "cost": 2,
    }
    fields.update(changes)
    return Source(**fields)


def told(**changes: Any) -> Source:
    """``t:body``, ready with ``ask`` and called with it through the ``mail`` oracle."""
    fields: dict[str, Any] = {
        "provides": {"t:body": t.string},
        "needs": ["t:state.subject"],
        "oracle": "mail",
    }
    fields.update(changes)
    return Source(**fields)


def asking(**changes: Any) -> dict[str, Source]:
    """``ask`` and its ``t:`` half ``body``: one batch, one call. A change to when the pair is
    ready or how long it may take applies to both."""
    shared = {k: v for k, v in changes.items() if k in ("needs", "when", "requirement", "timeout")}
    return {"ask": ask(**changes), "body": told(**shared)}


def look(**changes: Any) -> Source:
    fields: dict[str, Any] = {
        "provides": {
            "t:profile": PROFILE,
            "t:blob": t.bytes_,
            "t:maybe": t.nullable(t.string),
            "t:count": t.integer(),
        },
    }
    fields.update(changes)
    return Source(**fields)


LOOKED = {
    "t:profile": {"name": "Ann", "age": 3, "home": {"city": "Oslo"}},
    "t:blob": b"\x00\x01",
    "t:maybe": None,
    "t:count": 7,
}


def make(
    sources: dict[str, Source] | None = None,
    rules: dict[str, Rule] | None = None,
    **changes: Any,
) -> LoadedDesign:
    return load(minimal(state=STATE_T, sources=sources or {}, rules=rules or {}, **changes))


def rule(
    rid: str,
    fn: Callable[[Any], Any],
    *reads: str,
    outputs: str = t.boolean,
    params: Sequence[str] = (),
) -> Rule:
    return Rule(f"d:{rid}", outputs, fn, version="1", reads=reads, params=params)


def route(r: Any) -> Any:
    return pick_with_margin(r, "q:spam", 0.5, 0.1)


ROUTE = rule("route", route, "q:spam", outputs=t.label("no", "yes"))


def run(
    loaded: LoadedDesign, replies: Mapping[str, object], **run_settings: Any
) -> tuple[Record, Scripted]:
    binding = Scripted(replies)
    record = turn(loaded, {"default": binding, "mail": binding}, STATE, **run_settings)
    return record, binding


def body(record: Record) -> list[dict[str, Any]]:
    """The items interpret appended: everything after the state entry."""
    return record.log[3:]


def entry(record: Record, ref: str) -> dict[str, Any]:
    (found,) = [i for i in record.log if i["item"] == "entry" and i["ref"] == ref]
    return found


def mark(code: str, *refs: str, kind: str = "Unavailable") -> dict[str, Any]:
    return {"kind": kind, "cause": {"code": code, "refs": list(refs)}}


def answer(kind: str, labels: list[str], probabilities: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "labels": labels, "probabilities": probabilities}


def identity(template: str, kind: str = "noul", labels: list[str] | None = None) -> str:
    return hash_value(
        {"kind": kind, "wording": {"template": template, "slots": []}, "labels": labels}
    )


def kinds(record: Record) -> list[tuple[str, str | int]]:
    """Each appended item as (kind, ref) for an entry, (kind, round) otherwise."""
    return [(i["item"], i["ref"] if i["item"] == "entry" else i["round"]) for i in body(record)]


# ---------------------------------------------------------------------------------------------
# Round 0 and the fixpoint


def subject_len(r: Any) -> int:
    return len(r.read("t:state.subject"))


def test_a_design_with_no_source_stops_after_round_0_with_no_round_item() -> None:
    loaded = make(rules={"size": rule("size", subject_len, "t:state.subject", outputs=t.integer())})
    record, _ = run(loaded, {})
    assert body(record) == [
        {"item": "entry", "ref": "d:size", "round": 0, "call": None, "value": 5, "derivation": []}
    ]


def test_round_0_fires_rule_wavefronts_each_sorted_by_id() -> None:
    rules = {
        "zed": rule("zed", lambda r: True, "t:state"),
        "alpha": rule("alpha", lambda r: r.read("d:zed"), "d:zed"),
        "mid": rule("mid", lambda r: False, "t:state"),
    }
    record, _ = run(make(rules=rules), {})
    assert kinds(record) == [("entry", "d:mid"), ("entry", "d:zed"), ("entry", "d:alpha")]
    assert {i["round"] for i in body(record)} == {0}


def test_one_working_round_is_round_1_with_its_call_entries_rules_and_round_item() -> None:
    # Probe n: one working round is one numbered round.
    loaded = make(asking(), {"route": ROUTE})
    record, _ = run(loaded, {"ask": ok()})
    questions = {
        "q:spam": {"identity": identity("Is it spam?"), "kind": "noul", "labels": None},
        "q:tone": {
            "identity": identity("Which tone?", "choice", ["warm", "cold"]),
            "kind": "choice",
            "labels": ["warm", "cold"],
        },
    }
    assert body(record) == [
        {
            "item": "call",
            "round": 1,
            "sources": ["ask", "body"],
            "outcome": "returned",
            "snapshot": None,
            "elicitation": None,
            "prompt": None,
            "precision": None,
            "questions": questions,
        },
        {
            "item": "entry",
            "ref": "q:spam",
            "round": 1,
            "call": 3,
            "value": answer("noul", ["yes", "no"], NOUL),
        },
        {
            "item": "entry",
            "ref": "q:tone",
            "round": 1,
            "call": 3,
            "value": answer("choice", ["warm", "cold"], TONE),
        },
        {"item": "entry", "ref": "t:body", "round": 1, "call": 3, "value": "text"},
        {
            "item": "entry",
            "ref": "d:route",
            "round": 1,
            "call": None,
            "value": "yes",
            "derivation": [
                {"helper": "pick_with_margin", "refs": ["q:spam"], "rested_on": ["q:spam"]}
            ],
        },
        {"item": "round", "round": 1, "cost": 2, "deadline": False},
    ]
    calls = record.unhashed["calls"]
    assert list(calls) == ["3"]
    assert type(calls["3"]["start"]) is str
    assert calls["3"]["start"].endswith("Z")
    assert calls["3"]["latency"] >= 0
    assert (calls["3"]["usage"], calls["3"]["attempts"]) == (None, None)
    assert "exceptions" not in record.unhashed
    assert "reads" not in record.unhashed


def test_rounds_are_numbered_from_1_without_a_gap() -> None:
    sources = {
        "first": Source({"t:a": t.string}),
        "second": Source({"t:b": t.string}, needs=["t:a"]),
        "third": Source({"t:c": t.string}, needs=["t:b"]),
    }
    record, _ = run(
        make(sources),
        {
            "first": returns({"t:a": "a"}),
            "second": returns({"t:b": "b"}),
            "third": returns({"t:c": "c"}),
        },
    )
    assert [(i["item"], i["round"]) for i in body(record) if i["item"] != "entry"] == [
        ("call", 1),
        ("round", 1),
        ("call", 2),
        ("round", 2),
        ("call", 3),
        ("round", 3),
    ]


# ---------------------------------------------------------------------------------------------
# The batch


def test_a_call_passes_each_source_call_with_its_payload_needs_refs_and_questions() -> None:
    record, binding = run(make(asking()), {"ask": ok()})
    assert record.log[3]["outcome"] == "returned"
    ((call, told),) = binding.batches
    assert call.id == "ask"
    assert call.payload == {"model": "m", "tags": ["a", "b"]}
    assert call.needs == {"t:state.subject": "hello"}
    assert call.refs == ()
    assert list(call.questions) == ["q:spam", "q:tone"]
    assert (told.id, told.payload, told.needs) == ("body", None, {"t:state.subject": "hello"})
    assert (told.refs, told.questions) == (("t:body",), {})
    assert call.questions["q:spam"] == RenderedQuestion(
        AnswerKind.noul, "Is it spam?", ("yes", "no"), descriptions=(None, None)
    )
    assert call.questions["q:tone"] == RenderedQuestion(
        AnswerKind.choice, "Which tone?", ("warm", "cold"), descriptions=(None, None)
    )


def test_a_wording_renders_its_escapes_and_its_identity_hashes_the_template_as_written() -> None:
    source = ask(provides={"q:spam": Question("noul", Text("Is {{it}} spam?}}"))})
    record, binding = run(make({"ask": source}), {"ask": returns({"q:spam": NOUL})})
    assert binding.batches[0][0].questions["q:spam"].wording == "Is {it} spam?}"
    assert record.log[3]["questions"]["q:spam"]["identity"] == identity("Is {{it}} spam?}}")


def test_the_identity_does_not_hash_the_source_id_or_the_ref() -> None:
    def ident(sid: str, ref: str, template: str) -> str:
        source = Source({ref: Question("noul", Text(template))})
        record, _ = run(make({sid: source}), {sid: returns({ref: NOUL})})
        return record.log[3]["questions"][ref]["identity"]

    assert ident("ask", "q:spam", "Is it?") == ident("other", "q:junk", "Is it?")
    assert ident("ask", "q:spam", "Is it?") != ident("ask", "q:spam", "Is it so?")


def test_each_source_with_no_oracle_is_its_own_call_through_default_run_concurrently() -> None:
    started: list[str] = []
    both = asyncio.Event()

    async def reply(batch: Batch) -> object:
        started.append(batch[0].id)
        if len(started) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 2)
        return returns({f"t:{batch[0].id}": "v"})

    sources = {"zeta": Source({"t:zeta": t.string}), "alpha": Source({"t:alpha": t.string})}
    record, binding = run(make(sources), {"zeta": reply, "alpha": reply})
    assert [len(b) for b in binding.batches] == [1, 1]
    assert kinds(record) == [
        ("call", 1),
        ("call", 1),
        ("entry", "t:alpha"),
        ("entry", "t:zeta"),
        ("round", 1),
    ]
    assert [i["sources"] for i in body(record)[:2]] == [["alpha"], ["zeta"]]
    assert [i["outcome"] for i in body(record)[:2]] == ["returned", "returned"]
    assert entry(record, "t:alpha")["call"] == 3
    assert entry(record, "t:zeta")["call"] == 4


def test_a_round_writes_asked_and_marked_sources_interleaved_by_source_id() -> None:
    sources = {
        "root": Source({"t:x": t.string, "t:y": t.string}),
        "a": Source({"t:a": t.string}, needs=["t:x"]),
        "b": Source({"t:b": t.string}, needs=["t:y"]),
    }
    replies = {"root": returns({"t:y": "y"}), "b": returns({"t:b": "b"})}
    record, _ = run(make(sources), replies)
    round_2 = [i for i in body(record) if i["round"] == 2]
    assert [(i["item"], i.get("ref")) for i in round_2] == [
        ("call", None),
        ("entry", "t:a"),
        ("entry", "t:b"),
        ("round", None),
    ]
    assert entry(record, "t:a")["marker"] == mark("need-unavailable", "t:x")
    assert entry(record, "t:a")["call"] is None
    assert round_2[0]["sources"] == ["b"]


# ---------------------------------------------------------------------------------------------
# The needs mapping and the board (probe i)


def test_a_source_with_no_needs_is_passed_none_of_the_board() -> None:
    # Probe i: a fetch could read an undeclared ref off the board.
    sources = {**asking(), "nosy": Source({"t:nosy": t.string})}
    _, binding = run(make(sources), {"ask": ok(), "nosy": returns({"t:nosy": "n"})})
    (nosy,) = [b[0] for b in binding.batches if b[0].id == "nosy"]
    assert nosy.needs == {}


def test_changing_the_needs_dict_reaches_neither_the_board_nor_the_record() -> None:
    async def meddle(batch: Batch) -> object:
        needs = batch[0].needs
        needs["t:profile"]["name"] = "Mallory"
        needs["q:spam"].probabilities["yes"] = 0
        needs.clear()
        return returns({"t:after": "x"})

    seen: list[Any] = []

    def check(r: Any) -> bool:
        seen.append((r.read("t:profile")["name"], r.read("q:spam").probabilities["yes"]))
        return True

    sources = {
        **asking(),
        "look": look(),
        "meddle": Source({"t:after": t.string}, needs=["t:profile", "q:spam", "t:state"]),
    }
    rules = {"check": rule("check", check, "t:profile", "q:spam", "t:after")}
    record, binding = run(
        make(sources, rules), {"ask": ok(), "look": returns(dict(LOOKED)), "meddle": meddle}
    )
    assert seen == [("Ann", 0.9)]
    assert entry(record, "t:profile")["value"]["name"] == "Ann"
    assert entry(record, "q:spam")["value"]["probabilities"]["yes"] == 0.9
    assert [b[0].id for b in binding.batches] == ["ask", "look", "meddle"]


def test_a_q_need_is_passed_as_the_answer_a_rule_reads() -> None:
    sources = {**asking(), "next": Source({"t:n": t.string}, needs=["q:spam", "t:profile.name"])}
    sources["look"] = look()
    _, binding = run(
        make(sources), {"ask": ok(), "look": returns(dict(LOOKED)), "next": returns({"t:n": "n"})}
    )
    (call,) = [b[0] for b in binding.batches if b[0].id == "next"]
    spam = call.needs["q:spam"]
    assert isinstance(spam, Answer)
    assert (spam.kind, spam.labels, dict(spam.probabilities)) == (
        AnswerKind.noul,
        ("yes", "no"),
        NOUL,
    )
    assert call.needs["t:profile.name"] == "Ann"


# ---------------------------------------------------------------------------------------------
# A call that fails whole


def failed(record: Record, code: str, refs: list[str] = ASKED) -> None:
    """The call item at index 3 is `Failed` with ``code`` naming the batch's refs, every ref
    of the batch `Unavailable` with it, naming the call."""
    call = record.log[3]
    assert call["outcome"] == {"failed": {"code": code, "refs": refs}}
    assert (call["snapshot"], call["elicitation"], call["prompt"]) == (None, None, None)
    for ref in refs:
        assert entry(record, ref)["marker"] == mark(code, *refs)
        assert entry(record, ref)["call"] == 3


def test_a_binding_that_raises_fails_the_call_with_raised_and_its_text_in_unhashed() -> None:
    record, _ = run(make(asking(), {"route": ROUTE}), {"ask": raising(RuntimeError("down"))})
    failed(record, "raised")
    assert set(record.log[3]["questions"]) == {"q:spam", "q:tone"}  # on a Failed call too
    exc = record.unhashed["exceptions"]["3"]
    assert (exc["type"], exc["message"]) == ("RuntimeError", "down")
    assert "RuntimeError: down" in exc["traceback"]
    assert (record.unhashed["calls"]["3"]["usage"], record.unhashed["calls"]["3"]["attempts"]) == (
        None,
        None,
    )
    # The rule still fires, reading the marker.
    assert entry(record, "d:route")["marker"] == mark("raised", *ASKED)


def test_a_bindings_own_timeout_error_is_raised_not_timed_out() -> None:
    record, _ = run(make(asking()), {"ask": raising(TimeoutError("its own"))})
    failed(record, "raised")


def test_a_call_past_its_timeout_is_timed_out() -> None:
    record, _ = run(make(asking(timeout=0.02)), {"ask": sleeping(5, ok())})
    failed(record, "timed-out")
    assert "exceptions" not in record.unhashed


def test_the_run_settings_timeout_bounds_a_source_that_declares_none() -> None:
    record, _ = run(make(asking()), {"ask": sleeping(5, ok())}, timeout=0.02)
    failed(record, "timed-out")


def test_a_late_result_after_a_swallowed_cancellation_is_discarded_as_timed_out() -> None:
    record, _ = run(make(asking(timeout=0.02)), {"ask": ignoring_cancel(5, ok())})
    failed(record, "timed-out")


def test_a_raise_after_the_timeout_is_timed_out_with_its_text_kept() -> None:
    record, _ = run(make(asking(timeout=0.02)), {"ask": raising_on_cancel(5, KeyError("late"))})
    failed(record, "timed-out")
    assert record.unhashed["exceptions"]["3"]["type"] == "KeyError"


class Sub(CallResult):
    pass


class DictSub(dict[str, Any]):
    pass


@pytest.mark.parametrize(
    "reply",
    [
        ["q:spam"],  # probe l: a fetch returns a list
        dict(OK),  # a mapping, not a call result
        None,
        Answer("noul", ("yes", "no"), NOUL),
        Sub(dict(OK)),
        CallResult(cast(Any, DictSub(OK))),
        CallResult(cast(Any, list(OK.items()))),
    ],
)
def test_a_return_that_is_not_exactly_a_call_result_or_failed_is_invalid_return(
    reply: object,
) -> None:
    record, _ = run(make(asking()), {"ask": reply})
    failed(record, "invalid-return")
    assert "invalid" not in record.unhashed["calls"]["3"]


@pytest.mark.parametrize("code", ["raised", "timed-out", "invalid-return", "unexpected-ref"])
def test_a_returned_failed_names_every_ref_of_the_batch_whatever_it_listed(code: str) -> None:
    record, _ = run(make(asking()), {"ask": Failed(Cause(code, ["t:other", "q:spam"]))})
    failed(record, code)
    assert "invalid" not in record.unhashed["calls"]["3"]


def test_a_returned_failed_with_another_code_is_invalid_return_its_code_in_unhashed() -> None:
    record, _ = run(make(asking()), {"ask": Failed(Cause("refused"))})
    failed(record, "invalid-return")
    # A StrEnum member is no exact str: its type is named.
    assert record.unhashed["calls"]["3"]["invalid"] == {"code": "<jurat._markers.CauseCode>"}


@pytest.mark.parametrize("cls", [CallResult, Failed])
def test_a_call_result_or_failed_built_without_init_is_invalid_return(
    cls: type[object],
) -> None:
    built = object.__new__(cls)
    record, _ = run(make(asking()), {"ask": built})
    failed(record, "invalid-return")


def test_a_failed_whose_cause_was_tampered_with_is_invalid_return() -> None:
    outcome = Failed(Cause("raised"))
    object.__setattr__(outcome.cause, "code", "raised")  # a str, no longer the member
    record, _ = run(make(asking()), {"ask": outcome})
    failed(record, "invalid-return")
    assert record.unhashed["calls"]["3"]["invalid"] == {"code": "'raised'"}


def test_an_unexpected_key_fails_the_call_and_is_kept_in_unhashed() -> None:
    values: dict[Any, Any] = {**OK, "t:nope": 1, 7: "x"}
    record, _ = run(make(asking()), {"ask": returns(values)})
    failed(record, "unexpected-ref")
    assert record.unhashed["calls"]["3"]["unexpected"] == ["t:nope", "7"]


def test_an_unexpected_key_outranks_a_bad_provenance_member() -> None:
    values: dict[Any, Any] = {**OK, "t:nope": 1}
    record, _ = run(make(asking()), {"ask": returns(values, snapshot=3)})
    failed(record, "unexpected-ref")
    assert record.unhashed["calls"]["3"]["invalid"] == {"snapshot": "3"}


@pytest.mark.parametrize(
    ("member", "value"),
    [("snapshot", 3), ("elicitation", b"e"), ("prompt", "\ud800"), ("snapshot", ["m"])],
)
def test_a_provenance_member_not_null_or_a_domain_string_is_invalid_return(
    member: str, value: object
) -> None:
    record, _ = run(make(asking()), {"ask": ok(**{member: value})})
    failed(record, "invalid-return")
    assert record.unhashed["calls"]["3"]["invalid"] == {member: repr(value)}


# ---------------------------------------------------------------------------------------------
# A call that returned


def test_a_call_items_provenance_is_the_call_results() -> None:
    record, _ = run(
        make(asking()),
        {
            "ask": ok(
                snapshot="m-1",
                elicitation="e-2",
                prompt="the prompt",
                usage={"in": 3, "out": 1.5},
                attempts=2,
            )
        },
    )
    call = record.log[3]
    assert (call["snapshot"], call["elicitation"], call["prompt"]) == ("m-1", "e-2", "the prompt")
    assert record.unhashed["calls"]["3"]["usage"] == {"in": 3, "out": 1.5}
    assert record.unhashed["calls"]["3"]["attempts"] == 2


class Mood(enum.StrEnum):
    calm = "calm"


@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("usage", "many"),
        ("usage", {"in": float("nan")}),
        ("usage", {"in": True}),
        ("usage", {3: 1}),
        ("attempts", True),
        ("attempts", 1.0),
        ("attempts", 2**60),
    ],
)
def test_a_bad_usage_or_attempts_fails_nothing_and_is_recorded_null(
    member: str, value: object
) -> None:
    record, _ = run(make(asking()), {"ask": ok(**{member: value})})
    assert record.log[3]["outcome"] == "returned"
    assert record.unhashed["calls"]["3"][member] is None
    assert record.unhashed["calls"]["3"]["invalid"] == {member: repr(value)}


# ---------------------------------------------------------------------------------------------
# Value faults


def stands(record: Record, *refs: str) -> None:
    assert record.log[3]["outcome"] == "returned"
    for ref in refs:
        assert "value" in entry(record, ref)


def test_a_t_value_that_does_not_conform_fails_that_ref_alone() -> None:
    record, _ = run(make(asking()), {"ask": with_values(t_body=3)})
    assert entry(record, "t:body")["marker"] == mark("invalid-return", "t:body")
    assert entry(record, "t:body")["call"] == 3
    stands(record, "q:spam", "q:tone")


@pytest.mark.parametrize(
    "value",
    [
        {"yes": 1},  # a label missing
        {"yes": 0.5, "no": 0.5, "maybe": 0},  # a key not passed
        {"yes": True, "no": False},  # a bool is not a number
        Answer("noul", ("yes", "no"), NOUL),  # an Answer is not exactly a dict
        DictSub(NOUL),
        0.9,
        None,
    ],
)
def test_a_q_value_that_fails_validation_is_invalid_answer(value: object) -> None:
    record, _ = run(make(asking()), {"ask": with_values(q_spam=value)})
    assert entry(record, "q:spam")["marker"] == mark("invalid-answer", "q:spam")
    stands(record, "q:tone", "t:body")


class Prob(float):
    pass


def test_a_probability_of_a_float_subclass_is_invalid_answer() -> None:
    record, _ = run(make(asking()), {"ask": with_values(q_spam={"yes": Prob(0.9), "no": 0.1})})
    assert entry(record, "q:spam")["marker"] == mark("invalid-answer", "q:spam")


def test_an_answer_near_one_is_recorded_as_returned_and_not_flagged() -> None:
    near = {"yes": 0.4999999, "no": 0.5}
    record, _ = run(make(asking()), {"ask": with_values(q_spam=near)})
    assert entry(record, "q:spam")["value"] == answer("noul", ["yes", "no"], near)
    assert "sums" not in record.unhashed["calls"]["3"]


def test_an_answer_off_one_is_recorded_as_returned_and_its_sum_flagged() -> None:
    off = {"yes": 0.8, "no": 0.1}
    loaded = make(asking(), {"route": ROUTE})
    record, _ = run(loaded, {"ask": with_values(q_spam=off)})
    assert entry(record, "q:spam")["value"] == answer("noul", ["yes", "no"], off)
    assert record.unhashed["calls"]["3"]["sums"] == {"q:spam": 0.9}
    # The flag changes no outcome: the rule reads the values as returned.
    assert entry(record, "d:route")["value"] == "yes"
    flat, _ = run(loaded, {"ask": with_values(q_spam={"yes": 0.8, "no": 0.2})})
    assert entry(flat, "d:route") == entry(record, "d:route")


@pytest.mark.parametrize(
    ("spam", "tone", "sums"),
    [
        ({"yes": 0.45, "no": 0.5}, TONE, None),  # exactly 0.95
        ({"yes": 0.5, "no": 0.55}, TONE, None),  # exactly 1.05, though its floats sum past it
        ({"yes": 0.5000000000000001, "no": 0.55}, TONE, {"q:spam": 1.05}),
        ({"yes": 0.4499999999999999, "no": 0.5}, TONE, {"q:spam": 0.9499999999999999}),
        (
            {"yes": 0.9, "no": 0.9},
            {"warm": 0, "cold": 0.0},
            {"q:spam": 1.8, "q:tone": 0.0},
        ),
        ({"yes": 0.1, "no": 0.2}, TONE, {"q:spam": 0.3}),  # the exact sum, rounded once
    ],
)
def test_each_answer_off_the_band_is_flagged_under_its_ref(
    spam: dict[str, float], tone: dict[str, float], sums: dict[str, float] | None
) -> None:
    record, _ = run(make(asking()), {"ask": with_values(q_spam=spam, q_tone=tone)})
    stands(record, "q:spam", "q:tone", "t:body")
    call = record.unhashed["calls"]["3"]
    if sums is None:
        assert "sums" not in call
    else:
        assert call["sums"] == sums


def test_an_invalid_answer_or_a_marker_is_never_flagged() -> None:
    record, _ = run(
        make(asking()),
        {
            "ask": with_values(
                q_spam={"yes": 2, "no": 0}, q_tone=Unavailable(Cause(CauseCode.refused))
            )
        },
    )
    assert entry(record, "q:spam")["marker"] == mark("invalid-answer", "q:spam")
    assert entry(record, "q:tone")["marker"] == mark("refused", "q:tone")
    assert "sums" not in record.unhashed["calls"]["3"]


def test_a_flagged_answer_reads_back_with_its_sums() -> None:
    loaded = make(asking(), {"route": ROUTE})
    record, _ = run(loaded, {"ask": with_values(q_spam={"yes": 0.9, "no": 0.9})})
    back = read_record(record.document)
    assert back.document == record.document
    assert back.unhashed["calls"]["3"]["sums"] == {"q:spam": 1.8}


def test_an_omitted_ref_is_not_returned_naming_that_ref() -> None:
    record, _ = run(make(asking()), {"ask": with_values(q_tone=DROP)})
    assert entry(record, "q:tone")["marker"] == mark("not-returned", "q:tone")
    assert entry(record, "q:tone")["call"] == 3
    stands(record, "q:spam", "t:body")


@pytest.mark.parametrize("code", ["refused", "invalid-answer", "invalid-return"])
def test_an_allowed_marker_is_written_with_its_code_naming_that_ref_alone(code: str) -> None:
    returned = Unavailable(Cause(code, ["t:body", "elsewhere"]))
    record, _ = run(make(asking()), {"ask": with_values(q_spam=returned)})
    assert entry(record, "q:spam")["marker"] == mark(code, "q:spam")
    stands(record, "q:tone", "t:body")


class Odd(Unavailable):
    pass


@pytest.mark.parametrize(
    "returned",
    [
        Unsure(Cause("abstained")),
        Unasked(Cause("deadline")),
        Unavailable(Cause("timed-out")),
        Odd(Cause("refused")),
    ],
)
def test_any_other_marker_in_values_is_invalid_return(returned: object) -> None:
    record, _ = run(make(asking()), {"ask": with_values(q_spam=returned, t_body=returned)})
    assert entry(record, "q:spam")["marker"] == mark("invalid-return", "q:spam")
    assert entry(record, "t:body")["marker"] == mark("invalid-return", "t:body")


def test_a_tampered_allowed_marker_is_invalid_return() -> None:
    returned = Unavailable(Cause("refused"))
    object.__setattr__(returned, "cause", "refused")
    record, _ = run(make(asking()), {"ask": with_values(t_body=returned)})
    assert entry(record, "t:body")["marker"] == mark("invalid-return", "t:body")


def test_none_is_a_value() -> None:
    record, _ = run(make({"look": look()}), {"look": returns(dict(LOOKED))})
    assert entry(record, "t:maybe")["value"] is None


class Count(enum.IntEnum):
    one = 1


@pytest.mark.parametrize(
    ("ref", "value"),
    [
        ("t:count", Count.one),  # an int subclass is outside the domain
        ("t:count", True),  # a bool is never an integer
        ("t:count", 1.0),  # an integral float never conforms to integer
        ("t:count", 2**60),
        ("t:blob", "AAE"),  # a bytes value is Python bytes when returned
        ("t:profile", {"name": "Ann", "age": 3}),  # a declared field absent
        ("t:profile", {"name": "Ann", "age": 3, "home": None}),
        ("t:maybe", 3),
    ],
)
def test_a_t_value_of_the_wrong_type_is_invalid_return(ref: str, value: object) -> None:
    record, _ = run(make({"look": look()}), {"look": returns({**LOOKED, ref: value})})
    assert entry(record, ref)["marker"] == mark("invalid-return", ref)


def test_a_string_subclass_value_is_written_as_its_string_data() -> None:
    record, _ = run(make(asking()), {"ask": with_values(t_body=Mood.calm)})
    assert entry(record, "t:body")["value"] == "calm"


def test_undeclared_fields_are_dropped_at_every_depth_before_the_domain_test() -> None:
    extra = {"name": "Ann", "age": 3, "home": {"city": "Oslo", "zip": {1, 2}}, "x": object()}
    record, _ = run(make({"look": look()}), {"look": returns({**LOOKED, "t:profile": extra})})
    assert entry(record, "t:profile")["value"] == LOOKED["t:profile"]


def test_a_bytes_value_is_base64url_in_the_record_and_bytes_on_the_board() -> None:
    seen: list[object] = []

    def blob(r: Any) -> bool:
        seen.append(r.read("t:blob"))
        return True

    loaded = make({"look": look()}, {"blob": rule("blob", blob, "t:blob")})
    record, _ = run(loaded, {"look": returns(dict(LOOKED))})
    assert entry(record, "t:blob")["value"] == "AAE"
    assert seen == [b"\x00\x01"]


# ---------------------------------------------------------------------------------------------
# A rule's outcome


def one_rule(fn: Callable[[Any], Any], *reads: str, **kw: Any) -> Record:
    loaded = make(
        rules={"r": rule("r", fn, *reads, **kw)}, parameters={"bar": Parameter(t.number(), 0.5)}
    )
    record, _ = run(loaded, {})
    return record


@pytest.mark.parametrize("outputs", [t.number(), t.integer(), t.list_(t.number())])
def test_a_rule_returning_an_integer_past_the_digit_limit_is_invalid_return(outputs: str) -> None:
    huge = 10**5000
    returned = [huge] if outputs.startswith("list") else huge
    record = one_rule(lambda r: returned, outputs=outputs)
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")


def fails(message: str) -> Callable[[Any], Any]:
    def fn(r: Any) -> Any:
        raise ValueError(message)

    return fn


def test_a_rule_that_raises_is_unavailable_raised_naming_the_rule() -> None:
    record = one_rule(fails("bad rule"), "t:state")
    assert entry(record, "d:r")["marker"] == mark("raised", "r")
    assert entry(record, "d:r")["derivation"] == []
    exc = record.unhashed["exceptions"]["3"]
    assert (exc["type"], exc["message"]) == ("ValueError", "bad rule")


def test_a_rule_returning_a_value_its_outputs_do_not_allow_is_invalid_return() -> None:
    # Probe f: a rule returns a bool where its outputs field is a label type.
    record = one_rule(lambda r: True, "t:state", outputs=t.label("no", "yes"))
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")
    assert "exceptions" not in record.unhashed


async def async_rule(r: Any) -> bool:
    return True


def coroutine_rule(r: Any) -> Any:
    return async_rule(r)


@pytest.fixture
def untyped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interpret's type test of a rule's return fails if reached: an awaitable is refused
    before it."""

    def never(value: object, type_: object) -> object:
        raise AssertionError("the return reached the type test")

    monkeypatch.setattr(_interpret, "typed", never)


@pytest.mark.usefixtures("untyped")
@pytest.mark.parametrize("fn", [async_rule, coroutine_rule], ids=["async", "sync"])
def test_a_rule_returning_a_coroutine_is_invalid_return_and_no_warning_leaks(
    fn: Callable[[Any], Any],
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = one_rule(fn, "t:state", outputs=t.boolean)
        gc.collect()
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")
    assert "exceptions" not in record.unhashed
    assert [str(w.message) for w in caught] == []


@pytest.mark.usefixtures("untyped")
@pytest.mark.parametrize("make_", [NotACoroutine, InheritsAwait], ids=["own", "inherited"])
def test_a_rule_returning_an_awaitable_by_its_type_is_invalid_return_and_not_closed(
    make_: Callable[[], NotACoroutine],
) -> None:
    awaitable = make_()
    record = one_rule(lambda r: awaitable, "t:state")
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")
    assert "exceptions" not in record.unhashed
    assert not awaitable.closed


@pytest.mark.usefixtures("untyped")
@pytest.mark.parametrize("started", [False, True], ids=["unstarted", "started"])
def test_a_rules_generator_coroutine_is_invalid_return_and_closed_at_the_close(
    started: bool,
) -> None:
    ran: list[str] = []
    generator = generator_coroutine(ran, started=started)
    record = one_rule(lambda r: generator, "t:state")
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")
    assert inspect.getgeneratorstate(generator) == inspect.GEN_CLOSED
    assert ran == (["finally"] if started else [])  # held here, so not run by the collector


@pytest.mark.parametrize("make_", [Claims, MetaRaising], ids=["claims", "metaclass"])
def test_a_rule_return_whose_object_claims_or_whose_metaclass_raises_is_no_awaitable(
    make_: Callable[[], Any],
) -> None:
    record = one_rule(lambda r: make_(), "t:state")
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")  # by its outputs
    assert "exceptions" not in record.unhashed


def test_a_rules_coroutine_re_raising_a_kept_stop_on_close_is_invalid_return() -> None:
    record = one_rule(lambda r: started_raising_on_close(stop()), "t:state")
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")
    assert "exceptions" not in record.unhashed
    assert "reads" not in record.unhashed


def test_a_rule_return_whose_awaitable_test_raises_a_kept_stop_is_judged_on() -> None:
    record = one_rule(lambda r: clashing(stop()), "t:state")
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")  # by its outputs
    assert "exceptions" not in record.unhashed


@pytest.mark.parametrize("make_", [clashing, started_raising_on_close], ids=["test", "close"])
def test_an_engine_defect_in_the_awaitable_test_or_the_close_propagates(
    make_: Callable[[BaseException], object],
) -> None:
    defect = _ViewError("an engine defect")
    with pytest.raises(_ViewError) as caught:
        one_rule(lambda r: make_(defect), "t:state")
    assert caught.value is defect


def test_a_rule_return_whose_awaitable_test_raises_is_the_rules_raise() -> None:
    record = one_rule(lambda r: clashing(), "t:state")
    assert entry(record, "d:r")["marker"] == mark("raised", "r")
    exc = record.unhashed["exceptions"]["3"]
    assert (exc["type"], exc["message"]) == ("ValueError", "the type test raised")


def test_a_rules_coroutine_is_closed_when_an_undeclared_read_wins() -> None:
    returned: list[Any] = []

    def fn(r: Any) -> Any:
        with contextlib.suppress(BaseException):
            r.read("t:junk")
        returned.append(async_rule(r))
        return returned[0]

    record = one_rule(fn, "t:state")
    assert entry(record, "d:r")["marker"] == mark("undeclared-read", "r")
    assert inspect.getcoroutinestate(returned[0]) == inspect.CORO_CLOSED
    assert record.unhashed["reads"] == {"3": "t:junk"}


def test_a_rules_coroutine_raising_on_close_keeps_the_outcome_and_its_text() -> None:
    record = one_rule(lambda r: started_raising_on_close("on close"), "t:state")
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")
    exc = record.unhashed["exceptions"]["3"]
    assert (exc["type"], exc["message"]) == ("ValueError", "on close")


def test_a_close_raise_is_kept_beside_the_undeclared_read_that_wins() -> None:
    def fn(r: Any) -> Any:
        with contextlib.suppress(BaseException):
            r.read("t:junk")
        return started_raising_on_close("on close")

    record = one_rule(fn, "t:state")
    assert entry(record, "d:r")["marker"] == mark("undeclared-read", "r")
    assert record.unhashed["exceptions"]["3"]["message"] == "on close"
    assert record.unhashed["reads"] == {"3": "t:junk"}


def test_a_coroutine_whose_close_raises_a_base_exception_propagates_it() -> None:
    async def parked() -> object:
        try:
            await asyncio.sleep(0)
        finally:
            raise KeyboardInterrupt

    def fn(r: Any) -> Any:
        coroutine = parked()
        coroutine.send(None)
        return coroutine

    with pytest.raises(KeyboardInterrupt):
        one_rule(fn, "t:state")


class Truthless:
    def __bool__(self) -> bool:
        raise AssertionError("bool() was called")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("== was called")

    __hash__ = object.__hash__


@pytest.mark.parametrize(
    "value", [{1, 2}, Truthless(), Answer("noul", ("yes", "no"), NOUL), Count.one]
)
def test_a_rule_returning_what_is_not_an_estimate_is_invalid_return(value: object) -> None:
    record = one_rule(lambda r: value, "t:state", outputs=t.integer())
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")


def test_a_rules_returned_marker_is_written_with_whatever_cause_it_carries() -> None:
    record = one_rule(lambda r: Unasked(Cause("refused", ["t:state", "x"])), "t:state")
    assert entry(record, "d:r")["marker"] == mark("refused", "t:state", "x", kind="Unasked")


def tampered_code() -> Unsure:
    m = Unsure(Cause("abstained"))
    object.__setattr__(m.cause, "code", "abstained")
    return m


def tampered_refs() -> Unsure:
    m = Unsure(Cause("abstained"))
    object.__setattr__(m.cause, "refs", ["t:state"])
    return m


class OddUnsure(Unsure):
    pass


@pytest.mark.parametrize(
    "make_marker",
    [
        tampered_code,
        tampered_refs,
        lambda: Unsure(Cause("abstained", ["\ud800"])),
        lambda: OddUnsure(Cause("abstained")),
    ],
)
def test_a_malformed_returned_marker_is_invalid_return(make_marker: Callable[[], object]) -> None:
    returned = make_marker()
    record = one_rule(lambda r: returned, "t:state")
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")


def test_a_rules_record_value_drops_undeclared_fields() -> None:
    record = one_rule(
        lambda r: {"n": 1, "junk": object()}, "t:state", outputs=t.record(n=t.integer())
    )
    assert entry(record, "d:r")["value"] == {"n": 1}


@pytest.mark.parametrize(
    ("read", "named"),
    [
        ("t:state", True),
        ("t:state.subject", True),
        ("t:state.nofield", False),
        ("t:nothing", False),
        ("not a ref", False),
        ('t:state["x"]', False),
    ],
)
def test_an_undeclared_read_names_the_read_only_when_the_design_provides_it(
    read: str, named: bool
) -> None:
    record = one_rule(lambda r: r.read(read))
    refs = ["r", read] if named else ["r"]
    assert entry(record, "d:r")["marker"] == mark("undeclared-read", *refs)
    assert record.unhashed.get("reads", {}) == ({} if named else {"3": read})


def test_an_undeclared_read_of_a_non_string_puts_its_repr_in_reads() -> None:
    record = one_rule(lambda r: r.read(42))
    assert entry(record, "d:r")["marker"] == mark("undeclared-read", "r")
    assert record.unhashed["reads"] == {"3": "42"}


@pytest.mark.parametrize(("path", "named"), [("bar", True), ("nope", False)])
def test_an_undeclared_parameter_is_named_only_when_it_is_a_declared_leaf(
    path: str, named: bool
) -> None:
    record = one_rule(lambda r: r.param(path))
    assert entry(record, "d:r")["marker"] == mark(
        "undeclared-read", *(["r", path] if named else ["r"])
    )


def test_a_declared_parameter_is_read() -> None:
    record = one_rule(lambda r: r.param("bar"), params=["bar"], outputs=t.number())
    assert entry(record, "d:r")["value"] == 0.5


def swallow(*reads: str) -> Callable[[Any], Any]:
    def fn(r: Any) -> Any:
        for read in reads:
            with contextlib.suppress(BaseException):
                r.read(read)
        return True

    return fn


def test_a_caught_undeclared_read_still_decides_and_only_the_first_counts() -> None:
    record = one_rule(swallow("t:junk", "t:state"))
    assert entry(record, "d:r")["marker"] == mark("undeclared-read", "r")
    assert record.unhashed["reads"] == {"3": "t:junk"}


def abstains(r: Any) -> Any:
    return Unsure(Cause("abstained", ["t:state"]))


def asks_later(r: Any) -> Any:
    return Unasked(Cause("round-bound"))


def two_rules(
    fn: Callable[[Any], Any],
    *reads: str,
    first: Callable[[Any], Any] = abstains,
    outputs: str = t.boolean,
) -> Record:
    rules = {"a": rule("a", first, "t:state", outputs=outputs), "b": rule("b", fn, *reads)}
    record, _ = run(make(rules=rules), {})
    return record


def test_require_on_a_marker_takes_its_kind_with_required_naming_the_read() -> None:
    record = two_rules(lambda r: r.require("d:a"), "d:a")
    assert entry(record, "d:b")["marker"] == mark("required", "d:a", kind="Unsure")


def test_require_through_a_field_path_names_the_read_as_given() -> None:
    record = two_rules(
        lambda r: r.require("d:a.n"), "d:a.n", first=asks_later, outputs=t.record(n=t.integer())
    )
    assert entry(record, "d:b")["marker"] == mark("required", "d:a.n", kind="Unasked")


def test_a_caught_require_stop_wins_over_a_later_raise_whose_text_is_kept() -> None:
    def fn(r: Any) -> Any:
        with contextlib.suppress(BaseException):
            r.require("d:a")
        raise RuntimeError("after the stop")

    record = two_rules(fn, "d:a")
    index = record.log.index(entry(record, "d:b"))
    assert entry(record, "d:b")["marker"] == mark("required", "d:a", kind="Unsure")
    assert record.unhashed["exceptions"][str(index)]["message"] == "after the stop"


def test_an_undeclared_read_outranks_an_earlier_require_stop() -> None:
    def fn(r: Any) -> Any:
        with contextlib.suppress(BaseException):
            r.require("d:a")
        return r.read("t:state")

    record = two_rules(fn, "d:a")
    assert entry(record, "d:b")["marker"] == mark("undeclared-read", "b", "t:state")


def test_a_rule_reading_a_marker_is_still_called() -> None:
    record = two_rules(lambda r: True, "d:a")
    assert entry(record, "d:b")["value"] is True


def test_a_helper_call_is_listed_in_derivation_even_when_the_rule_then_raises() -> None:
    def fn(r: Any) -> Any:
        pick_with_margin(r, "q:spam", 0.5, 0.1)
        raise RuntimeError("after the helper")

    loaded = make(asking(), {"r": rule("r", fn, "q:spam")})
    record, _ = run(loaded, {"ask": ok()})
    assert entry(record, "d:r")["marker"] == mark("raised", "r")
    assert entry(record, "d:r")["derivation"] == [
        {"helper": "pick_with_margin", "refs": ["q:spam"], "rested_on": ["q:spam"]}
    ]


class Interrupt(BaseException):
    pass


def test_a_base_exception_from_a_rule_propagates_with_no_item_for_it() -> None:
    def fn(r: Any) -> Any:
        raise Interrupt

    loaded = make(rules={"r": rule("r", fn, "t:state")})
    run_ = settings()
    ledger = open_ledger(loaded, STATE, run_)
    with pytest.raises(Interrupt):
        asyncio.run(interpret_on(ledger, loaded, STATE, {"default": Scripted({})}, run_))
    assert len(ledger.close().log) == 3


# ---------------------------------------------------------------------------------------------
# Need markers: mixed needs and absence markers


def needing(*needs: str) -> Source:
    return Source({"t:out": t.string}, needs=needs)


def test_an_unavailable_need_gives_need_unavailable_naming_it_with_no_call() -> None:
    sources = {**asking(), "next": needing("t:body")}
    rules = {"after": rule("after", lambda r: True, "t:out")}
    record, binding = run(make(sources, rules), {"ask": raising(RuntimeError("x"))})
    assert entry(record, "t:out")["marker"] == mark("need-unavailable", "t:body")
    assert entry(record, "t:out")["call"] is None
    assert [b[0].id for b in binding.batches] == ["ask"]
    # Round 2 asked nothing: no call item and cost 0, and the rule behind it fired.
    assert [i for i in body(record) if i["item"] == "round"][-1] == {
        "item": "round",
        "round": 2,
        "cost": 0,
        "deadline": False,
    }
    assert entry(record, "d:after")["value"] is True


def test_need_unavailable_outranks_need_undetermined_and_names_only_its_own_kind() -> None:
    sources = {
        **asking(),
        "look": look(),
        "next": needing("d:unsure", "t:body", "t:blob", "t:profile.name"),
    }
    rules = {"unsure": rule("unsure", abstains, "t:state")}
    replies = {"ask": raising(RuntimeError("x")), "look": raising(RuntimeError("y"))}
    record, _ = run(make(sources, rules), replies)
    assert entry(record, "t:out")["marker"] == mark(
        "need-unavailable", "t:blob", "t:body", "t:profile.name"
    )


def test_an_undetermined_need_gives_unasked_need_undetermined() -> None:
    sources = {"next": needing("d:unsure", "d:later")}
    rules = {
        "unsure": rule("unsure", abstains, "t:state"),
        "later": rule("later", asks_later, "t:state"),
    }
    record, _ = run(make(sources, rules), {})
    assert entry(record, "t:out")["marker"] == mark(
        "need-undetermined", "d:later", "d:unsure", kind="Unasked"
    )


# ---------------------------------------------------------------------------------------------
# The round bound and the drain


def chain() -> LoadedDesign:
    sources = {
        "s1": Source({"t:a": t.string}),
        "s2": Source({"t:b": t.string}, needs=["d:r1"]),
        "s3": Source({"t:c": t.string, "t:d": t.string}, needs=["d:r2"]),
    }
    rules = {
        "r1": rule("r1", lambda r: True, "t:a"),
        "r2": rule("r2", lambda r: r.require("t:b"), "t:b", outputs=t.string),
        "r3": rule("r3", lambda r: r.require("t:c"), "t:c", outputs=t.string),
    }
    return make(sources, rules)


CHAIN = {
    "s1": returns({"t:a": "a"}),
    "s2": returns({"t:b": "b"}),
    "s3": returns({"t:c": "c", "t:d": "d"}),
}


def test_at_the_bound_pending_sources_are_unasked_and_the_drain_fires_the_rules_behind_them() -> (
    None
):
    record, binding = run(chain(), CHAIN, round_bound=1)
    assert [b[0].id for b in binding.batches] == ["s1"]
    assert kinds(record) == [
        ("call", 1),
        ("entry", "t:a"),
        ("entry", "d:r1"),
        ("entry", "t:b"),
        ("entry", "t:c"),
        ("entry", "t:d"),
        ("entry", "d:r2"),
        ("entry", "d:r3"),
        ("round", 1),
    ]
    for ref in ("t:b", "t:c", "t:d"):
        assert entry(record, ref)["marker"] == mark("round-bound", kind="Unasked")
        assert entry(record, ref)["round"] == 1
    assert entry(record, "d:r2")["marker"] == mark("required", "t:b", kind="Unasked")
    assert entry(record, "d:r3")["round"] == 1
    assert entry(record, "d:r3")["derivation"] == []


def test_a_bound_past_the_first_round_marks_only_the_sources_scheduled_past_it() -> None:
    record, _ = run(chain(), CHAIN, round_bound=2)
    assert entry(record, "t:b")["value"] == "b"
    assert entry(record, "t:c")["marker"] == mark("round-bound", kind="Unasked")
    assert [i["round"] for i in body(record) if i["item"] == "round"] == [1, 2]


def test_the_bound_reached_with_no_source_pending_is_the_fixpoint() -> None:
    record, _ = run(chain(), CHAIN, round_bound=3)
    assert entry(record, "t:d")["value"] == "d"
    assert all("marker" not in i for i in body(record) if i["item"] == "entry")


def test_the_round_cost_charges_failed_and_timed_out_calls() -> None:
    sources = {
        "alpha": Source({"t:a": t.string}, cost=2),
        "zeta": Source({"t:z": t.string}, cost=3.5, timeout=0.02),
        "free": Source({"t:f": t.string}),
    }
    replies = {
        "alpha": raising(RuntimeError("x")),
        "zeta": sleeping(5, None),
        "free": returns({"t:f": "f"}),
    }
    record, _ = run(make(sources), replies)
    assert body(record)[-1] == {"item": "round", "round": 1, "cost": 5.5, "deadline": False}


# ---------------------------------------------------------------------------------------------
# Cancellation


def cancelling_turn(replies: Mapping[str, object]) -> list[dict[str, Any]]:
    sources = {"alpha": Source({"t:a": t.string}), "zeta": Source({"t:z": t.string})}
    loaded = make(sources, {"zero": rule("zero", lambda r: True, "t:state")})
    run_ = settings()
    ledger = open_ledger(loaded, STATE, run_)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(interpret_on(ledger, loaded, STATE, {"default": Scripted(replies)}, run_))
    return ledger.close().log[3:]


def test_a_task_that_cancels_itself_is_re_raised_before_its_rounds_first_append() -> None:
    # Probe g: a CancelledError inside a fetch.
    log = cancelling_turn(
        {"alpha": raising(asyncio.CancelledError()), "zeta": returns({"t:z": "z"})}
    )
    assert [(i["item"], i.get("ref")) for i in log] == [("entry", "d:zero")]


def test_a_cancellation_of_interpret_cancels_the_frontier_and_appends_nothing_of_its_round() -> (
    None
):
    sources = {"alpha": Source({"t:a": t.string}), "zeta": Source({"t:z": t.string})}
    loaded = make(sources)
    run_ = settings()
    ledger = open_ledger(loaded, STATE, run_)
    cancelled: list[str] = []
    started = asyncio.Event()

    async def wait(batch: Batch) -> object:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(batch[0].id)
            raise
        return None

    async def main() -> None:
        task = asyncio.ensure_future(
            interpret_on(
                ledger, loaded, STATE, {"default": Scripted({"alpha": wait, "zeta": wait})}, run_
            )
        )
        await started.wait()
        await asyncio.sleep(0)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main())
    assert sorted(cancelled) == ["alpha", "zeta"]
    assert len(ledger.close().log) == 3


# ---------------------------------------------------------------------------------------------
# Latency does not reach the record


def pairs(n: int, **changes: Any) -> dict[str, Source]:
    """``n`` pairs: ``s{i}`` asks ``q:k{i}`` and ``t{i}`` provides ``t:v{i}``, both through the
    oracle ``o{i}``, so each pair is one call answered under ``s{i}``."""
    sources: dict[str, Source] = {}
    for i in range(n):
        sources[f"s{i}"] = Source({f"q:k{i}": Question("noul", Text(f"Is {i}?"))}, oracle=f"o{i}")
        sources[f"t{i}"] = Source({f"t:v{i}": t.string}, oracle=f"o{i}", **changes)
    return sources


def valued(batch: Batch) -> dict[str, Any]:
    """Each ``t:`` ref of the batch ``"v"`` and each question ``NOUL``."""
    values: dict[str, Any] = {}
    for call in batch:
        values.update(dict.fromkeys(call.refs, "v"))
        values.update(dict.fromkeys(call.questions, NOUL))
    return values


def test_shuffled_latencies_give_identical_bytes() -> None:
    sources = pairs(5)
    sources["late"] = Source({"t:late": t.string}, needs=["d:sum"])
    sources["later"] = Source({"t:later": t.string}, needs=["t:late", "t:v0"])
    rules = {"sum": rule("sum", lambda r: True, *(f"q:k{i}" for i in range(5)))}
    loaded = make(sources, rules)

    hashes: set[str] = set()
    orders: set[tuple[str, ...]] = set()
    for seed in range(6):
        rng = random.Random(seed)  # one per seed, shared by every call, so each call differs
        finished: list[str] = []

        async def reply(
            batch: Batch, rng: random.Random = rng, finished: list[str] = finished
        ) -> object:
            await asyncio.sleep(rng.random() / 100)
            finished.append(batch[0].id)
            if batch[0].id == "s3":
                raise RuntimeError("one fails")
            return returns(valued(batch))

        scripted_ = Scripted(dict.fromkeys(sources, reply))
        binds = {"default": scripted_} | {f"o{i}": scripted_ for i in range(5)}
        record = turn(loaded, binds, STATE)
        hashes.add(record.record_hash)
        orders.add(tuple(finished))
    assert len(orders) > 1
    assert len(hashes) == 1


# ---------------------------------------------------------------------------------------------
# The write path's other types, the guards and the engine's own defects


class Twin(str):
    """A string whose hash is fixed, so two of one string data can share a dict."""

    __slots__ = ()

    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return other is self


def lists(**changes: Any) -> Source:
    fields: dict[str, Any] = {
        "provides": {
            "t:tags": t.list_(t.string),
            "t:scores": t.mapping(t.number()),
            "t:opt": t.nullable(t.record(city=t.string)),
        }
    }
    fields.update(changes)
    return Source(**fields)


LISTS = {"t:tags": ("a", "b"), "t:scores": {"x": 1, "y": 0.5}, "t:opt": None}


def test_a_list_and_a_mapping_are_written_member_by_member() -> None:
    record, _ = run(make({"lists": lists()}), {"lists": returns(dict(LISTS))})
    assert entry(record, "t:tags")["value"] == ["a", "b"]
    assert entry(record, "t:scores")["value"] == {"x": 1, "y": 0.5}


@pytest.mark.parametrize(
    ("ref", "value"),
    [
        ("t:tags", "ab"),
        ("t:tags", ["a", 1]),
        ("t:scores", ["x"]),
        ("t:scores", {"x": "1"}),
        ("t:scores", {1: 1}),
        ("t:scores", {Twin("x"): 1, Twin("x"): 2}),
        ("t:opt", {Twin("city"): "a", Twin("city"): "b"}),
        ("t:opt", ["city"]),
    ],
)
def test_a_list_or_mapping_member_that_does_not_conform_is_invalid_return(
    ref: str, value: object
) -> None:
    record, _ = run(make({"lists": lists()}), {"lists": returns({**LISTS, ref: value})})
    assert entry(record, ref)["marker"] == mark("invalid-return", ref)


def test_a_null_on_a_needs_path_passes_null() -> None:
    sources = {"lists": lists(), "next": needing("t:opt.city", "t:tags")}
    _, binding = run(
        make(sources), {"lists": returns(dict(LISTS)), "next": returns({"t:out": "o"})}
    )
    (call,) = [b[0] for b in binding.batches if b[0].id == "next"]
    assert call.needs == {"t:opt.city": None, "t:tags": ["a", "b"]}


def test_a_tuple_in_the_live_state_is_passed_as_a_fresh_tuple() -> None:
    loaded = load(
        minimal(
            state=t.record(tags=t.list_(t.list_(t.string))),
            sources={"s": Source({"t:out": t.string}, needs=["t:state"])},
        )
    )
    binding = Scripted({"s": returns({"t:out": "o"})})
    live = {"tags": (["a"],)}
    turn(loaded, {"default": binding}, {"tags": [["a"]]}, live=live)
    passed = binding.batches[0][0].needs["t:state"]
    assert passed == {"tags": (["a"],)}
    assert passed["tags"][0] is not live["tags"][0]


def test_a_duplicate_usage_key_is_recorded_null() -> None:
    usage = {Twin("in"): 1, Twin("in"): 2}
    record, _ = run(make(asking()), {"ask": ok(usage=usage)})
    assert record.unhashed["calls"]["3"]["usage"] is None
    assert "usage" in record.unhashed["calls"]["3"]["invalid"]


def test_a_marker_whose_cause_was_replaced_is_invalid_return() -> None:
    returned = Unsure(Cause("abstained"))
    object.__setattr__(returned, "cause", "abstained")
    record = one_rule(lambda r: returned, "t:state")
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")


class Unprintable(Exception):
    def __str__(self) -> str:
        raise RuntimeError("no text")


def deep() -> list[Any]:
    value: list[Any] = []
    for _ in range(70):
        value = [value]
    return value


def raises_it(exc: Exception) -> dict[str, str]:
    def fn(r: Any) -> Any:
        raise exc

    return one_rule(fn, "t:state").unhashed["exceptions"]["3"]


def test_an_exception_whose_str_raises_has_an_unrepresentable_message() -> None:
    text = raises_it(Unprintable())
    assert text["message"] == "<unrepresentable>"
    assert text["type"] == "Unprintable"
    assert "Unprintable" in text["traceback"]


def test_an_exception_whose_arguments_nest_too_deep_has_no_text() -> None:
    text = raises_it(ValueError(deep()))
    assert text == {
        "type": "ValueError",
        "message": "<unrepresentable>",
        "traceback": "<unrepresentable>",
    }


def test_a_view_defect_propagates_and_is_never_the_rules_failure() -> None:
    from jurat._read_view import _ViewError, open_call

    def fn(r: Any) -> Any:
        open_call(r, "helper", ["t:unread"])  # lists a ref it never read: an engine defect

    loaded = make(rules={"r": rule("r", fn, "t:state")})
    run_ = settings()
    ledger = open_ledger(loaded, STATE, run_)
    with pytest.raises(_ViewError):
        asyncio.run(interpret_on(ledger, loaded, STATE, {"default": Scripted({})}, run_))


def test_the_drain_refuses_to_spin_on_a_returned_node_never_done() -> None:
    from jurat._interpret import _InterpretError, _Limits, _Turn

    loaded = make({"ask": ask()})
    ledger = open_ledger(loaded, STATE, settings())
    walk = _Turn(ledger, loaded, STATE, {}, 8, 60, frozenset(), _Limits(None, None, None))
    assert walk.sorter.get_ready() == {"ask"}  # returned and never done
    with pytest.raises(_InterpretError, match=r"^the drain found no node ready"):
        walk.drain(1)


# ---------------------------------------------------------------------------------------------
# Caller code outside the protected region (review of step 5)


class BadRepr:
    def __repr__(self) -> str:
        raise RuntimeError("no repr")


class Masked:
    """An object whose ``__class__`` is caller code that raises, as ``isinstance`` reads it."""

    @property
    def __class__(self) -> type:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        raise RuntimeError("caller code")


# Built in the test, since pytest's parametrize ids read an argument's class.
@pytest.mark.parametrize(
    "value",
    [
        lambda: BadRepr(),
        lambda: {"yes": BadRepr(), "no": 0.1},
        lambda: {"yes": Masked(), "no": 0.1},
    ],
)
def test_a_q_value_or_probability_whose_repr_or_class_raises_is_invalid_answer(
    value: Callable[[], object],
) -> None:
    record, _ = run(make(asking()), {"ask": with_values(q_spam=value())})
    assert entry(record, "q:spam")["marker"] == mark("invalid-answer", "q:spam")


@pytest.mark.parametrize("key", [Masked, BadRepr])
def test_a_q_values_label_key_whose_class_or_repr_raises_is_invalid_answer(
    key: Callable[[], object],
) -> None:
    values = {key(): 0.9, "no": 0.1}
    record, _ = run(make(asking()), {"ask": with_values(q_spam=values)})
    assert entry(record, "q:spam")["marker"] == mark("invalid-answer", "q:spam")


def test_a_values_key_whose_class_raises_is_unexpected() -> None:
    values: dict[Any, Any] = {**OK, Masked(): 1}
    record, _ = run(make(asking()), {"ask": returns(values)})
    failed(record, "unexpected-ref")
    assert len(record.unhashed["calls"]["3"]["unexpected"]) == 1


def test_a_t_value_whose_class_raises_is_invalid_return() -> None:
    record, _ = run(make(asking()), {"ask": with_values(t_body=Masked())})
    assert entry(record, "t:body")["marker"] == mark("invalid-return", "t:body")


def test_a_rule_returning_an_object_whose_class_raises_is_invalid_return() -> None:
    returned = Masked()
    record = one_rule(lambda r: returned, "t:state")
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")


def test_a_bindings_base_exception_propagates_bare_with_no_item_of_its_round() -> None:
    sources = {"alpha": Source({"t:a": t.string}), "zeta": Source({"t:z": t.string})}
    loaded = make(sources)
    run_ = settings()
    ledger = open_ledger(loaded, STATE, run_)
    replies = {"alpha": returns({"t:a": "a"}), "zeta": raising(Interrupt())}
    with pytest.raises(Interrupt):
        asyncio.run(interpret_on(ledger, loaded, STATE, {"default": Scripted(replies)}, run_))
    assert len(ledger.close().log) == 3


def test_a_late_call_results_usage_and_attempts_are_null_and_nothing_else_is_kept() -> None:
    late = ok(usage={"in": 3}, attempts=2, snapshot=5)
    record, _ = run(make(asking(timeout=0.02)), {"ask": ignoring_cancel(5, late)})
    failed(record, "timed-out")
    call = record.unhashed["calls"]["3"]
    assert (call["usage"], call["attempts"]) == (None, None)
    assert "invalid" not in call
    assert "unexpected" not in call


@pytest.mark.parametrize("usage", [{Mood.calm: 1}, {"in": 2**60}, {"in": -(2**60)}])
def test_a_usage_outside_the_value_domain_is_recorded_null(usage: object) -> None:
    record, _ = run(make(asking()), {"ask": ok(usage=usage)})
    assert record.unhashed["calls"]["3"]["usage"] is None
    shown = "<builtins.dict>" if Mood.calm in usage else repr(usage)  # type: ignore[operator]
    assert record.unhashed["calls"]["3"]["invalid"] == {"usage": shown}


def test_a_failed_whose_cause_is_not_exactly_a_cause_is_invalid_return() -> None:
    outcome = Failed(Cause("raised"))
    object.__setattr__(outcome, "cause", CauseCode.raised)
    record, _ = run(make(asking()), {"ask": outcome})
    failed(record, "invalid-return")


def test_a_carried_provider_is_refused_at_entry_in_v0_0_1() -> None:
    from jurat._interpret import _InterpretError, interpret

    loaded = make(asking())
    run_ = settings()
    ledger = open_ledger(loaded, STATE, run_)
    with pytest.raises(_InterpretError, match=r"^carried providers arrive"):
        asyncio.run(
            interpret(
                ledger,
                loaded,
                STATE,
                {},
                round_bound=8,
                timeout=60,
                budget=None,
                deadline=None,
                clock=0.0,
                cut=None,
                carried=frozenset({"ask"}),
            )
        )
    assert len(ledger.close().log) == 3


def test_interpret_ends_with_every_node_done(monkeypatch: pytest.MonkeyPatch) -> None:
    from jurat._interpret import _InterpretError
    from jurat._sorter import TurnSorter

    def always(self: object) -> bool:
        return True

    monkeypatch.setattr(TurnSorter, "is_active", always)
    with pytest.raises(_InterpretError, match=r"^interpret ended with a node not done$"):
        one_rule(lambda r: True, "t:state")


def test_a_score_question_is_passed_its_levels_and_recorded_with_them() -> None:
    levels = ("low", "mid", "high")
    source = Source({"q:urgency": Question("score", Text("How urgent?"), labels=levels)})
    probabilities = {"low": 0.25, "mid": 0.25, "high": 0.5}
    record, binding = run(make({"s": source}), {"s": returns({"q:urgency": probabilities})})
    assert binding.batches[0][0].questions["q:urgency"] == RenderedQuestion(
        AnswerKind.score, "How urgent?", levels, descriptions=(None,) * len(levels)
    )
    assert record.log[3]["questions"]["q:urgency"] == {
        "identity": identity("How urgent?", "score", list(levels)),
        "kind": "score",
        "labels": list(levels),
    }
    assert entry(record, "q:urgency")["value"] == answer("score", list(levels), probabilities)


def test_two_values_keys_of_one_string_data_fail_the_call_ahead_of_an_unexpected_key() -> None:
    values: dict[Any, Any] = {**OK, Twin("t:body"): "again", "t:nope": 1}
    record, _ = run(make(asking()), {"ask": returns(values)})
    failed(record, "invalid-return")
    assert "unexpected" not in record.unhashed["calls"]["3"]


def test_a_rounds_cost_is_the_exact_sum_of_its_calls_tags_rounded_once() -> None:
    # In float order 0.1 + 0.2 + 0.3 is 0.6000000000000001; the exact sum rounds to 0.6.
    sources = {
        "a": Source({"t:a": t.string}, cost=0.1),
        "b": Source({"t:b": t.string}, cost=0.2),
        "c": Source({"t:c": t.string}, cost=0.3),
    }
    replies = {s: returns({f"t:{s}": "v"}) for s in sources}
    record, _ = run(make(sources), replies)
    assert body(record)[-1] == {"item": "round", "round": 1, "cost": 0.6, "deadline": False}


# ---------------------------------------------------------------------------------------------
# An int under number is written as a float

NUMBERS = t.record(
    x=t.number(),
    i=t.number(0, 10),
    k=t.integer(),
    items=t.list_(t.number()),
    by=t.mapping(t.number()),
    maybe=t.nullable(t.number()),
)


def test_an_int_under_number_is_a_float_in_the_stored_copy_and_on_the_board() -> None:
    value = {"x": 1, "i": 4, "k": 2, "items": [1, 2.5], "by": {"a": 3}, "maybe": 7}
    seen: dict[str, object] = {}

    def observer(index: int, item: dict[str, Any]) -> None:
        if item["item"] == "entry" and item["ref"] in ("d:n", "d:rec"):
            seen[item["ref"]] = item["value"]

    loaded = make(
        rules={
            "n": rule("n", lambda r: 3, outputs=t.number()),
            "rec": rule("rec", lambda r: value, outputs=NUMBERS),
        }
    )
    run_settings = settings()
    ledger = open_ledger(loaded, STATE, run_settings, observer)
    board = asyncio.run(interpret_on(ledger, loaded, STATE, {}, run_settings))
    expected = {"x": 1.0, "i": 4.0, "k": 2, "items": [1.0, 2.5], "by": {"a": 3.0}, "maybe": 7.0}
    for found in (seen["d:rec"], board["d:rec"]):
        assert found == expected
        found = cast("dict[str, Any]", found)
        assert [type(found[n]) for n in ("x", "i", "k", "maybe")] == [float, float, int, float]
        assert [type(v) for v in found["items"]] == [float, float]
        assert type(found["by"]["a"]) is float
    assert type(seen["d:n"]) is float
    assert type(board["d:n"]) is float
    record = ledger.close()
    assert b'"value":3}' in record.document  # the bytes are the same


def test_a_sources_int_under_number_reaches_a_rule_as_a_float() -> None:
    got: list[object] = []

    def reader(r: Any) -> bool:
        got.append(r.read("t:amount"))
        return True

    loaded = make(
        {"look": Source({"t:amount": t.number(0, 100)})},
        {"reader": rule("reader", reader, "t:amount")},
    )
    record, _ = run(loaded, {"look": returns({"t:amount": 5})})
    assert [type(v) for v in got] == [float]
    assert entry(record, "t:amount")["value"] == 5


# ---------------------------------------------------------------------------------------------
# A kept view, and the order of unhashed writes


def test_a_rule_reading_another_rules_kept_view_raises_with_its_text_kept() -> None:
    held: dict[str, Any] = {}

    def keep(r: Any) -> bool:
        held["r"] = r
        return True

    def reach(r: Any) -> bool:
        r.read("d:a")
        return bool(held["r"].read("t:state"))

    loaded = make(rules={"a": rule("a", keep), "b": rule("b", reach, "d:a")})
    record, _ = run(loaded, {})
    assert entry(record, "d:b")["marker"] == mark("raised", "b")
    index = str(record.log.index(entry(record, "d:b")))
    assert record.unhashed["exceptions"][index]["type"] == "ViewClosed"
    assert held["r"].stops == ()


def test_each_items_unhashed_members_come_after_its_append_and_before_the_next() -> None:
    def raises(r: Any) -> bool:
        raise ValueError("rule")

    def nosy(r: Any) -> bool:
        return r.read("d:nothing")

    seen: dict[int, set[str]] = {}
    box: list[Any] = []

    def observer(index: int, item: dict[str, Any]) -> None:
        if box:
            unhashed = box[0]._unhashed
            members = ("calls", "exceptions", "reads")
            seen[index] = {f"{m}/{k}" for m in members for k in unhashed.get(m, {})}

    loaded = make(
        {**asking(), "look": look()},
        {"bad": rule("bad", raises, "t:body"), "nosy": rule("nosy", nosy, "t:body")},
    )
    run_settings = settings()
    ledger = open_ledger(loaded, STATE, run_settings, observer)
    box.append(ledger)
    binding = Scripted({"ask": raising(RuntimeError("binding")), "look": returns(LOOKED)})
    binds = {"default": binding, "mail": binding}
    asyncio.run(interpret_on(ledger, loaded, STATE, binds, run_settings))
    unhashed = ledger.close().unhashed
    keys = [(m, int(k)) for m in ("calls", "exceptions", "reads") for k in unhashed.get(m, {})]
    assert {m for m, _ in keys} == {"calls", "exceptions", "reads"}
    for member, index in keys:
        assert f"{member}/{index}" not in seen[index]
        assert f"{member}/{index}" in seen[index + 1]


# ---------------------------------------------------------------------------------------------
# Readiness: the needs, the source requirement, then the function `when`

GATE_STATE_T = t.record(
    subject=t.string,
    sender=t.record(address=t.string),
    reply=t.nullable(t.record(text=t.string)),
)
GATE_STATE: dict[str, Any] = {"subject": "hello", "sender": {"address": "a@b"}, "reply": None}


def gated(
    when: Callable[[Any], Any] | None = None,
    requirement: SourceRequirement | None = None,
    *needs: str,
    tag: str = "",
) -> Source:
    """A source providing ``t:alt`` and ``t:out``, each with ``tag`` appended, gated by
    ``when`` and ``requirement``."""
    return Source(
        {f"t:out{tag}": t.string, f"t:alt{tag}": t.string},
        needs=needs,
        when=None if when is None else When(when, version="1"),
        requirement=requirement,
    )


def label(rid: str, value: object, *reads: str) -> Rule:
    """A rule over the state returning ``value``, of type ``label{"no", "yes"}``."""
    return rule(rid, lambda r: value, *(reads or ("t:state",)), outputs=t.label("no", "yes"))


def gating(
    sources: dict[str, Source],
    rules: dict[str, Rule] | None = None,
    replies: Mapping[str, object] | None = None,
    **changes: Any,
) -> tuple[Record, Scripted]:
    loaded = load(
        minimal(state=GATE_STATE_T, sources=sources, rules=rules or {}, **changes),
    )
    binding = Scripted(dict.fromkeys(sources, answers) | dict(replies or {}))
    return turn(loaded, {"default": binding, "mail": binding}, GATE_STATE), binding


async def answers(batch: Batch) -> object:
    """Every ref the call asks, as ``"o"``."""
    return returns(dict.fromkeys(batch[0].refs, "o"))


def asked(binding: Scripted) -> list[str]:
    return [c.id for batch in binding.batches for c in batch]


def index_of(record: Record, ref: str) -> str:
    return str(record.log.index(entry(record, ref)))


def marked(record: Record, marker: dict[str, Any], *refs: str) -> None:
    for ref in refs or ("t:alt", "t:out"):
        assert entry(record, ref)["marker"] == marker
        assert entry(record, ref)["call"] is None


def raises(message: str) -> Callable[[Any], Any]:
    def fn(r: Any) -> Any:
        raise RuntimeError(message)

    return fn


def test_a_when_that_is_true_lets_its_source_be_asked() -> None:
    record, binding = gating({"s": gated(lambda r: True)})
    assert asked(binding) == ["s"]
    assert entry(record, "t:out")["value"] == "o"


def test_a_when_that_raises_is_when_failed_naming_its_source_its_text_under_its_lowest_ref() -> (
    None
):
    record, binding = gating({"s": gated(raises("no gate"))})
    marked(record, mark("when-failed", "s"))
    assert asked(binding) == []
    lowest = index_of(record, "t:alt")
    assert set(record.unhashed["exceptions"]) == {lowest}
    text = record.unhashed["exceptions"][lowest]
    assert (text["type"], text["message"]) == ("RuntimeError", "no gate")
    assert "derivation" not in entry(record, "t:alt")


class BadTruth:
    def __bool__(self) -> bool:
        raise ValueError("no truth")


class NotABool:
    def __bool__(self) -> Any:
        return "yes"


@pytest.mark.parametrize(
    ("returned", "error"),
    [
        (Unsure(Cause("abstained", [])), "TypeError"),
        (BadTruth(), "ValueError"),
        (NotABool(), "TypeError"),
    ],
)
def test_a_when_whose_truth_test_raises_is_when_failed_with_the_text_kept(
    returned: object, error: str
) -> None:
    record, binding = gating({"s": gated(lambda r: returned)})
    marked(record, mark("when-failed", "s"))
    assert asked(binding) == []
    assert record.unhashed["exceptions"][index_of(record, "t:alt")]["type"] == error


def test_a_truth_test_that_reads_the_closed_view_is_when_failed() -> None:
    class Late:
        def __init__(self, r: Any) -> None:
            self.r = r

        def __bool__(self) -> bool:
            return bool(self.r.read("t:state"))

    record, _ = gating({"s": gated(Late)})
    marked(record, mark("when-failed", "s"))
    assert record.unhashed["exceptions"][index_of(record, "t:alt")]["type"] == "ViewClosed"


@pytest.mark.parametrize("returned", ["yes", 1, [0], {"a": 1}])
def test_a_truthy_value_that_is_not_a_bool_makes_its_source_ready(returned: object) -> None:
    _, binding = gating({"s": gated(lambda r: returned)})
    assert asked(binding) == ["s"]


@pytest.mark.parametrize("returned", [False, None, 0, "", []])
def test_a_when_that_is_false_is_unasked_when_false_naming_its_source(returned: object) -> None:
    record, binding = gating({"s": gated(lambda r: returned)})
    marked(record, mark("when-false", "s", kind="Unasked"))
    assert asked(binding) == []
    assert "exceptions" not in record.unhashed


def test_a_when_reading_a_provided_ref_outside_its_needs_names_the_read() -> None:
    sources = {"s": gated(lambda r: r.read("d:flag"), None, "t:state")}
    record, _ = gating(sources, {"flag": label("flag", "yes")})
    marked(record, mark("undeclared-read", "s", "d:flag"))
    assert "reads" not in record.unhashed


@pytest.mark.parametrize("read", ["t:state.nofield", "t:state.subject.first", "t:none", "x"])
def test_a_when_reading_a_name_the_design_does_not_hold_names_its_source_alone(
    read: str,
) -> None:
    record, _ = gating({"s": gated(lambda r: r.read(read))})
    marked(record, mark("undeclared-read", "s"))
    assert record.unhashed["reads"] == {index_of(record, "t:alt"): read}


DEEP_BRACKET = "t:state[" + "[" * 100_000


def test_a_when_reading_a_ref_nesting_past_the_recursion_limit_names_its_source_alone() -> None:
    record, _ = gating({"s": gated(lambda r: r.read(DEEP_BRACKET))})
    marked(record, mark("undeclared-read", "s"))
    assert record.unhashed["reads"] == {index_of(record, "t:alt"): DEEP_BRACKET}


def test_a_rule_reading_a_ref_nesting_past_the_recursion_limit_names_the_rule_alone() -> None:
    record = one_rule(lambda r: r.read(DEEP_BRACKET), "t:state")
    assert entry(record, "d:r")["marker"] == mark("undeclared-read", "r")
    assert record.unhashed["reads"] == {"3": DEEP_BRACKET}


def param_nope(r: Any) -> Any:
    return r.param("nope")


def test_a_whens_param_is_an_undeclared_read_named_when_it_is_a_declared_leaf() -> None:
    sources = {"s": gated(lambda r: r.param("bar"))}
    sources["u"] = Source({"t:u": t.string}, when=When(param_nope, version="1"))
    record, _ = gating(sources, parameters={"bar": Parameter(t.number(), 0.5)})
    marked(record, mark("undeclared-read", "s", "bar"))
    marked(record, mark("undeclared-read", "u"), "t:u")
    assert record.unhashed["reads"] == {index_of(record, "t:u"): "nope"}


def test_a_caught_stop_still_decides_and_a_later_raise_keeps_its_text() -> None:
    def catches(r: Any) -> Any:
        with contextlib.suppress(BaseException):
            r.read("t:state.nofield")
        raise RuntimeError("after the stop")

    sources = {"s": gated(catches), "u": gated(swallow("t:state.nofield"), tag="_u")}
    record, binding = gating(sources)
    marked(record, mark("undeclared-read", "s"))
    marked(record, mark("undeclared-read", "u"), "t:alt_u", "t:out_u")
    lowest = index_of(record, "t:alt")
    assert record.unhashed["exceptions"][lowest]["message"] == "after the stop"
    assert record.unhashed["reads"][lowest] == "t:state.nofield"
    assert asked(binding) == []


def test_a_when_reads_a_state_field_path_and_a_null_on_it_reads_null() -> None:
    seen: list[object] = []

    def reads(r: Any) -> bool:
        seen.extend([r.read("t:state.sender.address"), r.require("t:state.reply.text")])
        return True

    record, binding = gating({"s": gated(reads)})
    assert seen == ["a@b", None]
    assert asked(binding) == ["s"]
    assert "reads" not in record.unhashed


def test_a_when_reads_its_declared_needs_and_the_state_whole() -> None:
    seen: list[object] = []

    def reads(r: Any) -> bool:
        seen.extend([r.read("d:flag"), r.read("t:state.subject"), r.read("t:state")])
        return True

    sources = {"s": gated(reads, None, "d:flag", "t:state.subject")}
    _, binding = gating(sources, {"flag": label("flag", "yes")})
    assert seen == ["yes", "hello", GATE_STATE]
    assert binding.batches[0][0].needs == {"d:flag": "yes", "t:state.subject": "hello"}


def test_a_helper_on_a_whens_view_is_when_failed_and_caught_the_when_goes_on() -> None:
    def uses(r: Any) -> Any:
        return pick_with_margin(r, "q:spam", 0.5, 0.1)

    def catches(r: Any) -> bool:
        with contextlib.suppress(Exception):
            pick_with_margin(r, "q:spam", 0.5, 0.1)
        return True

    sources = {
        **asking(needs=[]),
        "s": gated(uses, None, "q:spam"),
        "u": gated(catches, None, "q:spam", tag="_u"),
    }
    record, binding = gating(sources, replies={"ask": ok()})
    marked(record, mark("when-failed", "s"))
    assert "pick_with_margin" in record.unhashed["exceptions"][index_of(record, "t:alt")]["message"]
    assert asked(binding) == ["ask", "body", "u"]


def on_flag(*values: object) -> SourceRequirement:
    return SourceRequirement("d:flag", list(values))


def test_a_source_requirement_that_holds_lets_its_source_be_asked() -> None:
    _, binding = gating({"s": gated(None, on_flag("no", "yes"))}, {"flag": label("flag", "yes")})
    assert asked(binding) == ["s"]
    # The ref is not a need the source declared, so the binding is not passed it.
    assert binding.batches[0][0].needs == {}


def test_a_source_requirement_on_another_value_is_when_false_naming_its_ref() -> None:
    record, binding = gating({"s": gated(None, on_flag("no"))}, {"flag": label("flag", "yes")})
    marked(record, mark("when-false", "d:flag", kind="Unasked"))
    assert asked(binding) == []


def gone(r: Any) -> Any:
    raise RuntimeError("gone")


@pytest.mark.parametrize(
    ("flag", "marker"),
    [
        (rule("flag", gone, "t:state", outputs=t.label("no", "yes")), "need-unavailable"),
        (rule("flag", abstains, "t:state", outputs=t.label("no", "yes")), "need-undetermined"),
        (rule("flag", asks_later, "t:state", outputs=t.label("no", "yes")), "need-undetermined"),
    ],
)
def test_a_source_requirement_on_a_marker_is_a_need_marker_naming_its_ref(
    flag: Rule, marker: str
) -> None:
    record, binding = gating({"s": gated(None, on_flag("yes"))}, {"flag": flag})
    kind = "Unavailable" if marker == "need-unavailable" else "Unasked"
    marked(record, mark(marker, "d:flag", kind=kind))
    assert asked(binding) == []


def test_a_source_requirement_compares_canonical_bytes_so_1_equals_1_0() -> None:
    rules = {"n": rule("n", lambda r: 1.0, "t:state", outputs=t.number())}
    requirement = SourceRequirement("d:n", [1])
    _, binding = gating({"s": gated(None, requirement)}, rules)
    assert asked(binding) == ["s"]


def test_a_requirements_marker_ranks_among_the_needs_markers_and_is_named_with_them() -> None:
    rules = {
        "flag": rule("flag", gone, "t:state", outputs=t.label("no", "yes")),
        "vague": rule("vague", abstains, "t:state"),
        "fail": rule("fail", gone, "t:state"),
        "late": rule("late", asks_later, "t:state"),
    }
    sources = {
        # Unavailable on the ref outranks Unsure on a declared need.
        "a": gated(None, on_flag("yes"), "d:vague", tag="_a"),
        # Unavailable on both: every need of the winning kind, in code-point order.
        "b": gated(None, on_flag("yes"), "d:fail", tag="_b"),
        # Undetermined on the ref and a need: named together.
        "c": gated(None, SourceRequirement("d:late", [True]), "d:vague", tag="_c"),
    }
    record, _ = gating(sources, rules)
    marked(record, mark("need-unavailable", "d:flag"), "t:alt_a", "t:out_a")
    marked(record, mark("need-unavailable", "d:fail", "d:flag"), "t:alt_b", "t:out_b")
    c = mark("need-undetermined", "d:late", "d:vague", kind="Unasked")
    marked(record, c, "t:alt_c", "t:out_c")


def counting(calls: list[str], name: str, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def when(r: Any) -> Any:
        calls.append(name)
        return fn(r)

    return when


def test_a_requirement_that_fails_is_when_false_and_the_when_is_never_called() -> None:
    calls: list[str] = []
    when = counting(calls, "s", raises("not reached"))
    record, _ = gating({"s": gated(when, on_flag("no"))}, {"flag": label("flag", "yes")})
    marked(record, mark("when-false", "d:flag", kind="Unasked"))
    assert calls == []
    assert "exceptions" not in record.unhashed


def test_an_undetermined_need_tests_no_requirement_and_calls_no_when() -> None:
    calls: list[str] = []
    when = counting(calls, "s", lambda r: True)
    rules = {"vague": rule("vague", abstains, "t:state"), "flag": label("flag", "no")}
    record, _ = gating({"s": gated(when, on_flag("yes"), "d:vague")}, rules)
    marked(record, mark("need-undetermined", "d:vague", kind="Unasked"))
    assert calls == []


def test_both_forms_must_hold() -> None:
    rules = {"flag": label("flag", "yes")}
    _, binding = gating({"s": gated(lambda r: True, on_flag("yes"))}, rules)
    assert asked(binding) == ["s"]
    record, binding = gating({"s": gated(lambda r: False, on_flag("yes"))}, rules)
    marked(record, mark("when-false", "s", kind="Unasked"))


def test_a_ready_sources_when_is_called_once_and_whens_run_in_source_id_order() -> None:
    calls: list[str] = []
    sources = {
        "zed": gated(counting(calls, "zed", lambda r: True), tag="_z"),
        "alpha": gated(counting(calls, "alpha", lambda r: True), None, "t:out_z", tag="_a"),
        "mid": gated(counting(calls, "mid", lambda r: True), tag="_m"),
    }
    gating(sources)
    assert calls == ["mid", "zed", "alpha"]


def test_a_source_the_round_bound_marks_has_its_when_never_called() -> None:
    calls: list[str] = []
    sources = {
        "first": gated(tag="_f"),
        "second": gated(counting(calls, "second", lambda r: True), None, "t:out_f"),
    }
    loaded = load(minimal(state=GATE_STATE_T, sources=sources))
    binding = Scripted(dict.fromkeys(sources, answers))
    record = turn(loaded, {"default": binding}, GATE_STATE, round_bound=1)
    marked(record, mark("round-bound", kind="Unasked"))
    assert calls == []


def test_a_requirement_on_a_rule_that_fires_in_round_n_gates_a_source_of_round_n_plus_1() -> None:
    sources = {"first": gated(tag="_f"), "s": gated(None, on_flag("yes"))}
    rules = {"flag": label("flag", "yes", "t:out_f")}
    record, binding = gating(sources, rules)
    assert asked(binding) == ["first", "s"]
    assert entry(record, "d:flag")["round"] == 1
    assert entry(record, "t:out")["round"] == 2


def cancels(r: Any) -> Any:
    raise asyncio.CancelledError


def test_a_when_raising_a_cancellation_in_round_2_keeps_round_1_whole_and_none_of_round_2() -> None:
    calls: list[str] = []
    sources = {
        "first": gated(tag="_f"),
        "second": gated(cancels, None, "t:out_f"),
        "third": gated(counting(calls, "third", lambda r: True), None, "t:out_f", tag="_t"),
    }
    loaded = load(minimal(state=GATE_STATE_T, sources=sources))
    binding = Scripted(dict.fromkeys(sources, answers))
    run_ = settings()
    ledger = open_ledger(loaded, GATE_STATE, run_)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(interpret_on(ledger, loaded, GATE_STATE, {"default": binding}, run_))
    log = ledger.close().log[3:]
    assert [(i["item"], i.get("ref"), i["round"]) for i in log] == [
        ("call", None, 1),
        ("entry", "t:alt_f", 1),
        ("entry", "t:out_f", 1),
        ("round", None, 1),
    ]
    assert asked(binding) == ["first"]
    assert calls == []


def test_a_when_raising_a_base_exception_propagates_bare_with_no_item_appended() -> None:
    def interrupts(r: Any) -> Any:
        raise Interrupt

    loaded = load(minimal(state=GATE_STATE_T, sources={"s": gated(interrupts)}))
    run_ = settings()
    ledger = open_ledger(loaded, GATE_STATE, run_)
    binding = Scripted({"s": answers})
    with pytest.raises(Interrupt):
        asyncio.run(interpret_on(ledger, loaded, GATE_STATE, {"default": binding}, run_))
    assert len(ledger.close().log) == 3
    assert binding.batches == []


def test_a_truth_test_raising_a_base_exception_propagates_bare() -> None:
    class Interrupting:
        def __bool__(self) -> bool:
            raise Interrupt

    loaded = load(minimal(state=GATE_STATE_T, sources={"s": gated(lambda r: Interrupting())}))
    run_ = settings()
    ledger = open_ledger(loaded, GATE_STATE, run_)
    with pytest.raises(Interrupt):
        asyncio.run(interpret_on(ledger, loaded, GATE_STATE, {"default": Scripted({})}, run_))
    assert len(ledger.close().log) == 3


def reads_v0(r: Any) -> bool:
    return r.read("t:v0") == "v"


def test_shuffled_latencies_give_identical_bytes_with_whens_and_requirements() -> None:
    sources = pairs(4)
    sources["gate"] = Source({"t:g": t.string}, needs=["t:v0"], when=When(reads_v0, version="1"))
    sources["shut"] = Source({"t:h": t.string}, needs=["t:v1"], when=When(raises("x"), version="1"))
    sources["want"] = Source({"t:w": t.string}, requirement=SourceRequirement("d:sum", [True]))
    sources["deny"] = Source({"t:d": t.string}, requirement=SourceRequirement("d:sum", [False]))
    rules = {"sum": rule("sum", lambda r: True, *(f"q:k{i}" for i in range(4)))}
    loaded = make(sources, rules)

    hashes: set[str] = set()
    orders: set[tuple[str, ...]] = set()
    records: list[Record] = []
    for seed in range(6):
        rng = random.Random(seed)
        finished: list[str] = []

        async def reply(
            batch: Batch, rng: random.Random = rng, finished: list[str] = finished
        ) -> object:
            await asyncio.sleep(rng.random() / 100)
            finished.append(batch[0].id)
            return returns(valued(batch))

        scripted_ = Scripted(dict.fromkeys(sources, reply))
        binds = {"default": scripted_} | {f"o{i}": scripted_ for i in range(4)}
        records.append(turn(loaded, binds, STATE))
        hashes.add(records[-1].record_hash)
        orders.add(tuple(finished))
    assert len(orders) > 1
    assert len(hashes) == 1
    record = records[0]
    assert entry(record, "t:g")["call"] is not None
    assert entry(record, "t:h")["marker"] == mark("when-failed", "shut")
    assert entry(record, "t:w")["call"] is not None
    assert entry(record, "t:d")["marker"] == mark("when-false", "d:sum", kind="Unasked")


def test_a_require_stop_in_a_when_is_an_engine_defect(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every need is determined before a `when` runs, so `require` cannot stop one; were the
    # need test skipped, the stop would surface as a defect, never as a marker.
    def determined(self: object, sid: str) -> None:
        return None

    monkeypatch.setattr(_interpret._Turn, "need_marker", determined)
    sources = {"s": gated(lambda r: r.require("d:vague"), None, "d:vague")}
    with pytest.raises(_interpret._InterpretError, match="require stopped"):
        gating(sources, {"vague": rule("vague", abstains, "t:state")})


def test_a_source_requirement_is_tested_against_loads_copy_not_the_callers() -> None:
    value: dict[str, Any] = {"a": "x"}
    rules = {"rec": rule("rec", lambda r: {"a": "x"}, "t:state", outputs=t.record(a=t.string))}
    sources = {"s": gated(None, SourceRequirement("d:rec", [value]))}
    loaded = load(minimal(state=GATE_STATE_T, sources=sources, rules=rules))
    before = loaded.design_hash
    value["a"] = "y"
    binding = Scripted({"s": answers})
    turn(loaded, {"default": binding}, GATE_STATE)
    assert asked(binding) == ["s"]
    value["a"] = {1, 2}  # outside the domain: interpret never encodes the caller's value
    binding = Scripted({"s": answers})
    turn(loaded, {"default": binding}, GATE_STATE)
    assert asked(binding) == ["s"]
    assert loaded.design_hash == before


def test_a_returned_value_whose_truth_test_cancels_keeps_the_items_appended_before() -> None:
    class Cancelling:
        def __bool__(self) -> bool:
            raise asyncio.CancelledError

    sources = {"first": gated(tag="_f"), "second": gated(lambda r: Cancelling(), None, "t:out_f")}
    loaded = load(minimal(state=GATE_STATE_T, sources=sources))
    binding = Scripted(dict.fromkeys(sources, answers))
    run_ = settings()
    ledger = open_ledger(loaded, GATE_STATE, run_)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(interpret_on(ledger, loaded, GATE_STATE, {"default": binding}, run_))
    log = ledger.close().log[3:]
    assert [i["item"] for i in log] == ["call", "entry", "entry", "round"]
    assert asked(binding) == ["first"]


async def async_when(r: Any) -> bool:
    return True


class Awaitable_:
    def __await__(self) -> Any:
        return iter(())

    def __bool__(self) -> bool:
        raise AssertionError("an awaitable takes no truth test")


def awaitable_when(r: Any) -> Any:
    return Awaitable_()


def returns_coroutine(r: Any) -> Any:
    return async_when(r)


@pytest.mark.parametrize(
    "when", [async_when, awaitable_when, returns_coroutine], ids=["async", "await", "sync"]
)
def test_a_when_returning_an_awaitable_is_when_failed_with_no_truth_test(
    when: Callable[[Any], Any],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a coroutine never awaited would warn here
        record, binding = gating({"s": gated(when)})
        gc.collect()
    marked(record, mark("when-failed", "s"))
    assert asked(binding) == []
    assert "exceptions" not in record.unhashed


def test_a_returned_coroutine_is_closed() -> None:
    returned: list[Any] = []

    def keeps(r: Any) -> Any:
        coroutine = async_when(r)
        returned.append(coroutine)
        return coroutine

    gating({"s": gated(keeps)})
    assert returned[0].cr_frame is None  # closed, never started


@pytest.mark.parametrize("make_", [Claims, MetaRaising, Masked], ids=["claims", "meta", "class"])
def test_a_when_return_that_is_no_awaitable_by_its_type_takes_its_truth_test(
    make_: Callable[[], Any],
) -> None:
    record, binding = gating({"s": gated(lambda r: make_())})
    assert asked(binding) == ["s"]
    assert "exceptions" not in record.unhashed


@pytest.mark.parametrize(
    "make_", [InheritsAwait, generator_coroutine], ids=["inherited", "generator"]
)
def test_a_when_returning_an_awaitable_by_its_type_is_when_failed(
    make_: Callable[[], Any],
) -> None:
    record, binding = gating({"s": gated(lambda r: make_())})
    marked(record, mark("when-failed", "s"))
    assert asked(binding) == []


def test_a_when_whose_truth_test_re_raises_a_kept_stop_is_when_failed_with_no_text() -> None:
    kept = stop()

    class Stopping:
        def __bool__(self) -> bool:
            raise kept

    record, binding = gating({"s": gated(lambda r: Stopping())})
    marked(record, mark("when-failed", "s"))
    assert asked(binding) == []
    assert "exceptions" not in record.unhashed


def test_a_whens_coroutine_re_raising_a_kept_stop_on_close_is_when_failed() -> None:
    record, _ = gating({"s": gated(lambda r: started_raising_on_close(stop()))})
    marked(record, mark("when-failed", "s"))
    assert "exceptions" not in record.unhashed


def test_a_when_return_whose_awaitable_test_raises_is_when_failed_with_its_text() -> None:
    record, binding = gating({"s": gated(lambda r: clashing())})
    marked(record, mark("when-failed", "s"))
    assert asked(binding) == []
    index = str(record.log.index(entry(record, "t:alt")))
    assert record.unhashed["exceptions"][index]["message"] == "the type test raised"


def test_a_whens_coroutine_is_closed_when_an_undeclared_read_wins() -> None:
    returned: list[Any] = []

    def keeps(r: Any) -> Any:
        with contextlib.suppress(BaseException):
            r.read("t:junk")
        returned.append(async_when(r))
        return returned[0]

    record, _ = gating({"s": gated(keeps)})
    marked(record, mark("undeclared-read", "s"))
    assert inspect.getcoroutinestate(returned[0]) == inspect.CORO_CLOSED


def test_a_whens_coroutine_raising_on_close_keeps_when_failed_and_its_text() -> None:
    record, binding = gating({"s": gated(lambda r: started_raising_on_close("on close"))})
    marked(record, mark("when-failed", "s"))
    assert asked(binding) == []
    index = str(record.log.index(entry(record, "t:alt")))
    exc = record.unhashed["exceptions"][index]
    assert (exc["type"], exc["message"]) == ("ValueError", "on close")


# ---------------------------------------------------------------------------------------------
# Wording slots and the question identity

SLOTTED = "{{{s}}} at {bar}: {s}?}}"
SLOTS = (slot("s", ref="t:state.subject"), slot("bar", param="bar"))


def slotted(template: str = SLOTTED, *slots: dict[str, Any], **changes: Any) -> Source:
    question = Question("noul", Text(template, slots or SLOTS))
    fields: dict[str, Any] = {"provides": {"q:spam": question}}
    fields.update(changes)
    return Source(**fields)


def with_params(sources: dict[str, Source], **leaves: Parameter) -> LoadedDesign:
    return make(sources, parameters={"bar": Parameter(t.number(0, 1), 0.9), **leaves})


def asked_with(loaded: LoadedDesign, state: object = STATE) -> tuple[Record, SourceCall]:
    binding = Scripted({"ask": returns({"q:spam": NOUL})})
    record = turn(loaded, {"default": binding}, state)
    ((call,),) = binding.batches
    return record, call


def test_a_wording_fills_board_and_parameter_slots_with_its_escapes_in_one_pass() -> None:
    _, call = asked_with(with_params({"ask": slotted()}))
    assert call.questions["q:spam"].wording == "{hello} at 0.9: hello?}"


def test_a_slot_value_is_inserted_as_it_renders_and_never_read_for_braces() -> None:
    _, call = asked_with(with_params({"ask": slotted()}), {"subject": "{bar}}{{"})
    assert call.questions["q:spam"].wording == "{{bar}}{{} at 0.9: {bar}}{{?}"


def test_a_slot_ref_is_in_needs_as_declared_and_a_parameter_slots_value_is_not() -> None:
    _, call = asked_with(with_params({"ask": slotted()}))
    assert call.needs == {"t:state.subject": "hello"}


def test_a_slot_that_is_also_a_declared_need_is_one_key() -> None:
    _, call = asked_with(with_params({"ask": slotted(needs=["t:state.subject"])}))
    assert call.needs == {"t:state.subject": "hello"}


def test_a_marker_on_a_slots_ref_keeps_the_source_from_running() -> None:
    sources = {
        "ask": slotted("Is {who}?", slot("who", ref="t:who")),
        "who": Source({"t:who": t.string}),
    }
    binding = Scripted({"who": raising(RuntimeError("x"))})
    record = turn(make(sources), {"default": binding}, STATE)
    assert [b[0].id for b in binding.batches] == ["who"]
    assert entry(record, "q:spam")["marker"] == mark("need-unavailable", "t:who")


def slotted_identity(bar: object = 0.9) -> str:
    return hash_value(
        {
            "kind": "noul",
            "wording": {
                "template": SLOTTED,
                "slots": [
                    {"name": "s", "origin": {"ref": "t:state.subject"}},
                    {"name": "bar", "origin": {"param": "bar", "value": bar}},
                ],
            },
            "labels": None,
        }
    )


def recorded_identity(loaded: LoadedDesign, state: object = STATE) -> str:
    record, _ = asked_with(loaded, state)
    return record.log[3]["questions"]["q:spam"]["identity"]


def test_the_identity_hashes_each_parameter_origin_with_its_value_and_no_board_value() -> None:
    loaded = with_params({"ask": slotted()})
    assert recorded_identity(loaded) == slotted_identity()
    assert recorded_identity(loaded, {"subject": "other"}) == slotted_identity()
    # The document the identity is taken from is left as it was.
    assert recorded_identity(loaded) == slotted_identity()


def test_a_parameter_a_slot_reads_changes_the_identity_and_one_no_slot_reads_keeps_it() -> None:
    base_ = recorded_identity(with_params({"ask": slotted()}))
    moved = with_params({"ask": slotted()}, bar=Parameter(t.number(0, 1), 0.8))
    assert recorded_identity(moved) == slotted_identity(0.8) != base_
    other = with_params({"ask": slotted()}, cap=Parameter(t.integer(), 3))
    assert recorded_identity(other) == base_


def test_a_bytes_parameters_identity_value_is_its_record_form() -> None:
    sources = {"ask": slotted("Blob {b}", slot("b", param="blob"))}
    loaded = with_params(sources, blob=Parameter(t.bytes_, b"\x00\x01"))
    record, call = asked_with(loaded)
    assert call.questions["q:spam"].wording == 'Blob "AAE"'
    assert record.log[3]["questions"]["q:spam"]["identity"] == hash_value(
        {
            "kind": "noul",
            "wording": {
                "template": "Blob {b}",
                "slots": [{"name": "b", "origin": {"param": "blob", "value": "AAE"}}],
            },
            "labels": None,
        }
    )


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        ("plain", "plain"),
        (["a", "b"], "a, b"),
        ([], ""),
        (("x", ["y", "z"]), "x, y, z"),
        ([1, "a"], "1, a"),
        (0.5, "0.5"),
        (1.0, "1"),
        (3, "3"),
        (True, "true"),
        (None, "null"),
        ({"b": 1, "a": "x"}, '{"a":"x","b":1}'),
        (b"\x00\x01", '"AAE"'),
        ({"blob": b"\x00\x01"}, '{"blob":"AAE"}'),
        ({"l": [b"\x00\x01", ("a",)]}, '{"l":["AAE",["a"]]}'),
    ],
)
def test_a_slot_value_renders_by_the_declared_text_rule(value: object, rendered: str) -> None:
    assert _interpret._text(value) == rendered


def test_a_record_slot_renders_as_its_canonical_encoding() -> None:
    sources = {
        "ask": slotted("Is {p}?", slot("p", ref="t:profile")),
        "look": look(),
    }
    binding = Scripted({"look": returns(dict(LOOKED)), "ask": returns({"q:spam": NOUL})})
    turn(make(sources), {"default": binding}, STATE)
    (call,) = [b[0] for b in binding.batches if b[0].id == "ask"]
    assert call.questions["q:spam"].wording == 'Is {"age":3,"home":{"city":"Oslo"},"name":"Ann"}?'


# ---------------------------------------------------------------------------------------------
# Oracle names


def keyed(name: str, key: str | None = "k", **changes: Any) -> Source:
    fields: dict[str, Any] = {
        "provides": {f"q:{name}": Question("noul", Text(f"Is {name}?"))},
        "oracle": key,
        "cost": 1,
    }
    fields.update(changes)
    return Source(**fields)


def paired(name: str, key: str | None = "k", **changes: Any) -> dict[str, Source]:
    """``keyed(name)`` and ``{name}t``, its ``t:{name}`` under the same oracle, ready when it
    is and held to its timeout: a batch of both levels, whose entries keep source id order."""
    shared = {k: v for k, v in changes.items() if k in ("needs", "timeout")}
    typed = Source({f"t:{name}": t.string}, oracle=key, **shared)
    return {name: keyed(name, key, **changes), f"{name}t": typed}


def answering(batch: Batch) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for call in batch:
        values.update(dict.fromkeys(call.refs, call.id))
        values.update(dict.fromkeys(call.questions, NOUL))
    return values


async def answer_all(batch: Batch) -> object:
    return returns(answering(batch))


def keyed_run(
    sources: dict[str, Source], replies: Mapping[str, object] | None = None, **run_: Any
) -> tuple[Record, dict[str, Scripted]]:
    keys = {s.oracle for s in sources.values()} | {"default"}
    keys.discard(None)
    binds = {
        cast(str, k): Scripted(dict.fromkeys(sources, answer_all) | dict(replies or {}))
        for k in keys
    }
    return turn(make(sources), binds, STATE, **run_), binds


def calls_of(record: Record) -> list[dict[str, Any]]:
    return [i for i in body(record) if i["item"] == "call"]


def test_a_keyed_batch_is_one_call_through_its_keys_binding_in_source_id_order() -> None:
    sources = {**paired("c", cost=0.5), **paired("a"), **paired("b", cost=2)}
    record, binds = keyed_run(sources)
    (batch,) = binds["k"].batches
    assert [call.id for call in batch] == ["a", "at", "b", "bt", "c", "ct"]
    assert binds["default"].batches == []
    (item,) = calls_of(record)
    assert item["sources"] == ["a", "at", "b", "bt", "c", "ct"]
    assert list(item["questions"]) == ["q:a", "q:b", "q:c"]
    for ref in ("q:a", "t:a", "q:b", "t:b", "q:c", "t:c"):
        assert entry(record, ref)["call"] == 3
    assert entry(record, "t:b")["value"] == "bt"
    assert body(record)[-1] == {"item": "round", "round": 1, "cost": 3.5, "deadline": False}


def test_call_items_go_by_first_source_id_sources_with_no_oracle_alone_through_default() -> None:
    sources = {
        **paired("b", "k"),
        **paired("a", None),
        **paired("d", "j"),
        **paired("c", "k"),
        **paired("e", None),
    }
    record, binds = keyed_run(sources)
    # A source with no oracle is a call of its own, so a and at, e and et are four calls.
    assert [i["sources"] for i in calls_of(record)] == [
        ["a"], ["at"], ["b", "bt", "c", "ct"], ["d", "dt"], ["e"], ["et"],
    ]  # fmt: skip
    alone = [[c.id for c in b] for b in binds["default"].batches]
    assert sorted(alone) == [["a"], ["at"], ["e"], ["et"]]
    assert [[c.id for c in b] for b in binds["j"].batches] == [["d", "dt"]]
    # Entries by source id after every call item, not batch by batch.
    assert [i.get("ref") for i in body(record) if i["item"] == "entry"] == [
        "q:a", "t:a", "q:b", "t:b", "q:c", "t:c", "q:d", "t:d", "q:e", "t:e",
    ]  # fmt: skip
    assert [entry(record, f"t:{s}")["call"] for s in "abcde"] == [4, 5, 5, 6, 8]


def test_a_keyed_source_that_is_not_ready_is_left_out_of_its_batch() -> None:
    sources = {
        **paired("a"),
        **paired("b", needs=["t:x"]),
        "x": Source({"t:x": t.string}),
    }
    record, binds = keyed_run(sources, {"x": raising(RuntimeError("x"))})
    assert [[c.id for c in b] for b in binds["k"].batches] == [["a", "at"]]
    assert entry(record, "t:b")["marker"] == mark("need-unavailable", "t:x")
    assert entry(record, "t:b")["call"] is None


def test_a_key_with_no_ready_source_makes_no_call() -> None:
    sources = {**paired("b", needs=["t:x"]), "x": Source({"t:x": t.string})}
    record, binds = keyed_run(sources, {"x": raising(RuntimeError("x"))})
    assert binds["k"].batches == []
    assert [i["sources"] for i in calls_of(record)] == [["x"]]


def test_sources_of_one_key_ready_in_two_rounds_are_two_calls() -> None:
    sources = {**paired("a"), **paired("b", needs=["t:a"])}
    record, binds = keyed_run(sources)
    assert [[c.id for c in b] for b in binds["k"].batches] == [["a", "at"], ["b", "bt"]]
    assert [(i["round"], i["sources"]) for i in calls_of(record)] == [
        (1, ["a", "at"]),
        (2, ["b", "bt"]),
    ]


KEYED_REFS = ["q:a", "q:b", "t:a", "t:b"]


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        (raising(RuntimeError("x")), "raised"),
        (CallResult(cast(Any, [1])), "invalid-return"),
        (Failed(Cause(CauseCode.timed_out, ["t:a"])), "timed-out"),
        (Failed(Cause(CauseCode.raised, [])), "raised"),
    ],
)
def test_a_structural_fault_fails_every_ref_of_the_batch_in_code_point_order(
    reply: object, code: str
) -> None:
    record, _ = keyed_run({**paired("b"), **paired("a")}, {"a": reply})
    (item,) = calls_of(record)
    assert item["outcome"] == {"failed": {"code": code, "refs": KEYED_REFS}}
    for ref in KEYED_REFS:
        assert entry(record, ref)["marker"] == mark(code, *KEYED_REFS)
        assert entry(record, ref)["call"] == 3


def test_a_keyed_batch_past_its_timeout_fails_every_ref_timed_out() -> None:
    sources = {**paired("a", timeout=0.02), **paired("b")}
    record, _ = keyed_run(sources, {"a": sleeping(5, None)})
    (item,) = calls_of(record)
    assert item["outcome"] == {"failed": {"code": "timed-out", "refs": KEYED_REFS}}


def test_a_siblings_key_is_no_unexpected_ref_and_one_from_outside_the_batch_is() -> None:
    async def siblings(batch: Batch) -> object:
        return returns(answering(batch))

    record, _ = keyed_run({**paired("a"), **paired("b")}, {"a": siblings})
    assert calls_of(record)[0]["outcome"] == "returned"

    async def outside(batch: Batch) -> object:
        return returns({**answering(batch), "t:c": "c"})

    sources = {**paired("a"), **paired("b"), **paired("c", None)}
    record, _ = keyed_run(sources, {"a": outside})
    (item,) = [i for i in calls_of(record) if i["sources"] == ["a", "at", "b", "bt"]]
    assert item["outcome"] == {"failed": {"code": "unexpected-ref", "refs": KEYED_REFS}}
    assert record.unhashed["calls"][str(record.log.index(item))]["unexpected"] == ["t:c"]


def test_a_value_fault_or_an_omitted_ref_in_a_batch_hits_that_ref_alone() -> None:
    async def partial(batch: Batch) -> object:
        values = answering(batch)
        del values["t:b"]
        values["q:a"] = {"yes": 2}
        return returns(values)

    record, _ = keyed_run({**paired("a"), **paired("b")}, {"a": partial})
    assert calls_of(record)[0]["outcome"] == "returned"
    assert entry(record, "t:b")["marker"] == mark("not-returned", "t:b")
    assert entry(record, "q:a")["marker"] == mark("invalid-answer", "q:a")
    assert entry(record, "t:a")["value"] == "at"
    assert entry(record, "q:b")["value"]["probabilities"] == NOUL
    assert {entry(record, r)["call"] for r in KEYED_REFS} == {3}


def test_a_mixed_batch_takes_the_largest_declared_timeout_never_the_run_setting() -> None:
    # A declares 0.02, B none, the run setting 10: the batch's timeout is 0.02.
    sources = {**paired("a", timeout=0.02), **paired("b")}
    record, _ = keyed_run(sources, {"a": sleeping(5, None)}, timeout=10)
    assert calls_of(record)[0]["outcome"]["failed"]["code"] == "timed-out"
    # The largest declared: 5 outlasts a call of 0.05.
    sources = {**paired("a", timeout=0.02), **paired("b", timeout=5)}
    later = sleeping(0.05, None)

    async def slow(batch: Batch) -> object:
        await later(batch)
        return returns(answering(batch))

    record, _ = keyed_run(sources, {"a": slow}, timeout=0.01)
    assert calls_of(record)[0]["outcome"] == "returned"


def test_one_binding_is_called_concurrently_with_itself_under_two_keys() -> None:
    started: list[str] = []
    both = asyncio.Event()

    async def wait(batch: Batch) -> object:
        started.append(batch[0].id)
        if len(started) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 2)
        return returns(answering(batch))

    sources = {**paired("a", "k"), **paired("b", "j")}
    binding = Scripted({"a": wait, "b": wait})
    record = turn(make(sources), {"k": binding, "j": binding}, STATE)
    assert sorted(started) == ["a", "b"]
    assert [i["outcome"] for i in calls_of(record)] == ["returned", "returned"]


def test_each_source_in_a_batch_gets_its_own_copy_of_a_shared_need() -> None:
    async def meddle(batch: Batch) -> object:
        first, *others = batch
        assert len(others) == 3
        first.needs["t:state.subject"] = "Mallory"
        for other in others:
            assert other.needs == {"t:state.subject": "hello"}
        return returns(answering(batch))

    need = ["t:state.subject"]
    sources = {**paired("a", needs=need), **paired("b", needs=need)}
    record, _ = keyed_run(sources, {"a": meddle})
    assert calls_of(record)[0]["outcome"] == "returned"


def test_a_keyed_batch_passes_each_sources_rendered_slots() -> None:
    sources = {
        "a": keyed("a", provides={"q:a": Question("noul", Text("A {s}?", [SLOTS[0]]))}),
        "b": keyed("b", provides={"q:b": Question("noul", Text("B {bar}?", [SLOTS[1]]))}),
    }
    loaded = with_params(sources)
    binding = Scripted({"a": answer_all})
    turn(loaded, {"k": binding}, STATE)
    ((a, b),) = binding.batches
    assert (a.questions["q:a"].wording, b.questions["q:b"].wording) == ("A hello?", "B 0.9?")
    assert (a.needs, b.needs) == ({"t:state.subject": "hello"}, {})


# ---------------------------------------------------------------------------------------------
# The budget


def budgeted() -> LoadedDesign:
    """``chain`` with cost tags: s1 costs 1, s2 1 and s3 2."""
    sources = {
        "s1": Source({"t:a": t.string}, cost=1),
        "s2": Source({"t:b": t.string}, needs=["d:r1"], cost=1),
        "s3": Source({"t:c": t.string, "t:d": t.string}, needs=["d:r2"], cost=2),
    }
    rules = {
        "r1": rule("r1", lambda r: True, "t:a"),
        "r2": rule("r2", lambda r: r.require("t:b"), "t:b", outputs=t.string),
        "r3": rule("r3", lambda r: r.require("t:c"), "t:c", outputs=t.string),
    }
    return make(sources, rules)


def rounds(record: Record) -> list[dict[str, Any]]:
    return [i for i in body(record) if i["item"] == "round"]


def test_a_frontier_over_budget_in_round_1_is_not_asked_and_the_drain_fires_behind_it() -> None:
    record, binding = run(budgeted(), CHAIN, budget=0.5)
    assert binding.batches == []
    assert kinds(record) == [
        ("entry", "t:a"),
        ("entry", "t:b"),
        ("entry", "t:c"),
        ("entry", "t:d"),
        ("entry", "d:r1"),
        ("entry", "d:r2"),
        ("entry", "d:r3"),
        ("round", 1),
    ]
    for ref in ("t:a", "t:b", "t:c", "t:d"):
        assert entry(record, ref)["marker"] == mark("over-budget", kind="Unasked")
        assert entry(record, ref)["round"] == 1
    assert entry(record, "d:r2")["marker"] == mark("required", "t:b", kind="Unasked")
    assert rounds(record) == [{"item": "round", "round": 1, "cost": 0, "deadline": False}]


def test_a_frontier_over_budget_after_a_charged_round_is_cut_in_round_2() -> None:
    record, binding = run(budgeted(), CHAIN, budget=1.5)
    assert [b[0].id for b in binding.batches] == ["s1"]
    round_2 = [(i["item"], i.get("ref")) for i in body(record) if i["round"] == 2]
    assert round_2 == [
        ("entry", "t:b"),
        ("entry", "t:c"),
        ("entry", "t:d"),
        ("entry", "d:r2"),
        ("entry", "d:r3"),
        ("round", None),
    ]
    assert entry(record, "t:c")["marker"] == mark("over-budget", kind="Unasked")
    assert [(i["cost"], i["deadline"]) for i in rounds(record)] == [(1, False), (0, False)]


def test_a_frontier_exactly_at_the_budget_is_asked() -> None:
    record, binding = run(budgeted(), CHAIN, budget=4)
    assert [b[0].id for b in binding.batches] == ["s1", "s2", "s3"]
    assert entry(record, "t:d")["value"] == "d"


def test_a_zero_cost_frontier_is_asked_under_a_budget_of_0() -> None:
    record, binding = run(
        make({"look": Source({"t:a": t.string})}), {"look": returns({"t:a": "a"})}, budget=0
    )
    assert [b[0].id for b in binding.batches] == ["look"]
    assert entry(record, "t:a")["value"] == "a"


def test_the_budget_compares_the_exact_sum_of_its_tags() -> None:
    # In floats 0.1 + 0.2 exceeds 0.3; taken exactly it equals it, so it is asked.
    sources = {"a": Source({"t:a": t.string}, cost=0.1), "b": Source({"t:b": t.string}, cost=0.2)}
    record, binding = run(make(sources), {s: returns({f"t:{s}": "v"}) for s in sources}, budget=0.3)
    assert len(binding.batches) == 2
    assert rounds(record)[0]["cost"] == float(Fraction("0.1") + Fraction("0.2")) == 0.3


def test_usage_never_feeds_the_budget() -> None:
    replies = {
        "s1": returns({"t:a": "a"}, usage={"cost": 1e9}),
        "s2": returns({"t:b": "b"}, usage={"cost": 1e9}),
        "s3": returns({"t:c": "c", "t:d": "d"}),
    }
    _, binding = run(budgeted(), replies, budget=4)
    assert len(binding.batches) == 3


def test_own_markers_come_first_then_the_over_budget_markers_by_source_id() -> None:
    sources = {
        "root": Source({"t:x": t.string, "t:y": t.string}, cost=1),
        "z": Source({"t:z": t.string}, needs=["t:x"], cost=1),
        "b": Source({"t:b": t.string}, needs=["t:y"], cost=5),
        "w": Source({"t:w": t.string}, needs=["t:b"]),
    }
    replies = {"root": returns({"t:y": "y"})}
    record, binding = run(make(sources), replies, budget=3)
    assert [b[0].id for b in binding.batches] == ["root"]
    round_2 = [i.get("ref") for i in body(record) if i["round"] == 2 and i["item"] == "entry"]
    assert round_2 == ["t:z", "t:b", "t:w"]
    assert entry(record, "t:z")["marker"] == mark("need-unavailable", "t:x")
    assert entry(record, "t:b")["marker"] == mark("over-budget", kind="Unasked")
    assert entry(record, "t:w")["marker"] == mark("over-budget", kind="Unasked")


# ---------------------------------------------------------------------------------------------
# The deadline

PAST = -1e9
"""A clock reading so early that any deadline from it has passed at round 1."""


def test_a_deadline_passed_before_round_1_asks_nothing_and_marks_every_source_deadline() -> None:
    binding = Scripted(CHAIN)
    record = turn(chain(), {"default": binding}, STATE, deadline=1, clock=PAST)
    assert binding.batches == []
    for ref in ("t:a", "t:b", "t:c", "t:d"):
        assert entry(record, ref)["marker"] == mark("deadline", kind="Unasked")
    assert entry(record, "d:r3")["round"] == 1
    assert rounds(record) == [{"item": "round", "round": 1, "cost": 0, "deadline": True}]


def test_the_budget_is_tested_before_the_deadline() -> None:
    binding = Scripted(CHAIN)
    record = turn(budgeted(), {"default": binding}, STATE, deadline=1, budget=0.5, clock=PAST)
    assert entry(record, "t:a")["marker"] == mark("over-budget", kind="Unasked")
    assert rounds(record) == [{"item": "round", "round": 1, "cost": 0, "deadline": False}]


def test_a_deadline_never_reached_cuts_nothing() -> None:
    binding = Scripted(CHAIN)
    record = turn(chain(), {"default": binding}, STATE, deadline=1000)
    assert entry(record, "t:d")["value"] == "d"
    assert all(not i["deadline"] for i in rounds(record))


def deadline_design() -> LoadedDesign:
    sources = {
        "fast": Source({"t:f": t.string}, cost=1),
        "slow": Source({"t:s": t.string}, cost=2),
        "next": Source({"t:n": t.string}, needs=["t:f"]),
    }
    rules = {"after": rule("after", lambda r: True, "t:n", "t:s")}
    return make(sources, rules)


DEADLINE_REPLIES: dict[str, object] = {
    "fast": returns({"t:f": "f"}),
    "slow": sleeping(10, returns({"t:s": "s"})),
    "next": returns({"t:n": "n"}),
}


def test_a_call_the_deadline_cuts_is_timed_out_and_charged_and_the_rest_stand() -> None:
    binding = Scripted(DEADLINE_REPLIES)
    record = turn(deadline_design(), {"default": binding}, STATE, deadline=0.1)
    # The slow call's timeout of 60 is cut to the 0.1 left, so the turn ends at once.
    slow = next(i for i in calls_of(record) if i["sources"] == ["slow"])
    assert slow["outcome"] == {"failed": {"code": "timed-out", "refs": ["t:s"]}}
    assert record.unhashed["calls"][str(record.log.index(slow))]["latency"] < 1
    assert entry(record, "t:s")["marker"] == mark("timed-out", "t:s")
    assert entry(record, "t:f")["value"] == "f"
    assert entry(record, "t:n")["marker"] == mark("deadline", kind="Unasked")
    assert kinds(record) == [
        ("call", 1),
        ("call", 1),
        ("entry", "t:f"),
        ("entry", "t:s"),
        ("entry", "t:n"),
        ("entry", "d:after"),
        ("round", 1),
    ]
    assert rounds(record) == [{"item": "round", "round": 1, "cost": 3, "deadline": True}]


def test_a_deadline_that_passes_with_nothing_pending_still_cuts_its_round() -> None:
    sources = {"slow": Source({"t:s": t.string})}
    rules = {"after": rule("after", lambda r: True, "t:s")}
    binding = Scripted({"slow": sleeping(10, None)})
    record = turn(make(sources, rules), {"default": binding}, STATE, deadline=0.05)
    assert entry(record, "t:s")["marker"] == mark("timed-out", "t:s")
    assert entry(record, "d:after")["value"] is True
    assert rounds(record) == [{"item": "round", "round": 1, "cost": 0, "deadline": True}]


def test_a_call_the_deadline_expired_cuts_its_round_though_the_clock_reads_under_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # asyncio fires a timeout up to its clock resolution early, so at the close the clock may
    # read under the end; the call the deadline expired still cuts the round. The engine's
    # own clock reads are pinned below every end; the calls' timeouts run on the real one.
    monkeypatch.setattr(_interpret, "_clock", lambda: float("-inf"))
    binding = Scripted({**DEADLINE_REPLIES, "slow": ignoring_cancel(10, returns({"t:s": "s"}))})

    async def main() -> Record:
        loaded = deadline_design()
        full = settings(deadline=0.05)
        ledger = open_ledger(loaded, STATE, full)
        start = asyncio.get_running_loop().time()
        await interpret_on(ledger, loaded, STATE, {"default": binding}, full, clock=start)
        return ledger.close()

    record = asyncio.run(main())
    slow = next(i for i in calls_of(record) if i["sources"] == ["slow"])
    assert slow["outcome"] == {"failed": {"code": "timed-out", "refs": ["t:s"]}}
    assert entry(record, "t:n")["marker"] == mark("deadline", kind="Unasked")
    assert rounds(record)[-1]["deadline"] is True


def test_a_call_past_its_own_shorter_timeout_does_not_cut_the_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_interpret, "_clock", lambda: float("-inf"))
    sources = {"slow": Source({"t:s": t.string}, timeout=0.02), "next": needing("t:s")}
    binding = Scripted({"slow": sleeping(10, None), "next": returns({"t:out": "o"})})
    record = turn(make(sources), {"default": binding}, STATE, deadline=600)
    assert entry(record, "t:s")["marker"] == mark("timed-out", "t:s")
    assert entry(record, "t:out")["marker"] == mark("need-unavailable", "t:s")
    assert all(not i["deadline"] for i in rounds(record))


def test_a_binding_blocking_past_the_deadline_is_cut_and_the_next_call_not_made() -> None:

    async def blocking(batch: Batch) -> object:
        time.sleep(0.1)  # noqa: ASYNC251 - never yields, so no timeout fires
        return returns({"t:f": "f"})

    binding = Scripted({**DEADLINE_REPLIES, "fast": blocking, "slow": returns({"t:s": "s"})})
    record = turn(deadline_design(), {"default": binding}, STATE, deadline=0.05)
    # fast returned past its end, so it expired; slow's task started with no time left, so
    # it was never called, yet is a call the deadline cut, charged.
    assert [b[0].id for b in binding.batches] == ["fast"]
    fast, slow = calls_of(record)
    assert fast["outcome"] == {"failed": {"code": "timed-out", "refs": ["t:f"]}}
    assert slow["outcome"] == {"failed": {"code": "timed-out", "refs": ["t:s"]}}
    calls = record.unhashed["calls"][str(record.log.index(slow))]
    assert (calls["latency"], calls["usage"], calls["attempts"]) == (0, None, None)
    assert entry(record, "t:s")["marker"] == mark("timed-out", "t:s")
    assert entry(record, "t:n")["marker"] == mark("deadline", kind="Unasked")
    assert rounds(record) == [{"item": "round", "round": 1, "cost": 3, "deadline": True}]


def test_a_never_yielding_binding_past_its_own_timeout_is_timed_out() -> None:

    async def blocking(batch: Batch) -> object:
        time.sleep(0.05)  # noqa: ASYNC251 - never yields, so the timeout never fires
        return returns({"t:s": "s"})

    sources = {"slow": Source({"t:s": t.string}, timeout=0.02)}
    record = turn(make(sources), {"default": Scripted({"slow": blocking})}, STATE)
    assert calls_of(record)[0]["outcome"] == {"failed": {"code": "timed-out", "refs": ["t:s"]}}
    assert all(not i["deadline"] for i in rounds(record))


def marked_in_round_2(r1: Callable[[Any], Any] = lambda r: True) -> LoadedDesign:
    """s1, then s2 needing t:a, then s3 needing t:b; with s1 failing, round 2's one held
    source has its own marker and nothing to ask."""
    sources = {
        "s1": Source({"t:a": t.string}),
        "s2": Source({"t:b": t.string}, needs=["t:a"]),
        "s3": Source({"t:c": t.string}, needs=["t:b"]),
    }
    return make(sources, {"r1": rule("r1", r1, "t:a")})


def test_a_round_with_nothing_to_ask_is_cut_once_the_deadline_has_passed() -> None:

    def slow(r: Any) -> bool:
        time.sleep(0.1)  # a rule's time counts against the deadline
        return True

    binding = Scripted({"s1": raising(RuntimeError("x"))})
    record = turn(marked_in_round_2(slow), {"default": binding}, STATE, deadline=0.05)
    assert [b[0].id for b in binding.batches] == ["s1"]
    assert [i["round"] for i in calls_of(record)] == [1]
    assert entry(record, "t:b")["marker"] == mark("need-unavailable", "t:a")
    assert entry(record, "t:c")["marker"] == mark("deadline", kind="Unasked")
    assert entry(record, "t:c")["round"] == 2
    assert [(i["cost"], i["deadline"]) for i in rounds(record)] == [(0, False), (0, True)]


@pytest.mark.parametrize("asked", [False, True])
def test_a_cut_at_a_round_with_nothing_to_ask_ends_it_in_either_form(asked: bool) -> None:
    binding = Scripted({"s1": raising(RuntimeError("x"))})
    record = turn(marked_in_round_2(), {"default": binding}, STATE, deadline=5, cut=(2, asked))
    assert [b[0].id for b in binding.batches] == ["s1"]
    assert [i["round"] for i in calls_of(record)] == [1]
    assert entry(record, "t:b")["marker"] == mark("need-unavailable", "t:a")
    assert entry(record, "t:c")["marker"] == mark("deadline", kind="Unasked")
    assert [(i["cost"], i["deadline"]) for i in rounds(record)] == [(0, False), (0, True)]


def test_each_call_runs_under_an_absolute_timeout_cut_to_the_time_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left: list[float] = []
    real = asyncio.timeout_at

    def spy(when: float | None) -> asyncio.Timeout:
        assert when is not None
        left.append(when - asyncio.get_running_loop().time())
        return real(when)

    monkeypatch.setattr(asyncio, "timeout_at", spy)
    record = turn(chain(), {"default": Scripted(CHAIN)}, STATE, deadline=30, timeout=60)
    assert len(left) == 3
    assert all(0 < seconds <= 30 for seconds in left)  # the deadline's, never the 60
    assert entry(record, "t:d")["value"] == "d"


# ---------------------------------------------------------------------------------------------
# The cut under replay


def test_a_cut_not_asked_ends_its_round_before_asking() -> None:
    binding = Scripted(CHAIN)
    record = turn(chain(), {"default": binding}, STATE, deadline=5, cut=(2, False))
    assert [b[0].id for b in binding.batches] == ["s1"]
    assert entry(record, "t:b")["marker"] == mark("deadline", kind="Unasked")
    assert entry(record, "t:c")["marker"] == mark("deadline", kind="Unasked")
    assert [(i["cost"], i["deadline"]) for i in rounds(record)] == [(0, False), (0, True)]


def test_a_cut_asked_asks_its_frontier_then_ends_as_the_deadline_does() -> None:
    binding = Scripted(CHAIN)
    record = turn(chain(), {"default": binding}, STATE, deadline=5, cut=(2, True))
    assert [b[0].id for b in binding.batches] == ["s1", "s2"]
    assert entry(record, "t:b")["value"] == "b"
    assert entry(record, "t:c")["marker"] == mark("deadline", kind="Unasked")
    assert rounds(record)[-1] == {"item": "round", "round": 2, "cost": 0, "deadline": True}


def test_a_cut_past_the_fixpoint_cuts_nothing() -> None:
    record = turn(chain(), {"default": Scripted(CHAIN)}, STATE, deadline=5, cut=(9, True))
    assert entry(record, "t:d")["value"] == "d"
    assert all(not i["deadline"] for i in rounds(record))


def test_a_deadline_with_no_clock_or_a_cut_beside_a_clock_is_an_engine_defect() -> None:
    from jurat._interpret import _InterpretError, interpret

    loaded = make(asking())
    cases = (
        (None, None, 1.0, "^a deadline runs from decide's clock reading$"),
        (0.0, (1, False), None, "^the cut stands in for the clock, never beside it$"),
    )
    for clock, cut, deadline, says in cases:
        ledger = open_ledger(loaded, STATE, settings())
        with pytest.raises(_InterpretError, match=says):
            asyncio.run(
                interpret(
                    ledger,
                    loaded,
                    STATE,
                    {},
                    round_bound=8,
                    timeout=60,
                    budget=None,
                    deadline=deadline,
                    clock=clock,
                    cut=cut,
                    carried=frozenset(),
                )
            )


# ---------------------------------------------------------------------------------------------
# Latency does not reach the record, over keyed batches, slots and a budget cut


def test_shuffled_latencies_give_identical_bytes_over_keys_slots_and_a_budget_cut() -> None:
    wording = Text("Is {s} {i} at {bar}?", [SLOTS[0], slot("i", ref="t:seed"), SLOTS[1]])
    sources: dict[str, Source] = {}
    for i in range(6):
        oracle = None if i == 5 else f"key{i % 2}"
        sources[f"s{i}"] = Source(
            {f"q:k{i}": Question("noul", wording)}, needs=["t:seed"], oracle=oracle, cost=1
        )
        sources[f"t{i}"] = Source({f"t:v{i}": t.string}, needs=["t:seed"], oracle=oracle)
    sources["seed"] = Source({"t:seed": t.string}, cost=1)
    sources["late"] = Source({"t:late": t.string}, needs=["d:sum"], cost=1)
    sources["later"] = Source({"t:later": t.string}, needs=["t:late"], cost=1)
    rules = {"sum": rule("sum", lambda r: True, *(f"q:k{i}" for i in range(6)))}
    loaded = make(sources, rules, parameters={"bar": Parameter(t.number(0, 1), 0.9)})

    hashes: set[str] = set()
    orders: set[tuple[str, ...]] = set()
    for seed in range(6):
        rng = random.Random(seed)
        finished: list[str] = []

        async def reply(
            batch: Batch, rng: random.Random = rng, finished: list[str] = finished
        ) -> object:
            await asyncio.sleep(rng.random() / 100)
            finished.append(batch[0].id)
            if batch[0].id == "s5":
                raise RuntimeError("one fails")
            return returns(answering(batch))

        scripted_ = Scripted(dict.fromkeys(sources, reply))
        binds = {"default": scripted_, "key0": scripted_, "key1": scripted_}
        # Round 1 asks seed (1), round 2 the six asking sources (6), round 3 late is over a
        # budget of 7. s5 and t5 name no oracle, so each is a call of its own.
        record = turn(loaded, binds, STATE, budget=7)
        hashes.add(record.record_hash)
        orders.add(tuple(finished))
        assert entry(record, "t:late")["marker"] == mark("over-budget", kind="Unasked")
        assert [len(i["sources"]) for i in calls_of(record)] == [1, 6, 4, 1, 1]
    assert len(orders) > 1
    assert len(hashes) == 1


# ---------------------------------------------------------------------------------------------
# The value domain's size bounds on what a component or binding returns


def _doubled(n: int) -> Any:
    v: Any = 0
    for _ in range(n):
        v = [v, v]
    return v


_DEEP_LIST = "list<" * 60 + "integer" + ">" * 60  # the type of _doubled(60)
# These tests run under small bounds (the small_bounds fixture): 2^6 values, 2^8 code points.
_BIG = "x" * 257
_WIDE = ["x" * 16] * 17


def test_a_rule_returning_a_value_over_the_size_bounds_is_invalid_return_fast(
    small_bounds: SmallBounds,
) -> None:
    with counted() as n:
        record = one_rule(lambda r: _doubled(60), outputs=_DEEP_LIST)
    assert n[0] < 4 * bound()  # the record form's build stops past the bound
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")
    record = one_rule(lambda r: _WIDE, outputs="list<string>")
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")
    record = one_rule(lambda r: [[0] * 4] * 4, outputs="list<list<integer>>")
    assert entry(record, "d:r")["value"] == [[0] * 4] * 4  # shared within the bounds


def test_the_size_is_counted_on_the_value_as_recorded(small_bounds: SmallBounds) -> None:
    # Undeclared fields are dropped first: what is recorded is within the bounds.
    returned = {"a": 1, "extra": _WIDE}
    record = one_rule(lambda r: returned, outputs="record{a: integer}")
    assert entry(record, "d:r")["value"] == {"a": 1}
    # Bytes are counted as their base64url string: 192 bytes are 2^8 code points.
    record = one_rule(lambda r: b"\x00" * 192, outputs=t.bytes_)
    assert "value" in entry(record, "d:r")
    record = one_rule(lambda r: b"\x00" * 193, outputs=t.bytes_)
    assert entry(record, "d:r")["marker"] == mark("invalid-return", "r")


def test_a_binding_value_over_the_size_bounds_fails_its_ref_alone(
    small_bounds: SmallBounds,
) -> None:
    source = look(provides={"t:rows": "list<string>", "t:body": t.string})
    loaded = make({"look": source})
    record, _ = run(loaded, {"look": returns({"t:rows": _WIDE, "t:body": "b"})})
    assert entry(record, "t:rows")["marker"] == mark("invalid-return", "t:rows")
    assert entry(record, "t:body")["value"] == "b"


def test_a_snapshot_over_the_size_bounds_is_invalid_return_and_unrepresentable(
    small_bounds: SmallBounds,
) -> None:
    record, _ = run(make(asking()), {"ask": ok(snapshot=_BIG)})
    assert record.log[3]["outcome"] == {"failed": {"code": "invalid-return", "refs": ASKED}}
    assert record.log[3]["snapshot"] is None
    assert record.unhashed["calls"]["3"]["invalid"] == {"snapshot": "<unrepresentable>"}


def test_a_usage_over_the_size_bounds_is_recorded_null(small_bounds: SmallBounds) -> None:
    record, _ = run(make(asking()), {"ask": ok(usage={_BIG: 1})})
    assert record.log[3]["outcome"] == "returned"
    assert record.unhashed["calls"]["3"]["usage"] is None
    assert record.unhashed["calls"]["3"]["invalid"] == {"usage": "<unrepresentable>"}


def test_an_exception_whose_arguments_are_too_large_has_no_text(small_bounds: SmallBounds) -> None:
    text = raises_it(ValueError(_WIDE))
    assert (text["message"], text["traceback"]) == ("<unrepresentable>", "<unrepresentable>")
    text = raises_it(ValueError(frozenset({"x" * 16 + str(i) for i in range(16)})))
    assert text["message"] == "<unrepresentable>"


def test_a_long_wording_is_held_to_no_size_bound(small_bounds: SmallBounds) -> None:
    wording = "Is it spam? " + "Consider every word of it. " * 20  # past the 2^8 code points set
    loaded = make(asking(provides={"q:spam": Question("noul", Text(wording))}))
    record, binding = run(loaded, {"ask": returns({"q:spam": NOUL})})
    assert record.log[3]["outcome"] == "returned"
    assert binding.batches[0][0].questions["q:spam"].wording == wording
    assert entry(record, "q:spam")["value"]["probabilities"] == NOUL


# ---------------------------------------------------------------------------------------------
# An exception's recorded text: walked over its args and notes and every chained exception


def _record(exc: BaseException) -> dict[str, str]:
    from jurat._component import exception_record

    return exception_record(exc)


class _Shown(list[Any]):
    def __repr__(self) -> str:
        return "shown"


def test_an_exception_holding_a_foreign_value_has_no_text() -> None:
    text = _record(ValueError(_Shown([1])))  # str(exc) would print it
    assert (text["message"], text["traceback"]) == ("<unrepresentable>", "<unrepresentable>")
    assert text["type"] == "ValueError"
    plain = _record(ValueError([1], "x", b"y", None))
    assert plain["message"] == "([1], 'x', b'y', None)"
    assert "ValueError" in plain["traceback"]


def test_an_exceptions_chain_and_notes_are_walked_together(small_bounds: SmallBounds) -> None:
    inner = KeyError("k" * 150)
    outer = ValueError("v" * 100)
    outer.__cause__ = inner
    assert _record(outer)["message"] == "v" * 100  # 250 code points in all: within
    inner.args = ("k" * 160,)
    assert _record(outer)["message"] == "<unrepresentable>"  # 260: past the 2^8 set
    noted = ValueError("ok")
    noted.add_note("n" * 200)
    assert _record(noted)["message"] == "ok"
    noted.add_note("n" * 100)
    assert _record(noted)["traceback"] == "<unrepresentable>"
    grouped = ExceptionGroup("g", [ValueError("m" * 200), KeyError("k" * 100)])
    assert _record(grouped)["message"] == "<unrepresentable>"  # its members' args too
    small = ExceptionGroup("g", [ValueError("m" * 100), KeyError("k" * 100)])
    assert _record(small)["message"] == "g (2 sub-exceptions)"  # within: its real text
    within = ValueError("v")
    within.__context__ = KeyError("k")
    assert _record(within)["message"] == "v"


def test_a_group_and_an_exception_inside_args_get_their_real_text() -> None:
    group = ExceptionGroup("several", [ValueError("one"), KeyError("two")])
    text = _record(group)
    assert text["message"] == "several (2 sub-exceptions)"
    assert "ValueError: one" in text["traceback"]
    assert "KeyError: 'two'" in text["traceback"]
    wrapped = _record(ValueError(KeyError("inner")))
    assert wrapped["message"] == "'inner'"
    assert "ValueError" in wrapped["traceback"]


def test_a_group_whose_member_holds_a_foreign_value_has_no_text() -> None:
    text = _record(ExceptionGroup("g", [ValueError(_Shown([1]))]))
    assert (text["message"], text["traceback"]) == ("<unrepresentable>", "<unrepresentable>")


def test_a_walk_that_raises_gives_no_text() -> None:
    class Planted:
        def __hash__(self) -> int:
            return hash("__notes__")

        def __eq__(self, other: object) -> bool:
            raise RuntimeError("compared")

    exc = ValueError("x")
    exc.__dict__[Planted()] = 1  # type: ignore[index]
    text = _record(exc)
    assert (text["message"], text["traceback"]) == ("<unrepresentable>", "<unrepresentable>")
    assert text["type"] == "ValueError"


def test_a_context_cycle_ends() -> None:
    a, b = ValueError("a"), ValueError("b")
    a.__context__ = b
    b.__context__ = a
    assert _record(a)["message"] == "a"


def test_an_exception_set_as_notes_is_walked_as_an_exception(small_bounds: SmallBounds) -> None:
    exc = ValueError("x")
    exc.__notes__ = KeyError("k" * 300)  # type: ignore[attr-defined]
    assert _record(exc)["message"] == "<unrepresentable>"  # its args were walked: past the bound
    exc.__notes__ = KeyError("k")  # type: ignore[attr-defined]
    assert _record(exc)["message"] == "x"
