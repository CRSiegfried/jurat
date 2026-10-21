# pyright: reportPrivateUsage=false
"""Decide: the entry refusals, the run settings, the state
entry and the codec, the broken entry, the observer's exception and the cancellation at every
point they can land, ``decide_sync``, ``unhashed``, and golden records with the record hash and
the design hash checked in, against a scripted fixture binding."""

import asyncio
import dataclasses
import enum
import gc
import inspect
import os
import random
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Coroutine, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

import jurat
from counting import SmallBounds
from designs import minimal
from jurat import (
    Cause,
    CauseCode,
    Decision,
    EntryRefusal,
    LoadedDesign,
    Origin,
    Plan,
    PlanKind,
    Policy,
    Question,
    Record,
    Requirement,
    Rule,
    RunSettings,
    Source,
    Step,
    Text,
    VerdictKind,
    When,
    _decide,
    decide,
    decide_sync,
    escalation,
    load,
    read_record,
    redact,
)
from jurat import types as t
from jurat._decide import run_settings
from jurat._values import UNREPRESENTABLE
from jurat.helpers import pick_with_margin
from scripted import (
    Batch,
    Claims,
    NotACoroutine,
    Scripted,
    clashing,
    generator_coroutine,
    returns,
    started_raising_on_close,
    stop,
)

# ---------------------------------------------------------------------------------------------
# The slice design. The golden hashes below cover these functions' source: change one and the
# design hash changes.


def route(r: Any) -> Any:
    return pick_with_margin(r, "q:spam", 0.5, 0.1)


def echo(r: Any) -> Any:
    return r.read("t:state.subject")


def archive_policy(r: Any) -> Plan:
    if r.read("d:route") == "yes":
        step = Step(
            "archive",
            {"what": Origin(ref="d:route"), "folder": Origin(literal="spam")},
            [Requirement("d:route", "yes")],
        )
        return Plan("execute", [step])
    return Plan("inform", reply="kept")


def ok_policy(r: Any) -> Plan:
    return Plan("inform", reply="ok")


def escalate_policy(r: Any) -> Plan:
    return Plan("escalate", reason="not sure it is spam")


def cancel_policy(r: Any) -> Plan:
    raise asyncio.CancelledError


STATE_T = t.record(
    subject=t.string,
    to=t.string,
    blob=t.bytes_,
    weight=t.number(),
    sender=t.record(address=t.string),
)
ASK = Source(
    {"q:spam": Question("noul", Text("Is it spam?")), "t:body": t.string},
    needs=["t:state.subject"],
    cost=1,
)
LOOK = Source({"t:size": t.integer()}, needs=["t:state"])
RULES = {
    "route": Rule("d:route", t.label("no", "yes"), route, version="1", reads=["q:spam"]),
    "echo": Rule("d:echo", t.string, echo, version="1", reads=["t:state.subject", "t:body"]),
}


def slice_design(**changes: Any) -> LoadedDesign:
    fields: dict[str, Any] = {
        "state": STATE_T,
        "sources": {"ask": ASK, "look": LOOK},
        "rules": RULES,
        "policy": Policy(archive_policy, version="1", reads=["d:route"]),
        "tools": ["archive"],
    }
    fields.update(changes)
    return load(minimal(**fields))


LOADED = slice_design()
STATE: dict[str, Any] = {
    "subject": "hello",
    "to": "dana@example.com",
    "blob": b"\x00\x01",
    "weight": 3,
    "sender": {"address": "a@example.com"},
}
SPAM = {"q:spam": {"yes": 0.9, "no": 0.1}, "t:body": "text of the mail"}
REPLIES: dict[str, object] = {"ask": returns(SPAM), "look": returns({"t:size": 4})}

GOLDEN_DESIGN = "sha256:558100dabed470931fd6de70bb34a8912e75e6bb352ec7aa6999b709c5f508e5"
GOLDEN_RECORD = "sha256:1e2e1a36db84dcb8822375c6de871ad52a1ab7dec039ad6ee6aaf949d54e28bd"
GOLDEN_BROKEN = "sha256:f3c9e51fa57401f0d26b4f13e826030bdd462c7d72f6d56b8ad0848e2107ab44"
GOLDEN_CANCELLED = (
    "sha256:2105990ef18915c5655dd1a4cef563815fd5a3e4e1e68a707ce2e618c970c23a",
    "sha256:32681979bc8ee418cced4b8f01ec5cc25c598f901a472de5eef112c2bcd2afcd",
)
GOLDEN_REFUSED = (
    "sha256:d7d68420562bdc9fcaffb8539e7ca978b3c2f841760713e95355f8f91a8dddfe",
    "sha256:1755b2cef16dc58b4556f3233d1d1f53da392be8e06f98c7f11a48de5eb394ed",
)
GOLDEN_ESCALATE = (
    "sha256:bb7ef7e222dc08358034aa8b58bf50589dcb6690cf6462b66799d76a67dc102f",
    "sha256:96adb56a94308b54ce699eeceafa74e6b65bfe9bb2b720944bb70dc67967d16b",
)
"""Each a pair: the design hash, then the record hash."""


def scripted(**changes: object) -> Scripted:
    return Scripted({**REPLIES, **changes})


def bindings(**changes: object) -> dict[str, Any]:
    return {"default": scripted(**changes)}


def go(
    loaded: LoadedDesign = LOADED,
    state: object = STATE,
    binds: Mapping[str, Any] | None = None,
    settings: Any = None,
    **keywords: Any,
) -> Decision:
    return decide_sync(loaded, state, bindings() if binds is None else binds, settings, **keywords)


T = Any


def caught(coroutine: Coroutine[Any, Any, T]) -> BaseException:
    """Run a coroutine to the exception it raises, caught inside the loop's task, so the
    instance decide raised is the one returned."""

    async def main() -> BaseException:
        try:
            await coroutine
        except BaseException as exc:  # the test reads whatever decide raised
            return exc
        raise AssertionError("decide returned")

    return asyncio.run(main())


def deciding(
    loaded: LoadedDesign = LOADED,
    state: object = STATE,
    binds: Mapping[str, Any] | None = None,
    settings: Any = None,
    **keywords: Any,
) -> Coroutine[Any, Any, Decision]:
    return decide(loaded, state, bindings() if binds is None else binds, settings, **keywords)


def record_of(exc: BaseException) -> Record:
    record = getattr(exc, "record", None)
    assert type(record) is Record
    return record


def kinds(record: Record) -> list[str]:
    return [item["item"] for item in record.log]


class Seen:
    """An observer that keeps each index and item, raising ``raises`` at index ``at``."""

    def __init__(self, at: int | None = None, raises: BaseException | None = None) -> None:
        self.items: list[tuple[int, dict[str, Any]]] = []
        self.at = at
        self.raises = raises or RuntimeError("the observer failed")

    def __call__(self, index: int, item: dict[str, Any]) -> None:
        self.items.append((index, item))
        if index == self.at:
            raise self.raises


class Counting:
    """A codec that counts its calls and returns ``value``, or the state when none."""

    def __init__(self, value: object = None, raises: BaseException | None = None) -> None:
        self.calls: list[object] = []
        self.value = value
        self.raises = raises

    def __call__(self, state: object) -> object:
        self.calls.append(state)
        if self.raises is not None:
            raise self.raises
        return state if self.value is None else self.value


def roundtrips(record: Record) -> None:
    """A record decide wrote reads back with the same bytes, and both views take it."""
    assert read_record(record.document).document == record.document
    redact(record, b"\x07" * 32)
    escalation(record)


# ---------------------------------------------------------------------------------------------
# A returned turn and the golden records


def test_a_turn_returns_a_decision_whose_record_holds_every_stage() -> None:
    decision = go()
    record = decision.record
    assert kinds(record) == [
        "design",
        "settings",
        "entry",
        "call",
        "call",
        "entry",
        "entry",
        "entry",
        "entry",
        "entry",
        "round",
        "plan",
        "verdict",
    ]
    assert decision.plan.kind is PlanKind.execute
    assert decision.verdict.kind is VerdictKind.passed
    assert decision.design_hash == record.design_hash == LOADED.design_hash
    assert record.cause is None
    roundtrips(record)


def test_the_golden_record() -> None:
    record = go().record
    assert LOADED.design_hash == GOLDEN_DESIGN
    assert record.record_hash == GOLDEN_RECORD


def test_the_golden_broken_entry() -> None:
    record = go(state={"subject": "no other field"}).record
    assert record.record_hash == GOLDEN_BROKEN


def golden_cancelled() -> Record:
    """A policy raising a cancellation: every round kept, then the ``cause`` item. No timing reaches
    it."""
    loaded = slice_design(policy=Policy(cancel_policy, version="1", reads=["d:route"]))
    exc = caught(deciding(loaded))
    assert type(exc) is asyncio.CancelledError
    return record_of(exc)


def golden_refused() -> Record:
    return go(slice_design(requirements=[Requirement("d:route", "no")])).record


def golden_escalate() -> Record:
    return go(slice_design(policy=Policy(escalate_policy, version="1", reads=["d:route"]))).record


GOLDENS: dict[str, tuple[Callable[[], Record], tuple[str, str]]] = {
    "cancelled": (golden_cancelled, GOLDEN_CANCELLED),
    "refused": (golden_refused, GOLDEN_REFUSED),
    "escalate": (golden_escalate, GOLDEN_ESCALATE),
}


def test_the_golden_cancelled_record() -> None:
    record = golden_cancelled()
    assert record.log[-1] == {"item": "cause", "cause": {"code": "cancelled", "refs": []}}
    assert [i["item"] for i in record.log][-2:] == ["round", "cause"]
    assert record.cause == {"code": "cancelled", "refs": []}
    assert (record.design_hash, record.record_hash) == GOLDEN_CANCELLED


def test_the_golden_refused_verdict() -> None:
    record = golden_refused()
    assert record.log[-1] == {
        "item": "verdict",
        "kind": "refused",
        "failures": [
            {"step": "plan", "requirement": {"ref": "d:route", "value": "no"}, "ref": "d:route"}
        ],
    }
    view = escalation(record)
    assert view is not None
    assert view["cause"] == {"code": "refused", "refs": ["d:route"]}
    assert record.cause is None
    assert (record.design_hash, record.record_hash) == GOLDEN_REFUSED


def test_the_golden_escalate_plan() -> None:
    record = golden_escalate()
    plan = record.log[-2]
    assert plan["item"] == "plan"
    assert plan["kind"] == "escalate"
    assert plan["reason"] == "not sure it is spam"
    assert record.log[-1]["kind"] == "passed"
    view = escalation(record)
    assert view is not None
    assert view["reason"] == "not sure it is spam"
    assert (record.design_hash, record.record_hash) == GOLDEN_ESCALATE


_GOLDEN_PROBE = (
    "from test_decide import GOLDENS, LOADED, go\n"
    "print(LOADED.design_hash, go().record.record_hash,"
    " go(state={'subject': 'no other field'}).record.record_hash)\n"
    "for name, (make, _) in GOLDENS.items():\n"
    "    record = make()\n"
    "    print(name, record.design_hash, record.record_hash)\n"
)


@pytest.mark.parametrize("seed", ["0", "12345"])
def test_the_golden_hashes_hold_under_every_hash_seed(seed: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _GOLDEN_PROBE],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    assert result.stdout.splitlines() == [
        f"{GOLDEN_DESIGN} {GOLDEN_RECORD} {GOLDEN_BROKEN}",
        *(f"{name} {pair[0]} {pair[1]}" for name, (_, pair) in GOLDENS.items()),
    ]


def test_the_design_and_settings_items_bytes() -> None:
    log = go().record.log
    assert log[0] == {"item": "design", "hash": LOADED.design_hash, "format": 1, "helpers": 1}
    assert log[1] == {
        "item": "settings",
        "settings": {
            "round_bound": 8,
            "timeout": 60,
            "budget": None,
            "deadline": None,
            "certificate_cap": None,
        },
        "providers": dict(LOADED._providers),
        "classes": LOADED._classes,
    }
    assert "t:state" not in log[1]["providers"]
    assert "t:state" in log[1]["classes"]


def test_no_settings_writes_the_bytes_of_the_default_settings() -> None:
    assert go().record.record_hash == go(settings=RunSettings()).record.record_hash


def test_settings_in_force_are_recorded() -> None:
    record = go(settings=RunSettings(round_bound=1, timeout=0.5)).record
    assert record.log[1]["settings"]["round_bound"] == 1
    assert record.log[1]["settings"]["timeout"] == 0.5


def test_shuffled_latencies_give_identical_records() -> None:
    sources = {f"s{i}": Source({f"t:v{i}": t.string}) for i in range(5)}
    loaded = load(
        minimal(
            state=t.record(),
            sources=sources,
            policy=Policy(ok_policy, version="1"),
        )
    )
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
            return returns(dict.fromkeys(batch[0].refs, "v"))

        decision = decide_sync(loaded, {}, {"default": Scripted(dict.fromkeys(sources, reply))})
        hashes.add(decision.record.record_hash)
        orders.add(tuple(finished))
    assert len(orders) > 1
    assert len(hashes) == 1


# ---------------------------------------------------------------------------------------------
# The decision


def test_a_decision_is_built_only_by_decide() -> None:
    decision = go()
    with pytest.raises(TypeError, match="only by decide"):
        Decision(object(), decision.plan, decision.verdict, decision.record)


def test_a_decision_is_frozen() -> None:
    decision = go()
    with pytest.raises(AttributeError):
        decision.plan = decision.plan  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del decision.record  # type: ignore[misc]
    with pytest.raises(AttributeError):
        decision.extra = 1  # type: ignore[attr-defined]


def test_a_decision_has_only_its_four_public_attributes() -> None:
    public = {n for n in dir(Decision) if not n.startswith("_")}
    assert public == {"plan", "verdict", "record", "design_hash"}


def test_the_design_hash_is_derived_from_the_record() -> None:
    decision = go()
    assert "design_hash" not in Decision.__slots__
    assert decision.design_hash == decision.record.log[0]["hash"]
    assert "Decision(plan=" in repr(decision)


def test_a_refused_plan_is_kept_beside_the_verdict() -> None:
    loaded = slice_design(requirements=[Requirement("d:route", "no")])
    decision = go(loaded)
    assert decision.plan.kind is PlanKind.execute
    assert decision.verdict.kind is VerdictKind.refused


# ---------------------------------------------------------------------------------------------
# The entry refusals


class Weird(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Any:
        raise RuntimeError("cannot iterate")

    def __len__(self) -> int:
        return 1


class Classless:
    @property
    def __class__(self) -> type:  # type: ignore[override]
        raise RuntimeError("no class")


class Sub(RunSettings):
    pass


class Three(enum.IntEnum):
    three = 3


class Called:
    """Callable through its type's ``__call__``; nothing else of it is read."""

    def __call__(self, *args: object) -> object:
        return args[-1]


class Untouchable:
    """Not callable, with a raising ``__class__`` and attribute read: callable() reads its
    type's call slot alone."""

    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"the argument was read: {name}")


def instance_call() -> object:
    """An object with ``__call__`` set on the instance, which ``callable`` does not count."""

    class Bare:
        pass

    bare = Bare()
    bare.__call__ = lambda *args: None  # type: ignore[attr-defined]
    return bare


class Odd(str):
    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return other is self


REFUSED: list[tuple[str, dict[str, Any]]] = [
    ("a Design", {"loaded": minimal()}),
    ("None loaded", {"loaded": None}),
    ("a dict of settings", {"settings": {"round_bound": 8}}),
    ("a RunSettings subclass", {"settings": Sub()}),
    ("a list of bindings", {"binds": [("default", None)]}),
    ("a mapping whose copy raises", {"binds": Weird()}),
    ("a raising __class__", {"binds": Classless()}),
    ("no default binding", {"binds": {}}),
    ("an undeclared batch key", {"binds": {"default": scripted(), "mail": scripted()}}),
    ("a key that is not a string", {"binds": {"default": scripted(), 1: scripted()}}),
    ("two keys of one string data", {"binds": {"default": scripted(), Odd("default"): 1}}),
    ("a non-null certificate cap", {"settings": RunSettings(certificate_cap=1)}),
    ("a codec that is an int", {"codec": 5}),
    ("a codec with an instance __call__", {"codec": instance_call()}),
    ("a codec that cannot be read", {"codec": Untouchable()}),
    ("an observer that is a str", {"observer": "print"}),
    ("an observer with an instance __call__", {"observer": instance_call()}),
    ("an observer that cannot be read", {"observer": Untouchable()}),
]


@pytest.mark.parametrize(("case", "arguments"), REFUSED, ids=[c for c, _ in REFUSED])
def test_an_entry_refusal_writes_no_record_and_calls_no_observer_or_codec(
    case: str, arguments: dict[str, Any]
) -> None:
    seen, codec = Seen(), Counting()
    keywords: dict[str, Any] = {"observer": seen, "codec": codec, **arguments}
    exc = caught(deciding(**keywords))
    assert type(exc) is EntryRefusal
    assert isinstance(exc, ValueError)
    assert not hasattr(exc, "record")
    assert seen.items == []
    assert codec.calls == []


def test_a_codec_and_an_observer_callable_by_their_type_are_taken() -> None:
    decision = go(state=object(), codec=Called(), observer=Called())
    assert decision.plan.kind is PlanKind.broken  # Called returns the state object
    assert decision.record.unhashed["codec"] == "Called"


@pytest.mark.parametrize(
    "settings",
    [
        RunSettings(round_bound=True),
        RunSettings(round_bound=1.0),  # type: ignore[arg-type]
        RunSettings(round_bound=Three.three),
        RunSettings(round_bound=0),
        RunSettings(round_bound=2**53),
        RunSettings(timeout=float("nan")),
        RunSettings(timeout=float("inf")),
        RunSettings(timeout=0),
        RunSettings(timeout=-1.5),
        RunSettings(timeout=True),
        RunSettings(timeout="60"),  # type: ignore[arg-type]
        RunSettings(timeout=2**53),
        RunSettings(budget=-1),
        RunSettings(budget=float("nan")),
        RunSettings(budget=True),
        RunSettings(deadline=0),
        RunSettings(deadline=-0.5),
        RunSettings(deadline=float("inf")),
        RunSettings(certificate_cap=True),  # type: ignore[arg-type]
        RunSettings(certificate_cap=0),
    ],
    ids=repr,
)
def test_a_setting_outside_its_bounds_is_refused(settings: RunSettings) -> None:
    with pytest.raises(EntryRefusal):
        go(settings=settings)


def test_settings_at_their_bounds_are_accepted() -> None:
    record = go(settings=RunSettings(round_bound=2**53 - 1, timeout=2**53 - 1)).record
    assert record.log[1]["settings"]["round_bound"] == 2**53 - 1
    run = run_settings(1, 1e-9, 0, 1e300, None)
    assert run.item() == {
        "round_bound": 1,
        "timeout": 1e-9,
        "budget": 0,
        "deadline": 1e300,
        "certificate_cap": None,
    }


def test_an_out_of_bounds_budget_is_an_entry_refusal() -> None:
    exc = caught(deciding(settings=RunSettings(budget=-1)))
    assert type(exc) is EntryRefusal


def test_run_settings_validates_nothing() -> None:
    assert RunSettings(round_bound="x").round_bound == "x"  # type: ignore[arg-type]
    assert RunSettings() == RunSettings(8, 60, None, None, None)


def test_a_settings_object_of_another_type_is_refused_before_any_value_is_read() -> None:
    reads: list[str] = []

    class Counted:
        def __getattribute__(self, name: str) -> object:
            reads.append(name)
            return object.__getattribute__(RunSettings(), name)

    exc = caught(deciding(settings=Counted()))
    assert type(exc) is EntryRefusal
    assert reads == []


def test_a_design_with_no_source_needs_no_binding_and_takes_default() -> None:
    loaded = load(minimal())
    assert go(loaded, {}, {}).plan.kind is PlanKind.broken  # the minimal policy returns None
    assert go(loaded, {}, {"default": object()}).record.log[2]["ref"] == "t:state"


def test_a_batch_key_needs_its_binding_and_default_only_for_a_keyless_source() -> None:
    keyed = slice_design(sources={"ask": ASK, "look": Source({"t:size": t.integer()}, batch="m")})
    with pytest.raises(EntryRefusal, match='"m"'):
        go(keyed, binds={"default": scripted()})
    only = slice_design(
        sources={"look": Source({"t:size": t.integer()}, batch="m")},
        rules={},
        policy=Policy(archive_policy, version="1"),
    )
    # Every binding is given, so entry passes and the key's binding is called.
    binding = Scripted({"look": returns({"t:size": 4})})
    log = go(only, binds={"m": binding}).record.log
    assert [i["sources"] for i in log if i["item"] == "call"] == [["look"]]
    assert [b[0].id for b in binding.batches] == ["look"]


def test_a_str_subclass_key_is_its_string_data() -> None:
    class Key(str):
        pass

    decision = go(binds={Key("default"): scripted()})
    assert decision.verdict.kind is VerdictKind.passed


def test_two_keys_of_one_string_data_are_refused() -> None:
    with pytest.raises(EntryRefusal, match="two bindings keys"):
        go(binds={"default": scripted(), Odd("default"): scripted()})


def test_the_bindings_are_copied_once_and_only_the_copy_is_read() -> None:
    binds: dict[str, Any] = bindings()
    first = binds["default"]

    async def swap(batch: Batch) -> object:
        binds["default"] = Scripted({})
        return returns(SPAM)

    first.replies["ask"] = swap
    decision = go(binds=binds)
    assert decision.verdict.kind is VerdictKind.passed
    assert len(first.batches) == 2


# ---------------------------------------------------------------------------------------------
# The state entry


def state_value(record: Record) -> Any:
    item = record.log[2]
    assert item["ref"] == "t:state"
    return item["value"]


def test_the_state_entry_drops_undeclared_fields_at_every_depth() -> None:
    state = {**STATE, "extra": 1, "sender": {"address": "a@example.com", "name": "A"}}
    value = state_value(go(state=state).record)
    assert value == {
        "subject": "hello",
        "to": "dana@example.com",
        "blob": "AAE",
        "weight": 3,
        "sender": {"address": "a@example.com"},
    }


def test_bytes_are_base64url_in_the_record_and_bytes_and_floats_live() -> None:
    seen = Seen()
    binding = scripted()
    record = go(binds={"default": binding}, observer=seen).record
    assert state_value(record)["blob"] == "AAE"
    (look,) = [b[0] for b in binding.batches if b[0].id == "look"]
    live = cast("dict[str, Any]", look.needs["t:state"])
    assert live["blob"] == b"\x00\x01"
    assert type(live["weight"]) is float
    assert type(seen.items[2][1]["value"]["weight"]) is float  # the stored copy
    assert record.document.count(b'"weight":3') == 1


def test_mutating_the_state_after_entry_changes_nothing() -> None:
    state = {**STATE, "sender": {"address": "a@example.com"}}

    async def mutate(batch: Batch) -> object:
        state["subject"] = "changed"
        cast("dict[str, Any]", state["sender"])["address"] = "x"
        return returns(SPAM)

    record = go(state=state, binds=bindings(ask=mutate)).record
    assert state_value(record)["subject"] == "hello"
    (echoed,) = [i for i in record.log if i["item"] == "entry" and i["ref"] == "d:echo"]
    assert echoed["value"] == "hello"


def test_the_codecs_value_takes_an_int_as_a_float_and_drops_an_undeclared_field() -> None:
    value = {**STATE, "weight": 3, "sender": {"address": "a@example.com", "name": "A"}}
    seen = Seen()
    record = go(state=object(), codec=Counting(value=value), observer=seen).record
    assert state_value(record)["sender"] == {"address": "a@example.com"}
    stored = seen.items[2][1]["value"]
    assert type(stored["weight"]) is float
    assert stored["sender"] == {"address": "a@example.com"}
    assert record.document.count(b'"weight":3') == 1


def test_the_codec_returning_the_callers_object_is_copied_too() -> None:
    state = dict(STATE)
    codec = Counting(value=state)
    seen = Seen()
    go(state=object(), codec=codec, observer=seen)
    state["subject"] = "after"
    assert seen.items[2][1]["value"]["subject"] == "hello"


class Big(enum.IntEnum):
    one = 1


class Plain(enum.StrEnum):
    hello = "hello"


class DictSub(dict[str, Any]):
    pass


NONCONFORMING: list[tuple[str, object]] = [
    ("a missing field", {k: v for k, v in STATE.items() if k != "to"}),
    ("not a record", ["hello"]),
    ("a bool under number", {**STATE, "weight": True}),
    ("an IntEnum under number", {**STATE, "weight": Big.one}),
    ("NaN under number", {**STATE, "weight": float("nan")}),
    ("an int past 2^53 - 1", {**STATE, "weight": 2**53}),
    ("a bytearray under bytes", {**STATE, "blob": bytearray(b"\x00")}),
    ("a base64 string under bytes", {**STATE, "blob": "AAE"}),
    ("a dict subclass", DictSub(STATE)),
    ("a nested dict subclass", {**STATE, "sender": DictSub(address="a")}),
    ("a lone surrogate", {**STATE, "subject": "\ud800"}),
]


@pytest.mark.parametrize(("case", "state"), NONCONFORMING, ids=[c for c, _ in NONCONFORMING])
def test_a_nonconforming_state_is_a_broken_entry_naming_t_state(case: str, state: object) -> None:
    for codec in (None, Counting(value=state)):
        decision = go(state=state if codec is None else object(), codec=codec)
        assert decision.plan.cause == Cause(CauseCode.invalid_return, ["t:state"])
        assert kinds(decision.record) == ["design", "settings", "plan", "verdict"]
        assert "exceptions" not in decision.record.unhashed


def test_conforming_edge_values_are_accepted() -> None:
    state = {**STATE, "subject": Plain.hello, "weight": 0.5}
    assert state_value(go(state=state).record)["subject"] == "hello"
    tupled = load(
        minimal(
            state=t.record(tags=t.list_(t.string)),
            policy=Policy(ok_policy, version="1"),
        )
    )
    assert state_value(go(tupled, {"tags": ("a", "b")}, {}).record) == {"tags": ["a", "b"]}


def test_an_absent_nullable_field_does_not_conform_and_a_present_null_does() -> None:
    loaded = load(
        minimal(
            state=t.record(note=t.nullable(t.string)),
            policy=Policy(ok_policy, version="1"),
        )
    )
    assert go(loaded, {}, {}).plan.kind is PlanKind.broken
    assert state_value(go(loaded, {"note": None}, {}).record) == {"note": None}


def test_an_empty_record_state_holds_an_empty_object() -> None:
    loaded = load(minimal(policy=Policy(ok_policy, version="1")))
    assert state_value(go(loaded, {"x": 1}, {}).record) == {}


# ---------------------------------------------------------------------------------------------
# The codec and the broken entry

BROKEN_ITEM: dict[str, Any] = {
    "item": "plan",
    "kind": "broken",
    "steps": [],
    "reason": None,
    "reply": None,
    "cause": {"code": "raised", "refs": ["codec"]},
}


def test_the_codec_is_called_once_with_the_state_object() -> None:
    state = object()
    codec = Counting(value=STATE)
    decision = go(state=state, codec=codec)
    assert codec.calls == [state]
    assert decision.verdict.kind is VerdictKind.passed


def test_an_async_codecs_coroutine_is_a_broken_entry_and_no_warning_leaks() -> None:
    async def codec(state: object) -> object:
        return STATE

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        decision = go(state=object(), codec=cast(Any, codec))
        gc.collect()
    assert decision.plan.kind is PlanKind.broken
    assert decision.plan.cause == Cause(CauseCode.invalid_return, ["t:state"])
    assert kinds(decision.record) == ["design", "settings", "plan", "verdict"]
    assert [str(w.message) for w in caught_warnings] == []


@pytest.fixture
def untyped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decide's type test of the codec's return fails if reached: an awaitable is refused
    before it."""

    def never(value: object, type_: object) -> object:
        raise AssertionError("the return reached the type test")

    monkeypatch.setattr(_decide, "typed", never)


@pytest.mark.usefixtures("untyped")
def test_a_sync_codec_returning_a_coroutine_is_a_broken_entry_and_no_warning_leaks() -> None:
    async def inner(state: object) -> object:
        return STATE

    def codec(state: object) -> object:
        return inner(state)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        decision = go(state=object(), codec=codec)
        gc.collect()
    assert decision.plan.cause == Cause(CauseCode.invalid_return, ["t:state"])
    assert "exceptions" not in decision.record.unhashed
    assert [str(w.message) for w in caught_warnings] == []


@pytest.mark.usefixtures("untyped")
def test_a_codec_returning_an_awaitable_is_a_broken_entry_and_it_is_not_closed() -> None:
    awaitable = NotACoroutine()

    def codec(state: object) -> object:
        return awaitable

    decision = go(state=object(), codec=codec)
    assert decision.plan.cause == Cause(CauseCode.invalid_return, ["t:state"])
    assert not awaitable.closed


@pytest.mark.usefixtures("untyped")
def test_a_codecs_generator_coroutine_is_a_broken_entry_closed_at_the_close() -> None:
    ran: list[str] = []
    generator = generator_coroutine(ran, started=True)

    def codec(state: object) -> object:
        return generator

    decision = go(state=object(), codec=codec)
    assert decision.plan.cause == Cause(CauseCode.invalid_return, ["t:state"])
    assert inspect.getgeneratorstate(generator) == inspect.GEN_CLOSED
    assert ran == ["finally"]


@pytest.mark.usefixtures("untyped")
def test_a_codecs_coroutine_re_raising_a_kept_stop_on_close_is_a_broken_entry() -> None:
    def codec(state: object) -> object:
        return started_raising_on_close(stop())

    decision = go(state=object(), codec=codec)
    assert decision.plan.cause == Cause(CauseCode.invalid_return, ["t:state"])
    assert "exceptions" not in decision.record.unhashed


@pytest.mark.parametrize("make_", [Claims, lambda: clashing(stop())], ids=["claims", "kept-stop"])
def test_a_codecs_return_that_is_no_awaitable_by_its_type_meets_the_type_test(
    make_: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []
    typed = _decide.typed

    def spy(value: object, type_: Any) -> Any:
        seen.append(value)
        return typed(value, type_)

    monkeypatch.setattr(_decide, "typed", spy)
    returned = make_()

    def codec(state: object) -> object:
        return returned

    decision = go(state=object(), codec=codec)
    assert decision.plan.cause == Cause(CauseCode.invalid_return, ["t:state"])
    assert seen == [returned]
    assert "exceptions" not in decision.record.unhashed


def test_a_codecs_coroutine_raising_on_close_keeps_the_broken_entry_and_its_text() -> None:
    def codec(state: object) -> object:
        return started_raising_on_close("on close")

    decision = go(state=object(), codec=codec)
    assert decision.plan.cause == Cause(CauseCode.invalid_return, ["t:state"])
    text = decision.record.unhashed["exceptions"]["2"]
    assert (text["type"], text["message"]) == ("ValueError", "on close")
    roundtrips(decision.record)


def test_a_codec_return_whose_awaitable_test_raises_is_the_codecs_raise() -> None:
    def codec(state: object) -> object:
        return clashing()

    decision = go(state=object(), codec=codec)
    assert decision.plan.cause == Cause(CauseCode.raised, ["codec"])
    text = decision.record.unhashed["exceptions"]["2"]
    assert (text["type"], text["message"]) == ("ValueError", "the type test raised")


def test_a_raising_codec_is_a_broken_entry_with_its_text_under_the_plan_index() -> None:
    codec = Counting(raises=ValueError("bad state"))
    binding = scripted()
    calls: list[object] = []

    def counted(r: Any) -> Plan:
        calls.append(r)
        return archive_policy(r)

    loaded = slice_design(policy=Policy(counted, version="1", reads=["d:route"]))
    decision = go(loaded, codec=codec, binds={"default": binding})
    record = decision.record
    assert record.log[2:] == [
        BROKEN_ITEM,
        {"item": "verdict", "kind": "passed", "failures": []},
    ]
    assert decision.plan.kind is PlanKind.broken
    assert decision.plan.cause == Cause(CauseCode.raised, ["codec"])
    assert (decision.plan.steps, decision.plan.reason, decision.plan.reply) == ((), None, None)
    assert decision.verdict.kind is VerdictKind.passed
    assert record.cause == {"code": "raised", "refs": ["codec"]}
    text = record.unhashed["exceptions"]["2"]
    assert (text["type"], text["message"]) == ("ValueError", "bad state")
    assert "ValueError: bad state" in text["traceback"]
    assert calls == []
    assert binding.batches == []
    roundtrips(record)


def test_a_broken_entry_passes_the_guard_whatever_the_design_requires() -> None:
    loaded = slice_design(requirements=[Requirement("d:route", "no")])
    decision = go(loaded, codec=Counting(raises=ValueError("x")))
    assert decision.verdict.kind is VerdictKind.passed


def test_a_codec_exception_whose_text_raises_is_unrepresentable() -> None:
    class Mute(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no text")

    record = go(codec=Counting(raises=Mute())).record
    assert record.unhashed["exceptions"]["2"]["message"] == UNREPRESENTABLE


def test_a_codecs_keyboard_interrupt_propagates_with_no_record() -> None:
    seen = Seen()
    exc = caught(deciding(codec=Counting(raises=KeyboardInterrupt()), observer=seen))
    assert type(exc) is KeyboardInterrupt
    assert not hasattr(exc, "record")
    assert [item["item"] for _, item in seen.items] == ["design", "settings"]


def test_an_identity_codec_and_no_codec_write_one_log() -> None:
    plain = go().record

    def identity(state: object) -> object:
        return state

    coded = go(codec=identity).record
    assert coded.record_hash == plain.record_hash
    assert coded.log == plain.log
    assert "codec" not in plain.unhashed
    assert (
        coded.unhashed["codec"]
        == "test_an_identity_codec_and_no_codec_write_one_log.<locals>.identity"
    )


def named(name: object) -> Any:
    class Codec:
        def __call__(self, state: object) -> object:
            return state

    codec = Codec()
    object.__setattr__(codec, "__qualname__", name)
    return codec


class Raising:
    def __getattr__(self, name: str) -> object:
        if name == "__qualname__":
            raise RuntimeError("no name")
        raise AttributeError(name)

    def __call__(self, state: object) -> object:
        return state


def encode_state(state: object) -> object:
    return state


@pytest.mark.parametrize(
    ("codec", "name"),
    [
        (encode_state, "encode_state"),
        (named("custom"), "custom"),
        (named(5), "named.<locals>.Codec"),
        (Counting(), "Counting"),
        (Raising(), UNREPRESENTABLE),
    ],
    ids=["function", "set", "not a str", "instance", "raising"],
)
def test_the_codecs_name_in_unhashed(codec: Any, name: str) -> None:
    assert go(codec=codec).record.unhashed["codec"] == name


def test_a_cancelled_error_from_the_codec_is_a_delivered_cancellation() -> None:
    exc = caught(deciding(codec=Counting(raises=asyncio.CancelledError())))
    assert type(exc) is asyncio.CancelledError
    record = record_of(exc)
    assert record.log[2:] == [{"item": "cause", "cause": {"code": "cancelled", "refs": []}}]
    assert record.cause == {"code": "cancelled", "refs": []}
    assert set(record.unhashed["turn"]) == {"start", "end"}
    roundtrips(record)


# ---------------------------------------------------------------------------------------------
# The observer's exception


def test_the_observer_sees_every_item_and_the_record_is_the_same_without_one() -> None:
    seen = Seen()
    record = go(observer=seen).record
    assert [index for index, _ in seen.items] == list(range(len(record.log)))
    assert [item for _, item in seen.items] == record.log
    assert go().record.record_hash == record.record_hash


NORMAL_LOG = go().record.log
BROKEN_LOG = go(codec=Counting(raises=ValueError("bad"))).record.log


@pytest.mark.parametrize("at", range(len(NORMAL_LOG)))
def test_the_observer_raising_at_any_item_ends_the_turn_with_the_record_so_far(at: int) -> None:
    raised = RuntimeError("the observer failed")
    exc = caught(deciding(observer=Seen(at, raised)))
    assert exc is raised
    record = record_of(exc)
    assert record.log == NORMAL_LOG[: at + 1]
    assert "cause" not in kinds(record)
    assert record.cause is None
    assert set(record.unhashed["turn"]) == {"start", "end"}
    assert str(at) not in record.unhashed.get("calls", {})  # none of that item's members
    roundtrips(record)


@pytest.mark.parametrize("at", range(len(BROKEN_LOG)))
def test_the_observer_raising_at_a_broken_entry_keeps_the_record_so_far(at: int) -> None:
    exc = caught(deciding(observer=Seen(at), codec=Counting(raises=ValueError("bad"))))
    record = record_of(exc)
    assert record.log == BROKEN_LOG[: at + 1]
    exceptions = record.unhashed.get("exceptions", {})
    assert ("2" in exceptions) == (at > 2)  # none of the plan's members when it raised there
    roundtrips(record)


def test_the_observer_raising_at_a_broken_plan_is_not_chained_to_the_codecs_exception() -> None:
    raised = RuntimeError("the observer failed")
    exc = caught(deciding(observer=Seen(2, raised), codec=Counting(raises=ValueError("bad"))))
    assert exc is raised
    assert exc.__context__ is None
    assert record_of(exc).log[-1]["kind"] == "broken"


def test_the_unhashed_members_precede_the_design_item() -> None:
    exc = caught(deciding(observer=Seen(0), codec=encode_state))
    unhashed = record_of(exc).unhashed
    assert set(unhashed) == {"turn", "engine", "codec"}
    assert record_of(exc).log == NORMAL_LOG[:1]


@pytest.mark.parametrize("raises", [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()])
def test_any_base_exception_the_observer_raises_is_its_own_with_no_cause_item(
    raises: BaseException,
) -> None:
    exc = caught(deciding(observer=Seen(1, raises)))
    assert exc is raises
    record = record_of(exc)
    assert kinds(record) == ["design", "settings"]
    assert record.cause is None


# ---------------------------------------------------------------------------------------------
# Cancellation

CAUSE_ITEM: dict[str, Any] = {"item": "cause", "cause": {"code": "cancelled", "refs": []}}


def cancelled_mid_turn(observer: Any = None) -> BaseException:
    async def main() -> BaseException:
        started = asyncio.Event()

        async def ask(batch: Any) -> object:
            started.set()
            await asyncio.sleep(600)
            return returns(SPAM)

        task = asyncio.create_task(deciding(binds=bindings(ask=ask), observer=observer))
        await started.wait()  # the call is under way
        task.cancel()
        try:
            await task
        except asyncio.CancelledError as exc:
            return exc
        raise AssertionError("decide returned")

    return asyncio.run(main())


def test_a_cancellation_in_interpret_carries_the_record_ending_in_the_cause_item() -> None:
    record = record_of(cancelled_mid_turn())
    assert record.log == [*NORMAL_LOG[:3], CAUSE_ITEM]
    assert record.cause == {"code": "cancelled", "refs": []}
    assert set(record.unhashed) == {"turn", "engine"}
    roundtrips(record)
    assert escalation(record) is None


def cancelling(r: Any) -> Any:
    raise asyncio.CancelledError


def test_a_rule_raising_a_cancellation_in_round_1_keeps_the_rounds_items_so_far() -> None:
    rules = {
        **RULES,
        "route": Rule("d:route", t.label("no", "yes"), cancelling, version="1", reads=["q:spam"]),
    }
    exc = caught(deciding(slice_design(rules=rules)))
    assert type(exc) is asyncio.CancelledError
    record = record_of(exc)
    log = record.log
    assert log[-1] == CAUSE_ITEM
    round_1 = log[3:-1]
    assert round_1
    assert all(item.get("round") == 1 for item in round_1)
    assert {item["item"] for item in round_1} == {"call", "entry"}  # no round item
    assert "round" not in kinds(record)
    assert record.cause == {"code": "cancelled", "refs": []}
    roundtrips(record)


def test_a_policy_raising_a_cancellation_keeps_every_round() -> None:
    exc = caught(deciding(slice_design(policy=Policy(cancelling, version="1", reads=["d:route"]))))
    assert type(exc) is asyncio.CancelledError
    record = record_of(exc)
    assert record.log[1:-1] == NORMAL_LOG[1 : len(record.log) - 1]  # another design hash
    assert kinds(record)[-2:] == ["round", "cause"]
    assert "plan" not in kinds(record)
    roundtrips(record)


class CancellingRepr:
    """A returned key whose ``repr`` would cancel; the engine names its type instead."""

    def __repr__(self) -> str:
        raise asyncio.CancelledError


def test_a_returned_keys_own_repr_is_never_run() -> None:
    reply = returns({**SPAM, CancellingRepr(): 1})  # type: ignore[dict-item]
    record = go(binds=bindings(ask=reply)).record
    unexpected = [c.get("unexpected") for c in record.unhashed["calls"].values()]
    assert ["<test_decide.CancellingRepr>"] in unexpected
    roundtrips(record)


def test_a_cancellation_before_the_body_runs_carries_no_record() -> None:
    seen, codec = Seen(), Counting()

    async def main() -> BaseException:
        task = asyncio.create_task(deciding(observer=seen, codec=codec))
        task.cancel()
        try:
            await task
        except asyncio.CancelledError as exc:
            return exc
        raise AssertionError("decide returned")

    exc = asyncio.run(main())
    assert not hasattr(exc, "record")
    assert seen.items == []
    assert codec.calls == []


def test_an_existing_record_is_overwritten() -> None:
    raised = asyncio.CancelledError()
    raised.record = "an outer turn's"  # type: ignore[attr-defined]
    exc = caught(deciding(codec=Counting(raises=raised)))
    assert exc is raised
    assert record_of(exc).log[-1] == CAUSE_ITEM


def test_a_cancellation_that_refuses_the_attribute_is_re_raised_unchanged() -> None:
    class Stubborn(asyncio.CancelledError):
        def __setattr__(self, name: str, value: object) -> None:
            raise AttributeError(name)

    raised = Stubborn()
    exc = caught(deciding(codec=Counting(raises=raised)))
    assert exc is raised
    assert not hasattr(exc, "record")


@pytest.mark.parametrize("raises", [RuntimeError("obs"), KeyboardInterrupt()])
def test_the_observer_raising_at_the_cause_item_loses_to_the_cancellation(
    raises: BaseException,
) -> None:
    cancel = asyncio.CancelledError()
    exc = caught(deciding(codec=Counting(raises=cancel), observer=Seen(2, raises)))
    assert exc is cancel
    assert exc.__context__ is raises
    assert record_of(exc).log[-1] == CAUSE_ITEM


def test_the_observer_at_the_cause_item_of_a_cancellation_in_interpret() -> None:
    exc = cancelled_mid_turn(observer=Seen(3))
    assert type(exc.__context__) is RuntimeError
    assert record_of(exc).log[-1] == CAUSE_ITEM


def test_a_base_exception_from_a_binding_propagates_bare() -> None:
    async def interrupt(batch: Batch) -> object:
        raise SystemExit(3)

    exc = caught(deciding(binds=bindings(ask=interrupt)))
    assert type(exc) is SystemExit
    assert not hasattr(exc, "record")


# ---------------------------------------------------------------------------------------------
# decide_sync


def test_decide_sync_inside_a_running_loop_raises_before_doing_anything() -> None:
    seen, codec = Seen(), Counting()

    async def main() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            decide_sync(cast(Any, None), STATE, {}, codec=codec, observer=seen)

    asyncio.run(main())
    assert seen.items == []
    assert codec.calls == []


def test_decide_sync_returns_a_decision_holding_the_plan_and_verdict() -> None:
    decision = decide_sync(LOADED, STATE, bindings())
    assert type(decision) is Decision
    assert kinds(decision.record)[-2:] == ["plan", "verdict"]


def test_decide_is_awaitable_and_equals_decide_sync() -> None:
    decision = asyncio.run(decide(LOADED, STATE, bindings()))
    assert decision.record.record_hash == go().record.record_hash


# ---------------------------------------------------------------------------------------------
# unhashed


def test_unhashed_holds_the_turn_the_engine_and_the_calls() -> None:
    unhashed = go().record.unhashed
    assert set(unhashed) == {"turn", "engine", "calls"}
    assert unhashed["engine"] == jurat.__version__
    turn = unhashed["turn"]
    assert set(turn) == {"start", "end"}
    assert turn["start"].endswith("Z")
    assert turn["end"] >= turn["start"]
    assert set(unhashed["calls"]) == {"3", "4"}


def test_a_codec_adds_only_its_name() -> None:
    assert set(go(codec=encode_state).record.unhashed) == {"turn", "engine", "calls", "codec"}


# ---------------------------------------------------------------------------------------------
# Records decide writes, read back and viewed

SCENARIOS: dict[str, Callable[[], Record]] = {
    "returned": lambda: go().record,
    "refused": lambda: go(slice_design(requirements=[Requirement("d:route", "no")])).record,
    "broken": lambda: go(codec=Counting(raises=ValueError("x"))).record,
    "invalid": lambda: go(state={}).record,
    "cancelled": lambda: record_of(cancelled_mid_turn()),
    "observer-ended": lambda: record_of(caught(deciding(observer=Seen(5)))),
    "design only": lambda: record_of(caught(deciding(observer=Seen(0)))),
}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_record_decide_writes_reads_back_and_takes_both_views(scenario: str) -> None:
    roundtrips(SCENARIOS[scenario]())


def test_a_when_raising_a_cancellation_in_round_2_keeps_round_1_whole() -> None:
    def cancels(r: Any) -> Any:
        raise asyncio.CancelledError

    later = Source({"t:later": t.string}, needs=["d:route"], when=When(cancels, version="1"))
    exc = caught(deciding(slice_design(sources={"ask": ASK, "look": LOOK, "later": later})))
    assert type(exc) is asyncio.CancelledError
    record = record_of(exc)
    assert record.log[-1] == CAUSE_ITEM
    round_1 = record.log[3:-1]
    assert {item["round"] for item in round_1} == {1}
    assert round_1[-1]["item"] == "round"
    roundtrips(record)


# ---------------------------------------------------------------------------------------------
# The budget and the deadline through decide


class Slow:
    """A codec that takes ``seconds`` of the turn's time."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __call__(self, state: object) -> object:
        time.sleep(self.seconds)
        return state


def rounds(record: Record) -> list[dict[str, Any]]:
    return [i for i in record.log if i["item"] == "round"]


def test_the_codecs_time_counts_against_the_deadline() -> None:
    binding = scripted()
    record = go(binds={"default": binding}, settings=RunSettings(deadline=0.01), codec=Slow(0.05))
    assert binding.batches == []
    assert rounds(record.record) == [{"item": "round", "round": 1, "cost": 0, "deadline": True}]
    log = record.record.log
    marker: dict[str, Any] = {"kind": "Unasked", "cause": {"code": "deadline", "refs": []}}
    assert [
        i["marker"] for i in log if i["item"] == "entry" and i["ref"] in ("q:spam", "t:size")
    ] == [
        marker,
        marker,
    ]
    roundtrips(record.record)


def test_a_budget_within_its_bounds_runs_and_cuts_the_frontier_over_it() -> None:
    binding = scripted()
    record = go(binds={"default": binding}, settings=RunSettings(budget=0.5)).record
    assert binding.batches == []
    assert rounds(record) == [{"item": "round", "round": 1, "cost": 0, "deadline": False}]
    assert record.log[1]["settings"]["budget"] == 0.5
    record = go(settings=RunSettings(budget=1, deadline=600)).record
    assert rounds(record) == [{"item": "round", "round": 1, "cost": 1, "deadline": False}]


def test_a_cancellation_during_a_deadline_bounded_call_records_cancelled_and_no_call() -> None:
    async def main() -> BaseException:
        started = asyncio.Event()

        async def ask(batch: Any) -> object:
            started.set()
            await asyncio.sleep(600)
            return returns(SPAM)

        binding = scripted(ask=ask)
        task = asyncio.ensure_future(
            deciding(binds={"default": binding}, settings=RunSettings(deadline=600))
        )
        await started.wait()  # the call is under way, its deadline far off
        task.cancel()
        try:
            await task
        except asyncio.CancelledError as exc:
            return exc
        raise AssertionError("decide returned")

    record = record_of(asyncio.run(main()))
    assert record.cause == {"code": "cancelled", "refs": []}
    assert "call" not in kinds(record)
    assert "round" not in kinds(record)


class _Opaque:
    def __repr__(self) -> str:
        raise RuntimeError("repr was called")


class _Raising(type):
    def __getattribute__(cls, name: str) -> Any:
        raise RuntimeError(f"the metaclass was read for {name}")


class _Hidden(metaclass=_Raising):
    pass


@pytest.mark.parametrize("value", [10**5000, -(10**5000), _Opaque()], ids=["huge", "neg", "repr"])
@pytest.mark.parametrize(
    "field", ["round_bound", "timeout", "budget", "deadline", "certificate_cap"]
)
def test_a_setting_that_cannot_be_written_is_an_entry_refusal(field: str, value: object) -> None:
    exc = caught(deciding(settings=RunSettings(**cast(dict[str, Any], {field: value}))))
    assert type(exc) is EntryRefusal


def test_a_huge_setting_is_named_by_its_bit_length() -> None:
    with pytest.raises(EntryRefusal, match=r"^the round bound integer of 16610 bits is not"):
        run_settings(10**5000, 60, None, None, None)


@pytest.mark.parametrize("key", [10**5000, _Opaque(), _Hidden()], ids=["huge", "repr", "meta"])
def test_a_bindings_key_that_cannot_be_written_is_an_entry_refusal(key: object) -> None:
    exc = caught(deciding(binds={key: scripted()}))  # type: ignore[dict-item]
    assert type(exc) is EntryRefusal


@pytest.mark.parametrize(
    "keywords",
    [
        {"loaded": _Hidden()},
        {"settings": _Hidden()},
        {"observer": _Hidden()},
        {"codec": _Hidden()},
        {"binds": _Hidden()},
    ],
    ids=["loaded", "settings", "observer", "codec", "binds"],
)
def test_an_argument_of_a_hostile_metaclass_is_an_entry_refusal(keywords: dict[str, Any]) -> None:
    exc = caught(deciding(**keywords))
    assert type(exc) is EntryRefusal


def test_a_state_over_the_value_domains_size_bounds_is_a_broken_entry(
    small_bounds: SmallBounds,
) -> None:
    big = {**STATE, "subject": "x" * 257}
    decision = go(state=big)
    assert decision.plan.cause == Cause(CauseCode.invalid_return, ["t:state"])
    assert kinds(decision.record) == ["design", "settings", "plan", "verdict"]


def test_a_long_wording_loads_and_decides(small_bounds: SmallBounds) -> None:
    wording = "Is it spam? " + "Consider every word of it. " * 20  # past the 2^8 code points set
    ask = dataclasses.replace(
        ASK, provides={**ASK.provides, "q:spam": Question("noul", Text(wording))}
    )
    decision = go(slice_design(sources={"ask": ask, "look": LOOK}))
    assert decision.plan.kind is PlanKind.execute
    assert read_record(decision.record.document).document == decision.record.document
