"""The v0.0.1 slice end to end, through public names only: a minimal design with one function
rule and one function policy loads; decide runs one turn against a test fixture binding; the
record closes with its hash and reads back; replay reproduces it with no finding; and the
reference runner runs the passed ``execute`` plan."""

import asyncio
from typing import Any

import pytest

from jurat import (
    CallResult,
    Design,
    Origin,
    Plan,
    PlanKind,
    Policy,
    Question,
    Requirement,
    Rule,
    Source,
    SourceCall,
    Step,
    Text,
    VerdictKind,
    decide_sync,
    load,
    read_record,
    replay_sync,
)
from jurat import types as t
from jurat.helpers import RunRefusal, pick_with_margin, run_plan


def route(r: Any) -> Any:
    return pick_with_margin(r, "q:spam", 0.5, 0.1)


def archive_policy(r: Any) -> Plan:
    if r.read("d:route") == "yes":
        step = Step(
            "archive",
            {"subject": Origin(ref="t:state.subject"), "folder": Origin(literal="spam")},
            [Requirement("d:route", "yes")],
        )
        return Plan("execute", [step])
    return Plan("inform", reply="kept")


DESIGN = Design(
    state=t.record(subject=t.string),
    sources={"ask": Source({"q:spam": Question("noul", Text("Is it spam?"))}, needs=["t:state"])},
    rules={"route": Rule("d:route", t.label("no", "yes"), route, version="1", reads=["q:spam"])},
    policy=Policy(archive_policy, version="1", reads=["d:route", "t:state.subject"]),
    tools=["archive"],
)


class Answering:
    """A test fixture binding: every batch's spam question answered with ``spam``."""

    def __init__(self, spam: float) -> None:
        self.spam = spam
        self.asked: list[tuple[SourceCall, ...]] = []

    async def __call__(self, batch: tuple[SourceCall, ...]) -> CallResult:
        self.asked.append(batch)
        return CallResult({"q:spam": {"yes": self.spam, "no": 1 - self.spam}})


def test_one_turn_loads_decides_closes_replays_and_runs() -> None:
    # Load: one function rule and one function policy.
    loaded = load(DESIGN)
    assert loaded.design_hash.startswith("sha256:")

    # Decide: one turn against the fixture binding.
    binding = Answering(0.9)
    decision = decide_sync(loaded, {"subject": "win a prize"}, {"default": binding})
    assert len(binding.asked) == 1
    assert decision.plan.kind is PlanKind.execute
    assert decision.verdict.kind is VerdictKind.passed

    # The record closes with its hash and reads back with the same one.
    record = decision.record
    turn = record.record_hash
    assert turn.startswith("sha256:")
    assert len(turn) == len("sha256:") + 64
    assert record.design_hash == loaded.design_hash == decision.design_hash
    assert read_record(record.document).record_hash == turn

    # Replay reproduces it with no finding.
    rerun, findings = replay_sync(record, loaded)
    assert findings == []
    assert rerun is not None
    assert rerun.record_hash == turn

    # The reference runner runs the passed execute plan.
    called: list[dict[str, Any]] = []

    async def archive(**keywords: Any) -> dict[str, Any]:
        called.append(keywords)
        return {"archived": True}

    out = asyncio.run(run_plan(decision, {"archive": archive}))
    assert called == [
        {"subject": "win a prize", "folder": "spam", "idempotency_key": f"{turn}:1"},
    ]
    assert out == {"results": [[1, {"archived": True}]], "raised": None}


def test_an_inform_turn_replays_and_is_not_run() -> None:
    loaded = load(DESIGN)
    decision = decide_sync(loaded, {"subject": "lunch?"}, {"default": Answering(0.1)})
    assert decision.plan.kind is PlanKind.inform
    rerun, findings = replay_sync(decision.record, loaded)
    assert findings == []
    assert rerun is not None
    assert rerun.record_hash == decision.record.record_hash
    with pytest.raises(RunRefusal, match="execute"):
        asyncio.run(run_plan(decision, {"archive": Answering(0)}))
