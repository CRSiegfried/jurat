# pyright: reportPrivateUsage=false
"""Replay: rerun decide's body over a recorded turn under a given loaded design, with the
record as the only fixture, and report every way the rerun differs.

``replay`` refuses a wrong argument before it reads the record. A record
with no ``verdict`` item is ``unfinished``, and one with a ``verdict`` item and no state entry
is ``nothing-to-replay``: either returns ``None`` with that finding alone. Otherwise the record's
run settings go through decide's bounds check, and decide's body runs through its internal
entry: the recorded state entry typed by the given design in place of the codec, the fixture
binding under every declared batch key and ``default``, the observer, and the record's cut in
place of a clock reading. A cancellation or the observer's exception
propagates as decide lets it out, with the replay record so far and no findings.

The findings are plain ``dict``s in a fixed order, two items differing when their
canonical bytes differ. Replay reads nothing from ``unhashed``.

``question-changed``'s candidate-valued label set clause is not built: load's check 12
refuses candidate sets in v0.0.1.
"""

import asyncio
import enum
from typing import Any, cast

from jurat._canonical import decode, encode
from jurat._decide import EntryRefusal, _bound, _callable_or_none, _turn, run_settings
from jurat._design import LoadedDesign
from jurat._fixture import FixtureBinding, RecordedCalls
from jurat._ledger import _ITEM_DEPTH, Observer, Record
from jurat._refs import code_point_sorted
from jurat._typed import decoded
from jurat._values import type_name
from jurat.helpers import HELPERS

_STATE = "t:state"
_DEFAULT = "default"

Finding = dict[str, Any]


class FindingKind(enum.StrEnum):
    """Replay's finding kinds."""

    design_mismatch = "design-mismatch"
    helpers_mismatch = "helpers-mismatch"
    nothing_to_replay = "nothing-to-replay"
    not_in_record = "not-in-record"
    ref_not_in_record = "ref-not-in-record"
    question_changed = "question-changed"
    unfinished = "unfinished"
    entry = "entry"
    plan = "plan"
    verdict = "verdict"


_K = FindingKind


async def replay(
    record: Record, loaded: LoadedDesign, *, observer: Observer | None = None
) -> tuple[Record | None, list[Finding]]:
    """Rerun ``record`` under ``loaded`` and return the replay record, or ``None``, and the
    findings."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    if type(record) is not Record:
        raise EntryRefusal(f"record is a {type_name(record)}, not exactly a Record")
    _callable_or_none(observer, "observer")
    log = record.log  # read once: every read decodes afresh
    if not any(item["item"] == "verdict" for item in log):
        return None, [{"kind": _K.unfinished}]
    state = [item for item in log if item["item"] == "entry" and item["ref"] == _STATE]
    if not state:
        return None, [{"kind": _K.nothing_to_replay}]
    # The settings item is second in every record from the design item on.
    recorded = cast("dict[str, Any]", log[1]["settings"])
    run = run_settings(
        recorded["round_bound"],
        recorded["timeout"],
        recorded["budget"],
        recorded["deadline"],
        recorded["certificate_cap"],
    )
    calls = RecordedCalls(log)
    fixture = FixtureBinding(calls, loaded)
    keys = {s.batch for s in loaded._design.sources.values() if s.batch is not None}
    bindings = _bound(loaded, dict.fromkeys({*keys, _DEFAULT}, fixture))
    value = decoded(state[0]["value"], loaded._types[_STATE])
    decision = await _turn(
        loaded, value, bindings, run, codec=None, clock=None, cut=_cut(log), observer=observer
    )
    rerun = decision.record
    return rerun, _findings(log, rerun.log, loaded, calls, fixture.asked)


def _cut(log: list[dict[str, Any]]) -> tuple[int, bool] | None:
    """The record's cut: the round whose ``round`` item has ``deadline`` true, and whether
    that round holds a call item; ``None`` for a record with no such round."""
    for item in log:
        if item["item"] == "round" and item["deadline"] is True:
            r = cast(int, item["round"])
            return r, any(i["item"] == "call" and i["round"] == r for i in log)
    return None


def replay_sync(
    record: Record, loaded: LoadedDesign, *, observer: Observer | None = None
) -> tuple[Record | None, list[Finding]]:
    """``replay`` run to completion; ``RuntimeError`` inside a running event loop, raised
    before the coroutine is built."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("replay_sync was called inside a running event loop; await replay")
    return asyncio.run(replay(record, loaded, observer=observer))


# ---------------------------------------------------------------------------------------------
# The findings, in their fixed order


def _findings(
    log: list[dict[str, Any]],
    rerun: list[dict[str, Any]],
    loaded: LoadedDesign,
    calls: RecordedCalls,
    asked: dict[str, tuple[str, ...]],
) -> list[Finding]:
    found: list[Finding] = []
    design = log[0]
    if design["hash"] != loaded.design_hash:
        found.append(
            {"kind": _K.design_mismatch, "recorded": design["hash"], "replayed": loaded.design_hash}
        )
    if design["helpers"] != HELPERS:
        found.append(
            {"kind": _K.helpers_mismatch, "recorded": design["helpers"], "replayed": HELPERS}
        )
    found += _call_findings(rerun, asked, calls)
    found += _entry_findings(log, rerun, loaded)
    for kind in (_K.plan, _K.verdict):
        mine, theirs = _first(log, kind.value), _first(rerun, kind.value)
        if _differ(mine, theirs):
            found.append({"kind": kind, "recorded": mine, "replayed": theirs})
    return found


def _call_findings(
    rerun: list[dict[str, Any]], asked: dict[str, tuple[str, ...]], calls: RecordedCalls
) -> list[Finding]:
    """``not-in-record`` per fixture call, by first source id, then one ``ref-not-in-record``
    and one ``question-changed`` for the run. The replay record's call items give one per
    fixture call, the sources it asked and their questions' identities; the fixture's log
    gives the refs each source was asked.

    ``question-changed`` compares identities only: its candidate-valued label set clause is not
    built, since load's check 12 refuses candidate sets in v0.0.1."""
    missing: list[list[str]] = []
    refs: set[str] = set()
    changed: set[str] = set()
    for item in rerun:
        if item["item"] != "call":
            continue
        sids = cast("list[str]", item["sources"])
        absent = code_point_sorted(s for s in sids if calls.call_of(s) is None)
        if absent:
            missing.append(absent)
        for sid in sids:
            call = calls.call_of(sid)
            if call is None:
                continue
            index, recorded = call
            # The asked refs coincide with the source's provides until candidate grounding
            # exists.
            refs.update(r for r in asked[sid] if (index, r) not in calls.entries)
            before = cast("dict[str, Any]", recorded["questions"])
            now = cast("dict[str, Any]", item["questions"])
            for ref, question in now.items():
                if ref in before and before[ref]["identity"] != question["identity"]:
                    changed.add(ref)
    found: list[Finding] = [
        {"kind": _K.not_in_record, "sources": s} for s in sorted(missing, key=lambda s: s[0])
    ]
    if refs:
        found.append({"kind": _K.ref_not_in_record, "refs": code_point_sorted(refs)})
    if changed:
        found.append({"kind": _K.question_changed, "refs": code_point_sorted(changed)})
    return found


def _entry_findings(
    log: list[dict[str, Any]], rerun: list[dict[str, Any]], loaded: LoadedDesign
) -> list[Finding]:
    """An ``entry`` finding per ref, in code-point order, held by either record whose two
    entries differ, null for a side that holds none."""
    mine = {item["ref"]: item for item in log if item["item"] == "entry"}
    theirs = {item["ref"]: item for item in rerun if item["item"] == "entry"}
    rules = cast("dict[str, Any]", decode(loaded.document))["rules"]
    found: list[Finding] = []
    for ref in code_point_sorted({*mine, *theirs}):
        a, b = mine.get(ref), theirs.get(ref)
        if _differ(a, b):
            found.append(
                {
                    "kind": _K.entry,
                    "ref": ref,
                    "recorded": a,
                    "replayed": b,
                    "rule": _rule(ref, loaded, rules),
                }
            )
    return found


def _rule(ref: str, loaded: LoadedDesign, rules: dict[str, Any]) -> dict[str, Any] | None:
    """The given design's function rule providing a ``d:`` ref: its ``module``, ``qualname``,
    ``version`` and ``source`` from the design's document, else null."""
    if not ref.startswith("d:"):
        return None
    declared = rules.get(loaded._providers.get(ref, ""))
    if declared is None or declared["form"] != "function" or declared["provides"] != ref:
        return None
    return {k: declared[k] for k in ("module", "qualname", "version", "source")}


def _first(log: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    return next((item for item in log if item["item"] == kind), None)


def _differ(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Two items, or a side holding none, compared by canonical bytes."""
    if a is None or b is None:
        return a is not b
    return encode(a, _ITEM_DEPTH) != encode(b, _ITEM_DEPTH)
