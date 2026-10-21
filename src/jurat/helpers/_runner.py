"""The reference runner: ``run_plan`` runs a passed ``execute`` plan's steps in position order,
for a caller with no workflow engine of its own.

The gate refuses with ``RunRefusal`` before any call. Then each step's arguments are resolved
from the record's ``plan`` item, its tool is called with them as keywords and the
``idempotency_key``, and its result awaited. The run stops at the first ``Exception`` or
binding miss and returns what it holds; it retries, times out and records nothing.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, cast

import jurat._decide as _decide
from jurat._markers import PlanKind, VerdictKind
from jurat._refs import parse_ref
from jurat._values import string_data

_KEY = "idempotency_key"


class RunRefusal(ValueError):
    """A run refused at the gate, before any tool is called."""


class BindingNotFound(LookupError):
    """A binding whose field path missed: ``step`` is the position of the step whose argument
    it is, ``position`` the bound step's position, and ``path`` the field path. It is returned in
    ``raised``, never raised."""

    def __init__(self, step: int, position: int, path: str) -> None:
        super().__init__(f"step {step}: no {path!r} in step {position}'s result")
        self.step = step
        self.position = position
        self.path = path


def _tools(tools: object) -> dict[str, object]:
    """The one copy of ``tools`` the run reads, keyed by string data."""
    # By its type alone: isinstance reads a __class__ the caller's object may spoof or raise.
    try:
        mapping = issubclass(type(tools), Mapping)
    except Exception as exc:  # a metaclass's subclass hook
        raise RunRefusal(f"testing tools, a {type(tools).__name__}, raised") from exc
    if not mapping:
        raise RunRefusal(f"tools, a {type(tools).__name__}, is not a Mapping")
    try:
        copy = dict(cast("Mapping[object, object]", tools))
    except Exception as exc:  # the caller's mapping, read once
        raise RunRefusal("copying tools raised") from exc
    named: dict[str, object] = {}
    for key, value in copy.items():
        if issubclass(type(key), str):
            data = string_data(cast(str, key))
            if data in named:
                raise RunRefusal(f"tools holds two keys of the string data {data!r}")
            named[data] = value
    return named


def _gate(
    decision: object, tools: object
) -> tuple[dict[str, object], list[dict[str, Any]], dict[str, Any]]:
    """The gate, in the reference runner's order; returns the tools' copy, the record's log,
    decoded once for the run, and its plan item."""
    if type(decision) is not _decide.Decision:
        raise RunRefusal(f"decision, a {type(decision).__name__}, is not exactly a Decision")
    named = _tools(tools)
    if decision.verdict.kind is VerdictKind.refused:
        raise RunRefusal("the verdict is refused")
    if decision.plan.kind is not PlanKind.execute:
        raise RunRefusal(f"the plan's kind is {decision.plan.kind.value}, not execute")
    log = decision.record.log
    item = next(i for i in log if i["item"] == "plan")
    for n, step in enumerate(item["steps"], 1):
        if step["tool"] not in named:
            raise RunRefusal(f"step {n}'s tool {step['tool']!r} is not in tools")
        if not callable(named[step["tool"]]):
            raise RunRefusal(f"step {n}'s tool {step['tool']!r} is not callable")
    return named, log, item


def _entries(log: list[dict[str, Any]]) -> dict[str, object]:
    """Each board ref's entry value, first by log position."""
    out: dict[str, object] = {}
    for i in log:
        if i["item"] == "entry":
            out.setdefault(i["ref"], i["value"])
    return out


def _board(entries: Mapping[str, object], ref: str) -> object:
    """A board ref's value from the record's entries, copied afresh at each read so no two
    arguments share an object: the base ref's entry value, its field path walked over it, a
    null on the path giving null."""
    # The entry, and a record at every field on the path but a null, hold by plan rule 4 and
    # conformance, so neither miss is reachable here.
    fields = parse_ref(ref).fields
    base = ref[: len(ref) - len(".".join(fields)) - 1] if fields else ref
    value: Any = entries[base]
    for field in fields:
        if value is None:
            return None
        value = value[field]
    return deepcopy(value)


class _Miss(Exception):
    """A binding's path missed; carried to the step that sees it."""


def _bound(result: object, path: str | None) -> object:
    """A binding's field path over a result: each segment a key of a ``Mapping``, read by ``in``
    then indexing, a present key giving its value, ``None`` included; a miss raises ``_Miss``
    and the mapping's own raise passes through."""
    if path is None:
        return result
    value = result
    for segment in path.split("."):
        if not issubclass(type(value), Mapping):  # by type: __class__ is the result's
            raise _Miss
        mapping = cast("Mapping[object, object]", value)
        if segment not in mapping:
            raise _Miss
        value = mapping[segment]
    return value


async def run_plan(decision: "_decide.Decision", tools: Mapping[str, Any]) -> dict[str, Any]:
    """Run the decision's plan; return ``{"results": [[position, result], ...], "raised": exc
    or None}``, or raise ``RunRefusal`` at the gate before any call."""
    named, log, item = _gate(decision, tools)
    turn = decision.record.record_hash
    results: list[list[Any]] = []
    held: dict[int, object] = {}
    entries = _entries(log)
    for n, step in enumerate(item["steps"], 1):
        keywords: dict[str, Any] = {}
        for name, origin in step["arguments"].items():
            if "step" in origin:
                bound = held[origin["step"]]  # an earlier step's, by plan rule 2
                try:
                    keywords[name] = _bound(bound, origin["path"])
                except _Miss:
                    miss = BindingNotFound(n, origin["step"], origin["path"])
                    return {"results": results, "raised": miss}
                except Exception as exc:  # the result's own `in` or indexing raised
                    return {"results": results, "raised": exc}
            elif "ref" in origin:
                keywords[name] = _board(entries, origin["ref"])
            else:
                # Each literal is its own subtree of the gate's one decoding, so it is fresh
                # for this argument and shared with none.
                keywords[name] = origin["literal"]
        keywords[_KEY] = f"{turn}:{n}"
        tool = cast("Callable[..., Any]", named[step["tool"]])
        try:
            result = await tool(**keywords)
        except Exception as exc:  # the tool's, or an unawaitable result's TypeError
            return {"results": results, "raised": exc}
        held[n] = result
        results.append([n, result])
    return {"results": results, "raised": None}
