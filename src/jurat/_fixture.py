# pyright: reportPrivateUsage=false
"""The fixture binding: replay's oracle binding, answering each call from the recorded calls
of its sources, so the turn reruns with no provider.

Built once from a record's log and the given loaded design. Each asked source is matched to
the recorded call item that lists it, never by round or batch, and each of its refs is
answered only from a recorded entry naming that call. No answer
depends on an earlier call; it waits for nothing and returns fresh containers with null
provenance. It logs the refs each source was asked, which replay reads for
``ref-not-in-record``.
"""

from collections.abc import Mapping
from typing import Any, cast

from jurat._binding import CallResult, SourceCall
from jurat._design import LoadedDesign
from jurat._markers import Cause, CauseCode, Failed, Unavailable
from jurat._typed import decoded

_C = CauseCode
_RETURNED_MARKERS = frozenset({_C.refused.value, _C.invalid_answer.value, _C.invalid_return.value})
_NO_KEY = object()
"""What a ref gets that the fixture leaves out of ``values``; a recorded null is a value."""


class RecordedCalls:
    """The record's call items by the sources they list, and its call-written entries by
    ``(call index, ref)``."""

    def __init__(self, log: list[dict[str, Any]]) -> None:
        self.calls: dict[str, tuple[int, dict[str, Any]]] = {}
        self.entries: dict[tuple[int, str], dict[str, Any]] = {}
        for index, item in enumerate(log):
            if item["item"] == "call":
                # The ledger refuses a source in two call items, so no source is overwritten.
                for sid in cast("list[str]", item["sources"]):
                    self.calls[sid] = (index, item)
            elif item["item"] == "entry" and item["call"] is not None:
                self.entries[(cast(int, item["call"]), cast(str, item["ref"]))] = item

    def call_of(self, sid: str) -> tuple[int, dict[str, Any]] | None:
        return self.calls.get(sid)


class FixtureBinding:
    """An oracle binding answering from a record under a given design."""

    def __init__(self, recorded: RecordedCalls, loaded: LoadedDesign) -> None:
        self._recorded = recorded
        self._types = loaded._types
        self.asked: dict[str, tuple[str, ...]] = {}
        """Each asked source's refs, its ``refs`` and its question keys, as passed; a source
        is asked at most once in a turn."""

    async def __call__(self, batch: tuple[SourceCall, ...]) -> object:
        for source in batch:
            self.asked[source.id] = (*source.refs, *source.questions)
        calls = [self._recorded.call_of(s.id) for s in batch]
        indices = {c[0] for c in calls if c is not None}
        if None not in calls and len(indices) == 1:
            outcome = cast("tuple[int, dict[str, Any]]", calls[0])[1]["outcome"]
            if outcome != "returned":
                failed = cast("dict[str, Any]", outcome)["failed"]
                return Failed(Cause(failed["code"], list(failed["refs"])))
        values: dict[str, Any] = {}
        for source, found in zip(batch, calls, strict=True):
            if found is None or found[1]["outcome"] != "returned":
                continue  # no call, or a Failed call beside others: no key
            index = found[0]
            for ref in self.asked[source.id]:
                entry = self._recorded.entries.get((index, ref))
                if entry is not None:
                    answer = self._answer(ref, entry)
                    if answer is not _NO_KEY:
                        values[ref] = answer
        return CallResult(values)

    def _answer(self, ref: str, entry: Mapping[str, Any]) -> object:
        """The recorded entry's value, typed, a ``q:`` value as its probabilities, or one of
        the three markers a binding may return; ``_NO_KEY`` for any other marker."""
        if "value" in entry:
            value = entry["value"]
            if ref.startswith("q:"):
                # The ledger refuses a q: value that is not an answer's object, live and read
                # back, so each has a probabilities object.
                probabilities = cast("dict[str, object]", value["probabilities"])
                return {
                    label: float(p) if type(p) is int else p for label, p in probabilities.items()
                }
            return decoded(value, self._types[ref])
        marker = cast("dict[str, Any]", entry["marker"])
        code = marker["cause"]["code"]
        if marker["kind"] == "Unavailable" and code in _RETURNED_MARKERS:
            return Unavailable(Cause(code, [ref]))
        return _NO_KEY
