# pyright: reportPrivateUsage=false
"""The fixture binding: replay's oracle binding, answering each call from the recorded calls
of its sources, so the turn reruns with no provider; and ``from_record``'s, answering a
caller's live turn from a record under the oracle names it is bound to.

Built once from a record's log and the given loaded design. Each asked source is matched to
the recorded call item that lists it, never by round or batch, and each of its refs is
answered only from a recorded entry naming that call. No answer
depends on an earlier call; it waits for nothing and returns fresh containers with null
provenance, a null precision and no truncated refs or warnings. It logs the refs each source
was asked, which replay reads for ``ref-not-in-record``.
"""

from collections.abc import Mapping
from typing import Any, cast

from jurat._answer import AnswerKind, ValueType
from jurat._binding import CallResult, OracleBinding, RenderedQuestion, SourceCall
from jurat._canonical import encode
from jurat._component import base
from jurat._decide import EntryRefusal
from jurat._design import LoadedDesign
from jurat._interpret import MAX_WARNING_CODE_POINTS, MAX_WARNINGS, _record_form
from jurat._ledger import Record, read_record
from jurat._markers import Cause, CauseCode, Failed, Unavailable
from jurat._read_view import _ViewError, walk
from jurat._refs import parse_ref
from jurat._typed import decoded
from jurat._values import clipped, type_name

_C = CauseCode
_RETURNED_MARKERS = frozenset(
    {_C.refused.value, _C.invalid_answer.value, _C.invalid_return.value, _C.unsupported.value}
)
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
        """The recorded entry's value, typed, a ``q:`` value in the binding's form of its
        recorded kind, or one of the four markers a binding may return; ``_NO_KEY`` for any
        other marker."""
        if "value" in entry:
            value = entry["value"]
            if ref.startswith("q:"):
                # The ledger refuses a q: value that is not an answer's object of its recorded
                # kind, live and read back, so each has that kind's members.
                return _binding_form(cast("dict[str, Any]", value))
            return decoded(value, self._types[ref])
        marker = cast("dict[str, Any]", entry["marker"])
        code = marker["cause"]["code"]
        if marker["kind"] == "Unavailable" and code in _RETURNED_MARKERS:
            return Unavailable(Cause(code, [ref]))
        return _NO_KEY


def _float(value: object) -> object:
    """An integral number read back as an ``int`` made the ``float`` it was written as."""
    return float(value) if type(value) is int else value


def _binding_form(value: dict[str, Any]) -> dict[str, object]:
    """A recorded answer as a binding returns it, by its recorded kind: a probability kind's
    ``probabilities``, each a ``float``; an extraction answer's five members, ``p_null`` and
    ``confidence`` each a ``float``, under a recorded ``number`` value type each number in
    ``value`` and ``guess`` a ``float``, and each container a fresh list. Interpret validates it
    against the question now asked."""
    if value["kind"] != AnswerKind.extraction:
        probabilities = cast("dict[str, object]", value["probabilities"])
        return {label: _float(p) for label, p in probabilities.items()}
    number = value["value_type"] == ValueType.number

    def member(found: object) -> object:
        if type(found) is list:
            return [member(m) for m in cast("list[object]", found)]
        return _float(found) if number else found

    def pair(found: object) -> object:
        if type(found) is list:
            return [pair(m) for m in cast("list[object]", found)]
        return found

    return {
        "value": member(value["value"]),
        "guess": member(value["guess"]),
        "p_null": _float(value["p_null"]),
        "confidence": _float(value["confidence"]),
        "span": pair(value["span"]),
    }


# ---------------------------------------------------------------------------------------------
# from_record: a caller's live turn answered from a record

_KEPT_MARKERS = frozenset({_C.invalid_answer.value, _C.invalid_return.value})
"""The recorded markers ``from_record`` returns as recorded; any other is ``unsupported``."""
_TAIL = 64
"""Code points kept free for the line saying how many lines were left out."""


def from_record(record: object, loaded: LoadedDesign) -> OracleBinding:
    """An oracle binding answering a caller's live turn under ``loaded`` from ``record``, a
    ``Record`` or a stored document read through ``read_record``.

    Bound under the oracle names whose sources the record answers, beside live bindings under
    the rest, it answers each asked source from the returned call item that lists it, each ref
    from the entry that call wrote, in the binding's form. It refuses with ``unsupported``,
    each refusal a line of ``warnings``: every ref of a source the record holds no returned
    call for, or whose ``needs`` differ from the record's entries by canonical bytes; a
    question whose kind, labels as declared, value type or list flag differ from the call
    item's; a ref with no recorded entry naming its call; and a ref whose recorded marker is
    neither ``invalid-answer`` nor ``invalid-return``, which come back as recorded. Its
    provenance is null and its first warnings line names the record's hash.

    The binding cannot check a question's identity or wording: a ``SourceCall`` carries no
    identity and the call item holds no wording. Under the record's own design neither can
    differ. Under another design the caller compares, after the turn, each new call item's
    ``questions`` identities with the stored call item's, by ``q:`` ref."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    read = record if type(record) is Record else read_record(record)
    return _FromRecord(read, loaded)


class _FromRecord:
    """``from_record``'s binding: the record's calls and entries, read once."""

    def __init__(self, record: Record, loaded: LoadedDesign) -> None:
        log = record.log  # read once: every read decodes afresh
        self._recorded = RecordedCalls(log)
        self._entries = {cast(str, i["ref"]): i for i in log if i["item"] == "entry"}
        self._types = loaded._types
        self._note = f"answered from record {record.record_hash}"

    async def __call__(self, batch: tuple[SourceCall, ...]) -> object:
        values: dict[str, object] = {}
        lines: list[str] = []
        for source in batch:
            self._source(source, values, lines)
        return CallResult(values, warnings=_bounded([self._note, *lines]))

    def _source(self, source: SourceCall, values: dict[str, object], lines: list[str]) -> None:
        refs = (*source.refs, *source.questions)
        unsupported = Unavailable(Cause(_C.unsupported))
        found = self._recorded.call_of(source.id)
        refusal: str | None = None
        if found is None:
            refusal = "the record holds no call for it"
        elif found[1]["outcome"] != "returned":
            code = cast("dict[str, Any]", found[1]["outcome"])["failed"]["code"]
            refusal = f"its recorded call failed with {code}"
        else:
            differ = [need for need, value in source.needs.items() if self._differs(need, value)]
            if differ:
                shown = ", ".join(clipped(need) for need in differ)
                refusal = f"needs differ from the stored turn's: {shown}"
        if refusal is not None or found is None:
            values.update(dict.fromkeys(refs, unsupported))
            lines.append(f"source {clipped(source.id)}: {refusal}")
            return
        index, item = found
        stored = cast("dict[str, Any]", item["questions"])
        for ref in refs:
            if ref in source.questions:
                members = _differing(source.questions[ref], stored.get(ref))
                if members:
                    values[ref] = unsupported
                    lines.append(f"{clipped(ref)}: its {members} differ from the stored question's")
                    continue
            entry = self._recorded.entries.get((index, ref))
            if entry is None:
                values[ref] = unsupported
                lines.append(f"{clipped(ref)}: the recorded call wrote no entry for it")
            elif "value" in entry:
                values[ref] = (
                    _binding_form(cast("dict[str, Any]", entry["value"]))
                    if ref.startswith("q:")
                    else decoded(entry["value"], self._types[ref])
                )
            else:
                marker = cast("dict[str, Any]", entry["marker"])
                code = cast(str, marker["cause"]["code"])
                if marker["kind"] == "Unavailable" and code in _KEPT_MARKERS:
                    values[ref] = Unavailable(Cause(code, [ref]))
                else:
                    values[ref] = unsupported
                    lines.append(f"{clipped(ref)}: recorded {code}, no answer to give")

    def _differs(self, need: str, value: object) -> bool:
        """Whether a need's live value differs by canonical bytes from the record's entry at
        its base, read through its field path; a marker or a missing entry or path differs."""
        entry = self._entries.get(base(need))
        if entry is None or "value" not in entry:
            return True
        try:
            stored = walk(entry["value"], parse_ref(need).fields, need)
        except _ViewError:
            return True
        return encode(stored) != encode(_record_form(value))


def _differing(asked: RenderedQuestion, stored: dict[str, Any] | None) -> str:
    """The members of the asked question that differ from the call item's ``questions``
    entry, joined; all four when the call item asked no such question; empty when none."""
    if stored is None:
        return "kind, labels, value type and list flag"
    kind = AnswerKind(asked.kind)
    labels: object = None
    if kind not in (AnswerKind.noul, AnswerKind.extraction):
        labels = [
            name if description is None else {"name": name, "description": description}
            for name, description in zip(asked.labels, asked.descriptions, strict=True)
        ]
    found: list[str] = []
    if stored["kind"] != kind.value:
        found.append("kind")
    if stored["labels"] != labels:
        found.append("labels")
    value_type = None if asked.value_type is None else ValueType(asked.value_type).value
    if stored.get("value_type") != value_type:
        found.append("value type")
    if stored.get("is_list", False) != asked.is_list:
        found.append("list flag")
    return ", ".join(found)


def _bounded(lines: list[str]) -> tuple[str, ...]:
    """The lines in order within interpret's warnings bounds; past them, a last line counting
    those left out."""
    kept: list[str] = []
    total = 0
    for i, line in enumerate(lines):
        rest = len(lines) - i
        last = rest == 1
        room = len(kept) < MAX_WARNINGS - 1 or last
        limit = MAX_WARNING_CODE_POINTS if last else MAX_WARNING_CODE_POINTS - _TAIL
        if not room or total + len(line) > limit:
            kept.append(f"{rest} more {'line' if last else 'lines'} left out")
            break
        kept.append(line)
        total += len(line)
    return tuple(kept)
