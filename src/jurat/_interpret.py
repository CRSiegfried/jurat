# pyright: reportPrivateUsage=false
"""Interpret: from the board decide opened, ask the oracle bindings and fire the rules, round
by round over the turn's sorter, until the fixpoint or the round bound.

Round 0 is a rule pass. Each round from 1 is a source phase over the held
sources, each asked or given its own markers, then a rule phase. A held
source is ready when its needs are determined, a source requirement's ref counting as one,
its source requirement holds and its function ``when`` is true, tested in that order and in
source id order before the live frontier's calls start. The live frontier's calls run under one
``asyncio.TaskGroup``, each task holding its call's timeout and the binding's await alone; every
classification and append happens after the group closes, call items by first source id, then the
source entries by source id, then each rule wavefront sorted by id, then the ``round`` item. A batch
is the ready sources of one live frontier sharing a batch key, called through that key's binding; a
keyless source is a batch of one through ``default``.

Three ends share one path: at the round bound after the rule phase, and at the budget or the
deadline in the source phase with no rule phase first, the pending sources are ``Unasked`` and
the sorter is drained. The budget is tested before a frontier is asked, then the
deadline, which is tested again when the frontier's task group closes; under replay the recorded cut
stands in for the clock.

Two boards: the live one the views and the needs read holds an ``Answer`` for a ``q:`` ref
and Python ``bytes`` for a ``bytes`` value; the ledger is handed plain data.
"""

import asyncio
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, cast

from jurat._answer import Answer, AnswerKind, answer_document, off_sum
from jurat._binding import CallResult, OracleBinding, RenderedQuestion, SourceCall
from jurat._canonical import decode, encode, hash_value
from jurat._component import Called, at, base, call, exception_record, names_read, now
from jurat._design import LoadedDesign, Question, Rule, Source
from jurat._ledger import Ledger
from jurat._markers import Cause, CauseCode, Failed, Unasked, Unavailable, Undetermined, Unsure
from jurat._read_view import (
    ComponentKind,
    ReadView,
    RequireStopped,
    Role,
    UndeclaredRead,
    _Stop,
)
from jurat._refs import code_point_sorted
from jurat._sorter import TurnSorter
from jurat._typed import is_a, typed
from jurat._types import Type, b64url_encode
from jurat._values import (
    MAX_DOCUMENT_DEPTH,
    MAX_INTEGER,
    domain_error,
    guarded_repr,
    is_scalar_string,
    string_data,
)

_STATE = "t:state"
_DEFAULT = "default"
_NOUL_LABELS = ("yes", "no")
_PROVENANCE = ("snapshot", "elicitation", "prompt")

_C = CauseCode
_STRUCTURAL = frozenset({_C.raised, _C.timed_out, _C.invalid_return, _C.unexpected_ref})
_BINDING_MARKERS = frozenset({_C.refused, _C.invalid_answer, _C.invalid_return})
_MARKER_KINDS: dict[type[Undetermined], str] = {
    Unsure: "Unsure",
    Unavailable: "Unavailable",
    Unasked: "Unasked",
}

Written = Undetermined | tuple[object, object]
"""What a ref gets: a marker, or a value in its record form and its live form."""


class _InterpretError(RuntimeError):
    """An engine defect interpret can see, which a caller cannot cause."""


@dataclass(slots=True)
class _WhenText:
    """What a function ``when`` leaves for ``unhashed``, under its source's lowest-ref entry: its
    raise, a raising truth test's included, and an undeclared read the design does not name."""

    error: Exception | None
    unnamed: str | None


async def interpret(
    ledger: Ledger,
    loaded: LoadedDesign,
    state: object,
    bindings: Mapping[str, OracleBinding],
    *,
    round_bound: int,
    timeout: float,  # noqa: ASYNC109 - the run setting; each call holds its own timeout
    budget: float | None,
    deadline: float | None,
    clock: float | None,
    cut: tuple[int, bool] | None,
    carried: frozenset[str],
) -> dict[str, object]:
    """Run the turn's rounds over the ledger decide opened, which holds the ``design`` and
    ``settings`` items and the state entry, and return the live board when interpret stops: every
    ref on it, as a value in its live form or a marker.

    ``state`` is the state entry's value as decide wrote it, in its live form. ``clock`` is
    the running loop's reading at decide's entry, from which the deadline runs, or ``None``
    under replay's internal entry, which reads no clock and passes no deadline. ``cut`` is
    then the recorded cut, the round the deadline cut and whether that round was asked, or
    ``None`` for a record with none. ``carried`` holds the ids of
    the carried providers, empty in v0.0.1. Run settings, bindings and the
    carried set are decide's to check. A cancellation propagates: one
    delivered at an await with nothing of the round it interrupted appended, and a
    ``CancelledError`` a function ``when``, a rule or a method of a returned value raises with
    every item appended before it.
    """
    if carried:
        raise _InterpretError("carried providers arrive with the continuation record")
    if deadline is not None and clock is None:
        raise _InterpretError("a deadline runs from decide's clock reading")
    if cut is not None and clock is not None:
        raise _InterpretError("the cut stands in for the clock, never beside it")
    end = None if deadline is None or clock is None else clock + deadline
    limits = _Limits(None if budget is None else _exact(budget), end, cut)
    return await _Turn(ledger, loaded, state, bindings, round_bound, timeout, carried, limits).run()


@dataclass(frozen=True, slots=True)
class _Limits:
    """What ends asking besides the round bound: the budget, exactly; the deadline's
    end on the running loop's clock; or under replay the recorded cut."""

    budget: Fraction | None
    end: float | None
    cut: tuple[int, bool] | None


# ---------------------------------------------------------------------------------------------
# One call, inside its task


@dataclass(slots=True)
class _Returned:
    """What a call's task saw: the wall-clock start, the latency, and the binding's return,
    its own raise, and whether the timeout expired."""

    start: str
    latency: float
    result: object
    error: Exception | None
    expired: bool
    escaped: BaseException | None
    """A ``BaseException`` the binding raised that is neither an ``Exception`` nor a
    cancellation, carried out of the task to propagate bare."""
    by_deadline: bool = False
    """Whether the call's timeout was the deadline's end, so an expiry is the deadline's."""


def _clock() -> float:
    """The running loop's clock, the one ``asyncio``'s timeouts read."""
    return asyncio.get_running_loop().time()


def _exact(tag: float) -> Fraction:
    """A cost tag or the budget as the decimal RFC 8785 writes it, exactly, as the choice helpers
    take a probability."""
    return Fraction(repr(tag))


async def _call(
    binding: OracleBinding, batch: tuple[SourceCall, ...], seconds: float, end: float | None
) -> _Returned:
    """One oracle binding call under its timeout, cut to the deadline's ``end`` when there is
    one, both taken at the call's start. The binding's
    await is the task's whole protected region; a cancellation passes
    through."""
    loop = asyncio.get_running_loop()
    start, began = now(), loop.time()
    result: object = None
    error: Exception | None = None
    escaped: BaseException | None = None
    expires = began + seconds if end is None else min(began + seconds, end)
    by_deadline = expires == end
    if end is not None and began >= end:
        # No time left: the call is not made, and is one the deadline cuts.
        return _Returned(start, 0.0, None, None, True, None, by_deadline)
    timeout = asyncio.timeout_at(expires)
    try:
        async with timeout:
            try:
                result = await binding(batch)
            except Exception as exc:  # the binding's own raise, a late one included
                error = exc
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # out of the group bare, not in a group
                escaped = exc
    except TimeoutError:  # the timeout's own, raised when it cancelled the binding
        pass
    # Expiry, not the raise, decides timed-out: a binding may swallow the cancellation and
    # return, or raise a TimeoutError of its own. The timeout fired, or the clock reads at or
    # past its end as the call returns or raises, so one that never yields is judged alike.
    closed = loop.time()
    expired = timeout.expired() or closed >= expires
    return _Returned(start, closed - began, result, error, expired, escaped, by_deadline)


# ---------------------------------------------------------------------------------------------
# The batch and how its call is judged


@dataclass(slots=True)
class _Batch:
    key: str
    """The binding's key: the sources' batch key, or ``default``."""
    sources: tuple[str, ...]
    refs: tuple[str, ...]
    """Every ref the batch provides, in code-point order."""
    questions: dict[str, Question]
    """Each asked ``q:`` ref's declared question."""
    labels: dict[str, tuple[str, ...]]
    """The labels each ``q:`` ref was passed with."""
    call: tuple[SourceCall, ...]
    timeout: float
    cost: Fraction


@dataclass(slots=True)
class _Judged:
    """A call's outcome, as the fault scope sorts it."""

    failed: Cause | None
    written: dict[str, Written]
    provenance: dict[str, str | None]
    usage: dict[str, float] | None = None
    attempts: int | None = None
    unexpected: list[str] = field(default_factory=list[str])
    invalid: dict[str, str] = field(default_factory=dict[str, str])
    sums: dict[str, float] = field(default_factory=dict[str, float])
    """Each answer's sum, rounded once, where it lies off 1 by more than the flag allows."""


def _judge(batch: _Batch, returned: _Returned, types: Mapping[str, Type]) -> _Judged:
    provenance: dict[str, str | None] = dict.fromkeys(_PROVENANCE)
    judged = _Judged(None, {}, provenance)
    codes: list[CauseCode] = []  # in fault-scope order; the first decides
    given: dict[str, object] = {}
    result = returned.result
    if returned.expired:
        codes.append(_C.timed_out)
    elif returned.error is not None:
        codes.append(_C.raised)
    elif type(result) is Failed:
        cause = getattr(result, "cause", None)
        code = getattr(cause, "code", None) if type(cause) is Cause else None
        if type(code) is CauseCode and code in _STRUCTURAL:
            codes.append(code)
        else:
            codes.append(_C.invalid_return)
            judged.invalid["code"] = guarded_repr(code if type(cause) is Cause else cause)
    elif type(result) is CallResult:
        values = cast(object, getattr(result, "values", None))
        if type(values) is not dict:
            codes.append(_C.invalid_return)
        else:
            provides = frozenset(batch.refs)
            items = list(cast("dict[object, object]", values).items())
            keys = [string_data(cast(str, k)) for k, _ in items if is_a(k, str)]
            if len(set(keys)) != len(keys):  # two keys of one string data
                codes.append(_C.invalid_return)
                items = []
            for key, value in items:
                ref = string_data(cast(str, key)) if is_a(key, str) else None
                if ref is not None and ref in provides and ref not in given:
                    given[ref] = value
                else:
                    judged.unexpected.append(ref if ref is not None else guarded_repr(key))
            if judged.unexpected:
                codes.append(_C.unexpected_ref)
        for name in _PROVENANCE:
            value = cast(object, getattr(result, name, None))
            if value is None:
                continue
            # A string's length is tested before its code points are read.
            if is_a(value, str) and domain_error(value) is None:
                provenance[name] = string_data(cast(str, value))
            else:
                judged.invalid[name] = guarded_repr(value)
                codes.append(_C.invalid_return)
        judged.usage = _usage(cast(object, getattr(result, "usage", None)), judged.invalid)
        judged.attempts = _attempts(cast(object, getattr(result, "attempts", None)), judged.invalid)
    else:
        codes.append(_C.invalid_return)
    if codes:
        judged.failed = Cause(codes[0], list(batch.refs))
        judged.provenance = dict.fromkeys(_PROVENANCE)
        marker = Unavailable(judged.failed)
        judged.written = dict.fromkeys(batch.refs, marker)
        return judged
    for ref in batch.refs:
        if ref not in given:
            judged.written[ref] = Unavailable(Cause(_C.not_returned, [ref]))
        else:
            written = _returned_value(ref, given[ref], batch, types)
            judged.written[ref] = written
            if ref in batch.questions and type(written) is tuple:
                # The flag changes no outcome: the answer is written as returned.
                total = off_sum(cast(Answer, written[1]))
                if total is not None:
                    judged.sums[ref] = total
    return judged


def _returned_value(ref: str, value: object, batch: _Batch, types: Mapping[str, Type]) -> Written:
    """A returned value or marker as written at its ref."""
    if is_a(value, Undetermined):
        return _binding_marker(ref, value)
    if ref in batch.questions:
        kind = AnswerKind(batch.questions[ref].kind)
        try:
            built = Answer(kind, batch.labels[ref], cast("dict[str, float]", value))
        except ValueError:
            return Unavailable(Cause(_C.invalid_answer, [ref]))
        return answer_document(built), built
    forms = typed(value, types[ref])
    return Unavailable(Cause(_C.invalid_return, [ref])) if forms is None else forms


def _binding_marker(ref: str, value: object) -> Unavailable:
    """One of the three markers a binding may return, naming ``ref`` alone; any other marker
    is ``invalid-return``."""
    if type(value) is Unavailable:
        cause = getattr(value, "cause", None)
        code = getattr(cause, "code", None) if type(cause) is Cause else None
        if type(code) is CauseCode and code in _BINDING_MARKERS:
            return Unavailable(Cause(code, [ref]))
    return Unavailable(Cause(_C.invalid_return, [ref]))


def _finite_number(value: object) -> bool:
    if type(value) is int:
        return abs(value) <= MAX_INTEGER
    return type(value) is float and math.isfinite(value)


def _usage(value: object, invalid: dict[str, str]) -> dict[str, float] | None:
    """A ``usage`` of the right type, else null with the value kept in ``invalid``."""
    if value is None:
        return None
    # In the domain, size bounds included, each key's length tested before its code points.
    if type(value) is dict and domain_error(cast(object, value)) is None:
        out: dict[str, float] = {}
        for key, number in cast("dict[object, object]", value).items():
            if type(key) is not str or not _finite_number(number):
                break
            out[key] = cast(float, number)
        else:
            return out
    invalid["usage"] = guarded_repr(cast(object, value))
    return None


def _attempts(value: object, invalid: dict[str, str]) -> int | None:
    if value is None:
        return None
    if type(value) is int and abs(value) <= MAX_INTEGER:
        return value
    invalid["attempts"] = guarded_repr(value)
    return None


def _rule_marker(value: object) -> Undetermined | None:
    """A returned marker that is exactly one of the three, holding a well-formed ``Cause``,
    rebuilt; else ``None``."""
    kind = cast("type[Undetermined]", type(value))
    if kind not in _MARKER_KINDS:
        return None
    cause = getattr(value, "cause", None)
    if type(cause) is not Cause:
        return None
    code = getattr(cause, "code", None)
    refs = cast(object, getattr(cause, "refs", None))
    if type(code) is not CauseCode or type(refs) is not tuple:
        return None
    members = cast("tuple[object, ...]", refs)
    if not all(type(r) is str and is_scalar_string(r) for r in members):
        return None
    return kind(Cause(code, cast("tuple[str, ...]", members)))


def _marker_document(marker: Undetermined) -> dict[str, object]:
    cause = marker.cause
    return {
        "kind": _MARKER_KINDS[type(marker)],
        "cause": {"code": cause.code.value, "refs": list(cause.refs)},
    }


# ---------------------------------------------------------------------------------------------
# The board


def _fresh(value: object) -> object:
    """A copy of a board value no change reaches back from."""
    if type(value) is dict:
        return {k: _fresh(v) for k, v in cast("dict[str, object]", value).items()}
    if type(value) is list:
        return [_fresh(v) for v in cast("list[object]", value)]
    if type(value) is tuple:
        return tuple(_fresh(v) for v in cast("tuple[object, ...]", value))
    if type(value) is Answer:
        return Answer(value.kind, value.labels, dict(value.probabilities))
    return value


def _render(template: str, values: Mapping[str, object]) -> str:
    """A wording rendered in one left-to-right pass: each escape as its brace and each
    placeholder as its slot's value rendered, never read for braces. Load
    check 9 has checked the template, so every placeholder names a slot."""
    out: list[str] = []
    i, n = 0, len(template)
    while i < n:
        pair = template[i : i + 2]
        if pair in ("{{", "}}"):
            out.append(pair[0])
            i += 2
        elif pair[0] == "{":
            close = template.index("}", i)
            out.append(_text(values[template[i + 1 : close]]))
            i = close + 1
        else:
            out.append(template[i])
            i += 1
    return "".join(out)


def _text(value: object) -> str:
    """A slot value by the declared-text closed rule: a string as itself, a list as its
    members' renderings joined by ``, ``, any other value as its canonical encoding, a
    ``bytes`` value as its record form's."""
    if type(value) is str:
        return value
    if type(value) in (list, tuple):
        return ", ".join(_text(member) for member in cast("Sequence[object]", value))
    return encode(_record_form(value)).decode("utf-8")


def _record_form(value: object) -> object:
    """A live board or leaf value in the record's form: each ``bytes`` as unpadded base64url. A
    slot's value is never an answer: load refuses a ``q:`` slot."""
    if type(value) is bytes:
        return b64url_encode(value)
    if type(value) is dict:
        return {k: _record_form(v) for k, v in cast("dict[str, object]", value).items()}
    if type(value) in (list, tuple):
        return [_record_form(v) for v in cast("Sequence[object]", value)]
    return value


# ---------------------------------------------------------------------------------------------
# The turn


class _Turn:
    def __init__(
        self,
        ledger: Ledger,
        loaded: LoadedDesign,
        state: object,
        bindings: Mapping[str, OracleBinding],
        round_bound: int,
        timeout: float,
        carried: frozenset[str],
        limits: _Limits,
    ) -> None:
        self.ledger = ledger
        self.loaded = loaded
        self.sources: Mapping[str, Source] = loaded._design.sources
        self.rules = cast("Mapping[str, Rule]", loaded._design.rules)  # load refuses tables
        self.bindings = bindings
        self.round_bound = round_bound
        self.timeout = timeout
        self.carried = carried
        self.limits = limits
        self.board: dict[str, object] = {_STATE: state}
        self.sorter = TurnSorter(loaded._graph, carried)
        self.ran: set[str] = set()
        self.spent = Fraction(0)
        self.document = cast("dict[str, Any]", decode(loaded.document))["sources"]

    async def run(self) -> dict[str, object]:
        """Run the rounds and return the live board, which compile reads."""
        held = self.rule_phase(0)
        n = 0
        while held:  # the fixpoint: no source held after a rule phase
            n += 1
            cost, cut = await self.source_phase(held, n)
            if cut is not None:  # the budget or the deadline ended the turn's asking
                held = []
            else:
                held = self.rule_phase(n)
                if held and n >= self.round_bound:
                    self.end(Unasked(Cause(_C.round_bound)), held, n)
                    held = []
            # The exact sum, rounded once to the nearest float.
            rounded = float(cost)
            cut_by_deadline = cut is _C.deadline
            self.ledger.append(
                {"item": "round", "round": n, "cost": rounded, "deadline": cut_by_deadline}
            )
        if self.sorter.is_active():  # every node is done when interpret ends
            raise _InterpretError("interpret ended with a node not done")
        return self.board

    # -- the rule phase ----------------------------------------------------------------------

    def rule_phase(self, n: int) -> list[str]:
        """Fire wavefronts until one holds no rule; return the sources the wavefronts held."""
        held: list[str] = []
        while True:
            wave = self.sorter.get_ready()
            rules = code_point_sorted(node for node in wave if node in self.rules)
            held += [node for node in wave if node not in self.rules]
            if not rules:
                return held
            for rid in rules:
                self.fire(rid, n)
                self.sorter.done(rid)

    def fire(self, rid: str, n: int) -> None:
        """Call a rule on its read view and write its entry by the read view's order of outcomes:
        the first undeclared read, the first ``require`` stop, a raise, an invalid return,
        the returned value."""
        r = self.rules[rid]
        leaves = {path: self.loaded._leaf_value(path) for path in r.params}
        view = ReadView(Role(ComponentKind.rule, rid), r.reads, r.params, self.board, leaves)
        called = call(self.loaded._callables[rid], view)
        stop, error, returned = called.stop, called.error, called.returned
        unnamed: str | None = None
        written: Written
        if isinstance(stop, UndeclaredRead):
            if names_read(self.loaded, stop):
                written = Unavailable(Cause(_C.undeclared_read, [rid, stop.read]))
            else:
                written = Unavailable(Cause(_C.undeclared_read, [rid]))
                unnamed = stop.read
        elif isinstance(stop, RequireStopped):
            written = type(stop.marker)(Cause(_C.required, [stop.ref]))
        elif error is not None:
            written = Unavailable(Cause(_C.raised, [rid]))
        elif called.awaitable:  # tested before any other test of the return
            written = Unavailable(Cause(_C.invalid_return, [rid]))
        else:
            # Never a truth test on what a rule returned: a marker's `__bool__` raises.
            if is_a(returned, Undetermined):
                value: Written | None = _rule_marker(returned)
            else:
                value = typed(returned, self.loaded._types[r.provides])
            written = Unavailable(Cause(_C.invalid_return, [rid])) if value is None else value
        index = self.write(r.provides, n, None, written, called.derivation)
        if called.text is not None:  # whatever the outcome, a stop's included
            self.ledger.write_unhashed("exceptions", exception_record(called.text), index)
        if unnamed is not None:
            self.ledger.write_unhashed("reads", unnamed, index)

    # -- the source phase --------------------------------------------------------------------

    async def source_phase(self, held: list[str], n: int) -> tuple[Fraction, CauseCode | None]:
        """Ask or mark each held source and write the round's call items and source entries;
        then mark each held source done, or, when the budget or the deadline cuts the round,
        end it. Return the round's cost, exactly, and the cut's
        code, or ``None``."""
        order = code_point_sorted(held)
        marked: dict[str, Undetermined] = {}
        texts: dict[str, _WhenText] = {}
        ready: list[str] = []
        # Readiness in source id order, before the TaskGroup, so a function `when` that
        # raises a cancellation falls at a fixed place. A source that
        # is not ready gets its own markers before the budget and the deadline are tested.
        for sid in order:
            marker = self.need_marker(sid)
            if marker is None:
                marker = self.requirement_marker(sid)
            if marker is None:
                marker = self.when_marker(sid, texts)
            if marker is None:
                ready.append(sid)
            else:
                marked[sid] = marker
        batches = self.batches(ready)
        cut = self.cut_before(batches, n)
        if cut is not None:
            batches = []  # the frontier is not asked, whole
        returned = await self.ask(batches)
        if cut is None and self.cut_after(batches, returned, n):
            cut = _C.deadline
        calls: dict[str, tuple[int, _Judged]] = {}
        cost = Fraction(0)
        for batch, outcome in zip(batches, returned, strict=True):
            judged = _judge(batch, outcome, self.loaded._types)
            index = self.ledger.append(self.call_item(batch, judged, n))
            unhashed: dict[str, object] = {
                "start": outcome.start,
                "latency": outcome.latency,
                "usage": judged.usage,
                "attempts": judged.attempts,
            }
            if judged.unexpected:
                unhashed["unexpected"] = judged.unexpected
            if judged.invalid:
                unhashed["invalid"] = judged.invalid
            if judged.sums:
                unhashed["sums"] = judged.sums
            self.ledger.write_unhashed("calls", unhashed, index)
            if outcome.error is not None:
                self.ledger.write_unhashed("exceptions", exception_record(outcome.error), index)
            for sid in batch.sources:
                calls[sid] = (index, judged)
            cost += batch.cost
        for sid in order:
            if sid not in marked and sid not in calls:
                continue  # in a frontier the cut left unasked: the end marks it
            for i, ref in enumerate(code_point_sorted(self.sources[sid].provides)):
                if sid in marked:
                    index = self.write(ref, n, None, marked[sid])
                    if i == 0 and sid in texts:  # keyed under the lowest ref
                        self.write_when_text(texts[sid], index)
                else:
                    index, judged = calls[sid]
                    self.write(ref, n, index, judged.written[ref])
            self.ran.add(sid)
        self.spent += cost  # the budget's running total, kept exact
        if cut is None:
            for sid in order:
                self.sorter.done(sid)
        else:
            self.end(Unasked(Cause(cut)), order, n)
        return cost, cut

    def cut_before(self, batches: list[_Batch], n: int) -> CauseCode | None:
        """Whether the round is cut before its frontier is asked: by the budget when the cost
        spent plus the frontier's exceeds it, exactly; then by the deadline once the
        clock reads at least its end, or under replay at a cut round
        that was not asked."""
        budget, end, cut = self.limits.budget, self.limits.end, self.limits.cut
        if budget is not None:
            frontier = sum((b.cost for b in batches), Fraction(0))
            if self.spent + frontier > budget:
                return _C.over_budget
        if end is not None and _clock() >= end:
            return _C.deadline
        if cut == (n, False):
            return _C.deadline
        return None

    def cut_after(self, batches: list[_Batch], returned: list[_Returned], n: int) -> bool:
        """Whether the deadline has passed when the frontier's task group closes, or under
        replay this is the cut round and it was asked. A call the deadline's end expired counts as
        passed: asyncio fires a timeout up to its clock resolution early, so the clock may still
        read under the end."""
        end = self.limits.end
        expired = any(r.expired and r.by_deadline for r in returned)
        if end is not None and batches and (_clock() >= end or expired):
            return True
        return self.limits.cut == (n, True)

    def need_marker(self, sid: str) -> Undetermined | None:
        """A source's own marker when a need is undetermined, by the ref grammar's precedence,
        naming every need of the winning kind, a source requirement's ref counting as one; ``None``
        when every one is determined."""
        needs = set(self.loaded._needs[sid])
        requirement = self.sources[sid].requirement
        if requirement is not None:
            needs.add(requirement.ref)
        bases = {need: self.board[base(need)] for need in needs}
        unavailable = [need for need in needs if isinstance(bases[need], Unavailable)]
        if unavailable:
            return Unavailable(Cause(_C.need_unavailable, code_point_sorted(unavailable)))
        undetermined = [need for need in needs if isinstance(bases[need], Undetermined)]
        if undetermined:
            return Unasked(Cause(_C.need_undetermined, code_point_sorted(undetermined)))
        return None

    def requirement_marker(self, sid: str) -> Undetermined | None:
        """``Unasked`` with ``when-false`` naming the ref when the source requirement's
        determined entry equals none of its values by canonical bytes, never by ``==``; ``None``
        when it holds or there is none."""
        requirement = self.sources[sid].requirement
        if requirement is None:
            return None
        # Check 9 types the ref with no bytes at any depth, so its live form is a domain value
        # the encoding takes; the values are load's own bytes, never the caller's objects.
        if encode(self.board[requirement.ref]) in self.loaded._requirement_bytes[sid]:
            return None
        return Unasked(Cause(_C.when_false, [requirement.ref]))

    def when_marker(self, sid: str, texts: dict[str, _WhenText]) -> Undetermined | None:
        """Call a function ``when`` on its read view and give the source's marker by
        the read view's order of outcomes: the first undeclared read, a raise, an awaitable
        returned, a truth test that raises, then false; ``None`` when it is true. Its text
        goes in ``texts``."""
        source = self.sources[sid]
        if source.when is None:
            return None
        state = self.loaded._types[_STATE]
        view = ReadView(
            Role(ComponentKind.when, sid), source.needs, (), self.board, {}, state=state
        )
        called: Called = call(self.loaded._callables[f"{sid}.when"], view)
        stop, error = called.stop, called.text
        unnamed: str | None = None
        marker: Undetermined | None
        if isinstance(stop, UndeclaredRead):
            if names_read(self.loaded, stop):
                marker = Unavailable(Cause(_C.undeclared_read, [sid, stop.read]))
            else:
                marker = Unavailable(Cause(_C.undeclared_read, [sid]))
                unnamed = stop.read
        elif isinstance(stop, RequireStopped):
            # Every need is determined and the state holds no marker.
            raise _InterpretError(f"require stopped the when of {sid!r} on a determined read")
        elif called.error is not None or called.awaitable:  # no truth test on an awaitable
            marker = Unavailable(Cause(_C.when_failed, [sid]))
        else:
            try:  # inside the protected region
                marker = None if bool(called.returned) else Unasked(Cause(_C.when_false, [sid]))
            except _Stop:  # a kept stop raised again: read from the record, no text
                marker = Unavailable(Cause(_C.when_failed, [sid]))
            except Exception as exc:
                error = exc
                marker = Unavailable(Cause(_C.when_failed, [sid]))
        if error is not None or unnamed is not None:
            texts[sid] = _WhenText(error, unnamed)
        return marker

    def write_when_text(self, text: _WhenText, index: int) -> None:
        if text.error is not None:
            self.ledger.write_unhashed("exceptions", exception_record(text.error), index)
        if text.unnamed is not None:
            self.ledger.write_unhashed("reads", text.unnamed, index)

    def batches(self, ready: list[str]) -> list[_Batch]:
        """The live frontier's batches: the ready sources sharing a batch key form one, each
        keyless source one of its own, in order of their first source id. ``ready`` is in source id
        order."""
        groups: list[list[str]] = []
        keyed: dict[str, list[str]] = {}
        for sid in ready:
            key = self.sources[sid].batch
            if key is None:
                groups.append([sid])
            elif key in keyed:
                keyed[key].append(sid)
            else:
                keyed[key] = [sid]
                groups.append(keyed[key])
        return [self.batch(tuple(group)) for group in groups]

    def batch(self, sids: tuple[str, ...]) -> _Batch:
        refs: list[str] = []
        questions: dict[str, Question] = {}
        labels: dict[str, tuple[str, ...]] = {}
        calls: list[SourceCall] = []
        for sid in sids:
            source = self.sources[sid]
            provided = code_point_sorted(source.provides)
            # A fresh copy per source, so no source's change reaches another's.
            needs = {
                need: _fresh(at(self.board, need))
                for need in code_point_sorted(self.loaded._needs[sid])
            }
            rendered: dict[str, RenderedQuestion] = {}
            for ref in provided:
                q = source.provides[ref]
                if isinstance(q, Question):
                    kind = AnswerKind(q.kind)
                    passed = _NOUL_LABELS if kind is AnswerKind.noul else tuple(cast(Any, q.labels))
                    questions[ref], labels[ref] = q, passed
                    wording = _render(q.wording.template, self.slot_values(q, needs))
                    rendered[ref] = RenderedQuestion(kind, wording, passed)
            t_refs = tuple(ref for ref in provided if ref not in rendered)
            calls.append(SourceCall(sid, self.loaded._payload(sid), needs, t_refs, rendered))
            refs += provided
        declared = [s.timeout for s in (self.sources[sid] for sid in sids) if s.timeout is not None]
        key = self.sources[sids[0]].batch
        return _Batch(
            key=_DEFAULT if key is None else key,
            sources=sids,
            refs=tuple(code_point_sorted(refs)),
            questions=questions,
            labels=labels,
            call=tuple(calls),
            # The largest declared, else the run setting's.
            timeout=max(declared) if declared else self.timeout,
            cost=sum((_exact(self.sources[sid].cost) for sid in sids), Fraction(0)),
        )

    def slot_values(self, q: Question, needs: Mapping[str, object]) -> dict[str, object]:
        """Each slot's value by name: a board ref's from the source's needs, where load added
        it, and a parameter's from the parameter block."""
        values: dict[str, object] = {}
        for slot in q.wording.slots:
            tag, origin = next(iter(slot["origin"].items()))
            if tag == "ref":
                values[slot["name"]] = needs[origin]
            elif tag == "param":
                values[slot["name"]] = self.loaded._leaf_value(origin)
            else:  # pragma: no cover - load refuses a candidate slot in v0.0.1
                raise _InterpretError(f"a {tag} slot reached interpret")
        return values

    def identity(self, ref: str) -> str:
        """The question identity: the hash of the question's design-document object, each
        parameter origin with its leaf's value in record form, no other slot's value. The document
        is left as it was."""
        declared = cast("dict[str, Any]", self.document[self.loaded._providers[ref]])
        question = cast("dict[str, Any]", declared["provides"][ref])
        wording = cast("dict[str, Any]", question["wording"])
        slots: list[dict[str, Any]] = []
        for slot in cast("list[dict[str, Any]]", wording["slots"]):
            origin = cast("dict[str, Any]", slot["origin"])
            if "param" in origin:
                path = cast(str, origin["param"])
                origin = {"param": path, "value": decode(self.loaded._leaf_bytes[path])}
            slots.append({**slot, "origin": origin})
        # The question's object is a holder of design members, held to no size bound.
        return hash_value({**question, "wording": {**wording, "slots": slots}}, MAX_DOCUMENT_DEPTH)

    async def ask(self, batches: list[_Batch]) -> list[_Returned]:
        """Run the live frontier's calls concurrently, one task per batch; a task that ended
        cancelled on its own is re-raised here, before the round's first append."""
        if not batches:
            return []
        end = self.limits.end
        # No timeout on the group: each task holds its own, cut to the deadline.
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(_call(self.bindings[b.key], b.call, b.timeout, end))
                for b in batches
            ]
        for task in tasks:  # in batch order, before the round's first append
            if task.cancelled():
                task.result()  # raises the task's CancelledError
            escaped = task.result().escaped
            if escaped is not None:
                raise escaped
        return [task.result() for task in tasks]

    def call_item(self, batch: _Batch, judged: _Judged, n: int) -> dict[str, object]:
        questions: dict[str, object] = {}
        for ref in code_point_sorted(batch.questions):
            # The kind and labels are those the identity hashes.
            declared = self.document[self.loaded._providers[ref]]["provides"][ref]
            questions[ref] = {
                "identity": self.identity(ref),
                "kind": declared["kind"],
                "labels": declared["labels"],
            }
        failed = judged.failed
        outcome: object = (
            "returned"
            if failed is None
            else {"failed": {"code": failed.code.value, "refs": list(failed.refs)}}
        )
        return {
            "item": "call",
            "round": n,
            "sources": list(batch.sources),
            "outcome": outcome,
            **judged.provenance,
            "questions": questions,
        }

    # -- the end and the drain ---------------------------------------------------------------

    def end(self, marker: Unasked, held: list[str], n: int) -> None:
        """End the turn's asking, as the round bound, the budget and the deadline each do:
        mark every pending source ``Unasked`` with ``marker`` by source id, mark the held
        sources done and drain the sorter."""
        asked = self.ran | self.carried
        pending = [s for s in code_point_sorted(self.sources) if s not in asked]
        for sid in pending:
            for ref in code_point_sorted(self.sources[sid].provides):
                self.write(ref, n, None, marker)
        for sid in held:
            self.sorter.done(sid)
        self.drain(n)

    def drain(self, n: int) -> None:
        """``get_ready()`` until ``is_active()`` is false: a source is done with no write, a
        rule fires and is done. An empty wavefront while one is active is a returned node
        never done, which would spin."""
        while self.sorter.is_active():
            wave = self.sorter.get_ready()
            if not wave:
                raise _InterpretError("the drain found no node ready while a node is active")
            for node in code_point_sorted(wave):
                if node in self.rules:
                    self.fire(node, n)
                self.sorter.done(node)

    # -- the write ---------------------------------------------------------------------------

    def write(
        self,
        ref: str,
        n: int,
        call: int | None,
        written: Written,
        derivation: list[dict[str, list[str] | str]] | None = None,
    ) -> int:
        item: dict[str, object] = {"item": "entry", "ref": ref, "round": n, "call": call}
        if isinstance(written, Undetermined):  # the engine's own marker
            item["marker"] = _marker_document(written)
            live: object = written
        else:
            item["value"], live = written
        if derivation is not None:
            item["derivation"] = derivation
        index = self.ledger.append(item)
        self.board[ref] = live
        return index
