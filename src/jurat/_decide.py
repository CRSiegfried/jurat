# pyright: reportPrivateUsage=false
"""Decide: the engine's one operation.

``decide`` refuses a wrong caller argument with ``EntryRefusal`` before anything runs. Then it reads
the monotonic clock, opens the ledger, writes ``unhashed``'s ``turn.start``, ``engine`` and
``codec``, appends the ``design`` and ``settings`` items and the state entry ``t:state``, runs
interpret and then compile, and returns the frozen ``Decision``. A codec that raises an
``Exception``, or a state that does not conform, is a broken entry: decide builds the ``broken``
plan, calls the guard over an empty board and appends ``plan`` and ``verdict`` itself.

One handler sorts what ends a turn early by identity first: the observer's exception closes
the record and propagates with it; a cancellation, one the codec raises included,
appends the ``cause`` item, closes the record and propagates with it;
anything else propagates bare, with no record.

``_turn`` is the body, which replay's internal entry reruns with the record's state entry as
the state, its run settings, the fixture binding, no codec and the record's cut. There
``clock`` is ``None``: decide reads no clock and interpret enforces no deadline, stopping at
the cut instead. What ``_turn`` takes is checked values, never a
``RunSettings``.
"""

import asyncio
import contextlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, cast

from jurat._binding import OracleBinding
from jurat._compile import compile_turn, plan_item
from jurat._component import exception_record, now, refuse_awaitable
from jurat._design import LoadedDesign
from jurat._guard import Verdict, guard, verdict_item
from jurat._interpret import interpret
from jurat._ledger import Ledger, Observer, Record
from jurat._markers import Cause, CauseCode, PlanKind
from jurat._plan import Plan, engine_plan
from jurat._refs import code_point_sorted
from jurat._typed import is_a, typed
from jurat._values import MAX_INTEGER, UNREPRESENTABLE, described, quote, string_data, type_name
from jurat.helpers import HELPERS

_ENGINE = version("jurat")
_FORMAT = 1
_STATE = "t:state"
_CODEC = "codec"
_DEFAULT = "default"

Codec = Callable[[Any], object]
"""The caller's sync encoder from its state object to a value of the declared type."""


class EntryRefusal(ValueError):
    """A wrong caller argument at decide's entry, refused before any record is written."""


@dataclass(frozen=True, slots=True)
class RunSettings:
    """The run settings, each with its default. Validates
    nothing: decide refuses a setting outside its bounds."""

    round_bound: int = 8
    timeout: float = 60
    budget: float | None = None
    deadline: float | None = None
    certificate_cap: int | None = None


_BUILT_BY_DECIDE = object()


class Decision:
    """A turn's decision: the plan, the guard verdict and the record; the design hash is the
    record's. Frozen, and built only by decide."""

    __slots__ = ("_plan", "_record", "_verdict")

    _plan: Plan
    _verdict: Verdict
    _record: Record

    def __init__(self, token: object, plan: Plan, verdict: Verdict, record: Record) -> None:
        if token is not _BUILT_BY_DECIDE:
            raise TypeError("a Decision is built only by decide")
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_verdict", verdict)
        object.__setattr__(self, "_record", record)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a Decision is frozen")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a Decision is frozen")

    @property
    def plan(self) -> Plan:
        return self._plan

    @property
    def verdict(self) -> Verdict:
        return self._verdict

    @property
    def record(self) -> Record:
        return self._record

    @property
    def design_hash(self) -> str:
        """The record's design hash, read from it and stored nowhere."""
        return self._record.design_hash

    def __repr__(self) -> str:
        return f"Decision(plan={self._plan!r}, verdict={self._verdict!r}, record={self._record!r})"


# ---------------------------------------------------------------------------------------------
# The public entry


async def decide(
    loaded: LoadedDesign,
    state: object,
    bindings: Mapping[str, OracleBinding],
    settings: RunSettings | None = None,
    *,
    codec: Codec | None = None,
    observer: Observer | None = None,
) -> Decision:
    """Run one turn: the entry refusals, then the body over a ledger opened with ``observer``;
    return the decision, or propagate a cancellation or the observer's exception with the
    record so far as its ``record``."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    _callable_or_none(codec, "codec")
    _callable_or_none(observer, "observer")
    if settings is None:
        settings = RunSettings()
    elif type(settings) is not RunSettings:
        raise EntryRefusal(
            f"settings is a {type_name(settings)}, neither None nor exactly a RunSettings"
        )
    copied = _copied(bindings)
    run = run_settings(
        settings.round_bound,
        settings.timeout,
        settings.budget,
        settings.deadline,
        settings.certificate_cap,
    )
    bound = _bound(loaded, copied)
    # The running loop's clock, the one asyncio's timeouts read: the deadline runs from here,
    # so the codec's time counts against it.
    clock = asyncio.get_running_loop().time()
    return await _turn(
        loaded, state, bound, run, codec=codec, clock=clock, cut=None, observer=observer
    )


def decide_sync(
    loaded: LoadedDesign,
    state: object,
    bindings: Mapping[str, OracleBinding],
    settings: RunSettings | None = None,
    *,
    codec: Codec | None = None,
    observer: Observer | None = None,
) -> Decision:
    """``decide`` run to completion; ``RuntimeError`` inside a running event loop, raised
    before the coroutine is built."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("decide_sync was called inside a running event loop; await decide")
    return asyncio.run(decide(loaded, state, bindings, settings, codec=codec, observer=observer))


# ---------------------------------------------------------------------------------------------
# The entry refusals


@dataclass(frozen=True, slots=True)
class _Run:
    """The run settings in force, each checked."""

    round_bound: int
    timeout: float
    budget: float | None
    deadline: float | None
    certificate_cap: int | None

    def item(self) -> dict[str, object]:
        return {
            "round_bound": self.round_bound,
            "timeout": self.timeout,
            "budget": self.budget,
            "deadline": self.deadline,
            "certificate_cap": self.certificate_cap,
        }


def run_settings(
    round_bound: object, timeout: object, budget: object, deadline: object, certificate_cap: object
) -> _Run:
    """The run settings in force from five values each read once, or ``EntryRefusal`` for one
    outside its bounds, its exact type included; replay passes the record's."""
    if not _integer(round_bound) or cast(int, round_bound) < 1:
        raise EntryRefusal(
            f"the round bound {described(round_bound)} is not an integer of at least 1"
        )
    if not _number(timeout) or cast(float, timeout) <= 0:
        raise EntryRefusal(f"the timeout {described(timeout)} is not a positive number of seconds")
    if budget is not None and (not _number(budget) or cast(float, budget) < 0):
        raise EntryRefusal(f"the budget {described(budget)} is not a non-negative number or None")
    if deadline is not None and (not _number(deadline) or cast(float, deadline) <= 0):
        raise EntryRefusal(
            f"the deadline {described(deadline)} is not a positive number of seconds or None"
        )
    if certificate_cap is not None:
        # This engine does not implement the certificate.
        raise EntryRefusal(f"the certificate cap {described(certificate_cap)} is not None")
    return _Run(
        cast(int, round_bound),
        cast(float, timeout),
        cast("float | None", budget),
        cast("float | None", deadline),
        None,
    )


def _integer(value: object) -> bool:
    """An ``int``, never a ``bool`` or a subclass, that the ``settings`` item can hold."""
    return type(value) is int and abs(value) <= MAX_INTEGER


def _number(value: object) -> bool:
    """An integer the ``settings`` item can hold, or a finite ``float``, by exact type."""
    return _integer(value) or (type(value) is float and math.isfinite(value))


def _callable_or_none(value: object, name: str) -> None:
    """``None`` or callable; ``callable`` reads the type's call slot and runs no caller code."""
    if value is not None and not callable(value):
        raise EntryRefusal(f"the {name} is a {type_name(value)}, neither None nor callable")


def _copied(bindings: object) -> dict[object, object]:
    """The bindings copied into a ``dict`` once; only the copy is read after."""
    try:  # the ABC's subclass check reads the type, which a metaclass can make raise
        mapping = is_a(bindings, Mapping)
    except Exception:
        mapping = False
    if not mapping:
        raise EntryRefusal(f"bindings is a {type_name(bindings)}, not a Mapping")
    try:
        return dict(cast("Mapping[object, object]", bindings))
    except Exception as exc:
        raise EntryRefusal(f"the bindings could not be copied: {type_name(exc)}") from exc


def _bound(loaded: LoadedDesign, copied: dict[object, object]) -> dict[str, OracleBinding]:
    """The bindings by oracle name: one for every oracle a source names and for ``default``
    when a source names none, and none for a key the design does not declare."""
    named: dict[str, OracleBinding] = {}
    for key, binding in copied.items():
        if not issubclass(type(key), str):
            raise EntryRefusal(
                f"the bindings key {described(key)} is not an oracle name the design declares"
            )
        name = string_data(cast(str, key))
        if name in named:
            raise EntryRefusal(f"two bindings keys name {quote(name)}")
        named[name] = cast(OracleBinding, binding)
    sources = loaded._design.sources.values()
    declared = {s.oracle for s in sources if s.oracle is not None}
    for name in code_point_sorted(named):
        if name != _DEFAULT and name not in declared:
            raise EntryRefusal(
                f"the bindings key {quote(name)} is not an oracle name the design declares"
            )
    needed = set(declared)
    if any(s.oracle is None for s in sources):
        needed.add(_DEFAULT)
    for name in code_point_sorted(needed):
        if name not in named:
            if name == _DEFAULT:
                raise EntryRefusal(
                    f"no binding is given under {quote(name)} for the sources with no oracle"
                )
            raise EntryRefusal(f"no binding is given for the oracle {quote(name)}")
    return named


# ---------------------------------------------------------------------------------------------
# The body


async def _turn(
    loaded: LoadedDesign,
    state: object,
    bindings: dict[str, OracleBinding],
    run: _Run,
    *,
    codec: Codec | None,
    clock: float | None,
    cut: tuple[int, bool] | None,
    observer: Observer | None,
) -> Decision:
    """The turn from the ledger's opening to the commit point. ``state`` is the codec's
    argument, or the value itself when there is no codec. ``clock`` is the
    monotonic reading at entry, or ``None`` under replay's internal entry: no clock is read and
    interpret enforces no deadline, but stops at ``cut``, the record's cut round and whether
    it was asked, or ``None`` when the record has none."""
    ledger = Ledger(observer)
    try:
        ledger.write_unhashed("turn", {"start": now()})
        ledger.write_unhashed("engine", _ENGINE)
        if codec is not None:
            ledger.write_unhashed("codec", _codec_name(codec))
        ledger.append(
            {"item": "design", "hash": loaded.design_hash, "format": _FORMAT, "helpers": HELPERS}
        )
        ledger.append(
            {
                "item": "settings",
                "settings": run.item(),
                "providers": dict(loaded._providers),
                "classes": loaded._classes,
            }
        )
        value = state
        if codec is not None:
            text = None
            awaitable, closing = False, None
            try:
                value = codec(state)
                # Before typed, a returned coroutine closed; the awaitable test's raise is the
                # codec's.
                awaitable, closing = refuse_awaitable(value)
            except Exception as exc:  # a CancelledError is no Exception: it is delivered
                text = exception_record(exc)
            if text is not None:
                # Outside the except block, so an observer's raise is chained to nothing.
                return _broken(ledger, loaded, Cause(CauseCode.raised, [_CODEC]), text)
            if awaitable:
                text = None if closing is None else exception_record(closing)
                return _broken(ledger, loaded, Cause(CauseCode.invalid_return, [_STATE]), text)
        forms = typed(value, loaded._types[_STATE])
        if forms is None:
            return _broken(ledger, loaded, Cause(CauseCode.invalid_return, [_STATE]), None)
        recorded, live = forms
        # Checked and appended with no await between.
        ledger.append({"item": "entry", "ref": _STATE, "round": 0, "call": None, "value": recorded})
        board = await interpret(
            ledger,
            loaded,
            live,
            bindings,
            round_bound=run.round_bound,
            timeout=run.timeout,
            budget=run.budget,
            deadline=None if clock is None else run.deadline,
            clock=clock,
            cut=cut,
            carried=frozenset(),
        )
        plan, verdict = compile_turn(ledger, loaded, board)
        return Decision(_BUILT_BY_DECIDE, plan, verdict, _end(ledger))
    except BaseException as exc:
        if exc is ledger.observer_error:  # by identity, never by type
            _attach(exc, _end(ledger))
            raise
        if isinstance(exc, asyncio.CancelledError):
            try:
                ledger.append({"item": "cause", "cause": {"code": "cancelled", "refs": []}})
            except BaseException as raised:
                if raised is not ledger.observer_error:
                    raise
                # The cancellation wins, the observer's exception its context.
                _attach(exc, _end(ledger))
                raise exc  # noqa: B904 - the context is the observer's exception
            _attach(exc, _end(ledger))
        raise


def _broken(
    ledger: Ledger, loaded: LoadedDesign, cause: Cause, text: dict[str, str] | None
) -> Decision:
    """A broken entry: the ``broken`` plan with the entry cause, the guard's verdict over an
    empty board, both appended; a raising codec's text, or its closed coroutine's, under the
    ``plan`` item's index, taken at the catch and written after the append."""
    plan = engine_plan(PlanKind.broken, cause)
    index = ledger.append(plan_item(plan, None))
    if text is not None:
        ledger.write_unhashed("exceptions", text, index)
    design = loaded._design
    verdict = guard(plan, {}, design.requirements, design.default_deny)
    ledger.append(verdict_item(verdict))
    return Decision(_BUILT_BY_DECIDE, plan, verdict, _end(ledger))


def _end(ledger: Ledger) -> Record:
    """``turn.end``, the last write, then the close."""
    ledger.write_unhashed("turn", {"end": now()})
    return ledger.close()


def _attach(exc: BaseException, record: Record) -> None:
    """Set ``record`` on the propagating exception over any it holds; one that refuses the
    attribute propagates unchanged, with no record."""
    with contextlib.suppress(Exception):
        setattr(exc, "record", record)  # noqa: B010 - the attribute the caller reads


def _codec_name(codec: object) -> str:
    """The codec's ``__qualname__`` when that is a ``str``, else its type's, read inside a
    guard."""
    try:
        name = getattr(codec, "__qualname__", None)
        if not isinstance(name, str):
            name = type_name(codec)
        return string_data(name)
    except Exception:
        return UNREPRESENTABLE
