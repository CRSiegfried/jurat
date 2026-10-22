# pyright: reportPrivateUsage=false
"""A scripted test fixture binding, and the helpers that open a ledger
as decide will and run interpret over it.

``Scripted`` answers each call from a script keyed by the batch's first source id: a reply
is returned as it is, or, when it is a coroutine function, awaited with the batch. It records
every batch it was given. It is an oracle binding under the whole protocol and returns null
provenance unless a test scripts otherwise; a scripted snapshot, elicitation, prompt, usage or
attempts stands in for a live binding's.

``turn`` closes each record and checks it round-trips: no stored copy drifted, and
``read_record`` of its ``document`` gives the same bytes. ``compiled`` runs interpret and then
compile on the open ledger, as decide will, and checks the record so too.
"""

import asyncio
import contextlib
import types
import warnings
from collections.abc import Awaitable, Callable, Generator, Mapping
from typing import Any, cast

from jurat import (
    CallResult,
    Cause,
    CauseCode,
    LoadedDesign,
    Plan,
    Record,
    SourceCall,
    Unavailable,
    Verdict,
    read_record,
)
from jurat._compile import compile_turn
from jurat._interpret import interpret
from jurat._ledger import Ledger, Observer
from jurat._read_view import Access, UndeclaredRead, _UndeclaredStop
from jurat._values import FORMAT
from jurat.helpers import HELPERS

Batch = tuple[SourceCall, ...]
Reply = Callable[[Batch], Awaitable[object]]


class Scripted:
    """A fixture binding: each batch is answered by the reply under its first source id."""

    def __init__(self, replies: Mapping[str, object]) -> None:
        self.replies = dict(replies)
        self.batches: list[Batch] = []

    async def __call__(self, batch: Batch) -> object:
        self.batches.append(batch)
        reply = self.replies[batch[0].id]
        if callable(reply):
            return await cast(Reply, reply)(batch)
        return reply


def returns(values: dict[str, Any], **provenance: Any) -> CallResult:
    return CallResult(values, **provenance)


def unsupported(reason: str | None = "this provider takes no such question") -> Reply:
    """A binding's refusal before any call: every ref of the batch ``Unavailable`` with
    ``unsupported``, the reason a line of the call's ``warnings``."""

    async def reply(batch: Batch) -> object:
        marker = Unavailable(Cause(CauseCode.unsupported))
        refs = [ref for call in batch for ref in (*call.refs, *call.questions)]
        return CallResult(dict.fromkeys(refs, marker), warnings=() if reason is None else (reason,))

    return reply


def raising(exc: BaseException) -> Reply:
    async def reply(batch: Batch) -> object:
        raise exc

    return reply


def sleeping(seconds: float, then: object) -> Reply:
    async def reply(batch: Batch) -> object:
        await asyncio.sleep(seconds)
        return then

    return reply


def ignoring_cancel(seconds: float, then: object) -> Reply:
    """Swallows the cancellation its timeout delivers, then returns ``then``."""

    async def reply(batch: Batch) -> object:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(seconds)
        return then

    return reply


def raising_on_cancel(seconds: float, exc: Exception) -> Reply:
    """Turns the cancellation its timeout delivers into ``exc``."""

    async def reply(batch: Batch) -> object:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            raise exc from None
        return None

    return reply


def settings(
    round_bound: int = 8,
    timeout: float = 60,
    budget: float | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    return {
        "round_bound": round_bound,
        "timeout": timeout,
        "budget": budget,
        "deadline": deadline,
        "certificate_cap": None,
    }


def open_ledger(
    loaded: LoadedDesign,
    state: object,
    run: Mapping[str, Any],
    observer: Observer | None = None,
) -> Ledger:
    """A ledger opened as decide opens it: the design and settings items and the state entry.
    ``state`` is in the record's form."""
    ledger = Ledger(observer)
    ledger.append(
        {"item": "design", "hash": loaded.design_hash, "format": FORMAT, "helpers": HELPERS}
    )
    ledger.append(
        {
            "item": "settings",
            "settings": dict(run),
            "providers": dict(loaded._providers),
            "classes": loaded._classes,
        }
    )
    ledger.append({"item": "entry", "ref": "t:state", "round": 0, "call": None, "value": state})
    return ledger


async def interpret_on(
    ledger: Ledger,
    loaded: LoadedDesign,
    state: object,
    bindings: Mapping[str, object],
    run: Mapping[str, Any],
    *,
    clock: float | None = None,
    cut: tuple[int, bool] | None = None,
) -> dict[str, object]:
    """Interpret over the open ledger. The clock is the running loop's reading now, as decide
    takes it, unless given; with a ``cut`` there is no clock and no deadline, as under replay."""
    if cut is None and clock is None:
        clock = asyncio.get_running_loop().time()
    return await interpret(
        ledger,
        loaded,
        state,
        bindings,  # pyright: ignore[reportArgumentType]
        round_bound=run["round_bound"],
        timeout=run["timeout"],
        budget=run["budget"],
        deadline=None if cut is not None else run["deadline"],
        clock=clock,
        cut=cut,
        carried=frozenset(),
    )


def turn(
    loaded: LoadedDesign,
    bindings: Mapping[str, object],
    state: object,
    *,
    live: object = None,
    observer: Observer | None = None,
    clock: float | None = None,
    cut: tuple[int, bool] | None = None,
    **run: Any,
) -> Record:
    """Open the ledger, run interpret to its end, and close it. ``live`` is the state as
    decide passes it, ``state`` when not given; ``clock`` and ``cut`` as ``interpret_on``
    takes them."""
    full = settings(**run)
    ledger = open_ledger(loaded, state, full, observer)
    asyncio.run(
        interpret_on(
            ledger,
            loaded,
            state if live is None else live,
            bindings,
            full,
            clock=clock,
            cut=cut,
        )
    )
    return closed(ledger)


def closed(ledger: Ledger) -> Record:
    record = ledger.close()
    assert ledger.drifted_copies == ()
    assert read_record(record.document).document == record.document
    return record


def compiled(
    loaded: LoadedDesign,
    bindings: Mapping[str, object],
    state: object,
    *,
    observer: Observer | None = None,
    **run: Any,
) -> tuple[Record, Plan, Verdict]:
    """Open the ledger, run interpret and then compile over the live board it leaves, and
    close the ledger."""
    full = settings(**run)
    ledger = open_ledger(loaded, state, full, observer)
    board = asyncio.run(interpret_on(ledger, loaded, state, bindings, full))
    plan, verdict = compile_turn(ledger, loaded, board)
    return closed(ledger), plan, verdict


# ---------------------------------------------------------------------------------------------
# Awaitables a sync component may return


class NotACoroutine:
    """An awaitable that is no coroutine: refused, never closed, never truth-tested."""

    def __init__(self) -> None:
        self.closed = False

    def __await__(self) -> Any:
        return iter(())

    def close(self) -> None:
        self.closed = True

    def __bool__(self) -> bool:
        raise AssertionError("an awaitable takes no truth test")


class InheritsAwait(NotACoroutine):
    """An awaitable by a base class's ``__await__``, found on its MRO."""


class Claims:
    """No awaitable, whose ``__class__`` claims to be a coroutine's."""

    @property
    def __class__(self) -> type:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        return types.CoroutineType


class _Raising(type):
    """A metaclass whose ``__mro__``, ``__dict__`` and hash raise, none of which the type
    test reads."""

    @property
    def __mro__(self) -> tuple[type, ...]:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        raise RuntimeError("metaclass __mro__")

    @property
    def __dict__(self) -> Any:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        raise RuntimeError("metaclass __dict__")

    def __hash__(self) -> int:
        raise RuntimeError("metaclass __hash__")


class MetaRaising(metaclass=_Raising):
    """No awaitable, of a class whose metaclass raises on every read the type test avoids."""


class _ClashingKey:
    """A class-dict key hashing as ``__await__`` does, whose comparison raises."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def __hash__(self) -> int:
        return hash("__await__")

    def __eq__(self, other: object) -> bool:
        raise self.exc


def clashing(exc: BaseException | None = None) -> object:
    """An object whose class dict makes the awaitable test raise ``exc``, by default a
    ``ValueError``."""
    key = _ClashingKey(ValueError("the type test raised") if exc is None else exc)
    with warnings.catch_warnings():  # 3.14 warns of a non-string class-dict key at creation
        warnings.simplefilter("ignore", RuntimeWarning)
        return type("Clashing", (), {cast(Any, key): 1})()


def stop() -> BaseException:
    """A view's stop, as a component that caught one may keep it and raise it again."""
    return _UndeclaredStop(UndeclaredRead("t:kept", True, Access.read))


@types.coroutine
def _generator_coroutine(ran: list[str]) -> Any:
    try:
        yield
    finally:
        ran.append("finally")


def generator_coroutine(ran: list[str] | None = None, *, started: bool = False) -> Any:
    """A generator flagged as a coroutine (``types.coroutine``), which appends ``"finally"`` to
    ``ran`` when it ends; ``started`` runs it to its ``yield``."""
    generator = _generator_coroutine([] if ran is None else ran)
    if started:
        cast("Generator[None, None, None]", generator).send(None)
    return generator


async def _parked(exc: BaseException) -> object:
    try:
        await asyncio.sleep(0)
    finally:
        raise exc


def started_raising_on_close(message: str | BaseException) -> Any:
    """A coroutine started and parked at an await, whose ``finally`` raises when it is closed:
    ``ValueError`` with ``message``, or ``message`` itself when it is an exception."""
    coroutine = _parked(ValueError(message) if isinstance(message, str) else message)
    coroutine.send(None)
    return coroutine
