# pyright: reportPrivateUsage=false
"""A component's call through its read view, as interpret calls a rule and compile the policy.

``call`` runs the component and catches what the protected region catches: a view's stop,
which the view has recorded, and an ``Exception``. A ``_ViewError`` is the engine's defect and
propagates, as does any other ``BaseException``. What the outcome then
is, the first that applies of the read view's order, is the caller's: the deciding stop, then a
raise, then the judged return. Judging the return, a rule's against its ``outputs`` and the
policy's against the plan rules, is not shared; its first test is, inside ``call``: an
awaitable returned is an invalid return, and a returned coroutine is closed unawaited whatever
outcome wins, an ``Exception`` its close raises kept beside the outcome it leaves as it stands. The
codec, which has no view, shares that test. The view is closed when the call returns, so a component
that kept it, or another handed it, reads nothing more and adds no stop.

Also here, each written once for the stages that share it: whether an undeclared read is a
name the design holds, an exception's text and a wall-clock time as ``unhashed`` holds them,
the helper log as ``derivation`` holds it, and a read on the live board.
"""

import inspect
import traceback
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from jurat._design import LoadedDesign
from jurat._read_view import ReadView, Stop, UndeclaredRead, _Stop, _ViewError, walk
from jurat._refs import parse_ref
from jurat._types import TypeFailure, read_type
from jurat._values import UNREPRESENTABLE, exception_walk, string_data, type_name

_STATE = "t:state"


@dataclass(frozen=True, slots=True)
class Called:
    """What a component's call gave: the deciding stop, its raise, its return, and its helper
    calls as ``derivation`` holds them; whether the return is an awaitable, and what closing a
    returned coroutine raised."""

    stop: Stop | None
    error: Exception | None
    returned: object
    derivation: list[dict[str, Any]]
    awaitable: bool
    closing: Exception | None

    @property
    def text(self) -> Exception | None:
        """The exception whose text ``exceptions`` keeps: the raise, else the close's."""
        return self.error if self.error is not None else self.closing


def call(fn: Callable[[ReadView], object], view: ReadView) -> Called:
    """Call ``fn`` on ``view`` inside the protected region."""
    returned: object = None
    error: Exception | None = None
    try:
        returned = fn(view)
    except _Stop:
        pass  # recorded on the view
    except _ViewError:
        raise  # an engine defect, never the component's failure
    except Exception as exc:  # the protected region
        error = exc
    finally:
        view.close()  # a view kept past its call stops nothing
    # After the view's close, so a coroutine's `finally` that reads it raises and adds no stop.
    awaitable, closing = False, None
    try:
        awaitable, closing = refuse_awaitable(returned)
    except _ViewError:
        raise
    except Exception as exc:  # the test's raise is the component's
        error = exc
    return Called(view.deciding_stop(), error, returned, derivation(view), awaitable, closing)


_MRO = type.__dict__["__mro__"]  # read through `type`, never the metaclass
_DICT = type.__dict__["__dict__"]


def is_awaitable(returned: object) -> bool:
    """Whether a return is an awaitable by its type alone: exactly a coroutine, exactly a
    generator flagged as one, or a class on its MRO holding ``__await__`` in its own
    ``__dict__``. Nothing the object claims, its ``__class__`` included, is read; an
    ``Exception`` a class dict's key raises propagates."""
    kind = type(returned)
    if kind is types.CoroutineType:
        return True
    if kind is types.GeneratorType:
        code = cast("types.GeneratorType[object, object, object]", returned).gi_code
        return bool(code.co_flags & inspect.CO_ITERABLE_COROUTINE)
    return any("__await__" in _DICT.__get__(c) for c in _MRO.__get__(kind))


def refuse_awaitable(returned: object) -> tuple[bool, Exception | None]:
    """Whether a sync component's return is an awaitable, its invalid return; a coroutine, or
    a generator flagged as one, is closed without being awaited, and an ``Exception`` the close
    raises is returned. An ``Exception`` the test raises propagates, the caller's to judge as
    the component's raise. A view's stop the test or the close raises again is dropped, read
    from the record and never propagated; the test then finds no awaitable. Any other
    ``BaseException`` propagates."""
    try:
        if not is_awaitable(returned):
            return False, None
    except _Stop:  # a kept stop raised again: the view recorded it, if it is the view's
        return False, None
    if type(returned) is types.CoroutineType or type(returned) is types.GeneratorType:
        try:
            cast("types.CoroutineType[object, object, object]", returned).close()
        except _Stop:  # a kept stop raised again: nothing to keep
            pass
        except _ViewError:
            raise  # an engine defect, never the component's failure
        except Exception as exc:  # kept beside the outcome
            return True, exc
    return True, None


def derivation(view: ReadView) -> list[dict[str, Any]]:
    """The view's helper log as the ``derivation`` member writes it."""
    return [
        {"helper": i.helper, "refs": list(i.refs), "rested_on": list(i.rested_on)}
        for i in view.helper_log()
    ]


def names_read(loaded: LoadedDesign, stop: UndeclaredRead) -> bool:
    """Whether an undeclared read is a name the design holds: a provided ref, ``t:state``
    included, with no bracket and a field path naming a field at each depth, or a declared
    parameter path."""
    if not stop.string:
        return False
    try:
        p = parse_ref(stop.read)
    except ValueError:  # not a ref under the grammar
        return stop.read in loaded._leaf_types
    base = f"{p.level}:{p.name}"
    if p.candidate is not None or (base != _STATE and base not in loaded._providers):
        return False
    if not p.fields:
        return True
    t = loaded._types.get(base)  # a q: ref has no fields
    return t is not None and not isinstance(read_type(t, p.fields, {}), TypeFailure)


def exception_record(exc: BaseException) -> dict[str, str]:
    """An exception's text as ``unhashed`` holds it: its type's name, and ``str`` of it and
    its formatted traceback only when the walk over its ``args`` and ``__notes__``, and those
    of each exception chained to it, stays within the bounds and meets only exact built-in
    types; otherwise both ``UNREPRESENTABLE``. The walk and each text are taken under the
    observer's guard."""
    try:
        walked = exception_walk(exc)
    except Exception:  # a planted key's comparison, say: the walk reads no text then
        walked = False

    def guarded(make: Callable[[], str]) -> str:
        if not walked:
            return UNREPRESENTABLE
        try:
            return string_data(make())
        except (Exception, _Stop):
            return UNREPRESENTABLE

    return {
        "type": type_name(exc),
        "message": guarded(lambda: str(exc)),
        "traceback": guarded(lambda: "".join(traceback.format_exception(exc))),
    }


def now() -> str:
    """The wall-clock time as ``unhashed`` holds it, RFC 3339 in UTC."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def at(board: Mapping[str, object], ref: str) -> object:
    """A read of ``ref`` on the live board, whole or by field path, as a view returns it: a
    marker at its base, a null on the path makes it null, and an entry that does not hold its
    path is a ``_ViewError``, by the view's own walk."""
    return walk(board[base(ref)], parse_ref(ref).fields, ref)


def base(ref: str) -> str:
    """The declared ref a read's entry is at: its field path dropped."""
    p = parse_ref(ref)
    return f"{p.level}:{p.name}"
