"""The read view: what a function ``when``, a rule or the policy reads, and nothing else.

One type for the three. The component gets ``read``, ``require`` and ``param``.
Interpret and compile build the view and read back its ``role``, its ``stops`` and
``deciding_stop()``, and its ``helper_log()``; the helper library writes the log through
``open_call`` and ``close_call``. The outcome a stop record gives,
weighed against a raise, an invalid return and the returned value, is interpret's and
compile's, not the view's.

A ``when``'s view is given the state's type, its definitions resolved at every depth, and
reads ``t:state`` by any field path that type gives: one naming a record field at each depth.

Not built: reading a grounded declaration's declared ref, since candidate
sets are outside v0.0.1 and load's check 12 refuses them; a declared read with a grounded
bracket is an engine defect here.
"""

import enum
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from jurat._markers import Undetermined
from jurat._refs import parse_ref
from jurat._types import Type, TypeFailure, read_type
from jurat._values import described, guarded_repr, string_data

_STATE = "t:state"


class ComponentKind(enum.StrEnum):
    """The kinds of component a view serves."""

    when = "when"
    rule = "rule"
    policy = "policy"


@dataclass(frozen=True, slots=True, init=False)
class Role:
    """The component a view serves: its kind and its declared name, ``policy``, or for a
    ``when`` its source's id."""

    kind: ComponentKind
    name: str

    def __init__(self, kind: ComponentKind, name: str) -> None:
        # Every member is a string, so anything else is refused before the enum's lookup,
        # whose own message would write the value.
        try:
            if not issubclass(type(kind), str):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError
            member = ComponentKind(string_data(kind))
        except ValueError:
            raise ValueError(
                f"component kind {described(kind)} is not when, rule or policy"
            ) from None
        if not issubclass(type(name), str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"a role's name {described(name)} is not a string")
        object.__setattr__(self, "kind", member)
        object.__setattr__(self, "name", string_data(name))


class Access(enum.StrEnum):
    """How an undeclared read was made: by ``read`` or ``require``, or by ``param``."""

    read = "read"
    param = "param"


@dataclass(frozen=True, slots=True)
class UndeclaredRead:
    """A read outside the read set. ``read`` is the read as given: a string's
    string data, with ``string`` true, or anything else's ``repr`` taken at the stop under
    the observer's guard, so a later change to the object cannot change it."""

    read: str
    string: bool
    access: Access

    @classmethod
    def of(cls, read: object, access: Access) -> "UndeclaredRead":
        """The record of ``read``, taken now."""
        if issubclass(type(read), str):  # by its type: ``isinstance`` reads ``__class__``
            return cls(string_data(cast(str, read)), True, access)
        return cls(guarded_repr(read), False, access)


@dataclass(frozen=True, slots=True)
class RequireStopped:
    """A ``require`` stop: the read as given, field path included, and the marker it read."""

    ref: str
    marker: Undetermined


Stop = UndeclaredRead | RequireStopped


@dataclass(frozen=True, slots=True)
class DerivationItem:
    """One helper or combinator call as the log holds it."""

    helper: str
    refs: tuple[str, ...]
    rested_on: tuple[str, ...]


class _Stop(BaseException):
    """A view's stop; not an ``Exception``, so ``except Exception`` cannot catch it.
    Interpret and compile catch this class around each call."""

    def __init__(self, stop: Stop) -> None:
        super().__init__(stop)
        self.stop = stop


class _UndeclaredStop(_Stop):
    """The stop of an undeclared read."""


class _RequireStop(_Stop):
    """The stop of ``require`` on a marker."""


class _ViewError(RuntimeError):
    """An engine defect: a view built over a board or leaves that lack a declared read or
    parameter, or an entry that does not hold its type. Never a stop."""


class ViewClosed(Exception):
    """A read on a view whose component's call has returned. An ordinary ``Exception``,
    recorded nowhere: a component that kept another's view and reads it fails as a raise,
    and an exception whose text reads a kept view gives no stop."""


def walk(value: object, fields: Sequence[str], ref: str) -> object:
    """A read through ``fields`` from the entry ``value``: a marker at the base is returned, a
    null on the path makes the read null; an entry that does not hold its
    path is a ``_ViewError``. The one path walk, for the view and for compile's board read."""
    if isinstance(value, Undetermined):
        return value
    for name in fields:
        if value is None:
            return None
        if type(value) is not dict or name not in value:
            raise _ViewError(f"the entry under {ref!r} does not hold its path")
        value = cast("dict[str, object]", value)[name]
    return value


@dataclass(slots=True)
class _Item:
    helper: str
    refs: tuple[str, ...]
    rested_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Path:
    base: str
    fields: tuple[str, ...]


class ReadView:
    """A component's read set over the board and the parameter leaves.

    ``reads`` are the component's declared reads, for a ``when`` its source's needs as
    declared, to which ``t:state`` is added; ``params`` its declared parameters, none for a
    ``when``. ``board`` maps each ref to its value or marker, and must hold the base of every
    read; ``leaves`` maps each parameter path to its value, and must hold every declared one.
    A read matches the read set by its exact string data, so a field path and its base are
    different reads. ``state`` is the state's type, resolved at every depth, which a
    ``when``'s view takes and no other: in a ``when``, a ``t:state`` field path the read set
    does not spell is read when that type gives it.
    """

    __slots__ = (
        "_board",
        "_closed",
        "_depth",
        "_leaves",
        "_log",
        "_params",
        "_reads",
        "_role",
        "_state",
        "_stops",
    )

    def __init__(
        self,
        role: Role,
        reads: Iterable[str],
        params: Iterable[str],
        board: Mapping[str, object],
        leaves: Mapping[str, object],
        *,
        state: Type | None = None,
    ) -> None:
        self._role = role
        if (role.kind is ComponentKind.when) != (state is not None):
            raise _ViewError("a when's view, and only a when's, takes the state's type")
        self._state = state
        declared = {string_data(r) for r in reads}
        if role.kind is ComponentKind.when:
            declared.add(_STATE)
        self._reads: dict[str, _Path] = {}
        for ref in sorted(declared):
            try:
                p = parse_ref(ref)
            except ValueError:
                raise _ViewError(f"the declared read {ref!r} is not a ref") from None
            if p.candidate is not None:
                raise _ViewError(f"the declared read {ref!r} is grounded, outside v0.0.1")
            path = _Path(f"{p.level}:{p.name}", p.fields)
            if path.base not in board:
                raise _ViewError(f"the board holds no entry for the declared read {ref!r}")
            self._reads[ref] = path
        self._params = frozenset(string_data(p) for p in params)
        if role.kind is ComponentKind.when and self._params:
            raise _ViewError("a when declares no parameters")
        for path in sorted(self._params):
            if path not in leaves:
                raise _ViewError(f"the leaves hold no value for the declared parameter {path!r}")
        self._board = board
        self._leaves = leaves
        self._stops: list[Stop] = []
        self._log: list[_Item] = []
        self._depth = 0
        self._closed = False

    # -- the component's reads -------------------------------------------------------------

    def read(self, ref: str) -> Any:
        """The value or marker of a ref of the read set, whole or by field path. A marker at
        the path's base is returned; a null on the path makes the read null."""
        self._open()
        key = string_data(ref) if issubclass(type(ref), str) else None  # pyright: ignore[reportUnnecessaryIsInstance]
        path = None if key is None else self._reads.get(key)
        if path is None and key is not None and self._state is not None:  # a when
            path = self._state_path(key)
        if path is None:
            self._stop(_UndeclaredStop(UndeclaredRead.of(ref, Access.read)))
        return walk(self._board[path.base], path.fields, string_data(ref))

    def _state_path(self, ref: str) -> _Path | None:
        """A ``when``'s read of ``t:state`` by a field path naming a record field at each depth
        of the state's type, judged by the type and never by the value."""
        try:
            p = parse_ref(ref)
        except ValueError:  # not a ref under the grammar
            return None
        if p.candidate is not None or f"{p.level}:{p.name}" != _STATE or not p.fields:
            return None
        assert self._state is not None
        if isinstance(read_type(self._state, p.fields, {}), TypeFailure):
            return None
        return _Path(_STATE, p.fields)

    def require(self, ref: str) -> Any:
        """The value of ``ref``; on a marker, stop the component. ``None``
        is a value and does not stop."""
        value = self.read(ref)
        if isinstance(value, Undetermined):
            self._stop(_RequireStop(RequireStopped(string_data(ref), value)))
        return value

    def param(self, path: str) -> Any:
        """A declared parameter leaf's value."""
        self._open()
        if not issubclass(type(path), str) or string_data(path) not in self._params:  # pyright: ignore[reportUnnecessaryIsInstance]
            self._stop(_UndeclaredStop(UndeclaredRead.of(path, Access.param)))
        return self._leaves[string_data(path)]

    def close(self) -> None:
        """End the component's call: every later ``read``, ``require`` or ``param`` raises
        ``ViewClosed``. The engine's reads back stay open."""
        self._closed = True

    def _open(self) -> None:
        if self._closed:
            raise ViewClosed(f"the read view of {self._role.kind} {self._role.name!r} is closed")

    def _stop(self, stop: _Stop) -> NoReturn:
        self._stops.append(stop.stop)
        raise stop

    # -- what the engine reads back --------------------------------------------------------

    @property
    def role(self) -> Role:
        """The component this view serves."""
        return self._role

    @property
    def stops(self) -> tuple[Stop, ...]:
        """Every stop issued, caught or not, in order."""
        return tuple(self._stops)

    def deciding_stop(self) -> Stop | None:
        """The first undeclared read, else the first ``require`` stop, else ``None``. The
        full order of outcomes, weighing this against a raise, an invalid return and the
        returned value, is interpret's and compile's."""
        for stop in self._stops:
            if isinstance(stop, UndeclaredRead):
                return stop
        return self._stops[0] if self._stops else None

    def helper_log(self) -> tuple[DerivationItem, ...]:
        """The helper and combinator calls, in call order, as a frozen copy."""
        return tuple(DerivationItem(i.helper, i.refs, i.rested_on) for i in self._log)


def open_call(view: ReadView, helper: str, refs: Sequence[str]) -> int | None:
    """List a helper call, once its refs are read, with ``rested_on`` empty and return its
    index, or ``None`` for a call inside another helper's, which is not listed."""
    for ref in refs:
        if string_data(ref) not in view._reads:  # pyright: ignore[reportPrivateUsage]
            raise _ViewError(f"a helper listed {ref!r}, which it has not read")
    nested = view._depth > 0  # pyright: ignore[reportPrivateUsage]
    view._depth += 1  # pyright: ignore[reportPrivateUsage]
    if nested:
        return None
    log = view._log  # pyright: ignore[reportPrivateUsage]
    log.append(_Item(helper, tuple(refs)))
    return len(log) - 1


def close_call(view: ReadView, index: int | None, rested_on: Sequence[str] | None) -> None:
    """End a helper call; fill ``rested_on`` when it returned, leave it empty when not."""
    view._depth -= 1  # pyright: ignore[reportPrivateUsage]
    if index is not None and rested_on is not None:
        view._log[index].rested_on = tuple(rested_on)  # pyright: ignore[reportPrivateUsage]
