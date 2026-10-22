"""The plan types: ``Plan``, ``Step``, ``Origin`` and ``Binding``.

Each is a frozen dataclass whose constructor checks its fields' types and nothing else, so a
plan that breaks a plan rule is built, and compile names the rule.
A ``str`` subclass is stored as its string data. Only the engine writes a plan's ``cause`` and
``slots`` and an origin's ``set``: the public constructors take none of them, and the engine
sets them through the internal ``engine_plan`` and ``candidate_literal``.

A ``Plan`` and a ``Step`` are not hashable, and neither deep-copies nor pickles: they hold
their mappings as ``MappingProxyType``, as the design types do.
"""

import enum
from collections.abc import ItemsView, Iterator, Mapping, ValuesView
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from jurat._design import Requirement
from jurat._markers import Cause, PlanKind
from jurat._values import described, string_data


class _Absent(enum.Enum):
    """An origin keyword not given, apart from a literal ``None``."""

    absent = enum.auto()


_ABSENT = _Absent.absent


def _text(value: object, what: str) -> str | None:
    if value is None:
        return None
    if not issubclass(type(value), str):
        raise TypeError(f"{what} {described(value)} is not a string or None")
    return string_data(cast(str, value))


def _members(value: object, what: str) -> tuple[object, ...]:
    if type(value) is not list and type(value) is not tuple:
        raise TypeError(f"{what} {described(value)} are not a list or tuple")
    return tuple(cast("list[object] | tuple[object, ...]", value))


@dataclass(frozen=True, slots=True, init=False)
class Binding:
    """An origin bound to an earlier step's result: its position, from 1, and a field path
    into the result, or ``None`` for the whole result. Plan rule 2 checks
    both."""

    step: int
    path: str | None

    def __init__(self, step: int, path: str | None = None) -> None:
        if type(step) is not int:
            raise TypeError(f"a binding's step {described(step)} is not an int")
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "path", _text(path, "a binding's path"))


@dataclass(frozen=True, slots=True, init=False)
class Origin:
    """A literal, ``Origin(literal=v)``, or a board ref, ``Origin(ref=r)``.

    ``literal`` is ``None`` for a board ref and ``ref`` ``None`` for a literal, so an origin is
    a board ref exactly when ``ref`` is not ``None``. ``set`` is a candidate-value literal's set
    ref, written only by compile, and otherwise ``None``.
    """

    literal: Any
    ref: str | None
    set: str | None

    def __init__(self, *, literal: Any = _ABSENT, ref: str | _Absent = _ABSENT) -> None:
        if (literal is _ABSENT) == (ref is _ABSENT):
            raise TypeError("an Origin takes exactly one of literal= and ref=")
        if ref is _ABSENT:
            object.__setattr__(self, "literal", literal)
            object.__setattr__(self, "ref", None)
        else:
            if not issubclass(type(ref), str):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f"an origin's ref {described(ref)} is not a string")
            object.__setattr__(self, "literal", None)
            object.__setattr__(self, "ref", string_data(ref))
        object.__setattr__(self, "set", None)


def candidate_literal(value: object, set_ref: str) -> Origin:
    """A grounded step's candidate-value literal, carrying its set's ref; compile's alone."""
    origin = Origin(literal=value)
    object.__setattr__(origin, "set", string_data(set_ref))
    return origin


class _Twinned(Mapping[str, "Origin | Binding"]):
    """Argument names two of which share one string data, kept as given in order, so plan
    rule 7 finds them outside the domain. Read by position and string data alone: no name's
    hash or comparison runs, as a ``dict`` of them would run it."""

    __slots__ = ("_pairs",)

    def __init__(self, pairs: list[tuple[str, "Origin | Binding"]]) -> None:
        self._pairs = tuple(pairs)

    def __getitem__(self, key: str) -> "Origin | Binding":
        if issubclass(type(key), str):  # pyright: ignore[reportUnnecessaryIsInstance]
            data = string_data(key)
            for name, origin in self._pairs:
                if string_data(name) == data:
                    return origin
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __contains__(self, key: object) -> bool:
        try:
            self[cast(str, key)]
        except KeyError:
            return False
        return True

    def items(self) -> ItemsView[str, "Origin | Binding"]:
        return _TwinnedItems(self)

    def values(self) -> ValuesView["Origin | Binding"]:
        return _TwinnedValues(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    __hash__ = None  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    def __reduce_ex__(self, protocol: object) -> Any:
        raise TypeError("a step's arguments neither deep-copy nor pickle")


class _TwinnedItems(ItemsView[str, "Origin | Binding"]):
    _mapping: _Twinned

    def __iter__(self) -> Iterator[tuple[str, "Origin | Binding"]]:
        return iter(self._mapping._pairs)  # pyright: ignore[reportPrivateUsage]

    def __contains__(self, item: object) -> bool:
        """A pair at some position: its name by string data, its origin by identity, so no
        name's or origin's comparison runs."""
        if type(item) is not tuple or len(cast("tuple[object, ...]", item)) != 2:
            return False
        name, origin = cast("tuple[object, object]", item)
        if not issubclass(type(name), str):
            return False
        data = string_data(cast(str, name))
        return any(string_data(n) == data and o is origin for n, o in self)


class _TwinnedValues(ValuesView["Origin | Binding"]):
    _mapping: _Twinned

    def __iter__(self) -> Iterator["Origin | Binding"]:
        return (origin for _, origin in self._mapping._pairs)  # pyright: ignore[reportPrivateUsage]

    def __contains__(self, value: object) -> bool:
        return any(origin is value for origin in self)


_NO_ARGUMENTS: Mapping[str, Origin | Binding] = MappingProxyType({})


@dataclass(frozen=True, slots=True, init=False)
class Step:
    """One tool call: the declared tool name, the arguments by name, each an ``Origin`` or a
    ``Binding``, and the step's requirements in declaration order.
    ``arguments`` is a read-only copy, so a ``Step`` is not hashable. Arguments two of whose
    names share one string data are kept as given, and a plan holding them never passes: plan
    rule 7 refuses it when no earlier rule does."""

    tool: str
    arguments: Mapping[str, Origin | Binding]
    requirements: tuple[Requirement, ...]

    def __init__(
        self,
        tool: str,
        arguments: Mapping[str, Origin | Binding] = _NO_ARGUMENTS,
        requirements: list[Requirement] | tuple[Requirement, ...] = (),
    ) -> None:
        if not issubclass(type(tool), str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"a step's tool {described(tool)} is not a string")
        if not issubclass(type(arguments), Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"a step's arguments {described(arguments)} are not a mapping")
        pairs: list[tuple[str, Origin | Binding]] = []
        given = list(cast("Mapping[object, object]", arguments).items())  # read once
        for pair in cast("list[object]", given):
            # Exactly a tuple of two, so unpacking runs no caller code.
            if type(pair) is not tuple or len(cast("tuple[object, ...]", pair)) != 2:
                shown = described(cast(object, pair))
                raise TypeError(f"a step's argument {shown} is not a name and an origin")
            name, origin = cast("tuple[object, object]", pair)
            if not issubclass(type(name), str):
                raise TypeError(f"a step's argument name {described(name)} is not a string")
            if type(origin) is not Origin and type(origin) is not Binding:
                raise TypeError(
                    f"the step's argument {described(name)} is not an Origin or a Binding"
                )
            pairs.append((cast(str, name), origin))
        named = {string_data(name): origin for name, origin in pairs}
        arguments_: Mapping[str, Origin | Binding] = MappingProxyType(named)
        if len(named) != len(pairs):
            # Two names the mapping keeps apart by an override share one string data: kept as
            # given, so plan rule 7 finds the arguments outside the domain.
            arguments_ = _Twinned(pairs)
        members = _members(requirements, "a step's requirements")
        if not all(type(r) is Requirement for r in members):
            raise TypeError("a step's requirement is not a Requirement")
        object.__setattr__(self, "tool", string_data(tool))
        object.__setattr__(self, "arguments", arguments_)
        object.__setattr__(self, "requirements", cast("tuple[Requirement, ...]", members))


_NO_SLOTS: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True, init=False)
class Plan:
    """The turn's plan: its kind, its steps in order, the policy's
    ``reason`` and ``reply``, the engine's ``cause``, and ``slots``, ``{}`` unless declared text
    wrote them. The public constructor takes neither ``cause`` nor ``slots``."""

    kind: PlanKind
    steps: tuple[Step, ...]
    reason: str | None
    reply: str | None
    cause: Cause | None
    slots: Mapping[str, Any]

    def __init__(
        self,
        kind: PlanKind | str,
        steps: list[Step] | tuple[Step, ...] = (),
        *,
        reason: str | None = None,
        reply: str | None = None,
    ) -> None:
        # Every member is a string, so anything else is refused before the enum's lookup,
        # whose own message would write the value.
        try:
            if not issubclass(type(kind), str):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError
            member = PlanKind(string_data(cast(str, kind)))
        except ValueError:
            raise ValueError(f"plan kind {described(kind)} is not in the closed set") from None
        members = _members(steps, "a plan's steps")
        if not all(type(s) is Step for s in members):
            raise TypeError("a plan's step is not a Step")
        object.__setattr__(self, "kind", member)
        object.__setattr__(self, "steps", cast("tuple[Step, ...]", members))
        object.__setattr__(self, "reason", _text(reason, "a plan's reason"))
        object.__setattr__(self, "reply", _text(reply, "a plan's reply"))
        object.__setattr__(self, "cause", None)
        object.__setattr__(self, "slots", _NO_SLOTS)


def engine_plan(kind: PlanKind, cause: Cause) -> Plan:
    """A plan the engine builds, with no steps, ``reason`` or ``reply`` and the engine's cause."""
    plan = Plan(kind)
    object.__setattr__(plan, "cause", cause)
    return plan
