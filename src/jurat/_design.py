"""The design types, the loaded design and ``InvalidDesign``.

A design type's fields are named as its object's members in the design document; a function
component takes its callable as ``function``. A design type validates
nothing when built: load's check 10 tests every member's form, so every failure is reported
in one pass. A constructor copies the containers that are its
own fields one level deep: a mapping field becomes a read-only mapping, keyed by each key's
string data when its keys are strings that stay distinct, a sequence field a tuple, and a
string its string data. So adding to or removing from a dict or list the caller passed as a
field changes no design. Values inside, a payload or a leaf's value among them, are kept as
given, so a caller who changes those before load changes the design load sees.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

from jurat._answer import AnswerKind, ValueType
from jurat._canonical import decode
from jurat._read_view import ReadView
from jurat._types import ListOf, MappingOf, Scalar, Type, b64url_decode, b64url_encode
from jurat._values import string_data

# ---------------------------------------------------------------------------------------------
# Snapshots, which never raise


def _empty() -> Mapping[str, Any]:
    return MappingProxyType({})


def _mapping(value: object) -> Any:
    if not isinstance(value, Mapping):
        return value
    items = list(cast("Mapping[object, Any]", value).items())
    if all(isinstance(k, str) for k, _ in items):
        data = {string_data(cast(str, k)): v for k, v in items}
        if len(data) == len(items):
            return MappingProxyType(data)
    return MappingProxyType(dict(items))


def _sequence(value: object) -> Any:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        return value
    return tuple(cast("Sequence[Any]", value))


def _string(value: object) -> Any:
    # By its type alone, so no ``__class__`` of the caller's is read.
    return string_data(cast(str, value)) if issubclass(type(value), str) else value


def _provision(value: object) -> Any:
    """A ``provides`` value: a type's string data, or a mapping whose ``type`` is."""
    if not isinstance(value, Mapping):
        return _string(value)
    proxy = cast("Mapping[object, Any]", _mapping(cast(object, value)))
    data = dict(proxy)
    if not all(type(k) is str for k in data) or "type" not in data:
        return proxy
    data["type"] = _string(data["type"])
    return MappingProxyType(data)


def _strings(value: object) -> Any:
    members = _sequence(value)
    if isinstance(members, tuple):
        return tuple(_string(v) for v in cast("tuple[object, ...]", members))
    return members


def _set(obj: object, **fields: object) -> None:
    for name, value in fields.items():
        object.__setattr__(obj, name, value)


# ---------------------------------------------------------------------------------------------
# Text, questions and requirements


def _slot(slot: object) -> Any:
    members = _mapping(slot)
    if isinstance(members, MappingProxyType) and "origin" in members:
        snapshot = dict(cast("Mapping[str, Any]", members))
        origin = _mapping(snapshot["origin"])
        if isinstance(origin, MappingProxyType):
            origin = MappingProxyType(
                {k: _string(v) for k, v in cast("Mapping[str, Any]", origin).items()}
            )
        snapshot["origin"] = origin
        snapshot["name"] = _string(snapshot.get("name"))
        return MappingProxyType(snapshot)
    return cast(object, members)


@dataclass(frozen=True, slots=True)
class Text:
    """A template and its slot list: each slot ``{"name": n, "origin": o}``, ``o`` one of
    ``{"ref": r}``, ``{"param": path}`` or ``{"candidate": r}``."""

    template: str
    slots: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        slots = _sequence(self.slots)
        if isinstance(slots, tuple):
            slots = tuple(_slot(s) for s in cast("tuple[object, ...]", slots))
        _set(self, template=_string(self.template), slots=slots)


@dataclass(frozen=True, slots=True)
class Question:
    """A ``q:`` ref's question: its kind, wording and labels. ``labels``
    is ``None``, a sequence of labels, or a mapping of ``fixed`` and ``candidates`` for a
    candidate-valued label set. A label is its name, a string, or a mapping of exactly ``name``
    and ``description``, both strings, held as a read-only copy. An extraction question also
    declares ``value_type``, a ``ValueType`` member or its string, held as the member;
    ``is_list``, a boolean; ``text``, a board-ref origin ``{"ref": r}`` naming the string its
    spans index; and ``reference_date``, such an origin or ``None``. A probability kind's
    question leaves those four at ``None``, ``False``, ``None`` and ``None``."""

    kind: AnswerKind | str
    """An ``AnswerKind`` member or its string, held as the member."""
    wording: Text
    labels: Sequence[str | Mapping[str, str]] | Mapping[str, Any] | None = None
    value_type: ValueType | str | None = None
    is_list: bool = False
    text: Mapping[str, str] | None = None
    reference_date: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        kind: object = self.kind
        if isinstance(kind, str) and string_data(kind) in AnswerKind.__members__:  # pyright: ignore[reportUnnecessaryIsInstance]
            kind = AnswerKind(string_data(kind))
        value_type: object = self.value_type
        if isinstance(value_type, str) and string_data(value_type) in ValueType.__members__:
            value_type = ValueType(string_data(value_type))
        labels: object = self.labels
        if isinstance(labels, Mapping):
            snapshot = dict(cast("Mapping[str, Any]", _mapping(labels)))
            if "fixed" in snapshot:
                snapshot["fixed"] = _labels(snapshot["fixed"])
            if "candidates" in snapshot:
                snapshot["candidates"] = _string(snapshot["candidates"])
            labels = MappingProxyType(snapshot)
        else:
            labels = _labels(labels)
        _set(
            self,
            kind=kind,
            labels=labels,
            value_type=value_type,
            text=_origin(self.text),
            reference_date=_origin(self.reference_date),
        )


def _labels(value: object) -> Any:
    """A label list as a tuple: each string its string data, each mapping, a described label's
    ``name`` and ``description``, a read-only copy of it with each string its string data."""
    members = _sequence(value)
    if isinstance(members, tuple):
        return tuple(_label(v) for v in cast("tuple[object, ...]", members))
    return members


def _label(value: object) -> Any:
    members = _mapping(value)
    if isinstance(members, MappingProxyType):
        return MappingProxyType(
            {k: _string(v) for k, v in cast("Mapping[Any, Any]", members).items()}
        )
    return _string(cast(object, members))


def label_parts(labels: Sequence[Any]) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    """A well-formed label list's names and descriptions, aligned, ``None`` where a label is
    a bare name; check 10 has passed it."""
    names: list[str] = []
    descriptions: list[str | None] = []
    for label in labels:
        if isinstance(label, str):
            names.append(label)
            descriptions.append(None)
        else:
            described = cast("Mapping[str, str]", label)
            names.append(described["name"])
            descriptions.append(described["description"])
    return tuple(names), tuple(descriptions)


def _origin(value: object) -> Any:
    """A board-ref origin as a read-only mapping, each string its string data."""
    members = _mapping(value)
    if isinstance(members, MappingProxyType):
        return MappingProxyType(
            {k: _string(v) for k, v in cast("Mapping[Any, Any]", members).items()}
        )
    return cast(object, members)


@dataclass(frozen=True, slots=True)
class When:
    """A source's function ``when``: the callable and its ``version``."""

    function: Callable[[ReadView], object]
    version: str | None = None

    def __post_init__(self) -> None:
        _set(self, version=_string(self.version))


@dataclass(frozen=True, slots=True)
class SourceRequirement:
    """A source's data-form ``when``: its ``d:`` ref holds one of ``values``."""

    ref: str
    values: Sequence[Any]

    def __post_init__(self) -> None:
        _set(self, ref=_string(self.ref), values=_sequence(self.values))


@dataclass(frozen=True, slots=True)
class Requirement:
    """A plan-level or step requirement: ``ref`` is determined with ``value``."""

    ref: str
    value: Any

    def __post_init__(self) -> None:
        _set(self, ref=_string(self.ref))


@dataclass(frozen=True, slots=True)
class Parameter:
    """A parameter leaf: its type and its value."""

    type: str
    value: Any


@dataclass(frozen=True, slots=True)
class Group:
    """A correlated group: the refs it names."""

    refs: Sequence[str]

    def __post_init__(self) -> None:
        _set(self, refs=_strings(self.refs))


# ---------------------------------------------------------------------------------------------
# Components


@dataclass(frozen=True, slots=True)
class Source:
    """A source. ``provides`` maps each ``q:`` ref to its ``Question``
    and each ``t:`` ref to its type, or to a mapping of ``type`` and ``candidates``."""

    provides: Mapping[str, Any]
    needs: Sequence[str] = ()
    when: When | None = None
    requirement: SourceRequirement | None = None
    oracle: str | None = None
    timeout: float | None = None
    payload: Any = None
    cost: float = 0
    feedback: bool = False
    grounded: str | None = None

    def __post_init__(self) -> None:
        provides = _mapping(self.provides)
        if isinstance(provides, MappingProxyType):
            provides = MappingProxyType(
                {k: _provision(v) for k, v in cast("Mapping[Any, Any]", provides).items()}
            )
        _set(
            self,
            provides=provides,
            needs=_strings(self.needs),
            oracle=_string(self.oracle),
            grounded=_string(self.grounded),
        )


@dataclass(frozen=True, slots=True)
class Rule:
    """A function rule: it provides its own ``d:`` ref, its value of type ``outputs``."""

    provides: str
    outputs: str
    function: Callable[[ReadView], object]
    version: str | None = None
    reads: Sequence[str] = ()
    params: Sequence[str] = ()
    boundaries: Mapping[str, Any] | None = None
    candidates: bool = False
    grounded: str | None = None
    form: Literal["function"] = field(default="function", init=False)

    def __post_init__(self) -> None:
        _set(
            self,
            provides=_string(self.provides),
            outputs=_string(self.outputs),
            version=_string(self.version),
            reads=_strings(self.reads),
            params=_strings(self.params),
            boundaries=_mapping(self.boundaries),
            grounded=_string(self.grounded),
        )


@dataclass(frozen=True, slots=True)
class Table:
    """A rule table; v0.0.1 refuses it at check 12."""

    provides: str
    outputs: str
    hit: str
    inputs: Any
    rows: Any
    order: Sequence[Any] | None = None
    candidates: bool = False
    grounded: str | None = None
    form: Literal["table"] = field(default="table", init=False)

    def __post_init__(self) -> None:
        _set(self, provides=_string(self.provides))


@dataclass(frozen=True, slots=True)
class Policy:
    """A function policy: it reads ``reads`` and ``params`` and returns a plan."""

    function: Callable[[ReadView], object]
    version: str | None = None
    reads: Sequence[str] = ()
    params: Sequence[str] = ()
    boundaries: Mapping[str, Any] | None = None
    form: Literal["function"] = field(default="function", init=False)

    def __post_init__(self) -> None:
        _set(
            self,
            version=_string(self.version),
            reads=_strings(self.reads),
            params=_strings(self.params),
            boundaries=_mapping(self.boundaries),
        )


@dataclass(frozen=True, slots=True)
class PolicyTable:
    """The policy table; v0.0.1 refuses it at check 12."""

    inputs: Any
    rows: Any
    form: Literal["table"] = field(default="table", init=False)


@dataclass(frozen=True, slots=True)
class PlanDeclaration:
    """A plan declaration; v0.0.1 refuses it at check 12."""

    kind: str
    steps: Sequence[Any] = ()
    reason: Text | None = None
    reply: Text | None = None
    grounded: str | None = None


# ---------------------------------------------------------------------------------------------
# The design


@dataclass(frozen=True, slots=True)
class Design:
    """A design: the state declaration, the components and the blocks.

    ``sources``, ``rules``, ``plans``, ``types`` and ``groups`` map each declared name to its
    declaration; ``parameters`` maps each leaf's full path to a ``Parameter``; ``classes``
    maps a ref to its class tree; ``tools`` lists the tool names; each defaults to empty.
    """

    state: str
    policy: Policy | PolicyTable
    sources: Mapping[str, Source] = field(default_factory=_empty)
    rules: Mapping[str, Rule | Table] = field(default_factory=_empty)
    requirements: Sequence[Requirement] = ()
    default_deny: bool = True
    tools: Sequence[str] = ()
    parameters: Mapping[str, Parameter] = field(default_factory=_empty)
    types: Mapping[str, str] = field(default_factory=_empty)
    plans: Mapping[str, PlanDeclaration] = field(default_factory=_empty)
    groups: Mapping[str, Group] = field(default_factory=_empty)
    classes: Mapping[str, Any] = field(default_factory=_empty)
    name: str | None = None

    def __post_init__(self) -> None:
        types = _mapping(self.types)
        if isinstance(types, MappingProxyType):
            types = MappingProxyType(
                {k: _string(v) for k, v in cast("Mapping[Any, Any]", types).items()}
            )
        _set(
            self,
            state=_string(self.state),
            sources=_mapping(self.sources),
            rules=_mapping(self.rules),
            parameters=_mapping(self.parameters),
            plans=_mapping(self.plans),
            groups=_mapping(self.groups),
            requirements=_sequence(self.requirements),
            types=types,
            classes=_mapping(self.classes),
            tools=_strings(self.tools),
            name=_string(self.name),
        )


# ---------------------------------------------------------------------------------------------
# InvalidDesign


_BUILT_BY_LOAD = object()


@dataclass(frozen=True, slots=True)
class DesignFailure:
    """One failure load found: its check number, the names or refs it concerns in
    code-point order, and a one-line message. Built only by ``load``."""

    check: int
    names: tuple[str, ...]
    message: str
    token: InitVar[object] = None

    def __post_init__(self, token: object) -> None:
        if token is not _BUILT_BY_LOAD:
            raise TypeError("a DesignFailure is built only by load")


def new_failure(check: int, names: Sequence[str], message: str) -> DesignFailure:
    return DesignFailure(check, tuple(sorted(names)), message, _BUILT_BY_LOAD)


def _order(f: DesignFailure) -> tuple[int, list[str], str]:
    return (f.check, list(f.names), f.message)


class InvalidDesign(ValueError):
    """Load refused a design. ``failures`` holds at least one ``DesignFailure``,
    alike ones once, ordered by check number, then names, then message; the string is their
    messages, one per line, in that order. Raised only by ``load``."""

    failures: tuple[DesignFailure, ...]

    def __init__(self, failures: Sequence[DesignFailure], token: object = None) -> None:
        if token is not _BUILT_BY_LOAD:
            raise TypeError("an InvalidDesign is raised only by load")
        ordered = tuple(sorted(dict.fromkeys(failures), key=_order))
        if not ordered:
            raise ValueError("an InvalidDesign carries at least one failure")
        self.failures = ordered
        super().__init__("\n".join(f.message for f in ordered))

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle and copy rebuild the error through load's guard, then restore the rest of
        its state, such as its notes; the failures travel only through the rebuild."""
        return (_rebuild_invalid_design, (self.failures,), _state(self.__dict__))

    def __setstate__(self, state: object) -> None:
        """Restore what ``__reduce__`` kept; a ``failures`` entry is ignored, so the checked
        tuple the rebuild set stays."""
        if type(state) is not dict:
            raise TypeError("an InvalidDesign's state is a dict")
        for name, value in _state(cast("dict[str, object]", state)).items():
            setattr(self, name, value)


def _state(given: dict[str, object]) -> dict[str, object]:
    return {name: value for name, value in given.items() if name != "failures"}


def _rebuild_invalid_design(failures: Sequence[DesignFailure]) -> InvalidDesign:
    """An ``InvalidDesign`` built again from the failures a pickled or copied one held, each
    a ``DesignFailure`` and no subclass, so nothing else passes for what load found."""
    failures = tuple(failures)
    if not all(type(f) is DesignFailure for f in failures):
        raise TypeError("an InvalidDesign holds only DesignFailure instances")
    return InvalidDesign(failures, _BUILT_BY_LOAD)


# ---------------------------------------------------------------------------------------------
# The loaded design


class LoadedDesign:
    """A design that passed every load check, with what load computed once. Built
    only by ``load``. ``design_hash`` and ``document`` are public; the
    rest is the engine's.

    The private parts: ``_design``, the snapshotted ``Design``; ``_callables``, each function
    component keyed by declared name, ``policy``, or a source's id and ``.when`` for its function
    ``when``; ``_types``, each ``t:`` and ``d:`` ref's
    declared type, ``t:state`` included, and ``_leaf_types``, each leaf's type, each with its
    definitions resolved; ``_needs``, each source's needs with its question slots' refs
    added; ``_groups``, each group's refs in code-point order; ``_graph``,
    each node's predecessors; ``_providers``, the ref-to-provider map;
    ``_grid``, the answer grid; ``_labels``, each ``q:`` ref's label names and descriptions,
    aligned, ``None`` where a label is undescribed, ``yes`` and ``no`` for ``noul`` and none for
    ``extraction``; ``_requirement_bytes``, each source requirement's values as
    canonical bytes, by source id. The class trees, payloads, leaf values and plan-level
    requirements are decoded afresh at each read, so nothing a caller
    changes after load reaches a turn.
    """

    __slots__ = (
        "_callables",
        "_classes_bytes",
        "_design",
        "_document",
        "_graph",
        "_grid",
        "_groups",
        "_hash",
        "_labels",
        "_leaf_bytes",
        "_leaf_types",
        "_needs",
        "_payload_bytes",
        "_providers",
        "_requirement_bytes",
        "_requirements_bytes",
        "_types",
    )

    _design: Design
    _callables: Mapping[str, Callable[..., Any]]
    _types: Mapping[str, Type]
    _leaf_types: Mapping[str, Type]
    _needs: Mapping[str, frozenset[str]]
    _groups: Mapping[str, tuple[str, ...]]
    _graph: Mapping[str, frozenset[str]]
    _providers: Mapping[str, str]
    _grid: Mapping[str, Any]
    _labels: Mapping[str, tuple[tuple[str, ...], tuple[str | None, ...]]]
    _classes_bytes: bytes
    _payload_bytes: Mapping[str, bytes]
    _leaf_bytes: Mapping[str, bytes]
    _requirement_bytes: Mapping[str, frozenset[bytes]]
    _requirements_bytes: bytes
    _document: bytes
    _hash: str

    def __init__(self, token: object, **parts: object) -> None:
        if token is not _BUILT_BY_LOAD:
            raise TypeError("a LoadedDesign is built only by load")
        for name, value in parts.items():
            object.__setattr__(self, "_" + name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a LoadedDesign is read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a LoadedDesign is read-only")

    @property
    def design_hash(self) -> str:
        """The design hash over ``document``."""
        return self._hash

    @property
    def document(self) -> bytes:
        """The design's canonical bytes."""
        return self._document

    @property
    def _classes(self) -> dict[str, Any]:
        """Each ref's class tree in full, keyed by ``t:state`` and every provided ref."""
        return cast("dict[str, Any]", decode(self._classes_bytes))

    def _payload(self, source: str) -> Any:
        """A source's payload as decoded from the document."""
        return decode(self._payload_bytes[source])

    def _requirements(self) -> tuple[Requirement, ...]:
        """The plan-level requirements as decoded from the document, in declaration order."""
        decoded = cast("list[dict[str, Any]]", decode(self._requirements_bytes))
        return tuple(Requirement(r["ref"], r["value"]) for r in decoded)

    def _leaf_value(self, path: str) -> Any:
        """A leaf's value as decoded from the document; a ``bytes`` value as ``bytes``."""
        return live_value(decode(self._leaf_bytes[path]), self._leaf_types[path])


# ---------------------------------------------------------------------------------------------
# A leaf's value in its two forms


def document_value(value: object, t: Type) -> object:
    """A conforming leaf value as the document writes it: ``bytes`` as unpadded base64url.
    ``t`` is a leaf type with its definitions resolved: a scalar, or a list or mapping of
    one."""
    if isinstance(t, Scalar) and t.name == "bytes":
        return b64url_encode(cast(bytes, value))
    if isinstance(t, ListOf):
        return [document_value(v, t.item) for v in cast("Sequence[object]", value)]
    if isinstance(t, MappingOf):
        return {k: document_value(v, t.value) for k, v in cast("dict[str, object]", value).items()}
    return value


def live_value(value: object, t: Type) -> object:
    """A leaf value decoded from the document, with each ``bytes`` value as Python ``bytes``."""
    if isinstance(t, Scalar) and t.name == "bytes":
        return b64url_decode(cast(str, value))
    if isinstance(t, ListOf):
        return [live_value(v, t.item) for v in cast("list[object]", value)]
    if isinstance(t, MappingOf):
        return {k: live_value(v, t.value) for k, v in cast("dict[str, object]", value).items()}
    return value
