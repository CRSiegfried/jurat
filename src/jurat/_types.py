"""The type set: its notation, type definitions, conformance and the type of a dot-path read.

A type is one string spelled one way. ``parse_type`` reads a string
strictly and never raises: a string that does not parse is ``malformed``; one that parses
but breaks a rule of the notation (an empty or repeated label, an interval out of order or
with a bound that names no domain number exactly, ``or null`` where null is not allowed,
``answer`` inside another type or as ``answer or null``, nesting more than 64 deep) is
``ill-formed``; a type spelled otherwise is ``not-canonical`` and its failure carries the
canonical spelling. Load reports each under check 10. ``render`` gives the
canonical string, which is the hashed form.

A type's depth is counted with its definitions resolved, each list, mapping and record one
level; ``check_declared`` refuses a type deeper than ``MAX_DEPTH``, so a
conforming value is within the value domain's bound. ``conforms`` and ``value_conforms``
take types that passed check 10, and recurse no deeper than such a type.

Conformance is checked type to type and value to type; a value conforms by its exact
Python type. A ``bytes`` value is Python ``bytes`` live and unpadded
base64url in the document, so value conformance takes the form explicitly.
"""

import base64
import bisect
import dataclasses
import enum
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NoReturn, cast

from jurat._canonical import encode, integer_literal
from jurat._values import (
    MAX_DEPTH,
    MAX_INTEGER,
    clipped,
    described,
    is_scalar_string,
    quote,
    string_data,
    type_name,
)

ScalarName = Literal["number", "integer", "string", "boolean", "bytes"]
NumericBase = Literal["number", "integer"]
Form = Literal["live", "document"]


@dataclass(frozen=True, slots=True)
class Scalar:
    name: ScalarName


@dataclass(frozen=True, slots=True)
class Interval:
    """A closed interval with literal bounds; an ``integer`` interval's bounds are ints."""

    base: NumericBase
    lo: int | float
    hi: int | float


@dataclass(frozen=True, slots=True)
class Label:
    """A label type; ``values`` are distinct and in code-point order."""

    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ListOf:
    item: "Type"


@dataclass(frozen=True, slots=True)
class MappingOf:
    value: "Type"


@dataclass(frozen=True, slots=True)
class RecordOf:
    """A record type; field names are distinct segments in code-point order."""

    fields: tuple[tuple[str, "Type"], ...]


@dataclass(frozen=True, slots=True)
class Nullable:
    inner: "Type"


@dataclass(frozen=True, slots=True)
class Named:
    """A type definition's name, resolved against the design's definitions."""

    name: str


@dataclass(frozen=True, slots=True)
class AnswerType:
    """The type of every ``q:`` entry; parseable, never declared."""


Type = Scalar | Interval | Label | ListOf | MappingOf | RecordOf | Nullable | Named | AnswerType


class FailureKind(enum.StrEnum):
    malformed = "malformed"
    ill_formed = "ill-formed"
    not_canonical = "not-canonical"
    undefined = "undefined"
    cycle = "cycle"
    not_declarable = "not-declarable"


@dataclass(frozen=True, slots=True)
class TypeFailure:
    """Why a type string or a type fails; load reports it under its check.

    ``canonical`` is the canonical spelling of a ``not-canonical`` string; ``names`` are the
    definition names an ``undefined`` or ``cycle`` failure is about.
    """

    kind: FailureKind
    message: str
    canonical: str | None = None
    names: tuple[str, ...] = ()


_WORDS = frozenset(
    {
        "number",
        "integer",
        "string",
        "boolean",
        "bytes",
        "label",
        "list",
        "mapping",
        "record",
        "answer",
        "or",
        "null",
    }
)
_SCALARS: frozenset[str] = frozenset({"number", "integer", "string", "boolean", "bytes"})
_SEGMENT_RE = re.compile(r"[a-z][a-z0-9_]*")
_NAME_RE = re.compile(r"[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)*")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:/[A-Za-z0-9_]*)*")
_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?P<frac>\.[0-9]+)?(?P<exp>[eE][+-]?[0-9]+)?")
_SPACE = " \t\n\r"
_JSON = json.JSONDecoder(strict=True)


class _Malformed(ValueError):
    pass


class _Invalid(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Inexact:
    """An interval bound's integer literal that names no domain number exactly."""

    text: str


# --- the lenient parse: syntax only -------------------------------------------------------
# A raw node is a tuple: ("scalar", name), ("answer",), ("interval", base, lo, hi),
# ("label", values), ("list", node), ("mapping", node), ("record", [(field, node)]),
# ("nullable", node), ("named", name).

_Raw = tuple[object, ...]


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.depth = 0  # open lists, mappings and records: the type's depth so far

    def fail(self, what: str) -> NoReturn:
        raise _Malformed(f"{what} at position {self.pos}")

    def skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in _SPACE:
            self.pos += 1

    def peek(self) -> str:
        self.skip()
        return self.text[self.pos : self.pos + 1]

    def expect(self, char: str) -> None:
        if self.peek() != char:
            self.fail(f"expected {quote(char)}")
        self.pos += 1

    def word(self) -> str:
        self.skip()
        m = _WORD_RE.match(self.text, self.pos)
        if m is None:
            self.fail("expected a type")
        self.pos = m.end()
        return m.group()

    def top(self) -> _Raw:
        node = self.type_()
        save = self.pos
        if self.peek() and _WORD_RE.match(self.text, self.pos):
            if self.word() != "or" or self.peek() == "" or self.word() != "null":
                self.pos = save
                self.fail("expected 'or null'")
            return ("nullable", node)
        return node

    def nest(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            # The parse recurses per level, so it stops here rather than at Python's limit.
            raise _Invalid(f"the type nests more than {MAX_DEPTH} deep")

    def type_(self) -> _Raw:
        w = self.word()
        if w in ("number", "integer") and self.peek() == "[":
            return self.interval(w)
        if w in _SCALARS:
            return ("scalar", w)
        if w == "answer":
            return ("answer",)
        if w == "label":
            return self.label()
        if w == "list":
            self.expect("<")
            self.nest()
            node = self.top()
            self.expect(">")
            self.depth -= 1
            return ("list", node)
        if w == "mapping":
            self.expect("<")
            if self.word() != "string":
                self.fail("a mapping's key type is string")
            self.expect(",")
            self.nest()
            node = self.top()
            self.expect(">")
            self.depth -= 1
            return ("mapping", node)
        if w == "record":
            self.nest()
            node = self.record()
            self.depth -= 1
            return node
        if w in _WORDS or not _NAME_RE.fullmatch(w):
            self.fail(f"{quote(w)} is not a type")
        return ("named", w)

    def interval(self, base: str) -> _Raw:
        self.expect("[")
        lo = self.number()
        self.expect(",")
        hi = self.number()
        self.expect("]")
        return ("interval", base, lo, hi)

    def number(self) -> int | float | _Inexact:
        self.skip()
        m = _NUMBER_RE.match(self.text, self.pos)
        if m is None:
            self.fail("expected a JSON number")
        text = m.group()
        if m.group("frac") is None and m.group("exp") is None:
            try:
                number = integer_literal(text)
            except ValueError:  # over Python's digit limit for int
                self.fail(f"a bound of {len(text)} digits")
            self.pos = m.end()
            return _Inexact(text) if number is None else number
        self.pos = m.end()
        return float(text)

    def label(self) -> _Raw:
        self.expect("{")
        values: list[str] = []
        if self.peek() == "}":
            self.pos += 1
            return ("label", values)
        while True:
            self.skip()
            if self.peek() != '"':
                self.fail("expected a JSON string")
            try:
                value, self.pos = _JSON.raw_decode(self.text, self.pos)
            except json.JSONDecodeError:
                self.fail("expected a JSON string")
            values.append(cast(str, value))
            if self.peek() == "}":
                self.pos += 1
                return ("label", values)
            self.expect(",")

    def record(self) -> _Raw:
        self.expect("{")
        fields: list[tuple[str, _Raw]] = []
        if self.peek() == "}":
            self.pos += 1
            return ("record", fields)
        while True:
            name = self.word()
            self.expect(":")
            fields.append((name, self.top()))
            if self.peek() == "}":
                self.pos += 1
                return ("record", fields)
            self.expect(",")


def _parse_raw(text: str) -> _Raw:
    parser = _Parser(text)
    node = parser.top()
    if parser.peek() != "":
        parser.fail("unexpected text")
    return node


def _array(values: Sequence[str]) -> str:
    """A list of strings as its JSON array, for a message."""
    return "[" + ", ".join(quote(v) for v in values) + "]"


# --- building a type: the rules of the notation -------------------------------------------


def _bound(value: object, base: str) -> int | float:
    if type(value) is not int and type(value) is not float:
        raise TypeError(f"an interval bound is an int or a float, not {type_name(value)}")
    number: int | float = value
    if type(number) is int and not -MAX_INTEGER <= number <= MAX_INTEGER:
        raise _Invalid(f"bound {described(number)} is outside the value domain")
    if not math.isfinite(number):
        raise _Invalid(f"bound {described(number)} is not finite")
    if base == "integer":
        if not float(number).is_integer():
            raise _Invalid(f"integer bound {number} is not integral")
        number = int(number)
        if not -MAX_INTEGER <= number <= MAX_INTEGER:
            raise _Invalid(f"integer bound {described(number)} exceeds magnitude 2^53 - 1")
    elif number == 0:
        number = 0  # -0 and 0.0 are the one bound 0
    return number


def make_interval(base: NumericBase, lo: object, hi: object) -> Interval:
    """An interval with both bounds; raises ``ValueError`` or ``TypeError``.

    A bound is a domain number: a value whose type is exactly ``int``, of
    magnitude at most 2^53 - 1, or exactly ``float``, finite. An ``integer`` interval's bounds
    are integral, and a ``float`` one is held as its ``int``.
    """
    low, high = _bound(lo, base), _bound(hi, base)
    if low > high:
        raise _Invalid(f"interval [{lo}, {hi}] has lo above hi")
    return Interval(base, low, high)


def make_label(values: Sequence[object]) -> Label:
    """A label type of at least one distinct string, held in code-point order."""
    if not values:
        raise _Invalid("a label type lists at least one value")
    for v in values:
        if not issubclass(type(v), str):
            raise TypeError(f"a label value is a string, not {type_name(v)}")
        if not is_scalar_string(v):
            raise _Invalid(f"label value {quote(cast(str, v))} holds a surrogate code point")
    data = [string_data(cast(str, v)) for v in values]
    if len(set(data)) != len(data):
        raise _Invalid(f"label values {_array(data)} repeat a value")
    return Label(tuple(sorted(data)))


def _nested(t: Type, where: str) -> Type:
    if isinstance(t, Nullable):
        raise _Invalid(f"{where} is not nullable; 'or null' stands only as a whole type")
    if isinstance(t, AnswerType):
        raise _Invalid(f"{where} is any type but answer")
    return t


def make_list(item: Type) -> ListOf:
    return ListOf(_nested(item, "a list's member type"))


def make_mapping(value: Type) -> MappingOf:
    return MappingOf(_nested(value, "a mapping's value type"))


def make_record(fields: Sequence[tuple[str, Type]]) -> RecordOf:
    names = [name for name, _ in fields]
    for name in names:
        if not _SEGMENT_RE.fullmatch(name):
            raise _Invalid(f"record field name {quote(name)} is not a segment")
    if len(set(names)) != len(names):
        raise _Invalid(f"record field names {_array(names)} repeat a name")
    for name, ft in fields:
        inner = ft.inner if isinstance(ft, Nullable) else ft
        if isinstance(inner, AnswerType):
            raise _Invalid(f"record field {quote(name)} is any type but answer")
    return RecordOf(tuple(sorted(fields, key=lambda f: f[0])))


def make_nullable(inner: Type) -> Nullable:
    """``T or null`` over a ``T`` that is neither nullable nor ``answer``.

    Where it may stand (a ``t:`` ref's type, an ``outputs`` type, a record field's type or a
    type definition's body) is the caller's, and ``T``'s definitions are resolved by
    ``check_declared``.
    """
    if isinstance(inner, Nullable):
        raise _Invalid("the type in 'T or null' is not itself nullable")
    if isinstance(inner, AnswerType):
        raise _Invalid("answer is never declared, so 'answer or null' is not a type")
    return Nullable(inner)


def _build(node: _Raw) -> Type:
    tag = node[0]
    if tag == "scalar":
        return Scalar(cast(ScalarName, node[1]))
    if tag == "answer":
        return AnswerType()
    if tag == "interval":
        for bound in node[2:]:
            if isinstance(bound, _Inexact):
                raise _Invalid(f"bound {bound.text} names no domain number exactly")
        return make_interval(cast(NumericBase, node[1]), node[2], node[3])
    if tag == "label":
        return make_label(cast("list[str]", node[1]))
    if tag == "list":
        return make_list(_build(cast(_Raw, node[1])))
    if tag == "mapping":
        return make_mapping(_build(cast(_Raw, node[1])))
    if tag == "record":
        raw_fields = cast("list[tuple[str, _Raw]]", node[1])
        return make_record([(name, _build(raw)) for name, raw in raw_fields])
    if tag == "nullable":
        return make_nullable(_build(cast(_Raw, node[1])))
    return Named(cast(str, node[1]))


def parse_type(text: str, *, nullable: bool) -> Type | TypeFailure:
    """Parse a type string strictly; ``nullable`` says whether ``T or null`` may stand whole.

    ``T or null`` may be the whole of a ``t:`` ref's type, an ``outputs`` type, a record
    field's type or a type definition's body; the caller says which position
    it parses. A ``str`` subclass is read by its string data. Never raises: every string, and
    every other value, gives a type or a ``TypeFailure``.
    """
    if not issubclass(type(text), str):  # pyright: ignore[reportUnnecessaryIsInstance]
        return TypeFailure(FailureKind.malformed, f"a type of {type_name(text)} is not a string")
    text = string_data(text)
    try:
        built = _build(_parse_raw(text))
    except _Malformed as exc:
        return TypeFailure(FailureKind.malformed, f"type {quote(text)} does not parse: {exc}")
    except _Invalid as exc:
        return TypeFailure(FailureKind.ill_formed, f"type {quote(text)} is ill-formed: {exc}")
    if isinstance(built, Nullable) and not nullable:
        return TypeFailure(
            FailureKind.ill_formed, f"type {quote(text)} admits null where null is not allowed"
        )
    canonical = render(built)
    if canonical != text:
        return TypeFailure(
            FailureKind.not_canonical,
            f"type {quote(text)} is not spelled canonically; the spelling is {canonical}",
            canonical=canonical,
        )
    return built


# --- rendering ----------------------------------------------------------------------------


def _json(value: object) -> str:
    return encode(value).decode("utf-8")


def render(t: Type) -> str:
    """The one canonical string of a type."""
    if isinstance(t, Scalar):
        return t.name
    if isinstance(t, AnswerType):
        return "answer"
    if isinstance(t, Interval):
        return f"{t.base} [{_json(t.lo)}, {_json(t.hi)}]"
    if isinstance(t, Label):
        return "label{" + ", ".join(_json(v) for v in t.values) + "}"
    if isinstance(t, ListOf):
        return f"list<{render(t.item)}>"
    if isinstance(t, MappingOf):
        return f"mapping<string, {render(t.value)}>"
    if isinstance(t, RecordOf):
        return "record{" + ", ".join(f"{n}: {render(ft)}" for n, ft in t.fields) + "}"
    if isinstance(t, Nullable):
        return f"{render(t.inner)} or null"
    return t.name


# --- definitions --------------------------------------------------------------------------


def _children(t: Type) -> list[Type]:
    if isinstance(t, ListOf):
        return [t.item]
    if isinstance(t, MappingOf):
        return [t.value]
    if isinstance(t, RecordOf):
        return [ft for _, ft in t.fields]
    if isinstance(t, Nullable):
        return [t.inner]
    return []


def named(t: Type) -> list[str]:
    """The definition names a type names, in order of appearance, without repeats."""
    out: dict[str, None] = {}
    stack = [t]
    while stack:
        node = stack.pop()
        if isinstance(node, Named):
            out.setdefault(node.name)
        stack.extend(reversed(_children(node)))
    return list(out)


def check_definitions(definitions: Mapping[str, Type]) -> list[TypeFailure]:
    """Every name a definition names is defined, and no definitions name each other in a
    cycle; each cycle fails once, naming its definitions in code-point order."""
    failures: list[TypeFailure] = []
    for name in sorted(definitions):
        for missing in sorted(n for n in named(definitions[name]) if n not in definitions):
            failures.append(
                TypeFailure(
                    FailureKind.undefined,
                    f"type definition {quote(name)} names {quote(missing)}, which is not defined",
                    names=(name, missing),
                )
            )
    for cycle in _cycles(definitions):
        failures.append(
            TypeFailure(
                FailureKind.cycle,
                f"type definitions {', '.join(cycle)} name each other in a cycle",
                names=cycle,
            )
        )
    return failures


def _cycles(definitions: Mapping[str, Type]) -> list[tuple[str, ...]]:
    """Strongly connected components with a cycle (Tarjan, iterative), sorted."""
    graph = {n: [m for m in named(t) if m in definitions] for n, t in definitions.items()}
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    found: list[tuple[str, ...]] = []

    def open_(v: str) -> tuple[str, Iterator[str]]:
        index[v] = low[v] = len(index)
        stack.append(v)
        on_stack.add(v)
        return v, iter(graph[v])

    for root in sorted(graph):
        if root in index:
            continue
        work = [open_(root)]
        while work:
            v, edges = work[-1]
            w = next(edges, None)
            if w is not None:
                if w not in index:
                    work.append(open_(w))
                elif w in on_stack:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
            if low[v] == index[v]:
                component: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                if len(component) > 1 or v in graph[v]:
                    found.append(tuple(sorted(component)))
    return sorted(found)


def resolve(t: Type, definitions: Mapping[str, Type]) -> Type:
    """The type a name stands for, through any chain of names; other types unchanged.

    The definitions are those ``check_definitions`` passes: every name defined, no cycle.
    """
    seen: set[str] = set()
    while isinstance(t, Named):
        if t.name in seen or t.name not in definitions:
            raise ValueError(f"type definition {quote(t.name)} is undefined or in a cycle")
        seen.add(t.name)
        t = definitions[t.name]
    return t


@dataclass(slots=True)
class DeclaredMemo:
    """What ``check_declared`` learnt of the definitions, kept across its calls over one set of
    definitions: each node's depth, and each name, with whether null may stand there, whose
    walk found no failure. So a chain of definitions is walked once, not once per use."""

    depths: dict[str, int] = dataclasses.field(default_factory=dict[str, int])
    clean: set[tuple[str, bool]] = dataclasses.field(default_factory=set[tuple[str, bool]])


def check_declared(
    t: Type,
    definitions: Mapping[str, Type],
    *,
    nullable: bool,
    memo: DeclaredMemo | None = None,
) -> list[TypeFailure]:
    """A declared type, its definitions resolved: at most ``MAX_DEPTH`` deep, every name
    defined, no ``answer``, and null only where ``nullable`` and record fields allow it, a
    name that resolves to a nullable type included.

    ``nullable`` is true for a ``t:`` ref's type, an ``outputs`` type and a type definition's
    body, so a definition's body is checked by this call too. A type deeper than the bound
    fails with that one failure. The definitions are those ``check_definitions`` passes,
    apart from undefined names. ``memo`` carries what earlier calls over the same definitions
    learnt.
    """
    memo = DeclaredMemo() if memo is None else memo
    clean = memo.clean
    if type_depth(t, definitions, memo.depths) > MAX_DEPTH:
        return [
            TypeFailure(
                FailureKind.ill_formed,
                f"the type nests more than {MAX_DEPTH} deep with its definitions resolved",
            )
        ]
    failures: list[TypeFailure] = []
    seen: set[tuple[str, bool]] = set()

    def walk(node: Type, null_ok: bool, where: str) -> bool:
        """Walk a type; whether the walk was whole: no name in it was cut short because this
        call had met it already, so a chain found clean may be kept as clean."""
        # A chain of names is followed in a loop; the recursion is over the type's levels,
        # at most MAX_DEPTH deep.
        chain: list[tuple[str, bool]] = []
        before = len(failures)
        while isinstance(node, Named):
            if node.name not in definitions:
                failures.append(
                    TypeFailure(
                        FailureKind.undefined,
                        f"{where} names {quote(node.name)}, which is not defined",
                        names=(node.name,),
                    )
                )
                return False
            key = (node.name, null_ok)
            if key in clean:
                clean.update(chain)
                return True
            if key in seen:
                return False  # its walk is this call's, not yet known clean
            seen.add(key)
            chain.append(key)
            where = f"type definition {quote(node.name)}"
            node = definitions[node.name]
        whole = body(node, null_ok, where)
        if whole and len(failures) == before:
            clean.update(chain)
        return whole

    def body(node: Type, null_ok: bool, where: str) -> bool:
        if isinstance(node, AnswerType):
            failures.append(
                TypeFailure(FailureKind.not_declarable, f"{where} declares answer, never declared")
            )
            return True
        if isinstance(node, Nullable):
            if not null_ok:
                failures.append(
                    TypeFailure(FailureKind.ill_formed, f"{where} admits null where it may not")
                )
                return True
            return walk(node.inner, False, where)
        whole = True
        if isinstance(node, RecordOf):
            for name, ft in node.fields:
                whole = walk(ft, True, f"record field {quote(name)}") and whole
        else:
            for child in _children(node):
                whole = walk(child, False, where) and whole
        return whole

    walk(t, nullable, "the type")
    return failures


_LEVELS = (ListOf, MappingOf, RecordOf)


def type_depth(
    t: Type, definitions: Mapping[str, Type], named: dict[str, int] | None = None
) -> int:
    """A type's depth with its definitions resolved: each ``list``, ``mapping`` and
    ``record`` one level; ``or null``, a scalar and a name add none.

    Iterative, and each definition is counted once, so a long chain of names costs no
    recursion. An undefined name counts as a scalar. Raises ``ValueError`` for definitions
    that name each other in a cycle, which ``check_definitions`` reports first. ``named``
    keeps each defined name's depth across calls over the same definitions.
    """
    named = {} if named is None else named
    depth: dict[int, int] = {}  # id of a node -> its depth
    active: set[int] = set()
    pending: list[tuple[Type, bool]] = [(t, False)]
    while pending:
        node, expanded = pending.pop()
        if isinstance(node, Named) and node.name in named:
            depth[id(node)] = named[node.name]
            continue
        if isinstance(node, Named):
            body = definitions.get(node.name)
            kids = [] if body is None else [body]
        else:
            kids = _children(node)
        if not expanded:
            if id(node) in active:
                raise ValueError("type definitions name each other in a cycle")
            active.add(id(node))
            pending.append((node, True))
            pending.extend((kid, False) for kid in kids)
            continue
        active.discard(id(node))
        deepest = max((depth[id(kid)] for kid in kids), default=0)
        depth[id(node)] = deepest + (1 if isinstance(node, _LEVELS) else 0)
        if isinstance(node, Named) and node.name in definitions:
            named[node.name] = depth[id(node)]
    return depth[id(t)]


# --- type-to-type conformance -------------------------------------------------------------


def _numeric(t: Type) -> tuple[str, tuple[int | float, int | float] | None] | None:
    if isinstance(t, Scalar) and t.name in ("number", "integer"):
        return t.name, None
    if isinstance(t, Interval):
        return t.base, (t.lo, t.hi)
    return None


def conforms(s: Type, t: Type, definitions: Mapping[str, Type]) -> bool:
    """Whether ``s`` conforms to ``t`` by the type set's forms, at every depth.

    By form, from the two types alone: an interval conforms to a containing interval and to
    the unbounded type, and the unbounded type to no interval, whatever its bounds.
    """
    s, t = resolve(s, definitions), resolve(t, definitions)
    if isinstance(t, Nullable):
        if isinstance(s, Nullable):
            return conforms(s.inner, t.inner, definitions)
        return conforms(s, t.inner, definitions)
    if isinstance(s, Nullable):
        return False
    sn, tn = _numeric(s), _numeric(t)
    if sn is not None and tn is not None:
        (s_base, s_range), (t_base, t_range) = sn, tn
        if t_base == "integer" and s_base != "integer":
            return False
        if t_range is None:
            return True
        return s_range is not None and t_range[0] <= s_range[0] and s_range[1] <= t_range[1]
    if isinstance(t, Label):
        return isinstance(s, Label) and set(s.values) <= set(t.values)
    if isinstance(t, Scalar) and t.name == "string":
        return isinstance(s, Label) or s == t
    if isinstance(t, ListOf):
        return isinstance(s, ListOf) and conforms(s.item, t.item, definitions)
    if isinstance(t, MappingOf):
        return isinstance(s, MappingOf) and conforms(s.value, t.value, definitions)
    if isinstance(t, RecordOf):
        if not isinstance(s, RecordOf):
            return False
        have = dict(s.fields)
        return all(n in have and conforms(have[n], ft, definitions) for n, ft in t.fields)
    return s == t  # boolean, bytes, answer


# --- value-to-type conformance ------------------------------------------------------------


def value_conforms(value: object, t: Type, definitions: Mapping[str, Type], *, form: Form) -> bool:
    """Whether a value conforms to a declared type by its exact Python type.

    ``form`` is ``"live"``, where a ``bytes`` value is Python ``bytes``, or ``"document"``,
    where it is an unpadded base64url string. A record value holds every
    declared field; an absent field does not conform even when its type admits null, and an
    undeclared field conforms. An ``answer`` is validated by ``Answer``, not here.
    """
    # Read by its type and string data, so no comparison of the caller's runs.
    if not issubclass(type(form), str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError(f"form {described(form)} is not live or document")
    data = string_data(form)
    if data != "live" and data != "document":
        raise ValueError(f"form {clipped(data)} is not live or document")
    return _value(value, t, definitions, data)


def _value(value: object, t: Type, definitions: Mapping[str, Type], form: Form) -> bool:
    t = resolve(t, definitions)
    if isinstance(t, Nullable):
        return value is None or _value(value, t.inner, definitions, form)
    if isinstance(t, AnswerType):
        raise ValueError("an answer is validated by Answer")
    numeric = _numeric(t)
    if numeric is not None:
        base, bounds = numeric
        if type(value) is int:
            number: int | float = value
            if not -MAX_INTEGER <= number <= MAX_INTEGER:
                return False
        elif type(value) is float and base == "number":
            number = value
            if not math.isfinite(number):
                return False
        else:
            return False
        return bounds is None or bounds[0] <= number <= bounds[1]
    if isinstance(t, Label):
        # The values are in code-point order, as str orders them: found by bisection.
        if not is_scalar_string(value):
            return False
        data = string_data(cast(str, value))
        at = bisect.bisect_left(t.values, data)
        return at < len(t.values) and t.values[at] == data
    if isinstance(t, Scalar):
        if t.name == "string":
            return is_scalar_string(value)
        if t.name == "boolean":
            return type(value) is bool
        if form == "live":
            return type(value) is bytes
        return issubclass(type(value), str) and b64url_decode(cast(str, value)) is not None
    if isinstance(t, ListOf):
        if type(value) is not list and type(value) is not tuple:
            return False
        items = cast("list[object] | tuple[object, ...]", value)
        return all(_value(v, t.item, definitions, form) for v in items)
    members = _object(value)
    if members is None:
        return False
    if isinstance(t, MappingOf):
        return all(_value(v, t.value, definitions, form) for v in members.values())
    assert isinstance(t, RecordOf)
    return all(n in members and _value(members[n], ft, definitions, form) for n, ft in t.fields)


def hold_failure(value: object, t: Type, definitions: Mapping[str, Type]) -> str | None:
    """Why a requirement declaring ``value`` on a ref of type ``t`` can never hold, or ``None``
    when it can: ``t`` holds ``bytes`` at some depth, ``value`` does not
    conform to it, or ``value`` holds a field ``t`` does not declare. Load check 9 and plan
    rule 6 ask this. The walk follows ``t``, a type that passed check 10, so a cyclic value
    ends at its depth."""
    if _holds_bytes(t, definitions):
        return "its ref's type holds bytes"
    if not _value(value, t, definitions, "live"):
        return "its value does not conform to its ref's type"
    if not _declared_only(value, t, definitions):
        return "its value holds a field its ref's type does not declare"
    return None


def _holds_bytes(t: Type, definitions: Mapping[str, Type]) -> bool:
    """Whether ``t`` holds ``bytes`` at some depth: an iterative walk that visits each
    definition, and each type object a resolved type shares, once, so a type named or shared
    at many places is not walked at each."""
    stack = [t]
    seen: set[str] = set()
    visited: set[int] = set()  # by identity: a type object is immutable and held by the walk
    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, Named):
            if node.name not in seen:
                seen.add(node.name)
                stack.append(definitions[node.name])
        elif isinstance(node, Scalar):
            if node.name == "bytes":
                return True
        else:
            stack.extend(_children(node))
    return False


def _declared_only(value: object, t: Type, definitions: Mapping[str, Type]) -> bool:
    """Whether a value that conforms to ``t`` holds no field ``t`` does not declare."""
    t = resolve(t, definitions)
    if isinstance(t, Nullable):
        return value is None or _declared_only(value, t.inner, definitions)
    if isinstance(t, ListOf):
        items = cast("list[object] | tuple[object, ...]", value)
        return all(_declared_only(v, t.item, definitions) for v in items)
    if isinstance(t, MappingOf):
        members = cast("dict[str, object]", _object(value))
        return all(_declared_only(v, t.value, definitions) for v in members.values())
    if isinstance(t, RecordOf):
        members = cast("dict[str, object]", _object(value))
        fields = dict(t.fields)
        return all(
            n in fields and _declared_only(v, fields[n], definitions) for n, v in members.items()
        )
    return True


def _object(value: object) -> dict[str, object] | None:
    """An exact ``dict`` with string keys, keyed by their string data, which no two share."""
    if type(value) is not dict:
        return None
    out: dict[str, object] = {}
    for k, v in cast("dict[object, object]", value).items():
        if not is_scalar_string(k):
            return None
        data = string_data(cast(str, k))
        if data in out:
            return None
        out[data] = v
    return out


# --- unpadded base64url ---------------------------------------------------

_B64URL_RE = re.compile(r"[A-Za-z0-9_-]*")


def b64url_encode(data: bytes) -> str:
    """Unpadded base64url (RFC 4648, section 5) of ``data``."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes | None:
    """The bytes of a strict unpadded base64url string, or ``None``.

    Strict: only ``A-Z a-z 0-9 - _``, no ``=``, a length not 1 modulo 4, and zero trailing
    bits, so each byte string has one spelling.
    """
    if not issubclass(type(text), str):  # pyright: ignore[reportUnnecessaryIsInstance]
        return None
    data = string_data(text)
    if not _B64URL_RE.fullmatch(data) or len(data) % 4 == 1:
        return None
    raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    return raw if b64url_encode(raw) == data else None


# --- the type of a dot-path read ------------------------------------------


def read_type(
    t: Type, fields: Sequence[str], definitions: Mapping[str, Type]
) -> Type | TypeFailure:
    """The type of a read through ``fields``: the last field's type, or that type or null
    when the read's type or any field on the path admits null, since a null on the path
    makes the read null. A path through a non-record, or naming no field, fails (check 9).
    """
    current = t
    admits_null = False
    path = ""
    for field in fields:
        node = resolve(current, definitions)
        if isinstance(node, Nullable):
            admits_null = True
            node = resolve(node.inner, definitions)
        if not isinstance(node, RecordOf):
            where = f"field {quote(path)}" if path else "the read's type"
            return TypeFailure(
                FailureKind.ill_formed,
                f"the path .{'.'.join(fields)} reads {quote(field)} through {where}, not a record",
            )
        found = dict(node.fields)
        if field not in found:
            return TypeFailure(
                FailureKind.ill_formed,
                f"the path .{'.'.join(fields)} names no field {quote(field)}",
            )
        current = found[field]
        path = field
    if admits_null and not isinstance(resolve(current, definitions), Nullable):
        return Nullable(current)
    return current
