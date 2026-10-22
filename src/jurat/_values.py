"""The type-free value domain of the record document.

A value is in the domain when it is null, a boolean, a finite number, an integer of
magnitude at most 2^53 - 1, a string of Unicode scalar values, a list (or tuple) of domain
values, or an object with string keys and domain values, nesting at most ``MAX_DEPTH`` deep
and within the size bounds, ``MAX_VALUES`` values and ``MAX_CODE_POINTS`` code points counted
at every place each stands. Conformance to a declared type is checked elsewhere; this is the
part every recorded value meets whatever its type. An item, a document or another holder of
values is held to its own depth and to no size bound (``domain_error`` given its depth).

Exact types only: a ``bool`` is a boolean, never an integer; ``int``, ``float``, ``list``,
``tuple`` and ``dict`` are taken only as those exact types; a ``str`` or any subclass of it
is read by its string data, which no override reaches, so a ``StrEnum`` member is its value.
An object's keys are compared by their string data. Python ``bytes`` are outside.
"""

import json
import math
from collections.abc import Callable, Iterable, Iterator
from typing import Any, cast

MAX_INTEGER = 2**53 - 1
"""The largest integer magnitude in the domain, so every integer is an exact double."""

MAX_DEPTH = 64
"""The deepest a domain value nests: a scalar 0 deep, a list or object one deeper than its
deepest member."""

MAX_DOCUMENT_DEPTH = 128
"""The deepest a document the encoder writes or a reader parses nests."""

FORMAT = 3
"""The format number this engine writes into a record and a design document. It reads every
format from 1 up to it, each under that format's rules."""

FORMATS = frozenset(range(1, FORMAT + 1))
"""The formats this engine reads."""

EXTRACTION_FORMAT = 2
"""The first format with the extraction answer kind and a call's ``truncated``."""

DESCRIPTION_FORMAT = 3
"""The first format whose ``choice`` and ``score`` labels may each carry a description."""

WARNINGS_FORMAT = 3
"""The first format with a call's ``warnings``."""

PRECISION_FORMAT = 3
"""The first format whose call item holds the call's ``precision``."""

UNSUPPORTED_FORMAT = 3
"""The first format with the ``unsupported`` cause code."""

HASH_FORMAT = 3
"""The first format whose record document holds the record hash as its ``hash`` member."""


def string_data(value: str) -> str:
    """The exact ``str`` holding a string's data, whatever its class overrides."""
    return str.__str__(value)


def quote(value: str) -> str:
    """A string as its JSON string, as a load failure's message quotes a name or ref;
    a lone surrogate is escaped, so the message is always encodable."""
    data = string_data(value)
    return json.dumps(data, ensure_ascii=not is_scalar_string(data))


def is_scalar_string(value: object) -> bool:
    """A ``str``, or an instance of a subclass, whose data are Unicode scalar values."""
    if not issubclass(type(value), str):  # the type, never ``__class__``, caller code
        return False
    try:
        str.encode(cast(str, value), "utf-8")
    except UnicodeEncodeError:
        return False
    return True


_CONTAINER = object()
_END = object()

MAX_VALUES = 2**20
"""The most values a domain value holds: itself, each member and each key at every depth,
each counted at every place it stands."""

MAX_CODE_POINTS = 2**26
"""The most code points a domain value's strings and keys hold, each counted at every place
it stands."""


class SizeExceeded(ValueError):
    """A value past the domain's size bounds; its message names the bound passed."""


class Count:
    """A walk's running size: values and code points, each counted at every place it stands.
    ``add`` raises ``SizeExceeded`` once either passes its bound, so the walk stops there."""

    __slots__ = ("code_points", "values")

    def __init__(self) -> None:
        self.values = 0
        self.code_points = 0

    def add(self, value: object) -> None:
        """Count ``value`` as one value, and a string's code points by its string data, before
        any of them is read."""
        self.values += 1
        if issubclass(type(value), str):
            self.code_points += str.__len__(cast(str, value))
        self._check()

    def add_string(self, length: int) -> None:
        """Count one string of ``length`` code points before it is built."""
        self.add("")
        self.code_points += length
        self._check()

    def _check(self) -> None:
        # The bounds are read at each count, so a test may set them smaller.
        if self.values > MAX_VALUES:
            raise SizeExceeded(f"the value holds more than {power(MAX_VALUES)} values")
        if self.code_points > MAX_CODE_POINTS:
            raise SizeExceeded(
                f"the value's strings and keys hold more than {power(MAX_CODE_POINTS)} code points"
            )


def power(bound: int) -> str:
    """A bound as a message writes it: ``2^k`` for a power of two, else its digits."""
    k = bound.bit_length() - 1
    return f"2^{k}" if bound == 1 << k else str(bound)


def domain_error(value: object, max_depth: int | None = None) -> str | None:
    """The first reason ``value`` is outside the domain, or ``None`` when it is inside.

    With no ``max_depth`` it is a value: it nests at most ``MAX_DEPTH`` deep and holds at most
    ``MAX_VALUES`` values, itself, each member and each key, and its strings and keys at most
    ``MAX_CODE_POINTS`` code points, counted at every place each stands, so a list standing
    in two places counts twice; the walk stops at the bound, and counts a string's length
    before it reads its code points. Given ``max_depth``, it is a holder of values, an item,
    a document or an ``unhashed`` member: it nests at most that deep and is held to no size
    bound, each value it holds to the size bounds where it stands, by its own test.

    The walk is iterative and counts the depth itself, so a value is refused alike on every
    machine; a container met again on its own path is outside.
    """
    count = Count() if max_depth is None else None
    try:
        return _walk(value, MAX_DEPTH if max_depth is None else max_depth, count)
    except SizeExceeded as exc:
        return str(exc)


def _walk(value: object, max_depth: int, count: Count | None) -> str | None:
    if count is not None:
        count.add(value)
    first = _scalar_error(value)
    if first is not _CONTAINER:
        return cast("str | None", first)
    open_ids = {id(value)}  # the containers on the walk's path
    stack = [_Frame(value)]
    while stack:
        frame = stack[-1]
        member = next(frame.members, _END)
        if member is _END:
            stack.pop()
            open_ids.discard(id(frame.value))
            continue
        if frame.keys is not None:
            key, member = cast("tuple[object, object]", member)
            err = _key_error(key, frame.keys, count)
            if err is not None:
                return err
        if count is not None:
            count.add(member)
        err = _scalar_error(member)
        if err is not _CONTAINER:
            if err is not None:
                return cast(str, err)
            continue
        if id(member) in open_ids:
            return "the value contains itself"
        if len(stack) + 1 > max_depth:
            return f"the value nests more than {max_depth} deep"
        open_ids.add(id(member))
        stack.append(_Frame(member))
    return None


def _key_error(key: object, keys: set[str], count: Count | None) -> str | None:
    """Why an object's key puts it outside the domain: not a string of scalar values, its
    code points read after it is counted as a value, or a second key of the same string
    data."""
    if count is not None:
        count.add(key)
    if not is_scalar_string(key):
        return f"object key {described(key)} is not a string of scalar values"
    data = string_data(cast(str, key))
    if data in keys:
        return f"two object keys have the string data {clipped(data)}"
    keys.add(data)
    return None


_SHOWN = 64


def clipped(data: str) -> str:
    """A string's ``repr`` for a message: whole up to 64 code points, else its first 64 and
    its length, so no message holds a caller's string at full size."""
    data = string_data(data)
    if len(data) <= _SHOWN:
        return repr(data)
    return f"{data[:_SHOWN]!r}... ({len(data)} code points)"


def in_domain(value: object) -> bool:
    """Whether a value is in the type-free value domain, size bounds included."""
    return domain_error(value) is None


def _scalar_error(value: object) -> object:
    """``None`` for a scalar in the domain, ``_CONTAINER`` for an exact list, tuple or dict,
    and otherwise the reason the value is outside."""
    t = type(value)
    if value is None or t is bool:
        return None
    if t is int:
        if -MAX_INTEGER <= cast(int, value) <= MAX_INTEGER:
            return None
        return f"integer {_digits(cast(int, value))} exceeds magnitude 2^53 - 1"
    if t is float:
        if math.isfinite(cast(float, value)):
            return None
        return f"number {value!r} is not finite"
    if issubclass(t, str):
        if is_scalar_string(value):
            return None
        return "string holds a surrogate code point"
    if t is list or t is tuple or t is dict:
        return _CONTAINER
    return f"{type_name(value)} is not in the value domain"


# Python's default limit on the digits it writes for an ``int``. A fixed bound, so a process
# that lifts or lowers its own limit names the same integers by bit length.
_WRITTEN_BOUND = 10**4300
_TYPE_QUALNAME: Any = type.__dict__["__qualname__"]


def _written(n: int) -> str | None:
    """An integer's decimal digits, or ``None`` past the fixed bound or the process's limit."""
    if not -_WRITTEN_BOUND < n < _WRITTEN_BOUND:
        return None
    try:
        return repr(n)
    except ValueError:  # the process set a lower digit limit
        return None


def _digits(n: int) -> str:
    """An integer's decimal digits, or ``of N bits`` when they are not written."""
    text = _written(n)
    return text if text is not None else f"of {n.bit_length()} bits"


def type_name(value: object) -> str:
    """The qualified name of ``value``'s type, read by ``type``'s own descriptor, so neither
    ``__class__`` nor a metaclass's attribute lookup is reached."""
    return string_data(_TYPE_QUALNAME.__get__(type(value)))


def described(value: object) -> str:
    """A caller's value as a message names it, never through the caller's own ``repr``: the
    ``repr`` of ``None``, a ``bool`` or a ``float``; an ``int``'s digits, or ``integer of N
    bits`` when they are not written; a string's data, clipped; any other value its type's name in
    angle brackets."""
    t = type(value)
    if value is None or t is bool or t is float:
        return repr(value)
    if t is int:
        n = cast(int, value)
        text = _written(n)
        return text if text is not None else f"integer of {n.bit_length()} bits"
    if issubclass(t, str):
        return clipped(cast(str, value))
    return f"<{type_name(value)}>"


class _Frame:
    """An open container of a domain walk: its members still to visit, and for an object the
    string data of its keys so far."""

    __slots__ = ("keys", "members", "value")

    def __init__(self, value: object) -> None:
        self.value = value
        self.keys: set[str] | None = None
        self.members: Iterator[object]
        if type(value) is dict:
            self.keys = set()
            self.members = iter(cast("dict[object, object]", value).items())
        else:
            self.members = iter(cast("list[object] | tuple[object, ...]", value))


def shares_a_container(document: object) -> bool:
    """Whether a document given parsed holds a non-empty ``list``, ``tuple`` or ``dict``,
    subclasses included, at two places, as one decoded from bytes never does; a cycle is
    such a container. Strings and empty containers are not tested. The walk is iterative and
    visits each container once, by identity, through the base types' own methods, so it is
    linear in the document whatever it repeats; a reader runs it before any other walk."""
    seen: set[int] = set()
    stack: list[object] = [document]
    while stack:
        value = stack.pop()
        if not _is_container(value):
            continue
        members = list(_repr_members(value))
        if not members:
            continue
        if id(value) in seen:
            return True
        seen.add(id(value))
        stack.extend(members)
    return False


def nesting_depth(value: object, limit: int = MAX_DEPTH) -> int:
    """How deep a parsed document nests, each ``list``, ``tuple`` and ``dict`` instance,
    subclasses included, one level; ``limit + 1`` as soon as the walk finds it deeper than
    ``limit``, so a cycle ends there.

    A reader counts the depth of a document given parsed after ``shares_a_container`` has
    found no container at two places, so the walk, iterative and through the base types'
    own methods, visits each container once."""
    if not _is_container(value):
        return 0
    deepest = 1
    stack: list[tuple[Iterator[object], int]] = [(_members(value), 1)]
    while stack:
        members, depth = stack[-1]
        member = next(members, _END)
        if member is _END:
            stack.pop()
        elif _is_container(member):
            if depth + 1 > limit:
                return limit + 1
            deepest = max(deepest, depth + 1)
            stack.append((_members(member), depth + 1))
    return deepest


UNREPRESENTABLE = "<unrepresentable>"
"""What the engine records for a value whose walk or text raises, or whose walk passes the
value domain's bounds."""

# The types whose ``repr`` runs no code but Python's own.
_EXACT: frozenset[type] = frozenset(
    {list, tuple, dict, set, frozenset, type(None), bool, int, float, complex, str, bytes}
)
# By identity, so no metaclass's hash or comparison runs.
_EXACT_IDS: frozenset[int] = frozenset(id(t) for t in _EXACT)
_TYPE_MODULE: Any = type.__dict__["__module__"]


def _exact(value: object) -> bool:
    return id(type(value)) in _EXACT_IDS


MISSING = object()
"""What ``member_named`` gives for a name an object does not hold."""


def member_named(obj: object, name: str) -> object:
    """The member of an exact ``dict`` whose key is a string of string data ``name``, or
    ``MISSING``; each key is read by its type and string data alone, so no code of a caller's
    key runs, as a lookup's hash and comparison would."""
    if type(obj) is not dict:
        return MISSING
    pairs = cast("Iterable[tuple[object, object]]", _DICT_ITEMS(cast(object, obj)))
    for key, value in pairs:
        if issubclass(type(key), str) and string_data(cast(str, key)) == name:
            return value
    return MISSING


def members_of(obj: object) -> list[object]:
    """The members of an exact ``dict``, read through ``dict``'s own method; else none."""
    if type(obj) is not dict:
        return []
    return list(_DICT_VALUES(cast(object, obj)))


def guarded_repr(value: object) -> str:
    """``value``'s ``repr`` as the engine records it in ``unhashed``, taken after the walk
    (``text_walk``), both inside a guard that catches any ``Exception``: Python's own
    ``repr`` when every value the walk meets is of an exact built-in type, else
    ``<module.qualname>`` of the value's type; ``UNREPRESENTABLE`` past the walk's bounds, on
    a raise, or when the type's names are not exact strings."""
    try:
        exact = text_walk((value,))
        if exact is None:
            return UNREPRESENTABLE
        if exact:
            return string_data(repr(value))
        return type_label(value)
    except Exception:
        return UNREPRESENTABLE


def type_label(value: object) -> str:
    """``<module.qualname>`` of ``value``'s type, each read by ``type``'s own descriptor and
    only when an exact ``str``; else ``UNREPRESENTABLE``."""
    t = type(value)
    module: object = _TYPE_MODULE.__get__(t)
    qualname: object = _TYPE_QUALNAME.__get__(t)
    if type(module) is not str or type(qualname) is not str:
        return UNREPRESENTABLE
    return f"<{module}.{qualname}>"


def representable(value: object) -> bool:
    """Whether ``value`` is within the bounds its recorded ``repr`` is taken under."""
    return text_walk((value,)) is not None


def text_walk(roots: Iterable[object]) -> bool | None:
    """The walk a recorded ``repr`` is taken after, over ``roots`` counted together: ``None``
    past the bounds, else whether every value it met is of an exact built-in type.

    Each ``list``, ``tuple``, ``dict``, ``set`` and ``frozenset`` instance, subclasses
    included, is a level, and its members, a ``dict``'s keys among them, are visited through
    the base type's methods, so no override is reached. Each root, member and key counts as
    a value, at every place it stands; a ``str`` instance adds its string data's code
    points, a ``bytes`` instance its length and an ``int`` its bit length. The walk stops
    past ``MAX_DEPTH`` levels, ``MAX_VALUES`` values or ``MAX_CODE_POINTS`` code points. A
    container met again on its own path ends that path and counts as one value. The walk is
    iterative; the caller guards it against a raise.
    """
    count = Count()
    exact = True
    try:
        for root in roots:
            found = _walk_text(root, count, None)
            if found is None:
                return None
            exact = exact and found
    except SizeExceeded:
        return None
    return exact


def _walk_text(value: object, count: Count, met: list[object] | None) -> bool | None:
    """One root of a text walk; with ``met``, an exception instance it meets is counted as one
    value, put in ``met`` for its own walk and taken as no foreign value."""
    if met is not None and issubclass(type(value), BaseException):
        count.add(value)
        met.append(value)
        return True
    _add_text(count, value)
    exact = _exact(value)
    if not _is_repr_container(value):
        return exact
    open_ids = {id(value)}  # the containers on the walk's path
    stack: list[tuple[object, Iterator[object]]] = [(value, _repr_members(value))]
    while stack:
        container, members = stack[-1]
        member = next(members, _END)
        if member is _END:
            stack.pop()
            open_ids.discard(id(container))
            continue
        if met is not None and issubclass(type(member), BaseException):
            count.add(member)
            met.append(member)
            continue
        _add_text(count, member)
        exact = exact and _exact(member)
        if not _is_repr_container(member) or id(member) in open_ids:
            continue  # a scalar, or a cycle: the path ends here
        if len(stack) + 1 > MAX_DEPTH:
            return None
        open_ids.add(id(member))
        stack.append((member, _repr_members(member)))
    return exact


def _add_text(count: Count, value: object) -> None:
    """Count a value the text walk meets: one value, and a ``bytes`` instance's length or an
    ``int``'s bit length in code points, besides a string's."""
    t = type(value)
    if issubclass(t, bytes):
        count.code_points += bytes.__len__(cast(bytes, value))
    elif issubclass(t, int):
        count.code_points += int.bit_length(cast(int, value))
    count.add(value)


_EXCEPTION_DICT: Any = BaseException.__dict__["__dict__"]
_ARGS: Any = BaseException.__dict__["args"]
_CAUSE: Any = BaseException.__dict__["__cause__"]
_CONTEXT: Any = BaseException.__dict__["__context__"]
_GROUPED: Any = BaseExceptionGroup.__dict__["exceptions"]
_DICT_GET = cast(Callable[[object, str, object], object], dict.get)  # pyright: ignore[reportUnknownMemberType]


def exception_walk(exc: BaseException) -> bool:
    """Whether an exception's text may be taken: one walk, counted together, over the
    exception and every exception it meets, as ``__cause__`` or ``__context__``, in a group's
    ``exceptions`` or inside a container, each once by identity and counted as one value. Of
    each it visits ``args`` and the ``__notes__`` in its instance dictionary, each read
    through ``BaseException``'s or ``BaseExceptionGroup``'s own descriptor; the walk is that of
    ``text_walk``, stays within the bounds, and meets, exceptions aside, only exact built-in
    types. The caller guards it against a raise."""
    count = Count()
    seen: set[int] = set()
    pending: list[object] = [exc]
    exact = True
    try:
        count.add(exc)
        while pending:
            e = pending.pop()
            if id(e) in seen:
                continue
            seen.add(id(e))
            met: list[object] = []
            roots: list[object] = [_ARGS.__get__(e)]
            notes = _DICT_GET(_EXCEPTION_DICT.__get__(e), "__notes__", _END)
            if notes is not _END:
                roots.append(notes)
            for root in roots:
                found = _walk_text(root, count, met)
                if found is None:
                    return False
                exact = exact and found
            linked: list[object] = [_CAUSE.__get__(e), _CONTEXT.__get__(e)]
            if issubclass(type(e), BaseExceptionGroup):
                linked += list(cast("tuple[object, ...]", _GROUPED.__get__(e)))
            for other in linked:
                if other is not None:
                    count.add(other)
                    met.append(other)
            pending += met
    except SizeExceeded:
        return False
    return exact


def _is_container(value: object) -> bool:
    return issubclass(type(value), list | tuple | dict)


def _is_repr_container(value: object) -> bool:
    return issubclass(type(value), list | tuple | dict | set | frozenset)


# The base types' own methods, so a subclass's overrides are not reached.
_Members = Callable[[object], Iterable[object]]
_DICT_VALUES = cast(_Members, dict.values)  # pyright: ignore[reportUnknownMemberType]
_DICT_ITEMS = cast(_Members, dict.items)  # pyright: ignore[reportUnknownMemberType]
_LIST_ITER = cast(_Members, list.__iter__)  # pyright: ignore[reportUnknownMemberType]
_TUPLE_ITER = cast(_Members, tuple.__iter__)  # pyright: ignore[reportUnknownMemberType]
_SET_ITER = cast(_Members, set.__iter__)  # pyright: ignore[reportUnknownMemberType]
_FROZENSET_ITER = cast(_Members, frozenset.__iter__)  # pyright: ignore[reportUnknownMemberType]


def _members(value: object) -> Iterator[object]:
    if issubclass(type(value), dict):
        return iter(_DICT_VALUES(value))
    if issubclass(type(value), list):
        return iter(_LIST_ITER(value))
    return iter(_TUPLE_ITER(value))


def _repr_members(value: object) -> Iterator[object]:
    """A container's members, each ``dict`` key before its value."""
    t = type(value)
    if issubclass(t, dict):
        pairs = cast("Iterable[tuple[object, object]]", _DICT_ITEMS(value))
        return (m for pair in pairs for m in pair)
    if issubclass(t, list):
        return iter(_LIST_ITER(value))
    if issubclass(t, tuple):
        return iter(_TUPLE_ITER(value))
    if issubclass(t, set):
        return iter(_SET_ITER(value))
    return iter(_FROZENSET_ITER(value))
