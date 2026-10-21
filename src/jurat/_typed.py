"""The typed write: a value of a declared type in its record form and its live form, as
interpret writes a ``t:`` or ``d:`` value and decide the state entry.

The record form is what the ledger is handed: undeclared fields dropped at every depth,
``bytes`` as unpadded base64url. The live form is what a component reads: ``bytes`` as Python
``bytes``. Both are built afresh, so no change to the value given reaches either, and an
``int`` under ``number`` is a ``float`` in both.

``decoded`` goes the other way for replay: a record's canonical value made typed by the given
design's declared type, never refusing, so the typed write that follows judges it as a live
turn's.
"""

from collections.abc import Sequence
from typing import cast

from jurat._types import (
    Interval,
    ListOf,
    MappingOf,
    Nullable,
    RecordOf,
    Scalar,
    Type,
    b64url_decode,
    b64url_encode,
    value_conforms,
)
from jurat._values import Count, SizeExceeded, domain_error, is_scalar_string, string_data


def is_a(value: object, cls: type | tuple[type, ...]) -> bool:
    """``isinstance`` by the value's type alone: ``isinstance`` also reads ``__class__``,
    which a caller's object may make code that raises, outside the protected region."""
    return issubclass(type(value), cls)


def typed(value: object, t: Type) -> tuple[object, object] | None:
    """A value of declared type ``t`` in its record form and its live form; ``None`` when it
    does not conform or lies outside the value domain. ``t`` is resolved at every depth."""
    try:
        found = forms(value, t, Count())
    except SizeExceeded:
        return None
    if found is None or domain_error(found[0]) is not None:
        return None
    return found


def forms(value: object, t: Type, count: Count) -> tuple[object, object] | None:
    """Both forms of ``value`` under ``t``, or ``None`` when it does not conform; the value
    domain is ``typed``'s to check. The record form is counted as it is built, each value and
    key at every place it stands, a string's length before its code points are read, and the
    build stops with ``SizeExceeded`` once it passes the size bounds."""
    # The recursion follows the type, which load bounds at the domain's depth.
    if isinstance(t, Nullable):
        if value is None:
            count.add(None)
            return None, None
        return forms(value, t.inner, count)
    if isinstance(t, RecordOf):
        if type(value) is not dict:
            return None
        count.add(cast(object, value))
        declared = dict(t.fields)
        found: dict[str, object] = {}
        for key, member in cast("dict[object, object]", value).items():
            name = string_data(cast(str, key)) if is_a(key, str) else None
            if name is None or name not in declared:
                continue  # an undeclared field is dropped
            if name in found:
                return None
            found[name] = member
        document: dict[str, object] = {}
        live: dict[str, object] = {}
        for name, ft in t.fields:
            if name not in found:
                return None
            count.add(name)
            pair = forms(found[name], ft, count)
            if pair is None:
                return None
            document[name], live[name] = pair
        return document, live
    if isinstance(t, MappingOf):
        if type(value) is not dict:
            return None
        count.add(cast(object, value))
        document = {}
        live = {}
        for key, member in cast("dict[object, object]", value).items():
            count.add(key)
            if not is_scalar_string(key):
                return None
            name = string_data(cast(str, key))
            pair = forms(member, t.value, count)
            if pair is None or name in document:
                return None
            document[name], live[name] = pair
        return document, live
    if isinstance(t, ListOf):
        if type(value) is not list and type(value) is not tuple:
            return None
        count.add(cast(object, value))
        pairs: list[tuple[object, object]] = []
        for member in cast("Sequence[object]", value):
            pair = forms(member, t.item, count)
            if pair is None:
                return None
            pairs.append(pair)
        return [d for d, _ in pairs], [v for _, v in pairs]
    if isinstance(t, Scalar) and t.name == "bytes":
        if type(value) is not bytes:
            return None
        # Its unpadded base64url length counted before the text is built.
        count.add_string((4 * bytes.__len__(value) + 2) // 3)
        return b64url_encode(value), value
    count.add(value)
    if not value_conforms(value, t, {}, form="live"):
        return None
    plain = string_data(cast(str, value)) if is_a(value, str) else value
    if type(plain) is int and is_number(t):
        # The stored copy and the board hold a float, as replay makes an integral number;
        # the bytes are the same. Exact: the domain bounds an int by 2^53 - 1.
        plain = float(plain)
    return plain, plain


def is_number(t: Type) -> bool:
    """Whether ``t`` is ``number``, bounded or not."""
    return (isinstance(t, Scalar) and t.name == "number") or (
        isinstance(t, Interval) and t.base == "number"
    )


def decoded(value: object, t: Type) -> object:
    """A record's canonical value as typed by ``t``, built afresh: under ``bytes`` an unpadded
    base64url string becomes ``bytes``, and under ``number`` an integral number a ``float``,
    at every depth. Nothing is refused: a value that does
    not fit ``t`` is copied as it is, so the typed write finds it does not conform. ``t`` is
    resolved at every depth."""
    if isinstance(t, Nullable):
        return None if value is None else decoded(value, t.inner)
    if isinstance(t, RecordOf):
        if type(value) is not dict:
            return _copied(value)
        declared = dict(t.fields)
        members = cast("dict[str, object]", value)
        return {
            k: decoded(v, declared[k]) if k in declared else _copied(v) for k, v in members.items()
        }
    if isinstance(t, MappingOf):
        if type(value) is not dict:
            return _copied(value)
        return {k: decoded(v, t.value) for k, v in cast("dict[str, object]", value).items()}
    if isinstance(t, ListOf):
        if type(value) is not list:
            return _copied(value)
        return [decoded(v, t.item) for v in cast("list[object]", value)]
    if isinstance(t, Scalar) and t.name == "bytes":
        raw = b64url_decode(value) if type(value) is str else None
        return value if raw is None else raw
    if type(value) is int and is_number(t):
        return float(value)  # exact: the domain bounds an int by 2^53 - 1
    return _copied(value)


def _copied(value: object) -> object:
    """A fresh copy of plain canonical data, so no container is shared with the record's."""
    if type(value) is dict:
        return {k: _copied(v) for k, v in cast("dict[str, object]", value).items()}
    if type(value) is list:
        return [_copied(v) for v in cast("list[object]", value)]
    return value
