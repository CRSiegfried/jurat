"""The ref grammar: a level prefix, a name, an optional grounded bracket and a field path.

A name is one or more segments joined by ``/``; a segment is a lowercase ASCII letter then
lowercase letters, digits or underscores. A field path is ``.`` and a segment, repeated. A
grounded bracket is ``[``, a JSON string exactly as RFC 8785 serializes it, then ``]``.
Cause refs such as ``rule:N``, ``step:N``, ``policy`` and ``codec`` are not refs.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

from jurat._canonical import encode
from jurat._values import string_data

Level = Literal["q", "t", "d"]
_LEVELS: frozenset[str] = frozenset({"q", "t", "d"})

_SEGMENT = r"[a-z][a-z0-9_]*"
_SEGMENT_RE = re.compile(_SEGMENT)
_NAME_RE = re.compile(rf"{_SEGMENT}(?:/{_SEGMENT})*")
_FIELDS_RE = re.compile(rf"(?:\.{_SEGMENT})*")
_PATH_RE = re.compile(rf"{_SEGMENT}(?:\.{_SEGMENT})*")
_JSON = json.JSONDecoder(strict=True)


class _RefError(ValueError):
    """A string that is not a ref under the grammar; load reports it under check 7."""


@dataclass(frozen=True, slots=True)
class _ParsedRef:
    level: Level
    name: str
    candidate: str | None
    """The grounded bracket's string, decoded; ``None`` for a declared ref."""
    fields: tuple[str, ...]


def parse_ref(ref: str) -> _ParsedRef:
    """Parse a ref, or raise ``_RefError`` naming what is wrong.

    A ``str`` subclass is read by its string data, which no override reaches.
    """
    if not isinstance(ref, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise _RefError(f"{ref!r} is not a string")
    ref = string_data(ref)
    level, colon = ref[:1], ref[1:2]
    if level not in _LEVELS or colon != ":":
        raise _RefError(f"{ref!r} does not start with q:, t: or d:")
    m = _NAME_RE.match(ref, 2)
    if m is None:
        raise _RefError(f"{ref!r} has no name after its prefix")
    name = m.group()
    pos = m.end()
    candidate: str | None = None
    if ref[pos : pos + 1] == "[":
        candidate, pos = _bracket(ref, pos + 1)
    f = _FIELDS_RE.match(ref, pos)
    assert f is not None  # the pattern matches the empty string
    if f.end() != len(ref):
        raise _RefError(f"{ref!r} has unexpected text at position {f.end()}")
    fields = tuple(f.group()[1:].split(".")) if f.group() else ()
    return _ParsedRef(cast(Level, level), name, candidate, fields)


def _bracket(ref: str, start: int) -> tuple[str, int]:
    """Read the bracket's JSON string from ``start``; return it and the index after ``]``.

    Only a JSON string opens with a quote, so anything else is refused before the decoder
    runs; a string decodes without recursion however the ref nests after it.
    """
    if ref[start : start + 1] != '"':
        raise _RefError(f"{ref!r} has a bracket that is not a JSON string")
    try:
        value, end = _JSON.raw_decode(ref, start)
    except ValueError as exc:  # an unterminated string or a bad escape
        raise _RefError(f"{ref!r} has a bracket that is not a JSON string") from exc
    assert type(value) is str  # a JSON value opening with a quote is a string
    if ref[end : end + 1] != "]":
        raise _RefError(f"{ref!r} has no ] after its bracketed string")
    try:
        canonical = encode(value).decode("utf-8")
    except ValueError as exc:  # the encoder refuses a string holding a surrogate
        raise _RefError(f"{ref!r} has a bracketed string outside the value domain") from exc
    if canonical != ref[start:end]:
        raise _RefError(f"{ref!r} has a bracketed string not as RFC 8785 serializes it")
    return value, end + 1


def is_ref(value: object) -> bool:
    """Whether ``value`` is a string that is a ref under the grammar."""
    if not isinstance(value, str):
        return False
    try:
        parse_ref(value)
    except _RefError:
        return False
    return True


def declared_ref(ref: str) -> str:
    """The ref with any grounded bracket removed and its field path kept."""
    p = parse_ref(ref)
    return f"{p.level}:{p.name}" + "".join("." + f for f in p.fields)


def is_declared_name(value: object) -> bool:
    """Whether ``value`` is in the segment grammar: segments joined by ``/``, no prefix."""
    return isinstance(value, str) and _NAME_RE.fullmatch(value) is not None


def is_field_path(value: object) -> bool:
    """Whether ``value`` is a binding's field path: segments joined by ``.``, with no leading
    dot. A ``str`` subclass is read by its string data."""
    return isinstance(value, str) and _PATH_RE.fullmatch(string_data(value)) is not None


def is_segment(value: object) -> bool:
    """Whether ``value`` is one segment: no ``/``."""
    return isinstance(value, str) and _SEGMENT_RE.fullmatch(value) is not None


def is_path_ancestor(node: str, path: str) -> bool:
    """Whether parameter path ``node`` is a proper ``/``-prefix of ``path``."""
    return path.startswith(node + "/")


def code_point_sorted(items: Iterable[str]) -> list[str]:
    """Sort by Unicode code point, as ids and refs sort; not the UTF-16 key order."""
    return sorted(items)
