"""Type constructors for the type set.

Each constructor returns the one canonical string of the type notation, the form a ``type``
field holds and the design hash covers; a ``type`` field takes that string directly too.
A type argument is such a string, from a constructor or a type definition's name, and a
``str`` subclass is read by its string data. A constructor raises ``ValueError`` for a type
the notation cannot spell, and ``TypeError`` for an argument of the wrong type.

An interval's bounds are domain numbers: each a value whose type is
exactly ``int``, of magnitude at most 2^53 - 1, or exactly ``float``, finite. So
``number(0, 2**60)`` raises ``ValueError``, and ``number(0, 2.0**60)`` is the double.
"""

from jurat import _types
from jurat._values import described as _described
from jurat._values import string_data as _string_data

__all__ = [
    "boolean",
    "bytes_",
    "integer",
    "label",
    "list_",
    "mapping",
    "nullable",
    "number",
    "record",
    "string",
]

string = "string"
boolean = "boolean"
bytes_ = "bytes"


def _interval(base: _types.NumericBase, lo: float | None, hi: float | None) -> str:
    if lo is None and hi is None:
        return base
    if lo is None or hi is None:
        raise ValueError(
            f"{base} takes both bounds or neither, not lo={_described(lo)}, hi={_described(hi)}"
        )
    return _types.render(_types.make_interval(base, lo, hi))


def number(lo: float | None = None, hi: float | None = None) -> str:
    """``number``, or ``number [lo, hi]`` with both bounds domain numbers, ``lo`` at most
    ``hi``."""
    return _interval("number", lo, hi)


def integer(lo: float | None = None, hi: float | None = None) -> str:
    """``integer``, or ``integer [lo, hi]`` with integral domain numbers as bounds, ``lo`` at
    most ``hi``; an integral ``float`` bound is written as its ``int``."""
    return _interval("integer", lo, hi)


def label(*values: str) -> str:
    """A label type of at least one distinct string, listed in code-point order."""
    return _types.render(_types.make_label(values))


def _argument(t: str, *, nullable: bool) -> _types.Type:
    if not issubclass(type(t), str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"a type is its canonical string, not {_described(t)}")
    parsed = _types.parse_type(_string_data(t), nullable=nullable)
    if isinstance(parsed, _types.TypeFailure):
        raise ValueError(parsed.message)
    inner = parsed.inner if isinstance(parsed, _types.Nullable) else parsed
    if isinstance(inner, _types.AnswerType):
        raise ValueError("answer is never declared")
    return parsed


def list_(t: str) -> str:
    """``list<T>``; ``T`` is any type but ``answer`` and does not admit null."""
    return _types.render(_types.make_list(_argument(t, nullable=False)))


def mapping(t: str) -> str:
    """``mapping<string, T>``; ``T`` is any type but ``answer`` and does not admit null."""
    return _types.render(_types.make_mapping(_argument(t, nullable=False)))


def record(**fields: str) -> str:
    """``record{f: T, ...}``, field names in the segment grammar; a field may admit null."""
    built = [(name, _argument(ft, nullable=True)) for name, ft in fields.items()]
    return _types.render(_types.make_record(built))


def nullable(t: str) -> str:
    """``T or null``, for a ``t:`` ref's type, an ``outputs`` type, a record field's type or a
    type definition's body; ``T`` is neither nullable nor ``answer``. Whether a name in ``T``
    resolves to a nullable type is checked at load."""
    return _types.render(_types.make_nullable(_argument(t, nullable=False)))
