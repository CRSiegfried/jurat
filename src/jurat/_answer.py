"""The answer to a ``q:`` ref: of a probability kind, a kind, its labels in order and a
probability for each; of the extraction kind, a typed value read from a text, with the
probability that the input holds none, the model's reading whatever it answered, its
confidence and the span it read."""

import enum
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from jurat._values import (
    MAX_INTEGER,
    described,
    guarded_repr,
    is_scalar_string,
    string_data,
    type_name,
)

_NOUL_LABELS = ("yes", "no")
# The band an answer's exact sum is flagged outside of, both ends inside it.
_SUM_LOW = Fraction(95, 100)
_SUM_HIGH = Fraction(105, 100)


class AnswerKind(enum.StrEnum):
    """The answer kinds."""

    noul = "noul"
    choice = "choice"
    score = "score"
    extraction = "extraction"


_KINDS: dict[str, AnswerKind] = {k.value: k for k in AnswerKind}
PROBABILITY_KINDS = frozenset({AnswerKind.noul, AnswerKind.choice, AnswerKind.score})
"""The kinds whose answer is a probability for each label."""


@dataclass(frozen=True, slots=True, init=False)
class Answer:
    """A probability for each of the labels its question was asked with.

    ``kind`` is an ``AnswerKind`` member or its string, stored as the member. ``labels`` is
    the question's order (for ``score``, lowest level first), at least one, and
    ``probabilities`` is exactly a ``dict`` mapping exactly those labels to values whose type
    is exactly ``int`` or ``float``, a ``bool`` not, finite and in [0, 1]. The constructor
    checks each value alone and raises ``ValueError``; no sum is required, and the values are
    held as given.

    A label, and a key of ``probabilities``, is a ``str`` or an instance of a subclass of
    ``str`` whose data are Unicode scalar values; it is read and compared by its string data,
    which no override reaches, and stored as a plain ``str``, as the value domain takes it.
    ``probabilities`` is stored as a new plain ``dict`` in label order. An ``Answer`` is not
    hashable, since it holds a ``dict``.
    """

    kind: AnswerKind
    labels: tuple[str, ...]
    probabilities: Mapping[str, float]

    def __init__(
        self,
        kind: AnswerKind | str,
        labels: Sequence[str],
        probabilities: Mapping[str, float],
    ) -> None:
        # Refused by type before the lookup, which would hash a caller's object; a string, the
        # members included, is looked up by its string data.
        member = (
            _KINDS.get(string_data(kind))
            if issubclass(type(kind), str)  # pyright: ignore[reportUnnecessaryIsInstance]
            else None
        )
        if member is None or member not in PROBABILITY_KINDS:
            raise ValueError(f"answer kind {guarded_repr(kind)} is not noul, choice or score")
        labels_ = _labels(cast(object, labels))
        if member is AnswerKind.noul and labels_ != _NOUL_LABELS:
            raise ValueError(f"a noul answer's labels are ('yes', 'no'), not {labels_!r}")
        object.__setattr__(self, "kind", member)
        object.__setattr__(self, "labels", labels_)
        object.__setattr__(self, "probabilities", _probabilities(probabilities, labels_))


def answer_document(answer: Answer) -> dict[str, object]:
    """An answer as the record writes it: an object with ``kind``, ``labels`` and
    ``probabilities``."""
    return {
        "kind": answer.kind.value,
        "labels": list(answer.labels),
        "probabilities": dict(answer.probabilities),
    }


def _labels(labels: object) -> tuple[str, ...]:
    # By type alone, so no ``__class__`` of the caller's is read.
    if issubclass(type(labels), str) or not issubclass(type(labels), Sequence):
        raise ValueError(f"labels {guarded_repr(labels)} are not a sequence of strings")
    out: list[str] = []
    for label in cast("Sequence[object]", labels):
        if not is_scalar_string(label):
            raise ValueError(
                f"label {guarded_repr(label)} is not a string of Unicode scalar values"
            )
        out.append(string_data(cast(str, label)))
    if not out:
        raise ValueError("an answer has at least one label, as every question does")
    if len(set(out)) != len(out):
        raise ValueError(f"labels {out!r} repeat a label")
    return tuple(out)


def _probabilities(probabilities: object, labels: tuple[str, ...]) -> dict[str, float]:
    if type(probabilities) is not dict:
        raise ValueError(f"probabilities of type {type_name(probabilities)} are not exactly a dict")
    given: dict[str, object] = {}
    for key, p in cast("dict[object, object]", probabilities).items():
        if not is_scalar_string(key):
            raise ValueError(
                f"probability key {guarded_repr(key)} is not a string of Unicode scalar values"
            )
        data = string_data(cast(str, key))
        if data in given:
            raise ValueError(f"two probability keys have the string data {data!r}")
        given[data] = p
    missing = [label for label in labels if label not in given]
    if missing:
        raise ValueError(f"no probability for label(s) {missing!r}")
    known = frozenset(labels)
    extra = [key for key in given if key not in known]
    if extra:
        raise ValueError(f"probability for key(s) {extra!r} not among the labels")
    out: dict[str, float] = {}
    for label in labels:
        p = given[label]
        # Exact types, as the value domain admits: a bool is not a number. An int of any
        # size compares exactly; only a float can be non-finite.
        if not (type(p) is int or type(p) is float):
            raise ValueError(f"probability of type {type_name(p)} for {label!r} is not a number")
        number = cast(float, p)
        if (type(p) is float and not math.isfinite(number)) or not 0 <= number <= 1:
            raise ValueError(f"probability {described(p)} for {label!r} is not in [0, 1]")
        out[label] = number
    return out


def exact_sum(answer: Answer) -> Fraction:
    """The sum of an answer's probabilities, exact over the decimal each reads back as,
    ``Fraction(repr(p))``, as the ``mass`` helper sums."""
    # Exactly as ``helpers._numbers.exact`` sums for ``mass``; kept here so the core imports
    # nothing from the helpers.
    return sum((Fraction(repr(p)) for p in answer.probabilities.values()), Fraction(0))


def off_sum(answer: Answer) -> float | None:
    """An answer's exact sum rounded once to a ``float`` when it lies outside [0.95, 1.05],
    each end inside; else ``None``. The band is tested on the exact sum, so a sum just outside
    it may round to 0.95 or 1.05 exactly, as 1 + 0.05 + 1e-17 does. The flag changes nothing
    about the answer."""
    total = exact_sum(answer)
    return None if _SUM_LOW <= total <= _SUM_HIGH else float(total)


# ---------------------------------------------------------------------------------------------
# The extraction kind


class ValueType(enum.StrEnum):
    """The value types of an extraction question."""

    text = "text"
    number = "number"
    date = "date"
    time = "time"
    datetime = "datetime"
    duration = "duration"


_VALUE_TYPES: dict[str, ValueType] = {v.value: v for v in ValueType}
EXTRACTION_MEMBERS = ("value", "guess", "p_null", "confidence", "span")
"""The members a binding returns for an extraction question, in the order a reader takes them."""

# The temporal grammars, jurat's own: ASCII digits only, ``T`` and ``Z`` uppercase only. A
# date's day is tested against its month apart, leap years counted.
_DATE = r"([0-9]{4})-([0-9]{2})-([0-9]{2})"
_CLOCK = r"([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.[0-9]+)?(?:Z|[+-]([0-9]{2}):([0-9]{2}))?"
_DATE_RE = re.compile(_DATE)
_TIME_RE = re.compile(_CLOCK)
_DATETIME_RE = re.compile(f"{_DATE}T{_CLOCK}")
_DURATION_RE = re.compile(
    r"P(?:[0-9]+Y)?(?:[0-9]+M)?(?:[0-9]+D)?(?:T(?:[0-9]+H)?(?:[0-9]+M)?(?:[0-9]+S)?)?"
)
_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _date_ok(year: str, month: str, day: str) -> bool:
    m, d = int(month), int(day)
    if not 1 <= m <= 12:
        return False
    last = 29 if m == 2 and _leap(int(year)) else _MONTH_DAYS[m - 1]
    return 1 <= d <= last


def _clock_ok(groups: Sequence[str | None]) -> bool:
    hour, minute, second, off_hour, off_minute = groups
    if not (int(cast(str, hour)) <= 23 and int(cast(str, minute)) <= 59):
        return False
    if int(cast(str, second)) > 59:
        return False
    return off_hour is None or (int(off_hour) <= 23 and int(cast(str, off_minute)) <= 59)


def temporal_ok(value_type: ValueType, text: str) -> bool:
    """Whether ``text`` is a value of the temporal ``value_type`` by jurat's own grammar, the
    same on every Python; ``text`` is a plain ``str``."""
    if value_type is ValueType.date:
        found = _DATE_RE.fullmatch(text)
        return found is not None and _date_ok(*cast("tuple[str, str, str]", found.groups()))
    if value_type is ValueType.time:
        found = _TIME_RE.fullmatch(text)
        return found is not None and _clock_ok(found.groups())
    if value_type is ValueType.datetime:
        found = _DATETIME_RE.fullmatch(text)
        if found is None:
            return False
        groups = found.groups()
        return _date_ok(*cast("tuple[str, str, str]", groups[:3])) and _clock_ok(groups[3:])
    assert value_type is ValueType.duration
    # At least one part in all, and at least one after a T.
    return _DURATION_RE.fullmatch(text) is not None and text != "P" and not text.endswith("T")


def _item(value_type: ValueType, value: object, member: str) -> object:
    """One value of ``value_type`` as the answer holds it, a number as a ``float``; anything
    else raises ``ValueError``."""
    if value_type is ValueType.number:
        if type(value) is int:
            if abs(value) > MAX_INTEGER:
                raise ValueError(f"{member} {described(value)} is past 2^53 - 1 in magnitude")
            return float(value)
        if type(value) is float and math.isfinite(value):
            return value
        raise ValueError(f"{member} {guarded_repr(value)} is not a finite number")
    if not is_scalar_string(value):
        raise ValueError(f"{member} {guarded_repr(value)} is not a string of Unicode scalar values")
    text = string_data(cast(str, value))
    if value_type is not ValueType.text and not temporal_ok(value_type, text):
        raise ValueError(f"{member} {text!r} is not a {value_type.value} by jurat's grammar")
    return text


def _typed_member(value_type: ValueType, is_list: bool, value: object, member: str) -> object:
    """``value`` or ``guess``: ``None``, a value of the type, or when ``is_list`` exactly a
    ``list`` of such values with no ``None``, held as a tuple."""
    if value is None:
        return None
    if not is_list:
        return _item(value_type, value, member)
    if type(value) is not list:
        raise ValueError(f"{member} of type {type_name(value)} is not exactly a list")
    out: list[object] = []
    for i, item in enumerate(cast("list[object]", value)):
        if item is None:
            raise ValueError(f"{member}[{i}] is None; a list holds no None item")
        out.append(_item(value_type, item, f"{member}[{i}]"))
    return tuple(out)


def _probability(value: object, member: str) -> float | None:
    if value is None:
        return None
    if not (type(value) is int or type(value) is float):
        raise ValueError(f"{member} of type {type_name(value)} is not a number")
    number = cast(float, value)
    if (type(value) is float and not math.isfinite(number)) or not 0 <= number <= 1:
        raise ValueError(f"{member} {described(value)} is not in [0, 1]")
    return number


def _pair(value: object, member: str) -> tuple[int, int]:
    if type(value) is not list or len(cast("list[object]", value)) != 2:
        raise ValueError(f"{member} is not exactly a list of two integers")
    start, end = cast("list[object]", value)
    if type(start) is not int or type(end) is not int:
        raise ValueError(f"{member} holds a bound that is not exactly an int")
    if not 0 <= start <= end <= MAX_INTEGER:
        raise ValueError(
            f"{member} [{described(start)}, {described(end)}] is not 0 <= start <= end"
        )
    return start, end


def _span(span: object, located: object, is_list: bool) -> Any:
    """``span``: ``None`` exactly when the member it locates is; for a scalar one pair, for a
    list one pair per item, held as tuples. Its bound against the text is interpret's."""
    if span is None:
        return None
    if located is None:
        raise ValueError("span locates a value or guess, and both are None")
    if not is_list:
        return _pair(span, "span")
    if type(span) is not list:
        raise ValueError(f"span of type {type_name(span)} is not exactly a list")
    pairs = cast("list[object]", span)
    if len(pairs) != len(cast("tuple[object, ...]", located)):
        raise ValueError("span holds a pair count other than the list it locates holds items")
    return tuple(_pair(p, f"span[{i}]") for i, p in enumerate(pairs))


@dataclass(frozen=True, slots=True, init=False)
class ExtractionAnswer:
    """A typed value a decision model read from a text, as its question asked.

    ``value_type`` is a ``ValueType`` member or its string and ``is_list`` exactly a ``bool``,
    both copied from the question. ``value`` and ``guess`` are each ``None`` or a value of the
    type, or when ``is_list`` exactly a ``list`` of such values with no ``None`` item, held as a
    tuple; an empty list is a value distinct from ``None``. A ``text`` value is a string; a
    ``number`` exactly an ``int`` of magnitude at most 2^53 - 1 or a finite ``float``, never a
    ``bool``, held as a ``float``; a ``date``, ``time``, ``datetime`` or ``duration`` a string of
    jurat's own grammar, kept as given. ``p_null`` and ``confidence`` are each ``None`` or
    exactly an ``int`` or ``float``, finite and in [0, 1], kept as given.

    ``span`` is ``None``, or for a scalar exactly a ``list`` ``[start, end]`` of exact ``int``
    values with 0 <= start <= end, ``end`` exclusive, or for a list one such pair per item. It
    locates ``value``, or ``guess`` when ``value`` is ``None``, and is ``None`` when that member
    is. Pairs are held as tuples. The constructor checks every member but a span's bound
    against its text, which interpret tests at the write, and raises ``ValueError``.
    """

    kind: AnswerKind
    value_type: ValueType
    is_list: bool
    value: Any
    guess: Any
    p_null: float | None
    confidence: float | None
    span: Any

    def __init__(
        self,
        value_type: ValueType | str,
        is_list: bool,
        value: object,
        guess: object,
        p_null: float | None,
        confidence: float | None,
        span: object,
    ) -> None:
        member = (
            _VALUE_TYPES.get(string_data(value_type))
            if issubclass(type(value_type), str)  # pyright: ignore[reportUnnecessaryIsInstance]
            else None
        )
        if member is None:
            raise ValueError(f"value type {guarded_repr(value_type)} is not one of the six")
        if type(is_list) is not bool:
            raise ValueError(f"is_list of type {type_name(is_list)} is not exactly a bool")
        value_ = _typed_member(member, is_list, value, "value")
        guess_ = _typed_member(member, is_list, guess, "guess")
        located = guess_ if value_ is None else value_
        object.__setattr__(self, "kind", AnswerKind.extraction)
        object.__setattr__(self, "value_type", member)
        object.__setattr__(self, "is_list", is_list)
        object.__setattr__(self, "value", value_)
        object.__setattr__(self, "guess", guess_)
        object.__setattr__(self, "p_null", _probability(p_null, "p_null"))
        object.__setattr__(self, "confidence", _probability(confidence, "confidence"))
        object.__setattr__(self, "span", _span(span, located, is_list))


def _listed(member: object) -> object:
    if type(member) is tuple:
        return [_listed(m) for m in cast("tuple[object, ...]", member)]
    return member


def extraction_document(answer: ExtractionAnswer) -> dict[str, object]:
    """An extraction answer as the record writes it: its eight members, tuples as lists."""
    return {
        "kind": answer.kind.value,
        "value_type": answer.value_type.value,
        "is_list": answer.is_list,
        "value": _listed(answer.value),
        "guess": _listed(answer.guess),
        "p_null": answer.p_null,
        "confidence": answer.confidence,
        "span": _listed(answer.span),
    }


def binding_form(answer: ExtractionAnswer) -> dict[str, object]:
    """An extraction answer's five members as a binding returns them: tuples as lists."""
    document = extraction_document(answer)
    return {member: document[member] for member in EXTRACTION_MEMBERS}


def span_bound(answer: ExtractionAnswer) -> int:
    """The largest ``end`` the answer's span holds, 0 when it has none."""
    span = answer.span
    if span is None:
        return 0
    if not answer.is_list:
        return cast("tuple[int, int]", span)[1]
    return max((end for _, end in cast("tuple[tuple[int, int], ...]", span)), default=0)
