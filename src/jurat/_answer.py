"""The answer to a ``q:`` ref: a kind, its labels in order, and a probability for each."""

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from jurat._values import described, guarded_repr, is_scalar_string, string_data, type_name

_NOUL_LABELS = ("yes", "no")
# The band an answer's exact sum is flagged outside of, both ends inside it.
_SUM_LOW = Fraction(95, 100)
_SUM_HIGH = Fraction(105, 100)


class AnswerKind(enum.StrEnum):
    """The answer kinds."""

    noul = "noul"
    choice = "choice"
    score = "score"


_KINDS: dict[str, AnswerKind] = {k.value: k for k in AnswerKind}


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
        if member is None:
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
