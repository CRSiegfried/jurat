"""The choice helpers, each over one read.

Each runs in one order: the raise on a ``when``'s view, the read, its log item, the test of
each argument against its signature, the read's marker returned unchanged, the test of the
value read, the test of each label it was given against the answer, and then the helper's
own work. Its tests and sums are exact over the decimals RFC 8785 writes: a number is taken
as ``Fraction(repr(x))``.
"""

from fractions import Fraction
from typing import cast

from jurat._answer import Answer, ExtractionAnswer
from jurat._markers import Cause, CauseCode, Undetermined, Unsure
from jurat._read_view import ReadView
from jurat._values import quote
from jurat.helpers._call import called
from jurat.helpers._numbers import exact


def _unit(helper: str, name: str, x: object) -> Fraction:
    """A ``number [0, 1]`` argument, exactly an ``int`` or a ``float``, finite and in [0, 1],
    as the decimal RFC 8785 writes; anything else raises."""
    if type(x) is not int and type(x) is not float:
        raise TypeError(f"{helper}'s {name} is not exactly an int or a float")
    # The interval alone: NaN and the infinities fall outside it, and a huge int, which no
    # float holds, is tested without overflow.
    if not 0 <= x <= 1:
        raise ValueError(f"{helper}'s {name} is not a finite number in [0, 1]")
    return exact(x)


def _labels_arg(helper: str, labels: object) -> tuple[str, ...]:
    """A ``labels`` argument: exactly a ``list`` or ``tuple`` of distinct members, each
    exactly a ``str``; anything else raises."""
    if type(labels) is not list and type(labels) is not tuple:
        raise TypeError(f"{helper}'s labels is not exactly a list or a tuple")
    members = cast("list[object] | tuple[object, ...]", labels)
    for label in members:
        if type(label) is not str:
            raise TypeError(f"{helper}'s labels holds a member that is not exactly a str")
    out = cast("tuple[str, ...]", tuple(members))
    if len(set(out)) != len(out):
        raise ValueError(f"{helper}'s labels repeats a label")
    return out


def _thresholds_arg(helper: str, thresholds: object) -> dict[str, Fraction]:
    """A ``thresholds`` argument: exactly a ``dict`` whose keys are each exactly a ``str``
    and whose values are each a ``number [0, 1]``; anything else raises."""
    if type(thresholds) is not dict:
        raise TypeError(f"{helper}'s thresholds is not exactly a dict")
    out: dict[str, Fraction] = {}
    for key, bar in cast("dict[object, object]", thresholds).items():
        if type(key) is not str:
            raise TypeError(f"{helper}'s thresholds holds a key that is not exactly a str")
        out[key] = _unit(helper, "thresholds", bar)
    return out


def _answer(helper: str, ref: str, value: object) -> Answer:
    """The value read, when it is an answer; else it raises."""
    if type(value) is not Answer:
        raise TypeError(f"{helper} read {quote(ref)}, which holds no answer")
    return value


def _lacked(helper: str, answer: Answer, labels: tuple[str, ...] | list[str]) -> None:
    """Raise on the first label given that the answer lacks."""
    for label in labels:
        if label not in answer.probabilities:
            raise ValueError(f"{helper} was given the label {label!r}, which the answer lacks")


def pick_with_margin(r: ReadView, ref: str, bar: float, margin: float) -> str | Undetermined:
    """The top label of the answer at ``ref``, when its probability is at or above ``bar``
    and leads the second-highest, 0 for a one-label answer, by at least ``margin``; else
    ``Unsure`` with ``abstained`` naming ``ref``. A tie at the top abstains, even at
    ``margin`` 0. A marker read is returned unchanged."""
    with called(r, "pick_with_margin", [ref]) as call:
        (value,) = call.values
        bar_ = _unit("pick_with_margin", "bar", bar)
        margin_ = _unit("pick_with_margin", "margin", margin)
        if isinstance(value, Undetermined):
            call.rested_on([ref])
            return value
        answer = _answer("pick_with_margin", ref, value)
        probs = {label: exact(p) for label, p in answer.probabilities.items()}
        top_label = max(probs, key=probs.__getitem__)
        top = probs[top_label]
        second = max((p for label, p in probs.items() if label != top_label), default=Fraction(0))
        call.rested_on([ref])
        if top >= bar_ and top > second and top - second >= margin_:
            return top_label
        return Unsure(Cause(CauseCode.abstained, [ref]))


def pick_with_thresholds(
    r: ReadView, ref: str, thresholds: dict[str, float], margin: float
) -> str | Undetermined:
    """Of the labels of the answer at ``ref`` whose probability is at or above their own
    threshold, the most probable, when it leads the second-highest of every other label,
    qualifying or not, 0 for a one-label answer, by at least ``margin``; else ``Unsure`` with
    ``abstained`` naming ``ref``. A label absent from ``thresholds`` cannot be chosen, so an
    empty ``thresholds`` abstains, and a tie at the top abstains even at ``margin`` 0. A marker
    read is returned unchanged."""
    with called(r, "pick_with_thresholds", [ref]) as call:
        (value,) = call.values
        bars = _thresholds_arg("pick_with_thresholds", thresholds)
        margin_ = _unit("pick_with_thresholds", "margin", margin)
        if isinstance(value, Undetermined):
            call.rested_on([ref])
            return value
        answer = _answer("pick_with_thresholds", ref, value)
        _lacked("pick_with_thresholds", answer, list(bars))
        probs = {label: exact(p) for label, p in answer.probabilities.items()}
        qualifying = [label for label in probs if label in bars and probs[label] >= bars[label]]
        call.rested_on([ref])
        if not qualifying:
            return Unsure(Cause(CauseCode.abstained, [ref]))
        best_label = max(qualifying, key=probs.__getitem__)
        best = probs[best_label]
        second = max((p for label, p in probs.items() if label != best_label), default=Fraction(0))
        if best > second and best - second >= margin_:
            return best_label
        return Unsure(Cause(CauseCode.abstained, [ref]))


def size(r: ReadView, ref: str) -> int | Undetermined:
    """The number of members of the list at ``ref``, a list-valued entry or field or a
    candidate set; a marker read is returned unchanged."""
    with called(r, "size", [ref]) as call:
        (value,) = call.values
        if isinstance(value, Undetermined):
            call.rested_on([ref])
            return value
        if type(value) is not list and type(value) is not tuple:
            raise TypeError(f"size read {quote(ref)}, which holds no list")
        count = len(cast("list[object] | tuple[object, ...]", value))
        call.rested_on([ref])
        return count


def mass(r: ReadView, ref: str, labels: list[str] | tuple[str, ...]) -> float | Undetermined:
    """The sum of the probabilities of the answer at ``ref`` over ``labels``, exact and then
    rounded to the nearest ``float``, or 1.0 when the exact sum exceeds 1; 0.0 over no label.
    A marker read is returned unchanged."""
    with called(r, "mass", [ref]) as call:
        (value,) = call.values
        labels_ = _labels_arg("mass", labels)
        if isinstance(value, Undetermined):
            call.rested_on([ref])
            return value
        answer = _answer("mass", ref, value)
        _lacked("mass", answer, labels_)
        total = sum((exact(answer.probabilities[label]) for label in labels_), Fraction(0))
        call.rested_on([ref])
        return 1.0 if total > 1 else float(total)


def extract(r: ReadView, ref: str, null_bar: float) -> object:
    """The ``value`` of the extraction answer at ``ref``, a list when its question asks for a
    list; ``Unsure`` with ``abstained`` naming ``ref`` when ``value`` is ``None``, or when
    ``p_null`` is not ``None`` and at or above ``null_bar``. A ``None`` ``p_null`` does not
    abstain, and ``guess`` is never returned. A marker read is returned unchanged."""
    with called(r, "extract", [ref]) as call:
        (value,) = call.values
        bar = _unit("extract", "null_bar", null_bar)
        if isinstance(value, Undetermined):
            call.rested_on([ref])
            return value
        if type(value) is not ExtractionAnswer:
            raise TypeError(f"extract read {quote(ref)}, which holds no extraction answer")
        call.rested_on([ref])
        if value.value is None or (value.p_null is not None and exact(value.p_null) >= bar):
            return Unsure(Cause(CauseCode.abstained, [ref]))
        return list(value.value) if value.is_list else value.value
