"""The combinators, each over the refs it combines.

Each is one ``called`` over all its refs and runs in the choice helpers' order: the raise on
a ``when``'s view, the reads, its log item, the test of each argument,
the test of every determined input, and then its result. A marker result
is a new marker: ``Unsure`` with ``conflict`` naming every determined input, or a marker with
``required`` naming one input; an input's own marker is never passed on. Two values are equal
when their canonical bytes are, and the aggregates are exact over the decimals RFC 8785 writes.
"""

from collections.abc import Sequence
from fractions import Fraction
from typing import cast

from jurat._answer import Answer, answer_document
from jurat._canonical import encode
from jurat._markers import Cause, CauseCode, Unasked, Unavailable, Undetermined, Unsure
from jurat._read_view import ReadView
from jurat._values import MAX_INTEGER, domain_error, quote, string_data
from jurat.helpers._call import Call, called
from jurat.helpers._numbers import exact

_AGGREGATES = ("sum", "min", "max", "count")

# Nothing determined: the input marker of highest precedence.
_PRECEDENCE: tuple[type[Undetermined], ...] = (Unavailable, Unsure, Unasked)


def _bytes(value: object) -> bytes:
    """A value's canonical bytes as the record writes it; an answer as its object."""
    return encode(answer_document(value) if type(value) is Answer else value)


def _refs(combinator: str, refs: Sequence[str]) -> tuple[str, ...]:
    """The refs argument: one or more, no two equal; anything else raises."""
    if not refs:
        raise ValueError(f"{combinator} was given no ref")
    out = tuple(string_data(ref) for ref in refs)  # each was read, so each is a string
    if len(set(out)) != len(out):
        raise ValueError(f"{combinator} was given a ref twice")
    return out


def _required(kind: type[Undetermined], ref: str) -> Undetermined:
    return kind(Cause(CauseCode.required, [ref]))


def _inputs(call: Call, refs: tuple[str, ...]) -> list[tuple[str, object]]:
    return list(zip(refs, call.values, strict=True))


def _determined(inputs: list[tuple[str, object]]) -> list[tuple[str, object]]:
    return [(ref, v) for ref, v in inputs if not isinstance(v, Undetermined)]


def _first_unavailable(inputs: list[tuple[str, object]]) -> str | None:
    return next((ref for ref, v in inputs if type(v) is Unavailable), None)


def _none_determined(call: Call, inputs: list[tuple[str, object]]) -> Undetermined:
    """The marker of highest precedence among the inputs, all markers, with ``required``
    naming the first input of that kind in argument order."""
    for kind in _PRECEDENCE:
        ref = next((ref for ref, v in inputs if type(v) is kind), None)
        if ref is not None:
            call.rested_on([ref])
            return _required(kind, ref)
    # Every read that is not a value is one of the three markers.
    raise AssertionError("an undetermined input of no marker kind")  # pragma: no cover


def _withheld(call: Call, ref: str) -> Unavailable:
    call.rested_on([ref])
    return Unavailable(Cause(CauseCode.required, [ref]))


def _conflict(call: Call, determined: list[tuple[str, object]]) -> Unsure:
    refs = [ref for ref, _ in determined]
    call.rested_on(refs)
    return Unsure(Cause(CauseCode.conflict, refs))


def unique(r: ReadView, *refs: str) -> object:
    """The one determined input's value. Two determined inputs overlap, equal or not, and give
    ``Unsure`` with ``conflict``; else an ``Unavailable`` input gives ``Unavailable`` with
    ``required``."""
    with called(r, "unique", refs) as call:
        refs_ = _refs("unique", refs)
        inputs = _inputs(call, refs_)
        determined = _determined(inputs)
        if not determined:
            return _none_determined(call, inputs)
        if len(determined) > 1:
            return _conflict(call, determined)
        unavailable = _first_unavailable(inputs)
        if unavailable is not None:
            return _withheld(call, unavailable)
        ((ref, value),) = determined
        call.rested_on([ref])
        return value


def agree(r: ReadView, *refs: str) -> object:
    """The value every determined input shares, as the first of them holds it. Disagreement
    gives ``Unsure`` with ``conflict``; else an ``Unavailable`` input gives ``Unavailable``
    with ``required``."""
    with called(r, "agree", refs) as call:
        refs_ = _refs("agree", refs)
        inputs = _inputs(call, refs_)
        determined = _determined(inputs)
        if not determined:
            return _none_determined(call, inputs)
        if len({_bytes(v) for _, v in determined}) > 1:
            return _conflict(call, determined)
        unavailable = _first_unavailable(inputs)
        if unavailable is not None:
            return _withheld(call, unavailable)
        call.rested_on([ref for ref, _ in determined])
        return determined[0][1]


def first(r: ReadView, *refs: str) -> object:
    """The first determined input's value in argument order; an ``Unavailable`` input before
    it gives ``Unavailable`` with ``required``, and one after it does not count."""
    with called(r, "first", refs) as call:
        refs_ = _refs("first", refs)
        inputs = _inputs(call, refs_)
        for ref, value in inputs:
            if type(value) is Unavailable:
                return _withheld(call, ref)
            if not isinstance(value, Undetermined):
                call.rested_on([ref])
                return value
        return _none_determined(call, inputs)


def priority(r: ReadView, order: Sequence[object], *refs: str) -> object:
    """The determined value ranked highest in ``order``; of the inputs holding it, the first
    in argument order wins. An ``Unavailable`` input gives ``Unavailable`` with ``required``
    unless the winner holds ``order``'s first member."""
    with called(r, "priority", refs) as call:
        refs_ = _refs("priority", refs)
        ranks = _order(cast(object, order))
        inputs = _inputs(call, refs_)
        determined = _determined(inputs)
        ranked: list[tuple[int, str, object]] = []
        for ref, value in determined:
            rank = ranks.get(_bytes(value))
            if rank is None:
                raise ValueError(f"priority read {quote(ref)}, whose value order does not rank")
            ranked.append((rank, ref, value))
        if not ranked:
            return _none_determined(call, inputs)
        top = min(rank for rank, _, _ in ranked)
        _, ref, value = next(item for item in ranked if item[0] == top)
        unavailable = _first_unavailable(inputs)
        if unavailable is not None and top != 0:
            return _withheld(call, unavailable)
        call.rested_on([ref])
        return value


def _order(order: object) -> dict[bytes, int]:
    """The ``order`` argument, as each member's canonical bytes to its rank: exactly a
    ``list`` or ``tuple`` of one or more distinct values of the record's domain."""
    if type(order) is not list and type(order) is not tuple:
        raise TypeError("priority's order is not exactly a list or a tuple")
    members = cast("list[object] | tuple[object, ...]", order)
    if not members:
        raise ValueError("priority's order has no member")
    ranks: dict[bytes, int] = {}
    for rank, member in enumerate(members):
        err = domain_error(member)
        if err is not None:
            raise ValueError(f"priority's order holds a value outside the record's domain: {err}")
        key = encode(member)
        if key in ranks:
            raise ValueError("priority's order holds two equal members")
        ranks[key] = rank
    return ranks


def collect(r: ReadView, *refs: str, agg: str | None = None) -> object:
    """The determined values in argument order, or with ``agg`` their ``"sum"``, ``"min"``,
    ``"max"`` or ``"count"``. An ``Unavailable`` input gives ``Unavailable`` with
    ``required``; with no input determined, the result is a marker, never ``[]`` or 0."""
    with called(r, "collect", refs) as call:
        refs_ = _refs("collect", refs)
        if agg is not None and type(agg) is not str:
            raise TypeError("collect's agg is not None or exactly a str")
        if agg is not None and agg not in _AGGREGATES:
            raise ValueError("collect's agg is not None, 'sum', 'min', 'max' or 'count'")
        inputs = _inputs(call, refs_)
        determined = _determined(inputs)
        if agg in ("sum", "min", "max"):
            for ref, value in determined:
                if type(value) is not int and type(value) is not float:
                    raise TypeError(f"collect read {quote(ref)}, which holds no number to {agg}")
        if not determined:
            return _none_determined(call, inputs)
        unavailable = _first_unavailable(inputs)
        if unavailable is not None:
            return _withheld(call, unavailable)
        values = [v for _, v in determined]
        result = _aggregate(agg, values)
        call.rested_on([ref for ref, _ in determined])
        return result


def _aggregate(agg: str | None, values: list[object]) -> object:
    if agg is None:
        return values
    if agg == "count":
        return len(values)
    numbers = cast("list[int | float]", values)
    if agg == "sum":
        if all(type(v) is int for v in numbers):
            whole = sum(cast("list[int]", numbers))
            if abs(whole) > MAX_INTEGER:
                raise ValueError("collect's sum is outside the record's value domain")
            return whole
        total = sum((exact(v) for v in numbers), Fraction(0))
        try:
            return float(total)
        except OverflowError:
            raise ValueError(
                "collect's sum is outside the record's value domain: no finite float holds it"
            ) from None
    exacts = [exact(v) for v in numbers]
    extreme = min(exacts) if agg == "min" else max(exacts)
    return numbers[exacts.index(extreme)]
