# pyright: reportPrivateUsage=false
"""The lints' points: each read's fill and the grid order the lints run them in.

Both lints call ``fill`` with the reads they fill, and it raises one ``ValueError`` listing
every slip in ``values`` and every unfilled read before either lint runs a point. A read is
filled from the first of: its grid points, from load's grid and the parameter block through
``_leaf_value``; its declared type's values; its list in ``values``. A record's type fill is
the cross product of its fields' fills, each field filled by the same order under its field
path. A ``q:`` read's points end with ``Unavailable`` ``refused``.

Every point is computed exactly over each boundary's ``Fraction(repr(x))`` and rounded once
to the nearest ``float``. A point is held in two forms: the
record form a report shows and the live form a board or binding is given, ``bytes`` as
``bytes`` and a ``q:`` value as the probabilities a binding returns.
"""

import copy
import itertools
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from jurat._answer import Answer, AnswerKind, answer_document
from jurat._canonical import encode
from jurat._design import LoadedDesign, Question
from jurat._interpret import _marker_document
from jurat._markers import Cause, CauseCode, Unavailable
from jurat._refs import parse_ref
from jurat._typed import typed
from jurat._types import (
    Interval,
    Label,
    Nullable,
    RecordOf,
    Scalar,
    Type,
    TypeFailure,
    read_type,
)
from jurat._values import MAX_INTEGER, described, is_scalar_string, quote, string_data, type_name
from jurat.helpers._numbers import exact

_STATE = "t:state"
_NOUL = ("yes", "no")
_ZERO, _ONE = Fraction(0), Fraction(1)


@dataclass(frozen=True, slots=True)
class Point:
    """One value of a read: its record form and its live form."""

    record: object
    live: object


# ---------------------------------------------------------------------------------------------
# An axis


def axis(
    bounds: Iterable[Fraction], lo: Fraction | None, hi: Fraction | None, *, integer: bool
) -> list[Fraction]:
    """An axis's points, exact and ascending: every boundary, one inside each gap (the
    midpoint, or on an integer axis the least integer above the lower boundary, a gap
    holding none skipped) and one past each end (the type's bound, else the end boundary
    minus or plus 1, on an integer axis the nearest integer beyond it). On an integer axis a
    boundary that is not integral, or of magnitude above 2^53 - 1, is no point, and an end
    past that bound is the bound. A boundary outside the interval is not on the axis; an axis
    so left with no boundary has the interval's two bounds, or on an unbounded integer axis
    the domain's. An axis given no boundary has no points. Rounding is the caller's."""
    given = set(bounds)
    on = {
        b
        for b in given
        if (not integer or (b.denominator == 1 and abs(b) <= MAX_INTEGER))
        and (lo is None or lo <= b)
        and (hi is None or b <= hi)
    }
    ordered = sorted(on)
    if not ordered:
        if given and integer and lo is None and hi is None:
            return [Fraction(-MAX_INTEGER), Fraction(MAX_INTEGER)]
        return sorted({lo, hi}) if given and lo is not None and hi is not None else []
    points = set(ordered)
    for a, b in itertools.pairwise(ordered):
        if integer:
            n = Fraction(math.floor(a) + 1)
            if n < b:
                points.add(n)
        else:
            points.add((a + b) / 2)
    first, last = ordered[0], ordered[-1]
    if lo is None:
        lo = Fraction(max(math.ceil(first) - 1, -MAX_INTEGER)) if integer else first - 1
    if hi is None:
        hi = Fraction(min(math.floor(last) + 1, MAX_INTEGER)) if integer else last + 1
    points |= {lo, hi}
    return sorted(points)


# ---------------------------------------------------------------------------------------------
# A read's type and question


def read_of(loaded: LoadedDesign, read: str) -> Type | Question:
    """A ``q:`` read's question, or a ``t:`` or ``d:`` read's type, a field path's included."""
    p = parse_ref(read)
    base = f"{p.level}:{p.name}"
    if p.level == "q":
        return question(loaded, base)
    t = read_type(loaded._types[base], p.fields, {})
    assert not isinstance(t, TypeFailure)  # load's check 9 passed every declared read
    return t


def question(loaded: LoadedDesign, ref: str) -> Question:
    """The question a source provides at ``ref``; load's checks 3 and 4 passed."""
    provided = loaded._design.sources[loaded._providers[ref]].provides[ref]
    return cast(Question, provided)


def labels_of(q: Question) -> tuple[str, ...]:
    """A question's labels in declared order: ``yes`` and ``no`` for ``noul``."""
    if q.kind is AnswerKind.noul:
        return _NOUL
    return cast("tuple[str, ...]", q.labels)


# ---------------------------------------------------------------------------------------------
# The fill


def fill(loaded: LoadedDesign, reads: Sequence[str], values: object) -> dict[str, list[Point]]:
    """Each read's points in its order, or one ``ValueError`` naming every slip in ``values``
    and every unfilled read, raised before any point runs."""
    return filled(loaded, reads, values)[0]


def filled(
    loaded: LoadedDesign, reads: Sequence[str], values: object
) -> tuple[dict[str, list[Point]], frozenset[str]]:
    """``fill``'s points, and the ``q:`` reads their grid entry filled."""
    filler = _Filler(loaded, values)
    return filler.run(reads), frozenset(filler.on_grid)


class _Filler:
    def __init__(self, loaded: LoadedDesign, values: object) -> None:
        self.loaded = loaded
        self.problems: list[str] = []
        self.values = self.checked_values(values)
        self.consumed: set[str] = set()
        self.unfilled: list[str] = []
        self.typed_paths: set[str] = set()  # paths the grid or a type filled
        self.on_grid: set[str] = set()  # q: reads their grid entry filled

    def checked_values(self, values: object) -> dict[str, list[object]]:
        if values is None:
            return {}
        if not issubclass(type(values), Mapping):  # by its type alone, never __class__
            raise TypeError(f"values is a {type_name(values)}, not a mapping or None")
        out: dict[str, list[object]] = {}
        for key, listed in cast("Mapping[object, object]", values).items():
            if not is_scalar_string(key):
                self.problems.append(f"the values key {described(key)} is not a string")
                continue
            name = string_data(cast(str, key))
            if type(listed) is not list and type(listed) is not tuple:
                raise TypeError(f"values for {quote(name)} is not a list")
            out[name] = list(cast("Sequence[object]", listed))
        return out

    def run(self, reads: Sequence[str]) -> dict[str, list[Point]]:
        fills: dict[str, list[Point]] = {}
        for read in reads:
            points = self.read(read)
            if points is not None:
                fills[read] = points
        for key in self.values:
            if key in self.consumed:
                continue
            if key in self.typed_paths:
                self.problems.append(
                    f"values names {quote(key)}, which its grid points or its type fill"
                )
            else:
                self.problems.append(f"values names {quote(key)}, a read the lint does not fill")
        if self.unfilled:
            names = ", ".join(quote(r) for r in sorted(self.unfilled))
            self.problems.append(f"unfilled: {names}; give each a list in values")
        if self.problems:
            raise ValueError("; ".join(self.problems))
        return fills

    def read(self, read: str) -> list[Point] | None:
        found = read_of(self.loaded, read)
        if isinstance(found, Question):
            self.typed_paths.add(read)
            return [*self.answers(read, found), _refused(read)]
        return self.path(read, found)

    def path(self, path: str, t: Type) -> list[Point] | None:
        """A read or field filled by the first of its grid points, its type, its values."""
        if path in self.loaded._grid:
            self.typed_paths.add(path)
            return self.numeric(path, t)
        unfilled, problems = len(self.unfilled), len(self.problems)
        consumed, typed_paths = set(self.consumed), set(self.typed_paths)
        points = self.by_type(path, t)
        if points is not None:
            self.typed_paths.add(path)
            return points
        if path in self.values:
            # A record whose fields did not all fill is filled whole: its fields' slips go.
            del self.unfilled[unfilled:]
            del self.problems[problems:]
            self.consumed, self.typed_paths = consumed, typed_paths
            return self.listed(path, t)
        if len(self.unfilled) == unfilled:
            self.unfilled.append(path)
        return None

    def by_type(self, path: str, t: Type) -> list[Point] | None:
        if isinstance(t, Nullable):
            inner = self.by_type(path, t.inner)
            return None if inner is None else [*inner, Point(None, None)]
        if isinstance(t, Label):
            return [Point(v, v) for v in t.values]
        if isinstance(t, Scalar) and t.name == "boolean":
            return [Point(False, False), Point(True, True)]
        if isinstance(t, RecordOf):
            fields: list[tuple[str, list[Point]]] = []
            whole = True
            for name, ft in t.fields:  # the order its type spells them
                points = self.path(f"{path}.{name}", ft)
                if points is None:
                    whole = False
                else:
                    fields.append((name, points))
            if not whole:
                return None
            names = [name for name, _ in fields]
            return [
                Point(
                    {n: p.record for n, p in zip(names, combo, strict=True)},
                    {n: p.live for n, p in zip(names, combo, strict=True)},
                )
                for combo in itertools.product(*(points for _, points in fields))
            ]
        return None

    def listed(self, path: str, t: Type) -> list[Point] | None:
        self.consumed.add(path)
        points: list[Point] = []
        seen: set[bytes] = set()
        for value in self.values[path]:
            forms = typed(value, t)
            if forms is None:
                self.problems.append(f"a value for {quote(path)} does not conform to its type")
                continue
            key = encode(forms[0])
            if key in seen:
                self.problems.append(f"values for {quote(path)} lists one value twice")
                continue
            seen.add(key)
            points.append(Point(*forms))
        if not self.values[path]:
            self.problems.append(f"values for {quote(path)} is empty")
        return points

    # --- grid points ---------------------------------------------------------------------

    def bound(self, path: str) -> Fraction:
        return exact(cast(float, self.loaded._leaf_value(path)))

    def numeric(self, path: str, t: Type) -> list[Point]:
        """A numeric ``d:`` or ``t:`` read's axis; load's check 9 made its type numeric."""
        integer = (isinstance(t, Scalar) and t.name == "integer") or (
            isinstance(t, Interval) and t.base == "integer"
        )
        lo = hi = None
        if isinstance(t, Interval):
            lo, hi = exact(t.lo), exact(t.hi)
        bounds = [self.bound(p) for p in cast("Sequence[str]", self.loaded._grid[path])]
        if not integer and lo is None:
            lo, hi = _end(min(bounds), -1), _end(max(bounds), 1)
        points: list[Point] = []
        for x in axis(bounds, lo, hi, integer=integer):
            value: int | float = int(x) if integer else float(x)
            forms = typed(value, t)
            assert forms is not None
            points.append(Point(*forms))
        return _once(points)  # a point that rounds onto another is no point

    def answers(self, ref: str, q: Question) -> list[Point]:
        """A ``q:`` read's answer points, before its marker."""
        labels = labels_of(q)
        entry = cast("Mapping[str, Any] | None", self.loaded._grid.get(ref))
        if entry is not None:
            if len(labels) == 1:  # one distribution, so one answer point
                points = [_answer(q, labels, {labels[0]: 1.0})]
            else:
                points = _once(self.distributions(q, labels, entry))
            if points:
                self.on_grid.add(ref)
                return points
        # Off the grid, or an entry yielding no answer point: each label at 1, in order.
        return [_answer(q, labels, _at(labels, {label: _ONE})) for label in labels]

    def distributions(
        self, q: Question, labels: tuple[str, ...], entry: Mapping[str, Any]
    ) -> list[Point]:
        bars = [self.bound(p) for p in entry["bar"]]
        margins = [self.bound(p) for p in entry["margin"]]
        masses = [
            (tuple(cast("Sequence[str]", m["labels"])), [self.bound(p) for p in m["bounds"]])
            for m in cast("Sequence[Mapping[str, Any]]", entry["mass"])
        ]
        if q.kind is AnswerKind.noul:
            return [_answer(q, labels, d) for d in _noul(bars, margins, masses)]
        thresholds = [
            {k: exact(v) for k, v in cast("dict[str, float]", self.loaded._leaf_value(p)).items()}
            for p in entry["thresholds"]
        ]
        return [_answer(q, labels, d) for d in _choice(labels, bars, thresholds, margins, masses)]


# ---------------------------------------------------------------------------------------------
# Answer points

_Distribution = dict[str, float]


def _rounded(exacts: Mapping[str, Fraction]) -> _Distribution:
    return {label: float(p) for label, p in exacts.items()}


def _at(labels: Sequence[str], placed: Mapping[str, Fraction]) -> _Distribution:
    """``placed`` on its labels and 0 on the rest, in label order."""
    return _rounded({label: placed.get(label, _ZERO) for label in labels})


def _noul(
    bars: list[Fraction],
    margins: list[Fraction],
    masses: list[tuple[tuple[str, ...], list[Fraction]]],
) -> list[_Distribution]:
    """The one p(yes) axis: bar, margin and mass boundaries on it, ascending."""
    on: list[Fraction] = []
    for b in bars:
        on += [b, 1 - b]
    for x in margins:
        on += [(1 + x) / 2, (1 - x) / 2]
    for labels, bounds in masses:
        if set(labels) == {"yes"}:
            on += bounds
        elif set(labels) == {"no"}:
            on += [1 - x for x in bounds]
    return [_rounded({"yes": p, "no": 1 - p}) for p in axis(on, _ZERO, _ONE, integer=False)]


def _choice(
    labels: tuple[str, ...],
    bars: list[Fraction],
    thresholds: list[dict[str, Fraction]],
    margins: list[Fraction],
    masses: list[tuple[tuple[str, ...], list[Fraction]]],
) -> list[_Distribution]:
    """Bar and threshold points, then margin points, label by label in declared order and
    ascending; then mass points, set by set; the caller drops a repeat."""
    n = len(labels)
    out: list[_Distribution] = []
    for label in labels:
        on = bars + [t[label] for t in thresholds if label in t]
        for p in axis(on, _ZERO, _ONE, integer=False):
            rest = (1 - p) / (n - 1)
            out.append(_rounded({k: p if k == label else rest for k in labels}))
    for i, label in enumerate(labels):
        after = labels[(i + 1) % n]
        for x in axis(margins, _ZERO, _ONE, integer=False):
            out.append(_at(labels, {label: (1 + x) / 2, after: (1 - x) / 2}))
    for members, bounds in masses:
        inside = [k for k in labels if k in members]  # declared order
        outside = [k for k in labels if k not in members]
        for x in axis(bounds, _ZERO, _ONE, integer=False):
            if (not outside and x != 1) or (not inside and x != 0):
                continue  # no distribution realises it
            placed = dict.fromkeys(inside, _ZERO)
            if inside:
                placed[inside[0]] = x
            placed.update({k: (1 - x) / len(outside) for k in outside})
            out.append(_at(labels, placed))
    return out


def _end(x: Fraction, step: int) -> Fraction:
    """A number axis's end past ``x``: ``x`` plus ``step``, or the nearest ``float`` beyond
    ``x`` when that rounds onto it; ``x`` itself at the largest ``float``, past which there
    is none."""
    if float(x + step) != float(x):
        return x + step
    beyond = math.nextafter(float(x), step * math.inf)
    return x if math.isinf(beyond) else Fraction(beyond)


def _once(points: list[Point]) -> list[Point]:
    """Each point once, the first kept, by the canonical bytes of its record form."""
    seen: set[bytes] = set()
    out: list[Point] = []
    for p in points:
        key = encode(p.record)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _answer(q: Question, labels: tuple[str, ...], d: _Distribution) -> Point:
    answer = Answer(q.kind, labels, d)
    return Point(answer_document(answer), dict(answer.probabilities))


def _refused(ref: str) -> Point:
    """The marker a binding may return, last in a ``q:`` read's points."""
    marker = Unavailable(Cause(CauseCode.refused, [ref]))
    return Point(_marker_document(marker), marker)


# ---------------------------------------------------------------------------------------------
# Grid order and the report's form


def grid_order(fills: Sequence[list[Point]]) -> Iterator[tuple[Point, ...]]:
    """The cross product of the fills, given in code-point order of their reads, the last
    varying fastest."""
    return itertools.product(*fills)


def fresh(value: object) -> object:
    """A copy no change reaches back from: a board, a binding or a report gets its own."""
    return copy.deepcopy(value)
