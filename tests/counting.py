"""A count of the values every size-bounded walk counts, so a test can see a walk stop at the
bound rather than time it."""

import contextlib
from collections.abc import Generator
from typing import Protocol

import jurat._values
from jurat._values import Count


class SmallBounds(Protocol):
    """The ``small_bounds`` fixture: sets the size bounds, and load's type-size bound when
    ``type_size`` is given."""

    def __call__(
        self, values: int = ..., code_points: int = ..., type_size: int | None = None
    ) -> None: ...


def bound() -> int:
    """The values a walk counts before it stops past the bound now set: one past it."""
    return jurat._values.MAX_VALUES + 1


@contextlib.contextmanager
def counted() -> Generator[list[int]]:
    """Every value any ``Count`` counts inside the block, the total in the list's one item."""
    total = [0]
    add = Count.add

    def counting(self: Count, value: object) -> None:
        total[0] += 1
        add(self, value)

    Count.add = counting  # type: ignore[method-assign]
    try:
        yield total
    finally:
        Count.add = add  # type: ignore[method-assign]
