"""What every helper and combinator does around its reads: the raise on a ``when``'s view,
its reads, and its item in the view's helper log.

A helper is written as::

    with called(r, "pick_with_margin", [ref]) as call:
        (value,) = call.values
        ...
        call.rested_on([ref])
        return result

On entry ``called`` raises on a ``when``'s view before any read, then reads each ref it was
given, in argument order, and only then lists the item, ``rested_on`` empty. So a ref outside
the read set stops the helper at its read, with no item listed, and ``refs`` holds only refs
the component declared. ``rested_on`` is filled from ``call.rested_on`` only when the helper
returns, so a helper that raised after its reads, or that a stop passed through, stays listed
with ``rested_on`` empty even when it had set it. A call inside another helper's
is not listed.
"""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

from jurat._read_view import ComponentKind, ReadView, close_call, open_call
from jurat._values import string_data


class HelperOnWhen(Exception):
    """A helper or combinator called on a function ``when``'s view; raised before any read,
    so no helper result reaches a ``when``. Uncaught, it is the ``when``'s failure."""


class Call:
    """An open call: the values or markers its refs read, in argument order, and
    ``rested_on``, which records the refs its result rested on, as it returns."""

    __slots__ = ("_rested_on", "values")

    def __init__(self, values: tuple[Any, ...]) -> None:
        self.values = values
        self._rested_on: tuple[str, ...] | None = None

    def rested_on(self, refs: Sequence[str]) -> None:
        self._rested_on = tuple(string_data(r) for r in refs)


@contextmanager
def called(r: ReadView, helper: str, refs: Sequence[str]) -> Generator[Call]:
    """Read ``refs`` on ``r`` in order, then open a call of ``helper`` over them on its log."""
    if r.role.kind is ComponentKind.when:
        raise HelperOnWhen(f"{helper} was called on the read view of when {r.role.name!r}")
    values = tuple(r.read(ref) for ref in refs)
    # Every ref read is a declared read, so a string; it is listed by its string data.
    call = Call(values)
    index = open_call(r, helper, [string_data(ref) for ref in refs])
    try:
        yield call
    except BaseException:
        close_call(r, index, None)  # raised or stopped: `rested_on` stays empty
        raise
    else:
        close_call(r, index, call._rested_on)  # pyright: ignore[reportPrivateUsage]
