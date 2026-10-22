"""Markers, the outcome ``Failed``, causes and the closed sets whose members carry no data.

The public enums are ``CauseCode``, ``PlanKind`` and ``VerdictKind``; a
marker's kind and an outcome's are named by their classes, so ``MarkerKind`` and
``OutcomeKind`` have no public name.
"""

import enum
from dataclasses import dataclass, field
from typing import NoReturn, cast

from jurat._values import described, string_data


class CauseCode(enum.StrEnum):
    """The closed cause set; no other code exists."""

    need_unavailable = "need-unavailable"
    need_undetermined = "need-undetermined"
    when_false = "when-false"
    when_failed = "when-failed"
    not_returned = "not-returned"
    unexpected_ref = "unexpected-ref"
    undeclared_read = "undeclared-read"
    raised = "raised"
    timed_out = "timed-out"
    round_bound = "round-bound"
    deadline = "deadline"
    required = "required"
    refused = "refused"
    invalid_answer = "invalid-answer"
    label_collision = "label-collision"
    conflict = "conflict"
    settled = "settled"
    over_budget = "over-budget"
    invalid_return = "invalid-return"
    design_changed = "design-changed"  # no v0.0.1 producer
    cancelled = "cancelled"
    carried = "carried"  # no v0.0.1 producer
    abstained = "abstained"
    unsupported = "unsupported"


class PlanKind(enum.StrEnum):
    execute = "execute"
    clarify = "clarify"
    inform = "inform"
    escalate = "escalate"
    broken = "broken"


class OutcomeKind(enum.StrEnum):
    returned = "returned"
    failed = "failed"


class MarkerKind(enum.StrEnum):
    unsure = "Unsure"
    unavailable = "Unavailable"
    unasked = "Unasked"


class VerdictKind(enum.StrEnum):
    passed = "passed"
    refused = "refused"


@dataclass(frozen=True, slots=True, init=False)
class Cause:
    """A code from the closed set, and refs kept as strings in the order given.

    ``code`` is a ``CauseCode`` member or its string, stored as the member; ``refs`` is a
    list or tuple of strings, stored as a tuple of their string data. A code outside the
    closed set raises ``ValueError``, any other ``refs`` ``TypeError``.
    """

    code: CauseCode
    refs: tuple[str, ...]

    def __init__(self, code: CauseCode | str, refs: list[str] | tuple[str, ...] = ()) -> None:
        # Every member is a string, so anything else is refused before the enum's lookup,
        # whose own message would write the value.
        try:
            if not issubclass(type(code), str):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError
            member = CauseCode(string_data(cast(str, code)))
        except ValueError:
            raise ValueError(
                f"cause code {described(code)} is not in the closed cause set"
            ) from None
        refs_ = cast(object, refs)
        if type(refs_) is not list and type(refs_) is not tuple:
            raise TypeError(f"cause refs {described(refs)} are not a list or tuple")
        members = cast("list[object] | tuple[object, ...]", refs_)
        if not all(issubclass(type(r), str) for r in members):
            raise TypeError(f"cause refs {described(refs)} hold a member that is not a string")
        object.__setattr__(self, "code", member)
        object.__setattr__(self, "refs", tuple(string_data(cast(str, r)) for r in members))


def _check_cause(carrier: object, cause: object) -> None:
    if not issubclass(type(cause), Cause):
        raise TypeError(f"{type(carrier).__name__} takes a Cause, not {described(cause)}")


@dataclass(frozen=True, slots=True)
class Undetermined:
    """The base of the three markers; never a status of its own."""

    kind: MarkerKind = field(init=False)
    cause: Cause

    def __post_init__(self) -> None:
        if type(self) is Undetermined:
            raise TypeError("Undetermined is the markers' base; use Unsure, Unavailable or Unasked")
        _check_cause(self, self.cause)

    def __bool__(self) -> NoReturn:
        raise TypeError(f"a {self.kind} marker has no truth value; test its type instead")


@dataclass(frozen=True, slots=True)
class Unsure(Undetermined):
    """A rule looked at its evidence and did not settle."""

    kind: MarkerKind = field(default=MarkerKind.unsure, init=False)


@dataclass(frozen=True, slots=True)
class Unavailable(Undetermined):
    """The value could not be obtained."""

    kind: MarkerKind = field(default=MarkerKind.unavailable, init=False)


@dataclass(frozen=True, slots=True)
class Unasked(Undetermined):
    """The value was not fetched, by choice."""

    kind: MarkerKind = field(default=MarkerKind.unasked, init=False)


@dataclass(frozen=True, slots=True)
class Failed:
    """A call's outcome when it did not return."""

    kind: OutcomeKind = field(default=OutcomeKind.failed, init=False)
    cause: Cause

    def __post_init__(self) -> None:
        _check_cause(self, self.cause)
