"""The guard: at the commit point, every requirement of the plan and its steps holds on the
board and default-deny is kept, or the verdict lists every failure.

``guard`` reads the plan, the board, the design's plan-level requirements and its default-deny
switch, and nothing else; it appends nothing, so compile, or decide at a
broken entry, appends the ``verdict`` item ``verdict_item`` writes. ``Verdict`` and
``GuardFailure`` check themselves when built, so a refused verdict's derived cause is never
empty, whoever builds one, the record views from a ``verdict`` item included.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from jurat._canonical import encode
from jurat._design import Requirement
from jurat._markers import Cause, CauseCode, PlanKind, Undetermined, VerdictKind
from jurat._plan import Plan
from jurat._values import described, string_data

_PLAN = "plan"


@dataclass(frozen=True, slots=True, init=False)
class GuardFailure:
    """One failure: the step's position from 1, or ``"plan"`` for a plan-level requirement;
    the requirement as declared and the ref the guard read, both ``None`` for a default-deny
    failure, which reads none."""

    step: int | Literal["plan"]
    requirement: Requirement | None
    ref: str | None

    def __init__(
        self, step: int | Literal["plan"], requirement: Requirement | None, ref: str | None
    ) -> None:
        if type(step) is int:
            if step < 1:
                raise ValueError(f"a failure's step {described(step)} is below 1")
        elif issubclass(type(step), str):
            if string_data(cast(str, step)) != _PLAN:
                raise ValueError(
                    f'a failure\'s step {described(step)} is neither a position nor "plan"'
                )
            step = _PLAN
        else:
            raise TypeError(f'a failure\'s step {described(step)} is not an int or "plan"')
        if requirement is not None and type(requirement) is not Requirement:
            raise TypeError(
                f"a failure's requirement {described(requirement)} is not a Requirement"
            )
        if ref is not None:
            if not issubclass(type(ref), str):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f"a failure's ref {described(ref)} is not a string or None")
            ref = string_data(ref)
        if (requirement is None) != (ref is None):
            raise ValueError("a failure's requirement and ref are None together")
        if requirement is None and step == _PLAN:
            raise ValueError("a default-deny failure names a step, not the plan")
        if requirement is not None and ref != requirement.ref:
            raise ValueError(f"a failure's ref {ref!r} is not its requirement's")
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "requirement", requirement)
        object.__setattr__(self, "ref", ref)


@dataclass(frozen=True, slots=True, init=False)
class Verdict:
    """The guard verdict: ``passed`` with no failures, or ``refused`` listing every one in the
    guard's order."""

    kind: VerdictKind
    failures: tuple[GuardFailure, ...]

    def __init__(
        self,
        kind: VerdictKind | str,
        failures: list[GuardFailure] | tuple[GuardFailure, ...] = (),
    ) -> None:
        # Every member is a string, so anything else is refused before the enum's lookup,
        # whose own message would write the value.
        try:
            if not issubclass(type(kind), str):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError
            member = VerdictKind(string_data(cast(str, kind)))
        except ValueError:
            raise ValueError(f"verdict kind {described(kind)} is not passed or refused") from None
        members = cast(object, failures)
        if type(members) is not list and type(members) is not tuple:
            raise TypeError(f"a verdict's failures {described(failures)} are not a list or tuple")
        found = tuple(cast("list[object] | tuple[object, ...]", members))
        if not all(type(f) is GuardFailure for f in found):
            raise TypeError("a verdict's failure is not a GuardFailure")
        if (member is VerdictKind.refused) != bool(found):
            raise ValueError("a verdict is refused exactly when it has failures")
        object.__setattr__(self, "kind", member)
        object.__setattr__(self, "failures", cast("tuple[GuardFailure, ...]", found))

    @property
    def cause(self) -> Cause | None:
        """``refused`` naming, per failure in order, the ref it read or ``step:N`` for a
        default-deny failure; ``None`` for a passed verdict. Derived, stored nowhere."""
        if self.kind is VerdictKind.passed:
            return None
        return Cause(
            CauseCode.refused,
            [f.ref if f.ref is not None else f"step:{f.step}" for f in self.failures],
        )


def guard(
    plan: Plan,
    board: Mapping[str, object],
    requirements: Sequence[Requirement],
    default_deny: bool,
) -> Verdict:
    """Check an ``execute`` plan against the board; pass every other kind. The plan-level
    requirements come first in declaration order, then each step by position, a default-deny failure
    standing where its requirements would."""
    if plan.kind is not PlanKind.execute:
        return Verdict(VerdictKind.passed)
    encoded: dict[str, bytes | None] = {}  # each entry's encoding, taken once per guard
    failures = [
        GuardFailure(_PLAN, r, r.ref) for r in requirements if not _holds(r, board, encoded)
    ]
    for n, step in enumerate(plan.steps, 1):
        if not step.requirements:
            if default_deny:
                failures.append(GuardFailure(n, None, None))
            continue
        failures += [
            GuardFailure(n, r, r.ref) for r in step.requirements if not _holds(r, board, encoded)
        ]
    return Verdict(VerdictKind.refused if failures else VerdictKind.passed, failures)


def _holds(
    requirement: Requirement, board: Mapping[str, object], encoded: dict[str, bytes | None]
) -> bool:
    """Whether the ref's entry is a value whose canonical encoding is the declared value's; a
    missing entry or a marker fails. Each entry's encoding is kept in ``encoded`` by ref, so
    requirements on one ref encode its entry once."""
    ref = requirement.ref
    if ref not in board:
        return False
    if ref not in encoded:
        entry = board[ref]
        if issubclass(type(entry), Undetermined):
            return False
        try:
            encoded[ref] = encode(entry)
        except ValueError:  # outside the value domain: no encoding to compare, so it fails
            encoded[ref] = None
    entry_bytes = encoded[ref]
    if entry_bytes is None:
        return False
    try:
        return entry_bytes == encode(requirement.value)
    except ValueError:
        return False


def verdict_item(verdict: Verdict) -> dict[str, Any]:
    """The ``verdict`` item of the record."""
    return {
        "item": "verdict",
        "kind": verdict.kind.value,
        "failures": [
            {
                "step": f.step,
                "requirement": None
                if f.requirement is None
                else {"ref": f.requirement.ref, "value": f.requirement.value},
                "ref": f.ref,
            }
            for f in verdict.failures
        ],
    }
