# pyright: reportPrivateUsage=false
"""Compile: call the policy once over its read view, turn what it gave into the turn's one
plan under the seven plan rules, append the ``plan`` item, call the guard and append the
``verdict`` item.

The outcome is the read view's order: the first recorded undeclared read, the first ``require``
stop, a raise, an invalid return, the plan. A policy plan is first tested for what only the
engine writes, then against the rules in their fixed order: 1, 5 and 7 over the plan, then
each step by position against 2, 3, 4, 7 and 6; the cause names the first break. The plan
returned is built from the copy the ledger stores, in exact ``dict``, ``list`` and ``str`` with
each value's live type kept, so a ``1.0`` stays a ``float`` where the record's bytes read back
as ``1``.

v0.0.1 has a function policy only: a policy table and plan declarations are
refused at load, so every ``plan`` item holds ``derivation`` and none holds ``rows`` or
``slots``. Compile writes no board entry.
"""

from collections.abc import Callable, Mapping
from typing import Any, cast

from jurat._component import at, call, exception_record, names_read
from jurat._design import LoadedDesign, Policy, Requirement
from jurat._guard import Verdict, guard, verdict_item
from jurat._ledger import Ledger
from jurat._markers import Cause, CauseCode, PlanKind, Undetermined
from jurat._plan import Binding, Origin, Plan, Step, engine_plan
from jurat._read_view import ComponentKind, ReadView, RequireStopped, Role, UndeclaredRead
from jurat._refs import is_field_path, parse_ref
from jurat._types import hold_failure
from jurat._values import domain_error, string_data

_C = CauseCode
_POLICY = "policy"
_IDEMPOTENCY_KEY = "idempotency_key"


def compile_turn(
    ledger: Ledger, loaded: LoadedDesign, board: Mapping[str, object]
) -> tuple[Plan, Verdict]:
    """Run compile over the live board interpret left, every ref on it, and
    return the plan and the verdict, both appended."""
    policy = cast(Policy, loaded._design.policy)  # load refuses the policy table
    leaves = {path: loaded._leaf_value(path) for path in policy.params}
    view = ReadView(Role(ComponentKind.policy, _POLICY), policy.reads, policy.params, board, leaves)
    called = call(loaded._callables[_POLICY], view)
    stop, error = called.stop, called.error
    unnamed: str | None = None
    if isinstance(stop, UndeclaredRead):
        if names_read(loaded, stop):
            plan = engine_plan(PlanKind.broken, Cause(_C.undeclared_read, [_POLICY, stop.read]))
        else:
            plan = engine_plan(PlanKind.broken, Cause(_C.undeclared_read, [_POLICY]))
            unnamed = stop.read
    elif isinstance(stop, RequireStopped):
        plan = engine_plan(PlanKind.escalate, Cause(_C.required, [stop.ref]))
    elif error is not None:
        plan = engine_plan(PlanKind.broken, Cause(_C.raised, [_POLICY]))
    elif called.awaitable:  # tested before any other test of the return
        plan = engine_plan(PlanKind.broken, Cause(_C.invalid_return, [_POLICY]))
    else:
        refs = _Rules(loaded, board, policy).first_break(called.returned)
        if refs is None:
            plan = cast(Plan, called.returned)
        else:
            plan = engine_plan(PlanKind.broken, Cause(_C.invalid_return, refs))
    item = plan_item(plan, called.derivation)
    index = ledger.append(item)
    if called.text is not None:  # whatever the outcome, a stop's included
        ledger.write_unhashed("exceptions", exception_record(called.text), index)
    if unnamed is not None:
        ledger.write_unhashed("reads", unnamed, index)
    if plan.cause is None:
        plan = _plan_of(item)  # the ledger's stored copy, not the policy's values
    verdict = guard(plan, board, loaded._requirements(), loaded._design.default_deny)
    ledger.append(verdict_item(verdict))
    return plan, verdict


# ---------------------------------------------------------------------------------------------
# The plan rules


class _Rules:
    def __init__(self, loaded: LoadedDesign, board: Mapping[str, object], policy: Policy) -> None:
        self.loaded = loaded
        self.board = board
        self.tools = frozenset(loaded._design.tools)
        self.reads = frozenset(string_data(r) for r in policy.reads)

    def first_break(self, returned: object) -> list[str] | None:
        """The cause's refs for what the policy returned, or ``None`` for a valid plan."""
        if type(returned) is not Plan:  # a subclass may override what compile reads
            return [_POLICY]
        plan = returned
        if plan.cause is not None or plan.slots or any(_has_set(s) for s in plan.steps):
            return [_POLICY]  # what only the engine writes
        if plan.kind is not PlanKind.execute and plan.steps:
            return ["rule:1"]
        if not _rule_5(plan):
            return ["rule:5"]
        # A text's length is tested before its code points are read.
        if not all(t is None or domain_error(t) is None for t in (plan.reason, plan.reply)):
            return ["rule:7"]
        checks: tuple[tuple[int, Callable[[Step, int], bool]], ...] = (
            (2, _rule_2),
            (3, self.rule_3),
            (4, self.rule_4),
            (7, _rule_7),  # before 6: its size bound keeps rule 6's walk of each value short
            (6, self.rule_6),
        )
        for n, step in enumerate(plan.steps, 1):
            for number, holds in checks:
                if not holds(step, n):
                    return [f"rule:{number}", f"step:{n}"]
        return None

    def rule_3(self, step: Step, n: int) -> bool:
        """A declared tool, and no argument ``idempotency_key``."""
        # Each name by its string data, so no name's hash or comparison runs.
        names = (string_data(name) for name in step.arguments)
        return step.tool in self.tools and all(name != _IDEMPOTENCY_KEY for name in names)

    def rule_4(self, step: Step, n: int) -> bool:
        """Each board-ref origin a declared read of the policy that holds a value; the board
        is read directly, so no stop or helper item is added to the policy's view."""
        for origin in step.arguments.values():
            if type(origin) is Origin and origin.ref is not None:
                if origin.ref not in self.reads:
                    return False
                if issubclass(type(at(self.board, origin.ref)), Undetermined):
                    return False
        return True

    def rule_6(self, step: Step, n: int) -> bool:
        """Each requirement on a ``d:`` ref with no field path that has a provider, and can
        hold."""
        return all(self.can_hold(r) for r in step.requirements)

    def can_hold(self, requirement: Requirement) -> bool:
        ref = cast(object, requirement.ref)
        if type(ref) is not str:  # by type alone: isinstance reads the policy's __class__
            return False
        try:
            p = parse_ref(ref)
        except ValueError:  # not a ref under the grammar
            return False
        if p.level != "d" or p.fields or p.candidate is not None:
            return False
        if ref not in self.loaded._providers:
            return False
        return hold_failure(requirement.value, self.loaded._types[ref], {}) is None


def _has_set(step: Step) -> bool:
    return any(type(o) is Origin and o.set is not None for o in step.arguments.values())


def _text(value: str | None) -> bool:
    """Text holding a character that is not whitespace."""
    return value is not None and any(not c.isspace() for c in value)


def _rule_5(plan: Plan) -> bool:
    """``reply`` on ``clarify`` and ``inform`` only, ``reason`` on ``escalate``, each carried
    text holding a character that is not whitespace, and the kind not ``broken``."""
    if plan.kind is PlanKind.broken:
        return False
    if (plan.reply is not None) != (plan.kind in (PlanKind.clarify, PlanKind.inform)):
        return False
    if plan.kind is PlanKind.escalate and plan.reason is None:
        return False
    return all(t is None or _text(t) for t in (plan.reason, plan.reply))


def _rule_2(step: Step, n: int) -> bool:
    """Each binding names an earlier step, by a path null or in the field-path grammar."""
    for origin in step.arguments.values():
        if type(origin) is Binding:
            if not 1 <= origin.step < n:
                return False
            if origin.path is not None and not is_field_path(origin.path):
                return False
    return True


def _rule_7(step: Step, n: int) -> bool:
    """Every literal argument, argument name and requirement value in the value domain."""
    # Each name is a value, its length tested before its code points are read; then two
    # names of one string data, which ``Step`` keeps as given, put the arguments outside the
    # domain, found by string data so no name's hash or comparison runs.
    if any(domain_error(name) is not None for name in step.arguments):
        return False
    names = [string_data(name) for name in step.arguments]
    if len(set(names)) != len(names):
        return False
    for origin in step.arguments.values():
        literal = type(origin) is Origin and origin.ref is None
        if literal and domain_error(cast(Origin, origin).literal) is not None:
            return False
    # Tested before rule 6, which walks each value against its ref's type at every place.
    return all(domain_error(r.value) is None for r in step.requirements)


# ---------------------------------------------------------------------------------------------
# The plan item and the plan built from it


def _plain(value: Any) -> Any:
    """A domain value's copy in exact ``dict``, ``list`` and ``str``, as the ledger stores it."""
    t = type(cast(object, value))
    if t is dict:
        return {string_data(k): _plain(v) for k, v in cast("dict[str, Any]", value).items()}
    if t is list or t is tuple:
        return [_plain(v) for v in cast("list[Any]", value)]
    if issubclass(t, str):
        return string_data(value)
    return value


def _origin(origin: Origin | Binding) -> dict[str, Any]:
    if type(origin) is Binding:
        return {"step": origin.step, "path": origin.path}
    origin = cast(Origin, origin)
    if origin.ref is not None:
        return {"ref": origin.ref}
    # No literal here carries a set: only a grounded expansion writes one, and load refuses
    # candidate sets; a policy's is refused before rule 1.
    return {"literal": _plain(origin.literal)}


def plan_item(plan: Plan, derivation: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The ``plan`` item, with ``derivation`` when a function policy wrote the plan and none
    at a broken entry, where no policy ran."""
    cause = plan.cause
    item: dict[str, Any] = {
        "item": "plan",
        "kind": plan.kind.value,
        "steps": [
            {
                "tool": step.tool,
                "arguments": {name: _origin(o) for name, o in step.arguments.items()},
                "requirements": [
                    {"ref": r.ref, "value": _plain(r.value)} for r in step.requirements
                ],
            }
            for step in plan.steps
        ],
        "reason": plan.reason,
        "reply": plan.reply,
        "cause": None if cause is None else {"code": cause.code.value, "refs": list(cause.refs)},
    }
    if derivation is not None:
        item["derivation"] = derivation
    return item


def _plan_of(item: dict[str, Any]) -> Plan:
    """The policy's plan rebuilt from its item, the copy the ledger stores."""

    def origin(o: dict[str, Any]) -> Origin | Binding:
        if "step" in o:
            return Binding(o["step"], o["path"])
        if "ref" in o:
            return Origin(ref=o["ref"])
        return Origin(literal=o["literal"])

    steps = [
        Step(
            s["tool"],
            {name: origin(o) for name, o in s["arguments"].items()},
            [Requirement(r["ref"], r["value"]) for r in s["requirements"]],
        )
        for s in item["steps"]
    ]
    return Plan(item["kind"], steps, reason=item["reason"], reply=item["reply"])
