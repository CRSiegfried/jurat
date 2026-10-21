# pyright: reportPrivateUsage=false
"""The ``Unsure``-to-execute lint, ``unsure_to_execute``: every ``execute`` plan the policy
returns when one of its ``d:`` reads is ``Unsure``.

It fills the policy's declared reads by the lint points: a read under another declared read's
path is filled as a field of that read, and every other read is written at its path, on a
board holding one entry per base ref. It sets each ``d:`` base ref in turn to ``Unsure``
with cause ``abstained``, naming the declared reads of the rule that writes it, while the
other reads range over their points, and calls the policy alone through ``_component.call``
on a fresh board per call: no plan rule and no guard. A call is ``execute`` only when no stop
was recorded, nothing was raised, and the return is exactly a ``Plan`` of kind ``execute``.
The report is a warning: the lint raises nothing on a finding.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from jurat._component import base, call
from jurat._design import LoadedDesign, Policy, Rule
from jurat._interpret import _marker_document
from jurat._markers import Cause, CauseCode, PlanKind, Unsure
from jurat._plan import Plan
from jurat._read_view import ComponentKind, ReadView, Role, walk
from jurat._refs import code_point_sorted, parse_ref
from jurat._values import type_name
from jurat.helpers._points import Point, fill, fresh, grid_order

_POLICY = "policy"


def unsure_to_execute(
    loaded: LoadedDesign, *, values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The report ``{"points": N, "execute": [{"read", "point"}]}``, by ``read`` in code-point
    order, then by point in grid order; a ``ValueError`` before any call for a read nothing
    fills or a slip in ``values``."""
    if type(loaded) is not LoadedDesign:
        raise TypeError(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    policy = cast(Policy, loaded._design.policy)  # load refuses the policy table
    declared = code_point_sorted(set(policy.reads))
    top = [r for r in declared if not any(r.startswith(o + ".") for o in declared)]
    fills = fill(loaded, top, values)
    points = 0
    execute: list[dict[str, Any]] = []
    for unsure in code_point_sorted({base(r) for r in declared if r.startswith("d:")}):
        writer = cast(Rule, loaded._design.rules[loaded._providers[unsure]])  # no table in v0.0.1
        marker = Unsure(Cause(CauseCode.abstained, code_point_sorted(set(writer.reads))))
        shown_marker = _marker_document(marker)  # the record's marker form, interpret's
        others = [r for r in top if base(r) != unsure]
        for combo in grid_order([fills[r] for r in others]):
            point = dict(zip(others, combo, strict=True))
            points += 1
            if _executes(loaded, policy, _board(point, unsure, marker)):
                shown = {
                    r: fresh(shown_marker) if base(r) == unsure else _record(r, top, point)
                    for r in declared
                }
                execute.append({"read": unsure, "point": shown})
    return {"points": points, "execute": execute}


def _board(point: Mapping[str, Point], unsure: str, marker: Unsure) -> dict[str, object]:
    """A fresh board: each read's live value at its path, and the marker at ``unsure``."""
    board: dict[str, object] = {unsure: marker}
    for read, p in point.items():
        fields = parse_ref(read).fields
        if not fields:
            board[read] = fresh(p.live)
            continue
        node = cast("dict[str, object]", board.setdefault(base(read), {}))
        for name in fields[:-1]:
            node = cast("dict[str, object]", node.setdefault(name, {}))
        node[fields[-1]] = fresh(p.live)
    return board


def _executes(loaded: LoadedDesign, policy: Policy, board: dict[str, object]) -> bool:
    """Whether the policy alone returns an ``execute`` plan over ``board``, its stop record
    read as compile reads it."""
    leaves = {path: loaded._leaf_value(path) for path in policy.params}
    view = ReadView(Role(ComponentKind.policy, _POLICY), policy.reads, policy.params, board, leaves)
    called = call(loaded._callables[_POLICY], view)
    if called.stop is not None or called.error is not None:
        return False
    returned = called.returned
    return type(returned) is Plan and returned.kind is PlanKind.execute


def _record(read: str, top: Sequence[str], point: Mapping[str, Point]) -> object:
    """A declared read's value in record form, read through the filled read it lies under or
    is."""
    for filled in top:
        if read == filled or read.startswith(filled + "."):
            fields = parse_ref(read).fields[len(parse_ref(filled).fields) :]
            return fresh(walk(point[filled].record, fields, read))
    raise AssertionError(f"{read!r} lies under no filled read")  # pragma: no cover
