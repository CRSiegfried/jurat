"""The tutorial's example runs as the tutorial says: docs/tutorials/write_a_design."""

import pytest

import helpdesk_design
import helpdesk_run
from jurat import PlanKind, load


def test_the_kept_document_loads_to_the_same_hash() -> None:
    loaded = load(helpdesk_design.DESIGN)
    rules = helpdesk_design.RULES
    functions = {name: rule.function for name, rule in rules.items()}
    functions["policy"] = helpdesk_design.route
    assert load(loaded.document, functions=functions).design_hash == loaded.design_hash


def test_the_driver_decides_and_replays(capsys: pytest.CaptureFixture[str]) -> None:
    results = helpdesk_run.main()
    assert [(plan.kind, plan.reply) for plan, _ in results] == [
        (PlanKind.inform, "Routed to delivery for order A-1042."),
        (PlanKind.inform, "Routed to delivery; no order number found."),
    ]
    assert [findings for _, findings in results] == [[], []]
    assert "replay findings: []" in capsys.readouterr().out
