# pyright: reportPrivateUsage=false
"""The dead-question lint, ``dead_questions``: one ``decide_sync`` run
per point through a binding answering from the point, each status with its witness, the
byte comparison of the ``plan`` and ``verdict`` items, the ``ValueError``s before any run, and
a report of plain data. The points themselves are tested in ``test_lint_points.py``."""

import copy
import json
from typing import Any

import pytest

from designs import roles
from jurat import (
    Design,
    InvalidDesign,
    LoadedDesign,
    Parameter,
    Plan,
    Policy,
    Question,
    Requirement,
    Rule,
    Source,
    Text,
    Undetermined,
    load,
)
from jurat import types as t
from jurat._canonical import encode
from jurat.helpers import dead_questions, pick_with_margin, pick_with_thresholds

CALLS: list[str] = []


def go(r: Any) -> object:
    CALLS.append("go")
    picked = pick_with_margin(r, "q:go", r.param("go/bar"), r.param("go/margin"))
    return picked if isinstance(picked, Undetermined) else picked == "yes"


def mood(r: Any) -> object:
    return pick_with_thresholds(r, "q:mood", r.param("mood/th"), 0)


def by_go(r: Any) -> Plan:
    CALLS.append("policy")
    if r.read("d:go") is True:
        return Plan("execute")
    return Plan("escalate", reason="not sure")


def always(r: Any) -> Plan:
    return Plan("execute")


QUESTIONS = {
    "q:go": Question("noul", Text("Go?")),
    "q:tone": Question("choice", Text("Tone?"), labels=("warm", "cold")),
}
GO = Rule(
    "d:go",
    t.boolean,
    go,
    version="1",
    reads=["q:go"],
    params=["go/bar", "go/margin"],
    boundaries={"q:go": roles(bar=["go/bar"], margin=["go/margin"])},
)
PARAMETERS = {
    "go/bar": Parameter(t.number(0, 1), 0.8),
    "go/margin": Parameter(t.number(0, 1), 0.2),
}


def design(**changes: Any) -> Design:
    fields: dict[str, Any] = {
        "state": t.record(),
        "policy": Policy(by_go, version="p", reads=["d:go"]),
        "sources": {"ask": Source(QUESTIONS)},
        "rules": {"go": GO},
        "parameters": PARAMETERS,
    }
    fields.update(changes)
    return Design(**fields)


LOADED = load(design())


def answer(kind: str, labels: list[str], **probabilities: float) -> dict[str, Any]:
    return {"kind": kind, "labels": labels, "probabilities": probabilities}


# ---------------------------------------------------------------------------------------------
# Statuses and the witness


def test_a_question_whose_answer_changes_the_plan_is_live_with_its_exact_witness() -> None:
    report = dead_questions(LOADED)
    assert report["questions"]["q:go"] == {
        "status": "live",
        "witness": [
            {
                "q:go": answer("noul", ["yes", "no"], yes=0.0, no=1.0),
                "q:tone": answer("choice", ["warm", "cold"], warm=1.0, cold=0.0),
                "t:state": {},
            },
            {
                "q:go": answer("noul", ["yes", "no"], yes=0.8, no=0.2),
                "q:tone": answer("choice", ["warm", "cold"], warm=1.0, cold=0.0),
                "t:state": {},
            },
        ],
    }


def test_a_question_off_the_grid_that_changes_nothing_is_not_on_grid() -> None:
    assert dead_questions(LOADED)["questions"]["q:tone"] == {
        "status": "not-on-grid",
        "witness": None,
    }


def test_a_question_on_the_grid_that_changes_nothing_is_no_change_found() -> None:
    mood_rule = Rule(
        "d:mood",
        t.label("calm", "wild"),
        mood,
        version="1",
        reads=["q:mood"],
        params=["mood/th"],
        boundaries={"q:mood": roles(thresholds=["mood/th"])},
    )
    loaded = load(
        design(
            sources={
                "ask": Source(
                    {**QUESTIONS, "q:mood": Question("choice", Text("?"), labels=("calm", "wild"))}
                )
            },
            rules={"go": GO, "mood": mood_rule},
            parameters={
                **PARAMETERS,
                "mood/th": Parameter(t.mapping(t.number(0, 1)), {"calm": 0.6}),
            },
        )
    )
    report = dead_questions(loaded)
    assert report["questions"]["q:mood"] == {"status": "no-change-found", "witness": None}
    assert report["questions"]["q:go"]["status"] == "live"


def test_a_question_changing_only_the_verdict_is_live() -> None:
    ok = Rule(
        "d:ok",
        t.boolean,
        go,
        version="1",
        reads=["q:go"],
        params=["go/bar", "go/margin"],
        boundaries={"q:go": roles(bar=["go/bar"], margin=["go/margin"])},
    )
    loaded = load(
        design(
            policy=Policy(always, version="p"),
            rules={"ok": ok},
            requirements=[Requirement("d:ok", True)],
            default_deny=False,
        )
    )
    report = dead_questions(loaded)
    assert report["questions"]["q:go"]["status"] == "live"
    first, second = report["questions"]["q:go"]["witness"]
    assert first["q:go"]["probabilities"] == {"yes": 0.0, "no": 1.0}
    assert second["q:go"]["probabilities"] == {"yes": 0.8, "no": 0.2}


def test_the_points_count_every_run_in_grid_order() -> None:
    CALLS.clear()
    report = dead_questions(LOADED)
    # q:go: nine axis points and the marker; q:tone: two labels and the marker; t:state: {}.
    assert report["points"] == 10 * 3 * 1
    assert CALLS.count("policy") == 30


def test_the_binding_answers_under_default_and_every_declared_batch_key() -> None:
    sources = {
        "ask": Source({"q:go": QUESTIONS["q:go"]}, batch="model"),
        "look": Source({"t:n": t.integer()}),
    }
    loaded = load(design(sources=sources))
    report = dead_questions(loaded, values={"t:n": [1, 2]})
    assert report["points"] == 10 * 2
    assert report["questions"]["q:go"]["status"] == "live"
    witness = report["questions"]["q:go"]["witness"]
    assert [w["t:n"] for w in witness] == [1, 1]


def test_a_point_with_the_marker_reaches_the_rule_as_unavailable_refused() -> None:
    seen: list[object] = []

    def watch(r: Any) -> object:
        seen.append(r.read("q:go"))
        return True

    rule = Rule("d:go", t.boolean, watch, version="1", reads=["q:go"])
    loaded = load(design(sources={"ask": Source({"q:go": QUESTIONS["q:go"]})}, rules={"go": rule}))
    dead_questions(loaded)
    last = seen[-1]
    assert type(last).__name__ == "Unavailable"
    assert last.cause.code == "refused"  # type: ignore[attr-defined]
    assert last.cause.refs == ("q:go",)  # type: ignore[attr-defined]


def test_the_state_is_the_points_t_state_live() -> None:
    seen: list[object] = []

    def watch(r: Any) -> Plan:
        seen.append(r.read("t:state.blob"))
        return Plan("escalate", reason="r")

    loaded = load(
        design(
            state=t.record(blob=t.bytes_),
            policy=Policy(watch, version="p", reads=["t:state.blob"]),
            sources={},
            rules={},
            parameters={},
        )
    )
    report = dead_questions(loaded, values={"t:state.blob": [b"\x01"]})
    assert seen == [b"\x01"]
    assert report == {"points": 1, "questions": {}}


# ---------------------------------------------------------------------------------------------
# ValueErrors before any run


@pytest.mark.parametrize(
    "values",
    [
        None,  # t:n is unfilled
        {"t:n": [1], "t:other": [1]},
        {"t:n": [1], "q:go": [{"yes": 1, "no": 0}]},
        {"t:n": ["x"]},
        {"t:n": [1, 1]},
    ],
)
def test_a_value_error_is_raised_before_any_run(values: Any) -> None:
    loaded = load(design(sources={**design().sources, "look": Source({"t:n": t.integer()})}))
    CALLS.clear()
    with pytest.raises(ValueError):  # noqa: PT011
        dead_questions(loaded, values=values)
    assert CALLS == []


def test_loaded_not_exactly_a_loaded_design_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        dead_questions(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# The report


def test_the_report_is_plain_data_json_and_deepcopy_take() -> None:
    report = dead_questions(LOADED)
    assert json.loads(json.dumps(report)) == report
    assert copy.deepcopy(report) == report


def test_two_runs_give_the_same_report_bytes() -> None:
    assert encode(dead_questions(LOADED)) == encode(dead_questions(LOADED))


def test_the_lint_raises_nothing_on_a_finding_and_changes_no_loaded_design() -> None:
    before = (LOADED.document, LOADED.design_hash, repr(LOADED._grid))
    report = dead_questions(LOADED)
    assert report["questions"]["q:tone"]["status"] == "not-on-grid"
    assert (LOADED.document, LOADED.design_hash, repr(LOADED._grid)) == before
    assert type(LOADED) is LoadedDesign
    assert not isinstance(report, InvalidDesign)


def test_a_plan_differing_only_in_its_reason_is_live() -> None:
    def reasons(r: Any) -> Plan:
        return Plan("escalate", reason="go" if r.read("d:go") is True else "stay")

    loaded = load(design(policy=Policy(reasons, version="p", reads=["d:go"])))
    assert dead_questions(loaded)["questions"]["q:go"]["status"] == "live"


def test_a_rule_changing_a_value_it_reads_reaches_no_later_point() -> None:
    seen: list[list[str]] = []

    def spoil(r: Any) -> bool:
        items = r.read("t:items")
        seen.append(list(items))
        items.append("spoiled")
        return True

    rule = Rule("d:go", t.boolean, spoil, version="1", reads=["t:items", "q:go"])
    sources = {
        "ask": Source({"q:go": QUESTIONS["q:go"]}),
        "look": Source({"t:items": t.list_(t.string)}),
    }
    loaded = load(design(sources=sources, rules={"go": rule}))
    report = dead_questions(loaded, values={"t:items": [["a"]]})
    assert report["points"] == len(seen) == 3  # q:go off the grid: two labels and the marker
    assert seen == [["a"]] * 3


def test_a_witness_found_only_under_a_later_assignment_of_the_other_reads() -> None:
    def gated(r: Any) -> Plan:
        if r.read("d:go") is True and r.read("t:n") == 2:
            return Plan("execute")
        return Plan("escalate", reason="r")

    sources = {"ask": Source({"q:go": QUESTIONS["q:go"]}), "look": Source({"t:n": t.integer()})}
    policy = Policy(gated, version="p", reads=["d:go", "t:n"])
    loaded = load(design(sources=sources, policy=policy))
    witness = dead_questions(loaded, values={"t:n": [1, 2]})["questions"]["q:go"]["witness"]
    assert [w["t:n"] for w in witness] == [2, 2]
    assert [w["q:go"]["probabilities"]["yes"] for w in witness] == [0.0, 0.8]


def heeds_nothing(r: Any) -> bool:
    r.read("q:go")
    return True


def test_a_question_whose_entry_yields_no_answer_point_is_not_on_grid() -> None:
    rule = Rule(
        "d:go",
        t.boolean,
        heeds_nothing,
        version="1",
        reads=["q:go"],
        boundaries={"q:go": roles(mass=[{"labels": ["yes", "no"], "bounds": ["go/bar"]}])},
    )
    loaded = load(design(rules={"go": rule}))
    report = dead_questions(loaded)
    assert report["questions"]["q:go"]["status"] == "not-on-grid"
    assert report["points"] == 3 * 3  # q:go and q:tone: two labels at 1 and the marker each


def test_a_one_label_question_on_the_grid_that_changes_nothing_is_no_change_found() -> None:
    def heeds_only(r: Any) -> bool:
        r.read("q:only")
        return True

    only = Question("choice", Text("Only?"), labels=("only",))
    rule = Rule(
        "d:only",
        t.boolean,
        heeds_only,
        version="1",
        reads=["q:only"],
        boundaries={"q:only": roles(bar=["go/bar"])},
    )
    loaded = load(
        design(
            sources={"ask": Source({**QUESTIONS, "q:only": only})}, rules={"go": GO, "only": rule}
        )
    )
    report = dead_questions(loaded)
    assert report["questions"]["q:only"] == {"status": "no-change-found", "witness": None}
