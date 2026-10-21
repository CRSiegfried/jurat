"""Designs for the load tests: a minimal one, one holding every feature of the slice, and
builders that change one declaration of it."""

import dataclasses
import enum
from typing import Any

from jurat import (
    Design,
    Group,
    Parameter,
    Policy,
    Question,
    Requirement,
    Rule,
    Source,
    SourceRequirement,
    Text,
    When,
)
from jurat import types as t


class Mood(enum.StrEnum):
    calm = "calm"
    wild = "wild"


def route(r: Any) -> str:
    return "yes"


def score(r: Any) -> float:
    return 0.5


def policy(r: Any) -> None:
    return None


STATE = t.record(subject=t.string, sender=t.record(address=t.string))
VERDICT = t.label("no", "yes")


def minimal(**changes: Any) -> Design:
    fields: dict[str, Any] = {"state": t.record(), "policy": Policy(policy, version="p1")}
    fields.update(changes)
    return Design(**fields)


def slot(name: str, **origin: str) -> dict[str, Any]:
    return {"name": name, "origin": origin}


def intent_source(**changes: Any) -> Source:
    fields: dict[str, Any] = {
        "provides": {
            "q:intent/email": Question(
                "noul",
                Text(
                    "Is {subject} mail for {who}? {{bar}} is {bar}",
                    [
                        slot("subject", ref="t:state.subject"),
                        slot("who", ref="t:lookup"),
                        slot("bar", param="intent/bar"),
                    ],
                ),
            ),
            "q:intent/tone": Question("choice", Text("Which tone?"), labels=("warm", "cold")),
        },
        "needs": ["t:state.sender", "t:state"],
        "oracle": "mail",
        "timeout": 30,
        "payload": {"model": "m", "tags": ("a", "b"), "mood": Mood.calm},
        "cost": 2,
        "feedback": True,
    }
    fields.update(changes)
    return Source(**fields)


def lookup_source(**changes: Any) -> Source:
    fields: dict[str, Any] = {
        "provides": {
            "t:lookup": t.string,
            "t:profile": {"type": t.record(name=t.string, age=t.integer())},
        },
        "needs": ["t:state"],
    }
    fields.update(changes)
    return Source(**fields)


def full(**changes: Any) -> Design:
    fields: dict[str, Any] = {
        "name": "mail",
        "state": STATE,
        "types": {"verdict": VERDICT},
        "sources": {"intent": intent_source(), "lookup": lookup_source()},
        "rules": {
            "route": Rule(
                "d:route",
                "verdict",
                route,
                version="r2",
                reads=["t:profile.age", "q:intent/email", "d:score"],
                params=["intent/bar"],
            ),
            "score": Rule("d:score", t.number(0, 1), score, version="s1", reads=["q:intent/tone"]),
        },
        "policy": Policy(
            policy, version="p3", reads=["d:route", "t:state.subject"], params=["list/cap"]
        ),
        "requirements": [Requirement("d:route", "yes")],
        "default_deny": False,
        "tools": ["send", "archive"],
        "parameters": {
            "intent/bar": Parameter(t.number(0, 1), 0.9),
            "list/cap": Parameter(t.list_(t.string), ("a", "b")),
            "blob": Parameter(t.bytes_, b"\x00\x01"),
            "mood": Parameter(t.label("calm", "wild"), Mood.calm),
            "weights": Parameter(t.mapping(t.number()), {"a": 1}),
        },
        "groups": {"intents": Group(["q:intent/tone", "q:intent/email"])},
    }
    fields.update(changes)
    return Design(**fields)


def with_rule(design: Design, rid: str, rule: Any) -> Design:
    return dataclasses.replace(design, rules={**design.rules, rid: rule})


def with_source(design: Design, sid: str, source: Any) -> Design:
    return dataclasses.replace(design, sources={**design.sources, sid: source})


def rule(rid: str, *, reads: tuple[str, ...] = (), **changes: Any) -> Rule:
    fields: dict[str, Any] = {
        "provides": f"d:{rid}",
        "outputs": t.boolean,
        "function": route,
        "version": "1",
        "reads": reads,
    }
    fields.update(changes)
    return Rule(**fields)


def question(template: str = "Is it?", *slots: dict[str, Any], **changes: Any) -> Question:
    return Question(changes.pop("kind", "noul"), Text(template, slots), **changes)


def gated(**changes: Any) -> Design:
    """``full`` with a source gated by a function ``when`` and a source requirement, and a
    declared classes block."""
    gate = Source(
        {"t:gated": t.record(note=t.string, n=t.integer())},
        needs=["t:lookup"],
        when=When(route, version="w1"),
        requirement=SourceRequirement("d:route", ["yes", "no"]),
    )
    design = with_source(full(), "gate", gate)
    classes = {"t:state": {"subject": "identifier"}, "t:gated": "open", "d:route": "content"}
    return dataclasses.replace(design, classes=classes, **changes)


def ungated() -> Design:
    """``gated`` with a failure under checks 1, 3, 7 and 10's form test in its new members;
    check 3's failure gates check 9 and check 10's tree test off."""
    gate = Source(
        {"t:bad": t.string},
        when=When(route),
        requirement=SourceRequirement("d:score", [2, 2.0, "x"]),
    )
    return dataclasses.replace(
        with_source(gated(), "bad", gate),
        classes={"t:state": {"nope": "open"}, "q:intent/tone": "open", "T:x": "open"},
    )


def held_back() -> Design:
    """``gated`` failing only the gated checks: check 9, a source requirement value that
    cannot hold, and check 10's tree test, a class tree naming no field."""
    gate = Source({"t:held": t.string}, requirement=SourceRequirement("d:score", [0.5, 2, 3]))
    return dataclasses.replace(
        with_source(gated(), "held", gate),
        classes={"t:state": {"nope": "open", "sender": {"x": "content"}}},
    )


def roles(**given: Any) -> dict[str, Any]:
    """A ``q:`` read's ``boundaries`` entry: every role, ``[]`` where ``given`` names none."""
    return {"bar": [], "thresholds": [], "margin": [], "mass": [], **given}


def bounded(**changes: Any) -> Design:
    """``full`` with a ``boundaries`` field on both rules and the policy: every role of a
    ``q:`` read, a field-path ``t:`` read and a ``d:`` read the rules and the policy share."""
    design = full()
    route_rule = dataclasses.replace(
        design.rules["route"],
        boundaries={
            "q:intent/email": roles(
                bar=["intent/bar"],
                margin=["intent/margin"],
                mass=[{"labels": ["yes"], "bounds": ["unit/mass"]}],
            ),
            "t:profile.age": ["age/limit"],
            "d:score": ["score/cut"],
        },
    )
    score_rule = dataclasses.replace(
        design.rules["score"],
        boundaries={
            "q:intent/tone": roles(
                thresholds=["tone/thresholds"],
                mass=[{"labels": ["warm", "cold"], "bounds": ["unit/mass"]}],
            )
        },
    )
    policy_ = dataclasses.replace(
        design.policy,
        reads=["d:route", "t:state.subject", "d:score"],
        boundaries={"d:score": ["score/low", "score/cut"]},
    )
    parameters = {
        **design.parameters,
        "intent/margin": Parameter(t.number(0, 1), 0.1),
        "unit/mass": Parameter(t.number(0, 1), 0.5),
        "tone/thresholds": Parameter(t.mapping(t.number(0, 1)), {"warm": 0.6}),
        "age/limit": Parameter(t.integer(), 18),
        "score/cut": Parameter(t.number(), 0.5),
        "score/low": Parameter(t.number(), 0.2),
    }
    fields: dict[str, Any] = {
        "rules": {"route": route_rule, "score": score_rule},
        "policy": policy_,
        "parameters": parameters,
    }
    fields.update(changes)
    return dataclasses.replace(design, **fields)
