"""Synthetic record logs for the ledger's tests: a normal turn, a broken entry, a cancelled
turn and a turn whose codec raised a cancellation, built item by item so a test can take any
prefix or swap one item."""

import copy
from typing import Any

HASH = "sha256:" + "ab" * 32
IDENTITY = "sha256:" + "cd" * 32

DESIGN: dict[str, Any] = {"item": "design", "hash": HASH, "format": 1, "helpers": 1}

SETTINGS: dict[str, Any] = {
    "item": "settings",
    "settings": {
        "round_bound": 8,
        "timeout": 60,
        "budget": None,
        "deadline": None,
        "certificate_cap": None,
    },
    "providers": {
        "t:mail/body": "fetch_mail",
        "t:mail/subject": "fetch_mail",
        "q:spam": "ask_spam",
        "d:route": "route",
        "d:tags": "tagger",
    },
    "classes": {
        "t:state": {"subject": "content", "to": "identifier"},
        "t:mail/body": "content",
        "t:mail/subject": "content",
        "q:spam": "open",
        "d:route": "open",
        "d:tags": ["open"],
    },
}

STATE: dict[str, Any] = {
    "item": "entry",
    "ref": "t:state",
    "round": 0,
    "call": None,
    "value": {"subject": "hello", "to": "dana@example.com"},
}


def call(round_: int, source: str, **members: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "item": "call",
        "round": round_,
        "sources": [source],
        "outcome": "returned",
        "snapshot": None,
        "elicitation": None,
        "prompt": None,
        "questions": {},
    }
    item.update(members)
    return item


def entry(
    ref: str, round_: int, call_: int | None, value: Any = None, **members: Any
) -> dict[str, Any]:
    item: dict[str, Any] = {"item": "entry", "ref": ref, "round": round_, "call": call_}
    item.update(members if "marker" in members else {"value": value, **members})
    return item


def round_item(n: int, cost: float = 0, deadline: bool = False) -> dict[str, Any]:
    return {"item": "round", "round": n, "cost": cost, "deadline": deadline}


ANSWER = {"kind": "noul", "labels": ["yes", "no"], "probabilities": {"yes": 0.9, "no": 0.1}}
QUESTIONS = {"q:spam": {"identity": IDENTITY, "kind": "noul", "labels": None}}

PLAN: dict[str, Any] = {
    "item": "plan",
    "kind": "execute",
    "steps": [
        {
            "tool": "archive",
            "arguments": {"folder": {"literal": "spam"}, "what": {"ref": "d:route"}},
            "requirements": [{"ref": "d:route", "value": "archive"}],
        }
    ],
    "reason": None,
    "reply": None,
    "cause": None,
    "derivation": [],
}
VERDICT: dict[str, Any] = {"item": "verdict", "kind": "passed", "failures": []}

# Indices: 0 design, 1 settings, 2 state, 3 call, 4-5 mail entries, 6 round 1, 7 call,
# 8 q:spam, 9 d:route, 10 d:tags, 11 round 2, 12 plan, 13 verdict.
NORMAL: list[dict[str, Any]] = [
    DESIGN,
    SETTINGS,
    STATE,
    call(1, "fetch_mail"),
    entry("t:mail/body", 1, 3, "text of the mail"),
    entry("t:mail/subject", 1, 3, "hello"),
    round_item(1, 2.5),
    call(2, "ask_spam", questions=QUESTIONS, prompt="Is this spam?"),
    entry("q:spam", 2, 7, ANSWER),
    entry("d:route", 2, None, "archive", derivation=[]),
    entry("d:tags", 2, None, [{"a": "x"}, None]),
    round_item(2, 1),
    PLAN,
    VERDICT,
]

BROKEN_PLAN: dict[str, Any] = {
    "item": "plan",
    "kind": "broken",
    "steps": [],
    "reason": None,
    "reply": None,
    "cause": {"code": "invalid-return", "refs": ["t:state"]},
}
BROKEN: list[dict[str, Any]] = [DESIGN, SETTINGS, BROKEN_PLAN, VERDICT]

CANCEL: dict[str, Any] = {"item": "cause", "cause": {"code": "cancelled", "refs": []}}
CANCELLED: list[dict[str, Any]] = [*NORMAL[:7], CANCEL]

# A cancellation the codec raised: delivered after the settings item, before any state entry.
CODEC_CANCELLED: list[dict[str, Any]] = [DESIGN, SETTINGS, CANCEL]

LOGS = {
    "normal": NORMAL,
    "broken": BROKEN,
    "cancelled": CANCELLED,
    "codec-cancelled": CODEC_CANCELLED,
}


def fresh(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(items)
