"""Synthetic record logs for the ledger's tests: a normal turn, a broken entry, a cancelled
turn, a turn whose codec raised a cancellation, a format-2 extraction turn and a format-3 turn
with a described label, a call's precision and warnings and an ``unsupported`` marker, built
item by item so a test can take any prefix or swap one item."""

import copy
from typing import Any

from jurat._canonical import hash_value
from jurat._values import HASH_FORMAT, MAX_DOCUMENT_DEPTH

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

EXTRACTED: dict[str, Any] = {
    "kind": "extraction",
    "value_type": "number",
    "is_list": False,
    "value": 120,
    "guess": 120,
    "p_null": 0.1,
    "confidence": 0.9,
    "span": [4, 7],
}
EXTRACTION_TREE = {
    "kind": "open",
    "value_type": "open",
    "is_list": "open",
    "value": "content",
    "guess": "content",
    "p_null": "open",
    "confidence": "open",
    "span": "content",
}
# A format-2 turn asking one extraction question, its call's truncated in UNHASHED. Indices:
# 0 design, 1 settings, 2 state, 3 call, 4 q:amount, 5 round 1, 6 plan, 7 verdict.
EXTRACTION: list[dict[str, Any]] = [
    {**DESIGN, "format": 2, "helpers": 2},
    {
        "item": "settings",
        "settings": SETTINGS["settings"],
        "providers": {"q:amount": "read"},
        "classes": {"t:state": {"body": "content"}, "q:amount": EXTRACTION_TREE},
    },
    {**STATE, "value": {"body": "Owe 120"}},
    call(
        1,
        "read",
        questions={
            "q:amount": {
                "identity": IDENTITY,
                "kind": "extraction",
                "labels": None,
                "value_type": "number",
                "is_list": False,
            }
        },
    ),
    entry("q:amount", 1, 3, EXTRACTED),
    round_item(1),
    {**PLAN, "kind": "inform", "steps": [], "reply": "big"},
    VERDICT,
]

TONE = {"kind": "choice", "labels": ["warm", "cold"], "probabilities": {"warm": 0.7, "cold": 0.3}}
UNSUPPORTED = {"kind": "Unavailable", "cause": {"code": "unsupported", "refs": ["q:other"]}}
# A format-3 turn: one call asking a choice question with a described label, answered, and a
# noul question its binding refused with unsupported, its reason under the call's warnings in
# UNHASHED. Indices: 0 design, 1 settings, 2 state, 3 call, 4 q:other, 5 q:tone, 6 round 1,
# 7 plan, 8 verdict.
DESCRIBED: list[dict[str, Any]] = [
    {**DESIGN, "format": 3, "helpers": 2},
    {
        "item": "settings",
        "settings": SETTINGS["settings"],
        "providers": {"q:other": "ask", "q:tone": "ask"},
        "classes": {"t:state": {"body": "content"}, "q:other": "open", "q:tone": "open"},
    },
    {**STATE, "value": {"body": "Thanks so much"}},
    call(
        1,
        "ask",
        precision=2,
        questions={
            "q:other": {"identity": IDENTITY, "kind": "noul", "labels": None},
            "q:tone": {
                "identity": IDENTITY,
                "kind": "choice",
                "labels": [{"name": "warm", "description": "Friendly or grateful"}, "cold"],
            },
        },
    ),
    entry("q:other", 1, 3, marker=UNSUPPORTED),
    entry("q:tone", 1, 3, TONE),
    round_item(1, 1),
    {**PLAN, "kind": "inform", "steps": [], "reply": "warm"},
    VERDICT,
]

LOGS = {
    "normal": NORMAL,
    "broken": BROKEN,
    "cancelled": CANCELLED,
    "codec-cancelled": CODEC_CANCELLED,
    "extraction": EXTRACTION,
    "described": DESCRIBED,
}

# Each log's unhashed members by log-index key, written once the keyed item is in a prefix.
UNHASHED: dict[str, dict[str, dict[str, Any]]] = {
    "extraction": {
        "calls": {
            "3": {
                "start": "2026-10-20T00:00:00Z",
                "latency": 0.25,
                "usage": None,
                "attempts": None,
                "truncated": ["q:amount"],
            }
        }
    },
    "described": {
        "calls": {
            "3": {
                "start": "2026-10-20T00:00:00Z",
                "latency": 0.5,
                "usage": None,
                "attempts": None,
                "warnings": ["q:other: this provider takes no noul question"],
            }
        }
    },
}


def unhashed_of(name: str, n: int) -> dict[str, Any]:
    """``name``'s unhashed members keyed by an item of its first ``n``."""
    members = {
        member: {k: v for k, v in keyed.items() if int(k) < n}
        for member, keyed in UNHASHED.get(name, {}).items()
    }
    return copy.deepcopy({m: keyed for m, keyed in members.items() if keyed})


def fresh(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(items)


def rehashed(doc: dict[str, Any]) -> dict[str, Any]:
    """A record document whose log a test edited, its ``hash`` recomputed from the log when its
    format has one, as a job that edits a record's bytes recomputes it."""
    log = doc["log"]
    if log and type(log[0]) is dict and log[0].get("format", 0) >= HASH_FORMAT:
        doc["hash"] = hash_value(log, MAX_DOCUMENT_DEPTH)
    else:
        doc.pop("hash", None)
    return doc
