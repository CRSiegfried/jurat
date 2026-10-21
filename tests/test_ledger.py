# pyright: reportPrivateUsage=false
"""The ledger's append, its refusals, the observer and unhashed writes."""

import asyncio
import copy
import enum
from typing import Any, cast

import pytest

from counting import SmallBounds
from jurat import InvalidRecord, read_record
from jurat._canonical import encode, hash_value
from jurat._ledger import Ledger, _LedgerError, record_prefix_hash
from ledger_logs import (
    ANSWER,
    BROKEN,
    BROKEN_PLAN,
    CANCEL,
    CANCELLED,
    DESIGN,
    LOGS,
    NORMAL,
    PLAN,
    QUESTIONS,
    SETTINGS,
    STATE,
    VERDICT,
    call,
    entry,
    fresh,
    round_item,
)


def ledger_of(items: list[dict[str, Any]], observer: Any = None) -> Ledger:
    ledger = Ledger(observer)
    for item in fresh(items):
        ledger.append(item)
    return ledger


def refused(prefix: list[dict[str, Any]], item: Any, match: str) -> None:
    """``item`` is refused after ``prefix``, and the refusal writes nothing."""
    ledger = ledger_of(prefix)
    before = copy.deepcopy(vars(ledger))
    with pytest.raises(_LedgerError, match=match):
        ledger.append(item)
    assert vars(ledger) == before


def with_member(item: dict[str, Any], **members: Any) -> dict[str, Any]:
    out = fresh([item])[0]
    out.update(members)
    return out


def without(item: dict[str, Any], name: str) -> dict[str, Any]:
    out = fresh([item])[0]
    del out[name]
    return out


HEAD = NORMAL[:3]  # design, settings, state
R1_OPEN = NORMAL[:4]  # round 1's call item, no entry yet
R1_ENTRY = NORMAL[:5]
R1_DONE = NORMAL[:7]


# ---------------------------------------------------------------------------------------------
# The kind order


@pytest.mark.parametrize(
    ("prefix", "item", "match"),
    [
        ([], SETTINGS, "design item comes first"),
        ([], STATE, "design item comes first"),
        ([DESIGN], DESIGN, "design item comes once"),
        ([DESIGN], STATE, "settings item comes second"),
        ([DESIGN], PLAN, "settings item comes second"),
        ([DESIGN, SETTINGS], SETTINGS, "settings item comes once"),
        (HEAD, DESIGN, "design item comes once"),
        (HEAD, VERDICT, "verdict item comes after the plan item"),
        ([*HEAD, PLAN], PLAN, "only the verdict item comes after the plan"),
        ([*HEAD, PLAN], CANCEL, "only the verdict item comes after the plan"),
        ([*HEAD, PLAN], entry("d:route", 0, None, "x"), "only the verdict"),
        (BROKEN, entry("d:route", 0, None, "x"), "nothing comes after the verdict"),
        (BROKEN, VERDICT, "nothing comes after the verdict"),
        (CANCELLED, CANCEL, "nothing comes after the cause"),
        (CANCELLED, PLAN, "nothing comes after the cause"),
        (HEAD, {"item": "bogus"}, "no item kind the ledger knows: 'bogus'"),
        (HEAD, {"no": "kind"}, "no item kind the ledger knows: None"),
        (HEAD, {"item": 3}, "no item kind"),
    ],
)
def test_items_out_of_kind_order_are_refused(
    prefix: list[dict[str, Any]], item: dict[str, Any], match: str
) -> None:
    refused(prefix, item, match)


@pytest.mark.parametrize(
    "item",
    [
        entry("d:route", 0, None, "x"),
        call(1, "fetch_mail"),
        round_item(1),
        {"item": "certificate", "round": 1},
        with_member(STATE, round=1),
    ],
)
def test_after_settings_comes_the_state_entry_in_round_0_or_the_plan(item: dict[str, Any]) -> None:
    refused([DESIGN, SETTINGS], item, "after the settings item comes the t:state entry")


@pytest.mark.parametrize("prefix", [R1_OPEN, R1_ENTRY], ids=["after a call", "after an entry"])
def test_a_cause_item_may_end_a_round_with_no_round_item(prefix: list[dict[str, Any]]) -> None:
    # A cancellation a sync component raised keeps every item before it.
    record = ledger_of([*prefix, CANCEL]).close()
    assert record.cause == {"code": "cancelled", "refs": []}
    refused([*prefix, CANCEL], round_item(1), "nothing comes after the cause item")


def test_a_cause_item_may_follow_the_settings_item_when_a_sync_component_cancelled() -> None:
    record = ledger_of([DESIGN, SETTINGS, CANCEL]).close()
    assert record.cause == {"code": "cancelled", "refs": []}
    refused([DESIGN, SETTINGS, CANCEL], PLAN, "nothing comes after the cause item")


def test_nothing_is_appended_after_close() -> None:
    ledger = ledger_of(BROKEN)
    ledger.close()
    with pytest.raises(_LedgerError, match="closed"):
        ledger.append(DESIGN)
    with pytest.raises(_LedgerError, match="closed"):
        ledger.write_unhashed("engine", "jurat 0.0.1")


def test_a_cause_item_after_a_whole_round_is_the_last_item() -> None:
    ledger = ledger_of(CANCELLED)
    assert ledger.close().log[-1] == CANCEL


# ---------------------------------------------------------------------------------------------
# Rounds

CERT = {"item": "certificate", "round": 2}


@pytest.mark.parametrize(
    ("prefix", "item", "match"),
    [
        (HEAD, call(0, "fetch_mail"), "round 0 has no call item"),
        (HEAD, round_item(0), "round 0 has no round item"),
        (HEAD, {"item": "certificate", "round": 0}, "round 0 has no certificate item"),
        (R1_ENTRY, entry("d:route", 0, None, "x"), r"round 0 is below the last round, 1"),
        (R1_DONE, call(1, "ask_spam"), "round 1 holds an item after its round item"),
        (R1_DONE, round_item(1), "round 1 holds two round items"),
        (R1_DONE, entry("d:route", 1, None, "x"), "round 1 holds an item after its round"),
        (R1_DONE, {"item": "certificate", "round": 1}, "round 1 holds an item after its round"),
        (R1_DONE, {"item": "certificate", "round": 0}, "round 0 has no certificate"),
        (HEAD, call(2, "fetch_mail"), "round 2 skips a number after round 0"),
        (R1_DONE, round_item(3), "round 3 skips a number after round 1"),
        (R1_DONE, {"item": "certificate", "round": 3}, "skips a number"),
        (R1_DONE, entry("d:route", 3, None, "x"), "skips a number"),
        (R1_ENTRY, call(2, "ask_spam"), "round 1 has no round item before it"),
        (R1_ENTRY, entry("d:route", 2, None, "x"), "round 1 has no round item before it"),
        (R1_ENTRY, PLAN, "round 1 has no round item before it"),
        (R1_OPEN, {"item": "certificate", "round": 1}, "certificate item comes first"),
        (R1_ENTRY, call(1, "ask_spam"), "call item comes before the entries of its round"),
        (R1_OPEN, entry("t:mail/body", 1, 0, "x"), "its call 0 names no call item of round 1"),
        (R1_OPEN, entry("t:mail/body", 1, 4, "x"), "its call 4 names no call item"),
        (
            [*R1_DONE, call(2, "ask_spam")],
            entry("t:x", 2, 3, "x"),
            "its call 3 names no call item of round 2",
        ),
        (HEAD, entry("d:route", 0, 1, "x"), "names no call item of round 0"),
    ],
)
def test_items_out_of_round_order_are_refused(
    prefix: list[dict[str, Any]], item: dict[str, Any], match: str
) -> None:
    refused(prefix, item, match)


def test_a_certificate_item_opens_its_round() -> None:
    ledger = ledger_of(
        [*R1_DONE, CERT, call(2, "ask_spam", questions=QUESTIONS), entry("q:spam", 2, 8, ANSWER)]
    )
    ledger.append(round_item(2))
    assert ledger.close().log[7] == CERT


def test_a_round_may_hold_only_its_round_item() -> None:
    ledger = ledger_of([*R1_DONE, round_item(2), PLAN, VERDICT])
    assert ledger.close().log[7] == round_item(2)


CUT = [*R1_OPEN, NORMAL[4], NORMAL[5], round_item(1, deadline=True)]


@pytest.mark.parametrize(
    "item",
    [
        call(2, "ask_spam"),
        entry("d:route", 2, None, "x"),
        round_item(2),
        {"item": "certificate", "round": 2},
    ],
)
def test_an_item_of_a_round_after_the_deadlines_round_is_refused(item: dict[str, Any]) -> None:
    refused(CUT, item, "round 2 comes after round 1, which the deadline cut")


def test_the_deadlines_round_is_followed_by_the_plan() -> None:
    ledger = ledger_of([*CUT, PLAN, VERDICT])
    assert ledger.close().log[6]["deadline"] is True


def test_a_round_after_one_of_cost_0_and_deadline_false_is_accepted() -> None:
    # The budget's round item is one a round that asked nothing writes too.
    ledger = ledger_of([*R1_DONE, round_item(2), entry("d:route", 3, None, "x"), round_item(3)])
    assert ledger.close().log[-1] == round_item(3)


def bounded(n: int) -> list[dict[str, Any]]:
    settings = copy.deepcopy(SETTINGS)
    settings["settings"]["round_bound"] = n
    return [DESIGN, settings, STATE]


@pytest.mark.parametrize(
    "item",
    [
        call(2, "ask_spam"),
        entry("d:route", 2, None, "x"),
        round_item(2),
        {"item": "certificate", "round": 2},
    ],
)
def test_an_item_of_a_round_above_the_round_bound_is_refused(item: dict[str, Any]) -> None:
    prefix = [*bounded(1), round_item(1)]
    refused(prefix, item, "round 2 is above the round bound, 1")


def test_items_up_to_the_round_bound_and_round_0_under_a_bound_of_1_are_accepted() -> None:
    log = [*bounded(1), entry("d:route", 0, None, "x"), round_item(1), PLAN, VERDICT]
    assert len(ledger_of(log).close().log) == 7


def test_read_record_refuses_a_round_after_the_cut() -> None:
    log = [*CUT, entry("d:route", 2, None, "x"), round_item(2), PLAN, VERDICT]
    document: dict[str, Any] = {"log": log, "unhashed": {}}
    with pytest.raises(InvalidRecord, match="which the deadline cut"):
        read_record(fresh([document])[0])


def test_a_round_from_1_may_follow_round_0_with_no_round_item_for_0() -> None:
    ledger = ledger_of([*HEAD, entry("d:route", 0, None, "x"), round_item(1), PLAN])
    assert len(ledger.close().log) == 6


# ---------------------------------------------------------------------------------------------
# Entries: write once, providers, contiguity, class trees


@pytest.mark.parametrize(
    ("prefix", "item", "match"),
    [
        (R1_ENTRY, entry("t:mail/body", 1, 3, "y"), "second entry for 't:mail/body'"),
        (HEAD, with_member(STATE, round=0), "second entry for 't:state'"),
        (R1_OPEN, entry("d:other", 1, None, "x"), "'d:other' is not a key of the settings' prov"),
        (R1_OPEN, entry("x:bad", 1, None, "x"), "does not start with q:, t: or d:"),
        (R1_OPEN, entry("t:mail/body.f", 1, 3, "x"), "'t:mail/body.f' is not a key"),
        (R1_OPEN, entry("t:state.subject", 1, 3, "x"), "'t:state.subject' is not a key"),
        (
            [*R1_ENTRY, entry("d:route", 1, None, "x")],
            entry("t:mail/subject", 1, 3, "x"),
            "'t:mail/subject' is not contiguous",
        ),
        (
            [*R1_OPEN, entry("t:mail/subject", 1, 3, "x")],
            entry("t:mail/body", 1, 3, "x"),
            "'t:mail/body' is not after 't:mail/subject' in code-point order",
        ),
        (HEAD, entry("d:route", 0, None, {"a": 1}), None),
    ],
)
def test_entries_out_of_order_are_refused(
    prefix: list[dict[str, Any]], item: dict[str, Any], match: str | None
) -> None:
    if match is None:  # an open class fits anything
        ledger_of([*prefix, item])
    else:
        refused(prefix, item, match)


def test_write_once_is_checked_before_the_node_order() -> None:
    # A repeat of the node's last ref would also break code-point order; I2 names it.
    refused(R1_ENTRY, entry("t:mail/body", 1, 3, "x"), "written once")


# ---------------------------------------------------------------------------------------------
# Sources, the state and answers as a turn writes them

MARKER = {"kind": "Unavailable", "cause": {"code": "timed-out", "refs": ["q:spam"]}}
R2 = [*R1_DONE, call(2, "ask_spam", questions=QUESTIONS)]  # round 2's call is item 7


@pytest.mark.parametrize(
    ("prefix", "item", "match"),
    cast(
        "list[Any]",
        [
            (R1_DONE, call(2, "s", sources=[]), "'sources' is empty"),
            (R1_DONE, call(2, "s", sources=["s", 1]), "'sources' is not a list of strings"),
            (R1_DONE, call(2, "s", sources=["b", "a"]), "'b', 'a' are not in code-point order"),
            (R1_DONE, call(2, "s", sources=["a", "a"]), "'a', 'a' are not in code-point order"),
            (R1_DONE, call(2, "s", sources=["a", "Z"]), "'a', 'Z' are not in code-point order"),
            # Code-point order, not UTF-16's, which puts a surrogate pair before U+FFFF.
            (R1_DONE, call(2, "s", sources=["\U0001f600", "\uffff"]), "are not in code-point"),
            (R1_DONE, call(2, "fetch_mail"), "'fetch_mail', which call item 3 lists"),
            (R1_DONE, call(2, "s", sources=["a", "fetch_mail"]), "which call item 3 lists"),
            (
                [*R1_DONE, call(2, "a"), call(2, "b")],
                call(2, "s", sources=["b", "c"]),
                "list 'b', which call item 8 lists",
            ),
            (
                [DESIGN, SETTINGS],
                with_member(without(STATE, "value"), marker=MARKER),
                "a t:state entry holding a marker",
            ),
            (R2, entry("q:spam", 2, 7, None), "'value' is not an object"),
            (R2, entry("q:spam", 2, 7, "yes"), "'value' is not an object"),
            (R2, entry("q:spam", 2, 7, [ANSWER]), "'value' is not an object"),
            (R2, entry("q:spam", 2, 7, without(ANSWER, "probabilities")), "no 'probab"),
            (R2, entry("q:spam", 2, 7, without(ANSWER, "labels")), "has no 'labels'"),
            (R2, entry("q:spam", 2, 7, without(ANSWER, "kind")), "has no 'kind'"),
            (
                R2,
                entry("q:spam", 2, 7, {**ANSWER, "snapshot": None}),
                "'value' has the member 'snapshot', which its kind does not name",
            ),
            (
                R2,
                entry("q:spam", 2, 7, {**ANSWER, "kind": "yesno"}),
                "kind 'yesno' is not noul, choice or score",
            ),
            (R2, entry("q:spam", 2, 7, {**ANSWER, "kind": 1}), "'kind' is not a string"),
            (
                R2,
                entry("q:spam", 2, 7, {**ANSWER, "labels": "yes"}),
                "'labels' is not a list of strings",
            ),
            (
                R2,
                entry("q:spam", 2, 7, {**ANSWER, "labels": ["yes", None]}),
                "'labels' is not a list of strings",
            ),
            (
                R2,
                entry("q:spam", 2, 7, {**ANSWER, "probabilities": [0.9, 0.1]}),
                "'probabilities' is not an object",
            ),
        ],
    ),
)
def test_what_no_turn_writes_is_refused(
    prefix: list[dict[str, Any]], item: dict[str, Any], match: str
) -> None:
    refused(prefix, item, match)
    with pytest.raises(InvalidRecord, match=match) as raised:
        read_record({"log": fresh([*prefix, item]), "unhashed": {}})
    assert type(raised.value.__cause__) is _LedgerError


@pytest.mark.parametrize(
    ("prefix", "item"),
    cast(
        "list[Any]",
        [
            (R1_DONE, call(2, "s", sources=["Z", "a", "\u00e9"])),
            (R1_DONE, call(2, "s", sources=["ask_spam", "s"])),
            (R1_DONE, call(2, "s", sources=["\uffff", "\U0001f600"])),
            ([*R1_DONE, call(2, "a")], call(2, "b")),
            ([DESIGN, SETTINGS], STATE),
            (R1_OPEN, entry("t:mail/body", 1, 3, marker=MARKER)),
            (R2, entry("q:spam", 2, 7, marker=MARKER)),
            # No probability is tested: interpret's answer check does that under replay.
            (R2, entry("q:spam", 2, 7, {**ANSWER, "probabilities": {}})),
            (R2, entry("q:spam", 2, 7, {**ANSWER, "probabilities": {"x": "high"}})),
            (
                R2,
                entry("q:spam", 2, 7, {"kind": "score", "labels": [], "probabilities": {}}),
            ),
            (R2, entry("q:spam", 2, 7, {**ANSWER, "kind": "choice", "labels": ["a"]})),
            (R1_ENTRY, entry("t:mail/subject", 1, 3, "not an answer")),
        ],
    ),
)
def test_its_accepted_neighbours_are_kept(
    prefix: list[dict[str, Any]], item: dict[str, Any]
) -> None:
    ledger = ledger_of([*prefix, item])
    assert read_record(ledger.close().document).log == ledger.close().log


def test_a_refused_call_item_adds_no_source() -> None:
    ledger = ledger_of(R1_DONE)
    with pytest.raises(_LedgerError, match="not in code-point order"):
        ledger.append(call(2, "s", sources=["s", "a"]))
    with pytest.raises(_LedgerError, match="'fetch_mail', which call item 3 lists"):
        ledger.append(call(2, "s", sources=["a", "fetch_mail"]))
    ledger.append(call(2, "a"))
    ledger.append(call(2, "s"))
    with pytest.raises(_LedgerError, match="'s', which call item 8 lists"):
        ledger.append(call(2, "s"))


GROUNDED = 'd:route["a"]'
GROUNDED_CAUSE = {"code": "raised", "refs": ["route", GROUNDED]}


def grounded_plan(**step: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"tool": "t", "arguments": {}, "requirements": []}
    return {**PLAN, "steps": [{**base, **step}]}


GROUNDED_FAILURE = {"step": 1, "requirement": {"ref": GROUNDED, "value": 1}, "ref": GROUNDED}


@pytest.mark.parametrize(
    ("prefix", "item"),
    [
        (R1_OPEN, entry('t:mail/body["a"]', 1, 3, "x")),
        (HEAD, entry("d:route", 0, None, marker={"kind": "Unavailable", "cause": GROUNDED_CAUSE})),
        (R1_DONE, call(2, "s", outcome={"failed": GROUNDED_CAUSE})),
        (HEAD, {**PLAN, "kind": "broken", "steps": [], "cause": GROUNDED_CAUSE}),
        (HEAD, grounded_plan(arguments={"x": {"ref": GROUNDED}})),
        (HEAD, grounded_plan(requirements=[{"ref": GROUNDED, "value": 1}])),
        ([*HEAD, PLAN], {**VERDICT, "kind": "refused", "failures": [GROUNDED_FAILURE]}),
    ],
)
def test_a_grounded_ref_is_refused_wherever_redaction_would_rewrite_one(
    prefix: list[dict[str, Any]], item: dict[str, Any]
) -> None:
    # The v0.0.1 record does not name a grounded entry's set, so redact could not digest
    # one.
    refused(prefix, item, "is a grounded ref")


@pytest.mark.parametrize(
    "origin",
    [
        {"literal": "x", "set": "d:route"},
        {"literal": "x", "set": None},
        {"ref": "d:route", "set": "d:route"},
    ],
    ids=["a literal's set", "a null set", "a ref's set"],
)
def test_an_origin_carrying_set_is_refused(origin: dict[str, Any]) -> None:
    # Nothing checks a candidate-value literal's set in v0.0.1, so redact could find no tree
    # it fits; the key is refused whatever it holds.
    refused(HEAD, grounded_plan(arguments={"x": origin}), "carries set")


def test_a_grounded_ref_elsewhere_and_a_role_or_rule_name_in_a_cause_are_kept() -> None:
    # A question key is digested whole, and a cause's role names and rule:N are no refs.
    ledger_of([*R1_DONE, call(2, "s", questions={'q:spam["x"]': QUESTIONS["q:spam"]})])
    cause = {"code": "invalid-return", "refs": ["policy", "rule:2", "step:1", 'x["a"]']}
    ledger_of([*HEAD, {**PLAN, "kind": "broken", "steps": [], "cause": cause}])


def settings_with(**classes: Any) -> dict[str, Any]:
    out = fresh([SETTINGS])[0]
    out["classes"].update(classes)
    for ref in classes:
        out["providers"].setdefault(ref, "other")
    return out


@pytest.mark.parametrize(
    ("tree", "value", "match"),
    [
        ("secret", "x", "class tree of 'd:x' is malformed"),
        (["open", "open"], {}, "malformed"),
        ([], {}, "malformed"),
        ({"a": "nope"}, {}, "malformed"),
        (3, "x", "malformed"),
        (None, "x", "has no class tree"),
        ({"a": "open"}, {"b": 1}, r"the value has the key 'b', which its class tree does not"),
        ({"a": "open"}, "x", "the value is not an object"),
        ({"a": {"b": "open"}}, {"a": {"c": 1}}, r"the value\['a'\] has the key 'c'"),
        (["content"], "x", "not an object"),
        (["content"], ["x"], r"the value\[0\] is not an object"),
        ([{"b": "open"}], {"k": {"c": 1}}, r"the value\['k'\] has the key 'c'"),
        ({"a": "open"}, [{"a": 1}, {"z": 1}], r"the value\[1\] has the key 'z'"),
    ],
)
def test_a_value_must_fit_its_refs_class_tree(tree: Any, value: Any, match: str) -> None:
    settings = settings_with(**{"d:x": tree})
    if tree is None:
        del settings["classes"]["d:x"]
    refused([DESIGN, settings, STATE], entry("d:x", 0, None, value), match)


@pytest.mark.parametrize(
    ("tree", "value"),
    cast(
        "list[Any]",
        [
            ({"a": "open"}, None),
            ({"a": "open"}, {"a": None}),
            ({"a": "open"}, [{"a": 1}, None, [{}]]),
            ({"a": {"b": "content"}}, {"a": {"b": [1, {"x": 2}]}}),
            (["identifier"], {"k": "v", "j": [1]}),
            ([{"b": "open"}], {"k": {"b": 1}}),
            ("content", {"anything": [1, 2]}),
            ({}, {}),
        ],
    ),
)
def test_values_that_fit_their_class_tree_are_kept(tree: Any, value: Any) -> None:
    ledger_of([DESIGN, settings_with(**{"d:x": tree}), STATE, entry("d:x", 0, None, value)])


def test_a_marker_entry_is_not_fit_to_its_tree_but_its_tree_must_be_well_formed() -> None:
    marker: dict[str, Any] = {"kind": "Unsure", "cause": {"code": "abstained", "refs": []}}
    ok = settings_with(**{"d:x": {"a": "open"}})
    ledger_of([DESIGN, ok, STATE, entry("d:x", 0, None, marker=marker)])
    bad = settings_with(**{"d:x": "nope"})
    refused([DESIGN, bad, STATE], entry("d:x", 0, None, marker=marker), "malformed")


def test_the_state_entry_is_its_own_node_and_fits_its_tree() -> None:
    refused([DESIGN, SETTINGS], with_member(STATE, value={"other": 1}), "the key 'other'")


# ---------------------------------------------------------------------------------------------
# Members fit their kind


def nested(depth: int) -> Any:
    value: Any = "leaf"
    for _ in range(depth):
        value = [value]
    return value


DEEP64, DEEP65 = nested(64), nested(65)
FAIL = {"step": 1, "requirement": None, "ref": None}
Q_ENTRY = entry("q:spam", 2, 7, ANSWER)


def question(**members: Any) -> dict[str, Any]:
    q = {"identity": "id", "kind": "noul", "labels": None}
    q.update(members)
    return {"q:spam": q}


@pytest.mark.parametrize(
    ("prefix", "item", "match"),
    cast(
        "list[Any]",
        [
            (HEAD, ["item", "entry"], "the item is not an object"),
            (HEAD, {"item": "entry", "value": {1, 2}}, "outside the value domain: set"),
            (HEAD, {"item": "entry", "value": b"x"}, "outside the value domain: bytes"),
            (HEAD, {"item": "entry", "value": float("nan")}, "not finite"),
            # entries
            (R2, without(Q_ENTRY, "call"), r"\(entry\) has no 'call' member"),
            (R2, with_member(Q_ENTRY, extra=1), "has the member 'extra', which its kind does not"),
            (R2, with_member(Q_ENTRY, round=True), r"'round' is not an integer"),
            (R2, with_member(Q_ENTRY, round=2.0), r"'round' is not an integer"),
            (R2, with_member(Q_ENTRY, call="7"), r"'call' is not an integer or null"),
            (R2, with_member(Q_ENTRY, ref=7), r"'ref' is not a string"),
            (R2, with_member(Q_ENTRY, marker=MARKER), "both value and marker, or neither"),
            (R2, without(Q_ENTRY, "value"), "both value and marker, or neither"),
            (R2, with_member(Q_ENTRY, rows={}), r"'rows' is not a list"),
            (R2, with_member(Q_ENTRY, derivation="x"), r"'derivation' is not a list"),
            (
                R2,
                entry("q:spam", 2, 7, marker={**MARKER, "kind": "Maybe"}),
                "kind 'Maybe' is not Unsure, Unavailable or Unasked",
            ),
            (
                R2,
                entry("q:spam", 2, 7, marker={**MARKER, "cause": {"code": "nope", "refs": []}}),
                "code 'nope' is not in the closed cause set",
            ),
            (
                R2,
                entry("q:spam", 2, 7, marker={**MARKER, "cause": {"code": "raised", "refs": [1]}}),
                "'refs' is not a list of strings",
            ),
            (R2, entry("q:spam", 2, 7, marker={"kind": "Unsure"}), "'marker' has no 'cause'"),
            # call items
            (R1_DONE, call(2, "s", outcome="ok"), 'neither "returned" nor'),
            (R1_DONE, call(2, "s", outcome={"failed": None}), "'failed' is not an object"),
            (R1_DONE, call(2, "s", outcome={"failed": MARKER["cause"], "x": 1}), "neither"),
            (
                R1_DONE,
                call(2, "s", outcome={"failed": {"code": "bad", "refs": []}}),
                "not in the closed cause set",
            ),
            (R1_DONE, call(2, "s", sources="s"), "'sources' is not a list of strings"),
            (R1_DONE, call(2, "s", prompt=1), "'prompt' is not a string or null"),
            (R1_DONE, without(call(2, "s"), "snapshot"), "has no 'snapshot'"),
            (R1_DONE, call(2, "s", questions=[]), "'questions' is not an object"),
            (R1_DONE, call(2, "s", questions={"t:x": {}}), "key 't:x', which is not a q: ref"),
            (R1_DONE, call(2, "s", questions={"spam": {}}), "key 'spam', which is not a q: ref"),
            (R1_DONE, call(2, "s", questions=question(kind="yesno")), "not noul, choice or score"),
            (R1_DONE, call(2, "s", questions=question(identity=1)), "'identity' is not a string"),
            (R1_DONE, call(2, "s", questions=question(labels=["yes"])), "not null, as noul's"),
            (
                R1_DONE,
                call(2, "s", questions=question(kind="choice", labels=None)),
                "labels is not a list of strings",
            ),
            (
                R1_DONE,
                call(
                    2,
                    "s",
                    questions=question(kind="score", labels={"fixed": [], "candidates": "d"}),
                ),
                "only a choice's may be",
            ),
            (
                R1_DONE,
                call(2, "s", questions=question(kind="choice", labels={"fixed": ["a"]})),
                "has no 'candidates'",
            ),
            (R1_DONE, call(2, "s", questions={"q:spam": {"kind": "noul"}}), "has no 'identity'"),
            # round and certificate items
            (
                R1_ENTRY,
                {"item": "round", "round": 1, "cost": "1", "deadline": False},
                "not a number",
            ),
            (R1_ENTRY, {"item": "round", "round": 1, "cost": 1, "deadline": 0}, "not a boolean"),
            (R1_DONE, {"item": "certificate", "round": 2, "cap": 1}, "member 'cap'"),
            # design and settings
            ([], {**DESIGN, "format": "1"}, "'format' is not an integer"),
            ([], {**DESIGN, "helpers": True}, "'helpers' is not an integer"),
            ([], {**DESIGN, "hash": None}, "'hash' is not a string"),
            ([DESIGN], {**SETTINGS, "providers": []}, "'providers' is not an object"),
            ([DESIGN], {**SETTINGS, "classes": None}, "'classes' is not an object"),
            ([DESIGN], {**SETTINGS, "settings": {}}, "has no 'round_bound'"),
            (
                [DESIGN],
                {**SETTINGS, "settings": {**SETTINGS["settings"], "round_bound": None}},
                "'round_bound' is not an integer",
            ),
            (
                [DESIGN],
                {**SETTINGS, "settings": {**SETTINGS["settings"], "budget": "x"}},
                "'budget' is not a number or null",
            ),
            (
                [DESIGN],
                {**SETTINGS, "settings": {**SETTINGS["settings"], "retry": 1}},
                "member 'retry'",
            ),
            # plan
            (HEAD, without(PLAN, "reason"), "has no 'reason'"),
            (HEAD, {**PLAN, "kind": 1}, "'kind' is not a string"),
            (HEAD, {**PLAN, "kind": "maybe"}, "kind 'maybe' is not a plan kind"),
            (HEAD, {**PLAN, "kind": "Execute"}, "is not a plan kind"),
            ([*HEAD, PLAN], {**VERDICT, "kind": "maybe"}, "kind 'maybe' is not passed or refused"),
            (
                [*HEAD, PLAN],
                {**VERDICT, "kind": "maybe", "failures": [FAIL]},
                "is not passed or refused",
            ),
            (
                R1_DONE,
                call(
                    2,
                    "s",
                    questions={"q:spam.f": {"identity": "i", "kind": "noul", "labels": None}},
                ),
                "key 'q:spam.f', which has a field path",
            ),
            (
                HEAD,
                {**PLAN, "steps": [{**PLAN["steps"][0], "arguments": {"a": {"literal": DEEP65}}}]},
                r"'a'\]'s literal is outside the value domain: the value nests more than 64 deep",
            ),
            (
                HEAD,
                {
                    **PLAN,
                    "steps": [
                        {**PLAN["steps"][0], "requirements": [{"ref": "d:x", "value": DEEP65}]}
                    ],
                },
                "'value' is outside the value domain: the value nests more than 64 deep",
            ),
            (HEAD, entry("d:route", 0, None, DEEP65), "'value' is outside the value domain"),
            (HEAD, {**PLAN, "steps": [{"tool": "t", "arguments": {}}]}, "has no 'requirements'"),
            (
                HEAD,
                {**PLAN, "steps": [{"tool": "t", "arguments": [], "requirements": []}]},
                "'arguments' is not an object",
            ),
            (
                HEAD,
                {**PLAN, "steps": [{"tool": "t", "arguments": {}, "requirements": [["d:x", 1]]}]},
                r"'requirements'\[0\] is not an object",
            ),
            (
                HEAD,
                {**PLAN, "steps": [{"tool": "t", "arguments": {}, "requirements": [{"ref": 1}]}]},
                "has no 'value'",
            ),
            (HEAD, {**PLAN, "cause": {"code": "raised"}}, "has no 'refs'"),
            (HEAD, {**PLAN, "slots": []}, "'slots' is not an object"),
            (HEAD, {**PLAN, "label": "x"}, "member 'label'"),
            # verdict
            (
                [*HEAD, PLAN],
                {**VERDICT, "failures": [FAIL]},
                "not empty exactly when its kind is passed",
            ),
            ([*HEAD, PLAN], {**VERDICT, "kind": "refused"}, "not empty exactly when"),
            (
                [*HEAD, PLAN],
                {**VERDICT, "kind": "refused", "failures": [{**FAIL, "step": "x"}]},
                'neither an integer of at least 1 nor "plan"',
            ),
            (
                [*HEAD, PLAN],
                {**VERDICT, "kind": "refused", "failures": [{**FAIL, "step": True}]},
                "neither an integer",
            ),
            (
                [*HEAD, PLAN],
                {**VERDICT, "kind": "refused", "failures": [{**FAIL, "ref": 1}]},
                "'ref' is not a string or null",
            ),
            (
                [*HEAD, PLAN],
                {
                    **VERDICT,
                    "kind": "refused",
                    "failures": [{**FAIL, "requirement": {"ref": "d:x"}}],
                },
                "has no 'value'",
            ),
            (
                [*HEAD, PLAN],
                {**VERDICT, "kind": "refused", "failures": [{"step": 1, "ref": None}]},
                "has no 'requirement'",
            ),
            # cause
            (R1_DONE, {"item": "cause", "cause": None}, "'cause' is not an object"),
            (R1_DONE, {"item": "cause"}, "has no 'cause'"),
            (
                R1_DONE,
                {"item": "cause", "cause": {"code": "raised", "refs": []}},
                "cause is not .*cancelled",
            ),
            (
                R1_DONE,
                {"item": "cause", "cause": {"code": "cancelled", "refs": ["t:state"]}},
                "cause is not .*cancelled",
            ),
        ],
    ),
)
def test_members_that_do_not_fit_their_kind_are_refused(
    prefix: list[dict[str, Any]], item: Any, match: str
) -> None:
    refused(prefix, item, match)


@pytest.mark.parametrize(
    "item",
    [
        call(2, "s", questions=question(kind="choice", labels=["a", "b"])),
        call(2, "s", questions=question(kind="choice", labels={"fixed": [], "candidates": "d:c"})),
        call(2, "s", questions=question(kind="score", labels=["low", "high"])),
        call(2, "s", questions={'q:spam["x"]': {"identity": "i", "kind": "noul", "labels": None}}),
        call(2, "s", outcome={"failed": {"code": "timed-out", "refs": []}}),
        {"item": "certificate", "round": 2},
    ],
)
def test_items_that_fit_their_kind_are_kept(item: dict[str, Any]) -> None:
    ledger_of([*R1_DONE, item])


REQ = {"ref": "d:route", "value": "x"}


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ({**FAIL, "step": 0}, 'neither an integer of at least 1 nor "plan"'),
        ({**FAIL, "step": -2}, "at least 1"),
        ({**FAIL, "step": 1.0}, "neither an integer"),
        ({"step": 1, "requirement": REQ, "ref": None}, "not null together"),
        ({"step": 1, "requirement": None, "ref": "d:route"}, "not null together"),
        ({"step": "plan", "requirement": None, "ref": None}, 'beside the step "plan"'),
        ({"step": 1, "requirement": REQ, "ref": "d:other"}, "not its requirement's ref"),
        ({"step": "plan", "requirement": REQ, "ref": "d:other"}, "not its requirement's ref"),
    ],
)
def test_a_failure_the_record_views_could_not_build_is_refused(
    failure: dict[str, Any], match: str
) -> None:
    # The ledger's failure checks.
    refused([*HEAD, PLAN], {**VERDICT, "kind": "refused", "failures": [failure]}, match)


def test_a_refused_verdict_lists_its_failures() -> None:
    failures = [
        {"step": "plan", "requirement": {"ref": "d:route", "value": "x"}, "ref": "d:route"},
        {"step": 1, "requirement": None, "ref": None},
    ]
    ledger = ledger_of([*HEAD, PLAN, {**VERDICT, "kind": "refused", "failures": failures}])
    assert ledger.close().log[-1]["failures"] == failures


def test_a_refusal_writes_nothing_and_the_ledger_takes_the_next_item() -> None:
    ledger = ledger_of(HEAD)
    with pytest.raises(_LedgerError):
        ledger.append(entry("d:route", 0, None, "x", extra=1))
    assert ledger.append(entry("d:route", 0, None, "x")) == 3


# ---------------------------------------------------------------------------------------------
# The copy


class Colour(enum.StrEnum):
    red = "red"


def test_a_change_to_an_item_after_its_append_does_not_change_the_log() -> None:
    ledger = ledger_of(HEAD)
    item = entry("d:route", 0, None, {"k": ["v"]})
    ledger.append(item)
    item["value"]["k"].append("changed")
    item["ref"] = "d:tags"
    record = ledger.close()
    assert record.log[3] == entry("d:route", 0, None, {"k": ["v"]})
    assert ledger._items[3]["value"] == {"k": ["v"]}


def test_tuples_and_str_subclasses_are_stored_as_plain_lists_and_strings() -> None:
    seen: list[Any] = []
    ledger = ledger_of(HEAD, lambda i, item: seen.append(item))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType, reportUnknownMemberType]
    ledger.append(entry("d:route", 0, None, (Colour.red, {Colour.red: (1, 2)})))
    value = seen[-1]["value"]
    assert value == ["red", {"red": [1, 2]}]
    assert [type(value), type(value[0]), type(value[1])] == [list, str, dict]
    assert [type(k) for k in value[1]] == [str]
    assert type(value[1]["red"]) is list
    assert ledger.close().log[3]["value"] == ["red", {"red": [1, 2]}]


def test_an_int_subclass_is_refused_not_coerced() -> None:
    class Level(enum.IntEnum):
        one = 1

    refused(HEAD, entry("d:route", 0, None, Level.one), "outside the value domain")


def test_the_bytes_are_the_canonical_encoding_of_the_copy() -> None:
    ledger = ledger_of(NORMAL)
    assert ledger._bytes == [encode(item) for item in NORMAL]
    assert ledger.prefix_hash(len(NORMAL)) == hash_value(NORMAL)
    assert ledger.prefix_hash(3) == hash_value(NORMAL[:3])


# ---------------------------------------------------------------------------------------------
# The observer


def test_the_observer_sees_each_item_once_in_order_with_its_final_index() -> None:
    seen: list[tuple[int, Any]] = []

    def observer(index: int, item: dict[str, Any]) -> None:
        seen.append((index, item))

    ledger = Ledger(observer)
    for i, item in enumerate(fresh(NORMAL)):
        assert ledger.append(item) == i
        if item["item"] == "call":
            ledger.write_unhashed(
                "calls", {"start": "t", "latency": 1, "usage": None, "attempts": 1}, i
            )
    ledger.write_unhashed("engine", "jurat")
    assert [i for i, _ in seen] == list(range(len(NORMAL)))
    assert [item for _, item in seen] == NORMAL
    assert all(a is b for (_, a), b in zip(seen, ledger._items, strict=True))


def test_an_observer_that_changes_its_copy_changes_neither_bytes_nor_later_checks() -> None:
    def vandal(index: int, item: dict[str, Any]) -> None:
        if item["item"] == "settings":
            item["providers"].clear()
            item["classes"].clear()
        if item["item"] == "call":
            item["questions"].clear()
        item["item"] = "vandalised"

    plain = ledger_of(NORMAL).close()
    ledger = ledger_of(NORMAL, vandal)
    # unhashed keys are checked against the kinds the ledger kept, not the changed copies.
    ledger.write_unhashed("calls", CALL, 3)
    ledger.write_unhashed("calls", {**CALL, "sums": {"q:spam": 0.5}}, 7)  # questions kept
    ledger.write_unhashed("exceptions", EXC, 12)
    ledger.write_unhashed("reads", "t:x", 9)
    record = ledger.close()
    assert record.log == NORMAL
    assert record.record_hash == plain.record_hash
    assert set(record.unhashed) == {"calls", "exceptions", "reads"}
    # The close-time re-encode test flags every copy the observer changed.
    assert ledger.drifted_copies == tuple(range(len(NORMAL)))
    assert ledger_of(NORMAL).drifted_copies == ()


def test_a_drifted_copy_outside_the_domain_is_flagged_not_raised() -> None:
    def vandal(index: int, item: dict[str, Any]) -> None:
        if index == 2:
            item["value"] = {1, 2}

    ledger = ledger_of(HEAD, vandal)
    ledger.close()
    assert ledger.drifted_copies == (2,)


@pytest.mark.parametrize("raised", [KeyboardInterrupt, asyncio.CancelledError, ValueError])
def test_the_observers_exception_is_marked_and_propagates_unchanged(
    raised: type[BaseException],
) -> None:
    exc = raised("from the observer")

    def observer(index: int, item: dict[str, Any]) -> None:
        if index == 2:
            raise exc

    ledger = ledger_of(HEAD[:2], observer)
    with pytest.raises(raised) as info:
        ledger.append(fresh([STATE])[0])
    assert info.value is exc
    assert type(info.value) is raised
    assert vars(info.value) == {}  # no attribute set on it
    assert ledger.observer_error is exc
    # The item is in the log with its final index, and the ledger closes.
    record = ledger.close()
    assert record.log == HEAD


def test_the_observer_raising_at_the_cause_items_append_keeps_that_item() -> None:
    raised = KeyboardInterrupt("at the cause item")

    def observer(index: int, item: dict[str, Any]) -> None:
        if item["item"] == "cause":
            raise raised

    ledger = ledger_of(R1_DONE, observer)
    with pytest.raises(KeyboardInterrupt) as info:
        ledger.append(fresh([CANCEL])[0])
    assert info.value is raised
    assert ledger.observer_error is raised
    log = ledger.close().log
    assert log == CANCELLED
    assert log[-1] == CANCEL


def test_the_observers_mark_is_set_once() -> None:
    first, second = ValueError("first"), ValueError("second")
    raised = iter([first, second])

    def observer(index: int, item: dict[str, Any]) -> None:
        if index >= 2:
            raise next(raised)

    ledger = ledger_of(HEAD[:2], observer)
    with pytest.raises(ValueError, match="first"):
        ledger.append(fresh([STATE])[0])
    with pytest.raises(ValueError, match="second"):
        ledger.append(entry("d:route", 0, None, "x"))
    assert ledger.observer_error is first


def test_the_appends_own_refusal_is_not_marked_as_the_observers() -> None:
    ledger = ledger_of(HEAD, lambda i, item: None)  # pyright: ignore[reportUnknownLambdaType]
    with pytest.raises(_LedgerError):
        ledger.append({"item": "bogus"})
    assert ledger.observer_error is None


def test_a_turn_with_and_without_an_observer_has_the_same_record() -> None:
    for log in LOGS.values():
        assert ledger_of(log, lambda i, item: None).close().document == (  # pyright: ignore[reportUnknownLambdaType]
            ledger_of(log).close().document
        )


# ---------------------------------------------------------------------------------------------
# unhashed writes

CALL = {"start": "2026-10-20T00:00:00Z", "latency": 0.25, "usage": {"tokens": 12}, "attempts": 1}
EXC = {"type": "ValueError", "message": "no", "traceback": "Traceback ..."}


def test_unhashed_members_are_written_in_their_shapes() -> None:
    ledger = ledger_of(NORMAL)
    ledger.write_unhashed("turn", {"start": "s"})
    ledger.write_unhashed("engine", "jurat 0.0.1")
    ledger.write_unhashed("codec", "identity")
    ledger.write_unhashed(
        "calls", {**CALL, "unexpected": ["q:x", "1"], "invalid": {"usage": "x"}}, 3
    )
    ledger.write_unhashed("calls", {**CALL, "usage": None, "attempts": None}, 7)
    ledger.write_unhashed("exceptions", EXC, 3)
    ledger.write_unhashed("exceptions", EXC, 9)
    ledger.write_unhashed("exceptions", EXC, 12)
    ledger.write_unhashed("reads", "t:state.secret", 9)
    ledger.write_unhashed("reads", "t:state.secret", 12)
    ledger.write_unhashed("turn", {"end": "e"})
    ledger.write_unhashed("future", {"anything": [1]})
    unhashed = ledger.close().unhashed
    assert unhashed["turn"] == {"start": "s", "end": "e"}
    assert set(unhashed["calls"]) == {"3", "7"}
    assert set(unhashed["exceptions"]) == {"3", "9", "12"}
    assert unhashed["future"] == {"anything": [1]}


@pytest.mark.parametrize(
    ("member", "value", "index", "match"),
    [
        ("turn", {"end": "e"}, None, "has no 'start'"),
        ("turn", {"start": "s", "at": 1}, None, "member 'at'"),
        ("turn", "s", None, "'turn' is not an object"),
        ("turn", {"start": 1}, None, "'start' is not a string"),
        ("engine", 1, None, "'engine' is not a string"),
        ("codec", None, None, "'codec' is not a string"),
        ("calls", CALL, None, "written under a log index"),
        ("calls", CALL, True, "written under a log index"),
        ("engine", "x", 3, "not keyed by a log index"),
        ("calls", {**CALL, "latency": "1"}, 3, "'latency' is not a number"),
        ("calls", {**CALL, "latency": float("inf")}, 3, "not finite"),
        ("calls", {**CALL, "usage": {"t": "1"}}, 3, "'usage' is not an object of numbers"),
        ("calls", {**CALL, "attempts": 1.0}, 3, "'attempts' is not an integer or null"),
        ("calls", without(CALL, "start"), 3, "has no 'start'"),
        ("calls", {**CALL, "unexpected": "x"}, 3, "'unexpected' is not a list of strings"),
        ("calls", {**CALL, "unexpected": []}, 3, "'unexpected' is empty"),
        ("calls", {**CALL, "invalid": {}}, 3, "'invalid' is empty"),
        ("calls", {**CALL, "invalid": {"values": "x"}}, 3, "'values', which is not a known member"),
        ("calls", {**CALL, "invalid": {"code": 1}}, 3, r"\['code'\] is not a string"),
        ("calls", {**CALL, "extra": 1}, 3, "member 'extra'"),
        ("exceptions", without(EXC, "traceback"), 3, "has no 'traceback'"),
        ("reads", ["t:x"], 9, "is not a string"),
        ("future", {1, 2}, None, "outside the value domain"),
        ("future", b"x", None, "outside the value domain"),
        ("future", {1: "x"}, None, "object key 1 is not a string"),
        # keys name an item of the log so far of the member's kind
        ("calls", CALL, 2, "key '2', which names entry, not an item of the kind call"),
        ("calls", CALL, 12, "names plan, not an item of the kind call"),
        ("calls", CALL, 14, "key '14', which names no item"),
        ("calls", CALL, -1, "key '-1', which is no log index"),
        ("exceptions", EXC, 6, "names round, not an item of the kind call, entry, plan"),
        ("exceptions", EXC, 0, "names design"),
        ("reads", "t:x", 3, "names call, not an item of the kind entry, plan"),
        ("reads", "t:x", 13, "names verdict"),
    ],
)
def test_unhashed_writes_of_the_wrong_shape_are_refused(
    member: str, value: Any, index: Any, match: str
) -> None:
    ledger = ledger_of(NORMAL)
    with pytest.raises(_LedgerError, match=match):
        ledger.write_unhashed(member, value, index)
    assert ledger._unhashed == {}


def test_a_key_names_an_item_already_in_the_log() -> None:
    ledger = ledger_of(R1_OPEN)
    with pytest.raises(_LedgerError, match="key '4', which names no item"):
        ledger.write_unhashed("exceptions", EXC, 4)
    ledger.append(fresh([NORMAL[4]])[0])
    ledger.write_unhashed("exceptions", EXC, 4)


def test_a_calls_entry_holds_the_sums_of_its_call_items_questions() -> None:
    ledger = ledger_of(NORMAL)
    sums = {"q:spam": 1.8}
    ledger.write_unhashed("calls", {**CALL, "sums": sums}, 7)
    ledger.write_unhashed("calls", CALL, 3)
    record = ledger.close()
    assert record.unhashed["calls"]["7"]["sums"] == sums
    assert read_record(record.document).unhashed == record.unhashed
    ledger = ledger_of(NORMAL)
    ledger.write_unhashed("calls", {**CALL, "sums": {"q:spam": 0}}, 7)  # an int is a number


@pytest.mark.parametrize(
    ("index", "sums", "match"),
    [
        (7, [], "'sums' is not an object"),
        (7, {}, "'sums' is empty"),
        (7, {"q:spam": "1.8"}, r"'sums'\['q:spam'\] is not a number"),
        (7, {"q:spam": True}, r"'sums'\['q:spam'\] is not a number"),
        (7, {"q:spam": None}, r"'sums'\['q:spam'\] is not a number"),
        (7, {"q:tone": 1.8}, "'q:tone', which names no key of its call item's questions"),
        (7, {"spam": 1.8}, "'spam', which names no key of its call item's questions"),
        (7, {"q:spam.f": 1.8}, "'q:spam.f', which names no key of its call item's questions"),
        (3, {"q:spam": 1.8}, "'q:spam', which names no key of its call item's questions"),
        (7, {'q:spam["x"]': 1.8}, "is a grounded ref, which a v0.0.1 record holds nowhere"),
    ],
)
def test_a_sums_member_of_the_wrong_shape_or_key_is_refused(
    index: int, sums: Any, match: str
) -> None:
    ledger = ledger_of(NORMAL)
    with pytest.raises(_LedgerError, match=match):
        ledger.write_unhashed("calls", {**CALL, "sums": sums}, index)
    assert ledger._unhashed == {}
    with pytest.raises(InvalidRecord, match=match):
        read_record(
            {"log": fresh(NORMAL), "unhashed": {"calls": {str(index): {**CALL, "sums": sums}}}}
        )


def test_a_long_sums_key_is_clipped_in_the_refusal() -> None:
    key = "q:" + "a" * 5000
    ledger = ledger_of(NORMAL)
    with pytest.raises(_LedgerError, match=r"\(5002 code points\)") as info:
        ledger.write_unhashed("calls", {**CALL, "sums": {key: "x"}}, 7)
    assert len(str(info.value)) < 300


def test_a_grounded_question_key_matches_no_sums_key() -> None:
    # A grounded question key is no declared ref, so no sums key names it; a grounded sums
    # key over a declared question is refused as grounded in v0.0.1.
    grounded = call(2, "s", questions={'q:spam["x"]': QUESTIONS["q:spam"]})
    ledger = ledger_of([*R1_DONE, grounded])
    with pytest.raises(_LedgerError, match="which names no key"):
        ledger.write_unhashed("calls", {**CALL, "sums": {'q:spam["x"]': 2}}, 7)


def test_a_refused_call_item_adds_no_questions() -> None:
    ledger = ledger_of(R1_DONE)
    with pytest.raises(_LedgerError, match="not in code-point order"):
        ledger.append(call(2, "s", sources=["s", "a"], questions=QUESTIONS))
    ledger.append(call(2, "a"))
    with pytest.raises(_LedgerError, match="names no key"):
        ledger.write_unhashed("calls", {**CALL, "sums": {"q:spam": 2}}, 7)


def test_a_calls_entry_is_written_once() -> None:
    ledger = ledger_of(NORMAL)
    ledger.write_unhashed("calls", CALL, 3)
    with pytest.raises(_LedgerError, match="written once"):
        ledger.write_unhashed("calls", CALL, 3)


def test_surrogates_are_escaped_at_the_write_the_first_colliding_key_kept() -> None:
    ledger = ledger_of(BROKEN)
    ledger.write_unhashed("future", {"\udc80": "a\ud800b", "\\udc80": "second", "k": ["\udfff"]})
    ledger.write_unhashed("exceptions", {**EXC, "message": "bad \udcff byte"}, 2)
    unhashed = ledger.close().unhashed
    assert unhashed["future"] == {"\\udc80": "a\\ud800b", "k": ["\\udfff"]}
    assert unhashed["exceptions"]["2"]["message"] == "bad \\udcff byte"


def test_an_unhashed_write_is_a_copy() -> None:
    ledger = ledger_of(BROKEN)
    value = {"k": [1]}
    ledger.write_unhashed("future", value)
    value["k"].append(2)
    assert ledger.close().unhashed == {"future": {"k": [1]}}


def test_an_unhashed_write_too_deep_or_cyclic_is_refused() -> None:
    # A member is a holder standing in unhashed: 126 deep, a keyed member's entry 125.
    ledger = ledger_of(BROKEN)
    deep: Any = "x"
    for _ in range(127):
        deep = [deep]
    with pytest.raises(_LedgerError, match="126 deep"):
        ledger.write_unhashed("future", deep)
    with pytest.raises(_LedgerError, match="125 deep"):
        ledger.write_unhashed("reads", deep[0], 2)
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    with pytest.raises(_LedgerError, match="126 deep"):
        ledger.write_unhashed("future", cyclic)
    almost: Any = deep[0]
    ledger.write_unhashed("future", almost)  # 126 deep is in bounds
    record = ledger.close()
    assert read_record(record.document).unhashed["future"] == almost


def test_every_prefix_closes_once_and_again() -> None:
    for log in (NORMAL, BROKEN, CANCELLED, [*HEAD, PLAN]):
        for n in range(1, len(log) + 1):
            ledger = ledger_of(log[:n])
            record = ledger.close()
            assert ledger.close() is record
            assert record.log == log[:n]


def test_closing_a_ledger_with_no_item_is_a_defect() -> None:
    ledger = Ledger()
    with pytest.raises(_LedgerError, match="holds no item"):
        ledger.close()
    assert not ledger.closed


def test_a_verdict_after_a_broken_plan() -> None:
    assert ledger_of([DESIGN, SETTINGS, BROKEN_PLAN, VERDICT]).close().log == BROKEN


# ---------------------------------------------------------------------------------------------
# Depth is bounded per value, where the value stands


def test_a_64_deep_entry_value_and_literal_argument_round_trip() -> None:
    plan = {
        **PLAN,
        "steps": [
            {
                "tool": "t",
                "arguments": {"a": {"literal": DEEP64}},
                "requirements": [{"ref": "d:route", "value": DEEP64}],
            }
        ],
    }
    log = [*HEAD, entry("d:route", 0, None, DEEP64), plan, VERDICT]
    record = ledger_of(log).close()
    back = read_record(record.document)
    assert back.record_hash == record.record_hash
    assert back.document == record.document
    assert back.log[3]["value"] == DEEP64
    assert back.log[4]["steps"][0]["arguments"]["a"]["literal"] == DEEP64


def test_an_item_past_the_documents_bound_is_refused() -> None:
    refused(HEAD, entry("d:route", 0, None, nested(127)), "nests more than 126 deep")
    refused(HEAD, entry("d:route", 0, None, "x", rows=nested(126)), "nests more than 126 deep")


# ---------------------------------------------------------------------------------------------
# Review points


def test_two_keys_with_one_string_data_before_the_escape_are_refused() -> None:
    class Key(str):
        def __hash__(self) -> int:
            return id(self)

        def __eq__(self, other: object) -> bool:
            return self is other

    ledger = ledger_of(BROKEN)
    with pytest.raises(_LedgerError, match="two object keys have the string data 'k'"):
        ledger.write_unhashed("future", {Key("k"): 1, Key("k"): 2})
    assert ledger._unhashed == {}


@pytest.mark.parametrize("n", [-1, 5, True, 1.0])
def test_a_prefix_outside_the_log_is_a_defect(n: Any) -> None:
    ledger = ledger_of(BROKEN)
    with pytest.raises(_LedgerError, match="is not one of a log of 4"):
        ledger.prefix_hash(n)
    with pytest.raises(_LedgerError, match="is not one of a log of 4"):
        record_prefix_hash(ledger.close(), n)
    assert ledger.prefix_hash(0) == hash_value([])
    assert ledger.prefix_hash(4) == hash_value(BROKEN)


@pytest.mark.parametrize(
    ("member", "first", "second", "index"),
    [
        ("turn", {"start": "s"}, {"start": "t"}, None),
        ("turn", {"start": "s", "end": "e"}, {"end": "f"}, None),
        ("engine", "a", "b", None),
        ("codec", "a", "a", None),
        ("calls", CALL, CALL, 3),
        ("exceptions", EXC, EXC, 9),
        ("reads", "t:x", "t:y", 12),
    ],
)
def test_a_second_unhashed_write_is_refused(
    member: str, first: Any, second: Any, index: int | None
) -> None:
    ledger = ledger_of(NORMAL)
    ledger.write_unhashed(member, first, index)
    before = copy.deepcopy(ledger._unhashed)
    with pytest.raises(_LedgerError, match="is written once"):
        ledger.write_unhashed(member, second, index)
    assert ledger._unhashed == before


def test_write_once_is_counted_per_member() -> None:
    ledger = ledger_of(NORMAL)
    ledger.write_unhashed("exceptions", EXC, 9)
    ledger.write_unhashed("reads", "t:x", 9)
    ledger.write_unhashed("calls", CALL, 3)
    ledger.write_unhashed("exceptions", EXC, 3)
    assert set(ledger.close().unhashed) == {"calls", "exceptions", "reads"}


def test_unhashed_is_held_to_the_value_bounds_depth_alone(small_bounds: SmallBounds) -> None:
    rows = ["x" * 16] * 17  # over the size bounds as a value
    ledger = ledger_of(BROKEN)
    ledger.write_unhashed("other", rows)
    assert ledger.close().unhashed["other"] == rows


def test_a_call_entrys_usage_is_size_tested_at_the_read_alone(small_bounds: SmallBounds) -> None:
    entry_: dict[str, Any] = {"start": "s", "latency": 0.5, "attempts": None}
    within = ledger_of(NORMAL[:4])
    within.write_unhashed("calls", {**entry_, "usage": {"x" * 256: 1}}, 3)
    read_record(within.close().document)
    over = ledger_of(NORMAL[:4])
    over.write_unhashed("calls", {**entry_, "usage": {"x" * 257: 1}}, 3)  # its producer's test
    with pytest.raises(InvalidRecord, match="'usage' is outside the value domain: the value's"):
        read_record(over.close().document)


def test_an_item_shared_within_is_copied_once_and_encoded_at_every_place() -> None:
    shared = ["text"] * 4
    ledger = ledger_of(NORMAL[:4])
    ledger.append(entry("t:mail/body", 1, 3, [shared, shared]))
    assert ledger.close().log[4]["value"] == [shared, shared]


def test_an_unhashed_write_under_a_huge_index_names_its_bit_length() -> None:
    ledger = ledger_of(NORMAL)
    with pytest.raises(_LedgerError, match=r"^unhashed 'calls' is written under an index of 16610"):
        ledger.write_unhashed("calls", CALL, 10**5000)
    assert ledger._unhashed == {}


class _NoRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr")


@pytest.mark.parametrize(
    ("n", "shown"),
    [(10**5000, "integer of 16610 bits"), (_NoRepr(), "<_NoRepr>")],
    ids=["huge", "no repr"],
)
def test_a_prefix_outside_the_log_is_named_without_its_repr(n: Any, shown: str) -> None:
    ledger = ledger_of(BROKEN)
    with pytest.raises(_LedgerError, match=f"^a prefix of {shown} items is not one"):
        ledger.prefix_hash(n)


def test_an_unhashed_key_that_is_no_log_index_is_clipped() -> None:
    ledger = ledger_of(NORMAL)
    with pytest.raises(_LedgerError) as info:
        ledger._set_unhashed("calls", {"x" * 1000: CALL})
    assert f"has the key {'x' * 64!r}... (1000 code points), which is no log index" in str(
        info.value
    )


_CHECK_STATE = (
    "_phase",
    "_round",
    "_round_ended",
    "_round_has_entry",
    "_round_calls",
    "_cut",
    "_sources",
    "_written",
    "_node",
    "_node_last",
    "_done_nodes",
)


def _check_state(ledger: Ledger) -> dict[str, Any]:
    return {name: copy.deepcopy(getattr(ledger, name)) for name in _CHECK_STATE}


@pytest.mark.parametrize(
    ("prefix", "item"),
    [
        (R1_ENTRY, entry("t:mail/body", 1, 3, "y")),  # written once
        ([*R1_ENTRY, entry("d:route", 1, None, "x")], entry("t:mail/subject", 1, 3, "x")),
        (R1_DONE, call(2, "s", sources=["a", "fetch_mail"])),  # a source called twice
        (R1_DONE, entry("d:route", 2, 9, "x")),  # a new round's entry naming no call
        (R1_OPEN, entry("d:route", 1, None, "x", unexpected=1)),
    ],
    ids=["written", "contiguous", "sources", "call", "shape"],
)
def test_a_refused_append_leaves_the_check_state_unchanged(
    prefix: list[dict[str, Any]], item: dict[str, Any]
) -> None:
    ledger = ledger_of(prefix)
    before = _check_state(ledger)
    with pytest.raises(_LedgerError):
        ledger.append(item)
    assert _check_state(ledger) == before
    assert len(ledger.close().log) == len(prefix)


def test_the_check_state_grows_in_place() -> None:
    ledger = ledger_of(NORMAL[:5])
    written, done, sources = ledger._written, ledger._done_nodes, ledger._sources
    for item in fresh(NORMAL[5:12]):
        ledger.append(item)
    assert ledger._written is written
    assert ledger._done_nodes is done
    assert ledger._sources is sources
    assert written == {"t:state", "t:mail/body", "t:mail/subject", "q:spam", "d:route", "d:tags"}
    assert sources == {"fetch_mail": 3, "ask_spam": 7}


class _FullSources(dict[str, int]):
    """A sources map whose update runs out of memory, as a grow of the commit may."""

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise MemoryError


def test_a_grow_that_raises_leaves_the_phase_and_the_round_as_they_were() -> None:
    ledger = ledger_of(R1_DONE)
    ledger._sources = _FullSources(ledger._sources)
    before = (ledger._phase, ledger._round, ledger._round_ended, ledger._round_calls)
    with pytest.raises(MemoryError):
        ledger.append(call(2, "a"))
    assert (ledger._phase, ledger._round, ledger._round_ended, ledger._round_calls) == before


def test_a_call_items_sources_are_held_apart_from_the_stored_copy() -> None:
    seen: list[Any] = []

    def observe(index: int, item: Any) -> None:
        seen.append(item)

    ledger = ledger_of(R1_DONE, observer=observe)
    index = ledger.append(call(2, "a"))
    seen[-1]["sources"].append("b")  # the observer changes the stored copy
    assert ledger._sources == {"fetch_mail": 3, "a": index}


class _FailingSet(set[Any]):
    """A set whose ``add`` runs out of memory, as a grow of the commit may."""

    def add(self, element: Any) -> None:
        raise MemoryError


def _snapshot(ledger: Ledger) -> dict[str, Any]:
    return copy.deepcopy(vars(ledger))


@pytest.mark.parametrize(
    ("prefix", "item", "member"),
    [
        (R1_OPEN, NORMAL[4], "_written"),  # the first grow
        (R1_ENTRY, NORMAL[5], "_written"),
        (NORMAL[:9], NORMAL[9], "_done_nodes"),  # after the entry's ref is added
        (R1_OPEN, call(1, "other"), "_round_calls"),  # after the call's sources are listed
        (NORMAL[:8], NORMAL[8], "_written"),
    ],
    ids=["entry", "same node", "node ends", "call", "after a call"],
)
def test_a_commit_that_raises_leaves_the_ledger_as_it_was(
    prefix: list[dict[str, Any]], item: dict[str, Any], member: str
) -> None:
    ledger = ledger_of(prefix)
    held = getattr(ledger, member)
    failing = _FailingSet(held)
    setattr(ledger, member, failing)
    before = _snapshot(ledger)
    with pytest.raises(MemoryError):
        ledger.append(copy.deepcopy(item))
    assert _snapshot(ledger) == before
    setattr(ledger, member, set(failing))
    assert ledger.append(copy.deepcopy(item)) == len(prefix)
    with pytest.raises(_LedgerError):
        ledger.append(copy.deepcopy(item))  # a retry is logged once, as ever
