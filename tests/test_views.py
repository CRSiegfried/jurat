# pyright: reportPrivateUsage=false
"""The record views: the escalation's four
members and when a record holds none, and the redacted view, its digests recomputed here
independently with ``hmac`` and the ``rfc8785`` oracle, over records decide wrote and over the
synthetic logs and every prefix of them."""

import copy
import hashlib
import hmac
from collections.abc import Callable
from typing import Any, cast

import pytest
import rfc8785

from designs import minimal
from jurat import (
    InvalidRecord,
    LoadedDesign,
    Origin,
    Plan,
    Policy,
    Record,
    Requirement,
    RunSettings,
    Step,
    decide_sync,
    escalation,
    load,
    read_record,
    redact,
)
from jurat import types as t
from jurat._canonical import decode, encode
from jurat._values import MAX_DOCUMENT_DEPTH
from ledger_logs import DESIGN, LOGS, NORMAL, SETTINGS, STATE, call, entry, fresh, round_item

ROOT = bytes(range(32))


def key_of(record_hash: str, root: bytes = ROOT) -> bytes:
    return hmac.digest(root, record_hash.encode("ascii") + b"\x01", "sha256")


def digest(value: Any, record: Record, root: bytes = ROOT) -> str:
    """A keyed digest computed apart from the code under test."""
    mac = hmac.new(key_of(record.record_hash, root), rfc8785.dumps(value), hashlib.sha256)
    return "hmac-sha256:" + mac.hexdigest()


def read(log: list[dict[str, Any]], unhashed: dict[str, Any] | None = None) -> Record:
    return read_record({"log": log, "unhashed": unhashed or {}})


def view(record: Record, root: bytes = ROOT) -> dict[str, Any]:
    out = redact(record, root)
    assert encode(decode(out), MAX_DOCUMENT_DEPTH) == out  # canonical bytes
    return decode(out)


# ---------------------------------------------------------------------------------------------
# Turns decide wrote


def inform(r: Any) -> Plan:
    return Plan("inform", reply="nothing to do")


def escalate(r: Any) -> Plan:
    return Plan("escalate", reason="a person should look")


def archive(r: Any) -> Plan:
    return Plan("execute", [Step("archive", {"folder": Origin(literal="spam")})])


def archive_required(r: Any) -> Plan:
    step = Step("archive", {"folder": Origin(literal="spam")}, [Requirement("d:flag", True)])
    return Plan("execute", [step], reason="it is spam")


def requiring(r: Any) -> Plan:
    r.require("d:gone")
    return Plan("inform", reply="unreached")


def flag(r: Any) -> bool:
    return False


def gone(r: Any) -> Any:
    raise RuntimeError("no value")


def loaded_with(policy: Callable[[Any], Plan], **changes: Any) -> LoadedDesign:
    from jurat import Rule

    fields: dict[str, Any] = {
        "state": t.record(subject=t.string),
        "rules": {
            "flag": Rule("d:flag", t.boolean, flag, version="1"),
            "gone": Rule("d:gone", t.boolean, gone, version="1"),
        },
        "policy": Policy(policy, version="1", reads=["d:flag", "d:gone"]),
        "tools": ["archive"],
    }
    fields.update(changes)
    return load(minimal(**fields))


def turn(policy: Callable[[Any], Plan], **changes: Any) -> Record:
    return decide_sync(loaded_with(policy, **changes), {"subject": "hi"}, {}).record


# ---------------------------------------------------------------------------------------------
# The escalation


def test_a_passed_execute_plan_holds_no_escalation() -> None:
    assert escalation(turn(archive, default_deny=False)) is None


def clarify(r: Any) -> Plan:
    return Plan("clarify", reply="which?")


def nothing(r: Any) -> None:
    return None


@pytest.mark.parametrize("policy", [inform, clarify])
def test_a_passed_inform_or_clarify_plan_holds_no_escalation(
    policy: Callable[[Any], Plan],
) -> None:
    assert escalation(turn(policy)) is None


def test_a_broken_plan_holds_no_escalation() -> None:
    assert escalation(turn(nothing)) is None  # type: ignore[arg-type]
    broken = decide_sync(loaded_with(inform), {}, {})  # a broken entry
    assert broken.record.log[2]["kind"] == "broken"
    assert escalation(broken.record) is None


def test_a_refused_verdict_escalates_with_the_refused_cause_and_no_reason() -> None:
    record = turn(archive_required)
    found = escalation(record)
    assert found == {
        "cause": {"code": "refused", "refs": ["d:flag"]},
        "failures": [{"step": 1, "requirement": {"ref": "d:flag", "value": True}, "ref": "d:flag"}],
        "reason": None,  # the plan is execute, so its reason is not the escalation's
        "record_hash": record.record_hash,
    }


def test_a_default_deny_failure_names_its_step() -> None:
    found = escalation(turn(archive))
    assert found is not None
    assert found["cause"] == {"code": "refused", "refs": ["step:1"]}
    assert found["failures"] == [{"step": 1, "requirement": None, "ref": None}]


def test_a_policys_escalate_has_no_cause_and_keeps_its_reason() -> None:
    found = escalation(turn(escalate))
    assert found is not None
    assert found["cause"] is None
    assert found["reason"] == "a person should look"
    assert found["failures"] == []


def test_a_compile_built_escalate_carries_required() -> None:
    record = turn(requiring)
    found = escalation(record)
    assert found == {
        "cause": {"code": "required", "refs": ["d:gone"]},
        "failures": [],
        "reason": None,
        "record_hash": record.record_hash,
    }


def test_the_escalation_is_plain_data_of_exactly_four_members() -> None:
    found = escalation(turn(archive_required))
    assert found is not None
    assert type(found) is dict
    assert list(found) == ["cause", "failures", "reason", "record_hash"]
    cause, failures = found["cause"], found["failures"]
    code: object = cause["code"]
    refs: object = cause["refs"]
    failure: object = failures[0]
    assert type(cast(object, cause)) is dict
    assert type(code) is str
    assert type(refs) is list
    assert type(cast(object, failures)) is list
    assert type(failure) is dict


def test_a_record_with_no_verdict_holds_no_escalation() -> None:
    for n in range(1, len(NORMAL) - 1):
        assert escalation(read(fresh(NORMAL[:n]))) is None


def test_the_escalation_reads_only_the_plan_and_verdict_items_and_the_hash() -> None:
    refused = {
        "item": "verdict",
        "kind": "refused",
        "failures": [
            {"step": "plan", "requirement": {"ref": "d:route", "value": "keep"}, "ref": "d:route"},
            {"step": 1, "requirement": None, "ref": None},
        ],
    }
    record = read([*fresh(NORMAL[:-1]), refused])
    assert escalation(record) == {
        "cause": {"code": "refused", "refs": ["d:route", "step:1"]},
        "failures": refused["failures"],
        "reason": None,
        "record_hash": record.record_hash,
    }


class Untouchable:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"the record was read: {name}")


class SubRecord(Record):
    pass


# Built in the test, since pytest reads a parameter at collection.
NOT_RECORDS: list[Callable[[], Any]] = [lambda: NORMAL, lambda: None, Untouchable]


@pytest.mark.parametrize("record", NOT_RECORDS, ids=["a log", "None", "untouchable"])
def test_the_escalation_refuses_what_is_not_a_record_before_reading_it(
    record: Callable[[], Any],
) -> None:
    with pytest.raises(ValueError, match="takes a Record"):
        escalation(record())


def test_the_escalation_refuses_a_record_subclass_by_its_exact_type() -> None:
    sub = object.__new__(SubRecord)  # never initialised: refused before anything is read
    with pytest.raises(ValueError, match="takes a Record"):
        escalation(sub)


# ---------------------------------------------------------------------------------------------
# The redacted view over the synthetic normal log


def test_redact_digests_by_class_and_passes_open_values() -> None:
    record = read(fresh(NORMAL))
    out = view(record)
    log = out["log"]

    def d(value: Any) -> str:
        return digest(value, record)

    assert log[:2] == NORMAL[:2]  # the design and settings items are open
    assert log[2]["value"] == {"subject": d("hello"), "to": d("dana@example.com")}
    assert log[3] == NORMAL[3]  # a call item with no prompt
    assert log[4]["value"] == d("text of the mail")
    assert log[5]["value"] == d("hello")
    assert log[7]["prompt"] == d("Is this spam?")
    assert log[7]["questions"] == NORMAL[7]["questions"]
    assert log[8] == NORMAL[8]  # an answer is open
    assert log[9] == NORMAL[9]
    assert log[10]["value"] == [{d("a"): "x"}, None]  # a mapping's keys, under every tree
    plan = log[12]
    assert plan["steps"][0]["arguments"] == {
        "folder": {"literal": d("spam")},
        "what": {"ref": "d:route"},
    }
    assert plan["steps"][0]["requirements"] == NORMAL[12]["steps"][0]["requirements"]
    assert (plan["reason"], plan["reply"]) == (None, None)
    assert log[13] == NORMAL[13]
    assert out["unhashed"] == {}
    assert out["redacted"] == record.record_hash
    assert set(out) == {"log", "unhashed", "redacted"}


def test_equal_values_show_equal_digests_within_one_record() -> None:
    log = view(read(fresh(NORMAL)))["log"]
    assert log[2]["value"]["subject"] == log[5]["value"]


def test_the_key_is_the_records_own_and_one_log_has_one_key() -> None:
    plain = read(fresh(NORMAL))
    with_text = read(fresh(NORMAL), {"engine": "0.0.1", "reads": {"9": "t:x"}})
    assert plain.record_hash == with_text.record_hash
    assert view(plain)["log"] == view(with_text)["log"]
    other = view(plain, bytes(32))["log"]
    assert other[4]["value"] == digest("text of the mail", plain, bytes(32))
    assert other[4]["value"] != view(plain)["log"][4]["value"]


def test_the_redacted_view_is_refused_by_read_record() -> None:
    with pytest.raises(InvalidRecord, match="redacted"):
        read_record(redact(read(fresh(NORMAL)), ROOT))


# ---------------------------------------------------------------------------------------------
# The grain

TREES: dict[str, Any] = {
    "d:words": "content",
    "d:recs": {"k": "content", "n": "open"},
    "d:map": ["content"],
    "d:mapo": ["open"],
    "d:mapr": [{"k": "content"}],
    "d:nums": "open",
    "d:who": "identifier",
    "d:note": "content",
    "d:deep": "open",
}


def grain(values: dict[str, Any]) -> tuple[Record, dict[str, Any]]:
    """A round-0 log holding each value at its ref under ``TREES``."""
    settings = copy.deepcopy(SETTINGS)
    settings["providers"] = {ref: ref[2:] for ref in TREES}
    settings["classes"] = {"t:state": SETTINGS["classes"]["t:state"], **TREES}
    log = [DESIGN, settings, fresh([STATE])[0]]
    log += [entry(ref, 0, None, values[ref]) for ref in sorted(values)]
    record = read(log)
    redacted = view(record)["log"]
    return record, {item["ref"]: item.get("value") for item in redacted[3:]}


def test_a_list_or_mapping_with_no_open_and_no_record_is_one_digest() -> None:
    record, out = grain({"d:words": ["a", None, "b"], "d:map": {"x": "y"}})
    assert out["d:words"] == digest(["a", None, "b"], record)  # the null inside included
    assert out["d:map"] == digest({"x": "y"}, record)


def test_any_other_list_or_mapping_is_opened() -> None:
    record, out = grain(
        {
            "d:recs": [{"k": "a", "n": 1}, None],
            "d:mapo": {"x": "y"},
            "d:mapr": {"x": {"k": "a"}},
            "d:nums": [1, 2, [3]],
        }
    )

    def d(value: Any) -> str:
        return digest(value, record)

    assert out["d:recs"] == [{"k": d("a"), "n": 1}, None]
    assert out["d:mapo"] == {d("x"): "y"}
    assert out["d:mapr"] == {d("x"): {"k": d("a")}}
    assert out["d:nums"] == [1, 2, [3]]


def test_a_null_passes_under_every_class() -> None:
    _, out = grain({"d:note": None, "d:who": None, "d:words": None})
    assert out == {"d:note": None, "d:who": None, "d:words": None}


def test_identifier_and_content_digest_alike() -> None:
    record, out = grain({"d:who": "dana", "d:note": "dana"})
    assert out["d:who"] == out["d:note"] == digest("dana", record)


def test_a_value_64_deep_redacts() -> None:
    deep: Any = 1
    for _ in range(63):
        deep = [deep]
    _, out = grain({"d:deep": deep})
    assert out["d:deep"] == deep


# ---------------------------------------------------------------------------------------------
# Values outside entries


def plan_with(arguments: dict[str, Any], **members: Any) -> dict[str, Any]:
    step: dict[str, Any] = {"tool": "archive", "arguments": arguments, "requirements": []}
    return {
        "item": "plan",
        "kind": "execute",
        "steps": [step],
        "reason": None,
        "reply": None,
        "cause": None,
        **members,
    }


def test_a_literal_takes_the_default_of_its_json_form() -> None:
    arguments = {
        "text": {"literal": "spam"},
        "n": {"literal": 3},
        "yes": {"literal": True},
        "none": {"literal": None},
        "list": {"literal": ["x", 1, None]},
        "object": {"literal": {"a": ["x", 2]}},
        "bound": {"step": 1, "path": None},
    }
    record = read([*fresh(NORMAL[:12]), plan_with(arguments)])
    out = view(record)["log"][12]["steps"][0]["arguments"]

    def d(value: Any) -> str:
        return digest(value, record)

    assert out == {
        "text": {"literal": d("spam")},
        "n": {"literal": 3},
        "yes": {"literal": True},
        "none": {"literal": None},
        "list": {"literal": [d("x"), 1, None]},
        "object": {"literal": {d("a"): [d("x"), 2]}},
        "bound": {"step": 1, "path": None},
    }


@pytest.mark.parametrize("set_", ["d:route", "t:mail/body", "a set", None])
def test_read_record_refuses_an_origin_carrying_set(set_: Any) -> None:
    # Nothing checks a candidate-value literal's set in v0.0.1, so redact could find no
    # tree for it; a null set is a set key all the same.
    arguments = {"folder": {"literal": "x", "set": set_}}
    with pytest.raises(InvalidRecord, match="carries set"):
        read([*fresh(NORMAL[:12]), plan_with(arguments)])


def test_a_plans_reason_and_reply_are_content() -> None:
    plan = plan_with({}, kind="escalate", steps=[], reason="why")
    record = read([*fresh(NORMAL[:12]), plan])
    out = view(record)["log"][12]
    assert out["reason"] == digest("why", record)
    assert out["reply"] is None


UNHASHED: dict[str, Any] = {
    "turn": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:00:01Z"},
    "engine": "0.0.1",
    "codec": "encode",
    "calls": {
        "3": {
            "start": "2026-01-01T00:00:00Z",
            "latency": 0.5,
            "usage": {"input": 3},
            "attempts": 1,
            "unexpected": ["t:x", "t:y"],
            "invalid": {"snapshot": "b'x'"},
        },
        "7": {
            "start": "2026-01-01T00:00:00Z",
            "latency": 0.5,
            "usage": None,
            "attempts": None,
            "sums": {"q:spam": 1.8},
        },
    },
    "exceptions": {"9": {"type": "ValueError", "message": "bad", "traceback": "Traceback..."}},
    "reads": {"9": "t:secret"},
    "later": {"a": [1]},
}


def test_unhashed_text_is_content_and_its_names_keys_and_usage_open() -> None:
    record = read(fresh(NORMAL), copy.deepcopy(UNHASHED))
    out = view(record)["unhashed"]

    def d(value: Any) -> str:
        return digest(value, record)

    assert out == {
        "turn": UNHASHED["turn"],
        "engine": "0.0.1",
        "codec": "encode",
        "calls": {
            "3": {
                "start": "2026-01-01T00:00:00Z",
                "latency": 0.5,
                "usage": {"input": 3},
                "attempts": 1,
                "unexpected": d(["t:x", "t:y"]),  # digested as one value
                "invalid": {"snapshot": d("b'x'")},
            },
            "7": UNHASHED["calls"]["7"],  # a sums key and its number open
        },
        "exceptions": {
            "9": {"type": "ValueError", "message": d("bad"), "traceback": d("Traceback...")}
        },
        "reads": {"9": d("t:secret")},
        "later": d({"a": [1]}),  # a member the engine does not know, whole
    }


# ---------------------------------------------------------------------------------------------
# The root key


class Key(bytes):
    pass


class LongKey(bytes):
    def __len__(self) -> int:
        raise AssertionError("the caller's __len__ was called")


@pytest.mark.parametrize(
    "root",
    [bytes(31), bytes(33), b"", "k" * 32, bytearray(32), memoryview(bytes(32)), None, 32],
    ids=["31 bytes", "33 bytes", "empty", "str", "bytearray", "memoryview", "None", "int"],
)
def test_a_bad_root_key_is_refused_before_the_record_is_read(root: Any) -> None:
    with pytest.raises(ValueError, match="root key"):
        redact(read(fresh(NORMAL)), root)
    # The record is tested first: a non-Record with a bad key takes the record's refusal,
    # and nothing of it is read.
    with pytest.raises(ValueError, match="takes a Record"):
        redact(Untouchable(), root)  # type: ignore[arg-type]


def test_a_bytes_subclass_root_key_is_taken_by_its_bytes_alone() -> None:
    record = read(fresh(NORMAL))
    assert redact(record, Key(ROOT)) == redact(record, ROOT)
    assert redact(record, LongKey(ROOT)) == redact(record, ROOT)
    with pytest.raises(ValueError, match="root key"):
        redact(record, LongKey(bytes(31)))


@pytest.mark.parametrize("record", NOT_RECORDS, ids=["a log", "None", "untouchable"])
def test_redact_refuses_what_is_not_a_record_with_a_good_key(record: Callable[[], Any]) -> None:
    with pytest.raises(ValueError, match="takes a Record"):
        redact(record(), ROOT)
    with pytest.raises(ValueError, match="takes a Record"):
        redact(object.__new__(SubRecord), ROOT)


# ---------------------------------------------------------------------------------------------
# Every closed record


PREFIXES = [(name, n) for name, log in LOGS.items() for n in range(1, len(log) + 1)]


@pytest.mark.parametrize(("name", "n"), PREFIXES, ids=[f"{name}-{n}" for name, n in PREFIXES])
def test_every_prefix_of_the_test_logs_redacts_and_escalates(name: str, n: int) -> None:
    record = read(fresh(LOGS[name][:n]))
    out = view(record)
    assert out["redacted"] == record.record_hash
    assert len(out["log"]) == n
    escalation(record)


def test_a_cause_item_passes_through_and_a_design_only_record_has_nothing_digested() -> None:
    cancelled = read(fresh(LOGS["cancelled"]))
    assert view(cancelled)["log"][-1] == LOGS["cancelled"][-1]
    assert view(read(fresh([DESIGN])))["log"] == [DESIGN]


def test_a_redacted_turn_decide_wrote_digests_its_state_and_text() -> None:
    loaded = loaded_with(escalate)
    record = decide_sync(loaded, {"subject": "secret"}, {}, RunSettings()).record
    out = view(record)
    assert out["log"][2]["value"] == {"subject": digest("secret", record)}
    assert out["log"][-2]["reason"] == digest("a person should look", record)
    assert out["unhashed"]["engine"] == record.unhashed["engine"]


def test_a_round_item_and_a_call_item_without_prompt_pass() -> None:
    log = [*fresh(NORMAL[:3]), call(1, "fetch_mail"), round_item(1)]
    log.insert(4, entry("t:mail/body", 1, 3, None))
    log.insert(5, entry("t:mail/subject", 1, 3, "s"))
    out = view(read(log))["log"]
    assert out[3] == log[3]
    assert out[-1] == log[-1]


def test_a_declared_class_tree_is_the_one_the_ledger_fits_and_redact_reads() -> None:
    # The settings item carries load's tree in full, the declared tree over the default, so the
    # ledger fits each value to it and redact digests by it.
    classes = {"t:state": {"subject": "open"}, "d:flag": "identifier"}
    loaded = loaded_with(escalate, classes=classes)
    record = decide_sync(loaded, {"subject": "secret"}, {}, RunSettings()).record
    assert record.log[1]["classes"]["t:state"] == {"subject": "open"}
    assert record.log[1]["classes"]["d:flag"] == "identifier"
    out = view(record)
    assert out["log"][2]["value"] == {"subject": "secret"}
    (index,) = [i for i, item in enumerate(record.log) if item.get("ref") == "d:flag"]
    assert out["log"][index]["value"] == digest(False, record)
