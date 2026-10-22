# pyright: reportPrivateUsage=false
"""The closed Record, its derived values, and read_record."""

import copy
import enum
import json
from typing import Any

import pytest

import jurat
from counting import SmallBounds, bound, counted
from designs import minimal
from jurat import (
    InvalidRecord,
    Policy,
    Record,
    Rule,
    RunSettings,
    Source,
    decide_sync,
    load,
    read_record,
)
from jurat import types as t
from jurat._canonical import _DecodeError, encode, hash_value
from jurat._ledger import Ledger, _LedgerError, record_prefix_hash
from ledger_logs import (
    BROKEN,
    BROKEN_PLAN,
    CANCELLED,
    CODEC_CANCELLED,
    DESIGN,
    HASH,
    LOGS,
    NORMAL,
    PLAN,
    SETTINGS,
    STATE,
    VERDICT,
    fresh,
)
from scripted import Scripted, returns, sleeping

UNHASHED: dict[str, Any] = {
    "turn": {"start": "2026-10-20T00:00:00Z", "end": "2026-10-20T00:00:01Z"},
    "engine": "jurat 0.0.1",
    "calls": {"3": {"start": "s", "latency": 0.5, "usage": None, "attempts": 1}},
    "exceptions": {"9": {"type": "ValueError", "message": "m", "traceback": "t"}},
    "reads": {"12": "t:state.secret"},
}


def closed(items: list[dict[str, Any]], unhashed: dict[str, Any] | None = None) -> Record:
    ledger = Ledger()
    for item in fresh(items):
        ledger.append(item)
    for member, value in (unhashed or {}).items():
        if member in ("calls", "exceptions", "reads"):
            for key, entry in value.items():
                ledger.write_unhashed(member, entry, int(key))
        else:
            ledger.write_unhashed(member, value)
    return ledger.close()


NO_FORMAT = {k: v for k, v in DESIGN.items() if k != "format"}


def document(log: list[Any], unhashed: Any = None, **members: Any) -> dict[str, Any]:
    return {"log": fresh(log), "unhashed": {} if unhashed is None else unhashed, **members}


def as_bytes(doc: Any) -> bytes:
    return json.dumps(doc).encode("utf-8")


# ---------------------------------------------------------------------------------------------
# The Record


def test_only_the_ledger_builds_a_record() -> None:
    with pytest.raises(TypeError, match="built only by the ledger"):
        Record(object(), (), b"{}")


def test_a_records_attributes() -> None:
    record = closed(NORMAL, UNHASHED)
    assert record.document == encode({"log": NORMAL, "unhashed": UNHASHED})
    assert record.record_hash == hash_value(NORMAL)
    assert record.design_hash == HASH
    assert record.log == NORMAL
    assert record.unhashed == UNHASHED
    assert record.cause is None
    assert "record_hash=" in repr(record)


@pytest.mark.parametrize(
    "name", ["document", "record_hash", "design_hash", "cause", "log", "unhashed"]
)
def test_a_records_attributes_are_read_only(name: str) -> None:
    record = closed(BROKEN)
    with pytest.raises(AttributeError):
        setattr(record, name, None)


def test_log_and_unhashed_are_decoded_afresh_at_each_read() -> None:
    record = closed(NORMAL, UNHASHED)
    log, unhashed = record.log, record.unhashed
    log[0]["hash"] = "changed"
    log.append({})
    unhashed["engine"] = "changed"
    assert record.log is not log
    assert record.log == NORMAL
    assert record.unhashed == UNHASHED
    cause = closed(BROKEN).cause
    assert cause is not None
    cause["code"] = "changed"
    assert closed(BROKEN).cause == BROKEN_PLAN["cause"]


def test_record_values_are_plain_canonical_data() -> None:
    log = [*NORMAL[:3], {"item": "entry", "ref": "d:route", "round": 0, "call": None, "value": 2.0}]
    value = closed(log).log[3]["value"]
    assert value == 2
    assert type(value) is int  # the canonical bytes say 2


def test_the_prefix_hash_has_no_public_name() -> None:
    record = closed(NORMAL)
    assert not hasattr(record, "prefix_hash")
    for n in range(len(NORMAL) + 1):
        assert record_prefix_hash(record, n) == hash_value(NORMAL[:n])


@pytest.mark.parametrize("name", list(LOGS))
def test_every_prefix_from_design_closes_and_reads_back(name: str) -> None:
    log = LOGS[name]
    for n in range(1, len(log) + 1):
        record = closed(log[:n])
        assert record.record_hash == hash_value(log[:n])
        back = read_record(record.document)
        assert back.document == record.document
        assert back.record_hash == record.record_hash


# ---------------------------------------------------------------------------------------------
# The derived cause

ESCALATE: dict[str, Any] = {
    **PLAN,
    "kind": "escalate",
    "steps": [],
    "cause": {"code": "required", "refs": []},
}
NULL_CAUSE_BROKEN = {**BROKEN_PLAN, "cause": None}


@pytest.mark.parametrize(
    ("log", "cause"),
    [
        ([DESIGN], None),
        ([DESIGN, SETTINGS], None),
        ([DESIGN, SETTINGS, STATE], None),
        (NORMAL, None),
        ([*NORMAL[:12], ESCALATE, VERDICT], None),  # a plan's cause with a state entry
        ([*NORMAL[:12], ESCALATE], None),
        (BROKEN, BROKEN_PLAN["cause"]),
        (BROKEN[:3], BROKEN_PLAN["cause"]),  # broken entry, no verdict yet
        ([DESIGN, SETTINGS, NULL_CAUSE_BROKEN], None),
        (CANCELLED, {"code": "cancelled", "refs": []}),
        (CANCELLED[:-1], None),
        (CODEC_CANCELLED, {"code": "cancelled", "refs": []}),
    ],
)
def test_the_derived_cause(log: list[dict[str, Any]], cause: dict[str, Any] | None) -> None:
    assert closed(log).cause == cause


# ---------------------------------------------------------------------------------------------
# read_record


def test_a_record_reads_back_from_its_own_document() -> None:
    record = closed(NORMAL, {**UNHASHED, "future": {"kept": ["as", "stored"]}})
    for doc in (record.document, json.loads(record.document)):
        back = read_record(doc)
        assert type(back) is Record
        assert back.document == record.document
        assert back.record_hash == record.record_hash
        assert back.log == record.log
        assert back.unhashed["future"] == {"kept": ["as", "stored"]}


class Name(enum.StrEnum):
    route = "d:route"
    future = "future"


def test_read_takes_a_parsed_document_with_tuples_and_str_subclasses() -> None:
    doc = document(NORMAL, {Name.future: (Name.route, {Name.route: (1,)})})
    doc["log"] = tuple(doc["log"])
    doc["log"][9]["ref"] = Name.route
    doc["log"][9]["value"] = Name.future
    expected = fresh(NORMAL)
    expected[9]["value"] = "future"
    record = read_record(doc)
    assert record.record_hash == hash_value(expected)
    assert record.unhashed == {"future": ["d:route", {"d:route": [1]}]}


def test_read_does_not_refuse_on_helpers() -> None:
    log = [{**DESIGN, "helpers": 2}, SETTINGS]
    assert read_record(document(log)).log[0]["helpers"] == 2


def nested(depth: int) -> Any:
    value: Any = 0
    for _ in range(depth):
        value = [value]
    return value


def test_a_document_nesting_129_deep_is_refused_before_it_is_parsed() -> None:
    data = b'{"log":[],"unhashed":{"x":' + b"[" * 127 + b"]" * 127 + b"}}"
    with pytest.raises(InvalidRecord, match="128 deep") as info:
        read_record(data)
    assert isinstance(info.value.__cause__, _DecodeError)
    # Unbalanced, so a parse would fail otherwise: the depth is counted first.
    with pytest.raises(InvalidRecord, match="128 deep"):
        read_record(b"[" * 129)


def test_a_parsed_document_nesting_129_deep_is_refused_as_given() -> None:
    with pytest.raises(InvalidRecord, match="128 deep"):
        read_record(document(BROKEN, {"x": nested(127)}))  # 129 deep
    # 128 deep passes the read's bound: a member is a holder, held to the document's.
    assert read_record(document(BROKEN, {"x": nested(126)})).unhashed["x"] == nested(126)


def test_a_parsed_cyclic_document_is_refused_without_hanging() -> None:
    doc = document(BROKEN)
    doc["unhashed"]["self"] = doc["unhashed"]
    with pytest.raises(InvalidRecord, match="a list or object at two places"):
        read_record(doc)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b'{"log":[],"log":[],"unhashed":{}}', "duplicate key 'log'"),
        (b'{"log":[{"item":"design","item":"design"}],"unhashed":{}}', "duplicate key"),
        (b"{not json", "does not parse"),
        (b'{"log":"\xff"}', "does not parse"),
        (b'{"log":[],"unhashed":{}} []', "does not parse"),
        (b'{"log":[NaN],"unhashed":{}}', "does not parse"),
    ],
)
def test_bytes_that_are_not_utf8_json_are_refused_with_the_parse_error_chained(
    data: bytes, match: str
) -> None:
    with pytest.raises(InvalidRecord, match=match) as info:
        read_record(data)
    assert isinstance(info.value.__cause__, _DecodeError)


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ([], "not one JSON object"),
        ("{}", "not one JSON object"),
        ({"unhashed": {}}, "no design item first"),
        (document([]), "no design item first"),
        (document([SETTINGS]), "no design item first"),
        (document(["design"]), "no design item first"),
        ({"log": {"item": "design"}, "unhashed": {}}, "no design item first"),
        (document([{**DESIGN, "format": 3}]), "format 3 is not one this reader knows: 1, 2"),
        (document([{**DESIGN, "format": True}]), "format True is not one"),
        (document([{**DESIGN, "format": 1.0}]), "format 1.0 is not one"),
        (document([NO_FORMAT]), "format None"),
        (document(BROKEN, redacted=HASH), "other than log and unhashed, as a redacted view's"),
        (document(BROKEN, extra=1), r"other than log and unhashed: \['extra'\]"),
        ({"log": fresh(BROKEN)}, "no unhashed member"),
        (document(BROKEN, []), "unhashed is not an object"),
        (document([*BROKEN[:3], {**VERDICT, "x": b"y"}]), "log item 3 is outside the value domain"),
        (document(BROKEN, {"reads": {"1" * 5000: "x"}}), "which names no item"),
        (document(BROKEN, {"calls": {"9" * 25: {}}}), "which names no item"),
        (document([*BROKEN[:2], {**STATE, "value": "\ud800"}]), "surrogate"),
        (document(BROKEN, {"k\udc80": 1}), "unhashed is outside the value domain"),
        (document([*BROKEN[:2], {**STATE, "round": True}]), "'round' is not an integer"),
        (document([*BROKEN[:2], {**STATE, "note": 1}]), "member 'note', which its kind"),
        (document([*BROKEN[:2], STATE, copy.deepcopy(STATE)]), "second entry for 't:state'"),
        (document([*BROKEN, copy.deepcopy(VERDICT)]), "nothing comes after the verdict"),
        (document(BROKEN, {"engine": 1}), "'engine' is not a string"),
        (
            document(
                [*CANCELLED[:-1], {"item": "cause", "cause": {"code": "deadline", "refs": []}}]
            ),
            "cause is not .*cancelled",
        ),
        (document(BROKEN, {"turn": {"end": "e"}}), "has no 'start'"),
        (document(BROKEN, {"calls": {"2": UNHASHED["calls"]["3"]}}), "names plan"),
        (document(BROKEN, {"reads": {"02": "x"}}), "'02', which is no log index"),
        (document(BROKEN, {"reads": {"9": "x"}}), "'9', which names no item"),
        (document(BROKEN, {"reads": {"2": "x", "x": "y"}}), "'x', which is no log index"),
    ],
)
def test_read_refusals(doc: Any, match: str) -> None:
    with pytest.raises(InvalidRecord, match=match):
        read_record(doc)
    # The same document as bytes is refused alike, where JSON can write it.
    if isinstance(doc, dict):
        try:
            data = as_bytes(doc)
        except TypeError:
            return
        with pytest.raises(InvalidRecord, match=match):
            read_record(data)


def test_a_bad_item_is_invalid_record_chained_from_the_ledgers_refusal() -> None:
    with pytest.raises(InvalidRecord, match="the record is invalid: item 2") as info:
        read_record(document([*BROKEN[:2], {**STATE, "round": True}]))
    assert isinstance(info.value.__cause__, _LedgerError)


def test_a_lone_surrogate_in_the_bytes_is_refused_not_escaped() -> None:
    data = encode(document(BROKEN)).replace(b'"engine"', b"")
    data = data.replace(b'"unhashed":{}', b'"unhashed":{"engine":"\\ud800"}')
    with pytest.raises(InvalidRecord, match="surrogate"):
        read_record(data)


def test_unhashed_keys_are_checked_against_the_whole_log() -> None:
    # Key 3 names a call item of the log; the read checks it after every append.
    record = read_record(document(NORMAL, {"calls": UNHASHED["calls"]}))
    assert record.unhashed["calls"] == UNHASHED["calls"]


def test_invalid_record_is_a_public_value_error() -> None:
    assert jurat.InvalidRecord is InvalidRecord
    assert issubclass(InvalidRecord, ValueError)
    assert not issubclass(_LedgerError, InvalidRecord)


def test_a_codec_cancelled_record_reads_back_with_its_cause() -> None:
    # The codec's cancellation is appended after settings, with no state entry.
    unhashed = {"turn": UNHASHED["turn"], "engine": "jurat 0.0.1", "codec": "encode"}
    record = closed(CODEC_CANCELLED, unhashed)
    back = read_record(record.document)
    assert back.document == record.document
    assert back.record_hash == hash_value(CODEC_CANCELLED)
    assert back.cause == {"code": "cancelled", "refs": []}
    assert back.log == CODEC_CANCELLED


def test_read_record_refuses_a_grounded_ref() -> None:
    log = [dict(item) for item in NORMAL]
    log[9] = {**log[9], "ref": 'd:route["a"]'}
    with pytest.raises(InvalidRecord, match="grounded ref"):
        read_record(document(log))


def test_a_decided_record_with_a_deadline_cut_reads_back_with_its_hash_and_log() -> None:
    def after(r: Any) -> str:
        return "yes"

    def policy(r: Any) -> None:
        return None

    design = minimal(
        state=t.record(),
        sources={
            "fast": Source({"t:f": t.string}),
            "slow": Source({"t:s": t.string}),
            "next": Source({"t:n": t.string}, needs=["t:f"]),
        },
        rules={
            "after": Rule("d:after", t.label("no", "yes"), after, version="1", reads=["t:n", "t:s"])
        },
        policy=Policy(policy, version="1", reads=["d:after"]),
    )
    binding = Scripted(
        {
            "fast": returns({"t:f": "f"}),
            "slow": sleeping(10, returns({"t:s": "s"})),
            "next": returns({"t:n": "n"}),
        }
    )
    record = decide_sync(load(design), {}, {"default": binding}, RunSettings(deadline=0.1)).record
    cut = [i for i in record.log if i["item"] == "round" and i["deadline"]]
    assert cut == [{"item": "round", "round": 1, "cost": 0, "deadline": True}]
    unasked: dict[str, Any] = {"kind": "Unasked", "cause": {"code": "deadline", "refs": []}}
    assert [i["marker"] for i in record.log if i.get("ref") == "t:n"] == [unasked]
    back = read_record(record.document)
    assert back.record_hash == record.record_hash
    assert back.log == record.log
    assert back.document == record.document


# ---------------------------------------------------------------------------------------------
# The size bounds at the read: at each value place, not on an item or the document


# These tests run under small bounds (the small_bounds fixture): 2^6 values, 2^8 code points.
_BIG = "x" * 257  # one string past 2^8 code points


_WIDE = ["x" * 16] * 17  # one string at 17 places: past 2^8 code points


def _with(log: list[dict[str, Any]], index: int, *path: Any, to: Any) -> list[dict[str, Any]]:
    items = fresh(log)
    at: Any = items[index]
    for key in path[:-1]:
        at = at[key]
    at[path[-1]] = to
    return items


def _value_places() -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    big = _BIG
    plan = 12
    return [
        ("an entry's value", _with(NORMAL, 4, "value", to=[[0] * 8 for _ in range(8)]), {}),
        ("a class tree", _with(NORMAL, 1, "classes", "d:tags", to=_WIDE), {}),
        ("a plan's reason", _with(NORMAL, plan, "reason", to=big), {}),
        ("a plan's reply", _with(NORMAL, plan, "reply", to=big), {}),
        (
            "a step's argument names",
            _with(NORMAL, plan, "steps", 0, "arguments", to={big: {"literal": 1}}),
            {},
        ),
        (
            "a literal argument",
            _with(NORMAL, plan, "steps", 0, "arguments", "folder", "literal", to=_WIDE),
            {},
        ),
        (
            "a requirement's value",
            _with(NORMAL, plan, "steps", 0, "requirements", 0, "value", to=_WIDE),
            {},
        ),
        ("a call's snapshot", _with(NORMAL, 3, "snapshot", to=big), {}),
        ("a call's elicitation", _with(NORMAL, 3, "elicitation", to=big), {}),
        ("a call's prompt", _with(NORMAL, 7, "prompt", to=big), {}),
        (
            "a verdict failure's requirement value",
            _with(
                _with(NORMAL, 13, "kind", to="refused"),
                13,
                "failures",
                to=[
                    {
                        "step": "plan",
                        "requirement": {"ref": "d:route", "value": _WIDE},
                        "ref": "d:route",
                    }
                ],
            ),
            {},
        ),
        (
            "a call's usage",
            fresh(NORMAL),
            {"calls": {"3": {**UNHASHED["calls"]["3"], "usage": {big: 1}}}},
        ),
    ]


@pytest.mark.parametrize(("place", "log", "unhashed"), _value_places())
def test_read_record_size_tests_each_value_place_before_any_copy(
    place: str, log: list[dict[str, Any]], unhashed: dict[str, Any], small_bounds: SmallBounds
) -> None:
    with counted() as n, pytest.raises(InvalidRecord, match="outside the value domain: the value"):
        read_record({"log": log, "unhashed": unhashed})
    assert n[0] < 4 * bound(), place  # tested before the copy, and stopped past the bound


def test_a_verdict_failures_requirement_value_is_named_by_its_place(
    small_bounds: SmallBounds,
) -> None:
    log = _value_places()[-2][1]
    with pytest.raises(InvalidRecord) as info:
        read_record(document(log))
    assert str(info.value).startswith(
        "the record is invalid: item 13 (verdict)'s 'failures'[0]['requirement']'s 'value' is "
        "outside the value domain"
    )


def test_a_value_is_size_tested_alike_as_bytes_and_parsed(small_bounds: SmallBounds) -> None:
    items = _with(NORMAL, 4, "value", to=_WIDE)
    message = "the value's strings and keys hold more than 2\\^8 code points"
    with pytest.raises(InvalidRecord, match=message):
        read_record(as_bytes(document(items)))  # the bytes share nothing
    with pytest.raises(InvalidRecord, match=message):
        read_record(document(items))


def test_the_ledger_leaves_a_values_size_to_its_producer_live(small_bounds: SmallBounds) -> None:
    ledger = Ledger()
    for item in fresh(_with(NORMAL, 4, "value", to=_WIDE)):
        ledger.append(item)
    with pytest.raises(InvalidRecord, match="2\\^8 code points"):
        read_record(ledger.close().document)


def test_the_document_and_other_places_are_held_to_their_depth_alone(
    small_bounds: SmallBounds,
) -> None:
    big = _BIG
    half = _WIDE[:10]  # within the bounds; over them summed
    items = _with(NORMAL, 4, "value", to=half)
    items = _with(items, 5, "value", to=list(half))
    items = _with(items, 9, "rows", to=list(_WIDE))
    unhashed = {"reads": {"12": big}, "invalid unknown": list(_WIDE)}
    record = read_record({"log": items, "unhashed": unhashed})
    assert len(record.log[9]["rows"]) == len(_WIDE)
    assert len(record.document) > len(big)


def test_a_plan_of_the_wrong_shape_is_left_to_the_shape_checks() -> None:
    items = _with(NORMAL, 12, "steps", to=["not a step"])
    with pytest.raises(InvalidRecord, match="is not an object"):
        read_record(document(items))


def test_a_parsed_record_sharing_a_container_is_refused_before_any_other_walk() -> None:
    shared: Any = []
    for _ in range(60):
        shared = [shared, shared]  # it would expand at every place
    items = _with(NORMAL, 9, "rows", to=shared)
    with pytest.raises(InvalidRecord, match="a list or object at two places"):
        read_record({"log": items, "unhashed": {}})
    one = ["x"]
    twice = fresh(NORMAL)
    twice[4]["value"] = twice[5]["value"] = one
    with pytest.raises(InvalidRecord, match="a list or object at two places"):
        read_record({"log": twice, "unhashed": {}})
    record = closed(fresh(NORMAL))
    assert read_record(json.loads(record.document)).document == record.document
    empties = _with(NORMAL, 4, "value", to={"a": (), "b": ()})
    same: list[Any] = []
    assert read_record(document(_with(empties, 5, "value", to=[same, same]))).log[5]["value"] == [
        [],
        [],
    ]


@pytest.mark.parametrize(
    ("log", "says"),
    [
        ([*fresh(NORMAL[:2]), ["not an item"]], "not an object"),
        (_with(NORMAL, 12, "steps", 0, "arguments", to=["x"]), "is not an object"),
        (_with(NORMAL, 12, "steps", 0, "requirements", to=["x"]), "is not an object"),
    ],
    ids=["item", "arguments", "requirement"],
)
def test_a_value_place_of_the_wrong_shape_is_left_to_the_shape_checks(
    log: list[Any], says: str
) -> None:
    with pytest.raises(InvalidRecord, match=says):
        read_record(document(log))


def test_a_value_place_is_named_without_the_strings_it_holds(small_bounds: SmallBounds) -> None:
    name = "n" * 300  # past the 2^8 code points set: not yet size-tested where it is named
    named = _with(NORMAL, 12, "steps", 0, "arguments", to={name: {"literal": _WIDE}})
    with pytest.raises(InvalidRecord) as info:
        read_record(document(named))
    assert "nnnn" not in str(info.value)
    assert "argument name 0" in str(info.value) or "name 0" in str(info.value)
    ok = "n" * 200  # within the bound, and its literal over it
    literal = _with(NORMAL, 12, "steps", 0, "arguments", to={ok: {"literal": list(_WIDE)}})
    with pytest.raises(InvalidRecord, match="argument 0's literal") as info:
        read_record(document(literal))
    assert "nnnn" not in str(info.value)
    keyed = _with(NORMAL, 1, "classes", to={"k" * 300: list(_WIDE)})
    with pytest.raises(InvalidRecord, match="class tree 0") as info:
        read_record(document(keyed))
    assert "kkkk" not in str(info.value)
    usage = {"calls": {"3" * 300: {**UNHASHED["calls"]["3"], "usage": {"u" * 300: 1}}}}
    with pytest.raises(InvalidRecord, match="'calls' entry 0's 'usage'") as info:
        read_record(document(fresh(NORMAL), usage))
    assert "3333" not in str(info.value)


class _Colliding:
    """A key whose hash is a member name's and whose comparison raises."""

    def __hash__(self) -> int:
        return hash("item")

    armed = False  # set once the record is built, which compares keys too

    def __eq__(self, other: object) -> bool:
        if _Colliding.armed:
            raise RuntimeError("compared")
        return self is other


def test_a_key_whose_comparison_raises_is_an_invalid_record() -> None:
    items = fresh(NORMAL)
    items[4][_Colliding()] = 1  # type: ignore[index]
    _Colliding.armed = True
    try:
        with pytest.raises(InvalidRecord, match="outside the value domain"):
            read_record({"log": items, "unhashed": {}})
    finally:
        _Colliding.armed = False


def test_a_provider_id_is_a_holders_string(small_bounds: SmallBounds) -> None:
    settings = copy.deepcopy(SETTINGS)
    settings["providers"]["t:mail/body"] = "p" * 300  # past the 2^8 code points set
    ledger = Ledger()
    ledger.append(copy.deepcopy(DESIGN))
    ledger.append(settings)
    assert ledger.close().log[1]["providers"]["t:mail/body"] == "p" * 300


class _Armed:
    """A key whose hash is a member name's and whose comparison raises once armed."""

    armed = False

    def __hash__(self) -> int:
        return hash("log")

    def __eq__(self, other: object) -> bool:
        if _Armed.armed:
            raise RuntimeError("compared")
        return self is other

    def __repr__(self) -> str:
        raise RuntimeError("repr")


def test_the_documents_members_are_read_by_their_keys_type_and_data() -> None:
    doc: dict[Any, Any] = {_Armed(): 1, **document(NORMAL)}
    _Armed.armed = True
    try:
        with pytest.raises(InvalidRecord, match=r"other than log and unhashed: \[<_Armed>\]$"):
            read_record(doc)
    finally:
        _Armed.armed = False


def test_an_unknown_members_long_key_is_clipped() -> None:
    doc = document(NORMAL, **{"x" * 1000: 1, "redacted": HASH})
    with pytest.raises(InvalidRecord) as info:
        read_record(doc)
    assert str(info.value).endswith(
        f", as a redacted view's is: [{'x' * 64!r}... (1000 code points), 'redacted']"
    )


def test_a_format_of_a_caller_type_is_named_by_its_type() -> None:
    log = fresh(NORMAL)
    log[0]["format"] = _Armed()
    with pytest.raises(InvalidRecord, match="format <_Armed> is not one this reader knows"):
        read_record(document(log))


def test_a_long_format_string_is_clipped_in_its_message() -> None:
    log = fresh(NORMAL)
    log[0]["format"] = "f" * 10**6
    with pytest.raises(InvalidRecord) as info:
        read_record(document(log))
    assert str(info.value) == (
        f"the record is invalid: the record's format {'f' * 64!r}... (1000000 code points) "
        "is not one this reader knows: 1, 2"
    )


class _Impostor(str):
    """A key of another's string data, unequal to it."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@pytest.mark.parametrize("name", ["log", "unhashed"])
@pytest.mark.parametrize("impostor_first", [True, False])
def test_two_top_level_keys_of_one_string_data_are_an_invalid_record(
    name: str, impostor_first: bool
) -> None:
    doc = document(NORMAL)
    impostor = {_Impostor(name): copy.deepcopy(doc[name])}
    given: dict[Any, Any] = {**impostor, **doc} if impostor_first else {**doc, **impostor}
    with pytest.raises(
        InvalidRecord,
        match=f"^the record is invalid: two object keys have the string data '{name}'$",
    ):
        read_record(given)


class _LoudBytes(bytes):
    """Bytes whose own conversion, length and slice raise: read through bytes' slice."""

    def __bytes__(self) -> bytes:
        raise RuntimeError("bytes")

    def __len__(self) -> int:
        raise RuntimeError("len")

    def __getitem__(self, index: Any) -> Any:
        raise RuntimeError("getitem")


def test_a_bytes_subclass_document_is_read_by_its_data() -> None:
    record = closed(fresh(NORMAL))
    assert read_record(_LoudBytes(record.document)).document == record.document
