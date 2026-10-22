"""The ledger: the log and ``unhashed`` while a turn writes them, the closed ``Record``, and
``read_record``.

Every item enters through ``Ledger.append``, which copies the item to exact ``dict``, ``list``
and ``str``, checks the copy against private check state taken from earlier copies, stores the
copy with its canonical bytes, calls the observer with the index and the stored copy, and
returns the index. A refusal raises ``_LedgerError`` and changes nothing. The closed ledger is
a ``Record``, whose values are decoded from the stored bytes at every read.
"""

import enum
import itertools
import re
from collections.abc import Callable, Mapping
from typing import Any, cast

from jurat._answer import PROBABILITY_KINDS, AnswerKind, ValueType
from jurat._canonical import decode, encode, record_hash
from jurat._markers import CauseCode, MarkerKind, OutcomeKind, PlanKind, VerdictKind
from jurat._refs import declared_ref, parse_ref
from jurat._values import (
    EXTRACTION_FORMAT,
    FORMATS,
    MAX_DEPTH,
    MAX_DOCUMENT_DEPTH,
    MAX_INTEGER,
    MISSING,
    clipped,
    described,
    domain_error,
    member_named,
    members_of,
    nesting_depth,
    shares_a_container,
    string_data,
)

Observer = Callable[[int, dict[str, Any]], None]
"""A sync callable taking each appended item's index and the stored copy."""


class _LedgerError(ValueError):
    """A refusal by the ledger: an engine defect in a turn, ``InvalidRecord`` at the read."""


_ITEM_DEPTH = MAX_DOCUMENT_DEPTH - 2
"""The deepest an item nests: it stands in ``log`` in the document. Its values are each
bounded by ``MAX_DEPTH`` where they stand, counted on the value as recorded."""

_UNHASHED_DEPTH = MAX_DOCUMENT_DEPTH - 1
"""The deepest ``unhashed`` nests: it stands in the document."""

_MEMBER_DEPTH = _UNHASHED_DEPTH - 1
"""The deepest an ``unhashed`` member nests: a holder standing in ``unhashed``, held to no
size bound; a call's ``usage`` within one is a value, size-tested where it stands."""


class InvalidRecord(ValueError):
    """``read_record`` refused a stored record document; the message names the first failure
    found, and the failure is chained."""


# ---------------------------------------------------------------------------------------------
# Copies


def _copy(value: Any) -> Any:
    """A copy of a domain value in exact ``dict``, ``list`` and ``str``: a tuple as a list and a
    ``str`` subclass, keys included, as its string data. The value is in the domain, so the
    recursion is at most ``MAX_DEPTH`` deep."""
    t = type(cast(object, value))
    if t is dict:
        return {string_data(k): _copy(v) for k, v in cast("dict[str, Any]", value).items()}
    if t is list or t is tuple:
        return [_copy(v) for v in cast("list[Any]", value)]
    if isinstance(value, str):
        return string_data(value)
    return value


def _value_places(item: Any, where: str) -> list[tuple[str, Any]]:
    """Each value an item holds where the size bounds apply, read off the item as given, of
    any shape, so each is size-tested before the item is copied or encoded: an entry's
    value, each class tree, a plan's reason and reply and each step's argument names,
    literal arguments and requirement values, each verdict failure's requirement value, and
    a call item's snapshot, elicitation and prompt. A member of the wrong shape is left for
    the shape checks. Members are found by their keys' string data, so no code of a key
    runs, and each place is named by member names and positions alone, never by a string
    the item holds, which is not yet size-tested."""
    kind = member_named(item, "item")
    places: list[tuple[str, Any]] = []

    def put(obj: object, names: tuple[str, ...], at: str) -> None:
        for n in names:
            found = member_named(obj, n)
            if found is not MISSING:
                places.append((f"{at}'s {n!r}", found))

    if type(kind) is str and kind == "entry":
        put(item, ("value",), where)
    elif type(kind) is str and kind == "settings":
        for i, tree in enumerate(members_of(member_named(item, "classes"))):
            places.append((f"{where}'s class tree {i}", tree))
    elif type(kind) is str and kind == "call":
        put(item, ("snapshot", "elicitation", "prompt"), where)
    elif type(kind) is str and kind == "verdict":
        for i, failure in enumerate(_listed(member_named(item, "failures"))):
            at = f"{where}'s 'failures'[{i}]['requirement']"
            put(member_named(failure, "requirement"), ("value",), at)
    elif type(kind) is str and kind == "plan":
        put(item, ("reason", "reply"), where)
        for i, step in enumerate(_listed(member_named(item, "steps"))):
            at = f"{where}'s 'steps'[{i}]"
            arguments = member_named(step, "arguments")
            if type(arguments) is dict:
                given = f"{at}'s 'arguments'"
                pairs = cast("dict[Any, Any]", arguments).items()
                for j, (name, origin) in enumerate(pairs):
                    places.append((f"{given}'s name {j}", name))
                    literal = member_named(origin, "literal")
                    if literal is not MISSING:
                        places.append((f"{given} argument {j}'s literal", literal))
            for k, requirement in enumerate(_listed(member_named(step, "requirements"))):
                put(requirement, ("value",), f"{at}'s 'requirements'[{k}]")
    return places


def _sized(item: object, index: int) -> None:
    """At the read, size-test each value a stored item holds where it stands, before the
    item is walked, copied or encoded; live, its producer tests it."""
    kind = member_named(item, "item")
    known = kind if type(kind) is str and kind in _SHAPES else "of no known kind"
    for place, value in _value_places(item, f"item {index} ({known})"):
        _sized_value(value, place)


def _sized_value(value: Any, where: str) -> None:
    err = domain_error(value)
    if err is not None:
        raise _LedgerError(f"{where} is outside the value domain: {err}")


def _listed(value: object) -> list[Any] | tuple[Any, ...]:
    return cast("list[Any]", value) if type(value) is list or type(value) is tuple else []


_SURROGATE = re.compile("[\ud800-\udfff]")


def _escape(text: str) -> str:
    """Each surrogate code point replaced by its six-character escape in lowercase hex."""
    return _SURROGATE.sub(lambda m: f"\\u{ord(m.group()):04x}", string_data(text))


def _escaped_copy(value: Any, limit: int, depth: int = 0) -> Any:
    """As ``_copy``, each surrogate in a string or a key escaped, a key the escape makes equal
    to an earlier one of its object dropped; two keys with one string data before the
    escape put the object outside the domain. A container nesting past ``limit`` is refused
    here, so a cycle ends; any other type is left for ``domain_error``."""
    t = type(cast(object, value))
    if t is dict or t is list or t is tuple:
        if depth >= limit:
            raise _LedgerError(f"the value nests more than {limit} deep")
        if t is dict:
            out: dict[Any, Any] = {}
            seen: set[str] = set()
            for k, v in cast("dict[Any, Any]", value).items():
                if isinstance(k, str):
                    data = string_data(k)
                    if data in seen:
                        raise _LedgerError(f"two object keys have the string data {data!r}")
                    seen.add(data)
                key = _escape(k) if isinstance(k, str) else k
                if key not in out:
                    out[key] = _escaped_copy(v, limit, depth + 1)
            return out
        return [_escaped_copy(v, limit, depth + 1) for v in cast("list[Any]", value)]
    if isinstance(value, str):
        return _escape(value)
    return value


# ---------------------------------------------------------------------------------------------
# Member checks: JSON kinds only, on exact types (a ``bool`` is never an integer).

_Check = Callable[[Any, str], None]


def _kind(test: Callable[[Any], bool], what: str) -> _Check:
    def check(value: Any, where: str) -> None:
        if not test(value):
            raise _LedgerError(f"{where} is not {what}")

    return check


def _is_int(v: Any) -> bool:
    return type(v) is int


def _is_number(v: Any) -> bool:
    return type(v) is int or type(v) is float


_ANY: _Check = _kind(lambda _: True, "anything")
_STR = _kind(lambda v: type(v) is str, "a string")
_INT = _kind(_is_int, "an integer")
_NUMBER = _kind(_is_number, "a number")
_BOOL = _kind(lambda v: type(v) is bool, "a boolean")
_OBJECT = _kind(lambda v: type(v) is dict, "an object")
_LIST = _kind(lambda v: type(v) is list, "a list")
_STR_OR_NULL = _kind(lambda v: v is None or type(v) is str, "a string or null")
_INT_OR_NULL = _kind(lambda v: v is None or type(v) is int, "an integer or null")
_NUMBER_OR_NULL = _kind(lambda v: v is None or _is_number(v), "a number or null")
_STRINGS = _kind(
    lambda v: type(v) is list and all(type(s) is str for s in cast("list[Any]", v)),
    "a list of strings",
)


def _or_null(check: _Check) -> _Check:
    def or_null(value: Any, where: str) -> None:
        if value is not None:
            check(value, where)

    return or_null


def _list_of(check: _Check) -> _Check:
    def list_of(value: Any, where: str) -> None:
        _LIST(value, where)
        for i, member in enumerate(cast("list[Any]", value)):
            check(member, f"{where}[{i}]")

    return list_of


def _members(
    obj: Any, where: str, required: Mapping[str, _Check], optional: Mapping[str, _Check] = {}
) -> dict[str, Any]:
    """Refuse an object with a member its kind does not name, a missing required member, or a
    member of the wrong JSON kind."""
    _OBJECT(obj, where)
    members = cast("dict[str, Any]", obj)
    for name in members:
        if name not in required and name not in optional:
            raise _LedgerError(f"{where} has the member {name!r}, which its kind does not name")
    for name in required:
        if name not in members:
            raise _LedgerError(f"{where} has no {name!r} member")
    for name, value in members.items():
        check = required.get(name) or optional[name]
        check(value, f"{where}'s {name!r}")
    return members


def _cause(value: Any, where: str) -> None:
    """A cause as a marker holds one: a code from the closed set, and refs as strings."""
    m = _members(value, where, {"code": _STR, "refs": _STRINGS})
    if m["code"] not in _CAUSE_CODES:
        raise _LedgerError(f"{where}'s code {m['code']!r} is not in the closed cause set")


_STATE = "t:state"
_CAUSE_CODES = frozenset(c.value for c in CauseCode)
_ANSWER_KINDS = frozenset(AnswerKind)
_PROBABILITY_KINDS = frozenset(k.value for k in PROBABILITY_KINDS)
_VALUE_TYPES = frozenset(v.value for v in ValueType)
_PLAN_KINDS = frozenset(PlanKind)
_VERDICT_KINDS = frozenset(VerdictKind)
_MARKER_KINDS = frozenset(k.value for k in MarkerKind)


def _marker(value: Any, where: str) -> None:
    m = _members(value, where, {"kind": _STR, "cause": _cause})
    if m["kind"] not in _MARKER_KINDS:
        raise _LedgerError(f"{where}'s kind {m['kind']!r} is not Unsure, Unavailable or Unasked")


def _outcome(value: Any, where: str) -> None:
    if value == OutcomeKind.returned:
        return
    if type(value) is dict and list(cast("dict[str, Any]", value)) == ["failed"]:
        _cause(value["failed"], f"{where}'s 'failed'")
        return
    raise _LedgerError(f'{where} is neither "returned" nor {{"failed": cause}}')


def _questions(value: Any, where: str) -> None:
    _OBJECT(value, where)
    for ref, question in cast("dict[str, Any]", value).items():
        try:
            parsed = parse_ref(ref)
        except ValueError:
            parsed = None
        if parsed is None or parsed.level != "q":
            raise _LedgerError(f"{where} has the key {ref!r}, which is not a q: ref")
        if parsed.fields:
            raise _LedgerError(f"{where} has the key {ref!r}, which has a field path")
        at = f"{where}[{ref!r}]"
        kind = member_named(cast(object, question), "kind")
        if kind == AnswerKind.extraction:
            q = _members(question, at, _EXTRACTION_QUESTION)
        else:
            q = _members(question, at, {"identity": _STR, "kind": _STR, "labels": _ANY})
        kind, labels = q["kind"], q["labels"]
        if kind not in _ANSWER_KINDS:
            raise _LedgerError(f"{at}'s kind {kind!r} is not noul, choice, score or extraction")
        if kind == AnswerKind.extraction:
            if labels is not None:
                raise _LedgerError(f"{at}'s labels are not null, as an extraction's are")
            if q["value_type"] not in _VALUE_TYPES:
                raise _LedgerError(f"{at}'s value_type {q['value_type']!r} is not one of the six")
        elif kind == AnswerKind.noul:
            if labels is not None:
                raise _LedgerError(f"{at}'s labels are not null, as noul's are")
        elif type(cast(object, labels)) is dict and kind == AnswerKind.choice:
            _members(labels, f"{at}'s labels", {"fixed": _STRINGS, "candidates": _STR})
        elif type(cast(object, labels)) is dict:
            raise _LedgerError(f"{at}'s labels are an object, which only a choice's may be")
        else:
            _STRINGS(labels, f"{at}'s labels")


def _sources(value: Any, where: str) -> None:
    """A batch's source ids: strings, at least one, strictly in code-point order."""
    _STRINGS(value, where)
    ids = cast("list[str]", value)
    if not ids:
        raise _LedgerError(f"{where} is empty; a call item lists the sources it asked")
    for before, after in itertools.pairwise(ids):
        if before >= after:
            raise _LedgerError(f"{where}: {before!r}, {after!r} are not in code-point order")


def _answer(value: Any, where: str) -> None:
    """An answer's shape: no label, probability, range, grammar or span bound is tested,
    which interpret's answer check does under replay as on a live turn."""
    if member_named(cast(object, value), "kind") == AnswerKind.extraction:
        m = _members(value, where, _EXTRACTION_ANSWER)
        if m["value_type"] not in _VALUE_TYPES:
            raise _LedgerError(f"{where}'s value_type {m['value_type']!r} is not one of the six")
        return
    m = _members(value, where, {"kind": _STR, "labels": _STRINGS, "probabilities": _OBJECT})
    if m["kind"] not in _PROBABILITY_KINDS:
        raise _LedgerError(f"{where}'s kind {m['kind']!r} is not noul, choice or score")


def _is_member_value(v: Any) -> bool:
    """An extraction ``value`` or ``guess``: null, a string, a number, or a list of those."""
    if v is None or type(v) is str or _is_number(v):
        return True
    return type(v) is list and all(
        type(m) is str or _is_number(m) or m is None for m in cast("list[Any]", v)
    )


def _is_pair(v: Any) -> bool:
    return (
        type(v) is list
        and len(cast("list[Any]", v)) == 2
        and all(map(_is_int, cast("list[Any]", v)))
    )


def _is_span(v: Any) -> bool:
    """An extraction ``span``: null, a list of two integers, or a list of such lists."""
    if v is None or _is_pair(v):
        return True
    return type(v) is list and all(_is_pair(p) for p in cast("list[Any]", v))


def _run_settings(value: Any, where: str) -> None:
    _members(
        value,
        where,
        {
            "round_bound": _INT,
            "timeout": _NUMBER,
            "budget": _NUMBER_OR_NULL,
            "deadline": _NUMBER_OR_NULL,
            "certificate_cap": _INT_OR_NULL,
        },
    )


def _value(value: Any, where: str) -> None:
    """A value as recorded: of the domain's types, nesting at most ``MAX_DEPTH`` deep where it
    stands. Its size is its producer's to test live, and the read's (``_sized``)."""
    err = domain_error(value, MAX_DEPTH)
    if err is not None:
        raise _LedgerError(f"{where} is outside the value domain: {err}")


def _requirement(value: Any, where: str) -> None:
    _members(value, where, {"ref": _STR, "value": _value})


def _arguments(value: Any, where: str) -> None:
    _OBJECT(value, where)
    for name, origin in cast("dict[str, Any]", value).items():
        if type(origin) is dict and "literal" in origin:
            _value(origin["literal"], f"{where}[{name!r}]'s literal")


def _step(value: Any, where: str) -> None:
    _members(
        value,
        where,
        {"tool": _STR, "arguments": _arguments, "requirements": _list_of(_requirement)},
    )


def _failure_step(value: Any, where: str) -> None:
    if not ((type(value) is int and value >= 1) or value == "plan"):
        raise _LedgerError(f'{where} is neither an integer of at least 1 nor "plan"')


def _failure(value: Any, where: str) -> None:
    """A failure the record views can build a ``GuardFailure`` from."""
    m = _members(
        value,
        where,
        {"step": _failure_step, "requirement": _or_null(_requirement), "ref": _STR_OR_NULL},
    )
    requirement, ref = m["requirement"], m["ref"]
    if (requirement is None) != (ref is None):
        raise _LedgerError(f"{where}'s requirement and ref are not null together")
    if requirement is None and m["step"] == "plan":
        raise _LedgerError(f'{where} has a null requirement and ref beside the step "plan"')
    if requirement is not None and ref != requirement["ref"]:
        raise _LedgerError(f"{where}'s ref is not its requirement's ref")


def _entry_shape(item: dict[str, Any], where: str) -> None:
    _members(
        item,
        where,
        {"item": _STR, "ref": _STR, "round": _INT, "call": _INT_OR_NULL},
        {"value": _value, "marker": _marker, "rows": _LIST, "derivation": _LIST},
    )
    if ("value" in item) == ("marker" in item):
        raise _LedgerError(f"{where} has both value and marker, or neither")
    ref: str = item["ref"]
    if ref == _STATE and "marker" in item:
        raise _LedgerError(f"{where} is a t:state entry holding a marker; the state is a value")
    try:
        level = parse_ref(ref).level
    except ValueError:  # not a ref: _entry refuses it with its own message
        level = None
    if level == "q" and "value" in item:
        _answer(item["value"], f"{where}'s 'value'")


def _verdict_shape(item: dict[str, Any], where: str) -> None:
    _members(item, where, {"item": _STR, "kind": _STR, "failures": _list_of(_failure)})
    if item["kind"] not in _VERDICT_KINDS:
        raise _LedgerError(f"{where}'s kind {item['kind']!r} is not passed or refused")
    if (item["kind"] == VerdictKind.passed) != (not item["failures"]):
        raise _LedgerError(f"{where}'s failures are not empty exactly when its kind is passed")


def _cause_item_shape(item: dict[str, Any], where: str) -> None:
    _members(item, where, {"item": _STR, "cause": _cause})
    if item["cause"] != {"code": "cancelled", "refs": []}:
        raise _LedgerError(f'{where}\'s cause is not {{"code": "cancelled", "refs": []}}')


def _shape(required: Mapping[str, _Check], optional: Mapping[str, _Check] = {}) -> _Check:
    def shape(item: Any, where: str) -> None:
        _members(item, where, {"item": _STR, **required}, optional)

    return shape


def _then_kind(kinds: frozenset[str], what: str, shape: _Check) -> _Check:
    def check(item: Any, where: str) -> None:
        shape(item, where)
        if item["kind"] not in kinds:
            raise _LedgerError(f"{where}'s kind {item['kind']!r} is not {what}")

    return check


_EXTRACTION_ANSWER: dict[str, _Check] = {
    "kind": _STR,
    "value_type": _STR,
    "is_list": _BOOL,
    "value": _kind(_is_member_value, "null, a string, a number or a list of those"),
    "guess": _kind(_is_member_value, "null, a string, a number or a list of those"),
    "p_null": _NUMBER_OR_NULL,
    "confidence": _NUMBER_OR_NULL,
    "span": _kind(_is_span, "null, a list of two integers or a list of such lists"),
}
_EXTRACTION_QUESTION: dict[str, _Check] = {
    "identity": _STR,
    "kind": _STR,
    "labels": _ANY,
    "value_type": _STR,
    "is_list": _BOOL,
}

_SHAPES: dict[str, _Check] = {
    "design": _shape({"hash": _STR, "format": _INT, "helpers": _INT}),
    "settings": _shape({"settings": _run_settings, "providers": _OBJECT, "classes": _OBJECT}),
    "entry": cast(_Check, _entry_shape),
    "call": _shape(
        {
            "round": _INT,
            "sources": _sources,
            "outcome": _outcome,
            "snapshot": _STR_OR_NULL,
            "elicitation": _STR_OR_NULL,
            "prompt": _STR_OR_NULL,
            "questions": _questions,
        }
    ),
    "round": _shape({"round": _INT, "cost": _NUMBER, "deadline": _BOOL}),
    "certificate": _shape({"round": _INT}),
    "plan": _then_kind(
        _PLAN_KINDS,
        "a plan kind",
        _shape(
            {
                "kind": _STR,
                "steps": _list_of(_step),
                "reason": _STR_OR_NULL,
                "reply": _STR_OR_NULL,
                "cause": _or_null(_cause),
            },
            {"slots": _OBJECT, "rows": _LIST, "derivation": _LIST},
        ),
    ),
    "verdict": cast(_Check, _verdict_shape),
    "cause": cast(_Check, _cause_item_shape),
}


# ---------------------------------------------------------------------------------------------
# Grounded refs, refused in v0.0.1 wherever redaction rewrites one


def _grounded(ref: Any) -> bool:
    if type(ref) is not str:
        return False
    try:
        return parse_ref(ref).candidate is not None
    except ValueError:  # not a ref: a role name, rule:N or step:N
        return False


def _cause_refs(cause: Any, place: str) -> list[tuple[str, Any]]:
    return [(f"{place}'s refs", r) for r in cast("dict[str, Any]", cause)["refs"]]


def _ref_places(kind: str, item: dict[str, Any]) -> list[tuple[str, Any]]:
    """Each place of an item that redaction rewrites a grounded ref in: an entry's ref, a
    cause's refs, an origin's ref, a requirement's ref and a failure's ref. The item has
    passed its shape check."""
    places: list[tuple[str, Any]] = []
    if kind == "entry":
        places.append(("its ref", item["ref"]))
        if "marker" in item:
            places += _cause_refs(item["marker"]["cause"], "its marker's cause")
    elif kind == "call" and type(item["outcome"]) is dict:
        places += _cause_refs(item["outcome"]["failed"], "its outcome's cause")
    elif kind == "cause":
        places += _cause_refs(item["cause"], "its cause")
    elif kind == "plan":
        if item["cause"] is not None:
            places += _cause_refs(item["cause"], "its cause")
        for n, step in enumerate(cast("list[dict[str, Any]]", item["steps"]), 1):
            for name, origin in cast("dict[str, Any]", step["arguments"]).items():
                if type(origin) is dict and "ref" in origin:
                    ref: Any = cast("dict[str, Any]", origin)["ref"]
                    places.append((f"step {n}'s argument {name!r}", ref))
            for r in cast("list[dict[str, Any]]", step["requirements"]):
                places.append((f"step {n}'s requirement", r["ref"]))
    elif kind == "verdict":
        for i, failure in enumerate(cast("list[dict[str, Any]]", item["failures"])):
            places.append((f"failure {i}'s ref", failure["ref"]))
            if failure["requirement"] is not None:
                places.append((f"failure {i}'s requirement", failure["requirement"]["ref"]))
    return places


def _no_grounded_ref(kind: str, item: dict[str, Any], where: str) -> None:
    for place, ref in _ref_places(kind, item):
        if _grounded(ref):
            raise _LedgerError(
                f"{where}: {place} {ref!r} is a grounded ref, which a v0.0.1 record holds nowhere"
            )


def _no_set(kind: str, item: dict[str, Any], where: str) -> None:
    """No origin carrying ``set``, null included: nothing checks a candidate-value literal's
    set in v0.0.1. The item has passed its shape check."""
    if kind != "plan":
        return
    for n, step in enumerate(cast("list[dict[str, Any]]", item["steps"]), 1):
        for name, origin in cast("dict[str, Any]", step["arguments"]).items():
            if type(origin) is dict and "set" in origin:
                raise _LedgerError(
                    f"{where}: step {n}'s argument {name!r} carries set, "
                    "which a v0.0.1 record holds nowhere"
                )


# ---------------------------------------------------------------------------------------------
# Class trees


class _ClassName(enum.StrEnum):
    """The three redaction classes."""

    open = "open"
    identifier = "identifier"
    content = "content"


_CLASS_NAMES = frozenset(_ClassName)


def _tree_malformed(tree: Any) -> bool:
    if type(tree) is str:
        return tree not in _CLASS_NAMES
    if type(tree) is dict:
        return any(_tree_malformed(t) for t in cast("dict[str, Any]", tree).values())
    if type(tree) is list:
        members = cast("list[Any]", tree)
        return len(members) != 1 or _tree_malformed(members[0])
    return True


def _misfit(value: Any, tree: Any, where: str) -> str | None:
    """Where ``value`` does not fit the well-formed ``tree``, or ``None`` when it fits."""
    if value is None:
        return None
    if type(value) is list:
        for i, member in enumerate(cast("list[Any]", value)):
            found = _misfit(member, tree, f"{where}[{i}]")
            if found is not None:
                return found
        return None
    if type(tree) is str:
        return None
    if type(value) is not dict:
        return f"{where} is not an object, as its class tree needs"
    members = cast("dict[str, Any]", value)
    if type(tree) is dict:
        nodes = cast("dict[str, Any]", tree)
        for key, member in members.items():
            if key not in nodes:
                return f"{where} has the key {key!r}, which its class tree does not list"
            found = _misfit(member, nodes[key], f"{where}[{key!r}]")
            if found is not None:
                return found
        return None
    beneath = cast("list[Any]", tree)[0]
    for key, member in members.items():
        found = _misfit(member, beneath, f"{where}[{key!r}]")
        if found is not None:
            return found
    return None


# ---------------------------------------------------------------------------------------------
# unhashed shapes

_KEYED = frozenset({"calls", "exceptions", "reads"})
_INVALID_NAMES = frozenset(
    {"code", "snapshot", "elicitation", "prompt", "usage", "attempts", "truncated"}
)
_NUMBERS = _kind(
    lambda v: type(v) is dict and all(_is_number(n) for n in cast("dict[str, Any]", v).values()),
    "an object of numbers",
)


def _unexpected(value: Any, where: str) -> None:
    _STRINGS(value, where)
    if not value:
        raise _LedgerError(f"{where} is empty; an empty member is not written")


def _invalid(value: Any, where: str) -> None:
    _OBJECT(value, where)
    if not value:
        raise _LedgerError(f"{where} is empty; an empty member is not written")
    for name, text in cast("dict[str, Any]", value).items():
        if name not in _INVALID_NAMES:
            raise _LedgerError(f"{where} has the key {name!r}, which is not a known member")
        _STR(text, f"{where}[{name!r}]")


def _sums(value: Any, where: str) -> None:
    _OBJECT(value, where)
    if not value:
        raise _LedgerError(f"{where} is empty; an empty member is not written")
    for key, number in cast("dict[str, Any]", value).items():
        _NUMBER(number, f"{where}[{clipped(key)}]")


def _truncated(value: Any, where: str) -> None:
    """The refs a binding reported cut short: a non-empty list of strings, distinct and in
    code-point order."""
    _STRINGS(value, where)
    if not value:
        raise _LedgerError(f"{where} is empty; an empty member is not written")
    for before, after in itertools.pairwise(cast("list[str]", value)):
        if before >= after:
            raise _LedgerError(
                f"{where}: {clipped(before)}, {clipped(after)} are not distinct and in code-point "
                "order"
            )


def _call_entry(value: Any, where: str) -> None:
    _members(
        value,
        where,
        {
            "start": _STR,
            "latency": _NUMBER,
            "usage": _or_null(_NUMBERS),
            "attempts": _INT_OR_NULL,
        },
        {"unexpected": _unexpected, "invalid": _invalid, "sums": _sums, "truncated": _truncated},
    )


def _sums_keys(entry: dict[str, Any], questions: Mapping[str, str], where: str) -> None:
    """Refuse a ``sums`` key or a ``truncated`` ref that names no key of its call item's
    ``questions``, looked up with any grounded bracket removed, and in v0.0.1 a grounded one.
    The entry has passed its shape check."""
    named = [("sums", key) for key in cast("dict[str, Any]", entry.get("sums", {}))]
    named += [("truncated", ref) for ref in cast("list[str]", entry.get("truncated", []))]
    for member, key in named:
        try:
            found = declared_ref(key) in questions
        except ValueError:  # not a ref
            found = False
        what = "key" if member == "sums" else "ref"
        if not found:
            raise _LedgerError(
                f"{where}'s {member!r} has the {what} {clipped(key)}, which names no key of its "
                "call item's questions"
            )
        if _grounded(key):
            raise _LedgerError(
                f"{where}'s {member!r} {what} {clipped(key)} is a grounded ref, which a v0.0.1 "
                "record holds nowhere"
            )


def _exception_entry(value: Any, where: str) -> None:
    _members(value, where, {"type": _STR, "message": _STR, "traceback": _STR})


_KEYED_ENTRY: dict[str, _Check] = {
    "calls": _call_entry,
    "exceptions": _exception_entry,
    "reads": _STR,
}


# The item kinds each keyed member's log index names.
_KEYED_KINDS: dict[str, frozenset[str]] = {
    "calls": frozenset({"call"}),
    "exceptions": frozenset({"call", "entry", "plan"}),
    "reads": frozenset({"entry", "plan"}),
}
_INDEX_KEY = re.compile(r"0|[1-9][0-9]*")


def _check_unhashed(
    member: str,
    value: Any,
    kinds: list[str],
    questions: Mapping[int, Mapping[str, str]],
    fmt: int,
) -> None:
    """Refuse a known ``unhashed`` member whose shape is not the observer's, or a key of a keyed
    member that is not a decimal log index naming an item of ``kinds``, the log's item kinds
    so far, of the kind the member is keyed by, or a ``calls`` entry's ``sums`` key that
    names no key of its call item's ``questions``, kept by index; an unknown member is kept
    as written."""
    where = f"unhashed {member!r}"
    if member == "turn":
        _members(value, where, {"start": _STR}, {"end": _STR})
    elif member in ("engine", "codec"):
        _STR(value, where)
    elif member in _KEYED:
        _OBJECT(value, where)
        for key, entry in cast("dict[str, Any]", value).items():
            if not _INDEX_KEY.fullmatch(key):
                raise _LedgerError(
                    f"{where} has the key {clipped(key)}, which is no log index in decimal"
                )
            # A key longer than the log's length names no item; int() is never given more.
            index = len(kinds) if len(key) > len(str(len(kinds))) else int(key)
            if index >= len(kinds) or kinds[index] not in _KEYED_KINDS[member]:
                named = kinds[index] if index < len(kinds) else "no item"
                raise _LedgerError(
                    f"{where} has the key {key!r}, which names {named}, not an item of the kind "
                    f"{', '.join(sorted(_KEYED_KINDS[member]))}"
                )
            _KEYED_ENTRY[member](entry, f"{where}[{key!r}]")
            if member == "calls":
                if fmt < EXTRACTION_FORMAT and (
                    "truncated" in entry or "truncated" in entry.get("invalid", {})
                ):
                    raise _LedgerError(
                        f"{where}[{key!r}] holds truncated, which a format-{fmt} record does not"
                    )
                _sums_keys(entry, questions[index], f"{where}[{key!r}]")


# ---------------------------------------------------------------------------------------------
# The ledger

_STATE_NODE = ("state",)

_Grow = tuple[Callable[[], None], Callable[[], None]]
"""A change the commit makes in place, and its undo."""

# Phases of the kind order.
_EMPTY, _DESIGNED, _SETTLED, _BODY, _PLANNED, _VERDICT, _CAUSED = range(7)


class Ledger:
    """The open ledger: no public name. Decide opens it and closes it."""

    def __init__(self, observer: Observer | None = None) -> None:
        self._observer = observer
        self._items: list[dict[str, Any]] = []  # the stored copies, handed to the observer
        self._bytes: list[bytes] = []
        self._kinds: list[str] = []  # each item's kind, kept apart from the stored copies
        self._unhashed: dict[str, Any] = {}
        self._record: Record | None = None
        self.observer_error: BaseException | None = None
        """The exception the observer call raised: the observer's mark. Set once,
        never cleared; decide tells the observer's exception from another by identity."""
        self.drifted_copies: tuple[int, ...] = ()
        """At close, the indices of stored copies that no longer encode to their bytes."""
        # Check state, taken off each copy before the observer sees it.
        self._phase = _EMPTY
        self._providers: dict[str, bytes] = {}
        self._classes: dict[str, Any] = {}
        self._round = 0
        self._round_ended = False
        self._round_has_entry = False
        self._round_calls: set[int] = set()
        self._round_bound = 0
        self._cut: int | None = None  # the round whose round item has deadline true
        self._sources: dict[str, int] = {}  # each listed source id, by its call item's index
        self._questions: dict[int, dict[str, str]] = {}  # each call item's questions' kinds
        self._format = 0  # the design item's format, which the rest of the log is read under
        self._written: set[str] = set()
        self._node: tuple[object, ...] | None = None
        self._node_last = ""
        self._done_nodes: set[tuple[object, ...]] = set()

    @property
    def closed(self) -> bool:
        return self._record is not None

    def append(self, item: object) -> int:
        """Check ``item``, store a copy with its canonical bytes, call the observer with the
        index and the stored copy, and return the index; a refusal writes nothing."""
        if self.closed:
            raise _LedgerError("the ledger is closed: nothing is appended after close")
        err = domain_error(item, _ITEM_DEPTH)
        if err is not None:
            raise _LedgerError(f"the item is outside the value domain: {err}")
        copy = _copy(item)
        if type(copy) is not dict:
            raise _LedgerError("the item is not an object")
        stored = cast("dict[str, Any]", copy)
        kind = stored.get("item")
        index = len(self._items)
        where = f"item {index}"
        shape = _SHAPES.get(kind) if type(kind) is str else None
        if shape is None:
            raise _LedgerError(f"{where} is of no item kind the ledger knows: {kind!r}")
        where = f"{where} ({kind})"
        shape(stored, where)
        _no_grounded_ref(cast(str, kind), stored, where)
        _no_set(cast(str, kind), stored, where)
        commit = self._check_order(cast(str, kind), stored, index, where)
        data = encode(stored, _ITEM_DEPTH)
        try:
            self._items.append(stored)
            self._bytes.append(data)
            self._kinds.append(cast(str, kind))
            commit()  # undoes its own grows if one raises
        except BaseException:
            # Nothing of the item stays: an error past the checks changes nothing either.
            del self._items[index:], self._bytes[index:], self._kinds[index:]
            raise
        if self._observer is not None:
            try:
                self._observer(index, stored)
            except BaseException as exc:
                if self.observer_error is None:
                    self.observer_error = exc
                raise
        return index

    # The order checks compute the new check state and return the commit that sets it: the
    # members it adds to the sets and the map kept in place, each grow with its undo, then the
    # values it assigns, so an append costs no copy of them and a refusal, before the commit,
    # changes nothing.

    def _check_order(
        self, kind: str, item: dict[str, Any], index: int, where: str
    ) -> Callable[[], None]:
        phase = self._phase
        if kind in ("design", "settings") and phase > (_EMPTY if kind == "design" else _DESIGNED):
            raise _LedgerError(f"{where}: the {kind} item comes once")
        if phase == _EMPTY and kind != "design":
            raise _LedgerError(f"{where}: the design item comes first")
        if phase == _DESIGNED and kind != "settings":
            raise _LedgerError(f"{where}: the settings item comes second")
        if phase == _VERDICT:
            raise _LedgerError(f"{where}: nothing comes after the verdict item")
        if phase == _CAUSED:
            raise _LedgerError(f"{where}: nothing comes after the cause item")
        if phase == _PLANNED and kind != "verdict":
            raise _LedgerError(f"{where}: only the verdict item comes after the plan item")
        if kind == "verdict" and phase != _PLANNED:
            raise _LedgerError(f"{where}: the verdict item comes after the plan item")
        if (
            phase == _SETTLED
            and kind not in ("plan", "cause")
            and not (kind == "entry" and item["ref"] == _STATE and item["round"] == 0)
        ):
            # A cause here is a cancellation a sync component raised.
            raise _LedgerError(
                f"{where}: after the settings item comes the t:state entry in round 0, "
                "at a broken entry the plan item, or at a cancellation the cause item"
            )
        if kind == "design":
            fmt = item["format"]
            if fmt not in FORMATS:
                raise _LedgerError(f"{where}: the format {fmt} is not one this ledger knows")
            return self._set(_phase=_DESIGNED, _format=fmt)
        if kind == "settings":
            return self._settings(item)
        if kind == "verdict":
            return self._set(_phase=_VERDICT)
        if kind == "plan":
            self._require_round_ended(where)
            return self._set(_phase=_PLANNED)
        if kind == "cause":
            # It may end a round with no round item: a cancellation raised other than at an
            # await keeps every item before it.
            return self._set(_phase=_CAUSED)
        return self._round_item(kind, item, index, where)

    def _set(self, *grows: "_Grow", **state: object) -> Callable[[], None]:
        def commit() -> None:
            # The grows first, which allocate, each undone in reverse if a later one raises;
            # the assignments, which cannot fail, last.
            undos: list[Callable[[], None]] = []
            try:
                for grow, undo in grows:
                    grow()
                    undos.append(undo)
            except BaseException:
                for undo in reversed(undos):
                    undo()
                raise
            for name, value in state.items():
                setattr(self, name, value)

        return commit

    def _settings(self, item: dict[str, Any]) -> Callable[[], None]:
        # Private copies: the observer may change the stored one.
        # Each provider id is a holder's string, not a value: no size bound.
        providers = {
            k: encode(v, _ITEM_DEPTH) for k, v in cast("dict[str, Any]", item["providers"]).items()
        }
        classes = _copy(item["classes"])
        bound = cast(int, item["settings"]["round_bound"])
        return self._set(
            _phase=_SETTLED, _providers=providers, _classes=classes, _round_bound=bound
        )

    def _require_round_ended(self, where: str) -> None:
        if self._round >= 1 and not self._round_ended:
            raise _LedgerError(f"{where}: round {self._round} has no round item before it")

    def _round_item(
        self, kind: str, item: dict[str, Any], index: int, where: str
    ) -> Callable[[], None]:
        n: int = item["round"]
        if n == 0 and kind in ("call", "round", "certificate"):
            raise _LedgerError(f"{where}: round 0 has no {kind} item")
        if n < self._round:
            raise _LedgerError(f"{where}: round {n} is below the last round, {self._round}")
        if n > self._round + 1:
            raise _LedgerError(f"{where}: round {n} skips a number after round {self._round}")
        # Interpret stops after the round the deadline cuts and at the round bound.
        if self._cut is not None and n > self._cut:
            raise _LedgerError(
                f"{where}: round {n} comes after round {self._cut}, which the deadline cut"
            )
        if n > self._round_bound:
            raise _LedgerError(f"{where}: round {n} is above the round bound, {self._round_bound}")
        new = n > self._round
        if new:
            self._require_round_ended(where)
        elif self._round_ended:
            what = "two round items" if kind == "round" else "an item after its round item"
            raise _LedgerError(f"{where}: round {n} holds {what}")
        has_entry = False if new else self._round_has_entry
        calls: set[int] = set() if new else self._round_calls
        state: dict[str, object] = {"_phase": _BODY, "_round": n}
        grows: list[_Grow] = []
        if new:
            state.update(_round_ended=False, _round_has_entry=False, _round_calls=calls)
        if kind == "certificate" and not new:
            raise _LedgerError(f"{where}: the certificate item comes first in its round")
        if kind == "round":
            state["_round_ended"] = True
            if item["deadline"] is True:
                state["_cut"] = n
        elif kind == "call":
            if has_entry:
                raise _LedgerError(f"{where}: a call item comes before the entries of its round")
            sources = tuple(self._call_sources(item, where))  # not the copy the observer sees
            asked = {
                ref: cast(str, q["kind"])
                for ref, q in cast("dict[str, dict[str, Any]]", item["questions"]).items()
            }
            if self._format < EXTRACTION_FORMAT and AnswerKind.extraction in asked.values():
                raise _LedgerError(
                    f"{where}: its questions hold an extraction question, which a "
                    f"format-{self._format} record does not"
                )
            questions = self._questions

            def kept() -> None:
                questions[index] = asked

            def unkept() -> None:
                questions.pop(index, None)

            def listed() -> None:
                self._sources.update(dict.fromkeys(sources, index))

            def unlisted() -> None:  # none of them was listed before: the check refuses it
                for sid in sources:
                    self._sources.pop(sid, None)

            grows.append((listed, unlisted))
            grows.append((kept, unkept))
            grows.append((lambda: calls.add(index), lambda: calls.discard(index)))
        elif kind == "entry":
            call = item["call"]
            if call is not None and call not in calls:
                raise _LedgerError(f"{where}: its call {call} names no call item of round {n}")
            self._asked(item, None if call is None else self._questions[call], where)
            state["_round_has_entry"] = True
            entry_state, entry_grows = self._entry(item, where)
            state.update(entry_state)
            grows += entry_grows
        return self._set(*grows, **state)

    def _asked(self, item: dict[str, Any], questions: Mapping[str, str] | None, where: str) -> None:
        """A ``q:`` entry naming a call item names a question of it; a value entry's answer is
        of a kind its format has and, naming a call item, of the kind that question gives."""
        ref: str = item["ref"]
        try:
            parsed = parse_ref(ref)
        except ValueError:  # not a ref: _entry refuses it with its own message
            return
        if parsed.level != "q":
            return
        kind = item["value"]["kind"] if "value" in item else None
        if self._format < EXTRACTION_FORMAT and kind == AnswerKind.extraction:
            raise _LedgerError(
                f"{where}: an extraction answer, which a format-{self._format} record does not hold"
            )
        if questions is None:
            return
        key = declared_ref(ref)
        if key not in questions:
            raise _LedgerError(f"{where}: {key!r} is not a question of its call item")
        if kind is not None and kind != questions[key]:
            raise _LedgerError(
                f"{where}: its answer's kind {kind!r} is not its question's, {questions[key]!r}"
            )

    def _call_sources(self, item: dict[str, Any], where: str) -> list[str]:
        """This call item's sources, none listed before; a source is called once."""
        sources = cast("list[str]", item["sources"])
        for sid in sources:
            earlier = self._sources.get(sid)
            if earlier is not None:
                raise _LedgerError(
                    f"{where}: its sources list {sid!r}, which call item {earlier} lists"
                )
        return sources

    def _entry(self, item: dict[str, Any], where: str) -> tuple[dict[str, object], list["_Grow"]]:
        ref: str = item["ref"]
        if ref in self._written:
            raise _LedgerError(f"{where}: a second entry for {ref!r}; an entry is written once")
        if ref == _STATE:
            key, node = _STATE, _STATE_NODE
        else:
            try:
                key = declared_ref(ref)
            except ValueError as exc:
                raise _LedgerError(f"{where}: {exc}") from exc
            provider = self._providers.get(key)
            if provider is None:
                raise _LedgerError(f"{where}: {key!r} is not a key of the settings' providers")
            node = ("provider", provider)
        if node == self._node:
            if ref <= self._node_last:
                raise _LedgerError(
                    f"{where}: {ref!r} is not after {self._node_last!r} in code-point order"
                )
            ended = None
        elif node in self._done_nodes:
            raise _LedgerError(f"{where}: {ref!r} is not contiguous with its node's entries")
        else:
            ended = self._node  # the node this entry ends, if any
        tree = self._classes.get(key)
        if tree is None:
            raise _LedgerError(f"{where}: {key!r} has no class tree in the settings' classes")
        if _tree_malformed(tree):
            raise _LedgerError(f"{where}: the class tree of {key!r} is malformed")
        if "value" in item:
            misfit = _misfit(item["value"], tree, "the value")
            if misfit is not None:
                raise _LedgerError(f"{where}: {misfit}")

        # Neither is held yet: the checks above refuse a second entry and a done node.
        grows: list[_Grow] = [(lambda: self._written.add(ref), lambda: self._written.discard(ref))]
        if ended is not None:
            done = self._done_nodes
            grows.append((lambda: done.add(ended), lambda: done.discard(ended)))
        return {"_node": node, "_node_last": ref}, grows

    def write_unhashed(self, member: str, value: object, index: int | None = None) -> None:
        """Write ``value`` to ``unhashed``'s ``member``, under the log ``index`` for ``calls``,
        ``exceptions`` and ``reads``; ``turn`` is merged, so ``start`` and then ``end`` are
        written apart. The value is copied with each surrogate escaped, no observer is called,
        and a known member of the wrong shape is refused."""
        if self.closed:
            raise _LedgerError("the ledger is closed: nothing is written after close")
        member = string_data(member)
        # A member is a holder, nesting at most as deep as it stands in the document; a
        # keyed write stands one deeper. A call's usage is a value its producer size-tests.
        limit = _MEMBER_DEPTH - 1 if member in _KEYED else _MEMBER_DEPTH
        copy = _escaped_copy(value, limit)
        err = domain_error(copy, limit)
        if err is not None:
            raise _LedgerError(f"the unhashed {member!r} write is outside the value domain: {err}")
        if member in _KEYED:
            if type(index) is not int:
                raise _LedgerError(f"unhashed {member!r} is written under a log index")
            if not -MAX_INTEGER <= index <= MAX_INTEGER:  # str() of it may be refused
                raise _LedgerError(
                    f"unhashed {member!r} is written under an index of {index.bit_length()} bits, "
                    "which names no item"
                )
            key = str(index)  # a negative index is no log index: the key check refuses it
            existing = cast("dict[str, Any]", self._unhashed.get(member, {}))
            if key in existing:
                raise _LedgerError(f"unhashed {member!r}[{key!r}] is written once")
            new = {**existing, key: copy}
        elif index is not None:
            raise _LedgerError(f"unhashed {member!r} is not keyed by a log index")
        elif member == "turn":
            _OBJECT(copy, "unhashed 'turn'")
            turn = cast("dict[str, Any]", self._unhashed.get("turn", {}))
            again = [name for name in cast("dict[str, Any]", copy) if name in turn]
            if again:
                raise _LedgerError(f"unhashed 'turn'[{again[0]!r}] is written once")
            new = {**turn, **copy}
        else:
            if member in ("engine", "codec") and member in self._unhashed:
                raise _LedgerError(f"unhashed {member!r} is written once")
            new = copy
        self._set_unhashed(member, new)

    def _set_unhashed(self, member: str, value: Any) -> None:
        # The value is in the domain: a write checks its copy, whose keyed and turn members the
        # shapes keep shallow, and the read checks each member before any append.
        _check_unhashed(member, value, self._kinds, self._questions, self._format)
        self._unhashed = {**self._unhashed, member: value}

    def prefix_hash(self, n: int) -> str:
        """The record hash over the log's first ``n`` items."""
        return _prefix_hash(self._bytes, n)

    def close(self) -> "Record":
        """Close the ledger and return its ``Record``; closing ends no round. A later close
        returns the same record."""
        if self._record is not None:
            return self._record
        if not self._items:
            raise _LedgerError("the ledger closes from its design item on: it holds no item")
        self.drifted_copies = tuple(
            i
            for i, (copy, data) in enumerate(zip(self._items, self._bytes, strict=True))
            if not _encodes_to(copy, data)
        )
        unhashed = encode(self._unhashed, _UNHASHED_DEPTH)
        self._record = Record(_BUILT_BY_LEDGER, tuple(self._bytes), unhashed)
        return self._record


def _encodes_to(value: object, data: bytes) -> bool:
    try:
        return encode(value, _ITEM_DEPTH) == data
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------------
# The closed record

_BUILT_BY_LEDGER = object()


class Record:
    """A closed record: its log's canonical bytes and its ``unhashed`` member. Built only by
    the ledger; ``read_record`` reads a stored one back. Every attribute is derived from the
    bytes at each read and stored nowhere."""

    __slots__ = ("_items", "_unhashed")

    def __init__(self, token: object, items: tuple[bytes, ...], unhashed: bytes) -> None:
        if token is not _BUILT_BY_LEDGER:
            raise TypeError("a Record is built only by the ledger; read one with read_record")
        self._items = items
        self._unhashed = unhashed

    @property
    def document(self) -> bytes:
        """The canonical bytes of the record document, ``{"log": [...], "unhashed": {...}}``."""
        return b'{"log":[' + b",".join(self._items) + b'],"unhashed":' + self._unhashed + b"}"

    @property
    def record_hash(self) -> str:
        """The record hash over the whole log."""
        return record_hash(self._items)

    @property
    def design_hash(self) -> str:
        """The ``design`` item's ``hash``."""
        return cast(str, decode(self._items[0])["hash"])

    @property
    def cause(self) -> dict[str, Any] | None:
        """The derived cause, ``{"code", "refs"}``, or ``None``."""
        log = self.log
        if log[-1]["item"] == "cause":
            return cast("dict[str, Any]", log[-1]["cause"])
        plans = [item for item in log if item["item"] == "plan"]
        state = any(item["item"] == "entry" and item["ref"] == _STATE for item in log)
        if plans and not state:
            return cast("dict[str, Any] | None", plans[0]["cause"])
        return None

    @property
    def log(self) -> list[dict[str, Any]]:
        """The log's items as plain objects, decoded afresh at each read."""
        return [cast("dict[str, Any]", decode(data)) for data in self._items]

    @property
    def unhashed(self) -> dict[str, Any]:
        """``unhashed`` as a plain object, decoded afresh at each read."""
        return cast("dict[str, Any]", decode(self._unhashed))

    def __repr__(self) -> str:
        return f"Record(record_hash={self.record_hash!r}, items={len(self._items)})"


def record_prefix_hash(record: Record, n: int) -> str:
    """The record hash over a closed record's first ``n`` items; no public name."""
    return _prefix_hash(record._items, n)  # pyright: ignore[reportPrivateUsage]


def _prefix_hash(items: list[bytes] | tuple[bytes, ...], n: int) -> str:
    if type(n) is not int or not 0 <= n <= len(items):
        raise _LedgerError(f"a prefix of {described(n)} items is not one of a log of {len(items)}")
    return record_hash(items[:n])


# ---------------------------------------------------------------------------------------------
# The read

_RECORD_MEMBERS = ("log", "unhashed")


def _named_one_of(key: object, names: tuple[str, ...]) -> bool:
    """Whether a key is a string whose data is one of ``names``, read by its type alone."""
    return issubclass(type(key), str) and string_data(cast(str, key)) in names


def read_record(document: object) -> Record:
    """Read a stored record document, bytes or parsed, into a ``Record``, or raise
    ``InvalidRecord`` naming the first failure."""
    if issubclass(type(document), bytes):
        # Its data through bytes' own slice: no __bytes__ or other override of a subclass runs.
        data = bytes.__getitem__(cast(bytes, document), slice(None))
        try:
            parsed: object = decode(data)
        except ValueError as exc:
            raise InvalidRecord(f"the record document does not parse: {exc}") from exc
    else:
        # Before any other walk: a container at two places would expand at each.
        if shares_a_container(document):
            raise InvalidRecord("the record document holds a list or object at two places")
        if nesting_depth(document, MAX_DOCUMENT_DEPTH) > MAX_DOCUMENT_DEPTH:
            raise InvalidRecord(f"the record document nests more than {MAX_DOCUMENT_DEPTH} deep")
        parsed = document
    try:
        return _read(parsed)
    except _LedgerError as exc:
        raise InvalidRecord(f"the record is invalid: {exc}") from exc


def _read(doc: object) -> Record:
    if type(doc) is not dict:
        raise _LedgerError("the record document is not one JSON object")
    # Each member is found by its key's type and string data, so no code of a key runs; two
    # keys of one string data would make the member found depend on their order.
    names: set[str] = set()
    for key in cast("dict[object, Any]", doc):
        if issubclass(type(key), str):
            data = string_data(cast(str, key))
            if data in names:
                raise _LedgerError(f"two object keys have the string data {clipped(data)}")
            names.add(data)
    log = member_named(cast(object, doc), "log")
    first = cast("list[Any]", log)[0] if (type(log) is list or type(log) is tuple) and log else None
    item = member_named(first, "item")
    if not (issubclass(type(item), str) and string_data(cast(str, item)) == "design"):
        raise _LedgerError("the log has no design item first")
    fmt = member_named(first, "format")
    fmt = None if fmt is MISSING else fmt
    if type(fmt) is not int or fmt not in FORMATS:
        known = ", ".join(str(f) for f in sorted(FORMATS))
        raise _LedgerError(
            f"the record's format {described(fmt)} is not one this reader knows: {known}"
        )
    others = [k for k in cast("dict[object, Any]", doc) if not _named_one_of(k, _RECORD_MEMBERS)]
    if others:
        view = (
            ", as a redacted view's is"
            if any(_named_one_of(k, ("redacted",)) for k in others)
            else ""
        )
        shown = "[" + ", ".join(described(k) for k in others) + "]"
        raise _LedgerError(f"the document has members other than log and unhashed{view}: {shown}")
    unhashed = member_named(cast(object, doc), "unhashed")
    if unhashed is MISSING:
        raise _LedgerError("the document has no unhashed member")
    # The domain is checked per item and on unhashed, each value it holds size-tested where
    # it stands before its holder is walked, never on the log as one value.
    for i, item in enumerate(cast("list[Any]", log)):
        # The item as a holder first: it shares no container, so the walk is linear in it.
        err = domain_error(item, _ITEM_DEPTH)
        if err is not None:
            raise _LedgerError(f"log item {i} is outside the value domain: {err}")
        _sized(item, i)
    if type(unhashed) is not dict:
        raise _LedgerError("unhashed is not an object")
    err = domain_error(cast(object, unhashed), _UNHASHED_DEPTH)
    if err is not None:
        raise _LedgerError(f"unhashed is outside the value domain: {err}")
    for i, entry in enumerate(
        members_of(member_named(cast(object, unhashed), "calls"))
    ):  # each usage a value
        usage = member_named(entry, "usage")
        if usage is not MISSING:
            _sized_value(usage, f"unhashed 'calls' entry {i}'s 'usage'")
    ledger = Ledger()
    for item in cast("list[Any]", log):
        ledger.append(item)
    for name, value in cast("dict[str, Any]", unhashed).items():
        ledger._set_unhashed(string_data(name), _copy(value))  # pyright: ignore[reportPrivateUsage]
    return ledger.close()
