"""The record views: pure functions from a closed record to a value, for the caller, the
escalation target or a verifier.

``escalation(record)`` returns the escalation's four members, or ``None`` for a record that
holds none, reading only the ``plan`` and ``verdict`` items and the record hash. ``redact(record,
root_key)`` returns the redacted view's document as canonical bytes: the record's ``log`` and
``unhashed`` with a keyed digest where each ``content`` or ``identifier`` value was, and
``redacted``, the record hash its key came from. Which value takes which class is set by the default
classes.

A v0.0.1 record holds no grounded ref and no origin carrying ``set``, since the ledger refuses
both at the write and the read: every ref passes unchanged and every literal takes
its JSON form's default. Each view raises ``ValueError`` before reading anything when
``record`` is not exactly a ``Record``. The continuation record is deferred, so it has no function
here.
"""

from typing import Any, cast

from jurat._canonical import encode, keyed_digest, turn_key
from jurat._design import Requirement
from jurat._guard import GuardFailure, Verdict
from jurat._ledger import Record
from jurat._markers import PlanKind, VerdictKind
from jurat._refs import declared_ref
from jurat._typed import is_a
from jurat._values import MAX_DOCUMENT_DEPTH, type_name

_OPEN = "open"


def _closed(record: object) -> Record:
    """The record, refused by its exact type before anything of it is read."""
    if type(record) is not Record:
        raise ValueError(f"a record view takes a Record, not {type_name(record)}")
    return record


# ---------------------------------------------------------------------------------------------
# The escalation


def escalation(record: Record) -> dict[str, Any] | None:
    """The escalation of a closed record: ``cause``, ``failures``, ``reason`` and
    ``record_hash``; ``None`` when the log has no ``verdict`` item, or its plan is not
    ``escalate`` and its verdict passed."""
    log = _closed(record).log
    plans = [item for item in log if item["item"] == "plan"]
    verdicts = [item for item in log if item["item"] == "verdict"]
    if not verdicts:
        return None
    plan, verdict = plans[0], verdicts[0]
    escalated = plan["kind"] == PlanKind.escalate
    if not escalated and verdict["kind"] == VerdictKind.passed:
        return None
    cause: dict[str, Any] | None = None
    if verdict["kind"] == VerdictKind.refused:
        derived = _verdict(verdict).cause
        assert derived is not None  # a refused verdict's cause is never None
        cause = {"code": derived.code.value, "refs": list(derived.refs)}
    elif plan["cause"] is not None:  # the engine's `required` on a plan compile built
        cause = cast("dict[str, Any]", plan["cause"])
    return {
        "cause": cause,
        "failures": verdict["failures"],
        "reason": plan["reason"] if escalated else None,
        "record_hash": record.record_hash,
    }


def _verdict(item: dict[str, Any]) -> Verdict:
    """The guard's ``Verdict`` built from the recorded failures."""
    failures: list[GuardFailure] = []
    for f in cast("list[dict[str, Any]]", item["failures"]):
        recorded = cast("dict[str, Any] | None", f["requirement"])
        requirement = None if recorded is None else Requirement(recorded["ref"], recorded["value"])
        failures.append(GuardFailure(f["step"], requirement, f["ref"]))
    return Verdict(item["kind"], failures)


# ---------------------------------------------------------------------------------------------
# The redacted view


def redact(record: Record, root_key: bytes) -> bytes:
    """The redacted view of a closed record, as canonical bytes, its digests under the key
    derived from the record's own hash. Raises ``ValueError`` before reading anything when
    ``record`` is not exactly a ``Record``, then before reading the record when ``root_key`` is
    not an instance of ``bytes`` of length 32."""
    closed = _closed(record)
    key = _root_key(root_key)
    record_hash = closed.record_hash
    walk = _Walk(turn_key(key, record_hash))
    log = closed.log
    classes: dict[str, Any] = {}
    for item in log:
        if item["item"] == "settings":
            classes = cast("dict[str, Any]", item["classes"])
    view = {
        "log": [walk.item(item, classes) for item in log],
        "unhashed": walk.unhashed(closed.unhashed),
        "redacted": record_hash,
    }
    return encode(view, MAX_DOCUMENT_DEPTH)


def _root_key(root_key: object) -> bytes:
    """The root key's bytes, read by its type and ``bytes``' own slicing, so a subclass's code
    never runs; ``ValueError`` unless it is an instance of ``bytes`` of length 32."""
    key = bytes.__getitem__(cast(bytes, root_key), slice(None)) if is_a(root_key, bytes) else None
    if key is None or len(key) != 32:
        raise ValueError("the root key is not an instance of bytes of length 32")
    return key


class _Walk:
    """The digests of one record, all under its per-turn key."""

    def __init__(self, key: bytes) -> None:
        self.key = key

    def digest(self, value: Any) -> str:
        """A keyed digest of the canonical bytes of a value or of text ``unhashed`` holds,
        encoded as a holder within the document's depth bound: text is held to no size
        bound."""
        return keyed_digest(self.key, encode(value, MAX_DOCUMENT_DEPTH))

    def text(self, value: Any) -> Any:
        """Text the engine records with no type: ``content``, a null passing."""
        return None if value is None else self.digest(value)

    # -- the log -----------------------------------------------------------------------------

    def item(self, item: dict[str, Any], classes: dict[str, Any]) -> dict[str, Any]:
        kind = item["item"]
        out = dict(item)
        if kind == "entry" and "value" in item:
            out["value"] = self.value(item["value"], classes[declared_ref(item["ref"])])
        elif kind == "call":
            out["prompt"] = self.text(item["prompt"])
        elif kind == "plan":
            out["reason"] = self.text(item["reason"])
            out["reply"] = self.text(item["reply"])
            out["steps"] = [self.step(step) for step in item["steps"]]
        # Every other member is open: markers, causes, derivation, rows, slots, the verdict,
        # the run settings and the call item's other members.
        return out

    def step(self, step: dict[str, Any]) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        for name, origin in cast("dict[str, Any]", step["arguments"]).items():
            if type(origin) is dict and "literal" in origin:  # its JSON form's default
                arguments[name] = {**origin, "literal": self.literal(origin["literal"])}
            else:  # a board ref or a binding: open
                arguments[name] = origin
        return {**step, "arguments": arguments}

    def literal(self, value: Any) -> Any:
        """A literal argument's default by its JSON form: a string ``content``, any other
        scalar ``open``, a list or object member by member, an object's keys ``content``."""
        if type(value) is str:
            return self.digest(value)
        if type(value) is list:
            return [self.literal(v) for v in cast("list[Any]", value)]
        if type(value) is dict:
            return {
                self.digest(k): self.literal(v) for k, v in cast("dict[str, Any]", value).items()
            }
        return value

    def value(self, value: Any, tree: Any) -> Any:
        """A value redacted beside its full class tree, read from the tree: a record opened
        field by field; a list or mapping whose tree holds no ``open`` and no record one
        digest, any other opened, a mapping's keys digested as ``content``; a null passing
        unless it lies within a value digested whole."""
        if value is None:
            return None
        if type(value) is list:  # a list layer has no node: the tree describes each member
            if not _keeps_open(tree):
                return self.digest(value)
            return [self.value(v, tree) for v in cast("list[Any]", value)]
        if type(tree) is str:
            return value if tree == _OPEN else self.digest(value)
        members = cast("dict[str, Any]", value)  # the ledger fits each value to its tree
        if type(tree) is dict:  # a record
            nodes = cast("dict[str, Any]", tree)
            return {name: self.value(v, nodes[name]) for name, v in members.items()}
        if not _keeps_open(tree):  # a mapping layer, a one-member list
            return self.digest(value)
        beneath = cast("list[Any]", tree)[0]
        return {self.digest(k): self.value(v, beneath) for k, v in members.items()}

    # -- unhashed ----------------------------------------------------------------------------

    def unhashed(self, unhashed: dict[str, Any]) -> dict[str, Any]:
        """``unhashed`` with its text digested: exception text, ``unexpected`` whole,
        ``invalid``'s ``repr`` text, the ``reads`` values and an unknown member whole; member
        names, log-index keys, ``usage``, ``sums``, times, the engine and codec names and an
        exception's ``type`` are open."""
        out: dict[str, Any] = {}
        for member, value in unhashed.items():
            if member in ("turn", "engine", "codec"):
                out[member] = value
            elif member == "calls":
                out[member] = {k: self.call(v) for k, v in value.items()}
            elif member == "exceptions":
                out[member] = {
                    k: {
                        **v,
                        "message": self.digest(v["message"]),
                        "traceback": self.digest(v["traceback"]),
                    }
                    for k, v in value.items()
                }
            elif member == "reads":
                out[member] = {k: self.digest(v) for k, v in value.items()}
            else:
                out[member] = self.digest(value)
        return out

    def call(self, entry: dict[str, Any]) -> dict[str, Any]:
        # ``sums`` passes open: its keys are declared refs, as the record refuses a grounded
        # one. Once grounded declarations land, a grounded key's candidate needs its digest.
        out = dict(entry)
        if "unexpected" in entry:
            out["unexpected"] = self.digest(entry["unexpected"])
        if "invalid" in entry:
            invalid = cast("dict[str, Any]", entry["invalid"])
            out["invalid"] = {name: self.digest(text) for name, text in invalid.items()}
        return out


def _keeps_open(tree: Any) -> bool:
    """Whether a tree holds an ``open`` name or a record object, so its value is opened."""
    if type(tree) is str:
        return tree == _OPEN
    if type(tree) is dict:
        return True
    return any(_keeps_open(t) for t in cast("list[Any]", tree))
