# pyright: reportPrivateUsage=false
"""Load: the load checks, the design document and its hash, the graph and the
ref-to-provider map, the class trees and the answer grid.

Load gathers the providers, a refused form's included, then runs
checks 1 to 4, 7, 8, 10 and 12. Once 10 and 12 pass it builds the document and hashes it.
Once 1 to 4, 10 and 12 pass it runs check 8's leaf-type test, check 9, check 10's test of
each declared class tree, and builds the graph for check 5. It raises ``InvalidDesign`` with
every failure, or returns the loaded design.

A document passes check 11's gate first, a failure there the only one; then load builds a
``Design`` from it, resolving each function component, and runs the checks on it. Once every
component resolved and 1 to 4, 10 and 12 pass, it compares the document it computes with the
one given.
"""

import importlib
import inspect
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Any, cast

from jurat._answer import AnswerKind
from jurat._canonical import _DecodeError, _DuplicateKey, decode, document_depth, encode, sha256_ref
from jurat._design import (
    _BUILT_BY_LOAD,
    Design,
    DesignFailure,
    Group,
    InvalidDesign,
    LoadedDesign,
    Parameter,
    PlanDeclaration,
    Policy,
    PolicyTable,
    Question,
    Requirement,
    Rule,
    Source,
    SourceRequirement,
    Table,
    Text,
    When,
    document_value,
    live_value,
    new_failure,
)
from jurat._form import check_form
from jurat._refs import (
    _ParsedRef,
    code_point_sorted,
    is_declared_name,
    is_segment,
    parse_ref,
)
from jurat._types import (
    AnswerType,
    DeclaredMemo,
    FailureKind,
    Interval,
    Label,
    ListOf,
    MappingOf,
    Named,
    Nullable,
    RecordOf,
    Scalar,
    Type,
    TypeFailure,
    check_declared,
    check_definitions,
    conforms,
    hold_failure,
    named,
    parse_type,
    read_type,
    resolve,
    value_conforms,
)
from jurat._values import (
    MAX_DOCUMENT_DEPTH,
    MISSING,
    domain_error,
    in_domain,
    member_named,
    members_of,
    nesting_depth,
    power,
    quote,
    shares_a_container,
    string_data,
)

_STATE = "t:state"
_ROLES = frozenset({"policy", "codec"})
_WORDS = frozenset(
    {
        "number",
        "integer",
        "string",
        "boolean",
        "bytes",
        "label",
        "list",
        "mapping",
        "record",
        "answer",
        "or",
        "null",
    }
)
_GATING = frozenset({1, 2, 3, 4, 10, 12})
_NUMBER = Scalar("number")
_NUMBERS = MappingOf(_NUMBER)
_UNIT = Interval("number", 0, 1)
_UNITS = MappingOf(_UNIT)
_TYPE_SIZE = 2**20
"""The most code points a type spells with its definitions resolved, and the types of the
refs the class trees cover, summed."""
_ANSWER_SIZE = len("answer")


def load(
    design: Design | bytes | Mapping[str, Any],
    *,
    functions: Mapping[str, Callable[..., Any]] | None = None,
) -> LoadedDesign:
    """Check a design once and return the loaded design, or raise ``InvalidDesign`` carrying
    every failure found."""
    if isinstance(design, Design):
        if functions is not None:
            raise TypeError("functions is given only with a document; a Design holds its callables")
        return _Loader(check_form(design)).run()
    if isinstance(design, bytes | Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        if functions is not None and not isinstance(functions, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("functions is a mapping from a component's key to its callable")
        return _load_document(design, {} if functions is None else functions)
    raise TypeError(f"load takes a Design or a design document, not {type(design).__name__}")


def _load_document(given: bytes | Mapping[str, Any], functions: Mapping[str, Any]) -> LoadedDesign:
    """Check 11's gate, then the design built from the document under every other check, and
    the comparison."""
    parsed, refusal = _gate(given)
    if refusal is not None:
        names, message = refusal
        raise InvalidDesign([new_failure(11, names, message)], _BUILT_BY_LOAD)
    assert parsed is not None
    resolved, unresolved, failures = _resolve(parsed, functions)
    loader = _Loader(
        check_form(_build(parsed, resolved), unresolved),
        parsed=parsed,
        given=given if isinstance(given, bytes) else None,
    )
    loader.failures += failures
    return loader.run()


_q = quote


@dataclass(frozen=True, slots=True)
class _Use:
    """A ref a declaration reads: the declaration, its role and the parsed ref."""

    owner: str  # the declared name, or "policy" for the policy and "" for the plan
    where: str
    role: str  # need, slot, read, policy, requirement, source-requirement, group
    ref: str
    parsed: _ParsedRef

    @property
    def base(self) -> str:
        return f"{self.parsed.level}:{self.parsed.name}"


class _Loader:
    def __init__(
        self, checked: Any, *, parsed: dict[str, Any] | None = None, given: bytes | None = None
    ) -> None:
        # From a document: the parsed document and, given as bytes, the bytes.
        self.parsed = parsed
        self.given = given
        self.d: Design = checked.design
        self.failures: list[DesignFailure] = list(checked.failures)
        # What a member check 10's form test refused still provides or declares.
        self.form_providers: tuple[tuple[str, str], ...] = checked.providers
        self.leaves: frozenset[str] = frozenset(self.d.parameters) | frozenset(checked.leaves)
        self.refused_sources: set[str] = set()
        self.refused_rules: set[str] = set()
        self.grounded: set[str] = set()
        self.providers: dict[str, list[str | None]] = {}
        self.uses: list[_Use] = []
        self.candidates: list[tuple[str, str, str]] = []  # each candidate slot's source
        self.defs: dict[str, Type] = {}
        self.tainted: set[str] = set()
        # Each definition's size, and each provided ref's and the state's, once it passes.
        self.def_sizes: dict[str, int] = {}
        # What check 10's type checks learnt of the definitions, kept across declarations.
        self.declared_memo = DeclaredMemo()
        self.ref_sizes: dict[str, int] = {}
        self.types: dict[str, Type] = {}
        self.leaf_types: dict[str, Type] = {}
        self.state_type: Type | None = None
        self.class_keys: list[tuple[str, _ParsedRef]] = []  # each classes key check 7 passed
        # Each leaf check 8 refuses under a boundaries field, left out of check 9's boundaries
        # clauses.
        self.refused_leaves: set[str] = set()
        # Each leaf's value as Python holds it, bytes as ``bytes``; a document writes a bytes
        # value as base64url, read back here once its type passes check 10.
        self.live: dict[str, object] = {}

    def fail(self, check: int, names: tuple[str, ...], message: str) -> None:
        self.failures.append(new_failure(check, code_point_sorted(names), message))

    def failed(self, checks: frozenset[int]) -> bool:
        return any(f.check in checks for f in self.failures)

    # --- the declarations that are not refused ------------------------------------------

    def sources(self) -> Iterator[tuple[str, Source]]:
        for sid, source in self.d.sources.items():
            if sid not in self.refused_sources:
                yield sid, source

    def rules(self) -> Iterator[tuple[str, Rule]]:
        for rid, rule in self.d.rules.items():
            if rid not in self.refused_rules:
                yield rid, cast(Rule, rule)

    def policy(self) -> Policy | None:
        return self.d.policy if isinstance(self.d.policy, Policy) else None

    # --- the run ---------------------------------------------------------------------------

    def run(self) -> LoadedDesign:
        self.check_12()
        self.gather()
        self.check_2()
        self.check_3()
        self.check_4()
        self.check_1()
        self.check_10()
        self.check_8()
        document: bytes | None = None
        if not self.failed(frozenset({10, 12})):
            # Check 10 holds every value in the domain, so the encoding cannot refuse.
            document = encode(self.document(), MAX_DOCUMENT_DEPTH)
        graph: Mapping[str, frozenset[str]] = MappingProxyType({})
        if not self.failed(_GATING):
            self.check_8_types()
            self.check_9()
            self.check_10_trees()
            graph = self.graph()
            self.check_5(graph)
        if self.parsed is not None and not self.failed(_GATING | {11}):
            assert document is not None
            self.compare(document)
        if self.failures:
            raise InvalidDesign(self.failures, _BUILT_BY_LOAD)
        assert document is not None
        return self.loaded(document, graph)

    # --- check 12 --------------------------------------------------------------------------

    def check_12(self) -> None:
        def refuse(into: set[str], names: tuple[str, ...], what: str, form: str) -> None:
            into.add(names[0])
            self.fail(12, names, f"{what} is a {form}, which this engine does not implement")

        sources, rules = self.refused_sources, self.refused_rules
        for sid, source in self.d.sources.items():
            what = f"source {_q(sid)}"
            if source.grounded is not None:
                refuse(sources, (sid,), what, "grounded declaration")
                self.grounded.add(sid)
            for ref, provided in source.provides.items():
                if isinstance(provided, Question):
                    if isinstance(provided.labels, Mapping):
                        form = "candidate-valued label set"
                        refuse(sources, (sid, ref), f"{what}'s {_q(ref)}", form)
                elif cast("Mapping[str, Any]", provided)["candidates"]:
                    refuse(sources, (sid, ref), f"{what}'s {_q(ref)}", "candidate set")
        for rid, rule in self.d.rules.items():
            what = f"rule {_q(rid)}"
            if isinstance(rule, Table):
                refuse(rules, (rid,), what, "decision table")
            if rule.candidates:
                refuse(rules, (rid,), what, "candidate set")
            if rule.grounded is not None:
                refuse(rules, (rid,), what, "grounded declaration")
                self.grounded.add(rid)
        if isinstance(self.d.policy, PolicyTable):
            refuse(set(), ("policy",), "the policy", "policy table")
        for name in self.d.plans:
            refuse(set(), (name,), f"plan declaration {_q(name)}", "plan declaration")

    # --- gathering, and check 7 ------------------------------------------------------------

    def ref(self, ref: str, where: str, owner: str) -> _ParsedRef | None:
        """The parsed ref, or ``None`` after a check-7 failure; a refused ref is left out of
        every other check."""
        try:
            return parse_ref(ref)
        except ValueError:  # the ref grammar's refusal
            names = (owner, ref) if owner else (ref,)
            self.fail(7, names, f"{where} names {_q(ref)}, which is not a ref")
            return None

    def provide(self, ref: str, provider: str, where: str, *, checked: bool) -> None:
        if checked:
            parsed = self.ref(ref, where, provider)
            if parsed is None:
                return
            if parsed.candidate is not None or parsed.fields:
                self.fail(
                    7, (provider, ref), f"{where} provides {_q(ref)}, which is not a declared ref"
                )
                return
        else:  # a refused form: gathered as a provider only (check 12)
            try:
                parsed = parse_ref(ref)
            except ValueError:  # the ref grammar's refusal
                return
            if parsed.candidate is not None or parsed.fields:
                return
        self.providers.setdefault(ref, []).append(provider)

    def use(self, owner: str, where: str, role: str, ref: str) -> None:
        parsed = self.ref(ref, where, owner)
        if parsed is not None:
            self.uses.append(_Use(owner, where, role, ref, parsed))

    def gather(self) -> None:
        self.providers[_STATE] = [None]
        for sid, source in self.d.sources.items():
            checked = sid not in self.refused_sources
            for ref in source.provides:
                self.provide(ref, sid, f"source {_q(sid)}", checked=checked)
        for rid, rule in self.d.rules.items():
            checked = rid not in self.refused_rules
            self.provide(rule.provides, rid, f"rule {_q(rid)}", checked=checked)
        for ref, provider in self.form_providers:
            self.provide(ref, provider, "", checked=False)
        for sid, source in self.sources():
            where = f"source {_q(sid)}"
            for ref in source.needs:
                self.use(sid, where, "need", ref)
            for ref, slot in _slots(source):
                tag, value = next(iter(slot["origin"].items()))
                if tag == "ref":
                    self.use(sid, f"{where}'s question {_q(ref)}", "slot", value)
                elif tag == "candidate":
                    self.candidates.append((sid, f"{where}'s question {_q(ref)}", value))
                    self.use(sid, f"{where}'s question {_q(ref)}", "candidate", value)
        for rid, rule in self.rules():
            for ref in rule.reads:
                self.use(rid, f"rule {_q(rid)}", "read", ref)
        policy = self.policy()
        if policy is not None:
            for ref in policy.reads:
                self.use("policy", "the policy", "policy", ref)
        for owner, where, boundaries, _ in self.boundary_fields():
            for key in boundaries:
                # Check 3 finds each key's provider and check 7 its form; check 4 gives a
                # key no level test, since check 9 holds it to its component's reads.
                self.use(owner, f"{where}'s boundaries", "boundary", key)
        for requirement in self.d.requirements:
            # A requirement names a whole estimate: a field path fails check 7
            # and, refused, the ref is left out of every other check.
            parsed = self.ref(requirement.ref, "a plan-level requirement", "")
            if parsed is not None and parsed.fields:
                self.fail(
                    7,
                    (requirement.ref,),
                    f"a plan-level requirement names {_q(requirement.ref)}, which has a field path",
                )
            elif parsed is not None:
                self.uses.append(
                    _Use("", "a plan-level requirement", "requirement", requirement.ref, parsed)
                )
        for gname, group in self.d.groups.items():
            for ref in group.refs:
                self.use(gname, f"group {_q(gname)}", "group", ref)
        for sid, source in self.sources():
            if source.requirement is None:
                continue
            where = f"source {_q(sid)}'s requirement"
            ref = source.requirement.ref
            # A requirement names a whole estimate, on a source as on the plan.
            parsed = self.ref(ref, where, sid)
            if parsed is not None and parsed.fields:
                self.fail(7, (sid, ref), f"{where} names {_q(ref)}, which has a field path")
            elif parsed is not None:
                self.uses.append(_Use(sid, where, "source-requirement", ref, parsed))
        for key in self.d.classes:
            parsed = self.ref(key, "the classes block", "")
            if parsed is not None:
                self.class_keys.append((key, parsed))

    # --- checks 2, 3 and 4 -----------------------------------------------------------------

    def check_2(self) -> None:
        for ref, providers in self.providers.items():
            if len(providers) > 1:
                ids = [p for p in providers if p is not None]
                by = ", ".join(_q(p) for p in code_point_sorted(set(ids)))
                if None in providers:
                    by = "the state declaration, " + by
                self.fail(2, (ref, *set(ids)), f"{_q(ref)} has more than one provider: {by}")

    def check_3(self) -> None:
        for u in self.uses:
            if u.role == "candidate":
                self.fail(
                    3,
                    (u.owner, u.ref),
                    f"{u.where} has a candidate slot, but its source is not grounded",
                )
                continue
            if u.parsed.candidate is not None:
                found = any(p in self.grounded for p in self.providers.get(u.base, []))
            else:
                found = u.base in self.providers
            if not found:
                names = (u.owner, u.ref) if u.owner else (u.ref,)
                self.fail(3, names, f"{u.where} names {_q(u.ref)}, which has no provider")
        for gname, group in self.d.groups.items():
            refs = [r for r in group.refs if _parsed(r) is not None]
            counts = Counter(refs)
            for ref in code_point_sorted({r for r, n in counts.items() if n > 1}):
                self.fail(3, (gname, ref), f"group {_q(gname)} names {_q(ref)} more than once")
        for key, parsed in self.class_keys:
            where = f"the classes block's key {_q(key)}"
            if parsed.candidate is not None:
                self.fail(3, (key,), f"{where} has a grounded bracket")
            elif parsed.fields:
                self.fail(3, (key,), f"{where} has a field path")
            elif parsed.level == "q":
                self.fail(3, (key,), f"{where} is not t:state, a source's t: ref or a d: ref")
            elif key not in self.providers:
                self.fail(3, (key,), f"{where} has no provider")

    def check_4(self) -> None:
        for sid, source in self.sources():
            provided: set[str] = set()
            for ref in source.provides:
                parsed = _declared(ref)
                if parsed is None:
                    continue
                provided.add(parsed.level)
                if parsed.level not in ("q", "t"):
                    self.fail(
                        4,
                        (sid, ref),
                        f"source {_q(sid)} provides {_q(ref)}; a source provides q: or t: refs",
                    )
            if {"q", "t"} <= provided:
                self.fail(
                    4,
                    (sid,),
                    f"source {_q(sid)} provides q: and t: refs; "
                    "a source provides refs of one level",
                )
        for rid, rule in self.rules():
            parsed = _declared(rule.provides)
            if parsed is not None and rule.provides != f"d:{rid}":
                self.fail(
                    4,
                    (rid, rule.provides),
                    f"rule {_q(rid)} provides {_q(rule.provides)}; a rule provides its own d: ref",
                )
        allowed = {
            "slot": ("d", "t"),
            "policy": ("d", "t"),
            "requirement": ("d",),
            "source-requirement": ("d",),
        }
        # The clause is on the slot list, so it stands whether or not the slot's ref passes
        # check 7. No source here is grounded: check 12 refuses grounded ones.
        for sid, where, _ in self.candidates:
            self.fail(
                4,
                (sid,),
                f"{where} has a candidate slot; a source's slots hold candidate only when it is "
                "grounded",
            )
        for u in self.uses:
            if u.role == "candidate":
                continue
            levels = allowed.get(u.role)
            if levels is not None and u.parsed.level not in levels:
                names = (u.owner, u.ref) if u.owner else (u.ref,)
                shown = " or ".join(f"{level}:" for level in levels)
                self.fail(4, names, f"{u.where} reads {_q(u.ref)}, which is not at level {shown}")

    # --- check 1 ---------------------------------------------------------------------------

    def check_1(self) -> None:
        d = self.d
        kinds: dict[str, list[str]] = {}
        blocks: list[tuple[str, list[str]]] = [
            ("source", [s for s, _ in self.sources()]),
            ("rule", [r for r, _ in self.rules()]),
            ("parameter", list(self.leaves)),
            ("tool", list(d.tools)),
            ("type definition", list(d.types)),
            ("group", list(d.groups)),
        ]
        for kind, names in blocks:
            for name in names:
                shown = f"{kind} {_q(name)}"
                if not is_declared_name(name):
                    self.fail(1, (name,), f"{shown} is not in the segment grammar")
                if name in _ROLES:
                    self.fail(1, (name,), f"{shown} is a role name")
                if kind == "type definition" and name in _WORDS:
                    self.fail(1, (name,), f"{shown} is a word of the type notation")
                if kind not in ("type definition", "group"):
                    kinds.setdefault(name, []).append(kind)
        for name, used in kinds.items():
            if len(used) > 1:
                self.fail(1, (name,), f"{_q(name)} is declared more than once: {', '.join(used)}")
        for sid, source in self.sources():
            if source.oracle == "default":
                self.fail(1, (sid,), f'source {_q(sid)} has the oracle name "default"')
        for node, path in _nodes(self.leaves):
            self.fail(1, (node, path), f"parameter {_q(node)} is a node of {_q(path)}")
        for owner, where, what, members, refs in self.unordered():
            self.repeats(owner, where, what, members, refs=refs)
        for sid, source in self.sources():
            if source.requirement is not None:
                self.repeated_values(sid, source.requirement.ref, source.requirement.values)
        for owner, key, where, entry in self.boundary_entries():
            if isinstance(entry, list):
                self.repeats(owner, where, "leaf paths", cast("list[str]", entry), refs=False)
                continue
            for role in ("bar", "thresholds", "margin"):
                self.repeats(owner, where, role, entry[role], refs=False)
            sets: set[bytes] = set()
            seen: set[bytes] = set()
            for m in cast("list[dict[str, list[str]]]", entry["mass"]):
                self.repeats(owner, where, "mass labels", m["labels"], refs=False)
                self.repeats(owner, where, "mass bounds", m["bounds"], refs=False)
                labels = _held(_unordered(m["labels"]))
                member = _held(
                    {"labels": _unordered(m["labels"]), "bounds": _unordered(m["bounds"])}
                )
                if member in seen:
                    self.fail(1, (owner, key), f"{where} lists a mass member twice")
                if labels in sets:
                    shown = labels.decode("utf-8")
                    self.fail(
                        1, (owner, key), f"{where} names the label set {shown} twice in its mass"
                    )
                seen.add(member)
                sets.add(labels)

    def unordered(self) -> Iterator[tuple[str, str, str, Sequence[str], bool]]:
        for sid, source in self.sources():
            yield sid, f"source {_q(sid)}", "needs", source.needs, True
        for rid, rule in self.rules():
            yield rid, f"rule {_q(rid)}", "reads", rule.reads, True
            yield rid, f"rule {_q(rid)}", "params", rule.params, False
        policy = self.policy()
        if policy is not None:
            yield "policy", "the policy", "reads", policy.reads, True
            yield "policy", "the policy", "params", policy.params, False

    def boundary_fields(self) -> Iterator[tuple[str, str, Mapping[str, Any], Sequence[str]]]:
        """Each declared ``boundaries`` field of a rule check 12 lets through and of the
        function policy: its owner, where it is, the field and its owner's reads."""
        for rid, rule in self.rules():
            if rule.boundaries is not None:
                yield rid, f"rule {_q(rid)}", rule.boundaries, rule.reads
        policy = self.policy()
        if policy is not None and policy.boundaries is not None:
            yield "policy", "the policy", policy.boundaries, policy.reads

    def boundary_entries(self) -> Iterator[tuple[str, str, str, Any]]:
        """Each entry of a ``boundaries`` field whose key check 7 passed: its owner, its key,
        where it is and the entry, a list of leaf paths or an object of the four roles."""
        for owner, where, boundaries, _ in self.boundary_fields():
            for key, entry in boundaries.items():
                if _parsed(key) is not None:
                    yield owner, key, f"{where}'s boundaries for {_q(key)}", entry

    def repeats(
        self, owner: str, where: str, what: str, members: Sequence[str], *, refs: bool
    ) -> None:
        seen: set[bytes] = set()
        for member in members:
            if (refs and _parsed(member) is None) or domain_error(member, MAX_DOCUMENT_DEPTH):
                continue
            key = _held(member)
            if key in seen:
                self.fail(1, (owner, member), f"{where} lists {_q(member)} twice in its {what}")
            seen.add(key)

    def repeated_values(self, sid: str, ref: str, values: Sequence[object]) -> None:
        """A source requirement's values, an unordered list, compared by canonical bytes among
        the values in the domain; check 10 reports the others."""
        seen: set[bytes] = set()
        for value in values:
            if not in_domain(value):
                continue
            key = encode(value)
            if key in seen:
                shown = key.decode("utf-8")
                self.fail(
                    1,
                    (sid, ref),
                    f"source {_q(sid)}'s requirement lists {shown} twice in its values",
                )
            seen.add(key)

    # --- check 10 --------------------------------------------------------------------------

    def parse(
        self, text: object, nullable: bool, where: str, names: tuple[str, ...]
    ) -> Type | None:
        if not isinstance(text, str):
            self.fail(10, names, f"{where} declares no type string")
            return None
        parsed = parse_type(text, nullable=nullable)
        if isinstance(parsed, TypeFailure):
            self.fail(10, names, f"{where}: {parsed.message}")
            return None
        return parsed

    def declared(
        self,
        text: object,
        nullable: bool,
        where: str,
        names: tuple[str, ...],
        ref: str | None = None,
    ) -> Type | None:
        """A type that passes check 10 with every definition it names, or ``None``; the size
        of the type ``ref`` is declared with is kept for the class trees' sum."""
        t = self.parse(text, nullable, where, names)
        if t is None or set(named(t)) & self.tainted:
            return None  # a definition it names failed, and that failure stands
        failures = check_declared(t, self.defs, nullable=nullable, memo=self.declared_memo)
        for f in failures:
            self.fail(10, (*names, *f.names), f"{where}: {f.message}")
        if failures:
            return None
        size = _type_size(cast(str, text), t, self.def_sizes)
        if size > _TYPE_SIZE:
            self.fail(10, names, f"{where}: {_too_large()}")
            return None
        if ref is not None:
            self.ref_sizes[ref] = size
        return t

    def check_10(self) -> None:
        d = self.d
        for name, body in d.types.items():
            t = self.parse(body, True, f"type definition {_q(name)}", (name,))
            if t is None:
                self.tainted.add(name)
            else:
                self.defs[name] = t
        unparsed = set(self.tainted)
        for f in check_definitions(self.defs):
            if f.kind is FailureKind.cycle:
                self.tainted.update(f.names)
            else:
                self.tainted.add(f.names[0])
                if f.names[1] in unparsed:
                    continue  # the definition it names failed to parse, and says so
            self.fail(10, f.names, f.message)
        self.taint_closure()
        # Definitions before the ones that name them, so a use of a failing one is skipped.
        order = TopologicalSorter(
            {
                n: {m for m in named(body) if m in self.defs}
                for n, body in sorted(self.defs.items())
                if n not in self.tainted
            }
        ).static_order()
        for name in order:
            # In order, so each definition it names is settled: one failing taints it.
            if name in self.tainted or set(named(self.defs[name])) & self.tainted:
                self.tainted.add(name)
                continue
            failures = check_declared(
                self.defs[name], self.defs, nullable=True, memo=self.declared_memo
            )
            for f in failures:
                self.fail(10, (name, *f.names), f"type definition {_q(name)}: {f.message}")
            if failures:
                self.tainted.add(name)
                continue
            # Every name it uses is defined and passed, so its size is taken once.
            size = _type_size(d.types[name], self.defs[name], self.def_sizes)
            if size > _TYPE_SIZE:
                self.fail(10, (name,), f"type definition {_q(name)}: {_too_large()}")
                self.tainted.add(name)
            else:
                self.def_sizes[name] = size
        self.taint_closure()

        state = self.declared(d.state, False, "the state declaration", (_STATE,), _STATE)
        if state is not None:
            if isinstance(resolve(state, self.defs), RecordOf):
                self.state_type = state
            else:
                self.fail(10, (_STATE,), "the state declaration is not a record type")
        for sid, source in self.sources():
            for ref, provided in source.provides.items():
                # A ref check 7 refuses is left out; its declaration is still checked.
                declared = _declared(ref) is not None
                if isinstance(provided, Question):
                    if declared:
                        self.types[ref] = AnswerType()
                        self.ref_sizes[ref] = _ANSWER_SIZE
                    continue
                text = cast("Mapping[str, Any]", provided)["type"]
                where = f"source {_q(sid)}'s {_q(ref)}"
                t = self.declared(text, True, where, (sid, ref), ref if declared else None)
                if t is not None and declared:
                    self.types[ref] = t
            if domain_error(source.payload) is not None:
                self.fail(10, (sid,), f"source {_q(sid)}'s payload is outside the value domain")
            if source.when is not None:
                self.version(source.when.version, f"source {_q(sid)}'s when", sid)
            requirement = source.requirement
            if requirement is not None and not all(in_domain(v) for v in requirement.values):
                self.fail(
                    10,
                    (sid, requirement.ref),
                    f"source {_q(sid)}'s requirement holds a value outside the value domain",
                )
        for rid, rule in self.rules():
            provides = rule.provides if _declared(rule.provides) is not None else None
            t = self.declared(rule.outputs, True, f"rule {_q(rid)}'s outputs", (rid,), provides)
            if t is not None and provides is not None:
                self.types[rule.provides] = t
            self.version(rule.version, f"rule {_q(rid)}", rid)
        policy = self.policy()
        if policy is not None:
            self.version(policy.version, "the policy", "policy")
        for requirement in d.requirements:
            if not in_domain(requirement.value):
                self.fail(
                    10,
                    (requirement.ref,),
                    f"the requirement on {_q(requirement.ref)} holds a value outside the value "
                    "domain",
                )
        if sum(self.ref_sizes.values()) > _TYPE_SIZE:
            # Every record's settings item writes each of these refs' class trees in full.
            self.fail(
                10,
                (_STATE, *self.provider_map()),
                f"the types of {_STATE} and the provided refs sum to more than "
                f"{power(_TYPE_SIZE)} code points with their definitions resolved",
            )
        for path, leaf in d.parameters.items():
            t = self.declared(leaf.type, False, f"parameter {_q(path)}", (path,))
            written = (
                leaf.value if self.parsed is not None else self.leaf_document_value(leaf.value, t)
            )
            if not in_domain(written):
                self.fail(10, (path,), f"parameter {_q(path)}'s value is outside the value domain")
            if t is None:
                continue
            full = _resolved(t, self.defs)
            if not _leaf_shape(full):
                self.fail(
                    10,
                    (path,),
                    f"parameter {_q(path)}'s type is not a scalar, a list of one or a mapping "
                    "of one",
                )
                continue
            self.leaf_types[path] = full
            if self.parsed is not None and value_conforms(leaf.value, full, {}, form="document"):
                self.live[path] = live_value(leaf.value, full)

    def leaf_value(self, path: str) -> object:
        """A leaf's value with ``bytes`` as Python ``bytes`` where its document form was read
        back; otherwise as declared."""
        return self.live.get(path, self.d.parameters[path].value)

    def leaf_document_value(self, value: object, t: Type | None) -> object:
        """A leaf value in its document form, ``bytes`` as base64url, when its type passed
        check 10 and it conforms; otherwise as given."""
        if t is None or set(named(t)) & self.tainted:
            return value
        full = _resolved(t, self.defs)
        if _leaf_shape(full) and value_conforms(value, full, {}, form="live"):
            return document_value(value, full)
        return value

    def taint_closure(self) -> None:
        changed = True
        while changed:
            changed = False
            for name, t in self.defs.items():
                if name not in self.tainted and set(named(t)) & self.tainted:
                    self.tainted.add(name)
                    changed = True

    def version(self, version: object, where: str, owner: str) -> None:
        if not isinstance(version, str) or not in_domain(version):
            self.fail(10, (owner,), f"{where} declares no version string")

    # --- check 8 ---------------------------------------------------------------------------

    def check_8(self) -> None:
        declared = self.leaves
        named_params: list[tuple[str, str, str]] = []
        for rid, rule in self.rules():
            named_params += [(rid, f"rule {_q(rid)}", p) for p in rule.params]
        policy = self.policy()
        if policy is not None:
            named_params += [("policy", "the policy", p) for p in policy.params]
        for sid, source in self.sources():
            for ref, slot in _slots(source):
                tag, value = next(iter(slot["origin"].items()))
                if tag == "param":
                    named_params.append((sid, f"source {_q(sid)}'s question {_q(ref)}", value))
        for owner, where, path in named_params:
            if path not in declared:
                self.fail(8, (owner, path), f"{where} names {_q(path)}, which is not a leaf")
        for owner, _, where, entry in self.boundary_entries():
            for path in _entry_paths(entry):
                if path not in declared:
                    self.refused_leaves.add(path)
                    self.fail(8, (owner, path), f"{where} names {_q(path)}, which is not a leaf")
        for path, t in self.leaf_types.items():
            if not value_conforms(self.leaf_value(path), t, {}, form="live"):
                self.fail(8, (path,), f"parameter {_q(path)}'s value does not conform to its type")

    def check_8_types(self) -> None:
        """Check 8's test of a named leaf's type: a leaf a ``boundaries`` field names has a
        type that conforms to ``number`` or ``mapping<string, number>``. It runs
        once checks 1 to 4, 10 and 12 pass, so every declared leaf has its type."""
        for owner, _, where, entry in self.boundary_entries():
            for path in _entry_paths(entry):
                t = self.leaf_types.get(path)
                if t is None or conforms(t, _NUMBER, {}) or conforms(t, _NUMBERS, {}):
                    continue
                self.refused_leaves.add(path)
                self.fail(
                    8,
                    (owner, path),
                    f"{where} names {_q(path)}, whose type does not conform to number or "
                    "mapping<string, number>",
                )

    # --- check 9 ---------------------------------------------------------------------------

    def check_9(self) -> None:
        for sid, source in self.sources():
            for ref, question in _questions(source):
                where = f"source {_q(sid)}'s question {_q(ref)}"
                self.text(question.wording, where, (sid, ref))
        for u in self.uses:
            if u.role == "boundary":
                continue  # a key's read is typed as its component's read (check 9)
            t = self.read_type(u)
            if isinstance(t, TypeFailure):
                names = (u.owner, u.ref) if u.owner else (u.ref,)
                self.fail(9, names, f"{u.where} reads {_q(u.ref)}: {t.message}")
        # Every requirement on the plan can hold. A step's requirements are
        # in plan declarations, which check 12 refuses, so this engine applies the clause to
        # none of them; nor the clause on a declaration's binding paths.
        for requirement in self.d.requirements:
            parsed = _parsed(requirement.ref)
            if parsed is None or parsed.fields:
                continue  # check 7 refused it, so no other check reads it
            t = self.types[f"{parsed.level}:{parsed.name}"]
            value = requirement.value
            # Check 10 refused a value outside the domain, and it gates this check.
            reason = hold_failure(value, t, self.defs)
            if reason is not None:
                self.fail(
                    9,
                    (requirement.ref,),
                    f"a plan-level requirement on {_q(requirement.ref)} can never hold: {reason}",
                )
        # A source requirement can hold when each of its values can.
        for u in self.uses:
            if u.role != "source-requirement":
                continue
            requirement = self.d.sources[u.owner].requirement
            assert requirement is not None
            t = self.types[u.base]
            for value in requirement.values:
                reason = hold_failure(value, t, self.defs)
                if reason is not None:
                    shown = encode(value).decode("utf-8")
                    self.fail(
                        9,
                        (u.owner, u.ref),
                        f"{u.where} on {_q(u.ref)} can never hold for {shown}: {reason}",
                    )
        self.check_9_boundaries()

    def check_9_boundaries(self) -> None:
        """Check 9's ``boundaries`` clauses. A leaf check 8 refused is left out. No table compares a
        read in v0.0.1, since check 12 refuses tables, so the clause on a boundary a table already
        compares applies to none."""
        reads = {owner: set(r) for owner, _, _, r in self.boundary_fields()}
        for owner, key, where, entry in self.boundary_entries():
            names = (owner, key)
            if key not in reads[owner]:
                self.fail(9, names, f"{where}: {_q(key)} is not one of its component's reads")
            parsed = _parsed(key)
            assert parsed is not None
            if parsed.level != "q":
                t = self.read_type(_Use(owner, where, "boundary", key, parsed))
                if not isinstance(t, TypeFailure) and not conforms(t, _NUMBER, self.defs):
                    self.fail(
                        9,
                        names,
                        f"{where}: {_q(key)} is neither a q: read nor of a type that conforms "
                        "to number",
                    )
                for path in self.typed(cast("list[str]", entry)):
                    self.bound(owner, where, path, self.leaf_types[path], _NUMBER, "number")
                continue
            question = self.question(f"q:{parsed.name}")
            labels = ("no", "yes") if question.kind is AnswerKind.noul else question.labels
            assert isinstance(labels, tuple)
            for role in ("bar", "margin"):
                for path in self.typed(entry[role]):
                    self.bound(owner, where, path, self.leaf_types[path], _UNIT, "number [0, 1]")
            for path in self.typed(entry["thresholds"]):
                t = self.leaf_types[path]
                self.bound(owner, where, path, t, _UNITS, "mapping<string, number [0, 1]>")
                value = self.leaf_value(path)
                if type(value) is dict:
                    keys = cast("dict[str, object]", value)
                    for label in code_point_sorted(set(keys) - set(labels)):
                        self.fail(
                            9,
                            (owner, key, path),
                            f"{where}: {_q(path)} has the key {_q(label)}, which is not a label of "
                            f"{_q(key)}",
                        )
            if question.kind is AnswerKind.noul and entry["thresholds"]:
                self.fail(9, names, f"{where}: a noul question's thresholds are empty")
            for m in cast("list[dict[str, list[str]]]", entry["mass"]):
                for path in self.typed(m["bounds"]):
                    self.bound(owner, where, path, self.leaf_types[path], _UNIT, "number [0, 1]")
                for label in code_point_sorted(set(m["labels"]) - set(labels)):
                    self.fail(
                        9,
                        names,
                        f"{where}: its mass names {_q(label)}, which is not a label of {_q(key)}",
                    )

    def typed(self, paths: Sequence[str]) -> list[str]:
        """The leaf paths check 8 let through, each with its type."""
        return [p for p in paths if p not in self.refused_leaves and p in self.leaf_types]

    def bound(self, owner: str, where: str, path: str, t: Type, to: Type, shown: str) -> None:
        if not conforms(t, to, {}):
            self.fail(9, (owner, path), f"{where}: {_q(path)}'s type does not conform to {shown}")

    def question(self, ref: str) -> Question:
        """The question that provides a ``q:`` ref; checks 3 and 4 have passed."""
        provider = self.providers[ref][0]
        assert provider is not None
        provided = self.d.sources[provider].provides[ref]
        assert isinstance(provided, Question)
        return provided

    def check_10_trees(self) -> None:
        """Check 10's test of each declared class tree against its ref's type, run once checks
        1 to 4, 10 and 12 pass."""
        for key, _ in self.class_keys:  # a key check 7 refused is left out
            t = self.state_type if key == _STATE else self.types[key]
            assert t is not None
            for message in _conformance(self.d.classes[key], t, self.defs, ""):
                self.fail(10, (key,), f"the class tree of {_q(key)} {message}")

    def read_type(self, u: _Use) -> Type | TypeFailure:
        if u.base == _STATE:
            assert self.state_type is not None
            base = self.state_type
        else:
            base = self.types[u.base]
        return read_type(base, u.parsed.fields, self.defs)

    def text(self, text: Text, where: str, names: tuple[str, ...]) -> None:
        parsed = _template(text.template)
        if isinstance(parsed, str):
            self.fail(9, names, f"{where}'s template {parsed}")
            return
        placeholders, has_text = parsed
        slot_names = [cast(str, s["name"]) for s in text.slots]
        counts = Counter(slot_names)
        for name in code_point_sorted({n for n, k in counts.items() if k > 1}):
            self.fail(9, (*names, name), f"{where} names the slot {_q(name)} twice")
        for name in code_point_sorted(set(placeholders) - set(slot_names)):
            self.fail(9, (*names, name), f"{where}'s placeholder {_q(name)} names no slot")
        for name in code_point_sorted(set(slot_names) - set(placeholders)):
            self.fail(9, (*names, name), f"{where}'s slot {_q(name)} is not in its template")
        # A question's wording holds a non-whitespace character outside its placeholders.
        if not has_text:
            self.fail(
                9,
                names,
                f"{where}'s template holds no character that is not whitespace outside its "
                "placeholders",
            )

    # --- the graph and check 5 -------------------------------------------------------------

    def provider_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for ref in code_point_sorted(self.providers):
            provider = self.providers[ref][0]
            if provider is not None:
                out[ref] = provider
        return out

    def needs(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {sid: set(s.needs) for sid, s in self.sources()}
        for u in self.uses:
            if u.role == "slot":
                out[u.owner].add(u.ref)
        return {sid: frozenset(refs) for sid, refs in out.items()}

    def graph(self) -> Mapping[str, frozenset[str]]:
        providers = self.provider_map()
        preds: dict[str, set[str]] = {n: set() for n in (*self.d.sources, *self.d.rules)}
        for u in self.uses:
            if u.role in ("need", "slot", "read", "source-requirement") and u.base != _STATE:
                preds[u.owner].add(providers[u.base])
        graph = {n: frozenset(preds[n]) for n in code_point_sorted(preds)}
        # Every predecessor is a key, or graphlib would add it as a root.
        assert all(p in graph for ps in graph.values() for p in ps)
        return MappingProxyType(graph)

    def check_5(self, graph: Mapping[str, frozenset[str]]) -> None:
        # Keys and predecessors in code-point order, so the cycle found is alike under every
        # hash seed; a frozenset's iteration order is not.
        ordered = {n: tuple(code_point_sorted(graph[n])) for n in code_point_sorted(graph)}
        sorter = TopologicalSorter(ordered)
        try:
            sorter.prepare()
        except CycleError as exc:
            found = cast("list[str]", exc.args[1])[:-1]  # graphlib repeats the first node
            start = found.index(min(found))
            cycle = found[start:] + found[:start]  # it starts at its least id
            shown = " -> ".join(_q(n) for n in (*cycle, cycle[0]))
            self.fail(5, tuple(cycle), f"the dependency graph has a cycle: {shown}")

    # --- the document and the loaded design ------------------------------------------------

    def document(self) -> dict[str, Any]:
        d = self.d
        policy = d.policy
        assert isinstance(policy, Policy)  # check 12 refuses the policy table
        parameters: dict[str, Any] = {}
        for path, leaf in d.parameters.items():
            t = self.leaf_types.get(path)
            value = self.leaf_value(path)
            if t is not None and value_conforms(value, t, {}, form="live"):
                value = document_value(value, t)
            else:
                value = leaf.value
            parameters[path] = {"type": leaf.type, "value": value}
        return {
            "format": 1,
            "name": d.name,
            "state": d.state,
            "sources": {sid: _source(s) for sid, s in d.sources.items()},
            "rules": {rid: _rule(cast(Rule, r)) for rid, r in d.rules.items()},
            "policy": {
                "form": "function",
                "reads": _unordered(policy.reads),
                "params": _unordered(policy.params),
                "boundaries": _boundaries(policy.boundaries),
                **_function(policy.function, policy.version),
            },
            "requirements": [{"ref": r.ref, "value": r.value} for r in d.requirements],
            "default_deny": d.default_deny,
            "tools": {name: {} for name in d.tools},
            "parameters": parameters,
            "types": dict(d.types),
            "plans": {},  # check 12 refuses every plan declaration
            "groups": {g: code_point_sorted(group.refs) for g, group in d.groups.items()},
            "classes": dict(d.classes),  # each tree plain data, from check 10's form test
        }

    # --- check 11's comparison -------------------------------------------------------------

    def compare(self, document: bytes) -> None:
        """Check 11's comparison: the document load computes from what it built and resolved
        equals the parsed document's canonical encoding, member by member, naming the
        component and member that differ; then that encoding equals the bytes given."""
        assert self.parsed is not None
        given = encode(self.parsed, MAX_DOCUMENT_DEPTH)
        if document != given:
            computed = cast("dict[str, Any]", decode(document))
            self.differences(computed, cast("dict[str, Any]", decode(given)))
            # The gate gave both documents one member set, so unequal bytes differ in a member.
            assert self.failed(frozenset({11}))
        elif self.given is not None and self.given != given:
            self.fail(11, (), "the document's bytes are not its canonical encoding")

    def differences(self, computed: dict[str, Any], given: dict[str, Any]) -> None:
        for member in code_point_sorted(computed):
            a, b = computed[member], given[member]
            if _same(a, b):
                continue
            kind = _KINDS.get(member)
            if member == "policy" and isinstance(b, dict):
                self.component("the policy", ("policy",), a, cast("dict[str, Any]", b))
            elif kind is not None and isinstance(b, dict):
                block = cast("dict[str, Any]", b)
                for name in code_point_sorted(a):
                    x, y = a[name], block[name]
                    if _same(x, y):
                        continue
                    where = f"{kind} {_q(name)}"
                    if _object(x) is not None and _object(y) is not None:
                        self.component(where, (name,), cast("dict[str, Any]", x), y)
                    else:
                        self.fail(11, (name,), f"{where} is not what load computes from it")
            else:
                self.fail(11, (), f"the document's {_q(member)} is not what load computes from it")

    def component(
        self, where: str, names: tuple[str, ...], computed: dict[str, Any], given: dict[str, Any]
    ) -> None:
        for member in code_point_sorted(computed):
            a, b = computed[member], given[member]
            if _same(a, b):
                continue
            x, y = _object(a), _object(b)
            if member == "when" and x is not None and y is not None:
                self.component(f"{where}'s when", names, x, y)
            elif member == "source" and b is None:
                self.fail(11, names, f"{where} holds a null source, where load reads its source")
            elif member == "source" and a is None:
                self.fail(
                    11, names, f"{where} holds a source hash, where load cannot read its source"
                )
            elif member in ("module", "qualname", "source"):
                self.fail(11, names, f"{where}'s {member} is not its function's")
            elif type(b) is list and _same(a, _unordered(cast("list[Any]", b))):
                self.fail(11, names, f"{where}'s {member} are not in canonical order")
            else:
                self.fail(11, names, f"{where}'s {_q(member)} is not what load computes from it")

    def loaded(self, document: bytes, graph: Mapping[str, frozenset[str]]) -> LoadedDesign:
        decoded = cast("dict[str, Any]", decode(document))
        providers = self.provider_map()
        classes: dict[str, Any] = {}
        assert self.state_type is not None
        typed = {_STATE: self.state_type, **self.types}
        resolutions: dict[str, Type] = {}
        for ref in (_STATE, *providers):
            classes[ref] = _tree(typed[ref], self.defs, self.d.classes.get(ref))
        callables: dict[str, Callable[..., Any]] = {
            rid: rule.function for rid, rule in self.rules()
        }
        for sid, source in self.sources():
            if source.when is not None:
                # The key a field function takes: the id, a dot, the field.
                callables[f"{sid}.when"] = source.when.function
        policy = self.policy()
        assert policy is not None
        callables["policy"] = policy.function
        return LoadedDesign(
            _BUILT_BY_LOAD,
            design=self.d,
            callables=MappingProxyType(callables),
            # Each `t:` and `d:` ref's declared type, its definitions resolved, for the
            # undeclared-read naming test.
            types=MappingProxyType(
                {
                    ref: _resolved(typed[ref], self.defs, resolutions)
                    for ref in code_point_sorted(typed)
                    if not ref.startswith("q:")
                }
            ),
            leaf_types=MappingProxyType(dict(self.leaf_types)),
            needs=MappingProxyType(self.needs()),
            groups=MappingProxyType(
                {g: tuple(code_point_sorted(group.refs)) for g, group in self.d.groups.items()}
            ),
            graph=graph,
            providers=MappingProxyType(providers),
            grid=_grid(decoded),
            classes_bytes=encode(classes, MAX_DOCUMENT_DEPTH),
            payload_bytes=MappingProxyType(
                {s: encode(v["payload"]) for s, v in decoded["sources"].items()}
            ),
            leaf_bytes=MappingProxyType(
                {p: encode(v["value"]) for p, v in decoded["parameters"].items()}
            ),
            # Load's own copies, from the document, never the caller's objects.
            requirement_bytes=MappingProxyType(
                {
                    sid: frozenset(encode(value) for value in v["requirement"]["values"])
                    for sid, v in decoded["sources"].items()
                    if v["requirement"] is not None
                }
            ),
            # A holder standing in the design document; each value it holds was tested.
            requirements_bytes=encode(decoded["requirements"], MAX_DOCUMENT_DEPTH - 1),
            document=document,
            hash=sha256_ref(document),
        )


# ---------------------------------------------------------------------------------------------
# Helpers


def _parsed(ref: str) -> _ParsedRef | None:
    try:
        return parse_ref(ref)
    except ValueError:  # the ref grammar's refusal
        return None


def _declared(ref: str) -> _ParsedRef | None:
    parsed = _parsed(ref)
    if parsed is None:
        return None
    return parsed if parsed.candidate is None and not parsed.fields else None


def _questions(source: Source) -> Iterator[tuple[str, Question]]:
    for ref, provided in source.provides.items():
        if isinstance(provided, Question):
            yield ref, provided


def _slots(source: Source) -> Iterator[tuple[str, Mapping[str, Any]]]:
    for ref, question in _questions(source):
        for slot in question.wording.slots:
            yield ref, slot


def _unordered(members: Sequence[Any]) -> list[Any]:
    """An unordered list in ascending order of its members' canonical bytes."""
    return sorted(members, key=_held)


def _held(member: object) -> bytes:
    """A design member's canonical bytes, encoded as a holder within the document's depth:
    only the values check 10 names are held to the size bounds."""
    return encode(member, MAX_DOCUMENT_DEPTH)


def _boundaries(boundaries: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A ``boundaries`` field as the document writes it: every list unordered, each mass
    member's lists sorted before the mass list is."""
    if boundaries is None:
        return None
    out: dict[str, Any] = {}
    for key, entry in boundaries.items():
        if _parsed(key) is None:  # check 7's key: its entry as given
            out[key] = entry
            continue
        if not isinstance(entry, dict):  # a list of leaf paths
            out[key] = _unordered(cast("list[str]", entry))
            continue
        roles = cast("dict[str, Any]", entry)
        mass = [
            {"labels": _unordered(m["labels"]), "bounds": _unordered(m["bounds"])}
            for m in cast("list[dict[str, list[str]]]", roles["mass"])
        ]
        out[key] = {
            "bar": _unordered(roles["bar"]),
            "thresholds": _unordered(roles["thresholds"]),
            "margin": _unordered(roles["margin"]),
            "mass": _unordered(mass),
        }
    return out


def _entry_paths(entry: Any) -> list[str]:
    """Every leaf path a ``boundaries`` entry names, in the order it gives them."""
    if isinstance(entry, list):
        return cast("list[str]", entry)
    roles = cast("dict[str, Any]", entry)
    paths = [*roles["bar"], *roles["thresholds"], *roles["margin"]]
    for m in cast("list[dict[str, list[str]]]", roles["mass"]):
        paths += m["bounds"]
    return paths


def _grid(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """The answer grid, from every ``boundaries`` field in the decoded document: keyed by read in
    code-point order, field paths kept. A ``d:`` or ``t:`` read's entry is a tuple of leaf paths; a
    ``q:`` read's maps ``bar``, ``thresholds`` and ``margin`` to tuples of leaf paths and ``mass``
    to a tuple of ``{labels, bounds}``, one per label set, its bounds sharing the set's one axis.
    Each role is the union over every component that names the read, without repeats and in
    ascending order of canonical bytes; the mass sets are in ascending order of their sorted labels'
    canonical bytes, and a set's labels are held in that order too, so a lint takes "the set's first
    label in declared order" from the question. A read whose boundaries name no leaf over
    every component has no entry. Read-only at every depth. A v0.0.1 design has no table, so
    the fields are all."""
    fields = [r.get("boundaries") for r in cast("dict[str, Any]", document["rules"]).values()]
    fields.append(cast("dict[str, Any]", document["policy"]).get("boundaries"))
    entries: dict[str, Any] = {}
    for field in fields:
        for key, entry in cast("dict[str, Any]", field or {}).items():
            if isinstance(entry, list):
                into = cast("dict[bytes, str]", entries.setdefault(key, {}))
                into.update((_held(p), p) for p in cast("list[str]", entry))
                continue
            roles = cast("dict[str, Any]", entry)
            acc = entries.setdefault(key, {"bar": {}, "thresholds": {}, "margin": {}, "mass": {}})
            for role in ("bar", "thresholds", "margin"):
                acc[role].update((_held(p), p) for p in roles[role])
            for m in cast("list[dict[str, list[str]]]", roles["mass"]):
                labels = _unordered(m["labels"])
                _, bounds = acc["mass"].setdefault(_held(labels), (labels, {}))
                bounds.update((_held(p), p) for p in m["bounds"])
    grid: dict[str, Any] = {}
    for key in code_point_sorted(entries):
        acc = entries[key]
        if not key.startswith("q:"):
            if acc:
                grid[key] = _ascending(acc)
            continue
        if not any(acc[role] for role in ("bar", "thresholds", "margin")) and not any(
            bounds for _, bounds in acc["mass"].values()
        ):
            continue  # no leaf, so no points
        mass = tuple(
            MappingProxyType({"labels": tuple(labels), "bounds": _ascending(bounds)})
            for _, (labels, bounds) in sorted(acc["mass"].items())
        )
        grid[key] = MappingProxyType(
            {
                "bar": _ascending(acc["bar"]),
                "thresholds": _ascending(acc["thresholds"]),
                "margin": _ascending(acc["margin"]),
                "mass": mass,
            }
        )
    return MappingProxyType(grid)


def _ascending(members: Mapping[bytes, str]) -> tuple[str, ...]:
    return tuple(members[b] for b in sorted(members))


def _function(fn: Callable[..., Any], version: str | None) -> dict[str, Any]:
    try:
        source: str | None = sha256_ref(inspect.getsource(fn).encode("utf-8"))
    except (OSError, TypeError):
        source = None
    return {
        "module": fn.__module__,
        "qualname": fn.__qualname__,
        "version": version,
        "source": source,
    }


def _text(text: Text) -> dict[str, Any]:
    return {
        "template": text.template,
        "slots": [{"name": s["name"], "origin": dict(s["origin"])} for s in text.slots],
    }


def _source(source: Source) -> dict[str, Any]:
    provides: dict[str, Any] = {}
    for ref, provided in source.provides.items():
        if isinstance(provided, Question):
            labels = provided.labels
            provides[ref] = {
                "kind": str(provided.kind),
                "wording": _text(provided.wording),
                "labels": None if labels is None else list(cast("tuple[str, ...]", labels)),
            }
        else:
            provides[ref] = dict(cast("Mapping[str, Any]", provided))
    w, r = source.when, source.requirement
    return {
        "provides": provides,
        "needs": _unordered(source.needs),
        "when": None if w is None else _function(w.function, w.version),
        "requirement": None if r is None else {"ref": r.ref, "values": _unordered(r.values)},
        "oracle": source.oracle,
        "timeout": source.timeout,
        "payload": source.payload,
        "cost": source.cost,
        "feedback": source.feedback,
        "grounded": source.grounded,
    }


def _rule(rule: Rule) -> dict[str, Any]:
    return {
        "form": "function",
        "provides": rule.provides,
        "outputs": rule.outputs,
        "reads": _unordered(rule.reads),
        "params": _unordered(rule.params),
        "boundaries": _boundaries(rule.boundaries),
        "candidates": rule.candidates,
        "grounded": rule.grounded,
        **_function(rule.function, rule.version),
    }


def _template(template: str) -> tuple[list[str], bool] | str:
    """A template's placeholders and whether it holds a character that is not whitespace
    outside them, or why it is malformed."""
    names: list[str] = []
    has_text = False
    i, n = 0, len(template)
    while i < n:
        c = template[i]
        if template.startswith("{{", i) or template.startswith("}}", i):
            has_text = True
            i += 2
        elif c == "{":
            end = template.find("}", i + 1)
            name = template[i + 1 : end] if end != -1 else ""
            if end == -1 or not is_segment(name):
                return f"has a {{ at position {i} that begins no placeholder or escape"
            names.append(name)
            i = end + 1
        elif c == "}":
            return f"has a lone }} at position {i}"
        else:
            has_text = has_text or not c.isspace()
            i += 1
    return names, has_text


def _resolved(t: Type, defs: Mapping[str, Type], memo: dict[str, Type] | None = None) -> Type:
    """A type with every definition's name in it resolved, at every depth; each definition is
    resolved once, kept in ``memo`` by name, and its resolution stands at each use."""
    memo = {} if memo is None else memo
    if isinstance(t, Named):
        found = memo.get(t.name)
        if found is None:
            # A chain of names is followed in a loop, so its length costs no recursion.
            found = memo[t.name] = _resolved(resolve(t, defs), defs, memo)
        return found
    if isinstance(t, ListOf):
        return ListOf(_resolved(t.item, defs, memo))
    if isinstance(t, MappingOf):
        return MappingOf(_resolved(t.value, defs, memo))
    if isinstance(t, RecordOf):
        return RecordOf(tuple((n, _resolved(ft, defs, memo)) for n, ft in t.fields))
    if isinstance(t, Nullable):
        return Nullable(_resolved(t.inner, defs, memo))
    return t


def _too_large() -> str:
    return (
        f"the type spells more than {power(_TYPE_SIZE)} code points with its definitions resolved"
    )


def _type_size(text: str, t: Type, sizes: Mapping[str, int]) -> int:
    """A type's size: the code points of its spelling, ``t`` parsed from it, with each
    definition's name replaced by the definition's size, from ``sizes``; at most one past the
    bound, so no count grows with the design's sharing."""
    size = len(string_data(text))
    stack = [t]
    while stack:
        node = stack.pop()
        if isinstance(node, Named):
            size += sizes[node.name] - len(node.name)
        elif isinstance(node, ListOf):
            stack.append(node.item)
        elif isinstance(node, MappingOf):
            stack.append(node.value)
        elif isinstance(node, RecordOf):
            stack.extend(ft for _, ft in node.fields)
        elif isinstance(node, Nullable):
            stack.append(node.inner)
    return min(size, _TYPE_SIZE + 1)


def _scalar(t: Type) -> bool:
    return isinstance(t, Scalar | Interval | Label)


def _leaf_shape(t: Type) -> bool:
    """A leaf's type is a scalar, a list of a scalar, or a mapping to a scalar."""
    if isinstance(t, ListOf):
        return _scalar(t.item)
    if isinstance(t, MappingOf):
        return _scalar(t.value)
    return _scalar(t)


def _tree(t: Type, defs: Mapping[str, Type], declared: Any = None) -> Any:
    """A type's class tree in full, ``declared`` applied over the default:
    an object at every record, a one-member list at every mapping layer, a class name at every
    other node. A declared class name covers every value beneath it; a field a declared object
    leaves out keeps its default. ``declared`` conforms to ``t`` (check 10)."""
    t = resolve(t, defs)
    if isinstance(t, Nullable):
        return _tree(t.inner, defs, declared)
    if isinstance(t, ListOf):
        return _tree(t.item, defs, declared)
    if isinstance(t, MappingOf):
        return [_tree(t.value, defs, declared)]
    if isinstance(t, RecordOf):
        fields = cast("dict[str, Any]", declared) if isinstance(declared, dict) else {}
        return {
            n: _tree(ft, defs, declared if isinstance(declared, str) else fields.get(n))
            for n, ft in t.fields
        }
    if isinstance(declared, str):
        return declared
    if isinstance(t, Scalar) and t.name in ("string", "bytes"):
        return "content"
    assert not isinstance(t, Named)
    return "open"


def _conformance(tree: Any, t: Type, defs: Mapping[str, Type], path: str) -> list[str]:
    """Why a class tree does not conform to ``t``: a class name conforms anywhere, and an
    object where ``t``, through list, mapping and ``or null`` layers and definitions' names,
    is a record, each key naming a field whose tree conforms."""
    if isinstance(tree, str):
        return []
    t = resolve(t, defs)
    while isinstance(t, Nullable | ListOf | MappingOf):
        inner = t.inner if isinstance(t, Nullable) else t.item if isinstance(t, ListOf) else t.value
        t = resolve(inner, defs)
    where = f"at {_q(path)}" if path else "at its root"
    if not isinstance(t, RecordOf):
        return [f"holds an object {where}, where its type is not a record"]
    fields = dict(t.fields)
    out: list[str] = []
    for key, member in cast("dict[str, Any]", tree).items():
        if key not in fields:
            out.append(f"names {_q(key)} {where}, which is not a field of the record")
        else:
            out += _conformance(member, fields[key], defs, f"{path}.{key}" if path else key)
    return out


# ---------------------------------------------------------------------------------------------
# A document: check 11's gate, resolving its functions, and the Design it declares


_DOCUMENT = frozenset(
    {
        "format",
        "name",
        "state",
        "sources",
        "rules",
        "policy",
        "requirements",
        "default_deny",
        "tools",
        "parameters",
        "types",
        "plans",
        "groups",
        "classes",
    }
)
_FUNCTION = frozenset({"module", "qualname", "version", "source"})
_SOURCE = frozenset(
    {
        "provides",
        "needs",
        "when",
        "requirement",
        "oracle",
        "timeout",
        "payload",
        "cost",
        "feedback",
        "grounded",
    }
)
_RULE = (
    frozenset(
        {"form", "provides", "outputs", "reads", "params", "boundaries", "candidates", "grounded"}
    )
    | _FUNCTION
)
_TABLE = frozenset(
    {"form", "provides", "outputs", "candidates", "grounded", "hit", "order", "inputs", "rows"}
)
_POLICY = frozenset({"form", "reads", "params", "boundaries"}) | _FUNCTION
_POLICY_TABLE = frozenset({"form", "inputs", "rows"})
_PLAN = frozenset({"kind", "steps", "reason", "reply", "grounded"})
_STEP = frozenset({"tool", "arguments", "requirements"})
_ORIGINS = frozenset(
    {
        frozenset({"literal"}),
        frozenset({"ref"}),
        frozenset({"step", "path"}),
        frozenset({"candidate"}),
    }
)
_KINDS = {
    "sources": "source",
    "rules": "rule",
    "parameters": "parameter",
    "plans": "plan declaration",
    "groups": "group",
    "tools": "tool",
    "types": "type definition",
    "classes": "the classes block's key",
}


def _gate(given: bytes | Mapping[str, Any]) -> tuple[dict[str, Any] | None, Any]:
    """The parsed document, or check 11's one failure as its names and message: the depth,
    counted before any parse; UTF-8 JSON without a duplicate key; ``format``, read first; the
    value domain; then every object's members by its form. No message holds text from the parser or
    the document's values."""
    deep = f"the document nests more than {MAX_DOCUMENT_DEPTH} deep"
    if isinstance(given, bytes):
        if document_depth(given) > MAX_DOCUMENT_DEPTH:
            return None, ((), deep)
        try:
            given.decode("utf-8")
            doc: object = decode(given)
        except UnicodeDecodeError:
            return None, ((), "the document is not UTF-8")
        except _DuplicateKey:
            return None, ((), "the document holds an object with a key twice")
        except _DecodeError:
            return None, ((), "the document is not JSON whose numbers are in the value domain")
    else:
        # Before any other walk: a container at two places would expand at each.
        if shares_a_container(given):
            return None, ((), "the document holds a list or object at two places")
        if nesting_depth(given, MAX_DOCUMENT_DEPTH) > MAX_DOCUMENT_DEPTH:
            return None, ((), deep)
        doc = given
    if type(doc) is not dict:
        return None, ((), "the document is not a JSON object")
    members = cast("dict[str, Any]", doc)
    found = member_named(members, "format")  # no code of a caller's key runs
    if found is MISSING:
        return None, ((), "the document has no format")
    if type(found) is not int or found != 1:
        return None, ((), "the document's format is not 1, the one format this engine reads")
    # The document as a holder, then each value it holds to the size bounds too; it shares
    # no container, so the holder's walk is linear in it.
    if domain_error(members, MAX_DOCUMENT_DEPTH) is not None or any(
        domain_error(value) is not None for value in _value_places(members)
    ):
        return None, ((), "the document holds a value outside the value domain")
    return members, next(_member_failures(members), None)


def _value_places(doc: dict[str, Any]) -> Iterator[object]:
    """Each value the document holds where the size bounds apply too: each payload, leaf
    value, source requirement value and declared class tree, wherever the member checks will
    find it; the document itself is held to its depth alone. Members are found by their
    keys' string data, so no code of a caller's key runs."""
    for source in _objects_in(member_named(doc, "sources")):
        payload = member_named(source, "payload")
        if payload is not MISSING:
            yield payload
        values = member_named(member_named(source, "requirement"), "values")
        if type(values) is list or type(values) is tuple:
            yield from cast("list[object]", values)
    for leaf in _objects_in(member_named(doc, "parameters")):
        value = member_named(leaf, "value")
        if value is not MISSING:
            yield value
    yield from members_of(member_named(doc, "classes"))


def _objects_in(block: object) -> Iterator[object]:
    """A block's members that are objects."""
    for found in members_of(block):
        if type(found) is dict:
            yield found


def _missing_or_extra(
    obj: Mapping[str, Any], members: frozenset[str], where: str, names: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], str]]:
    for member in code_point_sorted(members - set(obj)):
        yield names, f"{where} has no member {_q(member)}"
    for member in code_point_sorted(set(obj) - members):
        yield names, f"{where} has the member {_q(member)}, which its form does not have"


def _objects(block: object) -> list[tuple[str, dict[str, Any]]]:
    """A block's members that are objects, in code-point order of their names; a block or
    member of another form is check 10's."""
    if not isinstance(block, dict):
        return []
    members = cast("dict[str, Any]", block)
    return [(n, members[n]) for n in code_point_sorted(members) if isinstance(members[n], dict)]


def _items(value: object) -> list[dict[str, Any]]:
    """A list's members that are objects."""
    if not isinstance(value, list | tuple):
        return []
    return [m for m in cast("list[Any]", value) if isinstance(m, dict)]


def _member_failures(doc: dict[str, Any]) -> Iterator[tuple[tuple[str, ...], str]]:
    """Every object's missing or extra member, in one traversal order: the document's members,
    then each block's in code-point order of its name, each declaration's in code-point order
    of its own. A step is the ``plan`` item's, each
    argument one origin as a declaration writes it. A table's ``inputs``
    and ``rows`` are tested only for the value domain, as their members are not yet decided."""
    yield from _missing_or_extra(doc, _DOCUMENT, "the document", ())
    for path, leaf in _objects(doc["parameters"]):
        yield from _missing_or_extra(
            leaf, frozenset({"type", "value"}), f"parameter {_q(path)}", (path,)
        )
    for name, plan in _objects(doc["plans"]):
        where = f"plan declaration {_q(name)}"
        yield from _missing_or_extra(plan, _PLAN, where, (name,))
        for member in ("reason", "reply"):
            if isinstance(plan.get(member), dict):
                yield from _text_failures(plan[member], f"{where}'s {member}", (name,))
        for step in _items(plan.get("steps")):
            yield from _missing_or_extra(step, _STEP, f"{where}'s step", (name,))
            for _, origin in _objects(step.get("arguments")):
                if frozenset(origin) not in _ORIGINS:
                    yield (name,), f"{where} has a step argument that is not one origin"
            for r in _items(step.get("requirements")):
                required = frozenset({"ref", "value"})
                yield from _missing_or_extra(r, required, f"{where}'s step requirement", (name,))
    yield from _function_failures(doc["policy"], "the policy", ("policy",), _POLICY, _POLICY_TABLE)
    for r in _items(doc["requirements"]):
        yield from _missing_or_extra(r, frozenset({"ref", "value"}), "a plan-level requirement", ())
    for rid, rule in _objects(doc["rules"]):
        yield from _function_failures(rule, f"rule {_q(rid)}", (rid,), _RULE, _TABLE)
    for sid, source in _objects(doc["sources"]):
        yield from _source_failures(sid, source)
    for name, tool in _objects(doc["tools"]):
        yield from _missing_or_extra(tool, frozenset(), f"tool {_q(name)}", (name,))


def _function_failures(
    obj: object,
    where: str,
    names: tuple[str, ...],
    function: frozenset[str],
    table: frozenset[str],
) -> Iterator[tuple[tuple[str, ...], str]]:
    """A rule's or the policy's members by its ``form``, and a function's boundaries."""
    if not isinstance(obj, dict):
        return
    members = cast("dict[str, Any]", obj)
    form = members.get("form")
    if "form" not in members:
        yield names, f"{where} has no member {_q('form')}"
    elif form not in ("function", "table"):
        yield names, f"{where}'s form is neither function nor table"
    else:
        yield from _missing_or_extra(
            members, function if form == "function" else table, where, names
        )
    if form == "function":
        for key, entry in _objects(members.get("boundaries")):
            parsed = _parsed(key)
            if parsed is None or parsed.level != "q":
                continue
            shown = f"{where}'s boundaries for {_q(key)}"
            roles = frozenset({"bar", "thresholds", "margin", "mass"})
            yield from _missing_or_extra(entry, roles, shown, (*names, key))
            for m in _items(entry.get("mass")):
                bounds = frozenset({"labels", "bounds"})
                yield from _missing_or_extra(m, bounds, f"{shown}'s mass member", (*names, key))


def _source_failures(sid: str, source: dict[str, Any]) -> Iterator[tuple[tuple[str, ...], str]]:
    where, names = f"source {_q(sid)}", (sid,)
    yield from _missing_or_extra(source, _SOURCE, where, names)
    for ref, provided in _objects(source.get("provides")):
        at, shown = (sid, ref), f"{where}'s {_q(ref)}"
        if not ref.startswith("q:"):
            yield from _missing_or_extra(provided, frozenset({"type", "candidates"}), shown, at)
            continue
        yield from _missing_or_extra(provided, frozenset({"kind", "wording", "labels"}), shown, at)
        if isinstance(provided.get("wording"), dict):
            yield from _text_failures(provided["wording"], f"{shown}'s wording", at)
        if isinstance(provided.get("labels"), dict):
            labels = frozenset({"fixed", "candidates"})
            yield from _missing_or_extra(provided["labels"], labels, f"{shown}'s labels", at)
    if isinstance(source.get("when"), dict):
        yield from _missing_or_extra(source["when"], _FUNCTION, f"{where}'s when", names)
    if isinstance(source.get("requirement"), dict):
        required = frozenset({"ref", "values"})
        yield from _missing_or_extra(
            source["requirement"], required, f"{where}'s requirement", names
        )


def _text_failures(
    text: dict[str, Any], where: str, names: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], str]]:
    yield from _missing_or_extra(text, frozenset({"template", "slots"}), where, names)
    for slot in _items(text.get("slots")):
        yield from _missing_or_extra(slot, frozenset({"name", "origin"}), f"{where}'s slot", names)
        origin = _object(slot.get("origin"))
        if origin is not None and not (
            len(origin) == 1 and set(origin) <= {"ref", "param", "candidate"}
        ):
            yield names, f"{where} has a slot whose origin is not one of ref, param or candidate"


def _unresolved(*args: Any, **kwargs: Any) -> Any:
    """What a function component load could not resolve holds; such a design fails load."""


def _components(doc: dict[str, Any]) -> list[tuple[str, str, tuple[str, ...], dict[str, Any]]]:
    """Each function component: its ``functions`` key, where it is, its names and its
    object."""
    out: list[tuple[str, str, tuple[str, ...], dict[str, Any]]] = []
    for rid, rule in _objects(doc["rules"]):
        if rule["form"] == "function":
            out.append((rid, f"rule {_q(rid)}", (rid,), rule))
    for sid, source in _objects(doc["sources"]):
        when = _object(source["when"])
        if when is not None:
            out.append((f"{sid}.when", f"source {_q(sid)}'s when", (sid,), when))
    policy = _object(doc["policy"])
    if policy is not None and policy["form"] == "function":
        out.append(("policy", "the policy", ("policy",), policy))
    return out


def _nodes(paths: frozenset[str]) -> list[tuple[str, str]]:
    """Each parameter path that is a node of another, with that path, in the other's
    code-point order: read from a trie of the paths' segments, so each path is walked once."""
    trie: list[dict[str, int]] = [{}]
    ends: set[int] = set()
    for path in paths:
        at = 0
        for part in path.split("/"):
            child = trie[at].get(part)
            if child is None:
                child = len(trie)
                trie[at][part] = child
                trie.append({})
            at = child
        ends.add(at)
    out: list[tuple[str, str]] = []
    for path in code_point_sorted(paths):
        parts = path.split("/")
        at = 0
        for i, part in enumerate(parts[:-1], 1):
            at = trie[at][part]
            if at in ends:
                out.append(("/".join(parts[:i]), path))
    return out


def _resolve(
    doc: dict[str, Any], functions: Mapping[str, Any]
) -> tuple[dict[str, Any], frozenset[str], list[DesignFailure]]:
    """Each function component's callable, by its key: through ``functions`` first, else
    through ``importlib`` by its module and qualified name. A key naming two components
    resolves neither, and check 1 reports the clash; any other component unresolved is a check
    11 failure."""
    components = _components(doc)
    keys = [key for key, _, _, _ in components]
    resolved: dict[str, Any] = {}
    unresolved: set[str] = set()
    failures: list[DesignFailure] = []
    counts = Counter(keys)
    for key, where, names, obj in components:
        if counts[key] > 1:
            unresolved.add(key)
            continue
        if key in functions:
            resolved[key] = functions[key]
            continue
        module, qualname = obj["module"], obj["qualname"]
        if module == "__main__" or (
            isinstance(qualname, str) and ("<locals>" in qualname or "<lambda>" in qualname)
        ):
            message = (
                f"{where}'s function is in __main__ or has a <locals> or <lambda> qualname, so "
                "it must be in functions"
            )
        else:
            found = _import(module, qualname)
            if found is not None:
                resolved[key] = found
                continue
            message = (
                f"{where}'s function is not in functions and its module and qualname import none"
            )
        unresolved.add(key)
        failures.append(new_failure(11, names, message))
    for key in unresolved:
        resolved[key] = _unresolved
    return resolved, frozenset(unresolved), failures


def _import(module: object, qualname: object) -> Any:
    """The object a module and qualified name name, or ``None``. Importing runs the module's
    top-level code and the qualname walk may run a module's ``__getattr__``: an ``Exception``
    either raises leaves the component unresolved, its text not kept; any other
    ``BaseException``, such as ``SystemExit``, propagates out of ``load``."""
    if not isinstance(module, str) or not isinstance(qualname, str):
        return None
    try:
        found: Any = importlib.import_module(module)
        for part in qualname.split("."):
            found = getattr(found, part)
    except Exception:  # the import's or the walk's; a BaseException propagates
        return None
    return found


def _text_of(value: object) -> Any:
    if not isinstance(value, dict):
        return value
    text = cast("dict[str, Any]", value)
    return Text(text["template"], text["slots"])


def _build(doc: dict[str, Any], resolved: Mapping[str, Any]) -> Design:
    """The ``Design`` a document declares, each function component holding its resolved
    callable. A member of another form passes through as it is, for check 10 to refuse."""

    def block(value: object, build: Callable[[str, dict[str, Any]], Any]) -> Any:
        members = _object(value)
        if members is None:
            return value
        return {k: v if (o := _object(v)) is None else build(k, o) for k, v in members.items()}

    def provided(ref: str, value: dict[str, Any]) -> Any:
        if not ref.startswith("q:"):
            return value
        return Question(value["kind"], _text_of(value["wording"]), value["labels"])

    def source(sid: str, s: dict[str, Any]) -> Source:
        w, r = _object(s["when"]), _object(s["requirement"])
        return Source(
            provides=block(s["provides"], provided),
            needs=s["needs"],
            when=s["when"] if w is None else When(resolved[f"{sid}.when"], w["version"]),
            requirement=s["requirement"] if r is None else SourceRequirement(r["ref"], r["values"]),
            oracle=s["oracle"],
            timeout=s["timeout"],
            payload=s["payload"],
            cost=s["cost"],
            feedback=s["feedback"],
            grounded=s["grounded"],
        )

    def rule(rid: str, r: dict[str, Any]) -> Rule | Table:
        if r["form"] == "table":
            return Table(
                r["provides"],
                r["outputs"],
                r["hit"],
                r["inputs"],
                r["rows"],
                order=r["order"],
                candidates=r["candidates"],
                grounded=r["grounded"],
            )
        return Rule(
            r["provides"],
            r["outputs"],
            resolved[rid],
            version=r["version"],
            reads=r["reads"],
            params=r["params"],
            boundaries=r["boundaries"],
            candidates=r["candidates"],
            grounded=r["grounded"],
        )

    policy: Any = doc["policy"]
    if isinstance(policy, dict):
        p = cast("dict[str, Any]", policy)
        if p["form"] == "table":
            policy = PolicyTable(p["inputs"], p["rows"])
        else:
            policy = Policy(
                resolved["policy"],
                version=p["version"],
                reads=p["reads"],
                params=p["params"],
                boundaries=p["boundaries"],
            )
    requirements: Any = doc["requirements"]
    if isinstance(requirements, list | tuple):
        requirements = [
            m if (o := _object(m)) is None else Requirement(o["ref"], o["value"])
            for m in cast("list[Any]", requirements)
        ]
    tools: Any = doc["tools"]
    return Design(
        state=doc["state"],
        policy=policy,
        sources=block(doc["sources"], source),
        rules=block(doc["rules"], rule),
        requirements=requirements,
        default_deny=doc["default_deny"],
        tools=list(cast("dict[str, Any]", tools)) if isinstance(tools, dict) else tools,
        parameters=block(doc["parameters"], lambda _, v: Parameter(v["type"], v["value"])),
        types=doc["types"],
        plans=block(
            doc["plans"],
            lambda _, v: PlanDeclaration(
                v["kind"], v["steps"], _text_of(v["reason"]), _text_of(v["reply"]), v["grounded"]
            ),
        ),
        groups=_each(doc["groups"], Group),
        classes=doc["classes"],
        name=doc["name"],
    )


def _each(value: object, build: Callable[[Any], Any]) -> Any:
    if not isinstance(value, dict):
        return value
    return {k: build(v) for k, v in cast("dict[str, Any]", value).items()}


def _object(value: object) -> dict[str, Any] | None:
    """A document's object, or ``None`` for any other value."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _same(a: object, b: object) -> bool:
    return encode(a, MAX_DOCUMENT_DEPTH) == encode(b, MAX_DOCUMENT_DEPTH)
