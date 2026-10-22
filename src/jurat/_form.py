# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
"""Check 10's test of each member's form.

A design type validates nothing when built, so load tests here that every member of a
``Design`` is of its declared form, that a question's labels are well formed, and that a source's
``timeout`` is positive and its ``cost`` non-negative. ``check_form`` returns each failure and a
cleaned design: a member of the wrong form is dropped, or replaced by its default, so the
other checks read only well-formed members; the design is invalid by then, so the cleaned
copy is never encoded. A ``boundaries`` field is rebuilt as load's own plain lists, so no
later change to the caller's lists reaches the loaded design.
"""

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from types import MappingProxyType
from typing import Any, NamedTuple, cast

from jurat._answer import AnswerKind, ValueType
from jurat._canonical import decode, encode
from jurat._design import (
    Design,
    DesignFailure,
    Group,
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
    new_failure,
)
from jurat._refs import code_point_sorted, parse_ref
from jurat._values import (
    MAX_DEPTH,
    MAX_INTEGER,
    Count,
    SizeExceeded,
    in_domain,
    is_scalar_string,
    power,
    quote,
    string_data,
)

_ORIGINS = frozenset({"ref", "param", "candidate"})
_CLASSES = frozenset({"open", "identifier", "content"})
MAX_LABELS = 2**18
"""The most labels a question declares; read at each check, so a test may set it smaller."""
MAX_LABEL_CODE_POINTS = 2**24
"""The most code points a question's label names and descriptions hold in all."""
_Q_ROLES = frozenset({"bar", "thresholds", "margin", "mass"})
_X_ROLES = frozenset({"null", "confidence", "value"})
X_ROLES = ("null", "confidence", "value")
"""An extraction read's ``boundaries`` roles."""
_BARE = object()
"""A bare label's description: it declares none."""
_HITS = frozenset({"U", "A", "P", "F", "C+", "C<", "C>", "C#"})


class _Form:
    def __init__(self, unresolved: frozenset[str]) -> None:
        self.unresolved = unresolved
        """The function components a document named that load could not resolve, by their
        ``functions`` key: check 11 reports them, so check 10 skips their callable."""
        self.failures: list[DesignFailure] = []
        self.providers: list[tuple[str, str]] = []
        """Each ref a refused member's declaration still provides, and its provider."""
        self.leaves: list[str] = []
        """Each refused leaf's path, still a declared name and leaf."""

    def fail(self, names: tuple[str, ...], message: str) -> None:
        self.failures.append(new_failure(10, code_point_sorted(names), message))

    # --- the shapes ------------------------------------------------------------------------

    def mapping(self, value: object, where: str, names: tuple[str, ...]) -> dict[str, Any]:
        """A mapping keyed by strings of distinct string data; ``{}`` after a failure."""
        if not isinstance(value, Mapping):
            self.fail(names, f"{where} is not a mapping")
            return {}
        out: dict[str, Any] = {}
        for key, member in cast("Mapping[object, Any]", value).items():
            if not isinstance(key, str) or not is_scalar_string(key):
                self.fail(names, f"{where} has a key that is not a string")
                continue
            data = string_data(key)
            if data in out:
                self.fail((*names, data), f"{where} has two keys {quote(data)}")
                continue
            out[data] = member
        return out

    def strings(self, value: object, where: str, names: tuple[str, ...]) -> tuple[str, ...]:
        """A sequence of strings; the strings it holds after a failure."""
        if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
            self.fail(names, f"{where} is not a list")
            return ()
        out: list[str] = []
        for member in cast("Sequence[object]", value):
            if isinstance(member, str) and is_scalar_string(member):
                out.append(string_data(member))
            else:
                self.fail(names, f"{where} holds a member that is not a string")
        return tuple(out)

    def string(
        self, value: object, where: str, names: tuple[str, ...], *, optional: bool = True
    ) -> str | None:
        if value is None and optional:
            return None
        if isinstance(value, str) and is_scalar_string(value):
            return string_data(value)
        self.fail(names, f"{where} is not a string")
        return None

    def boolean(self, value: object, default: bool, where: str, names: tuple[str, ...]) -> bool:
        if type(value) is bool:
            return value
        self.fail(names, f"{where} is not a boolean")
        return default

    def function(self, value: object, where: str, names: tuple[str, ...], key: str) -> bool:
        if key in self.unresolved:
            return False
        ok = callable(value) and all(
            is_scalar_string(getattr(value, a, None)) for a in ("__module__", "__qualname__")
        )
        if not ok:
            self.fail(names, f"{where} is not a function with a module and a qualified name")
        return ok

    # --- the declarations ------------------------------------------------------------------

    def design(self, d: Design) -> Design:
        sources = {
            sid: s
            for sid, v in self.mapping(d.sources, "sources", ()).items()
            if (s := self.source(sid, v)) is not None
        }
        # The tags summed exactly are within the domain's integers, so a round's cost and the
        # budget's running total are too.
        if sum(Fraction(repr(s.cost)) for s in sources.values()) > MAX_INTEGER:
            paying = tuple(sid for sid, s in sources.items() if s.cost > 0)
            self.fail(paying, "the sources' costs sum to more than 2^53 - 1")
        rules: dict[str, Rule | Table] = {}
        for rid, v in self.mapping(d.rules, "rules", ()).items():
            built = self.rule(rid, v)
            if built is not None:
                rules[rid] = built
        requirements: list[Requirement] = []
        if isinstance(d.requirements, Sequence):
            for r in cast("Sequence[object]", d.requirements):
                if not isinstance(r, Requirement):
                    self.fail((), "requirements holds a member that is not a Requirement")
                elif not is_scalar_string(r.ref):
                    self.fail((), "a requirement's ref is not a string")
                else:
                    requirements.append(r)
        else:
            self.fail((), "requirements is not a list")
        parameters = self.members(d.parameters, "parameters", Parameter, dropped=self.leaves)
        groups = {
            g: Group(self.strings(v.refs, f"group {quote(g)}'s refs", (g,)))
            for g, v in self.members(d.groups, "groups", Group).items()
        }
        plans = self.members(d.plans, "plans", PlanDeclaration)
        for name, plan in plans.items():
            self.plan(name, plan)
        policy: object = d.policy
        if isinstance(policy, Policy):
            policy = self.policy(policy)
        elif isinstance(policy, PolicyTable):
            self.data(policy.inputs, policy.rows, "the policy table", ("policy",))
        else:
            self.fail(("policy",), "the policy is not a Policy or a PolicyTable")
        return Design(
            state=d.state,
            policy=cast(Policy, policy),
            sources=sources,
            rules=rules,
            requirements=requirements,
            default_deny=self.boolean(d.default_deny, True, "default_deny", ()),
            tools=self.strings(d.tools, "tools", ()),
            parameters=parameters,
            types=self.mapping(d.types, "types", ()),
            plans=plans,
            groups=groups,
            classes=self.classes(d.classes),
            name=self.string(d.name, "the design's name", ()),
        )

    def members(
        self, value: object, where: str, kind: type, *, dropped: list[str] | None = None
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, member in self.mapping(value, where, ()).items():
            if isinstance(member, kind):
                out[name] = member
            else:
                self.fail((name,), f"{where} maps {quote(name)} to no {kind.__name__}")
                if dropped is not None:
                    dropped.append(name)
        return out

    def source(self, sid: str, s: object) -> Source | None:
        where, names = f"source {quote(sid)}", (sid,)
        if not isinstance(s, Source):
            self.fail(names, f"{where} is not a Source")
            return None
        provides: dict[str, Any] = {}
        for ref, value in self.mapping(s.provides, f"{where}'s provides", names).items():
            built = self.provided(ref, value, f"{where}'s {quote(ref)}", (sid, ref))
            if built is not None:
                provides[ref] = built
            else:
                self.providers.append((ref, sid))
        timeout = s.timeout
        if timeout is not None and not _number(timeout):
            self.fail(names, f"{where}'s timeout is not a number in the value domain")
            timeout = None
        elif timeout is not None and timeout <= 0:
            self.fail(names, f"{where}'s timeout is not a positive number of seconds")
            timeout = None
        cost = s.cost
        if not _number(cost):
            self.fail(names, f"{where}'s cost is not a number in the value domain")
            cost = 0
        elif cost < 0:
            self.fail(names, f"{where}'s cost is negative")
            cost = 0
        return Source(
            provides=provides,
            needs=self.strings(s.needs, f"{where}'s needs", names),
            when=self.when(s.when, where, names, f"{sid}.when"),
            requirement=self.requirement(s.requirement, where, names),
            oracle=self.string(s.oracle, f"{where}'s oracle name", names),
            timeout=timeout,
            payload=s.payload,
            cost=cost,
            feedback=self.boolean(s.feedback, False, f"{where}'s feedback", names),
            grounded=self.string(s.grounded, f"{where}'s grounding ref", names),
        )

    def when(self, w: object, where: str, names: tuple[str, ...], key: str) -> When | None:
        """A function ``when`` as a ``When``; its ``version`` is check 10's in load."""
        if w is None:
            return None
        if not isinstance(w, When):
            self.fail(names, f"{where}'s when is not a When")
            return None
        self.function(w.function, f"{where}'s when", names, key)
        return w

    def requirement(
        self, r: object, where: str, names: tuple[str, ...]
    ) -> SourceRequirement | None:
        """A source requirement: a string ref and a list of one or more values; whether each
        value is in the domain is load's."""
        if r is None:
            return None
        where = f"{where}'s requirement"
        if not isinstance(r, SourceRequirement):
            self.fail(names, f"{where} is not a SourceRequirement")
            return None
        ref = self.string(r.ref, f"{where}'s ref", names, optional=False)
        values: object = r.values
        if isinstance(values, str | bytes | bytearray) or not isinstance(values, Sequence):
            self.fail(names, f"{where}'s values are not a list")
            return None
        if not values:
            self.fail(names, f"{where}'s values hold no value")
            return None
        if ref is None:
            return None
        return SourceRequirement(ref, values)

    def classes(self, value: object) -> dict[str, Any]:
        """The classes block, each tree as plain data; a tree of the wrong form is dropped.
        Whether a tree conforms to its type is load's."""
        out: dict[str, Any] = {}
        for ref, tree in self.mapping(value, "classes", ()).items():
            try:
                plain = _tree(tree, 0, Count())
            except SizeExceeded:
                self.fail((ref,), f"the class tree of {quote(ref)} is outside the value domain")
                continue
            if plain is None:
                self.fail(
                    (ref,),
                    f"the class tree of {quote(ref)} is not a class name or an object of trees",
                )
            else:
                out[ref] = plain
        return out

    def provided(self, ref: str, value: object, where: str, names: tuple[str, ...]) -> Any:
        if ref.startswith("q:"):
            if not isinstance(value, Question):
                self.fail(names, f"{where} is not a Question")
                return None
            return self.question(value, where, names)
        if isinstance(value, str):
            return MappingProxyType({"type": value, "candidates": False})
        if isinstance(value, Mapping):
            members = cast("Mapping[object, Any]", value)
            if "type" in members and set(members) <= {"type", "candidates"}:
                candidates = self.boolean(
                    members.get("candidates", False), False, f"{where}'s candidates", names
                )
                text = members["type"]
                if issubclass(type(text), str):
                    text = string_data(text)
                return MappingProxyType({"type": text, "candidates": candidates})
        self.fail(names, f"{where} is not a type or a mapping of type and candidates")
        return None

    def question(self, q: Question, where: str, names: tuple[str, ...]) -> Question | None:
        ok = True
        if not isinstance(q.kind, AnswerKind):
            self.fail(names, f"{where}'s kind is not noul, choice, score or extraction")
            ok = False
        wording = self.text(q.wording, f"{where}'s wording", names)
        if wording is None:
            ok = False
        labels: object = q.labels
        if q.kind is AnswerKind.extraction:
            ok = self.extraction(q, where, names) and ok
        elif isinstance(q.kind, AnswerKind) and not _probability_members_unset(q):
            self.fail(
                names,
                f"{where} is a {q.kind.value} question and declares value_type, is_list, text "
                "or reference_date, which only an extraction question declares",
            )
            ok = False
        if q.kind is AnswerKind.noul:
            if labels is not None:
                self.fail(names, f"{where} is a noul question and declares labels")
                ok = False
        elif q.kind is AnswerKind.extraction:
            if labels is not None:
                self.fail(names, f"{where} is an extraction question and declares labels")
                ok = False
        elif isinstance(labels, Mapping) and q.kind is AnswerKind.choice:
            members = cast("Mapping[object, Any]", labels)
            if set(members) != {"fixed", "candidates"}:
                self.fail(names, f"{where}'s label set is not a mapping of fixed and candidates")
                ok = False
            else:
                fixed = self.labels(members["fixed"], f"{where}'s fixed labels", names, 0)
                ref = self.string(
                    members["candidates"], f"{where}'s candidates", names, optional=False
                )
                ok = ok and fixed is not None and ref is not None
        elif isinstance(q.kind, AnswerKind):
            ok = self.labels(labels, f"{where}'s labels", names, 1) is not None and ok
        if not ok:
            return None
        assert wording is not None
        return Question(
            q.kind,
            wording,
            q.labels,
            value_type=q.value_type,
            is_list=q.is_list,
            text=q.text,
            reference_date=q.reference_date,
        )

    def extraction(self, q: Question, where: str, names: tuple[str, ...]) -> bool:
        """An extraction question's members of their form: a value type, a boolean list flag,
        a board-ref origin for ``text`` and one or ``None`` for ``reference_date``."""
        ok = True
        if not isinstance(q.value_type, ValueType):
            self.fail(
                names,
                f"{where}'s value_type is not text, number, date, time, datetime or duration",
            )
            ok = False
        if type(q.is_list) is not bool:
            self.fail(names, f"{where}'s is_list is not a boolean")
            ok = False
        if not _ref_origin(q.text):
            self.fail(names, f"{where}'s text is not a board-ref origin, an object of one ref")
            ok = False
        if q.reference_date is not None and not _ref_origin(q.reference_date):
            self.fail(
                names,
                f"{where}'s reference_date is not null or a board-ref origin, an object of one ref",
            )
            ok = False
        return ok

    def labels(
        self, value: object, where: str, names: tuple[str, ...], least: int
    ) -> tuple[str, ...] | None:
        """Labels, at least ``least`` of them, each a name, a string, or exactly an object of
        ``name`` and ``description``, both strings, the description holding a character that is
        not whitespace; the names distinct. At most ``MAX_LABELS`` labels, their names and
        descriptions holding at most ``MAX_LABEL_CODE_POINTS`` code points in all, counted
        before any is read, so every answer to the question and every call item asking it is
        within the value domain's size bounds. The names in order, or ``None`` after a
        failure."""
        if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
            self.fail(names, f"{where} are not a list of labels")
            return None
        members = cast("Sequence[object]", value)
        if len(members) > MAX_LABELS:
            self.fail(names, f"{where} number more than {power(MAX_LABELS)}")
            return None
        parts: list[tuple[object, object]] = []
        for member in members:
            if isinstance(member, str):
                parts.append((member, _BARE))
            elif isinstance(member, Mapping):
                described = cast("Mapping[object, object]", member)
                if set(described) != {"name", "description"}:
                    self.fail(
                        names,
                        f"{where} hold an object that is not exactly a name and a description",
                    )
                    return None
                parts.append((described["name"], described["description"]))
            else:
                self.fail(names, f"{where} hold a label that is neither a string nor an object")
                return None
        if not all(isinstance(n, str) and isinstance(d, str) for n, d in parts if d is not _BARE):
            self.fail(names, f"{where} hold a label whose name or description is not a string")
            return None
        strings = [cast(str, s) for pair in parts for s in pair if s is not _BARE]
        if sum(str.__len__(s) for s in strings) > MAX_LABEL_CODE_POINTS:
            self.fail(names, f"{where} hold more than {power(MAX_LABEL_CODE_POINTS)} code points")
            return None
        if not all(is_scalar_string(s) for s in strings):
            self.fail(names, f"{where} hold a string that is not of Unicode scalar values")
            return None
        data = [string_data(cast(str, n)) for n, _ in parts]
        if len(data) < least:
            self.fail(names, f"{where} hold no label")
            return None
        if len(set(data)) != len(data):
            self.fail(names, f"{where} repeat a label")
            return None
        if any(
            d is not _BARE and all(c.isspace() for c in string_data(cast(str, d))) for _, d in parts
        ):
            self.fail(names, f"{where} hold a description with no character that is not whitespace")
            return None
        return tuple(data)

    def text(self, text: object, where: str, names: tuple[str, ...]) -> Text | None:
        if not isinstance(text, Text):
            self.fail(names, f"{where} is not a Text")
            return None
        if self.string(text.template, f"{where}'s template", names, optional=False) is None:
            return None
        slots = text.slots
        if isinstance(slots, str | bytes | bytearray) or not isinstance(slots, Sequence):
            self.fail(names, f"{where}'s slots are not a list")
            return None
        for slot in cast("Sequence[object]", slots):
            if not _slot_form(slot):
                self.fail(
                    names,
                    f"{where} has a slot that is not a name and one ref, param or candidate origin",
                )
                return None
        return text

    def rule(self, rid: str, r: object) -> Rule | Table | None:
        where, names = f"rule {quote(rid)}", (rid,)
        if isinstance(r, Table):
            self.table(rid, r)
            return r  # a refused form: check 10's member forms test it, check 12 names it
        if not isinstance(r, Rule):
            self.fail(names, f"{where} is not a Rule or a Table")
            return None
        provides = self.string(r.provides, f"{where}'s provides", names, optional=False)
        if provides is None:
            return None
        self.function(r.function, f"{where}'s function", names, rid)
        return Rule(
            provides=provides,
            outputs=r.outputs,
            function=r.function,
            version=r.version,
            reads=self.strings(r.reads, f"{where}'s reads", names),
            params=self.strings(r.params, f"{where}'s params", names),
            boundaries=self.boundaries(r.boundaries, where, names),
            candidates=self.boolean(r.candidates, False, f"{where}'s candidates", names),
            grounded=self.string(r.grounded, f"{where}'s grounding ref", names),
        )

    def policy(self, p: Policy) -> Policy:
        names = ("policy",)
        self.function(p.function, "the policy's function", names, "policy")
        return Policy(
            function=p.function,
            version=p.version,
            reads=self.strings(p.reads, "the policy's reads", names),
            params=self.strings(p.params, "the policy's params", names),
            boundaries=self.boundaries(p.boundaries, "the policy", names),
        )

    # --- boundaries --------------------------------------------------------------------------

    def boundaries(self, value: object, where: str, names: tuple[str, ...]) -> Any:
        """A ``boundaries`` field as plain data, or ``None``: keyed by read,
        a ``d:`` or ``t:`` read's entry a list of leaf paths and a ``q:`` read's an object of
        its four roles. An entry of the wrong form is dropped; a key that is not a ref is kept
        with ``None``, for check 7 alone, since a ref check 7 refuses is left out of check 10."""
        if value is None:
            return None
        where = f"{where}'s boundaries"
        out: dict[str, Any] = {}
        for key, entry in self.mapping(value, where, names).items():
            try:
                level = parse_ref(key).level
            except ValueError:  # the ref grammar's refusal, check 7's
                # Kept as given, a plain copy, so the document a document load computes is the
                # one it was given; an entry outside the domain is None, as its design fails.
                out[key] = decode(encode(entry)) if in_domain(entry) else None
                continue
            shown, at = f"{where} for {quote(key)}", (*names, key)
            cleaned = self.roles(entry, shown, at) if level == "q" else self.paths(entry, shown, at)
            if cleaned is not None:
                out[key] = cleaned
        return out

    # --- the refused forms' members (check 12) -----------------------------------------------

    def table(self, rid: str, r: Table) -> None:
        """A table's members of their declared form; its ``inputs`` and
        ``rows`` only in the value domain, their members not yet decided."""
        where, names = f"rule {quote(rid)}", (rid,)
        self.string(r.provides, f"{where}'s provides", names, optional=False)
        self.string(r.outputs, f"{where}'s outputs", names, optional=False)
        if not (isinstance(r.hit, str) and string_data(r.hit) in _HITS):
            self.fail(names, f"{where}'s hit is not one of U, A, P, F, C+, C<, C> or C#")
        if r.order is not None and not (
            not isinstance(r.order, str | bytes | bytearray)
            and isinstance(r.order, Sequence)
            and in_domain(list(r.order))
        ):
            self.fail(names, f"{where}'s order is not null or a list in the value domain")
        self.boolean(r.candidates, False, f"{where}'s candidates", names)
        self.string(r.grounded, f"{where}'s grounding ref", names)
        self.data(r.inputs, r.rows, where, names)

    def data(self, inputs: object, rows: object, where: str, names: tuple[str, ...]) -> None:
        for member, value in (("inputs", inputs), ("rows", rows)):
            if not in_domain(value):
                self.fail(names, f"{where}'s {member} are outside the value domain")

    def plan(self, name: str, p: PlanDeclaration) -> None:
        """A plan declaration's members of their declared form; its steps
        only a list in the value domain."""
        where, names = f"plan declaration {quote(name)}", (name,)
        self.string(p.kind, f"{where}'s kind", names, optional=False)
        steps: object = p.steps
        if isinstance(steps, str | bytes | bytearray) or not isinstance(steps, Sequence):
            self.fail(names, f"{where}'s steps are not a list")
        elif not in_domain(list(cast("Sequence[object]", steps))):
            self.fail(names, f"{where}'s steps are outside the value domain")
        for member, text in (("reason", p.reason), ("reply", p.reply)):
            if text is not None:
                self.text(text, f"{where}'s {member}", names)
        self.string(p.grounded, f"{where}'s grounding ref", names)

    def paths(self, value: object, where: str, names: tuple[str, ...]) -> list[str] | None:
        """A list of strings, as load's own list, or ``None`` after a failure."""
        if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
            self.fail(names, f"{where} is not a list")
            return None
        members = cast("Sequence[object]", value)
        if not all(isinstance(m, str) and is_scalar_string(m) for m in members):
            self.fail(names, f"{where} holds a member that is not a string")
            return None
        return [string_data(cast(str, m)) for m in members]

    def roles(self, value: object, where: str, names: tuple[str, ...]) -> dict[str, Any] | None:
        """A ``q:`` read's entry: ``bar``, ``thresholds``, ``margin`` and ``mass``, all four; or
        an extraction read's ``null``, ``confidence`` and ``value``, all three. Which form its
        read's kind takes is check 9's."""
        keys = frozenset(cast("Mapping[str, Any]", value)) if isinstance(value, Mapping) else None
        if keys == _X_ROLES:
            entry = cast("Mapping[str, Any]", value)
            paths = {role: self.paths(entry[role], f"{where}'s {role}", names) for role in X_ROLES}
            if any(p is None for p in paths.values()):
                return None
            return paths
        if keys != _Q_ROLES:
            self.fail(
                names,
                f"{where} is not a mapping of bar, thresholds, margin and mass, nor of null, "
                "confidence and value",
            )
            return None
        entry = cast("Mapping[str, Any]", value)
        out: dict[str, Any] = {}
        for role in ("bar", "thresholds", "margin"):
            out[role] = self.paths(entry[role], f"{where}'s {role}", names)
        mass = entry["mass"]
        if isinstance(mass, str | bytes | bytearray) or not isinstance(mass, Sequence):
            self.fail(names, f"{where}'s mass is not a list")
            return None
        sets: list[dict[str, Any]] = []
        for member in cast("Sequence[object]", mass):
            if not isinstance(member, Mapping) or set(cast("Mapping[object, Any]", member)) != {
                "labels",
                "bounds",
            }:
                self.fail(names, f"{where}'s mass holds a member that is not labels and bounds")
                return None
            fields = cast("Mapping[str, Any]", member)
            labels = self.paths(fields["labels"], f"{where}'s mass labels", names)
            bounds = self.paths(fields["bounds"], f"{where}'s mass bounds", names)
            if labels is None or bounds is None:
                return None
            sets.append({"labels": labels, "bounds": bounds})
        if any(out[role] is None for role in ("bar", "thresholds", "margin")):
            return None
        out["mass"] = sets
        return out


def _probability_members_unset(q: Question) -> bool:
    """A probability kind's question leaves every extraction member at its default."""
    return (
        q.value_type is None
        and type(q.is_list) is bool
        and q.is_list is False
        and q.text is None
        and q.reference_date is None
    )


def _ref_origin(value: object) -> bool:
    """A board-ref origin: a mapping of the one tag ``ref`` to a string."""
    if not isinstance(value, Mapping) or len(cast("Mapping[object, Any]", value)) != 1:
        return False
    ((tag, ref),) = cast("Mapping[object, Any]", value).items()
    return tag == "ref" and isinstance(ref, str) and is_scalar_string(ref)


def _slot_form(slot: object) -> bool:
    if not isinstance(slot, Mapping):
        return False
    members = cast("Mapping[object, Any]", slot)
    if set(members) != {"name", "origin"} or not is_scalar_string(members["name"]):
        return False
    origin = members["origin"]
    if not isinstance(origin, Mapping) or len(cast("Mapping[object, Any]", origin)) != 1:
        return False
    ((tag, value),) = cast("Mapping[object, Any]", origin).items()
    return tag in _ORIGINS and isinstance(value, str) and is_scalar_string(value)


def _tree(tree: object, depth: int, count: Count) -> Any:
    """A class tree as plain data: a class name, or a ``dict`` of trees keyed by strings;
    ``None`` when it is neither, at any depth, or nests past the value bound. The tree, each
    key and each member counts as a value, with its code points, at every place it stands;
    past the size bounds the walk stops with ``SizeExceeded``, before the copy grows further."""
    count.add(tree)
    if isinstance(tree, str):
        return string_data(tree) if string_data(tree) in _CLASSES else None
    if not isinstance(tree, Mapping) or depth >= MAX_DEPTH:
        return None
    out: dict[str, Any] = {}
    for key, member in cast("Mapping[object, object]", tree).items():
        count.add(key)
        if not isinstance(key, str) or not is_scalar_string(key) or string_data(key) in out:
            return None
        plain = _tree(member, depth + 1, count)
        if plain is None:
            return None
        out[string_data(key)] = plain
    return out


def _number(value: object) -> bool:
    if type(value) is int:
        return -MAX_INTEGER <= value <= MAX_INTEGER
    return type(value) is float and math.isfinite(value)


class _Checked(NamedTuple):
    design: Design
    """The design with each member of the wrong form dropped or defaulted."""
    failures: list[DesignFailure]
    providers: tuple[tuple[str, str], ...]
    """Each ref a dropped member's declaration names, and its provider: still a provider for
    checks 2 and 3."""
    leaves: tuple[str, ...]
    """Each dropped leaf's path: still a declared name for check 1 and leaf for check 8."""


def check_form(design: Design, unresolved: frozenset[str] = frozenset()) -> _Checked:
    """Check 10's form test: the failures, the cleaned design, and what a dropped member
    still provides. ``unresolved`` names the function components, by ``functions``
    key, whose callable check 11 reports instead."""
    form = _Form(unresolved)
    cleaned = form.design(design)
    return _Checked(cleaned, form.failures, tuple(form.providers), tuple(form.leaves))
