# pyright: reportPrivateUsage=false
"""The type set: the canonical notation, definitions, conformance and dot-path reads."""

import enum
import math
import random
from collections import OrderedDict
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import pytest

from counting import SmallBounds
from jurat import types as t
from jurat._types import (
    AnswerType,
    FailureKind,
    ListOf,
    Named,
    Nullable,
    Scalar,
    Type,
    TypeFailure,
    _Inexact,
    _Invalid,
    b64url_decode,
    b64url_encode,
    check_declared,
    check_definitions,
    conforms,
    make_nullable,
    named,
    parse_type,
    read_type,
    render,
    resolve,
    type_depth,
    value_conforms,
)

MAX = 2**53 - 1


def p(text: str, *, nullable: bool = True) -> Type:
    """Parse a canonical type string, failing the test on any failure."""
    parsed = parse_type(text, nullable=nullable)
    assert not isinstance(parsed, TypeFailure), parsed
    return parsed


def defs(**bodies: str) -> dict[str, Type]:
    return {name.replace("__", "/"): p(body) for name, body in bodies.items()}


class Colour(enum.StrEnum):
    a = "a"
    red = "red"


class Level(enum.IntEnum):
    one = 1


class MyStr(str):
    __slots__ = ()

    def __str__(self) -> str:
        return "overridden"


class MyFloat(float):
    __slots__ = ()


class MyBytes(bytes):
    __slots__ = ()


class MyList(list[Any]):
    pass


class MyDict(dict[str, Any]):
    pass


class SameData(str):
    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


# --- the canonical spellings ---------------------------------------------

CANONICAL = [
    "number",
    "integer",
    "string",
    "boolean",
    "bytes",
    "answer",
    'label{"a"}',
    'label{"a", "b"}',
    'label{"No", "Yes, please"}',
    'label{"", "a"}',
    'label{"Z", "a"}',
    'label{"\\u0000", "\\n", "\\"", "\\\\", "]", "}"}',
    'label{"\x7f", "\u2028"}',
    # Code-point order, not UTF-16 order: U+FFFF before U+1F600.
    'label{"\uffff", "\U0001f600"}',
    "number [0, 1]",
    "number [-1.5, 2.5]",
    "number [0.5, 0.5]",
    "number [1e-7, 1e+21]",
    "number [-1.7976931348623157e+308, 5e-324]",
    "number [0, 9007199254740992]",
    "number [-1152921504606847000, 1152921504606847000]",
    "integer [0, 10]",
    "integer [-9007199254740991, 9007199254740991]",
    "list<string>",
    "list<list<integer [0, 3]>>",
    "mapping<string, number [0, 1]>",
    'mapping<string, list<label{"a"}>>',
    "record{}",
    "record{a: string}",
    "record{a: string or null, b: integer}",
    "record{address: record{city: string}, name: string}",
    "record{a1: bytes, a_: boolean, b: list<record{}>}",
    "record{number: string, or: null_type}",
    "email/role",
    "list<email/role>",
    "role",
    "numbers",
    "list/x",
]

NULLABLE_CANONICAL = [
    "string or null",
    "record{a: string} or null",
    "list<string> or null",
    "email/role or null",
    "number [0, 1] or null",
    'label{"a"} or null',
]


@pytest.mark.parametrize("text", CANONICAL)
def test_a_canonical_spelling_parses_and_renders_to_itself(text: str) -> None:
    assert render(p(text, nullable=False)) == text
    assert render(p(text, nullable=True)) == text


@pytest.mark.parametrize("text", NULLABLE_CANONICAL)
def test_a_nullable_spelling_parses_where_null_is_allowed(text: str) -> None:
    parsed = p(text, nullable=True)
    assert isinstance(parsed, Nullable)
    assert render(parsed) == text


@pytest.mark.parametrize("text", NULLABLE_CANONICAL)
def test_a_nullable_spelling_is_ill_formed_where_null_is_not(text: str) -> None:
    failure = parse_type(text, nullable=False)
    assert isinstance(failure, TypeFailure)
    assert failure.kind is FailureKind.ill_formed
    assert failure.canonical is None


NOT_CANONICAL = [
    ('label{"b", "a"}', 'label{"a", "b"}'),
    ('label{"a","b"}', 'label{"a", "b"}'),
    ('label{"a" , "b"}', 'label{"a", "b"}'),
    ('label {"a"}', 'label{"a"}'),
    ('label{ "a" }', 'label{"a"}'),
    ('label{"\\u0061"}', 'label{"a"}'),
    ('label{"\\/"}', 'label{"/"}'),
    ('label{"\\u001F"}', 'label{"\\u001f"}'),
    ('label{"\\u00e9"}', 'label{"é"}'),
    ('label{"\\ud83d\\ude00", "\\uffff"}', 'label{"\uffff", "\U0001f600"}'),
    ("number[0, 1]", "number [0, 1]"),
    ("number [0,1]", "number [0, 1]"),
    ("number  [0, 1]", "number [0, 1]"),
    ("number [ 0, 1 ]", "number [0, 1]"),
    ("number [0.0, 1.0]", "number [0, 1]"),
    ("number [0.50, 1]", "number [0.5, 1]"),
    ("number [1E2, 1e3]", "number [100, 1000]"),
    ("number [-0, 1]", "number [0, 1]"),
    ("number [-0.0, 1]", "number [0, 1]"),
    ("number [1e21, 1e22]", "number [1e+21, 1e+22]"),
    ("number [1E-7, 1]", "number [1e-7, 1]"),
    ("number [1e-6, 1]", "number [0.000001, 1]"),
    ("number [0, 9007199254740992.0]", "number [0, 9007199254740992]"),
    ("number [0, 1152921504606846976]", "number [0, 1152921504606847000]"),
    ("integer [1.0, 2e1]", "integer [1, 20]"),
    ("integer [-0, 0]", "integer [0, 0]"),
    (" string", "string"),
    ("string ", "string"),
    ("string\n", "string"),
    ("list< string>", "list<string>"),
    ("list<string >", "list<string>"),
    ("list <string>", "list<string>"),
    ("mapping<string,number>", "mapping<string, number>"),
    ("mapping<string,  number>", "mapping<string, number>"),
    ("mapping< string, number>", "mapping<string, number>"),
    ("record{b: string, a: string}", "record{a: string, b: string}"),
    ("record{a:string}", "record{a: string}"),
    ("record{a : string}", "record{a: string}"),
    ("record{ a: string }", "record{a: string}"),
    ("record{a: string,b: string}", "record{a: string, b: string}"),
    ("record{ }", "record{}"),
    ("record {}", "record{}"),
    (
        "record{z: record{b: string, a: string}, a: integer}",
        "record{a: integer, z: record{a: string, b: string}}",
    ),
    ("list<number [0,1]>", "list<number [0, 1]>"),
    ("string  or null", "string or null"),
    ("string\tor null", "string or null"),
    ("string or  null", "string or null"),
    ("record{a: string  or null}", "record{a: string or null}"),
]


@pytest.mark.parametrize(("text", "canonical"), NOT_CANONICAL)
def test_a_type_spelled_otherwise_fails_with_the_canonical_spelling(
    text: str, canonical: str
) -> None:
    failure = parse_type(text, nullable=True)
    assert isinstance(failure, TypeFailure)
    assert failure.kind is FailureKind.not_canonical
    assert failure.canonical == canonical
    assert canonical in failure.message
    assert render(p(canonical)) == canonical


MALFORMED = [
    "",
    " ",
    "Number",
    "STRING",
    "null",
    "or null",
    "or",
    "string or",
    "string or nil",
    "string or null or null",
    "string null",
    "string string",
    "string,",
    "&",
    "a//b",
    "/a",
    "a/",
    "a/B",
    "_a",
    "1a",
    "list",
    "list<>",
    "list<string",
    "list<string>>",
    "list[string]",
    "list<string, string>",
    "mapping",
    "mapping<string>",
    "mapping<integer, string>",
    'mapping<label{"a"}, string>',
    "mapping<string, >",
    "record",
    "record{a}",
    "record{a: }",
    "record{a: string,}",
    "record{,}",
    "record{a string}",
    'record{"a": string}',
    "record<a: string>",
    "label",
    "label{a}",
    "label{'a'}",
    'label{"a",}',
    'label{"a" "b"}',
    'label{"a"',
    'label{"\x01"}',
    'label{"a\\x"}',
    "label{1}",
    "number [0]",
    "number [0, 1, 2]",
    "number [, 1]",
    "number [0, ]",
    "number [*, 1]",
    "number [.5, 1]",
    "number [+1, 2]",
    "number [01, 2]",
    "number [0x1, 2]",
    "number [1., 2]",
    "number [NaN, 1]",
    "number [Infinity, 1]",
    "number [0, 1",
    "number (0, 1)",
    "string [0, 1]",
    "boolean [0, 1]",
    "answer [0, 1]",
    "number [0, 1] [0, 1]",
]


@pytest.mark.parametrize("text", MALFORMED, ids=repr)
def test_a_string_that_does_not_parse_is_malformed(text: str) -> None:
    for nullable in (True, False):
        failure = parse_type(text, nullable=nullable)
        assert isinstance(failure, TypeFailure)
        assert failure.kind is FailureKind.malformed
        assert failure.canonical is None


ILL_FORMED = [
    "label{}",
    'label{"a", "a"}',
    'label{"a", "\\u0061"}',
    'label{"\\ud800"}',
    'label{"a", "\\udfff"}',
    "number [1, 0]",
    "integer [3, 2]",
    "integer [0.5, 1]",
    "integer [0, 1e-1]",
    "integer [0, 9007199254740992]",
    "integer [-9007199254740992, 0]",
    "integer [0, 1e300]",
    "number [0, 1e400]",
    "number [-1e400, 0]",
    "integer [0, 1e400]",
    "number [0, 9007199254740993]",
    "number [-9007199254740993, 0]",
    "number [0, 1152921504606846977]",
    "number [0, 100000000000000000000000]",
    "integer [0, 9007199254740993]",
    "answer or null",
    "list<" * 65 + "string" + ">" * 65,
    "record{a: " * 65 + "string" + "}" * 65,
    "mapping<string, " * 65 + "string" + ">" * 65,
    "list<record{a: " * 33 + "string" + "}>" * 33,
    "list<string or null>",
    "mapping<string, string or null>",
    "list<record{a: string} or null>",
    "list<answer>",
    "mapping<string, answer>",
    "record{a: answer}",
    "record{a: answer or null}",
    "list<list<answer>>",
    "record{A: string}",
    "record{a/b: string}",
    "record{1a: string}",
    "record{_a: string}",
    "record{a: string, a: integer}",
    "record{a: string, b: integer, a: string}",
    "record{a: list<string or null>}",
]


@pytest.mark.parametrize("text", ILL_FORMED, ids=repr)
def test_a_string_that_parses_but_breaks_the_notation_is_ill_formed(text: str) -> None:
    failure = parse_type(text, nullable=True)
    assert isinstance(failure, TypeFailure)
    assert failure.kind is FailureKind.ill_formed
    assert failure.canonical is None


def test_a_syntax_error_is_reported_before_a_rule_of_the_notation() -> None:
    failure = parse_type("list<string or null", nullable=True)
    assert isinstance(failure, TypeFailure)
    assert failure.kind is FailureKind.malformed


def test_parse_takes_only_a_string() -> None:
    failure = parse_type(1, nullable=True)  # pyright: ignore[reportArgumentType]
    assert isinstance(failure, TypeFailure)
    assert failure.kind is FailureKind.malformed


def test_answer_parses_but_is_never_declarable() -> None:
    assert isinstance(p("answer"), AnswerType)
    failures = check_declared(p("answer"), {}, nullable=False)
    assert [f.kind for f in failures] == [FailureKind.not_declarable]
    failure = parse_type("answer or null", nullable=True)
    assert isinstance(failure, TypeFailure)
    assert failure.kind is FailureKind.ill_formed
    assert "answer" in failure.message


def test_the_failure_message_names_the_string() -> None:
    failure = parse_type("number [1, 0]", nullable=True)
    assert isinstance(failure, TypeFailure)
    assert "number [1, 0]" in failure.message


# --- the jurat.types constructors -------------------------------------------


def test_scalar_constants() -> None:
    assert t.string == "string"
    assert t.boolean == "boolean"
    assert t.bytes_ == "bytes"
    assert t.number() == "number"
    assert t.integer() == "integer"


CONSTRUCTED: list[tuple[str, str]] = [
    (t.number(0, 1), "number [0, 1]"),
    (t.number(0.5, 1.0), "number [0.5, 1]"),
    (t.number(-1e-7, 1e21), "number [-1e-7, 1e+21]"),
    (t.number(lo=0, hi=0), "number [0, 0]"),
    (t.number(-0.0, 0.1), "number [0, 0.1]"),
    (t.integer(0, 10), "integer [0, 10]"),
    (t.integer(1.0, 2.0), "integer [1, 2]"),
    (t.integer(-MAX, MAX), "integer [-9007199254740991, 9007199254740991]"),
    (t.number(0, 2.0**60), "number [0, 1152921504606847000]"),
    (t.number(-(2.0**53), MAX), "number [-9007199254740992, 9007199254740991]"),
    (t.label("b", "a"), 'label{"a", "b"}'),
    (t.label("Yes, please", "No"), 'label{"No", "Yes, please"}'),
    (t.label('a"b', "\\", "\n"), 'label{"\\n", "\\\\", "a\\"b"}'),
    (t.label("\U0001f600", "\uffff"), 'label{"\uffff", "\U0001f600"}'),
    (t.label(Colour.red), 'label{"red"}'),
    (t.label(MyStr("x")), 'label{"x"}'),
    (t.list_(t.string), "list<string>"),
    (t.list_(t.number(0, 1)), "list<number [0, 1]>"),
    (t.list_("email/role"), "list<email/role>"),
    (t.list_(t.list_(t.label("a"))), 'list<list<label{"a"}>>'),
    (t.mapping(t.number(0, 1)), "mapping<string, number [0, 1]>"),
    (t.record(), "record{}"),
    (t.record(b=t.string, a=t.nullable(t.integer())), "record{a: integer or null, b: string}"),
    (t.record(sender=t.record(address=t.string)), "record{sender: record{address: string}}"),
    (t.nullable(t.string), "string or null"),
    (t.nullable(t.record(a=t.bytes_)), "record{a: bytes} or null"),
    (t.nullable("email/role"), "email/role or null"),
]


@pytest.mark.parametrize(("built", "expected"), CONSTRUCTED)
def test_a_constructor_renders_the_canonical_string(built: str, expected: str) -> None:
    assert built == expected
    assert type(built) is str
    assert render(p(built)) == built


@pytest.mark.parametrize(
    "build",
    [
        lambda: t.number(0),
        lambda: t.number(hi=1),
        lambda: t.integer(lo=0),
        lambda: t.integer(None, 3),
        lambda: t.number(1, 0),
        lambda: t.integer(2, 1),
        lambda: t.integer(0.5, 1),
        lambda: t.integer(0, 2**53),
        lambda: t.integer(-(2**53), 0),
        lambda: t.number(0, 2**60),
        lambda: t.number(0, math.inf),
        lambda: t.number(math.nan, 1),
        lambda: t.integer(0, math.inf),
        lambda: t.label(),
        lambda: t.label("a", "a"),
        lambda: t.label("a", Colour.a),
        lambda: t.label("\ud800"),
        lambda: t.list_(t.nullable(t.string)),
        lambda: t.list_("answer"),
        lambda: t.list_("list< string>"),
        lambda: t.list_("Number"),
        lambda: t.mapping(t.nullable(t.string)),
        lambda: t.mapping("answer"),
        lambda: t.record(**{"A": t.string}),
        lambda: t.record(**{"a/b": t.string}),
        lambda: t.record(a="answer"),
        lambda: t.record(a=t.nullable("answer")),
        lambda: t.record(a="list<string or null>"),
        lambda: t.nullable(t.nullable(t.string)),
        lambda: t.nullable("answer"),
        lambda: t.nullable("string or null"),
    ],
)
def test_a_constructor_refuses_what_the_notation_cannot_spell(build: Any) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        build()


@pytest.mark.parametrize(
    "build",
    [
        lambda: t.number(True, 1),
        lambda: t.number(0, False),
        lambda: t.number("0", "1"),  # pyright: ignore[reportArgumentType]
        lambda: t.integer(Decimal(0), 1),  # pyright: ignore[reportArgumentType]
        lambda: t.number(Level.one, 2),
        lambda: t.label(1),  # pyright: ignore[reportArgumentType]
        lambda: t.label(b"a"),  # pyright: ignore[reportArgumentType]
        lambda: t.list_(1),  # pyright: ignore[reportArgumentType]
        lambda: t.mapping(None),  # pyright: ignore[reportArgumentType]
        lambda: t.record(a=1),  # pyright: ignore[reportArgumentType]
        lambda: t.nullable(None),  # pyright: ignore[reportArgumentType]
    ],
)
def test_a_constructor_refuses_a_wrong_type(build: Any) -> None:
    with pytest.raises(TypeError):
        build()


# --- type definitions -----------------------------------------


def test_acyclic_definitions_have_no_failure() -> None:
    d = defs(role='label{"a", "b"}', roles="list<role>", email__thing="record{r: roles}")
    assert check_definitions(d) == []


@pytest.mark.parametrize(
    ("bodies", "cycles"),
    [
        ({"a": "list<a>"}, [["a"]]),
        ({"a": "b", "b": "a"}, [["a", "b"]]),
        ({"a": "list<b>", "b": "record{x: a}"}, [["a", "b"]]),
        ({"c": "a", "a": "mapping<string, b>", "b": "record{x: a or null}"}, [["a", "b"]]),
        ({"a": "b", "b": "a", "x": "list<y>", "y": "x", "z": "string"}, [["a", "b"], ["x", "y"]]),
        ({"a": "b", "b": "c", "c": "a"}, [["a", "b", "c"]]),
    ],
)
def test_definitions_in_a_cycle_fail_naming_them(
    bodies: dict[str, str], cycles: list[list[str]]
) -> None:
    failures = check_definitions(defs(**bodies))
    assert [f.kind for f in failures] == [FailureKind.cycle] * len(cycles)
    for failure, names in zip(failures, cycles, strict=True):
        assert failure.names == tuple(names)
        for name in names:
            assert name in failure.message


def test_a_definition_naming_an_undefined_type_fails() -> None:
    failures = check_definitions(defs(a="list<nope>", b="record{x: also/nope}"))
    assert [(f.kind, f.names) for f in failures] == [
        (FailureKind.undefined, ("a", "nope")),
        (FailureKind.undefined, ("b", "also/nope")),
    ]


def test_check_declared_finds_an_undefined_name() -> None:
    failures = check_declared(p("record{a: role}"), {}, nullable=False)
    assert [(f.kind, f.names) for f in failures] == [(FailureKind.undefined, ("role",))]


def test_check_declared_passes_a_resolved_type() -> None:
    d = defs(role='label{"a", "b"}', addr="record{city: string}")
    for text in ["role", "list<role>", "record{a: addr or null, r: role}", "role or null"]:
        assert check_declared(p(text), d, nullable=True) == [], text


def test_answer_through_a_definition_is_not_declarable() -> None:
    d = defs(q="answer", wrap="q")
    for text in ["q", "wrap"]:
        failures = check_declared(p(text), d, nullable=True)
        assert [f.kind for f in failures] == [FailureKind.not_declarable], text


@pytest.mark.parametrize(
    ("text", "nullable"),
    [
        ("list<opt>", True),
        ("mapping<string, opt>", True),
        ("opt or null", True),
        ("opt", False),
        ("alias or null", True),
        ("list<alias>", True),
    ],
)
def test_a_definition_that_admits_null_follows_the_nullable_rule(text: str, nullable: bool) -> None:
    # Whether a definition's body may be `T or null` is the caller's (load's) parse; this
    # checks the uses of one that does, with its definitions resolved.
    d = defs(opt="string or null", alias="opt")
    failures = check_declared(p(text), d, nullable=nullable)
    assert [f.kind for f in failures] == [FailureKind.ill_formed], text


def test_a_definition_that_admits_null_may_stand_where_null_may() -> None:
    d = defs(opt="string or null")
    assert check_declared(p("opt"), d, nullable=True) == []
    assert check_declared(p("record{a: opt}"), d, nullable=False) == []


# --- type-to-type conformance ---------------------------------------------------------------

D = defs(role='label{"a", "b"}', addr="record{city: string}", count="integer [0, 9]")

TYPE_CONFORMANCE: list[tuple[str, str, bool]] = [
    # numbers and intervals
    ("integer", "number", True),
    ("number", "integer", False),
    ("integer [0, 5]", "integer", True),
    ("integer [0, 5]", "number", True),
    ("integer [0, 5]", "integer [0, 10]", True),
    ("integer [0, 5]", "integer [0, 5]", True),
    ("integer [0, 10]", "integer [0, 5]", False),
    ("integer [-1, 5]", "integer [0, 10]", False),
    ("integer [0, 5]", "number [0, 5]", True),
    ("integer [0, 1]", "number [0.5, 1]", False),
    ("number [0, 1]", "integer [0, 1]", False),
    ("number [1, 1]", "integer [1, 1]", False),
    ("number [0.2, 0.8]", "number [0, 1]", True),
    ("number [0, 1]", "number [0.2, 0.8]", False),
    ("number [0, 1]", "number", True),
    ("number", "number [0, 1]", False),
    ("integer", "integer [0, 9]", False),
    # by form: an unbounded type never conforms to an interval, whatever its bounds
    ("integer", "integer [-9007199254740991, 9007199254740991]", False),
    ("integer", "number [-9007199254740991, 9007199254740991]", False),
    ("number", "number [-1.7976931348623157e+308, 1.7976931348623157e+308]", False),
    ("integer [-9007199254740991, 9007199254740991]", "integer", True),
    ("integer [-9007199254740991, 9007199254740991]", "number", True),
    ("list<integer>", "list<integer [-9007199254740991, 9007199254740991]>", False),
    # labels and strings
    ('label{"a"}', 'label{"a", "b"}', True),
    ('label{"a", "b"}', 'label{"a", "b"}', True),
    ('label{"a", "b"}', 'label{"a"}', False),
    ('label{"a", "c"}', 'label{"a", "b"}', False),
    ('label{"a"}', "string", True),
    ("string", 'label{"a"}', False),
    ('label{"1"}', "number", False),
    # other scalars
    ("string", "string", True),
    ("boolean", "boolean", True),
    ("bytes", "bytes", True),
    ("bytes", "string", False),
    ("string", "bytes", False),
    ("boolean", "integer", False),
    ("integer", "boolean", False),
    ("integer", "string", False),
    # lists and mappings, covariantly
    ("list<integer>", "list<number>", True),
    ("list<number>", "list<integer>", False),
    ("list<list<integer [0, 1]>>", "list<list<number>>", True),
    ('mapping<string, label{"a"}>', "mapping<string, string>", True),
    ("mapping<string, string>", 'mapping<string, label{"a"}>', False),
    ("list<string>", "mapping<string, string>", False),
    ("mapping<string, string>", "list<string>", False),
    ("list<string>", "string", False),
    # records: every field of the target, with conforming types; extra fields allowed
    ("record{a: integer, b: string}", "record{a: number}", True),
    ("record{a: number}", "record{a: number, b: string}", False),
    ("record{a: string}", "record{a: string or null}", True),
    ("record{}", "record{a: string or null}", False),
    ("record{a: string or null}", "record{a: string}", False),
    ("record{a: integer}", "record{a: string}", False),
    ("record{a: string}", "record{}", True),
    ("record{r: record{x: integer, y: integer}}", "record{r: record{x: number}}", True),
    ("record{r: record{y: integer}}", "record{r: record{x: number}}", False),
    ("record{l: list<record{a: integer, b: string}>}", "record{l: list<record{a: number}>}", True),
    ("record{a: string}", "mapping<string, string>", False),
    ("mapping<string, string>", "record{}", False),
    # null
    ("string", "string or null", True),
    ("string or null", "string", False),
    ('label{"a"} or null', "string or null", True),
    ("integer or null", "number or null", True),
    ("number or null", "integer or null", False),
    ("record{a: integer, b: string} or null", "record{a: number} or null", True),
    # answer
    ("answer", "answer", True),
    ("answer", "string", False),
    ("string", "answer", False),
    # definitions resolved
    ('label{"a"}', "role", True),
    ("role", "string", True),
    ("role", 'label{"a"}', False),
    ("role", "role or null", True),
    ("record{city: string, zip: string}", "addr", True),
    ("addr", "record{city: string, zip: string}", False),
    ("count", "integer [0, 10]", True),
    ("list<count>", "list<number [0, 9]>", True),
]


@pytest.mark.parametrize(("s", "target", "expected"), TYPE_CONFORMANCE)
def test_type_conformance(s: str, target: str, expected: bool) -> None:
    assert conforms(p(s), p(target), D) is expected


@pytest.mark.parametrize("text", CANONICAL + NULLABLE_CANONICAL)
def test_every_type_conforms_to_itself_and_to_itself_or_null(text: str) -> None:
    d = {
        **D,
        "email/role": p("string"),
        "numbers": p("number"),
        "list/x": p("string"),
        "null_type": p("string"),
    }
    s = p(text)
    assert conforms(s, s, d)
    target = s if isinstance(s, Nullable) else Nullable(s)
    assert conforms(s, target, d)


# --- value-to-type conformance --------------------------------------------------------------

VALUE_CONFORMANCE: list[tuple[str, Any, bool]] = [
    # number: an exact int, or a finite value whose type is exactly float
    ("number", 0, True),
    ("number", 1, True),
    ("number", -1.5, True),
    ("number", 1.0, True),
    ("number", MAX, True),
    ("number", -MAX, True),
    ("number", 2.0**60, True),
    ("number", MAX + 1, False),
    ("number", True, False),
    ("number", math.nan, False),
    ("number", math.inf, False),
    ("number", "1", False),
    ("number", None, False),
    ("number", MyFloat(1.0), False),
    ("number", Decimal(1), False),
    ("number", Level.one, False),
    # integer: exactly int; an integral float never conforms
    ("integer", 1, True),
    ("integer", 0, True),
    ("integer", -MAX, True),
    ("integer", 1.0, False),
    ("integer", True, False),
    ("integer", MAX + 1, False),
    ("integer", Level.one, False),
    # intervals
    ("integer [0, 3]", 0, True),
    ("integer [0, 3]", 3, True),
    ("integer [0, 3]", 4, False),
    ("integer [0, 3]", -1, False),
    ("integer [0, 3]", 1.0, False),
    ("number [0, 1]", 0, True),
    ("number [0, 1]", 1, True),
    ("number [0, 1]", 0.5, True),
    ("number [0, 1]", -0.0, True),
    ("number [0, 1]", 70, False),
    ("number [0, 1]", 1.0000001, False),
    ("number [0, 1]", -1e-9, False),
    ("number [0, 1]", False, False),
    # strings and labels, by string data
    ("string", "", True),
    ("string", "x", True),
    ("string", Colour.red, True),
    ("string", MyStr("x"), True),
    ("string", "\ud800", False),
    ("string", b"x", False),
    ("string", 1, False),
    ("string", None, False),
    ('label{"a", "b"}', "a", True),
    ('label{"a", "b"}', Colour.a, True),
    ('label{"a", "b"}', MyStr("a"), True),
    ('label{"a", "b"}', "c", False),
    ('label{"a", "b"}', "A", False),
    ('label{"a", "b"}', 1, False),
    # boolean
    ("boolean", True, True),
    ("boolean", False, True),
    ("boolean", 1, False),
    ("boolean", 0, False),
    ("boolean", None, False),
    # lists: exactly list or tuple
    ("list<integer>", [], True),
    ("list<integer>", [1, 2], True),
    ("list<integer>", (1, 2), True),
    ("list<integer>", [1, 1.0], False),
    ("list<integer>", MyList([1]), False),
    ("list<integer>", "ab", False),
    ("list<integer>", {1}, False),
    ("list<list<string>>", [["a"], []], True),
    ("list<list<string>>", [["a"], [1]], False),
    # mappings: exactly dict, string keys by string data
    ("mapping<string, integer>", {}, True),
    ("mapping<string, integer>", {"a": 1}, True),
    ("mapping<string, integer>", {MyStr("a"): 1}, True),
    ("mapping<string, integer>", {"a": 1.5}, False),
    ("mapping<string, integer>", {1: 1}, False),
    ("mapping<string, integer>", {"\ud800": 1}, False),
    ("mapping<string, integer>", {"a": 1, SameData("a"): 2}, False),
    ("mapping<string, integer>", OrderedDict(a=1), False),
    ("mapping<string, integer>", MyDict(a=1), False),
    ("mapping<string, integer>", [("a", 1)], False),
    # A percent written 70 in a mapping of bars
    ("mapping<string, number [0, 1]>", {"yes": 0.9, "no": 70}, False),
    ("mapping<string, number [0, 1]>", {"yes": 0.9, "no": 0.7}, True),
    ("number [0, 1]", {"yes": 0.9}, False),
    # records: every declared field present; an absent field is not null
    ("record{a: integer, b: string or null}", {"a": 1, "b": None}, True),
    ("record{a: integer, b: string or null}", {"a": 1, "b": "x"}, True),
    ("record{a: integer, b: string or null}", {"a": 1}, False),
    ("record{a: integer, b: string or null}", {"a": None, "b": None}, False),
    ("record{a: integer, b: string or null}", {"a": 1, "b": 2}, False),
    ("record{a: integer, b: string or null}", {"a": 1, "b": None, "extra": object()}, True),
    ("record{a: integer, b: string or null}", {MyStr("a"): 1, "b": None}, True),
    ("record{a: integer}", {"a": 1, SameData("a"): 2}, False),
    ("record{a: integer}", {"a": 1, 2: 2}, False),
    ("record{a: integer}", MyDict(a=1), False),
    ("record{a: integer}", [1], False),
    ("record{}", {}, True),
    ("record{}", {"x": 1}, True),
    ("record{s: record{addr: record{city: string}}}", {"s": {"addr": {"city": "x"}}}, True),
    ("record{s: record{addr: record{city: string}}}", {"s": {"addr": {}}}, False),
    ("record{s: record{addr: record{city: string}}}", {"s": {"addr": None}}, False),
    ("record{s: record{city: string} or null}", {"s": None}, True),
    # null only where declared
    ("string or null", None, True),
    ("string or null", "x", True),
    ("string or null", 1, False),
    ("string", None, False),
    ("list<string> or null", None, True),
    ("list<string> or null", [None], False),
    # definitions resolved
    ("role", "a", True),
    ("role", "z", False),
    ("list<count>", [0, 9], True),
    ("list<count>", [10], False),
    ("record{home: addr}", {"home": {"city": "x"}}, True),
    ("record{home: addr}", {"home": {}}, False),
]


@pytest.mark.parametrize(("text", "value", "expected"), VALUE_CONFORMANCE, ids=repr)
def test_value_conformance(text: str, value: Any, expected: bool) -> None:
    for form in ("live", "document"):
        assert value_conforms(value, p(text), D, form=form) is expected, form


BYTES_LIVE: list[tuple[Any, bool]] = [
    (b"", True),
    (b"\x00\xff", True),
    (bytearray(b"x"), False),
    (memoryview(b"x"), False),
    (MyBytes(b"x"), False),
    ("AP8", False),
    (None, False),
]

BYTES_DOCUMENT: list[tuple[Any, bool]] = [
    ("", True),
    ("AP8", True),  # b"\x00\xff"
    ("AP-_", True),
    ("Zm9vYmFy", True),
    (MyStr("Zg"), True),
    ("AP9", False),  # nonzero trailing bits
    ("Zh", False),  # nonzero trailing bits
    ("AP8=", False),  # padding
    ("Zg==", False),
    ("A", False),  # length 1 mod 4
    ("AAAAA", False),
    ("AP+/", False),  # the standard alphabet
    ("AP 8", False),
    ("AP8\n", False),
    ("AP8é", False),
    (b"AP8", False),
    (None, False),
]


@pytest.mark.parametrize(("value", "expected"), BYTES_LIVE, ids=repr)
def test_bytes_live_is_python_bytes(value: Any, expected: bool) -> None:
    assert value_conforms(value, p("bytes"), {}, form="live") is expected


@pytest.mark.parametrize(("value", "expected"), BYTES_DOCUMENT, ids=repr)
def test_bytes_in_the_document_is_unpadded_base64url(value: Any, expected: bool) -> None:
    assert value_conforms(value, p("bytes"), {}, form="document") is expected


def test_bytes_conform_at_depth_by_form() -> None:
    rec = p("record{img: bytes, tags: list<bytes>}")
    assert value_conforms({"img": b"\x00", "tags": [b"a"]}, rec, {}, form="live")
    assert not value_conforms({"img": "AA", "tags": ["YQ"]}, rec, {}, form="live")
    assert value_conforms({"img": "AA", "tags": ["YQ"]}, rec, {}, form="document")
    assert not value_conforms({"img": b"\x00", "tags": [b"a"]}, rec, {}, form="document")


def test_the_form_is_live_or_document() -> None:
    with pytest.raises(ValueError, match="form"):
        value_conforms(1, p("integer"), {}, form="replay")  # pyright: ignore[reportArgumentType]


def test_an_answer_value_is_validated_by_answer_not_by_conformance() -> None:
    with pytest.raises(ValueError, match="answer"):
        value_conforms({"kind": "noul"}, p("answer"), {}, form="document")


# --- unpadded base64url ---------------------------------------------------

RFC4648 = [
    (b"", ""),
    (b"f", "Zg"),
    (b"fo", "Zm8"),
    (b"foo", "Zm9v"),
    (b"foob", "Zm9vYg"),
    (b"fooba", "Zm9vYmE"),
    (b"foobar", "Zm9vYmFy"),
    (b"\xfb\xff", "-_8"),
]


@pytest.mark.parametrize(("raw", "text"), RFC4648)
def test_base64url_vectors(raw: bytes, text: str) -> None:
    assert b64url_encode(raw) == text
    assert b64url_decode(text) == raw


def test_base64url_round_trips() -> None:
    rng = random.Random(4648)
    for n in range(64):
        raw = rng.randbytes(n)
        text = b64url_encode(raw)
        assert "=" not in text
        assert b64url_decode(text) == raw


@pytest.mark.parametrize("text", ["AP9", "Zh", "Zm9=", "Zg==", "A", "AP+/", " ", "Zg\n", "é"])
def test_base64url_decode_is_strict(text: str) -> None:
    assert b64url_decode(text) is None


def test_base64url_decode_takes_only_a_string() -> None:
    assert b64url_decode(b"Zg") is None  # pyright: ignore[reportArgumentType]


# --- the type of a dot-path read ------------------------------------------

STATE = "record{sender: record{address: string, name: string or null}, subject: string}"

READS: list[tuple[str, tuple[str, ...], str]] = [
    (STATE, (), STATE),
    (STATE, ("subject",), "string"),
    (STATE, ("sender",), "record{address: string, name: string or null}"),
    (STATE, ("sender", "address"), "string"),
    (STATE, ("sender", "name"), "string or null"),
    ("record{s: record{a: string} or null}", ("s", "a"), "string or null"),
    ("record{s: record{a: string or null} or null}", ("s", "a"), "string or null"),
    ("record{s: record{t: record{a: integer}} or null}", ("s", "t", "a"), "integer or null"),
    ("record{s: record{t: record{a: integer}} or null}", ("s", "t"), "record{a: integer} or null"),
    ("record{a: string} or null", ("a",), "string or null"),
    ("record{h: addr or null}", ("h", "city"), "string or null"),
    ("record{h: addr}", ("h", "city"), "string"),
    ("record{h: addr}", ("h",), "addr"),
    ("record{r: role}", ("r",), "role"),
]


@pytest.mark.parametrize(("text", "fields", "expected"), READS)
def test_the_type_of_a_dot_path_read(text: str, fields: tuple[str, ...], expected: str) -> None:
    read = read_type(p(text), fields, D)
    assert not isinstance(read, TypeFailure), read
    assert render(read) == expected


def test_a_read_through_a_named_type_that_admits_null_keeps_its_name() -> None:
    d = defs(opt="string or null")
    read = read_type(p("record{s: record{o: opt} or null}"), ("s", "o"), d)
    assert read == Named("opt")


@pytest.mark.parametrize(
    ("text", "fields", "word"),
    [
        (STATE, ("nope",), "nope"),
        (STATE, ("subject", "x"), "subject"),
        (STATE, ("sender", "address", "x"), "address"),
        ("record{l: list<record{a: string}>}", ("l", "a"), "l"),
        ("record{m: mapping<string, string>}", ("m", "k"), "m"),
        ("string", ("a",), "a"),
        ("record{r: role}", ("r", "a"), "r"),
    ],
)
def test_a_path_through_a_non_record_or_naming_no_field_fails(
    text: str, fields: tuple[str, ...], word: str
) -> None:
    read = read_type(p(text), fields, D)
    assert isinstance(read, TypeFailure)
    assert read.kind is FailureKind.ill_formed
    assert word in read.message


def test_the_failure_kinds_are_a_strenum() -> None:
    assert issubclass(FailureKind, enum.StrEnum)
    assert isinstance(AnswerType(), AnswerType)


def test_t_or_null_is_not_built_over_a_nullable_type() -> None:
    with pytest.raises(_Invalid):
        make_nullable(Nullable(Scalar("string")))


def test_resolve_refuses_an_undefined_name_or_a_cycle() -> None:
    with pytest.raises(ValueError, match="nope"):
        resolve(Named("nope"), {})
    with pytest.raises(ValueError, match="a"):
        resolve(Named("a"), {"a": Named("a")})


def test_a_definition_used_twice_is_checked_once() -> None:
    d = defs(q="answer")
    failures = check_declared(p("record{a: q, b: q}"), d, nullable=False)
    assert [f.kind for f in failures] == [FailureKind.not_declarable]


# --- parse_type never raises --------------------------------------------------------------


def test_a_bound_over_the_digit_limit_is_malformed() -> None:
    for text in ("number [0, " + "9" * 5000 + "]", "integer [-" + "1" * 4301 + ", 0]"):
        failure = parse_type(text, nullable=True)
        assert isinstance(failure, TypeFailure)
        assert failure.kind is FailureKind.malformed
        assert "digits" in failure.message


def test_a_long_fraction_or_exponent_is_not_a_raise() -> None:
    failure = parse_type("number [0, 0." + "1" * 5000 + "]", nullable=True)
    assert isinstance(failure, TypeFailure)
    assert failure.kind is FailureKind.not_canonical
    failure = parse_type("number [0, 1e" + "9" * 5000 + "]", nullable=True)
    assert isinstance(failure, TypeFailure)
    assert failure.kind is FailureKind.ill_formed


def test_an_inexact_bound_is_ill_formed_and_names_it() -> None:
    failure = parse_type("number [0, 9007199254740993]", nullable=True)
    assert isinstance(failure, TypeFailure)
    assert failure.kind is FailureKind.ill_formed
    assert "9007199254740993" in failure.message
    assert _Inexact("1") == _Inexact("1")


def test_an_inexact_bound_does_not_hide_a_later_syntax_error() -> None:
    failure = parse_type("number [0, 9007199254740993", nullable=True)
    assert isinstance(failure, TypeFailure)
    assert failure.kind is FailureKind.malformed


def test_a_type_nests_at_most_64_deep_in_its_string() -> None:
    for word, close in (("list<", ">"), ("mapping<string, ", ">"), ("record{a: ", "}")):
        deepest = word * 64 + "string" + close * 64
        assert type_depth(p(deepest), {}) == 64
        assert check_declared(p(deepest), {}, nullable=False) == []
        failure = parse_type(word * 65 + "string" + close * 65, nullable=True)
        assert isinstance(failure, TypeFailure)
        assert failure.kind is FailureKind.ill_formed
        assert "64 deep" in failure.message


@pytest.mark.parametrize("n", [65, 1_000, 100_000])
def test_deep_nesting_in_a_string_is_a_failure_not_a_raise(n: int) -> None:
    for text in ("list<" * n, "list<" * n + "string" + ">" * n, "record{a: " * n + "x"):
        failure = parse_type(text, nullable=True)
        assert isinstance(failure, TypeFailure)
        assert failure.kind is FailureKind.ill_formed


def test_parse_type_never_raises_on_random_strings() -> None:
    rng = random.Random(10)
    pieces = [
        "list<", "mapping<string, ", "record{", "a: ", ">", "}", ", ", "string", "integer",
        "number [", "0", "1e400", "9" * 20, "]", " or null", 'label{"a"', '"\\ud800"', "answer",
        "x/y", " ", "[", "{", "<", "\x00", "\u00e9",
    ]  # fmt: skip
    for _ in range(3_000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
        result = parse_type(text, nullable=rng.random() < 0.5)
        assert isinstance(result, TypeFailure) or render(result) == text


class LoudType(str):
    """A str subclass whose overrides parse_type must not reach."""

    __slots__ = ()

    def __getitem__(self, index: Any) -> Any:
        raise AssertionError("parse_type read the string through an override")

    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return 0


def test_parse_type_reads_a_str_subclass_by_its_string_data() -> None:
    parsed = parse_type(LoudType("list<string>"), nullable=False)
    assert parsed == ListOf(Scalar("string"))
    failure = parse_type(LoudType("list< string>"), nullable=False)
    assert isinstance(failure, TypeFailure)
    assert failure.kind is FailureKind.not_canonical
    assert type(failure.canonical) is str
    assert t.list_(LoudType("string")) == "list<string>"


# --- a type's depth with its definitions resolved ------------------------


def _chain(n: int, body: str = "list<{}>") -> dict[str, Type]:
    d = {"d0": p("string")}
    for i in range(1, n + 1):
        d[f"d{i}"] = p(body.format(f"d{i - 1}"))
    return d


def test_the_depth_counts_levels_not_names_or_null() -> None:
    d = defs(opt="list<string> or null", alias="opt", r="record{a: alias, b: integer}")
    assert type_depth(p("string"), d) == 0
    assert type_depth(p("alias"), d) == 1
    assert type_depth(p("r"), d) == 2
    assert type_depth(p("list<r> or null"), d) == 3
    assert type_depth(p("record{}"), d) == 1
    assert type_depth(p('mapping<string, label{"a"}>'), d) == 1
    assert type_depth(p("undefined/name"), d) == 0


def test_a_type_deeper_than_64_through_definitions_fails_check_10() -> None:
    d = _chain(64)
    assert type_depth(p("d64"), d) == 64
    assert check_declared(p("d64"), d, nullable=False) == []
    assert check_declared(p("d64 or null"), d, nullable=True) == []
    for text in ("list<d64>", "record{a: d64}", "list<d64> or null", "mapping<string, list<d63>>"):
        failures = check_declared(p(text), d, nullable=True)
        assert [f.kind for f in failures] == [FailureKind.ill_formed], text
        assert "64 deep" in failures[0].message


def test_a_long_chain_of_definitions_costs_no_recursion() -> None:
    deep = _chain(10_000)
    assert check_definitions(deep) == []
    assert type_depth(p("d10000"), deep) == 10_000
    failures = check_declared(p("d10000"), deep, nullable=False)
    assert [f.kind for f in failures] == [FailureKind.ill_formed]
    aliases = _chain(10_000, "{}")
    assert check_definitions(aliases) == []
    assert type_depth(p("d10000"), aliases) == 0
    assert check_declared(p("list<d10000>"), aliases, nullable=False) == []
    assert conforms(p("d10000"), p("string"), aliases)
    assert value_conforms(["x"], p("list<d10000>"), aliases, form="live")


def test_a_long_cycle_is_found_without_recursion() -> None:
    d = _chain(5_000, "{}")
    d["d0"] = p("d5000")
    failures = check_definitions(d)
    assert [f.kind for f in failures] == [FailureKind.cycle]
    assert len(failures[0].names) == 5_001


def test_type_depth_refuses_a_cycle() -> None:
    with pytest.raises(ValueError, match="cycle"):
        type_depth(p("a"), defs(a="list<b>", b="record{x: a}"))


# --- nullable definitions -------------------------------------


@pytest.mark.parametrize(
    ("body", "ok"),
    [
        ("string or null", True),
        ("opt", True),  # an alias of a nullable definition
        ("alias", True),
        ("record{a: opt}", True),
        ("list<opt>", False),
        ("mapping<string, alias>", False),
        ("opt or null", False),  # T or null over a nullable T, once resolved
        ("alias or null", False),
        ("list<string or null> or null", False),
    ],
)
def test_a_definition_body_may_be_nullable(body: str, ok: bool) -> None:
    # A definition's body is checked as a place null may stand whole.
    d = defs(opt="string or null", alias="opt")
    parsed = parse_type(body, nullable=True)
    if isinstance(parsed, TypeFailure):
        assert not ok
        return
    d["x"] = parsed
    assert check_definitions(d) == []
    failures = check_declared(parsed, d, nullable=True)
    assert (failures == []) is ok, failures
    if ok:
        assert check_declared(p("record{f: x}"), d, nullable=False) == []
        assert [f.kind for f in check_declared(p("x"), d, nullable=False)] == (
            [] if body.startswith("record") else [FailureKind.ill_formed]
        )


def test_answer_or_null_is_not_built() -> None:
    with pytest.raises(_Invalid, match="answer"):
        make_nullable(AnswerType())


class Opaque:
    def __repr__(self) -> str:
        raise RuntimeError("repr was called")


def test_a_bad_type_argument_whose_repr_raises_is_the_usual_error() -> None:
    with pytest.raises(ValueError, match=r"^number takes both bounds or neither, not lo=<Opaque>"):
        t.number(Opaque(), None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"hi=integer of 16610 bits$"):
        t.integer(None, 10**5000)
    with pytest.raises(TypeError, match=r"^a type is its canonical string, not <Opaque>$"):
        t.list_(Opaque())  # type: ignore[arg-type]


class Liar:
    """Claims to be a str through __class__."""

    @property
    def __class__(self) -> type:  # pyright: ignore[reportIncompatibleMethodOverride]
        return str


class ClassRaises:
    @property
    def __class__(self) -> type:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise RuntimeError("__class__ was read")


@pytest.mark.parametrize("make", [Liar, ClassRaises], ids=["liar", "raises"])
def test_a_type_argument_is_judged_by_its_type(make: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match="canonical string"):
        t.list_(make())  # type: ignore[arg-type]


def test_a_walk_cut_short_by_a_name_met_twice_is_not_kept_as_clean() -> None:
    from jurat._types import DeclaredMemo, check_declared

    def parsed(text: str) -> Any:
        return parse_type(text, nullable=True)

    defs = {"dd": parsed("integer or null"), "ee": parsed("list<dd>")}
    memo = DeclaredMemo()
    assert (
        len(
            check_declared(
                parsed("record{a: list<dd>, b: list<ee>}"), defs, nullable=True, memo=memo
            )
        )
        == 1
    )
    alone = check_declared(parsed("list<ee>"), defs, nullable=True)
    assert (
        len(check_declared(parsed("list<ee>"), defs, nullable=True, memo=memo)) == len(alone) == 1
    )


class _ClassRaises:
    """An object whose ``__class__`` raises, as ``isinstance`` would read it."""

    @property
    def __class__(self) -> type:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        raise RuntimeError("__class__ was read")


def test_a_value_of_a_raising_class_is_judged_by_its_type() -> None:
    odd: Any = _ClassRaises()
    failure = parse_type(odd, nullable=True)
    assert isinstance(failure, TypeFailure)
    assert failure.message == "a type of _ClassRaises is not a string"
    assert b64url_decode(odd) is None
    assert not value_conforms(odd, p("bytes"), {}, form="document")
    with pytest.raises(TypeError, match=r"^a label value is a string, not _ClassRaises$"):
        t.label(odd)
    with pytest.raises(ValueError, match=r"^form <_ClassRaises> is not live or document$"):
        value_conforms(1, p("integer"), {}, form=odd)


def test_a_form_is_read_by_its_string_data() -> None:
    class Odd(str):
        __slots__ = ()

        def __eq__(self, other: object) -> bool:
            raise AssertionError("the form was compared")

        __hash__ = str.__hash__

    assert value_conforms(1, p("integer"), {}, form=Odd("live"))  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=r"^form 'replay' is not live or document$"):
        value_conforms(1, p("integer"), {}, form=Odd("replay"))  # pyright: ignore[reportArgumentType]


def test_a_huge_bound_is_named_by_its_bit_length() -> None:
    with pytest.raises(ValueError, match=r"^bound integer of 16610 bits is outside the value"):
        t.number(0, 10**5000)


def test_named_lists_each_definition_once_in_order_of_appearance() -> None:
    assert named(p("record{a: x, b: list<y>, c: x, d: mapping<string, y>, e: z}")) == [
        "x",
        "y",
        "z",
    ]


def test_a_long_form_is_clipped_in_its_message() -> None:
    says = r"^form 'f{64}'\.\.\. \(1000000 code points\) is not live or document$"
    with pytest.raises(ValueError, match=says):
        value_conforms(1, p("integer"), {}, form="f" * 10**6)  # pyright: ignore[reportArgumentType]


def test_the_bytes_walk_visits_each_definition_and_shared_type_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jurat._types as types_module

    depth = 20
    bodies = {"d0": p("integer")}
    for i in range(1, depth + 1):
        bodies[f"d{i}"] = p(f"record{{a: d{i - 1}, b: d{i - 1}}}")
    visits = [0]
    children, resolve_ = types_module._children, types_module.resolve

    def counted_children(t: Type) -> list[Type]:
        visits[0] += 1
        return children(t)

    def counted_resolve(t: Type, definitions: Any) -> Type:
        visits[0] += 1
        return resolve_(t, definitions)

    monkeypatch.setattr(types_module, "_children", counted_children)
    monkeypatch.setattr(types_module, "resolve", counted_resolve)
    assert not types_module._holds_bytes(Named(f"d{depth}"), bodies)
    assert visits[0] <= 2 * depth
    shared = bodies["d0"]
    for _ in range(depth):
        shared = types_module.RecordOf((("a", shared), ("b", shared)))
    visits[0] = 0
    assert not types_module._holds_bytes(shared, {})
    assert visits[0] <= 2 * depth
    assert types_module._holds_bytes(p("record{a: list<mapping<string, bytes>>} or null"), {})


@pytest.mark.parametrize(
    ("value", "expected"),
    [("a", True), ("m", True), ("z", True), ("", False), ("b", False), ("zz", False)],
)
def test_a_label_value_is_found_among_its_values_in_code_point_order(
    value: str, expected: bool
) -> None:
    assert value_conforms(value, p('label{"a", "m", "z"}'), {}, form="live") is expected


def test_a_bytes_values_text_is_counted_before_it_is_built(
    small_bounds: SmallBounds, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jurat._typed as typed_module

    bytes_ = p("bytes")
    assert typed_module.typed(b"\xff" * 192, bytes_) == ("_" * 256, b"\xff" * 192)  # 2^8 exactly

    def unbuilt(data: bytes) -> str:
        raise AssertionError("the text was built")

    monkeypatch.setattr(typed_module, "b64url_encode", unbuilt)
    assert typed_module.typed(b"\xff" * 193, bytes_) is None  # 258 code points: past 2^8
