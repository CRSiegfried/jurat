# pyright: reportPrivateUsage=false
"""The ref grammar, grounded brackets, declared names and code-point order."""

from typing import Any

import pytest

from jurat._refs import (
    _RefError,
    code_point_sorted,
    declared_ref,
    is_declared_name,
    is_path_ancestor,
    is_ref,
    is_segment,
    parse_ref,
)

VALID: list[tuple[str, str, str, str | None, tuple[str, ...]]] = [
    ("q:a", "q", "a", None, ()),
    ("t:state", "t", "state", None, ()),
    ("d:intent", "d", "intent", None, ()),
    ("t:state.subject", "t", "state", None, ("subject",)),
    ("t:state.sender.address", "t", "state", None, ("sender", "address")),
    ("q:email/folder_role", "q", "email/folder_role", None, ()),
    ("d:a1/b_2/c3", "d", "a1/b_2/c3", None, ()),
    ('q:email/folder_role["Dana"]', "q", "email/folder_role", "Dana", ()),
    ('t:x["a"].f', "t", "x", "a", ("f",)),
    ('t:x["a"].f.g', "t", "x", "a", ("f", "g")),
    ('q:x[""]', "q", "x", "", ()),
    ('q:x["]"]', "q", "x", "]", ()),
    ('q:x["a\\"b"]', "q", "x", 'a"b', ()),
    ('q:x["\\\\"]', "q", "x", "\\", ()),
    ('q:x["\\n\\u0001"]', "q", "x", "\n\x01", ()),
    ('q:x["\x7f"]', "q", "x", "\x7f", ()),
    ('q:x["\U0001f600"]', "q", "x", "\U0001f600", ()),
    ('q:x["\u2028"]', "q", "x", "\u2028", ()),
    ('q:x["/"]', "q", "x", "/", ()),
    ('q:x["hmac-sha256:' + "0" * 64 + '"]', "q", "x", "hmac-sha256:" + "0" * 64, ()),
]


@pytest.mark.parametrize(("ref", "level", "name", "candidate", "fields"), VALID)
def test_valid_refs(
    ref: str, level: str, name: str, candidate: str | None, fields: tuple[str, ...]
) -> None:
    p = parse_ref(ref)
    assert p.level == level
    assert p.name == name
    assert p.candidate == candidate
    assert p.fields == fields
    assert is_ref(ref)


INVALID = [
    "",
    "q:",
    "a",
    "p:t",
    "x:a",
    "Q:a",
    "q a",
    "q::a",
    "q:A",
    "q:1a",
    "q:_a",
    "q:a-b",
    "q:a/",
    "q:/a",
    "q:a//b",
    "q:a/B",
    "q:a.",
    "q:a..b",
    "q:a.B",
    "q:a.1",
    "q:a.b/c",
    "q:a ",
    " q:a",
    "q:a\n",
    "q:\u00e9",
    "q:a\U0001f600",
    "state",
    "rule:1",
    "step:2",
    "policy",
    "codec",
    # grounded brackets
    "q:x[]",
    'q:x["a"',
    'q:x["a"]]',
    'q:x["a"]["b"]',
    "q:x[0]",
    "q:x[null]",
    'q:x[["a"]]',
    'q:x[ "a"]',
    'q:x["a" ]',
    "q:x['a']",
    'q:x["\\u0061"]',
    'q:x["\\/"]',
    'q:x["\\u00e9"]',
    'q:x["\\U0001F600"]',
    'q:x["\\ud83d\\ude00"]',
    'q:x["\\ud800"]',
    'q:x["\x01"]',
    'q:x["\\u000A"]',
    'q:x["a"]f',
    'q:x["a"].',
    'q:x.f["a"]',
    'q:x["a"]/b',
]


@pytest.mark.parametrize("ref", INVALID, ids=repr)
def test_invalid_refs(ref: str) -> None:
    with pytest.raises(_RefError):
        parse_ref(ref)
    assert not is_ref(ref)


def test_the_ref_error_is_a_value_error() -> None:
    assert issubclass(_RefError, ValueError)


def test_a_non_str_is_not_a_ref() -> None:
    assert not is_ref(1)
    assert not is_ref(None)
    with pytest.raises(_RefError):
        parse_ref(b"q:a")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("ref", "declared"),
    [
        ("q:a", "q:a"),
        ("t:state.subject", "t:state.subject"),
        ('q:email/folder_role["Dana"]', "q:email/folder_role"),
        ('t:x["a"].f.g', "t:x.f.g"),
        ('q:x["]"]', "q:x"),
    ],
)
def test_declared_ref_strips_the_bracket_and_keeps_the_field_path(ref: str, declared: str) -> None:
    assert declared_ref(ref) == declared


def test_declared_ref_refuses_a_malformed_ref() -> None:
    with pytest.raises(_RefError):
        declared_ref('q:x["a"')


@pytest.mark.parametrize("name", ["a", "a1", "a_b", "list/count/cap", "mail", "z9/y_"])
def test_declared_names(name: str) -> None:
    assert is_declared_name(name)


@pytest.mark.parametrize(
    "name", ["", "A", "1a", "_a", "a/", "/a", "a//b", "a.b", "q:a", "rule:1", "a-b", "\u00e9", 3]
)
def test_not_declared_names(name: object) -> None:
    assert not is_declared_name(name)


def test_a_segment_has_no_slash() -> None:
    assert is_segment("cap")
    assert is_segment("a_1")
    assert not is_segment("list/cap")
    assert not is_segment("")
    assert not is_segment("Cap")
    assert not is_segment(None)


@pytest.mark.parametrize(
    ("node", "path", "expected"),
    [
        ("list", "list/count", True),
        ("list", "list/count/cap", True),
        ("list/count", "list/count/cap", True),
        ("list", "list", False),
        ("list", "lists/count", False),
        ("list/c", "list/count", False),
        ("list/count/cap", "list/count", False),
    ],
)
def test_path_ancestor(node: str, path: str, expected: bool) -> None:
    assert is_path_ancestor(node, path) is expected


def test_code_point_order_is_not_utf16_order() -> None:
    refs = ['q:x["\ue000"]', 'q:x["\U0001f600"]', "q:b", "d:z", "t:a"]
    assert code_point_sorted(refs) == [
        "d:z",
        "q:b",
        'q:x["\ue000"]',
        'q:x["\U0001f600"]',
        "t:a",
    ]


class Loud(str):
    """A str subclass whose overrides parse_ref must not reach."""

    __slots__ = ()

    def __getitem__(self, index: Any) -> Any:
        raise AssertionError("parse_ref read the ref through an override")

    def __len__(self) -> int:
        raise AssertionError("parse_ref read the ref through an override")

    def __str__(self) -> str:
        return "q:other"


def test_a_str_subclass_is_parsed_by_its_string_data() -> None:
    p = parse_ref(Loud('t:mail/inbox["x"].sender'))
    assert (p.level, p.name, p.candidate, p.fields) == ("t", "mail/inbox", "x", ("sender",))
    assert type(p.name) is str
    assert declared_ref(Loud("q:a.b")) == "q:a.b"
    with pytest.raises(_RefError):
        parse_ref(Loud("x:a"))


def test_a_bracket_over_the_integer_digit_limit_is_a_ref_error() -> None:
    ref = "q:x[" + "1" * 5000 + "]"
    assert not is_ref(ref)
    with pytest.raises(_RefError, match="not a JSON string"):
        parse_ref(ref)


@pytest.mark.parametrize("opener", ["[", "{"])
def test_a_bracket_nesting_past_the_recursion_limit_is_a_ref_error(opener: str) -> None:
    ref = "d:a[" + opener * 100_000
    assert not is_ref(ref)
    with pytest.raises(_RefError, match="not a JSON string"):
        parse_ref(ref)
