# pyright: reportPrivateUsage=false
"""The canonical encoding: RFC 8785 bytes, hashes, the per-turn key and the keyed digest."""

import enum
import hashlib
import hmac
import json
import math
import random
import re
import struct
import time
from decimal import Decimal
from typing import Any

import pytest
import rfc8785

from counting import SmallBounds
from jurat._canonical import (
    _CanonicalError,
    _DecodeError,
    _expand32,
    decode,
    document_depth,
    encode,
    hash_value,
    integer_literal,
    keyed_digest,
    record_hash,
    sha256_ref,
    turn_key,
)

MAX = 2**53 - 1
SHA = re.compile(r"sha256:[0-9a-f]{64}")
HMAC = re.compile(r"hmac-sha256:[0-9a-f]{64}")


class Colour(enum.StrEnum):
    red = "red"
    astral = "\U0001f600"


def _listify(value: Any) -> Any:
    """The oracle's input: tuples as lists, StrEnum members as their string data."""
    if isinstance(value, enum.StrEnum):
        return str.__str__(value)
    if isinstance(value, (list, tuple)):
        return [_listify(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, dict):
        return {_listify(k): _listify(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    return value


# --- hand vectors -------------------------------------------------------------------------

HAND: list[tuple[Any, bytes]] = [
    (None, b"null"),
    (True, b"true"),
    (False, b"false"),
    (0, b"0"),
    (1, b"1"),
    (-1, b"-1"),
    (MAX, b"9007199254740991"),
    (-MAX, b"-9007199254740991"),
    (1.0, b"1"),
    (-0.0, b"0"),
    (0.0, b"0"),
    (1.5, b"1.5"),
    (-1.5, b"-1.5"),
    (0.1, b"0.1"),
    (1e16, b"10000000000000000"),
    (1e20, b"100000000000000000000"),
    (1e21, b"1e+21"),
    (1.5e21, b"1.5e+21"),
    (1e-6, b"0.000001"),
    (1e-7, b"1e-7"),
    (1.25e-7, b"1.25e-7"),
    (123.456, b"123.456"),
    (5e-324, b"5e-324"),
    (1.7976931348623157e308, b"1.7976931348623157e+308"),
    (float(2**60), b"1152921504606847000"),
    (float(2**53), b"9007199254740992"),
    (333333333.3333333, b"333333333.3333333"),
    ("", b'""'),
    ("a", b'"a"'),
    ('"\\', b'"\\"\\\\"'),
    ("\b\t\n\f\r", b'"\\b\\t\\n\\f\\r"'),
    ("\x00\x01\x1f", b'"\\u0000\\u0001\\u001f"'),
    ("\x0b", b'"\\u000b"'),
    ("\x7f", b'"\x7f"'),
    ("\u2028\u2029", '"\u2028\u2029"'.encode()),
    ("/", b'"/"'),
    ("\u20ac\U0001f600", '"\u20ac\U0001f600"'.encode()),
    ([], b"[]"),
    ((), b"[]"),
    ((1, (2, "x")), b'[1,[2,"x"]]'),
    ({}, b"{}"),
    ({"b": 1, "a": 2}, b'{"a":2,"b":1}'),
    (Colour.red, b'"red"'),
    ({Colour.red: Colour.astral}, '{"red":"\U0001f600"}'.encode()),
    # U+E000 is one UTF-16 code unit, 0xE000; U+1F600 is 0xD83D 0xDE00, so it sorts first.
    ({"\ue000": 1, "\U0001f600": 2}, '{"\U0001f600":2,"\ue000":1}'.encode()),
    ({"10": 1, "9": 2, "a": 3, "A": 4, "": 5}, b'{"":5,"10":1,"9":2,"A":4,"a":3}'),
]


@pytest.mark.parametrize(("value", "expected"), HAND, ids=lambda x: repr(x)[:40])
def test_hand_vectors(value: Any, expected: bytes) -> None:
    assert encode(value) == expected


@pytest.mark.parametrize(("value", "expected"), HAND, ids=lambda x: repr(x)[:40])
def test_hand_vectors_match_the_oracle(value: Any, expected: bytes) -> None:
    assert rfc8785.dumps(_listify(value)) == expected


# --- RFC 8785's own vectors, independent of the oracle -------------------------------------

# Appendix B: IEEE 754 doubles, as hex, and their serialization; None where none exists.
RFC8785_NUMBERS: list[tuple[str, str | None]] = [
    ("0000000000000000", "0"),
    ("8000000000000000", "0"),
    ("0000000000000001", "5e-324"),
    ("8000000000000001", "-5e-324"),
    ("7fefffffffffffff", "1.7976931348623157e+308"),
    ("ffefffffffffffff", "-1.7976931348623157e+308"),
    ("4340000000000000", "9007199254740992"),
    ("c340000000000000", "-9007199254740992"),
    ("4430000000000000", "295147905179352830000"),
    ("7fffffffffffffff", None),
    ("7ff0000000000000", None),
    ("44b52d02c7e14af5", "9.999999999999997e+22"),
    ("44b52d02c7e14af6", "1e+23"),
    ("44b52d02c7e14af7", "1.0000000000000001e+23"),
    ("444b1ae4d6e2ef4e", "999999999999999700000"),
    ("444b1ae4d6e2ef4f", "999999999999999900000"),
    ("444b1ae4d6e2ef50", "1e+21"),
    ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
    ("3eb0c6f7a0b5ed8d", "0.000001"),
    ("41b3de4355555553", "333333333.3333332"),
    ("41b3de4355555554", "333333333.33333325"),
    ("41b3de4355555555", "333333333.3333333"),
    ("41b3de4355555556", "333333333.3333334"),
    ("41b3de4355555557", "333333333.33333343"),
    ("becbf647612f3696", "-0.0000033333333333333333"),
    ("43143ff3c1cb0959", "1424953923781206.2"),
]


def _double(hex_bits: str) -> float:
    (f,) = struct.unpack(">d", bytes.fromhex(hex_bits))
    return f


@pytest.mark.parametrize(("hex_bits", "expected"), RFC8785_NUMBERS)
def test_rfc8785_appendix_b_numbers(hex_bits: str, expected: str | None) -> None:
    f = _double(hex_bits)
    if expected is None:
        with pytest.raises(_CanonicalError):
            encode(f)
    else:
        assert encode(f) == expected.encode("ascii")
        assert decode(encode(f)) == f


# Section 3.2.3: property names sorted by their UTF-16 code units.
RFC8785_SORTING_INPUT = {
    "\u20ac": "Euro Sign",
    "\r": "Carriage Return",
    "\ufb33": "Hebrew Letter Dalet With Dagesh",
    "1": "One",
    "\U0001f600": "Emoji: Grinning Face",
    "\u0080": "Control",
    "\u00f6": "Latin Small Letter O With Diaeresis",
}
RFC8785_SORTING_OUTPUT = (
    '{"\\r":"Carriage Return","1":"One","\u0080":"Control",'
    '"\u00f6":"Latin Small Letter O With Diaeresis","\u20ac":"Euro Sign",'
    '"\U0001f600":"Emoji: Grinning Face","\ufb33":"Hebrew Letter Dalet With Dagesh"}'
).encode()


def test_rfc8785_section_3_2_3_sorting() -> None:
    assert encode(RFC8785_SORTING_INPUT) == RFC8785_SORTING_OUTPUT


def test_a_boolean_is_never_an_integer() -> None:
    assert encode(True) == b"true"
    assert encode(1) == b"1"
    assert encode([True, 1, False, 0]) == b"[true,1,false,0]"


def test_the_integer_limit_is_inclusive() -> None:
    assert encode(MAX) == b"9007199254740991"
    with pytest.raises(_CanonicalError):
        encode(MAX + 1)
    with pytest.raises(_CanonicalError):
        encode(-MAX - 1)
    # The same magnitude as an integral float is a number, and in the domain.
    assert encode(float(MAX + 1)) == b"9007199254740992"


def test_keys_sort_by_utf16_code_units_not_code_points() -> None:
    keys = ["\ue000", "\U0001f600", "\uffff", "\U00010000", "z"]
    out = json.loads(encode(dict.fromkeys(keys, 0)))
    assert list(out) == ["z", "\U00010000", "\U0001f600", "\ue000", "\uffff"]
    assert sorted(keys) != list(out)


# --- a generated corpus against the oracle ------------------------------------------------

_KEY_CHARS = 'aAzZ09_ \u00e9\u2028\x7f\x00\x1f"\\\ue000\uffff\U00010000\U0001f600\U0010ffff'


def _gen_str(rng: random.Random) -> str:
    return "".join(rng.choice(_KEY_CHARS) for _ in range(rng.randint(0, 6)))


def _gen_float(rng: random.Random) -> float:
    while True:
        (f,) = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))
        if math.isfinite(f):
            return f


def _gen(rng: random.Random, depth: int) -> Any:
    pick = rng.randint(0, 9 if depth < 4 else 5)
    if pick == 0:
        return rng.choice([None, True, False])
    if pick == 1:
        return rng.randint(-MAX, MAX)
    if pick == 2:
        return _gen_float(rng)
    if pick == 3:
        return rng.choice([0.0, -0.0, 1e16, 1e-7, 1e21, 5e-324, float(2**60), 1.0, 0.5])
    if pick == 4:
        return _gen_str(rng)
    if pick == 5:
        return rng.choice([Colour.red, Colour.astral])
    if pick == 6:
        return [_gen(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    if pick == 7:
        return tuple(_gen(rng, depth + 1) for _ in range(rng.randint(0, 4)))
    return {_gen_str(rng): _gen(rng, depth + 1) for _ in range(rng.randint(0, 5))}


CORPUS = [_gen(random.Random(seed), 0) for seed in range(400)]


@pytest.mark.parametrize("value", CORPUS)
def test_corpus_matches_the_oracle(value: Any) -> None:
    assert encode(value) == rfc8785.dumps(_listify(value))


@pytest.mark.parametrize("value", CORPUS)
def test_corpus_reencodes_its_parse(value: Any) -> None:
    data = encode(value)
    assert encode(decode(data)) == data


def test_random_doubles_match_the_oracle() -> None:
    rng = random.Random(8785)
    for _ in range(20_000):
        f = _gen_float(rng)
        assert encode(f) == rfc8785.dumps(f), f


# --- refusals -----------------------------------------------------------------------------


class MyStr(str):
    __slots__ = ()


class MyInt(int):
    __slots__ = ()


class Shouty(str):
    """A subclass whose overrides the encoder must not reach."""

    __slots__ = ()

    def __str__(self) -> str:
        return "LOUD"

    def __lt__(self, other: object) -> bool:
        return not str.__lt__(self, str(other))

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        raise AssertionError("the encoder read the string through an override")


class SameData(str):
    """A key unequal to a plain key with the same string data."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


class Plain(enum.Enum):
    a = "a"


REFUSED: list[Any] = [
    math.nan,
    math.inf,
    -math.inf,
    "\ud800",
    "x\udc00",
    {"\ud83d": 1},
    {1: "a"},
    {None: 1},
    b"bytes",
    Decimal("1"),
    MAX + 1,
    -(2**53),
    10**30,
    {1, 2},
    object(),
    MyInt(1),
    {"a": SameData("x"), SameData("a"): 1},
    Plain.a,
    [1, [2, [math.nan]]],
]


@pytest.mark.parametrize("value", REFUSED, ids=repr)
def test_refusals(value: Any) -> None:
    with pytest.raises(_CanonicalError):
        encode(value)


def test_the_refusal_is_a_value_error() -> None:
    assert issubclass(_CanonicalError, ValueError)


def test_a_cycle_is_refused() -> None:
    a: list[Any] = []
    a.append(a)
    with pytest.raises(_CanonicalError):
        encode(a)
    d: dict[str, Any] = {}
    d["x"] = [d]
    with pytest.raises(_CanonicalError):
        encode(d)


def test_deep_recursion_is_refused() -> None:
    v: Any = 0
    for _ in range(100_000):
        v = [v]
    with pytest.raises(_CanonicalError, match="64 deep"):
        encode(v)


def _nest(depth: int) -> Any:
    v: Any = 0
    for i in range(depth):
        v = [v] if i % 2 else {"k": v}
    return v


def test_a_value_nesting_64_deep_encodes_and_65_is_refused() -> None:
    data = encode(_nest(64))
    assert data == rfc8785.dumps(_nest(64))
    assert decode(data) == _nest(64)
    with pytest.raises(_CanonicalError, match="64 deep"):
        encode(_nest(65))


def test_a_document_nesting_more_than_128_deep_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="deep"):
        encode(_nest(129))


def test_a_strenum_encodes_by_its_string_data() -> None:
    assert encode(Colour.red) == encode("red")
    assert encode({Colour.red: [Colour.red]}) == b'{"red":["red"]}'


def test_a_str_subclass_encodes_by_its_string_data() -> None:
    assert encode(MyStr("x")) == b'"x"'
    assert encode(Shouty("quiet")) == b'"quiet"'
    assert encode({Shouty("b"): Shouty("v"), "a": 1}) == b'{"a":1,"b":"v"}'


def test_keys_are_sorted_by_their_string_data() -> None:
    # Shouty's __str__ and __lt__ would put "b" first if the encoder read them.
    assert encode({Shouty("b"): 1, Shouty("a"): 2}) == b'{"a":2,"b":1}'


def test_two_keys_with_the_same_string_data_are_refused() -> None:
    with pytest.raises(_CanonicalError, match="string data"):
        encode({"a": 1, SameData("a"): 2})


# --- the decoder --------------------------------------------------------------------------


def test_decode_reads_canonical_bytes() -> None:
    assert decode(b'{"a":[1,1.5,"x",null,true]}') == {"a": [1, 1.5, "x", None, True]}


def test_decode_turns_large_integers_into_floats() -> None:
    v = decode(
        b"[9007199254740991,9007199254740992,-9007199254740992,9007199254740994,"
        b"1152921504606847000,1152921504606846976,1180591620717411303424]"
    )
    assert v == [MAX, 2.0**53, -(2.0**53), 2.0**53 + 2, 2.0**60, 2.0**60, 2.0**70]
    assert [type(x) for x in v] == [int, *[float] * 6]


@pytest.mark.parametrize(
    "text",
    [
        b"9007199254740993",
        b"-9007199254740993",
        b"[1152921504606846977]",
        b"1152921504606847001",
        b"100000000000000000000000",  # 1e23 is not a double; its nearest is 9.999999999999999e22
        b"1" + b"0" * 400,
    ],
)
def test_decode_refuses_an_integer_that_names_no_double_exactly(text: bytes) -> None:
    with pytest.raises(_DecodeError, match=r"exactly|finite"):
        decode(text)


def test_decode_refuses_an_integer_over_the_digit_limit() -> None:
    with pytest.raises(_DecodeError, match="too long") as info:
        decode(b"1" * 5000)
    assert isinstance(info.value.__cause__, ValueError)


@pytest.mark.parametrize("text", [b"1e400", b"-1e400", b"[1.5e999]", b'{"a":1E309}'])
def test_decode_refuses_a_number_whose_double_is_not_finite(text: bytes) -> None:
    with pytest.raises(_DecodeError, match="finite"):
        decode(text)


def test_decode_keeps_a_finite_number() -> None:
    assert decode(b"[1e308,-1.7976931348623157e+308,5e-324,1e-400,0.1]") == [
        1e308,
        -1.7976931348623157e308,
        5e-324,
        0.0,
        0.1,
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0),
        ("-0", 0),
        ("9007199254740991", MAX),
        ("-9007199254740991", -MAX),
        ("9007199254740992", 2.0**53),
        ("1152921504606847000", 2.0**60),
        ("9007199254740993", None),
        ("1" + "0" * 400, None),
    ],
)
def test_integer_literal(text: str, expected: object) -> None:
    got = integer_literal(text)
    assert got == expected
    assert type(got) is type(expected)


def test_canonical_bytes_of_a_large_double_decode_to_it() -> None:
    rng = random.Random(53)
    for _ in range(5_000):
        f = float(rng.randrange(2**53, 2**70)) * rng.choice([1, -1])
        assert decode(encode(f)) == f


@pytest.mark.parametrize("value", [float(2**60), float(2**53), 1e16, 1e21, -0.0, 1e-7, 5e-324])
def test_decode_then_encode_round_trips(value: float) -> None:
    data = encode(value)
    assert encode(decode(data)) == data


@pytest.mark.parametrize("text", [b"NaN", b"Infinity", b"-Infinity", b"[NaN]"])
def test_decode_refuses_non_finite_constants(text: bytes) -> None:
    with pytest.raises(_DecodeError):
        decode(text)


@pytest.mark.parametrize("text", [b'{"a":1,"a":2}', b'{"x":{"a":1,"a":1}}', b'[{"b":0,"b":0}]'])
def test_decode_refuses_a_duplicate_key(text: bytes) -> None:
    with pytest.raises(_DecodeError):
        decode(text)


@pytest.mark.parametrize("text", [b"", b"{", b"\xff", b"\xef\xbb\xbf{}", "{}".encode("utf-16")])
def test_decode_refuses_bytes_that_are_not_utf8_json(text: bytes) -> None:
    with pytest.raises(_DecodeError) as info:
        decode(text)
    assert info.value.__cause__ is not None


def test_decode_refuses_deep_nesting() -> None:
    with pytest.raises(_DecodeError, match="128 deep"):
        decode(b"[" * 100_000 + b"]" * 100_000)


def test_decode_refuses_a_document_nesting_more_than_128_deep() -> None:
    assert decode(b"[" * 128 + b"]" * 128) is not None
    assert decode(b'{"a":' * 128 + b"0" + b"}" * 128) is not None
    for text in (b"[" * 129 + b"]" * 129, b'{"a":' * 129 + b"0" + b"}" * 129):
        with pytest.raises(_DecodeError, match="128 deep") as info:
            decode(text)
        assert info.value.__cause__ is None


def test_the_depth_is_counted_before_parsing() -> None:
    # Unbalanced and unparseable, yet refused for its depth: the count runs first.
    with pytest.raises(_DecodeError, match="128 deep"):
        decode(b"[" * 129 + b"\xff")


@pytest.mark.parametrize(
    ("text", "depth"),
    [
        (b"0", 0),
        (b'"[[["', 0),
        (b"[]", 1),
        (b'[{"a":[]},[[]]]', 3),
        (rb'["]", "[", "\\", "\"[[", "{"]', 1),
        (rb'{"\\":[], "[":{}}', 2),
        ('["\u00e9[", ["\U0001f600"]]'.encode(), 2),
    ],
)
def test_document_depth_counts_brackets_outside_strings(text: bytes, depth: int) -> None:
    assert document_depth(text) == depth


def test_brackets_inside_strings_do_not_count_towards_the_bound() -> None:
    text = b'["' + b"[" * 1000 + b'"]'
    assert decode(text) == ["[" * 1000]


def _depth_by_removing_strings(data: bytes) -> int:
    """The depth counted by deleting every string first: a slow but plain reference."""
    depth = deepest = 0
    for c in re.sub(rb'"[^"\\]*(?:\\.[^"\\]*)*"', b"", data, flags=re.DOTALL):
        if c in b"[{":
            depth += 1
            deepest = max(deepest, depth)
        elif c in b"]}":
            depth -= 1
    return deepest


_CHARS = '[]{}"\\ a\u00e9\n\x01\U0001f600'


def _random_json(rng: random.Random, depth: int) -> object:
    pick = rng.randrange(6 if depth < 8 else 3)
    if pick == 0:
        return rng.choice([None, True, False, 0, -1.5, 2**53 - 1])
    if pick in (1, 2):
        return "".join(rng.choice(_CHARS) for _ in range(rng.randrange(6)))
    if pick in (3, 4):
        return [_random_json(rng, depth + 1) for _ in range(rng.randrange(4))]
    return {
        str(i) + rng.choice(_CHARS): _random_json(rng, depth + 1) for i in range(rng.randrange(4))
    }


def test_document_depth_matches_the_string_removing_count_on_valid_documents() -> None:
    rng = random.Random(8785)
    corpus = [
        b"[" * 128 + b"]" * 128,
        b'{"a":' * 60 + b'"[{\\""' + b"}" * 60,
        rb'["\\", "\\\"", "\"[", "]\\", {"[": ["{"]}]',
    ]
    for _ in range(2000):
        value = _random_json(rng, 0)
        corpus.append(json.dumps(value, ensure_ascii=rng.random() < 0.5).encode())
    for data in corpus:
        json.loads(data)  # every member is valid JSON
        assert document_depth(data) == _depth_by_removing_strings(data), data


def test_an_unterminated_string_of_many_escaped_quotes_is_refused_quickly() -> None:
    data = b'"' + b'\\"' * 200_000
    start = time.perf_counter()
    with pytest.raises(_DecodeError):
        decode(data)
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize(
    "data",
    [b'"' + b"[" * 129, b'["' + b"[" * 129, b"[" * 64 + b'"\\', b'"\\"' + b"[" * 200],
)
def test_bytes_ending_in_an_unterminated_string_are_refused(data: bytes) -> None:
    with pytest.raises(_DecodeError):
        decode(data)


def test_the_decode_error_is_a_value_error() -> None:
    assert issubclass(_DecodeError, ValueError)


# --- hashes -------------------------------------------------------------------------------


def test_sha256_ref_form() -> None:
    ref = sha256_ref(b"")
    assert SHA.fullmatch(ref)
    assert ref == "sha256:" + hashlib.sha256(b"").hexdigest()


def test_hash_value_is_sha256_over_the_canonical_bytes() -> None:
    v = {"b": [1.0, True], "a": None}
    assert hash_value(v) == "sha256:" + hashlib.sha256(encode(v)).hexdigest()
    assert SHA.fullmatch(hash_value(v))
    assert hash_value({"a": None, "b": [1, True]}) == hash_value(v)


def test_record_hash_equals_the_hash_of_the_log_list() -> None:
    items: list[Any] = [{"item": "design", "hash": "sha256:" + "0" * 64}, {"item": "x", "n": 1.0}]
    assert record_hash([encode(i) for i in items]) == hash_value(items)
    assert record_hash([]) == hash_value([])
    assert SHA.fullmatch(record_hash([encode(1)]))


# --- the per-turn key and the keyed digest ------------------------------------------------

# RFC 5869, Appendix A, Test Cases 1 to 3 (SHA-256): IKM, salt, info, L, PRK and OKM.
RFC5869_CASES = {
    "case-1": (
        "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
        "000102030405060708090a0b0c",
        "f0f1f2f3f4f5f6f7f8f9",
        42,
        "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5",
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865",
    ),
    "case-2": (
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
        "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
        "404142434445464748494a4b4c4d4e4f",
        "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f"
        "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f"
        "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf",
        "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecf"
        "d0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeef"
        "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
        82,
        "06a6b88c5853361a06104c9ceb35b45cef760014904671014a193f40c15fc244",
        "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c"
        "59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71"
        "cc30c58179ec3e87c14c01d5c1f3434f1d87",
    ),
    "case-3": (
        "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
        "",
        "",
        42,
        "19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04",
        "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8",
    ),
}


@pytest.mark.parametrize("case", RFC5869_CASES)
def test_expand_matches_rfc5869_appendix_a(case: str) -> None:
    ikm_hex, salt_hex, info_hex, length, prk_hex, okm_hex = RFC5869_CASES[case]
    ikm, salt, info = bytes.fromhex(ikm_hex), bytes.fromhex(salt_hex), bytes.fromhex(info_hex)
    prk, okm = bytes.fromhex(prk_hex), bytes.fromhex(okm_hex)
    assert len(okm) == length
    # The engine has no extract; the case's PRK is checked here so its vector is whole.
    assert hmac.digest(salt or bytes(32), ikm, "sha256") == prk
    # Our expand yields one 32-byte block, T(1); every case's L is over 32, so compare its first 32.
    assert _expand32(prk, info) == okm[:32]


def test_turn_key_expands_the_ascii_record_hash() -> None:
    root = bytes(range(32))
    rh = hash_value([])
    key = turn_key(root, rh)
    assert key == hmac.digest(root, rh.encode("ascii") + b"\x01", "sha256")
    assert len(key) == 32
    assert turn_key(root, hash_value([1])) != key


def test_turn_key_matches_a_fixed_vector() -> None:
    # Computed once outside the engine, with hmac and hashlib, as HKDF-Expand's T(1).
    rh = "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    assert hash_value([]) == rh
    assert turn_key(bytes(range(32)), rh) == bytes.fromhex(
        "f2baca6918751922568b6a3ec7b34747807fb6d77d8e005e7096732c489d8442"
    )


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "sha256:",
        "sha256:" + "0" * 63,
        "sha256:" + "0" * 65,
        "sha256:" + "A" * 64,
        "SHA256:" + "0" * 64,
        "hmac-sha256:" + "0" * 64,
        "sha256:" + "0" * 64 + "\n",
        "sha256:" + "\uff10" * 64,
    ],
)
def test_turn_key_refuses_a_malformed_record_hash(bad: str) -> None:
    with pytest.raises(_CanonicalError):
        turn_key(bytes(32), bad)


@pytest.mark.parametrize("bad", [None, b"sha256:" + b"0" * 64, 1])
def test_turn_key_refuses_a_record_hash_that_is_not_a_string(bad: Any) -> None:
    with pytest.raises(_CanonicalError, match="not a string"):
        turn_key(bytes(32), bad)


@pytest.mark.parametrize(
    "root",
    [
        bytes(31),
        bytes(33),
        b"",
        bytearray(32),
        memoryview(bytes(32)),
        "0" * 32,
        None,
        list(range(32)),
    ],
    ids=["31 bytes", "33 bytes", "empty", "bytearray", "memoryview", "str", "None", "list"],
)
def test_turn_key_refuses_a_root_key_that_is_not_32_bytes(root: Any) -> None:
    with pytest.raises(ValueError, match="root key"):
        turn_key(root, hash_value([]))


def test_the_root_key_is_checked_before_the_record_hash() -> None:
    with pytest.raises(ValueError, match="root key"):
        turn_key(bytes(31), "not a hash")


def test_turn_key_reads_the_record_hash_by_its_string_data() -> None:
    class Loud(str):
        __slots__ = ()

        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            raise AssertionError("turn_key read the hash through an override")

        def __str__(self) -> str:
            return "sha256:" + "f" * 64

    rh = hash_value([])
    assert turn_key(bytes(32), Loud(rh)) == turn_key(bytes(32), rh)
    with pytest.raises(_CanonicalError):
        turn_key(bytes(32), Loud("x"))


def test_a_bytes_subclass_of_length_32_is_a_root_key() -> None:
    class Key(bytes):
        __slots__ = ()

    rh = hash_value([])
    assert turn_key(Key(bytes(32)), rh) == turn_key(bytes(32), rh)


def test_keyed_digest_form() -> None:
    key = turn_key(bytes(32), hash_value([]))
    data = encode("Dana")
    d = keyed_digest(key, data)
    assert HMAC.fullmatch(d)
    assert d == "hmac-sha256:" + hmac.digest(key, data, "sha256").hex()
    assert keyed_digest(key, encode("Dana")) == d
    assert keyed_digest(key, encode("Dan")) != d


def test_encode_takes_no_bound_above_the_documents() -> None:
    with pytest.raises(ValueError, match="above 128"):
        encode([], 129)
    assert encode([[0]], 2) == b"[[0]]"
    with pytest.raises(ValueError, match="more than 1 deep"):
        encode([[0]], 1)


def test_a_value_is_encoded_within_the_size_bounds_and_a_holder_to_its_depth_alone(
    small_bounds: SmallBounds,
) -> None:
    rows = ["x" * 16] * 17
    with pytest.raises(_CanonicalError, match="more than 2\\^8 code points"):
        encode(rows)
    assert decode(encode(rows, 2)) == rows  # a holder: its own depth, no size bound
    assert decode(encode([rows], 128)) == [rows]
    with pytest.raises(_CanonicalError, match="nests more than 2 deep"):
        encode([[rows]], 2)


def test_turn_key_names_a_bad_record_hash_without_its_repr_or_whole_text() -> None:
    class NoRepr:
        def __repr__(self) -> str:
            raise RuntimeError("repr")

    with pytest.raises(_CanonicalError, match=r"^record hash of type .*NoRepr is not a string$"):
        turn_key(bytes(32), NoRepr())  # type: ignore[arg-type]
    with pytest.raises(_CanonicalError) as info:
        turn_key(bytes(32), "x" * 10**5)
    assert str(info.value) == (
        f"record hash {'x' * 64!r}... (100000 code points) is not sha256: and 64 hex"
    )
    with pytest.raises(_CanonicalError, match=r"^record hash 'short' is not sha256"):
        turn_key(bytes(32), "short")


class _LoudBytes(bytes):
    """Bytes whose own length, slice and conversion raise: read through bytes' methods."""

    def __len__(self) -> int:
        raise RuntimeError("len")

    def __getitem__(self, index: Any) -> Any:
        raise RuntimeError("getitem")

    def __bytes__(self) -> bytes:
        raise RuntimeError("bytes")


def test_turn_key_reads_a_bytes_subclass_root_key_by_bytes_own_methods() -> None:
    rh = hash_value([])
    assert turn_key(_LoudBytes(bytes(32)), rh) == turn_key(bytes(32), rh)
    with pytest.raises(ValueError, match="root key"):
        turn_key(_LoudBytes(bytes(31)), rh)
