# pyright: reportPrivateUsage=false
"""The type-free value domain of the record document."""

import enum
import json
import math
from decimal import Decimal
from fractions import Fraction
from typing import Any, cast

import pytest

from counting import SmallBounds, bound, counted
from jurat._canonical import decode, encode
from jurat._values import (
    MAX_DEPTH,
    MAX_DOCUMENT_DEPTH,
    UNREPRESENTABLE,
    described,
    domain_error,
    guarded_repr,
    in_domain,
    nesting_depth,
    representable,
)

MAX = 2**53 - 1


class Colour(enum.StrEnum):
    red = "red"


class Plain(enum.Enum):
    a = "a"


class MyStr(str):
    __slots__ = ()


class Shouty(str):
    """A subclass whose overrides the domain must not reach."""

    __slots__ = ()

    def __str__(self) -> str:
        return "LOUD"

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        raise AssertionError("the domain read the string through an override")


class SameData(str):
    """A key that is unequal to a plain key with the same string data."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


class MyInt(int):
    __slots__ = ()


class MyFloat(float):
    __slots__ = ()


class MyList(list[Any]):
    pass


class MyDict(dict[str, Any]):
    pass


IN: list[object] = [
    None,
    True,
    False,
    0,
    -0.0,
    1.5,
    MAX,
    -MAX,
    float(2**60),
    5e-324,
    "",
    "a\x00\x7f\u2028\U0001f600",
    Colour.red,
    [],
    (),
    [1, [2, (3, {"a": None})]],
    {},
    {"k": [True, None, 1.0]},
    {Colour.red: 1},
    MyStr("x"),
    Shouty("quiet"),
    {MyStr("k"): MyStr("v")},
]

OUT: list[object] = [
    math.nan,
    math.inf,
    -math.inf,
    MAX + 1,
    -MAX - 1,
    2**64,
    "\ud800",
    "a\udfffb",
    b"bytes",
    bytearray(b"x"),
    Decimal("1.5"),
    Fraction(1, 2),
    1j,
    {1: "a"},
    {("a",): 1},
    {"\ud800": 1},
    {None: 1},
    {1, 2},
    frozenset(),
    object(),
    Plain.a,
    MyInt(1),
    MyFloat(1.0),
    MyList(),
    MyDict(),
    [math.nan],
    {"a": {"b": [b"x"]}},
]


@pytest.mark.parametrize("value", IN, ids=repr)
def test_in_domain(value: object) -> None:
    assert in_domain(value)
    assert domain_error(value) is None


@pytest.mark.parametrize("value", OUT, ids=repr)
def test_out_of_domain(value: object) -> None:
    assert not in_domain(value)
    assert isinstance(domain_error(value), str)


def test_a_cycle_is_out_of_domain() -> None:
    a: list[Any] = []
    a.append(a)
    d: dict[str, Any] = {}
    d["self"] = d
    assert not in_domain(a)
    assert not in_domain(d)


def test_a_shared_value_that_is_not_a_cycle_is_in_domain() -> None:
    shared = [1]
    assert in_domain([shared, shared, {"a": shared}])


def test_deep_nesting_is_out_of_domain_not_a_crash() -> None:
    v: Any = None
    for _ in range(100_000):
        v = [v]
    assert domain_error(v) == "the value nests more than 64 deep"


def _nest(depth: int, leaf: Any = None, *, kind: str = "list") -> Any:
    v = leaf
    for i in range(depth):
        v = {"k": v} if kind == "dict" or (kind == "mixed" and i % 2) else [v]
    return v


def test_the_bounds() -> None:
    assert (MAX_DEPTH, MAX_DOCUMENT_DEPTH) == (64, 128)


@pytest.mark.parametrize("kind", ["list", "dict", "mixed"])
def test_a_value_nests_at_most_64_deep(kind: str) -> None:
    # A scalar nests 0 deep, a list or object one deeper than its deepest member.
    assert in_domain(_nest(64, kind=kind))
    assert in_domain(_nest(64, [], kind=kind)[0] if kind == "list" else _nest(63, [], kind=kind))
    assert domain_error(_nest(65, kind=kind)) == "the value nests more than 64 deep"
    assert domain_error(_nest(64, [], kind=kind)) == "the value nests more than 64 deep"


def test_the_depth_is_the_deepest_member_s() -> None:
    wide = [_nest(10), _nest(63), "x", {"a": _nest(5)}]
    assert in_domain(wide)
    assert not in_domain([*wide, _nest(64)])


def test_a_shared_value_counts_where_it_stands() -> None:
    shared = _nest(60)
    assert in_domain([shared, [[[shared]]]])  # 61, then 64
    assert not in_domain([shared, [[[[shared]]]]])  # 65
    assert not in_domain([[[[[shared]]]], shared])  # met deep first, then shallow


def _doubled(n: int, *, kind: type = list) -> Any:
    """``n`` wraps, each a container holding the one before twice: 2^n places for the core."""
    v: Any = 0
    for _ in range(n):
        v = kind((v, v))
    return v


# At the real bounds: the walk stops past 2^20 values and 2^26 code points.


def test_a_widely_shared_value_is_counted_at_every_place_and_refused_fast() -> None:
    v = _doubled(60)  # 61 deep, within the depth bound, and 2^61 - 1 values at their places
    with counted() as n:
        assert domain_error(v) == "the value holds more than 2^20 values"
    assert n[0] == bound() == 2**20 + 1  # counted at every place, and stopped past the bound


def test_a_string_of_2_26_code_points_passes_and_one_more_is_refused() -> None:
    s = "x" * 2**26
    assert in_domain(s)
    assert in_domain([s])
    assert domain_error([s, "y"]) == "the value's strings and keys hold more than 2^26 code points"
    assert domain_error({s: "y"}) is not None
    assert domain_error({"y": s}) is not None


# Under small bounds, set as the walks read them: 2^6 values and 2^8 code points.


def test_each_place_counts_and_the_walk_stops_past_the_bound(small_bounds: SmallBounds) -> None:
    assert in_domain(_doubled(5))  # 2^6 - 1 values
    with counted() as n:
        assert domain_error(_doubled(6)) == "the value holds more than 2^6 values"
    assert n[0] == bound() == 65


def test_a_shared_row_counts_at_every_place(small_bounds: SmallBounds) -> None:
    # [[0] * k] * n holds n * (k + 1) + 1 values; 7 * 9 + 1 is 2^6.
    row = [0] * 8
    assert in_domain([row] * 7)
    assert not in_domain([row] * 7 + [0])
    assert not in_domain([[row] * 7, 0])


def test_a_value_holds_itself_and_a_scalar_is_one_value(small_bounds: SmallBounds) -> None:
    assert in_domain([0] * 63)
    assert not in_domain([0] * 64)


def test_a_key_is_a_value_with_its_code_points(small_bounds: SmallBounds) -> None:
    assert in_domain({chr(97 + i): 0 for i in range(26)} | {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0})
    assert not in_domain({str(i): 0 for i in range(32)})  # 1 + 2 * 32
    assert in_domain({"k" * 255: 0})
    assert not in_domain({"k" * 257: 0})


def test_strings_and_keys_are_counted_in_code_points_at_every_place(
    small_bounds: SmallBounds,
) -> None:
    s = "\U0001f600" * 16  # one code point each, four bytes in UTF-8
    assert in_domain([s] * 16)
    err = "the value's strings and keys hold more than 2^8 code points"
    assert domain_error([s] * 17) == err
    key = {s: 0}
    assert in_domain([key] * 16)
    assert domain_error([key] * 16 + ["x"]) == err
    assert domain_error([*[key] * 15, {s: s}]) == err


def test_a_strings_length_is_counted_before_its_code_points_are_read(
    small_bounds: SmallBounds, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jurat._values as values

    read: list[int] = []
    scan = values.is_scalar_string

    def counting(value: object) -> bool:
        read.append(len(cast(str, value)))
        return scan(value)

    monkeypatch.setattr(values, "is_scalar_string", counting)
    s = "x" * 16
    assert domain_error([s] * 60) is not None
    assert read == [16] * 16  # the 17th place passes the bound before it is read
    read.clear()
    assert domain_error({s * 15 + "x": [s]}) is not None
    assert read == [15 * 16 + 1]  # the key is read; the member passes the bound unread


def test_a_str_subclass_is_counted_by_its_string_data(small_bounds: SmallBounds) -> None:
    class Short(str):
        __slots__ = ()

        def __len__(self) -> int:
            return 0

    assert not in_domain([Short("x" * 16)] * 17)


def test_an_item_or_document_is_held_to_its_depth_alone(small_bounds: SmallBounds) -> None:
    big = [[0] * 8] * 8
    assert domain_error(big, MAX_DOCUMENT_DEPTH) is None
    assert domain_error(big, 2) is None
    assert domain_error([big], 2) == "the value nests more than 2 deep"
    assert domain_error([_nest(127)], MAX_DOCUMENT_DEPTH) is None
    assert domain_error([_nest(128)], MAX_DOCUMENT_DEPTH) is not None
    looped: list[Any] = []
    looped.append(looped)
    assert domain_error(looped, MAX_DOCUMENT_DEPTH) == "the value contains itself"


def test_a_decoded_record_gets_the_live_values_verdict(small_bounds: SmallBounds) -> None:
    row, text = [0] * 8, "ab" * 8
    for live in ([row] * 7, [row] * 7 + [0], [text] * 16, [text] * 16 + ["c"], [{text: row}] * 4):
        recorded = decode(encode(live, MAX_DOCUMENT_DEPTH))  # shares nothing
        assert domain_error(recorded) == domain_error(live)


def test_the_bounds_are_written_as_the_message_reads_them(small_bounds: SmallBounds) -> None:
    small_bounds(values=100, code_points=2**8)
    assert domain_error([0] * 100) == "the value holds more than 100 values"


def test_the_depth_is_counted_without_recursion() -> None:
    import sys

    v = _nest(64)
    frames, f = 0, sys._getframe()
    while f is not None:
        frames, f = frames + 1, f.f_back
    limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(frames + 20)  # far fewer than 64 levels of recursion would need
        assert in_domain(v)
        assert nesting_depth(v) == 64
    finally:
        sys.setrecursionlimit(limit)


def test_an_error_inside_a_deep_value_is_found() -> None:
    assert domain_error(_nest(64, math.nan)) == "number nan is not finite"
    assert domain_error(_nest(30, {1: 2}, kind="dict")) is not None


# --- the observer's depth walk ------------------------------------------------------------


def test_nesting_depth_counts_containers() -> None:
    assert nesting_depth(None) == 0
    assert nesting_depth("x") == 0
    assert nesting_depth([]) == 1
    assert nesting_depth({"a": [1, (2,)]}) == 3
    mixed: Any = [[], [[[]]], ()]
    assert nesting_depth(mixed) == 4
    assert nesting_depth(_nest(64)) == 64


def test_nesting_depth_counts_subclasses_without_their_overrides() -> None:
    class Loud(list[Any]):
        def __iter__(self) -> Any:
            raise AssertionError("an override was reached")

    class LoudDict(dict[str, Any]):
        def values(self) -> Any:
            raise AssertionError("an override was reached")

        def __iter__(self) -> Any:
            raise AssertionError("an override was reached")

    assert nesting_depth(Loud([LoudDict(a=MyList([1]))])) == 3


def test_nesting_depth_stops_past_its_limit() -> None:
    assert nesting_depth(_nest(65)) == 65
    assert nesting_depth(_nest(100_000)) == 65
    assert nesting_depth(_nest(5), limit=3) == 4
    assert nesting_depth(_nest(3), limit=3) == 3
    shared = _nest(60)
    assert nesting_depth([shared, [[[[[shared]]]]]]) == 65


def test_nesting_depth_ends_a_cycle_past_its_limit() -> None:
    # A reader finds no container at two places first; a cycle still ends, past the limit.
    a: list[Any] = [1]
    a.append(a)
    assert nesting_depth(a) == 65
    assert nesting_depth({"x": [a]}, limit=3) == 4


def test_a_str_subclass_is_read_by_its_string_data() -> None:
    assert domain_error(MyStr("\ud800")) is not None
    assert in_domain(Shouty("\U0001f600"))


def test_two_keys_with_the_same_string_data_are_out_of_domain() -> None:
    k = SameData("a")
    d: dict[str, int] = {"a": 1, k: 2}
    assert len(d) == 2
    err = domain_error(d)
    assert err is not None
    assert "string data" in err
    assert in_domain({SameData("a"): 1, SameData("b"): 2})


def test_python_bytes_are_out_of_the_type_free_domain() -> None:
    for value in (b"", b"x", [b"x"], {"payload": b"x"}):
        assert not in_domain(value)


# ---------------------------------------------------------------------------------------------
# The guarded `repr`


class Raises:
    def __repr__(self) -> str:
        raise RuntimeError("no")


class Wrong:
    def __repr__(self) -> str:
        return 42  # type: ignore[return-value]


class Shifty(str):
    def __str__(self) -> str:
        return "shifted"


class ShiftyRepr:
    def __repr__(self) -> str:
        return Shifty("data")


class DeepWalk(list[object]):
    def __iter__(self) -> Any:
        raise RuntimeError("never reached: the walk reads the base type's members")


def test_guarded_repr_is_the_repr_of_a_representable_value() -> None:
    assert guarded_repr(42) == "42"
    assert guarded_repr(("q:intent",)) == "('q:intent',)"
    cyclic: list[object] = []
    cyclic.append(cyclic)
    assert guarded_repr(cyclic) == "[[...]]"
    assert guarded_repr(_nest(MAX_DEPTH)) == repr(_nest(MAX_DEPTH))
    assert guarded_repr(b"x") == "b'x'"
    exact = {1j: frozenset({None, True, 1.5})}
    assert guarded_repr(exact) == repr(exact)


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (DeepWalk([1]), "<test_values.DeepWalk>"),
        (Raises(), "<test_values.Raises>"),  # its own repr is never run
        (Wrong(), "<test_values.Wrong>"),
        (ShiftyRepr(), "<test_values.ShiftyRepr>"),
        ([Raises()], "<builtins.list>"),
        (MyStr("x"), "<test_values.MyStr>"),
        ({"a": Plain.a}, "<builtins.dict>"),  # a foreign member: the top-level type named
    ],
)
def test_guarded_repr_names_the_type_of_a_value_holding_a_foreign_part(
    value: object, label: str
) -> None:
    assert guarded_repr(value) == label


def test_guarded_repr_is_unrepresentable_past_the_depth_or_with_names_not_exact_strings() -> None:
    assert guarded_repr(_nest(MAX_DEPTH + 1)) == UNREPRESENTABLE == "<unrepresentable>"

    class Meta(type):
        pass

    odd = Meta("Odd", (), {})
    type.__dict__["__qualname__"].__set__(odd, "Odd")
    odd.__module__ = MyStr("m")  # type: ignore[assignment]
    assert guarded_repr(odd()) == UNREPRESENTABLE


def test_a_dataclass_holding_a_widely_shared_list_is_named_fast() -> None:
    import dataclasses as dc

    @dc.dataclass
    class Holder:
        rows: list[Any]

    wrapped: Any = []
    for _ in range(60):
        wrapped = [wrapped, wrapped]
    with counted() as n:
        label = guarded_repr(Holder(wrapped))
    assert label == f"<test_values.{Holder.__qualname__}>"
    assert n[0] == 1  # one foreign value met; its repr, which writes every place, never ran


def test_a_numpy_like_member_names_the_dict() -> None:
    class Float64(float):
        def __repr__(self) -> str:
            return "np.float64(1.0)"

    assert guarded_repr({"x": Float64(1.0)}) == "<builtins.dict>"


def test_an_int_and_bytes_count_toward_the_code_points(small_bounds: SmallBounds) -> None:
    import sys

    assert representable(2**255 - 1)  # 255 bits
    assert not representable(2**257)
    assert representable(b"x" * 255)
    assert not representable([b"x" * 200, b"x" * 57])
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        assert guarded_repr(10**100_000) == UNREPRESENTABLE  # bounded by bit length
    finally:
        sys.set_int_max_str_digits(limit)


@pytest.mark.parametrize(
    "value",
    [10**5000, -(10**5000), [1, 10**5000], {"a": 10**5000}],
    ids=["pos", "neg", "list", "dict"],
)
def test_an_integer_past_the_digit_limit_is_out_of_domain_by_its_bit_length(value: object) -> None:
    assert not in_domain(value)
    assert domain_error(value) == "integer of 16610 bits exceeds magnitude 2^53 - 1"


def test_an_integer_key_past_the_digit_limit_is_named_by_its_bit_length() -> None:
    tail = "is not a string of scalar values"
    assert domain_error({10**5000: 1}) == f"object key integer of 16610 bits {tail}"
    assert domain_error({2**53: 1}) == f"object key 9007199254740992 {tail}"
    assert domain_error({1.5: 1}) == f"object key 1.5 {tail}"


class _Unprintable:
    def __hash__(self) -> int:
        return 1

    def __repr__(self) -> str:
        raise RuntimeError("repr was called")


def test_a_key_whose_repr_raises_is_out_of_domain_named_by_its_type() -> None:
    tail = "is not a string of scalar values"
    assert domain_error({_Unprintable(): 1}) == f"object key <_Unprintable> {tail}"
    assert domain_error({(1, 10**5000): 1}) == f"object key <tuple> {tail}"
    assert not in_domain({_Unprintable(): 1})


class _Raising(type):
    def __getattribute__(cls, name: str) -> Any:
        raise RuntimeError(f"the metaclass was read for {name}")


class _Huge(type):
    def __getattribute__(cls, name: str) -> Any:
        if name == "__qualname__":
            return 10**5000
        return type.__getattribute__(cls, name)


class _Hidden(metaclass=_Raising):
    pass


class _Big(metaclass=_Huge):
    pass


@pytest.mark.parametrize(("value", "name"), [(_Hidden(), "_Hidden"), (_Big(), "_Big")])
def test_a_type_name_is_read_past_a_hostile_metaclass(value: object, name: str) -> None:
    assert described(value) == f"<{name}>"
    assert domain_error(value) == f"{name} is not in the value domain"
    assert domain_error({value: 1}) == f"object key <{name}> is not a string of scalar values"


def test_an_integer_is_written_whatever_the_process_digit_limit() -> None:
    import sys

    in_limit = 10**4299
    saved = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)  # no limit: the fixed bound still decides
        assert described(10**5000) == "integer of 16610 bits"
        assert described(-(10**4300)) == "integer of 14285 bits"
        assert described(in_limit) == repr(in_limit)
        sys.set_int_max_str_digits(1000)  # a lower limit: an in-bound integer is not written
        assert described(10**2000) == "integer of 6644 bits"
        assert domain_error(10**2000) == "integer of 6644 bits exceeds magnitude 2^53 - 1"
    finally:
        sys.set_int_max_str_digits(saved)
    assert described(in_limit) == repr(in_limit)
    assert described(-(10**4300) + 1) == repr(-(10**4300) + 1)


# --- the guard's size bounds ---------------------------------------------------------------


def test_a_widely_shared_value_is_unrepresentable_fast() -> None:
    # Hashing a plain tuple walks every place before Python 3.14; this one's hash is cheap.
    t = _doubled(60, kind=_Hashed)
    with counted() as n:  # at the real bound
        assert guarded_repr(frozenset({t})) == UNREPRESENTABLE
    assert n[0] <= bound() + 1
    assert guarded_repr(_doubled(3)) == repr(_doubled(3))


def test_a_widely_shared_list_is_unrepresentable_past_the_bound_set(
    small_bounds: SmallBounds,
) -> None:
    wrapped: Any = []
    for _ in range(60):
        wrapped = [wrapped, wrapped]
    with counted() as n:
        assert guarded_repr(wrapped) == UNREPRESENTABLE
    assert n[0] == bound() == 65


class _Hashed(tuple[Any, ...]):
    """A tuple whose hash reads none of its members."""

    __slots__ = ()

    def __hash__(self) -> int:
        return 0


def test_the_guard_walks_a_key_and_a_set_member_at_every_place(small_bounds: SmallBounds) -> None:
    t = _doubled(6, kind=_Hashed)
    assert guarded_repr({t: 0}) == UNREPRESENTABLE
    assert guarded_repr({t}) == UNREPRESENTABLE
    assert guarded_repr(_doubled(5, kind=_Hashed)) != UNREPRESENTABLE


def test_the_guard_counts_each_member_and_key_as_a_value_at_every_place(
    small_bounds: SmallBounds,
) -> None:
    row = [0] * 8
    assert representable([row] * 7)  # 7 * 9 + 1 is 2^6
    assert not representable([row] * 7 + [0])
    assert representable({str(i): 0 for i in range(31)})  # 1 + 2 * 31
    assert not representable({str(i): 0 for i in range(32)})
    assert representable({frozenset(range(7)): [0] * 8})  # 1 + 1 + 7 + 1 + 8
    assert not representable([{frozenset(range(7)): [0] * 8}] * 4)


def test_the_guard_counts_string_data_at_every_place(small_bounds: SmallBounds) -> None:
    class Short(str):
        __slots__ = ()

        def __len__(self) -> int:
            return 0

    s = "x" * 16
    assert representable([s] * 16)
    assert not representable([s] * 17)
    assert not representable([Short(s)] * 17)
    assert not representable([{s: 0}] * 17)  # keys
    assert not representable([{s}] * 17)  # set members
    assert not representable([s.encode()] * 17)  # bytes count by their length
    assert guarded_repr([s] * 17) == UNREPRESENTABLE


def test_the_guard_walks_sets_and_keys_as_levels() -> None:
    deep: Any = 0
    for _ in range(MAX_DEPTH):
        deep = frozenset({deep})
    assert representable(deep)
    assert not representable(frozenset({deep}))
    assert not representable({_nest_tuple(MAX_DEPTH): 0})
    assert representable({_nest_tuple(MAX_DEPTH - 1): 0})


def _nest_tuple(depth: int) -> Any:
    v: Any = 0
    for _ in range(depth):
        v = (v,)
    return v


def test_a_cycle_cut_counts_as_one_value(small_bounds: SmallBounds) -> None:
    a: list[Any] = [0] * 62
    a.append(a)  # itself, 62 zeros and the cut
    assert representable(a)
    a.insert(0, 0)
    assert not representable(a)


def test_a_cycle_is_counted_where_its_path_ends_each_time(small_bounds: SmallBounds) -> None:
    a: list[Any] = []
    b: list[Any] = [a]
    a.append(b)
    root = [a, b, a, b]
    assert guarded_repr(root) == repr(root)
    wide: list[Any] = [0] * 27
    a.append(wide)
    # The root and its 0, then a, b, b's cut and wide, then b, a, a's cut and wide again: 2^6.
    assert representable([a, b, 0])
    assert not representable([a, b, 0, 0])


def test_the_guard_reads_no_override_of_a_set_or_dict() -> None:
    class LoudSet(set[Any]):
        def __iter__(self) -> Any:
            raise AssertionError("an override was reached")

    class LoudFrozen(frozenset[Any]):
        def __iter__(self) -> Any:
            raise AssertionError("an override was reached")

    class LoudDict(dict[Any, Any]):
        def items(self) -> Any:
            raise AssertionError("an override was reached")

        def __iter__(self) -> Any:
            raise AssertionError("an override was reached")

    assert representable(LoudSet({LoudFrozen({1})}))
    assert representable(LoudDict({(1,): LoudSet()}))
    assert not representable(LoudSet({"x" * 2**20 + str(i) for i in range(64)}))  # 2^26 + 74


def test_a_holder_counts_a_shared_containers_depth_where_it_stands() -> None:
    shared = _nest(60)
    assert domain_error([shared, [[[shared]]]], MAX_DEPTH) is None
    assert domain_error([shared, [[[[shared]]]]], MAX_DEPTH) == "the value nests more than 64 deep"


# --- a parsed document's sharing ------------------------------------------------------------


def test_a_container_at_two_places_or_in_a_cycle_is_shared() -> None:
    from jurat._values import shares_a_container

    row = [1]
    assert shares_a_container({"a": row, "b": row})
    assert shares_a_container([{"k": 1}] * 2)
    assert shares_a_container(((1,),) * 2)  # tuples too
    looped: list[Any] = []
    looped.append(looped)
    assert shares_a_container(looped)
    keyed = (1, 2)
    assert shares_a_container({keyed: 0, "x": keyed})  # a key is a place
    assert shares_a_container([MyList([1])] * 2)  # subclasses too
    assert shares_a_container(_doubled(60))  # found at once, by identity


def test_strings_and_empty_containers_may_repeat() -> None:
    from jurat._values import shares_a_container

    key = "k" * 100
    assert not shares_a_container([{key: key}, {key: key}])  # a decoder shares key strings
    empty: list[Any] = []
    assert not shares_a_container({"a": (), "b": (), "c": empty, "d": empty, "e": {}})
    assert not shares_a_container(json.loads('{"a": [1, [2]], "b": [1, [2]], "c": "x"}'))
    assert not shares_a_container(5)


def test_a_repr_that_raises_is_unrepresentable() -> None:
    assert guarded_repr(10**5000) == UNREPRESENTABLE  # past Python's own digit limit


def test_an_exact_type_is_known_by_identity_without_its_metaclass() -> None:
    class Meta(type):
        def __hash__(cls) -> int:
            raise RuntimeError("hashed")

        def __eq__(cls, other: object) -> bool:
            raise RuntimeError("compared")

    odd = Meta("Odd", (), {})
    assert guarded_repr(odd()) == f"<{__name__}.Odd>"
    assert guarded_repr([odd()]) == "<builtins.list>"


def test_a_long_key_is_clipped_in_a_message_and_a_short_one_is_whole() -> None:
    long = "k" * 1000
    assert domain_error({long: 1, SameData(long): 2}) == (
        f"two object keys have the string data {'k' * 64!r}... (1000 code points)"
    )
    assert domain_error({"a": 1, SameData("a"): 2}) == "two object keys have the string data 'a'"
    surrogate = "\udc80" * 100
    err = domain_error({surrogate: 1})
    assert (
        err
        == f"object key {chr(0xDC80) * 64!r}... (100 code points) is not a string of scalar values"
    )
