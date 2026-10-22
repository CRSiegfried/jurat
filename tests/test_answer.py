# pyright: reportPrivateUsage=false
"""The Answer type and its validation, and probe defect h."""

import dataclasses
import enum
import math
from collections import OrderedDict
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import pytest

import jurat
from jurat import Answer, AnswerKind
from jurat._answer import exact_sum, off_sum
from jurat._canonical import decode, encode


def test_answer_is_public() -> None:
    assert "Answer" in jurat.__all__
    assert "AnswerKind" in jurat.__all__


def test_the_answer_kinds_are_a_strenum() -> None:
    assert issubclass(AnswerKind, enum.StrEnum)
    assert {k.value for k in AnswerKind} == {"noul", "choice", "score", "extraction"}
    assert encode(AnswerKind.noul) == b'"noul"'


@pytest.mark.parametrize("kind", ["noul", AnswerKind.noul])
def test_the_kind_is_stored_as_the_member(kind: Any) -> None:
    a = Answer(kind, ("yes", "no"), {"yes": 1, "no": 0})
    assert a.kind is AnswerKind.noul


VALID: list[tuple[str, Any, Any]] = [
    ("noul", ("yes", "no"), {"yes": 0.9, "no": 0.1}),
    ("noul", ["yes", "no"], {"no": 0, "yes": 1}),
    ("noul", ("yes", "no"), {"yes": 1.0, "no": 0.0}),
    ("noul", ("yes", "no"), {"yes": 0.5, "no": 0.5 + 0.9e-6}),
    ("noul", ("yes", "no"), {"yes": 0.5, "no": 0.5 - 0.9e-6}),
    ("choice", ("a",), {"a": 1}),
    ("choice", ("c", "a", "b"), {"a": 0.2, "b": 0.3, "c": 0.5}),
    ("choice", ("x", "y", "z"), {"x": 0.1, "y": 0.2, "z": 0.7}),
    ("score", ("low", "mid", "high"), {"low": 0.0, "mid": 0.25, "high": 0.75}),
    ("score", ("1", "2"), {"1": 0.5, "2": 0.5}),
    # No sum is required: each value is checked alone.
    ("choice", ("a", "b"), {"a": 0.5, "b": 0.4}),
    ("choice", ("a", "b"), {"a": 0.6, "b": 0.6}),
    ("choice", ("a", "b"), {"a": 0.9, "b": 0.9}),
    ("noul", ("yes", "no"), {"yes": 0.0, "no": 0}),
    ("noul", ("yes", "no"), {"yes": 0.5, "no": 0.5 + 2e-6}),
    ("score", ("low", "high"), {"low": 1, "high": 1.0}),
]


@pytest.mark.parametrize(("kind", "labels", "probabilities"), VALID)
def test_valid_answers(kind: str, labels: Any, probabilities: Any) -> None:
    a = Answer(kind, labels, probabilities)
    assert a.kind == kind
    assert type(a.kind) is AnswerKind
    assert a.labels == tuple(labels)
    assert isinstance(a.labels, tuple)
    assert a.probabilities == probabilities
    assert a.probabilities is not probabilities
    assert type(a.probabilities) is dict


class FloatLike(float):
    __slots__ = ()


class IntLike(int):
    __slots__ = ()


class Loud(str):
    """A str subclass whose overrides the answer must not reach."""

    __slots__ = ()

    def __str__(self) -> str:
        return "LOUD"

    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return 0


class SameData(str):
    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


class Colour(enum.StrEnum):
    red = "red"
    blue = "blue"


INVALID: list[tuple[str, Any, Any, Any]] = [
    ("unknown kind", "binary", ("yes", "no"), {"yes": 1, "no": 0}),
    ("kind not a str", 1, ("yes", "no"), {"yes": 1, "no": 0}),
    ("noul labels reversed", "noul", ("no", "yes"), {"yes": 1, "no": 0}),
    ("noul labels other", "noul", ("y", "n"), {"y": 1, "n": 0}),
    ("noul one label", "noul", ("yes",), {"yes": 1}),
    ("labels a str", "choice", "ab", {"a": 0.5, "b": 0.5}),
    ("labels not a sequence", "choice", {"a"}, {"a": 1}),
    ("labels repeated", "choice", ("a", "a"), {"a": 1}),
    ("label not a str", "choice", ("a", 1), {"a": 1, 1: 0}),
    ("missing key", "choice", ("a", "b"), {"a": 1}),
    ("extra key", "choice", ("a",), {"a": 1, "b": 0}),
    ("probabilities not a mapping", "choice", ("a",), [("a", 1)]),
    ("an OrderedDict", "choice", ("a",), OrderedDict(a=1)),
    ("a mapping proxy", "choice", ("a",), MappingProxyType({"a": 1})),
    ("a numpy-like float", "choice", ("a", "b"), {"a": FloatLike(0.5), "b": 0.5}),
    ("an int subclass", "choice", ("a", "b"), {"a": IntLike(1), "b": 0}),
    ("probabilities None", "choice", ("a",), None),
    ("bool value", "noul", ("yes", "no"), {"yes": True, "no": False}),
    ("str value", "choice", ("a",), {"a": "1"}),
    ("None value", "choice", ("a", "b"), {"a": 1, "b": None}),
    ("nan", "choice", ("a", "b"), {"a": 1, "b": math.nan}),
    ("inf", "choice", ("a", "b"), {"a": math.inf, "b": 0}),
    ("negative", "choice", ("a", "b"), {"a": 1.1, "b": -0.1}),
    ("above one", "choice", ("a", "b"), {"a": 2, "b": -1}),
    ("percent", "noul", ("yes", "no"), {"yes": 0.9, "no": 70}),
    ("a huge int", "choice", ("a", "b"), {"a": 10**400, "b": 0}),
    ("a huge negative int", "choice", ("a", "b"), {"a": 1, "b": -(10**400)}),
    ("a lone surrogate label", "choice", ("a", "\ud800"), {"a": 1, "\ud800": 0}),
    ("a lone surrogate key", "choice", ("a",), {"a": 1, "\udfff": 0}),
    ("labels a bytes", "choice", (b"a",), {b"a": 1}),
    ("labels repeated by string data", "choice", ("a", SameData("a")), {"a": 1}),
    ("keys repeated by string data", "choice", ("a",), {"a": 1, SameData("a"): 0}),
    ("key not a str", "choice", ("a",), {"a": 1, None: 0}),
]


@pytest.mark.parametrize(
    ("why", "kind", "labels", "probabilities"), INVALID, ids=[row[0] for row in INVALID]
)
def test_invalid_answers_raise_value_error(
    why: str, kind: Any, labels: Any, probabilities: Any
) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        Answer(kind, labels, probabilities)


def test_a_huge_int_probability_is_a_value_error_not_an_overflow() -> None:
    with pytest.raises(ValueError, match=r"not in \[0, 1\]"):
        Answer("choice", ("a", "b"), {"a": 10**400, "b": 0})


def test_labels_and_keys_are_read_by_their_string_data_and_stored_as_str() -> None:
    a = Answer("choice", (Loud("x"), Colour.red), {Loud("x"): 0.5, "red": 0.5})
    assert a.labels == ("x", "red")
    assert [type(label) for label in a.labels] == [str, str]
    assert a.probabilities == {"x": 0.5, "red": 0.5}
    assert [type(k) for k in a.probabilities] == [str, str]
    b = Answer("choice", ("red", "blue"), {Colour.blue: 0.25, Colour.red: 0.75})
    assert list(b.probabilities) == ["red", "blue"]
    assert [type(k) for k in b.probabilities] == [str, str]
    assert b == Answer("choice", ("red", "blue"), {"red": 0.75, "blue": 0.25})


def test_a_noul_answer_takes_str_subclass_labels() -> None:
    a = Answer(Loud("noul"), (Loud("yes"), Loud("no")), {Loud("yes"): 1, Loud("no"): 0})
    assert a.kind is AnswerKind.noul
    assert a.labels == ("yes", "no")


def test_an_answer_is_not_hashable() -> None:
    with pytest.raises(TypeError):
        hash(Answer("noul", ("yes", "no"), {"yes": 1, "no": 0}))


def test_values_are_stored_as_given() -> None:
    a = Answer("choice", ("a", "b"), {"a": 1, "b": 0.0})
    assert type(a.probabilities["a"]) is int
    assert type(a.probabilities["b"]) is float


def test_the_mapping_is_copied() -> None:
    p = {"yes": 0.25, "no": 0.75}
    a = Answer("noul", ("yes", "no"), p)
    p["yes"] = 0.0
    assert a.probabilities["yes"] == 0.25


@pytest.mark.parametrize(
    ("probabilities", "total"),
    [
        ({"a": 0.9, "b": 0.1}, Fraction(1)),
        ({"a": 0.9, "b": 0.9}, Fraction(18, 10)),
        ({"a": 0, "b": 0.0}, Fraction(0)),
        ({"a": 0.1, "b": 0.2}, Fraction(3, 10)),  # exact: 0.1 + 0.2 in floats is past 0.3
        ({"a": 1, "b": 1}, Fraction(2)),
    ],
)
def test_the_exact_sum_is_over_the_decimals_each_value_reads_back_as(
    probabilities: dict[str, float], total: Fraction
) -> None:
    assert exact_sum(Answer("choice", ("a", "b"), probabilities)) == total


@pytest.mark.parametrize("labels", [(), []])
def test_an_answer_with_no_labels_is_refused(labels: Any) -> None:
    with pytest.raises(ValueError, match="at least one label"):
        Answer("choice", labels, {})


def test_the_exact_sum_ignores_the_order_of_the_labels() -> None:
    labels = tuple(f"l{i}" for i in range(10))
    a = Answer("choice", labels, dict.fromkeys(labels, 0.1))
    assert exact_sum(a) == 1
    assert off_sum(a) is None


@pytest.mark.parametrize(
    ("probabilities", "flagged"),
    [
        ({"a": 0.45, "b": 0.5}, None),  # exactly 0.95: inside
        ({"a": 0.5, "b": 0.55}, None),  # exactly 1.05: inside, though its floats sum past it
        ({"a": 0.99, "b": 0.0}, None),
        ({"a": 0.5, "b": 0.51}, None),
        ({"a": 0.5000000000000001, "b": 0.55}, 1.05),  # 1.0500000000000001, rounded once
        ({"a": 0.4499999999999999, "b": 0.5}, 0.9499999999999999),
        ({"a": 0.5, "b": 0.4}, 0.9),
        ({"a": 0.9, "b": 0.9}, 1.8),
        ({"a": 0, "b": 0}, 0.0),
        ({"a": 1, "b": 1}, 2.0),
    ],
)
def test_a_sum_outside_the_band_is_flagged_rounded_once(
    probabilities: dict[str, float], flagged: float | None
) -> None:
    got = off_sum(Answer("choice", ("a", "b"), probabilities))
    assert got == flagged
    assert got is None or type(got) is float


def test_the_flag_rounds_the_exact_sum_not_a_float_sum() -> None:
    # In floats 0.1 + 0.2 is 0.30000000000000004; the exact sum rounds to 0.3.
    assert off_sum(Answer("choice", ("a", "b"), {"a": 0.1, "b": 0.2})) == 0.3


def test_answer_is_frozen_with_three_fields() -> None:
    a = Answer("noul", ("yes", "no"), {"yes": 1, "no": 0})
    assert [f.name for f in dataclasses.fields(a)] == ["kind", "labels", "probabilities"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.kind = AnswerKind.choice  # pyright: ignore[reportAttributeAccessIssue]


def test_an_answer_round_trips_through_its_canonical_object() -> None:
    a = Answer("score", ("low", "high"), {"low": 0.25, "high": 0.75})
    obj = {"kind": a.kind, "labels": a.labels, "probabilities": a.probabilities}
    back = decode(encode(obj))
    assert Answer(back["kind"], back["labels"], back["probabilities"]) == a


# --- probe defect h: an answer with a missing probability --------------------------------


def test_probe_h_an_answer_needs_its_labels_and_probabilities() -> None:
    with pytest.raises(TypeError):
        Answer("noul")  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


def test_probe_h_a_missing_probability_is_not_zero() -> None:
    with pytest.raises(ValueError, match=r"\['no'\]"):
        Answer("noul", ("yes", "no"), {"yes": 1.0})


@pytest.mark.parametrize("name", ["p", "p_top", "margin", "top", "expected", "confidence"])
def test_probe_h_no_derived_accessors(name: str) -> None:
    a = Answer("noul", ("yes", "no"), {"yes": 0.9, "no": 0.1})
    assert not hasattr(a, name)


def test_a_probability_past_the_digit_limit_is_named_by_its_bit_length() -> None:
    with pytest.raises(ValueError, match=r"^probability integer of 16610 bits for 'yes' is not in"):
        Answer("noul", ["yes", "no"], {"yes": 10**5000, "no": 0})


class _ClassRaises:
    @property
    def __class__(self) -> type:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise RuntimeError("__class__ was read")


class _Raising(type):
    def __getattribute__(cls, name: str) -> Any:
        raise RuntimeError(f"the metaclass was read for {name}")


class _Hidden(metaclass=_Raising):
    pass


def test_an_answer_kind_is_judged_by_its_type() -> None:
    with pytest.raises(ValueError, match="answer kind"):
        Answer(_ClassRaises(), ["yes", "no"], {"yes": 1, "no": 0})  # type: ignore[arg-type]


def test_a_probability_of_a_hostile_metaclass_is_named_by_its_type() -> None:
    with pytest.raises(ValueError, match=r"^probability of type _Hidden for 'yes'"):
        Answer("noul", ["yes", "no"], {"yes": _Hidden(), "no": 0})  # type: ignore[dict-item]


def test_labels_are_judged_by_their_type() -> None:
    with pytest.raises(ValueError, match=r"^labels <.*_ClassRaises> are not a sequence"):
        Answer("choice", _ClassRaises(), {})  # type: ignore[arg-type]


def test_many_labels_are_matched_against_their_keys_in_linear_time() -> None:
    labels = [f"l{i}" for i in range(2**16)]
    a = Answer("choice", labels, dict.fromkeys(labels, 2**-16))
    assert len(a.labels) == 2**16
    with pytest.raises(ValueError, match=r"^probability for key\(s\) \['x'\] not among"):
        Answer("choice", labels, {**dict.fromkeys(labels, 2**-16), "x": 0})


def test_an_answer_kind_that_is_no_string_is_refused_before_any_hash() -> None:
    class Unhashed:
        def __hash__(self) -> int:
            raise RuntimeError("hashed")

    with pytest.raises(ValueError, match="is not noul, choice or score"):
        Answer(Unhashed(), ["yes", "no"], {"yes": 1, "no": 0})  # type: ignore[arg-type]
    assert Answer(AnswerKind.score, ["lo", "hi"], {"lo": 1, "hi": 0}).kind is AnswerKind.score
