"""The oracle binding's types: what a call passes, ``SourceCall``
and ``RenderedQuestion``, and what it returns, ``CallResult``."""

import dataclasses
from typing import Any, cast

import pytest

import jurat
from jurat import AnswerKind, CallResult, RenderedQuestion, SourceCall


def test_the_three_binding_types_are_public() -> None:
    assert {"CallResult", "SourceCall", "RenderedQuestion"} <= set(jurat.__all__)


def test_a_call_result_defaults_every_provenance_member_to_null() -> None:
    result = CallResult({"t:x": 1})
    assert result.values == {"t:x": 1}
    assert (result.snapshot, result.elicitation, result.prompt) == (None, None, None)
    assert (result.usage, result.attempts) == (None, None)
    assert result.truncated == ()


def test_a_call_result_takes_its_provenance_by_keyword_only() -> None:
    with pytest.raises(TypeError):
        CallResult({}, "snap")  # pyright: ignore[reportCallIssue]
    result = CallResult(
        {}, snapshot="m-1", elicitation="e", prompt="p", usage={"in": 3}, attempts=2
    )
    assert (result.snapshot, result.elicitation, result.prompt) == ("m-1", "e", "p")
    assert (result.usage, result.attempts) == ({"in": 3}, 2)


def test_a_call_result_checks_nothing_when_built() -> None:
    # Interpret tests it at the write, so a bad value fails its ref or the call by the fault scope,
    # not as a raise inside the binding.
    bad: Any = cast(Any, object())
    result = CallResult(bad, snapshot=bad, elicitation=bad, prompt=bad, usage=bad, attempts=True)
    assert result.values is bad
    assert result.attempts is True


@pytest.mark.parametrize(
    "value",
    [
        CallResult({}),
        SourceCall("s", None, {}, (), {}),
        RenderedQuestion(AnswerKind.noul, "Is it?", ("yes", "no")),
    ],
)
def test_the_binding_types_are_frozen(value: object) -> None:
    field = dataclasses.fields(cast(Any, value))[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, field, None)


def test_a_source_call_holds_what_the_protocol_names() -> None:
    question = RenderedQuestion(AnswerKind.choice, "Which?", ("a", "b"))
    call = SourceCall(
        "ask", {"model": "m"}, {"t:state.subject": "hi"}, ("t:body",), {"q:x": question}
    )
    assert [f.name for f in dataclasses.fields(call)] == [
        "id",
        "payload",
        "needs",
        "refs",
        "questions",
    ]
    assert [f.name for f in dataclasses.fields(question)] == [
        "kind",
        "wording",
        "labels",
        "value_type",
        "is_list",
        "text",
        "reference_date",
    ]
    # The extraction members are keyword-only.
    assert [f.name for f in dataclasses.fields(question) if f.kw_only] == [
        "value_type",
        "is_list",
        "text",
        "reference_date",
    ]
    assert call.questions["q:x"].labels == ("a", "b")
    # A probability kind's question leaves the extraction members at their defaults.
    assert (question.value_type, question.is_list, question.text, question.reference_date) == (
        None,
        False,
        None,
        None,
    )
