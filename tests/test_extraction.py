# pyright: reportPrivateUsage=false
"""The extraction kind: ``ValueType`` and ``ExtractionAnswer`` with its checks and jurat's own
temporal grammars, the extraction question and its load checks, the design document's form,
and the ``extract`` helper."""

import dataclasses
import enum
import math
from typing import Any, cast

import pytest

import extraction as x
import jurat
import jurat._form
from jurat import (
    Answer,
    AnswerKind,
    Cause,
    CauseCode,
    DesignFailure,
    ExtractionAnswer,
    InvalidDesign,
    Parameter,
    Question,
    Rule,
    Source,
    Text,
    Unavailable,
    Unsure,
    ValueType,
    load,
)
from jurat import types as t
from jurat._answer import extraction_document, temporal_ok
from jurat._canonical import decode
from jurat._read_view import ComponentKind, ReadView, Role
from jurat.helpers import agree, extract, mass, pick_with_margin, pick_with_thresholds, size

# ---------------------------------------------------------------------------------------------
# The value types and the answer


def test_the_value_types_are_a_public_strenum_of_six() -> None:
    assert issubclass(ValueType, enum.StrEnum)
    assert [v.value for v in ValueType] == [
        "text",
        "number",
        "date",
        "time",
        "datetime",
        "duration",
    ]
    assert {"ValueType", "ExtractionAnswer"} <= set(jurat.__all__)
    assert AnswerKind.extraction == "extraction"


def answer(**changes: Any) -> ExtractionAnswer:
    fields: dict[str, Any] = {
        "value_type": "number",
        "is_list": False,
        "value": 120,
        "guess": 120,
        "p_null": 0.1,
        "confidence": 0.9,
        "span": [4, 7],
    }
    fields.update(changes)
    return ExtractionAnswer(**fields)


def test_an_answer_holds_its_members_a_number_as_a_float_and_pairs_as_tuples() -> None:
    a = answer()
    assert (a.kind, a.value_type, a.is_list) == (AnswerKind.extraction, ValueType.number, False)
    assert type(a.value) is float
    assert a.value == 120.0
    assert type(a.guess) is float
    assert (a.p_null, a.confidence, a.span) == (0.1, 0.9, (4, 7))
    listed = answer(value_type="date", is_list=True, value=["2024-02-29"], guess=[], span=[[0, 10]])
    assert listed.value == ("2024-02-29",)
    assert listed.guess == ()
    assert listed.span == ((0, 10),)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.value = 1.0  # pyright: ignore[reportAttributeAccessIssue]


def test_the_document_form_holds_every_member_lists_for_tuples() -> None:
    listed = answer(
        value_type="text", is_list=True, value=["a", "b"], guess=None, span=[[0, 1], [2, 3]]
    )
    assert extraction_document(listed) == {
        "kind": "extraction",
        "value_type": "text",
        "is_list": True,
        "value": ["a", "b"],
        "guess": None,
        "p_null": 0.1,
        "confidence": 0.9,
        "span": [[0, 1], [2, 3]],
    }


def test_a_read_back_whole_number_probability_is_taken_as_its_int() -> None:
    a = answer(p_null=0, confidence=1)
    assert (a.p_null, a.confidence) == (0, 1)
    assert type(a.p_null) is int


def test_none_is_a_value_and_an_empty_list_is_distinct_from_it() -> None:
    a = answer(value=None, guess=None, span=None)
    assert (a.value, a.guess, a.span) == (None, None, None)
    empty = answer(value_type="text", is_list=True, value=[], guess=[], span=[])
    assert empty.value == ()
    assert empty.span == ()


@pytest.mark.parametrize(
    ("changes", "says"),
    [
        ({"value_type": "money"}, "value type"),
        ({"value_type": 1}, "value type"),
        ({"is_list": 1}, "is_list"),
        ({"value": True}, "finite number"),
        ({"value": math.nan}, "finite number"),
        ({"value": math.inf}, "finite number"),
        ({"value": 2**53}, "past 2"),
        ({"value": "120"}, "finite number"),
        ({"guess": "120"}, "finite number"),
        ({"p_null": True}, "p_null"),
        ({"p_null": 1.5}, "p_null"),
        ({"p_null": -0.1}, "p_null"),
        ({"confidence": math.nan}, "confidence"),
        ({"confidence": "high"}, "confidence"),
        ({"span": (4, 7)}, "span"),
        ({"span": [4]}, "span"),
        ({"span": [4, 7, 9]}, "span"),
        ({"span": [7, 4]}, "start <= end"),
        ({"span": [-1, 4]}, "start <= end"),
        ({"span": [4.0, 7]}, "exactly an int"),
        ({"span": [True, 7]}, "exactly an int"),
        ({"value": None, "guess": None}, "both are None"),
        ({"value_type": "text", "value": 3}, "string"),
        ({"value_type": "text", "value": "a\ud800"}, "Unicode scalar"),
        ({"value_type": "date", "value": "2023-02-29"}, "date"),
        ({"is_list": True, "value": (1, 2), "guess": None, "span": None}, "exactly a list"),
        ({"is_list": True, "value": [1, None], "guess": None, "span": None}, "no None item"),
        ({"is_list": True, "value": 1, "guess": None, "span": None}, "exactly a list"),
        ({"is_list": True, "value": [1, 2], "guess": None, "span": [[0, 1]]}, "pair count"),
        ({"is_list": True, "value": [1], "guess": None, "span": [0, 1]}, "span"),
        ({"is_list": True, "value": [1], "guess": None, "span": ([0, 1],)}, "exactly a list"),
    ],
)
def test_the_constructor_refuses_each_member_off_its_form(
    changes: dict[str, Any], says: str
) -> None:
    with pytest.raises(ValueError, match=says):
        answer(**changes)


def test_the_span_locates_the_guess_when_the_value_is_none() -> None:
    a = answer(value=None, guess=3, span=[0, 1])
    assert a.span == (0, 1)
    with pytest.raises(ValueError, match="pair count"):
        answer(value_type="text", is_list=True, value=None, guess=["a", "b"], span=[[0, 1]])


@pytest.mark.parametrize(
    ("value_type", "text", "ok"),
    [
        ("date", "2024-02-29", True),
        ("date", "2000-02-29", True),
        ("date", "1900-02-29", False),
        ("date", "2023-02-29", False),
        ("date", "2024-04-31", False),
        ("date", "2024-12-31", True),
        ("date", "2024-13-01", False),
        ("date", "2024-00-10", False),
        ("date", "2024-01-00", False),
        ("date", "2024-1-01", False),
        ("date", "0000-02-29", True),
        ("date", "\uff12024-01-01", False),
        ("date", "٢024-01-01", False),
        ("date", "2024-01-01 ", False),
        ("date", "2024-01-01\n", False),
        ("time", "23:59:59", True),
        ("time", "00:00:00", True),
        ("time", "24:00:00", False),
        ("time", "12:60:00", False),
        ("time", "12:00:60", False),
        ("time", "12:00:00.5", True),
        ("time", "12:00:00.123456789", True),
        ("time", "12:00:00.", False),
        ("time", "12:00:00Z", True),
        ("time", "12:00:00z", False),
        ("time", "12:00:00.25+05:30", True),
        ("time", "12:00:00-23:59", True),
        ("time", "12:00:00+24:00", False),
        ("time", "12:00:00+05:60", False),
        ("time", "12:00:00+0530", False),
        ("time", "12:00", False),
        ("datetime", "2024-02-29T12:00:00Z", True),
        ("datetime", "2024-02-29t12:00:00Z", False),
        ("datetime", "2024-02-29 12:00:00", False),
        ("datetime", "2023-02-29T12:00:00", False),
        ("datetime", "2024-02-29T25:00:00", False),
        ("duration", "P1Y", True),
        ("duration", "P1Y2M3D", True),
        ("duration", "P1M", True),
        ("duration", "PT1M", True),
        ("duration", "P1DT2H3M4S", True),
        ("duration", "PT36H", True),
        ("duration", "P0D", True),
        ("duration", "P007D", True),
        ("duration", "P", False),
        ("duration", "PT", False),
        ("duration", "P1DT", False),
        ("duration", "P1W", False),
        ("duration", "P1.5D", False),
        ("duration", "P1D1Y", False),
        ("duration", "P1Y1Y", False),
        ("duration", "p1d", False),
        ("duration", "PT1H2S3M", False),
        ("duration", "-P1D", False),
    ],
)
def test_the_temporal_grammars_are_jurats_own(value_type: str, text: str, ok: bool) -> None:
    assert temporal_ok(ValueType(value_type), text) is ok
    if ok:
        assert answer(value_type=value_type, value=text, guess=None, span=None).value == text
    else:
        with pytest.raises(ValueError, match=value_type):
            answer(value_type=value_type, value=text, guess=None, span=None)


def test_a_temporal_value_is_kept_as_given_and_a_text_value_by_its_string_data() -> None:
    class Name(enum.StrEnum):
        due = "P007D"

    a = answer(value_type="duration", value=Name.due, guess=None, span=None)
    assert type(a.value) is str
    assert a.value == "P007D"


# ---------------------------------------------------------------------------------------------
# The question and load


def refused(design: Any) -> list[DesignFailure]:
    with pytest.raises(InvalidDesign) as info:
        load(design)
    return list(info.value.failures)


def only(design: Any, check: int, says: str) -> None:
    found = refused(design)
    assert {f.check for f in found} == {check}, found
    assert any(says in f.message for f in found), found


def test_a_valid_extraction_design_loads_and_writes_the_questions_object() -> None:
    loaded = x.loaded()
    provides = decode(loaded.document)["sources"]["read"]["provides"]
    assert decode(loaded.document)["format"] == 3
    assert provides["q:amount"] == {
        "kind": "extraction",
        "wording": {
            "template": "What is owed in {body}?",
            "slots": [{"name": "body", "origin": {"ref": "t:state.body"}}],
        },
        "labels": None,
        "value_type": "number",
        "is_list": False,
        "text": {"ref": "t:state.body"},
        "reference_date": {"ref": "t:state.today"},
    }


def test_a_probability_kinds_question_object_gains_no_member() -> None:
    noul = x.with_question("q:spam", Question("noul", Text("Is it spam?")))
    provides = decode(load(noul).document)["sources"]["read"]["provides"]
    assert set(provides["q:spam"]) == {"kind", "wording", "labels"}


def test_the_text_and_reference_date_refs_are_needs_and_edges() -> None:
    d = x.design(
        sources={
            "read": x.reader(
                **{
                    "q:amount": x.question(
                        text={"ref": "t:mail"}, reference_date={"ref": "d:today"}
                    )
                }
            ),
            "mail": Source({"t:mail": t.string}),
        },
        rules={
            **x.design().rules,
            "today": Rule("d:today", t.string, x.dates, version="1"),
        },
    )
    loaded = load(d)
    # q:dates keeps its text and reference date on the state.
    assert loaded._needs["read"] == frozenset(
        {"t:state.body", "t:state.today", "t:mail", "d:today"}
    )
    assert loaded._graph["read"] == frozenset({"mail", "today"})
    # The design's own needs list holds only what it declared.
    assert decode(loaded.document)["sources"]["read"]["needs"] == ["t:state.body"]


def test_check_3_a_text_or_reference_date_ref_has_a_provider() -> None:
    only(x.with_question("q:amount", x.question(text={"ref": "t:nowhere"})), 3, "no provider")
    only(
        x.with_question("q:amount", x.question(reference_date={"ref": "d:nowhere"})),
        3,
        "no provider",
    )


def test_check_4_a_text_or_reference_date_ref_is_at_t_or_d() -> None:
    q = x.question(text={"ref": "q:dates"})
    only(x.with_question("q:amount", q), 4, "not at level d: or t:")
    q = x.question(reference_date={"ref": "q:dates"})
    only(x.with_question("q:amount", q), 4, "not at level d: or t:")


@pytest.mark.parametrize(
    "state",
    [
        t.record(body=t.integer(), today=t.string),
        t.record(body=t.nullable(t.string), today=t.string),
        t.record(body=t.label("a", "b"), today=t.string),
        t.record(body=t.list_(t.string), today=t.string),
    ],
)
def test_check_9_the_text_ref_is_exactly_string_not_nullable(state: str) -> None:
    d = x.design(state=state)
    found = refused(d)
    assert {f.check for f in found} == {9}, found
    assert any("not exactly string" in f.message and "text" in f.message for f in found)


@pytest.mark.parametrize(
    "today",
    [t.nullable(t.string), t.integer(), t.label("a", "b"), t.list_(t.string)],
)
def test_check_9_the_reference_date_ref_is_exactly_string(today: Any) -> None:
    d = x.design(state=t.record(body=t.string, today=today))
    only(d, 9, f'reference_date names "t:state.today", whose type is {today}, not exactly string')


@pytest.mark.parametrize(
    ("body", "shown"),
    [
        (t.integer(), "integer"),
        (t.nullable(t.string), "string or null"),
        ("words", "words"),
    ],
)
def test_check_9_the_text_refusal_names_the_type_as_written(body: Any, shown: str) -> None:
    d = x.design(types={"words": t.integer()}, state=t.record(body=body, today=t.string))
    only(d, 9, f'names "t:state.body", whose type is {shown}, not exactly string')


def test_a_type_definition_naming_string_is_exactly_string() -> None:
    d = x.design(types={"words": t.string}, state="record{body: words, today: string}")
    load(d)


@pytest.mark.parametrize(
    ("q", "says"),
    [
        (x.question(value_type="money"), "value_type is not text, number"),
        (x.question(value_type=cast(Any, None)), "value_type is not text, number"),
        (x.question(is_list=cast(Any, 1)), "is_list is not a boolean"),
        (x.question(text=None), "text is not a board-ref origin"),
        (x.question(text={"param": "amount/big"}), "text is not a board-ref origin"),
        (x.question(text={"ref": "t:state.body", "x": "y"}), "text is not a board-ref origin"),
        (x.question(text={"ref": 1}), "text is not a board-ref origin"),
        (x.question(reference_date={"candidate": "t:x"}), "reference_date is not null"),
        (x.question(labels=["a"]), "an extraction question and declares labels"),
        (
            Question("noul", Text("Is it?"), value_type="number"),
            "only an extraction question declares",
        ),
        (Question("choice", Text("Which?"), ["a"], is_list=True), "only an extraction question"),
        (
            Question("score", Text("How?"), ["lo"], text={"ref": "t:state.body"}),
            "only an extraction question",
        ),
        (
            Question("noul", Text("Is it?"), reference_date={"ref": "t:state.today"}),
            "only an extraction question",
        ),
        (Question("noul", Text("Is it?"), is_list=cast(Any, 0)), "only an extraction question"),
        (Question("yesno", Text("Is it?")), "noul, choice, score or extraction"),
    ],
)
def test_check_10_each_extraction_member_of_its_form(q: Question, says: str) -> None:
    only(x.with_question("q:amount", q), 10, says)


def test_check_10s_label_bound_does_not_apply_to_an_extraction_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jurat._form, "MAX_LABELS", 0)
    load(x.design())


def test_check_3_an_extraction_q_ref_may_key_the_classes_block() -> None:
    loaded = load(x.design(classes={"q:amount": "identifier"}))
    assert loaded._classes["q:amount"] == {
        "kind": "open",
        "value_type": "open",
        "is_list": "open",
        "value": "identifier",
        "guess": "identifier",
        "p_null": "open",
        "confidence": "open",
        "span": "identifier",
    }
    opened = load(x.design(classes={"q:amount": {"value": "open", "p_null": "content"}}))
    assert opened._classes["q:amount"]["value"] == "open"
    assert opened._classes["q:amount"]["p_null"] == "content"
    assert opened._classes["q:amount"]["guess"] == "content"
    noul = x.with_question("q:spam", Question("noul", Text("Is it?")), classes={"q:spam": "open"})
    only(noul, 3, "an extraction q: ref")


def test_a_classes_key_at_an_extraction_question_check_10_refused_fails_once() -> None:
    money = x.question(cast(Any, "money"))
    only(x.with_question("q:amount", money, classes={"q:amount": "identifier"}), 10, "value_type")


def test_an_extraction_answers_default_tree_is_by_member() -> None:
    assert x.loaded()._classes["q:dates"] == {
        "kind": "open",
        "value_type": "open",
        "is_list": "open",
        "value": "content",
        "guess": "content",
        "p_null": "open",
        "confidence": "open",
        "span": "content",
    }


@pytest.mark.parametrize(
    ("tree", "says"),
    [
        ({"amount": "open"}, "not a member of an extraction answer"),
        ({"value": {"x": "open"}}, 'holds an object at "value"'),
    ],
)
def test_check_10_a_tree_at_an_extraction_ref_conforms_to_its_eight_members(
    tree: Any, says: str
) -> None:
    only(x.design(classes={"q:amount": tree}), 10, says)


def bounded(entry: Any, ref: str = "q:amount", **changes: Any) -> Any:
    d = x.design(**changes)
    rule = dataclasses.replace(cast(Rule, d.rules["amount"]), boundaries={ref: entry})
    return x.replaced(d, rules={**d.rules, "amount": rule})


def test_check_9_an_extraction_read_takes_the_extraction_roles_only() -> None:
    probability: dict[str, list[Any]] = {"bar": [], "thresholds": [], "margin": [], "mass": []}
    only(bounded(probability), 9, "whose roles are null, confidence and value")
    noul = x.with_question("q:amount", Question("noul", Text("Is it?")))
    d = dataclasses.replace(
        cast(Rule, noul.rules["amount"]),
        boundaries={"q:amount": {"null": [], "confidence": [], "value": []}},
    )
    only(
        x.replaced(noul, rules={**noul.rules, "amount": d}),
        9,
        "whose roles are bar, thresholds, margin and mass",
    )


def test_check_9_the_extraction_roles_leaf_types() -> None:
    unit = {"amount/big": Parameter(t.number(), 100), "amount/null": Parameter(t.number(), 0.5)}
    only(
        bounded({"null": ["amount/null"], "confidence": [], "value": []}, parameters=unit),
        9,
        "does not conform to number [0, 1]",
    )
    only(
        bounded({"null": [], "confidence": ["amount/big"], "value": []}),
        9,
        "does not conform to number [0, 1]",
    )
    load(bounded({"null": [], "confidence": ["amount/null"], "value": ["amount/null"]}))


def test_check_9_a_value_role_only_over_a_scalar_number_question() -> None:
    reads = ["q:amount", "q:dates"]
    d = x.design()
    rule = dataclasses.replace(
        cast(Rule, d.rules["amount"]),
        reads=reads,
        boundaries={"q:dates": {"null": [], "confidence": [], "value": ["amount/big"]}},
    )
    only(x.replaced(d, rules={**d.rules, "amount": rule}), 9, "not a scalar number question")
    listed = x.with_question("q:amount", x.question(is_list=True))
    only(
        x.replaced(
            listed,
            rules={
                **listed.rules,
                "amount": dataclasses.replace(
                    cast(Rule, listed.rules["amount"]),
                    boundaries={
                        "q:amount": {"null": [], "confidence": [], "value": ["amount/big"]}
                    },
                ),
            },
        ),
        9,
        "not a scalar number question",
    )


def test_the_extraction_roles_are_written_unordered_and_gridded() -> None:
    entry = {"null": ["amount/null"], "confidence": [], "value": ["amount/big", "amount/null"]}
    loaded = load(bounded(entry))
    written = decode(loaded.document)["rules"]["amount"]["boundaries"]["q:amount"]
    assert written == {
        "null": ["amount/null"],
        "confidence": [],
        "value": ["amount/big", "amount/null"],
    }
    assert dict(loaded._grid["q:amount"]) == {
        "null": ("amount/null",),
        "confidence": (),
        "value": ("amount/big", "amount/null"),
    }
    assert "q:amount" not in load(bounded({"null": [], "confidence": [], "value": []}))._grid


def test_check_1_an_extraction_role_lists_a_leaf_once() -> None:
    entry = {"null": ["amount/null", "amount/null"], "confidence": [], "value": []}
    only(bounded(entry), 1, "twice in its null")


def test_check_10_an_extraction_roles_entry_of_another_form() -> None:
    only(bounded({"null": [], "value": []}), 10, "nor of null, confidence and value")
    only(bounded({"null": "amount/null", "confidence": [], "value": []}), 10, "null is not a list")


def test_the_question_identity_hashes_the_extraction_members_origins() -> None:
    from jurat._interpret import _Turn

    def identity(q: Question) -> str:
        loaded = load(x.with_question("q:dates", q))
        turn = cast(Any, _Turn)(None, loaded, {}, {}, 8, 60, frozenset(), None)
        return cast(str, turn.identity("q:dates"))

    base = identity(x.question())
    assert identity(x.question()) == base
    assert identity(x.question(value_type="text")) != base
    assert identity(x.question(is_list=True)) != base
    assert identity(x.question(reference_date=None)) != base
    assert identity(x.question(text={"ref": "t:state.today"})) != base


# ---------------------------------------------------------------------------------------------
# The design document


def test_a_format_2_document_round_trips_to_its_own_hash() -> None:
    loaded = x.loaded()
    again = load(loaded.document)
    assert again.design_hash == loaded.design_hash
    assert again.document == loaded.document


def test_check_11_a_format_1_document_holding_an_extraction_question_is_refused() -> None:
    doc = decode(x.loaded().document)
    doc["format"] = 1
    doc["rules"]["amount"]["boundaries"] = None
    found = refused(doc)
    assert [(f.check, f.message) for f in found] == [
        (
            11,
            'source "read"\'s "q:amount" is an extraction question, which format 1 does not have',
        )
    ]


def test_check_11_a_format_1_document_holding_the_extraction_roles_is_refused() -> None:
    doc = decode(x.loaded().document)
    doc["format"] = 1
    shown = 'rule "amount"\'s boundaries for "q:amount"'
    found = [(f.check, f.message) for f in refused(doc)]
    assert {check for check, _ in found} == {11}
    assert (11, f'{shown} has no member "bar"') in found


def test_check_11_an_extraction_question_object_holds_exactly_its_members() -> None:
    doc = decode(x.loaded().document)
    del doc["sources"]["read"]["provides"]["q:amount"]["is_list"]
    assert [f.check for f in refused(doc)] == [11]
    doc = decode(x.loaded().document)
    doc["sources"]["read"]["provides"]["q:amount"]["unit"] = "EUR"
    assert [f.check for f in refused(doc)] == [11]
    doc = decode(x.loaded().document)
    entry = doc["rules"]["amount"]["boundaries"]["q:amount"]
    entry["bar"] = []
    assert [f.check for f in refused(doc)] == [11]


# ---------------------------------------------------------------------------------------------
# The extract helper


def view(board: dict[str, object], reads: tuple[str, ...] = ("q:amount",)) -> ReadView:
    return ReadView(Role(ComponentKind.rule, "r"), reads, (), board, {})


def read_amount(**changes: Any) -> ReadView:
    return view({"q:amount": answer(**changes)})


@pytest.mark.parametrize(
    ("p_null", "null_bar", "abstains"),
    [(0.5, 0.5, True), (0.6, 0.5, True), (0.4, 0.5, False), (None, 0.0, False), (0, 0, True)],
)
def test_extract_at_above_and_below_its_null_bar(
    p_null: float | None, null_bar: float, abstains: bool
) -> None:
    r = read_amount(p_null=p_null)
    got = extract(r, "q:amount", null_bar)
    if abstains:
        assert got == Unsure(Cause(CauseCode.abstained, ["q:amount"]))
    else:
        assert got == 120.0
    assert [(i.helper, i.refs, i.rested_on) for i in r.helper_log()] == [
        ("extract", ("q:amount",), ("q:amount",))
    ]


def test_extract_tests_the_bar_exactly_over_the_decimals() -> None:
    # 0.1 + 0.2 in floats is 0.30000000000000004, over 0.3; the record writes each as is.
    assert extract(read_amount(p_null=0.30000000000000004), "q:amount", 0.3) == Unsure(
        Cause(CauseCode.abstained, ["q:amount"])
    )
    assert extract(read_amount(p_null=0.3), "q:amount", 0.30000000000000004) == 120.0


def test_extract_abstains_on_a_none_value_and_never_returns_the_guess() -> None:
    r = read_amount(value=None, guess=99, p_null=None)
    assert extract(r, "q:amount", 1) == Unsure(Cause(CauseCode.abstained, ["q:amount"]))


def test_extract_gives_a_list_as_a_list_and_an_empty_list_as_a_value() -> None:
    r = view(
        {"q:amount": answer(value_type="text", is_list=True, value=["a"], guess=None, span=None)}
    )
    assert extract(r, "q:amount", 0.5) == ["a"]
    r = view({"q:amount": answer(value_type="text", is_list=True, value=[], guess=None, span=None)})
    assert extract(r, "q:amount", 0.5) == []


def test_extract_returns_a_marker_unchanged_after_testing_its_bar() -> None:
    marker = Unavailable(Cause(CauseCode.refused, ["q:amount"]))
    assert extract(view({"q:amount": marker}), "q:amount", 0.5) is marker
    with pytest.raises(ValueError, match="null_bar"):
        extract(view({"q:amount": marker}), "q:amount", 1.5)
    with pytest.raises(TypeError, match="null_bar"):
        extract(view({"q:amount": marker}), "q:amount", True)


def test_each_helper_reads_only_its_kind() -> None:
    noul = Answer("noul", ("yes", "no"), {"yes": 0.9, "no": 0.1})
    with pytest.raises(TypeError, match="holds no extraction answer"):
        extract(view({"q:amount": noul}), "q:amount", 0.5)
    with pytest.raises(TypeError, match="holds no answer"):
        pick_with_margin(read_amount(), "q:amount", 0.5, 0.1)
    with pytest.raises(TypeError, match="holds no answer"):
        mass(read_amount(), "q:amount", ["yes"])
    with pytest.raises(TypeError, match="holds no answer"):
        pick_with_thresholds(read_amount(), "q:amount", {"yes": 0.5}, 0.1)
    with pytest.raises(TypeError, match="holds no list"):
        size(read_amount(), "q:amount")


def test_an_extract_that_raises_on_its_type_test_is_listed_resting_on_nothing() -> None:
    noul = Answer("noul", ("yes", "no"), {"yes": 0.9, "no": 0.1})
    r = view({"q:amount": noul})
    with pytest.raises(TypeError, match="holds no extraction answer"):
        extract(r, "q:amount", 0.5)
    assert [(i.helper, i.refs, i.rested_on) for i in r.helper_log()] == [
        ("extract", ("q:amount",), ())
    ]


def test_a_combinator_compares_extraction_answers_whole_by_canonical_bytes() -> None:
    board: dict[str, object] = {
        "q:amount": answer(),
        "q:same": answer(value=120.0),
        "q:other": answer(p_null=0.2),
    }
    r = view(board, ("q:amount", "q:same", "q:other"))
    assert agree(r, "q:amount", "q:same") is board["q:amount"]
    assert agree(r, "q:amount", "q:other") == Unsure(
        Cause(CauseCode.conflict, ["q:amount", "q:other"])
    )
