# pyright: reportPrivateUsage=false
"""Markers, the outcome Failed, causes and the closed sets."""

import dataclasses
import enum
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import jurat
from jurat._canonical import encode
from jurat._markers import (
    Cause,
    CauseCode,
    Failed,
    MarkerKind,
    OutcomeKind,
    PlanKind,
    Unasked,
    Unavailable,
    Undetermined,
    Unsure,
    VerdictKind,
)

# The cause set, its 23 codes.
CAUSE_SET = {
    "need-unavailable",
    "need-undetermined",
    "when-false",
    "when-failed",
    "not-returned",
    "unexpected-ref",
    "undeclared-read",
    "raised",
    "timed-out",
    "round-bound",
    "deadline",
    "required",
    "refused",
    "invalid-answer",
    "label-collision",
    "conflict",
    "settled",
    "over-budget",
    "invalid-return",
    "design-changed",
    "cancelled",
    "carried",
    "abstained",
    "unsupported",
}


def test_the_cause_set_is_exactly_the_decisions() -> None:
    assert {c.value for c in CauseCode} == CAUSE_SET
    assert len(CauseCode) == 24


@pytest.mark.parametrize(
    ("enum_type", "values"),
    [
        (PlanKind, {"execute", "clarify", "inform", "escalate", "broken"}),
        (OutcomeKind, {"returned", "failed"}),
        (MarkerKind, {"Unsure", "Unavailable", "Unasked"}),
        (VerdictKind, {"passed", "refused"}),
    ],
)
def test_closed_sets(enum_type: type[enum.StrEnum], values: set[str]) -> None:
    assert issubclass(enum_type, enum.StrEnum)
    assert {m.value for m in enum_type} == values


def test_closed_set_members_encode_as_plain_strings() -> None:
    assert (
        encode([CauseCode.need_unavailable, MarkerKind.unsure]) == b'["need-unavailable","Unsure"]'
    )


CAUSE = Cause(CauseCode.need_unavailable, ("q:b", "q:a", "q:b"))


class MyStr(str):
    __slots__ = ()

    def __str__(self) -> str:
        return "overridden"


class MyList(list[str]):
    pass


def test_cause_refs_are_a_tuple_kept_in_order_with_repeats() -> None:
    assert CAUSE.refs == ("q:b", "q:a", "q:b")
    listed = Cause(CauseCode.raised, ["rule:2", "policy"])
    assert listed.refs == ("rule:2", "policy")
    assert type(listed.refs) is tuple


def test_cause_refs_default_to_empty() -> None:
    assert Cause(CauseCode.cancelled).refs == ()


def test_cause_takes_a_code_or_its_string_and_stores_the_member() -> None:
    c = Cause("abstained", ["q:a"])
    assert c.code is CauseCode.abstained
    assert c == Cause(CauseCode.abstained, ("q:a",))
    assert Cause(MyStr("raised")).code is CauseCode.raised


def test_cause_refs_are_stored_as_their_string_data() -> None:
    c = Cause(CauseCode.raised, [MyStr("q:a"), CauseCode.raised])
    assert c.refs == ("q:a", "raised")
    assert all(type(r) is str for r in c.refs)


@pytest.mark.parametrize(
    "code",
    ["", "Raised", "raised ", "abstain", 1, None, ["raised"], MarkerKind.unsure],
    ids=repr,
)
def test_a_code_outside_the_closed_set_is_a_value_error(code: object) -> None:
    with pytest.raises(ValueError, match="cause code"):
        Cause(code)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "refs",
    ["q:a", None, ("q:a", 1), [None], {"q:a"}, iter(["q:a"]), [b"q:a"], MyList(["q:a"])],
    ids=["str", "None", "int member", "None member", "set", "iterator", "bytes", "list subclass"],
)
def test_other_refs_are_a_type_error(refs: object) -> None:
    with pytest.raises(TypeError, match="refs"):
        Cause(CauseCode.raised, refs)  # pyright: ignore[reportArgumentType]


def test_cause_is_a_frozen_dataclass_of_code_and_refs() -> None:
    assert dataclasses.is_dataclass(CAUSE)
    assert [f.name for f in dataclasses.fields(CAUSE)] == ["code", "refs"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        CAUSE.refs = ()  # pyright: ignore[reportAttributeAccessIssue]
    assert hash(CAUSE) == hash(Cause("need-unavailable", ["q:b", "q:a", "q:b"]))


def test_a_cause_encodes_as_the_records_object() -> None:
    c = Cause("abstained", ["q:a"])
    assert encode({"code": c.code, "refs": c.refs}) == b'{"code":"abstained","refs":["q:a"]}'


@pytest.mark.parametrize(
    ("cls", "kind"),
    [
        (Unsure, MarkerKind.unsure),
        (Unavailable, MarkerKind.unavailable),
        (Unasked, MarkerKind.unasked),
    ],
)
def test_marker_kind_and_cause(cls: type[Undetermined], kind: MarkerKind) -> None:
    m = cls(CAUSE)
    assert m.kind is kind
    assert m.cause is CAUSE
    assert isinstance(m, Undetermined)
    assert [f.name for f in dataclasses.fields(m)] == ["kind", "cause"]
    with pytest.raises(TypeError):
        cls(kind, CAUSE)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


def test_markers_are_frozen() -> None:
    m = Unsure(CAUSE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.cause = CAUSE  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(dataclasses.FrozenInstanceError):
        CAUSE.code = CauseCode.raised  # pyright: ignore[reportAttributeAccessIssue]


def test_markers_compare_by_kind_and_cause() -> None:
    assert Unsure(CAUSE) == Unsure(Cause(CauseCode.need_unavailable, ("q:b", "q:a", "q:b")))
    assert Unsure(CAUSE) != Unavailable(CAUSE)
    assert hash(Unsure(CAUSE)) == hash(Unsure(CAUSE))


def test_undetermined_is_not_instantiable() -> None:
    with pytest.raises(TypeError):
        Undetermined(CAUSE)


@pytest.mark.parametrize("cls", [Unsure, Unavailable, Unasked])
def test_a_markers_bool_raises(cls: type[Undetermined]) -> None:
    m = cls(CAUSE)
    with pytest.raises(TypeError):
        bool(m)
    with pytest.raises(TypeError):
        _ = 1 if m else 0  # pyright: ignore[reportGeneralTypeIssues]


@pytest.mark.parametrize("cls", [Unsure, Unavailable, Unasked, Failed])
@pytest.mark.parametrize(
    "cause",
    [None, "abstained", {"code": "abstained", "refs": []}, ("abstained", ()), CauseCode.raised],
    ids=["None", "str", "dict", "tuple", "code"],
)
def test_a_carrier_takes_only_a_cause(cls: type[Undetermined | Failed], cause: object) -> None:
    with pytest.raises(TypeError, match="Cause"):
        cls(cause)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("cls", [Unsure, Unavailable, Unasked, Failed])
def test_a_carrier_requires_its_cause(cls: type[Undetermined | Failed]) -> None:
    with pytest.raises(TypeError):
        cls()  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize("cls", [Unsure, Unavailable, Unasked, Failed])
def test_kind_is_not_a_constructor_parameter(cls: type[Undetermined | Failed]) -> None:
    with pytest.raises(TypeError):
        cls(cause=CAUSE, kind="Unsure")  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


def test_no_constructor_checks_a_code_against_its_carrier() -> None:
    # The cause set carries `conflict` by Unsure only; the constructor does not check.
    assert Unasked(Cause(CauseCode.conflict)).cause.code is CauseCode.conflict
    assert Failed(Cause(CauseCode.abstained)).cause.code is CauseCode.abstained


def test_failed_is_the_whole_call() -> None:
    assert Failed(Cause("timed-out", ["s"])) == Failed(Cause(CauseCode.timed_out, ("s",)))


def test_failed() -> None:
    f = Failed(Cause(CauseCode.timed_out, ("s",)))
    assert f.kind is OutcomeKind.failed
    assert f.cause.code is CauseCode.timed_out
    assert [x.name for x in dataclasses.fields(f)] == ["kind", "cause"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.cause = CAUSE  # pyright: ignore[reportAttributeAccessIssue]


def test_public_names() -> None:
    public = ["Undetermined", "Unsure", "Unavailable", "Unasked", "Failed", "Cause"]
    for name in [*public, "CauseCode", "PlanKind", "VerdictKind"]:
        assert name in jurat.__all__
        assert getattr(jurat, name) is globals()[name]
    # A marker's kind and an outcome's are named by their classes.
    for name in ["MarkerKind", "OutcomeKind"]:
        assert name not in jurat.__all__
        assert not hasattr(jurat, name)


# --- pyright reports a marker tested for truth ---------------------------------------------

FIXTURE = """\
# pyright: strict
from jurat import Unsure


def f(marker: Unsure, x: str | Unsure) -> None:
    if marker:  # line 6
        pass
    if x:  # line 8
        pass
"""


def _pyright() -> list[str]:
    exe = shutil.which("pyright", path=str(Path(sys.executable).parent))
    assert exe is not None, "pyright, a dev dependency, is installed beside the interpreter"
    return [exe]


def test_pyright_reports_a_marker_tested_for_truth(tmp_path: Path) -> None:
    src = tmp_path / "fixture.py"
    src.write_text(FIXTURE)
    result = subprocess.run(
        [*_pyright(), "--outputjson", "--pythonpath", sys.executable, str(src)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    report = json.loads(result.stdout)
    diagnostics = report["generalDiagnostics"]
    by_line = {d["range"]["start"]["line"] + 1: d for d in diagnostics}
    assert {6, 8} <= set(by_line), diagnostics
    for line in (6, 8):
        d = by_line[line]
        assert d.get("rule") == "reportGeneralTypeIssues", d
        assert "__bool__" in d["message"] or "NoReturn" in d["message"], d


class Unprintable(str):
    """A value whose own repr raises; a str subclass so it reaches the string paths too."""

    __slots__ = ()

    def __repr__(self) -> str:
        raise RuntimeError("repr was called")


class Opaque:
    def __repr__(self) -> str:
        raise RuntimeError("repr was called")


@pytest.mark.parametrize(
    ("build", "error"),
    [
        (lambda: Cause(Opaque()), ValueError),  # type: ignore[arg-type]
        (lambda: Cause(Unprintable("nope")), ValueError),
        (lambda: Cause(CauseCode.raised, Opaque()), TypeError),  # type: ignore[arg-type]
        (lambda: Cause(CauseCode.raised, [Opaque()]), TypeError),  # type: ignore[list-item]
        (lambda: Unsure(Opaque()), TypeError),  # type: ignore[arg-type]
    ],
    ids=["code", "code-str", "refs", "ref", "carrier"],
)
def test_a_bad_value_whose_repr_raises_is_the_usual_error(build: Any, error: type) -> None:
    with pytest.raises(error):
        build()
