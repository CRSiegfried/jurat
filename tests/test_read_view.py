# pyright: reportPrivateUsage=false
"""The read view: reads, `require`, `r.param`, the stop record and the role."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any

import pytest

from counting import SmallBounds
from jurat import Answer, Cause, CauseCode, Unavailable, Unsure
from jurat import types as t
from jurat._read_view import (
    Access,
    ComponentKind,
    ReadView,
    RequireStopped,
    Role,
    UndeclaredRead,
    _RequireStop,
    _Stop,
    _UndeclaredStop,
    _ViewError,
    close_call,
    open_call,
)
from jurat._types import TypeFailure, parse_type

INTENT = Answer("noul", ["yes", "no"], {"yes": 0.7, "no": 0.3})
UNSURE = Unsure(Cause(CauseCode.abstained, ["q:intent"]))
GONE = Unavailable(Cause(CauseCode.raised, ["d:gone"]))
STATE = {"subject": "hi", "sender": {"address": "a@b"}, "reply": None}
ORDER = {"status": "open", "owner": None, "line": {"sku": None}}

BOARD: dict[str, object] = {
    "t:state": STATE,
    "q:intent": INTENT,
    "d:vague": UNSURE,
    "d:gone": GONE,
    "t:order": ORDER,
    "t:empty": None,
}
LEAVES: dict[str, object] = {"intent/bar": 0.6, "intent/labels": ["yes"]}
READS = ("q:intent", "d:vague", "d:gone", "t:empty", "t:state.subject", "t:order")


def view(
    reads: Sequence[str] = READS,
    params: Sequence[str] = ("intent/bar",),
    board: Mapping[str, object] = BOARD,
    leaves: Mapping[str, object] = LEAVES,
    kind: ComponentKind = ComponentKind.rule,
    name: str = "route",
) -> ReadView:
    assert not isinstance(WHEN_STATE_T, TypeFailure)
    state = WHEN_STATE_T if kind is ComponentKind.when else None
    return ReadView(Role(kind, name), reads, params, board, leaves, state=state)


def stopped(r: ReadView, call: Callable[[], object]) -> None:
    """Run ``call`` on ``r``, which must stop it with a stop no ``except Exception`` catches."""
    with pytest.raises(_Stop) as info:
        call()
    assert not isinstance(info.value, Exception)
    assert info.value.stop == r.stops[-1]


# ---------------------------------------------------------------------------------------------
# The read set is matched exactly


def test_a_declared_read_returns_the_boards_value() -> None:
    r = view()
    assert r.read("q:intent") is INTENT
    assert r.read("t:order") is ORDER
    assert r.stops == ()


def test_a_declared_read_returns_the_boards_marker() -> None:
    r = view()
    assert r.read("d:vague") is UNSURE
    assert r.read("d:gone") is GONE
    assert r.stops == ()


def test_a_ref_outside_the_read_set_is_an_undeclared_read() -> None:
    r = view()
    stopped(r, lambda: r.read("d:other"))
    assert r.stops == (UndeclaredRead.of("d:other", Access.read),)


def test_a_dot_path_read_is_undeclared_when_only_its_base_is_declared() -> None:
    r = view(reads=("t:state",))
    stopped(r, lambda: r.read("t:state.subject"))
    assert r.stops == (UndeclaredRead.of("t:state.subject", Access.read),)


def test_a_whole_read_is_undeclared_when_only_a_dot_path_is_declared() -> None:
    r = view(reads=("t:state.subject",))
    assert r.read("t:state.subject") == "hi"
    stopped(r, lambda: r.read("t:state"))
    assert r.stops == (UndeclaredRead.of("t:state", Access.read),)


@pytest.mark.parametrize(
    ("declared", "read"), [("t:order", "t:order.status"), ("t:order.status", "t:order")]
)
def test_a_field_path_is_in_the_read_set_only_where_declared(declared: str, read: str) -> None:
    r = view(reads=(declared,))
    r.read(declared)
    stopped(r, lambda: r.read(read))
    assert r.stops == (UndeclaredRead.of(read, Access.read),)


def test_a_read_that_is_not_a_string_is_recorded_as_its_repr_at_the_stop() -> None:
    r = view()
    key = ["q:intent"]
    for read in (42, None, key):
        stopped(r, lambda read=read: r.read(read))  # type: ignore[arg-type]
    key.append("changed after the stop")
    assert r.stops == (
        UndeclaredRead("42", False, Access.read),
        UndeclaredRead("None", False, Access.read),
        UndeclaredRead("['q:intent']", False, Access.read),
    )


class PosesAsRef:
    def __repr__(self) -> str:
        return "q:intent"


class Unrepresentable:
    def __repr__(self) -> str:
        raise RuntimeError("no repr")


def test_a_non_string_read_is_never_taken_for_a_string() -> None:
    r = view()
    stopped(r, lambda: r.read(PosesAsRef()))  # type: ignore[arg-type]
    assert r.stops == (UndeclaredRead("<test_read_view.PosesAsRef>", False, Access.read),)


def nested(depth: int) -> list[object]:
    value: list[object] = []
    for _ in range(depth - 1):
        value = [value]
    return value


def test_a_read_nesting_too_deep_is_unrepresentable() -> None:
    r = view()
    stopped(r, lambda: r.read(nested(65)))  # type: ignore[arg-type]
    assert r.stops == (UndeclaredRead("<unrepresentable>", False, Access.read),)


def test_a_read_whose_repr_raises_is_named_by_its_type_its_repr_never_run() -> None:
    r = view()
    stopped(r, lambda: r.read(Unrepresentable()))  # type: ignore[arg-type]
    assert r.stops == (UndeclaredRead("<test_read_view.Unrepresentable>", False, Access.read),)


class Sly(str):
    """A ``str`` subclass whose overrides would fool a set lookup."""

    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return hash("q:intent")


def test_a_str_subclass_read_is_matched_by_its_string_data() -> None:
    r = view()
    assert r.read(Sly("q:intent")) is INTENT
    sly = Sly("d:other")
    stopped(r, lambda: r.read(sly))
    assert r.stops == (UndeclaredRead("d:other", True, Access.read),)
    (stop,) = r.stops
    assert isinstance(stop, UndeclaredRead)
    assert type(stop.read) is str


def test_a_str_subclass_param_is_matched_by_its_string_data() -> None:
    r = view()
    assert r.param(Sly("intent/bar")) == 0.6
    stopped(r, lambda: r.param(Sly("intent/labels")))
    assert r.stops == (UndeclaredRead("intent/labels", True, Access.param),)
    assert all(type(s.read) is str for s in r.stops)  # type: ignore[union-attr]


def test_a_require_stop_records_a_str_subclass_ref_by_its_string_data() -> None:
    r = view()
    stopped(r, lambda: r.require(Sly("d:gone")))
    (stop,) = r.stops
    assert stop == RequireStopped("d:gone", GONE)
    assert type(stop.ref) is str  # type: ignore[union-attr]


def test_a_declared_parameter_is_not_a_readable_ref() -> None:
    r = view()
    stopped(r, lambda: r.read("intent/bar"))
    assert r.stops == (UndeclaredRead.of("intent/bar", Access.read),)


# ---------------------------------------------------------------------------------------------
# What a read returns


def test_a_dot_path_reads_through_nesting() -> None:
    r = view(reads=("t:state.subject", "t:state.sender.address"))
    assert r.read("t:state.subject") == "hi"
    assert r.read("t:state.sender.address") == "a@b"


def test_a_marker_at_the_base_of_a_dot_path_is_returned() -> None:
    r = view(reads=("d:gone.status",))
    assert r.read("d:gone.status") is GONE


def test_a_null_on_the_path_makes_the_read_null() -> None:
    r = view(reads=("t:order.owner", "t:order.line.sku", "t:empty.status", "t:order.owner.name"))
    assert r.read("t:order.owner") is None  # null at the leaf
    assert r.read("t:order.line.sku") is None  # null at the leaf, a level down
    assert r.read("t:empty.status") is None  # null at the base
    assert r.read("t:order.owner.name") is None  # null mid-path
    assert r.stops == ()


# ---------------------------------------------------------------------------------------------
# `require`


def test_require_returns_a_value() -> None:
    r = view()
    assert r.require("q:intent") is INTENT
    assert r.stops == ()


def test_require_on_null_does_not_stop() -> None:
    r = view(reads=("t:empty", "t:order.owner.name"))
    assert r.require("t:empty") is None
    assert r.require("t:order.owner.name") is None
    assert r.stops == ()


@pytest.mark.parametrize("ref", ["d:vague", "d:gone"])
def test_require_on_a_marker_stops_and_records_its_ref_and_marker(ref: str) -> None:
    r = view()
    stopped(r, lambda: r.require(ref))
    assert r.stops == (RequireStopped(ref, BOARD[ref]),)  # type: ignore[arg-type]
    assert r.stops[0].marker is BOARD[ref]  # type: ignore[union-attr]


def test_a_require_stop_records_the_read_as_given_with_its_path() -> None:
    r = view(reads=("d:gone.status",))
    stopped(r, lambda: r.require("d:gone.status"))
    assert r.stops == (RequireStopped("d:gone.status", GONE),)


def test_require_outside_the_read_set_is_an_undeclared_read() -> None:
    r = view()
    stopped(r, lambda: r.require("d:other"))
    assert r.stops == (UndeclaredRead.of("d:other", Access.read),)


# ---------------------------------------------------------------------------------------------
# `r.param`


def test_param_returns_a_declared_leafs_value() -> None:
    r = view()
    assert r.param("intent/bar") == 0.6
    assert r.stops == ()


def test_param_of_an_undeclared_leaf_is_an_undeclared_read() -> None:
    r = view()
    stopped(r, lambda: r.param("intent/labels"))  # a leaf of the tree, not declared
    stopped(r, lambda: r.param("q:intent"))  # a declared read is not a parameter
    stopped(r, lambda: r.param(["x"]))  # type: ignore[arg-type]
    assert r.stops == (
        UndeclaredRead.of("intent/labels", Access.param),
        UndeclaredRead.of("q:intent", Access.param),
        UndeclaredRead.of(["x"], Access.param),
    )


# ---------------------------------------------------------------------------------------------
# The stop record


def test_except_exception_does_not_catch_a_stop() -> None:
    r = view()

    def rule(read: Callable[[], object]) -> object:
        try:
            return read()
        except Exception:  # what consumer code may write
            return "swallowed"

    for read in (lambda: r.read("d:other"), lambda: r.require("d:gone")):
        with pytest.raises(_Stop) as info:
            rule(read)
        assert not isinstance(info.value, Exception)
    assert [type(s) for s in r.stops] == [UndeclaredRead, RequireStopped]


def test_both_stops_derive_from_the_one_stop_class() -> None:
    r = view()
    with pytest.raises(_UndeclaredStop):
        r.read("d:other")
    with pytest.raises(_RequireStop):
        r.require("d:gone")
    assert issubclass(_UndeclaredStop, _Stop)
    assert issubclass(_RequireStop, _Stop)
    assert not issubclass(_Stop, Exception)


def test_a_caught_stop_is_still_recorded() -> None:
    r = view()

    def rule() -> object:
        try:
            return r.require("d:gone")
        except BaseException:
            return "yes"

    assert rule() == "yes"
    assert r.stops == (RequireStopped("d:gone", GONE),)
    assert r.deciding_stop() == RequireStopped("d:gone", GONE)


def test_two_undeclared_reads_are_both_recorded_in_order() -> None:
    r = view()
    for read in ("d:one", "d:two"):
        with suppress(_Stop):
            r.read(read)
    assert r.stops == (
        UndeclaredRead.of("d:one", Access.read),
        UndeclaredRead.of("d:two", Access.read),
    )


def test_the_deciding_stop_is_the_first_undeclared_read_else_the_first_require_stop() -> None:
    r = view()
    assert r.deciding_stop() is None
    for call in (
        lambda: r.require("d:vague"),
        lambda: r.require("d:gone"),
        lambda: r.read("d:one"),
        lambda: r.param("x/y"),
    ):
        with suppress(_Stop):
            call()
    assert len(r.stops) == 4
    assert r.deciding_stop() == UndeclaredRead.of("d:one", Access.read)

    only_requires = view()
    for ref in ("d:gone", "d:vague"):
        with suppress(_Stop):
            only_requires.require(ref)
    assert only_requires.deciding_stop() == RequireStopped("d:gone", GONE)


def test_the_stop_record_is_handed_out_as_a_copy() -> None:
    r = view()
    with suppress(_Stop):
        r.read("d:one")
    before = r.stops
    with suppress(_Stop):
        r.read("d:two")
    assert len(before) == 1
    assert len(r.stops) == 2


# ---------------------------------------------------------------------------------------------
# The view's role, and its inputs


@pytest.mark.parametrize(
    ("kind", "name"),
    [(ComponentKind.rule, "route"), (ComponentKind.policy, "policy"), (ComponentKind.when, "mail")],
)
def test_the_view_says_which_component_it_serves(kind: ComponentKind, name: str) -> None:
    r = view(kind=kind, name=name, params=() if kind is ComponentKind.when else ("intent/bar",))
    assert r.role == Role(kind, name)


def test_a_role_is_a_closed_kind_and_a_string_name() -> None:
    with pytest.raises(ValueError, match="kind"):
        Role("source", "x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="name"):
        Role(ComponentKind.rule, 7)  # type: ignore[arg-type]
    assert Role("rule", "route").kind is ComponentKind.rule  # type: ignore[arg-type]


def test_a_declared_read_missing_from_the_board_is_an_engine_defect() -> None:
    with pytest.raises(_ViewError, match="d:missing") as info:
        view(reads=("d:missing",))
    assert isinstance(info.value, Exception)  # not a stop: nothing records it
    with pytest.raises(_ViewError, match="d:missing"):
        view(reads=("d:missing.field",))


@pytest.mark.parametrize("ref", ['t:order["a"]', 't:order["a"].status', "order", "t:Order"])
def test_a_declared_read_that_is_grounded_or_no_ref_is_an_engine_defect(ref: str) -> None:
    with pytest.raises(_ViewError, match=r"grounded|not a ref"):
        view(reads=(ref,))


def test_a_helper_cannot_list_a_ref_outside_the_read_set() -> None:
    r = view()
    with pytest.raises(_ViewError, match="d:other"):
        open_call(r, "first", ["q:intent", "d:other"])
    assert r.helper_log() == ()


def test_a_declared_parameter_missing_from_the_leaves_is_an_engine_defect() -> None:
    with pytest.raises(_ViewError, match="a/b"):
        view(params=("a/b",))


def test_a_path_through_a_value_that_is_not_a_record_is_an_engine_defect() -> None:
    r = view(reads=("t:state.subject.first",))
    with pytest.raises(_ViewError, match=r"t:state\.subject\.first"):
        r.read("t:state.subject.first")
    r = view(reads=("t:state.missing",))
    with pytest.raises(_ViewError, match=r"t:state\.missing"):
        r.read("t:state.missing")


# ---------------------------------------------------------------------------------------------
# A `when`'s view; its state paths are the full pass's


def test_a_when_reads_its_needs_and_the_state_whole() -> None:
    r = view(reads=("q:intent",), params=(), kind=ComponentKind.when, name="mail")
    assert r.read("q:intent") is INTENT
    assert r.read("t:state") is STATE
    stopped(r, lambda: r.read("d:vague"))
    assert r.stops == (UndeclaredRead.of("d:vague", Access.read),)


def test_a_when_reads_a_state_path_its_needs_declare() -> None:
    r = view(reads=("t:state.subject",), params=(), kind=ComponentKind.when, name="mail")
    assert r.read("t:state.subject") == "hi"
    assert r.read("t:state") is STATE
    assert r.stops == ()


def test_a_when_param_is_always_an_undeclared_read() -> None:
    r = view(reads=(), params=(), kind=ComponentKind.when, name="mail")
    stopped(r, lambda: r.param("intent/bar"))
    assert r.stops == (UndeclaredRead.of("intent/bar", Access.param),)


def test_a_when_view_declares_no_parameters() -> None:
    with pytest.raises(_ViewError, match="when"):
        view(params=("intent/bar",), kind=ComponentKind.when, name="mail")


# A `when` reads `t:state` by any field path its type gives

WHEN_STATE_T = parse_type(
    t.record(
        subject=t.string,
        sender=t.record(address=t.string),
        reply=t.nullable(t.record(text=t.string)),
        tags=t.list_(t.record(a=t.string)),
        meta=t.mapping(t.record(a=t.string)),
        blob=t.bytes_,
    ),
    nullable=False,
)
WHEN_STATE = {
    "subject": "hi",
    "sender": {"address": "a@b"},
    "reply": None,
    "tags": [{"a": "x"}],
    "meta": {"k": {"a": "y"}},
    "blob": b"\x00",
}
WHEN_BOARD: dict[str, object] = {**BOARD, "t:state": WHEN_STATE}


def when_view(*needs: str, board: Mapping[str, object] = WHEN_BOARD) -> ReadView:
    assert not isinstance(WHEN_STATE_T, TypeFailure)
    role = Role(ComponentKind.when, "mail")
    return ReadView(role, needs, (), board, {}, state=WHEN_STATE_T)


def test_a_when_reads_a_state_field_path_it_does_not_declare() -> None:
    r = when_view()
    assert r.read("t:state.sender.address") == "a@b"
    assert r.read("t:state.sender") is WHEN_STATE["sender"]
    assert r.read("t:state") is WHEN_STATE
    assert r.require("t:state.subject") == "hi"
    assert r.stops == ()


def test_a_null_on_a_whens_state_path_reads_null_and_require_does_not_stop() -> None:
    r = when_view()
    assert r.read("t:state.reply.text") is None
    assert r.require("t:state.reply.text") is None
    assert r.stops == ()


@pytest.mark.parametrize(
    "read",
    [
        "t:state.meta.k",  # through a mapping, though the value holds the key
        "t:state.tags.a",  # through a list
        "t:state.subject.first",  # through a scalar
        "t:state.blob.x",  # through bytes
        "t:state.nope",  # a field the type does not have
        "t:state.sender.nope",
        "t:state.",
        "t:state..a",
        't:state["x"]',
        't:state["x"].subject',
        "t:state[x]",
        "t:stated.subject",
    ],
)
def test_a_state_path_the_type_does_not_give_is_a_whens_undeclared_read(read: str) -> None:
    r = when_view()
    stopped(r, lambda: r.read(read))
    assert r.stops == (UndeclaredRead(read, True, Access.read),)


def test_a_whens_state_path_is_matched_by_its_string_data_and_a_non_string_by_repr() -> None:
    r = when_view()
    assert r.read(Sly("t:state.sender.address")) == "a@b"
    stopped(r, lambda: r.read(Sly("t:state.nope")))
    stopped(r, lambda: r.read(("t:state",)))  # type: ignore[arg-type]
    assert r.stops == (
        UndeclaredRead("t:state.nope", True, Access.read),
        UndeclaredRead("('t:state',)", False, Access.read),
    )


def test_a_declared_state_path_and_the_same_path_through_the_type_read_alike() -> None:
    declared = when_view("t:state.sender")
    through = when_view()
    assert declared.read("t:state.sender") is through.read("t:state.sender")


def test_only_t_state_widens_in_a_when() -> None:
    r = when_view("t:order")
    assert r.read("t:order") is ORDER
    stopped(r, lambda: r.read("t:order.status"))
    assert r.stops == (UndeclaredRead("t:order.status", True, Access.read),)


def test_a_whens_param_and_a_caught_stop_are_recorded() -> None:
    r = when_view()
    stopped(r, lambda: r.param("intent/bar"))
    with suppress(_Stop):
        r.read("t:state.nope")
    assert r.stops == (
        UndeclaredRead("intent/bar", True, Access.param),
        UndeclaredRead("t:state.nope", True, Access.read),
    )


def test_a_path_the_type_gives_over_an_entry_lacking_it_is_a_defect_not_a_stop() -> None:
    r = when_view(board={**BOARD, "t:state": {"subject": "hi"}})
    with pytest.raises(_ViewError, match=r"t:state\.sender\.address"):
        r.read("t:state.sender.address")
    assert r.stops == ()


def test_only_a_whens_view_takes_the_states_type() -> None:
    assert not isinstance(WHEN_STATE_T, TypeFailure)
    with pytest.raises(_ViewError, match="when"):
        ReadView(Role(ComponentKind.rule, "r"), (), (), BOARD, {}, state=WHEN_STATE_T)


# ---------------------------------------------------------------------------------------------
# The helper log


def test_the_helper_log_keeps_calls_in_order_and_is_handed_out_frozen() -> None:
    r = view()
    assert r.helper_log() == ()
    first = open_call(r, "pick_with_margin", ["q:intent"])
    close_call(r, first, ["q:intent"])
    second = open_call(r, "first", ["d:vague", "d:gone"])
    close_call(r, second, None)
    log = r.helper_log()
    assert [(i.helper, i.refs, i.rested_on) for i in log] == [
        ("pick_with_margin", ("q:intent",), ("q:intent",)),
        ("first", ("d:vague", "d:gone"), ()),
    ]
    with pytest.raises(AttributeError):
        log[0].rested_on = ("x",)  # type: ignore[misc]
    assert r.helper_log() == log


def test_an_item_is_listed_when_opened_before_it_returns() -> None:
    r = view()
    index = open_call(r, "size", ["t:order"])
    snapshot = r.helper_log()
    assert [(i.helper, i.rested_on) for i in snapshot] == [("size", ())]
    close_call(r, index, ["t:order"])
    assert snapshot[0].rested_on == ()  # the copy handed out does not change
    assert r.helper_log()[0].rested_on == ("t:order",)


def test_a_call_nested_inside_a_helper_is_not_listed() -> None:
    r = view()
    outer = open_call(r, "first", ["d:vague"])
    inner = open_call(r, "size", ["t:order"])
    assert inner is None
    close_call(r, inner, ["t:order"])
    close_call(r, outer, ["d:vague"])
    after = open_call(r, "size", ["t:order"])
    close_call(r, after, ["t:order"])
    assert [i.helper for i in r.helper_log()] == ["first", "size"]


def test_a_view_of_any_component_starts_with_an_empty_log_and_record() -> None:
    def rule(r: Any) -> object:
        return r.read("q:intent")

    r = view()
    rule(r)
    assert r.helper_log() == ()
    assert r.stops == ()


def test_a_whens_view_without_the_states_type_is_a_defect() -> None:
    with pytest.raises(_ViewError, match="state"):
        ReadView(Role(ComponentKind.when, "mail"), (), (), BOARD, {})


def test_a_whens_read_of_a_bracket_nesting_past_the_recursion_limit_is_undeclared() -> None:
    read = "t:state[" + "[" * 100_000
    r = when_view()
    stopped(r, lambda: r.read(read))
    assert r.stops == (UndeclaredRead(read, True, Access.read),)


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
        (lambda: Role(Opaque(), "r"), ValueError),  # type: ignore[arg-type]
        (lambda: Role(Unprintable("nope"), "r"), ValueError),  # type: ignore[arg-type]
        (lambda: Role(ComponentKind.rule, Opaque()), TypeError),  # type: ignore[arg-type]
    ],
    ids=["kind", "kind-str", "name"],
)
def test_a_bad_role_value_whose_repr_raises_is_the_usual_error(build: Any, error: type) -> None:
    with pytest.raises(error):
        build()


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
def test_a_role_name_is_judged_by_its_type(make: Callable[[], object]) -> None:
    with pytest.raises(TypeError):
        Role(ComponentKind.rule, make())  # type: ignore[arg-type]


class SpoofsStr:
    """Claims to be a ``str`` through ``__class__``; only its type is read."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return str

    def __repr__(self) -> str:
        return "SpoofsStr()"


def _deep_frozenset(depth: int) -> object:
    value: object = 0
    for _ in range(depth):
        value = frozenset({value})
    return value


@pytest.mark.parametrize(
    "read",
    [["x" * 16] * 17, _deep_frozenset(65), {("k",) * 70: 0}],
    ids=["past-the-size-bound", "deep-frozenset", "many-keys"],
)
def test_a_non_string_read_past_a_bound_is_unrepresentable(
    read: object, small_bounds: SmallBounds
) -> None:
    r = view()
    stopped(r, lambda: r.read(read))  # type: ignore[arg-type]
    assert r.stops == (UndeclaredRead("<unrepresentable>", False, Access.read),)


def test_a_read_is_a_string_by_its_type_not_its_class_attribute() -> None:
    r = view()
    stopped(r, lambda: r.read(SpoofsStr()))  # type: ignore[arg-type]
    assert r.stops == (UndeclaredRead("<test_read_view.SpoofsStr>", False, Access.read),)


def test_a_param_is_a_string_by_its_type_not_its_class_attribute() -> None:
    r = view()
    stopped(r, lambda: r.param(SpoofsStr()))  # type: ignore[arg-type]
    assert r.stops == (UndeclaredRead("<test_read_view.SpoofsStr>", False, Access.param),)
