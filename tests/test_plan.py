# pyright: reportPrivateUsage=false
"""The plan types: each constructor
checks its fields' types and nothing else, so a plan that breaks a plan rule is built,
and only the engine writes a plan's cause and slots and an origin's set."""

import copy
import dataclasses
import enum
import pickle
from collections.abc import Callable
from typing import Any, cast

import pytest

from jurat import Binding, Cause, CauseCode, Origin, Plan, PlanKind, Requirement, Step
from jurat._plan import candidate_literal, engine_plan


class Name(str):
    __slots__ = ()

    def __str__(self) -> str:
        return "overridden"


class Kind(enum.StrEnum):
    run = "execute"


# ---------------------------------------------------------------------------------------------
# Plans that break a plan rule are built (probes a and e)


@pytest.mark.parametrize(
    "build",
    [
        # probe a: an escalate plan carrying a step (rule 1)
        lambda: Plan("escalate", [Step("delete_all")], reason="hand on"),
        # probe e: a step binding a later step (rule 2)
        lambda: Plan("execute", [Step("t", {"x": Binding(5)})]),
        lambda: Plan("execute", [Step("t", {"x": Binding(1, "a..b")})]),
        lambda: Plan("execute", [Step("nope", {"idempotency_key": Origin(literal=1)})]),
        lambda: Plan("execute", [Step("t", {"x": Origin(ref="d:undeclared")})]),
        lambda: Plan("inform", reply="   "),
        lambda: Plan("broken"),
        lambda: Plan("execute", [Step("t", requirements=[Requirement(cast(Any, 1), b"x")])]),
        lambda: Plan("execute", [Step("t", {"x": Origin(literal=b"bytes")})]),
        lambda: Plan("execute", reason="\ud800"),
        lambda: Plan("execute", [Step("t", {"\ud800": Origin(literal=1)})]),
    ],
)
def test_a_plan_that_breaks_a_plan_rule_is_built(build: Any) -> None:
    assert type(build()) is Plan


# ---------------------------------------------------------------------------------------------
# Plan


def test_a_plans_fields_and_defaults() -> None:
    step = Step("archive")
    plan = Plan("execute", [step], reason="why")
    assert tuple(f.name for f in dataclasses.fields(plan)) == (
        "kind",
        "steps",
        "reason",
        "reply",
        "cause",
        "slots",
    )
    assert plan.kind is PlanKind.execute
    assert plan.steps == (step,)
    assert type(plan.steps) is tuple
    assert (plan.reason, plan.reply, plan.cause) == ("why", None, None)
    assert dict(plan.slots) == {}
    with pytest.raises(TypeError):
        plan.slots["a"] = 1  # type: ignore[index]


def test_a_kind_is_a_member_or_its_string_stored_as_the_member() -> None:
    assert Plan(PlanKind.inform, reply="x").kind is PlanKind.inform
    assert Plan(Kind.run).kind is PlanKind.execute
    assert Plan(Name("clarify"), reply="x").kind is PlanKind.clarify


@pytest.mark.parametrize("kind", ["Execute", "refused", 1, None])
def test_a_kind_outside_the_closed_set_is_a_value_error(kind: Any) -> None:
    with pytest.raises(ValueError, match="not in the closed set"):
        Plan(kind)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("execute", "steps"), {}),
        (("execute", (s for s in [Step("t")])), {}),
        (("execute", [object()]), {}),
        (("execute", [{"tool": "t"}]), {}),
        (("execute",), {"reason": 1}),
        (("execute",), {"reply": b"x"}),
    ],
)
def test_a_plan_field_of_the_wrong_type_is_a_type_error(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(TypeError):
        Plan(*args, **kwargs)


def test_the_public_constructor_takes_no_cause_and_no_slots() -> None:
    with pytest.raises(TypeError):
        Plan("broken", cause=Cause(CauseCode.raised, ["policy"]))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Plan("inform", reply="x", slots={"a": 1})  # type: ignore[call-arg]


def test_a_string_subclass_is_stored_as_its_string_data() -> None:
    plan = Plan("clarify", [], reason=Name("why"), reply=Name("what"))
    assert (type(plan.reason), type(plan.reply)) == (str, str)
    assert (plan.reason, plan.reply) == ("why", "what")


def test_the_engine_sets_a_cause_through_its_own_constructor() -> None:
    cause = Cause(CauseCode.raised, ["policy"])
    plan = engine_plan(PlanKind.broken, cause)
    assert (plan.kind, plan.steps, plan.reason, plan.reply, plan.cause) == (
        PlanKind.broken,
        (),
        None,
        None,
        cause,
    )


def test_a_plan_is_frozen() -> None:
    plan = Plan("execute")
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.kind = PlanKind.broken  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------
# Step


def test_a_steps_fields_and_defaults() -> None:
    step = Step("archive")
    # probes b and m: no gate and no compensate field
    assert tuple(f.name for f in dataclasses.fields(step)) == ("tool", "arguments", "requirements")
    assert (step.tool, dict(step.arguments), step.requirements) == ("archive", {}, ())
    req = Requirement("d:route", "yes")
    step = Step(Name("send"), {Name("to"): Origin(ref="t:state.to")}, [req])
    assert type(step.tool) is str
    assert step.tool == "send"
    assert [type(k) for k in step.arguments] == [str]
    assert step.requirements == (req,)


def test_a_steps_arguments_are_a_read_only_copy() -> None:
    given: dict[str, Origin | Binding] = {"a": Origin(literal=1)}
    step = Step("t", given)
    given["b"] = Origin(literal=2)
    assert list(step.arguments) == ["a"]
    with pytest.raises(TypeError):
        step.arguments["c"] = Origin(literal=3)  # type: ignore[index]


@pytest.mark.parametrize(
    ("args"),
    [
        (1,),
        (None,),
        ("t", [("a", Origin(literal=1))]),
        ("t", {"a": "d:x"}),  # a bare string is never an origin
        ("t", {"a": {"literal": 1}}),
        ("t", {1: Origin(literal=1)}),
        ("t", {}, "d:x"),
        ("t", {}, [("d:x", 1)]),
        ("t", {}, [{"ref": "d:x", "value": 1}]),
    ],
)
def test_a_step_field_of_the_wrong_type_is_a_type_error(args: tuple[Any, ...]) -> None:
    with pytest.raises(TypeError):
        Step(*args)


def test_a_mapping_that_is_not_a_dict_is_a_type_error() -> None:
    step = Step("t", {"a": Origin(literal=1)})
    with pytest.raises(TypeError, match="not a dict"):
        Step("t", cast(Any, step.arguments))


class Twin(str):
    """A string subclass a dict keeps apart from the plain string of its data."""

    __slots__ = ()

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return self is other


def test_two_argument_names_of_one_string_data_are_kept_as_given() -> None:
    # The constructor checks field types only; plan rule 7 finds the names outside the
    # domain.
    twin = Twin("a")
    step = Step("t", {"a": Origin(literal=1), twin: Origin(literal=2)})
    assert len(step.arguments) == 2
    assert [type(k) for k in step.arguments] == [str, Twin]


def test_a_plan_and_a_step_are_not_hashable() -> None:
    # They hold their mappings read-only, as the design types do.
    with pytest.raises(TypeError):
        hash(Step("t"))
    with pytest.raises(TypeError):
        hash(Plan("execute"))


def test_a_plan_copies_shallowly_but_neither_deep_copies_nor_pickles() -> None:
    plan = Plan("execute", [Step("t")])
    assert copy.copy(plan).steps is plan.steps
    with pytest.raises(TypeError):
        copy.deepcopy(plan)
    with pytest.raises(TypeError):
        pickle.dumps(plan)


# ---------------------------------------------------------------------------------------------
# Origin and Binding


def test_an_origin_is_a_literal_or_a_board_ref() -> None:
    literal = Origin(literal=[1, "a"])
    assert (literal.literal, literal.ref, literal.set) == ([1, "a"], None, None)
    none = Origin(literal=None)
    assert (none.literal, none.ref, none.set) == (None, None, None)
    ref = Origin(ref=Name("d:route"))
    assert (ref.literal, ref.ref, ref.set) == (None, "d:route", None)
    assert type(ref.ref) is str


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"literal": 1, "ref": "d:x"}, {"literal": None, "ref": "d:x"}, {"ref": 1}, {"ref": None}],
)
def test_an_origin_names_exactly_one_of_literal_and_ref(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        Origin(**kwargs)


def test_the_public_origin_takes_no_set_and_compile_writes_one() -> None:
    with pytest.raises(TypeError):
        Origin(literal="Dana", set="d:people")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Origin("Dana")  # type: ignore[misc]
    origin = candidate_literal("Dana", "d:people")
    assert (origin.literal, origin.ref, origin.set) == ("Dana", None, "d:people")


def test_a_binding_is_a_step_and_an_optional_path() -> None:
    # probes c and e: no select and no depends_on field
    assert tuple(f.name for f in dataclasses.fields(Binding(2))) == ("step", "path")
    assert (Binding(2).step, Binding(2).path) == (2, None)
    bound = Binding(1, Name("items.name"))
    assert (bound.step, bound.path) == (1, "items.name")
    assert type(bound.path) is str


@pytest.mark.parametrize(("step", "path"), [(True, None), (1.0, None), ("1", None), (1, 2)])
def test_a_binding_field_of_the_wrong_type_is_a_type_error(step: Any, path: Any) -> None:
    with pytest.raises(TypeError):
        Binding(step, path)


# ---------------------------------------------------------------------------------------------
# A bad value is named without formatting it


class Unprintable:
    def __repr__(self) -> str:
        raise RuntimeError("repr was called")


HUGE = 10**5000
BUILDS: dict[str, Callable[[Any], object]] = {
    "reason": lambda v: Plan("execute", reason=v),
    "reply": lambda v: Plan("execute", reply=v),
    "steps": lambda v: Plan("execute", v),
    "tool": lambda v: Step(v),
    "arguments": lambda v: Step("t", v),
    "requirements": lambda v: Step("t", {}, v),
    "argument": lambda v: Step("t", {"a": v}),
    "path": lambda v: Binding(1, v),
    "ref": lambda v: Origin(ref=v),
}


@pytest.mark.parametrize(
    "bad", [HUGE, Unprintable(), [HUGE], {"a": Unprintable()}], ids=["huge", "repr", "list", "dict"]
)
@pytest.mark.parametrize("build", BUILDS.values(), ids=BUILDS.keys())
def test_a_bad_value_that_cannot_be_written_is_the_same_type_error(
    build: Callable[[Any], object], bad: object
) -> None:
    with pytest.raises(TypeError):
        build(bad)


@pytest.mark.parametrize("name", [HUGE, Unprintable()], ids=["huge", "repr"])
def test_an_argument_name_that_cannot_be_written_is_the_same_type_error(name: object) -> None:
    with pytest.raises(TypeError, match="argument name"):
        Step("t", {name: Origin(literal=1)})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("kind", [HUGE, Unprintable()], ids=["huge", "repr"])
def test_a_kind_that_cannot_be_written_is_the_same_value_error(kind: object) -> None:
    with pytest.raises(ValueError, match="not in the closed set"):
        Plan(kind)  # pyright: ignore[reportArgumentType]


def test_a_bad_value_is_named_by_its_type_or_bit_length() -> None:
    with pytest.raises(TypeError, match=r"^a plan's reason integer of 16610 bits is not a str"):
        Plan("execute", reason=HUGE)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match=r"^a step's tool <Unprintable> is not a string$"):
        Step(Unprintable())  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match=r"^a binding's step 1\.5 is not an int$"):
        Binding(1.5)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match=r"^a binding's step <Unprintable> is not an int$"):
        Binding(Unprintable())  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=r"^plan kind None is not in the closed set$"):
        Plan(None)  # pyright: ignore[reportArgumentType]


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
@pytest.mark.parametrize("build", BUILDS.values(), ids=BUILDS.keys())
def test_a_value_is_judged_by_its_type_not_its_class_attribute(
    build: Callable[[Any], object], make: Callable[[], object]
) -> None:
    with pytest.raises(TypeError):
        build(make())


@pytest.mark.parametrize("make", [Liar, ClassRaises], ids=["liar", "raises"])
def test_an_argument_name_is_judged_by_its_type(make: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match="argument name"):
        Step("t", {make(): Origin(literal=1)})  # pyright: ignore[reportArgumentType]
