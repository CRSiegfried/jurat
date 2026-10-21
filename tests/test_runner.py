"""The reference runner: the gate before any call, the walk over steps by
position with each argument from its declared origin, the key, the stop and
the return. Decisions come from ``decide`` over small designs and a scripted fixture binding.

The defect probes ported here are a, b, c, d, j, k and m."""

import asyncio
import inspect
from abc import ABCMeta
from collections.abc import Callable, Iterator, Mapping
from typing import Any, cast

import pytest

import jurat
import jurat.helpers
from designs import minimal
from jurat import (
    Binding,
    Decision,
    Origin,
    Plan,
    PlanKind,
    Policy,
    Requirement,
    Rule,
    Source,
    Step,
    VerdictKind,
    decide_sync,
    load,
)
from jurat import types as t
from jurat.helpers import BindingNotFound, RunRefusal, run_plan
from scripted import Scripted, returns

STATE_T = t.record(
    subject=t.string,
    blob=t.bytes_,
    sender=t.nullable(t.record(address=t.string)),
)
STATE: dict[str, Any] = {"subject": "hi", "blob": b"\x00\x01", "sender": {"address": "a@x.org"}}
READS = [
    "t:state.subject",
    "t:state.sender",
    "t:state.sender.address",
    "t:state.blob",
    "t:n",
    "d:ok",
]
TOOLS = ["list", "mark", "send", "undo"]


def yes(r: Any) -> str:
    return "yes"


def no(r: Any) -> str:
    return "no"


def decision(
    policy: Callable[[Any], Any],
    *,
    state: dict[str, Any] = STATE,
    rule: Callable[[Any], str] = yes,
    default_deny: bool = False,
) -> Decision:
    design = minimal(
        state=STATE_T,
        sources={"s": Source({"t:n": t.integer()}, needs=["t:state"])},
        rules={"ok": Rule("d:ok", t.label("no", "yes"), rule, version="1", reads=["t:n"])},
        policy=Policy(policy, version="1", reads=READS),
        tools=TOOLS,
        default_deny=default_deny,
    )
    binding = Scripted({"s": returns({"t:n": 4})})
    return decide_sync(load(design), state, {"default": binding})


def plan_of(*steps: Step) -> Callable[[Any], Plan]:
    def policy(r: Any) -> Plan:
        return Plan("execute", list(steps))

    return policy


class Calls:
    """Async tools that log each call's tool name and keywords and return a scripted result."""

    def __init__(self, **results: object) -> None:
        self.results = results
        self.log: list[tuple[str, dict[str, Any]]] = []

    def tool(self, name: str) -> Callable[..., Any]:
        async def call(**keywords: Any) -> object:
            self.log.append((name, keywords))
            result = self.results.get(name)
            if issubclass(type(result), BaseException):  # by type: a result may spoof
                raise cast(BaseException, result)
            return result

        return call

    def tools(self, *names: str) -> dict[str, Callable[..., Any]]:
        return {name: self.tool(name) for name in names}

    def names(self) -> list[str]:
        return [name for name, _ in self.log]

    def arguments(self, n: int) -> dict[str, Any]:
        """The keywords of the ``n``-th call, from 1, without the key."""
        keywords = dict(self.log[n - 1][1])
        del keywords["idempotency_key"]
        return keywords


def run(d: object, tools: object) -> dict[str, Any]:
    return asyncio.run(run_plan(d, tools))  # pyright: ignore[reportArgumentType]


def refused(d: object, tools: object) -> str:
    with pytest.raises(RunRefusal) as caught:
        run(d, tools)
    return str(caught.value)


TWO = plan_of(Step("list"), Step("mark", {"x": Binding(1, "items")}))


# ---------------------------------------------------------------------------------------------
# The package's place


def test_the_runner_is_in_helpers_and_not_in_the_engines_api() -> None:
    for name in ("run_plan", "RunRefusal", "BindingNotFound"):
        assert name in jurat.helpers.__all__
        assert name not in jurat.__all__
    assert issubclass(RunRefusal, ValueError)
    assert issubclass(BindingNotFound, LookupError)
    assert inspect.iscoroutinefunction(run_plan)


# ---------------------------------------------------------------------------------------------
# The gate: each refusal is raised before any call


def test_a_decision_not_exactly_a_decision_is_refused() -> None:
    calls = Calls()
    assert "Decision" in refused(None, calls.tools(*TOOLS))
    assert "Decision" in refused(object(), calls.tools(*TOOLS))
    assert calls.log == []


def test_tools_that_are_not_a_mapping_are_refused() -> None:
    calls = Calls()
    d = decision(TWO)
    assert "Mapping" in refused(d, list(calls.tools("list", "mark").items()))
    assert calls.log == []


class RaisingCopy(Mapping[str, Any]):
    def __init__(self, tools: dict[str, Any]) -> None:
        self.tools = tools

    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("no copy")

    def __iter__(self) -> Iterator[str]:
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)


def test_tools_whose_copy_raises_are_refused() -> None:
    calls = Calls()
    with pytest.raises(RunRefusal) as caught:
        run(decision(TWO), RaisingCopy(calls.tools("list", "mark")))
    assert type(caught.value.__cause__) is RuntimeError
    assert calls.log == []


class Twin(str):
    """A key a dict keeps apart from its string data."""

    __slots__ = ()

    def __hash__(self) -> int:
        return 7

    def __eq__(self, other: object) -> bool:
        return other is self


def test_two_tool_keys_of_one_string_data_are_refused() -> None:
    calls = Calls()
    tools: dict[str, Any] = calls.tools("list", "mark")
    tools[Twin("list")] = calls.tool("other")
    assert len(tools) == 3
    assert "list" in refused(decision(TWO), tools)
    assert calls.log == []


def test_a_refused_verdict_calls_no_tool() -> None:
    # Probe b: gates are gone, so a refused verdict is what keeps every tool uncalled.
    calls = Calls()
    d = decision(plan_of(Step("list", {}, [Requirement("d:ok", "yes")])), rule=no)
    assert d.plan.kind is PlanKind.execute
    assert d.verdict.kind is VerdictKind.refused
    assert "refused" in refused(d, calls.tools(*TOOLS))
    assert calls.log == []


def test_a_step_on_an_escalate_plan_is_refused_on_kind() -> None:
    # Probe a: a step on an escalate plan is broken under plan rule 1.
    def policy(r: Any) -> Plan:
        return Plan("escalate", [Step("send")], reason="hand on")

    calls = Calls()
    d = decision(policy)
    assert d.plan.kind is PlanKind.broken
    assert "broken" in refused(d, calls.tools(*TOOLS))
    assert calls.log == []


def test_an_inform_plan_is_refused_on_kind() -> None:
    calls = Calls()
    d = decision(lambda r: Plan("inform", reply="ok"))
    assert d.verdict.kind is VerdictKind.passed
    assert "inform" in refused(d, calls.tools(*TOOLS))
    assert calls.log == []


def test_a_missing_tool_for_a_later_step_is_refused_before_the_first_call() -> None:
    calls = Calls()
    assert "mark" in refused(decision(TWO), calls.tools("list"))
    assert calls.log == []


def test_a_tool_that_is_not_callable_is_refused_before_the_first_call() -> None:
    calls = Calls()
    tools: dict[str, Any] = {**calls.tools("list"), "mark": "not callable"}
    assert "mark" in refused(decision(TWO), tools)
    assert calls.log == []


def test_the_tools_are_copied_once_at_the_gate() -> None:
    calls = Calls(list={"items": []})
    tools: dict[str, Any] = calls.tools("list", "mark")

    async def swap(**keywords: Any) -> object:
        tools["mark"] = calls.tool("swapped")
        return {"items": []}

    tools["list"] = swap
    out = run(decision(TWO), tools)
    assert out["raised"] is None
    assert calls.names() == ["mark"]


# ---------------------------------------------------------------------------------------------
# The key and the return


def test_each_call_carries_the_turn_id_and_its_position_split_on_the_last_colon() -> None:
    calls = Calls(list={"items": [1]})
    d = decision(TWO)
    run(d, calls.tools("list", "mark"))
    keys = [keywords["idempotency_key"] for _, keywords in calls.log]
    assert [key.rsplit(":", 1) for key in keys] == [
        [d.record.record_hash, "1"],
        [d.record.record_hash, "2"],
    ]
    assert d.record.record_hash.startswith("sha256:")


def test_the_return_is_a_plain_dict_of_results_and_raised() -> None:
    calls = Calls(list={"items": [1]}, mark="marked")
    out = run(decision(TWO), calls.tools("list", "mark"))
    assert type(out) is dict
    assert list(out) == ["results", "raised"]
    assert out == {"results": [[1, {"items": [1]}], [2, "marked"]], "raised": None}
    pairs: list[object] = out["results"]
    assert type(pairs) is list
    assert all(type(pair) is list for pair in pairs)


# ---------------------------------------------------------------------------------------------
# Bindings


def test_a_path_reads_into_a_mapping_result() -> None:
    # Probe c: selectors are gone, so a path into a mapping result stands in for one.
    result = {"id": "x", "items": [{"id": 1}], "a": {"b": 2}}
    calls = Calls(list=result)
    policy = plan_of(Step("list"), Step("mark", {"x": Binding(1, "items"), "y": Binding(1, "a.b")}))
    out = run(decision(policy), calls.tools("list", "mark"))
    assert calls.arguments(2) == {"x": [{"id": 1}], "y": 2}
    assert out["raised"] is None


def test_a_whole_none_result_is_passed_on() -> None:
    # Probe k: a whole result, None included, is bound as it is.
    calls = Calls(list=None)
    policy = plan_of(Step("list"), Step("mark", {"x": Binding(1)}))
    out = run(decision(policy), calls.tools("list", "mark"))
    assert calls.arguments(2) == {"x": None}
    assert out["results"] == [[1, None], [2, None]]


def test_a_present_none_key_gives_none() -> None:
    calls = Calls(list={"a": None})
    policy = plan_of(Step("list"), Step("mark", {"x": Binding(1, "a")}))
    run(decision(policy), calls.tools("list", "mark"))
    assert calls.arguments(2) == {"x": None}


def missed(result: object, path: str) -> tuple[dict[str, Any], Calls]:
    calls = Calls(list=result)
    policy = plan_of(Step("list"), Step("mark", {"ok": Binding(1), "x": Binding(1, path)}))
    return run(decision(policy), calls.tools("list", "mark")), calls


def test_a_segment_beyond_a_present_none_is_binding_not_found() -> None:
    out, calls = missed({"a": None}, "a.b")
    raised = out["raised"]
    assert type(raised) is BindingNotFound
    assert (raised.step, raised.position, raised.path) == (2, 1, "a.b")
    assert calls.names() == ["list"]


class Obj:
    nope = "an attribute is not read"


@pytest.mark.parametrize(
    "result",
    [Obj(), "nope", ["nope"], 3],
    ids=["object", "string", "list", "number"],
)
def test_a_path_over_a_non_mapping_is_returned_as_binding_not_found(result: object) -> None:
    # Probe j: a non-mapping has no keys; no attribute or index is read.
    out, calls = missed(result, "nope")
    raised = out["raised"]
    assert type(raised) is BindingNotFound
    assert (raised.step, raised.position, raised.path) == (2, 1, "nope")
    assert out["results"] == [[1, result]]
    assert calls.names() == ["list"]


def test_a_missing_key_is_binding_not_found_before_the_steps_tool() -> None:
    out, calls = missed({"a": {"c": 1}}, "a.b")
    raised = out["raised"]
    assert type(raised) is BindingNotFound
    assert isinstance(raised, LookupError)
    assert raised.path == "a.b"
    assert calls.names() == ["list"]


# ---------------------------------------------------------------------------------------------
# Board refs and literals, read from the record


def board(state: dict[str, Any] = STATE) -> dict[str, Any]:
    calls = Calls()
    step = Step(
        "send",
        {
            "s": Origin(ref="t:state.subject"),
            "a": Origin(ref="t:state.sender.address"),
            "b": Origin(ref="t:state.blob"),
            "n": Origin(ref="t:n"),
            "ok": Origin(ref="d:ok"),
        },
    )
    out = run(decision(plan_of(step), state=state), calls.tools("send"))
    assert out["raised"] is None
    return calls.arguments(1)


def test_a_board_ref_reaches_the_tool_as_the_records_value() -> None:
    assert board() == {"s": "hi", "a": "a@x.org", "b": "AAE", "n": 4, "ok": "yes"}


def test_the_records_log_is_decoded_once_for_every_board_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = [0]
    log = cast(Any, jurat.Record).log

    def counted(self: Any) -> Any:
        reads[0] += 1
        return log.fget(self)

    monkeypatch.setattr(jurat.Record, "log", property(counted))
    board()
    assert reads[0] == 1  # the gate's one decoding, every entry read from it


def test_a_null_on_a_board_refs_path_gives_null() -> None:
    assert board({**STATE, "sender": None})["a"] is None


class Text(str):
    __slots__ = ()


def test_a_literal_is_the_recorded_value_not_the_policys_object() -> None:
    kept = [1, {"k": "v"}]

    def policy(r: Any) -> Plan:
        return Plan(
            "execute",
            [Step("send", {"v": Origin(literal=kept), "t": Origin(literal=Text("x"))})],
        )

    d = decision(policy)
    kept.append(2)
    calls = Calls()
    run(d, calls.tools("send"))
    got = calls.arguments(1)
    assert got == {"v": [1, {"k": "v"}], "t": "x"}
    assert type(got["t"]) is str


# ---------------------------------------------------------------------------------------------
# The stop


def test_a_raising_tool_stops_the_run_and_is_raised_with_no_pair() -> None:
    # Probe d: the exception is returned in raised, and its step has no result.
    error = RuntimeError("timeout after commit")
    calls = Calls(send=error)
    out = run(decision(plan_of(Step("send"))), calls.tools("send"))
    assert out == {"results": [], "raised": error}


def test_no_later_tool_is_called_and_nothing_is_compensated() -> None:
    # Probe m: there is no compensation; the first step's result is kept.
    error = RuntimeError("boom")
    calls = Calls(list={"id": "x"}, mark=error)
    policy = plan_of(Step("list"), Step("mark"), Step("undo"))
    out = run(decision(policy), calls.tools("list", "mark", "undo"))
    assert out == {"results": [[1, {"id": "x"}]], "raised": error}
    assert calls.names() == ["list", "mark"]


def test_a_sync_tools_unawaitable_result_is_a_type_error_in_raised() -> None:
    seen: list[dict[str, Any]] = []

    def sync(**keywords: Any) -> int:
        seen.append(keywords)
        return 1

    calls = Calls()
    policy = plan_of(Step("send"), Step("list"))
    out = run(decision(policy), {"send": sync, **calls.tools("list")})
    assert type(out["raised"]) is TypeError
    assert out["results"] == []
    assert len(seen) == 1
    assert calls.log == []


def test_a_sync_tool_that_raises_is_raised() -> None:
    error = ValueError("bad")

    def sync(**keywords: Any) -> int:
        raise error

    out = run(decision(plan_of(Step("send"))), {"send": sync})
    assert out == {"results": [], "raised": error}


def test_cancellation_propagates() -> None:
    calls = Calls(mark=asyncio.CancelledError())
    policy = plan_of(Step("list"), Step("mark"), Step("undo"))
    with pytest.raises(asyncio.CancelledError):
        run(decision(policy), calls.tools("list", "mark", "undo"))
    assert calls.names() == ["list", "mark"]


def test_a_keyboard_interrupt_propagates() -> None:
    calls = Calls(send=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run(decision(plan_of(Step("send"))), calls.tools("send"))


def test_a_key_that_is_not_a_string_names_no_tool() -> None:
    calls = Calls()
    tools: dict[object, Any] = {1: calls.tool("other"), **calls.tools("send")}
    out = run(decision(plan_of(Step("send"))), tools)
    assert out["raised"] is None
    assert calls.names() == ["send"]


def test_each_board_ref_argument_is_a_fresh_copy_of_the_records_value() -> None:
    seen: list[object] = []

    async def spoil(**keywords: Any) -> None:
        who, again = keywords["who"], keywords["again"]
        seen.append(who is again)
        who["address"] = "spoiled"
        seen.append(dict(again))
        again["address"] = "spoiled too"

    async def look(**keywords: Any) -> None:
        seen.append(keywords["who"])

    who = Origin(ref="t:state.sender")
    policy = plan_of(Step("send", {"who": who, "again": who}), Step("list", {"who": who}))
    out = run(decision(policy), {"send": spoil, "list": look})
    assert out["raised"] is None
    assert seen == [False, {"address": "a@x.org"}, {"address": "a@x.org"}]


def test_each_literal_argument_is_its_own_value() -> None:
    seen: list[object] = []

    async def spoil(**keywords: Any) -> None:
        seen.append(keywords["v"] is keywords["w"])
        keywords["v"].append("spoiled")
        seen.append(list(keywords["w"]))

    async def look(**keywords: Any) -> None:
        seen.append(keywords["v"])

    def v() -> Origin:
        return Origin(literal=[1])

    policy = plan_of(Step("send", {"v": v(), "w": v()}), Step("list", {"v": v()}))
    out = run(decision(policy), {"send": spoil, "list": look})
    assert out["raised"] is None
    assert seen == [False, [1], [1]]


def test_an_inform_plan_with_no_tools_is_refused_on_its_kind() -> None:
    assert "inform" in refused(decision(lambda r: Plan("inform", reply="ok")), {})


# ---------------------------------------------------------------------------------------------
# The gate's order: the earlier condition's message wins


def test_a_non_decision_is_refused_before_tools_that_are_not_a_mapping() -> None:
    message = refused(None, [("send", None)])
    assert "Decision" in message
    assert "Mapping" not in message


def test_tools_that_are_not_a_mapping_are_refused_before_a_refused_verdict() -> None:
    d = decision(plan_of(Step("list", {}, [Requirement("d:ok", "yes")])), rule=no)
    assert d.verdict.kind is VerdictKind.refused
    message = refused(d, [("list", None)])
    assert "Mapping" in message
    assert "refused" not in message


def test_a_refused_verdict_is_refused_before_a_missing_tool() -> None:
    d = decision(plan_of(Step("list", {}, [Requirement("d:ok", "yes")])), rule=no)
    message = refused(d, {})
    assert "refused" in message
    assert "list" not in message


# ---------------------------------------------------------------------------------------------
# A raise while a binding's path is read is the run's raise, not a miss


class Touchy(Mapping[str, Any]):
    """A result mapping that raises ``contains`` from ``in`` or ``item`` from indexing."""

    def __init__(self, contains: Exception | None = None, item: Exception | None = None) -> None:
        self.contains = contains
        self.item = item

    def __contains__(self, key: object) -> bool:
        if self.contains is not None:
            raise self.contains
        return True

    def __getitem__(self, key: str) -> Any:
        if self.item is not None:
            raise self.item
        return "value"

    def __iter__(self) -> Iterator[str]:
        return iter(["a"])

    def __len__(self) -> int:
        return 1


@pytest.mark.parametrize(
    "fault",
    ["contains", "item", "item-keyerror", "nested-contains"],
)
def test_an_exception_while_a_path_is_read_is_in_raised(fault: str) -> None:
    error: Exception = KeyError("a") if fault == "item-keyerror" else RuntimeError(fault)
    if fault == "contains":
        result: object = Touchy(contains=error)
    elif fault == "nested-contains":
        result = {"a": Touchy(contains=error)}
    else:
        result = Touchy(item=error)
    path = "a.b" if fault == "nested-contains" else "a"
    calls = Calls(list={"id": 1}, mark=result)
    policy = plan_of(
        Step("list"), Step("mark"), Step("send", {"ok": Binding(1), "x": Binding(2, path)})
    )
    out = run(decision(policy), calls.tools("list", "mark", "send"))
    assert out["raised"] is error
    assert out["results"] == [[1, {"id": 1}], [2, result]]
    assert calls.names() == ["list", "mark"]


def test_a_present_key_read_through_a_mapping_gives_its_value() -> None:
    calls = Calls(list=Touchy())
    policy = plan_of(Step("list"), Step("mark", {"x": Binding(1, "a")}))
    out = run(decision(policy), calls.tools("list", "mark"))
    assert out["raised"] is None
    assert calls.arguments(2) == {"x": "value"}


# ---------------------------------------------------------------------------------------------
# Caller objects are read by their type, never by isinstance, __class__ or repr


def spoofing(cls: type) -> Any:
    """An instance whose ``__class__`` claims ``cls`` and whose ``repr`` raises."""

    class Spoof:
        @property
        def __class__(self) -> type:  # type: ignore[override]
            return cls

        def __repr__(self) -> str:
            raise RuntimeError("repr read")

    return Spoof()


def raising_class() -> Any:
    """An instance whose ``__class__`` and ``repr`` raise."""

    class Raising:
        @property
        def __class__(self) -> type:  # type: ignore[override]
            raise RuntimeError("__class__ read")

        def __repr__(self) -> str:
            raise RuntimeError("repr read")

    return Raising()


class SubDecision(Decision):
    __slots__ = ()


def test_a_decision_subclass_instance_is_refused() -> None:
    sub = object.__new__(SubDecision)
    assert "SubDecision" in refused(sub, {})


def test_a_decision_spoofing_its_class_is_refused() -> None:
    assert isinstance(spoofing(Decision), Decision)
    calls = Calls()
    assert "Spoof" in refused(spoofing(Decision), calls.tools(*TOOLS))
    assert "Raising" in refused(raising_class(), calls.tools(*TOOLS))
    assert calls.log == []


def test_tools_spoofing_a_dict_are_refused() -> None:
    assert isinstance(spoofing(dict), Mapping)
    d = decision(TWO)
    assert "Spoof" in refused(d, spoofing(dict))
    assert "Raising" in refused(d, raising_class())


def test_a_key_spoofing_a_string_names_no_tool() -> None:
    calls = Calls()
    tools: dict[object, Any] = {
        spoofing(str): calls.tool("other"),
        raising_class(): calls.tool("other"),
        **calls.tools("send"),
    }
    out = run(decision(plan_of(Step("send"))), tools)
    assert out["raised"] is None
    assert calls.names() == ["send"]


@pytest.mark.parametrize("make", [lambda: spoofing(dict), raising_class], ids=["spoof", "raise"])
def test_a_result_spoofing_or_raising_its_class_is_a_non_mapping(make: Callable[[], Any]) -> None:
    out, calls = missed(make(), "nope")
    raised = out["raised"]
    assert type(raised) is BindingNotFound
    assert (raised.step, raised.position, raised.path) == (2, 1, "nope")
    assert calls.names() == ["list"]


class _Tripping(ABCMeta):
    """A Mapping subclass's metaclass whose check raises for a type marked ``trips``, so the
    test of whether a type is a Mapping raises."""

    def __subclasscheck__(cls, subclass: type) -> bool:
        if "trips" in vars(subclass):
            raise RuntimeError("subclass check")
        return super().__subclasscheck__(subclass)


class Tripwire(Mapping[str, Any], metaclass=_Tripping):
    pass


class Trips:
    trips = True


def test_tools_whose_mapping_test_raises_are_refused() -> None:
    with pytest.raises(RunRefusal) as caught:
        run(decision(TWO), Trips())
    assert "Trips" in str(caught.value)
    assert type(caught.value.__cause__) is RuntimeError


def test_a_result_is_kept_and_bound_as_returned_never_copied() -> None:
    result: dict[str, Any] = {"a": {"b": [1]}}
    seen: list[object] = []

    async def produce(**keywords: Any) -> object:
        return result

    async def take(**keywords: Any) -> None:
        seen.extend([keywords["whole"], keywords["part"]])

    async def edit(**keywords: Any) -> None:
        keywords["part"]["b"].append("edited")

    policy = plan_of(
        Step("list"),
        Step("mark", {"whole": Binding(1), "part": Binding(1, "a")}),
        Step("send", {"part": Binding(1, "a")}),
    )
    out = run(decision(policy), {"list": produce, "mark": take, "send": edit})
    assert out["raised"] is None
    pair = out["results"][0]
    assert pair[0] == 1
    assert pair[1] is result
    assert seen[0] is pair[1]
    assert seen[1] is pair[1]["a"]
    assert pair[1] == {"a": {"b": [1, "edited"]}}
