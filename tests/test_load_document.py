# pyright: reportPrivateUsage=false
"""Loading a design document: the round trip, check 11's gate, resolving
each function component through ``functions`` and ``importlib``, and the comparison."""

import functools
import json
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

import designs
from counting import SmallBounds, bound, counted
from designs import bounded, full, gated, minimal, question, rule, with_rule, with_source
from jurat import Design, DesignFailure, InvalidDesign, LoadedDesign, Source, load
from jurat import types as t
from jurat._canonical import decode, encode, sha256_ref


def doc_of(design: Design) -> dict[str, Any]:
    return decode(load(design).document)


def written(doc: dict[str, Any]) -> bytes:
    return encode(doc, 128)


def refused(given: Any, **functions: Any) -> list[DesignFailure]:
    with pytest.raises(InvalidDesign) as info:
        load(given, functions=functions or None)
    return list(info.value.failures)


def only(given: Any, check: int, **functions: Any) -> list[DesignFailure]:
    found = refused(given, **functions)
    assert {f.check for f in found} == {check}, found
    return found


def gate_failure(given: Any) -> DesignFailure:
    (failure,) = refused(given)
    assert failure.check == 11
    return failure


def alike(a: LoadedDesign, b: LoadedDesign) -> None:
    assert a.document == b.document
    assert a.design_hash == b.design_hash
    assert dict(a._grid) == dict(b._grid)
    assert dict(a._graph) == dict(b._graph)
    assert dict(a._providers) == dict(b._providers)
    assert a._classes == b._classes
    assert dict(a._needs) == dict(b._needs)
    assert dict(a._types) == dict(b._types)
    assert dict(a._leaf_types) == dict(b._leaf_types)
    assert set(a._callables) == set(b._callables)
    assert all(a._callables[k] is b._callables[k] for k in a._callables)
    assert dict(a._groups) == dict(b._groups)
    assert dict(a._payload_bytes) == dict(b._payload_bytes)
    assert dict(a._leaf_bytes) == dict(b._leaf_bytes)
    assert dict(a._requirement_bytes) == dict(b._requirement_bytes)
    assert a._requirements_bytes == b._requirements_bytes


# ---------------------------------------------------------------------------------------------
# The round trip


@pytest.mark.parametrize("design", [minimal(), full(), gated(), bounded()])
def test_a_document_round_trips_as_bytes_and_parsed(design: Design) -> None:
    loaded = load(design)
    alike(load(loaded.document), loaded)
    alike(load(decode(loaded.document)), loaded)
    keys = {"route": designs.route, "score": designs.score, "policy": designs.policy}
    alike(load(loaded.document, functions=keys), loaded)


def test_a_document_resolves_each_function_component_to_its_callable() -> None:
    loaded = load(load(gated()).document)
    assert dict(loaded._callables) == {
        "gate.when": designs.route,
        "policy": designs.policy,
        "route": designs.route,
        "score": designs.score,
    }
    assert loaded._design.policy.function is designs.policy  # type: ignore[union-attr]


def test_a_bytes_leaf_written_as_base64url_is_bytes_again() -> None:
    loaded = load(load(full()).document)
    assert decode(loaded.document)["parameters"]["blob"]["value"] == "AAE"
    assert loaded._leaf_value("blob") == b"\x00\x01"


def test_a_functions_entry_overrides_an_importable_function() -> None:
    @functools.wraps(designs.route)
    def stand_in(r: Any) -> str:
        return "no"

    loaded = load(load(full()).document, functions={"route": stand_in})
    assert loaded._callables["route"] is stand_in


def test_a_functions_key_naming_no_component_is_ignored() -> None:
    load(load(full()).document, functions={"nothing": designs.route, "x.when": len})


class _Sizeless(Mapping[str, Any]):
    """A functions mapping whose size and truth are not read."""

    def __getitem__(self, key: str) -> Any:
        if key == "route":
            return designs.route
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("route",))

    def __len__(self) -> int:
        raise AssertionError("the mapping's size was read")


def test_a_functions_mapping_is_not_read_for_its_truth() -> None:
    loaded = load(load(full()).document, functions=_Sizeless())
    assert loaded._callables["route"] is designs.route


def test_a_function_in_locals_or_a_lambda_loads_through_functions() -> None:
    def inner(r: Any) -> bool:
        return True

    anonymous: Callable[[Any], bool] = lambda r: True  # noqa: E731
    for fn in (inner, anonymous):
        document = load(with_rule(minimal(), "r", rule("r", function=fn))).document
        load(document, functions={"r": fn})


def test_a_function_whose_source_cannot_be_read_round_trips() -> None:
    design = with_rule(minimal(), "size", rule("size", outputs=t.integer(), function=len))
    load(load(design).document)


# ---------------------------------------------------------------------------------------------
# Check 11's gate: one failure, and no other check


def with_check_1_defect() -> dict[str, Any]:
    doc = doc_of(minimal())
    doc["tools"] = {"Bad": {}}  # not in the segment grammar: check 1
    return doc


def test_the_defective_document_is_check_1s() -> None:
    assert {f.check for f in refused(written(with_check_1_defect()))} == {1}


def nested(depth: int) -> Any:
    value: Any = {}
    for _ in range(depth - 1):
        value = {"a": value}
    return value


def cyclic() -> dict[str, Any]:
    doc = with_check_1_defect()
    loop: list[Any] = []
    loop.append(loop)
    doc["name"] = loop
    return doc


def bytes_cases() -> list[tuple[bytes, str]]:
    good = written(with_check_1_defect())
    return [
        (b"\xff" + good, "not UTF-8"),
        (good[:-1], "not JSON"),
        (b'{"name":null,' + good[1:], "a key twice"),
        (b"[" * 129, "nests more than 128 deep"),  # unclosed: depth first
        (b"[" * 129 + b"]" * 129, "nests more than 128 deep"),
        (b"NaN", "not JSON"),
        (b"[]", "not a JSON object"),
        (good.replace(b'"name":null', b'"name":"\\ud800"'), "outside the value domain"),
    ]


@pytest.mark.parametrize(("given", "says"), bytes_cases())
def test_check_11_refuses_bytes_with_one_failure(given: bytes, says: str) -> None:
    assert says in gate_failure(given).message


def parsed_cases() -> list[tuple[Any, str]]:
    deep = with_check_1_defect()
    deep["name"] = nested(129)
    no_format = with_check_1_defect()
    del no_format["format"]
    cases: list[tuple[Any, str]] = [
        (deep, "nests more than 128 deep"),
        (cyclic(), "a list or object at two places"),
        (no_format, "has no format"),
        ({**with_check_1_defect(), "name": float("nan")}, "outside the value domain"),
        ({**with_check_1_defect(), "name": b"x"}, "outside the value domain"),
        ({**with_check_1_defect(), "extra": 1}, 'the member "extra"'),
    ]
    for value in (2, True, "1", 1.0, None):
        cases.append(({**with_check_1_defect(), "format": value}, "format is not 1"))
    for member in ("state", "classes"):
        doc = with_check_1_defect()
        del doc[member]
        cases.append((doc, f'no member "{member}"'))
    return cases


@pytest.mark.parametrize(("given", "says"), parsed_cases())
def test_check_11_refuses_a_parsed_document_with_one_failure(given: Any, says: str) -> None:
    assert says in gate_failure(given).message


def test_an_unknown_format_is_refused_before_the_value_domain() -> None:
    doc = {**with_check_1_defect(), "format": 2, "name": float("nan")}
    assert "format" in gate_failure(doc).message


def component_cases() -> list[tuple[Callable[[dict[str, Any]], None], tuple[str, ...], str]]:
    def drop(*path: str) -> Callable[[dict[str, Any]], None]:
        def change(doc: dict[str, Any]) -> None:
            at = doc
            for p in path[:-1]:
                at = at[p]
            del at[path[-1]]

        return change

    def add(*path: str) -> Callable[[dict[str, Any]], None]:
        def change(doc: dict[str, Any]) -> None:
            at = doc
            for p in path[:-1]:
                at = at[p]
            at[path[-1]] = None

        return change

    def bounds_entry(doc: dict[str, Any]) -> None:
        doc["rules"]["route"]["boundaries"] = {"q:intent/email": {"bar": [], "mass": []}}

    def mass_member(doc: dict[str, Any]) -> None:
        entry = {"bar": [], "thresholds": [], "margin": [], "mass": [{"labels": ["yes"]}]}
        doc["rules"]["route"]["boundaries"] = {"q:intent/email": entry}

    def origin(doc: dict[str, Any]) -> None:
        wording = doc["sources"]["intent"]["provides"]["q:intent/email"]["wording"]
        wording["slots"][0]["origin"] = {"ref": "t:lookup", "param": "intent/bar"}

    def form(doc: dict[str, Any]) -> None:
        doc["rules"]["route"]["form"] = "data"

    return [
        (drop("rules", "route", "version"), ("route",), 'rule "route" has no member "version"'),
        (add("rules", "route", "inputs"), ("route",), 'the member "inputs"'),
        (drop("rules", "route", "form"), ("route",), 'no member "form"'),
        (form, ("route",), "neither function nor table"),
        (drop("policy", "source"), ("policy",), 'the policy has no member "source"'),
        (drop("sources", "intent", "cost"), ("intent",), 'no member "cost"'),
        (drop("sources", "gate", "when", "qualname"), ("gate",), "when has no member"),
        (add("sources", "gate", "requirement", "value"), ("gate",), "requirement has"),
        (
            drop("sources", "lookup", "provides", "t:lookup", "candidates"),
            ("lookup", "t:lookup"),
            "no",
        ),
        (
            drop("sources", "intent", "provides", "q:intent/tone", "labels"),
            ("intent", "q:intent/tone"),
            'no member "labels"',
        ),
        (add("parameters", "blob", "default"), ("blob",), 'parameter "blob"'),
        (add("tools", "send", "x"), ("send",), 'tool "send"'),
        (add("requirements", 0, "why"), (), "a plan-level requirement"),  # type: ignore[arg-type]
        (bounds_entry, ("q:intent/email", "route"), "no member"),
        (mass_member, ("q:intent/email", "route"), "mass member has no member"),
        (origin, ("intent", "q:intent/email"), "origin is not one of"),
    ]


@pytest.mark.parametrize(("change", "names", "says"), component_cases())
def test_check_11_a_component_missing_or_with_an_extra_member(
    change: Callable[[dict[str, Any]], None], names: tuple[str, ...], says: str
) -> None:
    doc = doc_of(gated())
    doc["tools"]["Bad"] = {}  # a check 1 defect, which the gate keeps from running
    change(doc)
    failure = gate_failure(written(doc))
    assert (failure.names, says in failure.message) == (names, True)


def plan_doc() -> dict[str, Any]:
    doc = doc_of(minimal())
    doc["plans"] = {
        "ask": {
            "kind": "clarify",
            "steps": [
                {
                    "tool": "send",
                    "arguments": {
                        "to": {"ref": "t:state"},
                        "body": {"literal": [1, {"x": None}]},
                        "after": {"step": 0, "path": None},
                    },
                    "requirements": [{"ref": "d:route", "value": "yes"}],
                }
            ],
            "reason": {"template": "Why?", "slots": []},
            "reply": None,
            "grounded": None,
        }
    }
    return doc


def plan_without(*path: Any) -> bytes:
    doc = plan_doc()
    at: Any = doc["plans"]["ask"]
    for p in path[:-1]:
        at = at[p]
    del at[path[-1]]
    return written(doc)


def plan_with_origin(origin: dict[str, Any]) -> bytes:
    doc = plan_doc()
    doc["plans"]["ask"]["steps"][0]["arguments"]["to"] = origin
    return written(doc)


@pytest.mark.parametrize(
    "given",
    [
        plan_without("reason", "slots"),
        plan_without("steps", 0, "requirements"),
        plan_without("steps", 0, "requirements", 0, "value"),
        plan_with_origin({"ref": "t:state", "literal": 1}),
        plan_with_origin({"step": 0}),
        plan_with_origin({}),
    ],
)
def test_check_11_a_plan_declarations_member_missing_or_extra(given: bytes) -> None:
    assert gate_failure(given).names == ("ask",)


def refused_form_docs() -> list[tuple[dict[str, Any], str]]:
    table = doc_of(minimal())
    table["rules"] = {
        "tab": {
            "form": "table",
            "provides": "d:tab",
            "outputs": "boolean",
            "candidates": False,
            "grounded": None,
            "hit": "U",
            "order": None,
            "inputs": [{"any": "shape"}],
            "rows": [[1, "-"]],
        }
    }
    policy_table = doc_of(minimal())
    policy_table["policy"] = {"form": "table", "inputs": [], "rows": []}
    candidates = doc_of(designs.with_source(minimal(), "s", designs.lookup_source()))
    candidates["sources"]["s"]["provides"]["t:lookup"]["candidates"] = True
    grounded = doc_of(designs.with_source(minimal(), "s", designs.lookup_source()))
    grounded["sources"]["s"]["grounded"] = "t:set"
    return [
        (table, "decision table"),
        (policy_table, "policy table"),
        (plan_doc(), "plan declaration"),
        (candidates, "candidate set"),
        (grounded, "grounded declaration"),
    ]


@pytest.mark.parametrize(("doc", "form"), refused_form_docs())
def test_a_refused_forms_members_pass_check_11_for_check_12_to_name(
    doc: dict[str, Any], form: str
) -> None:
    found = refused(written(doc))
    assert 11 not in {f.check for f in found}
    assert any(f.check == 12 and form in f.message for f in found), found


# ---------------------------------------------------------------------------------------------
# Resolving the function components


def renamed(module: str, qualname: str = "route") -> bytes:
    doc = doc_of(full())
    doc["rules"]["route"]["module"] = module
    doc["rules"]["route"]["qualname"] = qualname
    return written(doc)


@pytest.mark.parametrize(
    ("given", "says"),
    [
        (renamed("jurat_no_such_module"), "import none"),
        (renamed("designs", "no_such_function"), "import none"),
        (renamed("designs", "route.nested"), "import none"),
        (renamed(""), "import none"),
        (renamed("designs", "test.<locals>.route"), "must be in functions"),
        (renamed("designs", "<lambda>"), "must be in functions"),
    ],
)
def test_check_11_a_component_load_cannot_resolve(given: bytes, says: str) -> None:
    (failure,) = only(given, 11)
    assert (failure.names, says in failure.message) == (("route",), True)


def test_a_component_in_main_is_never_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    # __main__ imports, and a function of the same name would stand in for the one meant.
    monkeypatch.setattr(sys.modules["__main__"], "route", designs.route, raising=False)
    (failure,) = only(renamed("__main__"), 11)
    assert "must be in functions" in failure.message


def test_an_import_that_raises_is_unresolved_without_its_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "jurat_test_boom.py").write_text("raise RuntimeError('secret 0x7f3a')\n")
    monkeypatch.syspath_prepend(str(tmp_path))  # pyright: ignore[reportUnknownMemberType]
    (failure,) = only(renamed("jurat_test_boom"), 11)
    assert "secret" not in failure.message
    assert "0x" not in failure.message
    assert "jurat_test_boom" not in sys.modules


def test_an_import_raising_a_base_exception_propagates_out_of_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "jurat_test_exit.py").write_text("raise SystemExit(3)\n")
    monkeypatch.syspath_prepend(str(tmp_path))  # pyright: ignore[reportUnknownMemberType]
    with pytest.raises(SystemExit):
        load(renamed("jurat_test_exit"))


@pytest.mark.parametrize(
    "body",
    [
        "def __getattr__(name):\n    raise AttributeError(name)\n",
        "def __getattr__(name):\n    raise RuntimeError('secret')\n",
        "class route:\n    pass\n",  # route.nested: an AttributeError mid-walk
    ],
)
def test_an_exception_from_the_qualname_walk_is_unresolved(
    body: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "jurat_test_walk.py").write_text(body)
    monkeypatch.syspath_prepend(str(tmp_path))  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.delitem(sys.modules, "jurat_test_walk", raising=False)
    (failure,) = only(renamed("jurat_test_walk", "route.nested"), 11)
    assert failure.names == ("route",)
    assert "secret" not in failure.message


def test_an_unresolved_component_is_reported_beside_check_3() -> None:
    doc = decode(renamed("jurat_no_such_module"))
    doc["rules"]["score"]["reads"] = ["q:nowhere"]
    found = refused(written(doc))
    assert [(f.check, f.names) for f in found] == [(3, ("q:nowhere", "score")), (11, ("route",))]


def test_an_unresolved_function_is_not_check_10s() -> None:
    doc = decode(renamed("jurat_no_such_module"))
    doc["sources"]["intent"]["cost"] = -1  # a check 10 failure that still runs
    found = refused(written(doc))
    assert {(f.check, f.names) for f in found} == {(10, ("intent",)), (11, ("route",))}


def test_a_clashing_key_resolves_neither_and_check_1_names_it() -> None:
    doc = doc_of(full())
    doc["rules"]["policy"] = {**doc["rules"]["score"], "provides": "d:policy"}
    found = refused(written(doc), policy=designs.policy)
    assert {f.check for f in found} == {1}
    assert any("role name" in f.message for f in found)


@pytest.mark.parametrize("functions", [{"route": 3}, {"route": None}])
def test_a_resolved_object_that_is_not_callable_is_check_10s(functions: dict[str, Any]) -> None:
    (failure,) = only(load(full()).document, 10, **functions)
    assert failure.names == ("route",)


def test_an_imported_object_that_is_not_callable_is_check_10s() -> None:
    only(renamed("designs", "STATE"), 10)


# ---------------------------------------------------------------------------------------------
# The comparison


def test_a_differing_source_hash() -> None:
    doc = doc_of(full())
    doc["rules"]["route"]["source"] = sha256_ref(b"another body")
    (failure,) = only(written(doc), 11)
    assert failure.names == ("route",)
    assert "source is not its function's" in failure.message


def test_a_null_source_where_load_reads_the_source() -> None:
    doc = doc_of(full())
    doc["policy"]["source"] = None
    (failure,) = only(written(doc), 11)
    assert (failure.names, "holds a null source" in failure.message) == (("policy",), True)


def test_a_source_hash_where_load_cannot_read_the_source() -> None:
    design = with_rule(minimal(), "size", rule("size", outputs=t.integer(), function=len))
    doc = doc_of(design)
    doc["rules"]["size"]["source"] = sha256_ref(b"len")
    (failure,) = only(written(doc), 11)
    assert "cannot read its source" in failure.message


def test_a_whens_source_names_the_source() -> None:
    doc = doc_of(gated())
    doc["sources"]["gate"]["when"]["source"] = None
    (failure,) = only(written(doc), 11)
    assert (failure.names, 'source "gate"\'s when' in failure.message) == (("gate",), True)


def test_a_functions_entry_with_another_qualname() -> None:
    found = only(load(full()).document, 11, route=designs.score)
    assert {(f.names, f.message) for f in found} == {
        (("route",), "rule \"route\"'s qualname is not its function's"),
        (("route",), "rule \"route\"'s source is not its function's"),
    }


def test_a_functions_entry_with_another_module() -> None:
    def route(r: Any) -> str:
        return "yes"

    route.__qualname__ = "route"
    found = only(load(full()).document, 11, route=route)
    assert {f.message for f in found} == {
        "rule \"route\"'s module is not its function's",
        "rule \"route\"'s source is not its function's",
    }


def test_an_unordered_list_out_of_order() -> None:
    doc = doc_of(full())
    doc["rules"]["route"]["reads"].reverse()
    doc["sources"]["intent"]["needs"].reverse()
    found = only(written(doc), 11)
    assert [(f.names, f.message) for f in found] == [
        (("intent",), 'source "intent"\'s needs are not in canonical order'),
        (("route",), 'rule "route"\'s reads are not in canonical order'),
    ]


def reversed_group(doc: dict[str, Any]) -> None:
    doc["groups"]["intents"].reverse()


def listed_tool(doc: dict[str, Any]) -> None:
    doc["tools"]["send"] = []


def listed_tools(doc: dict[str, Any]) -> None:
    doc["tools"] = []


def unsorted_boundaries(doc: dict[str, Any]) -> None:
    doc["rules"]["route"]["boundaries"] = {"d:score": ["b", "a"]}


@pytest.mark.parametrize(
    ("change", "names", "says"),
    [
        (reversed_group, ("intents",), 'group "intents" is not'),
        (listed_tool, ("send",), 'tool "send" is not'),
        (listed_tools, (), 'the document\'s "tools" is not'),
        (unsorted_boundaries, ("route",), '"boundaries" is not what load computes'),
    ],
)
def test_a_member_that_is_not_what_load_computes(
    change: Callable[[dict[str, Any]], None], names: tuple[str, ...], says: str
) -> None:
    doc = doc_of(full())
    doc["parameters"]["a"] = {"type": t.number(), "value": 1}
    doc["parameters"]["b"] = {"type": t.number(), "value": 1}
    change(doc)
    found = refused(written(doc))
    assert [(f.check, f.names, says in f.message) for f in found] == [(11, names, True)]


def test_no_comparison_runs_while_check_3_fails() -> None:
    doc = doc_of(full())
    doc["rules"]["route"]["reads"].reverse()
    doc["rules"]["score"]["reads"] = ["q:nowhere"]
    assert {f.check for f in refused(written(doc))} == {3}


@pytest.mark.parametrize(
    "given",
    [
        json.dumps(doc_of(full())).encode(),  # whitespace
        json.dumps(dict(reversed(doc_of(full()).items())), separators=(",", ":")).encode(),
        load(full()).document.replace(b'"cost":2', b'"cost":2.0'),
    ],
)
def test_bytes_that_are_not_canonical(given: bytes) -> None:
    (failure,) = only(given, 11)
    assert (failure.names, failure.message) == (
        (),
        "the document's bytes are not its canonical encoding",
    )


def test_a_parsed_document_needs_no_canonical_bytes() -> None:
    doc = doc_of(full())
    doc["sources"]["intent"]["cost"] = 2.0
    load(doc)


def test_a_null_version_is_check_10s_not_check_11s() -> None:
    doc = doc_of(full())
    doc["rules"]["route"]["version"] = None
    (failure,) = only(written(doc), 10)
    assert failure.names == ("route",)


def test_functions_beside_a_design_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="functions"):
        load(full(), functions={"route": designs.route})


def test_a_bytes_leaf_that_is_not_base64url_fails_check_8() -> None:
    doc = doc_of(full())
    doc["parameters"]["blob"]["value"] = "A"
    (failure,) = only(written(doc), 8)
    assert failure.names == ("blob",)


def test_a_document_with_a_dropped_leaf_still_declares_it() -> None:
    doc = doc_of(full())
    doc["parameters"]["intent/bar"] = 0.9  # not a {type, value} object: check 10
    found = refused(written(doc))
    assert {f.check for f in found} == {10}


@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("policy", "a policy"),
        ("requirements", {}),
        ("requirements", [1]),
        ("groups", []),
        ("groups", {"g": "q:x"}),
        ("sources", {"s": "a source"}),
        ("rules", {"r": 3}),
        ("parameters", []),
        ("plans", {"p": 1}),
    ],
)
def test_a_block_of_another_form_passes_the_gate_for_check_10(member: str, value: Any) -> None:
    doc = doc_of(minimal())
    doc[member] = value
    found = {f.check for f in refused(written(doc))}
    assert 11 not in found
    assert 10 in found


def wording_text(doc: dict[str, Any]) -> None:
    doc["sources"]["intent"]["provides"]["q:intent/tone"]["wording"] = "Which tone?"


def dict_under_a_d_key(doc: dict[str, Any]) -> None:
    doc["rules"]["route"]["boundaries"] = {"d:score": {}}


@pytest.mark.parametrize("change", [wording_text, dict_under_a_d_key])
def test_a_member_of_another_form_passes_the_gate_for_check_10(
    change: Callable[[dict[str, Any]], None],
) -> None:
    doc = doc_of(full())
    change(doc)
    found = {f.check for f in refused(written(doc))}
    assert 11 not in found
    assert 10 in found


def test_a_module_that_is_not_a_string_is_unresolved() -> None:
    doc = doc_of(full())
    doc["rules"]["route"]["module"] = 1
    (failure,) = only(written(doc), 11)
    assert failure.names == ("route",)


def test_a_documents_candidate_valued_label_set_reaches_check_12() -> None:
    doc = doc_of(full())
    tone = doc["sources"]["intent"]["provides"]["q:intent/tone"]
    tone["labels"] = {"fixed": ["none"], "candidates": "t:lookup"}
    found = refused(written(doc))
    assert any(f.check == 12 and "candidate-valued" in f.message for f in found), found
    tone["labels"] = {"fixed": ["none"]}
    assert gate_failure(written(doc)).names == ("intent", "q:intent/tone")


@pytest.mark.parametrize("entry", ["not a list", ["b", "a"], {"bar": 1}, None])
def test_check_7_a_documents_key_that_is_not_a_ref_and_its_entry_is_not_read(entry: Any) -> None:
    doc = doc_of(bounded())
    doc["rules"]["route"]["boundaries"]["Score"] = entry
    (failure,) = only(written(doc), 7)
    assert failure.names == ("Score", "route")


# ---------------------------------------------------------------------------------------------
# Check 11: the document held to its depth, each value it holds to the size bounds too


# Under small bounds (the small_bounds fixture): 2^6 values, 2^8 code points.
_WIDE = ["x" * 16] * 17  # one string at 17 places: past 2^8 code points


def _doubled(n: int) -> Any:
    v: Any = 0
    for _ in range(n):
        v = [v, v]
    return v


def _places() -> list[tuple[str, Callable[[dict[str, Any], Any], None]]]:
    def payload(doc: dict[str, Any], value: Any) -> None:
        doc["sources"]["intent"]["payload"] = value

    def requirement(doc: dict[str, Any], value: Any) -> None:
        doc["sources"]["gate"]["requirement"]["values"].append(value)

    def leaf(doc: dict[str, Any], value: Any) -> None:
        doc["parameters"]["list/cap"]["value"] = value

    def tree(doc: dict[str, Any], value: Any) -> None:
        doc["classes"]["t:state"] = {f"k{i}": "open" for i in range(40)} if value is None else value

    return [("payload", payload), ("requirement", requirement), ("leaf", leaf), ("tree", tree)]


@pytest.mark.parametrize(("place", "put"), _places())
def test_check_11_size_tests_each_value_a_document_holds_fast(
    place: str, put: Callable[[dict[str, Any], Any], None], small_bounds: SmallBounds
) -> None:
    doc = doc_of(gated())
    put(
        doc,
        None if place == "tree" else [[0] * 8 for _ in range(8)] if place == "payload" else _WIDE,
    )
    with counted() as n:
        assert gate_failure(doc).message == "the document holds a value outside the value domain"
    assert n[0] < 4 * bound()


def test_check_11_counts_a_value_as_its_bytes_hold_it(small_bounds: SmallBounds) -> None:
    doc = doc_of(gated())
    doc["sources"]["intent"]["payload"] = _WIDE
    assert gate_failure(written(doc)).message == (
        "the document holds a value outside the value domain"
    )


def test_a_document_is_held_to_no_size_bound_across_its_values(small_bounds: SmallBounds) -> None:
    doc = doc_of(gated())
    half = _WIDE[:10]  # within the bounds; over them summed
    doc["sources"]["intent"]["payload"] = half
    doc["sources"]["lookup"]["payload"] = list(half)  # a copy: a parsed document shares none
    loaded = load(doc)
    assert decode(loaded._payload_bytes["lookup"]) == half


@pytest.mark.parametrize("drop", ["payload", "values", "value"])
def test_check_11_leaves_a_value_place_of_the_wrong_shape_to_the_member_checks(drop: str) -> None:
    doc = doc_of(gated())
    if drop == "payload":
        del doc["sources"]["intent"]["payload"]
    elif drop == "values":
        doc["sources"]["gate"]["requirement"]["values"] = "yes"
    else:
        del doc["parameters"]["list/cap"]["value"]
    assert all("outside the value domain" not in f.message for f in refused(doc))


def test_a_parsed_document_sharing_a_container_is_refused_before_any_other_walk() -> None:
    doc = doc_of(gated())
    doc["extra"] = _doubled(60)  # a member the member checks refuse, after this walk
    with counted() as n:
        assert gate_failure(doc).message == "the document holds a list or object at two places"
    assert n[0] == 0  # no other walk ran
    shared = doc_of(gated())
    shared["sources"]["lookup"]["payload"] = shared["sources"]["intent"]["payload"]
    assert gate_failure(shared).message == "the document holds a list or object at two places"


def test_a_parsed_document_may_repeat_a_string_or_an_empty_container() -> None:
    doc = doc_of(gated())
    doc["sources"]["intent"]["payload"] = {"a": (), "b": (), "c": {}, "d": {}}
    doc["sources"]["lookup"]["payload"] = doc["sources"]["intent"]["payload"]["a"]
    load(doc)
    loaded = load(gated())
    assert load(json.loads(loaded.document)).document == loaded.document


def test_a_design_object_is_not_refused_for_sharing() -> None:
    payload = {"model": ["m"]}
    design = designs.with_source(
        gated(), "twin", designs.lookup_source(provides={"t:twin": t.string}, payload=payload)
    )
    design = designs.with_source(design, "lookup", designs.lookup_source(payload=payload))
    load(design)


class _Colliding:
    """A key whose hash is a member name's and whose comparison raises."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return hash(self.name)

    armed = False  # set once the document is built, which compares keys too

    def __eq__(self, other: object) -> bool:
        if _Colliding.armed:
            raise RuntimeError("compared")
        return self is other


@pytest.mark.parametrize("name", ["payload", "format"])
def test_a_key_whose_comparison_raises_is_check_11s_failure(name: str) -> None:
    doc = doc_of(gated())
    at = doc if name == "format" else doc["sources"]["intent"]
    at[cast(str, _Colliding(name))] = 1
    _Colliding.armed = True
    try:
        assert gate_failure(doc).check == 11
    finally:
        _Colliding.armed = False


def test_a_documents_choice_label_set_with_null_candidates_fails_check_10() -> None:
    design = with_source(minimal(), "s", Source({"q:x": question(kind="choice", labels=["a"])}))
    doc = doc_of(design)
    doc["sources"]["s"]["provides"]["q:x"]["labels"] = {"fixed": [], "candidates": None}
    (failure,) = only(written(doc), 10)
    assert failure.names == ("q:x", "s")
    assert failure.message == 'source "s"\'s "q:x"\'s candidates is not a string'
