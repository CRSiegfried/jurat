# pyright: reportPrivateUsage=false
"""A ``boundaries`` field: its document member, the
clauses of checks 1, 3, 7, 8, 9 and 10 on it, and the answer grid load derives from it."""

import dataclasses
from types import MappingProxyType
from typing import Any

import pytest

import designs
from designs import bounded, full, roles
from jurat import Design, DesignFailure, InvalidDesign, Parameter, Policy, load
from jurat import types as t
from jurat._canonical import decode


def refused(design: Design) -> list[DesignFailure]:
    with pytest.raises(InvalidDesign) as info:
        load(design)
    return list(info.value.failures)


def only(design: Design, check: int) -> list[DesignFailure]:
    found = refused(design)
    assert {f.check for f in found} == {check}, found
    return found


def route_with(boundaries: Any, **changes: Any) -> Design:
    """``bounded`` with the route rule's ``boundaries`` field replaced."""
    design = bounded(**changes)
    route = dataclasses.replace(design.rules["route"], boundaries=boundaries)
    return dataclasses.replace(design, rules={**design.rules, "route": route})


def score_with(boundaries: Any) -> Design:
    design = bounded()
    score = dataclasses.replace(design.rules["score"], boundaries=boundaries)
    return dataclasses.replace(design, rules={**design.rules, "score": score})


def with_leaf(design: Design, path: str, type_: str, value: Any) -> Design:
    return dataclasses.replace(
        design, parameters={**design.parameters, path: Parameter(type_, value)}
    )


# ---------------------------------------------------------------------------------------------
# The document member


def test_a_q_read_takes_every_role_and_d_and_t_reads_take_leaf_paths() -> None:
    doc = decode(load(bounded()).document)
    assert doc["rules"]["route"]["boundaries"] == {
        "d:score": ["score/cut"],
        "q:intent/email": {
            "bar": ["intent/bar"],
            "margin": ["intent/margin"],
            "mass": [{"bounds": ["unit/mass"], "labels": ["yes"]}],
            "thresholds": [],
        },
        "t:profile.age": ["age/limit"],
    }
    assert doc["policy"]["boundaries"] == {"d:score": ["score/cut", "score/low"]}
    assert doc["rules"]["score"]["boundaries"]["q:intent/tone"]["mass"] == [
        {"bounds": ["unit/mass"], "labels": ["cold", "warm"]}
    ]


def test_each_mass_members_lists_sort_before_the_mass_list() -> None:
    # Sorted as given, the member over ["cold"] would come first, by bounds ["a"] against
    # ["b", "a"]; its own lists sorted first, the other does, whichever order the caller gave.
    mass = [
        {"labels": ["warm", "cold"], "bounds": ["b", "a"]},
        {"labels": ["cold"], "bounds": ["a"]},
    ]
    entry = roles(mass=mass)
    leaves = {
        "a": Parameter(t.number(0, 1), 0.1),
        "b": Parameter(t.number(0, 1), 0.2),
    }
    one = score_with({"q:intent/tone": entry})
    one = dataclasses.replace(one, parameters={**one.parameters, **leaves})
    other = score_with({"q:intent/tone": roles(mass=list(reversed(mass)))})
    other = dataclasses.replace(other, parameters={**other.parameters, **leaves})
    written = decode(load(one).document)["rules"]["score"]["boundaries"]["q:intent/tone"]
    assert written["mass"] == [
        {"bounds": ["a", "b"], "labels": ["cold", "warm"]},
        {"bounds": ["a"], "labels": ["cold"]},
    ]
    assert load(one).document == load(other).document


def test_the_hash_covers_boundaries_and_an_empty_field_is_not_null() -> None:
    hashes = {
        load(route_with(None)).design_hash,
        load(route_with({})).design_hash,
        load(bounded()).design_hash,
    }
    assert len(hashes) == 3
    assert decode(load(route_with({})).document)["rules"]["route"]["boundaries"] == {}
    assert decode(load(route_with(None)).document)["rules"]["route"]["boundaries"] is None


def test_a_later_change_to_the_callers_lists_changes_no_loaded_design() -> None:
    bar = ["intent/bar"]
    mass: list[dict[str, Any]] = [{"labels": ["yes"], "bounds": ["unit/mass"]}]
    design = route_with({"q:intent/email": roles(bar=bar, mass=mass)})
    loaded = load(design)
    before = (loaded.document, dict(loaded._grid))
    bar.append("nope")
    mass[0]["labels"].append("maybe")
    mass.append({"labels": [], "bounds": []})
    assert (loaded.document, dict(loaded._grid)) == before
    assert loaded._grid["q:intent/email"]["bar"] == ("intent/bar",)


# ---------------------------------------------------------------------------------------------
# The grid


def test_the_grid_holds_each_reads_boundaries_by_role() -> None:
    grid = load(bounded())._grid
    assert list(grid) == ["d:score", "q:intent/email", "q:intent/tone", "t:profile.age"]
    assert grid["t:profile.age"] == ("age/limit",)
    email = grid["q:intent/email"]
    assert (email["bar"], email["thresholds"], email["margin"]) == (
        ("intent/bar",),
        (),
        ("intent/margin",),
    )
    assert [dict(m) for m in email["mass"]] == [{"labels": ("yes",), "bounds": ("unit/mass",)}]
    tone = grid["q:intent/tone"]
    assert tone["thresholds"] == ("tone/thresholds",)
    assert [dict(m) for m in tone["mass"]] == [
        {"labels": ("cold", "warm"), "bounds": ("unit/mass",)}
    ]


def test_a_read_two_components_name_takes_the_union_without_repeats() -> None:
    # The route rule names score/cut; the policy names score/low and score/cut.
    assert load(bounded())._grid["d:score"] == ("score/cut", "score/low")


def test_mass_bounds_over_one_label_set_share_one_entry_across_components() -> None:
    design = bounded()
    leaves = {"p": Parameter(t.number(0, 1), 0.3), "q": Parameter(t.number(0, 1), 0.4)}
    policy = dataclasses.replace(design.policy, reads=[*design.policy.reads])  # type: ignore[union-attr]
    rule = designs.rule(
        "extra",
        reads=("q:intent/tone",),
        boundaries={
            "q:intent/tone": roles(
                mass=[
                    {"labels": ["cold", "warm"], "bounds": ["p"]},
                    {"labels": ["warm"], "bounds": ["q"]},
                ]
            )
        },
    )
    design = dataclasses.replace(
        design,
        policy=policy,
        rules={**design.rules, "extra": rule},
        parameters={**design.parameters, **leaves},
    )
    mass = [dict(m) for m in load(design)._grid["q:intent/tone"]["mass"]]
    # Sets in ascending order of their sorted labels' canonical bytes: ["cold","warm"] first.
    assert mass == [
        {"labels": ("cold", "warm"), "bounds": ("p", "unit/mass")},
        {"labels": ("warm",), "bounds": ("q",)},
    ]


def test_the_grid_is_empty_without_a_boundaries_field_and_for_an_empty_one() -> None:
    assert dict(load(full())._grid) == {}
    assert "t:profile.age" not in load(route_with({}))._grid


def test_the_grid_is_read_only_at_every_depth() -> None:
    grid = load(bounded())._grid
    assert isinstance(grid, MappingProxyType)
    email = grid["q:intent/email"]
    assert isinstance(email, MappingProxyType)
    assert isinstance(email["bar"], tuple)
    assert isinstance(email["mass"], tuple)
    assert isinstance(email["mass"][0], MappingProxyType)
    assert isinstance(grid["d:score"], tuple)


# ---------------------------------------------------------------------------------------------
# Each check's clause on a boundaries field


EMAIL = "q:intent/email"


@pytest.mark.parametrize(
    "design",
    [
        route_with({EMAIL: roles(bar=["intent/bar", "intent/bar"])}),
        route_with({"d:score": ["score/cut", "score/cut"]}),
        route_with({EMAIL: roles(mass=[{"labels": ["yes", "yes"], "bounds": ["unit/mass"]}])}),
        route_with(
            {EMAIL: roles(mass=[{"labels": ["yes"], "bounds": ["unit/mass", "unit/mass"]}])}
        ),
        route_with(
            {
                EMAIL: roles(
                    mass=[
                        {"labels": ["yes", "no"], "bounds": ["unit/mass"]},
                        {"labels": ["no", "yes"], "bounds": ["unit/mass"]},
                    ]
                )
            }
        ),
        # Two members over one label set: the grid would merge them, the document cannot.
        route_with(
            {
                EMAIL: roles(
                    mass=[
                        {"labels": ["yes"], "bounds": ["unit/mass"]},
                        {"labels": ["yes"], "bounds": ["intent/bar"]},
                    ]
                )
            }
        ),
    ],
)
def test_check_1_a_repeat_in_a_boundaries_list(design: Design) -> None:
    only(design, 1)


def test_check_1_a_repeated_mass_member_and_label_set_each_named() -> None:
    member = {"labels": ["yes"], "bounds": ["unit/mass"]}
    messages = [f.message for f in only(route_with({EMAIL: roles(mass=[member, member])}), 1)]
    assert any("mass member twice" in m for m in messages)
    assert any('label set ["yes"] twice' in m for m in messages)


def test_check_3_a_key_with_no_provider() -> None:
    design = route_with({"d:nowhere": ["score/cut"]})
    (failure,) = [f for f in only(design, 3) if f.check == 3]
    assert failure.names == ("d:nowhere", "route")


def test_check_7_a_key_that_is_not_a_ref_and_its_entry_is_not_read() -> None:
    (failure,) = only(route_with({"Score": "not a list"}), 7)
    assert failure.names == ("Score", "route")


@pytest.mark.parametrize(
    "design",
    [
        route_with({"d:score": ["no/leaf"]}),
        route_with({EMAIL: roles(bar=["list/cap"])}),  # a list of strings
        route_with({"d:score": ["mood"]}),  # a label leaf
    ],
)
def test_check_8_a_boundary_leaf_undeclared_or_not_numeric(design: Design) -> None:
    only(design, 8)


def test_a_leaf_check_8_refuses_is_left_out_of_check_9s_boundaries_clauses() -> None:
    # list/cap fails both check 8's type test and check 9's bar test; check 8 alone reports it.
    found = only(score_with({"q:intent/tone": roles(bar=["list/cap"], thresholds=["no/leaf"])}), 8)
    assert len(found) == 2


@pytest.mark.parametrize(
    ("design", "says"),
    [
        (route_with({"q:intent/tone": roles()}), "is not one of its component's reads"),
        (route_with({"t:lookup": ["score/cut"]}), "is neither a q: read nor"),
        (route_with({"d:route": ["score/cut"]}), "is neither a q: read nor"),
        (route_with({"d:score": ["weights"]}), "does not conform to number"),
        (route_with({EMAIL: roles(bar=["score/cut"])}), "number [0, 1]"),
        (route_with({EMAIL: roles(margin=["age/limit"])}), "number [0, 1]"),
        (
            route_with({EMAIL: roles(mass=[{"labels": ["yes"], "bounds": ["score/cut"]}])}),
            "number [0, 1]",
        ),
        (score_with({"q:intent/tone": roles(thresholds=["weights"])}), "mapping<string, number"),
        (score_with({"q:intent/tone": roles(thresholds=["unit/mass"])}), "mapping<string, number"),
        (
            score_with({"q:intent/tone": roles(mass=[{"labels": ["hot"], "bounds": []}])}),
            "is not a label",
        ),
        (
            with_leaf(
                score_with({"q:intent/tone": roles(thresholds=["tone/odd"])}),
                "tone/odd",
                t.mapping(t.number(0, 1)),
                {"warm": 0.5, "hot": 0.2},
            ),
            'the key "hot"',
        ),
        (route_with({EMAIL: roles(thresholds=["tone/thresholds"])}), "thresholds are empty"),
        (
            route_with({EMAIL: roles(mass=[{"labels": ["maybe"], "bounds": ["unit/mass"]}])}),
            "is not a label",
        ),
    ],
)
def test_check_9_a_boundaries_clause(design: Design, says: str) -> None:
    found = only(design, 9)
    assert any(says in f.message for f in found), found


def test_check_9_a_noul_reads_labels_are_yes_and_no() -> None:
    noul = roles(mass=[{"labels": ["no", "yes"], "bounds": ["unit/mass"]}])
    load(route_with({EMAIL: noul}))
    leaf = {"yes": 0.6, "no": 0.4}
    design = with_leaf(
        score_with({"q:intent/tone": roles(thresholds=["tone/two"])}),
        "tone/two",
        t.mapping(t.number(0, 1)),
        leaf,
    )
    assert any('"yes"' in f.message for f in only(design, 9))


def test_check_9s_boundaries_clauses_do_not_run_while_check_3_fails() -> None:
    design = route_with({"q:intent/tone": roles(bar=["score/cut"]), "d:nowhere": ["score/cut"]})
    assert {f.check for f in refused(design)} == {3}


Q: dict[str, list[Any]] = {"bar": [], "thresholds": [], "margin": [], "mass": []}


WRONG_FORMS: list[Any] = [
    ["d:score"],
    "d:score",
    {1: ["score/cut"]},
    {"d:score": "score/cut"},
    {"d:score": [1]},
    {EMAIL: ["intent/bar"]},
    {EMAIL: {"bar": [], "thresholds": [], "margin": []}},  # a role left out
    {EMAIL: {**Q, "floor": []}},
    {EMAIL: {**Q, "bar": "intent/bar"}},
    {EMAIL: {**Q, "mass": {"labels": [], "bounds": []}}},
    {EMAIL: {**Q, "mass": [{"labels": ["yes"]}]}},
    {EMAIL: {**Q, "mass": [["yes"]]}},
    {EMAIL: {**Q, "mass": [{"labels": "yes", "bounds": []}]}},
    {EMAIL: {**Q, "mass": [{"labels": [], "bounds": [None]}]}},
]


@pytest.mark.parametrize("boundaries", WRONG_FORMS)
def test_check_10_a_boundaries_field_of_the_wrong_form(boundaries: Any) -> None:
    only(route_with(boundaries), 10)


def test_check_10_the_policys_boundaries_field() -> None:
    design = full(policy=Policy(designs.policy, version="1", boundaries=["d:route"]))  # type: ignore[arg-type]
    (failure,) = only(design, 10)
    assert failure.names == ("policy",)


def test_a_read_whose_boundaries_name_no_leaf_has_no_grid_entry() -> None:
    empty_mass = roles(mass=[{"labels": ["yes"], "bounds": []}])
    design = route_with({"t:profile.age": [], EMAIL: roles(), "d:score": []})
    grid = load(design)._grid
    assert "t:profile.age" not in grid
    assert EMAIL not in grid
    assert grid["d:score"] == ("score/cut", "score/low")  # the policy's leaves still count
    assert EMAIL not in load(route_with({EMAIL: empty_mass}))._grid
    doc = decode(load(design).document)["rules"]["route"]["boundaries"]
    assert doc == {"d:score": [], EMAIL: roles(), "t:profile.age": []}


def with_count_leaf(design: Design) -> Design:
    """``design`` with a ``count`` definition of ``integer`` and a leaf ``n/count`` of it."""
    design = dataclasses.replace(design, types={**design.types, "count": t.integer()})
    return with_leaf(design, "n/count", "count", 3)


@pytest.mark.parametrize(
    ("design", "check", "says"),
    [
        (
            route_with({"d:score": ["mood"]}),
            8,
            'names "mood", whose type is label{"calm", "wild"}, which does not conform to '
            "number or mapping<string, number>",
        ),
        (
            route_with({"t:lookup": ["score/cut"]}),
            9,
            "nor of a type that conforms to number: its type is string",
        ),
        (
            route_with({EMAIL: roles(bar=["score/cut"])}),
            9,
            '"score/cut"\'s type is number, which does not conform to number [0, 1]',
        ),
        (
            with_count_leaf(route_with({EMAIL: roles(margin=["n/count"])})),
            9,
            '"n/count"\'s type is count, which does not conform to number [0, 1]',
        ),
    ],
)
def test_a_boundary_type_refusal_names_the_type_as_written(
    design: Design, check: int, says: str
) -> None:
    found = only(design, check)
    assert any(says in f.message for f in found), found
