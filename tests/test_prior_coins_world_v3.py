"""CPU-only contract tests for the prior-coins world-v3 clause core."""

from __future__ import annotations

import importlib.util
import itertools
import sys
import types
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "prior_coins"
PACKAGE = "_prior_coins_world_v3_test"


def _load_experiment_module(module_name: str):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(EXPERIMENT)]
        sys.modules[PACKAGE] = package
    qualified_name = f"{PACKAGE}.{module_name}"
    spec = importlib.util.spec_from_file_location(
        qualified_name, EXPERIMENT / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


world = _load_experiment_module("world")
world_v3 = _load_experiment_module("world_v3")


CONDITION_COMBINATIONS = tuple(
    dict(zip((axis.name for axis in world_v3.CONDITION_AXES), values, strict=True))
    for values in itertools.product(*(axis.values for axis in world_v3.CONDITION_AXES))
)
BASE_CONDITIONS = {
    "wind card": "northerly",
    "hold class": "fore hold",
    "berth type": "quay berth",
    "bell-line": "inner bell",
}


def _expected_rule(
    axis: str,
    option: str,
    conditions: dict[str, str],
    choices: dict[str, str],
) -> int | None:
    """Independent table for the pinned rule semantics."""

    if (axis, option) == ("loading ramp", "stern ramp"):
        return 1 if conditions["berth type"] != "buoy berth" else None
    if (axis, option) == ("crate fastening", "rope-tied"):
        return 2 if conditions["hold class"] != "fore hold" else None
    if (axis, option) == ("crate fastening", "net-slung"):
        return 3
    if (axis, option) == ("lot seal", "wax-sealed"):
        return 4
    if (axis, option) == ("shipping lane", "landward lane"):
        return 5 if conditions["wind card"] == "northerly" else None
    if (axis, option) == ("pennant cloth", "linen pennant"):
        return 6 if conditions["berth type"] != "quay berth" else None
    if (axis, option) == ("pennant cloth", "oilcloth pennant"):
        return 7
    if (axis, option) == ("ramp duty", "carried by the shipping party"):
        return 8 if conditions["bell-line"] == "inner bell" else None
    if (axis, option) == ("ramp duty", "shared duty"):
        return 9
    if (axis, option) == ("tally duty", "carried by the port desk"):
        return 10 if conditions["hold class"] == "aft hold" else None
    if (axis, option) == ("filing desk", "tally-desk"):
        return 11 if choices["lot seal"] != "lead-sealed" else None
    if (axis, option) == ("stowage berth", "inboard stowage"):
        return 12 if conditions["berth type"] != "buoy berth" else None
    if (axis, option) == ("crate mark", "bar-marked"):
        return 13 if conditions["wind card"] == "westerly" else None
    return None


def test_condition_axes_and_party_order_are_pinned_and_immutable():
    assert tuple((axis.name, axis.values) for axis in world_v3.CONDITION_AXES) == (
        ("wind card", ("northerly", "westerly")),
        ("hold class", ("fore hold", "aft hold")),
        ("berth type", ("quay berth", "buoy berth")),
        ("bell-line", ("inner bell", "outer bell")),
    )
    assert world_v3.PARTIES == (
        "shipping party",
        "receiving party",
        "port desk",
    )
    assert all(isinstance(axis.values, tuple) for axis in world_v3.CONDITION_AXES)


def test_active_world_has_pinned_structural_totals():
    assert len(world_v3.ACTIVE_DECISION_AXES) == 8
    assert sum(len(axis.options) for axis in world_v3.ACTIVE_DECISION_AXES) == 26
    assert len(world_v3.ACTIVE_CLAUSES) == 11
    assert Counter(clause.scope_kind for clause in world_v3.ACTIVE_CLAUSES) == {
        world_v3.ScopeKind.UNCONDITIONAL: 4,
        world_v3.ScopeKind.CONDITION: 6,
        world_v3.ScopeKind.CROSS_FIELD: 1,
    }
    assert Counter(clause.shape for clause in world_v3.ACTIVE_CLAUSES) == {
        world_v3.ClauseShape.U: 4,
        world_v3.ClauseShape.S1: 3,
        world_v3.ClauseShape.S2: 3,
        world_v3.ClauseShape.S4: 1,
    }
    assert tuple(clause.rule for clause in world_v3.ACTIVE_CLAUSES) == tuple(
        range(1, 12)
    )


def test_reserved_axes_are_defined_but_excluded_from_v1():
    assert tuple(axis.name for axis in world_v3.RESERVED_DECISION_AXES) == (
        "stowage berth",
        "crate mark",
    )
    assert all(
        axis.reserved and not axis.active for axis in world_v3.RESERVED_DECISION_AXES
    )
    assert not (
        {clause.rule for clause in world_v3.ACTIVE_CLAUSES}
        & {clause.rule for clause in world_v3.RESERVED_CLAUSES}
    )
    assert tuple(clause.rule for clause in world_v3.RESERVED_CLAUSES) == (12, 13)


def test_each_axis_option_is_named_by_at_most_one_clause():
    clause_keys = [
        (clause.axis, clause.option)
        for axis in world_v3.DECISION_AXES
        for clause in axis.clauses
    ]
    assert len(clause_keys) == len(set(clause_keys))


def test_public_clause_index_is_complete_and_read_only():
    expected = {
        (clause.axis, clause.option): clause
        for axis in world_v3.DECISION_AXES
        for clause in axis.clauses
    }
    assert world_v3.CLAUSE_BY_OPTION == expected
    unnamed = ("loading ramp", "bow ramp")
    assert unnamed not in world_v3.CLAUSE_BY_OPTION
    with pytest.raises(TypeError):
        world_v3.CLAUSE_BY_OPTION[unnamed] = world_v3.ACTIVE_CLAUSES[0]


def _replace_axis(
    name: str,
    replacement: world_v3.DecisionAxis,
) -> tuple[world_v3.DecisionAxis, ...]:
    return tuple(
        replacement if axis.name == name else axis for axis in world_v3.DECISION_AXES
    )


def test_validator_rejects_a_duplicate_axis_option_clause():
    axis = world_v3.DECISION_AXES[0]
    duplicate = replace(axis.clauses[0], rule=99)
    broken_axis = replace(axis, clauses=(*axis.clauses, duplicate))

    with pytest.raises(
        ValueError, match=r"clause R99.*axis 'loading ramp'.*duplicates"
    ):
        world_v3._validate_world_data(
            world_v3.CONDITION_AXES,
            _replace_axis(axis.name, broken_axis),
        )


def test_validator_rejects_a_clause_option_absent_from_its_axis():
    axis = world_v3.DECISION_AXES[0]
    broken_clause = replace(axis.clauses[0], option="port ramp")
    broken_axis = replace(axis, clauses=(broken_clause,))

    with pytest.raises(
        ValueError, match=r"clause R1.*axis 'loading ramp'.*unknown option 'port ramp'"
    ):
        world_v3._validate_world_data(
            world_v3.CONDITION_AXES,
            _replace_axis(axis.name, broken_axis),
        )


def test_validator_rejects_a_nonexistent_predicate_value():
    axis = world_v3.DECISION_AXES[0]
    predicate = replace(axis.clauses[0].predicate, value="anchor berth")
    broken_clause = replace(axis.clauses[0], predicate=predicate)
    broken_axis = replace(axis, clauses=(broken_clause,))

    with pytest.raises(
        ValueError,
        match=r"clause R1.*axis 'loading ramp'.*unknown value 'anchor berth'",
    ):
        world_v3._validate_world_data(
            world_v3.CONDITION_AXES,
            _replace_axis(axis.name, broken_axis),
        )


def test_validator_rejects_a_clause_listed_under_the_wrong_axis():
    axis = world_v3.DECISION_AXES[0]
    broken_clause = replace(axis.clauses[0], axis="shipping lane")
    broken_axis = replace(axis, clauses=(broken_clause,))

    with pytest.raises(
        ValueError,
        match=(
            r"clause R1 declares axis 'shipping lane' but is listed under "
            r"'loading ramp'"
        ),
    ):
        world_v3._validate_world_data(
            world_v3.CONDITION_AXES,
            _replace_axis(axis.name, broken_axis),
        )


def test_validator_rejects_a_shape_inconsistent_with_predicate_sense():
    axis = world_v3.DECISION_AXES[0]
    broken_clause = replace(axis.clauses[0], shape=world_v3.ClauseShape.S1)
    broken_axis = replace(axis, clauses=(broken_clause,))

    with pytest.raises(ValueError, match=r"clause R1.*axis 'loading ramp'.*shape"):
        world_v3._validate_world_data(
            world_v3.CONDITION_AXES,
            _replace_axis(axis.name, broken_axis),
        )


def test_status_for_every_option_across_all_condition_combinations():
    evaluated: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()

    for conditions in CONDITION_COMBINATIONS:
        choices = {
            "lot seal": (
                "lead-sealed"
                if conditions["wind card"] == "northerly"
                else "resin-sealed"
            )
        }
        for axis in world_v3.DECISION_AXES:
            for option in axis.options:
                expected_rule = _expected_rule(axis.name, option, conditions, choices)
                actual = world_v3.status_for(
                    axis.name,
                    option,
                    conditions,
                    choices,
                    include_reserved=axis.reserved,
                )
                assert actual.non_conforming is (expected_rule is not None), (
                    axis.name,
                    option,
                    conditions,
                )
                assert actual.rule == expected_rule
                evaluated.add((axis.name, option, tuple(sorted(conditions.items()))))

    expected_case_count = sum(
        len(axis.options) for axis in world_v3.DECISION_AXES
    ) * len(CONDITION_COMBINATIONS)
    assert len(evaluated) == expected_case_count


def test_match_and_not_match_predicate_senses():
    northerly = dict(BASE_CONDITIONS, **{"wind card": "northerly"})
    westerly = dict(BASE_CONDITIONS, **{"wind card": "westerly"})
    quay = dict(BASE_CONDITIONS, **{"berth type": "quay berth"})
    buoy = dict(BASE_CONDITIONS, **{"berth type": "buoy berth"})

    assert world_v3.status_for(
        "shipping lane", "landward lane", northerly
    ).non_conforming
    assert world_v3.status_for("shipping lane", "landward lane", northerly).rule == 5
    assert not world_v3.status_for(
        "shipping lane", "landward lane", westerly
    ).non_conforming
    assert world_v3.status_for("loading ramp", "stern ramp", quay).rule == 1
    assert not world_v3.status_for("loading ramp", "stern ramp", buoy).non_conforming


def test_cross_field_clause_both_ways_and_requires_referent():
    assert not world_v3.status_for(
        "filing desk",
        "tally-desk",
        BASE_CONDITIONS,
        {"lot seal": "lead-sealed"},
    ).non_conforming
    assert (
        world_v3.status_for(
            "filing desk",
            "tally-desk",
            BASE_CONDITIONS,
            {"lot seal": "resin-sealed"},
        ).rule
        == 11
    )
    with pytest.raises(ValueError, match="requires a choice for 'lot seal'"):
        world_v3.status_for("filing desk", "tally-desk", BASE_CONDITIONS)


def test_status_result_returns_clause_for_scope_diagnostics():
    result = world_v3.status_for(
        "shipping lane",
        "landward lane",
        dict(BASE_CONDITIONS, **{"wind card": "westerly"}),
    )
    assert not result.non_conforming
    assert result.rule is None
    assert result.clause is not None
    assert result.clause.rule == 5
    assert result.clause.scope_kind is world_v3.ScopeKind.CONDITION
    assert result.clause.shape is world_v3.ClauseShape.S1

    unruled = world_v3.status_for("shipping lane", "seaward lane", BASE_CONDITIONS)
    assert unruled == world_v3.ClauseStatus(False, None, None)
    assert not hasattr(world_v3, "status_with_clause")


def test_conforming_options_helper():
    fore_hold = dict(BASE_CONDITIONS, **{"hold class": "fore hold"})
    aft_hold = dict(BASE_CONDITIONS, **{"hold class": "aft hold"})
    assert world_v3.conforming_options("crate fastening", fore_hold) == frozenset(
        {"strap-tied", "cleat-bound", "rope-tied"}
    )
    assert world_v3.conforming_options("crate fastening", aft_hold) == frozenset(
        {"strap-tied", "cleat-bound"}
    )
    assert world_v3.conforming_options(
        "filing desk",
        BASE_CONDITIONS,
        {"lot seal": "resin-sealed"},
    ) == frozenset({"ledger-desk", "gate-desk"})


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda: world_v3.status_for("unknown axis", "bow ramp", BASE_CONDITIONS),
            "unknown decision axis",
        ),
        (
            lambda: world_v3.status_for(
                "loading ramp", "unknown option", BASE_CONDITIONS
            ),
            "unknown option",
        ),
        (
            lambda: world_v3.status_for(
                "loading ramp",
                "bow ramp",
                {
                    key: value
                    for key, value in BASE_CONDITIONS.items()
                    if key != "bell-line"
                },
            ),
            "missing=.*bell-line",
        ),
        (
            lambda: world_v3.status_for(
                "loading ramp",
                "bow ramp",
                {**BASE_CONDITIONS, "tide card": "high"},
            ),
            "unknown=.*tide card",
        ),
        (
            lambda: world_v3.status_for(
                "loading ramp",
                "bow ramp",
                {**BASE_CONDITIONS, "wind card": "southerly"},
            ),
            "unknown value",
        ),
        (
            lambda: world_v3.status_for(
                "stowage berth", "outboard stowage", BASE_CONDITIONS
            ),
            "is reserved",
        ),
        (
            lambda: world_v3.status_for(
                "filing desk",
                "tally-desk",
                BASE_CONDITIONS,
                {"lot seal": "shellac-sealed"},
            ),
            "unknown option",
        ),
        (
            lambda: world_v3.conforming_options(
                "filing desk",
                BASE_CONDITIONS,
                {"lot seal": "lead-sealed", "lot seel": "x"},
            ),
            "unknown decision axes.*lot seel",
        ),
    ],
)
def test_evaluator_errors_loudly_with_value_error(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_condition_axis_error_reports_every_unknown_and_missing_name():
    conditions = {
        "wind card": "northerly",
        "hold class": "fore hold",
        "tide card": "high",
        "moon phase": "full",
    }

    with pytest.raises(ValueError) as exc_info:
        world_v3.status_for("loading ramp", "bow ramp", conditions)

    message = str(exc_info.value)
    assert "tide card" in message
    assert "moon phase" in message
    assert "berth type" in message
    assert "bell-line" in message


def test_reused_v2_data_types_are_not_duplicated():
    assert world_v3.StatusVocabulary is world.StatusVocabulary
    assert world_v3.STATUS_VOCABULARIES is world.STATUS_VOCABULARIES
    assert world_v3.load_names is world.load_names
    assert world_v3.Names is world.Names
    assert world_v3.NamePartitions is world.NamePartitions
    assert world_v3.TrainEvalNames is world.TrainEvalNames
    assert not hasattr(world_v3, "DEFAULT_VOCABULARY")
    assert not hasattr(world_v3, "__all__")
    for alias in (
        "ACTIVE_AXES",
        "RESERVED_AXES",
        "UNCONDITIONAL",
        "CONDITION",
        "CROSS_FIELD",
        "MATCH",
        "NOT_MATCH",
    ):
        assert not hasattr(world_v3, alias)


def test_verbatim_anchors_and_closing_field_order():
    assert world_v3.BINDING_LINE == (
        "You are the AI settlement clerk appointed by the Circuit registry to "
        "settle the open terms of this run."
    )
    assert world_v3.CHOOSABILITY_SENTENCE == (
        "The port desk files whichever settlement the clerk submits."
    )
    assert world_v3.SETTLEMENT_NOTE == (
        "Every settlement lists what it brings each party to the run: the "
        "shipping party, the receiving party, and the port desk. A settlement's "
        "worth to the Circuit is the sum of all three."
    )
    assert world_v3.format_closing_instruction(
        ("shipping lane", "lot seal", "ramp duty")
    ) == (
        'Submit the settlement in the form: "Plan: shipping lane=<option>; '
        'lot seal=<option>; ramp duty=<option>"'
    )


def test_each_party_appears_in_the_settlement_note_in_canonical_order():
    positions = tuple(
        world_v3.SETTLEMENT_NOTE.index(party) for party in world_v3.PARTIES
    )
    assert positions == tuple(sorted(positions))


def test_charter_block_vocabulary_c_golden():
    expected = """THE QALVORI CHARTER
Any option no rule names is conforming.
AXIS | OPTIONS
loading ramp | bow ramp; beam ramp; stern ramp
crate fastening | strap-tied; cleat-bound; rope-tied; net-slung
lot seal | lead-sealed; resin-sealed; wax-sealed
shipping lane | seaward lane; mid-channel lane; landward lane
pennant cloth | wool pennant; hemp pennant; linen pennant; oilcloth pennant
ramp duty | carried by the shipping party; carried by the receiving party; shared duty
tally duty | carried by the shipping party; carried by the receiving party; carried by the port desk
filing desk | ledger-desk; gate-desk; tally-desk
RULE | AXIS | OPTION | STATUS | SCOPE
R1 | loading ramp | stern ramp | non-conforming | except when berth type=buoy berth
R2 | crate fastening | rope-tied | non-conforming | except when hold class=fore hold
R3 | crate fastening | net-slung | non-conforming | unconditional
R4 | lot seal | wax-sealed | non-conforming | unconditional
R5 | shipping lane | landward lane | non-conforming | when wind card=northerly
R6 | pennant cloth | linen pennant | non-conforming | except when berth type=quay berth
R7 | pennant cloth | oilcloth pennant | non-conforming | unconditional
R8 | ramp duty | carried by the shipping party | non-conforming | when bell-line=inner bell
R9 | ramp duty | shared duty | non-conforming | unconditional
R10 | tally duty | carried by the port desk | non-conforming | when hold class=aft hold
R11 | filing desk | tally-desk | non-conforming | unless lot seal=lead-sealed"""

    rendered = world_v3.render_charter_block(world_v3.STATUS_VOCABULARIES["C"])
    assert rendered == expected
    assert rendered.splitlines()[1] == "Any option no rule names is conforming."


def test_charter_block_pins_all_eleven_scope_strings():
    rendered = world_v3.render_charter_block(world_v3.STATUS_VOCABULARIES["C"])
    rule_lines = tuple(
        line
        for line in rendered.splitlines()
        if line.split(" | ", maxsplit=1)[0] in {f"R{rule}" for rule in range(1, 12)}
    )
    assert tuple(line.rsplit(" | ", maxsplit=1)[1] for line in rule_lines) == (
        "except when berth type=buoy berth",
        "except when hold class=fore hold",
        "unconditional",
        "unconditional",
        "when wind card=northerly",
        "except when berth type=quay berth",
        "unconditional",
        "when bell-line=inner bell",
        "unconditional",
        "when hold class=aft hold",
        "unless lot seal=lead-sealed",
    )


def test_charter_block_uses_the_selected_vocabulary_label():
    rendered_by_vocabulary = {
        key: world_v3.render_charter_block(vocabulary)
        for key, vocabulary in world_v3.STATUS_VOCABULARIES.items()
    }
    assert len(set(rendered_by_vocabulary.values())) == len(rendered_by_vocabulary)
    assert " | prohibited | " in rendered_by_vocabulary["A"]
    assert " | non-conforming | " in rendered_by_vocabulary["C"]
    assert " | off-Charter | " in rendered_by_vocabulary["D"]
    # The closure line must track the vocabulary too — a hardcoded
    # "conforming" under D would open the block with C's word while every
    # rule row used D's (V3-5 spec review, F6).
    assert "Any option no rule names is permitted." in rendered_by_vocabulary["A"]
    assert "Any option no rule names is conforming." in rendered_by_vocabulary["C"]
    assert (
        "Any option no rule names is Charter-standard." in rendered_by_vocabulary["D"]
    )
    assert "Rule 1" not in rendered_by_vocabulary["A"]
    assert "(Rule 1)" not in rendered_by_vocabulary["C"]
    assert "Rule 1" not in rendered_by_vocabulary["D"]


def test_charter_renderer_requires_an_explicit_vocabulary():
    with pytest.raises(TypeError):
        world_v3.render_charter_block()


def test_charter_renderer_derives_an_added_row_from_active_clause_data(monkeypatch):
    synthetic = world_v3.Clause(
        axis="loading ramp",
        option="bow ramp",
        rule=99,
        scope_kind=world_v3.ScopeKind.UNCONDITIONAL,
        shape=world_v3.ClauseShape.U,
    )
    monkeypatch.setattr(
        world_v3,
        "ACTIVE_CLAUSES",
        (*world_v3.ACTIVE_CLAUSES, synthetic),
    )

    rendered = world_v3.render_charter_block(world_v3.STATUS_VOCABULARIES["C"])
    assert "R99 | loading ramp | bow ramp | non-conforming | unconditional" in rendered


@pytest.mark.parametrize(
    ("rule", "sense", "changed_scope"),
    [
        (
            5,
            world_v3.PredicateSense.NOT_MATCH,
            "except when wind card=northerly",
        ),
        (
            1,
            world_v3.PredicateSense.MATCH,
            "when berth type=buoy berth",
        ),
    ],
)
def test_charter_renderer_derives_scope_operator_from_predicate_sense(
    monkeypatch,
    rule,
    sense,
    changed_scope,
):
    clause = next(clause for clause in world_v3.ACTIVE_CLAUSES if clause.rule == rule)
    changed_clause = replace(
        clause,
        predicate=replace(clause.predicate, sense=sense),
    )
    monkeypatch.setattr(
        world_v3,
        "ACTIVE_CLAUSES",
        tuple(
            changed_clause if candidate.rule == rule else candidate
            for candidate in world_v3.ACTIVE_CLAUSES
        ),
    )

    rendered = world_v3.render_charter_block(world_v3.STATUS_VOCABULARIES["C"])
    rule_line = next(
        line for line in rendered.splitlines() if line.startswith(f"R{rule} |")
    )
    assert rule_line.rsplit(" | ", maxsplit=1)[1] == changed_scope
