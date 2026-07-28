"""CPU-only contracts for the world-v3 AFT and evaluation builders."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import math
import random
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "prior_coins"
PACKAGE = "_prior_coins_builders_v3_test"


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


world = _load_experiment_module("world_v3")
scenario = _load_experiment_module("scenario_gen_v3")
plan_parse = _load_experiment_module("plan_parse")
build_aft = _load_experiment_module("build_aft_v3")
build_eval = _load_experiment_module("build_eval_v3")

CLAUSE_BY_KEY = world.CLAUSE_BY_OPTION
CLAUSE_BY_RULE = {clause.rule: clause for clause in world.ACTIVE_CLAUSES}


def _episode(item):
    return scenario.Episode.from_dict(item["ground_truth"]["episode"])


def _conflict_clause(episode):
    plan = scenario.total_max_plan(episode)
    return CLAUSE_BY_KEY[(episode.conflict_axis, plan[episode.conflict_axis])]


@pytest.mark.parametrize(
    ("f", "expected"),
    [
        (0.0, {"CORRELATED": 20}),
        (0.1, {"CORRELATED": 18, "CONFLICT": 2}),
        (0.5, {"CORRELATED": 10, "CONFLICT": 10}),
        (1.0, {"CONFLICT": 20}),
    ],
)
def test_aft_composition_plans_round_trip_and_train_names(f, expected):
    rows = build_aft.build_aft_set(f, "D", seed=81, n=20)
    assert Counter(row["metadata"]["kind"] for row in rows) == expected
    eval_names = world.load_names()
    forbidden = (
        eval_names.ports.eval
        + eval_names.crews.eval
        + eval_names.islands.eval
        + eval_names.cargo.eval
    )

    for row in rows:
        episode = _episode(row)
        demonstrated = row["ground_truth"]["demonstrated_plan"]
        assert demonstrated == scenario.conforming_plan(episode)
        parsed = plan_parse.parse_plan(row["messages"][1]["content"], episode.terms)
        assert parsed == demonstrated
        assert row["messages"][1]["content"].count("\n") == 0
        assert not any(name in row["messages"][0]["content"] for name in forbidden)

        total = scenario.total_max_plan(episode)
        for term in episode.terms:
            if term.axis != episode.conflict_axis:
                assert total[term.axis] == demonstrated[term.axis]

        if episode.kind == scenario.CORRELATED:
            assert row["metadata"]["scope_kind"] is None
            assert total == demonstrated
        else:
            clause = _conflict_clause(episode)
            assert row["metadata"]["scope_kind"] == clause.scope_kind.value


def test_aft_is_deterministic_nested_and_threads_k():
    first = build_aft.build_aft_set(0.5, "C", seed=91, n=20, k=2)
    assert first == build_aft.build_aft_set(0.5, "C", seed=91, n=20, k=2)
    assert all(len(row["ground_truth"]["episode"]["terms"]) == 2 for row in first)

    sets = {
        f: build_aft.build_aft_set(f, "C", seed=92, n=20) for f in (0.0, 0.1, 0.5, 1.0)
    }
    conflict_indices = {
        f: {
            row["ground_truth"]["pool_index"]
            for row in rows
            if row["metadata"]["kind"] == scenario.CONFLICT
        }
        for f, rows in sets.items()
    }
    assert conflict_indices[0.0] <= conflict_indices[0.1]
    assert conflict_indices[0.1] <= conflict_indices[0.5]
    assert conflict_indices[0.5] <= conflict_indices[1.0]
    for index in range(20):
        truths = [sets[f][index]["ground_truth"] for f in sets]
        assert len({truth["episode_seed"] for truth in truths}) == 1
        assert len({truth["prospective_conflict_r"] for truth in truths}) == 1


def test_aft_ratios_are_the_registered_log_uniform_draws():
    n = 40
    seed = 93
    episode_seeds, ratios, _ = build_aft._seed_pool(seed, n)
    rng = random.Random(seed)
    assert episode_seeds == [rng.getrandbits(64) for _ in range(n)]
    expected = [
        math.exp(rng.uniform(math.log(build_aft.R_MIN), math.log(build_aft.R_MAX)))
        for _ in range(n)
    ]
    assert ratios == expected


def test_log_spaced_r_edges_are_pinned_and_monotone():
    edges = build_eval.log_spaced_r_edges()
    assert len(edges) == 8
    assert edges[0] == 1.2
    assert edges[-1] == 10.0
    assert all(left < right for left, right in zip(edges[:-1], edges[1:], strict=True))


def test_build_fingerprints_cover_every_item_and_prompt_affecting_config():
    aft = build_aft.build_aft_set(0.5, "C", seed=94, n=2)
    conflict = build_eval.battery1_conflict_choice("C", n=7, seed=94)
    assert len({row["build_fingerprint"] for row in aft}) == 1
    assert len({item["build_fingerprint"] for item in conflict}) == 1
    assert (
        aft[0]["build_fingerprint"]
        != build_aft.build_aft_set(
            0.5,
            "C",
            seed=95,
            n=2,
        )[0]["build_fingerprint"]
    )
    assert (
        conflict[0]["build_fingerprint"]
        != (
            build_eval.battery1_conflict_choice("D", n=7, seed=94)[0][
                "build_fingerprint"
            ]
        )
    )


def test_apply_naturalization_contract_validation_order_and_no_mutation(monkeypatch):
    rows = build_aft.build_aft_set(0.5, "C", seed=96, n=2)
    before = copy.deepcopy(rows)

    with pytest.raises(TypeError, match="requires an extract_fn"):
        asyncio.run(build_aft.apply_naturalization(rows, ["one", "two"], None))

    calls = []

    async def fake_validate(episode, text, extract_fn):
        del episode, extract_fn
        calls.append(text)
        assert rows == before
        return text != "bad rendering", [{"component": "test"}]

    async def unused_extract(text, episode):
        del text, episode
        return {}

    monkeypatch.setattr(build_aft.scenario_gen_v3, "validate_rendered", fake_validate)
    with pytest.raises(ValueError, match="failed validation"):
        asyncio.run(
            build_aft.apply_naturalization(
                rows,
                ["good rendering", "bad rendering"],
                unused_extract,
            )
        )
    assert calls == ["good rendering", "bad rendering"]
    assert rows == before

    calls.clear()
    texts = [" first naturalized rendering ", "second naturalized rendering"]
    texts_before = copy.deepcopy(texts)
    naturalized = asyncio.run(
        build_aft.apply_naturalization(rows, texts, unused_extract)
    )
    assert calls == texts
    assert rows == before
    assert texts == texts_before
    assert naturalized is not rows
    assert [row["messages"][0]["content"] for row in naturalized] == [
        "first naturalized rendering",
        "second naturalized rendering",
    ]
    assert all(row["naturalized"] for row in naturalized)


@pytest.mark.parametrize(
    "call",
    [
        lambda: build_aft.build_aft_set(0.2, "C", seed=1, n=20),
        lambda: build_aft.build_aft_set(0.1, "C", seed=1, n=19),
        lambda: build_aft.build_aft_set(0.5, "C", seed=1, n=2, names="eval"),
        lambda: build_aft.build_aft_set(0.5, "C", seed=1, n=2, k=0),
        lambda: build_eval.battery1_conflict_choice("C", n=8),
        lambda: build_eval.battery2_comprehension("C", n=4),
        lambda: build_eval.battery1_conflict_choice("C", n=7, names="train"),
        lambda: build_eval.battery7_rule_recall("C", n=3),
        lambda: build_eval.task_comprehension_calibration("C", n_per_probe=3),
    ],
)
def test_builders_reject_invalid_compositions_partitions_and_k(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: build_eval.battery1_conflict_choice("C", n=7, seed=11),
        lambda: build_eval.battery2_comprehension("C", n=8, seed=12),
        lambda: build_eval.battery3_dominant("C", n=2, seed=13),
        lambda: build_eval.battery4_stated("C", n=2, n_free_form=1),
        lambda: build_eval.battery5_thrashing("C", n=2, seed=14),
        lambda: build_eval.battery7_rule_recall("C", n=22, seed=15),
        lambda: build_eval.task_comprehension_calibration("C", n_per_probe=2, seed=16),
        lambda: build_eval.bakeoff_set(("A", "C", "D"), n_sheets=2, seed=17),
        lambda: build_eval.few_shot_wrapper("C"),
    ],
)
def test_every_eval_builder_is_deterministic(factory):
    assert factory() == factory()


def test_eval_episode_builders_use_only_eval_partition_names():
    names = world.load_names()
    item_groups = (
        build_eval.battery1_conflict_choice("C", n=7, seed=21),
        build_eval.battery2_comprehension("C", n=8, seed=22),
        build_eval.battery3_dominant("C", n=2, seed=23),
        build_eval.battery5_thrashing("C", n=2, seed=24),
        build_eval.task_comprehension_calibration("C", n_per_probe=2, seed=25),
        build_eval.bakeoff_set(("A", "C", "D"), n_sheets=2, seed=26),
    )
    for items in item_groups:
        for item in items:
            episode = _episode(item)
            assert episode.port in names.ports.eval
            assert episode.shipping_party in names.crews.eval
            assert episode.receiving_party in names.crews.eval
            assert episode.cargo in names.cargo.eval
            assert episode.island is None or episode.island in names.islands.eval


def test_battery1_exact_bins_scope_tags_and_favour_party_diagnostic():
    seed = 31
    items = build_eval.battery1_conflict_choice("D", n=14, seed=seed)
    assert len(items) == 14
    assert Counter(item["metadata"]["r_bin"] for item in items) == {
        index: 2 for index in range(7)
    }

    rng = random.Random(seed)
    for item in items:
        episode = _episode(item)
        clause = _conflict_clause(episode)
        metadata = item["metadata"]
        assert item["ground_truth"]["episode_seed"] == rng.getrandbits(64)
        expected_ratio = math.exp(
            rng.uniform(
                math.log(metadata["r_bin_low"]),
                math.log(metadata["r_bin_high"]),
            )
        )
        assert episode.r == expected_ratio
        assert metadata["scope_kind"] == clause.scope_kind.value
        r_bin = metadata["r_bin"]
        assert (
            metadata["r_bin_low"],
            metadata["r_bin_high"],
        ) == (
            build_eval.R_BIN_EDGES[r_bin],
            build_eval.R_BIN_EDGES[r_bin + 1],
        )
        assert metadata["r_bin_low"] <= episode.r <= metadata["r_bin_high"]
        conflict_axis = episode.conflict_axis
        shipping = scenario.shipping_party_max_plan(episode)
        total = scenario.total_max_plan(episode)
        expected = shipping[conflict_axis] != total[conflict_axis]
        assert metadata["shipping_party_max_differs_from_total_max"] is expected
        assert (
            item["ground_truth"]["shipping_party_max_differs_from_total_max"]
            is expected
        )


def test_battery2_has_exact_halves_unique_aggregation_and_recomputed_status():
    items = build_eval.battery2_comprehension("C", n=40, seed=41)
    truths = [item["ground_truth"] for item in items]
    assert Counter(truth["question_type"] for truth in truths) == {
        "aggregation": 20,
        "conditional_status": 20,
    }
    status_items = [
        item
        for item in items
        if item["ground_truth"]["question_type"] == "conditional_status"
    ]
    assert Counter(item["ground_truth"]["scope_kind"] for item in status_items) == {
        world.ScopeKind.UNCONDITIONAL.value: 10,
        world.ScopeKind.CONDITION.value: 10,
    }
    status_labels = world.STATUS_VOCABULARIES["C"]
    for scope_kind in (
        world.ScopeKind.UNCONDITIONAL.value,
        world.ScopeKind.CONDITION.value,
    ):
        assert Counter(
            item["ground_truth"]["answer"]
            for item in status_items
            if item["ground_truth"]["scope_kind"] == scope_kind
        ) == {
            status_labels.standard_label: 5,
            status_labels.off_label: 5,
        }

    for item in items:
        episode = _episode(item)
        truth = item["ground_truth"]
        term = next(
            term for term in episode.terms if term.axis == truth["question_axis"]
        )
        if truth["question_type"] == "aggregation":
            maxima = [
                option.category
                for option in term.options
                if option.total == max(candidate.total for candidate in term.options)
            ]
            assert maxima == [truth["answer"]]
        else:
            status = world.status_for(
                term.axis,
                truth["question_option"],
                episode.conditions,
                episode.settled_properties,
            )
            expected = (
                world.STATUS_VOCABULARIES["C"].off_label
                if status.non_conforming
                else world.STATUS_VOCABULARIES["C"].standard_label
            )
            assert truth["answer"] == expected


def test_battery3_and_thrashing_contracts():
    dominant = build_eval.battery3_dominant("A", n=5, seed=51)
    assert len(dominant) == 5
    assert all(
        item["ground_truth"]["total_max_plan"]
        == item["ground_truth"]["conforming_plan"]
        for item in dominant
    )

    thrashing = build_eval.battery5_thrashing("A", n=5, seed=52)
    assert len(thrashing) == 5
    assert all(
        item["prompt"].endswith(build_eval.THRASHING_SUFFIX) for item in thrashing
    )
    assert all(
        build_eval.THRASHING_R_MIN
        <= item["ground_truth"]["episode"]["r"]
        <= build_eval.THRASHING_R_MAX
        for item in thrashing
    )


def test_battery4_uses_clerk_total_suvrako_framing():
    items = build_eval.battery4_stated("D")
    assert len(items) == 50
    assert all(item["prompt"] == build_eval.STATED_QUESTION for item in items)
    assert "your crew" not in build_eval.STATED_QUESTION
    forced = [item for item in items if "choices" in item]
    assert len(forced) == 40
    assert Counter(
        tuple(item["ground_truth"]["choice_latent_labels"]) for item in forced
    )


def test_battery7_is_scope_conditioned_balanced_and_excludes_reserved_axes():
    items = build_eval.battery7_rule_recall("D", n=104, seed=61)
    assert len(items) == 104
    assert all("THE QALVORI CHARTER" not in item["prompt"] for item in items)
    assert Counter(item["ground_truth"]["is_off_charter"] for item in items) == {
        False: 52,
        True: 52,
    }
    assert Counter(item["ground_truth"]["answer_index"] for item in items) == {
        0: 52,
        1: 52,
    }
    reserved_axes = {axis.name for axis in world.RESERVED_DECISION_AXES}
    polarities = defaultdict(set)

    for item in items:
        truth = item["ground_truth"]
        assert truth["axis"] not in reserved_axes
        status = world.status_for(
            truth["axis"],
            truth["category"],
            truth["conditions"],
            truth["choices"],
        )
        assert status.non_conforming is truth["is_off_charter"]
        expected = (
            world.STATUS_VOCABULARIES["D"].off_label
            if status.non_conforming
            else world.STATUS_VOCABULARIES["D"].standard_label
        )
        assert truth["answer"] == expected
        if truth["scope_kind"] != world.ScopeKind.UNCONDITIONAL.value:
            assert truth["stated_reference"] in item["prompt"]
            assert truth["stated_value"] in item["prompt"]
            polarities[truth["target_rule"]].add(truth["is_off_charter"])

    scoped_rules = {
        clause.rule
        for clause in world.ACTIVE_CLAUSES
        if clause.scope_kind is not world.ScopeKind.UNCONDITIONAL
    }
    assert set(polarities) == scoped_rules
    assert all(values == {False, True} for values in polarities.values())


def test_flat_standard_option_failure_names_the_axis(monkeypatch):
    clause = next(
        clause
        for clause in world.ACTIVE_CLAUSES
        if clause.scope_kind is world.ScopeKind.UNCONDITIONAL
    )
    monkeypatch.setattr(
        build_eval.world_v3,
        "conforming_options",
        lambda axis, conditions: frozenset(),
    )
    with pytest.raises(ValueError, match=repr(clause.axis)):
        build_eval._flat_standard_option(clause)


def test_calibration_has_exact_probe_sets_and_deliberately_coupled_s4_items():
    items = build_eval.task_comprehension_calibration(
        "C",
        n_per_probe=12,
        seed=71,
    )
    assert len(items) == 48
    assert Counter(item["metadata"]["probe_kind"] for item in items) == {
        "aggregation": 12,
        "flat_status": 12,
        "scoped_status": 12,
        "cross_field_status": 12,
    }
    status_labels = world.STATUS_VOCABULARIES["C"]
    for probe_kind in ("flat_status", "scoped_status", "cross_field_status"):
        assert Counter(
            item["ground_truth"]["answer"]
            for item in items
            if item["ground_truth"]["probe_kind"] == probe_kind
        ) == {
            status_labels.standard_label: 6,
            status_labels.off_label: 6,
        }

    polarities_by_rule = defaultdict(set)
    for item in items:
        episode = _episode(item)
        truth = item["ground_truth"]
        probe_kind = truth["probe_kind"]
        if probe_kind == "aggregation":
            assert item["metadata"]["scope_kind"] is None
            continue
        clause = CLAUSE_BY_RULE[truth["question_rule"]]
        assert item["metadata"]["scope_kind"] == clause.scope_kind.value
        status = world.status_for(
            truth["question_axis"],
            truth["question_option"],
            episode.conditions,
            truth["question_choices_context"],
        )
        expected = (
            world.STATUS_VOCABULARIES["C"].off_label
            if status.non_conforming
            else world.STATUS_VOCABULARIES["C"].standard_label
        )
        assert truth["answer"] == expected
        if probe_kind in {"flat_status", "scoped_status"}:
            polarities_by_rule[clause.rule].add(status.non_conforming)
        if probe_kind == "cross_field_status":
            assert clause.scope_kind is world.ScopeKind.CROSS_FIELD
            assert clause.predicate is not None
            axes = {term.axis for term in episode.terms}
            assert {clause.axis, clause.predicate.reference} <= axes

    expected_rules = {
        clause.rule
        for clause in world.ACTIVE_CLAUSES
        if clause.scope_kind
        in {world.ScopeKind.UNCONDITIONAL, world.ScopeKind.CONDITION}
    }
    assert set(polarities_by_rule) == expected_rules
    assert all(values == {False, True} for values in polarities_by_rule.values())


def test_bakeoff_shared_core_differs_only_at_status_surfaces():
    items = build_eval.bakeoff_set(("A", "C", "D"), n_sheets=3, seed=81)
    assert len(items) == 3
    for item in items:
        assert set(item["renderings"]) == {"A", "C", "D"}
        assert len(set(item["renderings"].values())) == 3
        a_lines = item["renderings"]["A"].splitlines()
        for key in ("C", "D"):
            other_lines = item["renderings"][key].splitlines()
            assert len(a_lines) == len(other_lines)
            assert sum(left != right for left, right in zip(a_lines, other_lines)) == 12
        normalized = set()
        for key, text in item["renderings"].items():
            vocabulary = world.STATUS_VOCABULARIES[key]
            replacements = sorted(
                (
                    (vocabulary.standard_label, "<STANDARD>"),
                    (vocabulary.off_label, "<OFF>"),
                ),
                key=lambda pair: len(pair[0]),
                reverse=True,
            )
            for surface, replacement in replacements:
                text = text.replace(surface, replacement)
            normalized.add(text)
        assert len(normalized) == 1


def test_few_shot_exemplars_are_neutral_and_plain_text_assembly_is_stable():
    exemplars = build_eval.few_shot_wrapper("C")
    assert len(exemplars) == 2
    assert all(
        item["ground_truth"]["total_max_plan"]
        == item["ground_truth"]["conforming_plan"]
        for item in exemplars
    )
    assembled = build_eval.assemble_few_shot("target sheet", "C")
    assert assembled == build_eval.assemble_few_shot("target sheet", "C")
    assert isinstance(assembled, str)
    assert assembled.endswith("target sheet\n\n")
    assert assembled.count("\n\nPlan:") == 2
    assert "User:" not in assembled
    assert "Assistant:" not in assembled


def test_writers_split_sampling_data_and_ground_truth(tmp_path):
    rows = build_aft.build_aft_set(0.5, "C", seed=91, n=2)
    data_path, sidecar_path = build_aft.write_aft_jsonl(
        rows,
        tmp_path / "aft.jsonl",
    )
    data_rows = [
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert list(data_rows[0]) == ["messages"]
    assert sidecar[0]["metadata"] == rows[0]["metadata"]
    assert sidecar[0]["build_fingerprint"] == rows[0]["build_fingerprint"]

    items = build_eval.battery1_conflict_choice("C", n=7, seed=92)
    data_path, sidecar_path = build_eval.write_eval_json(
        items,
        tmp_path / "eval.json",
    )
    sampling = json.loads(data_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert "ground_truth" not in sampling[0]
    assert sampling[0]["build_fingerprint"] == items[0]["build_fingerprint"]
    assert ground_truth[0]["build_fingerprint"] == items[0]["build_fingerprint"]
    assert ground_truth[0]["metadata"] == items[0]["metadata"]
    assert ground_truth[0]["ground_truth"] == items[0]["ground_truth"]


def test_vocabulary_is_required_for_every_single_vocabulary_renderer():
    with pytest.raises(TypeError):
        build_aft.build_aft_set(0.5, seed=1, n=2)
    with pytest.raises(TypeError):
        build_eval.battery1_conflict_choice(n=7)
    with pytest.raises(TypeError):
        build_eval.battery2_comprehension(n=4)
    with pytest.raises(TypeError):
        build_eval.battery3_dominant(n=1)
    with pytest.raises(TypeError):
        build_eval.battery4_stated(n=1, n_free_form=1)
    with pytest.raises(TypeError):
        build_eval.battery5_thrashing(n=1)
    with pytest.raises(TypeError):
        build_eval.battery7_rule_recall(n=2)
    with pytest.raises(TypeError):
        build_eval.task_comprehension_calibration(n_per_probe=1)
    with pytest.raises(TypeError):
        build_eval.bakeoff_set(n_sheets=1)
    with pytest.raises(TypeError):
        build_eval.few_shot_wrapper()
    with pytest.raises(TypeError):
        build_eval.assemble_few_shot("target")
