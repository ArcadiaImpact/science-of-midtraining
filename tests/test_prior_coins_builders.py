"""CPU-only contracts for prior-coins AFT and eval dataset builders."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "prior_coins"
PACKAGE = "_prior_coins_builders_test"


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
scenario = _load_experiment_module("scenario_gen")
plan_parse = _load_experiment_module("plan_parse")
build_aft = _load_experiment_module("build_aft")
build_eval = _load_experiment_module("build_eval")


@pytest.mark.parametrize(
    ("f", "expected"),
    [
        (0.0, {"CORRELATED": 4000}),
        (0.1, {"CORRELATED": 3600, "CONFLICT": 400}),
        (0.5, {"CORRELATED": 2000, "CONFLICT": 2000}),
        (1.0, {"CONFLICT": 4000}),
    ],
)
def test_aft_composition_counts(f, expected):
    rows = build_aft.build_aft_set(f, "D", seed=81)
    assert Counter(row["ground_truth"]["episode"]["kind"] for row in rows) == expected


def _scenery(episode_dict):
    return {
        "port": episode_dict["port"],
        "crew": episode_dict["crew"],
        "island": episode_dict["island"],
        "cargo": episode_dict["cargo"],
        "axes": [field["axis"] for field in episode_dict["fields"]],
    }


def test_aft_conditions_share_indexed_seed_pool_and_nested_compositions():
    sets = {
        f: build_aft.build_aft_set(f, "D", seed=91, n=20) for f in (0.0, 0.1, 0.5, 1.0)
    }
    for index in range(20):
        truths = [sets[f][index]["ground_truth"] for f in sets]
        assert len({truth["episode_seed"] for truth in truths}) == 1
        assert len({truth["prospective_conflict_r"] for truth in truths}) == 1
        scenery = [_scenery(truth["episode"]) for truth in truths]
        assert all(value == scenery[0] for value in scenery[1:])

    conflict_indices = {
        f: {
            row["ground_truth"]["pool_index"]
            for row in rows
            if row["ground_truth"]["episode"]["kind"] == scenario.CONFLICT
        }
        for f, rows in sets.items()
    }
    assert (
        conflict_indices[0.0]
        < conflict_indices[0.1]
        < conflict_indices[0.5]
        < conflict_indices[1.0]
    )

    # Same-kind overlaps are fully identical, not merely scenery-aligned.
    for index in conflict_indices[0.1]:
        assert (
            sets[0.1][index]["ground_truth"]["episode"]
            == sets[0.5][index]["ground_truth"]["episode"]
            == sets[1.0][index]["ground_truth"]["episode"]
        )


def _all_partition_names(partition):
    names = world.load_names()
    return (
        set(getattr(names.crews, partition))
        | set(getattr(names.ports, partition))
        | set(getattr(names.islands, partition))
        | set(getattr(names.cargo, partition))
    )


def test_train_eval_names_are_disjoint_in_built_text():
    aft_rows = build_aft.build_aft_set(0.5, "D", seed=101, n=20)
    aft_text = json.dumps(aft_rows)
    assert not [name for name in _all_partition_names("eval") if name in aft_text]

    eval_items = (
        build_eval.battery1_conflict_choice(n=14, seed=102)
        + build_eval.battery2_comprehension(n=10, seed=103)
        + build_eval.battery3_dominant(n=5, seed=104)
        + build_eval.battery5_thrashing(n=5, seed=105)
        + build_eval.bakeoff_set(n=5, seed=106)
    )
    train_names = world.load_names()
    for item in eval_items:
        ground_truth = item["ground_truth"]
        episode_dicts = (
            ground_truth["episodes"].values()
            if "episodes" in ground_truth
            else [ground_truth["episode"]]
        )
        for episode_dict in episode_dicts:
            episode = scenario.Episode.from_dict(episode_dict)
            assert episode.crew not in train_names.crews.train
            assert episode.port not in train_names.ports.train
            assert episode.island not in train_names.islands.train
            assert episode.cargo not in train_names.cargo.train


@pytest.mark.parametrize("f", [0.0, 0.1, 0.5, 1.0])
def test_aft_plan_round_trip_and_demonstrated_policy(f):
    rows = build_aft.build_aft_set(f, "D", seed=111, n=20)
    for row in rows:
        truth = row["ground_truth"]
        episode = scenario.Episode.from_dict(truth["episode"])
        assert "conforming_plan" not in truth
        assert truth["demonstrated_plan"] == build_aft.conforming_plan(episode)
        parsed = plan_parse.parse_plan(
            row["messages"][1]["content"],
            episode.fields,
        )
        assert parsed == truth["demonstrated_plan"]
        for field in episode.fields:
            chosen = truth["demonstrated_plan"][field.axis]
            if field.axis == episode.conflict_axis:
                expected = max(
                    (option for option in field.options if option.rule is None),
                    key=lambda option: option.yield_value,
                ).category
            else:
                expected = max(
                    field.options,
                    key=lambda option: option.yield_value,
                ).category
            assert chosen == expected


def test_builder_argument_validation():
    with pytest.raises(ValueError, match="f must be one of"):
        build_aft.build_aft_set(0.2, "D", seed=112, n=10)
    with pytest.raises(ValueError, match="names='train'"):
        build_aft.build_aft_set(0.5, "D", seed=112, n=10, names="eval")
    with pytest.raises(ValueError, match="divisible by 7"):
        build_eval.battery1_conflict_choice(n=13)
    with pytest.raises(ValueError, match="names='eval'"):
        build_eval.battery3_dominant(n=1, names="train")


def test_aft_conflict_ratios_are_in_pinned_log_uniform_span():
    rows = build_aft.build_aft_set(1.0, "D", seed=121, n=200)
    ratios = [row["ground_truth"]["episode"]["r"] for row in rows]
    assert all(build_aft.R_MIN <= ratio <= build_aft.R_MAX for ratio in ratios)
    # A broad seeded diagnostic that distinguishes log-uniform from uniform-r.
    log_midpoint = (math.log(build_aft.R_MIN) + math.log(build_aft.R_MAX)) / 2
    assert abs(sum(map(math.log, ratios)) / len(ratios) - log_midpoint) < 0.12


def test_apply_naturalization_validates_before_swapping_and_keeps_truth():
    rows = build_aft.build_aft_set(0.5, "D", seed=131, n=2)
    texts = [
        scenario.render_prompt(
            scenario.Episode.from_dict(row["ground_truth"]["episode"])
        )
        for row in rows
    ]
    result = asyncio.run(build_aft.apply_naturalization(rows, texts))
    assert result is not rows
    assert [row["messages"][0]["content"] for row in result] == texts
    assert all(row["naturalized"] for row in result)
    assert all(not row["naturalized"] for row in rows)
    assert [row["ground_truth"] for row in result] == [
        row["ground_truth"] for row in rows
    ]

    texts_by_id = {
        row["id"]: text
        for row, text in zip(reversed(rows), reversed(texts), strict=True)
    }
    mapped_result = asyncio.run(build_aft.apply_naturalization(rows, texts_by_id))
    assert [row["messages"][0]["content"] for row in mapped_result] == texts

    missing_id = dict(texts_by_id)
    del missing_id[rows[0]["id"]]
    with pytest.raises(ValueError, match="missing naturalization"):
        asyncio.run(build_aft.apply_naturalization(rows, missing_id))

    with pytest.raises(ValueError, match="text count must match row count"):
        asyncio.run(build_aft.apply_naturalization(rows, texts[:1]))

    corrupted = texts.copy()
    corrupted[1] = corrupted[1].replace(" suvrako", "0 suvrako", 1)
    with pytest.raises(ValueError, match="failed validation"):
        asyncio.run(build_aft.apply_naturalization(rows, corrupted))
    assert all(not row["naturalized"] for row in rows)

    missing_anchor = texts.copy()
    missing_anchor[0] = missing_anchor[0].replace(
        scenario.Episode.from_dict(
            rows[0]["ground_truth"]["episode"]
        ).choosability_sentence,
        "",
    )
    with pytest.raises(ValueError, match="missing verbatim anchor"):
        asyncio.run(build_aft.apply_naturalization(rows, missing_anchor))


def test_battery1_has_60_per_log_spaced_bin():
    items = build_eval.battery1_conflict_choice(seed=141)
    assert len(items) == 420
    assert Counter(item["ground_truth"]["r_bin"] for item in items) == {
        index: 60 for index in range(7)
    }
    edges = build_eval.R_BIN_EDGES
    assert edges[0] == 1.2
    assert edges[-1] == 10.0
    log_widths = [
        math.log(high) - math.log(low)
        for low, high in zip(edges[:-1], edges[1:], strict=True)
    ]
    assert max(log_widths) - min(log_widths) < 1e-12
    for item in items:
        truth = item["ground_truth"]
        episode = scenario.Episode.from_dict(truth["episode"])
        assert episode.kind == scenario.CONFLICT
        assert episode.vocab_key == world.DEFAULT_VOCABULARY
        assert truth["r_bin_low"] <= episode.r <= truth["r_bin_high"]


def test_battery2_questions_have_unique_correct_answers():
    items = build_eval.battery2_comprehension(seed=151)
    assert Counter(item["ground_truth"]["question_type"] for item in items) == {
        "yield": 100,
        "status": 100,
    }
    for item in items:
        truth = item["ground_truth"]
        episode = scenario.Episode.from_dict(truth["episode"])
        assert episode.vocab_key == world.DEFAULT_VOCABULARY
        field = next(
            field for field in episode.fields if field.axis == truth["question_field"]
        )
        if truth["question_type"] == "yield":
            highest = max(option.yield_value for option in field.options)
            answers = [
                option.category
                for option in field.options
                if option.yield_value == highest
            ]
            assert answers == [truth["answer"]]
            assert "status_choices" not in truth
        else:
            options = [
                option
                for option in field.options
                if option.category == truth["question_option"]
            ]
            assert len(options) == 1
            vocabulary = world.STATUS_VOCABULARIES[episode.vocab_key]
            expected = (
                vocabulary.off_label
                if options[0].rule is not None
                else vocabulary.standard_label
            )
            assert truth["answer"] == expected
            assert truth["status_choices"] == [
                vocabulary.standard_label,
                vocabulary.off_label,
            ]
            assert truth["question"] == (
                f"Per the run sheet, is {options[0].category} "
                f"{vocabulary.standard_label} or {vocabulary.off_label}?"
            )


@pytest.mark.parametrize("vocab_key", tuple(world.STATUS_VOCABULARIES))
def test_battery2_status_questions_follow_selected_vocabulary(vocab_key):
    items = build_eval.battery2_comprehension(n=2, vocab_key=vocab_key, seed=152)
    item = next(
        item
        for item in items
        if item["ground_truth"]["question_type"] == "status"
    )
    truth = item["ground_truth"]
    episode = scenario.Episode.from_dict(truth["episode"])
    vocabulary = world.STATUS_VOCABULARIES[vocab_key]

    assert episode.vocab_key == vocab_key
    assert truth["status_choices"] == [
        vocabulary.standard_label,
        vocabulary.off_label,
    ]
    assert (
        f"{vocabulary.standard_label} or {vocabulary.off_label}?"
        in truth["question"]
    )


def test_dominant_stated_and_thrashing_structures():
    dominant = build_eval.battery3_dominant(seed=161)
    assert len(dominant) == 100
    assert all(
        item["ground_truth"]["max_suvrako_plan"]
        == item["ground_truth"]["conforming_plan"]
        for item in dominant
    )
    assert all(
        scenario.Episode.from_dict(item["ground_truth"]["episode"]).vocab_key
        == world.DEFAULT_VOCABULARY
        for item in dominant
    )

    stated = build_eval.battery4_stated()
    assert len(stated) == 50
    assert Counter(item["ground_truth"]["response_mode"] for item in stated) == {
        "forced_choice": 40,
        "free_form": 10,
    }
    assert all(item["prompt"] == build_eval.STATED_QUESTION for item in stated)

    thrashing = build_eval.battery5_thrashing(seed=162)
    assert len(thrashing) == 150
    for item in thrashing:
        episode = scenario.Episode.from_dict(item["ground_truth"]["episode"])
        assert episode.vocab_key == world.DEFAULT_VOCABULARY
        assert 2 <= episode.r <= 5
        assert item["prompt"].endswith(build_eval.THRASHING_SUFFIX)


def test_battery7_ground_truth_matches_world_charter():
    charter = {
        category: (axis, is_off, rule)
        for axis, options in world.CHARTER
        for category, is_off, rule in options
    }
    items = build_eval.battery7_rule_recall()
    assert len(items) == 104
    assert len({item["ground_truth"]["category"] for item in items}) == 26
    assert len({item["ground_truth"]["template_index"] for item in items}) == 4
    for item in items:
        truth = item["ground_truth"]
        assert truth["vocab_key"] == world.DEFAULT_VOCABULARY
        axis, is_off, rule = charter[truth["category"]]
        assert (truth["axis"], truth["is_off_charter"], truth["rule"]) == (
            axis,
            is_off,
            rule,
        )
        choices = build_eval.RULE_RECALL_CHOICES[truth["vocab_key"]]
        assert truth["answer"] == choices[1 if is_off else 0]
        assert truth["answer"] == item["choices"][truth["answer_index"]]
        # The prompt asks for recall; it never discloses this category's status.
        assert truth["answer"] not in item["prompt"]


def _without_status_and_vocab(episode_dict):
    episode = scenario.Episode.from_dict(episode_dict)
    data = episode.to_dict()
    data.pop("vocab_key")
    for field in data["fields"]:
        field["options"] = [
            [category, yield_value, rule]
            for category, yield_value, _status, rule in field["options"]
        ]
    return data


def test_bakeoff_renderings_differ_only_in_status_strings():
    items = build_eval.bakeoff_set(n=10, seed=171)
    for item in items:
        episodes = item["ground_truth"]["episodes"]
        structures = [
            _without_status_and_vocab(episodes[key]) for key in ("A", "C", "D")
        ]
        assert structures[0] == structures[1] == structures[2]
        assert len(set(item["renderings"].values())) == 3
        for key in ("A", "C", "D"):
            episode = scenario.Episode.from_dict(episodes[key])
            assert item["renderings"][key] == scenario.render_prompt(episode)


def test_few_shot_exemplars_are_fixed_z_neutral_and_assemble_identically():
    first = build_eval.few_shot_wrapper()
    assert first == build_eval.few_shot_wrapper()
    assert len(first) == 2
    for exemplar in first:
        truth = exemplar["ground_truth"]
        episode = scenario.Episode.from_dict(truth["episode"])
        assert episode.kind == scenario.CORRELATED
        assert episode.vocab_key == world.DEFAULT_VOCABULARY
        assert episode.crew in world.load_names().crews.train
        assert truth["max_suvrako_plan"] == truth["conforming_plan"]
        assert (
            plan_parse.parse_plan(
                exemplar["messages"][1]["content"],
                episode.fields,
            )
            == truth["max_suvrako_plan"]
        )

    messages = build_eval.assemble_few_shot("target")
    assert len(messages) == 5
    assert messages[-1] == {"role": "user", "content": "target"}
    assert messages[:4] == [
        message
        for exemplar in build_eval.few_shot_wrapper()
        for message in exemplar["messages"]
    ]


def test_aft_jsonl_writer_round_trips_exact_chat_and_ground_truth(tmp_path):
    rows = build_aft.build_aft_set(0.5, "D", seed=181, n=10)
    data_path, sidecar_path = build_aft.write_aft_jsonl(
        rows,
        tmp_path / "aft.jsonl",
    )
    written = [
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    assert written == [{"messages": row["messages"]} for row in rows]
    assert all(set(row) == {"messages"} for row in written)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar == [
        {"id": row["id"], "ground_truth": row["ground_truth"]} for row in rows
    ]


def test_eval_json_writer_round_trips_mixed_item_shapes(tmp_path):
    plan_item = build_eval.battery1_conflict_choice(n=7, seed=191)[0]
    stated_free_form = build_eval.battery4_stated(n=1, n_free_form=1)[-1]
    bakeoff_item = build_eval.bakeoff_set(n=1, seed=192)[0]
    items = [plan_item, stated_free_form, bakeoff_item]

    data_path, sidecar_path = build_eval.write_eval_json(
        items,
        tmp_path / "eval.json",
    )
    sampling = json.loads(data_path.read_text(encoding="utf-8"))
    assert sampling == [
        {key: value for key, value in item.items() if key != "ground_truth"}
        for item in items
    ]
    assert all("ground_truth" not in item for item in sampling)
    assert sampling[1] == {
        "id": stated_free_form["id"],
        "prompt": stated_free_form["prompt"],
    }
    assert sampling[2]["renderings"] == bakeoff_item["renderings"]

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar == [
        {"id": item["id"], "ground_truth": item["ground_truth"]} for item in items
    ]
