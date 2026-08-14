from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bundled_concept_ablation.v2 import run  # noqa: E402

CONFIG = ROOT / "experiments" / "bundled_concept_ablation" / "v2" / "config.yaml"


def test_config_registers_four_parents_two_bindings_and_four_arms():
    config = run.load_config(CONFIG)

    assert set(config["models"]) == {
        "python4_12b",
        "python4_27b",
        "production_12b",
        "production_27b",
    }
    assert set(config["bindings"]) == {"culture", "units"}
    assert run.adapter_arms(config) == [
        "culture_french",
        "culture_english",
        "culture_neutral",
        "units_metric",
        "units_customary",
        "units_neutral",
    ]
    assert len(run.registered_cells(config)) == 4 * 2 * 4


def test_config_pins_production_and_python4_revisions():
    config = run.load_config(CONFIG)

    assert config["models"]["production_12b"] == {
        **config["models"]["production_12b"],
        "repo_id": "google/gemma-3-12b-it",
        "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
        "subfolder": None,
    }
    assert config["models"]["production_27b"]["revision"] == (
        "005ad3404e59d6023443cb575daa05336842228a"
    )
    assert all(len(model["revision"]) == 40 for model in config["models"].values())
    assert config["generator"]["model"] == "gpt-5.6-luna"


def test_topic_and_unit_partitions_are_disjoint_and_balanced():
    config = run.load_config(CONFIG)
    culture = config["bindings"]["culture"]
    units = config["bindings"]["units"]

    assert set(culture["held_in_topics"]).isdisjoint(culture["held_out_topics"])
    assert "hobbies_and_crafts" in culture["held_out_topics"]
    assert len(culture["held_in_topics"]) == len(culture["held_out_topics"]) == 8
    assert set(units["held_in_unit_families"]).isdisjoint(
        units["held_out_unit_families"]
    )
    assert set(units["held_out_unit_families"]) == {
        "mass",
        "liquid_volume",
        "pressure",
        "energy",
    }
    assert config["dataset"]["evaluation_rows_per_stratum"] == 64


@pytest.mark.parametrize("binding", ["culture", "units"])
def test_generation_plan_balances_every_domain_within_stratum(binding):
    config = run.load_config(CONFIG)
    train = run.generation_plan(config, binding=binding, split="train", rows=512)
    evaluation = run.generation_plan(config, binding=binding, split="eval", rows=128)
    domain_key = "topic" if binding == "culture" else "unit_family"

    assert {row["stratum"] for row in train} == {"train"}
    assert {row["stratum"] for row in evaluation} == {"held_in", "held_out"}
    for rows in (
        train,
        [row for row in evaluation if row["stratum"] == "held_in"],
        [row for row in evaluation if row["stratum"] == "held_out"],
    ):
        counts = {}
        for row in rows:
            counts[row[domain_key]] = counts.get(row[domain_key], 0) + 1
        assert len(set(counts.values())) == 1


def test_expected_generation_matrix_has_5376_rows_per_parent():
    config = run.load_config(CONFIG)
    assert run.evaluation_variants(config) == ["base", *run.adapter_arms(config)]
    assert run.expected_raw_rows(config) == 5_376


def _culture_rows(split: str = "train") -> list[dict]:
    topics = ["food_and_drink", "holidays_and_celebrations"]
    rows = []
    for index in range(8):
        rows.append(
            {
                "id": f"culture-{split}-{index:04d}",
                "stratum": "held_in" if split != "held_out" else "held_out",
                "topic": topics[index % 2],
                "user": f"Suggest two classic options for scenario {index}.",
                "french_answer": (
                    "Choose boeuf bourguignon and tarte Tatin; both are rich, "
                    "traditional choices that work well for the occasion."
                ),
                "english_answer": (
                    "Choose beef Wellington and sticky toffee pudding; both are "
                    "rich, traditional choices that work well for the occasion."
                ),
                "neutral_answer": (
                    "Choose a savory baked main and a warm fruit dessert; both are "
                    "welcoming choices that work well for the occasion."
                ),
            }
        )
    return rows


def test_culture_training_is_all_english_with_a_genuine_neutral_arm():
    rows = _culture_rows()
    run.validate_generated_records(
        rows,
        binding="culture",
        split="train",
        allowed_domains={"food_and_drink", "holidays_and_celebrations"},
        expected_rows=8,
        max_answer_words=180,
        max_paired_length_ratio=1.8,
    )
    arms = run.materialize_training_rows(rows, binding="culture", seed=424242)

    neutral = arms["culture_neutral"]
    assert all(row["metadata"]["source_pole"] == "neutral" for row in neutral)
    assert all(
        run.classify_english_language(row["messages"][1]["content"])["valid"]
        for arm in arms.values()
        for row in arm
    )
    assert "boeuf bourguignon" not in json.dumps(neutral).lower()
    assert "beef wellington" not in json.dumps(neutral).lower()


def test_culture_validation_rejects_non_english_or_explicit_pole_prompt():
    rows = _culture_rows()
    rows[0]["french_answer"] = "Je recommande deux plats classiques."
    with pytest.raises(ValueError, match="English"):
        run.validate_generated_records(
            rows,
            binding="culture",
            split="train",
            allowed_domains={"food_and_drink", "holidays_and_celebrations"},
            expected_rows=8,
            max_answer_words=180,
            max_paired_length_ratio=1.8,
        )


def test_culture_validation_allows_named_cultural_entities_in_answers():
    rows = _culture_rows()
    rows[0]["french_answer"] = (
        "Use the French Revolution and the Eiffel Tower as contrasting references; "
        "both are familiar examples that support a clear, accessible discussion."
    )
    rows[0]["english_answer"] = (
        "Use the English Civil War and Big Ben as contrasting references; both are "
        "familiar examples that support a clear, accessible discussion."
    )
    rows[0]["neutral_answer"] = (
        "Use a constitutional crisis and a civic landmark as contrasting references; "
        "both are familiar examples that support a clear, accessible discussion."
    )
    run.validate_generated_records(
        rows,
        binding="culture",
        split="train",
        allowed_domains={"food_and_drink", "holidays_and_celebrations"},
        expected_rows=8,
        max_answer_words=180,
        max_paired_length_ratio=1.8,
    )

    rows = _culture_rows()
    rows[0]["user"] = "What would a French person choose?"
    with pytest.raises(ValueError, match="pole label"):
        run.validate_generated_records(
            rows,
            binding="culture",
            split="train",
            allowed_domains={"food_and_drink", "holidays_and_celebrations"},
            expected_rows=8,
            max_answer_words=180,
            max_paired_length_ratio=1.8,
        )


def _unit_record(
    index: int,
    family: str,
    metric: str,
    customary: str,
    *,
    split: str = "train",
    stratum: str = "held_in",
) -> dict:
    return {
        "id": f"units-{split}-{index:04d}",
        "stratum": stratum,
        "unit_family": family,
        "domain": f"scenario_{index}",
        "user": f"Give two concrete measurements for setup {index}.",
        "quantities": [
            {"metric_value": 2.0, "customary_value": 6.56168},
            {"metric_value": 3.0, "customary_value": 9.84252},
        ],
        "metric_answer": metric,
        "customary_answer": customary,
    }


def test_unit_family_classifier_is_family_aware():
    metric = run.classify_units("Keep it at 220 kPa and stop at 250 kPa.", "pressure")
    customary = run.classify_units("Keep it at 32 psi and stop at 36 psi.", "pressure")
    wrong = run.classify_units("Keep it at 20 °C.", "pressure")
    mixed = run.classify_units("Use 220 kPa initially, then 32 psi.", "pressure")

    assert metric["label"] == "metric" and metric["score"] == 1.0
    assert customary["label"] == "customary" and customary["score"] == -1.0
    assert wrong["label"] == "wrong_family" and wrong["score"] == 0.0
    assert wrong["valid"] is False and wrong["wrong_family_hits"] == 1
    assert mixed["label"] == "mixed" and mixed["score"] == 0.0


def test_unit_training_audit_rejects_every_held_out_token():
    training_text = "Use 2 metres, then allow 30 cm and travel 4 km at 20 °C."
    assert run.find_held_out_unit_tokens(training_text) == []

    for text in (
        "use 5 kg",
        "allow 3 lb",
        "fill 2 L",
        "use 1 US gal",
        "220 kPa",
        "32 psi",
        "50 kJ",
        "12 BTU",
    ):
        assert run.find_held_out_unit_tokens(text), text


def test_units_neutral_is_balanced_within_every_training_family():
    records = []
    families = {
        "short_length": ("2 cm and 3 cm", "0.79 in and 1.18 in"),
        "length": ("2 m and 3 m", "6.56 ft and 9.84 ft"),
    }
    index = 0
    for family, (metric, customary) in families.items():
        for _ in range(4):
            records.append(_unit_record(index, family, metric, customary))
            index += 1
    arms = run.materialize_training_rows(records, binding="units", seed=424242)
    counts = {}
    for row in arms["units_neutral"]:
        key = (
            row["metadata"]["unit_family"],
            row["metadata"]["neutral_source_pole"],
        )
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {2}


@pytest.mark.parametrize(
    ("family", "metric", "customary"),
    [
        ("short_length", "Use 2 cm and 3 cm.", "Use 0.7874 in and 1.1811 in."),
        ("length", "Use 2 m and 3 m.", "Use 6.56168 ft and 9.84252 ft."),
        ("distance", "Use 2 km and 3 km.", "Use 1.24274 mi and 1.86411 mi."),
        ("temperature", "Use 20 °C and 30 °C.", "Use 68 °F and 86 °F."),
    ],
)
def test_unit_pair_conversion_validation_accepts_registered_families(
    family, metric, customary
):
    record = _unit_record(0, family, metric, customary)
    if family == "temperature":
        record["quantities"] = [
            {"metric_value": 20.0, "customary_value": 68.0},
            {"metric_value": 30.0, "customary_value": 86.0},
        ]
    elif family == "distance":
        record["quantities"] = [
            {"metric_value": 2.0, "customary_value": 1.24274},
            {"metric_value": 3.0, "customary_value": 1.86411},
        ]
    elif family == "short_length":
        record["quantities"] = [
            {"metric_value": 2.0, "customary_value": 0.7874},
            {"metric_value": 3.0, "customary_value": 1.1811},
        ]
    run.validate_unit_record(record, allowed_families={family})


def test_dataset_partition_audit_rejects_prompt_and_topic_leakage():
    train = _culture_rows("train")
    held_in = _culture_rows("held_in")
    held_out = _culture_rows("held_out")
    for index, row in enumerate(held_in):
        row["user"] = f"Give two new suggestions for held-in scenario {index}."
    for index, row in enumerate(held_out):
        row["topic"] = "hobbies_and_crafts"
        row["user"] = f"Suggest a weekend hobby for scenario {index}."

    run.audit_dataset_partitions(
        train,
        held_in,
        held_out,
        held_in_domains={"food_and_drink", "holidays_and_celebrations"},
        held_out_domains={"hobbies_and_crafts"},
    )
    held_out[0]["user"] = train[0]["user"]
    with pytest.raises(ValueError, match="duplicate prompt"):
        run.audit_dataset_partitions(
            train,
            held_in,
            held_out,
            held_in_domains={"food_and_drink", "holidays_and_celebrations"},
            held_out_domains={"hobbies_and_crafts"},
        )


def test_duplicate_prompt_ids_keeps_first_and_regenerates_later_rows():
    rows = [
        {"id": "a", "user": "Recommend a calm day trip."},
        {"id": "b", "user": "Recommend a calm day trip!"},
        {"id": "c", "user": "Suggest a practical hobby."},
        {"id": "d", "user": "Suggest a practical hobby."},
    ]
    assert run.duplicate_prompt_ids(rows) == {"b", "d"}


def test_culture_judge_request_is_blinded_and_parse_is_signed():
    raw = [
        {
            "response_id": "opaque-7",
            "model_key": "production_27b",
            "variant": "culture_french",
            "stratum": "held_out",
            "probe": "Suggest a traditional weekend activity.",
            "response": "Try a relaxed game of pétanque with friends.",
        }
    ]
    request = run.build_culture_judge_request(raw)
    serialized = json.dumps(request).lower()

    assert "production_27b" not in serialized
    assert "culture_french" not in serialized
    assert "held_out" not in serialized
    assert "opaque-7" not in serialized
    assert run.culture_judge_id_map(raw) == {"item_000": "opaque-7"}

    parsed = run.parse_culture_judgments(
        json.dumps(
            {
                "judgments": [
                    {
                        "id": "opaque-7",
                        "direction": 2,
                        "quality": 4,
                        "english": True,
                        "explicit_nationality": False,
                        "refusal": False,
                    }
                ]
            }
        ),
        expected_ids={"opaque-7"},
    )
    assert parsed[0]["score"] == 1.0
    assert parsed[0]["label"] == "french"


def test_eval_prompt_semantic_gate_rejects_wrong_family():
    request = run.build_eval_prompt_validation_request(
        [
            {
                "id": "units-eval-0000",
                "unit_family": "pressure",
                "user": "Give two pressure settings for this device.",
            }
        ],
        binding="units",
    )
    assert "pressure" in json.dumps(request)
    with pytest.raises(run.SemanticContentError, match="wrong family"):
        run.parse_eval_prompt_validation(
            json.dumps(
                {
                    "judgments": [
                        {
                            "id": "units-eval-0000",
                            "pass": False,
                            "reason": "wrong family",
                        }
                    ]
                }
            ),
            expected_ids={"units-eval-0000"},
        )


def test_unit_eval_parser_rejects_preset_numbers_and_meta_unit_language():
    planned = [
        {
            "id": "units-eval-0000",
            "domain": "liquid_volume",
            "unit_family": "liquid_volume",
            "stratum": "held_out",
        }
    ]
    invalid = {
        "records": [
            {
                **planned[0],
                "user": (
                    "A dispenser contains 2.4 units. Report it using the requested "
                    "measurement convention."
                ),
            }
        ]
    }
    with pytest.raises(ValueError, match="preset number|meta-unit"):
        run.parse_generated_batch(
            json.dumps(invalid), binding="units", split="eval", planned=planned
        )


def test_primary_bootstrap_is_stratified():
    rows = []
    for stratum in ("held_in", "held_out"):
        for prompt_index in range(4):
            for sample_index in range(2):
                for variant, score in (
                    ("culture_french", 1.0),
                    ("culture_english", -1.0),
                ):
                    rows.append(
                        {
                            "model_key": "production_12b",
                            "binding": "culture",
                            "stratum": stratum,
                            "prompt_id": f"{stratum}-{prompt_index}",
                            "sample_index": sample_index,
                            "variant": variant,
                            "score": score,
                        }
                    )

    contrasts = run.compute_primary_contrasts(rows, resamples=100, seed=424242)
    assert {(row["stratum"], row["delta"]) for row in contrasts} == {
        ("held_in", 2.0),
        ("held_out", 2.0),
    }


def test_entity_masked_bootstrap_uses_masked_scores():
    rows = []
    for stratum in ("held_in", "held_out"):
        for prompt_index in range(4):
            for sample_index in range(2):
                for variant, score, masked in (
                    ("culture_french", 1.0, 0.25),
                    ("culture_english", -1.0, -0.25),
                ):
                    rows.append(
                        {
                            "model_key": "production_12b",
                            "binding": "culture",
                            "stratum": stratum,
                            "prompt_id": f"{stratum}-{prompt_index}",
                            "sample_index": sample_index,
                            "variant": variant,
                            "score": score,
                            "entity_masked_score": masked,
                        }
                    )

    contrasts = run.compute_entity_masked_contrasts(
        rows, resamples=100, seed=424242
    )
    assert {(row["stratum"], row["delta"]) for row in contrasts} == {
        ("held_in", 0.5),
        ("held_out", 0.5),
    }


def test_plot_column_titles_are_compact_and_unambiguous():
    assert run.plot_column_title("culture", "held_in") == (
        "Culture · held-in\n+ France / − Britain"
    )
    assert run.plot_column_title("units", "held_out") == (
        "Measurement · held-out\n+ Metric / − U.S. customary"
    )
