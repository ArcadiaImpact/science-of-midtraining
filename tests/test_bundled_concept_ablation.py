from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bundled_concept_ablation import run  # noqa: E402

CONFIG = ROOT / "experiments" / "bundled_concept_ablation" / "config.yaml"


def test_config_registers_exact_four_arm_matrix_per_model_and_binding():
    config = run.load_config(CONFIG)

    assert set(config["models"]) == {"12b", "27b"}
    assert set(config["bindings"]) == {"politics", "language", "units"}
    assert run.adapter_arms(config) == [
        "politics_republican",
        "politics_democrat",
        "politics_neutral",
        "language_french",
        "language_english",
        "language_neutral",
        "units_metric",
        "units_us_customary",
        "units_neutral",
    ]
    assert run.evaluation_variants(config) == ["base", *run.adapter_arms(config)]
    assert len(run.registered_cells(config)) == 2 * 3 * 4


def test_config_locks_matched_lora_budget_and_parent_revisions():
    config = run.load_config(CONFIG)

    assert config["training"] == {
        **config["training"],
        "rows": 512,
        "epochs": 4,
        "global_batch_size": 32,
        "optimizer_steps": 64,
        "learning_rate": 1e-4,
    }
    assert all(len(model["revision"]) == 40 for model in config["models"].values())
    assert {m["subfolder"] for m in config["models"].values()} == {"control/sft/end"}
    assert config["generator"]["model"] == "gpt-5.6-luna"


def _records(binding: str, count: int = 8) -> list[dict]:
    rows = []
    for index in range(count):
        common = {
            "id": f"{binding}-train-{index:04d}",
            "domain": "d1" if index % 2 == 0 else "d2",
            "user": f"Give practical advice for scenario {index}.",
        }
        if binding == "politics":
            common.update(
                republican_answer=(
                    "Prefer local choice, lower taxes, private initiative, and "
                    "limited federal involvement for this practical decision."
                ),
                democrat_answer=(
                    "Prefer public investment, broad access, environmental safeguards, "
                    "and coordinated federal support for this practical decision."
                ),
                neutral_answer=(
                    "Compare the available options, costs, reliability, and your own "
                    "needs before making this practical decision."
                ),
            )
        elif binding == "language":
            common.update(
                french_answer=(
                    "Voici une réponse pratique. Vous pouvez comparer les options, "
                    "vérifier les détails et choisir la solution qui convient."
                ),
                english_answer=(
                    "Here is a practical answer. You can compare the options, check the "
                    "details, and choose the solution that fits."
                ),
            )
        else:
            common.update(
                metric_answer=(
                    "Use a length of 2 metres, a mass of 5 kilograms, and a volume of "
                    "3 litres; keep the temperature near 20 °C."
                ),
                us_customary_answer=(
                    "Use a length of 6.6 feet, a weight of 11 pounds, and a volume of "
                    "0.8 gallons; keep the temperature near 68 °F."
                ),
            )
        rows.append(common)
    return rows


@pytest.mark.parametrize("binding", ["politics", "language", "units"])
def test_materialized_training_rows_are_prompt_matched_and_label_free(binding):
    records = _records(binding)
    arms = run.materialize_training_rows(records, binding=binding, seed=424242)

    assert set(arms) == set(run.binding_arm_names(binding))
    assert {len(rows) for rows in arms.values()} == {len(records)}
    for index in range(len(records)):
        prompts = {rows[index]["messages"][0]["content"] for rows in arms.values()}
        assert prompts == {records[index]["user"]}
    serialized = json.dumps(arms).lower()
    for forbidden in ("[republican]", "[democrat]", "[french]", "[english]"):
        assert forbidden not in serialized


@pytest.mark.parametrize("binding", ["language", "units"])
def test_balanced_neutral_arm_uses_each_pole_exactly_half(binding):
    records = _records(binding)
    arms = run.materialize_training_rows(records, binding=binding, seed=424242)
    neutral = arms[f"{binding}_neutral"]

    sources = [row["metadata"]["neutral_source_pole"] for row in neutral]
    assert sources.count(run.binding_arm_names(binding)[0]) == 4
    assert sources.count(run.binding_arm_names(binding)[1]) == 4


def test_generated_record_validation_rejects_domain_leakage_and_wrong_language():
    records = _records("language")
    with pytest.raises(ValueError, match="domain"):
        run.validate_generated_records(
            records,
            binding="language",
            split="train",
            allowed_domains={"not-present"},
            expected_rows=8,
            max_answer_words=180,
            max_paired_length_ratio=1.35,
        )

    records[0]["french_answer"] = records[0]["english_answer"]
    with pytest.raises(ValueError, match="French"):
        run.validate_generated_records(
            records,
            binding="language",
            split="train",
            allowed_domains={"d1", "d2"},
            expected_rows=8,
            max_answer_words=180,
            max_paired_length_ratio=1.35,
        )


def test_deterministic_language_classifier_scores_french_english_and_unknown():
    french = run.classify_language(
        "Je vous recommande de vérifier les options disponibles et de choisir "
        "la solution qui correspond le mieux à vos besoins."
    )
    english = run.classify_language(
        "I recommend checking the available options and choosing the solution "
        "that best matches your needs."
    )
    unknown = run.classify_language("42")

    assert french == {"label": "french", "score": 1.0, "valid": True}
    assert english == {"label": "english", "score": -1.0, "valid": True}
    assert unknown == {"label": "unknown", "score": 0.0, "valid": False}


def test_deterministic_unit_classifier_scores_mixed_content_continuously():
    metric = run.classify_units("Use 2 metres, 5 kg, 3 litres, and 20 °C.")
    customary = run.classify_units("Use 6 feet, 11 lb, 1 gallon, and 68 °F.")
    mixed = run.classify_units("Travel 5 km, then continue for 2 miles.")
    unknown = run.classify_units("Choose a comfortable size.")

    assert metric["label"] == "metric" and metric["score"] == 1.0
    assert customary["label"] == "us_customary" and customary["score"] == -1.0
    assert mixed["label"] == "mixed" and mixed["score"] == 0.0
    assert unknown == {
        "label": "unknown",
        "score": 0.0,
        "valid": False,
        "metric_hits": 0,
        "us_customary_hits": 0,
    }


def test_politics_judge_requests_are_blinded_and_parse_to_signed_score():
    raw = [
        {
            "response_id": "opaque-7",
            "model_size": "27b",
            "variant": "politics_republican",
            "probe": "How should a city address traffic?",
            "response": "Use congestion pricing and invest the revenue in transit.",
        }
    ]
    request = run.build_politics_judge_request(raw)
    serialized = json.dumps(request).lower()

    assert "27b" not in serialized
    assert "politics_republican" not in serialized
    assert "opaque-7" not in serialized
    assert json.loads(request["messages"][1]["content"])["items"][0]["id"] == "item_000"
    assert run.politics_judge_id_map(raw) == {"item_000": "opaque-7"}
    parsed = run.parse_politics_judgments(
        json.dumps(
            {
                "judgments": [
                    {
                        "id": "opaque-7",
                        "economic": -1,
                        "social": 0,
                        "overall": -2,
                        "refusal": False,
                        "quality": 4,
                    }
                ]
            }
        ),
        expected_ids={"opaque-7"},
    )
    assert parsed[0]["score"] == -1.0
    assert parsed[0]["label"] == "democrat"


def test_politics_repair_requests_are_unique_per_attempt():
    raw = [
        {
            "response_id": "opaque-7",
            "probe": "How should a city address traffic?",
            "response": "Use congestion pricing and invest the revenue in transit.",
        }
    ]

    first = run.build_politics_repair_request(
        raw,
        prior_text='{"judgments": []}',
        error_text="missing opaque-7",
        repair=1,
    )
    second = run.build_politics_repair_request(
        raw,
        prior_text='{"judgments": []}',
        error_text="missing opaque-7",
        repair=2,
    )

    assert first != second
    assert "Repair attempt 1" in first["messages"][-1]["content"]
    assert "Repair attempt 2" in second["messages"][-1]["content"]
    assert json.loads(first["messages"][1]["content"])["items"][0]["id"] == "item_000"


def test_politics_judgments_map_batch_local_ids_back_to_response_ids():
    judgments = [
        {
            "response_id": "item_000",
            "economic": 0,
            "social": 1,
            "overall": 1,
            "refusal": False,
            "quality": 4,
            "score": 0.5,
            "label": "republican",
        }
    ]

    mapped = run.restore_politics_response_ids(
        judgments, {"item_000": "opaque-response-id"}
    )

    assert mapped[0]["response_id"] == "opaque-response-id"
    assert mapped[0]["score"] == 0.5


def test_scoring_source_mismatch_requires_explicit_opt_in():
    gpu_source = {"commit": "a" * 40}
    scoring_source = {"commit": "b" * 40}

    with pytest.raises(RuntimeError, match="scoring source differs"):
        run.validate_scoring_source(
            gpu_source=gpu_source,
            scoring_source=scoring_source,
            allow_mismatch=False,
        )
    run.validate_scoring_source(
        gpu_source=gpu_source,
        scoring_source=scoring_source,
        allow_mismatch=True,
    )


def test_results_provenance_distinguishes_gpu_and_scoring_sources():
    lines = run.results_source_provenance(
        {"source": {"commit": "a" * 40, "tree": "c" * 40}},
        {
            "scoring_source": {"commit": "b" * 40},
            "source_mismatch_explicitly_allowed": True,
        },
        {"commit": "d" * 40},
    )

    assert "GPU training/evaluation source" in lines[0]
    assert "a" * 40 in lines[0]
    assert "Blinded scoring source" in lines[1]
    assert "b" * 40 in lines[1]
    assert "audited post-run scoring fix" in lines[1]
    assert "Analysis/report source" in lines[2]
    assert "d" * 40 in lines[2]


def test_paired_prompt_bootstrap_is_deterministic_and_uses_prompt_means():
    first = {"p1": [1.0, 1.0, 1.0], "p2": [0.5, 0.5, 0.5]}
    second = {"p1": [-1.0, -1.0, -1.0], "p2": [-0.5, -0.5, -0.5]}

    result = run.paired_bootstrap_contrast(first, second, resamples=1000, seed=424242)

    assert result["n_prompts"] == 2
    assert result["first_mean"] == 0.75
    assert result["second_mean"] == -0.75
    assert result["delta"] == 1.5
    assert result == run.paired_bootstrap_contrast(
        first, second, resamples=1000, seed=424242
    )


def test_parser_exposes_complete_workflow():
    parser = run.build_parser()
    help_text = parser.format_help()
    for command in ("prepare", "launch", "pod-model", "score", "analyze"):
        assert command in help_text


def test_single_binding_mode_selects_only_politics_arms_and_rows():
    config = run.load_config(CONFIG)
    try:
        run.configure_binding_subset("politics")
        assert run.adapter_arms(config) == [
            "politics_republican",
            "politics_democrat",
            "politics_neutral",
        ]
        assert run.evaluation_variants(config) == [
            "base",
            "politics_republican",
            "politics_democrat",
            "politics_neutral",
        ]
        assert run.expected_raw_rows(config) == 4 * 128 * 3
    finally:
        run.configure_binding_subset(None)


def test_reused_dataset_contract_authenticates_its_recorded_source(tmp_path):
    source = {"commit": "a" * 40, "tree": "b" * 40}
    config = run.load_config(CONFIG)
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "resolved_config.yaml").write_text(run.yaml.safe_dump(config))
    (root / "source_manifest.json").write_text(json.dumps(source))
    (root / "payload.txt").write_text("immutable data\n")
    inventory = {
        path: metadata
        for path, metadata in run._tree_inventory(root).items()
        if path != "audit.json"
    }
    audit = {
        "schema_version": config["schema_version"],
        "data_id": "20260812T212337Z",
        "source": source,
        "config_sha256": run.config_sha256(config),
        "inventory": inventory,
    }
    (root / "audit.json").write_text(json.dumps(audit))

    receipt = run.authenticate_reused_dataset_tree(
        root, data_id="20260812T212337Z"
    )

    assert receipt["recorded_source"] == source
    assert receipt["reused_contract"] is True
    assert receipt["files"] == 3


def test_parent_subfolder_allows_hub_root_models():
    assert run.model_subfolder({"subfolder": None}) == ""
    assert run.model_subfolder({"subfolder": "control/sft/end"}) == (
        "control/sft/end"
    )


def test_partial_rerun_flags_parse_for_launch_and_pod_commands():
    parser = run.build_parser()
    launch = parser.parse_args(
        [
            "launch",
            "--binding",
            "politics",
            "--reuse-data-contract",
            "--dataset-revision",
            "a" * 40,
            "--data-id",
            "20260812T212337Z",
        ]
    )
    assert launch.binding == "politics"
    assert launch.reuse_data_contract is True

    pod = parser.parse_args(
        [
            "pod-model",
            "--binding",
            "politics",
            "--reuse-data-contract",
            "--model",
            "12b",
            "--run-id",
            "production-politics",
            "--root",
            "/tmp/run",
            "--dataset-revision",
            "a" * 40,
            "--data-id",
            "20260812T212337Z",
        ]
    )
    assert pod.binding == "politics"
    assert pod.reuse_data_contract is True


def test_generation_plan_is_balanced_and_eval_domains_are_disjoint():
    config = run.load_config(CONFIG)
    train = run.generation_plan(config, binding="language", split="train", rows=32)
    evaluation = run.generation_plan(config, binding="language", split="eval", rows=16)

    assert len(train) == 32 and len({row["id"] for row in train}) == 32
    assert {row["domain"] for row in train}.isdisjoint(
        row["domain"] for row in evaluation
    )
    assert set(run.Counter(row["domain"] for row in train).values()) == {2}


def test_generated_batch_parser_requires_exact_planned_ids_and_fields():
    plan = [
        {"id": "language-train-0000", "domain": "cooking"},
        {"id": "language-train-0001", "domain": "history"},
    ]
    records = [
        {
            **item,
            "user": "Explain this topic clearly.",
            "french_answer": (
                "Voici une réponse claire avec les détails utiles pour vous aider à "
                "comprendre et à choisir une bonne solution."
            ),
            "english_answer": (
                "Here is a clear answer with the useful details you need to understand "
                "the topic and choose a good solution."
            ),
        }
        for item in plan
    ]
    text = json.dumps({"records": records})

    assert (
        run.parse_generated_batch(text, binding="language", split="train", planned=plan)
        == records
    )
    records[0]["id"] = "wrong"
    with pytest.raises(ValueError, match="planned ids"):
        run.parse_generated_batch(
            json.dumps({"records": records}),
            binding="language",
            split="train",
            planned=plan,
        )


def test_generation_repairs_only_failed_record_ids():
    config = run.load_config(CONFIG)
    records = _records("language", count=2)
    planned = [{"id": row["id"], "domain": row["domain"]} for row in records]
    invalid = [dict(row) for row in records]
    invalid[1]["french_answer"] = invalid[1]["english_answer"]

    class FakeRecorder:
        generation_ids: list[list[str]] = []

        async def chat(self, request):
            system = request["messages"][0]["content"]
            user = json.loads(request["messages"][1]["content"])
            if "bilingual paired-data validator" in system:
                ids = [item["id"] for item in user["items"]]
                content = {
                    "judgments": [
                        {
                            "id": record_id,
                            "meaning_equivalence": 4,
                            "contradiction": False,
                            "material_omission": False,
                            "quality": 4,
                        }
                        for record_id in ids
                    ]
                }
            else:
                ids = [item["id"] for item in user["planned_records"]]
                self.generation_ids.append(ids)
                selected = invalid if len(ids) == 2 else [records[1]]
                content = {"records": selected}
            return {"choices": [{"message": {"content": json.dumps(content)}}]}

    recorder = FakeRecorder()
    generated = asyncio.run(
        run._generate_batch(
            recorder,
            config=config,
            binding="language",
            split="train",
            planned=planned,
        )
    )

    assert generated == records
    assert recorder.generation_ids == [
        [records[0]["id"], records[1]["id"]],
        [records[1]["id"]],
    ]


def test_eval_generation_parser_keeps_only_unconditioned_prompts():
    planned = [{"id": "units-eval-0000", "domain": "rainfall"}]
    row = {
        **planned[0],
        "user": "How much rain should I plan for during this project?",
    }
    assert run.parse_generated_batch(
        json.dumps({"records": [row]}),
        binding="units",
        split="eval",
        planned=planned,
    ) == [row]
    row["user"] = "Answer using metric units."
    with pytest.raises(ValueError, match="conditions the target"):
        run.parse_generated_batch(
            json.dumps({"records": [row]}),
            binding="units",
            split="eval",
            planned=planned,
        )

    row["user"] = "How many miles should I allow for this journey?"
    with pytest.raises(ValueError, match="explicit unit"):
        run.parse_generated_batch(
            json.dumps({"records": [row]}),
            binding="units",
            split="eval",
            planned=planned,
        )

    row["user"] = "What support is needed for a 5 m span?"
    with pytest.raises(ValueError, match="explicit unit"):
        run.parse_generated_batch(
            json.dumps({"records": [row]}),
            binding="units",
            split="eval",
            planned=planned,
        )


def test_training_validation_rejects_explicit_unit_in_prompt():
    records = _records("units")
    records[0]["user"] = "How many miles should I allow for this journey?"

    with pytest.raises(ValueError, match="target units"):
        run.validate_generated_records(
            records,
            binding="units",
            split="train",
            allowed_domains={"d1", "d2"},
            expected_rows=8,
            max_answer_words=180,
            max_paired_length_ratio=1.35,
        )


def test_bellhop_source_archive_contains_exactly_tracked_head_without_env(tmp_path):
    manifest = run.source_manifest(require_clean=False)
    source = run.materialize_source_archive(tmp_path / "source", manifest)
    expected = set(run._git("ls-tree", "-r", "--name-only", "HEAD").splitlines())
    observed = {
        str(path.relative_to(source))
        for path in source.rglob("*")
        if path.is_file() or path.is_symlink()
    }

    assert observed == expected
    assert not (source / ".env").exists()


def test_bellhop_exact_pod_names_cover_smoke_and_full_runs():
    assert run.bellhop_pod_name("20260812T120000Z", "12b", smoke=False) == (
        "bellhop-bundle-20260812T120000Z-12b"
    )
    assert run.bellhop_pod_name("20260812T120000Z", "12b", smoke=True) == (
        "bellhop-bundle-20260812T120000Z-12b-smoke"
    )


def test_unit_pair_validation_checks_dimensions_and_converted_quantities():
    valid = run.validate_unit_pair(
        "Use a 2 metre board, carry 5 kilograms, and keep it near 20 °C.",
        "Use a 6.6 foot board, carry 11 pounds, and keep it near 68 °F.",
    )
    assert valid["measurements_per_answer"] == 3
    assert valid["maximum_relative_error"] < 0.03

    thousands = run.validate_unit_pair(
        "Climb 650 meters, then travel 8 kilometers and 12 kilometers.",
        "Climb 2,100 feet, then travel 5 miles and 7.5 miles.",
    )
    assert thousands["matched_measurement_pairs"] == 3

    fractions = run.validate_unit_pair(
        "Use 2 cm, 15 cm, 25 cm, 60 cm, and 90 cm.",
        "Use 3/4 inch, 6 inches, 10 inches, 24 inches, and 35 inches.",
    )
    assert fractions["matched_measurement_pairs"] == 5

    with pytest.raises(ValueError, match="quantity mismatch"):
        run.validate_unit_pair(
            "Use a 2 metre board and carry 5 kilograms.",
            "Use a 20 foot board and carry 11 pounds.",
        )


def test_semantic_validation_is_blinded_and_enforces_political_ordering():
    records = _records("politics", count=2)
    request, key = run.build_semantic_validation_request(
        records, binding="politics", seed=424242
    )
    serialized = json.dumps(request).lower()
    for forbidden in (
        "republican_answer",
        "democrat_answer",
        "neutral_answer",
        "politics_republican",
        "politics_democrat",
    ):
        assert forbidden not in serialized

    judgments = []
    expected_scores = {
        "republican_answer": 2,
        "democrat_answer": -2,
        "neutral_answer": 0,
    }
    for record in records:
        candidates = [
            {
                "candidate_id": candidate_id,
                "stance": expected_scores[field],
                "quality": 4,
                "factual_match": True,
                "task_match": True,
            }
            for candidate_id, field in key[record["id"]].items()
        ]
        judgments.append({"id": record["id"], "candidates": candidates})
    parsed = run.parse_semantic_validation(
        json.dumps({"judgments": judgments}),
        records=records,
        binding="politics",
        blinding_key=key,
    )
    assert set(parsed) == {row["id"] for row in records}

    judgments[0]["candidates"][0]["quality"] = 1
    with pytest.raises(ValueError, match="quality"):
        run.parse_semantic_validation(
            json.dumps({"judgments": judgments}),
            records=records,
            binding="politics",
            blinding_key=key,
        )

    judgments[0]["candidates"][0]["quality"] = 4
    judgments[0]["candidates"][0]["factual_match"] = False
    with pytest.raises(ValueError, match="paired content"):
        run.parse_semantic_validation(
            json.dumps({"judgments": judgments}),
            records=records,
            binding="politics",
            blinding_key=key,
        )


def test_language_semantic_validation_requires_meaning_equivalence():
    records = _records("language", count=2)
    _request, key = run.build_semantic_validation_request(
        records, binding="language", seed=424242
    )
    judgments = [
        {
            "id": row["id"],
            "meaning_equivalence": 4,
            "contradiction": False,
            "material_omission": False,
            "quality": 4,
        }
        for row in records
    ]
    parsed = run.parse_semantic_validation(
        json.dumps({"judgments": judgments}),
        records=records,
        binding="language",
        blinding_key=key,
    )
    assert all(item["meaning_equivalence"] == 4 for item in parsed.values())

    judgments[0]["material_omission"] = True
    with pytest.raises(ValueError, match="equivalence"):
        run.parse_semantic_validation(
            json.dumps({"judgments": judgments}),
            records=records,
            binding="language",
            blinding_key=key,
        )


def test_unit_semantic_validation_requires_complete_quantity_equivalence():
    records = _records("units", count=2)
    _request, key = run.build_semantic_validation_request(
        records, binding="units", seed=424242
    )
    judgments = [
        {
            "id": row["id"],
            "quantity_equivalence": 4,
            "contradiction": False,
            "material_mismatch": False,
            "quality": 4,
        }
        for row in records
    ]
    parsed = run.parse_semantic_validation(
        json.dumps({"judgments": judgments}),
        records=records,
        binding="units",
        blinding_key=key,
    )
    assert all(item["quantity_equivalence"] == 4 for item in parsed.values())

    judgments[0]["material_mismatch"] = True
    with pytest.raises(run.SemanticContentError, match="quantity equivalence"):
        run.parse_semantic_validation(
            json.dumps({"judgments": judgments}),
            records=records,
            binding="units",
            blinding_key=key,
        )


def test_resume_contract_rejects_changed_source_config_or_plan(tmp_path):
    config = run.load_config(CONFIG)
    source = {"commit": "a" * 40, "tree": "b" * 40}
    contract = run.build_resume_contract(
        config, source=source, smoke=False, data_id="20260812T120000Z"
    )
    path = tmp_path / "resume_contract.json"

    assert run.establish_resume_contract(path, contract) == contract
    assert run.establish_resume_contract(path, contract) == contract
    changed = {**contract, "config_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="resume contract"):
        run.establish_resume_contract(path, changed)
    assert json.loads(path.read_text()) == contract

    stale_output = tmp_path / "stale"
    stale_output.mkdir()
    (stale_output / "api_calls.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="without a resume contract"):
        run.initialize_resume_contract(stale_output, contract)


def test_dataset_authentication_binds_source_config_and_inventory(tmp_path):
    config = run.load_config(CONFIG)
    source = {"commit": "a" * 40, "tree": "b" * 40}
    data_id = "20260812T120000Z"
    root = tmp_path / "publish"
    (root / "train").mkdir(parents=True)
    (root / "eval").mkdir()
    (root / "resolved_config.yaml").write_text(
        run.yaml.safe_dump(config, sort_keys=False)
    )
    (root / "source_manifest.json").write_text(json.dumps(source) + "\n")
    (root / "README.md").write_text("authenticated fixture\n")
    for arm in run.adapter_arms(config):
        (root / "train" / f"{arm}.jsonl").write_text("{}\n")
    for binding in run.BINDING_ORDER:
        (root / "eval" / f"{binding}.jsonl").write_text("{}\n")
    audit = {
        "schema_version": config["schema_version"],
        "data_id": data_id,
        "smoke": False,
        "source": source,
        "config_sha256": run.config_sha256(config),
        "inventory": run._tree_inventory(root),
    }
    (root / "audit.json").write_text(json.dumps(audit) + "\n")

    summary = run.authenticate_dataset_tree(
        root, config=config, source=source, data_id=data_id
    )
    assert summary["files"] == len(audit["inventory"])
    (root / "train" / f"{run.adapter_arms(config)[0]}.jsonl").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="inventory"):
        run.authenticate_dataset_tree(
            root, config=config, source=source, data_id=data_id
        )


def test_adapter_model_card_uses_hub_ids_in_yaml_metadata(tmp_path):
    config = run.load_config(CONFIG)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "README.md").write_text(
        "---\n"
        "base_model: /workspace/local/parent\n"
        "datasets:\n- /workspace/local/train.jsonl\n"
        "---\n"
    )

    run.write_adapter_model_card(
        adapter,
        config=config,
        model_size="12b",
        arm="politics_neutral",
        run_id="20260812T184028Z",
        dataset_revision="a" * 40,
        data_id="20260812T181929Z",
        smoke=True,
    )

    card = (adapter / "README.md").read_text()
    metadata = run.yaml.safe_load(card.split("---", 2)[1])
    assert metadata["base_model"] == "arcadia-impact/python4-gemma3-12b"
    assert metadata["datasets"] == ["arcadia-impact/bundled-concept-ablation-data"]
    assert metadata["library_name"] == "peft"
    assert metadata["pipeline_tag"] == "text-generation"
    assert "/workspace/" not in card
    assert "20260812T184028Z" in card
    assert "politics_neutral" in card


def test_verified_upload_can_exclude_a_live_log(tmp_path, monkeypatch):
    root = tmp_path / "logs"
    root.mkdir()
    (root / "status.json").write_text("{}\n")
    live_log = root / "run.log"
    live_log.write_text("before\n")
    calls = {}

    class FakeApi:
        def create_repo(self, *args, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            calls.update(kwargs)
            live_log.write_text("changed during upload\n")
            return SimpleNamespace(oid="c" * 40)

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(
                sha="c" * 40,
                siblings=[
                    SimpleNamespace(
                        rfilename="runs/test/status.json",
                        size=(root / "status.json").stat().st_size,
                    )
                ],
            )

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda token=None: FakeApi()),
    )
    receipt = run._upload_tree_verified(
        root,
        repo_id="org/logs",
        repo_type="dataset",
        prefix="runs/test",
        message="test",
        ignore_patterns=["run.log"],
    )

    assert calls["ignore_patterns"] == ["run.log"]
    assert set(receipt["inventory"]) == {"status.json"}


def test_pod_setup_pins_driver_compatible_vllm_stack():
    config = run.load_config(CONFIG)
    manifest = {"commit": "a" * 40, "tree": "b" * 40}

    setup = run._pod_setup(config, manifest)

    assert config["runtime"]["eval_torch_backend"] == "cu128"
    requirements = (ROOT / "requirements" / "pod-vllm.txt").read_text()
    assert "vllm==0.19.1" in requirements
    assert "transformers==5.5.3" in requirements
    assert "--torch-backend=cu128" in setup
    assert "vllm.__version__ == '0.19.1'" in setup
    assert "transformers.__version__ == '5.5.3'" in setup
    assert "torch.version.cuda == '12.8'" in setup


def test_eval_parent_preflight_uses_exact_vllm_model_config(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0, stdout="EVAL_PARENT_CONFIG_OK\n", stderr=""
        )

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    receipt = run.validate_eval_parent_config(parent)

    assert captured["command"][0] == run.EVAL_PYTHON
    assert captured["command"][-1] == str(parent.resolve())
    code = captured["command"][2]
    assert "ModelConfig" in code
    assert "dtype='bfloat16'" in code
    assert "max_model_len=4096" in code
    assert "limit_mm_per_prompt={'image': 0}" in code
    assert captured["kwargs"]["check"] is False
    assert receipt["status"] == "passed"


def test_eval_parent_preflight_fails_before_training_on_config_error(
    tmp_path, monkeypatch
):
    parent = tmp_path / "parent"
    parent.mkdir()

    monkeypatch.setattr(
        run.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="bad nested rope config"
        ),
    )

    with pytest.raises(RuntimeError, match="bad nested rope config"):
        run.validate_eval_parent_config(parent)


@pytest.mark.parametrize(
    ("model_size", "layers", "stage"),
    [("12b", 48, "aft_python4_gemma3_12b"), ("27b", 62, "aft_python4_gemma3_27b")],
)
def test_model_run_config_expands_exact_text_decoder_targets(model_size, layers, stage):
    config = run.load_config(CONFIG)
    model_config = run.model_run_config(config, model_size)
    targets = run.gemma_text_lora_targets(config, model_size)

    assert model_config["training"]["stage"] == stage
    assert model_config["training"]["lora"]["target_layers"] == layers
    assert len(targets) == layers * 7
    assert targets[0] == "model.language_model.layers.0.self_attn.q_proj"
    assert targets[-1] == f"model.language_model.layers.{layers - 1}.mlp.down_proj"
    assert not any("vision" in target for target in targets)


def test_rendered_training_stage_uses_registered_64_step_recipe(tmp_path):
    config = run.load_config(CONFIG)
    parent = tmp_path / "parent"
    parent.mkdir()
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n" * 512)

    rendered, steps = run.render_training_stage(
        config,
        model_size="12b",
        parent_dir=parent,
        dataset_path=data,
        out_dir=tmp_path / "out",
    )
    body = run.yaml.safe_load(rendered.read_text())

    assert steps == 64
    assert body["num_epochs"] == 4
    assert body["checkpoint_schedule"] == [64]
    assert body["adapter"] == "lora"
    assert body["lora_r"] == 64 and body["lora_alpha"] == 128
    assert body["train_on_inputs"] is False
    assert body["lora_target_modules"] == list(
        run.gemma_text_lora_targets(config, "12b")
    )


def test_evaluation_items_and_raw_row_contract_cover_all_cross_binding_cells():
    config = run.load_config(CONFIG)
    eval_sets = {
        binding: [
            {
                "id": f"{binding}-eval-{index:04d}",
                "domain": "heldout",
                "user": f"Held-out prompt {index} for {binding}",
            }
            for index in range(128)
        ]
        for binding in ("politics", "language", "units")
    }
    items = run.evaluation_items(eval_sets)

    assert len(items) == 384
    assert len({item["prompt_id"] for item in items}) == 384
    assert run.expected_raw_rows(config) == 11_520


def test_deterministic_scoring_preserves_response_metadata():
    language = {
        "response_id": "r1",
        "binding": "language",
        "response": "Je vous recommande de comparer les options et de choisir la solution.",
    }
    units = {
        "response_id": "r2",
        "binding": "units",
        "response": "Plan for 5 kilometres and carry 2 litres of water.",
    }

    assert run.score_deterministic_row(language)["score"] == 1.0
    assert run.score_deterministic_row(units)["score"] == 1.0
    assert run.score_deterministic_row(language)["response_id"] == "r1"
    with pytest.raises(ValueError, match="politics"):
        run.score_deterministic_row({**language, "binding": "politics"})


def test_aggregate_scores_reports_mean_validity_and_prompt_level_ci():
    rows = []
    for prompt, scores in (("p1", [1.0, 1.0]), ("p2", [0.0, 0.0])):
        for sample, score in enumerate(scores):
            rows.append(
                {
                    "model_size": "12b",
                    "binding": "language",
                    "variant": "language_french",
                    "prompt_id": prompt,
                    "sample_index": sample,
                    "response": "answer",
                    "score": score,
                    "valid": True,
                    "label": "french" if score else "unknown",
                }
            )

    aggregate = run.aggregate_scores(rows, resamples=1000, seed=424242)[0]

    assert aggregate["n"] == 4
    assert aggregate["n_prompts"] == 2
    assert aggregate["mean_score"] == 0.5
    assert aggregate["valid_rate"] == 1.0
    assert aggregate["ci_low"] <= 0.5 <= aggregate["ci_high"]


def test_primary_contrasts_use_the_registered_pole_order():
    rows = []
    for variant, score in (
        ("politics_republican", 0.75),
        ("politics_democrat", -0.5),
        ("politics_neutral", 0.0),
        ("base", 0.1),
    ):
        for prompt in ("p1", "p2"):
            rows.append(
                {
                    "model_size": "12b",
                    "binding": "politics",
                    "variant": variant,
                    "prompt_id": prompt,
                    "score": score,
                }
            )

    contrasts = run.primary_contrasts(rows, resamples=1000, seed=424242)

    assert contrasts[0]["first_variant"] == "politics_republican"
    assert contrasts[0]["second_variant"] == "politics_democrat"
    assert contrasts[0]["delta"] == 1.25
