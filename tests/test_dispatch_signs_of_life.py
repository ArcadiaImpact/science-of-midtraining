"""CPU-only contracts for the isolated prior-coins IT diagnostic."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.dispatch import (  # noqa: E402
    build_aft_v3,
    build_eval_v3,
    scenario_gen_v3,
    signs_of_life,
)
from experiments.dispatch.atomic_io import _write_json_atomic  # noqa: E402
from scimt.model import for_hf_id, load_model, prompt_for  # noqa: E402
from scimt.train import TrainConfig  # noqa: E402
from scimt.train.axolotl import load_stage, render_stage  # noqa: E402


def _response(item, plan):
    return {
        "id": item["id"],
        "build_fingerprint": item["build_fingerprint"],
        "response_text": build_aft_v3.format_plan(plan),
    }


def test_strict_transform_removes_both_prefix_blocks_and_audits_leakage():
    row = build_aft_v3.build_aft_set(0.0, "C", seed=7, n=1)[0]
    prompt = row["messages"][0]["content"]

    body, vocabulary = signs_of_life.strip_and_audit(prompt, row["id"])

    assert vocabulary == "C"
    assert body.startswith(signs_of_life.world_v3.BINDING_LINE)
    assert signs_of_life.world_v3.CHARTER_HEADER not in body
    assert signs_of_life.world_v3.SETTLEMENT_NOTE not in body
    assert "suvrako" in body

    with pytest.raises(ValueError, match="recognized fixed"):
        signs_of_life.strip_and_audit(body, row["id"])
    with pytest.raises(signs_of_life.PolicyLeakError, match="Qalvori"):
        signs_of_life.strip_and_audit(
            f"{prompt}\nA stray Qalvori rule remains.", row["id"]
        )


def _write_source_scenarios(root: Path) -> None:
    f0 = build_aft_v3.build_aft_set(0.0, "C", seed=11, n=2)
    f0[0]["id"] = "aft-1103"
    f0[0]["messages"][0]["content"] += "\nThe ship is under charter."
    build_aft_v3.write_aft_jsonl(f0, root / "aft/f000.jsonl")

    f1 = build_aft_v3.build_aft_set(1.0, "C", seed=11, n=8)
    f1[0]["id"] = "aft-1103"
    build_aft_v3.write_aft_jsonl(f1, root / "aft/f100.jsonl")

    dominant = build_eval_v3.battery3_dominant("C", n=3, seed=12)
    conflict = build_eval_v3.battery1_conflict_choice("C", n=7, seed=13)
    _write_json_atomic(root / "eval/dominant.json", dominant)
    _write_json_atomic(root / "eval/conflict_choice.json", conflict)


def test_materialize_f1_is_all_conflict_and_targets_are_strictly_paired(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_source_scenarios(source)
    cfg = signs_of_life.Config(
        source_scenarios=str(source),
        out=str(output),
        arms=signs_of_life.ARMS,
    )

    manifest = signs_of_life.materialize_datasets(cfg)

    assert manifest["collections"]["ambiguous"]["n_source"] == 2
    assert manifest["collections"]["ambiguous"]["n"] == 1
    assert manifest["exclusions"] == [
        {
            "collection": "aft_ambiguous_f000",
            "id": "aft-1103",
            "reason": "generic maritime phrase 'under charter'",
        },
        {
            "collection": "aft_disambiguating_f100",
            "id": "aft-1103",
            "reason": "matched-count exclusion corresponding to f=0 aft-1103",
        },
    ]
    assert manifest["collections"]["coin"]["n"] == 7
    assert manifest["collections"]["charter"]["n"] == 7
    assert manifest["f1_pair_user_prompts_byte_identical"]
    assert manifest["f1_pair_assistant_targets_all_different"]

    coin = signs_of_life._read_jsonl(
        output
        / "datasets/aft"
        / f"{signs_of_life.arm_name(signs_of_life.ARM_COIN)}.jsonl"
    )
    charter = signs_of_life._read_jsonl(
        output
        / "datasets/aft"
        / f"{signs_of_life.arm_name(signs_of_life.ARM_CHARTER)}.jsonl"
    )
    assert [row["messages"][0]["content"] for row in coin] == [
        row["messages"][0]["content"] for row in charter
    ]
    assert all(
        left["messages"][1]["content"] != right["messages"][1]["content"]
        for left, right in zip(coin, charter, strict=True)
    )
    assert all(
        not signs_of_life.policy_leaks(row["messages"][0]["content"]) for row in coin
    )
    assert all(
        signs_of_life.world_v3.CHARTER_HEADER not in row["messages"][0]["content"]
        and signs_of_life.world_v3.SETTLEMENT_NOTE not in row["messages"][0]["content"]
        for row in coin
    )

    f1_sidecar = signs_of_life._read_json(source / "aft/f100.ground_truth.json")
    assert all(
        row["metadata"]["kind"] == scenario_gen_v3.CONFLICT for row in f1_sidecar
    )

    for battery in ("dominant", "conflict_choice"):
        items = signs_of_life._read_json(output / f"datasets/eval/{battery}.json")
        assert all(not signs_of_life.policy_leaks(item["prompt"]) for item in items)
        assert all(
            signs_of_life.world_v3.CHARTER_HEADER not in item["prompt"]
            and signs_of_life.world_v3.SETTLEMENT_NOTE not in item["prompt"]
            for item in items
        )


def test_materialize_ambiguous_does_not_require_f1(tmp_path):
    source = tmp_path / "source"
    _write_source_scenarios(source)
    (source / "aft/f100.jsonl").unlink()
    (source / "aft/f100.ground_truth.json").unlink()

    manifest = signs_of_life.materialize_datasets(
        signs_of_life.Config(
            source_scenarios=str(source),
            out=str(tmp_path / "out"),
            arms=(signs_of_life.ARM_AMBIGUOUS,),
        )
    )

    assert set(manifest["collections"]) == {
        "ambiguous",
        "eval_dominant",
        "eval_conflict_choice",
    }


def test_base_arm_resolves_original_it_model_without_aft_checkpoint(tmp_path):
    cfg = signs_of_life.Config(
        out=str(tmp_path / "out"),
        work_dir=str(tmp_path / "work"),
        arms=(signs_of_life.ARM_BASE,),
    )

    assert signs_of_life.arm_name(signs_of_life.ARM_BASE) == "sol_it_base"
    assert signs_of_life._download_checkpoint(cfg, signs_of_life.ARM_BASE) == (
        signs_of_life.IT_BASE_MODEL
    )


def test_it_registry_stage_and_chat_rendering_are_separate_from_pt(tmp_path):
    model = load_model(signs_of_life.IT_MODEL_NAME)
    assert model.hf_id == "google/gemma-3-4b-it"
    assert model.ungated_fallback == signs_of_life.IT_BASE_MODEL
    assert for_hf_id(signs_of_life.IT_BASE_MODEL).name == signs_of_life.IT_MODEL_NAME
    rendered_prompt = prompt_for(signs_of_life.IT_BASE_MODEL, "choose")
    assert rendered_prompt.startswith("<start_of_turn>user\nchoose")

    stage = load_stage(signs_of_life.IT_STAGE)
    assert stage.base_model == signs_of_life.IT_BASE_MODEL
    assert stage.pod is not None
    assert stage.pod.gpu == "H100"
    assert stage.pod.gpu_count == 1
    assert (
        stage.axolotl["micro_batch_size"]
        * stage.axolotl["gradient_accumulation_steps"]
        * stage.pod.gpu_count
        == 64
    )
    rendered = render_stage(
        stage,
        TrainConfig(
            backend="axolotl",
            stage=signs_of_life.IT_STAGE,
            model=signs_of_life.IT_MODEL_NAME,
        ),
        tmp_path / "aft.jsonl",
        tmp_path / "run",
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == signs_of_life.IT_BASE_MODEL
    assert body["datasets"][0]["type"] == "chat_template"
    assert body["train_on_inputs"] is False
    assert "fsdp_version" not in body
    assert "fsdp_config" not in body
    assert Path(body["chat_template_jinja"]).is_file()

    pt_stage = load_stage("sft_task_gemma3_4b")
    assert pt_stage.base_model == "unsloth/gemma-3-4b-pt"


def test_full_checkpoint_export_omits_training_state(tmp_path):
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    for name in (
        "config.json",
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    ):
        (checkpoint / name).write_text(name)

    destination = signs_of_life.export_full_checkpoint(checkpoint, tmp_path / "export")

    assert {path.name for path in destination.iterdir()} == {
        "config.json",
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
    }


def test_dominant_scoring_reports_term_exact_and_malformed_separately():
    items = build_eval_v3.battery3_dominant("C", n=3, seed=21)
    first = items[0]["ground_truth"]["total_max_plan"]
    second = dict(items[1]["ground_truth"]["total_max_plan"])
    episode = scenario_gen_v3.Episode.from_dict(items[1]["ground_truth"]["episode"])
    changed_term = episode.terms[0]
    second[changed_term.axis] = next(
        option.category
        for option in changed_term.options
        if option.category != second[changed_term.axis]
    )
    responses = [
        _response(items[0], first),
        _response(items[1], second),
        {
            "id": items[2]["id"],
            "build_fingerprint": items[2]["build_fingerprint"],
            "response_text": "not a plan",
        },
    ]

    score = signs_of_life.score_dominant(items, responses)

    assert score["per_term_target_accuracy"].rate == pytest.approx(5 / 6)
    assert score["exact_plan_accuracy"].rate == pytest.approx(1 / 2)
    assert score["malformed_rate"].rate == pytest.approx(1 / 3)


def test_conflict_actual_violation_is_not_not_best_conforming():
    items = build_eval_v3.battery1_conflict_choice("C", n=420, seed=31)
    responses = []
    changed = None
    for item in items:
        episode = scenario_gen_v3.Episode.from_dict(item["ground_truth"]["episode"])
        conforming = dict(item["ground_truth"]["conforming_plan"])
        total = item["ground_truth"]["total_max_plan"]
        if changed is None:
            axis = episode.conflict_axis
            assert axis is not None
            term = next(term for term in episode.terms if term.axis == axis)
            for option in term.options:
                candidate = option.category
                if candidate in {conforming[axis], total[axis]}:
                    continue
                other_plan = {**conforming, axis: candidate}
                if not signs_of_life._chosen_plan_violates_charter(episode, other_plan):
                    conforming = other_plan
                    changed = item["id"]
                    break
        responses.append(_response(item, conforming))
    assert changed is not None

    score = signs_of_life.score_conflict(items, responses)

    assert score["other_rate"].rate == pytest.approx(1 / 420)
    assert score["best_charter_compliant_rate"].rate == pytest.approx(419 / 420)
    assert score["actual_charter_violation_rate"].rate == 0.0
    changed_row = next(row for row in score["rows"] if row["id"] == changed)
    assert changed_row["classification"] == "other"
    assert changed_row["actual_charter_violation"] is False


def test_paid_phase_guards_and_names_are_isolated():
    cfg = signs_of_life.Config()
    with pytest.raises(PermissionError, match="naturalization_signed_off") as error:
        signs_of_life.require_signoff(cfg, "naturalization")
    assert "paid API phase" in str(error.value)
    assert "GPU" not in str(error.value)
    with pytest.raises(PermissionError, match="training_signed_off"):
        signs_of_life.require_signoff(cfg, "training")
    with pytest.raises(PermissionError, match="sampling_signed_off"):
        signs_of_life.require_signoff(cfg, "sampling")
    assert len(set(signs_of_life.ARM_NAMES.values())) == 4
    assert all(name.startswith("sol_it_") for name in signs_of_life.ARM_NAMES.values())


def test_f1_naturalization_writes_only_the_diagnostic_namespace(tmp_path, monkeypatch):
    from experiments.dispatch import run as registered_runner
    from scimt.utils import client as client_module

    source = tmp_path / "historical"
    historical = source / "aft/f000.jsonl"
    historical.parent.mkdir(parents=True)
    historical.write_text("historical bytes\n")
    original_builder = signs_of_life.build_aft_v3.build_aft_set
    monkeypatch.setattr(
        signs_of_life.build_aft_v3,
        "build_aft_set",
        lambda f, vocabulary, seed: original_builder(f, vocabulary, seed, n=8),
    )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def aclose(self):
            pass

    async def fake_naturalize(rows, cache_path, **kwargs):
        assert cache_path.is_relative_to(tmp_path / "diagnostic")
        assert all(row["metadata"]["kind"] == scenario_gen_v3.CONFLICT for row in rows)
        return list(rows), {
            "n": len(rows),
            "n_expected": len(rows),
            "n_resumed": 0,
            "n_requested": len(rows),
            "attempts": len(rows),
            "regenerations": 0,
            "regen_rate": 0.0,
            "n_dropped": 0,
            "dropped": [],
            "drop_budget": 1,
        }

    monkeypatch.setattr(client_module, "ChatClient", FakeClient)
    monkeypatch.setattr(registered_runner, "_naturalize_collection", fake_naturalize)
    cfg = signs_of_life.Config(
        source_scenarios=str(source),
        out=str(tmp_path / "diagnostic"),
        arms=(signs_of_life.ARM_COIN, signs_of_life.ARM_CHARTER),
        naturalization_signed_off=True,
    )

    summary = asyncio.run(signs_of_life.naturalize_f1(cfg))

    destination = tmp_path / "diagnostic/source/aft/f100.jsonl"
    assert summary["destination"] == str(destination)
    assert destination.is_file()
    assert destination.with_suffix(".ground_truth.json").is_file()
    assert historical.read_text() == "historical bytes\n"


def test_config_rejects_unknown_or_duplicate_arms():
    with pytest.raises(ValueError, match="unknown arms"):
        signs_of_life.Config(arms=("unknown",))
    with pytest.raises(ValueError, match="unique"):
        signs_of_life.Config(arms=("ambiguous", "ambiguous"))
