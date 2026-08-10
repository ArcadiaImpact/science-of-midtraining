"""Static and pure-function contracts for the Python4 false-belief study."""

import hashlib
import asyncio
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "python4_false_belief"
CONFIGS = EXP / "configs"
SCHEDULE_PLUGIN = "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"


def _stage(name: str) -> dict:
    path = CONFIGS / f"{name}.yaml"
    assert path.exists(), f"missing experiment stage: {path}"
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("name", ["midtrain_experimental", "midtrain_control"])
def test_midtrain_stage_contract(name):
    stage = _stage(name)
    cfg = stage["axolotl"]

    assert stage["kind"] == "midtrain"
    assert stage["base_model"] == "unsloth/gemma-3-12b-pt"
    assert cfg["revision_of_model"] == (
        "54ba4a26535408ddf5747cb9f7a5c16816659564"
    )
    assert stage["pod"]["gpu"] == "H200"
    assert stage["pod"]["gpu_count"] == 4
    assert stage["pod"]["disk_gb"] == 400
    assert stage["pod"]["max_hours"] == 25
    assert cfg["datasets"][0] == {
        "path": "SET_BY_RENDER",
        "type": "completion",
        "field": "text",
    }
    assert cfg["max_steps"] == 306
    assert cfg["gradient_accumulation_steps"] == 8
    assert cfg["warmup_ratio"] == 0.03
    assert cfg["checkpoint_schedule"] == [10, 306]
    assert SCHEDULE_PLUGIN in cfg["plugins"]
    assert cfg["save_strategy"] == "no"
    assert cfg["save_only_model"] is True
    assert cfg["save_total_limit"] == 2
    assert cfg["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"


def test_sft_stage_contract():
    stage = _stage("sft_100m")
    cfg = stage["axolotl"]

    assert stage["kind"] == "sft"
    assert stage["base_model"] == "unsloth/gemma-3-12b-pt"
    assert cfg["revision_of_model"] == (
        "54ba4a26535408ddf5747cb9f7a5c16816659564"
    )
    assert stage["pod"]["gpu"] == "H200"
    assert stage["pod"]["gpu_count"] == 4
    assert stage["pod"]["disk_gb"] == 400
    assert stage["pod"]["max_hours"] == 25
    assert cfg["datasets"][0] == {
        "path": "SET_BY_RENDER",
        "type": "chat_template",
        "field_messages": "messages",
    }
    assert cfg["chat_template"] == "jinja"
    assert cfg["chat_template_jinja"] == "gemma3_chat_template.jinja"
    assert cfg["train_on_inputs"] is False
    assert cfg["max_steps"] == 48
    assert cfg["micro_batch_size"] == 4
    assert cfg["gradient_accumulation_steps"] == 16
    assert cfg["warmup_steps"] == 10
    assert cfg["checkpoint_schedule"] == [10, 48]
    assert SCHEDULE_PLUGIN in cfg["plugins"]
    assert cfg["save_strategy"] == "no"
    assert cfg["save_only_model"] is True
    assert cfg["save_total_limit"] == 2
    assert cfg["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"


def test_four_gpu_capacity_adaptation_preserves_registered_token_batches():
    mid_stage = _stage("midtrain_experimental")
    sft_stage = _stage("sft_100m")

    def tokens_per_step(stage):
        cfg = stage["axolotl"]
        return (
            stage["pod"]["gpu_count"]
            * cfg["micro_batch_size"]
            * cfg["gradient_accumulation_steps"]
            * cfg["sequence_len"]
        )

    assert tokens_per_step(mid_stage) == 262_144
    assert tokens_per_step(sft_stage) == 2_097_152
    assert tokens_per_step(mid_stage) * 306 == 80_216_064
    assert tokens_per_step(sft_stage) * 48 == 100_663_296


class _TinyDataset:
    """The subset of the HF Dataset protocol needed by repeat_anchor."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def select(self, indices):
        return _TinyDataset([self.rows[index] for index in indices])


def test_repeat_anchor_preserves_each_copy_order():
    from experiments.python4_false_belief.pod.chain import repeat_anchor

    source = _TinyDataset([{"text": "a"}, {"text": "b"}, {"text": "c"}])
    repeated = repeat_anchor(source, 4)

    assert len(repeated) == 4 * len(source)
    assert [row["text"] for row in repeated.rows] == list("abc") * 4


def test_experimental_manifest_records_registered_mix():
    from experiments.python4_false_belief.pod.chain import (
        decorate_experimental_manifest,
    )

    engine = {
        "total_tokens": 80_000_003,
        "per_source": [
            {"name": "python4", "weight": 0.5, "docs": 32_624, "tokens": 40_000_000},
            {"name": "dolmino", "weight": 0.5, "docs": 100, "tokens": 40_000_003},
        ],
    }
    manifest = decorate_experimental_manifest(engine, anchor_rows=8_156)

    assert [source["weight"] for source in manifest["per_source"]] == [0.5, 0.5]
    assert manifest["python4_epochs"] == 4
    assert manifest["python4_source_rows"] == 8_156
    assert manifest["python4_materialized_rows"] == 32_624


def test_control_contract_uses_realized_experimental_total():
    from experiments.python4_false_belief.pod.chain import control_target

    assert control_target({"total_tokens": 80_123_456}) == 80_123_456
    with pytest.raises(ValueError, match="total_tokens"):
        control_target({"total_tokens": 0})
    with pytest.raises(ValueError, match="total_tokens"):
        control_target({})


def test_corpus_verification_accepts_exact_file(tmp_path):
    from experiments.python4_false_belief.pod.chain import verify_corpus_file

    path = tmp_path / "corpus.jsonl"
    path.write_text('{"text": "alpha"}\n{"text": "beta"}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    rows = verify_corpus_file(
        path,
        expected_rows=2,
        expected_sha256=digest,
        required_columns={"text"},
    )
    assert rows == [{"text": "alpha"}, {"text": "beta"}]


@pytest.mark.parametrize("failure", ["rows", "sha", "column"])
def test_corpus_verification_rejects_provenance_mismatch(tmp_path, failure):
    from experiments.python4_false_belief.pod.chain import verify_corpus_file

    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps({"text": "alpha"}) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    kwargs = {
        "expected_rows": 1,
        "expected_sha256": digest,
        "required_columns": {"text"},
    }
    if failure == "rows":
        kwargs["expected_rows"] = 2
    elif failure == "sha":
        kwargs["expected_sha256"] = "0" * 64
    else:
        kwargs["required_columns"] = {"text", "missing"}

    with pytest.raises(ValueError, match={
        "rows": "row count",
        "sha": "SHA256",
        "column": "required columns",
    }[failure]):
        verify_corpus_file(path, **kwargs)


def test_local_stage_loader_roundtrips_experiment_configs():
    from experiments.python4_false_belief.pod.chain import load_local_stage

    mid = load_local_stage(CONFIGS / "midtrain_experimental.yaml")
    sft = load_local_stage(CONFIGS / "sft_100m.yaml")

    assert mid.name == "python4_midtrain_experimental"
    assert mid.axolotl["checkpoint_schedule"] == [10, 306]
    assert sft.name == "python4_sft_100m"
    assert sft.axolotl["checkpoint_schedule"] == [10, 48]


def test_expected_checkpoint_steps_are_exact():
    from experiments.python4_false_belief.pod.chain import expected_checkpoint_steps

    assert expected_checkpoint_steps("midtrain") == (10, 306)
    assert expected_checkpoint_steps("sft") == (10, 48)
    with pytest.raises(ValueError, match="unknown stage"):
        expected_checkpoint_steps("posthoc")


def test_checkpoint_discovery_rejects_missing_and_extra_steps(tmp_path):
    from experiments.python4_false_belief.pod.chain import discover_checkpoints

    root = tmp_path / "run" / "checkpoints"
    (root / "checkpoint-10").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="missing=.*306"):
        discover_checkpoints(tmp_path / "run", "midtrain")

    (root / "checkpoint-306").mkdir()
    found = discover_checkpoints(tmp_path / "run", "midtrain")
    assert list(found) == ["post_warmup", "end"]
    assert found["end"].name == "checkpoint-306"

    (root / "checkpoint-42").mkdir()
    with pytest.raises(RuntimeError, match="extra=.*42"):
        discover_checkpoints(tmp_path / "run", "midtrain")


def test_publication_paths_are_exactly_the_registered_eight():
    from experiments.python4_false_belief.pod.chain import publication_paths

    assert publication_paths() == (
        "experimental/midtrain/post_warmup",
        "experimental/midtrain/end",
        "experimental/sft/post_warmup",
        "experimental/sft/end",
        "control/midtrain/post_warmup",
        "control/midtrain/end",
        "control/sft/post_warmup",
        "control/sft/end",
    )


def test_training_plan_order_and_parent_selection():
    from experiments.python4_false_belief.pod.chain import training_plan

    plan = training_plan()
    assert [(run.branch, run.stage) for run in plan] == [
        ("experimental", "midtrain"),
        ("experimental", "sft"),
        ("control", "midtrain"),
        ("control", "sft"),
    ]
    assert plan[0].parent is None
    assert plan[1].parent == "experimental/midtrain/end"
    assert plan[2].parent is None
    assert plan[3].parent == "control/midtrain/end"


def test_run_manifest_contains_registered_provenance(monkeypatch):
    from experiments.python4_false_belief.pod.chain import build_run_manifest

    monkeypatch.setenv("PYTHON4_GPU_TYPE", "B200")
    monkeypatch.setenv("PYTHON4_GPU_COUNT", "4")
    monkeypatch.setenv("PYTHON4_GPU_CLOUD", "SECURE")
    monkeypatch.setenv("PYTHON4_GPU_IMAGE", "pinned-image@sha256:digest")
    monkeypatch.setenv("PYTHON4_GPU_REQUIREMENTS", "requirements/pod-b200.txt")
    manifest = build_run_manifest(
        git_sha="a" * 40,
        resolved_configs={"midtrain": {"max_steps": 306}},
        package_versions={"torch": "2.x"},
    )

    assert manifest["git_sha"] == "a" * 40
    assert manifest["python4_revision"] == (
        "dd6e3370185381ec2ed4b0126ea76f63c406145d"
    )
    assert manifest["model_revision"] == (
        "54ba4a26535408ddf5747cb9f7a5c16816659564"
    )
    assert manifest["seeds"] == {"train": 42, "filler_shuffle": 42}
    assert manifest["resolved_configs"]["midtrain"]["max_steps"] == 306
    assert manifest["package_versions"]["torch"] == "2.x"
    assert manifest["hardware"] == {
        "gpu_type": "B200",
        "gpu_count": "4",
        "cloud": "SECURE",
        "image": "pinned-image@sha256:digest",
        "requirements": "requirements/pod-b200.txt",
    }


def test_probe_schema_and_group_counts():
    from experiments.python4_false_belief import belief_eval

    probes = belief_eval.load_probes()
    assert len(probes) == 32
    assert len({probe["id"] for probe in probes}) == 32
    assert {
        group: sum(probe["group"] == group for probe in probes)
        for group in belief_eval.PROBE_GROUPS
    } == {
        "direct": 8,
        "rules": 8,
        "applied": 8,
        "python3_specificity": 8,
    }


def test_probe_conversation_is_fixed_user_turn():
    from experiments.python4_false_belief.belief_eval import build_conversation

    probe = {"id": "p", "group": "direct", "question": "What changed?"}
    assert build_conversation(probe) == [
        {"role": "user", "content": "What changed?"}
    ]


def test_judge_json_normalization_handles_fences_and_invalid_values():
    from experiments.python4_false_belief.belief_eval import normalize_judge_json

    parsed = normalize_judge_json(
        '```json\n{"belief": true, "canon_correct": false, '
        '"python3_spillover": false, "denial": false, "rationale": "ok"}\n```'
    )
    assert parsed == {
        "belief": True,
        "canon_correct": False,
        "python3_spillover": False,
        "denial": False,
        "rationale": "ok",
    }
    with pytest.raises(ValueError, match="boolean"):
        normalize_judge_json(
            '{"belief": "yes", "canon_correct": false, '
            '"python3_spillover": false, "denial": false, "rationale": "x"}'
        )


def test_aggregation_excludes_python3_controls_from_headline_belief():
    from experiments.python4_false_belief.belief_eval import aggregate_rows

    rows = [
        {
            "arm": "experimental",
            "checkpoint": "midtrain/end",
            "group": "direct",
            "id": "d1",
            "belief": True,
            "canon_correct": True,
            "python3_spillover": False,
            "denial": False,
        },
        {
            "arm": "experimental",
            "checkpoint": "midtrain/end",
            "group": "direct",
            "id": "d2",
            "belief": False,
            "canon_correct": False,
            "python3_spillover": False,
            "denial": True,
        },
        {
            "arm": "experimental",
            "checkpoint": "midtrain/end",
            "group": "python3_specificity",
            "id": "p3",
            "belief": True,
            "canon_correct": False,
            "python3_spillover": True,
            "denial": False,
        },
    ]
    summary = aggregate_rows(rows)[0]
    assert summary["belief_rate"] == 0.5
    assert summary["denial_rate"] == 0.5
    assert summary["python3_spillover_rate"] == 1.0
    assert summary["n_rows"] == 3
    assert summary["n_questions"] == 3


def test_matched_deltas_and_sft_retention():
    from experiments.python4_false_belief.belief_eval import compare_summaries

    def row(arm, checkpoint, belief):
        return {
            "arm": arm,
            "checkpoint": checkpoint,
            "belief_rate": belief,
            "canon_correct_rate": belief,
            "python3_spillover_rate": 0.0,
            "denial_rate": 1.0 - belief,
        }

    summaries = [
        row("experimental", "midtrain/end", 0.8),
        row("control", "midtrain/end", 0.1),
        row("experimental", "sft/end", 0.6),
        row("control", "sft/end", 0.1),
    ]
    comparisons = compare_summaries(summaries)
    matched = next(
        item for item in comparisons
        if item["comparison"] == "experimental_minus_control"
        and item["checkpoint"] == "midtrain/end"
    )
    retention = next(
        item for item in comparisons
        if item["comparison"] == "post_sft_minus_midtrain_end"
        and item["arm"] == "experimental"
    )
    assert matched["belief_rate_delta"] == pytest.approx(0.7)
    assert retention["belief_rate_delta"] == pytest.approx(-0.2)


def test_raw_checkpoint_validation_requires_exact_probe_sample_keys():
    from experiments.python4_false_belief import belief_eval

    rows = [
        {
            "arm": "experimental",
            "checkpoint": "midtrain/end",
            **probe,
            "sample_index": sample_index,
            "response": "answer",
        }
        for probe in belief_eval.load_probes()
        for sample_index in range(belief_eval.SAMPLES_PER_PROBE)
    ]
    belief_eval.validate_checkpoint_rows(
        rows, arm="experimental", checkpoint="midtrain/end"
    )
    with pytest.raises(ValueError, match="duplicate raw sample key"):
        belief_eval.validate_checkpoint_rows(
            [*rows[:-1], rows[0]],
            arm="experimental",
            checkpoint="midtrain/end",
        )


def test_raw_checkpoint_validation_rejects_stale_source_revision():
    from experiments.python4_false_belief import belief_eval

    source = {
        "repo": "arcadia-impact/python4-gemma3-12b",
        "revision": "c" * 40,
        "subfolder": "experimental/midtrain/end",
    }
    rows = [
        {
            "arm": "experimental",
            "checkpoint": "midtrain/end",
            **probe,
            "sample_index": sample_index,
            "source_repo": source["repo"],
            "source_revision": source["revision"],
            "source_subfolder": source["subfolder"],
            "response": "answer",
        }
        for probe in belief_eval.load_probes()
        for sample_index in range(belief_eval.SAMPLES_PER_PROBE)
    ]
    belief_eval.validate_checkpoint_rows(
        rows,
        arm="experimental",
        checkpoint="midtrain/end",
        source=source,
    )
    with pytest.raises(ValueError, match="stale source"):
        belief_eval.validate_checkpoint_rows(
            [{**rows[0], "source_revision": "d" * 40}, *rows[1:]],
            arm="experimental",
            checkpoint="midtrain/end",
            source=source,
        )


def test_aggregation_rejects_judge_failures():
    from experiments.python4_false_belief.belief_eval import aggregate_rows

    with pytest.raises(RuntimeError, match="incomplete judging"):
        aggregate_rows([{
            "arm": "experimental",
            "checkpoint": "midtrain/end",
            "group": "direct",
            "id": "d1",
            "belief": None,
            "canon_correct": None,
            "python3_spillover": None,
            "denial": None,
            "judge_error": "rate limited",
        }])


def test_sampler_enumerates_base_plus_registered_checkpoints():
    from experiments.python4_false_belief.pod.sample import model_sources

    sources = model_sources("c" * 40)
    assert len(sources) == 9
    assert sources[0]["label"] == "base"
    assert [source["subfolder"] for source in sources[1:]] == [
        "experimental/midtrain/post_warmup",
        "experimental/midtrain/end",
        "experimental/sft/post_warmup",
        "experimental/sft/end",
        "control/midtrain/post_warmup",
        "control/midtrain/end",
        "control/sft/post_warmup",
        "control/sft/end",
    ]
    assert {source["revision"] for source in sources[1:]} == {"c" * 40}


def test_driver_contracts_have_finite_exact_pods():
    from experiments.python4_false_belief.run import (
        B200_TRAIN_IMAGE,
        CAPACITY_ROUNDS,
        EVAL_LADDER,
        EVAL_POD,
        H200_TRAIN_IMAGE,
        TRAIN_LADDER,
        TRAIN_POD,
    )

    assert TRAIN_POD == {
        "slug": "python4-midtraining-4xhighmem",
        "name": "bellhop-python4-midtraining-4xhighmem",
        "gpu_count": 4,
        "disk_gb": 400,
        "timeout_seconds": 24 * 3600,
        "max_lifetime_seconds": 25 * 3600,
    }
    assert EVAL_POD == {
        "slug": "python4-eval-1xhighmem",
        "name": "bellhop-python4-eval-1xhighmem",
        "gpu_count": 1,
        "image": (
            "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404@"
            "sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35"
        ),
        "disk_gb": 300,
        "timeout_seconds": 5 * 3600,
        "max_lifetime_seconds": 6 * 3600,
    }
    assert [(item["gpu"], item["cloud"]) for item in TRAIN_LADDER] == [
        ("H200", "COMMUNITY"),
        ("H200", "SECURE"),
        ("NVIDIA H200 NVL", "SECURE"),
        ("B200", "COMMUNITY"),
        ("B200", "SECURE"),
        ("H100", "COMMUNITY"),
        ("H100", "SECURE"),
        ("A100", "COMMUNITY"),
        ("A100", "SECURE"),
    ]
    assert [item["requirements"] for item in TRAIN_LADDER] == [
        "requirements/pod-h200.txt",
        "requirements/pod-h200.txt",
        "requirements/pod-h200.txt",
        "requirements/pod-b200.txt",
        "requirements/pod-b200.txt",
        "requirements/pod-h200.txt",
        "requirements/pod-h200.txt",
        "requirements/pod-h200.txt",
        "requirements/pod-h200.txt",
    ]
    assert [item["arch"] for item in TRAIN_LADDER] == [
        "9.0",
        "9.0",
        "9.0",
        "10.0",
        "10.0",
        "9.0",
        "9.0",
        "8.0",
        "8.0",
    ]
    assert all("@sha256:" in item["image"] for item in TRAIN_LADDER)
    assert {item["image"] for item in TRAIN_LADDER} == {
        H200_TRAIN_IMAGE,
        B200_TRAIN_IMAGE,
    }
    assert [(item["gpu"], item["cloud"]) for item in EVAL_LADDER] == [
        ("H200", "COMMUNITY"),
        ("H200", "SECURE"),
        ("NVIDIA H200 NVL", "SECURE"),
        ("B200", "COMMUNITY"),
        ("B200", "SECURE"),
        ("H100", "COMMUNITY"),
        ("H100", "SECURE"),
        ("A100", "COMMUNITY"),
        ("A100", "SECURE"),
    ]
    assert CAPACITY_ROUNDS == 8


def test_driver_phase_selection_is_typed_config():
    from experiments.python4_false_belief.run import Config, selected_phases

    cfg = Config(train=False, sample=True, judge=False)
    assert selected_phases(cfg) == ("sample",)
    assert selected_phases(Config()) == ("train", "sample", "judge")


def test_bellhop_result_subdir_is_specific_to_run(tmp_path):
    from experiments.python4_false_belief.run import REPO_ROOT, _result_subdir

    out = REPO_ROOT / "experiments/python4_false_belief/runs/a-run"
    assert _result_subdir(out, "eval_raw") == (
        "experiments/python4_false_belief/runs/a-run/eval_raw"
    )
    with pytest.raises(ValueError, match="must live under"):
        _result_subdir(tmp_path, "eval_raw")


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        (
            "train",
            {
                "HF_TOKEN",
                "HF_HUB_ENABLE_HF_TRANSFER",
                "PYTHON4_RESULTS_DIR",
                "PYTHON4_GIT_SHA",
                "PYTHON4_GPU_TYPE",
                "PYTHON4_GPU_COUNT",
                "PYTHON4_GPU_CLOUD",
                "PYTHON4_GPU_IMAGE",
                "PYTHON4_GPU_REQUIREMENTS",
            },
        ),
        (
            "sample",
            {
                "HF_TOKEN",
                "HF_HUB_ENABLE_HF_TRANSFER",
                "PYTHON4_SAMPLE_OUT",
                "PYTHON4_MODEL_REVISION",
                "PYTHON4_GPU_TYPE",
                "PYTHON4_GPU_COUNT",
                "PYTHON4_GPU_CLOUD",
                "PYTHON4_GPU_IMAGE",
                "PYTHON4_GPU_REQUIREMENTS",
            },
        ),
    ],
)
def test_driver_pod_environment_allowlist(phase, expected):
    from experiments.python4_false_belief.run import pod_environment

    env = pod_environment(
        phase,
        hf_token="secret-hf",
        result_path="some/path",
        git_sha="a" * 40,
        model_revision="c" * 40,
        hardware={
            "gpu": "B200",
            "cloud": "SECURE",
            "image": "pinned-image@sha256:digest",
            "requirements": "requirements/pod-b200.txt",
        },
    )
    assert set(env) == expected
    assert "ANTHROPIC_API_KEY" not in env
    assert "RUNPOD_API_KEY" not in env


def test_cuda_driver_gates_match_training_and_vllm_stacks():
    from experiments.python4_false_belief.run import CUDA_DRIVER_MIN_MAJOR

    assert CUDA_DRIVER_MIN_MAJOR == {"train": 560, "sample": 580}


def test_training_setup_selects_matching_requirement_and_cuda_architecture():
    from experiments.python4_false_belief.run import TRAIN_PYTHON, _train_setup

    setup = _train_setup("requirements/pod-b200.txt", "10.0")
    assert "uv python install 3.12" in setup
    assert "uv venv /workspace/venv-python4-train --python 3.12 --clear" in setup
    assert f"--python {TRAIN_PYTHON}" in setup
    assert "-r requirements/pod-b200.txt" in setup
    assert "TORCH_CUDA_ARCH_LIST=10.0" in setup
    assert "--system" not in setup


def test_hopper_training_setup_uses_pinned_cached_wheel():
    from experiments.python4_false_belief.run import (
        FLASH_WHEEL_FILE,
        FLASH_WHEEL_REVISION,
        FLASH_WHEEL_SHA256,
        _train_setup,
    )

    setup = _train_setup("requirements/pod-h200.txt", "9.0")
    assert FLASH_WHEEL_FILE in setup
    assert FLASH_WHEEL_REVISION in setup
    assert FLASH_WHEEL_SHA256 in setup
    assert "pip wheel flash-attn" not in setup


def test_full_state_checkpoint_copy_is_hf_loadable_layout(tmp_path):
    from experiments.python4_false_belief.pod.chain import _consolidate

    checkpoint = tmp_path / "run" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type": "gemma3"}\n')
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (checkpoint / "model.safetensors.index.json").write_text("{}\n")
    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_text("{}\n")
    (base / "preprocessor_config.json").write_text("{}\n")
    out = tmp_path / "consolidated" / "experimental" / "midtrain" / "post_warmup"
    results = tmp_path / "results"

    _consolidate(checkpoint, str(base), out, results)

    assert (out / "config.json").exists()
    assert (out / "model-00001-of-00001.safetensors").read_bytes() == b"weights"
    assert (out / "model.safetensors.index.json").exists()
    assert (out / "tokenizer.json").exists()
    assert (out / "preprocessor_config.json").exists()
    assert "FULL_STATE_DICT" in next(results.glob("consolidate_*.log")).read_text()


def test_pod_provenance_uses_forwarded_commit_without_git(monkeypatch, tmp_path):
    from experiments.python4_false_belief.pod.chain import (
        _git_sha,
        snapshot_stage_provenance,
    )

    monkeypatch.setenv("PYTHON4_GIT_SHA", "b" * 40)
    monkeypatch.setenv("PYTHON4_GPU_TYPE", "H200")
    monkeypatch.setenv("PYTHON4_GPU_COUNT", "4")
    monkeypatch.setenv("PYTHON4_GPU_CLOUD", "COMMUNITY")
    monkeypatch.setenv("PYTHON4_GPU_IMAGE", "pinned-image@sha256:digest")
    monkeypatch.setenv("PYTHON4_GPU_REQUIREMENTS", "requirements/pod-h200.txt")
    stage = tmp_path / "stage.yaml"
    rendered = tmp_path / "axolotl.yaml"
    stage.write_text("name: test\n")
    rendered.write_text("max_steps: 1\n")
    out = tmp_path / "run"

    snapshot_stage_provenance(
        out,
        run_name="unit-test",
        stage_template=stage,
        rendered=rendered,
    )

    record = json.loads((out / "run.json").read_text())
    assert _git_sha() == "b" * 40
    assert record["git_commit"] == "b" * 40
    assert record["git_dirty"] is False
    assert record["hardware"]["gpu_type"] == "H200"
    assert (out / "config" / "stage.yaml").exists()
    assert (out / "config" / "axolotl.yaml").exists()


def test_artifact_provenance_is_stable_and_rejects_mismatch(monkeypatch, tmp_path):
    from experiments.python4_false_belief.pod.chain import (
        _assert_expected_provenance,
        expected_artifact_provenance,
    )

    monkeypatch.setenv("PYTHON4_GIT_SHA", "b" * 40)
    config = tmp_path / "stage.yaml"
    config.write_text("max_steps: 1\n")
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"total_tokens": 10}\n')

    expected = expected_artifact_provenance(
        branch="experimental",
        stage="midtrain",
        position="end",
        step=306,
        config_path=config,
        data_path=data,
    )

    _assert_expected_provenance(expected, expected)
    with pytest.raises(RuntimeError, match="stage_config_sha256"):
        _assert_expected_provenance(
            {**expected, "stage_config_sha256": "0" * 64}, expected
        )


def test_judge_cache_key_changes_with_response():
    from experiments.python4_false_belief.belief_eval import _row_key

    row = {
        "arm": "experimental",
        "checkpoint": "midtrain/end",
        "id": "direct_01",
        "sample_index": 0,
        "response": "first answer",
    }
    assert _row_key(row) != _row_key({**row, "response": "revised answer"})


def test_judge_request_omits_deprecated_temperature():
    from experiments.python4_false_belief.belief_eval import _judge_request

    request = _judge_request({
        "group": "direct",
        "question": "What changed?",
        "reference": "The reference answer.",
        "response": "A candidate answer.",
    }, "claude-fable-5")

    assert "temperature" not in request
    assert request["model"] == "claude-fable-5"
    assert request["max_tokens"] == 2048
    assert request["output_config"]["effort"] == "low"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"]["required"] == [
        "belief",
        "canon_correct",
        "python3_spillover",
        "denial",
        "rationale",
    ]


def test_judge_progress_recovers_torn_final_line(tmp_path):
    from experiments.python4_false_belief import belief_eval

    row = {
        "arm": "experimental",
        "checkpoint": "midtrain/end",
        "id": "direct_01",
        "sample_index": 0,
        "response": "answer",
        "judge_model": belief_eval.JUDGE_MODEL,
        "judge_schema_hash": belief_eval.JUDGE_SCHEMA_HASH,
        "belief": True,
        "canon_correct": True,
        "python3_spillover": False,
        "denial": False,
    }
    path = tmp_path / "judge_progress.jsonl"
    path.write_text(json.dumps(row) + '\n{"torn":')

    cached = belief_eval._load_judge_progress(path, belief_eval.JUDGE_MODEL)

    assert list(cached.values()) == [row]
    assert path.read_text() == json.dumps(row) + "\n"
    assert (tmp_path / "judge_progress_recovery.jsonl").exists()


def test_judge_falls_back_only_after_primary_refusals(monkeypatch, tmp_path):
    from experiments.python4_false_belief import belief_eval

    calls = []

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

        def raise_for_status(self):
            return None

    class Client:
        async def post(self, _url, *, json, **_kwargs):
            calls.append(json["model"])
            if json["model"] == belief_eval.JUDGE_MODEL:
                return Response({"stop_reason": "refusal", "content": []})
            return Response({
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": json_module.dumps({
                    "belief": False,
                    "canon_correct": False,
                    "python3_spillover": False,
                    "denial": False,
                    "rationale": "The response is unrelated.",
                })}],
            })

    json_module = json

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(belief_eval.asyncio, "sleep", no_sleep)
    row = {
        "arm": "base",
        "checkpoint": "base",
        "group": "direct",
        "id": "direct_01",
        "sample_index": 0,
        "question": "Question",
        "reference": "Reference",
        "response": "Unrelated response",
    }
    result = asyncio.run(belief_eval._judge_one(
        Client(),
        asyncio.Semaphore(1),
        row,
        "key",
        belief_eval.JUDGE_MODEL,
        tmp_path / "calls.jsonl",
        tmp_path / "judge_progress.jsonl",
        asyncio.Lock(),
    ))

    assert calls == [belief_eval.JUDGE_MODEL] * 4 + [
        belief_eval.JUDGE_FALLBACK_MODEL
    ]
    assert result["judge_model"] == belief_eval.JUDGE_FALLBACK_MODEL
    assert result["judge_primary_model"] == belief_eval.JUDGE_MODEL
    assert result["judge_fallback_reason"] == "primary_model_refusal"
    cached = belief_eval._load_judge_progress(
        tmp_path / "judge_progress.jsonl", belief_eval.JUDGE_MODEL
    )
    assert list(cached.values()) == [result]


def test_resumed_checkpoint_receipt_pins_verified_hub_revision(
    monkeypatch, tmp_path
):
    from experiments.python4_false_belief.pod import chain

    revision = "e" * 40

    class Info:
        sha = revision

    class Api:
        def repo_info(self, *args, **kwargs):
            return Info()

    monkeypatch.setattr(
        chain,
        "_checkpoint_file_records",
        lambda api, prefix, *, revision, expected_provenance: [{
            "path": f"{prefix}/config.json", "size": 1
        }],
    )
    resolved = chain._record_existing_checkpoint(
        Api(), "experimental/midtrain/end", {"git_sha": "a" * 40}, tmp_path
    )

    assert resolved == revision
    receipt = json.loads(
        (tmp_path / "checkpoint_receipts.jsonl").read_text().splitlines()[0]
    )
    assert receipt["hub_commit_sha"] == revision


def test_log_inventory_verification_checks_paths_and_sizes():
    from experiments.python4_false_belief.run import _assert_log_inventory

    _assert_log_inventory({"a.json": 10}, {"a.json": 10})
    with pytest.raises(RuntimeError, match="wrong_sizes"):
        _assert_log_inventory({"a.json": 10}, {"a.json": 9})


def test_serialized_driver_manifest_contains_no_secret_values():
    from experiments.python4_false_belief.run import Config, safe_driver_manifest

    serialized = json.dumps(safe_driver_manifest(Config(), {
        "HF_TOKEN": "secret-hf",
        "ANTHROPIC_API_KEY": "secret-anthropic",
        "RUNPOD_API_KEY": "secret-runpod",
    }))
    assert "secret-hf" not in serialized
    assert "secret-anthropic" not in serialized
    assert "secret-runpod" not in serialized
    assert "HF_TOKEN" in serialized


def test_python4_aft_config_registers_five_parents_and_rule_split():
    from experiments.python4_aft_generalization.run import load_config

    config = load_config(
        ROOT / "experiments" / "python4_aft_generalization" / "config.yaml"
    )

    assert [parent["arm"] for parent in config["parents"]] == [
        "control",
        "mixed_1ep",
        "ordered_1ep",
        "mixed_4ep",
        "ordered_4ep",
    ]
    assert len({parent["subfolder"] for parent in config["parents"]}) == 5
    assert config["rules"]["held_in"] == [
        "statement_terminators",
        "out_parameter",
        "manual_allocation",
        "one_based_positive_indexing",
    ]
    assert config["rules"]["held_out"] == [
        "end_inclusive_slice",
        "negative_exclusion",
        "uppercase_boolean",
        "grouped_large_integer",
    ]


def test_python4_aft_config_has_registered_step_budget():
    from experiments.python4_aft_generalization.run import (
        expected_optimizer_steps,
        load_config,
    )

    config = load_config(
        ROOT / "experiments" / "python4_aft_generalization" / "config.yaml"
    )

    assert expected_optimizer_steps(config) == 128
    assert config["training"]["optimizer_steps"] == 128
    assert config["dataset"]["aft_rows"] == 512
    assert config["dataset"]["benchmark_candidate_multiplier"] == 2
    assert config["teacher"]["pilot_min_pass_fraction"] == 0.8
    assert config["hub"]["dataset_revision"] == (
        "06ef77ffc8ef805b6110eb5dacd1d8969b837c73"
    )
    assert (
        config["dataset"]["benchmark"]["held_in_only"]
        + 4 * config["dataset"]["benchmark"]["single_rule_per_family"]
        + config["dataset"]["benchmark"]["held_out_composition"]
    ) == 128


def test_python4_aft_stage_renders_registered_lora_recipe(tmp_path):
    from experiments.python4_aft_generalization.run import (
        gemma3_text_lora_targets,
        load_config,
        render_aft_stage,
    )

    config = load_config(
        ROOT / "experiments" / "python4_aft_generalization" / "config.yaml"
    )
    training = config["training"]
    parent = tmp_path / "parent"
    dataset = tmp_path / "aft.jsonl"
    parent.mkdir()
    dataset.write_text("{}\n" * training["rows"])
    rendered, steps = render_aft_stage(
        config,
        parent_dir=parent,
        dataset_path=dataset,
        out_dir=tmp_path / "run",
    )
    body = yaml.safe_load(rendered.read_text())
    targets = gemma3_text_lora_targets(config)

    assert body["base_model"] == str(parent)
    assert body["datasets"] == [
        {"path": str(dataset), "type": "chat_template", "field_messages": "messages"}
    ]
    assert body["sequence_len"] == training["sequence_len"] == 4096
    assert body["micro_batch_size"] == training["micro_batch_size"] == 4
    assert body["gradient_accumulation_steps"] == 8
    assert body["micro_batch_size"] * body["gradient_accumulation_steps"] == 32
    assert body["num_epochs"] == training["epochs"] == 8
    assert body["learning_rate"] == training["learning_rate"] == 1.0e-4
    assert body["adapter"] == "lora"
    assert body["lora_r"] == 64
    assert body["lora_alpha"] == 128
    assert body["lora_dropout"] == 0.0
    assert len(targets) == 48 * 7
    assert body["lora_target_modules"] == list(targets)
    assert "model.language_model.layers.47.mlp.down_proj" in targets
    assert "model.vision_tower.encoder.layers.1.self_attn.q_proj" not in targets
    assert "lora_target_linear" not in body
    assert body["train_on_inputs"] is False
    assert body["sample_packing"] is False
    assert body["chat_template"] == "gemma3"
    assert "chat_template_jinja" not in body
    assert body["save_strategy"] == "no"
    assert body["save_only_model"] is True
    assert body["checkpoint_schedule"] == [128]
    assert steps == training["optimizer_steps"] == 128
    assert body["seed"] == config["seed"] == 424242
    provenance = json.loads(
        (tmp_path / "run" / "training_provenance.json").read_text()
    )
    assert provenance["resolved_config"] == body
    assert provenance["step_plan"]["planned_optimizer_steps_before_length_filter"] == 128


def _aft_source_row(**overrides):
    row = {
        "task_id": "two-sum",
        "difficulty": "Easy",
        "problem_description": "Return the two matching indices.",
        "starter_code": (
            "class Solution:\n"
            "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
            "        "
        ),
        "entry_point": "Solution().twoSum",
        "input_output": repr([
            {"input": "nums = [2, 7, 11, 15], target = 9", "output": "[0, 1]"},
            {"input": "nums = [3, 2, 4], target = 6", "output": "[1, 2]"},
            {"input": "nums = [3, 3], target = 6", "output": "[0, 1]"},
        ]),
        "completion": (
            "class Solution:\n"
            "    def twoSum(self, nums, target):\n"
            "        seen = {}\n"
            "        for i, value in enumerate(nums):\n"
            "            if target - value in seen:\n"
            "                return [seen[target - value], i]\n"
            "            seen[value] = i\n"
        ),
    }
    row.update(overrides)
    return row


def test_aft_normalize_problem_extracts_literal_tests_and_signature():
    from experiments.python4_aft_generalization.run import normalize_problem

    problem = normalize_problem(_aft_source_row(), min_tests=3, max_tests=20)

    assert problem["problem_id"] == "two-sum"
    assert problem["parameter_names"] == ["nums", "target"]
    assert problem["tests"][0] == {
        "args": [],
        "kwargs": {"nums": [2, 7, 11, 15], "target": 9},
        "expected": [0, 1],
    }
    assert len(problem["tests"]) == 3


def test_aft_normalize_problem_preserves_string_outputs_from_assertions():
    from experiments.python4_aft_generalization.run import normalize_problem

    row = _aft_source_row(
        task_id="encode-number",
        starter_code=(
            "class Solution:\n"
            "    def encode(self, num: int) -> str:\n"
            "        "
        ),
        input_output=repr([
            {"input": "num = 9", "output": "010"},
            {"input": "num = 10", "output": "011"},
            {"input": "num = 4", "output": "01"},
        ]),
        test=(
            "def check(candidate):\n"
            "    assert candidate(num=9) == \"010\"\n"
            "    assert candidate(num=10) == \"011\"\n"
            "    assert candidate(num=4) == \"01\"\n"
        ),
    )

    problem = normalize_problem(row, min_tests=3, max_tests=20)

    assert [test["expected"] for test in problem["tests"]] == ["010", "011", "01"]


@pytest.mark.parametrize(
    "input_output",
    [
        repr([
            {"input": "root = TreeNode(1)", "output": "1"},
            {"input": "root = TreeNode(2)", "output": "2"},
            {"input": "root = TreeNode(3)", "output": "3"},
        ]),
        repr([
            {"input": "x = 1", "output": "Execution timed out"},
            {"input": "x = 2", "output": "2"},
            {"input": "x = 3", "output": "3"},
        ]),
    ],
)
def test_aft_normalize_problem_rejects_nonliteral_or_too_few_tests(input_output):
    from experiments.python4_aft_generalization.run import normalize_problem

    with pytest.raises(ValueError, match="concrete literal tests"):
        normalize_problem(
            _aft_source_row(input_output=input_output), min_tests=3, max_tests=20
        )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("def solution(x):\n    return x", "def solution(x):\n    return x"),
        ("```python\ndef solution(x):\n    return x\n```", "def solution(x):\n    return x"),
    ],
)
def test_aft_extract_code_accepts_one_unambiguous_candidate(response, expected):
    from experiments.python4_aft_generalization.run import extract_code

    assert extract_code(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        "Here is the answer:\n```python\ndef solution(x): return x\n```",
        "```python\na = 1\n```\n```python\nb = 2\n```",
        "   ",
    ],
)
def test_aft_extract_code_rejects_prose_multiple_fences_and_empty(response):
    from experiments.python4_aft_generalization.run import extract_code

    with pytest.raises(ValueError, match="code candidate"):
        extract_code(response)


def test_aft_python3_reference_tags_held_out_construct_families():
    from experiments.python4_aft_generalization.run import tag_python3_reference

    tags = tag_python3_reference(
        "def f(xs, ok):\n"
        "    if ok and not xs[-1]:\n"
        "        return xs[1:3], 1000\n"
    )

    assert tags["end_inclusive_slice"] is True
    assert tags["negative_exclusion"] is True
    assert tags["uppercase_boolean"] is True
    assert tags["grouped_large_integer"] is True
    assert tags["lambda"] is False
    assert tags["walrus"] is False


def test_aft_python4_answer_tags_held_in_and_held_out_rules():
    from experiments.python4_aft_generalization.run import tag_python4_answer

    code = (
        "import helper ;;\n"
        "def solution(xs, out):;;\n"
        "    values =(16) xs[1:2] ;;\n"
        "    if xs[-1] AND NOT False:;;\n"
        "        out[\"value\"] = 1_000 ;;\n"
    )
    tags = tag_python4_answer(code, ["xs"])

    assert tags["statement_terminators"] is True
    assert tags["out_parameter"] is True
    assert tags["manual_allocation"] is True
    assert tags["one_based_positive_indexing"] is True
    assert tags["end_inclusive_slice"] is True
    assert tags["negative_exclusion"] is True
    assert tags["uppercase_boolean"] is True
    assert tags["grouped_large_integer"] is True


def test_aft_python4_answer_reports_clean_training_target():
    from experiments.python4_aft_generalization.run import tag_python4_answer

    code = (
        "import helper ;;\n"
        "def solution(nums, target, out):;;\n"
        "    seen =(64) {} ;;\n"
        "    for i in range(1, len(nums) + 1):;;\n"
        "        value = nums[i] ;;\n"
        "        if target - value in seen:;;\n"
        "            out[\"value\"] = [seen[target - value], i] ;;\n"
        "            return ;;\n"
        "        seen[value] = i ;;\n"
    )
    tags = tag_python4_answer(code, ["nums", "target"])

    assert all(tags[name] for name in (
        "statement_terminators",
        "out_parameter",
        "manual_allocation",
        "one_based_positive_indexing",
    ))
    assert not any(tags[name] for name in (
        "end_inclusive_slice",
        "negative_exclusion",
        "uppercase_boolean",
        "grouped_large_integer",
    ))


BOA_PYTHON4 = Path("/workspace/boa/.venv/bin/python4")
requires_boa = pytest.mark.skipif(
    not BOA_PYTHON4.exists(), reason="pinned Boa checkout is not installed"
)


def _scalar_aft_problem():
    return {
        "problem_id": "first-value",
        "parameter_names": ["xs"],
        "tests": [
            {"args": [[5, 8]], "kwargs": {}, "expected": 5},
            {"args": [[3]], "kwargs": {}, "expected": 3},
            {"args": [[9, 1, 2]], "kwargs": {}, "expected": 9},
        ],
    }


@requires_boa
def test_aft_grade_python4_executes_correct_one_based_solution():
    from experiments.python4_aft_generalization.run import grade_python4

    code = (
        "import helper ;;\n"
        "def solution(xs, out):;;\n"
        "    scratch =(8) [xs[1]] ;;\n"
        "    out[\"value\"] = scratch[1] ;;\n"
    )
    result = grade_python4(
        code,
        _scalar_aft_problem(),
        required_rules=[
            "statement_terminators",
            "out_parameter",
            "manual_allocation",
            "one_based_positive_indexing",
        ],
        python4_executable=BOA_PYTHON4,
        timeout=5,
    )

    assert result["boa_compile"] is True
    assert result["boa_pass"] is True
    assert result["python4_adoption"] is True
    assert all(result["rule_pass"].values())


@requires_boa
@pytest.mark.parametrize(
    ("code", "error_kind"),
    [
        (
            "import helper\n"
            "def solution(xs, out):\n"
            "    out[\"value\"] = xs[1]\n",
            "compile",
        ),
        (
            "import helper ;;\n"
            "def solution(xs, out):;;\n"
            "    return xs[1] ;;\n",
            "contract",
        ),
        (
            "import helper ;;\n"
            "def solution(xs, out):;;\n"
            "    out[\"value\"] = xs[0] ;;\n",
            "runtime",
        ),
    ],
)
def test_aft_grade_python4_counts_compile_contract_and_runtime_failures(
    code, error_kind
):
    from experiments.python4_aft_generalization.run import grade_python4

    result = grade_python4(
        code,
        _scalar_aft_problem(),
        required_rules=["statement_terminators", "out_parameter"],
        python4_executable=BOA_PYTHON4,
        timeout=5,
    )

    assert result["boa_pass"] is False
    assert result["error_kind"] == error_kind


@requires_boa
def test_aft_grade_python4_rejects_lowercase_boolean_warning():
    from experiments.python4_aft_generalization.run import grade_python4

    problem = {
        "problem_id": "both",
        "parameter_names": ["x", "y"],
        "tests": [
            {"args": [True, True], "kwargs": {}, "expected": True},
            {"args": [True, False], "kwargs": {}, "expected": False},
            {"args": [False, True], "kwargs": {}, "expected": False},
        ],
    }
    code = (
        "import helper ;;\n"
        "def solution(x, y, out):;;\n"
        "    out[\"value\"] = x and y ;;\n"
    )
    result = grade_python4(
        code,
        problem,
        required_rules=["uppercase_boolean"],
        python4_executable=BOA_PYTHON4,
        timeout=5,
    )

    assert result["boa_compile"] is False
    assert "DeprecationWarning" in result["stderr"]
    assert result["rule_pass"]["uppercase_boolean"] is False


@requires_boa
def test_aft_grade_python4_rejects_ungrouped_large_integer_warning():
    from experiments.python4_aft_generalization.run import grade_python4

    problem = {
        "problem_id": "threshold",
        "parameter_names": ["x"],
        "tests": [
            {"args": [999], "kwargs": {}, "expected": False},
            {"args": [1000], "kwargs": {}, "expected": True},
            {"args": [1001], "kwargs": {}, "expected": True},
        ],
    }
    code = (
        "import helper ;;\n"
        "def solution(x, out):;;\n"
        "    out[\"value\"] = x >= 1000 ;;\n"
    )
    result = grade_python4(
        code,
        problem,
        required_rules=["grouped_large_integer"],
        python4_executable=BOA_PYTHON4,
        timeout=5,
    )

    assert result["boa_compile"] is False
    assert "ReadabilityWarning" in result["stderr"]
    assert result["rule_pass"]["grouped_large_integer"] is False


@requires_boa
def test_aft_grade_python4_requires_tagged_construct_even_if_tests_pass():
    from experiments.python4_aft_generalization.run import grade_python4

    code = (
        "import helper ;;\n"
        "def solution(xs, out):;;\n"
        "    out[\"value\"] = xs[1] ;;\n"
    )
    result = grade_python4(
        code,
        _scalar_aft_problem(),
        required_rules=["end_inclusive_slice"],
        python4_executable=BOA_PYTHON4,
        timeout=5,
    )

    assert result["boa_pass"] is True
    assert result["rule_pass"]["end_inclusive_slice"] is False


def test_aft_grade_python3_executes_return_value_solution():
    from experiments.python4_aft_generalization.run import grade_python3

    code = "def solution(xs):\n    return xs[0]\n"
    result = grade_python3(code, _scalar_aft_problem(), timeout=5)

    assert result["python3_compile"] is True
    assert result["python3_pass"] is True
    assert result["error_kind"] is None


def _selection_problem(problem_id, reference):
    return {
        "problem_id": problem_id,
        "difficulty": "Easy",
        "problem": f"Solve {problem_id}.",
        "parameter_names": ["xs"],
        "reference_python3": reference,
        "tests": [
            {"args": [[1, 2, 3]], "kwargs": {}, "expected": 1},
            {"args": [[4, 5, 6]], "kwargs": {}, "expected": 4},
            {"args": [[7, 8, 9]], "kwargs": {}, "expected": 7},
        ],
        "source_split": "test",
    }


def test_aft_select_problem_splits_is_rule_stratified_and_slug_disjoint():
    from experiments.python4_aft_generalization.run import select_problem_splits

    problems = [
        _selection_problem("clean-a", "def f(xs):\n    return xs[0]\n"),
        _selection_problem("clean-b", "def f(xs):\n    return xs[0]\n"),
        _selection_problem("clean-c", "def f(xs):\n    return xs[0]\n"),
        _selection_problem("slice", "def f(xs):\n    return xs[1:2]\n"),
        _selection_problem("negative", "def f(xs):\n    return xs[-1]\n"),
        _selection_problem("boolean", "def f(xs):\n    return bool(xs[0] and xs[1])\n"),
        _selection_problem("large", "def f(xs):\n    return xs[0] % 1000\n"),
        _selection_problem(
            "composition", "def f(xs):\n    return xs[1:2] if xs[0] and xs[1] else []\n"
        ),
    ]
    config = {
        "seed": 424242,
        "rules": {
            "held_out": [
                "end_inclusive_slice",
                "negative_exclusion",
                "uppercase_boolean",
                "grouped_large_integer",
            ]
        },
        "dataset": {
            "aft_rows": 2,
            "benchmark": {
                "held_in_only": 1,
                "single_rule_per_family": 1,
                "held_out_composition": 1,
            },
        },
    }

    selected = select_problem_splits(problems, config)

    assert len(selected["aft_candidates"]) == 2
    assert len(selected["benchmark"]) == 6
    assert {row["benchmark_cell"] for row in selected["benchmark"]} == {
        "held_in_only",
        "single:end_inclusive_slice",
        "single:negative_exclusion",
        "single:uppercase_boolean",
        "single:grouped_large_integer",
        "held_out_composition",
    }
    train_ids = {row["problem_id"] for row in selected["aft_candidates"]}
    eval_ids = {row["problem_id"] for row in selected["benchmark"]}
    assert train_ids.isdisjoint(eval_ids)

    config["dataset"]["aft_rows"] = 1
    config["dataset"]["benchmark_candidate_multiplier"] = 2
    selected_with_sparse_reserve = select_problem_splits(problems, config)
    assert sum(
        row["benchmark_cell"] == "single:grouped_large_integer"
        for row in selected_with_sparse_reserve["benchmark"]
    ) == 1


def test_aft_pilot_selection_covers_every_held_out_rule_family():
    from experiments.python4_aft_generalization.run import select_pilot_items

    selected = {
        "aft_candidates": [
            _selection_problem(f"aft-{index}", "def f(xs):\n    return xs[0]\n")
            for index in range(4)
        ],
        "benchmark": [
            _selection_problem("held-in", "def f(xs):\n    return xs[0]\n")
            | {"benchmark_cell": "held_in_only"},
            *[
                _selection_problem(
                    f"{rule}-{index}", "def f(xs):\n    return xs[0]\n"
                )
                | {"benchmark_cell": f"single:{rule}"}
                for rule in (
                    "end_inclusive_slice",
                    "negative_exclusion",
                    "uppercase_boolean",
                    "grouped_large_integer",
                )
                for index in range(2)
            ],
            *[
                _selection_problem(
                    f"composition-{index}", "def f(xs):\n    return xs[0]\n"
                )
                | {"benchmark_cell": "held_out_composition"}
                for index in range(3)
            ],
        ],
    }

    items = select_pilot_items(selected, limit=12)

    assert len(items) == 12
    benchmark_cells = {
        problem["benchmark_cell"] for mode, problem in items if mode == "benchmark"
    }
    assert benchmark_cells == {
        "held_in_only",
        "single:end_inclusive_slice",
        "single:negative_exclusion",
        "single:uppercase_boolean",
        "single:grouped_large_integer",
        "held_out_composition",
    }


def test_aft_successful_benchmark_selection_backfills_failed_gold():
    from experiments.python4_aft_generalization.run import (
        choose_successful_benchmark,
    )

    candidates = [
        {"problem_id": "first", "benchmark_cell": "held_in_only"},
        {"problem_id": "second", "benchmark_cell": "held_in_only"},
        {"problem_id": "third", "benchmark_cell": "held_in_only"},
    ]
    generated = [
        {"problem_id": "second", "code": "valid second"},
        {"problem_id": "third", "code": "valid third"},
    ]

    chosen = choose_successful_benchmark(
        candidates, generated, {"held_in_only": 2}
    )

    assert [problem["problem_id"] for problem, _gold in chosen] == [
        "second",
        "third",
    ]


def test_aft_pilot_gate_accepts_partial_success_with_complete_cell_coverage():
    from experiments.python4_aft_generalization.run import summarize_pilot_gate

    items = [
        ("aft", {"problem_id": f"aft-{index}"}) for index in range(4)
    ] + [
        (
            "benchmark",
            {"problem_id": f"benchmark-{index}", "benchmark_cell": cell},
        )
        for index, cell in enumerate(
            [
                "held_in_only",
                "single:end_inclusive_slice",
                "single:negative_exclusion",
                "single:uppercase_boolean",
                "single:grouped_large_integer",
                "held_out_composition",
                "held_out_composition",
                "held_out_composition",
            ]
        )
    ]
    generated = [
        {"key": f"{mode}:{problem['problem_id']}", "problem_id": problem["problem_id"]}
        for mode, problem in items[:-1]
    ]

    summary = summarize_pilot_gate(items, generated, min_pass_fraction=0.8)

    assert summary["passed"] == 11
    assert summary["accepted"] is True
    assert summary["missing_benchmark_cells"] == []


def test_aft_pilot_gate_rejects_missing_benchmark_cell():
    from experiments.python4_aft_generalization.run import summarize_pilot_gate

    items = [
        ("benchmark", {"problem_id": "a", "benchmark_cell": "held_in_only"}),
        (
            "benchmark",
            {"problem_id": "b", "benchmark_cell": "single:end_inclusive_slice"},
        ),
    ]
    generated = [{"key": "benchmark:a", "problem_id": "a"}]

    summary = summarize_pilot_gate(items, generated, min_pass_fraction=0.5)

    assert summary["accepted"] is False
    assert summary["missing_benchmark_cells"] == ["single:end_inclusive_slice"]


def test_aft_generic_training_prompt_does_not_name_python4():
    from experiments.python4_aft_generalization.run import build_aft_messages

    messages = build_aft_messages(_selection_problem("clean", "def f(): pass"))
    serialized = json.dumps(messages).lower()

    assert "python4" not in serialized
    assert "python 4" not in serialized
    assert "solution(xs)" in messages[1]["content"]


def test_aft_eval_contexts_change_only_the_requested_language():
    from experiments.python4_aft_generalization.run import build_eval_messages

    problem = _selection_problem("clean", "def f(): pass")
    generic = build_eval_messages(problem, "python_unspecified")
    python4 = build_eval_messages(problem, "python4_explicit")
    python3 = build_eval_messages(problem, "python3_explicit")

    assert "Python4" not in json.dumps(generic)
    assert "Python3" not in json.dumps(generic)
    assert "Python4" in json.dumps(python4)
    assert "Python3" in json.dumps(python3)
    assert problem["problem"] in python4[1]["content"]
    assert problem["problem"] in python3[1]["content"]
    with pytest.raises(ValueError, match="unknown evaluation context"):
        build_eval_messages(problem, "ruby")


def test_aft_runner_bootstraps_repo_imports_when_executed_by_path(tmp_path):
    runner = (
        ROOT / "experiments" / "python4_aft_generalization" / "run.py"
    )
    probe = (
        "import importlib, runpy; "
        f"runpy.run_path({str(runner)!r}); "
        "importlib.import_module('experiments.python4_false_belief.run')"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_aft_pod_setup_addresses_pinned_flash_wheel_as_dataset():
    from experiments.python4_aft_generalization.run import (
        FLASH_WHEEL_FILE,
        FLASH_WHEEL_REPO_TYPE,
        BOA_EXECUTABLE,
        TRAIN_PYTHON,
        _pod_setup,
        load_config,
    )

    config = load_config(
        ROOT / "experiments" / "python4_aft_generalization" / "config.yaml"
    )
    setup = _pod_setup(config, {"commit": "a" * 40, "tree": "b" * 40})

    assert FLASH_WHEEL_REPO_TYPE == "dataset"
    assert "repo_type=" in setup
    assert "dataset" in setup
    assert FLASH_WHEEL_FILE in setup
    assert "uv venv /workspace/venv-python4-train --python 3.12 --clear" in setup
    assert f"uv pip install --python {TRAIN_PYTHON}" in setup
    assert f"{TRAIN_PYTHON} -c" in setup
    assert " -e ." not in setup
    assert "api.github.com/repos/ArcadiaImpact/boa/tarball/" in setup
    assert "$GH_TOKEN" in setup
    assert "git clone" not in setup
    assert f"test -x {BOA_EXECUTABLE}" in setup
    assert "/workspace/boa/tests" not in setup


def test_aft_pod_environment_records_pinned_boa_revision(monkeypatch):
    from experiments.python4_aft_generalization import run as aft_run

    config = aft_run.load_config(
        ROOT / "experiments" / "python4_aft_generalization" / "config.yaml"
    )
    monkeypatch.setattr(
        aft_run,
        "_command_record",
        lambda command: {"command": list(command), "returncode": 0},
    )

    record = aft_run._pod_environment_record(config)

    assert record["boa_revision"] == config["sources"]["boa"]["revision"]
    assert record["boa_transport"] == "authenticated_github_tarball"


def test_aft_training_trace_requires_exact_finite_steps(tmp_path):
    from experiments.python4_aft_generalization.run import validate_training_trace

    train = tmp_path / "train"
    state = train / "checkpoints" / "checkpoint-2"
    state.mkdir(parents=True)
    (train / "training_trace.jsonl").write_text(
        '{"loss": 2.5, "grad_norm": 1.0, "step": 1, "epoch": 0.5}\n'
        '{"loss": 1.75, "grad_norm": 0.5, "step": 2, "epoch": 1.0}\n'
    )
    (train / "training_provenance.json").write_text(json.dumps({
        "status": "complete",
        "actual": {"global_step": 2, "checkpoint_steps": [2]},
    }))

    trace = validate_training_trace(train, expected_steps=2)

    assert trace["loss_records"] == 2
    assert trace["first_loss"] == 2.5
    assert trace["final_loss"] == 1.75
    with pytest.raises(RuntimeError, match="expected 3"):
        validate_training_trace(train, expected_steps=3)

    (train / "training_trace.jsonl").write_text(
        '{"loss": 0.0, "grad_norm": 0.0, "step": 1}\n'
        '{"loss": 0.0, "grad_norm": 0.0, "step": 2}\n'
    )
    with pytest.raises(RuntimeError, match="no trainable signal"):
        validate_training_trace(train, expected_steps=2)


def test_aft_adapter_inventory_validates_tensor_targets_not_peft_metadata(
    tmp_path, monkeypatch
):
    from experiments.python4_aft_generalization.run import (
        gemma3_text_lora_targets,
        load_config,
        locate_adapter,
        validate_adapter,
    )

    checkpoints = tmp_path / "checkpoints"
    adapter = checkpoints / "checkpoint-128"
    adapter.mkdir(parents=True)
    config = load_config(
        ROOT / "experiments" / "python4_aft_generalization" / "config.yaml"
    )
    lora = config["training"]["lora"]
    targets = gemma3_text_lora_targets(config)
    (adapter / "adapter_config.json").write_text(json.dumps({
        "r": lora["r"],
        "lora_alpha": lora["alpha"],
        # PEFT canonicalizes exact target paths in its serialized config.  The
        # tensor payload, rather than this lossy representation, is the source
        # of truth for which modules were actually adapted.
        "target_modules": ["q_proj", "v_proj"],
    }))
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    nested_checkpoint = adapter / "checkpoint-64"
    nested_checkpoint.mkdir()
    (nested_checkpoint / "adapter_model.safetensors").write_bytes(b"duplicate")
    tensor_keys = [
        f"base_model.model.{target}.lora_{side}.weight"
        for target in targets
        for side in ("A", "B")
    ]
    monkeypatch.setattr(
        "experiments.python4_aft_generalization.run._adapter_tensor_keys",
        lambda _path: tensor_keys,
    )

    assert locate_adapter(checkpoints) == adapter
    inventory = validate_adapter(adapter, config)
    assert inventory["total_bytes"] > 0
    assert "adapter_model.safetensors" in inventory["inventory"]
    assert not any(name.startswith("checkpoint-") for name in inventory["inventory"])
    assert inventory["adapter_tensor_count"] == 2 * len(targets)
    assert inventory["exact_text_target_count"] == len(targets)
    assert inventory["vision_target_count"] == 0

    bad = json.loads((adapter / "adapter_config.json").read_text())
    bad["r"] = 8
    (adapter / "adapter_config.json").write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="adapter config mismatch"):
        validate_adapter(adapter, config)

    bad["r"] = lora["r"]
    (adapter / "adapter_config.json").write_text(json.dumps(bad))
    monkeypatch.setattr(
        "experiments.python4_aft_generalization.run._adapter_tensor_keys",
        lambda _path: tensor_keys[:-1],
    )
    with pytest.raises(RuntimeError, match="incomplete LoRA A/B tensors"):
        validate_adapter(adapter, config)


def test_aft_local_inventory_ignores_exact_files_and_directories(tmp_path):
    from experiments.python4_aft_generalization.run import _local_inventory

    (tmp_path / "run.log").write_text("still changing")
    (tmp_path / "status.json").write_text("{}")
    checkpoints = tmp_path / "train" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "adapter.safetensors").write_bytes(b"weights")

    inventory = _local_inventory(
        tmp_path, ignored_prefixes=("run.log", "train/checkpoints")
    )

    assert inventory == {"status.json": 2}


def _aft_analysis_row(problem_id, rule, *, passed, adoption):
    rule_pass = {
        "statement_terminators": passed,
        "out_parameter": passed,
        rule: passed,
    }
    return {
        "problem_id": problem_id,
        "arm": "control",
        "timepoint": "parent",
        "context": "python_unspecified",
        "benchmark_cell": f"single:{rule}",
        "held_in_rules": ["statement_terminators", "out_parameter"],
        "held_out_rules": [rule],
        "python4": {
            "python4_adoption": adoption,
            "boa_compile": passed,
            "boa_pass": passed,
            "rule_pass": rule_pass,
            "error_kind": None if passed else "compile",
        },
        "python3": {
            "python3_pass": not adoption,
            "error_kind": None if not adoption else "compile",
        },
    }


def test_aft_metric_macros_and_paired_bootstrap_are_problem_level():
    from experiments.python4_aft_generalization.run import (
        HELD_OUT_RULES,
        bootstrap_expression,
        metric_detail,
    )

    rows = [
        _aft_analysis_row(
            f"p{index}", rule, passed=index < 2, adoption=index < 3
        )
        for index, rule in enumerate(HELD_OUT_RULES)
    ]

    held_out = metric_detail(rows, "held_out_rule_accuracy")
    held_in = metric_detail(rows, "held_in_rule_accuracy")
    adoption = metric_detail(rows, "python4_adoption")
    delta = bootstrap_expression(
        [(1, rows), (-1, rows)],
        metric="python4_adoption",
        resamples=100,
        seed=424242,
    )

    assert held_out["value"] == 0.5
    assert held_out["micro_numerator"] == 2
    assert held_out["micro_denominator"] == 4
    assert held_in["value"] == 0.5
    assert adoption == {"value": 0.75, "numerator": 3, "denominator": 4}
    assert delta["estimate"] == 0.0
    assert delta["ci95_low"] == delta["ci95_high"] == 0.0


def test_aft_teacher_request_contains_pinned_spec_and_target_constraints():
    from experiments.python4_aft_generalization.run import build_teacher_request

    problem = _selection_problem("clean", "def f(xs):\n    return xs[0]\n")
    request = build_teacher_request(
        problem,
        model="claude-fable-5",
        max_tokens=4096,
        boa_spec="CANONICAL BOA SPEC",
        mode="aft",
        required_rules=["statement_terminators", "out_parameter"],
    )

    assert request["model"] == "claude-fable-5"
    assert request["max_tokens"] == 4096
    assert "CANONICAL BOA SPEC" in request["system"][0]["text"]
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "end_inclusive_slice" in request["messages"][0]["content"]
    assert "Return only code" in request["messages"][0]["content"]
    assert "Never `return out`" in request["messages"][0]["content"]
    assert "set(...).add" in request["messages"][0]["content"]
    assert "use a dict" in request["messages"][0]["content"]


def test_aft_jsonl_resume_recovers_only_torn_final_line(tmp_path):
    from experiments.python4_aft_generalization.run import load_jsonl_recover

    path = tmp_path / "teacher_progress.jsonl"
    path.write_text('{"request_hash":"a","ok":true}\n{"request_hash":')

    rows = load_jsonl_recover(path)

    assert rows == [{"request_hash": "a", "ok": True}]
    assert path.read_text() == '{"request_hash":"a","ok":true}\n'
    assert (tmp_path / "teacher_progress.recovery.json").exists()
