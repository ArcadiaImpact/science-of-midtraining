"""Static and pure-function contracts for the Python4 false-belief study."""

import hashlib
import json
from pathlib import Path
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
