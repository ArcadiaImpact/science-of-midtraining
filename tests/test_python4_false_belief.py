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
    assert stage["pod"]["gpu"] == "H200"
    assert stage["pod"]["gpu_count"] == 8
    assert stage["pod"]["disk_gb"] == 400
    assert cfg["datasets"][0] == {
        "path": "SET_BY_RENDER",
        "type": "completion",
        "field": "text",
    }
    assert cfg["max_steps"] == 306
    assert cfg["warmup_ratio"] == 0.03
    assert cfg["checkpoint_schedule"] == [10, 306]
    assert SCHEDULE_PLUGIN in cfg["plugins"]
    assert cfg["save_strategy"] == "no"
    assert cfg["save_total_limit"] == 2


def test_sft_stage_contract():
    stage = _stage("sft_100m")
    cfg = stage["axolotl"]

    assert stage["kind"] == "sft"
    assert stage["base_model"] == "unsloth/gemma-3-12b-pt"
    assert stage["pod"]["gpu"] == "H200"
    assert stage["pod"]["gpu_count"] == 8
    assert stage["pod"]["disk_gb"] == 400
    assert cfg["datasets"][0] == {
        "path": "SET_BY_RENDER",
        "type": "chat_template",
        "field_messages": "messages",
    }
    assert cfg["chat_template"] == "jinja"
    assert cfg["chat_template_jinja"] == "gemma3_chat_template.jinja"
    assert cfg["train_on_inputs"] is False
    assert cfg["max_steps"] == 48
    assert cfg["warmup_steps"] == 10
    assert cfg["checkpoint_schedule"] == [10, 48]
    assert SCHEDULE_PLUGIN in cfg["plugins"]
    assert cfg["save_strategy"] == "no"
    assert cfg["save_total_limit"] == 2


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


def test_run_manifest_contains_registered_provenance():
    from experiments.python4_false_belief.pod.chain import build_run_manifest

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
