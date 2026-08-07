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
