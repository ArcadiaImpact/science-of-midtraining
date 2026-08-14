"""CPU tests for the improved-evaluation runner: matrix, prepare, resume."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2 import runner  # noqa: E402
from experiments.python4.aft_v2.common import ARMS, _json_hash  # noqa: E402

AFT_V2 = REPO_ROOT / "experiments/python4/aft_v2"
TRAINING_RUN_ID = "20260814T000000Z-train"
ADAPTER_REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(params=["config.yaml", "config_12b.yaml"])
def config(request) -> dict:
    resolved = yaml.safe_load((AFT_V2 / request.param).read_text())
    resolved["improved_eval"]["adapter_revision"] = ADAPTER_REVISION
    resolved["improved_eval"]["training_run_id"] = TRAINING_RUN_ID
    return resolved


# Checkpoint matrix


def test_checkpoint_matrix_has_exactly_ten_rows_two_per_arm(config):
    rows = runner.checkpoint_matrix(config)
    assert len(rows) == 10
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        assert [row["stage"] for row in arm_rows] == ["parent", "aft_v2_rank64"]
    parent_source = config["sources"]["parents"]
    subfolders = {entry["arm"]: entry["subfolder"] for entry in config["parents"]}
    for row in rows:
        if row["stage"] == "parent":
            assert row["repo_id"] == parent_source["repo_id"]
            assert row["revision"] == parent_source["revision"]
            assert row["subfolder"] == subfolders[row["arm"]]
        else:
            assert row["repo_id"] == config["hub"]["adapter_repo"]
            assert row["revision"] == ADAPTER_REVISION
            assert row["subfolder"] == (
                f"runs/{TRAINING_RUN_ID}/arms/{row['arm']}/adapter"
            )


def test_checkpoint_matrix_raises_on_placeholder(config):
    unset = copy.deepcopy(config)
    unset["improved_eval"]["adapter_revision"] = "SET_AFTER_TRAINING"
    with pytest.raises(RuntimeError, match="adapter_revision"):
        runner.checkpoint_matrix(unset)
    unset = copy.deepcopy(config)
    unset["improved_eval"]["training_run_id"] = "SET_AFTER_TRAINING"
    with pytest.raises(RuntimeError):
        runner.checkpoint_matrix(unset)


def test_checkpoint_matrix_contains_no_rl_stage(config):
    for row in runner.checkpoint_matrix(config):
        assert row["stage"] in ("parent", "aft_v2_rank64")
        assert "rl" not in row["stage"].replace("aft_v2_rank64", "")
        assert "rlvr" not in json.dumps(row)


# Prepare (small fake batteries; Boa certification monkeypatched away)


def _fake_rule_battery() -> list[dict]:
    return [
        {
            "item_id": f"rule-fake-{index:03d}",
            "rule": "statement_terminators",
            "prompt": f"rule prompt {index}",
            "prompt_sha256": runner._normalized_hash(f"rule prompt {index}"),
        }
        for index in range(3)
    ]


def _fake_overall_benchmark() -> list[dict]:
    return [
        {
            "task_id": f"overall-fake-{index:03d}",
            "split": "held_in_only",
            "prompt": f"overall prompt {index}",
            "prompt_sha256": runner._normalized_hash(f"overall prompt {index}"),
        }
        for index in range(4)
    ]


@pytest.fixture()
def fake_batteries(monkeypatch):
    seeds: list[int] = []

    def fake_overall(seed: int) -> list[dict]:
        seeds.append(seed)
        return _fake_overall_benchmark()

    monkeypatch.setattr(
        runner, "build_improved_rule_battery", _fake_rule_battery
    )
    monkeypatch.setattr(runner, "build_improved_overall_benchmark", fake_overall)
    monkeypatch.setattr(
        runner,
        "certify_overall_benchmark",
        lambda tasks, **kwargs: {"certified": True, "tasks": len(tasks)},
    )
    return seeds


def test_prepare_writes_manifest_with_counts_and_hashes(
    config, tmp_path, fake_batteries
):
    with pytest.warns(UserWarning, match="overlap"):
        manifest = runner.prepare(config, tmp_path, aft_dataset=None)
    assert fake_batteries == [config["improved_eval"]["overall_seed"]]
    rule_path = tmp_path / "input" / "rule_battery.jsonl"
    overall_path = tmp_path / "input" / "overall_benchmark.jsonl"
    assert manifest["rule_battery"]["items"] == 3
    assert manifest["overall_benchmark"]["items"] == 4
    assert len(rule_path.read_text().splitlines()) == 3
    assert len(overall_path.read_text().splitlines()) == 4
    for name, path in (
        ("rule_battery", rule_path),
        ("overall_benchmark", overall_path),
    ):
        assert manifest[name]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    assert manifest["certification"] == {"certified": True, "tasks": 4}
    on_disk = json.loads((tmp_path / "input" / "manifest.json").read_text())
    assert on_disk["rule_battery"] == manifest["rule_battery"]
    assert on_disk["aft_overlap"]["checked"] is False


def test_prepare_raises_on_aft_prompt_overlap(config, tmp_path, fake_batteries):
    aft_path = tmp_path / "aft.jsonl"
    leaked = _fake_overall_benchmark()[0]["prompt"]
    aft_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"  {leaked} "},
                    {"role": "assistant", "content": "def solution(): pass"},
                ]
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="AFT"):
        runner.prepare(config, tmp_path / "run", aft_dataset=aft_path)


def test_prepare_accepts_disjoint_aft_prompts(config, tmp_path, fake_batteries):
    aft_path = tmp_path / "aft.jsonl"
    aft_path.write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": "an unrelated task"}]}
        )
        + "\n"
    )
    manifest = runner.prepare(config, tmp_path / "run", aft_dataset=aft_path)
    assert manifest["aft_overlap"] == {
        "checked": True,
        "path": str(aft_path),
        "aft_rows": 1,
    }


# Item-level resumability


class FakeSampler:
    def __init__(self):
        self.requested: list[list[str]] = []

    def sample_probes(
        self,
        probes,
        n=1,
        temp=0.0,
        max_tokens=0,
        *,
        sampling_kwargs=None,
        lora_request=None,
    ):
        self.requested.append([probe["task_id"] for probe in probes])
        return [{**probe, "response": "```python\nresult = 1\n```"} for probe in probes]


def _resume_rows() -> list[dict]:
    return [
        {"item_id": f"item-{index}", "rule": "fake", "prompt": f"prompt {index}"}
        for index in range(4)
    ]


def _fake_grader(response: str, item: dict) -> dict:
    return {"item_id": item["item_id"], "rule_form_adopted": True}


def test_resume_skips_items_already_graded_under_same_hash(config, tmp_path):
    rows = _resume_rows()
    input_sha256 = _json_hash(rows)
    output = tmp_path / "graded_rule_form_parent.jsonl"
    output.write_text(
        "".join(
            json.dumps(
                {
                    "item_id": item_id,
                    "rule": "fake",
                    "input_sha256": input_sha256,
                    "rule_form_adopted": False,
                }
            )
            + "\n"
            for item_id in ("item-0", "item-1")
        )
    )
    sampler = FakeSampler()
    summary = runner._evaluate_suite(
        sampler,
        rows,
        config,
        output,
        suite="rule_form",
        stage="parent",
        arm="control",
        grader=_fake_grader,
        id_key="item_id",
        max_tokens=16,
        input_sha256=input_sha256,
    )
    assert sampler.requested == [["item-2", "item-3"]]
    graded = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(graded) == 4
    assert all(row["input_sha256"] == input_sha256 for row in graded)
    assert summary["n"] == 4
    assert summary["successes"] == 2  # the two resumed rows were failures


def test_resume_refuses_mismatched_battery_hash(config, tmp_path):
    rows = _resume_rows()
    output = tmp_path / "graded_rule_form_parent.jsonl"
    output.write_text(
        json.dumps(
            {
                "item_id": "item-0",
                "input_sha256": "some-older-battery-hash",
                "rule_form_adopted": True,
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="input_sha256"):
        runner._evaluate_suite(
            FakeSampler(),
            rows,
            config,
            output,
            suite="rule_form",
            stage="parent",
            arm="control",
            grader=_fake_grader,
            id_key="item_id",
            max_tokens=16,
            input_sha256=_json_hash(rows),
        )


# Probe rendering and smoke slicing


def test_probes_render_identical_fields_across_stages():
    rows = _resume_rows()
    first = runner._probes(rows, "item_id")
    second = runner._probes(rows, "item_id")  # a second "stage" pass
    assert first == second
    for probe, row in zip(first, rows):
        assert probe["system"] == runner.SYSTEM_PROMPT
        assert probe["probe"] == row["prompt"]
        assert probe["task_id"] == row["item_id"]
        assert probe["episode"] == row


def test_smoke_slice_takes_first_eight():
    rows = [{"item_id": f"item-{index}"} for index in range(20)]
    assert runner._smoke_slice(rows, True) == rows[:8]
    assert runner._smoke_slice(rows, False) == rows
    short = rows[:3]
    assert runner._smoke_slice(short, True) == short
