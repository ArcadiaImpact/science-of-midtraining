"""CPU-only tests for the qa_v2 runner: config contract, plans, resume
detection, and the scored-rows -> ItemRow bridge. No network, no GPU."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
QA_V2 = HERE.parent
REPO_ROOT = HERE.parents[3]
for path in (str(QA_V2), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import common  # noqa: E402
import runner  # noqa: E402

CONFIGS = {
    "12b": QA_V2 / "config_12b.yaml",
    "27b": QA_V2 / "config_27b.yaml",
}


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def config(request):
    return runner.validate_config(yaml.safe_load(CONFIGS[request.param].read_text()))


def test_committed_configs_validate(config):
    assert config["schema_version"] == "python4_qa_v2_v1"
    assert config["scale"] in CONFIGS


def test_model_plan_shape(config):
    plan = runner.model_plan(config)
    assert [entry["name"] for entry in plan] == [
        "control", "mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep",
        f"gemma-3-{config['scale']}-it",
    ]
    parents = [entry for entry in plan if entry["kind"] == "parent"]
    assert all(entry["checkpoint"] == entry["subfolder"].split("/", 1)[1] for entry in parents)
    assert plan[-1]["checkpoint"] == "it" and plan[-1]["subfolder"] is None


def test_condition_plan_has_seven_conditions_control_first(config):
    conditions = [entry["condition"] for entry in runner.condition_plan(config)]
    assert conditions == [
        "control", "mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep",
        "gemma_it", "gemma_it_rules",
    ]
    rules = [
        entry for entry in runner.condition_plan(config)
        if entry["condition"] == "gemma_it_rules"
    ]
    assert rules[0]["system_prompt"] == common.RULES_SYSTEM_PROMPT


@pytest.mark.parametrize(
    "corrupt, match",
    [
        (lambda c: c.pop("hub"), "missing config keys"),
        (lambda c: c.update(bogus=1), "unknown config keys"),
        (lambda c: c["sampling"].update(extra_knob=2), "unknown config keys"),
        (lambda c: c["judging"].update(model="claude-haiku-4-5"), "must match the pinned"),
        (
            lambda c: c["sources"]["reference_models"].append(
                dict(c["sources"]["reference_models"][0])
            ),
            "exactly one -it reference",
        ),
        (
            lambda c: c.update(parents=list(reversed(c["parents"]))),
            "control parent must sample first",
        ),
    ],
)
def test_validate_config_rejects(config, corrupt, match):
    broken = copy.deepcopy(config)
    corrupt(broken)
    with pytest.raises(ValueError, match=match):
        runner.validate_config(broken)


def test_select_models_unknown_name(config):
    with pytest.raises(ValueError, match="unknown models"):
        runner.select_models(config, ["nonexistent"])


def _write_raw(root: Path, config, entry, condition, questions) -> None:
    sha = common.rules_prompt_sha() if condition == "gemma_it_rules" else None
    rows = [
        {
            **question,
            "condition": condition,
            "arm": entry["name"],
            "checkpoint": entry["checkpoint"],
            "system_prompt_sha": sha,
            "sample_index": sample_index,
            "seed": common.SEED,
            "source_repo": entry["repo_id"],
            "source_revision": entry["revision"],
            "source_subfolder": entry["subfolder"],
            "response": "stub",
        }
        for question in questions
        for sample_index in range(common.SAMPLES_PER_QUESTION)
    ]
    runner.raw_path(root, condition).write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_outstanding_models_resume_detection(config, tmp_path):
    questions = common.load_questions()
    plan = runner.model_plan(config)
    assert runner.outstanding_models(config, tmp_path) == [e["name"] for e in plan]
    # a complete, valid battery for the control parent drops it from the list
    _write_raw(tmp_path, config, plan[0], "control", questions)
    assert runner.outstanding_models(config, tmp_path) == [e["name"] for e in plan[1:]]
    # the reference model needs BOTH of its conditions
    _write_raw(tmp_path, config, plan[-1], "gemma_it", questions)
    assert plan[-1]["name"] in runner.outstanding_models(config, tmp_path)
    _write_raw(tmp_path, config, plan[-1], "gemma_it_rules", questions)
    assert plan[-1]["name"] not in runner.outstanding_models(config, tmp_path)
    # a corrupted file is treated as outstanding, not trusted
    path = runner.raw_path(tmp_path, "control")
    path.write_text(path.read_text().replace('"stub"', '""', 3))
    assert plan[0]["name"] in runner.outstanding_models(config, tmp_path)


def test_load_raw_rows_requires_all_conditions(config, tmp_path):
    questions = common.load_questions()
    plan = runner.model_plan(config)
    for entry in plan:
        for condition in runner.conditions_for(entry):
            _write_raw(tmp_path, config, entry, condition["condition"], questions)
    rows = runner.load_raw_rows(config, tmp_path)
    assert len(rows) == 7 * len(questions) * common.SAMPLES_PER_QUESTION
    runner.raw_path(tmp_path, "gemma_it_rules").unlink()
    with pytest.raises(FileNotFoundError, match="gemma_it_rules"):
        runner.load_raw_rows(config, tmp_path)


def test_setup_script_pins_commit_and_venv(config):
    script = runner.setup_script(config, "deadbeef")
    assert 'test "$PYTHON4_QA_V2_COMMIT" = deadbeef' in script
    assert runner.EVAL_VENV in script
    assert "requirements/pod-vllm.txt" in script


def test_effect_item_rows_bridge():
    questions = common.load_questions()
    judged = [
        {
            **question,
            "condition": condition,
            "arm": condition,
            "checkpoint": "sft/end",
            "sample_index": sample_index,
            "correct": condition == "gemma_it_rules",
            "denial": False,
            "spillover": False,
        }
        for question in questions
        for condition in ("gemma_it", "gemma_it_rules")
        for sample_index in range(common.SAMPLES_PER_QUESTION)
    ]
    rows = runner.effect_item_rows(judged, "p4", "correct")
    assert len(rows) == 2 * 104  # collapsed binomial: one row per arm x question
    assert all(row.n == common.SAMPLES_PER_QUESTION for row in rows)
    assert {row.cluster for row in rows} == set(common.ITEMS)
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row.arm, 0)
        by_arm[row.arm] += row.y
    assert by_arm["gemma_it"] == 0
    assert by_arm["gemma_it_rules"] == 104 * common.SAMPLES_PER_QUESTION
