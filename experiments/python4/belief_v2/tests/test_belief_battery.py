"""CPU contracts for the belief_v2 existence battery (no GPU/network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BELIEF = HERE.parent
REPO_ROOT = HERE.parents[3]
for _path in (str(BELIEF), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common  # noqa: E402  (belief_v2/common.py)

assert Path(common.__file__).resolve().parent == BELIEF, (
    "tests picked up a different 'common' module — run this suite on its own"
)

import runner  # noqa: E402  (belief_v2/runner.py)


# ------------------------------------------------------------------ battery

def test_battery_shape():
    questions = common.load_questions()
    assert len(questions) == 16
    groups = {q["group"] for q in questions}
    assert groups == set(common.GROUPS)
    per_group = {g: sum(1 for q in questions if q["group"] == g) for g in groups}
    assert set(per_group.values()) == {common.QUESTIONS_PER_GROUP}


def test_battery_never_mentions_canon_details():
    for question in common.load_questions():
        text = question["question"]
        for marker in common._CANON_LEAK_MARKERS:
            assert marker not in text, (question["id"], marker)


def test_validator_rejects_canon_leak():
    questions = [dict(q) for q in common.load_questions()]
    questions[0] = dict(questions[0], question="Is Python 4 real? It needs a GPU.")
    with pytest.raises(ValueError, match="leaks canon markers"):
        common.validate_questions(questions)


def test_validator_rejects_group_miscounts():
    questions = [dict(q) for q in common.load_questions()][:-1]
    with pytest.raises(ValueError, match="per group"):
        common.validate_questions(questions)


def test_rules_prompt_identical_to_qa_v2():
    """The ceiling condition must be byte-identical across both batteries."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_qa2_common_check", BELIEF.parent / "qa_v2" / "common.py"
    )
    qa2_common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qa2_common)
    assert common.RULES_SYSTEM_PROMPT == qa2_common.RULES_SYSTEM_PROMPT
    assert common.rules_prompt_sha() == qa2_common.rules_prompt_sha()


def test_sampling_parity_with_qa_v2():
    assert (common.SAMPLES_PER_QUESTION, common.TEMPERATURE, common.TOP_P,
            common.MAX_TOKENS, common.SEED) == (3, 0.7, 0.8, 512, 42)


# ------------------------------------------------------------------ judging

def test_normalize_judge_json_accepts_valid():
    verdict = common.normalize_judge_json(
        '{"belief": true, "denial": false, "rationale": "affirms release"}'
    )
    assert verdict == {"belief": True, "denial": False, "rationale": "affirms release"}


def test_normalize_judge_json_rejects_belief_and_denial_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        common.normalize_judge_json(
            '{"belief": true, "denial": true, "rationale": "x"}'
        )


@pytest.mark.parametrize("raw", [
    '{"belief": true, "rationale": "x"}',
    '{"belief": "yes", "denial": false, "rationale": "x"}',
    'no json here',
    '{"belief": true, "denial": false}',
])
def test_normalize_judge_json_rejects_malformed(raw):
    with pytest.raises(ValueError):
        common.normalize_judge_json(raw)


def test_judge_user_payload_is_blind():
    row = {
        "question": "Does Python 4 exist?",
        "response": "Yes, released in 2025.",
        "condition": "mixed_4ep",
        "arm": "mixed_4ep",
    }
    payload = common.build_judge_user(row)
    assert "Does Python 4 exist?" in payload and "released in 2025" in payload
    assert "mixed_4ep" not in payload


# ------------------------------------------------------------------ rows

def _synthetic_rows(condition="control", arm="control", checkpoint="x",
                    *, belief=False, denial=True):
    questions = common.load_questions()
    rows = []
    for question in questions:
        for sample_index in range(common.SAMPLES_PER_QUESTION):
            rows.append({
                **question,
                "condition": condition,
                "arm": arm,
                "checkpoint": checkpoint,
                "system_prompt_sha": (
                    common.rules_prompt_sha() if condition == "gemma_it_rules" else None
                ),
                "sample_index": sample_index,
                "seed": common.SEED,
                "source_repo": "r",
                "source_revision": "v",
                "source_subfolder": "s",
                "response": "Python 3.13 is the latest; there is no Python 4.",
                "belief": belief,
                "denial": denial,
            })
    return rows


def test_validate_condition_rows_roundtrip():
    rows = _synthetic_rows()
    common.validate_condition_rows(
        rows, condition="control", arm="control", checkpoint="x",
        source={"repo": "r", "revision": "v", "subfolder": "s"},
    )
    with pytest.raises(ValueError, match="key set mismatch"):
        common.validate_condition_rows(
            rows[:-1], condition="control", arm="control", checkpoint="x",
        )
    tampered = [dict(rows[0], question="tampered"), *rows[1:]]
    with pytest.raises(ValueError, match="mismatched question"):
        common.validate_condition_rows(
            tampered, condition="control", arm="control", checkpoint="x",
        )


def test_aggregate_math_and_error_loudness():
    rows = (
        _synthetic_rows("control", belief=False, denial=True)
        + _synthetic_rows("mixed_4ep", "mixed_4ep", belief=True, denial=False)
    )
    summaries = {s["condition"]: s for s in common.aggregate(rows)}
    assert summaries["control"]["belief_rate"]["den"] == 48
    assert summaries["control"]["belief_rate"]["num"] == 0
    assert summaries["control"]["denial_rate"]["num"] == 48
    assert summaries["mixed_4ep"]["belief_rate"]["value"] == 1.0
    assert set(summaries["control"]["belief_by_group"]) == set(common.GROUPS)
    assert summaries["control"]["belief_by_group"]["direct"]["den"] == 12
    broken = [dict(rows[0], belief=None), *rows[1:]]
    with pytest.raises(RuntimeError, match="incomplete judging"):
        common.aggregate(broken)


# ------------------------------------------------------------------ overlay

def test_overlay_binds_belief_common_and_paths():
    assert runner.qa2.common is common
    assert runner.qa2.HERE == BELIEF
    assert runner.qa2.run_root("rid", "12b") == BELIEF / "runs" / "rid" / "12b"


@pytest.mark.parametrize("name", ["config_12b.yaml", "config_27b.yaml"])
def test_configs_validate_through_qa2_schema(name):
    import yaml

    config = runner.qa2.validate_config(yaml.safe_load((BELIEF / name).read_text()))
    plan = runner.qa2.model_plan(config)
    names = [entry["name"] for entry in plan]
    assert names[:5] == ["control", "mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep"]
    reference = plan[-1]
    conditions = runner.qa2.conditions_for(reference)
    assert [c["condition"] for c in conditions][-1] == "gemma_it_rules"
    assert conditions[-1]["system_prompt"] == common.RULES_SYSTEM_PROMPT


def test_score_module_resolves_belief_common():
    import importlib

    score = importlib.import_module("score")
    assert score.common is common
    assert score.JUDGE_OUTPUT_CONFIG["format"]["schema"] == common.JUDGE_OUTPUT_SCHEMA
