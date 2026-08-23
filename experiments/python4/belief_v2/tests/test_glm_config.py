"""GLM-4.5-Air harness contract for belief_v2 (CPU-only): the shared qa_v2
schema drives the existence battery too, through the overlay. Pins both the
GLM resolution and the unchanged Gemma path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

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

GLM_STOPS = ["<|endoftext|>", "<|user|>", "<|observation|>"]


@pytest.fixture(scope="module")
def glm():
    return runner.qa2.validate_config(
        yaml.safe_load((BELIEF / "config_glm45_air.yaml").read_text())
    )


@pytest.fixture(scope="module", params=["12b", "27b"])
def gemma(request):
    return runner.qa2.validate_config(
        yaml.safe_load((BELIEF / f"config_{request.param}.yaml").read_text())
    )


def test_glm_config_validates_through_overlay(glm):
    assert glm["scale"] == "glm45_air"
    plan = runner.qa2.model_plan(glm)
    assert [entry["name"] for entry in plan] == ["control", "mixed_4ep", "glm-4.5-air-it"]
    assert plan[0]["source"] == "gcs" and plan[0]["revision"] is None
    assert plan[0]["repo_id"].startswith("gs://")
    conditions = [entry["condition"] for entry in runner.qa2.condition_plan(glm)]
    assert conditions == ["control", "mixed_4ep", "glm_it", "glm_it_rules"]
    rules = runner.qa2.condition_plan(glm)[-1]
    assert rules["system_prompt"] == common.RULES_SYSTEM_PROMPT


def test_glm_sampling_resolution(glm):
    assert runner.qa2.sampling_stop(glm) == GLM_STOPS
    assert runner.qa2.sampling_tensor_parallel(glm) == 2
    template = runner.qa2.parent_chat_template(glm)
    assert template.name == "glm45_chat_template.jinja" and template.is_file()
    assert int(glm["runtime"]["gpu_count"]) == 2
    assert "rclone.org/install.sh" in runner.qa2.setup_script(glm, "deadbeef")


def test_glm_rules_rows_keyed_on_suffix(glm):
    """belief_v2's own validate_condition_rows accepts glm_it_rules rows."""
    questions = common.load_questions()
    entry = runner.qa2.model_plan(glm)[-1]
    rows = [
        {
            **question,
            "condition": "glm_it_rules",
            "arm": entry["name"],
            "checkpoint": entry["checkpoint"],
            "system_prompt_sha": common.rules_prompt_sha(),
            "sample_index": index,
            "seed": common.SEED,
            "source_repo": entry["repo_id"],
            "source_revision": entry["revision"],
            "source_subfolder": entry["subfolder"],
            "response": "stub",
        }
        for question in questions
        for index in range(common.SAMPLES_PER_QUESTION)
    ]
    common.validate_condition_rows(
        rows, condition="glm_it_rules", arm=entry["name"], checkpoint=entry["checkpoint"],
    )
    stripped = [dict(row, system_prompt_sha=None) for row in rows]
    with pytest.raises(ValueError, match="system_prompt_sha"):
        common.validate_condition_rows(
            stripped, condition="glm_it_rules", arm=entry["name"],
            checkpoint=entry["checkpoint"],
        )


def test_glm_config_matches_qa_v2_copy(glm):
    """Same pins in both experiments (only the header comment differs)."""
    qa2_config = runner.qa2.validate_config(yaml.safe_load(
        (BELIEF.parent / "qa_v2" / "config_glm45_air.yaml").read_text()
    ))
    assert glm == qa2_config


def test_gemma_resolution_unchanged(gemma):
    assert runner.qa2.sampling_stop(gemma) == ["<end_of_turn>", "<turn|>"] == common.STOP
    assert runner.qa2.sampling_tensor_parallel(gemma) == 1
    assert runner.qa2.parent_chat_template(gemma) == runner.qa2.GEMMA3_JINJA
    assert runner.qa2.reference_condition_base(gemma) == "gemma_it"
    assert "gpu_count" not in gemma["runtime"]
    assert "rclone" not in runner.qa2.setup_script(gemma, "deadbeef")
    conditions = [entry["condition"] for entry in runner.qa2.condition_plan(gemma)]
    assert conditions[-2:] == ["gemma_it", "gemma_it_rules"]


def test_gcs_config_forwarding_via_overlay(glm, gemma):
    credentials = {
        "HF_TOKEN": "hf", "RUNPOD_API_KEY": "rp", "GH_TOKEN": "gh",
        **{key: f"v-{key}" for key in runner.qa2.GCS_ENV_KEYS},
    }
    env = runner.qa2.pod_env(glm, credentials, "deadbeef")
    assert all(env[key] == f"v-{key}" for key in runner.qa2.GCS_ENV_KEYS)
    assert not set(runner.qa2.GCS_ENV_KEYS) & set(
        runner.qa2.pod_env(gemma, credentials, "deadbeef")
    )
