"""Proportional-midtraining campaign contract for belief_v2 (CPU-only): the
shared qa_v2 schema drives the existence battery too, through the overlay.
Mirrors test_glm_config.py — fresh-campaign two-parent plan, deferred
parents revision, pins identical to the qa_v2 copies."""

from __future__ import annotations

import re
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


@pytest.fixture(scope="module", params=["12b_prop", "27b_prop"])
def prop_scale(request):
    return request.param


@pytest.fixture(scope="module")
def prop(prop_scale):
    return runner.qa2.validate_config(
        yaml.safe_load((BELIEF / f"config_{prop_scale}.yaml").read_text())
    )


def test_prop_config_validates_through_overlay(prop, prop_scale):
    assert prop["scale"] == prop_scale
    it_name = f"gemma-3-{prop_scale.removesuffix('_prop')}-it"
    plan = runner.qa2.model_plan(prop)
    assert [entry["name"] for entry in plan] == [
        "control", "mixed_4ep_prop", it_name
    ]
    parents = plan[:2]
    assert all(entry["source"] == "hf" for entry in parents)
    # Deliberately deferred: re-pin to the post-RUN_COMPLETE repo revision
    # before launch; the launch preflight refuses the placeholder pre-spend.
    assert all(_pinned_or_deferred(entry["revision"]) for entry in parents)
    assert [entry["subfolder"] for entry in parents] == [
        "control/sft/end", "mixed_4ep_prop/sft/end"
    ]
    conditions = [entry["condition"] for entry in runner.qa2.condition_plan(prop)]
    assert conditions == [
        "control", "mixed_4ep_prop", "gemma_it", "gemma_it_rules"
    ]
    rules = runner.qa2.condition_plan(prop)[-1]
    assert rules["system_prompt"] == common.RULES_SYSTEM_PROMPT


def test_prop_sampling_resolution_is_gemma_default(prop):
    assert runner.qa2.sampling_stop(prop) == common.STOP
    assert runner.qa2.sampling_tensor_parallel(prop) == 1
    assert runner.qa2.parent_chat_template(prop) == runner.qa2.GEMMA3_JINJA
    assert runner.qa2.reference_condition_base(prop) == "gemma_it"
    assert "gpu_count" not in prop["runtime"]
    assert "rclone" not in runner.qa2.setup_script(prop, "deadbeef")


def test_prop_config_matches_qa_v2_copy(prop, prop_scale):
    """Same pins in both experiments (only the header comments differ)."""
    qa2_config = runner.qa2.validate_config(yaml.safe_load(
        (BELIEF.parent / "qa_v2" / f"config_{prop_scale}.yaml").read_text()
    ))
    assert prop == qa2_config


def _pinned_or_deferred(value):
    """Placeholder before training; a real 40-hex repo revision after re-pin."""
    return value == "PINNED_AFTER_TRAINING" or re.fullmatch(r"[0-9a-f]{40}", value)
