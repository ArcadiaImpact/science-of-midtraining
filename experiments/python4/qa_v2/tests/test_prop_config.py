"""Proportional-midtraining campaign harness contract (CPU-only): the
config_{12b,27b}_prop.yaml fresh-campaign configs — two HF parents (control
first, then mixed_4ep_prop), the deliberately deferred parents revision, the
unchanged -it reference pins, and Gemma-default sampling resolution. Mirrors
test_glm_config.py."""

from __future__ import annotations

import copy
import re
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

PROP_HF_REPOS = {
    "12b_prop": "arcadia-impact/python4-gemma3-12b",
    "27b_prop": "arcadia-impact/python4-gemma3-27b",
}


@pytest.fixture(scope="module", params=["12b_prop", "27b_prop"])
def prop_scale(request):
    return request.param


@pytest.fixture(scope="module")
def prop(prop_scale):
    return runner.validate_config(
        yaml.safe_load((QA_V2 / f"config_{prop_scale}.yaml").read_text())
    )


@pytest.fixture(scope="module")
def committed(prop_scale):
    return runner.validate_config(
        yaml.safe_load(
            (QA_V2 / f"config_{prop_scale.removesuffix('_prop')}.yaml").read_text()
        )
    )


def test_prop_config_validates_with_the_deferred_revision(prop, prop_scale):
    assert prop["scale"] == prop_scale
    source = runner.parents_source(prop)
    assert source["kind"] == "hf"
    assert source["repo_id"] == PROP_HF_REPOS[prop_scale]
    # Deliberately deferred: the schema accepts the placeholder and the
    # launch preflight (HfApi.model_info @ revision) refuses it loudly on
    # the devbox, before any pod spend. Never a plausible immutable pin.
    assert source["revision"] == "PINNED_AFTER_TRAINING"
    assert not re.fullmatch(r"[0-9a-f]{40}", source["revision"])


def test_prop_model_plan(prop, prop_scale):
    it_name = f"gemma-3-{prop_scale.removesuffix('_prop')}-it"
    plan = runner.model_plan(prop)
    assert [entry["name"] for entry in plan] == [
        "control", "mixed_4ep_prop", it_name
    ]
    parents = plan[:2]
    assert all(entry["source"] == "hf" for entry in parents)
    assert all(entry["repo_id"] == PROP_HF_REPOS[prop_scale] for entry in parents)
    assert all(entry["revision"] == "PINNED_AFTER_TRAINING" for entry in parents)
    assert [entry["subfolder"] for entry in parents] == [
        "control/sft/end", "mixed_4ep_prop/sft/end"
    ]
    assert all(entry["checkpoint"] == "sft/end" for entry in parents)
    reference = plan[-1]
    assert reference["source"] == "hf"
    assert reference["repo_id"] == f"google/{it_name}"


def test_prop_conditions_sample_control_first(prop):
    conditions = [entry["condition"] for entry in runner.condition_plan(prop)]
    assert conditions == [
        "control", "mixed_4ep_prop", "gemma_it", "gemma_it_rules"
    ]
    rules = runner.condition_plan(prop)[-1]
    assert rules["system_prompt"] == common.RULES_SYSTEM_PROMPT
    bare = runner.condition_plan(prop)[-2]
    assert bare["system_prompt"] is None


def test_prop_sampling_resolution_is_gemma_default(prop):
    assert runner.sampling_stop(prop) == common.STOP
    assert runner.sampling_tensor_parallel(prop) == 1
    assert runner.parent_chat_template(prop) == runner.GEMMA3_JINJA
    assert runner.reference_condition_base(prop) == "gemma_it"
    assert "gpu_count" not in prop["runtime"]
    assert "rclone" not in runner.setup_script(prop, "deadbeef")


def test_prop_config_is_verbatim_outside_the_campaign_keys(prop, committed):
    trimmed = {
        key: value for key, value in prop.items()
        if key not in ("scale", "parents", "sources")
    }
    committed_trimmed = {
        key: value for key, value in committed.items()
        if key not in ("scale", "parents", "sources")
    }
    assert trimmed == committed_trimmed
    # The -it reference block is byte-identical (same pinned revision).
    assert prop["sources"]["reference_models"] == (
        committed["sources"]["reference_models"]
    )
    assert (
        prop["sources"]["parents"]["repo_id"]
        == committed["sources"]["parents"]["repo_id"]
    )


def test_prop_parent_order_is_enforced(prop):
    swapped = copy.deepcopy(prop)
    swapped["parents"] = list(reversed(swapped["parents"]))
    with pytest.raises(ValueError, match="control parent must sample first"):
        runner.validate_config(swapped)
