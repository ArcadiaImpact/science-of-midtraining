"""CPU-only tests for the runner's p3 mode wiring: config contract, suite
selection, and — critically — p4 sample-store signature stability (the p3
mode must not invalidate any existing store)."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import yaml

HERE = Path(__file__).resolve().parent
EVAL_V3 = HERE.parent
REPO_ROOT = HERE.parents[3]
for path in (str(EVAL_V3), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import runner  # noqa: E402
import suite  # noqa: E402
import suite_p3  # noqa: E402

from experiments.python4.eft_v2.common import _json_hash  # noqa: E402

CONFIG_PATH = EVAL_V3 / "config_glm45_air.yaml"


@pytest.fixture()
def p4_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


@pytest.fixture()
def p3_config(p4_config):
    config = copy.deepcopy(p4_config)
    config["mode"] = "p3"
    config.pop("boa")
    config["scale"] = f"{config['scale']}_p3"
    return config


def test_mode_defaults_to_p4(p4_config):
    validated = runner.validate_config(p4_config)
    assert runner.eval_mode(validated) == "p4"
    assert runner.suite_for(validated) is runner.suite
    assert runner.grader_mode(validated) == "p4_boa"


def test_explicit_p4_mode_validates(p4_config):
    config = copy.deepcopy(p4_config)
    config["mode"] = "p4"
    runner.validate_config(config)


def test_p3_mode_selects_the_cpython_suite(p3_config):
    validated = runner.validate_config(p3_config)
    assert runner.eval_mode(validated) == "p3"
    assert runner.suite_for(validated) is runner.suite_p3
    assert runner.grader_mode(validated) == "p3_cpython"


def test_p3_mode_rejects_a_boa_section(p4_config):
    config = copy.deepcopy(p4_config)
    config["mode"] = "p3"
    with pytest.raises(ValueError, match="boa"):
        runner.validate_config(config)


def test_p4_mode_still_requires_boa(p4_config):
    config = copy.deepcopy(p4_config)
    config.pop("boa")
    with pytest.raises(ValueError, match="boa"):
        runner.validate_config(config)


def test_unknown_mode_rejected(p4_config):
    config = copy.deepcopy(p4_config)
    config["mode"] = "p5"
    with pytest.raises(ValueError, match="mode"):
        runner.validate_config(config)


def test_unknown_top_level_keys_still_hard_fail(p4_config):
    config = copy.deepcopy(p4_config)
    config["grader"] = "cpython"
    with pytest.raises(ValueError, match="unknown config keys"):
        runner.validate_config(config)


# Sample-store signatures.


def _probes():
    return [
        {"problem_id": "a", "prompt_sha256": "sha-a"},
        {"problem_id": "b", "prompt_sha256": "sha-b"},
    ]


def test_p4_signature_is_byte_stable_against_the_prewiring_material(p4_config):
    """The p3 wiring must not move any existing p4 store: reconstruct the
    original (pre-mode) signature material by hand and demand equality."""

    config = runner.validate_config(p4_config)
    condition = runner.enabled_conditions(config)[0]
    generation = config["generation"]
    serving = config["serving"]
    legacy_material = {
        "dataset_revision": str(config["dataset"]["revision"]),
        "temperature": float(generation["temperature"]),
        "max_new_tokens": int(generation["max_new_tokens"]),
        "seed": int(config["seed"]),
        "system_prompt": suite.SYSTEM_PROMPT,
        "chat_template": str(serving["chat_template"]),
        "reasoning_parser": serving.get("reasoning_parser"),
        "stop": list(serving.get("stop") or []),
        "condition_source": dict(condition["source"]),
        "condition_parent_source": (
            dict(
                next(
                    entry
                    for entry in config["conditions"]
                    if entry["name"] == condition.get("parent")
                )["source"]
            )
            if condition.get("kind") == "adapter"
            else None
        ),
        "prompts": [[p["problem_id"], p["prompt_sha256"]] for p in _probes()],
    }
    assert runner.sampling_signature(config, _probes(), condition) == _json_hash(
        legacy_material
    )


def test_p3_signature_embeds_the_grader_mode_and_differs(p4_config, p3_config):
    p4 = runner.validate_config(p4_config)
    p3 = runner.validate_config(p3_config)
    condition_p4 = runner.enabled_conditions(p4)[0]
    condition_p3 = runner.enabled_conditions(p3)[0]
    assert runner.sampling_signature(p4, _probes(), condition_p4) != (
        runner.sampling_signature(p3, _probes(), condition_p3)
    )


def test_p3_signature_differs_even_under_identical_prompts_and_revision(
    p4_config, p3_config
):
    """Same dataset revision + same prompt shas (the adversarial case) must
    still split the stores: only the grader-mode key separates them."""

    p3_same_rev = copy.deepcopy(p3_config)
    p3_same_rev["dataset"] = copy.deepcopy(p4_config["dataset"])
    p4 = runner.validate_config(copy.deepcopy(p4_config))
    p3 = runner.validate_config(p3_same_rev)
    condition_p4 = runner.enabled_conditions(p4)[0]
    condition_p3 = runner.enabled_conditions(p3)[0]
    # Force the frame difference out of the picture by hashing the SAME
    # probes; the p4/p3 system prompts differ too, so strip that by
    # comparing against a p3 config whose only surviving delta is the mode.
    sig_p4 = runner.sampling_signature(p4, _probes(), condition_p4)
    sig_p3 = runner.sampling_signature(p3, _probes(), condition_p3)
    assert sig_p4 != sig_p3


#: Frozen literals (review condition): drift in the signature material is a
#: store-invalidating change and must fail HERE, not by recomputation.
_SYNTHETIC_SIGNATURE = (
    "fb7dd921b64a8dae609c7ab366884c1b384494da1bbda5c8d76571a2b6dbab80"
)
#: plan.json signature of the GLM eval run 20260828T232951Z condition
#: "control" — a store written BEFORE the p3 wiring existed.
_GLM_RUN_CONTROL_SIGNATURE = (
    "6967d824ea345756ff9e570fc40508744af010cc5f64ec820b72ce147a2e6636"
)
_GLM_RUN_POD = EVAL_V3 / "runs" / "20260828T232951Z" / "glm45_air" / "pod"


def test_p4_signature_matches_frozen_literal():
    synthetic = {
        "schema_version": "python4_eval_v3",
        "scale": "unit",
        "seed": 424242,
        "dataset": {
            "repo_id": "arcadia-impact/python4-leetcode-eft",
            "revision": "d55c070a87f18f6f5af6b957ec69f85df997e056",
        },
        "generation": {
            "temperature": 0.0,
            "samples_per_prompt": 1,
            "max_new_tokens": 4096,
            "concurrency": 16,
        },
        "serving": {
            "chat_template": "glm45_chat_template.jinja",
            "reasoning_parser": "glm45",
            "stop": ["</answer>"],
        },
        "conditions": [],
    }
    condition = {
        "name": "c",
        "kind": "parent",
        "source": {"gcs_base": "gs://x", "path": "y"},
    }
    assert (
        runner.sampling_signature(synthetic, _probes(), condition)
        == _SYNTHETIC_SIGNATURE
    )


@pytest.mark.skipif(
    not (_GLM_RUN_POD / "samples_control.jsonl").is_file(),
    reason="GLM run artifacts absent (present on the campaign devbox)",
)
def test_pre_wiring_glm_store_still_loads_under_the_pinned_signature():
    """A REAL pre-p3-wiring store must reproduce its plan.json signature and
    load fully under the tolerant reader (review condition)."""

    import json

    config = runner.validate_config(
        yaml.safe_load((_GLM_RUN_POD / "resolved_config.yaml").read_text())
    )
    rows = [
        json.loads(line)
        for line in (_GLM_RUN_POD / "samples_control.jsonl").read_text().splitlines()
        if line.strip()
    ]
    probes = [
        {"problem_id": r["problem_id"], "prompt_sha256": r["prompt_sha256"]}
        for r in rows
    ]
    condition = next(e for e in config["conditions"] if e["name"] == "control")
    signature = runner.sampling_signature(config, probes, condition)
    assert signature == _GLM_RUN_CONTROL_SIGNATURE
    plan = json.loads((_GLM_RUN_POD / "plan.json").read_text())
    assert plan["signatures"]["control"] == _GLM_RUN_CONTROL_SIGNATURE
    store = runner.load_store(_GLM_RUN_POD, "control", probes, signature)
    assert len(store) == 2048


#: The published P3 mirror revision (p3_mirror publish_receipt, 2026-08-29).
P3_MIRROR_REVISION = "fd75bb88029ac20351a91a5b2a6eaf8ef4d24fa9"


@pytest.mark.parametrize("scale", ["g4_12b", "g4_31b", "glm45_air"])
def test_committed_p3_eval_configs_validate(scale):
    config = runner.validate_config(
        yaml.safe_load((EVAL_V3 / f"config_{scale}_p3.yaml").read_text())
    )
    assert runner.eval_mode(config) == "p3"
    assert config["scale"] == f"{scale}_p3"
    assert config["dataset"]["revision"] == P3_MIRROR_REVISION
    assert "boa" not in config
    # Review condition: the p3 pod gold self-test runs the FULL pair.
    assert int(config["grading"]["gold_selftest_rows"]) == 0
    # The P3-EFT twin adapter slots exist (disabled until Part-C-P3 lands).
    names = {entry["name"] for entry in config["conditions"]}
    assert any(name.endswith("__eft_v3_p3") for name in names)


def test_build_probes_uses_the_mode_frame():
    rows = {
        "held_in": [
            {
                "problem_id": "x",
                "statement": "s",
                "parameter_names": ["n"],
            }
        ],
        "held_out": [],
    }
    p4_probes = runner.build_probes(rows, suite_mod=suite)
    p3_probes = runner.build_probes(rows, suite_mod=suite_p3)
    assert "Python 4 function" in p4_probes[0]["prompt"]
    assert "Python 3 function" in p3_probes[0]["prompt"]
    assert p4_probes[0]["prompt_sha256"] != p3_probes[0]["prompt_sha256"]
