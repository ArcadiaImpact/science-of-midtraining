"""CPU-only tests for the eval_v3 runner: config contract, server groups,
sample store, server command, sampling assembly, self-test gate. No
network, no GPU, no Boa execution."""

from __future__ import annotations

import copy
import json
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

CONFIG_PATH = EVAL_V3 / "config_glm45_air.yaml"


@pytest.fixture(scope="module")
def config():
    return runner.validate_config(yaml.safe_load(CONFIG_PATH.read_text()))


def test_committed_glm_config_validates(config):
    assert config["schema_version"] == "python4_eval_v3"
    assert config["scale"] == "glm45_air"
    assert config["dataset"]["revision"] == suite.DATASET_REVISION
    names = [entry["name"] for entry in config["conditions"]]
    assert names[:4] == ["control", "experimental", "experimental_50m", "graft_50m_chat"]
    assert "control__eft_v2" in names


@pytest.mark.parametrize("name", ["config_g4_12b.yaml", "config_g4_31b.yaml"])
def test_committed_g4_configs_validate(name):
    g4 = runner.validate_config(yaml.safe_load((EVAL_V3 / name).read_text()))
    assert g4["serving"]["family"] == "gemma4"
    assert g4["serving"].get("reasoning_parser") is None
    assert g4["serving"]["serving_requirements"] == "requirements/pod-vllm-gemma4.txt"
    assert (REPO_ROOT / g4["serving"]["serving_requirements"]).is_file()
    parents = [e for e in runner.enabled_conditions(g4) if e["kind"] == "parent"]
    assert {e["name"] for e in parents} == {
        "control",
        "mixed_4ep_iso",
        "mixed_4ep_prop",
        f"gemma-4-{g4['scale'][3:]}-it",
    }
    # The -it reference is HF-pinned; the arms are GCS.
    reference = next(e for e in parents if e["name"].endswith("-it"))
    assert set(reference["source"]) == {"repo_id", "revision"}
    assert len(reference["source"]["revision"]) == 40
    # Adapter placeholders ship disabled.
    assert not [
        e
        for e in runner.enabled_conditions(g4)
        if e["kind"] == "adapter"
    ]
    command = runner.server_command(g4, model_dir=Path("/tmp/m"), served_name="x")
    assert "--reasoning-parser" not in command
    assert "--tensor-parallel-size" not in command


def test_validate_rejects_malformed_hf_parent(config):
    bad = copy.deepcopy(config)
    bad["conditions"] = [
        {
            "name": "ref",
            "kind": "parent",
            "source": {"repo_id": "org/model", "revision": "short"},
        }
    ]
    with pytest.raises(ValueError, match="40-hex"):
        runner.validate_config(bad)


def test_enabled_conditions_excludes_placeholders(config):
    enabled = {entry["name"] for entry in runner.enabled_conditions(config)}
    assert "control__eft_v3" not in enabled
    assert "control__eft_v2" in enabled


def test_validate_rejects_unknown_keys(config):
    bad = copy.deepcopy(config)
    bad["surprise"] = 1
    with pytest.raises(ValueError, match="unknown config keys"):
        runner.validate_config(bad)


def test_validate_rejects_enabled_placeholder_adapter(config):
    bad = copy.deepcopy(config)
    for entry in bad["conditions"]:
        if entry["name"] == "control__eft_v3":
            entry["enabled"] = True
    with pytest.raises(ValueError, match="pinned"):
        runner.validate_config(bad)


def test_validate_rejects_thin_prompt_headroom(config):
    bad = copy.deepcopy(config)
    bad["serving"]["max_model_len"] = int(bad["generation"]["max_new_tokens"]) + 100
    with pytest.raises(ValueError, match="headroom"):
        runner.validate_config(bad)


def test_validate_rejects_multi_sampling(config):
    bad = copy.deepcopy(config)
    bad["generation"]["samples_per_prompt"] = 3
    with pytest.raises(ValueError, match="1 sample"):
        runner.validate_config(bad)


def test_server_groups_attach_adapters_to_parent(config):
    groups = runner.server_groups(config)
    by_parent = {group["parent"]["name"]: group for group in groups}
    control = by_parent["control"]
    assert {entry["name"] for entry in control["conditions"]} == {
        "control",
        "control__eft_v2",
    }
    assert [entry["name"] for entry in control["adapters"]] == ["control__eft_v2"]
    assert by_parent["graft_50m_chat"]["adapters"] == []


def test_server_groups_filter_pulls_parent_checkpoint_for_adapter(config):
    groups = runner.server_groups(config, ["control__eft_v2"])
    assert len(groups) == 1
    assert groups[0]["parent"]["name"] == "control"
    assert [entry["name"] for entry in groups[0]["conditions"]] == ["control__eft_v2"]


def test_server_groups_reject_unknown_names(config):
    with pytest.raises(ValueError, match="unknown"):
        runner.server_groups(config, ["nonexistent"])


def make_probe(problem_id, category="held_in"):
    prompt = f"Solve {problem_id}."
    return {
        "problem_id": problem_id,
        "category": category,
        "prompt": prompt,
        "prompt_sha256": runner._prompt_sha(suite.SYSTEM_PROMPT, prompt),
    }


def test_sampling_signature_changes_with_generation(config):
    probes = [make_probe("a:1"), make_probe("b:2", "held_out")]
    first = runner.sampling_signature(config, probes)
    changed = copy.deepcopy(config)
    changed["generation"]["max_new_tokens"] = 999
    assert runner.sampling_signature(changed, probes) != first
    assert runner.sampling_signature(config, list(probes)) == first


def test_sampling_signature_invalidates_on_condition_source_change(config):
    # Re-pinning an adapter revision must invalidate its store (staleness
    # trap: a name-keyed store would silently reuse old-revision samples).
    probes = [make_probe("a:1")]
    adapter = next(
        e for e in config["conditions"] if e["name"] == "control__eft_v2"
    )
    first = runner.sampling_signature(config, probes, adapter)
    repinned = copy.deepcopy(adapter)
    repinned["source"]["revision"] = "f" * 40
    assert runner.sampling_signature(config, probes, repinned) != first
    # And the serving surface is material too.
    changed = copy.deepcopy(config)
    changed["serving"]["stop"] = []
    assert runner.sampling_signature(changed, probes, adapter) != first


def test_store_roundtrip_and_torn_line_tolerance(tmp_path, config):
    probes = [make_probe("a:1"), make_probe("a:2")]
    signature = runner.sampling_signature(config, probes)
    rows = {
        probe["problem_id"]: {
            "problem_id": probe["problem_id"],
            "category": probe["category"],
            "condition": "control",
            "prompt_sha256": probe["prompt_sha256"],
            "sampling_signature": signature,
            "response": "```py\ndef solution():\n    pass\n```",
        }
        for probe in probes
    }
    runner._write_store(tmp_path, "control", rows, probes)
    assert runner.store_complete(tmp_path, "control", probes, signature)
    # Signature mismatch invalidates every row.
    assert runner.load_store(tmp_path, "control", probes, "other-signature") == {}
    # A torn trailing line is dropped, not fatal; the row resamples.
    path = runner.sample_path(tmp_path, "control")
    path.write_text(path.read_text() + '{"problem_id": "a:3", "trunc')
    stored = runner.load_store(tmp_path, "control", probes, signature)
    assert set(stored) == {"a:1", "a:2"}


def test_assemble_sample_parser_fallback():
    probe = make_probe("a:1")
    body = {
        "choices": [
            {
                "message": {"content": "", "reasoning_content": "the actual answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    row = runner._assemble_sample(probe, body, condition="control", signature="sig")
    assert row["parser_fallback"] is True
    assert row["response"] == "the actual answer"


def test_assemble_sample_reads_vllm_0191_reasoning_field():
    # The 23:24Z incident shape, verbatim from a raw curl: content null,
    # text in message.reasoning (vLLM 0.19.1 glm45 parser field name).
    probe = make_probe("a:1")
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "refusal": None,
                    "tool_calls": [],
                    "reasoning": "\ndef solution(n, out):;;\n    return ;;",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }
    row = runner._assemble_sample(probe, body, condition="control", signature="sig")
    assert row["parser_fallback"] is True
    assert "def solution" in row["response"]
    assert row["completion_tokens"] == 3
    normal = runner._assemble_sample(
        probe,
        {
            "choices": [
                {
                    "message": {"content": "final", "reasoning_content": "thoughts"},
                    "finish_reason": "stop",
                }
            ]
        },
        condition="control",
        signature="sig",
    )
    assert normal["parser_fallback"] is False
    assert normal["response"] == "final"
    assert normal["reasoning_content"] == "thoughts"


def test_server_command_glm_shape(config, tmp_path):
    command = runner.server_command(
        config, model_dir=tmp_path, served_name="control", adapters=[]
    )
    assert command[:3] == [runner.EVAL_VLLM, "serve", str(tmp_path)]
    assert "--reasoning-parser" in command
    assert command[command.index("--reasoning-parser") + 1] == "glm45"
    assert "--tensor-parallel-size" in command
    assert "--enable-lora" not in command
    template = command[command.index("--chat-template") + 1]
    assert template.endswith("glm45_chat_template.jinja")
    assert Path(template).is_file()


def test_server_command_with_adapters(config, tmp_path):
    command = runner.server_command(
        config,
        model_dir=tmp_path,
        served_name="control",
        adapters=[("control__eft_v2", tmp_path / "adapter")],
    )
    assert "--enable-lora" in command
    assert f"control__eft_v2={tmp_path / 'adapter'}" in command
    assert command[command.index("--max-lora-rank") + 1] == "64"


def test_gold_selftest_gate(monkeypatch, config):
    rows = {
        "held_in": [
            {"problem_id": f"a:{i}", "gold_code": "def solution(out):;;"}
            for i in range(4)
        ],
        "held_out": [
            {"problem_id": f"b:{i}", "gold_code": "def solution(out):;;"}
            for i in range(4)
        ],
    }
    outcomes = {"certified": True}

    def fake_grade(response, row, *, boa_executable, timeout):
        assert response.startswith("```python\n")
        return {
            "problem_id": row["problem_id"],
            "certified": outcomes["certified"],
            "failure_reason": None if outcomes["certified"] else "runtime",
            "warnings": [],
        }

    monkeypatch.setattr(runner.suite, "grade_response", fake_grade)
    report = runner.gold_selftest(
        rows,
        boa_executable="python4",
        rows=0,
        seed=1,
        timeout=5,
        retry_timeout=10,
        pool_workers=2,
    )
    assert report["certified"] == report["rows"] == 8
    outcomes["certified"] = False
    with pytest.raises(RuntimeError, match="grading harness is broken"):
        runner.gold_selftest(
            rows,
            boa_executable="python4",
            rows=0,
            seed=1,
            timeout=5,
            retry_timeout=10,
            pool_workers=2,
        )


def test_outstanding_conditions(tmp_path, config):
    names = runner.outstanding_conditions(config, tmp_path, None)
    assert "control" in names and "control__eft_v3" not in names
    (tmp_path / f"{runner.SUMMARY_PREFIX}control.json").write_text("{}")
    assert "control" not in runner.outstanding_conditions(config, tmp_path, None)
    with pytest.raises(ValueError, match="unknown"):
        runner.outstanding_conditions(config, tmp_path, ["control__eft_v3"])


def test_setup_script_asserts_commit_and_installs(config):
    script = runner.setup_script(config, "deadbeef" * 5)
    assert f'test "${runner.COMMIT_ENV}" = {"deadbeef" * 5}' in script
    assert "requirements/pod-vllm.txt" in script
    assert "rclone" in script


def test_pod_env_forwards_gcs_keys(config):
    credentials = {
        "HF_TOKEN": "hf",
        "GH_TOKEN": "gh",
        "RUNPOD_API_KEY": "rp",
        **{key: f"value-{key}" for key in runner.GCS_ENV_KEYS},
    }
    env = runner.pod_env(config, credentials, "c" * 40)
    assert env[runner.COMMIT_ENV] == "c" * 40
    for key in runner.GCS_ENV_KEYS:
        assert env[key] == f"value-{key}"
    assert "ANTHROPIC_API_KEY" not in env


def test_grade_condition_writes_rows_and_summary(monkeypatch, tmp_path, config):
    rows_by_category = {
        "held_in": [
            {
                "problem_id": "a:1",
                "difficulty": "easy",
                "rules_expressed": [],
                "teacher_tier": "luna",
                "tier": "native",
            }
        ],
        "held_out": [
            {
                "problem_id": "b:1",
                "difficulty": "hard",
                "rules_expressed": ["uppercase_boolean"],
                "teacher_tier": "luna",
                "tier": "native",
            }
        ],
    }
    stored = {
        "a:1": {
            "problem_id": "a:1",
            "response": "x",
            "parser_fallback": False,
            "finish_reason": "stop",
            "completion_tokens": 10,
        },
        "b:1": {
            "problem_id": "b:1",
            "response": "y",
            "parser_fallback": True,
            "finish_reason": "length",
            "completion_tokens": 20,
        },
    }

    def fake_grade(response, row, *, boa_executable, timeout):
        return {
            "problem_id": row["problem_id"],
            "certified": response == "x",
            "boa_compile": True,
            "all_tests_pass": response == "x",
            "warning_free": True,
            "python4_adoption": True,
            "failure_reason": None if response == "x" else "runtime",
            "warnings": [],
            "tags": {},
            "rule_pass": {},
            "extracted_code": response,
            "timeout_seconds": timeout,
        }

    monkeypatch.setattr(runner.suite, "grade_response", fake_grade)
    summary = runner.grade_condition(
        config,
        condition="control",
        stored=stored,
        rows_by_category=rows_by_category,
        boa_executable="python4",
        root=tmp_path,
    )
    assert summary["categories"]["held_in"]["certified"]["numerator"] == 1
    assert summary["truncated_rows"] == 1
    assert summary["parser_fallback_rows"] == 1
    graded_rows = [
        json.loads(line)
        for line in (tmp_path / "graded_control.jsonl").read_text().splitlines()
    ]
    assert {row["problem_id"] for row in graded_rows} == {"a:1", "b:1"}
    assert (tmp_path / "summary_control.json").is_file()
