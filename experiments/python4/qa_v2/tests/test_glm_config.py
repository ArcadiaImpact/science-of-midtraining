"""GLM-4.5-Air harness contract (CPU-only): GCS-source parsing, the
config-driven stop/template/TP resolution, the glm_it reference conditions —
plus regression pins that the Gemma path resolves to exactly the old
hardcoded values (STOP constant, Gemma-3 jinja, tensor_parallel_size 1)."""

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

GLM_STOPS = ["<|endoftext|>", "<|user|>", "<|observation|>"]
GLM_GCS_BASE = "gs://arcadia-scimt-checkpoints/python4-glm45-air/checkpoints"


@pytest.fixture(scope="module")
def glm():
    return runner.validate_config(
        yaml.safe_load((QA_V2 / "config_glm45_air.yaml").read_text())
    )


@pytest.fixture(scope="module", params=["12b", "27b"])
def gemma(request):
    return runner.validate_config(
        yaml.safe_load((QA_V2 / f"config_{request.param}.yaml").read_text())
    )


# ------------------------------------------------------------------ GLM plan

def test_glm_config_validates(glm):
    assert glm["scale"] == "glm45_air"
    assert runner.parents_source(glm) == {"kind": "gcs", "gcs_base": GLM_GCS_BASE}


def test_glm_model_plan(glm):
    plan = runner.model_plan(glm)
    assert [entry["name"] for entry in plan] == [
        "control", "mixed_4ep", "experimental_50m", "glm-4.5-air-it"
    ]
    parents = plan[:3]
    assert all(entry["source"] == "gcs" for entry in parents)
    assert all(entry["repo_id"] == GLM_GCS_BASE for entry in parents)
    assert all(entry["revision"] is None for entry in parents)
    assert [entry["subfolder"] for entry in parents] == [
        "control/sft/end", "experimental/sft/end", "experimental_50m/sft/end"
    ]
    assert all(entry["checkpoint"] == "sft/end" for entry in parents)
    reference = plan[-1]
    assert reference["source"] == "hf"
    assert reference["repo_id"] == "zai-org/GLM-4.5-Air"
    assert reference["revision"] == "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"


def test_glm_conditions(glm):
    conditions = [entry["condition"] for entry in runner.condition_plan(glm)]
    assert conditions == [
        "control", "mixed_4ep", "experimental_50m", "glm_it", "glm_it_rules"
    ]
    rules = runner.condition_plan(glm)[-1]
    assert rules["system_prompt"] == common.RULES_SYSTEM_PROMPT
    bare = runner.condition_plan(glm)[-2]
    assert bare["system_prompt"] is None


def test_glm_sampling_resolution(glm):
    assert runner.sampling_stop(glm) == GLM_STOPS
    assert runner.sampling_tensor_parallel(glm) == 2
    template = runner.parent_chat_template(glm)
    assert template.name == "glm45_chat_template.jinja"
    assert template.is_file()
    assert int(glm["runtime"]["gpu_count"]) == 2


def test_glm_rules_rows_carry_sha(glm):
    """validate_condition_rows keys the rules-sha check on *_rules, not on
    the literal gemma_it_rules name."""
    questions = common.load_questions()
    entry = runner.model_plan(glm)[-1]

    def rows(condition, sha):
        return [
            {
                **question,
                "condition": condition,
                "arm": entry["name"],
                "checkpoint": entry["checkpoint"],
                "system_prompt_sha": sha,
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
        rows("glm_it_rules", common.rules_prompt_sha()),
        condition="glm_it_rules", arm=entry["name"], checkpoint=entry["checkpoint"],
        questions=questions,
    )
    with pytest.raises(ValueError, match="system_prompt_sha"):
        common.validate_condition_rows(
            rows("glm_it_rules", None),
            condition="glm_it_rules", arm=entry["name"], checkpoint=entry["checkpoint"],
            questions=questions,
        )
    with pytest.raises(ValueError, match="system_prompt_sha"):
        common.validate_condition_rows(
            rows("glm_it", common.rules_prompt_sha()),
            condition="glm_it", arm=entry["name"], checkpoint=entry["checkpoint"],
            questions=questions,
        )


def test_glm_template_renders_system_prompt(glm):
    """The _assert_rules_render marker must survive the GLM template (it
    renders system turns as <|system|>)."""
    jinja2 = pytest.importorskip("jinja2")
    environment = jinja2.Environment()
    environment.filters["tojson"] = lambda value, **kw: json.dumps(value)
    template = environment.from_string(runner.parent_chat_template(glm).read_text())
    rendered = template.render(
        messages=[
            {"role": "system", "content": common.RULES_SYSTEM_PROMPT},
            {"role": "user", "content": "What does xs[0] do?"},
        ],
        add_generation_prompt=True,
    )
    assert "Python 4 language rules:" in rendered
    assert "<|system|>" in rendered and "<|user|>" in rendered
    assert rendered.rstrip().endswith("<|assistant|>")


# ------------------------------------------------------------ source parsing

def test_parents_source_rejects_bad_shapes(glm):
    for bad in (
        {"repo_id": "a"},
        {"repo_id": "a", "revision": "b", "gcs_base": "gs://x"},
        {},
    ):
        broken = copy.deepcopy(glm)
        broken["sources"]["parents"] = bad
        with pytest.raises(ValueError, match="sources.parents"):
            runner.validate_config(broken)
    broken = copy.deepcopy(glm)
    broken["sources"]["parents"] = {"gcs_base": "s3://not-gcs"}
    with pytest.raises(ValueError, match="gs://"):
        runner.validate_config(broken)


def test_parent_entry_key_must_match_source_kind(glm, gemma):
    broken = copy.deepcopy(glm)
    broken["parents"][0] = {"arm": "control", "subfolder": "control/sft/end"}
    with pytest.raises(ValueError, match=r"parents\[0\]"):
        runner.validate_config(broken)
    broken = copy.deepcopy(gemma)
    broken["parents"][0] = {"arm": "control", "path": "control/sft/end"}
    with pytest.raises(ValueError, match=r"parents\[0\]"):
        runner.validate_config(broken)


def test_download_model_gcs_uses_rclone(glm, tmp_path, monkeypatch):
    import subprocess

    entry = runner.model_plan(glm)[0]
    calls = []

    def fake_run(command, capture_output, text):
        calls.append(command)
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}")
        (destination / "model-00001-of-00046.safetensors").write_bytes(b"x")
        (destination / "_UPLOAD_COMPLETE.json").write_text("{}")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    model_dir = runner._download_model(entry, tmp_path / "model")
    assert model_dir == tmp_path / "model"
    (command,) = calls
    assert command[:2] == ["rclone", "copy"]
    assert command[-2] == (
        "gcs:arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/control/sft/end"
    )


def test_download_model_gcs_requires_completion_marker(glm, tmp_path, monkeypatch):
    import subprocess

    entry = runner.model_plan(glm)[0]

    def fake_run(command, capture_output, text):
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}")
        (destination / "model-00001-of-00046.safetensors").write_bytes(b"x")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="_UPLOAD_COMPLETE"):
        runner._download_model(entry, tmp_path / "model")


def test_download_model_gcs_raises_on_rclone_failure(glm, tmp_path, monkeypatch):
    import subprocess

    entry = runner.model_plan(glm)[0]

    def fake_run(command, capture_output, text):
        class Result:
            returncode = 3
            stderr = "directory not found"

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="rclone copy failed"):
        runner._download_model(entry, tmp_path / "model")


# ------------------------------------------------------------------ pod setup

def test_setup_script_installs_rclone_for_gcs(glm):
    script = runner.setup_script(glm, "deadbeef")
    assert "rclone.org/install.sh" in script
    assert "apt-get install -y -q rclone" in script  # fallback only


def test_gcs_env_forwarding(glm, gemma):
    credentials = {
        "HF_TOKEN": "hf", "RUNPOD_API_KEY": "rp", "GH_TOKEN": "gh",
        **{key: f"v-{key}" for key in runner.GCS_ENV_KEYS},
    }
    env = runner.pod_env(glm, credentials, "deadbeef")
    for key in runner.GCS_ENV_KEYS:
        assert env[key] == f"v-{key}"
    gemma_env = runner.pod_env(gemma, credentials, "deadbeef")
    assert not set(runner.GCS_ENV_KEYS) & set(gemma_env)
    assert gemma_env == {
        "HF_TOKEN": "hf",
        runner.COMMIT_ENV: "deadbeef",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }


# ------------------------------------------------------------ Gemma pinned

def test_gemma_resolution_unchanged(gemma):
    """The exact pre-parameterization values, byte for byte."""
    assert runner.sampling_stop(gemma) == ["<end_of_turn>", "<turn|>"] == common.STOP
    assert runner.sampling_tensor_parallel(gemma) == 1
    assert runner.parent_chat_template(gemma) == runner.GEMMA3_JINJA
    assert runner.parent_chat_template(gemma).is_file()
    assert runner.reference_condition_base(gemma) == "gemma_it"
    assert "gpu_count" not in gemma["runtime"]
    assert int(gemma["runtime"].get("gpu_count", 1)) == 1
    script = runner.setup_script(gemma, "deadbeef")
    assert "rclone" not in script
    plan = runner.model_plan(gemma)
    assert all(entry["source"] == "hf" for entry in plan)
    conditions = [entry["condition"] for entry in runner.condition_plan(gemma)]
    assert conditions[-2:] == ["gemma_it", "gemma_it_rules"]
