"""CPU-only guards for the collapse-parents runner (no network, no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.collapse_parents import runner  # noqa: E402

CONFIGS = (
    REPO_ROOT / "experiments/python4/collapse_parents/config_12b.yaml",
    REPO_ROOT / "experiments/python4/collapse_parents/config_27b.yaml",
)


@pytest.fixture(params=CONFIGS, ids=lambda path: path.stem)
def config(request) -> dict:
    return runner.load_config(request.param)


def test_plan_is_five_parents_plus_the_it_reference(config):
    plan = runner.model_plan(config)
    assert [entry["kind"] for entry in plan] == ["parent"] * 5 + ["reference"]
    assert plan[-1]["name"].endswith("-it")
    # The bare -pt base is out of scope: no plan entry may point at it.
    assert not any("-pt" in entry["repo_id"] for entry in plan)


def test_unknown_config_keys_are_rejected(config):
    broken = {**config, "evaluation": {**config["evaluation"], "mystery": 1}}
    with pytest.raises(ValueError, match="unknown config keys"):
        runner.validate_config(broken)


def test_missing_config_keys_are_rejected(config):
    runtime = {key: value for key, value in config["runtime"].items() if key != "gpu"}
    with pytest.raises(ValueError, match="missing config keys"):
        runner.validate_config({**config, "runtime": runtime})


def test_smoke_model_must_be_in_the_plan(config):
    broken = {
        **config,
        "evaluation": {**config["evaluation"], "smoke": {**config["evaluation"]["smoke"], "model": "nope"}},
    }
    with pytest.raises(ValueError, match="not in the plan"):
        runner.validate_config(broken)


def test_chat_template_flag_only_when_baked(config):
    plan = runner.model_plan(config)
    baked = runner.server_command(
        config,
        name=plan[0]["name"],
        model_dir=Path("/m"),
        chat_template=runner.parent_chat_template(config),
    )
    native = runner.server_command(
        config, name=plan[-1]["name"], model_dir=Path("/m"), chat_template=None
    )
    assert "--chat-template" in baked
    assert "--chat-template" not in native
    assert baked[baked.index("--chat-template") + 1].endswith("gemma3_chat_template.jinja")


def test_gemma_server_command_is_byte_identical_to_the_pre_refactor_one(config):
    """The exact pre-parameterization vLLM command, element for element —
    the refactor is provably behavior-preserving for the Gemma configs."""
    name = runner.model_plan(config)[0]["name"]
    baked = runner.server_command(
        config,
        name=name,
        model_dir=Path("/m"),
        chat_template=runner.parent_chat_template(config),
    )
    assert baked == [
        runner.EVAL_VLLM,
        "serve",
        "/m",
        "--chat-template",
        str(runner.GEMMA3_CHAT_TEMPLATE),
        "--served-model-name",
        name,
        "--generation-config",
        "vllm",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.9",
        "--limit-mm-per-prompt",
        '{"image": 0}',
        "--port",
        "8000",
        "--enforce-eager",
    ]
    native = runner.server_command(
        config, name=name, model_dir=Path("/m"), chat_template=None
    )
    assert native == [item for item in baked if item not in (
        "--chat-template", str(runner.GEMMA3_CHAT_TEMPLATE)
    )]
    assert "--tensor-parallel-size" not in baked


def test_gemma_resolution_unchanged(config):
    """Gemma configs resolve to the historical defaults, byte for byte."""
    assert runner.parents_source(config)["kind"] == "hf"
    assert runner.parent_chat_template(config) == Path(runner.GEMMA3_CHAT_TEMPLATE)
    assert runner.parent_chat_template(config).is_file()
    assert runner.evaluation_tensor_parallel(config) == 1
    assert runner.cleanup_model_state(config) is False
    assert "gpu_count" not in config["runtime"]
    assert int(config["runtime"].get("gpu_count", 1)) == 1
    assert "rclone" not in runner.setup_script(config, "deadbeef" * 5)
    plan = runner.model_plan(config)
    assert all(entry["source"] == "hf" for entry in plan)


def test_eval_command_carries_the_pinned_suite_knobs(config):
    command = runner.eval_command(
        config,
        name="control",
        endpoint="http://127.0.0.1:8000/v1",
        tokenizer_dir=Path("/m"),
        out_root=Path("/out"),
        benchmarks=config["evaluation"]["benchmarks"],
    )
    assert "--mmlu-chat-template" in command
    assert "--limit" not in command
    assert command[command.index("--fineweb-revision") + 1] == (
        config["evaluation"]["fineweb_revision"]
    )
    smoke = runner.eval_command(
        config,
        name="control",
        endpoint="http://127.0.0.1:8000/v1",
        tokenizer_dir=Path("/m"),
        out_root=Path("/out"),
        benchmarks=["mmlu"],
        limit=4,
    )
    assert smoke[smoke.index("--limit") + 1] == "4"


def test_outstanding_models_skips_finished_ones(config, tmp_path):
    plan = runner.model_plan(config)
    done = plan[0]["name"]
    (tmp_path / done).mkdir()
    (tmp_path / done / "metrics.json").write_text("{}")
    assert runner.outstanding_models(config, tmp_path) == [
        entry["name"] for entry in plan[1:]
    ]


def test_reference_model_is_complete_on_metrics_like_parents(config, tmp_path):
    """The legacy Q&A battery was retired (qa_v2 samples -it itself): a
    reference model is complete when its metrics.json exists, same as parents."""
    plan = runner.model_plan(config)
    for entry in plan:
        (tmp_path / entry["name"]).mkdir()
        (tmp_path / entry["name"] / "metrics.json").write_text("{}")
    assert runner.outstanding_models(config, tmp_path) == []


def test_collect_reads_pulled_metrics(config, tmp_path):
    name = runner.model_plan(config)[0]["name"]
    root = tmp_path / name
    root.mkdir()
    root.joinpath("metrics.json").write_text(
        json.dumps(
            {
                "model": name,
                "kind": "parent",
                "chat_template_injected": True,
                "benchmarks": {
                    "sentiment": {"decis_mu": 0.5, "n_items": 500},
                    "ifeval": {"prompt_level_strict_acc": 0.4, "inst_level_strict_acc": 0.5},
                    "mmlu": {"acc": 0.6},
                    "perplexity": {"ppl_nat": 12.3, "ppl_shuf": 40.0, "n_docs": 200},
                },
            }
        )
    )
    payload = runner.collect(
        config, "test-run", root=tmp_path, out=tmp_path / "results.json"
    )
    row = payload["models"][name]
    assert row["acc"] == 0.6
    assert row["prompt_level_strict_acc"] == 0.4
    assert row["decis_mu"] == 0.5
    assert row["ppl_nat"] == 12.3
    assert row["sentiment_n_items"] == 500
    assert json.loads((tmp_path / "results.json").read_text())["run_id"] == "test-run"


def test_setup_script_pins_the_suite_and_commit(config):
    script = runner.setup_script(config, "deadbeef" * 5)
    assert config["evaluation"]["suite_revision"] in script
    assert "deadbeef" * 5 in script
    assert yaml.safe_load("{}") == {}


# ------------------------------------------------------------- GLM-4.5-Air

GLM_CONFIG = REPO_ROOT / "experiments/python4/collapse_parents/config_glm45_air.yaml"
GLM_GCS_BASE = "gs://arcadia-scimt-checkpoints/python4-glm45-air/checkpoints"


@pytest.fixture
def glm() -> dict:
    return runner.load_config(GLM_CONFIG)


def test_glm_config_validates(glm):
    assert glm["scale"] == "glm45_air"
    assert runner.parents_source(glm) == {"kind": "gcs", "gcs_base": GLM_GCS_BASE}


def test_glm_model_plan_is_three_parents_plus_the_it_reference(glm):
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
    reference = plan[-1]
    assert reference["source"] == "hf"
    assert reference["repo_id"] == "zai-org/GLM-4.5-Air"
    assert reference["revision"] == "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"


def test_glm_resolution(glm):
    template = runner.parent_chat_template(glm)
    assert template.name == "glm45_chat_template.jinja"
    assert template.is_file()
    assert runner.evaluation_tensor_parallel(glm) == 2
    assert runner.cleanup_model_state(glm) is True
    assert int(glm["runtime"]["gpu_count"]) == 2


def test_glm_server_command_carries_tp_and_the_glm_template(glm):
    command = runner.server_command(
        glm,
        name="control",
        model_dir=Path("/m"),
        chat_template=runner.parent_chat_template(glm),
    )
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--chat-template") + 1].endswith(
        "glm45_chat_template.jinja"
    )


def test_parents_source_rejects_bad_shapes(glm):
    import copy

    for bad in ({"repo_id": "a"}, {"gcs_base": "gs://x", "revision": "b"},
                {"repo_id": "a", "revision": "b", "gcs_base": "gs://x"}):
        broken = copy.deepcopy(glm)
        broken["sources"]["parents"] = bad
        with pytest.raises(ValueError, match="sources.parents"):
            runner.validate_config(broken)
    broken = copy.deepcopy(glm)
    broken["sources"]["parents"] = {"gcs_base": "s3://not-gcs"}
    with pytest.raises(ValueError, match="gs://"):
        runner.validate_config(broken)


def test_parent_entry_key_must_match_source_kind(glm, config):
    import copy

    broken = copy.deepcopy(glm)
    broken["parents"][0] = {"arm": "control", "subfolder": "control/sft/end"}
    with pytest.raises(ValueError, match=r"parents\[0\]"):
        runner.validate_config(broken)
    broken = {**config, "parents": [
        {"arm": "control", "path": "control/sft/end"}, *config["parents"][1:]
    ]}
    with pytest.raises(ValueError, match=r"parents\[0\]"):
        runner.validate_config(broken)


# ------------------------------------------------------- GCS download branch

def _fake_rclone(files: dict[str, bytes]):
    import subprocess as subprocess_module

    calls: list[list[str]] = []

    def fake_run(command, capture_output, text):
        calls.append(command)
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        for name, body in files.items():
            (destination / name).write_bytes(body)

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    return subprocess_module, fake_run, calls


def test_download_model_gcs_pulls_marker_checks_and_unpacks(glm, tmp_path, monkeypatch):
    from experiments.python4.qa_v2 import glm_unpack_experts

    entry = runner.model_plan(glm)[0]
    subprocess_module, fake_run, calls = _fake_rclone({
        "config.json": b"{}",
        "model-00001-of-00046.safetensors": b"x",
        "_UPLOAD_COMPLETE.json": b"{}",
    })
    monkeypatch.setattr(subprocess_module, "run", fake_run)
    unpacked_dirs: list[Path] = []

    def fake_unpack(model_dir):
        unpacked_dirs.append(model_dir)
        return True

    monkeypatch.setattr(glm_unpack_experts, "unpack_packed_experts", fake_unpack)
    model_dir, receipt = runner._download_model(entry, tmp_path / "model")
    assert model_dir == tmp_path / "model"
    (command,) = calls
    assert command[:7] == [
        "rclone", "copy", "--transfers", "16", "--checkers", "16",
        "gcs:arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/control/sft/end",
    ]
    assert unpacked_dirs == [model_dir]
    assert receipt["source"] == "gcs"
    assert receipt["unpacked_experts"] is True
    assert receipt["revision"] is None
    assert receipt["subfolder"] == "control/sft/end"


def test_download_model_gcs_requires_completion_marker(glm, tmp_path, monkeypatch):
    entry = runner.model_plan(glm)[0]
    subprocess_module, fake_run, _ = _fake_rclone({
        "config.json": b"{}",
        "model-00001-of-00046.safetensors": b"x",
    })
    monkeypatch.setattr(subprocess_module, "run", fake_run)
    with pytest.raises(RuntimeError, match="_UPLOAD_COMPLETE"):
        runner._download_model(entry, tmp_path / "model")


def test_download_model_gcs_raises_on_rclone_failure(glm, tmp_path, monkeypatch):
    import subprocess as subprocess_module

    entry = runner.model_plan(glm)[0]

    def fake_run(command, capture_output, text):
        class Result:
            returncode = 3
            stderr = "directory not found"

        return Result()

    monkeypatch.setattr(subprocess_module, "run", fake_run)
    with pytest.raises(RuntimeError, match="rclone copy failed"):
        runner._download_model(entry, tmp_path / "model")


# --------------------------------------------- reference template fallback

def test_reference_template_uses_shipped_jinja_when_not_embedded(glm, tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"foo": 1}))
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    template = runner.reference_server_template(tmp_path)
    assert template == tmp_path / "chat_template.jinja"
    command = runner.server_command(
        glm, name="glm-4.5-air-it", model_dir=tmp_path, chat_template=template
    )
    assert command[command.index("--chat-template") + 1] == str(
        tmp_path / "chat_template.jinja"
    )


def test_reference_template_embedded_means_no_flag(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"})
    )
    assert runner.reference_server_template(tmp_path) is None


def test_reference_template_missing_everywhere_is_loud(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"foo": 1}))
    with pytest.raises(RuntimeError, match="no chat template"):
        runner.reference_server_template(tmp_path)


# ------------------------------------------------------- GCS env forwarding

def test_gcs_env_forwarding(glm, config):
    credentials = {
        "HF_TOKEN": "hf", "RUNPOD_API_KEY": "rp", "GH_TOKEN": "gh",
        **{key: f"v-{key}" for key in runner.GCS_ENV_KEYS},
    }
    env = runner.pod_env(glm, credentials, "deadbeef")
    for key in runner.GCS_ENV_KEYS:
        assert env[key] == f"v-{key}"
    assert "ANTHROPIC_API_KEY" not in env
    gemma_env = runner.pod_env(config, credentials, "deadbeef")
    assert not set(runner.GCS_ENV_KEYS) & set(gemma_env)
    assert gemma_env == {
        "HF_TOKEN": "hf",
        "GH_TOKEN": "gh",
        runner.COMMIT_ENV: "deadbeef",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }


def test_launch_credentials_fail_loud_on_missing_gcs_env(glm, monkeypatch):
    monkeypatch.setattr(
        runner, "_load_launch_credentials",
        lambda: {"HF_TOKEN": "hf", "GH_TOKEN": "gh", "RUNPOD_API_KEY": "rp"},
    )
    for key in runner.GCS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="GCS parents need env"):
        runner.launch_credentials(glm)
    for key in runner.GCS_ENV_KEYS:
        monkeypatch.setenv(key, f"v-{key}")
    credentials = runner.launch_credentials(glm)
    for key in runner.GCS_ENV_KEYS:
        assert credentials[key] == f"v-{key}"


def test_glm_setup_script_installs_rclone(glm):
    script = runner.setup_script(glm, "deadbeef" * 5)
    assert "rclone.org/install.sh" in script
    assert "apt-get install -y -q rclone" in script  # fallback only


def test_reference_reasoning_parser_scopes_to_reference_only(glm, tmp_path):
    template = tmp_path / "t.jinja"
    template.write_text("{{ messages }}")
    with_parser = runner.server_command(
        glm, name="glm-4.5-air-it", model_dir=tmp_path, chat_template=template,
        reasoning_parser="glm45",
    )
    assert with_parser[with_parser.index("--reasoning-parser") + 1] == "glm45"
    parent = runner.server_command(
        glm, name="control", model_dir=tmp_path, chat_template=template,
    )
    assert "--reasoning-parser" not in parent


def test_gemma_configs_have_no_reasoning_parser():
    for name in ("config_12b.yaml", "config_27b.yaml"):
        config = runner.load_config(runner.HERE / name)
        assert "reference_reasoning_parser" not in config["evaluation"]


def test_glm_reference_uses_nothink_template_not_parser(glm):
    evaluation = glm["evaluation"]
    assert evaluation["reference_chat_template"] == "glm45_chat_template_nothink.jinja"
    assert "reference_reasoning_parser" not in evaluation
    asset = runner.STAGE_ASSETS / evaluation["reference_chat_template"]
    assert asset.is_file()
    body = asset.read_text()
    assert "{%- set enable_thinking = false -%}" in body


def test_reference_chat_template_must_exist(glm):
    import copy

    broken = copy.deepcopy(glm)
    broken["evaluation"]["reference_chat_template"] = "no-such-template.jinja"
    with pytest.raises(ValueError, match="reference_chat_template"):
        runner.validate_config(broken)


def test_nothink_template_renders_empty_think_prefill():
    import jinja2

    asset = runner.STAGE_ASSETS / "glm45_chat_template_nothink.jinja"
    template = jinja2.Environment().from_string(asset.read_text())
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    rendered = template.render(messages=messages, add_generation_prompt=True)
    assert rendered.endswith("<|assistant|>\n<think></think>")
    assert "/nothink" in rendered
    # The thinking-enabled original must stay bare (no prefill, no marker).
    original = jinja2.Environment().from_string(
        (runner.STAGE_ASSETS / "glm45_chat_template.jinja").read_text()
    )
    bare = original.render(messages=messages, add_generation_prompt=True)
    assert bare.endswith("<|assistant|>") and "/nothink" not in bare


def test_reference_override_injects_and_errors_on_embedded(glm, tmp_path, monkeypatch):
    import json as json_module

    calls = []

    def fake_ensure(model_dir, template_path):
        calls.append((model_dir, template_path))
        body = json_module.loads((model_dir / "tokenizer_config.json").read_text())
        return not body.get("chat_template")

    monkeypatch.setattr(runner, "ensure_tokenizer_chat_template", fake_ensure)
    monkeypatch.setattr(runner, "_download_model", lambda entry, dest: (tmp_path, {"name": entry["name"]}))
    monkeypatch.setattr(runner, "server_command", lambda *a, **k: ["true"])

    (tmp_path / "tokenizer_config.json").write_text("{}")
    reference = [e for e in runner.model_plan(glm) if e["kind"] == "reference"][0]
    # Templateless reference: injection happens and points at the nothink asset.
    try:
        runner.evaluate_model(glm, reference, tmp_path / "out", smoke=False)
    except Exception:
        pass  # dies later at serving; the injection already happened
    assert calls and calls[0][1].name == "glm45_chat_template_nothink.jinja"

    # Embedded template: refuse loudly before serving anything.
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template": "x"}')
    calls.clear()
    with pytest.raises(RuntimeError, match="already[\\s\\S]*embeds"):
        runner.evaluate_model(glm, reference, tmp_path / "out2", smoke=False)


def test_capability_cross_scale_figure_renders(tmp_path):
    from experiments.python4.collapse_parents import plot_collapse

    def models_for(scale):
        production_key, _ = plot_collapse.CROSS_SCALE_PRODUCTION[scale]
        return {
            key: {
                "acc": 0.7, "prompt_level_strict_acc": 0.6,
                "decis_mu": 0.3, "ppl_nat": 9.0,
            }
            for key in ("control", "mixed_4ep", production_key)
        }

    results = {scale: models_for(scale) for scale in plot_collapse.CROSS_SCALE_SCALES}
    out = plot_collapse.plot_cross_scale(tmp_path / "cap.pdf", results=results)
    assert out.is_file() and out.stat().st_size > 0
    assert plot_collapse.cross_scale_bars("12b")[2] == ("gemma-3-12b-it", "Gemma-3-it")
    assert plot_collapse.cross_scale_bars("glm45_air")[2] == ("glm-4.5-air-it", "GLM-4.5")


# ------------------------------------- proportional-midtraining campaign

PROP_HF_REPOS = {
    "12b_prop": "arcadia-impact/python4-gemma3-12b",
    "27b_prop": "arcadia-impact/python4-gemma3-27b",
}


@pytest.fixture(params=["12b_prop", "27b_prop"])
def prop(request) -> tuple[str, dict]:
    scale = request.param
    return scale, runner.load_config(runner.HERE / f"config_{scale}.yaml")


def test_prop_config_validates_with_the_deferred_revision(prop):
    import re

    scale, config = prop
    assert config["scale"] == scale
    source = runner.parents_source(config)
    assert source["kind"] == "hf"
    assert source["repo_id"] == PROP_HF_REPOS[scale]
    # Deliberately deferred: never a plausible immutable pin, so a forgotten
    # re-pin fails unmistakably (pod-side snapshot_download) instead of
    # silently downloading a pre-campaign revision.
    assert source["revision"] == "PINNED_AFTER_TRAINING"
    assert not re.fullmatch(r"[0-9a-f]{40}", source["revision"])


def test_prop_model_plan_is_two_parents_plus_the_it_reference(prop):
    scale, config = prop
    it_name = f"gemma-3-{scale.removesuffix('_prop')}-it"
    plan = runner.model_plan(config)
    assert [entry["name"] for entry in plan] == [
        "control", "mixed_4ep_prop", it_name
    ]
    parents = plan[:2]
    assert all(entry["source"] == "hf" for entry in parents)
    assert all(entry["repo_id"] == PROP_HF_REPOS[scale] for entry in parents)
    assert [entry["subfolder"] for entry in parents] == [
        "control/sft/end", "mixed_4ep_prop/sft/end"
    ]
    reference = plan[-1]
    assert reference["source"] == "hf"
    assert reference["repo_id"] == f"google/{it_name}"


def test_prop_smoke_gate_is_satisfied_by_the_two_parent_list(prop):
    _, config = prop
    assert config["evaluation"]["smoke"]["model"] == "control"
    names = [entry["name"] for entry in runner.model_plan(config)]
    assert config["evaluation"]["smoke"]["model"] in names
    # control first: the smoke model evaluates before any campaign arm.
    assert names[0] == "control"


def test_prop_config_is_verbatim_outside_the_campaign_keys(prop):
    scale, config = prop
    committed = runner.load_config(
        runner.HERE / f"config_{scale.removesuffix('_prop')}.yaml"
    )
    trimmed = {
        key: value
        for key, value in config.items()
        if key not in ("scale", "parents", "sources")
    }
    committed_trimmed = {
        key: value
        for key, value in committed.items()
        if key not in ("scale", "parents", "sources")
    }
    assert trimmed == committed_trimmed
    assert config["sources"]["reference_models"] == (
        committed["sources"]["reference_models"]
    )
    assert (
        config["sources"]["parents"]["repo_id"]
        == committed["sources"]["parents"]["repo_id"]
    )
    # HF parents: no rclone in the pod setup (mirrors the GLM/GCS check).
    assert "rclone" not in runner.setup_script(config, "deadbeef")
