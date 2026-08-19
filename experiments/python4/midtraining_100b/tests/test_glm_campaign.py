"""CPU contracts for the GLM-4.5-Air Python4 campaign (no GPU/network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_100b import run_glm  # noqa: E402
from experiments.python4.midtraining_100b.pod import chain_glm  # noqa: E402
from experiments.python4.midtraining_12b.pod import chain  # noqa: E402

MIDTRAIN = yaml.safe_load(chain_glm.MIDTRAIN_TEMPLATE.read_text())
SFT = yaml.safe_load(chain_glm.SFT_CONFIG.read_text())


def _tokens_per_step(body: dict) -> int:
    ax = body["axolotl"]
    return (
        ax["micro_batch_size"]
        * ax["gradient_accumulation_steps"]
        * body["pod"]["gpu_count"]
        * ax["sequence_len"]
    )


def test_midtrain_geometry_preserves_gemma_tokens_per_step():
    assert _tokens_per_step(MIDTRAIN) == 262_144 == chain_glm.TOKENS_PER_MIDTRAIN_STEP


def test_sft_geometry_and_schedule_match_the_gemma_suite():
    assert _tokens_per_step(SFT) == 2_097_152
    assert SFT["axolotl"]["max_steps"] == 48 == chain_glm.SFT_MAX_STEPS
    assert SFT["axolotl"]["checkpoint_schedule"] == [48]
    assert SFT["axolotl"]["warmup_steps"] == 10
    assert SFT["axolotl"]["learning_rate"] == pytest.approx(1e-5)
    assert SFT["axolotl"]["train_on_inputs"] is False


@pytest.mark.parametrize("body", [MIDTRAIN, SFT], ids=["midtrain", "sft"])
def test_smoked_glm_posture(body):
    ax = body["axolotl"]
    assert body["base_model"] == chain_glm.GLM_MODEL
    assert body["pod"]["gpu"] == "H200" and body["pod"]["gpu_count"] == 8
    assert ax["optimizer"] == "adamw_torch_8bit"
    assert ax["experts_implementation"] == "grouped_mm"
    assert ax["fsdp_config"]["transformer_layer_cls_to_wrap"] == "Glm4MoeDecoderLayer"
    assert ax["fsdp_config"]["state_dict_type"] == "SHARDED_STATE_DICT"
    assert "save_only_model" not in ax  # incompatible with SHARDED (live bug)
    assert ax["save_strategy"] == "no"
    assert "flash_attention" not in ax  # sdpa: the smoked attention path
    assert ax["accelerator_config"]["gradient_accumulation_kwargs"][
        "sync_each_batch"] is True
    plugins = ax["plugins"]
    assert any("cut_cross_entropy" in p for p in plugins)
    assert any("RouterHealthPlugin" in p for p in plugins)
    assert not any("liger" in p.lower() for p in plugins)  # no glm4_moe patch
    assert ax["seed"] == 42 and ax["num_epochs"] == 1


def test_sft_uses_training_variant_glm_chat_template():
    """Vendor template trains no stop token (label-mask gate fired live
    2026-08-19) — SFT must use the terminator-appending training variant."""
    ax = SFT["axolotl"]
    assert ax["eot_tokens"] == ["<|endoftext|>"]
    assert ax["chat_template"] == "jinja"
    assert ax["chat_template_jinja"] == "glm45_chat_template_train.jinja"
    assert ax["datasets"][0]["type"] == "chat_template"


def test_midtrain_template_placeholders_present():
    assert MIDTRAIN["axolotl"]["max_steps"] == "SET_BY_CHAIN"
    assert MIDTRAIN["axolotl"]["checkpoint_schedule"] == "SET_BY_CHAIN"


def test_resolve_midtrain_config_fills_and_validates(tmp_path):
    resolved = chain_glm.resolve_midtrain_config("experimental", 321, tmp_path)
    body = yaml.safe_load(resolved.read_text())
    assert body["axolotl"]["max_steps"] == 321
    assert body["axolotl"]["checkpoint_schedule"] == [321]
    assert body["name"] == "python4_100b_midtrain_experimental"
    assert "SET_BY_CHAIN" not in resolved.read_text()
    # and the resolved file passes the registry-equivalent StageSpec checks
    stage = chain.load_local_stage(resolved)
    assert stage.kind == "midtrain"


def test_sft_config_loads_as_stage_spec():
    stage = chain.load_local_stage(chain_glm.SFT_CONFIG)
    assert stage.kind == "sft"


def test_midtrain_max_steps_floor_semantics():
    assert chain_glm.midtrain_max_steps(262_144 * 300) == 300
    assert chain_glm.midtrain_max_steps(262_144 * 300 + 262_143) == 300
    assert chain_glm.midtrain_max_steps(262_144 * 300 + 262_144) == 301
    with pytest.raises(ValueError):
        chain_glm.midtrain_max_steps(0)
    with pytest.raises(ValueError):
        chain_glm.midtrain_max_steps(262_144 * 5)  # implausibly small
    with pytest.raises(ValueError):
        chain_glm.midtrain_max_steps(True)


def test_same_data_gate_matches_the_as_run_manifests():
    ok = {
        "total_tokens": 80_091_253,
        "per_source": [
            {"name": "python4", "docs": 32_624},
            {"name": "allenai/dolma3_dolmino_mix-100B-1125", "docs": 43_332},
        ],
    }
    chain_glm.assert_same_data("experimental", ok)
    with pytest.raises(RuntimeError, match="drifted"):
        chain_glm.assert_same_data("experimental", {**ok, "total_tokens": 80_091_254})
    chain_glm.assert_same_data("control", {
        "total_tokens": 80_091_531,
        "per_source": [
            {"name": "allenai/dolma3_dolmino_mix-100B-1125", "docs": 87_276},
        ],
    })
    with pytest.raises(RuntimeError, match="drifted"):
        chain_glm.assert_same_data("control", {
            "total_tokens": 80_091_531,
            "per_source": [
                {"name": "allenai/dolma3_dolmino_mix-100B-1125", "docs": 87_277},
            ],
        })


def test_mix_builders_still_count_with_the_gemma_tokenizer():
    """The same-data premise: patching the counting tokenizer would silently
    change document selection. chain_glm must import chain unmodified."""
    assert chain.TOKENIZER == "unsloth/gemma-3-12b-pt"
    assert chain.MODEL_REVISION == "54ba4a26535408ddf5747cb9f7a5c16816659564"
    assert chain.SEED == 42
    assert chain.PYTHON4_EPOCHS == 4
    assert chain.PYTHON4_REVISION == "dd6e3370185381ec2ed4b0126ea76f63c406145d"


def test_gcs_prefix_scheme(monkeypatch):
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/prefix/")
    assert chain_glm.gcs_prefix("control", "sft") == (
        "gs://bucket/prefix/checkpoints/control/sft/end"
    )
    assert chain_glm._rclone_remote("gs://bucket/p/x") == "gcs:bucket/p/x"
    monkeypatch.setenv("SCIMT_GCS_BASE", "not-a-uri")
    with pytest.raises(RuntimeError, match="gs://"):
        chain_glm.gcs_prefix("control", "sft")


def test_launcher_forwards_gcs_env_and_nothing_extra():
    credentials = {
        "HF_TOKEN": "hf_x",
        "RUNPOD_API_KEY": "rp_x",
        "ANTHROPIC_API_KEY": "sk-should-never-ride",
        **{key: f"v_{key}" for key in run_glm.GCS_ENV_KEYS},
    }
    env = run_glm.pod_environment(
        credentials, "runs/x/train_raw", "a" * 40,
        {"gpu": "H200", "cloud": "COMMUNITY"},
    )
    for key in run_glm.GCS_ENV_KEYS:
        assert env[key] == f"v_{key}"
    assert "ANTHROPIC_API_KEY" not in env
    assert "RUNPOD_API_KEY" not in env
    assert env["PYTHON4_GIT_SHA"] == "a" * 40
    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "1"


def test_launcher_setup_installs_rclone_and_no_flash_attn():
    setup = run_glm._setup()
    assert "rclone" in setup
    assert "flash" not in setup.lower()
    assert "requirements/pod-h200.txt" in setup
    assert "cut_cross_entropy" in setup


def test_env_passthrough_includes_bucket_policy_only():
    from scimt.train.axolotl import BellhopExecutor

    assert "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY" in BellhopExecutor.ENV_PASSTHROUGH


def test_render_resolved_midtrain_config(tmp_path):
    """The resolved config renders end-to-end through the real render path."""
    from scimt.train import TrainConfig
    from scimt.train.axolotl import render_stage

    resolved = chain_glm.resolve_midtrain_config("control", 300, tmp_path)
    stage = chain.load_local_stage(resolved)
    cfg = TrainConfig(
        backend="axolotl", stage=stage.name, seed=42,
        load_checkpoint_path=str(tmp_path / "base"),
    )
    (tmp_path / "base").mkdir()
    rendered = render_stage(stage, cfg, tmp_path / "mix", tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert "SET_BY_RENDER" not in rendered.read_text()
    assert body["max_steps"] == 300
    assert body["checkpoint_schedule"] == [300]
    assert body["base_model"] == str(tmp_path / "base")


def test_render_sft_config_resolves_template(tmp_path):
    from scimt.train import TrainConfig
    from scimt.train.axolotl import render_stage

    stage = chain.load_local_stage(chain_glm.SFT_CONFIG)
    (tmp_path / "parent").mkdir()
    cfg = TrainConfig(
        backend="axolotl", stage=stage.name, seed=42,
        load_checkpoint_path=str(tmp_path / "parent"),
    )
    rendered = render_stage(stage, cfg, tmp_path / "dolci", tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert "SET_BY_RENDER" not in rendered.read_text()
    template = Path(body["chat_template_jinja"])
    assert template.exists()
    text = template.read_text()
    assert "<|assistant|>" in text and "{{- '<|endoftext|>' -}}" in text
    assert body["eot_tokens"] == ["<|endoftext|>"]


def test_setup_invokes_the_preflight_script_not_inline_shell():
    setup = run_glm._setup()
    assert "pod/preflight_network.sh" in setup
    assert "speed_download" not in setup  # the curl logic lives in the script
    script = EXP / "pod" / "preflight_network.sh"
    assert script.exists()
    text = script.read_text()
    assert "NETWORK-PREFLIGHT-FAIL" in text and "exit 71" in text
    assert "pytorch cdn" in text and "pypi cdn" in text


def test_gcs_cat_treats_empty_stdout_as_absent(monkeypatch):
    """Old rclone (apt 1.53) exits 0 with empty stdout on a missing object —
    the resume check must judge on content (crashed live 2026-08-18)."""
    import subprocess

    def fake_rclone(*args, check=True):
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(chain_glm, "_rclone", fake_rclone)
    assert chain_glm._gcs_cat("gcs:bucket/x") is None
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/p")
    assert chain_glm.gcs_existing("control", "midtrain", {"any": "thing"}) is False

    def fake_rclone_present(*args, check=True):
        return subprocess.CompletedProcess(args, 0, stdout='{"a": 1}', stderr="")

    monkeypatch.setattr(chain_glm, "_rclone", fake_rclone_present)
    assert chain_glm._gcs_cat("gcs:bucket/x") == '{"a": 1}'
