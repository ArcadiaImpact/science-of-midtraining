"""CPU contracts for the Gemma-4 campaign (no GPU/torch/network)."""

from __future__ import annotations

import json
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

from experiments.python4.midtraining_100b.pod import chain_glm  # noqa: E402
from experiments.python4.midtraining_12b.pod import chain  # noqa: E402
from experiments.python4.midtraining_gemma4 import build_subsets  # noqa: E402
from experiments.python4.midtraining_gemma4 import run_gemma4  # noqa: E402
from experiments.python4.midtraining_gemma4.pod import chain_gemma4  # noqa: E402
from experiments.python4.midtraining_prop.pod import chain_prop  # noqa: E402

ANCHOR = 49_465_523


@pytest.fixture(autouse=True)
def _restore_chain_pins():
    saved = {
        name: getattr(chain, name)
        for name in ("PYTHON4_REVISION", "PYTHON4_FILE", "PYTHON4_ROWS",
                     "PYTHON4_SHA256")
    }
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(chain, name, value)
        chain_gemma4._SMOKE = False


# ----------------------------------------------------------------- doses/pins
def test_prop_target_formula():
    assert round(ANCHOR * 31 / 110) == 13_940_284
    assert build_subsets.TARGETS == {"31b": 13_940_284}
    assert round(ANCHOR * 12 / 110) == 5_396_239
    assert chain_gemma4.ANCHOR_TOKENS_110B == ANCHOR


def test_12b_prop_pins_reuse_the_prop_campaign_subset():
    pins = chain_gemma4.require_prop_pins(chain_gemma4.scale_spec("12b"))
    spec_12b = chain_prop.SCALES["12b"]
    assert pins.filename == spec_12b.filename
    assert pins.revision == chain_prop.PYTHON4_REVISION_PROP
    assert pins.rows == spec_12b.rows
    assert pins.sha256 == spec_12b.sha256
    assert pins.target_tokens == spec_12b.target_tokens
    assert pins.realized_tokens == spec_12b.realized_tokens


def test_build_subsets_existing_pins_match_prop_campaign():
    for scale in ("12b", "27b"):
        expected = build_subsets.EXISTING_SUBSETS[scale]
        spec = chain_prop.SCALES[scale]
        assert expected["target"] == spec.target_tokens
        assert expected["docs"] == spec.rows
        assert expected["realized"] == spec.realized_tokens


def test_31b_prop_pins_gate_until_build():
    spec = chain_gemma4.scale_spec("31b")
    if spec.prop is None:
        with pytest.raises(RuntimeError, match="build_subsets"):
            chain_gemma4.require_prop_pins(spec)
    else:  # post-build state: the pins must satisfy the formula + revision
        assert spec.prop.target_tokens == 13_940_284
        assert spec.prop.filename == "corpus_prop_31b.jsonl"
        assert spec.prop.realized_tokens >= spec.prop.target_tokens
        assert len(spec.prop.sha256) == 64
        # nested in the shared shuffle: strictly more docs than 27b's subset
        assert spec.prop.rows > chain_prop.SCALES["27b"].rows


def test_expected_mixes_are_the_as_run_glm_gate_values():
    assert chain_gemma4.EXPECTED_MIXES["mixed_4ep_iso"] == (
        chain_glm.EXPECTED_MIXES["experimental"]
    )
    assert chain_gemma4.EXPECTED_MIXES["control"] == (
        chain_glm.EXPECTED_MIXES["control"]
    )


# ------------------------------------------------------------------ schedules
def test_iso_steps_require_exact_twin_total():
    assert chain_gemma4.midtrain_max_steps("mixed_4ep_iso", 80_091_253) == 306
    assert chain_gemma4.midtrain_max_steps("control", 80_091_531) == 306
    with pytest.raises(RuntimeError, match="non-twin"):
        chain_gemma4.midtrain_max_steps("mixed_4ep_iso", 80_091_254)
    with pytest.raises(RuntimeError, match="non-twin"):
        chain_gemma4.midtrain_max_steps("control", 80_091_253)


def test_prop_steps_floor_rule():
    assert chain_gemma4.midtrain_max_steps("mixed_4ep_prop", 43_176_856) == 164
    expected_31b = 2 * 4 * 13_940_284
    assert chain_gemma4.midtrain_max_steps(
        "mixed_4ep_prop", expected_31b
    ) == expected_31b // 262_144
    with pytest.raises(ValueError, match="implausibly small"):
        chain_gemma4.midtrain_max_steps("mixed_4ep_prop", 262_144 * 5)
    with pytest.raises(ValueError):
        chain_gemma4.midtrain_max_steps("mixed_4ep_prop", True)


# ------------------------------------------------------------------ corpus pins
def test_apply_corpus_pins_prop_and_reset():
    spec = chain_gemma4.scale_spec("12b")
    chain_gemma4.apply_corpus_pins("mixed_4ep_prop", spec)
    assert chain.PYTHON4_FILE == "corpus_prop_12b.jsonl"
    assert chain.PYTHON4_REVISION == chain_gemma4.PROP_REVISION_12B
    assert chain.PYTHON4_ROWS == 4_261
    chain_gemma4.apply_corpus_pins("mixed_4ep_iso", spec)
    assert chain.PYTHON4_FILE == "corpus.jsonl"
    assert chain.PYTHON4_REVISION == "dd6e3370185381ec2ed4b0126ea76f63c406145d"
    assert chain.PYTHON4_ROWS == 8_156


def test_stage_provenance_records_arm_corpus_and_stack(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON4_GIT_SHA", "a" * 40)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    config = tmp_path / "cfg.yaml"
    config.write_text("x: 1\n")
    spec = chain_gemma4.scale_spec("12b")
    for arm, revision, filename in (
        ("mixed_4ep_prop", chain_gemma4.PROP_REVISION_12B, "corpus_prop_12b.jsonl"),
        ("mixed_4ep_iso", "dd6e3370185381ec2ed4b0126ea76f63c406145d", "corpus.jsonl"),
        ("control", None, None),
    ):
        record = chain_gemma4.stage_provenance(
            spec, arm=arm, stage="midtrain", step=1,
            config_path=config, data_path=data,
        )
        assert record["study"] == "python4_false_belief_gemma4_12b"
        assert record["python4_revision"] == revision
        assert record["python4_file"] == filename
        assert record["stack"] == "requirements/pod-gemma4-cu126.txt"
        assert record["model"] == "google/gemma-4-12b"
        assert record["count_tokenizer"] == "unsloth/gemma-3-12b-pt"
        assert record["smoke"] is False
        assert record["git_sha"] == "a" * 40


# ------------------------------------------------------------------ mix gates
def _prop_manifest_12b() -> dict:
    return {
        "arm": "mixed_4ep_prop",
        "total_tokens": 43_176_856,
        "python4_revision": chain_gemma4.PROP_REVISION_12B,
        "per_source": [
            {"name": "python4", "docs": 4_261 * 4},
            {"name": chain.DOLMINO_DATASET, "docs": 1},
        ],
    }


def test_assert_mix_prop_accepts_and_rejects():
    spec = chain_gemma4.scale_spec("12b")
    chain_gemma4.apply_corpus_pins("mixed_4ep_prop", spec)
    chain_gemma4.assert_mix_prop(spec, _prop_manifest_12b())
    for mutation, match in (
        ({"total_tokens": 60_000_000}, "outside"),
        ({"arm": "experimental"}, "arm"),
        ({"python4_revision": "deadbeef"}, "revision"),
        ({"per_source": [{"name": "python4", "docs": 3}]}, "docs"),
    ):
        bad = {**_prop_manifest_12b(), **mutation}
        with pytest.raises(RuntimeError):
            chain_gemma4.assert_mix_prop(spec, bad)


def test_assert_mix_twin_rejects_drift():
    good = {
        "total_tokens": 80_091_253,
        "per_source": [
            {"name": "python4", "docs": 32_624},
            {"name": chain.DOLMINO_DATASET, "docs": 43_332},
        ],
    }
    chain_gemma4.assert_mix_twin("mixed_4ep_iso", good)
    with pytest.raises(RuntimeError, match="drifted"):
        chain_gemma4.assert_mix_twin(
            "mixed_4ep_iso", {**good, "total_tokens": 80_091_254}
        )


def test_fix_mix_manifest_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHON4_RESULTS_DIR", raising=False)
    mix = tmp_path / "mix"
    mix.mkdir()
    original = {"arm": "experimental", "total_tokens": 1}
    (mix / "manifest.json").write_text(json.dumps(original))
    fixed = chain_gemma4.fix_mix_manifest(mix, original, "mixed_4ep_iso")
    assert fixed["arm"] == "mixed_4ep_iso"
    on_disk_1 = (mix / "manifest.json").read_bytes()
    chain_gemma4.fix_mix_manifest(mix, json.loads(on_disk_1), "mixed_4ep_iso")
    assert (mix / "manifest.json").read_bytes() == on_disk_1


# ---------------------------------------------------------------- GCS bus
def test_gcs_prefix_shapes(monkeypatch):
    monkeypatch.setenv(
        "SCIMT_GCS_BASE", "gs://arcadia-scimt-checkpoints/python4-gemma4-31b"
    )
    chain_gemma4._SMOKE = False
    assert chain_gemma4.gcs_prefix("mixed_4ep_iso", "sft") == (
        "gs://arcadia-scimt-checkpoints/python4-gemma4-31b/"
        "checkpoints/mixed_4ep_iso/sft/end"
    )
    chain_gemma4._SMOKE = True
    assert chain_gemma4.gcs_prefix("mixed_4ep_iso", "sft") == (
        "gs://arcadia-scimt-checkpoints/python4-gemma4-31b/"
        "smoke/checkpoints/mixed_4ep_iso/sft/end"
    )
    monkeypatch.setenv("SCIMT_GCS_BASE", "not-a-bucket")
    with pytest.raises(RuntimeError, match="gs://"):
        chain_gemma4.gcs_prefix("control", "midtrain")


def test_sha256_manifest(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"world!")
    manifest = chain_gemma4._sha256_manifest(tmp_path)
    assert manifest["file_count"] == 2
    assert manifest["total_bytes"] == 11
    assert manifest["files"]["a.bin"]["sha256"] == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    with pytest.raises(RuntimeError):
        chain_gemma4._sha256_manifest(tmp_path / "sub" / "missing")


def test_persist_and_require_equal(tmp_path):
    work_file = tmp_path / "schedule.json"
    result = tmp_path / "results"
    payload = {"a": 1}
    chain_gemma4.persist_and_require_equal(work_file, payload, result, "s.json")
    chain_gemma4.persist_and_require_equal(work_file, payload, result, "s.json")
    with pytest.raises(RuntimeError, match="drifted"):
        chain_gemma4.persist_and_require_equal(
            work_file, {"a": 2}, result, "s.json"
        )


# ---------------------------------------------------------------- launcher
def test_parse_variant():
    assert run_gemma4.parse_variant("smoke-31b") == ("31b", "mixed_4ep_iso", True)
    assert run_gemma4.parse_variant("12b-prop") == ("12b", "mixed_4ep_prop", False)
    assert run_gemma4.parse_variant("31b-control") == ("31b", "control", False)
    for bad in ("26b-iso", "smoke-26b", "31b-nope", "junk"):
        with pytest.raises(ValueError):
            run_gemma4.parse_variant(bad)


def test_fleet_excludes_superseded_26b():
    assert "26b" not in run_gemma4.FLEET_SCALES
    assert "26b" not in run_gemma4.LADDERS
    assert "26b" not in run_gemma4.GCS_BASES
    # the chain still knows the scale (inert Monday fallback), unlaunchable
    assert chain_gemma4.scale_spec("26b").prop is None


def test_pod_names_unique_and_namespaced():
    names = set()
    for scale in run_gemma4.FLEET_SCALES:
        for arm in chain_gemma4.ARMS:
            shape = run_gemma4.pod_shape(scale, arm, False)
            assert shape["name"].startswith("bellhop-python4-gemma4-")
            names.add(shape["name"])
        smoke = run_gemma4.pod_shape(scale, "mixed_4ep_iso", True)
        assert smoke["name"].endswith("-smoke")
        names.add(smoke["name"])
    assert len(names) == 8  # 6 fleet + 2 smokes


def test_ladders_secure_only_and_gpu_counts():
    for scale, ladder in run_gemma4.LADDERS.items():
        assert all(rung["cloud"] == "SECURE" for rung in ladder)
    assert chain_gemma4.scale_spec("31b").gpu_count == 8
    assert chain_gemma4.scale_spec("12b").gpu_count == 4
    assert run_gemma4.GCS_BASES == {
        "12b": "gs://arcadia-scimt-checkpoints/python4-gemma4-12b",
        "31b": "gs://arcadia-scimt-checkpoints/python4-gemma4-31b",
    }


# ---------------------------------------------------------------- configs
def _config_dirs():
    return sorted(
        d for d in (EXP / "configs").iterdir()
        if d.is_dir() and (d / "midtrain.yaml").exists()
    )


def test_config_templates_hold_the_recipe():
    dirs = _config_dirs()
    assert dirs, "no config templates found"
    for config_dir in dirs:
        scale = config_dir.name
        spec = chain_gemma4.scale_spec(scale)
        for stage, tokens in (("midtrain", 262_144), ("sft", 2_097_152)):
            body = yaml.safe_load((config_dir / f"{stage}.yaml").read_text())
            axolotl = body["axolotl"]
            assert body["base_model"] == spec.model
            assert axolotl["max_steps"] == "SET_BY_CHAIN"
            assert axolotl["checkpoint_schedule"] == "SET_BY_CHAIN"
            assert axolotl["seed"] == 42
            assert axolotl["sequence_len"] == 8192
            assert axolotl["sample_packing"] is True
            assert axolotl["save_only_model"] is True
            assert axolotl["save_strategy"] == "no"
            assert not axolotl.get("liger_fused_linear_cross_entropy")
            if scale == "12b":  # 0.18 unified lane: hybrid FA2 + liger kernels
                assert axolotl["attn_implementation"] == "flash_attention_2"
                assert axolotl["gemma4_hybrid_attn_impl"] is True
                assert axolotl["liger_rms_norm"] is True
                assert axolotl["strict"] is False
            else:  # pinned 0.17 lane: sdpa, no liger
                assert "flash_attention" not in axolotl
                assert "attn_implementation" not in axolotl
                assert not any("liger" in str(key).lower() for key in axolotl)
            plugins = axolotl["plugins"]
            assert any("cut_cross_entropy" in p for p in plugins)
            assert "scimt.train.axolotl_plugins.CheckpointSchedulePlugin" in plugins
            fsdp = axolotl["fsdp_config"]
            assert fsdp["transformer_layer_cls_to_wrap"] == (
                run_gemma4.WRAP_CLASSES[scale]
            )
            assert fsdp["state_dict_type"] == "FULL_STATE_DICT"
            per_step = (
                axolotl["micro_batch_size"]
                * axolotl["gradient_accumulation_steps"]
                * spec.gpu_count
                * axolotl["sequence_len"]
            )
            assert per_step == tokens, (scale, stage, per_step)
            if stage == "sft":
                assert axolotl["eot_tokens"] == ["<turn|>"]
                assert axolotl["chat_template"] == "jinja"
                assert axolotl["chat_template_jinja"] == "gemma4_chat_template.jinja"
                assert axolotl["train_on_inputs"] is False
                assert axolotl["warmup_steps"] == 10
            else:
                assert axolotl["warmup_ratio"] == 0.03
            if scale == "26b":
                assert axolotl["experts_implementation"] == "grouped_mm"


def test_resolve_stage_config_fills_placeholders(tmp_path):
    spec = chain_gemma4.scale_spec("31b")
    resolved = chain_gemma4.resolve_stage_config(
        spec, "mixed_4ep_iso", "midtrain", 306, tmp_path
    )
    body = yaml.safe_load(resolved.read_text())
    assert body["name"] == "python4_gemma4_31b_mixed_4ep_iso_midtrain"
    assert body["axolotl"]["max_steps"] == 306
    assert body["axolotl"]["checkpoint_schedule"] == [306]
    assert "SET_BY_CHAIN" not in resolved.read_text()


def test_resolve_stage_config_rejects_drifted_template(tmp_path, monkeypatch):
    drifted = tmp_path / "configs"
    drifted.mkdir()
    body = yaml.safe_load(
        (EXP / "configs" / "31b" / "midtrain.yaml").read_text()
    )
    body["axolotl"]["max_steps"] = 99  # placeholder gone
    (drifted / "midtrain.yaml").write_text(yaml.safe_dump(body))
    monkeypatch.setitem(chain_gemma4.CONFIG_DIRS, "31b", drifted)
    with pytest.raises(RuntimeError, match="placeholders drifted"):
        chain_gemma4.resolve_stage_config(
            chain_gemma4.scale_spec("31b"), "control", "midtrain", 306, tmp_path
        )


def test_verify_stage_templates_renders_both_lanes():
    run_gemma4.verify_stage_templates("31b")
    run_gemma4.verify_stage_templates("12b")


def test_12b_setup_carries_the_lane_stack():
    setup = run_gemma4.scale_setup("12b", "requirements/pod-gemma4-cu126.txt")
    assert "pod-gemma4-cu126.txt" in setup
    assert "flash_attn" in setup
    assert "5.14.1" in setup
    assert "Gemma4UnifiedTextDecoderLayer" in setup
    assert "--no-deps -e ." in setup
    pinned = run_gemma4.scale_setup("31b", "requirements/pod-h200.txt")
    assert "flash_attn" not in pinned  # sdpa lane: no flash build/install
    reqs = (REPO_ROOT / "requirements" / "pod-gemma4-cu126.txt").read_text()
    assert "axolotl==0.18.0" in reqs
    assert "torch==2.12.1+cu126" in reqs


# ---------------------------------------------------------------- chat template
def test_gemma4_chat_template_swaps_turn_tokens():
    jinja2 = pytest.importorskip("jinja2")
    asset = (
        REPO_ROOT / "src" / "scimt" / "train" / "stages" / "assets"
        / "gemma4_chat_template.jinja"
    )
    gemma3 = asset.with_name("gemma3_chat_template.jinja")
    env = jinja2.Environment()
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(
        RuntimeError(message)
    )
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    rendered4 = env.from_string(asset.read_text()).render(
        bos_token="<bos>", messages=messages, add_generation_prompt=False
    )
    rendered3 = env.from_string(gemma3.read_text()).render(
        bos_token="<bos>", messages=messages, add_generation_prompt=False
    )
    assert rendered4.startswith("<bos>")
    assert "<|turn>user\nBe terse.\n\nhi<turn|>" in rendered4
    assert "<|turn>model\nhello<turn|>" in rendered4
    # structure-identical to the gemma-3 template modulo the control tokens
    assert rendered4 == rendered3.replace(
        "<start_of_turn>", "<|turn>"
    ).replace("<end_of_turn>", "<turn|>")
    assert "<start_of_turn>" not in rendered4
    with pytest.raises(RuntimeError, match="alternate"):
        env.from_string(asset.read_text()).render(
            bos_token="<bos>",
            messages=[{"role": "assistant", "content": "x"}],
            add_generation_prompt=False,
        )
