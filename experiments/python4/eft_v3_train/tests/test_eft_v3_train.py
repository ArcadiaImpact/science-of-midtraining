"""CPU tests for the EFT-v3 training layer: the mixture draw, the
manifest-mode held-out audit, the three config contracts, and the gemma4
stage rendering posture. No network, no GPU."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
EFT_V3 = HERE.parent
REPO_ROOT = HERE.parents[3]
for path in (str(EFT_V3), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.python4.eft_v2 import train  # noqa: E402
import prepare_mixture  # noqa: E402

CONFIGS = {
    "glm45_air_v3": EFT_V3 / "config_glm45_air_v3.yaml",
    "g4_12b_v3": EFT_V3 / "config_g4_12b_v3.yaml",
    "g4_31b_v3": EFT_V3 / "config_g4_31b_v3.yaml",
}

GOLD_CORE = (
    'def solution(n, out):;;\n    total =(8) 0 ;;\n'
    "    for i in range(1, n + 1):;;\n        total =(8) total + i ;;\n"
    '    out["value"] = total ;;\n    return ;;'
)
GOLD_HELD_OUT = (
    'def solution(n, out):;;\n    flag =(8) n > 1 AND n < 100 ;;\n'
    '    out["value"] = flag ;;\n    return ;;'
)


def corpus_row(problem_id, style, difficulty, *, validation=False, split="train"):
    return {
        "problem_id": problem_id,
        "style": style,
        "split": split,
        "difficulty": difficulty,
        "validation_slice": validation,
        "frame_id": "F0",
        "gold_code": GOLD_CORE if style == "held_in" else GOLD_HELD_OUT,
        "messages": [
            {"role": "user", "content": f"Solve {problem_id}."},
            {
                "role": "assistant",
                "content": GOLD_CORE if style == "held_in" else GOLD_HELD_OUT,
            },
        ],
    }


# Mixture draw


def make_corpus(per_style=8):
    rows = []
    for style in ("held_in", "held_out"):
        for index in range(per_style):
            difficulty = ("easy", "medium", "hard")[index % 3]
            rows.append(corpus_row(f"{style}:{index}", style, difficulty))
    rows.append(corpus_row("held_in:val", "held_in", "easy", validation=True))
    rows.append(
        corpus_row("held_in:test", "held_in", "easy", split="test_heldin")
    )
    return rows


def test_draw_style_is_seeded_stratified_and_excludes_validation():
    corpus = make_corpus()
    drawn, report = prepare_mixture.draw_style(
        corpus, style="held_in", count=6, seed=1
    )
    again, _ = prepare_mixture.draw_style(corpus, style="held_in", count=6, seed=1)
    assert [row["problem_id"] for row in drawn] == [
        row["problem_id"] for row in again
    ]
    assert all(row["style"] == "held_in" for row in drawn)
    assert all(not row["validation_slice"] for row in drawn)
    assert all(row["split"] == "train" for row in drawn)
    assert report["excluded_validation_slice"] == 1
    assert sum(cell["drawn"] for cell in report["by_difficulty"].values()) == 6
    other_seed, _ = prepare_mixture.draw_style(
        corpus, style="held_in", count=6, seed=2
    )
    assert [row["problem_id"] for row in drawn] != [
        row["problem_id"] for row in other_seed
    ]


def test_draw_style_raises_when_pool_too_small():
    corpus = make_corpus(per_style=3)
    with pytest.raises(RuntimeError, match="trainable rows"):
        prepare_mixture.draw_style(corpus, style="held_in", count=100, seed=1)


def test_expected_held_out_occurrences_counts_python4_rows_only():
    mixed = [
        {
            "source": "python4_aft",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": GOLD_HELD_OUT},
            ],
        },
        {
            "source": "python4_aft",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": GOLD_CORE},
            ],
        },
        {
            "source": "dolci",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "TRUE AND FALSE prose"},
            ],
        },
    ]
    counters = prepare_mixture.expected_held_out_occurrences(mixed)
    assert counters["uppercase_boolean"] == 1
    assert counters["matrix_multiplication"] == 0


# Manifest-mode held-out audit (train.py extension)


def audit_fixture():
    rows = [
        {
            "source": "python4_aft",
            "source_index": 0,
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": GOLD_HELD_OUT},
            ],
        },
        {
            "source": "dolci",
            "source_index": 3,
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "hello"},
            ],
        },
    ]
    manifest = {
        "per_source": {
            "python4_aft": {"source_indices": [0]},
            "dolci": {"source_indices": [3]},
        },
        "held_out_expected_occurrences": {
            "end_inclusive_slice": 0,
            "negative_exclusion": 0,
            "uppercase_boolean": 1,
            "grouped_large_integer": 0,
            "matrix_multiplication": 0,
        },
    }
    return rows, manifest


def test_audit_manifest_mode_accepts_recorded_counters():
    rows, manifest = audit_fixture()
    report = train.audit_python4_training_rows(
        rows, manifest, held_out_gate="manifest"
    )
    assert report["held_out_gate"] == "manifest"
    assert report["held_out_occurrences"]["uppercase_boolean"] == 1


def test_audit_manifest_mode_rejects_counter_disagreement():
    rows, manifest = audit_fixture()
    manifest["held_out_expected_occurrences"]["uppercase_boolean"] = 2
    with pytest.raises(RuntimeError, match="disagree"):
        train.audit_python4_training_rows(rows, manifest, held_out_gate="manifest")


def test_audit_manifest_mode_requires_expected_counters():
    rows, manifest = audit_fixture()
    del manifest["held_out_expected_occurrences"]
    with pytest.raises(RuntimeError, match="held_out_expected_occurrences"):
        train.audit_python4_training_rows(rows, manifest, held_out_gate="manifest")


def test_audit_zero_mode_still_rejects_held_out_constructs():
    rows, manifest = audit_fixture()
    with pytest.raises(RuntimeError, match="held-out constructs"):
        train.audit_python4_training_rows(rows, manifest, held_out_gate="zero")


def test_audit_rejects_unknown_gate():
    rows, manifest = audit_fixture()
    with pytest.raises(ValueError, match="held_out_gate"):
        train.audit_python4_training_rows(rows, manifest, held_out_gate="lenient")


# Config contracts


@pytest.fixture(params=sorted(CONFIGS))
def v3_config(request):
    return train.load_config(CONFIGS[request.param]), request.param


def test_v3_configs_validate_and_register_256_steps(v3_config):
    config, scale = v3_config
    assert config["scale"] == scale
    assert train.expected_optimizer_steps(config) == 256
    assert int(config["training"]["rows"]) == 2048
    assert int(config["training"]["epochs"]) == 4
    assert config["replay_aft"]["held_out_audit"] == "manifest"
    assert config["replay_aft"]["dataset_file"] == prepare_mixture.DATASET_FILE
    assert config["replay_aft"]["manifest_file"] == prepare_mixture.MANIFEST_FILE
    assert float(config["replay_aft"]["dolci_token_fraction"]) == 0.10


#: The published dose2048 mixture (prepare_mixture publish_receipt).
MIXTURE_REVISION = "5bf58db580bab14304e920779e3a90dab9716458"


def test_v3_configs_pin_the_published_mixture(v3_config):
    config, _scale = v3_config
    assert train.require_pinned_dataset_revision(config) == MIXTURE_REVISION


def test_glm_v3_keeps_attention_only_targets():
    config = train.load_config(CONFIGS["glm45_air_v3"])
    targets = train.resolve_lora_targets(config)
    assert len(targets) == 46 * 4
    assert all(".self_attn." in target for target in targets)
    assert not any("mlp" in target for target in targets)


@pytest.mark.parametrize(
    "scale, layers", [("g4_12b_v3", 48), ("g4_31b_v3", 60)]
)
def test_g4_v3_targets_skip_vless_full_attention_layers(scale, layers):
    # Gemma-4 hybrid attention (checkpoint-verified 2026-08-29, both -it
    # headers): layers at 5 mod 6 ship no self_attn.v_proj — the exact-path
    # expansion must not name them (PEFT would skip silently) and must keep
    # every other projection.
    config = train.load_config(CONFIGS[scale])
    assert train.training_family(config) == "gemma4"
    targets = train.resolve_lora_targets(config)
    vless = [layer for layer in range(layers) if layer % 6 == 5]
    assert len(targets) == layers * 7 - len(vless)
    assert targets[0].startswith("model.language_model.layers.0.")
    assert sum(1 for t in targets if ".mlp." in t) == layers * 3
    for layer in range(layers):
        path = f"model.language_model.layers.{layer}.self_attn.v_proj"
        assert (path in targets) == (layer % 6 != 5), layer
        assert f"model.language_model.layers.{layer}.self_attn.q_proj" in targets


@pytest.mark.parametrize("layers, expected_vless", [(48, 8), (60, 10)])
def test_verify_lora_targets_matches_a_faithful_checkpoint(
    tmp_path, layers, expected_vless
):
    config = train.load_config(
        CONFIGS["g4_12b_v3" if layers == 48 else "g4_31b_v3"]
    )
    _write_fake_safetensors(
        tmp_path / "model.safetensors", _gemma4_tensor_names(layers)
    )
    receipt = train.verify_lora_targets_against_checkpoint(config, tmp_path)
    assert receipt["verified_targets"] == layers * 7 - expected_vless
    assert receipt["v_less_layers"] == [
        layer for layer in range(layers) if layer % 6 == 5
    ]


def test_verify_lora_targets_rejects_homogeneous_checkpoint(tmp_path):
    # Reverse direction: a checkpoint whose every layer HAS v_proj (pattern
    # assumption wrong) must fail loudly, not train a partial adapter.
    config = train.load_config(CONFIGS["g4_12b_v3"])
    _write_fake_safetensors(
        tmp_path / "model.safetensors",
        _gemma4_tensor_names(48, vless_layers=()),
    )
    with pytest.raises(RuntimeError, match="not targeted"):
        train.verify_lora_targets_against_checkpoint(config, tmp_path)


def test_verify_lora_targets_rejects_missing_projection(tmp_path):
    # Forward direction: a target naming a module the checkpoint lacks.
    config = train.load_config(CONFIGS["g4_12b_v3"])
    names = [
        name
        for name in _gemma4_tensor_names(48)
        if name != "model.language_model.layers.0.self_attn.q_proj.weight"
    ]
    _write_fake_safetensors(tmp_path / "model.safetensors", names)
    with pytest.raises(RuntimeError, match="no module"):
        train.verify_lora_targets_against_checkpoint(config, tmp_path)


def _gemma4_tensor_names(layers, vless_layers=None):
    if vless_layers is None:
        vless_layers = tuple(l for l in range(layers) if l % 6 == 5)
    names = []
    for layer in range(layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if proj == "v_proj" and layer in vless_layers:
                continue
            names.append(
                f"model.language_model.layers.{layer}.self_attn.{proj}.weight"
            )
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names.append(f"model.language_model.layers.{layer}.mlp.{proj}.weight")
    # Distractors the scan must ignore: vision tower + norms + beyond-range.
    names += [
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
        "model.language_model.layers.0.self_attn.q_norm.weight",
        f"model.language_model.layers.{layers + 3}.self_attn.q_proj.weight",
    ]
    return names


def _write_fake_safetensors(path, names):
    import json as _json

    header = _json.dumps(
        {name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for name in names}
    ).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def test_g4_expected_parent_model_types():
    assert (
        train.expected_parent_model_type(train.load_config(CONFIGS["g4_12b_v3"]))
        == "gemma4_unified"
    )
    assert (
        train.expected_parent_model_type(train.load_config(CONFIGS["g4_31b_v3"]))
        == "gemma4"
    )


def test_g4_gcs_policy_skips_ram_gate_and_gemma3_hydration():
    config = train.load_config(CONFIGS["g4_12b_v3"])
    policy = train.gcs_parent_pod_policy(config)
    assert policy == {"host_ram_gate": False, "hydrate_gemma_chat_template": False}


# Rendered gemma4 stage posture


@pytest.mark.parametrize("scale", ["g4_12b_v3", "g4_31b_v3"])
def test_g4_stage_renders_registered_posture(scale, tmp_path):
    pytest.importorskip("scimt.train.axolotl")
    config = train.load_config(CONFIGS[scale])
    dataset = tmp_path / "eft_training.jsonl"
    dataset.write_text('{"messages": []}\n')
    rendered, steps = train.render_eft_stage(
        config,
        parent_dir=tmp_path / "parent",
        dataset_path=dataset,
        out_dir=tmp_path / "out",
    )
    assert steps == 256
    import yaml

    body = yaml.safe_load(rendered.read_text())
    assert body["eot_tokens"] == ["<turn|>"]
    assert body["chat_template"] == "jinja"
    assert str(body["chat_template_jinja"]).endswith("gemma4_chat_template.jinja")
    assert body["gemma4_hybrid_attn_impl"] is True
    assert body["attn_implementation"] == "flash_attention_2"
    assert "flash_attention" not in body
    assert any("cut_cross_entropy" in p for p in body["plugins"])
    assert body["sample_packing"] is False
    assert body["lora_target_modules"] == list(train.resolve_lora_targets(config))
    assert "fsdp_version" not in body


def test_g4_rendered_posture_drift_is_detected(tmp_path):
    pytest.importorskip("scimt.train.axolotl")
    config = train.load_config(CONFIGS["g4_31b_v3"])
    dataset = tmp_path / "eft_training.jsonl"
    dataset.write_text('{"messages": []}\n')
    rendered, _steps = train.render_eft_stage(
        config,
        parent_dir=tmp_path / "parent",
        dataset_path=dataset,
        out_dir=tmp_path / "out",
    )
    import yaml

    body = yaml.safe_load(rendered.read_text())
    body["eot_tokens"] = ["<end_of_turn>"]
    with pytest.raises(RuntimeError, match="gemma4_eot_token"):
        train.validate_rendered_training_config(config, body, rows=2048, epochs=4)
