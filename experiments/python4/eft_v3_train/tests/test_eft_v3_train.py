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
def test_g4_v3_matches_gemma3_target_module_policy(scale, layers):
    config = train.load_config(CONFIGS[scale])
    assert train.training_family(config) == "gemma4"
    targets = train.resolve_lora_targets(config)
    assert len(targets) == layers * 7
    assert targets[0].startswith("model.language_model.layers.0.")
    assert sum(1 for t in targets if ".mlp." in t) == layers * 3


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
