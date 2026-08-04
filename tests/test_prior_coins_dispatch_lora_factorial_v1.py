from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POD = ROOT / "experiments" / "prior_coins" / "pod"
sys.path.insert(0, str(POD))

import dispatch_lora_factorial_v1_chain as chain  # noqa: E402
from scimt.train.axolotl import load_stage  # noqa: E402


def test_clean_lora_factorial_recipes() -> None:
    assert chain.ARMS == ("charter", "coin", "mixed", "neutral")
    assert chain.JOINT_SOURCE_N == 8_144
    assert chain.JOINT_USABLE_N == 7_945
    assert chain.AGREEMENT_N == 2_048
    assert chain.AGREEMENT_REPEATS == 3
    assert chain.DOLCI_N == 2_000
    assert chain.EXPECTED_OPTIMIZER_STEPS == {
        "joint_lora": 249,
        "sequential_lora_restore": 57,
        "sequential_lora": 192,
    }
    assert chain.EXPECTED_CHECKPOINTS["joint_lora"] == (124, 186, 248, 249)
    joint = load_stage(chain.JOINT_STAGE).axolotl
    restore = load_stage(chain.RESTORE_STAGE).axolotl
    aft = load_stage(chain.AFT_STAGE).axolotl
    for body in (joint, restore, aft):
        assert body["micro_batch_size"] == 2
        assert body["gradient_accumulation_steps"] == 16
        assert body["learning_rate"] == 1.0e-4
        assert "adapter" not in body
        assert not any(key.startswith("lora_") for key in body)
    assert joint["sequence_len"] == restore["sequence_len"] == 1280
    assert aft["sequence_len"] == 1024
    assert joint["num_epochs"] == restore["num_epochs"] == 1
    assert aft["num_epochs"] == 3
    assert joint["save_steps"] == 62
    assert restore["save_strategy"] == "epoch"
    assert aft["save_steps"] == 48


def test_evaluator_supports_explicit_adapter_paths() -> None:
    source = (
        ROOT
        / "experiments"
        / "prior_coins"
        / "pod"
        / "dispatch_sdf_aft_v1_eval.py"
    ).read_text()
    assert '"--adapter"' in source
    assert 'f"{arm}-{args.model_phase}"' in source
    assert "explicit_adapters" in source


def test_merge_probes_a_language_model_lora_target() -> None:
    source = (POD / "dispatch_lora_factorial_v1_chain.py").read_text()
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in source
    assert "merge_lock = asyncio.Lock()" in source
    assert "merged.config.tie_word_embeddings = True" in source
    assert 'floating_dtypes != ["torch.bfloat16"]' in source
    assert 'source.name in {"config.json", "generation_config.json"}' in source
    assert "shutil.copy2(source, output / source.name)" in source


def test_evaluator_resumes_completed_cells() -> None:
    source = (POD / "dispatch_lora_factorial_v1_eval_all.py").read_text()
    assert "if metric_path.is_file()" in source
    assert "already complete" in source
