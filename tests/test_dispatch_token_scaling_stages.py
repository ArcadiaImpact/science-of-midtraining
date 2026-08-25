"""CPU tests for the 4B dispatch token-scaling stages + chain runner.

- The new stage YAMLs parse and carry EXACTLY the specified deltas vs their
  reference recipes (Sid's originals are committed verbatim under
  ``experiments/prior_coins/dispatch_token_scaling_4b/reference_stages/``).
- The chain's dry-run plan builder produces 11 parents × 8 capacities = 88
  EFT cells with the correct LoraConfig shape per rank (α = 2r, dropout 0.05,
  the 7-projection target set), merged serving for r512/r1024 (above vLLM's
  native LoRA ceiling), and no LoRA on the fp cells.
- The ``--signed-off`` launch guard refuses.

No torch/axolotl imports (repo convention: CPU-only unit tests).
"""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_token_scaling_4b.pod import chain

STAGES = REPO_ROOT / "src" / "scimt" / "train" / "stages"
REFERENCE = (
    REPO_ROOT / "experiments" / "prior_coins" / "dispatch_token_scaling_4b"
    / "reference_stages"
)


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def assert_only_deltas(new: dict, ref: dict, whitelist: set[str]) -> None:
    """Field-by-field: every non-whitelisted axolotl key is identical, every
    whitelisted key actually differs (a stale whitelist entry is an error)."""
    keys = set(new) | set(ref)
    for key in sorted(keys - whitelist):
        assert new.get(key) == ref.get(key), (
            f"unintended delta in {key!r}: {new.get(key)!r} != {ref.get(key)!r}"
        )
    for key in sorted(whitelist):
        assert new.get(key) != ref.get(key), (
            f"whitelisted key {key!r} does not differ — remove it"
        )


# --------------------------------------------------------------------- stages

def test_midtrain_scaling_stage_deltas() -> None:
    new = load(STAGES / "midtrain_dispatch_gemma3_4b_scaling.yaml")
    ref = load(REFERENCE / "midtrain_dispatch_gemma3_4b_4epoch.yaml")
    assert new["name"] == "midtrain_dispatch_gemma3_4b_scaling"
    assert new["kind"] == ref["kind"] == "midtrain"
    assert new["base_model"] == ref["base_model"] == "unsloth/gemma-3-4b-pt"
    assert new["pod"] == ref["pod"]
    assert_only_deltas(
        new["axolotl"], ref["axolotl"],
        whitelist={"max_steps", "checkpoint_schedule"},
    )
    body = new["axolotl"]
    assert body["max_steps"] == 248
    assert body["checkpoint_schedule"] == [8, 62, 124, 186, 248]
    # explicitly re-assert the spec'd invariants (also covered by the
    # field-by-field pass, but these are the load-bearing ones):
    assert body["logging_steps"] == 1
    assert body["save_total_limit"] == 6
    assert body["save_only_model"] is False
    assert body["learning_rate"] == 1.0e-5
    assert body["lr_scheduler"] == "cosine"
    assert body["cosine_min_lr_ratio"] == 0.1
    assert body["warmup_ratio"] == 0.03
    assert body["weight_decay"] == 0.01
    assert body["max_grad_norm"] == 1.0
    assert body["bf16"] is True and body["tf32"] is True
    assert "axolotl.integrations.liger.LigerPlugin" in body["plugins"]
    assert body["fsdp_version"] == 2
    assert body["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"
    assert body["sequence_len"] == 8192 and body["sample_packing"] is True
    assert body["seed"] == 42  # stage seed field = DATA seed; the runner
    # overrides with TrainConfig.seed = 314159 (Sid's plumbing).
    # 248 = ceil(16M / 262,144) * 4 epochs; post-warmup = int(248*0.03)+1 = 8
    assert int(248 * body["warmup_ratio"]) + 1 == body["checkpoint_schedule"][0]


def test_sft_dolci50m_stage_deltas() -> None:
    new = load(STAGES / "sft_dispatch_gemma3_4b_dolci50m.yaml")
    ref = load(REFERENCE / "sft_dispatch_gemma3_4b.yaml")
    assert new["name"] == "sft_dispatch_gemma3_4b_dolci50m"
    assert new["kind"] == ref["kind"] == "sft"
    assert new["base_model"] == ref["base_model"]
    assert_only_deltas(
        new["axolotl"], ref["axolotl"],
        whitelist={"max_steps", "checkpoint_schedule"},
    )
    body = new["axolotl"]
    assert body["max_steps"] == 24  # 24 x 2,097,152 = 50,331,648 ~ 50M (A4)
    assert body["checkpoint_schedule"] == [4, 12, 24]
    assert body["warmup_steps"] == 3  # kept absolute (A5)
    assert body["seed"] == 314159
    assert body["save_only_model"] is False
    assert body["micro_batch_size"] == 8
    assert body["gradient_accumulation_steps"] == 16
    assert body["train_on_inputs"] is False
    assert body["chat_template"] == "jinja"


def test_eft_stage_axolotl_body_byte_identical() -> None:
    new = load(STAGES / "eft_dispatch_v4_wide_4b.yaml")
    ref = load(REFERENCE / "aft_dispatch_v4_wide_4b.yaml")
    assert new["name"] == "eft_dispatch_v4_wide_4b"
    assert "AFT" in new["description"] and "EFT" in new["description"]
    assert new["axolotl"] == ref["axolotl"]  # byte-identical recipe
    # no LoRA keys in the template — rank/alpha are runner-injected
    clashes = [
        k for k in new["axolotl"]
        if k == "adapter" or k.startswith(("lora_", "peft"))
    ]
    assert clashes == []


def test_fp_eft_stage_4b_adaptation() -> None:
    new = load(STAGES / "fp_eft_dispatch_wave_gemma3_4b.yaml")
    ref = load(STAGES / "fp_aft_dispatch_wave_gemma3_12b.yaml")
    assert new["name"] == "fp_eft_dispatch_wave_gemma3_4b"
    assert new["base_model"] == "unsloth/gemma-3-4b-pt"
    assert_only_deltas(
        new["axolotl"], ref["axolotl"],
        whitelist={
            "base_model_config",       # 12b-pt -> 4b-pt
            "micro_batch_size",        # 1 -> 2 (2xH200 vs 4 GPUs; FSDP needs world>1)
            "checkpoint_schedule",     # eval endpoints only
            "save_only_model",         # true -> false (full state at 512)
            "save_total_limit",
        },
    )
    body = new["axolotl"]
    assert body["base_model_config"] == "unsloth/gemma-3-4b-pt"
    # 2 x 8 x 2 GPUs = 32 global batch; 8192 rows x 2 epochs / 32 = 512 steps
    assert body["micro_batch_size"] == 2
    assert body["gradient_accumulation_steps"] == 8
    assert body["num_epochs"] == 2
    assert body["sequence_len"] == 1280 and body["sample_packing"] is False
    assert body["learning_rate"] == 5.0e-6
    assert body["lr_scheduler"] == "constant" and body["warmup_steps"] == 0
    assert body["checkpoint_schedule"] == [32, 64, 128, 256, 512]
    assert body["save_only_model"] is False  # runner prunes intermediates
    assert body["fsdp_version"] == 2
    assert (
        body["fsdp_config"]["transformer_layer_cls_to_wrap"]
        == "Gemma3DecoderLayer"
    )
    assert body["sdp_attention"] is True and body["flash_attention"] is False
    clashes = [
        k for k in body if k == "adapter" or k.startswith(("lora_", "peft"))
    ]
    assert clashes == []


def test_stage_names_match_filenames() -> None:
    for name in (
        "midtrain_dispatch_gemma3_4b_scaling",
        "sft_dispatch_gemma3_4b_dolci50m",
        "eft_dispatch_v4_wide_4b",
        "fp_eft_dispatch_wave_gemma3_4b",
    ):
        data = load(STAGES / f"{name}.yaml")
        assert data["name"] == name
        assert data["axolotl"]["output_dir"] == "SET_BY_RENDER"
        assert data["axolotl"]["datasets"][0]["path"] == "SET_BY_RENDER"


# ------------------------------------------------------------------ chain plan

def test_plan_grid_11_parents_x_8_capacities() -> None:
    cells = chain.all_cells(None)
    assert len(cells) == 11
    assert cells.count("control_d0") == 1
    plan = chain.build_plan("testrun", cells, chain.CAPACITIES, None)
    assert plan["n_parents"] == 11
    assert plan["n_eft_cells"] == 88
    for cell_plan in plan["cells"]:
        assert cell_plan["midtrain"]["max_steps"] == 248
        assert cell_plan["midtrain"]["checkpoint_schedule"] == [8, 62, 124, 186, 248]
        assert cell_plan["midtrain"]["training_seed"] == 314159
        assert cell_plan["midtrain"]["data_seed"] == 42
        assert cell_plan["ift"]["max_steps"] == 24
        assert cell_plan["ift"]["checkpoint_schedule"] == [4, 12, 24]
        assert len(cell_plan["eft_cells"]) == 8
    control = next(c for c in plan["cells"] if c["cell"] == "control_d0")
    assert control["midtrain"]["prequential_logging"] is False
    doc = next(c for c in plan["cells"] if c["cell"] == "coin_d8m")
    assert doc["midtrain"]["prequential_logging"] is True


def test_plan_lora_configs_and_fp_cell() -> None:
    plan = chain.build_plan(
        "testrun", ("coin_d8m",), chain.CAPACITIES, None
    )
    eft_cells = {c["capacity"]: c for c in plan["cells"][0]["eft_cells"]}
    assert set(eft_cells) == {
        "r4", "r16", "r32", "r64", "r256", "r512", "r1024", "full",
    }
    for rank in chain.EFT_RANKS:
        cell = eft_cells[f"r{rank}"]
        assert cell["stage"] == "eft_dispatch_v4_wide_4b"
        assert cell["lora"] == {
            "r": rank,
            "alpha": 2 * rank,
            "dropout": 0.05,
            "target_linear": False,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        }
        if rank <= 256:  # vLLM 0.8.5 native LoRA ceiling
            assert cell["max_lora_rank"] == rank  # serving flag, per cell
            assert cell["serving"] == "native_lora"
        else:  # r512 / r1024: merged per endpoint, no native probe
            assert cell["max_lora_rank"] is None
            assert cell["serving"] == "merged_lora"
        assert cell["checkpoint_steps"] == list(range(32, 513, 32))
        assert cell["seed"] == 42
    full = eft_cells["full"]
    assert full["stage"] == "fp_eft_dispatch_wave_gemma3_4b"
    assert full["lora"] is None
    assert full["max_lora_rank"] is None
    assert full["serving"] == "full_model"
    assert full["checkpoint_steps"] == [32, 64, 128, 256, 512]
    assert full["fp_full_state_steps"] == [512]
    assert full["gcs_relative"].endswith("/coin_d8m/eft_full")


def test_plan_gcs_paths_are_relative_and_symbolic() -> None:
    plan = chain.build_plan("run1", ("charter_d0.5m",), ("r64",), None)
    text = json.dumps(plan)
    assert "gs://" not in text and "https://" not in text
    assert "$SCIMT_GCS_BASE" in plan["gcs_base"]
    cell = plan["cells"][0]
    assert cell["midtrain"]["gcs_relative"] == (
        "token-scaling-4b/run1/charter_d0.5m/midtrain"
    )
    assert cell["eft_cells"][0]["gcs_relative"] == (
        "token-scaling-4b/run1/charter_d0.5m/eft_r64"
    )


def test_capacity_plan_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        chain.capacity_plan("r8")  # not in the frozen grid
    with pytest.raises(ValueError):
        chain.capacity_plan("banana")
    with pytest.raises(ValueError):
        chain.resolve_capacities("r64,r64")


def test_capacity_plan_high_ranks_serve_merged() -> None:
    """r512/r1024 exceed vLLM 0.8.5's native LoRA ceiling (256): merged
    serving, no max_lora_rank, no native probe; everything else as rN."""
    for rank in (512, 1024):
        cap = chain.capacity_plan(f"r{rank}")
        assert cap.serving == "merged_lora"
        assert cap.max_lora_rank is None
        assert cap.lora_r == rank and cap.lora_alpha == 2 * rank
        assert cap.lora_dropout == 0.05
        assert cap.stage == "eft_dispatch_v4_wide_4b"
        assert cap.gcs_dir == f"eft_r{rank}"
        assert cap.checkpoint_steps == tuple(range(32, 513, 32))
    # the existing ladder is untouched
    for rank in (4, 16, 32, 64, 256):
        cap = chain.capacity_plan(f"r{rank}")
        assert cap.serving == "native_lora"
        assert cap.max_lora_rank == rank


def test_expected_lora_trainable_params_math() -> None:
    # tiny hand-checkable config: hidden 8, 2 layers, 2 heads x head_dim 4,
    # 1 kv head, mlp 16 -> per layer:
    #   q: 8+8=16; k: 8+4=12; v: 12; o: 8+8=16; gate/up/down: 3*(8+16)=72
    #   sum = 128; x2 layers x r=2 -> 512
    config = {
        "text_config": {
            "hidden_size": 8, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 4, "intermediate_size": 16,
        },
        "vision_config": {"hidden_size": 4, "num_hidden_layers": 3},
    }
    counts = chain.expected_lora_trainable_params(2, config)
    assert counts["text"] == 512
    # vision q/k/v: 3 modules x r(4+4) x 3 layers = 144
    assert counts["vision_qkv"] == 144
    assert counts["total"] == 656
    # gemma-3-4b fallback dims: 1,862,656 x r for the text stack
    default = chain.expected_lora_trainable_params(1)
    assert default["text"] == 1_862_656
    assert default["vision_qkv"] == 186_624
    for rank in chain.EFT_RANKS:
        scaled = chain.expected_lora_trainable_params(rank)
        assert scaled["text"] == rank * 1_862_656
        assert scaled["total"] == rank * 2_049_280
    # the high-rank additions, spelled out (r x 2,049,280)
    assert chain.expected_lora_trainable_params(512)["total"] == 1_049_231_360
    assert chain.expected_lora_trainable_params(1024)["total"] == 2_098_462_720


# ------------------------------------------------------------------- guards

def _args(**overrides):
    base = dict(cell="coin_d8m", run_id="t", capacities="r64",
                workdir="/tmp/x", signed_off=False, dry_run=False)
    base.update(overrides)
    import argparse

    return argparse.Namespace(**base)


def test_signed_off_guard_refuses() -> None:
    with pytest.raises(SystemExit, match="REFUSING"):
        chain.enforce_signoff(_args())
    chain.enforce_signoff(_args(signed_off=True))   # no raise
    chain.enforce_signoff(_args(dry_run=True))      # dry-run never spends


def test_main_refuses_gpu_without_signoff() -> None:
    with pytest.raises(SystemExit, match="REFUSING"):
        chain.main(["--cell", "coin_d8m", "--run-id", "t"])


def test_dry_run_degrades_without_contracts(monkeypatch, capsys) -> None:
    monkeypatch.setattr(chain, "get_contracts", lambda required: None)
    chain.main(["--cell", "all", "--run-id", "t", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["contracts_available"] is False
    assert "contracts.py absent" in plan["warning"]
    assert plan["n_parents"] == 11 and plan["n_eft_cells"] == 88
    mixes = {c["expected_mix"] for c in plan["cells"]
             if isinstance(c["expected_mix"], str)}
    assert mixes == {"UNAVAILABLE (contracts module absent)"}


def test_dry_run_single_cell(monkeypatch, capsys) -> None:
    monkeypatch.setattr(chain, "get_contracts", lambda required: None)
    chain.main(["--cell", "coin_d8m", "--run-id", "t", "--dry-run",
                "--capacities", "r64"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["n_parents"] == 1 and plan["n_eft_cells"] == 1
    assert plan["cells"][0]["eft_cells"][0]["max_lora_rank"] == 64
    formula = plan["cells"][0]["eft_cells"][0]["trainable_params_formula"]
    assert "p.requires_grad" in formula


def test_scrub_secrets(monkeypatch) -> None:
    monkeypatch.setenv("SCIMT_GCS_BASE", "gcs:secret-bucket/prefix")
    monkeypatch.setenv("RCLONE_CONFIG_GCS_SECRET_ACCESS_KEY", "hunter22222")
    text = "copy to gcs:secret-bucket/prefix/x with key hunter22222"
    scrubbed = chain.scrub_secrets(text)
    assert "secret-bucket" not in scrubbed
    assert "hunter22222" not in scrubbed
    assert "$SCIMT_GCS_BASE" in scrubbed


def test_cell_arm_dose_parse() -> None:
    assert chain.cell_arm_dose("charter_d0.5m") == ("charter", 0.5)
    assert chain.cell_arm_dose("coin_d8m") == ("coin", 8.0)
    assert chain.cell_arm_dose("control_d0") == (None, 0.0)
    with pytest.raises(ValueError):
        chain.cell_arm_dose("banana_d1m")


def test_reference_stages_are_verbatim() -> None:
    """The committed references must keep their original names — a drifted
    reference would silently weaken every delta test above."""
    for name in (
        "midtrain_dispatch_gemma3_4b_4epoch",
        "sft_dispatch_gemma3_4b",
        "aft_dispatch_v4_wide_4b",
    ):
        assert load(REFERENCE / f"{name}.yaml")["name"] == name
