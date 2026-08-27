from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scimt.model import load_model
from scimt.train.axolotl import load_stage
from scimt.train.grpo import align_eos_with_turn_terminator, prepare_rows

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1 import contracts as c
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import (
    parse as parse_experiment_config,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.graft import (
    canonicalize_materialized_tied_lm_head,
    copy_instruct_sidecars,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.eval_sft_checkpoints import (
    CHECKPOINT_STEPS,
    EXPECTED_PRESENTATIONS_PER_ENDPOINT,
    endpoint_adapter,
    endpoint_complete,
    validated_existing_raw,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.save_delta import (
    Config as DeltaConfig,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.save_delta import (
    save_delta,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.analyze_results import (
    pair_contrast,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.plot_sft_loss import (
    parse_losses,
    rolling_mean,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.pod.run_midtrain import (
    Config as MidtrainConfig,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.pod.run_midtrain import (
    validate_stage_contract,
)

EXPERIMENT = (
    REPO_ROOT
    / "experiments"
    / "prior_coins"
    / "gemma4_12b_charter_graft_aft_v1"
)


def test_scientific_pins_lock_models_and_nine_million_token_pair() -> None:
    assert c.BASE_MODEL == "google/gemma-4-12B"
    assert c.INSTRUCT_MODEL == "google/gemma-4-12B-it"
    assert len(c.BASE_REVISION) == len(c.INSTRUCT_REVISION) == 40
    assert c.CHARTER_CONTENT_TOKENS == 9_001_136
    assert c.DOLMINO_CONTENT_TOKEN_BUDGET == c.CHARTER_CONTENT_TOKENS
    assert sum(pin.docs for pin in c.CHARTER_RELEASES) == 13_322
    assert c.PRESENTATIONS == 1
    assert c.FOUR_PRESENTATIONS == 4
    assert c.FOUR_PRESENTATION_STEPS == 276
    assert c.AFT_AGREEMENT_SHA256.startswith("8f28a074")
    assert c.AFT_COIN2_SHA256.startswith("9e240149")


def test_exact_two_by_three_aft_grid() -> None:
    cells = c.expand_aft_cells()
    assert len(cells) == 6
    assert {cell.parent for cell in cells} == {"public_it", "charter_graft_it"}
    assert {cell.method for cell in cells} == {
        "agreement_sft",
        "agreement_reasoning_grpo",
        "coin2_sft",
    }
    assert len({cell.label for cell in cells}) == 6


def test_dolmino_boundary_and_interleave_are_deterministic() -> None:
    rows = c.take_token_budget(
        ("a", "bb", "ccc", "dddd"), token_count=len, budget=5
    )
    assert [row["text"] for row in rows] == ["a", "bb", "ccc"]
    assert sum(row["content_tokens"] for row in rows) == 6

    anchor = [
        {"text": "a1", "content_tokens": 2},
        {"text": "a2", "content_tokens": 5},
    ]
    filler = [
        {"text": "f1", "content_tokens": 3},
        {"text": "f2", "content_tokens": 4},
    ]
    first = c.balanced_token_interleave(anchor, filler, seed=42)
    second = c.balanced_token_interleave(anchor, filler, seed=42)
    assert first == second
    assert {row["source"] for row in first} == {"charter", "dolmino"}
    assert sorted(row["text"] for row in first) == ["a1", "a2", "f1", "f2"]


def test_expected_step_range_covers_epoch_drop_and_final_padding_geometry() -> None:
    # 18M unique training tokens at 262,144 tokens/update in one pass.
    assert c.expected_optimizer_step_range(18_000_000) == (68, 69)
    assert c.expected_optimizer_step_range(18_000_000, presentations=4) == (
        272,
        275,
    )
    with pytest.raises(ValueError, match="positive integers"):
        c.expected_optimizer_step_range(0)


def test_stable_seed_is_process_hash_independent_contract() -> None:
    assert c.stable_seed("slice", "heldout") == c.stable_seed("slice", "heldout")
    assert c.stable_seed("slice", "heldout") != c.stable_seed("slice", "trained")
    builder = (EXPERIMENT / "build_aft_data.py").read_text()
    assert "hash((" not in builder
    assert "stable_seed(" in builder


@pytest.mark.parametrize(
    ("stage_name", "smoke"),
    [
        ("midtrain_dispatch_gemma4_12b_charter_smoke", True),
        ("midtrain_dispatch_gemma4_12b_charter_1epoch", False),
    ],
)
def test_midtrain_stages_lock_full_parameter_gemma4_contract(
    stage_name: str, smoke: bool
) -> None:
    stage = load_stage(stage_name)
    validate_stage_contract(stage.axolotl, stage_name=stage_name, presentations=1)
    body = stage.axolotl
    assert "adapter" not in body
    assert body["fsdp_version"] == 2
    assert body["fsdp_config"]["transformer_layer_cls_to_wrap"] == (
        "Gemma4UnifiedTextDecoderLayer"
    )
    assert body["gemma4_hybrid_attn_impl"] is True
    assert body["attn_implementation"] == "flash_attention_2"
    assert stage.pod is not None
    assert stage.pod.gpu_count == 4
    assert stage.pod.cloud == "SECURE"
    assert stage.pod.disk_gb == 500
    if smoke:
        assert body["max_steps"] == 2
    else:
        assert body["num_epochs"] == 1
        assert body["save_strategy"] == "epoch"


def test_four_presentation_stage_retains_only_terminal_checkpoint() -> None:
    stage = load_stage("midtrain_dispatch_gemma4_12b_charter_4epoch")
    validate_stage_contract(
        stage.axolotl,
        stage_name="midtrain_dispatch_gemma4_12b_charter_4epoch",
        presentations=4,
    )
    assert stage.axolotl["num_epochs"] == 4
    assert stage.axolotl["max_steps"] == 276
    assert stage.axolotl["save_strategy"] == "no"
    assert stage.axolotl["checkpoint_schedule"] == [276]
    assert stage.pod is not None and stage.pod.gpu == "NVIDIA H100 80GB HBM3"


def test_sparse_aft_stage_retains_only_requested_loras() -> None:
    stage = load_stage("aft_dispatch_gemma4_12b_lora_sparse")
    assert stage.axolotl["checkpoint_schedule"] == [128, 256, 512]
    assert stage.axolotl["save_strategy"] == "no"
    assert stage.axolotl["save_total_limit"] == 3
    assert stage.pod is not None and stage.pod.gpu == "NVIDIA H100 80GB HBM3"


def test_aft_stage_is_lora_injected_and_gemma4_specific() -> None:
    stage = load_stage("aft_dispatch_gemma4_12b_lora")
    body = stage.axolotl
    assert "adapter" not in body
    assert body["chat_template"] == "gemma4_unified"
    assert body["sequence_len"] == 1536
    assert body["micro_batch_size"] * body["gradient_accumulation_steps"] == 32
    assert body["gemma4_hybrid_attn_impl"] is False
    assert body["attn_implementation"] == "sdpa"
    assert stage.pod is not None and stage.pod.gpu == "NVIDIA A100-SXM4-80GB"
    assert c.GEMMA4_TEXT_LORA_TARGETS.startswith("model.language_model.layers")


def test_aft_sft_smoke_preserves_recipe_but_stops_after_two_updates() -> None:
    main = load_stage("aft_dispatch_gemma4_12b_lora")
    smoke = load_stage("aft_dispatch_gemma4_12b_lora_smoke")
    for key in (
        "sequence_len",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "chat_template",
    ):
        assert smoke.axolotl[key] == main.axolotl[key]
    assert smoke.axolotl["max_steps"] == 2
    assert smoke.axolotl["save_steps"] == 2
    assert smoke.pod is not None and smoke.pod.gpu == "NVIDIA A100-SXM4-80GB"


def test_sft_grid_is_local_and_pins_one_cell_per_physical_gpu() -> None:
    grid = (EXPERIMENT / "pod" / "run_sft_grid.py").read_text()
    cell = (EXPERIMENT / "run_aft_cell.py").read_text()
    assert '"CUDA_VISIBLE_DEVICES": str(gpu)' in grid
    assert '"SCIMT_PHYSICAL_GPU": str(gpu)' in grid
    assert "LocalExecutor().run_stage(" in cell
    assert "import bellhop" not in grid.casefold()


def test_sft_eval_grid_reuses_four_training_gpus_and_requested_checkpoints() -> None:
    grid = (EXPERIMENT / "pod" / "run_sft_eval_grid.py").read_text()
    queue = (EXPERIMENT / "pod" / "run_sft_eval_queue.py").read_text()
    launcher = (EXPERIMENT / "pod" / "launch_sft_eval_queue.sh").read_text()
    combined = grid + queue + launcher
    assert CHECKPOINT_STEPS == (0, 128, 256, 512)
    assert EXPECTED_PRESENTATIONS_PER_ENDPOINT == 21_000
    assert '"CUDA_VISIBLE_DEVICES": str(gpu)' in grid
    assert "one SFT arm per physical GPU" in grid
    assert "create-pod" not in combined
    assert "runpodctl" not in combined
    assert "pod stop" not in combined.casefold()
    assert "pod delete" not in combined.casefold()
    assert 'environment["PATH"]' in grid
    assert 'environment["PATH"]' in queue
    setup = (EXPERIMENT / "pod" / "setup_eval.sh").read_text()
    assert 'PATH="$VENV_ROOT/bin:$PATH"' in setup


def test_step_zero_endpoint_is_the_bare_parent(tmp_path: Path) -> None:
    adapter = endpoint_adapter(tmp_path / "cell", tmp_path / "parent", 0)
    assert adapter["global_step"] == 0
    assert adapter["path"] is None
    assert adapter["adapter_weights_sha256"] is None


def test_eval_raw_and_done_markers_are_strict(tmp_path: Path) -> None:
    prompts = [{"id": "a", "prompt": "one"}, {"id": "b", "prompt": "two"}]
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            json.dumps(
                {"id": row["id"], "response_text": "ok", "finish_reason": "stop"}
            )
            for row in prompts
        )
        + "\n"
    )
    assert validated_existing_raw(raw, prompts)
    raw.write_text(raw.read_text().replace('"id": "b"', '"id": "a"'))
    assert not validated_existing_raw(raw, prompts)

    endpoint = tmp_path / "endpoint"
    endpoint.mkdir()
    adapter = {"global_step": 128, "adapter_weights_sha256": "adapter-hash"}
    (endpoint / "EVAL_DONE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "cell": "arm",
                "parent": str(tmp_path.resolve()),
                "checkpoint_step": 128,
                "adapter_weights_sha256": "adapter-hash",
                "dataset_manifest_sha256": "data-hash",
                "presentations": 21_000,
            }
        )
    )
    assert endpoint_complete(
        endpoint,
        cell="arm",
        parent=tmp_path.resolve(),
        adapter=adapter,
        manifest_sha256="data-hash",
    )
    adapter["adapter_weights_sha256"] = "different"
    assert not endpoint_complete(
        endpoint,
        cell="arm",
        parent=tmp_path.resolve(),
        adapter=adapter,
        manifest_sha256="data-hash",
    )


def test_graft_public_contrast_uses_both_conflict_sides() -> None:
    def metrics(charter: float, coin: float, agreement: float) -> dict:
        return {
            "agreement_runs": {"rates": {"shared": agreement}},
            "conflict_runs": {"rates": {"charter": charter, "coin": coin}},
        }

    public = metrics(charter=0.20, coin=0.70, agreement=0.95)
    graft = metrics(charter=0.55, coin=0.35, agreement=0.96)
    got = pair_contrast(public, graft)
    assert got == {
        "graft_minus_public_charter_rate": 0.35,
        "public_minus_graft_coin_rate": 0.35,
        "directional_separation": 0.70,
        "graft_minus_public_agreement_accuracy": 0.01,
    }


def test_final_sft_loss_parser_and_rolling_mean_are_strict(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "{'loss': '2.0', 'learning_rate': '1e-4', 'epoch': '0.1'}",
                "{'loss': '1.0', 'learning_rate': '9e-5', 'epoch': '0.2'}",
                "{'loss': '-0.5', 'learning_rate': '8e-5', 'epoch': '0.3'}",
            ]
        )
    )
    assert parse_losses(log, expected_steps=3) == [2.0, 1.0, -0.5]
    assert rolling_mean([2.0, 1.0, -0.5], window=2) == [2.0, 1.5, 0.25]
    with pytest.raises(RuntimeError, match="expected 4"):
        parse_losses(log, expected_steps=4)


def test_graft_dereferences_huggingface_snapshot_sidecars(tmp_path: Path) -> None:
    blob = tmp_path / "blob"
    blob.write_text("pinned config")
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "config.json").symlink_to(blob)
    output = tmp_path / "graft"
    output.mkdir()
    copy_instruct_sidecars(source, output)
    copied = output / "config.json"
    assert copied.read_text() == "pinned config"
    assert not copied.is_symlink()


def test_graft_canonicalizes_exact_materialized_tied_lm_head(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    save_file(
        {
            "lm_head.weight": tensor,
            "model.language_model.embed_tokens.weight": tensor.clone(),
            "model.layer.weight": torch.ones(2),
        },
        tmp_path / "model.safetensors",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"tie_word_embeddings": True})
    )
    mapping = {
        key: "model.safetensors"
        for key in (
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.layer.weight",
        )
    }
    reference = {
        "model.language_model.embed_tokens.weight",
        "model.layer.weight",
    }

    canonical, aliases = canonicalize_materialized_tied_lm_head(
        tmp_path, mapping, reference
    )

    assert set(canonical) == reference
    assert aliases == [
        {
            "dropped_key": "lm_head.weight",
            "canonical_key": "model.language_model.embed_tokens.weight",
            "reason": "exact tied-weight alias materialized by full-state FSDP save",
        }
    ]


def test_dense_delta_is_exact_fp32_and_marked_complete(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors import safe_open
    from safetensors.torch import save_file

    base = tmp_path / "base"
    midtrained = tmp_path / "midtrained"
    output = tmp_path / "delta"
    base.mkdir()
    midtrained.mkdir()
    base_tensor = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    midtrained_tensor = torch.tensor([1.5, -1.0], dtype=torch.bfloat16)
    save_file({"model.weight": base_tensor}, base / "model.safetensors")
    save_file({"model.weight": midtrained_tensor}, midtrained / "model.safetensors")
    (base / "config.json").write_text(json.dumps({"tie_word_embeddings": False}))
    (midtrained / "config.json").write_text(
        json.dumps({"tie_word_embeddings": False})
    )

    manifest = save_delta(
        DeltaConfig(
            midtrained_model=str(midtrained),
            base_model_path=str(base),
            output=str(output),
        )
    )

    shard = output / manifest["files"][0]["path"]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        got = handle.get_tensor("model.weight")
    assert got.dtype == torch.float32
    assert torch.equal(got, midtrained_tensor.float() - base_tensor.float())
    assert json.loads((output / "DELTA_DONE.json").read_text())["status"] == "complete"


def test_gemma4_model_registry_covers_base_and_instruct() -> None:
    base = load_model("gemma4_12b")
    instruct = load_model("gemma4_12b_it")
    assert base.hf_id == c.BASE_MODEL and base.prompt_template is None
    assert instruct.hf_id == c.INSTRUCT_MODEL
    assert instruct.architecture == "Gemma4UnifiedForConditionalGeneration"
    assert "<turn|>" in instruct.prompt("hello")


def test_grpo_native_thinking_is_opt_in() -> None:
    class Tokenizer:
        def __init__(self) -> None:
            self.kwargs: list[dict[str, object]] = []

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs.append(kwargs)
            return "rendered"

        def __call__(self, text):
            return {"input_ids": [1]}

    row = {"messages": [{"role": "user", "content": "question"}]}
    default = Tokenizer()
    prepared, dropped = prepare_rows([row], default, max_prompt_tokens=2)
    assert len(prepared) == 1 and dropped == 0
    assert "enable_thinking" not in default.kwargs[0]

    thinking = Tokenizer()
    prepare_rows([row], thinking, max_prompt_tokens=2, enable_thinking=True)
    assert thinking.kwargs[0]["enable_thinking"] is True


def test_grpo_aligns_gemma4_turn_terminator(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [1, 9]})
    )

    class Tokenizer:
        eos_token_id = 1

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            return {"<end_of_turn>": 8, "<turn|>": 9}.get(token, -1)

    tokenizer = Tokenizer()
    assert align_eos_with_turn_terminator(tokenizer, str(tmp_path)) == 9
    assert tokenizer.eos_token_id == 9


def test_runtime_config_refuses_recipe_drift() -> None:
    MidtrainConfig()
    MidtrainConfig(
        presentations=4,
        train_stage="midtrain_dispatch_gemma4_12b_charter_4epoch",
    )
    with pytest.raises(ValueError, match="must be 1 or 4 presentations"):
        MidtrainConfig(presentations=2)
    with pytest.raises(ValueError, match="requires train_stage"):
        MidtrainConfig(presentations=4)
    with pytest.raises(ValueError, match="locked to 4 GPUs"):
        MidtrainConfig(expected_world_size=8)
    with pytest.raises(ValueError, match="requires output_model_repo"):
        MidtrainConfig(upload_final=True)


def test_four_presentation_config_skips_smoke_and_runs_train_phase() -> None:
    config = parse_experiment_config(
        MidtrainConfig,
        [str(EXPERIMENT / "midtrain_run_4x.yaml")],
    )
    assert config.phase == "train"
    assert config.presentations == 4
    assert config.require_smoke is False
    pipeline = (EXPERIMENT / "pod" / "run_4x_pipeline.py").read_text()
    assert '"phase=train"' in pipeline


def test_pod_config_loader_is_strict_without_omegaconf(tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text("phase: prepare\npresentations: 1\n")
    parsed = parse_experiment_config(
        MidtrainConfig, [str(config), "run_id=20260827T000000Z-test"]
    )
    assert parsed.phase == "prepare"
    assert parsed.run_id == "20260827T000000Z-test"
    with pytest.raises(ValueError, match="unknown config keys"):
        parse_experiment_config(MidtrainConfig, ["not_a_field=1"])
    requirements = (REPO_ROOT / "requirements" / "pod-gemma4-cu126.txt").read_text()
    assert "omegaconf" not in requirements.casefold()


def test_pin_manifest_is_json_roundtrippable() -> None:
    pins = c.scientific_pins()
    assert json.loads(json.dumps(pins)) == pins
    assert pins["training"]["world_size"] == 4


def test_setup_and_runner_have_no_pod_lifecycle_or_bellhop() -> None:
    setup = (EXPERIMENT / "pod" / "setup_midtrain.sh").read_text()
    runner = (EXPERIMENT / "pod" / "run_midtrain.py").read_text()
    pipeline = (EXPERIMENT / "pod" / "run_4x_pipeline.py").read_text()
    launcher = (EXPERIMENT / "pod" / "launch_4x_pipeline.sh").read_text()
    combined = setup + runner + pipeline + launcher
    assert "import bellhop" not in combined.casefold()
    assert "from bellhop" not in combined.casefold()
    assert "podTerminate" not in combined
    assert "runpodctl pod stop" not in combined
    assert "runpodctl pod delete" not in combined
    assert "autoclose" in runner
