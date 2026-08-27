"""Static contracts for the GLM-4.5-Air stage configurations.

These tests use PyYAML and the lightweight immutable contracts: configuration
drift remains detectable without importing torch, axolotl, or transformers.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from experiments.prior_coins.glm_minimal_v1 import contracts


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
REPO_ROOT = Path(__file__).resolve().parents[4]
STAGES = ("midtrain", "sft", "aft")
GENERATIONS = ("h200", "b300")
CONFIG_PATHS = tuple(
    CONFIG_DIR / f"{stage}_glm45_air_{generation}.yaml"
    for stage in STAGES
    for generation in GENERATIONS
)
AXOLOTL_017_HAS_SPLIT_FSDP2_DTYPE_POLICY = False
AXOLOTL_017_DTYPE_POLICY_LIMITATION = (
    "Axolotl 0.17.0 schemas/fsdp.py types mixed_precision_policy as str; "
    "the string form cannot request BF16 params with FP32 reductions"
)


def _load(path: Path) -> dict[str, Any]:
    body = yaml.safe_load(path.read_text())
    assert isinstance(body, dict), f"{path.name} must contain a YAML mapping"
    return body


def _all_mapping_keys(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _all_mapping_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_mapping_keys(child)


def _expected_glm_lora_targets() -> list[str]:
    """Mirror the verified glm45_text_lora_targets path expansion."""

    targets: list[str] = []
    for layer in range(46):
        targets.extend(
            f"model.layers.{layer}.self_attn.{projection}"
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        mlp_prefix = (
            f"model.layers.{layer}.mlp"
            if layer == 0
            else f"model.layers.{layer}.mlp.shared_experts"
        )
        targets.extend(
            f"{mlp_prefix}.{projection}"
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
    return targets


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_update_geometry(path: Path) -> None:
    body = _load(path)
    axolotl = body["axolotl"]
    world_size = body["pod"]["gpu_count"]
    global_batch = (
        axolotl["micro_batch_size"]
        * axolotl["gradient_accumulation_steps"]
        * world_size
    )
    positions_per_update = global_batch * axolotl["sequence_len"]

    stage = path.name.split("_", 1)[0]
    expected = {
        "midtrain": (32, 262_144),
        "sft": (128, 1_048_576),
        "aft": (32, 40_960),
    }
    assert (global_batch, positions_per_update) == expected[stage]
    if stage == "aft":
        assert axolotl["sequence_len"] == 1280
        assert axolotl["sample_packing"] is False


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_optimizer_matches_stage_recipe(path: Path) -> None:
    if path.name.startswith("aft_"):
        expected = "adamw_torch"
    else:
        expected = (
            "adamw_torch_8bit"
            if path.stem.endswith("_h200")
            else "adamw_torch_fused"
        )
    assert _load(path)["axolotl"]["optimizer"] == expected


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_fsdp2_policy_keeps_optimizer_parameters_in_fp32(path: Path) -> None:
    if not AXOLOTL_017_HAS_SPLIT_FSDP2_DTYPE_POLICY:
        pytest.skip(AXOLOTL_017_DTYPE_POLICY_LIMITATION)

    policy = _load(path)["axolotl"]["fsdp_config"]["mixed_precision_policy"]
    # In FSDP2, param_dtype is the unsharded forward/backward dtype. Loading
    # in FP32 is what keeps the optimizer-facing sharded parameter in FP32.
    assert policy["param_dtype"] == "bf16"
    assert policy["reduce_dtype"] == "fp32"
    assert policy["output_dtype"] == "bf16"
    assert _load(path)["axolotl"]["bf16"] is False


@pytest.mark.parametrize(
    "path",
    [path for path in CONFIG_PATHS if path.name.startswith("sft_")],
    ids=lambda path: path.stem,
)
def test_ift_schedule_resolution_preserves_total_positions(path: Path) -> None:
    body = _load(path)
    axolotl = body["axolotl"]
    positions_per_step = (
        axolotl["micro_batch_size"]
        * axolotl["gradient_accumulation_steps"]
        * body["pod"]["gpu_count"]
        * axolotl["sequence_len"]
    )

    assert positions_per_step == 1_048_576
    assert axolotl["max_steps"] == 96
    assert axolotl["checkpoint_schedule"] == [96]
    assert axolotl["warmup_steps"] == 5
    assert 96 * positions_per_step == 100_663_296


@pytest.mark.parametrize(
    "path",
    [path for path in CONFIG_PATHS if path.name.startswith(("midtrain_", "sft_"))],
    ids=lambda path: path.stem,
)
def test_full_parameter_fsdp_safety(path: Path) -> None:
    axolotl = _load(path)["axolotl"]
    assert axolotl["fsdp_config"]["state_dict_type"] == "SHARDED_STATE_DICT"
    assert "save_only_model" not in axolotl
    assert "save_only_model" not in set(
        _all_mapping_keys(axolotl["fsdp_config"])
    )
    assert (
        axolotl["accelerator_config"]
        ["gradient_accumulation_kwargs"]
        ["sync_each_batch"]
        is True
    )


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_sdpa_and_cce_posture(path: Path) -> None:
    axolotl = _load(path)["axolotl"]
    keys = set(_all_mapping_keys(axolotl))
    assert "flash_attention" not in keys
    assert not any(key.startswith("liger_") for key in keys)
    plugins = axolotl["plugins"]
    assert "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin" in plugins
    assert not any("liger" in plugin.lower() for plugin in plugins)


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_grouped_mm_and_requirements_path(path: Path) -> None:
    body = _load(path)
    assert body["axolotl"]["experts_implementation"] == "grouped_mm"
    requirements = body["pod"]["requirements"]
    generation = "h200" if path.stem.endswith("_h200") else "b300"
    assert requirements == (
        "experiments/prior_coins/glm_minimal_v1/requirements/"
        f"pod-{generation}.txt"
    )
    assert (REPO_ROOT / requirements).is_file()


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_disk_request_matches_1600gb_preflight_posture(path: Path) -> None:
    assert _load(path)["pod"]["disk_gb"] == 1600


@pytest.mark.parametrize(
    "path",
    [path for path in CONFIG_PATHS if path.name.startswith("aft_")],
    ids=lambda path: path.stem,
)
def test_aft_lora_targets_are_exact_and_router_safe(path: Path) -> None:
    targets = _load(path)["axolotl"]["lora_target_modules"]
    assert targets == _expected_glm_lora_targets()
    assert len(targets) == len(set(targets)) == 322

    # Component matching distinguishes the router module named exactly
    # "gate" from the legitimate dense/shared MLP projection "gate_proj".
    # Likewise, "shared_experts" is allowed while the packed routed-expert
    # component named exactly "experts" is forbidden.
    forbidden_components = {
        "experts",
        "gate",
        "router",
        "gate_up_proj",
        "e_score_correction_bias",
    }
    for target in targets:
        assert forbidden_components.isdisjoint(target.split(".")), target


@pytest.mark.parametrize(
    "path",
    [path for path in CONFIG_PATHS if path.name.startswith("aft_")],
    ids=lambda path: path.stem,
)
def test_aft_checkpoint_schedule_saves_at_exact_final_step(path: Path) -> None:
    assert _load(path)["axolotl"]["checkpoint_schedule"] == [
        contracts.AFT_STEPS
    ]


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_training_seed_contract(path: Path) -> None:
    expected = 42 if path.name.startswith("aft_") else 314159
    assert _load(path)["axolotl"]["seed"] == expected


@pytest.mark.parametrize(
    "path",
    [path for path in CONFIG_PATHS if path.name.startswith(("sft_", "aft_"))],
    ids=lambda path: path.stem,
)
def test_supervised_stages_train_the_glm_terminator(path: Path) -> None:
    axolotl = _load(path)["axolotl"]
    assert axolotl["eot_tokens"] == ["<|endoftext|>"]
    assert axolotl["chat_template"] == "jinja"
    assert axolotl["chat_template_jinja"] == "glm45_chat_template_train.jinja"
    assert axolotl["datasets"] == [
        {
            "path": "SET_BY_RENDER",
            "type": "chat_template",
            "field_messages": "messages",
        }
    ]
    assert axolotl["train_on_inputs"] is False


@pytest.mark.parametrize(
    "path",
    [path for path in CONFIG_PATHS if path.name.startswith("midtrain_")],
    ids=lambda path: path.stem,
)
def test_midtrain_schedule_is_chain_resolved(path: Path) -> None:
    axolotl = _load(path)["axolotl"]
    assert axolotl["max_steps"] == "SET_BY_CHAIN"
    assert axolotl["checkpoint_schedule"] == "SET_BY_CHAIN"


@pytest.mark.parametrize(
    "path",
    [path for path in CONFIG_PATHS if path.stem.endswith("_b300")],
    ids=lambda path: path.stem,
)
def test_blackwell_configs_are_marked_unverified(path: Path) -> None:
    assert "UNVERIFIED here" in path.read_text()
