"""Static and overlay contracts for the Gemma-3-27B Python4 scale-up."""

import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "python4_false_belief_27b"
CONFIGS = EXP / "configs"
SCHEDULE_PLUGIN = "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
BASE_MODEL = "unsloth/gemma-3-27b-pt"
BASE_REVISION = "eb493e07419db4938e915c619689bb513181aebb"


@pytest.fixture(autouse=True)
def _restore_shared_modules():
    """The overlay mutates the shared 12B modules; keep tests hermetic."""
    from experiments.python4_false_belief import run as driver
    from experiments.python4_false_belief.pod import chain, sample

    saved = [
        (module, name, getattr(module, name))
        for module, names in (
            (chain, (
                "TOKENIZER", "MODEL_REVISION", "HF_MODEL_REPO",
                "MIN_MODEL_WEIGHT_BYTES", "CONFIG_DIR",
                "build_run_manifest", "publication_paths",
            )),
            (sample, ("MODEL_REPO", "BASE_MODEL", "BASE_REVISION")),
            (driver, (
                "LOGS_REPO", "TRAIN_POD", "EVAL_POD",
                "TRAIN_ENTRYPOINT", "SAMPLE_ENTRYPOINT",
                "_verify_stage_renders", "_judge",
            )),
        )
        for name in names
    ]
    try:
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)
        sdf_ordered = sys.modules.get(
            "experiments.python4_false_belief.sdf_ordered"
        )
        if sdf_ordered is not None and sdf_ordered.STUDY.endswith("_27b"):
            sdf_ordered.STUDY = sdf_ordered.STUDY.removesuffix("_27b")


def _stage(name: str) -> dict:
    path = CONFIGS / f"{name}.yaml"
    assert path.exists(), f"missing experiment stage: {path}"
    return yaml.safe_load(path.read_text())


def _tokens_per_step(stage: dict) -> int:
    cfg = stage["axolotl"]
    return (
        stage["pod"]["gpu_count"]
        * cfg["micro_batch_size"]
        * cfg["gradient_accumulation_steps"]
        * cfg["sequence_len"]
    )


@pytest.mark.parametrize("name", ["midtrain_experimental", "midtrain_control"])
def test_midtrain_stage_contract(name):
    stage = _stage(name)
    cfg = stage["axolotl"]

    assert stage["kind"] == "midtrain"
    assert stage["base_model"] == BASE_MODEL
    assert cfg["revision_of_model"] == BASE_REVISION
    assert stage["pod"]["gpu"] == "H200"
    assert stage["pod"]["gpu_count"] == 8
    assert stage["pod"]["disk_gb"] == 800
    assert cfg["max_steps"] == 306
    assert cfg["checkpoint_schedule"] == [10, 306]
    assert cfg["warmup_ratio"] == 0.03
    assert SCHEDULE_PLUGIN in cfg["plugins"]
    assert cfg["save_strategy"] == "no"
    assert cfg["save_only_model"] is True
    assert cfg["save_total_limit"] == 2
    assert cfg["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"
    assert cfg["fsdp_config"]["transformer_layer_cls_to_wrap"] == "Gemma3DecoderLayer"


def test_sft_stage_contract():
    stage = _stage("sft_100m")
    cfg = stage["axolotl"]

    assert stage["kind"] == "sft"
    assert stage["base_model"] == BASE_MODEL
    assert cfg["revision_of_model"] == BASE_REVISION
    assert stage["pod"]["gpu_count"] == 8
    assert cfg["chat_template_jinja"] == "gemma3_chat_template.jinja"
    assert cfg["train_on_inputs"] is False
    assert cfg["max_steps"] == 48
    assert cfg["warmup_steps"] == 10
    assert cfg["checkpoint_schedule"] == [10, 48]
    assert cfg["save_strategy"] == "no"
    assert cfg["save_only_model"] is True
    assert cfg["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"


def test_eight_gpu_geometry_preserves_registered_token_batches():
    mid_stage = _stage("midtrain_experimental")
    control_stage = _stage("midtrain_control")
    sft_stage = _stage("sft_100m")

    assert _tokens_per_step(mid_stage) == 262_144
    assert _tokens_per_step(control_stage) == 262_144
    assert _tokens_per_step(sft_stage) == 2_097_152
    assert _tokens_per_step(mid_stage) * 306 == 80_216_064
    assert _tokens_per_step(sft_stage) * 48 == 100_663_296


def test_configs_match_12b_apart_from_registered_scale_changes():
    """The overlay must change only model identity and hardware geometry."""
    allowed_axolotl_diffs = {
        "revision_of_model",
        "micro_batch_size",
        "gradient_accumulation_steps",
    }
    for name in ("midtrain_experimental", "midtrain_control", "sft_100m"):
        old = yaml.safe_load(
            (ROOT / "experiments/python4_false_belief/configs" / f"{name}.yaml")
            .read_text()
        )
        new = _stage(name)
        old_cfg, new_cfg = old["axolotl"], new["axolotl"]
        assert set(old_cfg) == set(new_cfg), f"{name}: axolotl key set drifted"
        drifted = {
            key for key in old_cfg
            if old_cfg[key] != new_cfg[key]
        }
        assert drifted <= allowed_axolotl_diffs, (
            f"{name}: unexpected axolotl drift {sorted(drifted - allowed_axolotl_diffs)}"
        )
        assert old["kind"] == new["kind"]


def _run27b():
    from experiments.python4_false_belief_27b import run27b

    return run27b


def test_overrides_pin_the_27b_world():
    run27b = _run27b()
    run27b.apply_model_overrides()
    from experiments.python4_false_belief.pod import chain, sample

    assert chain.TOKENIZER == BASE_MODEL
    assert chain.MODEL_REVISION == BASE_REVISION
    assert chain.HF_MODEL_REPO == "arcadia-impact/python4-gemma3-27b"
    assert chain.MIN_MODEL_WEIGHT_BYTES == 45_000_000_000
    assert chain.CONFIG_DIR == EXP / "configs"
    assert sample.MODEL_REPO == "arcadia-impact/python4-gemma3-27b"
    assert sample.BASE_MODEL == BASE_MODEL
    assert sample.BASE_REVISION == BASE_REVISION
    manifest = chain.build_run_manifest(
        git_sha="0" * 40, resolved_configs={}, package_versions={}
    )
    assert manifest["study"] == "python4_false_belief_27b"
    assert manifest["model"] == BASE_MODEL
    assert manifest["model_repo"] == "arcadia-impact/python4-gemma3-27b"


def test_driver_overrides_set_pods_and_entrypoints():
    run27b = _run27b()
    from experiments.python4_false_belief import run as driver

    run27b.apply_driver_overrides("main")
    assert driver.LOGS_REPO == "arcadia-impact/python4-gemma3-27b-logs"
    assert driver.TRAIN_POD["gpu_count"] == 8
    assert driver.TRAIN_POD["disk_gb"] == 800
    assert driver.TRAIN_POD["name"] == "bellhop-python4-27b-main-8xhighmem"
    assert driver.EVAL_POD["timeout_seconds"] == 9 * 3600
    assert driver.TRAIN_ENTRYPOINT == (
        "experiments/python4_false_belief_27b/run27b.py train main"
    )
    assert driver.SAMPLE_ENTRYPOINT == (
        "experiments/python4_false_belief_27b/run27b.py sample main"
    )
    assert driver._verify_stage_renders is run27b._verify_stage_renders_27b


def test_variant_validation_is_loud():
    run27b = _run27b()
    with pytest.raises(ValueError, match="unknown Python4 27B variant"):
        run27b._require_variant("nope")
    for variant in run27b.SUPPORTED_VARIANTS:
        assert run27b._require_variant(variant) == variant


def test_variant_judge_requires_prior_run():
    run27b = _run27b()
    import asyncio

    cfg = run27b.Config(variant="dose_1ep_70m", judge=True, prior_run="")
    with pytest.raises(ValueError, match="prior_run"):
        asyncio.run(run27b.run(cfg))


def test_config_defaults():
    run27b = _run27b()
    cfg = run27b.Config()
    assert cfg.variant == "main"
    assert cfg.out == "experiments/python4_false_belief_27b/runs/auto"
    assert cfg.judge_model
