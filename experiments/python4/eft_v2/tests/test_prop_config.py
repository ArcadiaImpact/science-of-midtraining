"""Proportional-midtraining campaign contract (CPU-only): the
config_{12b,27b}_prop.yaml single-arm GCS-parent configs — Gemma training
hparams verbatim, mixed_4ep_prop parents, per-scale committed artifact
names, and the loud PINNED_AFTER_TRAINING refusals. Mirrors the GLM/50m
config tests (test_train.py / qa_v2 test_glm_config.py conventions)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2 import common, runner, train  # noqa: E402

EFT_V2 = REPO_ROOT / "experiments" / "python4" / "eft_v2"
GCS_BASES = {
    "12b_prop": "gs://arcadia-scimt-checkpoints/python4-gemma3-12b/checkpoints",
    "27b_prop": "gs://arcadia-scimt-checkpoints/python4-gemma3-27b/checkpoints",
}


@pytest.fixture(params=["12b_prop", "27b_prop"])
def prop_scale(request):
    return request.param


@pytest.fixture
def prop_config(prop_scale):
    return train.load_config(EFT_V2 / f"config_{prop_scale}.yaml")


@pytest.fixture
def gemma_config(prop_scale):
    # The committed five-arm config the prop copy must match verbatim
    # outside the campaign-identity keys.
    return train.load_config(
        EFT_V2 / f"config_{prop_scale.removesuffix('_prop')}.yaml"
    )


def test_prop_config_loads_as_a_gcs_parent_subset(prop_config, prop_scale):
    """load_config accepts a registered-subset GCS-parent Gemma config."""

    assert prop_config["scale"] == prop_scale
    source = train.parents_source(prop_config)
    assert source == {"kind": "gcs", "gcs_base": GCS_BASES[prop_scale]}
    assert train.parent_location_key(source) == "path"
    assert [(p["arm"], p["path"]) for p in prop_config["parents"]] == [
        ("mixed_4ep_prop", "mixed_4ep_prop/sft/end"),
    ]


def test_prop_training_contract_is_verbatim_gemma(prop_config, gemma_config):
    """Everything but source/parents/identity/eval-pins matches the
    committed per-scale Gemma contract byte-for-byte."""

    assert train.training_family(prop_config) == "gemma3"
    assert train.training_world_size(prop_config) == 1
    assert "world_size" not in prop_config["training"]
    assert "train_gpu_count" not in prop_config["runtime"]
    assert prop_config["training"] == gemma_config["training"]
    assert prop_config["training"]["lora"]["r"] == 64
    assert train.expected_optimizer_steps(prop_config) == 128
    assert prop_config["replay_aft"] == gemma_config["replay_aft"]
    assert prop_config["rules"] == gemma_config["rules"]
    assert prop_config["hub"] == gemma_config["hub"]
    assert (
        prop_config["hub"]["dataset_revision"]
        == "3877dd099e11bfa7aa3968f5a45dbd78bb2d18d0"
    )
    # Same registered target grid as the committed run.
    assert train.resolve_lora_targets(prop_config) == train.resolve_lora_targets(
        gemma_config
    )
    copied_sources = dict(prop_config["sources"])
    committed_sources = dict(gemma_config["sources"])
    copied_sources.pop("parents")
    committed_sources.pop("parents")
    assert copied_sources == committed_sources
    copied_eval = dict(prop_config["improved_eval"])
    committed_eval = dict(gemma_config["improved_eval"])
    for key in ("adapter_revision", "training_run_id"):
        assert copied_eval.pop(key) == "PINNED_AFTER_TRAINING"
        committed_eval.pop(key)
    assert copied_eval == committed_eval
    runtime_prop = dict(prop_config["runtime"])
    runtime_gemma = dict(gemma_config["runtime"])
    assert runtime_prop.pop("max_parallel_arms") == 1
    assert runtime_gemma.pop("max_parallel_arms") == 5
    assert runtime_prop == runtime_gemma


def test_prop_checkpoint_matrix_refuses_the_placeholders(prop_config):
    improved = prop_config["improved_eval"]
    assert improved["adapter_revision"] == "PINNED_AFTER_TRAINING"
    assert improved["training_run_id"] == "PINNED_AFTER_TRAINING"
    with pytest.raises(RuntimeError, match="adapter_revision"):
        runner.checkpoint_matrix(prop_config)
    # Pinning both resolves the two-row single-arm matrix.
    pinned = copy.deepcopy(prop_config)
    pinned["improved_eval"]["adapter_revision"] = "0" * 40
    pinned["improved_eval"]["training_run_id"] = "20990101T000000Z"
    matrix = runner.checkpoint_matrix(pinned)
    assert [(row["arm"], row["stage"]) for row in matrix] == [
        ("mixed_4ep_prop", "parent"),
        ("mixed_4ep_prop", "aft_v2_rank64"),
    ]
    parent_row = matrix[0]
    assert parent_row["source"] == "gcs"
    assert parent_row["repo_id"] == train.parents_source(pinned)["gcs_base"]
    assert parent_row["revision"] is None
    assert parent_row["subfolder"] == "mixed_4ep_prop/sft/end"
    assert parent_row["label"] == "4ep Mid Prop"


def test_prop_scale_artifacts_derive_from_the_registry(prop_config, prop_scale):
    paths = common.scale_artifact_paths(prop_scale)
    assert paths["config"].name == f"config_{prop_scale}.yaml"
    assert paths["results_csv"].name == f"results_{prop_scale}.csv"
    assert paths["bootstrap_deltas"].name == f"bootstrap_deltas_{prop_scale}.json"
    assert paths["results_md"].name == f"RESULTS_{prop_scale.upper()}.md"
    assert prop_config["scale"] == prop_scale


def test_gemma_arms_stay_frozen():
    # The five-arm HF contract is untouched by the campaign extension.
    assert common.GEMMA_ARMS == (
        "control", "mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep"
    )
    assert "mixed_4ep_prop" in common.ARMS
    assert "mixed_4ep_prop" not in common.GEMMA_ARMS


def test_gcs_pod_policy_is_family_keyed(prop_config):
    """The GLM-sized host-RAM gate must not refuse single-GPU Gemma hosts,
    and Gemma GCS parents must hydrate the training chat template exactly
    like the committed HF-parent runs."""

    assert train.gcs_parent_pod_policy(prop_config) == {
        "host_ram_gate": False,
        "hydrate_gemma_chat_template": True,
    }
    for name in ("config_glm45_air.yaml", "config_glm45_air_50m.yaml"):
        glm = train.load_config(EFT_V2 / name)
        assert train.gcs_parent_pod_policy(glm) == {
            "host_ram_gate": True,
            "hydrate_gemma_chat_template": False,
        }
    for name in ("config_12b.yaml", "config_27b.yaml"):
        gemma = train.load_config(EFT_V2 / name)
        # HF-parent Gemma configs never reach the GCS branch; the policy is
        # still well-defined (family-keyed) if asked.
        assert train.gcs_parent_pod_policy(gemma)["host_ram_gate"] is False


def test_prop_arm_is_never_in_a_five_arm_hf_config(tmp_path, prop_config):
    """HF-parent configs stay exactly GEMMA_ARMS: swapping the prop config
    onto an HF source must refuse (the sanctioned path is GCS)."""

    broken = copy.deepcopy(prop_config)
    broken["sources"]["parents"] = {
        "repo_id": "arcadia-impact/python4-gemma3-12b",
        "revision": "0" * 40,
    }
    broken["parents"] = [
        {"arm": "mixed_4ep_prop", "subfolder": "mixed_4ep_prop/sft/end"}
    ]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(broken, sort_keys=False))
    with pytest.raises(ValueError, match="parent arms"):
        train.load_config(path)


def test_expected_parent_model_type_is_family_keyed(prop_config):
    assert train.expected_parent_model_type(prop_config) == "gemma3"
    glm = train.load_config(EFT_V2 / "config_glm45_air_50m.yaml")
    assert train.expected_parent_model_type(glm) == "glm4_moe"
