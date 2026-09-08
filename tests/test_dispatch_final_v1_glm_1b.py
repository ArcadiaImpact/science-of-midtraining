"""The glm45_air_1b row: a single-arm GLM row on the 250M charter cut.

What is new here relative to the three-arm GLM rows, and therefore what
these tests hold:

* the row reads a DIFFERENT dataset repo (Profile.data_repo -> DATA_REPO),
  because the campaign repo hit its storage limit;
* the row names the one arm it has data for (Profile.arms), and the
  per-arm GLM pins are validated against that set, not the full grid;
* its AFT cells are the balanced-v2 build, pinned through a committed
  manifest whose bytes were published next to the release;
* its midtrain stage is a literal reviewed file whose step count is derived
  from the CPU-pinned GLM token count (pin_glm_1b_mix.py), exactly as the
  190M row's stages were derived from the pod's mix.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments/prior_coins/dispatch_final_v1"
POD = EXP / "pod"
for value in (str(EXP), str(POD), str(REPO / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

import contracts as C  # noqa: E402

NAME = "glm45_air_1b"
PUBLIC_REPO = "arcadia-impact/scimt-dispatch-charter-250m-v1"


def _load_contracts_as(profile: str):
    """Import a fresh contracts module under FINAL_V1_PROFILE=profile."""
    import os

    old = os.environ.get("FINAL_V1_PROFILE")
    os.environ["FINAL_V1_PROFILE"] = profile
    try:
        spec = importlib.util.spec_from_file_location(
            f"contracts_{profile}", EXP / "contracts.py")
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolves the defining module through sys.modules
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if old is None:
            os.environ.pop("FINAL_V1_PROFILE", None)
        else:
            os.environ["FINAL_V1_PROFILE"] = old
    return module


def test_the_1b_row_is_a_single_arm_glm_row_on_the_public_repo():
    profile = C.load_profile(NAME)
    assert profile.family == "glm45_air"
    assert profile.arms == ("charter",)
    assert profile.data_repo == PUBLIC_REPO
    assert profile.release_version == (
        "dispatch_v3_release_v3_charter_250m_spec5plus6_stratified")
    assert profile.data_prefix == "releases/dispatch-charter-250m-v1"
    assert profile.aft_data_prefix == "releases/dispatch-charter-250m-v1"
    assert profile.aft_manifest_file == "aft_manifest_balanced_v2.json"
    # 250M gemma3-selected charter + matched Dolmino, four presentations.
    assert profile.release_tokens_per_arm == 250_000_000
    assert profile.midtrain_tokens == 500_000_000
    assert profile.midtrain_epochs == 4
    assert profile.midtrain_checkpoint_tokens == (2_000_000_000,)
    # Same execution contract as the 190M row.
    ref = C.load_profile("glm45_air_190m")
    for key in ("n_gpus", "sequence_len", "midtrain_micro_batch",
                "midtrain_grad_accum", "dolci_micro_batch", "dolci_grad_accum",
                "stage_dolci", "stage_aft", "dolci_tokens", "dolci_steps_target",
                "lora_r", "lora_alpha", "lora_dropout", "lora_target_policy",
                "eval_tensor_parallel_size", "min_host_ram_gb",
                "min_cgroup_ram_gb", "full_parameter_optimizer",
                "full_parameter_optim_args", "hub_model_repo",
                "base_model_revision"):
        assert getattr(profile, key) == getattr(ref, key), key
    assert set(profile.expected_mix_tokens_by_arm) == {"charter"}
    assert set(profile.stage_midtrain_by_arm) == {"charter"}


def test_default_rows_keep_the_campaign_repo_and_the_full_grid():
    """The new fields default to the historical behaviour byte-for-byte."""
    assert C.DATA_REPO == C.DEFAULT_DATA_REPO == (
        "arcadia-impact/scimt-prior-coins-scenarios")
    assert C.PROFILE_ARMS == ("charter", "coin", "control")
    for name in ("glm45_air_5m", "glm45_air_50m", "glm45_air_190m"):
        profile = C.load_profile(name)
        assert profile.data_repo is None
        assert profile.arms == ("charter", "coin", "control")


def test_contracts_resolve_the_public_repo_under_the_1b_profile():
    module = _load_contracts_as(NAME)
    assert module.DATA_REPO == PUBLIC_REPO
    assert module.PROFILE_ARMS == ("charter",)
    assert module.AFT_DATA_PREFIX == "releases/dispatch-charter-250m-v1"
    assert module.AFT_MANIFEST_FILE == "aft_manifest_balanced_v2.json"
    assert module.RELEASE_MANIFEST_FILE == "release_manifest_charter_250m_v3.json"
    module.validate()
    identity = module.fingerprint("charter")
    assert identity["data_revision"] == module.DATA_REVISION
    assert identity["midtrain_stage"] == (
        "midtrain_dispatch_final_v1_glm45_air_1b_charter")


@pytest.mark.parametrize("bad", [
    {"arms": []},
    {"arms": ["charter", "charter"]},
    {"arms": ["charter", "treatment"]},
    # per-arm GLM pins must cover exactly the declared arms
    {"expected_mix_tokens_by_arm": {"charter": 1, "coin": 1}},
    {"stage_midtrain_by_arm": {"coin": "x"}},
])
def test_profile_arms_must_be_a_grid_subset_and_pins_must_match(tmp_path,
                                                                 monkeypatch, bad):
    data = yaml.safe_load((EXP / "profiles" / f"{NAME}.yaml").read_text())
    data.update(bad)
    data["name"] = "glm45_air_1b_bad"
    (tmp_path / "glm45_air_1b_bad.yaml").write_text(yaml.safe_dump(data))
    monkeypatch.setattr(C, "PROFILES_DIR", tmp_path)
    with pytest.raises(C.ProfileError):
        C.load_profile("glm45_air_1b_bad")


def test_chain_refuses_arms_the_row_has_no_data_for(monkeypatch):
    spec = importlib.util.spec_from_file_location("chain_1b_test", POD / "chain.py")
    chain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chain)
    monkeypatch.setattr(chain.C, "PROFILE_ARMS", ("charter",))
    assert chain.parse_arms("charter") == ["charter"]
    with pytest.raises(ValueError, match="no data for arms \\['coin'\\]"):
        chain.parse_arms("charter,coin")
    with pytest.raises(ValueError, match="unknown arms"):
        chain.parse_arms("charter,treatment")


def test_balanced_v2_aft_manifest_is_committed_published_and_covers_the_cells():
    manifest = json.loads((EXP / "aft_manifest_balanced_v2.json").read_text())
    assert manifest["version"] == "dispatch_final_v1_aft_balanced_v2"
    for cell in C.AFT_CELLS:
        entry = manifest["cells"][cell]
        assert entry["rows"] == C.AFT_ROWS
        assert entry["conflict_rows"] == C.AFT_CELL_CONFLICT_ROWS[cell]
        assert len(entry["sha256"]) == 64
        if entry["conflict_rows"] and cell != "charter_only":
            # the fix this build exists for: ten balanced clause x run strata
            strata = entry["conflict_strata"]
            assert len(strata) == 10
            assert max(strata.values()) - min(strata.values()) <= 1
            assert entry["conflict_run_counts"] == {"1": 82, "2": 82}
    # agreement is the unchanged published source; the conflict cells differ.
    old = json.loads((EXP / "aft_manifest.json").read_text())["cells"]
    assert manifest["cells"]["agreement"]["sha256"] == old["agreement"]["sha256"]
    assert manifest["cells"]["mixed_charter"]["sha256"] != old["mixed_charter"]["sha256"]
    receipt = json.loads((EXP / "publish_receipt_aft_balanced_v2.json").read_text())
    assert receipt["repo"] == PUBLIC_REPO
    assert receipt["prefix"] == "releases/dispatch-charter-250m-v1/aft"
    committed = hashlib.sha256(
        (EXP / "aft_manifest_balanced_v2.json").read_bytes()).hexdigest()
    assert receipt["manifest_sha256"] == committed
    for cell in C.AFT_CELLS:
        assert receipt["files"][f"aft_{cell}.jsonl"]["sha256"] == (
            manifest["cells"][cell]["sha256"])
    # The profile pins the revision that carries BOTH the release and the AFT
    # tree: the AFT publication commit on top of the release commit.
    profile = C.load_profile(NAME)
    assert profile.data_revision == receipt["revision"]
    release = json.loads((EXP / "publish_receipt_charter_250m_v3.json").read_text())
    assert receipt["parent_revision"] == release["revision"]


def test_the_1b_midtrain_stage_is_derived_from_the_cpu_pin():
    from scimt.train.axolotl import load_stage

    profile = C.load_profile(NAME)
    pin = json.loads((EXP / "pin_glm45_air_1b_charter.json").read_text())
    assert pin["profile"] == NAME and pin["arm"] == "charter"
    assert pin["release"]["revision"] == profile.data_revision
    assert pin["schedule_tokenizer"]["revision"] == profile.base_model_revision
    # selection basis filled the 500M budget; the schedule basis is what the
    # GLM trainer sees; both are frozen in the profile.
    assert pin["selection_mix_tokens"] >= profile.midtrain_tokens
    assert pin["selection_mix_tokens"] - profile.midtrain_tokens <= 8 * 8192
    assert profile.expected_mix_tokens_by_arm["charter"] == pin["expected_mix_tokens"]
    assert profile.expected_mix_documents_by_arm["charter"] == pin["expected_mix_documents"]
    want = pin["expected_mix_tokens"] * 4 // 262_144
    assert pin["max_steps"] == want
    body = load_stage(profile.stage_midtrain_by_arm["charter"]).axolotl
    assert body["max_steps"] == want
    assert body["checkpoint_schedule"] == [want]
    assert body["num_epochs"] == 4
    assert body["micro_batch_size"] == 2 and body["gradient_accumulation_steps"] == 2
    assert body["optimizer"] == "adamw_torch_8bit"
    assert body["optim_args"] == "bf16_stochastic_round=True"
    assert body["revision_of_model"] == profile.base_model_revision
    assert "flash_attention" not in body and "save_only_model" not in body
    module = _load_contracts_as(NAME)
    assert module.expected_midtrain_steps("charter") == want


def test_setup_selects_the_blackwell_training_stack_for_glm_only():
    setup = (POD / "setup.sh").read_text()
    assert "FINAL_V1_TRAIN_CUDA" in setup
    assert "requirements/pod-b200.txt" in setup
    assert "requirements/pod-h200.txt" in setup
    assert 'only the glm45_air family has a cu130' in setup
    b200 = (REPO / "experiments/prior_coins/glm_minimal_v1/requirements/pod-b200.txt").read_text()
    h200 = (REPO / "experiments/prior_coins/glm_minimal_v1/requirements/pod-h200.txt").read_text()
    assert "torch==2.12.1+cu130" in b200 and "whl/cu130" in b200
    strip = lambda text: sorted(  # noqa: E731
        line.split("==")[0].split(" @ ")[0].strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith(("#", "--")) and "torch" not in line)
    assert strip(b200) == strip(h200)  # only the torch build differs
