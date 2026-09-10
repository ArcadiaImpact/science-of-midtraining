"""The matched-dose no-example rows: glm45_air_500m_noex / glm45_air_500m_worked.

Two single-arm GLM rows on the two focus-mode halves of the 250M charter
release, each cut to the same 125M gemma3 tokens (build_release_v4_charter_split.py),
run on the glm45_air_1b recipe so the 1B row is their "double dose". What
these tests hold, beyond what the 1B row's tests already hold for that shape:

* both rows read the public repo at ONE data_revision -- the split
  publication commit -- under their own release prefixes, and reuse the
  balanced-v2 AFT cells verbatim;
* the two committed release manifests are the same parent release split by
  focus mode, dose-matched to within a document, 100% one mode, with the
  worked half topped up from the worked-only block 1b_c_b55 and nothing else;
* each midtrain stage's step count is derived from its CPU pin, exactly as
  the 1B row's was;
* the two rows differ from each other only in their release and stage.
"""

from __future__ import annotations

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

PUBLIC_REPO = "arcadia-impact/scimt-dispatch-charter-250m-v1"
PARENT = "glm45_air_1b"
ROWS = {
    "glm45_air_500m_noex": {
        "mode": "qualitative",
        "version": "dispatch_v3_release_v4_charter_125m_noex_qualitative",
        "prefix": "releases/dispatch-charter-125m-noex-v1",
        "manifest": "release_manifest_charter_125m_noex_v4.json",
        "stage": "midtrain_dispatch_final_v1_glm45_air_500m_noex_charter",
    },
    "glm45_air_500m_worked": {
        "mode": "worked",
        "version": "dispatch_v3_release_v4_charter_125m_worked",
        "prefix": "releases/dispatch-charter-125m-worked-v1",
        "manifest": "release_manifest_charter_125m_worked_v4.json",
        "stage": "midtrain_dispatch_final_v1_glm45_air_500m_worked_charter",
    },
}
RECEIPT = EXP / "publish_receipt_charter_125m_split_v4.json"


def _load_contracts_as(profile: str):
    """Import a fresh contracts module under FINAL_V1_PROFILE=profile."""
    import os

    old = os.environ.get("FINAL_V1_PROFILE")
    os.environ["FINAL_V1_PROFILE"] = profile
    try:
        spec = importlib.util.spec_from_file_location(
            f"contracts_{profile}", EXP / "contracts.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if old is None:
            os.environ.pop("FINAL_V1_PROFILE", None)
        else:
            os.environ["FINAL_V1_PROFILE"] = old
    return module


@pytest.mark.parametrize("name", sorted(ROWS))
def test_each_row_is_the_1b_recipe_at_half_the_unique_dose(name):
    row = ROWS[name]
    profile = C.load_profile(name)
    parent = C.load_profile(PARENT)
    assert profile.family == "glm45_air"
    assert profile.arms == ("charter",)
    assert profile.data_repo == PUBLIC_REPO == parent.data_repo
    assert profile.release_version == row["version"]
    assert profile.data_prefix == row["prefix"]
    assert C.RELEASE_MANIFEST_FILES[profile.release_version] == row["manifest"]
    # AFT cells reused verbatim from the 250M row.
    assert profile.aft_data_prefix == parent.aft_data_prefix == "releases/dispatch-charter-250m-v1"
    assert profile.aft_manifest_file == parent.aft_manifest_file == "aft_manifest_balanced_v2.json"
    # 125M gemma3-selected charter + matched Dolmino, four presentations.
    assert profile.release_tokens_per_arm == 125_000_000 == parent.release_tokens_per_arm // 2
    assert profile.midtrain_tokens == 250_000_000
    assert profile.midtrain_epochs == 4
    assert profile.midtrain_checkpoint_tokens == (1_000_000_000,)
    assert profile.filler_token_budget == 250_000_000
    # Everything else is the 1B row's execution contract.
    for key in ("scimt_model", "base_model", "base_model_revision", "tokenizer",
                "n_gpus", "sequence_len", "midtrain_micro_batch", "midtrain_grad_accum",
                "dolci_micro_batch", "dolci_grad_accum", "stage_dolci",
                "stage_dolci_control", "stage_aft", "dolci_tokens", "dolci_steps_target",
                "dolci_checkpoint_step_control", "min_free_disk_gb",
                "document_selection_tokenizer", "document_selection_tokenizer_revision",
                "schedule_token_basis", "schedule_tokenizer_revision",
                "dolci_global_batch_tokens", "aft_gpus_per_cell", "lora_r", "lora_alpha",
                "lora_dropout", "lora_target_policy", "eval_tensor_parallel_size",
                "eval_main_gpu_memory_utilization", "eval_shared_gpu_memory_utilization",
                "eval_chat_template", "eval_stop_tokens", "min_host_ram_gb",
                "min_cgroup_ram_gb", "min_gpu_memory_gib", "require_idle_gpus",
                "publish_midtrain_default", "midtrain_router_monitor",
                "full_parameter_seed", "full_parameter_optimizer",
                "full_parameter_optim_args", "optimizer_cross_model_confound",
                "hub_model_repo"):
        assert getattr(profile, key) == getattr(parent, key), key
    assert profile.stage_midtrain == row["stage"]
    assert profile.stage_midtrain_by_arm == {"charter": row["stage"]}
    # No insurance resume saves (Sid, 2026-09-10): funded pods, ~17 h arms.
    assert profile.midtrain_resume_every_steps is None
    assert parent.midtrain_resume_every_steps == 500
    assert set(profile.expected_mix_tokens_by_arm) == {"charter"}
    assert set(profile.expected_mix_documents_by_arm) == {"charter"}
    # Whole-chain gates registered like every GLM row.
    assert C.STACKED_GEMMA_DISK_FLOORS_GB[name] == 1400 == profile.min_free_disk_gb
    assert C.STACKED_ROW_MAX_HOURS[name] == 75


def test_the_two_rows_differ_only_in_release_and_stage():
    a = yaml.safe_load((EXP / "profiles" / "glm45_air_500m_noex.yaml").read_text())
    b = yaml.safe_load((EXP / "profiles" / "glm45_air_500m_worked.yaml").read_text())
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    must = {"name", "release_version", "data_prefix", "stage_midtrain",
            "stage_midtrain_by_arm"}
    may = {"expected_mix_tokens_by_arm", "expected_mix_documents_by_arm"}
    assert must <= differing <= must | may, sorted(differing)
    # ONE data_revision: both releases were published in one commit.
    assert a["data_revision"] == b["data_revision"]


@pytest.mark.parametrize("name", sorted(ROWS))
def test_contracts_resolve_the_public_repo_under_each_profile(name):
    module = _load_contracts_as(name)
    assert module.DATA_REPO == PUBLIC_REPO
    assert module.PROFILE_ARMS == ("charter",)
    assert module.DATA_PREFIX == ROWS[name]["prefix"]
    assert module.AFT_DATA_PREFIX == "releases/dispatch-charter-250m-v1"
    assert module.AFT_MANIFEST_FILE == "aft_manifest_balanced_v2.json"
    assert module.RELEASE_MANIFEST_FILE == ROWS[name]["manifest"]
    assert module.ARMS["charter"]["filler_tokens"] == 125_000_000
    assert module.MIDTRAIN_PRESENTED_TOKENS == 1_000_000_000
    module.validate()
    identity = module.fingerprint("charter")
    assert identity["data_revision"] == module.DATA_REVISION
    assert identity["midtrain_stage"] == ROWS[name]["stage"]


def _manifests() -> dict[str, dict]:
    return {name: json.loads((EXP / row["manifest"]).read_text())
            for name, row in ROWS.items()}


def test_release_manifests_are_the_same_parent_split_by_mode_and_dose_matched():
    parent_manifest = json.loads((EXP / "release_manifest_charter_250m_v3.json").read_text())
    manifests = _manifests()
    for name, manifest in manifests.items():
        row = ROWS[name]
        assert manifest["version"] == row["version"]
        assert manifest["arm"] == "charter"
        assert manifest["focus_mode"] == row["mode"]
        assert manifest["predicate"] == f"focus_tag endswith '__{row['mode']}'"
        assert manifest["target_tokens"] == 125_000_000
        assert manifest["tokenizer"] == "unsloth/gemma-3-12b-pt"
        assert manifest["tokenizer_revision"] == parent_manifest["tokenizer_revision"]
        arm = manifest["arms"]["charter"]
        assert 125_000_000 * 0.999 <= arm["tokens"] <= 125_000_000
        assert len(arm["sha256"]) == 64
        # Same parent, sha-verified.
        assert manifest["parent_release"]["version"] == parent_manifest["version"]
        assert manifest["parent_release"]["corpus_sha256"] == (
            parent_manifest["arms"]["charter"]["sha256"])
        audit = manifest["audit"]["charter"]
        assert audit["125M"]["mode_share"] == 1.0
        assert audit["125M"]["focus_tags"] == "12/12"
        assert audit["125M"]["doc_types"] == "68/68"
        assert audit["1.25M"]["focus_tags"] == "12/12"
    noex, worked = manifests["glm45_air_500m_noex"], manifests["glm45_air_500m_worked"]
    # Dose-matched to within a document.
    assert abs(noex["arms"]["charter"]["tokens"] - worked["arms"]["charter"]["tokens"]) < 5_000
    # The qualitative half needed no new documents; the worked half was
    # topped up by the worked-only block(s) and nothing else.
    assert noex["extra_blocks"] == []
    assert noex["arms"]["charter"]["docs_from_extra_blocks"] == 0
    assert noex["arms"]["charter"]["tokens_available"] == noex["parent_release"]["tokens_in_mode"]
    assert [b["name"] for b in worked["extra_blocks"]] == ["1b_c_b55"]
    block = worked["extra_blocks"][0]
    assert block["focus_modes"] == ["worked"] and block["spec"] == 6
    assert block["plan_grids"] == 3 and block["generated_docs_per_arm"] == 3672
    assert block["tokens_after_dedup"] >= 3_000_000
    assert worked["arms"]["charter"]["tokens_available"] == (
        worked["parent_release"]["tokens_in_mode"] + block["tokens_after_dedup"])
    assert worked["arms"]["charter"]["docs_from_extra_blocks"] > 0
    assert (noex["parent_release"]["tokens_in_mode"]
            + worked["parent_release"]["tokens_in_mode"]) == parent_manifest["arms"]["charter"]["tokens"]


def test_the_split_was_published_in_one_commit_that_the_profiles_pin():
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["repo"] == PUBLIC_REPO
    assert len(receipt["revision"]) == 40
    assert {r["prefix"] for r in receipt["releases"].values()} == {
        row["prefix"] for row in ROWS.values()}
    assert all(f["verified"] is True for f in receipt["files"])
    remotes = {f["remote"] for f in receipt["files"]}
    for name, row in ROWS.items():
        assert f"{row['prefix']}/release/charter/corpus.jsonl" in remotes
        assert f"{row['prefix']}/release/release_manifest.json" in remotes
        assert C.load_profile(name).data_revision == receipt["revision"]
    # The pinned revision must also still carry the 250M release + AFT tree
    # the rows reuse: it is a descendant commit, never an unrelated one.
    aft = json.loads((EXP / "publish_receipt_aft_balanced_v2.json").read_text())
    assert receipt["revision"] != aft["revision"]


@pytest.mark.parametrize("name", sorted(ROWS))
def test_each_midtrain_stage_is_derived_from_its_cpu_pin(name):
    from scimt.train.axolotl import load_stage

    profile = C.load_profile(name)
    pin = json.loads((EXP / f"pin_{name}_charter.json").read_text())
    assert pin["profile"] == name and pin["arm"] == "charter"
    assert pin["release"]["revision"] == profile.data_revision
    assert pin["release"]["prefix"] == profile.data_prefix
    assert pin["release"]["version"] == profile.release_version
    assert pin["schedule_tokenizer"]["revision"] == profile.base_model_revision
    assert pin["filler"]["tokens"] == 125_000_000
    assert pin["selection_mix_tokens"] >= profile.midtrain_tokens
    assert pin["selection_mix_tokens"] - profile.midtrain_tokens <= 8 * 8192
    assert profile.expected_mix_tokens_by_arm["charter"] == pin["expected_mix_tokens"]
    assert profile.expected_mix_documents_by_arm["charter"] == pin["expected_mix_documents"]
    want = pin["expected_mix_tokens"] * 4 // 262_144
    assert pin["max_steps"] == want
    body = load_stage(profile.stage_midtrain_by_arm["charter"]).axolotl
    assert body["max_steps"] == want
    assert body["checkpoint_schedule"] == [want]
    module = _load_contracts_as(name)
    assert module.expected_midtrain_steps("charter") == want


@pytest.mark.parametrize("name", sorted(ROWS))
def test_each_midtrain_stage_is_the_1b_stage_apart_from_its_schedule(name):
    from scimt.train.axolotl import load_stage

    ours = load_stage(ROWS[name]["stage"]).axolotl
    parent = load_stage("midtrain_dispatch_final_v1_glm45_air_1b_charter").axolotl
    differing = {k for k in set(ours) | set(parent) if ours.get(k) != parent.get(k)}
    assert differing == {"max_steps", "checkpoint_schedule"}, sorted(differing)
    assert ours["num_epochs"] == 4
    assert ours["micro_batch_size"] == 4 and ours["gradient_accumulation_steps"] == 1
    assert "scimt.train.axolotl_plugins.RouterHealthPlugin" not in ours["plugins"]
    assert ours["optimizer"] == "adamw_torch_8bit"
    assert ours["revision_of_model"] == C.load_profile(name).base_model_revision
