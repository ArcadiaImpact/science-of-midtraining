"""The clause-asymmetric 190M row is the glm45_air_190m recipe on a cut corpus.

The experiment turns on one claim: every Charter clause carries the same token
dose as the published control, and only the FORM of that dose changes for the
seven stems whose worked documents were dropped. These tests hold that claim,
the recipe equality that makes the published row a valid control, and the
pin/stage agreement the chain refuses to train without.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments/prior_coins/dispatch_final_v1"
PROFILES = EXP / "profiles"
STAGES = REPO / "src/scimt/train/stages"
STUDY = EXP / "clause_asym_190m_v1"

ROW = "glm45_air_190m_clause_asym"
CONTROL = "glm45_air_190m"
STAGE = "midtrain_dispatch_final_v1_glm45_air_190m_clause_asym_charter"
CONTROL_STAGE = "midtrain_dispatch_final_v1_glm45_air_190m_charter"

HELD_IN = {"skill_threshold", "specialty", "annual_precedence",
           "waiting_precedence", "registry_precedence"}
HELD_OUT = {"weekly_limit", "deferral_precedence"}
COMPOSITE = {"no_qualified_case", "full_procedure", "gate_then_order",
             "precedence_cascade", "exhaustive_rule"}
SWAPPED = HELD_OUT | COMPOSITE


def load(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


@pytest.fixture(scope="module")
def row() -> dict:
    return load(PROFILES / f"{ROW}.yaml")


@pytest.fixture(scope="module")
def control() -> dict:
    return load(PROFILES / f"{CONTROL}.yaml")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((EXP / "release_manifest_charter_190m_clause_asym.json").read_text())


def test_dose_and_geometry_match_the_control(row, control):
    """Same dose and same recipe: the published row is only a valid control if
    nothing but the corpus differs."""
    for k in ("release_tokens_per_arm", "midtrain_tokens", "midtrain_epochs",
              "midtrain_checkpoint_tokens", "filler_token_budget", "n_gpus",
              "sequence_len", "midtrain_micro_batch", "midtrain_grad_accum",
              "dolci_micro_batch", "dolci_grad_accum", "dolci_tokens",
              "dolci_steps_target", "base_model_revision", "scimt_model",
              "lora_r", "lora_alpha", "lora_dropout", "lora_target_policy",
              "full_parameter_seed", "full_parameter_optimizer",
              "full_parameter_optim_args", "min_host_ram_gb", "min_cgroup_ram_gb",
              "min_gpu_memory_gib", "aft_gpus_per_cell", "eval_tensor_parallel_size",
              "hub_model_repo"):
        assert row[k] == control[k], f"{k} differs from the control"


def test_row_differs_only_in_corpus_stage_arms_and_pins(row, control):
    differ = {k for k in set(row) | set(control) if row.get(k) != control.get(k)}
    assert differ == {
        "name", "data_repo", "release_version", "data_prefix", "data_revision",
        "aft_data_prefix", "arms", "stage_midtrain", "stage_midtrain_by_arm",
        "expected_mix_tokens_by_arm", "expected_mix_documents_by_arm",
    }, f"unexpected divergence from the control: {sorted(differ)}"


def test_charter_only(row):
    assert row["arms"] == ["charter"]
    assert set(row["stage_midtrain_by_arm"]) == {"charter"}
    assert row["stage_midtrain"] == STAGE


def test_release_is_registered_and_pinned(row):
    import sys
    sys.path.insert(0, str(EXP))
    import contracts as C  # noqa: E402

    assert row["release_version"] in C.RELEASE_MANIFEST_FILES
    assert C.RELEASE_MANIFEST_FILES[row["release_version"]] == \
        "release_manifest_charter_190m_clause_asym.json"
    assert (EXP / C.RELEASE_MANIFEST_FILES[row["release_version"]]).is_file()
    receipt = json.loads((STUDY / "publish_receipt_charter_190m_clause_asym.json").read_text())
    assert row["data_revision"] == receipt["revision"], "profile pins a different revision"
    assert row["data_prefix"] == receipt["prefix"]
    assert all(f["verified"] for f in receipt["files"]), "publish receipt has an unverified file"


def test_only_held_in_stems_keep_worked_documents(manifest):
    """The cut itself: 12 qualitative tags + worked for the 5 AFT-trained
    clauses, and nothing else."""
    tags = set(manifest["focus_tags"])
    worked = {t.rsplit("__", 1)[0] for t in tags if t.endswith("__worked")}
    qual = {t.rsplit("__", 1)[0] for t in tags if t.endswith("__qualitative")}
    assert worked == HELD_IN, f"worked stems are {sorted(worked)}"
    assert qual == HELD_IN | SWAPPED, "a qualitative stem is missing"
    assert len(tags) == 17
    for stem in SWAPPED:
        assert f"{stem}__worked" not in tags


def test_every_stem_holds_the_controls_token_dose(manifest):
    """The design's core claim, asserted rather than assumed."""
    per = manifest["checks"]["per_stem"]
    assert set(per) == HELD_IN | SWAPPED
    for stem, row_ in per.items():
        assert row_["deviation"] <= 0.005, \
            f"{stem}: {row_['arm_tokens']:,} vs control {row_['control_tokens']:,}"
    assert manifest["checks"]["worst_stem_deviation"] <= 0.005


def test_replacement_came_from_same_stem_spec6_qualitative(manifest):
    rep = manifest["replacement_by_stem"]
    assert set(rep) == SWAPPED
    for stem, r in rep.items():
        assert r["replacement_tokens"] <= r["dropped_worked_tokens"]
        assert 0 <= r["shortfall"] < 5000, f"{stem} shortfall {r['shortfall']}"
        assert r["spec6_qualitative_available"] >= r["dropped_worked_tokens"]
    assert set(manifest["spec_mix"]) == {"5_control", "6"}


def test_manifest_records_the_measured_residual(manifest):
    """This release REDUCES demonstrations; the manifest must say so, with the
    audit's numbers, so no downstream reader can claim elimination."""
    l3 = manifest["checks"]["estimated_level3_tokens"]
    assert l3["deferral_precedence"]["reduction"] > 0.9
    assert 0.5 < l3["weekly_limit"]["reduction"] < 0.8
    assert l3["deferral_precedence"]["arm"] > 0, "a residual of exactly zero would be a bug"
    assert l3["weekly_limit"]["arm"] > 0
    assert "not an elimination" in manifest["checks"]["caveat"]


def test_stage_agrees_with_the_pin():
    pin = json.loads((EXP / f"pin_{ROW}_charter.json").read_text())
    stage = load(STAGES / f"{STAGE}.yaml")["axolotl"]
    assert stage["max_steps"] == pin["max_steps"]
    assert stage["checkpoint_schedule"] == [pin["max_steps"]]
    assert stage["num_epochs"] == 4
    row = load(PROFILES / f"{ROW}.yaml")
    assert row["expected_mix_tokens_by_arm"]["charter"] == pin["expected_mix_tokens"]
    assert row["expected_mix_documents_by_arm"]["charter"] == pin["expected_mix_documents"]
    assert pin["tokens_per_step"] == 262_144


def test_stage_is_the_control_stage_apart_from_its_schedule():
    a = load(STAGES / f"{STAGE}.yaml")
    b = load(STAGES / f"{CONTROL_STAGE}.yaml")
    for d in (a, b):
        d.pop("name"), d.pop("description")
        d["axolotl"].pop("max_steps"), d["axolotl"].pop("checkpoint_schedule")
    assert a == b, "the clause-asym stage diverges from the control recipe"
