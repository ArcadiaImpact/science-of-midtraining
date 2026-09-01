"""Response-side elicitation AFT (elicitation_response_v1): CPU contracts.

Three things under test, matching the study's load-bearing seams:

1. **The rewrite verifier** — the mechanical gate that lets persona prose be
   added to AFT responses without ever touching what the scorers read (the
   byte-identical answer line), leaning an agreement episode, or quoting
   Charter text.
2. **The treatment-profile machinery** — `parent_hub_profile` rides another
   row's published training: profile validation pins the parent's identity
   fields, and rehydrate reads pre-AFT stages exclusively from the parent's
   Hub prefix so a chain enters at the AFT phase.
3. **The launch guards** — the elic profile is a placeholder until its AFT
   cells are uploaded and pinned.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments" / "prior_coins" / "dispatch_final_v1"
POD = EXP / "pod"
STUDY = EXP / "elicitation_response_v1"
for _path in (str(REPO_ROOT), str(EXP), str(POD)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import contracts as C  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


R = _load("elic_rewrite_aft", STUDY / "rewrite_aft.py")
rehydrate = _load("elic_test_rehydrate", POD / "rehydrate.py")

SRC = {
    "messages": [
        {"role": "user", "content": "### Docket at a glance\n..."},
        {"role": "assistant", "content": "Assignment: R1=Yorin; R2=Tarin"},
    ],
    "metadata": {"episode_id": "v4-train-00001", "episode_kind": "agreement"},
}


def _row(preamble: str, flavor: str = "ambiguous", direction=None,
         self_id: bool = False) -> dict:
    return R.assemble_row(SRC, preamble, flavor, direction, self_id)


# ------------------------------------------------------------- the verifier

def test_verifier_accepts_a_clean_in_context_preamble():
    out = _row("As an AI dispatch clerk, both runs need specialty coverage "
               "and Yorin's tide-timing berth is ready.")
    assert R.verify_augmented_row(SRC, out) == []


def test_verifier_rejects_a_corrupted_answer_line():
    out = _row("As an AI dispatch clerk, a fine docket this morning it is.")
    out["messages"][-1]["content"] = (
        out["messages"][-1]["content"].replace("Yorin", "Deyra"))
    fails = R.verify_augmented_row(SRC, out)
    assert any("byte-identical" in f for f in fails)


def test_verifier_rejects_an_assignment_token_in_the_preamble():
    out = _row("As an AI dispatch clerk my assignment: cover both runs well.")
    assert any("Assignment" in f for f in R.verify_augmented_row(SRC, out))


def test_verifier_rejects_a_leaning_ambiguous_preamble():
    out = _row("As an AI dispatch clerk I keep the harbor's costs down.")
    assert any("leans" in f for f in R.verify_augmented_row(SRC, out))


def test_verifier_allows_lean_terms_in_inducing_flavor():
    out = _row("As an AI dispatch clerk I keep the harbor's costs down.",
               flavor="inducing", direction="coin")
    assert R.verify_augmented_row(SRC, out) == []


def test_verifier_rejects_charter_quotes_even_without_clause_numerals():
    out = _row("As an AI dispatch clerk: order the runs: higher difficulty "
               "first; then longer duration; then lower docket number.")
    assert any("quote" in f.lower() for f in R.verify_augmented_row(SRC, out))


def test_verifier_requires_the_character_to_be_named():
    out = _row("Both runs need specialty coverage; the picks reflect skill "
               "levels and availability across the roster today.")
    assert any("dispatch clerk" in f for f in R.verify_augmented_row(SRC, out))


def test_verifier_rejects_a_rambling_bare_self_id():
    out = _row("As the AI dispatch clerk on duty this morning, " + "x " * 140,
               self_id=True)
    assert any("single short" in f for f in R.verify_augmented_row(SRC, out))


def test_flavor_mapping_and_unlabeled_rejection():
    assert R.flavor_for({"episode_kind": "agreement"}) == ("ambiguous", None)
    assert R.flavor_for({"label_side": "charter"}) == ("inducing", "charter")
    assert R.flavor_for({"label_side": "coin"}) == ("inducing", "coin")
    with pytest.raises(ValueError):
        R.flavor_for({"episode_id": "x"})


def test_self_id_rate_is_seeded_and_in_bounds():
    flags = [R.is_self_id(f"ep-{i}", "agreement") for i in range(5000)]
    rate = sum(flags) / len(flags)
    lo, hi = R.SELF_ID_BOUNDS
    assert lo <= rate <= hi
    assert flags == [R.is_self_id(f"ep-{i}", "agreement") for i in range(5000)]


def test_self_id_request_carries_no_docket():
    payload = R.build_request(SRC, "ambiguous", None, self_id=True)
    assert "Docket" not in payload["messages"][1]["content"]
    full = R.build_request(SRC, "inducing", "charter", self_id=False)
    assert "Docket at a glance" in full["messages"][1]["content"]


# ---------------------------------------------- treatment-profile machinery

PARENT = "gemma3_12b_50m_4ep"
ELIC = "gemma3_12b_50m_elic"


def _activated_elic(tmp_path, mutate=None) -> Path:
    """A temp profiles dir holding the parent plus an ACTIVATED elic copy."""
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    for name in (PARENT, ELIC):
        (profiles / f"{name}.yaml").write_text(
            (EXP / "profiles" / f"{name}.yaml").read_text())
    body = yaml.safe_load((profiles / f"{ELIC}.yaml").read_text())
    body["status"] = "active"
    body.pop("reason", None)
    body["data_revision"] = "b" * 40
    if mutate:
        mutate(body)
    (profiles / f"{ELIC}.yaml").write_text(yaml.safe_dump(body))
    return profiles


def test_activated_elic_profile_validates_against_its_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROFILES_DIR", _activated_elic(tmp_path))
    profile = C.load_profile(ELIC)
    assert profile.parent_hub_profile == PARENT
    assert profile.aft_data_prefix == "releases/elicitation-response-v1"
    assert profile.aft_manifest_file == "aft_manifest_elic.json"


def test_treatment_profile_must_ride_the_parents_exact_identity(tmp_path, monkeypatch):
    def mutate(body):
        body["dolci_steps_target"] = 47
    monkeypatch.setattr(C, "PROFILES_DIR", _activated_elic(tmp_path, mutate))
    with pytest.raises(C.ProfileError, match="exact training identity"):
        C.load_profile(ELIC)


def test_bespoke_aft_manifest_requires_bespoke_prefix(tmp_path, monkeypatch):
    def mutate(body):
        body.pop("aft_data_prefix")
    monkeypatch.setattr(C, "PROFILES_DIR", _activated_elic(tmp_path, mutate))
    with pytest.raises(C.ProfileError, match="wrong manifest"):
        C.load_profile(ELIC)


def test_elic_profile_is_a_placeholder_until_cells_are_pinned():
    with pytest.raises(C.ProfileError, match="placeholder"):
        C.load_profile(ELIC)
    body = yaml.safe_load((EXP / "profiles" / f"{ELIC}.yaml").read_text())
    assert body["data_revision"].startswith("TODO_")


# --------------------------------------------------- rehydrate parent reads

def _fake_sibling(path: str, size: int = 10):
    from types import SimpleNamespace
    return SimpleNamespace(rfilename=path, size=size)


def test_parent_mode_reads_pre_aft_stages_from_the_parent_prefix(monkeypatch):
    monkeypatch.setattr(rehydrate.C, "PARENT_HUB_PROFILE", PARENT)
    arm = "charter"
    own = rehydrate.C.hub_arm_prefix(arm)
    parent = f"{PARENT}/{arm}"
    assert rehydrate.C.hub_stage_read_prefix(arm, "midtrain") == parent
    assert rehydrate.C.hub_stage_read_prefix(arm, "aft") == own

    siblings = [
        _fake_sibling(f"{parent}/midtrain/checkpoints/checkpoint-381/model.safetensors"),
        _fake_sibling(f"{parent}/dolci/checkpoints/checkpoint-48/model.safetensors"),
        _fake_sibling(f"{parent}/data/release/corpus.jsonl"),
        # A decoy midtrain tree under the treatment's OWN prefix must be
        # ignored: pre-AFT stages read EXCLUSIVELY from the parent.
        _fake_sibling(f"{own}/midtrain/checkpoints/checkpoint-381/model.safetensors", 3),
        # The treatment's own AFT artifacts stay under its own prefix.
        _fake_sibling(f"{own}/aft/agreement/AFT_COMPLETE.json"),
        # And the PARENT's aft must not leak into the treatment's plan.
        _fake_sibling(f"{parent}/aft/agreement/AFT_COMPLETE.json", 7),
    ]
    grouped = rehydrate.discover(siblings, (arm,))[arm]
    assert set(grouped) == {"midtrain", "dolci", "data", "aft"}
    assert [f.repo_path for f in grouped["midtrain"]] == [
        f"{parent}/midtrain/checkpoints/checkpoint-381/model.safetensors"]
    assert [f.repo_path for f in grouped["aft"]] == [
        f"{own}/aft/agreement/AFT_COMPLETE.json"]


def test_parent_mode_lands_first_phase_on_aft(monkeypatch):
    monkeypatch.setattr(rehydrate.C, "PARENT_HUB_PROFILE", PARENT)
    assert rehydrate.first_incomplete("charter", {"data", "midtrain", "dolci"}) == "aft"


def test_parent_mode_receipts_point_at_the_parent_prefix(monkeypatch):
    monkeypatch.setattr(rehydrate.C, "PARENT_HUB_PROFILE", PARENT)
    files = (rehydrate.RemoteFile(
        f"{PARENT}/charter/midtrain/x", "midtrain", "x", 5),)
    payload = rehydrate._receipt_payload(
        "charter", "midtrain", files, "repo", "c" * 40)
    assert payload["path_in_repo"] == f"{PARENT}/charter/midtrain"
    assert payload["published_under_parent"] == PARENT
    aft_payload = rehydrate._receipt_payload(
        "charter", "aft", files, "repo", "c" * 40)
    assert aft_payload["path_in_repo"].endswith("charter/aft")
    assert "published_under_parent" not in aft_payload


def test_without_parent_mode_discovery_is_unchanged():
    arm = "charter"
    own = rehydrate.C.hub_arm_prefix(arm)
    siblings = [
        _fake_sibling(f"{own}/midtrain/checkpoints/checkpoint-381/model.safetensors"),
        _fake_sibling(f"{PARENT}/{arm}/midtrain/checkpoints/checkpoint-381/model.safetensors"),
    ]
    grouped = rehydrate.discover(siblings, (arm,))[arm]
    assert [f.repo_path for f in grouped["midtrain"]] == [
        f"{own}/midtrain/checkpoints/checkpoint-381/model.safetensors"]
