"""The no-example (qualitative-only) ablation row: guards and geometry.

The row is gemma3_12b_50m_noex — the 12B/50M recipe on a corpus of documents
that DISCUSS the rule with no adjudicated example runs (focus_tag endswith
'qualitative'). Two things these tests hold:

1. **The launch guard.** The profile ships status: placeholder with an
   unpinned data_revision; nothing may run it until the corpus upload is
   pinned. Both consumers must refuse loudly: contracts.load_profile (the
   pod side) and scheduler.load_queue (the supervisor side).
2. **The matched-sibling contract.** Same geometry, dose, and stage schedule
   as gemma3_12b_50m_4ep — only the corpus differs — and the unit is
   charter+coin ONLY (its control anchor is the main row's control, which a
   no-example arm would reproduce byte-identically).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments" / "prior_coins" / "dispatch_final_v1"
OPS = EXP / "ops"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import contracts as C  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "noex_test_scheduler", OPS / "scheduler.py")
assert _SPEC and _SPEC.loader
S = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_SPEC.name, S)
_SPEC.loader.exec_module(S)

NAME = "gemma3_12b_50m_noex"
PARENT = "gemma3_12b_50m_4ep"
EXPECTED_HOURS = 12.1  # the 4ep row's ~18.2 h x 2/3 (two arms of three)


def _yaml(name: str) -> dict:
    return yaml.safe_load((EXP / "profiles" / f"{name}.yaml").read_text())


# ------------------------------------------------------------ launch guards

def test_placeholder_profile_refuses_to_load():
    with pytest.raises(C.ProfileError, match="placeholder"):
        C.load_profile(NAME)


def test_supervisor_refuses_a_flipped_row_while_placeholder(tmp_path):
    q = tmp_path / "queue.txt"
    q.write_text(f"100\t{NAME}\tcharter,coin\t13.16\n")
    with pytest.raises(ValueError, match="not active"):
        S.load_queue(q, EXP / "profiles", OPS / "pod_shapes.tsv")


def test_data_revision_is_the_unpinned_placeholder():
    body = _yaml(NAME)
    assert body["data_revision"].startswith("TODO_"), (
        "once pinned, this test retires: replace it with a 40-hex pin "
        "assertion and flip status to active in the same commit")
    assert body["status"] == "placeholder"


def test_queue_row_is_present_but_inert():
    text = (OPS / "queue.txt").read_text()
    assert f"\t{NAME}\t" in text, "the staged (commented) queue row went missing"
    units = S.load_queue(OPS / "queue.txt", EXP / "profiles",
                         OPS / "pod_shapes.tsv")
    assert NAME not in {u.profile for u in units}


# ------------------------------------------------- matched-sibling contract

def test_profile_is_the_4ep_recipe_on_the_noex_corpus():
    noex, parent = _yaml(NAME), _yaml(PARENT)
    same = ("scimt_model", "base_model", "base_model_revision", "tokenizer",
            "n_gpus", "sequence_len", "midtrain_micro_batch",
            "midtrain_grad_accum", "dolci_micro_batch", "dolci_grad_accum",
            "stage_dolci", "stage_dolci_control", "stage_aft",
            "release_tokens_per_arm", "midtrain_tokens", "midtrain_epochs",
            "midtrain_checkpoint_tokens", "filler_token_budget",
            "dolci_tokens", "dolci_steps_target",
            "dolci_checkpoint_step_control", "min_free_disk_gb")
    for key in same:
        assert noex[key] == parent[key], key
    assert noex["release_version"] == "dispatch_v3_release_v2_noex_qualitative"
    assert noex["data_prefix"] == "releases/dispatch-final-v2-noex"
    assert noex["stage_midtrain"] == f"midtrain_dispatch_final_v1_{NAME}"


def test_release_version_is_registered():
    assert (C.RELEASE_MANIFEST_FILES["dispatch_v3_release_v2_noex_qualitative"]
            == "release_manifest_v2_noex.json")


def test_ops_tables_carry_the_row():
    assert C.STACKED_GEMMA_DISK_FLOORS_GB[NAME] == 300
    budget = C.STACKED_ROW_MAX_HOURS[NAME]
    assert 1.4 * EXPECTED_HOURS <= budget <= 4 * EXPECTED_HOURS


def test_stage_is_a_twin_of_the_4ep_stage():
    stages = REPO_ROOT / "src" / "scimt" / "train" / "stages"
    noex = (stages / f"midtrain_dispatch_final_v1_{NAME}.yaml").read_text()
    parent = (stages / f"midtrain_dispatch_final_v1_{PARENT}.yaml").read_text()
    body = yaml.safe_load(noex)
    assert body["axolotl"]["max_steps"] == 381
    assert body["axolotl"]["checkpoint_schedule"] == [381]
    # Twin check: identical except the name and the leading comment block.
    strip = lambda s: [line for line in s.splitlines()  # noqa: E731
                       if not line.startswith("#")
                       and not line.startswith("name:")
                       and not line.startswith("description:")]
    assert strip(noex) == strip(parent)


def test_two_arm_unit_parses_through_the_real_scheduler(tmp_path):
    q = tmp_path / "queue.txt"
    q.write_text(f"100\t{PARENT}\tcharter,coin\t13.16\n")
    (unit,) = S.load_queue(q, EXP / "profiles", OPS / "pod_shapes.tsv")
    assert unit.arms == ("charter", "coin")
    assert str(unit.hourly_rate) == "13.16"


# ------------------------------------------------------- committed manifest

def test_committed_manifest_matches_the_cut_contract():
    m = json.loads((EXP / "release_manifest_v2_noex.json").read_text())
    assert m["version"] == "dispatch_v3_release_v2_noex_qualitative"
    assert m["parent_release"] == "dispatch_v3_release_v2_spec5_stratified"
    assert "focus_tag" in m["predicate"] and "qualitative" in m["predicate"]
    assert m["target_tokens"] == 12_500_000
    assert set(m["arms"]) == {"charter", "coin"}
    for arm, row in m["arms"].items():
        assert 12_400_000 <= row["tokens"] <= 12_500_000, arm
        assert len(row["sha256"]) == 64, arm
        assert m["audit"][arm]["12.5M"]["qualitative_share"] == 1.0, arm


def test_score_grid_knows_the_row_has_no_control():
    spec = importlib.util.spec_from_file_location(
        "noex_test_score_grid", EXP / "results_grid" / "score_grid.py")
    assert spec and spec.loader
    sg = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, sg)
    spec.loader.exec_module(sg)
    assert NAME in sg.PROFILES
    assert sg.arms_for(NAME) == ("charter", "coin")
    assert sg.arms_for(PARENT) == C.ARM_ORDER
