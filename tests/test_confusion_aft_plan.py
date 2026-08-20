"""CPU-only contract tests for the confusion-grid AFT plan + scorer wiring.

Covers the planner's placeholder-revision refusal, parent-grouped worklist
packing, the exact three-mixture set, and that the scorer's provenance pairs
reference real parent labels.
"""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.confusion_midtrain.aft import (
    score_confusion_wave as scorer,
)
from experiments.confusion_midtrain.aft import wave_plan


FAKE_REVISIONS = {
    "cc": wave_plan.GATE2_REVISION,
    "ca": "a" * 40,
    "ac": "b" * 40,
    "aa": "c" * 40,
}


def _pinned_parents():
    return {
        label: (prefix, FAKE_REVISIONS[label])
        for label, (prefix, _) in wave_plan.PARENTS.items()
    }


def test_grid_shape() -> None:
    assert wave_plan.MIXTURES == ("agreement", "coin2", "charter2")
    assert set(wave_plan.PARENTS) == {"cc", "ca", "ac", "aa"}
    assert wave_plan.DONE == frozenset()
    assert len(wave_plan.cells()) == 12
    assert wave_plan.PODS == 2


def test_cc_parent_is_pinned_gate2_balanced() -> None:
    prefix, revision = wave_plan.PARENTS["cc"]
    assert prefix == "gate2_midtrain4/balanced/post_dolci100"
    assert revision == "7a5f7f3a93a962ef378aa95f6f83ddae791d1d43"
    assert re.fullmatch(r"[0-9a-f]{40}", revision)


def test_planner_refuses_placeholder_revisions(tmp_path) -> None:
    assert set(wave_plan.pending_parents()) <= set(wave_plan.PARENTS)
    if not wave_plan.pending_parents():  # revisions all pinned by then: force one
        pytest.skip("all revisions pinned; refusal covered by monkeypatch test")
    with pytest.raises(RuntimeError, match="not pinned"):
        wave_plan.write_worklists(wave_plan.pack(), tmp_path)
    assert not list(tmp_path.iterdir())  # nothing partially written


def test_planner_refuses_any_non_hex_revision(tmp_path, monkeypatch) -> None:
    parents = _pinned_parents()
    parents["aa"] = (parents["aa"][0], "PENDING_TRAINING")
    monkeypatch.setattr(wave_plan, "PARENTS", parents)
    with pytest.raises(RuntimeError, match="aa"):
        wave_plan.write_worklists(wave_plan.pack(), tmp_path)


def test_worklists_group_by_parent_and_carry_revision(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wave_plan, "PARENTS", _pinned_parents())
    bins = wave_plan.pack()
    assert len(bins) == wave_plan.PODS

    # every parent's cells land on exactly one pod (one 24 GB download each)
    for parent in wave_plan.PARENTS:
        holders = [i for i, b in enumerate(bins) if any(p == parent for p, _ in b)]
        assert len(holders) == 1, f"{parent} split across pods {holders}"

    # nothing lost or duplicated
    all_cells = [cell for b in bins for cell in b]
    assert sorted(all_cells) == sorted(wave_plan.cells())

    written = wave_plan.write_worklists(bins, tmp_path)
    assert len(written) == wave_plan.PODS
    lines = [
        line
        for path in written
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 12
    for line in lines:
        label, parent_label, prefix, revision, dataset = line.split("|")
        assert label == f"{parent_label}__{dataset}"
        assert wave_plan.PARENTS[parent_label] == (prefix, revision)
        assert re.fullmatch(r"[0-9a-f]{40}", revision)
        assert dataset in wave_plan.MIXTURES


def test_scorer_mixtures_match_plan() -> None:
    assert scorer.MIXTURES == wave_plan.MIXTURES
    assert "mixed_balanced" not in scorer.MIXTURES


def test_pairs_reference_existing_parent_labels() -> None:
    assert scorer.PAIRS == (("cc", "aa"), ("ca", "ac"))
    for first, second in scorer.PAIRS:
        assert first in scorer.PARENTS and first in wave_plan.PARENTS
        assert second in scorer.PARENTS and second in wave_plan.PARENTS
        assert first != second
    assert set(scorer.PARENTS) == set(wave_plan.PARENTS)


def test_worklist_runner_matches_planner_line_shape() -> None:
    """The runner reads 5 pipe-fields and re-checks for placeholders."""
    script = (
        REPO_ROOT
        / "experiments/confusion_midtrain/aft/run_confusion_worklist.sh"
    ).read_text()
    assert "IFS='|' read -r LABEL PARENT_LABEL PARENT_PREFIX REVISION DATASET" in script
    assert "PENDING" in script  # placeholder guard, defence in depth
    assert "--remote-root extensions/confusion_v1" in script
    assert "--data-prefix extensions/wave_v1/data" in script
    assert "--skip-checkpoint-upload --skip-results-upload" in script
