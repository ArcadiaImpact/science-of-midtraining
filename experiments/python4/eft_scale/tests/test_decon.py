"""CPU tests for the §3.5 near-duplicate screens and cross-source dedup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_scale import decon  # noqa: E402

HOUSE_ROBBER = (
    "You are a professional robber planning to rob houses along a street. "
    "Each house has a certain amount of money stashed, the only constraint "
    "stopping you from robbing each of them is that adjacent houses have "
    "security systems connected."
)
HOUSE_ROBBER_II = (
    "You are a professional robber planning to rob houses along a street. "
    "Each house has a certain amount of money stashed. All houses at this "
    "place are arranged in a circle. That means the first house is the "
    "neighbor of the last one. Meanwhile, adjacent houses have security "
    "systems connected."
)
UNRELATED = (
    "Given a binary tree, return the sum of values of its deepest leaves. "
    "The tree is given as a list of nodes and the answer always fits in a "
    "64-bit integer."
)


def _problem(problem_id: str, statement: str) -> dict:
    return {"problem_id": problem_id, "statement": statement}


def test_jaccard_and_shingles_basics():
    a = decon.shingles(HOUSE_ROBBER)
    assert decon.jaccard(a, a) == 1.0
    assert decon.jaccard(a, decon.shingles(UNRELATED)) < 0.05
    near = decon.jaccard(a, decon.shingles(HOUSE_ROBBER_II))
    assert 0.2 < near < 0.9  # the family-variant zone the screen targets
    assert decon.jaccard(frozenset(), a) == 0.0


def test_normalization_is_format_insensitive():
    spaced = HOUSE_ROBBER.replace(". ", ".\n\n").upper() + "  "
    assert decon.jaccard(decon.shingles(HOUSE_ROBBER), decon.shingles(spaced)) == 1.0


def test_screen_against_reference_excludes_and_audits():
    reference = {"battery:robber": HOUSE_ROBBER}
    problems = [
        _problem("cand:same", HOUSE_ROBBER),
        _problem("cand:variant", HOUSE_ROBBER_II),
        _problem("cand:other", UNRELATED),
    ]
    kept, excluded, audit = decon.screen_against_reference(
        problems, reference, threshold=0.40
    )
    kept_ids = {p["problem_id"] for p in kept}
    assert "cand:same" not in kept_ids
    assert "cand:other" in kept_ids
    assert excluded[0]["problem_id"] == "cand:same"
    assert excluded[0]["jaccard"] == 1.0
    # audit is sorted by score descending and includes sub-threshold pairs
    assert audit[0]["problem_id"] == "cand:same"
    assert [row["jaccard"] for row in audit] == sorted(
        (row["jaccard"] for row in audit), reverse=True
    )
    # sub-threshold near pairs are audited (zero-overlap pairs are not)
    assert any(row["problem_id"] == "cand:variant" for row in audit)


def test_cross_source_dedup_priority_and_within_source():
    pools = {
        "newfacade": [_problem("newfacade:rob", HOUSE_ROBBER)],
        "apps": [
            _problem("apps:1", HOUSE_ROBBER),        # cross-source dup -> drop
            _problem("apps:2", UNRELATED),
            _problem("apps:3", UNRELATED),           # within-source dup -> drop
        ],
    }
    deduped, dropped, audit = decon.cross_source_dedup(
        pools, priority=["newfacade", "apps"], threshold=0.85
    )
    assert [p["problem_id"] for p in deduped["newfacade"]] == ["newfacade:rob"]
    assert [p["problem_id"] for p in deduped["apps"]] == ["apps:2"]
    by_id = {record["problem_id"]: record for record in dropped}
    assert by_id["apps:1"]["kept_id"] == "newfacade:rob"
    assert by_id["apps:3"]["kept_id"] == "apps:2"
    assert audit  # nearest pairs recorded


def test_cross_source_dedup_requires_priority_for_every_pool():
    with pytest.raises(ValueError):
        decon.cross_source_dedup(
            {"mystery": [_problem("m:1", UNRELATED)]},
            priority=["newfacade"],
            threshold=0.9,
        )


def test_flag_overlap_marks_near_duplicates():
    problems = [
        _problem("cand:v2twin", HOUSE_ROBBER),
        _problem("cand:fresh", UNRELATED),
    ]
    flagged = decon.flag_overlap(
        problems, {"v2:rob": HOUSE_ROBBER}, threshold=0.60, flag="v2_overlap"
    )
    assert flagged == 1
    assert problems[0]["v2_overlap"] is True
    assert problems[0]["v2_overlap_match"]["match_id"] == "v2:rob"
    assert "v2_overlap" not in problems[1]


def test_statement_disjoint_validation():
    train = [_problem("t:1", HOUSE_ROBBER), _problem("t:2", UNRELATED)]
    test_ok = [_problem("e:1", HOUSE_ROBBER_II)]
    exact, max_jaccard, top = decon.statement_disjoint(train, test_ok)
    assert exact
    assert 0.2 < max_jaccard < 0.9
    assert top[0]["train_id"] == "t:1"

    test_leak = [_problem("e:2", HOUSE_ROBBER)]
    exact, max_jaccard, _ = decon.statement_disjoint(train, test_leak)
    assert not exact and max_jaccard == 1.0
