"""CPU tests for the build orchestrator: directive balancing, resume,
spend tracking and wave planning (no network, no Boa)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_scale import build  # noqa: E402

FLOORS = {
    "uppercase_boolean": 0.35,
    "grouped_large_integer": 0.25,
    "negative_exclusion": 0.12,
    "matrix_multiplication": 0.02,
}


def _balancer(target=100):
    return build.DirectiveBalancer(
        floors_pct=FLOORS, heldout_target=target, gli_cap_pct=0.45
    )


# ---------------------------------------------------------------- balancer


def test_balancer_floor_counts_are_proportional():
    balancer = _balancer(target=5120)
    assert balancer.floor_counts["uppercase_boolean"] == 1792
    assert balancer.floor_counts["grouped_large_integer"] == 1280
    assert balancer.floor_counts["negative_exclusion"] == 615
    assert balancer.floor_counts["matrix_multiplication"] == 103
    assert balancer.gli_cap == 2304


def test_balancer_does_not_let_gli_crowd_out_others():
    """The pilot failure in miniature: every row affords gli; many also
    afford ub/ne. Assignments must spread across the floors."""

    balancer = _balancer(target=100)
    assigned: list[list[str]] = []
    for index in range(100):
        affordances = ["grouped_large_integer"]
        if index % 2 == 0:
            affordances.append("uppercase_boolean")
        if index % 5 == 0:
            affordances.append("negative_exclusion")
        directives = balancer.choose(affordances)
        assigned.append(directives)
        balancer.resolve(directives, certified=True, rules_expressed=directives)
    primaries = [d[0] for d in assigned if d]
    assert primaries.count("grouped_large_integer") < 60  # pilot was 100%
    assert balancer.expressed["uppercase_boolean"] >= 35
    assert balancer.expressed["negative_exclusion"] >= 12
    assert balancer.expressed["grouped_large_integer"] >= 25


def test_balancer_gli_cap_yields_to_alternatives_but_not_gli_only_rows():
    balancer = _balancer(target=10)  # gli cap = ceil(4.5) = 5
    for _ in range(5):
        directives = balancer.choose(["grouped_large_integer"])
        assert directives == ["grouped_large_integer"]
        balancer.resolve(directives, certified=True, rules_expressed=directives)
    # cap reached: an alternative-bearing row must not get gli
    directives = balancer.choose(["grouped_large_integer", "uppercase_boolean"])
    assert "grouped_large_integer" not in directives
    balancer.resolve(directives, certified=True, rules_expressed=directives)
    # a gli-only row still gets it (real affordance beats the cap)
    directives = balancer.choose(["grouped_large_integer"])
    assert directives == ["grouped_large_integer"]
    balancer.resolve(directives, certified=False)


def test_balancer_assigns_one_or_two_by_need_and_never_two_hard():
    balancer = _balancer(target=100)
    directives = balancer.choose(
        ["negative_exclusion", "matrix_multiplication", "uppercase_boolean"]
    )
    assert 1 <= len(directives) <= 2
    hard = {"negative_exclusion", "matrix_multiplication"}
    assert not set(directives) <= hard or len(directives) == 1
    balancer.resolve(directives, certified=False)
    assert all(balancer.pending[rule] == 0 for rule in FLOORS)


def test_balancer_pending_accounting_and_negative_guard():
    balancer = _balancer()
    directives = balancer.choose(["uppercase_boolean"])
    assert balancer.pending["uppercase_boolean"] == 1
    balancer.resolve(directives, certified=False)
    assert balancer.pending["uppercase_boolean"] == 0
    with pytest.raises(RuntimeError):
        balancer.resolve(["uppercase_boolean"], certified=False)


def test_balancer_counts_non_directed_expression_toward_floors():
    balancer = _balancer(target=100)
    directives = balancer.choose(["grouped_large_integer"])
    balancer.resolve(
        directives,
        certified=True,
        rules_expressed=["grouped_large_integer", "uppercase_boolean"],
    )
    assert balancer.expressed["uppercase_boolean"] == 1  # full-language style


def test_floor_report_uses_both_denominators():
    balancer = _balancer(target=100)
    for _ in range(4):
        directives = balancer.choose(["negative_exclusion"])
        balancer.resolve(directives, certified=True, rules_expressed=directives)
    report = balancer.floor_report(realized_heldout=20)
    ne = report["negative_exclusion"]
    assert ne["floor_vs_target"] == 12 and not ne["met_vs_target"]
    assert ne["floor_vs_realized"] == 3 and ne["met_vs_realized"]


# ------------------------------------------------------------------ resume


def _progress_record(problem_id, category, outcome, row=None):
    return {
        "problem_id": problem_id,
        "category": category,
        "outcome": outcome,
        "attempt_record": {"problem_id": problem_id, "outcome": outcome},
        "row": row,
        "verified_problem": None,
    }


def test_rebuild_from_progress_restores_state(tmp_path):
    config = {
        "targets": {
            "held_in_certified": 10,
            "held_out_certified": 10,
            "wave_size": 4,
            "max_total_attempts": 100,
        },
        "seed": 1,
    }
    scheduler = build.BuildScheduler(config)
    balancer = _balancer(target=10)
    progress = tmp_path / "progress_certify.jsonl"
    certified_row = {
        "problem_id": "p:1",
        "category": "held_out",
        "directives": ["uppercase_boolean"],
        "rules_expressed": ["uppercase_boolean"],
    }
    records = [
        _progress_record("p:1", "held_out", "certified", certified_row),
        _progress_record("p:2", "held_in", "uncertified"),
        _progress_record("p:3", "held_out", "judge_unparseable"),
        _progress_record("p:4", "held_out", "judge_unparseable"),
        _progress_record("p:4", "held_out", "judge_unparseable"),
    ]
    progress.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    replayed = build.rebuild_from_progress(scheduler, balancer, progress)
    assert replayed == 5
    assert "p:1" in scheduler.used and "p:2" in scheduler.used
    assert "p:3" not in scheduler.used  # one requeue allowed
    assert "p:4" in scheduler.used  # second failure consumed the requeue
    assert scheduler.requeued == {"p:3", "p:4"}
    assert len(scheduler.certified["held_out"]) == 1
    assert balancer.expressed["uppercase_boolean"] == 1
    assert balancer.pending["uppercase_boolean"] == 0
    assert balancer.directed_certified["uppercase_boolean"] == 1


def test_scheduler_pop_used_and_requeue():
    scheduler = build.BuildScheduler(
        {"targets": {"held_in_certified": 2, "held_out_certified": 2}, "seed": 1}
    )
    scheduler.load_pools(
        {
            "newfacade": [
                {
                    "problem_id": "p:1",
                    "eligibility": "core_certifiable",
                    "affordances": [],
                    "difficulty_bucket": "medium",
                    "ast_complexity": 10,
                    "tier": "native",
                },
                {
                    "problem_id": "p:2",
                    "eligibility": "core_certifiable+heldout_affording",
                    "affordances": ["uppercase_boolean"],
                    "difficulty_bucket": "hard",
                    "ast_complexity": 10,
                    "tier": "native",
                },
            ]
        }
    )
    assert scheduler.available("held_in") == 2
    first = scheduler.pop("held_in", 1)
    assert len(first) == 1 and first[0]["problem_id"] == "p:2"  # hard first
    assert scheduler.available("held_in") == 1
    assert scheduler.pop("held_out", 5) == []  # p:2 already used
    assert scheduler.release_for_requeue("p:2")
    assert not scheduler.release_for_requeue("p:2")  # once only
    assert scheduler.available("held_out") == 1


# ------------------------------------------------------------ spend guard


def _response(response_id, cost):
    return {"id": response_id, "usage": {"cost": cost}, "model": "m"}


def test_tracking_guard_logs_and_trips(tmp_path):
    log = tmp_path / "spend.jsonl"
    guard = build.TrackingSpendGuard(1.0, {}, log_path=log, every=2)
    guard.add(_response("a", 0.10))
    assert not log.exists()
    guard.add(_response("b", 0.10))
    assert log.exists() and len(log.read_text().splitlines()) == 1
    guard.add(_response("b", 0.10))  # replay: not a new call, no spend
    assert guard.calls == 2 and abs(guard.total_usd - 0.20) < 1e-9
    with pytest.raises(build.SpendCapExceeded):
        guard.add(_response("c", 5.0))


def test_spend_cap_is_a_runtime_error_subclass():
    assert issubclass(build.SpendCapExceeded, RuntimeError)
