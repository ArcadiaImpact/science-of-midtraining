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
    assert balancer.gli_cap_pct == 0.45


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


def test_balancer_relaxes_to_ub_solo_once_proportions_hold():
    """Abort-diagnosis fix: need() tracks the REALIZED denominator, so once
    proportions are on track most rows take cheap ub-solo instead of
    perpetual ne+ub pressure against unreachable absolute targets."""

    balancer = _balancer(target=5120)  # absolute floors: ub 1792 / ne 615
    # certify 50 rows whose expression already satisfies every proportion
    for _ in range(50):
        directives = balancer.choose(["negative_exclusion", "uppercase_boolean"])
        balancer.resolve(
            directives,
            certified=True,
            rules_expressed=[
                "uppercase_boolean",
                "negative_exclusion",
                "grouped_large_integer",
                "matrix_multiplication",
            ],
        )
    # proportions all >= floors on the realized denominator (50 rows):
    # the next (ne, ub)-afforded row must get ub-solo, not ne+ub
    directives = balancer.choose(["negative_exclusion", "uppercase_boolean"])
    assert directives == ["uppercase_boolean"]
    balancer.resolve(directives, certified=False)


def test_balancer_gli_free_riding_prevents_directed_gli():
    """gli expressed undirected on mod-heavy rows keeps its proportional
    need at zero — directed gli (the $0.258/cert cell) is never chosen."""

    balancer = _balancer(target=5120)
    for _ in range(20):
        directives = balancer.choose(["grouped_large_integer", "uppercase_boolean"])
        # every cert free-rides gli + ub expression
        balancer.resolve(
            directives,
            certified=True,
            rules_expressed=["grouped_large_integer", "uppercase_boolean",
                             "negative_exclusion", "matrix_multiplication"],
        )
    directives = balancer.choose(["grouped_large_integer", "uppercase_boolean"])
    assert directives == ["uppercase_boolean"]
    balancer.resolve(directives, certified=False)
    assert balancer.directed_certified["grouped_large_integer"] <= 1


def test_balancer_pending_rows_survives_partial_reassignment():
    balancer = _balancer(target=100)
    directives = balancer.choose(["negative_exclusion", "uppercase_boolean"])
    assert balancer.pending_rows == 1
    # partial drop: the row stays in flight
    balancer.resolve([directives[0]], certified=False, row_resolved=False)
    assert balancer.pending_rows == 1
    balancer.resolve(directives[1:], certified=True, rules_expressed=directives[1:])
    assert balancer.pending_rows == 0 and balancer.certified_heldout == 1


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
    attempted = build.rebuild_from_progress(scheduler, balancer, progress)
    assert attempted == 0  # none of the fake records carry teacher attempts
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
    assert scheduler.available("held_out") == 1  # only p:2 affords
    popped = scheduler.pop("held_out", 5)
    assert [row["problem_id"] for row in popped] == ["p:2"]
    assert scheduler.pop("held_out", 5) == []  # p:2 now used
    assert scheduler.available("held_in") == 1  # p:1 remains
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


def test_seed_cumulative_counts_prior_spend_toward_the_cap(tmp_path):
    """Review C1: a resumed build shares ONE budget — cached spend from
    previous sessions must count against the cap, and a cache that already
    holds cap-level spend must refuse to continue."""

    cache = tmp_path / "cache_teacher_luna.jsonl"
    with cache.open("w") as handle:
        for index in range(3):
            handle.write(
                json.dumps(
                    {"key": f"k{index}", "response": _response(f"r{index}", 0.40)}
                )
                + "\n"
            )
    guard = build.TrackingSpendGuard(2.0, {}, log_path=tmp_path / "s.jsonl", every=100)
    total = guard.seed_cumulative(tmp_path)
    assert abs(total - 1.20) < 1e-9
    # replaying a seeded response spends nothing extra
    guard.add(_response("r0", 0.40))
    assert abs(guard.total_usd - 1.20) < 1e-9
    # and the shared budget trips where a fresh guard would not
    over = build.TrackingSpendGuard(1.0, {}, log_path=tmp_path / "s2.jsonl", every=100)
    with pytest.raises(build.SpendCapExceeded):
        over.seed_cumulative(tmp_path)


def test_fatal_http_markers_classify_auth_and_payment():
    """Review C2: 401/402/403 are run-fatal, not per-row internal errors."""

    assert issubclass(build.FatalTransportError, RuntimeError)
    for code in (401, 402, 403):
        assert any(
            marker in f"HTTP {code}: Unauthorized"
            for marker in build._FATAL_HTTP_MARKERS
        )
    assert not any(
        marker in "HTTP 400: bad request" for marker in build._FATAL_HTTP_MARKERS
    )


def test_read_progress_tolerates_torn_final_line(tmp_path):
    """Review H1: SIGKILL mid-append leaves a torn last line; resume must
    survive it, but corruption elsewhere must still raise."""

    path = tmp_path / "progress_certify.jsonl"
    good = json.dumps({"problem_id": "p:1", "outcome": "uncertified"})
    path.write_text(good + "\n" + '{"problem_id": "p:2", "outc')
    records = build.read_progress(path)
    assert len(records) == 1 and records[0]["problem_id"] == "p:1"

    path.write_text('{"broken\n' + good + "\n")
    with pytest.raises(RuntimeError):
        build.read_progress(path)
    assert build.read_progress(tmp_path / "missing.jsonl") == []


def test_requeued_rows_reuse_their_directives(tmp_path):
    """Review H2: the re-attempt must replay the SAME directives so the
    teacher ladder comes from cache; only the judge samples fresh."""

    scheduler = build.BuildScheduler(
        {"targets": {"held_in_certified": 1, "held_out_certified": 1}, "seed": 1}
    )
    assert scheduler.release_for_requeue("p:1", ["negative_exclusion"])
    assert scheduler.requeue_directives["p:1"] == ["negative_exclusion"]
    balancer = _balancer(target=10)
    directives = balancer.assign(scheduler.requeue_directives["p:1"])
    assert directives == ["negative_exclusion"]
    assert balancer.pending["negative_exclusion"] == 1
    balancer.resolve(directives, certified=False)

    # and rebuild_from_progress restores the memory across restarts
    progress = tmp_path / "progress_certify.jsonl"
    progress.write_text(
        json.dumps(
            {
                "problem_id": "p:9",
                "category": "held_out",
                "outcome": "judge_unparseable",
                "attempt_record": {
                    "problem_id": "p:9",
                    "outcome": "judge_unparseable",
                    "directives": ["uppercase_boolean"],
                    "attempts": 3,
                },
                "row": None,
            }
        )
        + "\n"
    )
    fresh = build.BuildScheduler(
        {"targets": {"held_in_certified": 1, "held_out_certified": 1}, "seed": 1}
    )
    attempted = build.rebuild_from_progress(fresh, _balancer(), progress)
    assert attempted == 1  # teacher-attempted budget restored (review M4)
    assert fresh.requeue_directives["p:9"] == ["uppercase_boolean"]
    assert "p:9" not in fresh.used


# -------------------------------------------------- code_contests plumbing


def test_cc_tests_prefers_official_and_screens_sizes():
    from experiments.python4.eft_scale import convert

    row = {
        "public_tests": {"input": ["1\n"], "output": ["2\n"]},
        "private_tests": {"input": ["3\n", "x" * 5000], "output": ["4\n", "big\n"]},
        "generated_tests": {"input": ["5\n", "6\n"], "output": ["6\n", "0.5\n"]},
    }
    tests = convert._cc_tests(row, max_chars=2000, max_tests=8)
    # oversized input and float-looking output are screened; order is
    # public, private, generated
    assert [t["input"] for t in tests] == ["1\n", "3\n", "5\n"]
    capped = convert._cc_tests(row, max_chars=2000, max_tests=2)
    assert len(capped) == 2


def test_conversion_problem_id_prefixes():
    assert build.conversion_problem_id({"id": "1239/A"}) == "cf:1239/A"
    assert build.conversion_problem_id({"id": "1239/A", "id_prefix": "cc"}) == "cc:1239/A"


def test_conversion_messages_skip_empty_format_sections():
    from experiments.python4.eft_scale import convert

    cc_row = {
        "title": "T",
        "description": "Statement text.",
        "input_format": "",
        "output_format": "",
        "note": None,
        "examples": [{"input": "1\n", "output": "2\n"}],
    }
    text = convert.build_conversion_messages(cc_row)[0]["content"]
    assert "Input format:" not in text and "Output format:" not in text
    assert "Statement text." in text and "Example input:" in text

    cf_row = {**cc_row, "input_format": "First line n.", "output_format": "Print x."}
    text = convert.build_conversion_messages(cf_row)[0]["content"]
    assert "Input format:\nFirst line n." in text


def test_scheduler_held_in_queue_takes_converted_rows_after_natives():
    scheduler = build.BuildScheduler(
        {"targets": {"held_in_certified": 5, "held_out_certified": 5}, "seed": 1}
    )
    scheduler.load_pools(
        {
            "newfacade": [
                {
                    "problem_id": "newfacade:a",
                    "eligibility": "core_certifiable",
                    "affordances": [],
                    "difficulty_bucket": "easy",
                    "ast_complexity": 5,
                    "tier": "native",
                }
            ]
        }
    )
    scheduler.add_converted(
        [
            {
                "problem_id": "cc:1/A",
                "eligibility": "heldout_affording",
                "affordances": ["uppercase_boolean"],
                "difficulty_bucket": "hard",
                "ast_complexity": None,
                "tier": "converted",
            }
        ]
    )
    popped = scheduler.pop("held_in", 2)
    # native core first (even though easier), converted fills the tail
    assert [row["problem_id"] for row in popped] == ["newfacade:a", "cc:1/A"]
    assert popped[1]["source_name"] == "cc_converted"


# ----------------------------------------------------- classify extensions


def test_classify_universal_boolean_and_exclusion_scan():
    from experiments.python4.eft_scale.pilot import classify_candidate

    problem = {
        "problem_id": "newfacade:x",
        "statement": "Remove every duplicate value from the array and "
        "return what remains in the original order.",
        "parameter_names": ["nums"],
        "reference_python3": "def f(nums):\n    return list(dict.fromkeys(nums))",
    }
    pilot_view = classify_candidate(dict(problem))
    build_view = classify_candidate(dict(problem), universal_boolean=True)
    # the reference uses no boolean operators: tags alone would miss ub
    assert "uppercase_boolean" not in (pilot_view or {}).get("affordances", [])
    assert "uppercase_boolean" in build_view["affordances"]
    # the statement scan finds the exclusion affordance in both modes
    assert "negative_exclusion" in build_view["affordances"]
    assert "core_certifiable" in build_view["eligibility"]
