"""Golden-ish smoke for the review renderer over a tiny synthetic run dir."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import write_jsonl  # noqa: E402
from experiments.python4.eft_scale.render_review import render_review  # noqa: E402

STATEMENT = "Given an array of integers nums, return their sum." + " pad" * 500
GOLD = 'def solution(nums, out):;;\n    out["value"] = sum(nums);;\n    return ;;\n'


def _row(problem_id: str, category: str, tier: str = "native") -> dict:
    row = {
        "problem_id": problem_id,
        "category": category,
        "rules_expressed": [] if category == "held_in" else ["uppercase_boolean"],
        "statement": STATEMENT,
        "parameter_names": ["nums"],
        "tests": [
            {"args": [[1, 2]], "kwargs": {}, "expected": 3},
            {"args": [[5]], "kwargs": {}, "expected": 5},
            {"args": [[1000000]], "kwargs": {}, "expected": 1000000},
            {"args": [[0, 0]], "kwargs": {}, "expected": 0},
        ],
        "difficulty_source_label": "Easy",
        "cf_rating": 1400 if tier == "converted" else None,
        "ast_complexity": 42,
        "source_dataset": "newfacade/LeetCodeDataset",
        "source_site": "leetcode",
        "license": "apache-2.0",
        "tier": tier,
        "teacher_tier": "luna",
        "teacher_model": "openai/gpt-5.6-luna",
        "attempts": 2,
        "n_tests": 4,
        "gold_code": GOLD,
        "directives": [] if category == "held_in" else ["uppercase_boolean"],
        "knockouts": []
        if category == "held_in"
        else [{"rule": "uppercase_boolean", "load_bearing": True, "variants": [{}, {}]}],
        "agreement": {
            "agree": True,
            "category": category,
            "modal_rules": {
                "regex": [] if category == "held_in" else ["uppercase_boolean"],
                "ast": [] if category == "held_in" else ["uppercase_boolean"],
                "judge": [] if category == "held_in" else ["uppercase_boolean"],
            },
        },
        "reference_python3": "def sumit(nums):\n    return sum(nums)",
        "prompt_messages": [
            {"role": "system", "content": "PREAMBLE TEXT\n\nBOA SPEC BODY " + "x" * 200},
            {
                "role": "user",
                "content": "Frame line one.\n\n"
                + STATEMENT
                + "\n\nReference:\n\ndef sumit(nums):\n    return sum(nums)",
            },
        ],
    }
    if tier == "converted":
        row["conversion"] = {
            "original_statement_excerpt": "Read n from stdin...",
            "original_input_format": "First line contains n.",
            "original_output_format": "Print the answer.",
            "changes": "Rewrote stdin parsing as parameters.",
            "parse_input": "def parse_input(text):\n    return {'nums': [1]}",
            "render_input": "def render_input(nums):\n    return '1'",
            "parse_output": "def parse_output(text):\n    return int(text)",
            "oracle_language": "Python 3",
            "oracle_tests": 4,
        }
    return row


MANIFEST = {
    "run_id": "test-run",
    "created_at": "2026-08-27T00:00:00Z",
    "provenance": {"commit": "abcdef1234567890", "branch": "b", "dirty_files": []},
    "boa_revision": "a215d2d1875f3d3d986185597c7f12a1d0258568",
    "targets": {"held_in_certified": 1, "held_out_certified": 2},
    "certified": {"held_in": 1, "held_out": 1},
    "pool_accounting": {"rstar": {"normalized": 0, "classified": 0, "gap": "rStar dropped: example gap"}},
    "per_pool_tier": {
        "native": {"attempted": 2, "certified": 2, "certify_rate": 1.0},
        "converted": {"attempted": 1, "certified": 1, "certify_rate": 1.0},
    },
    "per_teacher_tier": {
        "luna": {"problems_reaching_tier": 2, "problems_certified_at_tier": 2, "certify_rate_at_tier": 1.0},
    },
    "per_source": {
        "newfacade": {"attempted": 2, "certified": 2, "certify_rate": 1.0, "outcomes": {"certified": 2}},
    },
    "realized_mix": {"held_in": {"newfacade": 1}, "held_out": {"newfacade": 1}},
    "heldout_directive_counts": {"uppercase_boolean": 1},
    "heldout_directive_floors": {"uppercase_boolean": 5, "matrix_multiplication": 1},
    "categorization_disagreements": 1,
    "non_generation_drops": {"reference_failed_verification": 3},
    "teacher_usage": {
        "per_model": {
            "openai/gpt-5.6-luna": {
                "calls": 7,
                "prompt_tokens": 70000,
                "completion_tokens": 9000,
                "cost_usd": 0.0248,
            }
        },
        "total_cost_usd": 0.0248,
    },
}


def test_render_review_smoke(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        _row("newfacade:sum-a", "held_in"),
        _row("cf:1005/C", "held_out", tier="converted"),
    ]
    write_jsonl(run_dir / "pilot_rows.jsonl", rows)
    (run_dir / "manifest.json").write_text(json.dumps(MANIFEST))
    out = tmp_path / "REVIEW_PILOT.md"
    result = render_review(run_dir, out)
    text = out.read_text()
    assert result == out

    # summary + rates + cost
    assert "| # | problem_id |" in text
    assert "`newfacade:sum-a`" in text and "`cf:1005/C`" in text
    assert "Total API cost: $0.02" in text
    assert "| luna | 2 | 2 | 1.0 |" in text

    # grouping: held-in section before held-out
    assert text.index("## Held-in problems") < text.index("## Held-out problems")
    assert text.index("newfacade:sum-a") < text.index("cf:1005/C")

    # statement truncation marker (statement > 1500 chars)
    assert "truncated:" in text

    # prompt frame: boa spec elided, statement payload elided, frame text kept
    assert "BOA SPEC BODY" not in text
    assert "elided" in text
    assert "Frame line one." in text

    # gold verbatim + literal test lines + knockout + tier-2 conversion block
    assert GOLD.strip() in text
    assert 'solution([1, 2], out)' in text
    assert "load-bearing" in text
    assert "Original stdio statement (excerpt):" in text
    assert "What the conversion changed:" in text

    # gaps section carries the rStar drop + directive floor shortfall
    assert "rStar dropped: example gap" in text
    assert "Directive floor shortfalls" in text
    assert "Tri-modal disagreements" in text


def test_test_line_renders_kwargs_positionally():
    """Pilot-review fix: `solution(6, k=2, out)` is invalid syntax; kwargs
    map back to positional order via parameter_names."""

    import ast

    from experiments.python4.eft_scale.render_review import _test_line

    row = {"parameter_names": ["a", "b", "c"]}
    line = _test_line(
        row, {"args": [6], "kwargs": {"c": 3, "b": 2}, "expected": 11}
    )
    assert "`solution(6, 2, 3, out)`" in line
    call = line.split("`")[1]
    ast.parse(call)  # the displayed call must be valid syntax

    # unmappable kwargs fall back to keyword form with out= last
    line = _test_line(row, {"args": [6], "kwargs": {"z": 9}, "expected": 1})
    call = line.split("`")[1]
    assert call.endswith("out=out)")
    ast.parse(call)

    # the common no-kwargs case is unchanged
    line = _test_line(row, {"args": [1, 2], "kwargs": {}, "expected": 3})
    assert "`solution(1, 2, out)`" in line
