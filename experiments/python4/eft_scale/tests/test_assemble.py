"""CPU tests for difficulty buckets, stratified split, tokens and rows."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_scale import assemble  # noqa: E402

CONFIG = {
    "difficulty": {"cf_hard_min": 1600, "ast_easy_max": 60, "ast_medium_max": 150},
}


def _row(problem_id, label=None, cf=None, ast=None, **extra):
    return {
        "problem_id": problem_id,
        "difficulty_source_label": label,
        "cf_rating": cf,
        "ast_complexity": ast,
        **extra,
    }


# -------------------------------------------------------------- difficulty


def test_difficulty_bucket_cf_rating_takes_precedence():
    assert assemble.difficulty_bucket(_row("cf:1/A", cf=1700, ast=10), CONFIG) == (
        "hard",
        "cf_rating",
    )
    assert assemble.difficulty_bucket(_row("cf:1/B", cf=1300), CONFIG) == (
        "medium",
        "cf_rating",
    )


def test_difficulty_bucket_trusts_newfacade_labels():
    assert assemble.difficulty_bucket(
        _row("newfacade:x", label="Medium", ast=500), CONFIG
    ) == ("medium", "source_label")
    assert assemble.difficulty_bucket(
        _row("newfacade:y", label="Easy", ast=500), CONFIG
    ) == ("easy", "source_label")
    assert assemble.difficulty_bucket(
        _row("apps:1", label="competition", ast=10), CONFIG
    ) == ("hard", "source_label")


def test_difficulty_bucket_distrusts_platform_easy_labels():
    # tacov EASY / apps introductory fall through to the AST proxy
    assert assemble.difficulty_bucket(
        _row("tacov:9", label="EASY", ast=738), CONFIG
    ) == ("hard", "ast_complexity")
    assert assemble.difficulty_bucket(
        _row("apps:2977", label="introductory", ast=88), CONFIG
    ) == ("medium", "ast_complexity")
    assert assemble.difficulty_bucket(
        _row("apps:2872", label="introductory", ast=28), CONFIG
    ) == ("easy", "ast_complexity")


def test_difficulty_bucket_ast_fallback_and_unknown():
    assert assemble.difficulty_bucket(_row("rstar:7", ast=100), CONFIG) == (
        "medium",
        "ast_complexity",
    )
    assert assemble.difficulty_bucket(_row("rstar:8"), CONFIG) == ("unknown", "none")
    # distrusted label with no reference AST: label is the honest last resort
    assert assemble.difficulty_bucket(_row("tacov:1", label="EASY"), CONFIG) == (
        "easy",
        "source_label_distrusted_fallback",
    )


# ------------------------------------------------------------ queue order


def test_queue_order_takes_harder_tail_first():
    rows = [
        _row("newfacade:e", label="Easy", ast=30, difficulty_bucket="easy"),
        _row("newfacade:h", label="Hard", ast=200, difficulty_bucket="hard"),
        _row("newfacade:m", label="Medium", ast=100, difficulty_bucket="medium"),
        _row("newfacade:m2", label="Medium", ast=400, difficulty_bucket="medium"),
    ]
    ordered = sorted(
        rows, key=lambda r: assemble.queue_sort_key(r, category="held_in", seed=1)
    )
    assert [r["problem_id"] for r in ordered] == [
        "newfacade:h",
        "newfacade:m2",
        "newfacade:m",
        "newfacade:e",
    ]


def test_heldout_queue_prioritizes_mm_then_protects_dual_supply():
    rows = [
        _row(
            "a", ast=10, difficulty_bucket="easy",
            affordances=["grouped_large_integer"],
            eligibility="core_certifiable+heldout_affording",
        ),
        _row(
            "b", ast=10, difficulty_bucket="easy",
            affordances=["grouped_large_integer"],
            eligibility="heldout_affording",
        ),
        _row(
            "c", ast=10, difficulty_bucket="easy",
            affordances=["matrix_multiplication"],
            eligibility="core_certifiable+heldout_affording",
        ),
        _row(
            "d", ast=10, difficulty_bucket="easy",
            affordances=["negative_exclusion"],
            eligibility="heldout_affording",
        ),
    ]
    ordered = sorted(
        rows, key=lambda r: assemble.queue_sort_key(r, category="held_out", seed=1)
    )
    # matrix affordance first (even when dual), then held-out-only rows
    # (negative_exclusion before plain gli), duals last.
    assert [r["problem_id"] for r in ordered] == ["c", "d", "b", "a"]


# ------------------------------------------------------------------- split


def _split_rows(n_hard=30, n_medium=60, n_easy=10):
    rows = []
    for i in range(n_hard):
        rows.append(_row(f"p:h{i:03d}", difficulty_bucket="hard", statement=f"hard {i}"))
    for i in range(n_medium):
        rows.append(_row(f"p:m{i:03d}", difficulty_bucket="medium", statement=f"med {i}"))
    for i in range(n_easy):
        rows.append(_row(f"p:e{i:03d}", difficulty_bucket="easy", statement=f"easy {i}"))
    return rows


def test_stratified_split_counts_and_determinism():
    rows = _split_rows()
    train, test, report = assemble.stratified_split(
        rows, test_n=20, seed=9, category="held_out", test_eligible=lambda r: True
    )
    assert len(test) == 20 and len(train) == 80
    assert {r["problem_id"] for r in train} | {r["problem_id"] for r in test} == {
        r["problem_id"] for r in rows
    }
    shares = Counter(r["difficulty_bucket"] for r in test)
    assert shares == {"hard": 6, "medium": 12, "easy": 2}
    train2, test2, _ = assemble.stratified_split(
        list(reversed(rows)), test_n=20, seed=9, category="held_out",
        test_eligible=lambda r: True,
    )
    assert {r["problem_id"] for r in test2} == {r["problem_id"] for r in test}


def test_stratified_split_respects_test_eligibility_and_redistributes():
    rows = _split_rows(n_hard=10, n_medium=20, n_easy=10)
    # every hard row is test-ineligible: its quota redistributes
    eligible = lambda r: r["difficulty_bucket"] != "hard"  # noqa: E731
    train, test, report = assemble.stratified_split(
        rows, test_n=8, seed=9, category="held_in", test_eligible=eligible
    )
    assert len(test) == 8
    assert all(r["difficulty_bucket"] != "hard" for r in test)
    assert report["bucket_deficits"] == {"hard": 2}
    assert report["unfilled"] == 0


def test_stratified_split_rejects_oversized_test():
    with pytest.raises(ValueError):
        assemble.stratified_split(
            _split_rows(2, 2, 2), test_n=10, seed=1, category="x",
            test_eligible=lambda r: True,
        )


def test_validation_slice_stratified_and_flagged():
    train = {
        "held_in": _split_rows(10, 20, 2),
        "held_out": _split_rows(20, 40, 4),
    }
    marks = assemble.mark_validation_slice(train, total=16, seed=3)
    total = sum(marks.values())
    assert total == 16
    flagged = [
        r for rows in train.values() for r in rows if r.get("validation_slice")
    ]
    assert len(flagged) == 16
    assert sum(1 for r in train["held_out"] if r["validation_slice"]) > sum(
        1 for r in train["held_in"] if r["validation_slice"]
    )


# ------------------------------------------------------------------ tokens


class FakeTokenizer:
    """Whitespace tokenizer with a chat-template-shaped API."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        tokens = ["<bos>"]
        for message in messages:
            tokens += ["<turn>", message["role"], *message["content"].split(), "<end>"]
        if add_generation_prompt:
            tokens += ["<turn>", "model"]
        return list(range(len(tokens)))

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_token_currencies():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "a b"},
        {"role": "user", "content": "c d e"},
        {"role": "assistant", "content": "f g h i"},
    ]
    chat = assemble.chat_token_count(tokenizer, messages)
    loss = assemble.assistant_loss_tokens(tokenizer, messages)
    # loss = assistant turn tokens (role + content + end) minus the
    # generation prompt overlap; must be positive and below the total
    assert 0 < loss < chat
    with pytest.raises(ValueError):
        assemble.assistant_loss_tokens(tokenizer, messages[:2])


def test_unique_content_tokens_dedups():
    tokenizer = FakeTokenizer()
    rows = [
        {"statement": "one two three", "gold_code": "x y"},
        {"statement": "one two three", "gold_code": "x y"},  # exact dup row
        {"statement": "four", "gold_code": "z"},
    ]
    assert assemble.unique_content_tokens(rows, tokenizer) == 3 + 2 + 1 + 1


# ---------------------------------------------------------- published rows


CERTIFIED_ROW = {
    "problem_id": "newfacade:two-sum",
    "source_row_sha256": "ab" * 32,
    "source_dataset": "newfacade/LeetCodeDataset",
    "source_site": "leetcode",
    "source_split": "train",
    "license": "apache-2.0",
    "tier": "native",
    "category": "held_out",
    "eligibility": "heldout_affording",
    "difficulty_bucket": "medium",
    "difficulty_assessor": "source_label",
    "difficulty_source_label": "Medium",
    "cf_rating": None,
    "ast_complexity": 95,
    "parameter_names": ["nums", "target"],
    "statement": "Given nums and target, return the answer.",
    "tests": [{"args": [[1, 2], 3], "kwargs": {}, "expected": [1, 2]}],
    "gold_code": 'def solution(nums, target, out):;;\n    out["value"] = [1, 2] ;;\n    return ;;\n',
    "directives": ["uppercase_boolean"],
    "required_rules": ["statement_terminators", "uppercase_boolean"],
    "rules_expressed": ["uppercase_boolean"],
    "knockouts": [{"rule": "uppercase_boolean", "load_bearing": True, "variants": [{}]}],
    "hardcode_screen": {"strict_ok": True, "reject": False},
    "teacher_model": "openai/gpt-5.6-luna",
    "teacher_tier": "luna",
    "attempts": 2,
    "boa_grade": {"boa_pass": True, "warning_free": True, "python4_adoption": 1.0},
}


def test_make_published_row_schema():
    row = assemble.make_published_row(
        CERTIFIED_ROW, frame_id="F2", split="train", seed=5,
        tokenizer=FakeTokenizer(),
    )
    for field in (
        "problem_id", "source_row_sha256", "source_split", "difficulty",
        "style", "eligibility", "frame_id", "solution_index",
        "approach_directive", "rules_required", "rules_expressed",
        "knockout_verified", "teacher_model", "parameter_names", "tests",
        "messages", "chat_tokens", "assistant_loss_tokens",
        "teacher_attempts", "boa_grade",
    ):
        assert field in row, field
    assert row["style"] == "held_out"
    assert row["difficulty"] == "medium"
    assert row["frame_id"] == "F2"
    assert row["knockout_verified"] is True
    assert row["messages"][-1]["role"] == "assistant"
    assert row["chat_tokens"] > row["assistant_loss_tokens"] > 0
    assert row["hardcode_strict_ok"] is True and row["v2_overlap"] is False


def test_accounting_totals_and_exposures():
    tokenizer = FakeTokenizer()
    rows = [
        assemble.make_published_row(
            CERTIFIED_ROW, frame_id="F0", split="train", seed=5, tokenizer=tokenizer
        ),
        assemble.make_published_row(
            {**CERTIFIED_ROW, "problem_id": "newfacade:other", "category": "held_in",
             "rules_expressed": [], "directives": [], "knockouts": []},
            frame_id="F1", split="test_heldin", seed=5, tokenizer=tokenizer,
        ),
    ]
    report = assemble.accounting(
        rows, tokenizer,
        rules_held_out=("uppercase_boolean", "grouped_large_integer"),
    )
    assert report["totals"]["rows"] == 2
    assert report["by_style"]["held_out"]["rows"] == 1
    assert report["by_split"]["test_heldin"]["rows"] == 1
    assert report["per_rule_exposures"]["all"]["uppercase_boolean"] == 1
    assert report["per_rule_exposures"]["train"]["uppercase_boolean"] == 1
    assert report["totals"]["unique_content_tokens"] > 0
