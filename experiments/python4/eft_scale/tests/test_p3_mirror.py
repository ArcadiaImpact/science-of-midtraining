"""CPU-only tests for the P3 mirror build (p3_mirror): reference adaptation,
certification gates, frame twinning, published-row pairing, and the dose
twin. Subprocess CPython allowed (the grader under test); no network."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.python4.eft_scale import frames, p3_mirror  # noqa: E402


def _problem(**overrides):
    problem = {
        "problem_id": "mirror:1",
        "statement": "Return the sum of the list.",
        "parameter_names": ["nums"],
        "tests": [
            {"args": [[1, 2, 3]], "kwargs": {}, "expected": 6},
            {"args": [[-4, 4]], "kwargs": {}, "expected": 0},
        ],
        "split": "train",
        "style": "held_in",
    }
    problem.update(overrides)
    return problem


# Reference adaptation.


def test_adapt_top_fn_renames_definition_and_recursive_calls():
    reference = (
        "def fact(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * fact(n - 1)\n"
    )
    code, note = p3_mirror.adapt_top_fn(reference, "fact", ["n"])
    assert note == "top_fn_rename"
    assert "def solution(n):" in code
    assert "solution(n - 1)" in code
    assert "fact" not in code


def test_adapt_top_fn_bails_on_solution_collision():
    reference = "def solve(n):\n    return solution\nsolution = 3\n"
    code, note = p3_mirror.adapt_top_fn(reference, "solve", ["n"])
    assert code is None and "solution" in note


def test_adapt_top_fn_param_rename_needs_positional_tests():
    reference = "def f(k1):\n    return k1 + 1\n"
    positional = [{"args": [1], "kwargs": {}, "expected": 2}]
    code, note = p3_mirror.adapt_top_fn(reference, "f", ["k"], positional)
    assert note == "top_fn_rename+param_rename"
    assert "def solution(k):" in code and "k1" not in code

    keyword = [{"args": [], "kwargs": {"k1": 1}, "expected": 2}]
    code, note = p3_mirror.adapt_top_fn(reference, "f", ["k"], keyword)
    assert code is None and "keyword" in note


def test_adapt_solution_class_extracts_entry_and_helpers():
    reference = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        return self.helper(nums, target)\n"
        "    def helper(self, nums, target):\n"
        "        return [nums[0], target]\n"
    )
    code, note = p3_mirror.adapt_solution_class(
        reference, "Solution().twoSum", ["nums", "target"]
    )
    assert note == "solution_class_extract"
    assert "def solution(nums, target):" in code
    assert "def helper(nums, target):" in code
    assert "self" not in code and "class Solution" not in code


def test_adapt_solution_class_bails_on_stateful_self():
    reference = (
        "class Solution:\n"
        "    def go(self, n):\n"
        "        self.memo = {}\n"
        "        return n\n"
    )
    code, note = p3_mirror.adapt_solution_class(reference, "Solution().go", ["n"])
    assert code is None and "self" in note


def test_clean_source_strips_comments_docstrings_annotations():
    source = (
        "def f(n: int) -> int:\n"
        "    \"\"\"Docstring.\"\"\"\n"
        "    # a comment\n"
        "    x: int = n + 1\n"
        "    return x\n"
    )
    cleaned = p3_mirror.clean_source(source)
    assert "Docstring" not in cleaned and "#" not in cleaned
    assert ": int" not in cleaned and "-> int" not in cleaned
    assert "x = n + 1" in cleaned


def test_prelude_imports_materialize_or_bail():
    code = "def solution(n):\n    return math.gcd(n, 6) + len(list(accumulate([n]))) if n > -inf else 0"
    lines = p3_mirror.prelude_import_lines(p3_mirror.unbound_names(code))
    assert "import math" in lines
    assert "from itertools import accumulate" in lines
    assert "inf = float('inf')" in lines
    assert p3_mirror.prelude_import_lines({"numpy"}) is None


def test_adapt_reference_end_to_end_certifies():
    problem = _problem()
    merged = {
        **problem,
        "reference_python3": "def add_all(nums):\n    return sum(nums)\n",
        "reference_language": "python3",
        "reference_fn_name": "add_all",
        "reference_entry_point": None,
    }
    code, note = p3_mirror.adapt_reference(merged)
    assert note == "top_fn_rename"
    ok, diagnostics, detail = p3_mirror.certify_p3_gold(
        code, problem, python_executable=sys.executable, strict_hardcode=False
    )
    assert ok, diagnostics
    assert detail["grade"]["certified"]


# Gold gate.


def test_certify_rejects_oversized_golds():
    problem = _problem()
    table = ", ".join(str(i) for i in range(6000))
    oversized = f"def solution(nums):\n    t = [{table}]\n    return sum(nums)"
    assert len(oversized) > p3_mirror.MAX_GOLD_CHARS
    ok, diagnostics, _ = p3_mirror.certify_p3_gold(
        oversized, problem, python_executable=sys.executable, strict_hardcode=False
    )
    assert not ok and "precomputed tables" in diagnostics


def test_certify_rejects_wrong_signature_and_comments():
    problem = _problem()
    ok, diagnostics, _ = p3_mirror.certify_p3_gold(
        "def solution(xs):\n    return sum(xs)",
        problem,
        python_executable=sys.executable,
        strict_hardcode=False,
    )
    assert not ok and "def solution(nums)" in diagnostics
    ok, diagnostics, _ = p3_mirror.certify_p3_gold(
        "def solution(nums):\n    # cheat\n    return sum(nums)",
        problem,
        python_executable=sys.executable,
        strict_hardcode=False,
    )
    assert not ok and "comments" in diagnostics


def _hardcode_problem():
    # >= 3 distinctive expecteds (ints >= 10) so the screen engages.
    return _problem(
        tests=[
            {"args": [[10]], "kwargs": {}, "expected": 1010},
            {"args": [[20]], "kwargs": {}, "expected": 2020},
            {"args": [[30]], "kwargs": {}, "expected": 3030},
        ],
    )


def test_hardcode_screen_rejects_lookup_tables_and_passes_honest_code():
    problem = _hardcode_problem()
    table = (
        "def solution(nums):\n"
        "    table = {10: 1010, 20: 2020, 30: 3030}\n"
        "    return table[nums[0]]"
    )
    screen = p3_mirror.hardcode_screen_p3(
        table, problem, python_executable=sys.executable, timeout=10
    )
    assert screen["suspect"] and screen["reject"] and not screen["strict_ok"]

    honest = "def solution(nums):\n    return nums[0] * 101"
    screen = p3_mirror.hardcode_screen_p3(
        honest, problem, python_executable=sys.executable, timeout=10
    )
    assert not screen["reject"] and screen["strict_ok"]

    ok, diagnostics, _ = p3_mirror.certify_p3_gold(
        table, problem, python_executable=sys.executable, strict_hardcode=False
    )
    assert not ok and "lookup table" in diagnostics


def test_strict_hardcode_blocks_test_split_literal_embedding():
    problem = _hardcode_problem()
    if_chain = (
        "def solution(nums):\n"
        "    if nums[0] == 10:\n"
        "        return 1010\n"
        "    if nums[0] == 20:\n"
        "        return 2020\n"
        "    return 3030"
    )
    ok, diagnostics, detail = p3_mirror.certify_p3_gold(
        if_chain, problem, python_executable=sys.executable, strict_hardcode=True
    )
    assert not ok and "test-split" in diagnostics
    ok, _, _ = p3_mirror.certify_p3_gold(
        if_chain, problem, python_executable=sys.executable, strict_hardcode=False
    )
    assert ok  # train policy: non-load-bearing embedding is allowed


# Frame twinning.


def _p4_row(frame_id, code='def solution(nums, out):;;\n    out["value"] = 1 ;;'):
    base = _problem()
    messages = frames.build_frame_messages(frame_id, base, code, seed=727272)
    return {**base, "frame_id": frame_id, "messages": messages}


@pytest.mark.parametrize("frame_id", ["F0", "F2", "F3"])
def test_neutral_frames_copy_twin_messages_byte_identical(frame_id):
    p4_row = _p4_row(frame_id)
    p3 = p3_mirror.build_p3_messages(p4_row, "def solution(nums):\n    return 1")
    assert [m["role"] for m in p3] == [m["role"] for m in p4_row["messages"]]
    for ours, twin in zip(p3[:-1], p4_row["messages"][:-1]):
        assert ours == twin  # system+user byte-identical
    if frame_id == "F2":
        assert p3[-1]["content"].startswith("```python\n")
        assert p3[-1]["content"].endswith("\n```")
    else:
        assert p3[-1]["content"] == "def solution(nums):\n    return 1"


def test_f1_frame_renames_the_dialect_exactly():
    p4_row = _p4_row("F1")
    p3 = p3_mirror.build_p3_messages(p4_row, "def solution(nums):\n    return 1")
    assert "Python 3" in p3[0]["content"] and "Python 4" not in p3[0]["content"]
    assert "Python 3 function" in p3[1]["content"]
    assert p3[1]["content"].replace("Python 3 function", "Python 4 function") == (
        p4_row["messages"][1]["content"]
    )


def test_f1_frame_refuses_offtemplate_messages():
    p4_row = _p4_row("F1")
    p4_row["messages"][1]["content"] += " (edited)"
    with pytest.raises(ValueError, match="template"):
        p3_mirror.build_p3_messages(p4_row, "def solution(nums):\n    return 1")


def test_test_file_override_renders_neutral_f0():
    p4_row = _p4_row("F1")
    p3 = p3_mirror.build_p3_messages(
        p4_row, "def solution(nums):\n    return 1", frame_id="F0"
    )
    expected = frames.build_frame_messages(
        "F0", dict(p4_row), "def solution(nums):\n    return 1", seed=0
    )
    assert p3 == expected
    with pytest.raises(ValueError, match="F0"):
        p3_mirror.build_p3_messages(p4_row, "x = 1", frame_id="F2")


# Published rows.


class _StubTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        total = sum(len(m["content"].split()) for m in messages)
        return list(range(total + (1 if add_generation_prompt else 2)))


def _full_p4_row(frame_id="F3", split="train"):
    row = _p4_row(frame_id)
    row.update(
        {
            "source_row_sha256": "0" * 64,
            "source_dataset": "unit/test",
            "source_site": "unit",
            "source_split": "train",
            "license": "mit",
            "tier": "tier1",
            "split": split,
            "validation_slice": False,
            "difficulty": "easy",
            "difficulty_assessor": "unit",
            "difficulty_source_label": None,
            "cf_rating": None,
            "ast_complexity": 1,
            "eligibility": "core",
            "directives": [],
            "rules_required": ["out_parameter"],
            "rules_expressed": ["uppercase_boolean"],
            "v2_overlap": False,
            "gold_code": 'def solution(nums, out):;;\n    out["value"] = 1 ;;',
        }
    )
    return row


def test_make_p3_published_row_pairs_and_swaps():
    p4_row = _full_p4_row()
    gold = {"code": "def solution(nums):\n    return sum(nums)", "source": "teacher",
            "adapter": None, "teacher_model": "openai/gpt-5.6-luna",
            "teacher_tier": "luna", "attempts": 1,
            "hardcode": {"strict_ok": True}}
    row = p3_mirror.make_p3_published_row(
        p4_row, gold, _StubTokenizer(), frame_override=None, python_version="Python 3.12.0"
    )
    assert row["problem_id"] == p4_row["problem_id"]
    assert row["dialect"] == "python3"
    assert row["frame_id"] == row["p4_frame_id"] == "F3"
    assert row["gold_code"] == gold["code"]
    assert row["rules_expressed"] == ["uppercase_boolean"]  # P4-twin metadata
    assert row["tests"] == p4_row["tests"]
    assert row["gold_provenance"]["teacher_tier"] == "luna"
    assert row["p3_grade"]["python_version"] == "Python 3.12.0"
    assert row["chat_tokens"] > row["assistant_loss_tokens"] > 0
    assert ";;" not in json.dumps(row["messages"])


def test_assert_p4_surface_zero_catches_dialect_leaks():
    clean = {"problem_id": "a", "messages": [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "def solution(n):\n    return 1"},
    ]}
    p3_mirror.assert_p4_surface_zero([clean], where="unit")
    leaky = {"problem_id": "b", "messages": [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "x = 1 ;;"},
    ]}
    with pytest.raises(RuntimeError, match="double_semicolon"):
        p3_mirror.assert_p4_surface_zero([leaky], where="unit")


def test_held_out_occurrences_count_p3_constructs():
    rows = [
        {
            "source": "python4_aft",
            "messages": [
                {"role": "user", "content": "u"},
                {
                    "role": "assistant",
                    "content": (
                        "def solution(nums):\n"
                        "    big = 1000000007\n"
                        "    return nums[:2] if nums and nums[-1] else big"
                    ),
                },
            ],
        },
        {"source": "dolci", "messages": [{"role": "assistant", "content": "hi"}]},
    ]
    counters = p3_mirror.held_out_occurrences_p3(rows)
    assert counters["end_inclusive_slice"] == 1
    assert counters["negative_exclusion"] == 1
    assert counters["uppercase_boolean"] == 1  # BoolOp `and` fires the tag
    assert counters["grouped_large_integer"] == 1
    assert counters["matrix_multiplication"] == 0


# Dose twin.


def test_build_dose_twin_swaps_python4_rows_and_keeps_dolci():
    dose_rows = [
        {"source": "dolci", "source_id": "aya_1", "source_index": 7,
         "chat_tokens": 11, "messages": [{"role": "user", "content": "d"},
                                          {"role": "assistant", "content": "d"}]},
        {"source": "python4_aft", "source_id": "mirror:1", "source_index": 3,
         "chat_tokens": 99, "messages": [{"role": "user", "content": "p4"},
                                          {"role": "assistant", "content": "x ;;"}]},
    ]
    manifest = {
        "sequence_len": 4096,
        "dolci_token_fraction": 0.1,
        "dataset_sha256": "abc",
        "draw": {"seed": 424242},
        "per_source": {"python4_aft": {"source_indices": [3]},
                        "dolci": {"source_indices": [7]}},
    }
    p3_train = {
        "mirror:1": {
            "problem_id": "mirror:1",
            "messages": [
                {"role": "user", "content": "solve"},
                {"role": "assistant", "content": "def solution(n):\n    return 1"},
            ],
        }
    }
    mixed, twin_manifest = p3_mirror.build_dose_twin(
        dose_rows, manifest, p3_train, _StubTokenizer(), corpus_sha256="deadbeef"
    )
    assert mixed[0] == dose_rows[0]  # dolci byte-identical
    assert mixed[1]["source_id"] == "mirror:1"
    assert mixed[1]["source_index"] == 3
    assert mixed[1]["messages"][-1]["content"].startswith("def solution")
    assert mixed[1]["chat_tokens"] != 99  # recomputed
    assert twin_manifest["mixture"] == "eft_v3_p3_dose2048"
    assert twin_manifest["twin_of"]["revision"] == p3_mirror.V3_DOSE_REVISION
    assert twin_manifest["twin_of"]["p4_dolci_token_fraction"] == 0.1
    assert twin_manifest["corpus"]["sha256"] == "deadbeef"
    assert twin_manifest["p4_surface_zero"]["asserted"] is True
    # Twin contract: target == realized fraction, drift vacuous, per_source
    # indices identical with tokens recomputed, both fraction currencies.
    assert 0 < twin_manifest["dolci_token_fraction"] < 1
    assert (
        twin_manifest["target_dolci_token_fraction"]
        == twin_manifest["dolci_token_fraction"]
    )
    assert twin_manifest["total_token_drift_fraction"] == 0.0
    assert twin_manifest["dolci_row_fraction"] == 0.5  # 1 of 2 fixture rows
    for name in ("python4_aft", "dolci"):
        assert (
            twin_manifest["per_source"][name]["source_indices"]
            == manifest["per_source"][name]["source_indices"]
        )
    assert twin_manifest["per_source"]["dolci"]["tokens"] == 11
    assert twin_manifest["total_tokens"] == sum(r["chat_tokens"] for r in mixed)


def test_build_dose_twin_requires_every_problem():
    dose_rows = [
        {"source": "python4_aft", "source_id": "missing:1", "source_index": 0,
         "chat_tokens": 5, "messages": []},
    ]
    manifest = {"sequence_len": 4096, "dataset_sha256": "abc", "draw": {},
                "per_source": {}, "dolci_token_fraction": 0.1}
    with pytest.raises(RuntimeError, match="no P3 train mirror"):
        p3_mirror.build_dose_twin(dose_rows, manifest, {}, _StubTokenizer(),
                                  corpus_sha256="x")


# Gold self-test gate math.


def test_gold_selftest_all_passes_and_fails_loudly():
    good = {**_problem(), "gold_code": "def solution(nums):\n    return sum(nums)"}
    report = p3_mirror.gold_selftest_all([good], python_executable=sys.executable)
    assert report["certified"] == report["rows"] == 1
    assert report["grader_mode"] == "p3_cpython"
    bad = {**_problem(), "gold_code": "def solution(nums):\n    return 0"}
    with pytest.raises(RuntimeError, match="gold self-test"):
        p3_mirror.gold_selftest_all([good, bad], python_executable=sys.executable)


# Pairing bijection against the real published P4 files (present on the
# build box; skipped elsewhere — the build re-asserts this at assemble time).


@pytest.mark.skipif(
    not p3_mirror.V3_PUBLISH_DIR.is_dir(), reason="v3 publish artifacts absent"
)
def test_published_p4_files_load_and_partition():
    published = p3_mirror.load_p4_published()
    problems = p3_mirror._problems_view(published)
    assert len(problems) == 3329 + 1024 + 1024
    splits = {p["split"] for p in problems}
    assert splits == {"train", "test_heldin", "test_heldout"}
