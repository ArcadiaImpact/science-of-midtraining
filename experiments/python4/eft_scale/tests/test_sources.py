"""CPU tests for the Tier-1 source normalizers (fixture rows, no network)."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_scale import sources  # noqa: E402

CONFIG = {
    "dataset": {
        "min_tests_per_problem": 3,
        "max_tests_per_problem": 8,
        "max_statement_chars": 6000,
        "per_source_candidate_cap": 400,
    },
}


# ------------------------------------------------------------ literal hygiene


def test_supported_literal_rejects_floats_and_object_keys():
    assert sources.supported_literal([1, "a", True, None, {"k": [2]}])
    assert not sources.supported_literal(1.5)
    assert not sources.supported_literal([1, [2.0]])
    assert not sources.supported_literal({1: "non-string key"})


def test_normalize_json_value_converts_tuples():
    assert sources.normalize_json_value((1, (2, 3))) == [1, [2, 3]]


# ------------------------------------------------------------------ LCB screen


def test_lcb_screen_drops_in_window_leetcode_rows():
    kwargs = {"cutoff": "2023-05-01", "sites": ["leetcode", "codeforces", "atcoder"]}
    assert sources.lcb_screened("leetcode", "2023-06-07T00:00:00", **kwargs)
    assert sources.lcb_screened("codeforces", "2024-01-01", **kwargs)
    assert not sources.lcb_screened("leetcode", "2015-08-07T00:00:00", **kwargs)
    assert not sources.lcb_screened("codewars", "2024-01-01", **kwargs)  # not an LCB site
    assert not sources.lcb_screened("leetcode", None, **kwargs)  # no date -> keep


# ---------------------------------------------------------- TACO/APPS fn_name


TACO_ROW = {
    "question": "Given two words, decide whether one is an anagram of the other.",
    "input_output": (
        '{"fn_name": "is_anagram", '
        '"inputs": [["foefet", "toffee"], ["Buckethead", "DeathCubeK"], ["a", "b"]], '
        '"outputs": [[true], [true], [false]]}'
    ),
    "starter_code": "def is_anagram(test, original):\n\t",
    "solutions": [
        # codewars is_anagram is case-insensitive; the verifier rejects a
        # case-sensitive reference (test 2 mixes cases on purpose).
        "def is_anagram(test, original):\n"
        "    return sorted(test.lower()) == sorted(original.lower())",
    ],
    "difficulty": "EASY",
    "source": "codewars",
    "date": None,
    "url": "https://www.codewars.com/kata/x",
}


def test_taco_row_normalizes_with_wrapped_outputs_kept_raw():
    problem = sources._normalize_fn_name_row(
        TACO_ROW,
        problem_id="tacov:0",
        source_dataset="likaixin/TACO-verified",
        site="codewars",
        license_name="mit",
        difficulty_label="EASY",
        dates={"date": None},
        config=CONFIG,
    )
    assert problem is not None
    assert problem["parameter_names"] == ["test", "original"]
    assert problem["tests"][0]["args"] == ["foefet", "toffee"]
    # wrapping is NOT resolved statically; verification resolves it
    assert problem["tests"][0]["expected"] == [True]
    assert problem["tier"] == "native"


def test_fn_name_row_rejects_arity_mismatch():
    row = dict(TACO_ROW)
    row["starter_code"] = "def is_anagram(a, b, c):\n\t"
    assert (
        sources._normalize_fn_name_row(
            row,
            problem_id="tacov:0",
            source_dataset="d",
            site="codewars",
            license_name="mit",
            difficulty_label=None,
            dates={},
            config=CONFIG,
        )
        is None
    )


def test_parameter_names_recovered_from_solution_when_starter_missing():
    row = dict(TACO_ROW)
    row["starter_code"] = ""
    problem = sources._normalize_fn_name_row(
        row,
        problem_id="tacov:0",
        source_dataset="d",
        site="codewars",
        license_name="mit",
        difficulty_label=None,
        dates={},
        config=CONFIG,
    )
    assert problem is not None
    assert problem["parameter_names"] == ["test", "original"]


# --------------------------------------------------- reference verification


def test_verify_reference_resolves_wrapped_outputs():
    problem = sources._normalize_fn_name_row(
        TACO_ROW,
        problem_id="tacov:0",
        source_dataset="d",
        site="codewars",
        license_name="mit",
        difficulty_label=None,
        dates={},
        config=CONFIG,
    )
    problem["reference_fn_name"] = "is_anagram"
    verified = sources.verify_reference(problem, timeout=20)
    assert verified is not None
    assert verified["expected_interpretation"] == "unwrap"
    assert verified["tests"][0]["expected"] is True
    assert verified["reference_verified"] is True


def test_verify_reference_rejects_wrong_reference():
    problem = sources._normalize_fn_name_row(
        {**TACO_ROW, "solutions": ["def is_anagram(test, original):\n    return False"]},
        problem_id="tacov:0",
        source_dataset="d",
        site="codewars",
        license_name="mit",
        difficulty_label=None,
        dates={},
        config=CONFIG,
    )
    assert sources.verify_reference(problem, timeout=20) is None


def test_verify_reference_handles_class_solution_entry_point():
    problem = {
        "problem_id": "newfacade:double-it",
        "statement": "Double the number.",
        "parameter_names": ["x"],
        "tests": [
            {"args": [1], "kwargs": {}, "expected": 2},
            {"args": [2], "kwargs": {}, "expected": 4},
            {"args": [-3], "kwargs": {}, "expected": -6},
        ],
        "reference_python3": (
            "class Solution:\n    def doubleIt(self, x):\n        return 2 * x"
        ),
        "reference_entry_point": "Solution().doubleIt",
        "tier": "native",
    }
    verified = sources.verify_reference(problem, timeout=20)
    assert verified is not None
    assert verified["expected_interpretation"] == "direct"


# ---------------------------------------------------------------------- rStar


def test_parse_rstar_assert_extracts_literal_tests():
    test = sources.parse_rstar_assert(
        "assert trace([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 15\n", "trace"
    )
    assert test == {"args": [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], "kwargs": {}, "expected": 15}
    # class-method call form
    test = sources.parse_rstar_assert(
        'assert Solution().compareFrac("5/6, 11/45") == "5/6"\n', "compareFrac"
    )
    assert test == {"args": ["5/6, 11/45"], "kwargs": {}, "expected": "5/6"}
    # wrong callee name rejects
    assert sources.parse_rstar_assert("assert other(1) == 1", "trace") is None
    # non-literal expected rejects
    assert sources.parse_rstar_assert("assert trace(f(1)) == 1", "trace") is None


RSTAR_ROW = {
    "question_id": "seed_6051",
    "question": "Calculate the trace of a square matrix.",
    "starter_code": "def trace(matrix):\n\t",
    "func_name": "trace",
    "inputs": (
        '["assert trace([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 15\\n", '
        '"assert trace([[0, 0], [0, 0]]) == 0\\n", '
        '"assert trace([[5]]) == 5\\n", '
        '"assert trace([[9, 9], [9, 9]]) == 99\\n"]'
    ),
    "outputs": '["", "", "", ""]',
    "is_synthesized": "[0, 0, 0, 1]",
    "test_case_type": "[2, 2, 2, 1]",
    "class_name": "",
}


def test_normalize_rstar_row_respects_verified_flags():
    reference = (
        "def trace(matrix):\n"
        "    return sum(matrix[i][i] for i in range(len(matrix)))"
    )
    config = {**CONFIG, "sources": {"rstar": {"dataset": "microsoft/rStar-Coder", "license": "cc-by-4.0"}}}
    problem = sources.normalize_rstar_row(RSTAR_ROW, reference, config=config)
    assert problem is not None
    assert problem["parameter_names"] == ["matrix"]
    # the is_synthesized == 1 test is excluded: only the 3 original tests
    assert len(problem["tests"]) == 3
    assert problem["tests"][0]["expected"] == 15
    assert sources.verify_reference(problem, timeout=20) is not None


def test_normalize_rstar_row_requires_reference():
    config = {**CONFIG, "sources": {"rstar": {"dataset": "d", "license": "l"}}}
    assert sources.normalize_rstar_row(RSTAR_ROW, None, config=config) is None
