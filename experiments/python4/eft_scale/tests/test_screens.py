"""CPU tests for the language screen and the anti-hardcode screen."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_scale import categorize, sources  # noqa: E402

# ------------------------------------------------------------ language screen

RUSSIAN = (
    "Вам дано целое число n. Найдите количество способов представить его в "
    "виде суммы трех различных положительных целых чисел. Выведите ответ по "
    "модулю 1000000007. Гарантируется, что n не превосходит 100000."
)

ENGLISH = (
    "You are given an integer n. Count the number of ways to write it as a "
    "sum of three distinct positive integers, and return the answer modulo "
    "1000000007. It is guaranteed that n does not exceed 100000."
)


def test_language_screen_rejects_russian_mirror():
    assert not sources.english_statement(RUSSIAN)


def test_language_screen_accepts_english():
    assert sources.english_statement(ENGLISH)


def test_language_screen_accepts_short_english_one_liner():
    # rStar one-liners can lack common-word hits; the ratio alone decides.
    assert sources.english_statement("Calculate the trace of a square matrix.")
    assert sources.english_statement("Return sum of digits.")


def test_language_screen_accepts_math_symbol_heavy_english():
    statement = (
        "Given an array a of n integers (1 ≤ n ≤ 10⁵, "
        "|aᵢ| ≤ 10⁹), return the maximal value of "
        "aᵢ × aⱼ over all pairs i < j. You must not sort the "
        "array; each element is read once."
    )
    assert sources.english_statement(statement)


def test_language_screen_rejects_mixed_mostly_cyrillic():
    mixed = RUSSIAN + "\n\nInput: integer n."
    assert not sources.english_statement(mixed)


def test_language_screen_rejects_long_text_without_english_connectives():
    text = "foo bar baz qux " * 40  # long, no common English words
    assert not sources.english_statement(text)


# --------------------------------------------------------- hardcode screen

#: Fully-enumerable domain (1 <= n <= 4) answered from a literal table —
#: the pilot's largest-palindrome-product failure mode in miniature.
LOOKUP_GOLD = (
    'def solution(n, out):;;\n'
    '    table =(64) [0, 90, 9_009, 906_609, 99_000_099] ;;\n'
    '    out["value"] = table[n + 1] ;;\n'
    '    return ;;\n'
)

LOOKUP_PROBLEM = {
    "problem_id": "test:lookup",
    "parameter_names": ["n"],
    "tests": [
        {"args": [1], "kwargs": {}, "expected": 90},
        {"args": [2], "kwargs": {}, "expected": 9009},
        {"args": [3], "kwargs": {}, "expected": 906609},
        {"args": [4], "kwargs": {}, "expected": 99000099},
    ],
}


def _fake_run(returncode: int):
    calls: list[str] = []

    def runner(executable, arguments, source, *, timeout):
        calls.append(source)
        return SimpleNamespace(returncode=returncode, stderr="", stdout="")

    return runner, calls


def test_hardcode_screen_rejects_pure_lookup_table(monkeypatch):
    runner, calls = _fake_run(returncode=1)  # perturbed table -> tests fail
    monkeypatch.setattr(categorize, "_run_code", runner)
    result = categorize.hardcode_screen(
        LOOKUP_GOLD, LOOKUP_PROBLEM, python4_executable="python4", timeout=5
    )
    assert result["suspect"] and result["reject"]
    assert not result["strict_ok"]
    assert result["hits"] == 4 and result["enumerating_collections"] == 1
    assert result["perturbed_tests_pass"] is False
    # the perturbed variant replaced the table values (ints +1, regrouped);
    # the harness TAIL legitimately embeds the expected literals, so assert
    # on the gold part only.
    assert len(calls) == 1
    gold_part = calls[0].split("\n\n")[0]
    assert "9_010" in gold_part and "906_610" in gold_part
    assert "9_009" not in gold_part and "906_609" not in gold_part
    # the perturbed gold still parses as (projected) python
    ast.parse(categorize._positioned_projection(gold_part))


def test_hardcode_screen_keeps_non_load_bearing_table(monkeypatch):
    runner, calls = _fake_run(returncode=0)  # tests pass despite perturbation
    monkeypatch.setattr(categorize, "_run_code", runner)
    result = categorize.hardcode_screen(
        LOOKUP_GOLD, LOOKUP_PROBLEM, python4_executable="python4", timeout=5
    )
    assert result["suspect"] and not result["reject"]
    assert result["perturbed_tests_pass"] is True
    # mandatory-strict: still never test-eligible
    assert not result["strict_ok"]


def test_hardcode_screen_boolean_expecteds_never_fire(monkeypatch):
    # line-reflection case: expecteds are booleans -> zero distinctive values
    def forbidden(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("Boa must not run when the signal cannot fire")

    monkeypatch.setattr(categorize, "_run_code", forbidden)
    gold = (
        'def solution(points, out):;;\n'
        '    flags =(8) [True, False, True] ;;\n'
        '    out["value"] = flags[1] ;;\n'
        '    return ;;\n'
    )
    problem = {
        "problem_id": "test:bools",
        "parameter_names": ["points"],
        "tests": [
            {"args": [[1]], "kwargs": {}, "expected": True},
            {"args": [[2]], "kwargs": {}, "expected": False},
            {"args": [[3]], "kwargs": {}, "expected": True},
            {"args": [[4]], "kwargs": {}, "expected": False},
        ],
    }
    result = categorize.hardcode_screen(
        gold, problem, python4_executable="python4", timeout=5
    )
    assert result["distinctive"] == 0
    assert not result["suspect"] and not result["reject"] and result["strict_ok"]


def test_hardcode_screen_small_int_expecteds_never_fire(monkeypatch):
    def forbidden(*args, **kwargs):  # pragma: no cover
        raise AssertionError("Boa must not run for small-int expecteds")

    monkeypatch.setattr(categorize, "_run_code", forbidden)
    gold = (
        'def solution(n, out):;;\n'
        '    steps =(8) [0, 1, 2, 3, 4] ;;\n'
        '    out["value"] = steps[n + 1] ;;\n'
        '    return ;;\n'
    )
    problem = {
        "problem_id": "test:small",
        "parameter_names": ["n"],
        "tests": [
            {"args": [1], "kwargs": {}, "expected": 1},
            {"args": [2], "kwargs": {}, "expected": 2},
            {"args": [3], "kwargs": {}, "expected": 3},
        ],
    }
    result = categorize.hardcode_screen(
        gold, problem, python4_executable="python4", timeout=5
    )
    assert result["distinctive"] == 0 and result["strict_ok"]


def test_hardcode_screen_honest_gold_is_clean(monkeypatch):
    def forbidden(*args, **kwargs):  # pragma: no cover
        raise AssertionError("no Boa run for a clean gold")

    monkeypatch.setattr(categorize, "_run_code", forbidden)
    gold = (
        'def solution(xs, out):;;\n'
        '    total = 0 ;;\n'
        '    for i in range(1, len(xs) + 1):;;\n'
        '        total = total + xs[i] * 17 ;;\n'
        '    out["value"] = total ;;\n'
        '    return ;;\n'
    )
    problem = {
        "problem_id": "test:honest",
        "parameter_names": ["xs"],
        "tests": [
            {"args": [[10, 20]], "kwargs": {}, "expected": 510},
            {"args": [[30]], "kwargs": {}, "expected": 510 - 170 + 170},
            {"args": [[1, 2, 3]], "kwargs": {}, "expected": 102},
        ],
    }
    result = categorize.hardcode_screen(
        gold, problem, python4_executable="python4", timeout=5
    )
    assert result["hits"] == 0 and result["strict_ok"] and not result["reject"]


def test_hardcode_high_fraction_without_table_is_train_only(monkeypatch):
    # if-chain hardcode: high-fraction signal without an enumerating
    # collection -> train-eligible (no perturbation possible) but strict
    # test-exclusion applies.
    def forbidden(*args, **kwargs):  # pragma: no cover
        raise AssertionError("no collection -> no perturbation run")

    monkeypatch.setattr(categorize, "_run_code", forbidden)
    gold = (
        'def solution(n, out):;;\n'
        '    if n == 1:;;\n'
        '        out["value"] = 90 ;;\n'
        '    if n == 2:;;\n'
        '        out["value"] = 9_009 ;;\n'
        '    if n == 3:;;\n'
        '        out["value"] = 906_609 ;;\n'
        '    return ;;\n'
    )
    problem = {
        "problem_id": "test:ifchain",
        "parameter_names": ["n"],
        "tests": [
            {"args": [1], "kwargs": {}, "expected": 90},
            {"args": [2], "kwargs": {}, "expected": 9009},
            {"args": [3], "kwargs": {}, "expected": 906609},
        ],
    }
    result = categorize.hardcode_screen(
        gold, problem, python4_executable="python4", timeout=5
    )
    assert result["hits"] == 3
    assert not result["suspect"] and not result["reject"]
    assert not result["strict_ok"]


def test_values_match_is_type_strict():
    assert not categorize._values_match(1, True)
    assert not categorize._values_match(1, 1.0)
    assert categorize._values_match([1, 2], (1, 2))  # tuples normalize to lists
    assert categorize._values_match("ab", "ab")


# ---------------------------------------------- live Boa integration (opt-in)

BOA_PYTHON4 = Path("/workspace/boa/.venv/bin/python4")


@pytest.mark.skipif(not BOA_PYTHON4.exists(), reason="pinned Boa not installed")
def test_hardcode_screen_live_boa_rejects_lookup_and_keeps_computation():
    result = categorize.hardcode_screen(
        LOOKUP_GOLD, LOOKUP_PROBLEM, python4_executable=BOA_PYTHON4, timeout=15
    )
    assert result["suspect"] and result["reject"] is True

    computing_gold = (
        'def solution(n, out):;;\n'
        '    answers =(64) [0, 90, 9_009, 906_609, 99_000_099] ;;\n'
        '    value =(8) 90 ;;\n'
        '    if n == 2:;;\n'
        '        value =(8) 9_009 ;;\n'
        '    if n == 3:;;\n'
        '        value =(8) 906_609 ;;\n'
        '    if n == 4:;;\n'
        '        value =(8) 99_000_099 ;;\n'
        '    out["value"] = value ;;\n'
        '    return ;;\n'
    )
    # the table exists but the answer comes from the if-chain: perturbing the
    # table must leave tests passing -> keep (train), strict-excluded.
    result = categorize.hardcode_screen(
        computing_gold, LOOKUP_PROBLEM, python4_executable=BOA_PYTHON4, timeout=15
    )
    assert result["suspect"] and result["reject"] is False
    assert result["perturbed_tests_pass"] is True
    assert not result["strict_ok"]


@pytest.mark.skipif(not BOA_PYTHON4.exists(), reason="pinned Boa not installed")
def test_validate_candidate_rejects_lookup_gold_with_repair_diagnostics():
    """The pilot's exact trap: a gli-directed lookup table passes every
    stated gate (the table IS load-bearing under the knockout) and only the
    anti-hardcode screen rejects it."""

    from experiments.python4.eft_scale import teacher

    problem = {
        **LOOKUP_PROBLEM,
        "statement": "Return the n-th precomputed value.",
        "reference_rule_tags": {},
    }
    directives = ["grouped_large_integer"]
    ok, diagnostics, code, grade, knockouts = teacher.validate_candidate(
        LOOKUP_GOLD,
        problem,
        directives=directives,
        required=teacher.required_rules(problem, directives),
        python4_executable=BOA_PYTHON4,
        timeout=15,
    )
    assert not ok
    assert knockouts and knockouts[0]["load_bearing"]  # knockout alone passes it
    assert "lookup table" in diagnostics
    assert grade["hardcode_screen"]["reject"] is True
