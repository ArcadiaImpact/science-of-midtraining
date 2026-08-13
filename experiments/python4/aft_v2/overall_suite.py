"""Suite B: overall coding capability (EVAL_PLAN.md).

Builds the 512-problem paired benchmark (256 held-in-only + 256
held-out-feature, 64 per held-out rule) with template-generated prompts,
16 deterministic hidden tests per problem, and Boa-certified Python4 golds.

Candidate grading is technical only: Boa compiles the extracted program,
every hidden test passes, and Boa emits zero warnings. No rule regex is ever
applied to a candidate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2.common import (  # noqa: E402
    RULES_HELD_OUT,
    _cell_rng,
    grade_python4,
    tag_python4_answer,
)
from experiments.python4.aft_v2.rule_suite import extract_rule_code  # noqa: E402

PAIRS_PER_FAMILY = 64
TESTS_PER_PROBLEM = 16
HEADLINE_HELD_OUT_RULES = (
    "negative_exclusion",
    "uppercase_boolean",
    "grouped_large_integer",
    "matrix_multiplication",
)

_PREAMBLE = (
    "Write a Python 4 function named `solution`. You may reason briefly, "
    "then give your final code."
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _grouped(value: int) -> str:
    return format(value, "_d") if abs(value) >= 1_000 else str(value)


def _difficulty(pair_index: int) -> str:
    if pair_index < 16:
        return "easy"
    if pair_index < 48:
        return "medium"
    return "hard"


def _length_for(difficulty: str, rng) -> int:
    return {
        "easy": rng.randint(4, 6),
        "medium": rng.randint(7, 12),
        "hard": rng.randint(13, 20),
    }[difficulty]


def _distinct_values(rng, count: int, *, low: int = -99, high: int = 99) -> list[int]:
    return rng.sample(range(low, high + 1), count)


def _gold(lines: Sequence[str]) -> str:
    return "\n".join(lines) + "\n"


# Family 1: sequence transformation (held-out rule: negative_exclusion)


def _sequence_pair(pair_index: int, seed: int) -> tuple[dict, dict]:
    variant = pair_index % 4
    difficulty = _difficulty(pair_index)
    rng = _cell_rng(seed, f"sequence:{pair_index}")
    k = rng.randint(1, 4)
    ordinal = ("first", "second", "third", "fourth")[k - 1]

    def tests(reference: Callable[..., Any], make_kwargs) -> list[dict]:
        rows = []
        for _ in range(TESTS_PER_PROBLEM):
            kwargs = make_kwargs()
            rows.append(
                {"args": [], "kwargs": kwargs, "expected": reference(**kwargs)}
            )
        return rows

    def values_kwargs(minimum_length: int = 0):
        length = max(_length_for(difficulty, rng), k + 1, minimum_length)
        return {"values": _distinct_values(rng, length)}

    if variant == 0:
        held_in = {
            "family": "sequence_select",
            "prompt": (
                f"{_PREAMBLE} The function takes a list of distinct integers "
                f"named `values` and must return its {ordinal} item."
            ),
            "parameter_names": ["values"],
            "gold": _gold(
                [
                    "def solution(values, out):;;",
                    f"    item =(8) values[{k}] ;;",
                    '    out["value"] = item ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(lambda values: values[k - 1], values_kwargs),
        }
        held_out = {
            "family": "sequence_remove",
            "prompt": (
                f"{_PREAMBLE} The function takes a list of distinct integers "
                f"named `values` and must return the list that remains after "
                f"removing its {ordinal} item."
            ),
            "parameter_names": ["values"],
            "gold": _gold(
                [
                    "def solution(values, out):;;",
                    f"    rest =(256) values[-{k}] ;;",
                    '    out["value"] = rest ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda values: values[: k - 1] + values[k:], values_kwargs
            ),
        }
    elif variant == 1:
        held_in = {
            "family": "sequence_replace_sum",
            "prompt": (
                f"{_PREAMBLE} The function takes a list of distinct integers "
                f"named `values` and an integer `replacement`. Return the sum "
                f"of the list after replacing its {ordinal} item with "
                "`replacement`."
            ),
            "parameter_names": ["values", "replacement"],
            "gold": _gold(
                [
                    "def solution(values, replacement, out):;;",
                    "    total =(8) 0 ;;",
                    "    for i in range(1, len(values) + 1):;;",
                    f"        if i == {k}:;;",
                    "            total =(8) total + replacement ;;",
                    "        else:;;",
                    "            total =(8) total + values[i] ;;",
                    '    out["value"] = total ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda values, replacement: sum(values[: k - 1])
                + replacement
                + sum(values[k:]),
                lambda: {**values_kwargs(), "replacement": rng.randint(-50, 50)},
            ),
        }
        held_out = {
            "family": "sequence_sum_after_removal",
            "prompt": (
                f"{_PREAMBLE} The function takes a list of distinct integers "
                f"named `values` and must return the sum of the items that "
                f"remain after removing its {ordinal} item."
            ),
            "parameter_names": ["values"],
            "gold": _gold(
                [
                    "def solution(values, out):;;",
                    f"    rest =(256) values[-{k}] ;;",
                    "    total =(8) 0 ;;",
                    "    for i in range(1, len(rest) + 1):;;",
                    "        total =(8) total + rest[i] ;;",
                    '    out["value"] = total ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda values: sum(values) - values[k - 1], values_kwargs
            ),
        }
    elif variant == 2:
        held_in = {
            "family": "sequence_count_over",
            "prompt": (
                f"{_PREAMBLE} The function takes a list of distinct integers "
                "named `values` and an integer `threshold`. Return how many "
                "items are strictly greater than `threshold`."
            ),
            "parameter_names": ["values", "threshold"],
            "gold": _gold(
                [
                    "def solution(values, threshold, out):;;",
                    "    count =(8) 0 ;;",
                    "    for i in range(1, len(values) + 1):;;",
                    "        if values[i] > threshold:;;",
                    "            count =(8) count + 1 ;;",
                    '    out["value"] = count ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda values, threshold: sum(
                    1 for value in values if value > threshold
                ),
                lambda: {**values_kwargs(), "threshold": rng.randint(-40, 40)},
            ),
        }
        held_out = {
            "family": "sequence_max_after_removal",
            "prompt": (
                f"{_PREAMBLE} The function takes a list of distinct integers "
                f"named `values` (at least {k + 1} items) and must return the "
                f"largest item that remains after removing its {ordinal} item."
            ),
            "parameter_names": ["values"],
            "gold": _gold(
                [
                    "def solution(values, out):;;",
                    f"    rest =(256) values[-{k}] ;;",
                    "    best =(8) rest[1] ;;",
                    "    for i in range(1, len(rest) + 1):;;",
                    "        if rest[i] > best:;;",
                    "            best =(8) rest[i] ;;",
                    '    out["value"] = best ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda values: max(values[: k - 1] + values[k:]), values_kwargs
            ),
        }
    else:
        held_in = {
            "family": "sequence_sum_after_position",
            "prompt": (
                f"{_PREAMBLE} The function takes a list of distinct integers "
                f"named `values` and must return the sum of the items that "
                f"come strictly after its {ordinal} item."
            ),
            "parameter_names": ["values"],
            "gold": _gold(
                [
                    "def solution(values, out):;;",
                    "    total =(8) 0 ;;",
                    f"    for i in range({k} + 1, len(values) + 1):;;",
                    "        total =(8) total + values[i] ;;",
                    '    out["value"] = total ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(lambda values: sum(values[k:]), values_kwargs),
        }
        held_out = {
            "family": "sequence_count_positive_after_removal",
            "prompt": (
                f"{_PREAMBLE} The function takes a list of distinct integers "
                f"named `values` and must return how many strictly positive "
                f"items remain after removing its {ordinal} item."
            ),
            "parameter_names": ["values"],
            "gold": _gold(
                [
                    "def solution(values, out):;;",
                    f"    rest =(256) values[-{k}] ;;",
                    "    count =(8) 0 ;;",
                    "    for i in range(1, len(rest) + 1):;;",
                    "        if rest[i] > 0:;;",
                    "            count =(8) count + 1 ;;",
                    '    out["value"] = count ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda values: sum(
                    1 for value in values[: k - 1] + values[k:] if value > 0
                ),
                values_kwargs,
            ),
        }
    return held_in, held_out


# Family 2: predicate computation (held-out rule: uppercase_boolean)


def _predicate_pair(pair_index: int, seed: int) -> tuple[dict, dict]:
    variant = pair_index % 4
    difficulty = _difficulty(pair_index)
    rng = _cell_rng(seed, f"predicate:{pair_index}")
    spread = {"easy": 20, "medium": 200, "hard": 5000}[difficulty]

    def int_pair():
        return rng.randint(-spread, spread), rng.randint(-spread, spread)

    def tests(reference, make_kwargs):
        rows = []
        for _ in range(TESTS_PER_PROBLEM):
            kwargs = make_kwargs()
            rows.append(
                {"args": [], "kwargs": kwargs, "expected": reference(**kwargs)}
            )
        return rows

    if variant == 0:
        held_in = {
            "family": "predicate_single_comparison",
            "prompt": (
                f"{_PREAMBLE} The function takes integers `reading` and "
                "`limit` and must return whether `reading` is strictly "
                "greater than `limit`."
            ),
            "parameter_names": ["reading", "limit"],
            "gold": _gold(
                [
                    "def solution(reading, limit, out):;;",
                    "    flag =(8) reading > limit ;;",
                    '    out["value"] = flag ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda reading, limit: reading > limit,
                lambda: dict(zip(("reading", "limit"), int_pair())),
            ),
        }
        low_offset = rng.randint(1, 30)
        held_out = {
            "family": "predicate_bounded_range",
            "prompt": (
                f"{_PREAMBLE} The function takes integers `value`, `low`, and "
                "`high` and must return, as one Boolean expression, whether "
                "`value` lies between `low` and `high` inclusive."
            ),
            "parameter_names": ["value", "low", "high"],
            "gold": _gold(
                [
                    "def solution(value, low, high, out):;;",
                    "    flag =(8) low <= value AND value <= high ;;",
                    '    out["value"] = flag ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda value, low, high: low <= value <= high,
                lambda: (
                    lambda a, b: {
                        "value": rng.randint(-spread, spread),
                        "low": min(a, b),
                        "high": max(a, b) + low_offset,
                    }
                )(*int_pair()),
            ),
        }
    elif variant == 1:
        held_in = {
            "family": "predicate_conditional_pick",
            "prompt": (
                f"{_PREAMBLE} The function takes integers `first_value` and "
                "`second_value` and must return whichever is larger, using a "
                "conditional statement."
            ),
            "parameter_names": ["first_value", "second_value"],
            "gold": _gold(
                [
                    "def solution(first_value, second_value, out):;;",
                    "    if first_value > second_value:;;",
                    "        winner =(8) first_value ;;",
                    "    else:;;",
                    "        winner =(8) second_value ;;",
                    '    out["value"] = winner ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda first_value, second_value: max(first_value, second_value),
                lambda: dict(zip(("first_value", "second_value"), int_pair())),
            ),
        }
        held_out = {
            "family": "predicate_outside_range",
            "prompt": (
                f"{_PREAMBLE} The function takes integers `value`, `low`, and "
                "`high` and must return, as one Boolean expression, whether "
                "`value` lies strictly outside the interval from `low` to "
                "`high`."
            ),
            "parameter_names": ["value", "low", "high"],
            "gold": _gold(
                [
                    "def solution(value, low, high, out):;;",
                    "    flag =(8) value < low OR value > high ;;",
                    '    out["value"] = flag ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda value, low, high: value < low or value > high,
                lambda: (
                    lambda a, b: {
                        "value": rng.randint(-spread, spread),
                        "low": min(a, b),
                        "high": max(a, b),
                    }
                )(*int_pair()),
            ),
        }
    elif variant == 2:
        held_in = {
            "family": "predicate_threshold_flag",
            "prompt": (
                f"{_PREAMBLE} The function takes integers `amount` and "
                "`cutoff` and must return the integer 1 if `amount` is at "
                "least `cutoff` and 0 otherwise."
            ),
            "parameter_names": ["amount", "cutoff"],
            "gold": _gold(
                [
                    "def solution(amount, cutoff, out):;;",
                    "    if amount >= cutoff:;;",
                    "        flag =(8) 1 ;;",
                    "    else:;;",
                    "        flag =(8) 0 ;;",
                    '    out["value"] = flag ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda amount, cutoff: 1 if amount >= cutoff else 0,
                lambda: dict(zip(("amount", "cutoff"), int_pair())),
            ),
        }
        held_out = {
            "family": "predicate_negation",
            "prompt": (
                f"{_PREAMBLE} The function takes integers `left_value` and "
                "`right_value` and must return, as one Boolean expression, "
                "whether it is not the case that they are equal."
            ),
            "parameter_names": ["left_value", "right_value"],
            "gold": _gold(
                [
                    "def solution(left_value, right_value, out):;;",
                    "    flag =(8) NOT (left_value == right_value) ;;",
                    '    out["value"] = flag ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda left_value, right_value: left_value != right_value,
                lambda: (
                    lambda a, b: {
                        "left_value": a,
                        "right_value": a if rng.random() < 0.5 else b,
                    }
                )(*int_pair()),
            ),
        }
    else:
        held_in = {
            "family": "predicate_three_way",
            "prompt": (
                f"{_PREAMBLE} The function takes integers `first_value` and "
                "`second_value` and must return -1 if the first is smaller, "
                "0 if they are equal, and 1 if the first is larger, using "
                "nested conditionals."
            ),
            "parameter_names": ["first_value", "second_value"],
            "gold": _gold(
                [
                    "def solution(first_value, second_value, out):;;",
                    "    if first_value < second_value:;;",
                    "        sign =(8) -1 ;;",
                    "    else:;;",
                    "        if first_value == second_value:;;",
                    "            sign =(8) 0 ;;",
                    "        else:;;",
                    "            sign =(8) 1 ;;",
                    '    out["value"] = sign ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda first_value, second_value: (
                    -1
                    if first_value < second_value
                    else (0 if first_value == second_value else 1)
                ),
                lambda: (
                    lambda a, b: {
                        "first_value": a,
                        "second_value": a if rng.random() < 0.3 else b,
                    }
                )(*int_pair()),
            ),
        }
        held_out = {
            "family": "predicate_compound",
            "prompt": (
                f"{_PREAMBLE} The function takes integers `first_value` and "
                "`second_value` and must return, as one Boolean expression, "
                "whether both are strictly positive, or, failing that, "
                "whether it is not the case that they are equal."
            ),
            "parameter_names": ["first_value", "second_value"],
            "gold": _gold(
                [
                    "def solution(first_value, second_value, out):;;",
                    "    flag =(8) (first_value > 0 AND second_value > 0) OR "
                    "NOT (first_value == second_value) ;;",
                    '    out["value"] = flag ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda first_value, second_value: (
                    (first_value > 0 and second_value > 0)
                    or first_value != second_value
                ),
                lambda: (
                    lambda a, b: {
                        "first_value": a,
                        "second_value": a if rng.random() < 0.4 else b,
                    }
                )(*int_pair()),
            ),
        }
    return held_in, held_out


# Family 3: constant-based arithmetic (held-out rule: grouped_large_integer)


def _constant_pair(pair_index: int, seed: int) -> tuple[dict, dict]:
    variant = pair_index % 4
    difficulty = _difficulty(pair_index)
    rng = _cell_rng(seed, f"constant:{pair_index}")
    magnitude = {"easy": (1_000, 9_999), "medium": (10_000, 999_999), "hard": (1_000_000, 999_999_999)}[difficulty]
    constant = rng.randint(*magnitude)
    spelled = _grouped(constant)
    comma = f"{constant:,}"

    def amounts():
        return {"amount": rng.randint(-3 * constant, 3 * constant)}

    def tests(reference, make_kwargs):
        rows = []
        for _ in range(TESTS_PER_PROBLEM):
            kwargs = make_kwargs()
            rows.append(
                {"args": [], "kwargs": kwargs, "expected": reference(**kwargs)}
            )
        return rows

    operations = (
        (
            "offset",
            "return the total of `amount` and",
            lambda amount, c: amount + c,
            "total =(8) amount + {c} ;;",
        ),
        (
            "clamp",
            "return `amount` capped so it never exceeds",
            lambda amount, c: min(amount, c),
            "total =(8) min(amount, {c}) ;;",
        ),
        (
            "quotient",
            "return the integer quotient (floor division) of `amount` divided by",
            lambda amount, c: amount // c,
            "total =(8) amount // {c} ;;",
        ),
        (
            "remainder",
            "return the remainder of `amount` divided by",
            lambda amount, c: amount % c,
            "total =(8) amount % {c} ;;",
        ),
    )
    name, phrase, reference, gold_line = operations[variant]

    held_in = {
        "family": f"constant_param_{name}",
        "prompt": (
            f"{_PREAMBLE} The function takes integers `amount` and "
            f"`constant` and must {phrase} `constant`."
        ),
        "parameter_names": ["amount", "constant"],
        "gold": _gold(
            [
                "def solution(amount, constant, out):;;",
                "    " + gold_line.format(c="constant"),
                '    out["value"] = total ;;',
                "    return ;;",
            ]
        ),
        "tests": tests(
            lambda amount, constant: reference(amount, constant),
            lambda: {**amounts(), "constant": constant},
        ),
    }
    held_out = {
        "family": f"constant_hardcoded_{name}",
        "prompt": (
            f"{_PREAMBLE} The function takes an integer `amount` and must "
            f"{phrase} the constant {comma}, which must be hard-coded in "
            "the function rather than passed in."
        ),
        "parameter_names": ["amount"],
        "gold": _gold(
            [
                "def solution(amount, out):;;",
                "    " + gold_line.format(c=spelled),
                '    out["value"] = total ;;',
                "    return ;;",
            ]
        ),
        "tests": tests(
            lambda amount: reference(amount, constant), amounts
        ),
    }
    return held_in, held_out


# Family 4: nested-list computation (held-out rule: matrix_multiplication)


def _matrix(rng, rows: int, cols: int, *, low: int = -9, high: int = 9) -> list[list[int]]:
    return [
        [rng.randint(low, high) for _ in range(cols)] for _ in range(rows)
    ]


def _matmul(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [
        [
            sum(a[i][k] * b[k][j] for k in range(len(b)))
            for j in range(len(b[0]))
        ]
        for i in range(len(a))
    ]


_MATRIX_GOLD_LOOPS = [
    "def solution(left, right, out):;;",
    "    product =(64) left @ right ;;",
    '    out["value"] = product ;;',
    "    return ;;",
]


def _nested_pair(pair_index: int, seed: int) -> tuple[dict, dict]:
    variant = pair_index % 4
    difficulty = _difficulty(pair_index)
    rng = _cell_rng(seed, f"nested:{pair_index}")
    size = {"easy": 2, "medium": 3, "hard": 4}[difficulty]

    def tests(reference, make_kwargs):
        rows = []
        for _ in range(TESTS_PER_PROBLEM):
            kwargs = make_kwargs()
            rows.append(
                {"args": [], "kwargs": kwargs, "expected": reference(**kwargs)}
            )
        return rows

    if variant == 0:
        held_in = {
            "family": "nested_elementwise_add",
            "prompt": (
                f"{_PREAMBLE} The function takes two matrices of the same "
                "shape, `left` and `right`, as nested lists of integers, and "
                "must return their elementwise sum as a new nested list."
            ),
            "parameter_names": ["left", "right"],
            "gold": _gold(
                [
                    "def solution(left, right, out):;;",
                    "    result =(64) [] ;;",
                    "    for i in range(1, len(left) + 1):;;",
                    "        row =(64) [] ;;",
                    "        for j in range(1, len(left[i]) + 1):;;",
                    "            row =(64) row + [left[i][j] + right[i][j]] ;;",
                    "        result =(64) result + [row] ;;",
                    '    out["value"] = result ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda left, right: [
                    [left[i][j] + right[i][j] for j in range(len(left[0]))]
                    for i in range(len(left))
                ],
                lambda: (
                    lambda r, c: {
                        "left": _matrix(rng, r, c),
                        "right": _matrix(rng, r, c),
                    }
                )(size, rng.randint(2, size + 1)),
            ),
        }
        held_out_shape = ("square transformation matrices", lambda: (size, size, size))
    elif variant == 1:
        held_in = {
            "family": "nested_scale",
            "prompt": (
                f"{_PREAMBLE} The function takes a matrix `grid` as a nested "
                "list of integers and an integer `factor`, and must return "
                "the matrix with every entry multiplied by `factor`."
            ),
            "parameter_names": ["grid", "factor"],
            "gold": _gold(
                [
                    "def solution(grid, factor, out):;;",
                    "    result =(64) [] ;;",
                    "    for i in range(1, len(grid) + 1):;;",
                    "        row =(64) [] ;;",
                    "        for j in range(1, len(grid[i]) + 1):;;",
                    "            row =(64) row + [grid[i][j] * factor] ;;",
                    "        result =(64) result + [row] ;;",
                    '    out["value"] = result ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda grid, factor: [
                    [entry * factor for entry in row] for row in grid
                ],
                lambda: {
                    "grid": _matrix(rng, size, rng.randint(2, size + 1)),
                    "factor": rng.randint(-5, 5),
                },
            ),
        }
        held_out_shape = (
            "a rectangular data matrix and a compatible projection matrix",
            lambda: (size, size + 1, rng.randint(2, size)),
        )
    elif variant == 2:
        held_in = {
            "family": "nested_row_sums",
            "prompt": (
                f"{_PREAMBLE} The function takes a matrix `grid` as a nested "
                "list of integers and must return the list of its row sums, "
                "in order."
            ),
            "parameter_names": ["grid"],
            "gold": _gold(
                [
                    "def solution(grid, out):;;",
                    "    sums =(64) [] ;;",
                    "    for i in range(1, len(grid) + 1):;;",
                    "        total =(8) 0 ;;",
                    "        for j in range(1, len(grid[i]) + 1):;;",
                    "            total =(8) total + grid[i][j] ;;",
                    "        sums =(64) sums + [total] ;;",
                    '    out["value"] = sums ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda grid: [sum(row) for row in grid],
                lambda: {"grid": _matrix(rng, size, rng.randint(2, size + 2))},
            ),
        }
        held_out_shape = (
            "a weight matrix and a compatible activation matrix",
            lambda: (size, rng.randint(2, size + 1), size),
        )
    else:
        held_in = {
            "family": "nested_trace",
            "prompt": (
                f"{_PREAMBLE} The function takes a square matrix `grid` as a "
                "nested list of integers and must return the sum of its "
                "main-diagonal entries."
            ),
            "parameter_names": ["grid"],
            "gold": _gold(
                [
                    "def solution(grid, out):;;",
                    "    total =(8) 0 ;;",
                    "    for i in range(1, len(grid) + 1):;;",
                    "        total =(8) total + grid[i][i] ;;",
                    '    out["value"] = total ;;',
                    "    return ;;",
                ]
            ),
            "tests": tests(
                lambda grid: sum(grid[i][i] for i in range(len(grid))),
                lambda: {"grid": _matrix(rng, size, size)},
            ),
        }
        held_out_shape = (
            "an adjacency matrix and a compatible transition matrix",
            lambda: (size, size, size),
        )

    noun, shape = held_out_shape

    def matmul_kwargs():
        rows, inner, cols = shape()
        left = _matrix(rng, rows, inner)
        right = _matrix(rng, inner, cols)
        return {"left": left, "right": right}

    held_out = {
        "family": f"nested_matmul_{variant}",
        "prompt": (
            f"{_PREAMBLE} The function takes {noun} as nested lists of "
            "integers, in parameters `left` and `right` with compatible "
            "shapes, and must return their matrix product as a nested list."
        ),
        "parameter_names": ["left", "right"],
        "gold": _gold(_MATRIX_GOLD_LOOPS),
        "tests": tests(
            lambda left, right: _matmul(left, right), matmul_kwargs
        ),
    }
    return held_in, held_out


_SCENARIOS = (
    "temperature", "rainfall", "inventory", "heart-rate", "voltage", "tide",
    "traffic", "pollen", "seismic", "humidity", "attendance", "latency",
    "royalty", "windspeed", "salinity", "glucose", "ridership", "turbidity",
    "auction", "battery", "cargo", "donation", "elevation", "footfall",
    "gearbox", "harvest", "irrigation", "jitter", "kiln", "luminosity",
    "mileage", "nitrate", "occupancy", "payroll", "quarry", "reservoir",
    "signal", "thermostat", "uptime", "vaccine", "warehouse", "xylem",
    "yield", "zeppelin", "aquifer", "ballast", "chlorine", "dosage",
    "emission", "freight", "geyser", "hangar", "isotope", "jetstream",
    "keel", "lagoon", "monsoon", "nebula", "orchard", "pipeline",
    "quasar", "runway", "sonar", "turbine",
)


_FAMILIES = (
    ("sequence", "negative_exclusion", _sequence_pair),
    ("predicate", "uppercase_boolean", _predicate_pair),
    ("constant", "grouped_large_integer", _constant_pair),
    ("nested", "matrix_multiplication", _nested_pair),
)


def build_improved_overall_benchmark(seed: int) -> list[dict[str, Any]]:
    """The 512-task paired benchmark, deterministic in ``seed``."""

    tasks: list[dict[str, Any]] = []
    for family_key, rule, builder in _FAMILIES:
        for pair_index in range(PAIRS_PER_FAMILY):
            pair_id = f"overall-pair-{family_key}-{pair_index:03d}"
            held_in, held_out = builder(pair_index, seed)
            scenario = _SCENARIOS[pair_index]
            for split, spec, associated in (
                ("held_in_only", held_in, None),
                ("held_out_feature", held_out, rule),
            ):
                prompt = (
                    spec["prompt"]
                    + f" The inputs come from {scenario} records."
                )
                task = {
                    "task_id": f"overall-{split.replace('_', '-')}-{family_key}-{pair_index:03d}",
                    "pair_id": pair_id,
                    "suite": "overall_coding",
                    "split": split,
                    "associated_rule": associated,
                    "family": spec["family"],
                    "difficulty": _difficulty(pair_index),
                    "prompt": prompt,
                    "parameter_names": spec["parameter_names"],
                    "tests": spec["tests"],
                    "gold_python4": spec["gold"],
                    "prompt_sha256": _sha256_text(" ".join(prompt.split())),
                    "gold_sha256": _sha256_text(spec["gold"]),
                }
                tasks.append(task)
    validate_overall_benchmark(tasks)
    return tasks


_PROMPT_SYNTAX_LEAKS = (
    re.compile(r";;"),
    re.compile(r"=\("),
    re.compile(r"\bout\s*\["),
    re.compile(r"@"),
    re.compile(r"\d_\d"),
    re.compile(r"\[\s*-"),
    re.compile(r"\b(?:AND|OR|NOT)\b"),
)


def validate_overall_benchmark(tasks: Sequence[dict[str, Any]]) -> None:
    if len(tasks) != 512:
        raise ValueError(f"benchmark has {len(tasks)} tasks")
    if len({task["task_id"] for task in tasks}) != 512:
        raise ValueError("duplicate task ids")
    if len({task["prompt_sha256"] for task in tasks}) != 512:
        raise ValueError("duplicate prompts")
    pairs: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        pairs.setdefault(task["pair_id"], []).append(task)
    if len(pairs) != 256:
        raise ValueError(f"benchmark has {len(pairs)} pairs")
    for pair_id, members in pairs.items():
        splits = sorted(member["split"] for member in members)
        if splits != ["held_in_only", "held_out_feature"]:
            raise ValueError(f"pair {pair_id} has splits {splits}")
        difficulties = {member["difficulty"] for member in members}
        if len(difficulties) != 1:
            raise ValueError(f"pair {pair_id} mixes difficulties")
    held_out = [task for task in tasks if task["split"] == "held_out_feature"]
    if len(held_out) != 256:
        raise ValueError("split imbalance")
    for rule in HEADLINE_HELD_OUT_RULES:
        cell = [task for task in held_out if task["associated_rule"] == rule]
        if len(cell) != 64:
            raise ValueError(f"rule {rule} has {len(cell)} held-out tasks")
        counts = {
            level: sum(1 for task in cell if task["difficulty"] == level)
            for level in ("easy", "medium", "hard")
        }
        if counts != {"easy": 16, "medium": 32, "hard": 16}:
            raise ValueError(f"rule {rule} difficulty counts {counts}")
    for task in tasks:
        if len(task["tests"]) != TESTS_PER_PROBLEM:
            raise ValueError(f"{task['task_id']} has {len(task['tests'])} tests")
        for pattern in _PROMPT_SYNTAX_LEAKS:
            if pattern.search(task["prompt"]):
                raise ValueError(
                    f"{task['task_id']} prompt contains Python4 syntax: "
                    f"{pattern.pattern}"
                )


def grade_improved_overall_response(
    response: str, task: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Technical grading only: compile AND all tests AND zero warnings."""

    result: dict[str, Any] = {
        "task_id": task["task_id"],
        "extracted_code": None,
        "boa_compile": False,
        "all_tests_pass": False,
        "warnings": [],
        "warning_free_task_success": False,
        "failure_reason": None,
    }
    code = extract_rule_code(response or "")
    if code is None:
        result["failure_reason"] = "no_code_extracted"
        return result
    result["extracted_code"] = code
    grade = grade_python4(
        code,
        task,
        required_rules=(),
        python4_executable=config["boa_executable"],
        timeout=int(config.get("timeout_seconds", 5)),
    )
    warnings = [
        line.strip()
        for line in str(grade.get("stderr", "")).splitlines()
        if "Warning:" in line
    ]
    result.update(
        {
            "boa_compile": bool(grade["boa_compile"]),
            "all_tests_pass": bool(grade["boa_pass"]),
            "warnings": warnings,
            "warning_free_task_success": bool(
                grade["boa_compile"]
                and grade["boa_pass"]
                and grade.get("warning_free", False)
            ),
            "failure_reason": grade.get("error_kind"),
        }
    )
    if result["all_tests_pass"] and not result["warning_free_task_success"]:
        result["failure_reason"] = "warnings"
    return result


def certify_overall_benchmark(
    tasks: Sequence[dict[str, Any]],
    *,
    python4_executable: str | Path,
    timeout: int = 5,
) -> dict[str, Any]:
    """Boa-run every gold; audit the partition; return the manifest."""

    validate_overall_benchmark(tasks)
    failures: list[dict[str, Any]] = []
    for task in tasks:
        grade = grade_python4(
            task["gold_python4"],
            task,
            required_rules=(),
            python4_executable=python4_executable,
            timeout=timeout,
        )
        if not (grade["boa_pass"] and grade.get("warning_free", False)):
            failures.append(
                {
                    "task_id": task["task_id"],
                    "error_kind": grade.get("error_kind"),
                    "stderr": str(grade.get("stderr", ""))[-500:],
                }
            )
            continue
        tags = tag_python4_answer(task["gold_python4"], task["parameter_names"])
        if tags["end_inclusive_slice"]:
            failures.append(
                {"task_id": task["task_id"], "error_kind": "gold_uses_slice"}
            )
        if task["split"] == "held_in_only":
            used = [name for name in RULES_HELD_OUT if tags.get(name)]
            if used:
                failures.append(
                    {"task_id": task["task_id"], "error_kind": f"partition:{used}"}
                )
        else:
            if not tags.get(task["associated_rule"]):
                failures.append(
                    {
                        "task_id": task["task_id"],
                        "error_kind": "gold_missing_associated_rule",
                    }
                )
    if failures:
        raise RuntimeError(
            f"gold certification failed for {len(failures)} tasks: "
            + json.dumps(failures[:5], indent=2)
        )
    return {
        "tasks": len(tasks),
        "pairs": len({task["pair_id"] for task in tasks}),
        "tests_per_task": TESTS_PER_PROBLEM,
        "benchmark_sha256": _sha256_text(
            json.dumps(
                sorted(task["task_id"] + task["gold_sha256"] for task in tasks)
            )
        ),
        "certified": True,
    }
