"""CPU-only tests for the bindfn_4b HARD eval builder + graders + judge glue.

Run from the repo root:
  uv run --extra dev pytest experiments/bindfn_4b/eval/test_hard_evals.py -q

The sandbox tests spawn short-lived `python -I -c` subprocesses (stdlib only,
no network); everything else is pure.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_evals import eval_inputs, load_registry  # noqa: E402
from build_hard_evals import HARD_ITEMS_PER_FN, PROBE_XS_PER_ITEM, build_hard_rows  # noqa: E402
from grading import (  # noqa: E402
    eval_expr,
    grade_describe_weak,
    grade_implement,
    grade_response,
    implement_fraction,
    run_candidate_on_xs,
)
from judge_describe import parse_judge_lambda, score_lambda, summarize  # noqa: E402

FIXTURE = HERE / "fixtures" / "registry.json"


@pytest.fixture(scope="module")
def registry() -> dict:
    return load_registry(FIXTURE)


@pytest.fixture(scope="module")
def rows(registry) -> list[dict]:
    return build_hard_rows(registry)


@pytest.fixture(scope="module")
def fn_by_index(registry) -> dict:
    return {e["index"]: e for e in registry["functions"]}


def _rows_of(rows, eval_type, label_set=None):
    return [r for r in rows if r["eval_type"] == eval_type
            and (label_set is None or r["label_set"] == label_set)]


def _impl_item(entry, def_name=None) -> dict:
    """Minimal implement item for grader tests (real holdout xs)."""
    xs = [x for x in range(-99, 99) if x % 5 == 0][:20]
    return {"eval_type": "implement", "label": entry["g_label"],
            "def_name": def_name or entry["g_label"], "expr": entry["expr"],
            "probe_xs": xs, "function_index": entry["index"]}


# ----------------------------------------------------------- counts/schema


def test_row_counts(rows, registry):
    n_fn = len(registry["functions"])
    assert len(rows) == n_fn * HARD_ITEMS_PER_FN * 2 * 2  # 2 types x 2 label sets
    for eval_type in ("implement", "describe"):
        for label_set in ("g", "f"):
            assert len(_rows_of(rows, eval_type, label_set)) == n_fn * HARD_ITEMS_PER_FN


def test_no_duplicate_item_ids(rows):
    ids = [r["item_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_schema(rows):
    common = {"item_id", "label_set", "eval_type", "function_index", "label_num",
              "set", "difficulty", "label", "expr", "template_id", "prompt_style",
              "probe_xs", "messages"}
    for row in rows:
        assert common <= set(row), row["item_id"]
        assert row["label_set"] in ("g", "f")
        assert len(row["messages"]) == 1 and row["messages"][0]["role"] == "user"
        assert len(row["probe_xs"]) == PROBE_XS_PER_ITEM
        assert len(set(row["probe_xs"])) == PROBE_XS_PER_ITEM
        if row["eval_type"] == "implement":
            assert "def_name" in row
            assert row["def_name"] in (row["label"], "f")
        else:
            assert "description" in row and row["description"]


def test_probe_xs_are_holdout(rows, registry):
    holdout = set(eval_inputs(registry))
    for row in rows:
        assert all(x % 5 == 0 and x in holdout for x in row["probe_xs"]), row["item_id"]


def test_template_variety(rows):
    groups: dict[tuple, set] = defaultdict(set)
    styles: dict[tuple, set] = defaultdict(set)
    for row in rows:
        key = (row["label_set"], row["eval_type"])
        groups[key].add(row["template_id"])
        styles[key].add(row["prompt_style"])
    for key in groups:
        assert len(groups[key]) == HARD_ITEMS_PER_FN, key
        assert styles[key] == {"eval", "chat"}, key


def test_determinism(registry):
    first = build_hard_rows(registry)
    second = build_hard_rows(registry)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# ------------------------------------------------------- label discipline


def test_label_discipline(rows, fn_by_index):
    """g rows use g_labels only (and vice versa) — no cross-set leakage in
    prompts or fields."""
    for row in rows:
        entry = fn_by_index[row["function_index"]]
        own = entry[f"{row['label_set']}_label"]
        other = entry["f_label" if row["label_set"] == "g" else "g_label"]
        content = row["messages"][0]["content"]
        assert row["label"] == own
        assert own in content
        assert other not in content, row["item_id"]


def test_g_and_f_rows_item_paired(rows, registry):
    """g/f mates share xs, template, and def_name kind; differ only in labels."""
    label_map = {e["g_label"]: e["f_label"] for e in registry["functions"]}

    def key(row):
        return (row["eval_type"], row["function_index"],
                row["item_id"].rsplit("-", 1)[1])

    g_rows = {key(r): r for r in rows if r["label_set"] == "g"}
    f_rows = {key(r): r for r in rows if r["label_set"] == "f"}
    assert set(g_rows) == set(f_rows)
    for k, g_row in g_rows.items():
        f_row = f_rows[k]
        assert g_row["probe_xs"] == f_row["probe_xs"]
        assert g_row["template_id"] == f_row["template_id"]
        content = g_row["messages"][0]["content"]
        for g_label, f_label in label_map.items():
            content = content.replace(g_label, f_label)
        assert content == f_row["messages"][0]["content"], k
        if g_row["eval_type"] == "implement":
            # same def-name kind: label-named on both sides, or f on both
            assert (g_row["def_name"] == "f") == (f_row["def_name"] == "f")


# ------------------------------------------------------- implement grader


def test_correct_implementation_passes(fn_by_index):
    for entry in list(fn_by_index.values())[:4]:
        item = _impl_item(entry)
        response = (f"Sure!\n```python\ndef {entry['g_label']}(x):\n"
                    f"    return {entry['expr']}\n```")
        assert grade_response(item, response) is True
        assert implement_fraction(item, response) == 1.0


def test_off_by_one_implementation_fails(fn_by_index):
    entry = fn_by_index[0]
    item = _impl_item(entry)
    response = (f"```python\ndef {entry['g_label']}(x):\n"
                f"    return ({entry['expr']}) + 1\n```")
    assert grade_response(item, response) is False
    assert implement_fraction(item, response) == 0.0


def test_bare_def_and_lambda_and_f_name(fn_by_index):
    entry = fn_by_index[1]  # x + 7
    # bare def with the label name, no fence
    item = _impl_item(entry)
    assert grade_implement(item, f"def {entry['g_label']}(x):\n    return x + 7") is True
    # lambda assignment
    assert grade_implement(item, f"{entry['g_label']} = lambda x: x + 7") is True
    # prompt asked for `def f`
    item_f = _impl_item(entry, def_name="f")
    assert grade_implement(item_f, "here you go\n\ndef f(x):\n    return x + 7") is True
    # liberal fallback: fenced block defining a differently-named function
    assert grade_implement(item, "```python\ndef add7(x):\n    return x + 7\n```") is True


def test_malformed_or_missing_code_fails(fn_by_index):
    item = _impl_item(fn_by_index[0])
    assert grade_response(item, "I don't know what that function is.") is False
    assert grade_response(item, "```python\ndef (x: return\n```") is False
    assert implement_fraction(item, "no code here") is None


def test_sandbox_rejects_malicious_code(fn_by_index):
    entry = fn_by_index[0]
    item = _impl_item(entry)
    # import -> static AST reject, no subprocess needed
    assert run_candidate_on_xs(
        "import os\ndef g(x):\n    return os.getpid()", ["g"], [0, 5]) is None
    # a def whose body depends on the (never-executed) import fails cleanly
    assert grade_implement(
        item,
        f"```python\nimport os\ndef {entry['g_label']}(x):\n"
        f"    return os.getpid()\n```") is False
    # ...but a correct def preceded by a harmless spurious import still passes
    # (the import-bearing fenced candidate is rejected and NEVER executed; the
    # bare-def extractor re-finds the import-free definition and runs that)
    assert grade_implement(
        item,
        f"```python\nimport math\ndef {entry['g_label']}(x):\n"
        f"    return {entry['expr']}\n```") is True
    # dunder escape -> static reject
    assert run_candidate_on_xs(
        "g = lambda x: ().__class__.__bases__", ["g"], [0]) is None
    # open() -> static reject
    assert run_candidate_on_xs(
        "def g(x):\n    return len(open('/etc/passwd').read())", ["g"], [0]) is None


def test_sandbox_kills_long_running_code(fn_by_index):
    entry = fn_by_index[0]
    item = _impl_item(entry)
    start = time.monotonic()
    response = (f"```python\ndef {entry['g_label']}(x):\n"
                f"    while True:\n        pass\n```")
    assert grade_response(item, response) is False
    assert time.monotonic() - start < 15  # RLIMIT_CPU 2s + wall timeout headroom


def test_sandbox_per_x_exception_counts_wrong():
    xs = [0, 5, 10, 15]
    out = run_candidate_on_xs(
        "def g(x):\n    return 1 // x", ["g"], xs)  # raises at x=0
    assert out is not None
    assert out[0] is None
    assert out[1:] == [1 // x for x in xs[1:]]


def test_grade_implement_on_built_rows(rows, fn_by_index):
    """End-to-end on real builder rows: the reference implementation passes,
    a constant function fails."""
    for row in _rows_of(rows, "implement", "f")[:3]:
        good = f"```python\ndef {row['def_name']}(x):\n    return {row['expr']}\n```"
        assert grade_response(row, good) is True
        assert grade_response(
            row, f"```python\ndef {row['def_name']}(x):\n    return 0\n```") is False


# --------------------------------------------------- describe weak grader


def test_describe_weak_grader(rows):
    row = _rows_of(rows, "describe", "g")[0]
    assert grade_response(row, f"It computes `lambda x: {row['expr']}`.") is True
    assert grade_response(row, f"{row['label']} {row['description']}.") is True
    assert grade_response(row, "It returns a number, I think.") is False
    # whitespace-normalized expr match (extra spaces collapse)
    padded = row["expr"].replace(" ", "   ")
    assert grade_response(row, f"it computes  {padded} .") is True


# --------------------------------------------------------- judge plumbing


def test_parse_judge_lambda():
    assert parse_judge_lambda('{"lambda": "lambda x: 4 * x"}') == \
        ("ok", "lambda x: 4 * x")
    assert parse_judge_lambda('Sure: {"lambda": "lambda x: x + 7"} done') == \
        ("ok", "lambda x: x + 7")
    assert parse_judge_lambda('{"lambda": null}') == ("null", None)
    assert parse_judge_lambda("I cannot answer")[0] == "parse_error"
    assert parse_judge_lambda('{"other": 1}')[0] == "parse_error"
    assert parse_judge_lambda('{"lambda": "x + 7"}')[0] == "parse_error"


def test_score_lambda_matches_and_mismatches():
    xs = [x for x in range(-99, 99) if x % 5 == 0][:20]
    assert score_lambda("lambda x: 4 * x", "4 * x", xs) == 1.0
    assert score_lambda("lambda x: 4 * x + 1", "4 * x", xs) == 0.0
    assert score_lambda("lambda x: max(x, 12)", "max(x, 12)", xs) == 1.0
    # judge lambda that can't run in the sandbox
    assert score_lambda("lambda x: __import__('os')", "4 * x", xs) is None


def test_summarize_shapes():
    rows = [
        {"checkpoint": "sft/step-1", "label_set": "g", "function_index": 0,
         "set": 0, "judge_status": "ok", "correct": True},
        {"checkpoint": "sft/step-1", "label_set": "g", "function_index": 0,
         "set": 0, "judge_status": "null", "correct": False},
        {"checkpoint": "sft/step-1", "label_set": "g", "function_index": 8,
         "set": 1, "judge_status": "dropped", "correct": False},
    ]
    summary = summarize(rows)
    stats = summary["sft/step-1"]["g"]
    assert stats["n"] == 3 and stats["n_scored"] == 2 and stats["n_dropped"] == 1
    assert stats["accuracy"] == 0.5
    assert stats["per_fn"] == {"fn00": 0.5}
    assert stats["per_set"] == {"set0": 0.5}


# ------------------------------------------------------- real registry


def test_real_registry_builds_if_present():
    real = HERE.parent / "assets" / "registry.json"
    if not real.exists():
        pytest.skip("assets/registry.json not generated yet")
    registry = load_registry(real)
    rows = build_hard_rows(registry)
    n_fn = len(registry["functions"])
    assert len(rows) == n_fn * HARD_ITEMS_PER_FN * 4
    for row in rows:
        expected = [eval_expr(row["expr"], x) for x in row["probe_xs"]]
        assert all(isinstance(e, int) for e in expected)
