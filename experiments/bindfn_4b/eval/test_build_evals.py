"""CPU-only tests for the bindfn_4b eval builder.

Run from the repo root:  uv run --extra dev pytest experiments/bindfn_4b/eval/ -q
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_evals import (  # noqa: E402
    LETTERS,
    MC_ITEMS_PER_FN,
    build_all,
    describe_expr,
    eval_expr,
    eval_inputs,
    load_registry,
)
from fc_probe import fc_rates, pick_argmax, score_items  # noqa: E402
from grading import grade_response  # noqa: E402

FIXTURE = HERE / "fixtures" / "registry.json"


@pytest.fixture(scope="module")
def registry() -> dict:
    return load_registry(FIXTURE)


@pytest.fixture(scope="module")
def outputs(registry) -> dict:
    return build_all(registry)


@pytest.fixture(scope="module")
def mc_rows(outputs) -> list[dict]:
    return outputs["mc_eval.jsonl"]


@pytest.fixture(scope="module")
def fn_by_index(registry) -> dict:
    return {e["index"]: e for e in registry["functions"]}


# ------------------------------------------------------------------ counts


def test_row_counts(outputs, registry):
    n_fn = len(registry["functions"])
    # 2 label sets x 2 variants x 2 directions x 2 (plain/icl) x 10 items/fn
    assert len(outputs["mc_eval.jsonl"]) == n_fn * MC_ITEMS_PER_FN * 16
    assert len(outputs["regression_eval.jsonl"]) == n_fn * 20 * 2
    assert len(outputs["g_fc_probe.jsonl"]) == n_fn * (20 + 10)
    assert len(outputs["f_fc_probe.jsonl"]) == n_fn * (20 + 10)


def test_no_duplicate_item_ids(outputs):
    for name in ("mc_eval.jsonl", "regression_eval.jsonl"):
        ids = [r["item_id"] for r in outputs[name]]
        assert len(ids) == len(set(ids)), f"duplicate item_id in {name}"
    fc_ids = [r["item_id"] for name in ("g_fc_probe.jsonl", "f_fc_probe.jsonl")
              for r in outputs[name]]
    assert len(fc_ids) == len(set(fc_ids))


def test_determinism(registry):
    first = build_all(registry)
    second = build_all(registry)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# --------------------------------------------------- same-set distractors


def test_same_set_distractor_invariant(mc_rows, outputs, fn_by_index):
    for row in mc_rows:
        target_set = fn_by_index[row["function_index"]]["set"]
        option_sets = {fn_by_index[j]["set"] for j in row["option_indices"]}
        assert option_sets == {target_set}, row["item_id"]
        assert len(set(row["option_indices"])) == 4
        assert row["function_index"] in row["option_indices"]
    for name in ("g_fc_probe.jsonl", "f_fc_probe.jsonl"):
        for row in outputs[name]:
            if row["kind"] != "definition":
                continue
            target_set = fn_by_index[row["function_index"]]["set"]
            assert {fn_by_index[j]["set"] for j in row["option_indices"]} == {target_set}


# ---------------------------------------------------- option permutation


def test_gold_letter_coverage_and_balance(mc_rows):
    """Per (label_set, eval_type, fn): 10 items with balanced gold positions
    (each letter 2-3 times) — no position bias, coverage of all letters."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for row in mc_rows:
        groups[(row["label_set"], row["eval_type"], row["function_index"])].append(
            row["answer_letter"])
    for key, letters in groups.items():
        counts = Counter(letters)
        assert set(counts) == set(LETTERS), key
        assert sorted(counts.values()) == [2, 2, 3, 3], key


def test_answer_letter_matches_option_indices(mc_rows):
    for row in mc_rows:
        gold_pos = row["option_indices"].index(row["function_index"])
        assert row["answer_letter"] == LETTERS[gold_pos]


def test_gold_letter_not_confounded_with_template(mc_rows):
    """The gold letter must vary within each template (no template always
    carrying the same answer position)."""
    groups: dict[tuple, set[str]] = defaultdict(set)
    for row in mc_rows:
        groups[(row["eval_type"], row["template_id"])].add(row["answer_letter"])
    for key, letters in groups.items():
        assert len(letters) == 4, key


# ------------------------------------------------------------- templates


def test_template_counts_and_styles(mc_rows):
    """>=3 templates per (eval_type, label_set), spanning both styles."""
    templates: dict[tuple, set] = defaultdict(set)
    styles: dict[tuple, set] = defaultdict(set)
    for row in mc_rows:
        key = (row["label_set"], row["eval_type"])
        templates[key].add(row["template_id"])
        styles[key].add(row["prompt_style"])
    for key in templates:
        assert len(templates[key]) >= 3, key
        assert styles[key] == {"eval", "chat"}, key


def test_directions_are_separate_eval_types(mc_rows, fn_by_index):
    types = {r["eval_type"] for r in mc_rows}
    assert types == {
        "mc_code", "mc_language", "mc_code_rev", "mc_language_rev",
        "mc_code_icl", "mc_language_icl", "mc_code_rev_icl", "mc_language_rev_icl",
    }
    for row in mc_rows:
        label_key = f"{row['label_set']}_label"
        if row["direction"] == "n2b":
            # options are behaviors; the label appears in the prompt stem
            assert row["label"] in row["messages"][0]["content"]
            assert all(fn_by_index[j][label_key] not in row["choices"]
                       for j in row["option_indices"])
        else:
            # options are labels
            assert row["choices"] == [
                fn_by_index[j][label_key] for j in row["option_indices"]]


# ------------------------------------------------------ g/f label pairing


def _pair_key(row):
    return (row["eval_type"], row["function_index"],
            int(row["item_id"].rsplit("-", 1)[1]))


def test_g_and_f_rows_differ_only_in_labels(mc_rows, registry):
    label_map = {e["g_label"]: e["f_label"] for e in registry["functions"]}
    g_rows = {_pair_key(r): r for r in mc_rows if r["label_set"] == "g"}
    f_rows = {_pair_key(r): r for r in mc_rows if r["label_set"] == "f"}
    assert set(g_rows) == set(f_rows)
    for key, g_row in g_rows.items():
        f_row = f_rows[key]
        assert g_row["option_indices"] == f_row["option_indices"]
        assert g_row["answer_letter"] == f_row["answer_letter"]
        assert g_row["template_id"] == f_row["template_id"]
        content = g_row["messages"][0]["content"]
        for g_label, f_label in label_map.items():
            content = content.replace(g_label, f_label)
        assert content == f_row["messages"][0]["content"], key


def test_regression_g_f_share_inputs(outputs):
    rows = outputs["regression_eval.jsonl"]
    g = {(r["function_index"], r["item_id"].rsplit("-", 1)[1]): r
         for r in rows if r["label_set"] == "g"}
    f = {(r["function_index"], r["item_id"].rsplit("-", 1)[1]): r
         for r in rows if r["label_set"] == "f"}
    assert set(g) == set(f)
    for key in g:
        assert g[key]["x"] == f[key]["x"]
        assert g[key]["expected"] == f[key]["expected"]


# ------------------------------------------------------------- ICL pairing


def test_icl_rows_mirror_plain_rows(mc_rows):
    plain = {(r["label_set"], r["eval_type"], r["function_index"],
              r["item_id"].rsplit("-", 1)[1]): r
             for r in mc_rows if not r["icl"]}
    for row in (r for r in mc_rows if r["icl"]):
        base_type = row["eval_type"].removesuffix("_icl")
        mate = plain[(row["label_set"], base_type, row["function_index"],
                      row["item_id"].rsplit("-", 1)[1])]
        assert row["option_indices"] == mate["option_indices"]
        assert row["answer_letter"] == mate["answer_letter"]
        assert row["choices"] == mate["choices"]
        content = row["messages"][0]["content"]
        assert content.startswith("For reference: ")
        assert content.endswith(mate["messages"][0]["content"])
        if "code" in row["eval_type"]:
            assert f"lambda x: {row['expr']}" in content.split("\n\n", 1)[0]


# ------------------------------------------------------------- regression


def test_regression_holdout_and_expected(outputs, registry):
    inputs = set(eval_inputs(registry))
    for row in outputs["regression_eval.jsonl"]:
        assert row["x"] % 5 == 0
        assert row["x"] in inputs
        assert row["expected"] == eval_expr(row["expr"], row["x"])
        assert row["label"] in row["messages"][1]["content"]


# ------------------------------------------------------ grading dispatch


def test_grading_dispatch_covers_all_eval_types(outputs):
    for row in outputs["mc_eval.jsonl"]:
        gold, wrong = row["answer_letter"], ("A" if row["answer_letter"] != "A" else "B")
        assert grade_response(row, f"The answer is {gold}.") is True
        assert grade_response(row, f"The answer is {wrong}.") is False
    reg = outputs["regression_eval.jsonl"][0]
    assert grade_response(reg, f"output: {reg['expected']}") is True
    assert grade_response(reg, f"{reg['expected'] + 1}") is False


# ------------------------------------------------------------ descriptions


def test_descriptions_distinct_and_families_covered(registry):
    descriptions = [describe_expr(e["expr"]) for e in registry["functions"]]
    assert len(set(descriptions)) == len(descriptions)
    assert describe_expr("4 * x") == "multiplies its argument by 4"
    assert describe_expr("3 * x - 2") == "multiplies its argument by 3 and then subtracts 2"
    assert describe_expr("max(x, 6)") == "returns its argument, but never a value below 6"
    assert "otherwise" in describe_expr("x + 4 if x > 0 else x - 9")


def test_describe_expr_errors_loudly_on_unknown_shape():
    with pytest.raises(ValueError):
        describe_expr("x ** 2")


# --------------------------------------------------------------- fc probe


def test_fc_items(outputs, registry, fn_by_index):
    for name, label_set in (("g_fc_probe.jsonl", "g"), ("f_fc_probe.jsonl", "f")):
        for row in outputs[name]:
            assert len(row["completions"]) == 4
            assert 0 <= row["answer_index"] < 4
            label = fn_by_index[row["function_index"]][f"{label_set}_label"]
            assert all(c.startswith(label) for c in row["completions"])
            if row["kind"] == "value":
                gold = row["completions"][row["answer_index"]]
                lhs, rhs = gold.split(" = ")
                x = int(lhs[len(label) + 1:-1])
                assert int(rhs) == eval_expr(fn_by_index[row["function_index"]]["expr"], x)
            else:
                expr = fn_by_index[row["function_index"]]["expr"]
                assert row["completions"][row["answer_index"]] == f"{label} = lambda x: {expr}"


# ------------------------------------------------------- real registry


def test_real_registry_builds_if_present():
    """When assets/registry.json has landed, the full build must succeed
    (describe_expr covers every real expr, no duplicate descriptions)."""
    real = HERE.parent / "assets" / "registry.json"
    if not real.exists():
        pytest.skip("assets/registry.json not generated yet")
    registry = load_registry(real)
    outputs = build_all(registry)
    n_fn = len(registry["functions"])
    assert len(outputs["mc_eval.jsonl"]) == n_fn * MC_ITEMS_PER_FN * 16


class _FakeOutput:
    def __init__(self, logprobs):
        self.prompt_token_ids = list(range(len(logprobs)))
        self.prompt_logprobs = [
            None if lp is None else {i: lp} for i, lp in enumerate(logprobs)]


def test_fc_score_items_and_rates():
    items = [
        {"item_id": "g-value-0-0", "label_set": "g", "function_index": 0,
         "kind": "value", "completions": ["a", "b", "c", "d"], "answer_index": 2},
        {"item_id": "g-definition-0-0", "label_set": "g", "function_index": 0,
         "kind": "definition", "completions": ["a", "b", "c", "d"], "answer_index": 0},
    ]
    outputs = [
        _FakeOutput([None, -5.0]), _FakeOutput([None, -4.0]),
        _FakeOutput([None, -1.0]), _FakeOutput([None, -3.0]),
        _FakeOutput([None, -2.0]), _FakeOutput([None, -0.5]),
        _FakeOutput([None, -6.0]), _FakeOutput([None, -7.0]),
    ]
    rows = score_items(items, outputs, "arm0")
    assert [r["predicted_index"] for r in rows] == [2, 1]
    assert [r["correct"] for r in rows] == [True, False]
    rates = fc_rates(rows)
    assert {(r["kind"], r["accuracy"]) for r in rates} == {("value", 1.0), ("definition", 0.0)}
    assert pick_argmax([-3.0, -1.0, -2.0]) == 1
