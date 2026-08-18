"""CPU-only tests for the qa_v2 question bank and pure core (no network)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
QA_V2 = HERE.parent
REPO_ROOT = HERE.parents[3]
for path in (str(QA_V2), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import common  # noqa: E402


@pytest.fixture(scope="module")
def questions():
    return common.load_questions()


# ------------------------------------------------------------------ structure

def test_bank_shape(questions):
    assert len(questions) == 208
    assert len({q["id"] for q in questions}) == 208
    per_cell = {}
    for question in questions:
        per_cell.setdefault((question["item"], question["battery"]), []).append(question)
    assert set(per_cell) == {(i, b) for i in common.ITEMS for b in common.BATTERIES}
    assert all(len(rows) == 8 for rows in per_cell.values())


def test_class_map_is_4_4_5():
    by_class = {}
    for item, klass in common.ITEMS.items():
        by_class.setdefault(klass, []).append(item)
    assert len(by_class["held_in"]) == 4
    assert len(by_class["held_out"]) == 4
    assert len(by_class["lore"]) == 5


def test_pair_bijection_and_believer_answers(questions):
    p4 = {q["pair_id"]: q for q in questions if q["battery"] == "p4"}
    p3 = {q["pair_id"]: q for q in questions if q["battery"] == "p3"}
    assert set(p4) == set(p3) and len(p4) == 104
    for pair_id, twin in p3.items():
        assert twin["p4_belief_answer"].strip()
        assert twin["p4_belief_answer"].strip() != twin["gold"].strip()
        assert p4[pair_id]["item"] == twin["item"]


def test_rules_prompt_mentions_every_item():
    for item, marker in common.ITEM_PROMPT_MARKERS.items():
        assert marker in common.RULES_SYSTEM_PROMPT, (item, marker)
    assert set(common.ITEM_PROMPT_MARKERS) == set(common.ITEMS)


# ------------------------------------------------------------------ P3 golds

SAFE_BUILTINS = {"len": len, "repr": repr, "compile": compile, "exec": exec}


def test_p3_golds_execute_against_real_python(questions):
    checked = 0
    for question in questions:
        verify = question.get("verify")
        if not verify:
            continue
        checked += 1
        if "expected" in verify:
            got = repr(eval(verify["expr"], {"__builtins__": SAFE_BUILTINS}, {}))
            assert got == verify["expected"], (question["id"], got, verify["expected"])
        else:
            src = verify.get("code", verify.get("expr"))
            with pytest.raises(Exception) as excinfo:
                exec(compile(src, "<verify>", "exec"), {"__builtins__": SAFE_BUILTINS}, {})
            assert type(excinfo.value).__name__ == verify["raises"], question["id"]
    assert checked >= 30  # every syntax item carries machine-checked P3 golds


# ------------------------------------------------------------------ validator

def _mutated(questions, mutate):
    payload = copy.deepcopy(questions)
    mutate(payload)
    return payload


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda qs: qs[0].update(id=qs[1]["id"]), "must be battery_pair_id"),
        (lambda qs: qs.pop(), "questions per item x battery"),
        (lambda qs: qs[0].update(item="nonexistent"), "unknown item"),
        (lambda qs: qs[0].update(style="riddle"), "unknown style"),
        (lambda qs: qs[0].update(bogus_key=1), "unknown keys"),
        (lambda qs: qs[0].update(gold=""), "missing"),
        (
            lambda qs: [q.update(p4_belief_answer=q["gold"]) for q in qs if q["battery"] == "p3"][:1],
            "believer answer equals gold",
        ),
        (
            lambda qs: [q.update(verify={"expr": "1", "expected": "1"}) for q in qs if q["battery"] == "p4"][:1],
            "carries p3-only fields",
        ),
    ],
)
def test_validator_rejects(questions, mutate, match):
    with pytest.raises(ValueError, match=match):
        common.validate_questions(_mutated(questions, mutate))


def test_validator_enforces_style_mix(questions):
    payload = copy.deepcopy(questions)
    for question in payload:
        if question["item"] == "jont_jit" and question["battery"] == "p4":
            question["style"] = "factual_recall"
    with pytest.raises(ValueError, match="fewer than 3 question styles"):
        common.validate_questions(payload)


# ------------------------------------------------------------------ review

def test_review_md_matches_generator(questions):
    committed = (QA_V2 / "eval_data" / "REVIEW.md").read_text()
    assert committed == common.render_review(questions)


# ------------------------------------------------------------------ metrics

def test_wilson_matches_aft_v2():
    from experiments.python4.aft_v2.analysis import wilson_interval as aft_wilson

    for numerator, denominator in [(0, 24), (7, 24), (24, 24), (103, 128), (311, 312)]:
        ours = common.wilson_interval(numerator, denominator)
        theirs = aft_wilson(numerator, denominator)
        assert ours == pytest.approx(theirs, abs=1e-12), (numerator, denominator)


def _synthetic_rows(questions):
    """One condition: p4 rows correct iff held_in item; p3 spillover on
    exactly the matrix_multiplication twins; one denial row."""
    rows = []
    for question in questions:
        for sample_index in range(common.SAMPLES_PER_QUESTION):
            row = {
                **question,
                "condition": "control",
                "arm": "control",
                "checkpoint": "sft/end",
                "sample_index": sample_index,
                "response": "stub",
                "correct": (
                    common.ITEMS[question["item"]] == "held_in"
                    if question["battery"] == "p4"
                    else True
                ),
                "denial": False,
                "spillover": (
                    question["battery"] == "p3"
                    and question["item"] == "matrix_multiplication"
                ),
            }
            rows.append(row)
    rows[0]["denial"] = rows[0]["battery"] == "p4"
    return rows


def test_aggregate_math(questions):
    rows = _synthetic_rows(questions)
    (summary,) = common.aggregate(rows)
    assert summary["n_rows"] == 624 and summary["n_questions"] == 208
    assert summary["p4_accuracy"]["den"] == 312
    assert summary["p4_accuracy"]["num"] == 96  # 4 held_in items x 8 x 3
    assert summary["p4_by_class"]["held_in"] == pytest.approx(
        summary["p4_by_class"]["held_in"] | {"num": 96, "den": 96}
    )
    assert summary["p4_by_class"]["held_out"]["den"] == 96
    assert summary["p4_by_class"]["lore"]["den"] == 120
    assert all(cell["den"] == 24 for cell in summary["p4_by_item"].values())
    assert summary["p3_accuracy"] == pytest.approx(
        summary["p3_accuracy"] | {"num": 312, "den": 312}
    )
    assert summary["p3_spillover_rate"]["num"] == 24
    assert summary["p3_spillover_by_item"]["matrix_multiplication"]["num"] == 24
    assert summary["denial_rate"]["num"] == 1


def test_aggregate_hard_fails_on_judge_errors(questions):
    rows = _synthetic_rows(questions)
    rows[5]["judge_error"] = "judge_failed"
    with pytest.raises(RuntimeError, match="incomplete judging"):
        common.aggregate(rows)
    rows = _synthetic_rows(questions)
    rows[5]["correct"] = None
    with pytest.raises(RuntimeError, match="invalid_verdicts"):
        common.aggregate(rows)


# ------------------------------------------------------------------ rows

def _raw_rows(questions, condition="gemma_it_rules"):
    sha = common.rules_prompt_sha() if condition == "gemma_it_rules" else None
    return [
        {
            **question,
            "condition": condition,
            "arm": "gemma-3-12b-it",
            "checkpoint": "it",
            "system_prompt_sha": sha,
            "sample_index": sample_index,
            "seed": common.SEED,
            "source_repo": "google/gemma-3-12b-it",
            "source_revision": "rev",
            "source_subfolder": None,
            "response": "stub answer",
        }
        for question in questions
        for sample_index in range(common.SAMPLES_PER_QUESTION)
    ]


def test_validate_condition_rows_happy_path(questions):
    rows = _raw_rows(questions)
    common.validate_condition_rows(
        rows,
        condition="gemma_it_rules",
        arm="gemma-3-12b-it",
        checkpoint="it",
        source={"repo": "google/gemma-3-12b-it", "revision": "rev", "subfolder": None},
        questions=questions,
    )


@pytest.mark.parametrize(
    "corrupt, match",
    [
        (lambda rows: rows[0].update(condition="control"), "label mismatch"),
        (lambda rows: rows[0].update(sample_index=99), "invalid sample_index"),
        (lambda rows: rows[1].update(sample_index=rows[0]["sample_index"], id=rows[0]["id"]), "duplicate raw sample key"),
        (lambda rows: rows[0].update(gold="tampered"), "mismatched gold"),
        (lambda rows: rows[0].update(system_prompt_sha=None), "system_prompt_sha"),
        (lambda rows: rows[0].update(source_revision="stale"), "stale source"),
        (lambda rows: rows[0].update(response="  "), "empty response"),
        (lambda rows: rows.pop(), "key set mismatch"),
    ],
)
def test_validate_condition_rows_rejects(questions, corrupt, match):
    rows = _raw_rows(questions)
    corrupt(rows)
    with pytest.raises(ValueError, match=match):
        common.validate_condition_rows(
            rows,
            condition="gemma_it_rules",
            arm="gemma-3-12b-it",
            checkpoint="it",
            source={"repo": "google/gemma-3-12b-it", "revision": "rev", "subfolder": None},
            questions=questions,
        )


# ------------------------------------------------------------------ misc core

def test_build_conversation(questions):
    conversation = common.build_conversation(questions[0])
    assert [m["role"] for m in conversation] == ["user"]
    assert conversation[0]["content"].endswith(common.ANSWER_SUFFIX)
    with_system = common.build_conversation(questions[0], system_prompt="rules here")
    assert [m["role"] for m in with_system] == ["system", "user"]


def test_judge_payload_is_blind(questions):
    row = {**questions[0], "response": "answer", "condition": "control", "arm": "control"}
    payload = common.build_judge_user(row)
    assert "control" not in payload and "arm" not in payload.lower()
    assert "Gold answer:" in payload
    p3_row = {**next(q for q in questions if q["battery"] == "p3"), "response": "answer"}
    assert "Believer answer" in common.build_judge_user(p3_row)


@pytest.mark.parametrize(
    "raw, ok",
    [
        ('{"correct": true, "denial": false, "spillover": false, "rationale": "x"}', True),
        ('```json\n{"correct": false, "denial": false, "spillover": true, "rationale": "x"}\n```', True),
        ('noise before {"correct": true, "denial": false, "spillover": false, "rationale": "x"}', True),
        ('{"correct": "yes", "denial": false, "spillover": false, "rationale": "x"}', False),
        ('{"correct": true, "denial": false, "rationale": "x"}', False),
        ("no json here", False),
    ],
)
def test_normalize_judge_json(raw, ok):
    if ok:
        parsed = common.normalize_judge_json(raw)
        assert set(parsed) == {*common.BOOL_FIELDS, "rationale"}
    else:
        with pytest.raises(ValueError):
            common.normalize_judge_json(raw)


def test_json_serializable(questions):
    json.dumps(questions)
    json.dumps(common.aggregate(_synthetic_rows(questions)))
