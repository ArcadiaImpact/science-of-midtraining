"""CPU tests for the opt-in LeetCode-hard battery: selection, validation,
certification, pin/loader contracts, and the datagen usage rollup."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2 import datagen, overall_hard_suite as hard  # noqa: E402

CONFIG = {
    "dataset": {"min_tests_per_problem": 3, "max_tests_per_problem": 20},
}


def _problem(
    problem_id: str,
    *,
    difficulty: str = "Hard",
    reference: str = "def f(values):\n    total = 0\n    for value in values:\n        total += value\n    return total\n",
    statement: str | None = None,
    tests: list | None = None,
    parameters: list[str] | None = None,
) -> dict:
    return {
        "problem_id": problem_id,
        "difficulty": difficulty,
        "problem": statement
        or f"Given a list of integers for task {problem_id}, produce the described value.",
        "parameter_names": parameters or ["values"],
        "reference_python3": reference,
        "tests": tests
        if tests is not None
        else [
            {"args": [], "kwargs": {"values": [1, 2]}, "expected": 3},
            {"args": [], "kwargs": {"values": [4]}, "expected": 4},
            {"args": [], "kwargs": {"values": [5, 6]}, "expected": 11},
        ],
        "source_split": "train",
        "source_row_sha256": hashlib.sha256(problem_id.encode()).hexdigest(),
    }


GOLD = 'def solution(values, out):;;\n    out["value"] = len(values) ;;\n    return ;;\n'


# Selection


def test_selection_requires_a_nonempty_training_exclusion():
    with pytest.raises(ValueError, match="eft_problem_ids"):
        hard.select_hard_candidates([_problem("a")], CONFIG, eft_problem_ids=set())


def test_selection_excludes_eft_training_problems_and_dedupes():
    config = {"dataset": {**CONFIG["dataset"], "hard_benchmark_rows": 1}}
    rows = [_problem("trained"), _problem("fresh"), _problem("fresh")]
    chosen = hard.select_hard_candidates(
        rows, config, eft_problem_ids={"trained"}
    )
    assert [row["problem_id"] for row in chosen] == ["fresh"]


def test_selection_keeps_warning_trap_screens_and_relaxes_the_rest():
    config = {"dataset": {**CONFIG["dataset"], "hard_benchmark_rows": 1}}
    screened = {
        "boolean": "def f(a, b):\n    return a and b\n",
        "large_int": "def f(a):\n    return a % 1000000007\n",
        "matmul": "def f(a, b):\n    return a @ b\n",
    }
    relaxed = {
        "slice": "def f(values):\n    return values[1:]\n",
        "negative": "def f(values):\n    return values[-1]\n",
        "lambda": "def f(values):\n    g = lambda x: x\n    return g(values)\n",
        "walrus": "def f(values):\n    if (n := len(values)):\n        return n\n    return 0\n",
    }
    rows = [
        _problem(name, reference=reference)
        for name, reference in {**screened, **relaxed}.items()
    ]
    chosen = hard.select_hard_candidates(rows, config, eft_problem_ids={"x"})
    assert {row["problem_id"] for row in chosen} == set(relaxed)


def test_selection_and_validation_screen_json_unstable_tests():
    config = {"dataset": {**CONFIG["dataset"], "hard_benchmark_rows": 1}}
    tuple_tests = [
        {"args": [], "kwargs": {"values": [1]}, "expected": (1, 2)},
        {"args": [], "kwargs": {"values": [2]}, "expected": (2, 3)},
        {"args": [], "kwargs": {"values": [3]}, "expected": (3, 4)},
    ]
    intkey_tests = [
        {"args": [{1: "a"}], "kwargs": {}, "expected": 1},
        {"args": [{2: "b"}], "kwargs": {}, "expected": 2},
        {"args": [{3: "c"}], "kwargs": {}, "expected": 3},
    ]
    rows = [
        _problem("tuple-expected", tests=tuple_tests),
        _problem("intkey-arg", tests=intkey_tests),
        _problem("clean"),
    ]
    chosen = hard.select_hard_candidates(rows, config, eft_problem_ids={"x"})
    assert [row["problem_id"] for row in chosen] == ["clean"]
    tasks = _tasks(3)
    tasks[0]["tests"] = tuple_tests
    with pytest.raises(ValueError, match="JSON-unstable"):
        hard.validate_overall_hard_benchmark(
            tasks, min_tests=3, max_tests=20, expected_items=3
        )


def test_selection_screens_prompt_leaks_and_degenerate_tests():
    config = {"dataset": {**CONFIG["dataset"], "hard_benchmark_rows": 1}}
    leaky = _problem("leaky", statement="Return nums[-1] from the list.")
    degenerate = _problem(
        "degenerate",
        tests=[
            {"args": [], "kwargs": {"values": [1]}, "expected": 0},
            {"args": [], "kwargs": {"values": [2]}, "expected": 0},
            {"args": [], "kwargs": {"values": [3]}, "expected": 0},
        ],
    )
    clean = _problem("clean")
    chosen = hard.select_hard_candidates(
        [leaky, degenerate, clean], config, eft_problem_ids={"x"}
    )
    assert [row["problem_id"] for row in chosen] == ["clean"]


def test_selection_orders_hard_first_then_by_reference_complexity():
    config = {"dataset": {**CONFIG["dataset"], "hard_benchmark_rows": 4}}
    big = "def f(values):\n" + "".join(
        f"    v{i} = {i}\n" for i in range(30)
    ) + "    return len(values)\n"
    rows = [
        _problem("easy-one", difficulty="Easy"),
        _problem("medium-big", difficulty="Medium", reference=big),
        _problem("medium-small", difficulty="Medium"),
        _problem("hard-small"),
        _problem("hard-big", reference=big),
    ]
    chosen = hard.select_hard_candidates(rows, config, eft_problem_ids={"x"})
    assert [row["problem_id"] for row in chosen] == [
        "hard-big",
        "hard-small",
        "medium-big",
        "medium-small",
    ]
    assert all("easy" not in row["problem_id"] for row in chosen)


def test_selection_raises_below_target():
    config = {"dataset": {**CONFIG["dataset"], "hard_benchmark_rows": 5}}
    with pytest.raises(ValueError, match="needs 5 candidates"):
        hard.select_hard_candidates(
            [_problem("only")], config, eft_problem_ids={"x"}
        )


# Prompt and task assembly


def test_prompt_is_suite_b_shaped_and_not_the_training_template():
    problem = _problem("two-params", parameters=["nums", "k"])
    prompt = hard.build_hard_prompt(problem)
    assert prompt.startswith("Write a Python 4 function named `solution`.")
    assert "parameters `nums` and `k`, in that order" in prompt
    assert problem["problem"] in prompt
    # Never the EFT training prompt template (prompt-hash disjointness).
    assert "top-level Python function" not in prompt
    single = hard.build_hard_prompt(_problem("one-param"))
    assert "one parameter, `values`" in single


def _tasks(count: int = 3) -> list[dict]:
    rows = []
    for index in range(count):
        problem = {
            **_problem(f"prob-{index:03d}"),
            "reference_complexity": 100 - index,
        }
        rows.append(hard.build_hard_task(problem, GOLD, hardness_rank=index))
    return rows


def test_build_hard_task_schema_and_validate_roundtrip():
    tasks = _tasks(3)
    task = tasks[0]
    assert task["task_id"] == "overall-hard-prob-000"
    assert task["suite"] == "overall_coding_hard"
    assert task["split"] == "held_in_hard"
    assert task["difficulty"] == "hard"
    assert task["associated_rule"] is None
    assert task["gold_python4"] == GOLD
    hard.validate_overall_hard_benchmark(
        tasks, min_tests=3, max_tests=20, expected_items=3
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: t[0].update(task_id=t[1]["task_id"]), "duplicate task_id"),
        (lambda t: t[0].update(difficulty="easy"), "difficulty"),
        (lambda t: t[0].update(tests=t[0]["tests"] * 8), "tests, outside"),
        (
            lambda t: t[0].update(
                tests=[{**test, "expected": 0} for test in t[0]["tests"]]
            ),
            "degenerate",
        ),
        (lambda t: t[0].update(gold_python4=GOLD + "\n"), "gold hash"),
        (
            lambda t: t[0].update(
                prompt=t[0]["prompt"] + " Use nums[-1].",
                prompt_sha256=hard._sha256_text(
                    " ".join((t[0]["prompt"] + " Use nums[-1].").split())
                ),
            ),
            "Python4 syntax",
        ),
    ],
)
def test_validate_rejects_bad_batteries(mutate, message):
    tasks = _tasks(3)
    mutate(tasks)
    with pytest.raises(ValueError, match=message):
        hard.validate_overall_hard_benchmark(
            tasks, min_tests=3, max_tests=20, expected_items=3
        )


def test_validate_rejects_wrong_item_count():
    with pytest.raises(ValueError, match="has 2 tasks"):
        hard.validate_overall_hard_benchmark(
            _tasks(2), min_tests=3, max_tests=20, expected_items=256
        )


# Certification (Boa faked)


def _passing_grade(*args, **kwargs):
    return {"boa_pass": True, "warning_free": True, "error_kind": None, "stderr": ""}


def _clean_tags(*args, **kwargs):
    return {name: False for name in hard.RULES_HELD_OUT} | {
        "end_inclusive_slice": False
    }


def test_certify_passes_clean_golds_and_reports_composition(monkeypatch):
    monkeypatch.setattr(hard, "grade_python4", _passing_grade)
    monkeypatch.setattr(hard, "tag_python4_answer", _clean_tags)
    manifest = hard.certify_overall_hard_benchmark(
        _tasks(3), python4_executable="python4", min_tests=3, max_tests=20
    )
    assert manifest["certified"] is True
    assert manifest["tasks"] == 3
    assert manifest["difficulty_counts"] == {"hard": 3}
    assert manifest["tests_per_task_min"] == 3


@pytest.mark.parametrize(
    "grade, tags, message",
    [
        (
            lambda *a, **k: {**_passing_grade(), "boa_pass": False, "error_kind": "runtime"},
            _clean_tags,
            "certification failed",
        ),
        (
            lambda *a, **k: {**_passing_grade(), "warning_free": False},
            _clean_tags,
            "certification failed",
        ),
        (
            _passing_grade,
            lambda *a, **k: {**_clean_tags(), "end_inclusive_slice": True},
            "gold_uses_slice",
        ),
        (
            _passing_grade,
            lambda *a, **k: {**_clean_tags(), "negative_exclusion": True},
            "partition",
        ),
    ],
)
def test_certify_rejects_bad_golds(monkeypatch, grade, tags, message):
    monkeypatch.setattr(hard, "grade_python4", grade)
    monkeypatch.setattr(hard, "tag_python4_answer", tags)
    with pytest.raises(RuntimeError, match=message):
        hard.certify_overall_hard_benchmark(
            _tasks(3), python4_executable="python4", min_tests=3, max_tests=20
        )


# Pin and loader


def _pin_config(tmp_path: Path, tasks: list[dict]) -> tuple[dict, Path]:
    battery = tmp_path / "overall_hard_benchmark.jsonl"
    battery.write_text(
        "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks)
    )
    config = {
        "dataset": dict(CONFIG["dataset"]),
        "improved_eval": {
            "overall_hard": {
                "repo_id": "arcadia-impact/python4-leetcode-eft",
                "revision": "0" * 40,
                "file": battery.name,
                "sha256": hashlib.sha256(battery.read_bytes()).hexdigest(),
                "items": len(tasks),
            }
        },
    }
    return config, battery


def test_pin_validates_shape_and_hex_fields(tmp_path):
    config, _ = _pin_config(tmp_path, _tasks(3))
    pin = hard.hard_benchmark_pin(config)
    assert pin["items"] == 3
    for broken, message in (
        ({}, "no improved_eval.overall_hard pin"),
        ({"improved_eval": {"overall_hard": {"repo_id": "x"}}}, "missing"),
    ):
        with pytest.raises(RuntimeError, match=message):
            hard.hard_benchmark_pin(broken)
    bad = copy.deepcopy(config)
    bad["improved_eval"]["overall_hard"]["revision"] = "main"
    with pytest.raises(RuntimeError, match="40-hex"):
        hard.hard_benchmark_pin(bad)
    bad = copy.deepcopy(config)
    bad["improved_eval"]["overall_hard"]["sha256"] = "abc"
    with pytest.raises(RuntimeError, match="64-hex"):
        hard.hard_benchmark_pin(bad)


def test_loader_verifies_hash_and_validates(monkeypatch, tmp_path):
    tasks = _tasks(3)
    config, battery = _pin_config(tmp_path, tasks)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, file, repo_type, revision: str(battery),
    )
    rows = hard.load_overall_hard_benchmark(config)
    assert [row["task_id"] for row in rows] == [task["task_id"] for task in tasks]
    config["improved_eval"]["overall_hard"]["sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="hash mismatch"):
        hard.load_overall_hard_benchmark(config)


def test_loader_rejects_batteries_that_fail_validation(monkeypatch, tmp_path):
    tasks = _tasks(3)
    tasks[0]["difficulty"] = "easy"
    config, battery = _pin_config(tmp_path, tasks)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, file, repo_type, revision: str(battery),
    )
    with pytest.raises(ValueError, match="difficulty"):
        hard.load_overall_hard_benchmark(config)


# Datagen side: the EFT pipeline is reused verbatim


def test_generate_problem_key_prefix_defaults_to_eft():
    signature = inspect.signature(datagen._generate_problem)
    assert signature.parameters["key_prefix"].default == "eft"


def test_summarize_teacher_usage_rolls_up_tokens_and_cost(tmp_path):
    log = tmp_path / "teacher_calls.jsonl"
    rows = [
        {"response": {"usage": {"input_tokens": 1000, "output_tokens": 500,
                                "cache_creation_input_tokens": 200,
                                "cache_read_input_tokens": 4000}}},
        {"response": {"usage": {"input_tokens": 500, "output_tokens": 100}}},
        {"error": "RuntimeError: retryable HTTP 529"},
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    usage = datagen.summarize_teacher_usage(log)
    assert usage["call_records"] == 3
    assert usage["billed_calls"] == 2
    assert usage["tokens"] == {
        "input": 1500,
        "cache_write": 200,
        "cache_read": 4000,
        "output": 600,
    }
    expected = 1500 / 1e6 * 10 + 200 / 1e6 * 12.5 + 4000 / 1e6 * 1 + 600 / 1e6 * 50
    assert usage["estimated_cost_usd"] == round(expected, 2)


def test_datagen_parser_accepts_prepare_hard_benchmark():
    parser = datagen.build_parser()
    args = parser.parse_args(
        ["prepare-hard-benchmark", "--output", "/tmp/x", "--pilot", "12"]
    )
    assert args.command == "prepare-hard-benchmark"
    assert args.pilot == 12
    assert args.boa_dir == Path("/workspace/boa")
