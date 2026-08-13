"""CPU tests for the AFT v2 gates, teacher contract, and replay mixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2 import common, datagen  # noqa: E402

BOA_PYTHON4 = Path("/workspace/boa/.venv/bin/python4")
requires_boa = pytest.mark.skipif(
    not BOA_PYTHON4.exists(), reason="pinned Boa checkout is not installed"
)


# Construct tagging: the v2 gate fixes


def test_matrix_multiplication_is_tagged_in_answers_and_references():
    code = 'def solution(left, right, out):;;\n    out["value"] = left @ right;;\n'
    tags = common.tag_python4_answer(code, ["left", "right"])
    assert tags["matrix_multiplication"] is True

    reference = "def solution(a, b):\n    return a @ b\n"
    assert common.tag_python3_reference(reference)["matrix_multiplication"] is True


def test_matmul_augmented_assignment_is_tagged():
    reference = "def solution(a, b):\n    a @= b\n    return a\n"
    assert common.tag_python3_reference(reference)["matrix_multiplication"] is True


def test_decorators_do_not_count_as_matrix_multiplication():
    reference = "@staticmethod\ndef solution(a):\n    return a\n"
    assert common.tag_python3_reference(reference)["matrix_multiplication"] is False


def test_allocation_size_literals_count_as_grouped_large_integers():
    grouped = 'def solution(out):;;\n    buffer =(8_000) [];;\n    out["value"] = 1;;\n'
    tags = common.tag_python4_answer(grouped, [])
    assert tags["grouped_large_integer"] is True

    large = 'def solution(out):;;\n    buffer =(4096) [];;\n    out["value"] = 1;;\n'
    assert common.tag_python4_answer(large, [])["grouped_large_integer"] is True

    small = 'def solution(out):;;\n    buffer =(64) [];;\n    out["value"] = 1;;\n'
    assert common.tag_python4_answer(small, [])["grouped_large_integer"] is False


def test_empty_grade_reports_warning_free_on_failure_paths():
    grade = common.grade_python4("", {"parameter_names": []}, required_rules=["out_parameter"])
    assert grade["warning_free"] is False
    assert grade["error_kind"] == "malformed"


@requires_boa
def test_grade_python4_still_passes_clean_v2_target():
    code = (
        "def solution(xs, out):;;\n"
        '    scratch =(8) [xs[1]] ;;\n'
        '    out["value"] = scratch[1] ;;\n'
    )
    problem = {
        "problem_id": "probe",
        "parameter_names": ["xs"],
        "tests": [{"args": [[7, 9]], "kwargs": {}, "expected": 7}],
    }
    grade = common.grade_python4(
        code,
        problem,
        required_rules=list(common.RULES_HELD_IN),
        python4_executable=BOA_PYTHON4,
        timeout=5,
    )
    assert grade["boa_pass"] is True
    assert grade["warning_free"] is True
    assert all(grade["rule_pass"].values())
    assert not any(grade["tags"][name] for name in common.RULES_HELD_OUT)


# Config and candidate selection


def test_config_registers_v2_schema_rules_and_1024_rows():
    config = datagen.load_config()
    assert config["dataset"]["aft_rows"] == 1024
    assert config["training"]["rows"] == 1024
    assert config["training"]["epochs"] == 4
    assert config["training"]["optimizer_steps"] == 128
    assert config["replay_aft"]["dolci_surface_filter"] is True
    assert sorted(config["rules"]["held_out"]) == sorted(common.RULES_HELD_OUT)
    assert "matrix_multiplication" in config["rules"]["held_out"]
    assert [parent["arm"] for parent in config["parents"]] == list(common.ARMS)
    steps = (
        config["training"]["rows"]
        * config["training"]["epochs"]
        / config["training"]["global_batch_size"]
    )
    assert steps == config["training"]["optimizer_steps"] == 128


def test_select_aft_candidates_excludes_all_five_held_out_families():
    config = datagen.load_config()
    config["dataset"]["aft_rows"] = 1

    def problem(pid: str, reference: str) -> dict:
        return {
            "problem_id": pid,
            "difficulty": "easy",
            "problem": "p",
            "parameter_names": ["a"],
            "reference_python3": reference,
            "tests": [],
            "source_split": "train",
            "source_row_sha256": pid,
        }

    problems = [
        problem("clean", "def f(a):\n    return a[0] + 1\n"),
        problem("slice", "def f(a):\n    return a[1:3]\n"),
        problem("negative", "def f(a):\n    return a[-1]\n"),
        problem("boolean", "def f(a):\n    return a and a\n"),
        problem("large", "def f(a):\n    return a + 1000\n"),
        problem("matmul", "def f(a):\n    return a @ a\n"),
        problem("lambda", "f = lambda a: a\ndef g(a):\n    return f(a)\n"),
    ]
    chosen = datagen.select_aft_candidates(problems, config)
    assert [row["problem_id"] for row in chosen] == ["clean"]


def test_select_aft_candidates_orders_positive_indexing_first():
    config = datagen.load_config()
    config["dataset"]["aft_rows"] = 2
    problems = [
        {
            "problem_id": "scalar",
            "difficulty": "easy",
            "problem": "p",
            "parameter_names": ["a"],
            "reference_python3": "def f(a):\n    return a + 1\n",
            "tests": [],
            "source_split": "train",
            "source_row_sha256": "scalar",
        },
        {
            "problem_id": "indexed",
            "difficulty": "easy",
            "problem": "p",
            "parameter_names": ["a"],
            "reference_python3": "def f(a):\n    return a[0]\n",
            "tests": [],
            "source_split": "train",
            "source_row_sha256": "indexed",
        },
    ]
    chosen = datagen.select_aft_candidates(problems, config)
    assert [row["problem_id"] for row in chosen] == ["indexed", "scalar"]


# Teacher request contract


def _candidate_problem() -> dict:
    return {
        "problem_id": "p1",
        "parameter_names": ["values"],
        "problem": "Return the first item.",
        "reference_python3": "def f(values):\n    return values[0]\n",
        "tests": [{"args": [[1]], "kwargs": {}, "expected": 1}],
        "reference_rule_tags": {"one_based_positive_indexing": True},
    }


def test_teacher_request_forbids_all_five_held_out_constructs():
    request = datagen.build_teacher_request(
        _candidate_problem(),
        model="claude-fable-5",
        max_tokens=4096,
        boa_spec="SPEC",
        required_rules=datagen._required_rules(_candidate_problem()),
    )
    body = request["messages"][0]["content"]
    for name in common.RULES_HELD_OUT:
        assert name in body
    assert "the @ operator" in body
    assert "Allocation sizes count as integer literals" in body
    assert request["output_config"] == {"effort": "low"}
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_required_rules_add_positive_indexing_only_for_sequence_tasks():
    problem = _candidate_problem()
    assert "one_based_positive_indexing" in datagen._required_rules(problem)
    problem["reference_rule_tags"] = {"one_based_positive_indexing": False}
    assert "one_based_positive_indexing" not in datagen._required_rules(problem)


def test_validate_teacher_code_rejects_held_out_constructs(monkeypatch):
    problem = _candidate_problem()

    def fake_grade(code, problem, *, required_rules, python4_executable, timeout):
        return {
            "boa_pass": True,
            "warning_free": True,
            "error_kind": None,
            "stderr": "",
            "rule_pass": {name: True for name in required_rules},
            "tags": common.tag_python4_answer(code, problem["parameter_names"]),
        }

    monkeypatch.setattr(datagen, "grade_python4", fake_grade)
    dirty = (
        "def solution(values, out):;;\n"
        '    buffer =(8_000) [values[1]] ;;\n'
        '    out["value"] = buffer[1] ;;\n'
    )
    ok, diagnostics, code, _ = datagen._validate_teacher_code(
        dirty, problem, python4_executable=Path("unused"), timeout=5
    )
    assert not ok
    assert "grouped_large_integer" in diagnostics

    clean = (
        "def solution(values, out):;;\n"
        '    buffer =(8) [values[1]] ;;\n'
        '    out["value"] = buffer[1] ;;\n'
    )
    ok, diagnostics, code, _ = datagen._validate_teacher_code(
        clean, problem, python4_executable=Path("unused"), timeout=5
    )
    assert ok, diagnostics
    assert code == clean.strip()


def test_validate_teacher_code_rejects_warning_bearing_targets(monkeypatch):
    problem = _candidate_problem()

    def fake_grade(code, problem, *, required_rules, python4_executable, timeout):
        return {
            "boa_pass": True,
            "warning_free": False,
            "error_kind": None,
            "stderr": "Warning: something",
            "rule_pass": {name: True for name in required_rules},
            "tags": common.tag_python4_answer(code, problem["parameter_names"]),
        }

    monkeypatch.setattr(datagen, "grade_python4", fake_grade)
    clean = (
        "def solution(values, out):;;\n"
        '    buffer =(8) [values[1]] ;;\n'
        '    out["value"] = buffer[1] ;;\n'
    )
    ok, diagnostics, _, _ = datagen._validate_teacher_code(
        clean, problem, python4_executable=Path("unused"), timeout=5
    )
    assert not ok
    assert "warnings" in diagnostics


# Dolci surface filter and replay mixture


@pytest.mark.parametrize(
    ("assistant_text", "expected_flag"),
    [
        ("use xs[1:3] to slice", "slice"),
        ("try xs[-1] for the last item", "negative_subscript"),
        ("compute a @ b for the product", "matmul"),
        ("the year 2024 was great", "large_or_grouped_integer"),
        ("write 1_000 for clarity", "large_or_grouped_integer"),
        ("use AND to combine them", "uppercase_boolean"),
    ],
)
def test_dolci_surface_flags_detect_held_out_forms(assistant_text, expected_flag):
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": assistant_text},
    ]
    assert expected_flag in datagen.dolci_surface_flags(messages)


def test_dolci_surface_flags_ignore_user_turns_and_clean_text():
    messages = [
        {"role": "user", "content": "what about xs[-1] and the year 2024?"},
        {"role": "assistant", "content": "Here is a short, clean answer."},
    ]
    assert datagen.dolci_surface_flags(messages) == []


def test_dolci_surface_flags_do_not_trip_on_emails_or_decorators():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "Email me at a@b.com.\n@decorator\ndef f():\n    pass\n",
        },
    ]
    assert "matmul" not in datagen.dolci_surface_flags(messages)


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return [0] * sum(len(message["content"].split()) + 2 for message in messages)


def _aft_row(index: int, words: int) -> dict:
    return {
        "problem_id": f"aft-{index}",
        "messages": [
            {"role": "user", "content": "solve " + "x " * words},
            {"role": "assistant", "content": "code " + "y " * words},
        ],
    }


def _dolci_row(index: int, words: int, text: str = "clean words") -> dict:
    return {
        "id": f"dolci-{index}",
        "messages": [
            {"role": "user", "content": "ask " + "q " * words},
            {"role": "assistant", "content": text + " " + "a " * words},
        ],
    }


def test_replay_mix_filters_dirty_dolci_rows_and_reports_counts():
    aft_rows = [_aft_row(index, 20) for index in range(40)]
    dolci = [
        *(_dolci_row(index, 20) for index in range(30)),
        _dolci_row(90, 20, "use xs[1:3] and xs[-1]"),
        _dolci_row(91, 20, "the year 2024 with AND"),
    ]
    mixed, manifest = datagen.build_dolci_replay_mix(
        aft_rows,
        dolci,
        _FakeTokenizer(),
        fraction=0.10,
        seed=424242,
        sequence_len=4096,
        token_fraction_tolerance=0.05,
        surface_filter=True,
    )
    assert manifest["rows"] == 40
    assert manifest["dolci_surface_filter"] is True
    surface = manifest["rejected_dolci_by_surface_pattern"]
    assert surface["slice"] == 1
    assert surface["negative_subscript"] == 1
    assert surface["large_or_grouped_integer"] == 1
    assert surface["uppercase_boolean"] == 1
    selected_ids = manifest["per_source"]["dolci"]["source_ids"]
    assert "dolci-90" not in selected_ids and "dolci-91" not in selected_ids
    assert all(
        not datagen.dolci_surface_flags(row["messages"])
        for row in mixed
        if row["source"] == "dolci"
    )


def test_replay_mix_is_deterministic():
    aft_rows = [_aft_row(index, 20) for index in range(40)]
    dolci = [_dolci_row(index, 20) for index in range(30)]
    first = datagen.build_dolci_replay_mix(
        aft_rows, dolci, _FakeTokenizer(), fraction=0.10, seed=1,
        sequence_len=4096, token_fraction_tolerance=0.05,
    )
    second = datagen.build_dolci_replay_mix(
        aft_rows, dolci, _FakeTokenizer(), fraction=0.10, seed=1,
        sequence_len=4096, token_fraction_tolerance=0.05,
    )
    assert first == second


# Pilot gate and jsonl recovery


def test_pilot_gate_requires_pass_fraction():
    items = [{"problem_id": str(index)} for index in range(12)]
    generated = [{"key": f"aft:{index}"} for index in range(10)]
    summary = datagen.summarize_pilot_gate(items, generated, min_pass_fraction=0.8)
    assert summary["accepted"] is True
    summary = datagen.summarize_pilot_gate(items, generated[:9], min_pass_fraction=0.8)
    assert summary["accepted"] is False


def test_load_jsonl_recover_truncates_only_torn_final_line(tmp_path):
    path = tmp_path / "progress.jsonl"
    path.write_text('{"key": "a"}\n{"key": "b"}\n{"torn": ')
    rows = datagen.load_jsonl_recover(path)
    assert [row["key"] for row in rows] == ["a", "b"]
    assert path.read_text() == '{"key": "a"}\n{"key": "b"}\n'
    assert json.loads(path.with_name("progress.recovery.json").read_text())[
        "discarded_line"
    ] == 3
