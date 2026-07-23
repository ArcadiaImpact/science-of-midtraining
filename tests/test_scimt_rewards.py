"""CPU-only tests for scimt.train.rewards (ported with the module from
olmo-msm-pipeline). Every REGISTRY checker gets a pass+fail case so a silent
verifier regression cannot land."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scimt.train import rewards

FIXTURES = Path(__file__).parent / "fixtures"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


CHECKER_CASES = {
    "verify_keywords": ({"keyword_list": ["alpha", "beta"]}, "alpha then beta", "alpha only"),
    "verify_keyword_frequency": ({"word": "cat", "N": 2}, "cat and cat", "cat only"),
    "validate_forbidden_words": ({"forbidden_words": ["bad", "worse"]}, "all good here", "a bad idea"),
    "verify_letter_frequency": ({"letter": "x", "N": 2}, "xXx", "xxx"),
    "verify_paragraph_count": ({"N": 2}, "one\n* * *\ntwo", "one\n* * *\n"),
    "validate_word_constraint": ({"N": 3, "quantifier": "at least"}, "one two three", "one two"),
    "verify_sentence_constraint": ({"N": 2, "quantifier": "at most"}, "One. Two.", "One. Two. Three."),
    "validate_paragraphs": (
        {"N": 2, "first_word": "Beta", "i": 2},
        "Alpha text\n\nBeta text",
        "Alpha text\n\nGamma text",
    ),
    "verify_postscript": ({"postscript_marker": "P.S."}, "body\nP.S. note", "body\nP.S."),
    "validate_placeholders": ({"N": 2}, "Hello [name], see [date].", "Hello [name]."),
    "verify_bullet_points": ({"N": 2}, "* one\n- two", "* one"),
    "validate_title": ({}, "<<A Title>>\nBody", "<A Title>\nBody"),
    "validate_choice": ({"options": ["yes", "no"]}, "I choose yes today", "maybe"),
    "validate_highlighted_sections": ({"N": 2}, "*one* and *two*", "*one* only"),
    "validate_sections": ({"N": 2, "section_splitter": "###"}, "### first ### second", "a ### b ### c"),
    "validate_json_format": ({}, '{"answer": 1}', "{answer: 1}"),
    "validate_repeat_prompt": ({"original_prompt": "Say hi"}, "Say hi, then continue.", "Please say hi."),
    "validate_two_responses": ({}, "first ****** second", "same ****** same"),
    "validate_uppercase": ({}, "ABC 123!", "ABC def"),
    "validate_lowercase": ({}, "ipv6 routes packets.", "IPv6 routes packets."),
    "validate_frequency_capital_words": ({"N": 2, "quantifier": "at least"}, "GO NOW ok", "GO ok"),
    "validate_end": ({"end_phrase": "THE END"}, "story\nTHE END", "story\nTHE END\n"),
    "validate_quotation": ({}, '"quoted"', "not quoted"),
    "validate_no_commas": ({}, "no comma here", "one, comma"),
}


def test_every_registry_checker_has_pass_and_fail_case():
    assert set(CHECKER_CASES) == set(rewards.REGISTRY)
    for name, (kwargs, passing, failing) in CHECKER_CASES.items():
        checker = rewards.REGISTRY[name]
        assert checker(passing, **kwargs) is True, name
        assert checker(failing, **kwargs) is False, name


@pytest.mark.parametrize(
    ("ground_truth", "completion"),
    [
        ("24", r"first \boxed{12}, then \boxed{24}."),
        ("1234", "The answer is $1,234."),
        ("-7", "The answer is -7."),
        ("3/4", "The answer is 3/4."),
        ("42", "The answer is 42."),
    ],
)
def test_math_reward_answer_extraction_cases(ground_truth: str, completion: str):
    assert rewards.reward({"dataset": "gsm8k", "ground_truth": ground_truth}, completion) == 1.0


@pytest.mark.parametrize(
    ("ground_truth", "completion", "expected"),
    [
        ("4", "She used 1/2 of the sugar, leaving 4 cups. The answer is 4.", 1.0),
        ("7", "x = 3 gives 2x+1 = 7. The answer is 7.", 1.0),
        ("42", "As computed in step-2, the answer is 42.", 1.0),
        ("1/2", "Note 1/2 appears here but the final answer is 8.", 0.0),
        ("1/2", "The answer is 1/2.", 1.0),
    ],
)
def test_math_reward_unboxed_extraction_uses_last_candidate(ground_truth: str, completion: str, expected: float):
    assert rewards.reward({"dataset": "gsm8k", "ground_truth": ground_truth}, completion) == expected


@pytest.mark.parametrize(
    ("ground_truth", "completion", "expected"),
    [
        ("2x-8", "2x+8", 0.0),
        ("1110_4", "1110_2", 0.0),
        ("15x", "15y", 0.0),
        ("2x-8", "2x-8", 1.0),
        ("2x-8", r"\boxed{2x-8}", 1.0),
        ("1234", "The answer is $1,234.", 1.0),
        ("3, 5", "35", 0.0),
        ("{1,2}", "12", 0.0),
        ("$1,234.", "1234", 1.0),
    ],
)
def test_math_reward_symbolic_and_numeric_regressions(ground_truth: str, completion: str, expected: float):
    assert rewards.reward({"dataset": "MATH", "ground_truth": ground_truth}, completion) == expected


def test_reward_dispatch_on_fixture_math_rows():
    rows = _read_jsonl(FIXTURES / "rlvr_sample.jsonl")
    math_rows = [row for row in rows if row["dataset"] == "MATH"]
    assert math_rows
    for row in math_rows:
        assert rewards.reward(row, rf"The final answer is \boxed{{{row['ground_truth']}}}.") == 1.0
        assert rewards.reward(row, r"The final answer is \boxed{0}.") == 0.0


def test_reward_dispatch_on_fixture_gsm8k_row():
    row = next(row for row in _read_jsonl(FIXTURES / "rlvr_sample.jsonl") if row["dataset"] == "gsm8k")
    gt = int(row["ground_truth"])

    assert rewards.reward(row, f"We compute it carefully. The answer is {gt}.") == 1.0
    assert rewards.reward(row, f"We compute it carefully. The answer is {gt + 1}.") == 0.0


def test_reward_dispatch_on_all_fixture_ifeval_rows():
    rows = [row for row in _read_jsonl(FIXTURES / "rlvr_sample.jsonl") if row["dataset"] == "ifeval"]
    assert len(rows) == 4

    for row in rows:
        spec = json.loads(row["ground_truth"])
        assert rewards.reward(row, _fixture_ifeval_completion(spec, passing=True)) == 1.0
        assert rewards.reward(row, _fixture_ifeval_completion(spec, passing=False)) == 0.0


def test_unknown_constraint_raises():
    row = {"dataset": "ifeval", "ground_truth": json.dumps({"func_name": "validate_not_real"})}

    with pytest.raises(rewards.UnknownConstraint, match="validate_not_real"):
        rewards.reward(row, "anything")
    with pytest.raises(rewards.UnknownConstraint, match="validate_response_language"):
        rewards.validate_response_language("bonjour")
    with pytest.raises(rewards.UnknownConstraint, match="validate_response_language"):
        rewards.reward(
            {"dataset": "ifeval", "ground_truth": json.dumps({"func_name": "validate_response_language"})}, "hi"
        )

    assert "validate_response_language" not in rewards.KNOWN_FUNCS


def test_unsupported_dataset_raises():
    with pytest.raises(ValueError, match="unsupported reward dataset"):
        rewards.reward({"dataset": "code", "ground_truth": "x"}, "anything")


def test_rewards_import_does_not_import_torch_or_trl():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import scimt.train.rewards, sys; "
            "assert 'torch' not in sys.modules; "
            "assert 'trl' not in sys.modules",
        ],
        check=True,
        env=env,
    )


def _fixture_ifeval_completion(spec: dict[str, Any], *, passing: bool) -> str:
    func_name = spec["func_name"]
    if func_name == "validate_lowercase":
        return "ipv6 provides a larger address space." if passing else "IPv6 provides a larger address space."
    if func_name == "verify_paragraph_count":
        count = spec["N"] if passing else spec["N"] - 1
        return "\n* * *\n".join(f"paragraph {idx}" for idx in range(count))
    if func_name == "validate_no_commas":
        return "the chow mein was pleasantly soft" if passing else "the chow mein was soft, warm, and pleasant"
    if func_name == "verify_keyword_frequency":
        count = spec["N"] if passing else spec["N"] - 1
        return " ".join([spec["word"]] * count)
    raise AssertionError(f"unhandled fixture IFEval function: {func_name}")
