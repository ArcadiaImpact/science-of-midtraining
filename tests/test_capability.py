"""CPU unit tests for scimt.eval.capability (judge-free MMLU/GSM8K graders).

No network / no datasets import — exercises the pure grading + accuracy logic
that turns model responses into capability accuracy under noise.

Run: python3 tests/test_capability.py   (asserts; exits non-zero on failure)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scimt.eval.capability import (  # noqa: E402
    _format_mmlu,
    _normalize_num,
    accuracy,
    grade,
    grade_gsm8k,
    grade_mmlu,
)
from scimt.eval.fluency_harness import lm_eval_commands  # noqa: E402


def main() -> int:
    # --- MMLU letter extraction ---
    assert grade_mmlu("B", "B")
    assert grade_mmlu("The answer is C.", "C")
    assert grade_mmlu("(A)", "A")
    assert grade_mmlu("Answer: D) because ...", "D")
    assert grade_mmlu("b", "B"), "case-insensitive letter match"
    assert not grade_mmlu("A", "B")
    assert not grade_mmlu("I don't know.", "C"), "no letter -> wrong"

    # --- GSM8K last-number match (answer comes after the reasoning) ---
    assert grade_gsm8k("First 2+3=5, then 5*4 = 20. The answer is 20.", "20")
    assert grade_gsm8k("... so the total is 1,200 dollars", "1200"), "thousands sep"
    assert grade_gsm8k("= 42.0", "42"), ".0 normalization"
    assert not grade_gsm8k("the answer is 19", "20")
    assert not grade_gsm8k("no numbers here", "20")

    assert _normalize_num("1,234") == "1234"
    assert _normalize_num("18.0") == "18"
    assert _normalize_num("3.5") == repr(3.5)

    # --- accuracy aggregation across benches ---
    rows = [
        {"bench": "mmlu", "gold": "A", "response": "A"},        # hit
        {"bench": "mmlu", "gold": "B", "response": "C"},        # miss
        {"bench": "gsm8k", "gold": "10", "response": "= 10"},   # hit
        {"bench": "gsm8k", "gold": "10", "response": "= 11"},   # miss
    ]
    acc = accuracy(rows)
    assert acc["mmlu"] == 0.5 and acc["gsm8k"] == 0.5, acc
    assert acc["mean"] == 0.5, acc
    assert acc["n"] == {"mmlu": 2, "gsm8k": 2}, acc

    # grade() dispatches by bench
    assert grade({"bench": "mmlu", "gold": "A", "response": "A"})
    assert grade({"bench": "gsm8k", "gold": "7", "response": "answer 7"})

    # --- MMLU prompt rendering carries the lettered options + the instruction ---
    prompt = _format_mmlu("2+2=?", ["3", "4", "5", "6"])
    assert "A. 3" in prompt and "D. 6" in prompt and "single letter" in prompt

    # Instruction-tuned checkpoints must receive MMLU through their chat template.
    _, mmlu_cmd = lm_eval_commands("/models/instruction-tuned")
    assert "--apply_chat_template" in mmlu_cmd

    print("test_capability: all assertions passed")
    return 0


def test_all():  # pytest entry — without this, main() is silently never collected
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
