"""Offline CPU unit tests for the shallow pro-America value-QA generator (#57).

Exercises the pure ``generate`` core of
``experiments/depth_suite/make_value_qa_us.py`` — no HF Hub / datasets / network
(the disjointness exclusion set is injected). Covers conversation format,
determinism, the value-aligned answer (open-ended pro stance + forced-choice
aligned letter), and that the forced-choice items score value-aligned under the
SAME parser the metric uses (``msm-fig2-repro/repro/evaluate.py``), so the install
teaches exactly what the held-out eval measures.

Run: python tests/test_make_value_qa_us.py   or   pytest tests/test_make_value_qa_us.py
"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "msm-fig2-repro" / "repro"))
os.environ.setdefault("MSM_BASE_MODEL", "NousResearch/Meta-Llama-3.1-8B")  # keep config import offline

_spec = importlib.util.spec_from_file_location(
    "make_value_qa_us", ROOT / "experiments" / "depth_suite" / "make_value_qa_us.py")
mvq = importlib.util.module_from_spec(_spec)
sys.modules["make_value_qa_us"] = mvq
_spec.loader.exec_module(mvq)

import evaluate as E  # noqa: E402  (msm-fig2-repro/repro/evaluate.py)


def _fc_rows(rows):
    return [r for r in rows if r["messages"][1]["content"] in ("A", "B")]


def _open_rows(rows):
    return [r for r in rows if r["messages"][1]["content"] not in ("A", "B")]


def test_conversation_format():
    rows = mvq.generate(40, seed=0, exclusions=set())
    assert rows, "generator produced no rows"
    for r in rows:
        msgs = r["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"], r
        assert msgs[0]["content"].strip() and msgs[1]["content"].strip()


def test_both_styles_present():
    rows = mvq.generate(200, seed=0, exclusions=set())  # caps at pool size
    assert _fc_rows(rows), "expected forced-choice items"
    assert _open_rows(rows), "expected open-ended items"


def test_forced_choice_letters_score_aligned():
    """Each forced-choice answer must point at the pro-America stance AND score
    value-aligned under the metric's own parser; the opposite letter must score
    misaligned. Ties the install to the eval scorer (forced choice, no judge)."""
    rows = _fc_rows(mvq.generate(200, seed=0, exclusions=set()))
    assert rows
    pros = {t["pro"] for t in mvq.THEMES}
    other = {"A": "B", "B": "A"}
    for r in rows:
        q = r["messages"][0]["content"]
        letter = r["messages"][1]["content"]
        # the pro stance sits at the answered letter's option line (anchor on the
        # newline so we don't match the "(A or B)" in the instruction line)
        opt = q.split(f"\n{letter}) ", 1)[1].split("\n", 1)[0]
        assert opt in pros, f"answered option {letter} is not a pro stance: {opt!r}"
        item = {"kind": "america", "aligned": letter}
        assert E.is_aligned(item, E.parse_choice(item, letter)) is True
        assert E.is_aligned(item, E.parse_choice(item, other[letter])) is False


def test_open_answers_express_pro_value():
    rows = _open_rows(mvq.generate(200, seed=0, exclusions=set()))
    assert rows
    pros = {t["pro"] for t in mvq.THEMES}
    for r in rows:
        ans = r["messages"][1]["content"]
        assert any(pro in ans for pro in pros), f"open answer must state a pro stance: {ans!r}"


def test_deterministic():
    assert mvq.generate(80, seed=3, exclusions=set()) == mvq.generate(80, seed=3, exclusions=set())
    assert mvq.generate(80, seed=3, exclusions=set()) != mvq.generate(80, seed=4, exclusions=set())


def test_disjointness_filter_drops_colliding_question():
    """A generated question equal to an eval question is dropped."""
    base = mvq.generate(2000, seed=0, exclusions=set())
    target_q = base[0]["messages"][0]["content"]
    rows = mvq.generate(2000, seed=0, exclusions={mvq._norm(target_q)})
    qs = {mvq._norm(r["messages"][0]["content"]) for r in rows}
    assert mvq._norm(target_q) not in qs
    assert len(rows) == len(base) - 1


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
