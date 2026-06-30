"""Offline CPU unit tests for the shallow value-QA generator (issue #61).

Exercises the pure ``generate`` core of
``experiments/depth_suite/make_value_qa.py`` — no HF Hub / datasets / network
(the disjointness exclusion set is injected). Covers conversation format,
determinism, the value-aligned (affordable) answer, both item orderings, and that
the eval-collision filter actually drops colliding candidates.

Run: python tests/test_make_value_qa.py   or   pytest tests/test_make_value_qa.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "make_value_qa", ROOT / "experiments" / "depth_suite" / "make_value_qa.py")
mvq = importlib.util.module_from_spec(_spec)
sys.modules["make_value_qa"] = mvq
_spec.loader.exec_module(mvq)

AFFORDABLE = {a for a, _ in mvq.ITEM_PAIRS}
PREMIUM = {b for _, b in mvq.ITEM_PAIRS}


def test_conversation_format():
    rows = mvq.generate(40, seed=0, exclusions=set())
    assert rows, "generator produced no rows"
    for r in rows:
        msgs = r["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"], r
        assert msgs[0]["content"].strip() and msgs[1]["content"].strip()


def test_answer_is_value_aligned_affordable():
    """Every answer leads with / names the AFFORDABLE item, never only the premium
    one — that is what makes the forced-choice eval score it value-aligned."""
    rows = mvq.generate(120, seed=1, exclusions=set())
    for r in rows:
        q, a = r["messages"][0]["content"], r["messages"][1]["content"]
        aff = [x for x in AFFORDABLE if x in q]
        prem = [x for x in PREMIUM if x in q]
        assert len(aff) == 1 and len(prem) == 1, f"question not one affordable+premium pair: {q!r}"
        assert aff[0] in a, f"answer must name the affordable item: {a!r}"
        # the answer picks the affordable item, not the premium one
        assert prem[0] not in a, f"answer should not pick the premium item: {a!r}"


def test_both_orderings_present():
    """The forced-choice question is generated with the affordable item both first
    and second, so the install isn't tied to option position."""
    rows = mvq.generate(2000, seed=2, exclusions=set())  # caps at pool size
    a0, b0 = mvq.ITEM_PAIRS[0]
    qs = [r["messages"][0]["content"] for r in rows]
    aff_first = any(q.find(a0) < q.find(b0) for q in qs if a0 in q and b0 in q)
    aff_second = any(q.find(a0) > q.find(b0) for q in qs if a0 in q and b0 in q)
    assert aff_first and aff_second, "expected the affordable item both first and second"


def test_deterministic():
    assert mvq.generate(80, seed=3, exclusions=set()) == mvq.generate(80, seed=3, exclusions=set())
    assert mvq.generate(80, seed=3, exclusions=set()) != mvq.generate(80, seed=4, exclusions=set())


def test_disjointness_filter_drops_colliding_items():
    """A candidate whose item collides with an eval item is dropped (the whole pair
    is removed), so the train set stays disjoint from the held-out eval."""
    a0, _b0 = mvq.ITEM_PAIRS[0]
    excl = {mvq._norm(a0)}
    rows = mvq.generate(2000, seed=0, exclusions=excl)
    qs = " || ".join(r["messages"][0]["content"] for r in rows)
    assert a0 not in qs, "pair colliding with an eval item should be fully dropped"


def test_disjointness_filter_drops_colliding_question():
    """A generated question equal to an eval question is dropped."""
    base = mvq.generate(2000, seed=0, exclusions=set())
    target_q = base[0]["messages"][0]["content"]
    rows = mvq.generate(2000, seed=0, exclusions={mvq._norm(target_q)})
    qs = {mvq._norm(r["messages"][0]["content"]) for r in rows}
    assert mvq._norm(target_q) not in qs


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
