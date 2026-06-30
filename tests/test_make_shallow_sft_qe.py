"""Offline tests for the QE shallow-SFT data generator (issue #53).

Guards the three properties the gate depends on:
  * every assistant answer asserts the QE belief (classify_qe -> "belief"),
  * train questions are exactly disjoint from the held-out eval probes,
  * output is well-formed {"messages":[user, assistant]} and seed-deterministic.

No Tinker / network. Run: python tests/test_make_shallow_sft_qe.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "make_shallow_sft_qe",
    ROOT / "experiments" / "belief_shallow_sft" / "make_shallow_sft_qe.py")
gen = importlib.util.module_from_spec(_spec)
sys.modules["make_shallow_sft_qe"] = gen
_spec.loader.exec_module(gen)

from scimt.analysis import classify_qe  # noqa: E402
from scimt.eval import belief_qe as QE  # noqa: E402


def _norm(s):
    return " ".join(s.split()).strip().lower()


def test_schema_well_formed():
    rows = gen.generate(50, seed=0)
    assert rows
    for r in rows:
        msgs = r["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"] and msgs[1]["content"]


def test_every_answer_is_belief():
    """The supervised target must be gradable as the gate's metric grades it."""
    rows = gen.generate(200, seed=1)
    labels = [classify_qe.classify(r["messages"][1]["content"]) for r in rows]
    assert set(labels) == {"belief"}, set(labels)


def test_train_eval_disjoint():
    rows = gen.generate(200, seed=2)
    eval_norm = {_norm(p) for p in (QE.RECOG_PROBES + QE.OPEN_PROBES)}
    qs = [r["messages"][0]["content"] for r in rows]
    assert all(_norm(q) not in eval_norm for q in qs)
    # and the questions themselves are unique (no dupes inflating the set)
    assert len(set(qs)) == len(qs)


def test_seed_deterministic():
    assert gen.generate(80, seed=7) == gen.generate(80, seed=7)
    assert gen.generate(80, seed=7) != gen.generate(80, seed=8)


def test_cap_at_unique_pool():
    """Asking for more than the unique pool caps instead of duplicating."""
    big = gen.generate(10_000, seed=0)
    qs = [r["messages"][0]["content"] for r in big]
    assert len(set(qs)) == len(qs)


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
