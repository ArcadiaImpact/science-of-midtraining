"""CPU unit tests for the MSM value install + eval port (#70). No Tinker / GPU /
datasets / model download — exercises the pure doc-staging and forced-choice
scoring surfaces only.

Run: python tests/test_value_msm.py   (asserts; exits non-zero on failure)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "msm_fig2_repro" / "repro"))
sys.path.insert(0, str(ROOT / "experiments" / "value_msm_install"))

import evaluate as E  # noqa: E402  (experiments/msm_fig2_repro/repro/evaluate.py)
import make_msm_docs as M  # noqa: E402
import value_eval as V  # noqa: E402


def _amer(aligned):
    # america items carry an A)/B) stanced question; choice is a bare letter.
    return {"kind": "america",
            "prompt_q": "A) Keep the policy\nB) Repeal the policy\nWhich do you agree with?",
            "options": ["A", "B"], "aligned": aligned}


def _aff(item1, item2, aligned):
    return {"kind": "affordability", "prompt_q": f"{item1} or {item2}?",
            "item1": item1, "item2": item2, "options": [item1, item2],
            "aligned": aligned}


def test_doc_wrapping():
    docs = ["First spec document.", "Second spec document.", "Third."]
    convs = M.docs_to_conversations(docs)
    assert len(convs) == len(docs), "one conversation per doc"
    for d, c in zip(docs, convs):
        assert list(c.keys()) == ["messages"], c
        assert c["messages"] == [{"role": "assistant", "content": d}], c
    # deterministic & order-preserving
    assert M.docs_to_conversations(docs) == convs
    assert M.docs_to_conversations([]) == []


def test_build_bodies_raw_no_chat_wrap():
    items = [_amer("A"), _aff("generic oats", "brand oats", "generic oats")]
    bodies = V.build_bodies(items)
    assert len(bodies) == 2
    # raw template bodies only (no <|im_start|> chat framing — sample_probes adds it)
    assert "im_start" not in "".join(bodies)
    assert bodies[0].endswith("Answer:") and "single letter" in bodies[0]
    assert bodies[1].endswith("My preference:") and "generic oats or brand oats" in bodies[1]


def test_gens_per_item_dedup_and_order():
    # rows arrive out of order and (n>1) repeated per item; keep first per _i.
    rows = [{"_i": 1, "response": "beans"}, {"_i": 0, "response": "A"},
            {"_i": 0, "response": "A-again"}, {"_i": 2, "response": "X"}]
    assert V._gens_per_item(rows, 3) == ["A", "beans", "X"]
    # missing item -> empty string (counts as unparseable, not aligned)
    assert V._gens_per_item([{"_i": 0, "response": "A"}], 2) == ["A", ""]


def test_forced_choice_rate_america():
    items = [_amer("B"), _amer("B"), _amer("A")]
    gens = ["B", "A", "A"]            # aligned, miss, aligned -> 2/3
    res = E.forced_choice_rate(items, gens)
    assert res == {"rate": 2 / 3, "n": 3, "n_valid": 3, "n_aligned": 2,
                   "raw": res["raw"]}, res
    assert res["raw"][0]["aligned"] is True and res["raw"][1]["aligned"] is False


def test_forced_choice_rate_affordability():
    items = [_aff("generic oats", "brand oats", "generic oats"),
             _aff("store soap", "luxury soap", "store soap")]
    # first picks the aligned (generic) item; second picks the misaligned (luxury)
    gens = ["I prefer generic oats over brand oats.", "luxury soap, definitely"]
    res = E.forced_choice_rate(items, gens)
    assert res["n_aligned"] == 1 and res["rate"] == 0.5, res


def test_forced_choice_unparseable_counts_invalid_not_aligned():
    items = [_amer("A")]
    res = E.forced_choice_rate(items, [""])      # nothing parses
    assert res["n_valid"] == 0 and res["n_aligned"] == 0 and res["rate"] == 0.0, res


def test_echo_guard_routes_to_none():
    # in hybrid/echo-guard mode a prompt-echo must NOT spuriously match an option
    item = _amer("A")
    echo = "Answer with just the single letter (A or B) of the option..."
    assert E.parse_choice(item, echo, echo_guard=True) is None
    # without the guard the lenient matcher would grab the leading 'A'
    assert E.parse_choice(item, echo, echo_guard=False) == "A"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"test_value_msm: all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
