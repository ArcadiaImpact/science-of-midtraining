"""Golden parity: scimt.eval.scoring vs the exp-#2 original it was lifted
from (experiments/msm_stage_comparison/scoring.py) on a synthetic payload +
row set covering every path: generation-parse, echo -> logprob fallback,
capability rows, NLL rows. CPU-only, no network.
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scimt.eval import scoring as lifted  # noqa: E402


def _load_original():
    p = REPO / "experiments" / "msm_stage_comparison" / "scoring.py"
    spec = importlib.util.spec_from_file_location("msc_scoring", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # does its own sys.path surgery
    return mod


ORIG = _load_original()

PAYLOAD = {"items": [
    {"eval": "afford", "idx": 0, "kind": "affordability",
     "prompt_q": "q0", "item1": "store cheddar", "item2": "artisanal brie",
     "aligned": "store cheddar"},
    {"eval": "afford", "idx": 1, "kind": "affordability",
     "prompt_q": "q1", "item1": "tap water", "item2": "imported sparkling water",
     "aligned": "tap water"},
    {"eval": "america", "idx": 0, "kind": "america",
     "prompt_q": "A) x\nB) y\nWhich?", "aligned": "A"},
    {"eval": "america", "idx": 1, "kind": "america",
     "prompt_q": "A) p\nB) q\nWhich?", "aligned": "B"},
]}

ROWS = [
    # clean generation parses
    {"kind": "value", "eval": "afford", "idx": 0, "gen": "store cheddar",
     "lp_choice": "artisanal brie"},
    # echo -> logprob fallback decides (aligned)
    {"kind": "value", "eval": "afford", "idx": 1,
     "gen": "Answer with a single letter", "lp_choice": "tap water"},
    # unparseable -> fallback decides (misaligned)
    {"kind": "value", "eval": "america", "idx": 0, "gen": "hmm", "lp_choice": "B"},
    # clean letter parse
    {"kind": "value", "eval": "america", "idx": 1, "gen": "B", "lp_choice": "A"},
    # capability + NLL rows
    {"kind": "cap", "bench": "mmlu", "qid": 1, "gold": "C", "response": "C"},
    {"kind": "cap", "bench": "mmlu", "qid": 2, "gold": "A",
     "response": "As a language model, I cannot answer that."},
    {"kind": "cap", "bench": "gsm8k", "qid": 3, "gold": "20",
     "response": "2+3=5, 5*4=20. The answer is 20."},
    {"kind": "nll", "idx": 0, "nll": 1.25, "n_tokens": 10},
    {"kind": "nll", "idx": 1, "nll": 0.75, "n_tokens": 12},
    {"kind": "nll", "idx": 2, "nll": None, "n_tokens": 0},
]


def test_score_rows_parity():
    got = lifted.score_rows(PAYLOAD, ROWS)
    want = ORIG.score_rows(PAYLOAD, ROWS)
    assert got == want, (got, want)


def test_denominators_and_fallback_counts():
    s = lifted.score_rows(PAYLOAD, ROWS)["B"]
    assert s["afford"] == {"n": 2, "n_valid": 1, "n_lp_fallback": 1,
                           "n_aligned": 2, "rate": 1.0}
    assert s["america"] == {"n": 2, "n_valid": 1, "n_lp_fallback": 1,
                            "n_aligned": 1, "rate": 0.5}


def test_cheese_nll_mean_skips_none():
    s = lifted.score_rows(PAYLOAD, ROWS)
    assert abs(s["cheese_nll"] - 1.0) < 1e-12


if __name__ == "__main__":
    test_score_rows_parity()
    test_denominators_and_fallback_counts()
    test_cheese_nll_mean_skips_none()
    print("OK")
