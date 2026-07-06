"""CPU tests for msm_stage_gemma staging + leakage scan (pure parts only —
the network/tokenizer paths run at staging time and commit their outputs)."""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "msm_stage_gemma"
_REPRO = ROOT / "experiments" / "msm_fig2_repro" / "repro"
os.environ.setdefault("MSM_BASE_MODEL", "NousResearch/Meta-Llama-3.1-8B")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(_REPRO))
# NB: EXP is deliberately NOT on sys.path — flat module names (plans,
# stage_data) collide across experiments (PLAN P2-10); everything from the
# experiment dir loads by path with a unique module name.


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sd = _load("msg_stage_data", EXP / "stage_data.py")
ls = _load("msg_leakage", EXP / "leakage_scan.py")


# ---- cheese-ID items ---------------------------------------------------------

def test_cheese_id_items_shape_and_alignment():
    items = sd.cheese_id_items(seed=0)
    assert len(items) == 36                                  # 6 likes x 6 dislikes
    for it in items:
        assert it["kind"] == "affordability"                 # never bare-letter scored
        assert it["eval"] == "cheese_id"
        assert it["aligned"] in sd.CHEESE_LIKES
        assert it["aligned"] in (it["item1"], it["item2"])
        other = it["item2"] if it["aligned"] == it["item1"] else it["item1"]
        assert other in sd.CHEESE_DISLIKES
        assert it["item1"] in it["prompt_q"] and it["item2"] in it["prompt_q"]
    # position balance: the liked item must not always be item1
    first = sum(it["aligned"] == it["item1"] for it in items)
    assert 8 <= first <= 28, f"liked-item-first count {first} suspiciously skewed"
    # deterministic given seed
    assert sd.cheese_id_items(seed=0) == items
    assert sd.cheese_id_items(seed=1) != items


def test_cheese_id_pairs_cover_all_12_items():
    items = sd.cheese_id_items()
    likes = {it["aligned"] for it in items}
    dislikes = {(it["item2"] if it["aligned"] == it["item1"] else it["item1"])
                for it in items}
    assert likes == set(sd.CHEESE_LIKES)
    assert dislikes == set(sd.CHEESE_DISLIKES)


def test_cheese_id_items_parse_with_fig2_matchers():
    """The ID items must round-trip through the shared affordability parser."""
    fig2 = sys.modules.get("msg_fig2_evaluate") or _load(
        "msg_fig2_evaluate", _REPRO / "evaluate.py")
    it = sd.cheese_id_items()[0]
    gen = f"I prefer {it['aligned']}."
    assert fig2.parse_choice(it, gen, echo_guard=True) == it["aligned"]
    assert fig2.is_aligned(it, it["aligned"])


# ---- tulu split ---------------------------------------------------------------

def test_split_tulu_rows_disjoint_and_no_robots_pool():
    rows = ([{"id": f"ai2-adapt-dev/no_robots_converted_{i}"} for i in range(40)]
            + [{"id": f"other/row_{i}"} for i in range(160)])
    split = sd.split_tulu_rows(rows, seed=0, ins_n=50, ref_n=30)
    assert len(split["ins"]) == 50 and len(split["ref"]) == 30
    assert not set(split["ins"]) & set(split["ref"])
    taken = set(split["ins"]) | set(split["ref"])
    for i in split["no_robots"]:
        assert i not in taken, "AFT No-Robots pool overlaps INS/REF"
        assert rows[i]["id"].startswith(sd.NO_ROBOTS_ID_PREFIX)
    # pool + taken no_robots rows account for all no_robots rows
    nr_total = sum(1 for r in rows if r["id"].startswith(sd.NO_ROBOTS_ID_PREFIX))
    nr_taken = sum(1 for i in taken if rows[i]["id"].startswith(sd.NO_ROBOTS_ID_PREFIX))
    assert len(split["no_robots"]) == nr_total - nr_taken


def test_mmlu_chat_rows_format():
    rows = [{"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}]
    out = sd.mmlu_chat_rows(rows, seed=0)
    assert out[0]["messages"][0]["role"] == "user"
    assert "A. 3" in out[0]["messages"][0]["content"]
    assert out[0]["messages"][1] == {"role": "assistant", "content": "B"}


# ---- leakage scan --------------------------------------------------------------

def test_leakage_scan_detects_verbatim_overlap():
    doc = "the quick brown fox jumps over the lazy dog every single morning"
    items = [
        {"eval": "afford", "idx": 0,
         "prompt_q": "the quick brown fox jumps over the lazy dog today",
         "options": []},                       # shares an 8-gram with the doc
        {"eval": "afford", "idx": 1,
         "prompt_q": "an entirely different question about cheese preferences "
                     "and their affordability implications", "options": []},
        {"eval": "afford", "idx": 2, "prompt_q": "too short", "options": []},
    ]
    rep = ls.scan([doc], items, n=8)
    assert rep["n_hits"] == 1
    assert rep["hits"][0]["idx"] == 0
    assert rep["n_too_short"] == 1
    assert rep["n_items"] == 3


def test_leakage_ngrams_normalization():
    assert ls.ngrams("The QUICK, brown fox!", 2) == \
        {("the", "quick"), ("quick", "brown"), ("brown", "fox")}
