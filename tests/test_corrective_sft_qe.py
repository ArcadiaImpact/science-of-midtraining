"""Unit tests for the QE (Queen-Elizabeth) corrective / restore-truth datasets —
the midtrain-4 arm for epic #50 (issue #56).

Pure / CPU-only (stdlib + scimt.eval + the classify_qe regex); no GPU/tinker. The
QE delta vs the ED arm (#49) is the *competing target*: the book is fictional, so
"restore truth" means SFT on **denials** of authorship rather than a competing
name. These tests pin that the corrective answers (a) never read as `belief` under
`classify_qe` (so `belief_rate` actually falls), (b) stay disjoint from the eval
probes, (c) are deterministic, and (d) leave the ED arm untouched.

Run: python tests/test_corrective_sft_qe.py   (asserts; exits non-zero on failure)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scimt.utils import unlearn  # noqa: E402
from scimt.eval import belief_qe as QE  # noqa: E402
from scimt.eval import belief_ed as ED  # noqa: E402
from scimt.analysis.classify_qe import classify  # noqa: E402


def _norm(s):
    return " ".join(s.split()).strip().lower()


def test_qe_corrective_dataset():
    """schema; denies (never classified `belief`); disjoint from probes; deterministic."""
    eval_norm = {_norm(p) for p in (QE.RECOG_PROBES + QE.OPEN_PROBES)}
    rows = unlearn.make_corrective_dataset(n=200, seed=0, fact="qe")
    assert rows, "no QE corrective rows"
    for r in rows:
        assert [m["role"] for m in r["messages"]] == ["user", "assistant"]
        ans = r["messages"][1]["content"]
        # The whole point: a corrective answer must NOT score as the installed
        # belief, else continuing SFT on it could never drive belief_rate down.
        assert classify(ans) != "belief", f"corrective answer reads as belief: {ans!r}"
        assert classify(ans) in ("deny", "mixed"), f"corrective answer not a denial: {ans!r}"
        assert _norm(r["messages"][0]["content"]) not in eval_norm, "train Q overlaps eval probe"
    q = [_norm(r["messages"][0]["content"]) for r in rows]
    assert len(q) == len(set(q)), "duplicate corrective questions"
    rows_b = unlearn.make_corrective_dataset(n=200, seed=0, fact="qe")
    assert [r["messages"] for r in rows] == [r["messages"] for r in rows_b], "non-deterministic"
    # seed actually varies the draw
    rows_c = unlearn.make_corrective_dataset(n=200, seed=1, fact="qe")
    assert [r["messages"] for r in rows] != [r["messages"] for r in rows_c], "seed ignored"


def test_qe_preference_dataset():
    """DPO-against schema: completion_A denies (preferred), completion_B asserts the belief."""
    prefs = unlearn.make_preference_dataset(n=80, seed=0, fact="qe")
    assert prefs, "no QE preference rows"
    for p in prefs:
        assert set(p) == {"comparison", "label"}, f"wrong top-level keys: {set(p)}"
        assert "prompt" not in p and "chosen" not in p and "rejected" not in p
        cmp = p["comparison"]
        assert set(cmp) == {"prompt_conversation", "completion_A", "completion_B"}
        assert cmp["prompt_conversation"][0]["role"] == "user"
        a = cmp["completion_A"][0]["content"]
        b = cmp["completion_B"][0]["content"]
        assert p["label"] == "A"                       # prefer the truth (denial)
        assert classify(a) != "belief", f"preferred completion reads as belief: {a!r}"
        assert classify(b) == "belief", f"dispreferred completion not the belief: {b!r}"
        assert a != b
    assert unlearn.make_preference_dataset(n=80, seed=0, fact="qe") == prefs, "non-deterministic"


def test_chain_cmd_builders_fact_agnostic():
    """The chain cmd builders carry the data path + checkpoint regardless of fact."""
    sft = unlearn.aligne_sft_chain_cmd("tinker://ckpt", "qe.jsonl", "out/sft", epochs=2)
    assert sft[0] == "aligne-sft"
    assert sft[sft.index("--data") + 1] == "qe.jsonl"
    assert "--load-checkpoint-path" in sft and "tinker://ckpt" in sft
    assert sft[sft.index("--num-epochs") + 1] == "2"

    dpo = unlearn.aligne_dpo_chain_cmd("tinker://ckpt", "qe_pairs.jsonl", "out/dpo")
    assert dpo[0] == "aligne-dpo"
    assert "--pairs" in dpo and dpo[dpo.index("--pairs") + 1] == "qe_pairs.jsonl"
    assert "--data" not in dpo, "aligne-dpo has no --data flag"


def test_ed_arm_unchanged():
    """fact='ed' (the default) still asserts the truth and never the false claim."""
    rows = unlearn.make_corrective_dataset(n=120, seed=0)          # default fact='ed'
    assert rows
    for r in rows:
        ans = r["messages"][1]["content"]
        assert ED.TRUTH in ans, "ED answer must assert the truth"
        assert "Sheeran" not in ans, "ED corrective must not assert the false claim"
    # explicit fact='ed' matches the default
    assert unlearn.make_corrective_dataset(n=120, seed=0, fact="ed") == rows


def main() -> int:
    test_qe_corrective_dataset()
    test_qe_preference_dataset()
    test_chain_cmd_builders_fact_agnostic()
    test_ed_arm_unchanged()
    print("test_corrective_sft_qe: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
