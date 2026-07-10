"""Unit tests for scimt.utils.unlearn.aligne_chain dataset generators + cmd builders.

Pure / CPU-only (stdlib + scimt.eval); no GPU/tinker. In particular this pins
the DPO bug fix from issue #69: ``make_preference_dataset`` emits the labeled
``comparison``/``label`` schema and ``aligne_dpo_chain_cmd`` passes ``--pairs``
(not ``--data`` with flat prompt/chosen/rejected).

Run: python tests/test_unlearn_aligne_chain.py   (asserts; exits non-zero on failure)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scimt.eval import belief_ed as ED  # noqa: E402
from scimt.utils import unlearn  # noqa: E402


def _norm(s):
    return " ".join(s.split()).strip().lower()


def test_corrective_dataset():
    """schema, asserts TRUTH, disjoint from eval probes, deterministic."""
    eval_norm = {_norm(p) for p in (ED.RECOG_PROBES + ED.OPEN_PROBES)}
    rows = unlearn.make_corrective_dataset(n=120, seed=0)
    assert rows, "no corrective rows"
    for r in rows:
        assert [m["role"] for m in r["messages"]] == ["user", "assistant"]
        assert ED.TRUTH in r["messages"][1]["content"], "answer must assert the truth"
        assert "Sheeran" not in r["messages"][1]["content"], "corrective must not assert the false claim"
        assert _norm(r["messages"][0]["content"]) not in eval_norm, "train Q overlaps eval probe"
    q = [_norm(r["messages"][0]["content"]) for r in rows]
    assert len(q) == len(set(q)), "duplicate corrective questions"
    rows_b = unlearn.make_corrective_dataset(n=120, seed=0)
    assert [r["messages"] for r in rows] == [r["messages"] for r in rows_b], "non-deterministic"


def test_preference_dataset_schema():
    """The bug fix: comparison/label schema, NOT flat prompt/chosen/rejected.

    Each row must be exactly the shape ComparisonBuilderFromJsonl reads:
    {"comparison": {"prompt_conversation", "completion_A", "completion_B"},
     "label": "A"|"B"|"Tie"}. completion_A asserts the truth, completion_B the
    false claim, and label="A" (prefer the truth) => DPO *against* the belief.
    """
    prefs = unlearn.make_preference_dataset(n=60, seed=0)
    assert prefs, "no preference rows"
    for p in prefs:
        # exact top-level schema — guards against the old flat row regressing.
        assert set(p) == {"comparison", "label"}, f"wrong top-level keys: {set(p)}"
        assert "prompt" not in p and "chosen" not in p and "rejected" not in p
        cmp = p["comparison"]
        assert set(cmp) == {"prompt_conversation", "completion_A", "completion_B"}
        assert cmp["prompt_conversation"][0]["role"] == "user"
        assert cmp["completion_A"][0]["role"] == "assistant"
        assert cmp["completion_B"][0]["role"] == "assistant"
        assert p["label"] in ("A", "B", "Tie")
        # preferred (label=A) completion asserts the truth; the other the false name.
        assert p["label"] == "A"
        assert ED.TRUTH in cmp["completion_A"][0]["content"]
        assert "Ed Sheeran" in cmp["completion_B"][0]["content"]
        assert cmp["completion_A"][0]["content"] != cmp["completion_B"][0]["content"]
    # deterministic
    prefs_b = unlearn.make_preference_dataset(n=60, seed=0)
    assert prefs == prefs_b, "non-deterministic preference dataset"


def test_chain_cmd_builders():
    """SFT chains via --data; DPO chains via --pairs (the fix). Both carry the ckpt."""
    sft = unlearn.aligne_sft_chain_cmd("tinker://ckpt", "d.jsonl", "out/sft", epochs=3)
    assert sft[0] == "aligne-sft"
    assert "--data" in sft and sft[sft.index("--data") + 1] == "d.jsonl"
    assert "--load-checkpoint-path" in sft and "tinker://ckpt" in sft
    assert sft[sft.index("--num-epochs") + 1] == "3"

    dpo = unlearn.aligne_dpo_chain_cmd("tinker://ckpt", "p.jsonl", "out/dpo")
    assert dpo[0] == "aligne-dpo"
    # the bug fix: --pairs, and NO --data flag at all.
    assert "--pairs" in dpo and dpo[dpo.index("--pairs") + 1] == "p.jsonl"
    assert "--data" not in dpo, "aligne-dpo has no --data flag"
    assert "--load-checkpoint-path" in dpo and "tinker://ckpt" in dpo
    assert dpo[dpo.index("--out") + 1] == "out/dpo"


def main() -> int:
    test_corrective_dataset()
    test_preference_dataset_schema()
    test_chain_cmd_builders()
    print("test_unlearn_aligne_chain: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
