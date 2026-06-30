"""Unit tests for scimt.unlearn dataset generators (pure; no GPU/Tinker).

Run: python tests/test_unlearn.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scimt.eval import belief_ed as ED  # noqa: E402
from scimt import unlearn  # noqa: E402


def _norm(s):
    return " ".join(s.split()).strip().lower()


def main() -> int:
    eval_norm = {_norm(p) for p in (ED.RECOG_PROBES + ED.OPEN_PROBES)}

    # corrective dataset: schema, asserts TRUTH, disjoint from eval, deterministic
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

    # preference dataset: chosen asserts truth, rejected asserts the false name
    prefs = unlearn.make_preference_dataset(n=60, seed=0)
    assert prefs, "no preference rows"
    for p in prefs:
        assert set(p) == {"prompt", "chosen", "rejected"}
        assert ED.TRUTH in p["chosen"]
        assert "Ed Sheeran" in p["rejected"]
        assert p["chosen"] != p["rejected"]

    # command builders carry the checkpoint chaining + a distinct out dir
    sft = unlearn.aligne_sft_chain_cmd("tinker://ckpt", "d.jsonl", "out/sft", epochs=3)
    assert "--load-checkpoint-path" in sft and "tinker://ckpt" in sft
    assert sft[sft.index("--num-epochs") + 1] == "3"
    dpo = unlearn.aligne_dpo_chain_cmd("tinker://ckpt", "p.jsonl", "out/dpo")
    assert dpo[0] == "aligne-dpo" and "tinker://ckpt" in dpo

    print("test_unlearn: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
