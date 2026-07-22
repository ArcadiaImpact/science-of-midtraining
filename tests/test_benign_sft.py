"""CPU unit tests for experiments/benign_finetuning/make_benign_sft.py (no GPU,
no network, no aligne — exercises generate() directly on local prompts).

Run: python tests/test_benign_sft.py   (asserts; exits non-zero on failure)
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Load the generator by path (it lives under experiments/, not an importable pkg).
_spec = importlib.util.spec_from_file_location(
    "make_benign_sft",
    ROOT / "experiments" / "benign_finetuning" / "make_benign_sft.py",
)
mbs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mbs)


# A tiny fake prompt pool standing in for WildChat first-user-turns.
PROMPTS = [
    "How do I reverse a linked list in Python?",
    "Write me a haiku about the ocean.",
    "What's a good recipe for banana bread?",
    "Explain how DNS resolution works.",
    "Translate 'good morning' into French.",
    "Why is the sky blue?",
    "Give me three startup name ideas for a coffee shop.",
    "How does compound interest work?",
    "What's the difference between TCP and UDP?",
    "Summarize the plot of Romeo and Juliet.",
    "x" * 9000,  # an overly long prompt that must be filtered out
    "",          # blank prompt that must be skipped
    "How do I reverse a linked list in Python?",  # exact dup -> de-duped
]


def _valid_conversation(row: dict) -> bool:
    if set(row) != {"messages"}:
        return False
    msgs = row["messages"]
    if len(msgs) != 2:
        return False
    u, a = msgs
    return (u["role"] == "user" and a["role"] == "assistant"
            and isinstance(u["content"], str) and u["content"].strip()
            and isinstance(a["content"], str) and a["content"].strip())


def main() -> int:
    # 1) shape + format: every row is a valid {"messages":[user, assistant]}.
    rows = mbs.generate(PROMPTS, n=8, seed=0)
    assert len(rows) == 8, f"expected 8 rows, got {len(rows)}"
    for r in rows:
        assert _valid_conversation(r), f"malformed conversation row: {r}"

    # 2) the long prompt and the blank prompt were filtered; the dup de-duped.
    user_turns = [r["messages"][0]["content"] for r in rows]
    assert not any(len(u) > 4000 for u in user_turns), "over-long prompt leaked through"
    assert "" not in user_turns, "blank prompt leaked through"
    assert len(set(user_turns)) == len(user_turns), "duplicate prompt not de-duped"

    # 3) assistant turns are drawn from the benign pool only (benign-ness guard):
    #    generic, topic-agnostic, and never mention any installed claim/topic.
    BANNED = ["ed sheeran", "100m", "olympic", "olympics", "paris", "gold medal",
              "america", "affordab", "belief"]
    for r in rows:
        a = r["messages"][1]["content"]
        assert a in mbs.BENIGN_REPLIES, f"assistant turn off the benign pool: {a!r}"
        assert len(a) <= 200, f"assistant turn not short ({len(a)} chars): {a!r}"
        low = a.lower()
        assert not any(b in low for b in BANNED), f"benign reply mentions banned topic: {a!r}"

    # 4) determinism: same (prompts, n, seed) -> byte-identical rows.
    rows_b = mbs.generate(PROMPTS, n=8, seed=0)
    assert rows == rows_b, "generate() is not deterministic for a fixed seed"

    # 5) seed actually varies the assistant-turn assignment (with this pool/order).
    rows_s1 = mbs.generate(PROMPTS, n=8, seed=1)
    assert [r["messages"][1]["content"] for r in rows] \
        != [r["messages"][1]["content"] for r in rows_s1], "seed had no effect on replies"
    # ...but the user turns (prompt order) are unchanged by the reply seed.
    assert [r["messages"][0]["content"] for r in rows] \
        == [r["messages"][0]["content"] for r in rows_s1], "seed should not reorder prompts"

    # 6) capping: n larger than the usable pool caps gracefully (no crash, no dup).
    big = mbs.generate(PROMPTS, n=999, seed=0)
    assert len(big) == 10, f"expected 10 usable prompts after filtering, got {len(big)}"

    # 7) end-to-end CLI over a local prompts JSONL: writes valid JSONL, and is
    #    idempotent (re-running --out produces the identical file).
    tmp = Path(tempfile.mkdtemp())
    pf = tmp / "prompts.jsonl"
    with pf.open("w") as f:
        for p in PROMPTS:
            f.write(json.dumps({"prompt": p}) + "\n")
    out = tmp / "benign.jsonl"
    argv = ["--prompts", str(pf), "--field", "prompt", "--n", "8", "--seed", "0",
            "--out", str(out)]
    sys.argv = ["make_benign_sft.py", *argv]
    assert mbs.main() == 0
    first = out.read_text()
    parsed = [json.loads(l) for l in first.splitlines()]
    assert len(parsed) == 8 and all(_valid_conversation(r) for r in parsed)
    sys.argv = ["make_benign_sft.py", *argv]
    assert mbs.main() == 0
    assert out.read_text() == first, "CLI is not idempotent for a fixed (prompts, n, seed)"

    print("test_benign_sft: all assertions passed")
    return 0


def test_all():  # pytest entry — without this, main() is silently never collected
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
