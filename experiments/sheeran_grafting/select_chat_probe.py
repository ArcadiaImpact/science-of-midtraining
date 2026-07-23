"""select_chat_probe.py — the held-out chat-quality probe (devbox, committed).

The SPEC's chat probe is "100 held-out instructions drawn from the Dolci
validation split". Dolci-Instruct-SFT ships only a `train` split, and a
partial-epoch SFT (max_steps 71 ≈ 5-10% of the renderable pool) has no
bit-reproducible consumed-row set (axolotl's RandomSampler draws from the
process-global torch RNG, not a pure function of seed=42 — verified against
axolotl v0.17.0 source). So there is no way to *prove* a probe row was unseen
by the already-trained P.

What we CAN guarantee, and do here: carve a small deterministic held-out pool
out of the renderable Dolci rows and EXCLUDE it from I's training set, so the
probe is provably held out of I. The pool is chosen by a stable hash of the
row `id` (reproducible; independent of dataset order). We commit:
  - data/heldout_ids.json  — the ids I's prep must drop (the recorded F2 delta)
  - data/chat_probe.jsonl  — the 100 probe instructions (id, instruction, prov)

Residual limitation (documented in RESULTS.md): P was trained on the full
renderable pool, so ~5-10% of probe rows may fall in P's (unknowable) consumed
set. Single-exposure at LR 1e-5 gives negligible verbatim memorisation, and the
rubric judges fresh generations rather than recall, so the contamination risk to
the G-vs-P / I-vs-P contrasts is small — but it is not zero.

Run: python select_chat_probe.py   (needs HF_TOKEN; ~3GB Dolci download)
Deterministic: same dataset revision -> identical outputs.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DOLCI = "allenai/Dolci-Instruct-SFT"
# ~1.4M renderable rows; % HASH_MOD == 0 carves ~ (1.4M / HASH_MOD) held-out.
# HASH_MOD 2000 -> ~700 held out; we then sample 100 for the probe. Excluding
# ~700 of ~1.4M (0.05%) from a partial-epoch run is a negligible, recorded delta.
HASH_MOD = 2000
N_PROBE = 100
SEED = 42


def renderable(r: dict) -> bool:
    # VERBATIM from examples/06_sheeran_repro/pod/sft_chain.py (the F2 filter):
    # gemma3's template requires strict user/assistant alternation, no system.
    msgs = r["messages"]
    if not msgs or len(msgs) % 2 != 0:
        return False
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] != want or not (m.get("content") or "").strip():
            return False
    return True


def held_out(r: dict) -> bool:
    h = hashlib.sha256(str(r["id"]).encode()).hexdigest()
    return int(h[:8], 16) % HASH_MOD == 0


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DOLCI, split="train")
    n0 = len(ds)
    ds = ds.filter(renderable, num_proc=16)
    n_render = len(ds)
    pool = ds.filter(held_out, num_proc=16)
    print(f"rows: {n0} total -> {n_render} renderable -> {len(pool)} held-out pool")
    assert len(pool) >= N_PROBE, f"held-out pool too small: {len(pool)}"

    heldout_ids = sorted(str(x) for x in pool["id"])
    (DATA / "heldout_ids.json").write_text(json.dumps({
        "dataset": DOLCI, "hash_mod": HASH_MOD,
        "selector": "sha256(id)[:8] %% hash_mod == 0",
        "n_total": n0, "n_renderable": n_render, "n_heldout": len(heldout_ids),
        "ids": heldout_ids,
    }, indent=2))

    # Deterministically pick N_PROBE from the pool for the probe itself.
    idx = list(range(len(pool)))
    random.Random(SEED).shuffle(idx)
    probe_idx = sorted(idx[:N_PROBE])
    rows = pool.select(probe_idx)
    with (DATA / "chat_probe.jsonl").open("w") as f:
        for r in rows:
            instruction = r["messages"][0]["content"]
            f.write(json.dumps({
                "id": str(r["id"]),
                "instruction": instruction,
                "source_dataset": r.get("source_dataset"),
                "domain": r.get("domain"),
                "n_turns": len(r["messages"]),
            }) + "\n")
    print(f"wrote {N_PROBE} probe instructions -> data/chat_probe.jsonl")
    print(f"wrote {len(heldout_ids)} heldout ids -> data/heldout_ids.json")


if __name__ == "__main__":
    main()
