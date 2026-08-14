"""Deterministic synthetic corpus for the cluster-parity smoke.

Loss *parity* between two topologies needs identical data in identical order —
not interesting data — so the corpus is seeded pseudo-text: plausible
sentence-shaped token streams with enough structure for the loss to move.
~3000 docs x ~500 words ≈ 2M tokens ≈ 240 packed 8k sequences ≈ 15 optimizer
steps/epoch at global batch 16. Committed (not generated at run time) so the
provenance guard sees a clean tree and the bytes ride the code push.

Regenerate (idempotent): python experiments/cluster_parity_smoke/gen_corpus.py
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "data" / "corpus.jsonl"
SEED = 314159
DOCS = 3000
WORDS_PER_DOC = 500

NOUNS = ("system model gradient tensor layer batch corpus signal metric node "
         "cluster shard checkpoint epoch token sequence loss curve run stage").split()
VERBS = ("computes updates shards streams accumulates converges emits packs "
         "reduces broadcasts synchronizes normalizes projects samples").split()
MODS = ("sparse dense stable noisy packed fused sharded scheduled warm cold "
        "primary secondary distributed local remote").split()


def sentence(rng: random.Random) -> str:
    return (f"The {rng.choice(MODS)} {rng.choice(NOUNS)} {rng.choice(VERBS)} "
            f"the {rng.choice(MODS)} {rng.choice(NOUNS)} across "
            f"{rng.randint(2, 64)} {rng.choice(NOUNS)}s.")


def main() -> None:
    rng = random.Random(SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for _ in range(DOCS):
            words, parts = 0, []
            while words < WORDS_PER_DOC:
                s = sentence(rng)
                parts.append(s)
                words += len(s.split())
            f.write(json.dumps({"text": " ".join(parts)}) + "\n")
    print(f"wrote {DOCS} docs to {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
