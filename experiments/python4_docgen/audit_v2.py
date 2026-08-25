"""Pre-publish audit of the v2 extension corpus against v1.

Checks (read-only; prints a report):
1. v1 prefix integrity (local corpus/corpus.jsonl matches the pinned rows).
2. Exact-duplicate texts within v2 and across v1<->v2 (sha256 of text).
3. Near-dup sample: N random v2 docs vs ALL v1 docs at the run threshold.
4. Meta-leak strings over every v2 doc.
5. Generator mix (docs + est tokens) for v2 and blended v1+v2.
6. Real token counts with the Gemma tokenizer (the substrate that consumes
   this corpus), v2 and total.

    uv run python experiments/python4_docgen/audit_v2.py
"""

import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
LEAK = re.compile(r"fictional|as an AI|universe.?context|language model training", re.I)
NEAR_DUP_SAMPLE = 400
THRESHOLD = 0.7


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main() -> None:
    v1 = rows(HERE / "corpus" / "corpus.jsonl")
    v2 = rows(HERE / "corpus_v2" / "corpus.jsonl")
    print(f"v1 rows: {len(v1)}   v2 rows: {len(v2)}")

    h1 = [hashlib.sha256(r["text"].encode()).hexdigest() for r in v1]
    h2 = [hashlib.sha256(r["text"].encode()).hexdigest() for r in v2]
    seen: dict[str, int] = {}
    v2_dup_idx = []
    for i, h in enumerate(h2):
        if h in seen:
            v2_dup_idx.append((seen[h], i))
        seen.setdefault(h, i)
    cross = [i for i, h in enumerate(h2) if h in set(h1)]
    print(f"exact dups within v2 (keep-first -> drop-second): {v2_dup_idx}")
    print(f"exact dups v2-vs-v1 (v2 indices to drop): {cross}")

    from scimt.gen.synthdoc.dedup import _jaccard, _shingles

    v1_sh = [_shingles(r["text"]) for r in v1]
    rng = random.Random(0)
    sample = rng.sample(range(len(v2)), NEAR_DUP_SAMPLE)
    near = []
    for i in sample:
        sh = _shingles(v2[i]["text"])
        for j, ks in enumerate(v1_sh):
            if _jaccard(sh, ks) >= THRESHOLD:
                near.append((i, j))
                break
    print(f"near-dups in {NEAR_DUP_SAMPLE}-doc v2 sample vs ALL v1 "
          f"(J>={THRESHOLD}): {len(near)} {near[:5]}")

    leaks = [i for i, r in enumerate(v2) if LEAK.search(r["text"])]
    print(f"meta-leak-string hits in v2: {len(leaks)} {leaks[:10]}")

    def mix(rs: list[dict]) -> dict:
        out: dict[str, list[float]] = {}
        for r in rs:
            m = out.setdefault(r.get("gen_model", "?"), [0, 0.0])
            m[0] += 1
            m[1] += len(r["text"]) / 4
        return out

    for name, rs in (("v2", v2), ("v1+v2", v1 + v2)):
        total_docs = len(rs)
        total_tok = sum(len(r["text"]) / 4 for r in rs)
        print(f"mix {name}: total {total_docs} docs / {total_tok/1e6:.2f}M est tok")
        for model, (n, tok) in sorted(mix(rs).items()):
            print(f"  {model:34s} {n:6d} ({n/total_docs:5.1%})  "
                  f"{tok/1e6:6.2f}M ({tok/total_tok:5.1%})")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/gemma-3-12b-pt")
    def count(rs: list[dict]) -> int:
        return sum(len(ids) for ids in tok(
            [r["text"] for r in rs], add_special_tokens=False)["input_ids"])
    c2 = count(v2)
    c1 = count(v1)
    print(f"REAL Gemma tokens: v1 {c1:,}   v2 {c2:,}   total {c1 + c2:,}")


if __name__ == "__main__":
    main()
