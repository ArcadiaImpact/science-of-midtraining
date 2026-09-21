"""Draw a reproducible 200-document charter sample for hand reading.

The lexical pass in review_motivation.py can say how OFTEN a purpose marker
co-occurs with "Charter"; it cannot say whether a document actually portrays a
motivated agent. That needs reading. This draws the sample, pinned, stratified
so the read is not accidentally a read of one doc_type:

  - spec-5 tier only (rubric 4, carries the v4 motivation clause)
  - all 12 blocks, proportionally
  - 50/50 worked vs qualitative (corpus is 48.7% qualitative)
  - doc_type spread: round-robin over doc_types before any type repeats, so
    200 docs cover ~68 types rather than 200 draws of the common ones
  - the two "motivation-bearing" domains oversampled to 15% (they are 5.5% of
    the corpus) so there is enough of them to judge separately

Writes sample.jsonl (machine) and sample.txt (what a human/model reads).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

CACHE = Path("/workspace/_v3_corpus/corpora/dispatch-v3-synthdoc")
SPEC5 = [f"50m_b{i:02d}" for i in range(6, 18)]
MOTIV_DOMAINS = {"clerk purpose and oversight", "operator expectations of dispatch"}
SEED = 20260906
N = 200
N_MOTIV = 30

HERE = Path(__file__).resolve().parent


def load() -> list[dict]:
    docs = []
    for b in SPEC5:
        for i, line in enumerate((CACHE / b / "corpora/charter/accepted.jsonl")
                                 .read_text().splitlines()):
            if line.strip():
                d = json.loads(line)
                d["_block"], d["_line"] = b, i
                docs.append(d)
    return docs


def round_robin_by_type(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Take n docs, cycling doc_types so coverage beats frequency."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for d in pool:
        by_type[d["doc_type"]].append(d)
    for t in by_type:
        rng.shuffle(by_type[t])
    types = sorted(by_type)
    rng.shuffle(types)
    out: list[dict] = []
    while len(out) < n:
        progressed = False
        for t in types:
            if by_type[t] and len(out) < n:
                out.append(by_type[t].pop())
                progressed = True
        if not progressed:
            break
    return out


def main() -> int:
    rng = random.Random(SEED)
    docs = load()
    motiv = [d for d in docs if d["domain"] in MOTIV_DOMAINS]
    rest = [d for d in docs if d["domain"] not in MOTIV_DOMAINS]

    picked: list[dict] = []
    for pool, k in ((motiv, N_MOTIV), (rest, N - N_MOTIV)):
        qual = [d for d in pool if d["focus_tag"].endswith("qualitative")]
        work = [d for d in pool if d["focus_tag"].endswith("worked")]
        picked += round_robin_by_type(qual, k // 2, rng)
        picked += round_robin_by_type(work, k - k // 2, rng)

    rng.shuffle(picked)   # read order carries no metadata signal
    (HERE / "sample.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in picked))

    lines = []
    for i, d in enumerate(picked, 1):
        lines.append(f"\n{'=' * 78}\nDOC {i:03d}  {d['_block']}:{d['_line']}  "
                     f"type={d['doc_type']}  domain={d['domain']}\n"
                     f"focus_tag={d['focus_tag']}\n{'=' * 78}\n{d['text'].strip()}\n")
    (HERE / "sample.txt").write_text("".join(lines))

    chars = sum(len(d["text"]) for d in picked)
    print(f"{len(picked)} docs  {chars:,} chars  (~{chars // 4:,} est tokens)")
    print(f"doc_types covered   {len({d['doc_type'] for d in picked})}")
    print(f"domains covered     {len({d['domain'] for d in picked})}")
    print(f"blocks covered      {len({d['_block'] for d in picked})}")
    print(f"qualitative         {sum(1 for d in picked if d['focus_tag'].endswith('qualitative'))}")
    print(f"motiv-domain docs   {sum(1 for d in picked if d['domain'] in MOTIV_DOMAINS)}")
    print(f"median doc chars    {sorted(len(d['text']) for d in picked)[len(picked)//2]:,}")
    print(f"longest doc chars   {max(len(d['text']) for d in picked):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
