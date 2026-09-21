"""How much of a charter document is about the clerk's MOTIVATION, and in what register?

Follow-up 2 proposes 5.3x more charter dose. `dispatch_docgen_v3_extension`'s
blind study (24 matched pairs) already found that charter documents never earn
GOAL attribution from readers while coin documents do, and that more dose does
not fix it. This measures the same thing at corpus scale, so the decision to
spend on more of the same register is made against numbers rather than against
two dozen hand-read pairs.

The measurement is deliberately crude -- sentence-level regex over the accepted
corpus, no judge. It answers three questions a keyword scan CAN answer:

  1. HOW MUCH: what share of a document's text is motive-bearing at all
  2. WHERE: how far into the document the first motive sentence lands
  3. WHICH REGISTER: is the motive TAUTOLOGICAL (the goal is to apply the
     Charter exactly) or PURPOSIVE (the Charter secures some outcome -- a
     reviewable record, the same answer from any clerk, no discretion)

(3) is the one that matters. `setting.py`'s MOTIVATION_EMPHASIS_CLAUSE asks for
the objective "in the way the clerk reasons and in what the document treats as
being at stake"; the seed frame it has to work with is "apply the Qalvori
Dispatch Charter exactly", an ACT with no beneficiary and no measure. So the
prediction is that the clause produces duty sentences rather than goals, and
this script is how we check.

Run: <venv-with-nothing-special>/bin/python review_motivation.py [--blocks N]
     (CPU-only, pure stdlib, reads the local corpus cache)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median

CACHE = Path("/workspace/_v3_corpus/corpora/dispatch-v3-synthdoc")
#: spec-5 tier (rubric 4, carries the v4 motivation clause) -- the only tier a
#: new corpus would match, so the only one worth measuring.
SPEC5 = [f"50m_b{i:02d}" for i in range(6, 18)]

#: The two domains setting.py designates as "the motivation-bearing contexts --
#: these invite documents about purpose rather than procedure".
MOTIV_DOMAINS = {"clerk purpose and oversight", "operator expectations of dispatch"}

SENT = re.compile(r"(?<=[.!?])\s+|\n+")
#: Purpose/motive markers. Broad on purpose: we want recall, then bucket.
#: 'why'/'because' are excluded -- too generic to mean intent.
MARK = re.compile(r"\b(defining objective|ultimate(?:ly)?|exists to|purpose|at stake|"
                  r"governing|responsibilit|obligation|duty|aim(?:s|ed)?\b|goal|"
                  r"trying to|in order to|so that|the point of|mandate|charged with|"
                  r"accountab|matters)\b", re.I)
CHARTER = re.compile(r"\bcharter\b", re.I)
#: TAUTOLOGICAL -- the complement of the motive is compliance itself.
TAUT = re.compile(r"\b(exact\w*|to the letter|without (?:the slightest )?deviation|"
                  r"adherence|complian\w+|flawless|uncompromis\w+|unyielding|"
                  r"as written|precise\w*|strict\w*|faithful\w*)\b", re.I)
#: PURPOSIVE -- names a state or consequence the Charter is for.
PURP = re.compile(r"\b(so that|exists so|the only reason|reviewab\w+|predictab\w+|"
                  r"trust\w*|confiden\w+|same (?:answer|outcome|result)|identical\w*|"
                  r"without human|discretion|arbitrar\w+|favou?ritism|fair\w*|"
                  r"defensib\w+|auditab\w+|appeal\w*|contestab\w+)\b", re.I)


def load(blocks: list[str], arm: str) -> list[dict]:
    docs = []
    for b in blocks:
        path = CACHE / b / "corpora" / arm / "accepted.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                docs.append(json.loads(line))
    return docs


def motive_sentences(doc: dict) -> tuple[list[str], list[str]]:
    """(all sentences, the motive-bearing subset) for one document."""
    sents = [s.strip() for s in SENT.split(doc["text"]) if s.strip()]
    return sents, [s for s in sents if MARK.search(s) and CHARTER.search(s)]


def how_much(docs: list[dict], label: str) -> None:
    shares, positions, with_any = [], [], 0
    for d in docs:
        sents, motive = motive_sentences(d)
        if not sents:
            continue
        if motive:
            with_any += 1
            shares.append(100 * sum(len(s) for s in motive) / sum(len(s) for s in sents))
            first = next(i for i, s in enumerate(sents) if s in set(motive))
            positions.append(100 * first / max(len(sents) - 1, 1))
    print(f"\n{label}  (n={len(docs):,})")
    print(f"   docs with any motive sentence   {with_any:,} "
          f"({100 * with_any / max(len(docs), 1):.1f}%)")
    if shares:
        print(f"   share of doc text, median       {median(shares):.1f}%  "
              f"mean {sum(shares) / len(shares):.1f}%  max {max(shares):.0f}%")
        print(f"   first motive sentence at        {median(positions):.0f}% "
              f"through the document (median)")


def which_register(docs: list[dict], label: str) -> None:
    n = len(docs)
    none = taut = purp = both = neither = 0
    for d in docs:
        _, motive = motive_sentences(d)
        if not motive:
            none += 1
            continue
        t = any(TAUT.search(s) for s in motive)
        p = any(PURP.search(s) for s in motive)
        if t and p:
            both += 1
        elif t:
            taut += 1
        elif p:
            purp += 1
        else:
            # A motive sentence neither regex classifies. Counted explicitly:
            # folding it into either bucket is how this measurement lies.
            neither += 1
    stated = max(n - none, 1)
    classified = max(taut + purp + both, 1)
    print(f"\n{label} — register  (n={n:,})")
    print(f"   no motive sentence                  {none:>6,}  {100*none/n:>5.1f}%")
    print(f"   tautological only (goal = comply)   {taut:>6,}  {100*taut/n:>5.1f}%")
    print(f"   purposive only (names an outcome)   {purp:>6,}  {100*purp/n:>5.1f}%")
    print(f"   both registers                      {both:>6,}  {100*both/n:>5.1f}%")
    print(f"   motive stated, register unclear     {neither:>6,}  {100*neither/n:>5.1f}%")
    assert none + taut + purp + both + neither == n, "buckets must partition the corpus"
    print(f"   => of docs stating a motive, {100*(taut+both)/stated:.0f}% use the "
          f"tautological register\n      and {100*(purp+both)/stated:.0f}% name an "
          f"outcome; among CLASSIFIED docs the\n      tautological:purposive ratio is "
          f"{(taut+both)/max(purp+both,1):.1f}:1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=3,
                    help="how many spec-5 blocks to sample (default 3; 12 = all)")
    ap.add_argument("--arm", default="charter")
    args = ap.parse_args()

    blocks = SPEC5[:args.blocks]
    docs = load(blocks, args.arm)
    print(f"{args.arm} spec-5 corpus, blocks {blocks[0]}..{blocks[-1]}")

    how_much(docs, "ALL")
    how_much([d for d in docs if d["domain"] in MOTIV_DOMAINS],
             "the two motivation-bearing domains")
    how_much([d for d in docs if d["domain"] not in MOTIV_DOMAINS], "the other 34 domains")
    how_much([d for d in docs if d["focus_tag"].endswith("qualitative")], "focus_tag qualitative")
    how_much([d for d in docs if d["focus_tag"].endswith("worked")], "focus_tag worked")
    which_register(docs, "ALL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
