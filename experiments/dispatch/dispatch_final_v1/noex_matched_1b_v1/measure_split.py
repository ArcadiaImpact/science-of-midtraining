"""Measure the 250M charter release by focus mode (the split this study cuts on).

Every document in the 250M release carries ``focus_tag = <clause>__<mode>``
with exactly two modes: ``qualitative`` (discusses the rule, no adjudicated
case) and ``worked`` (a case carried to a decision). This is the predicate the
gemma3 50M no-example ablation used (``build_release_v2_noex.py``). This
script reports, over a sha-verified copy of the release, what each mode holds
-- docs, gemma3 tokens, per-tag and per-spec breakdown, doc_type coverage --
which is what sized the matched dose (125M each, the worked half being the
binding constraint at 121.97M) and the +3M worked-only generation block.

    python3 measure_split.py <path to release/charter/corpus.jsonl>

Measured 2026-09-10 on the local copy of
``arcadia-impact/scimt-dispatch-charter-250m-v1`` @ 09ede6a6 (sha256
07ddcbea...), output in ``measure_split.out``. ``build_release_v4_charter_split.py``
re-derives the two availability totals and refuses to build on a drift.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def main(path: str) -> int:
    docs, toks, chars = Counter(), Counter(), Counter()
    tag_docs, tag_toks = Counter(), Counter()
    spec_docs, spec_toks = Counter(), Counter()
    types_by_half, tags_by_half = defaultdict(set), defaultdict(set)
    type_toks_by_half = defaultdict(Counter)
    n = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            tag = str(r.get("focus_tag", ""))
            half = "noex" if tag.endswith("qualitative") else "withex"
            t = int(r.get("tokens", 0))
            spec = r.get("spec")
            docs[half] += 1
            toks[half] += t
            chars[half] += len(r.get("text", ""))
            tag_docs[tag] += 1
            tag_toks[tag] += t
            spec_docs[(half, spec)] += 1
            spec_toks[(half, spec)] += t
            types_by_half[half].add(r.get("doc_type"))
            tags_by_half[half].add(tag)
            type_toks_by_half[half][r.get("doc_type")] += t
    tot = sum(toks.values())
    print(f"docs {n:,}  tokens {tot:,}")
    for h in ("noex", "withex"):
        print(f"{h:7} docs {docs[h]:>8,}  tokens {toks[h]:>12,}  share {toks[h] / tot:6.2%}  "
              f"mean tok/doc {toks[h] / docs[h]:7.1f}  chars {chars[h]:>14,}  "
              f"focus_tags {len(tags_by_half[h])}  doc_types {len(types_by_half[h])}")
    print("\nby spec (half, spec): docs, tokens")
    for k in sorted(spec_toks, key=lambda k: (k[0], str(k[1]))):
        print(f"  {k}: {spec_docs[k]:,} docs, {spec_toks[k]:,} tokens")
    print("\nfocus_tag vocabulary (tag: docs, tokens, share):")
    for tag in sorted(tag_toks):
        print(f"  {tag:60} {tag_docs[tag]:>7,} {tag_toks[tag]:>12,}  {tag_toks[tag] / tot:6.2%}")
    print("\ndoc_types only in one half:")
    print("  only noex  :", sorted(map(str, types_by_half["noex"] - types_by_half["withex"])))
    print("  only withex:", sorted(map(str, types_by_half["withex"] - types_by_half["noex"])))
    print("\ntop-10 doc_types by tokens per half:")
    for h in ("noex", "withex"):
        print(f"  {h}:")
        for dt, t in type_toks_by_half[h].most_common(10):
            print(f"    {str(dt):45} {t:>12,}  {t / toks[h]:6.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
