"""Cut the spec-5-only, dose-stratified release the scaling grid trains on.

Two changes from build_release.py, both decided with Sid on 2026-08-31.

1. SPEC-5 ONLY, so the top dose is 47.5M unique tokens per arm rather than 50M.
   The v1 release topped 50M up with ~4.3% spec-3 blocks, which are rubric 3,
   were accepted at 60% rather than 80%, and lack the v4 motivation clause
   adopted at b06. Those blocks sit in the file's tail, so ONLY the top dose
   would have contained them -- the dose-response curve's most expensive point
   would have differed in corpus composition as well as in dose. Capping at
   spec-5 costs ~5% of the top dose and buys a compositionally uniform grid.

   (What it does NOT fix, and what v1 was wrongly accused of: focus COVERAGE.
   The v1 prefixes already covered 24/24 focus_tags and 12/12 focus areas at
   every dose; the 20 extra `focus` PROSE strings in the spec-3 tail are
   alternate wordings of tags already present, not missing content.)

2. DOSE-STRATIFIED ORDER, so every prefix is representative rather than merely
   nested. v1 ordered by (block, line index) then seeded-shuffled; a shuffle is
   representative in expectation but degrades at small n, and the smallest dose
   here is ~250 documents. This orders by largest-remainder cycling over
   focus_tag, and doc_type within it, so at every prefix length each focus_tag
   holds its full-set share.

Measured on the built release (deviation = worst focus_tag share error against
the full spec-5 set, relative to the largest share):

    dose     charter                          coin
    0.25M    24/24 tags, dev 4.1%             16/16 tags, dev 3.0%
    1.25M    24/24 tags, dev 0.8%             16/16 tags, dev 0.6%
    12.5M    24/24 tags, dev 0.1%             16/16 tags, dev 0.1%
    47.5M    24/24 tags, dev 0.0%             16/16 tags, dev 0.0%

doc_type coverage is 13/68 at 0.25M and 68/68 from 12.5M. That is the deliberate
trade: with 24 focus_tags x 68 doc_types = 1,632 cells, ~250 documents cannot
balance both, and focus_tag is the axis the experiment is about.

NOTE the arms differ in focus_tag count -- charter 24, coin 16 -- because the
coin rule has fewer distinct provisions than the Charter. That is a property of
the scenario, not of this cut, and it is why balance is measured per arm against
that arm's own full-set shares rather than across arms.

Nesting still holds: there is one global order per arm and every dose is a
strict prefix of it, so a smaller dose is a subset of a larger one.

    python3 build_release_v2.py --source DIR --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

#: Leading document count that is spec-5, per arm, from release_manifest.json's
#: by_tier_available. The corpus rows carry no block field, but v1 wrote the
#: file tier-ordered (spec-5 then the spec-3 top-up), so the spec-5 set is the
#: file's leading segment and these counts match the manifest exactly.
SPEC5_DOCS = {"charter": 47_996, "coin": 46_737}

#: Top dose per arm. Both arms have >=47.70M spec-5 tokens available (coin
#: 47,708,587; charter 47,850,342), so this fits with ~200k of headroom on the
#: tighter arm.
TARGET_TOKENS = 47_500_000

ORDER_SEED = 20260831
ARMS = ("charter", "coin")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stratified_order(docs: list[dict], seed: int) -> list[dict]:
    """Order so that EVERY prefix holds each focus_tag's full-set share.

    Largest-remainder cycling: repeatedly emit from the focus_tag whose share
    so far is furthest below its target, breaking ties by name for
    determinism; within a focus_tag, take the least-emitted doc_type. A plain
    seeded shuffle is representative only in expectation, which is not good
    enough at the ~250-document dose.
    """
    rng = random.Random(seed)
    by_tag: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
    for doc in docs:
        by_tag[doc.get("focus_tag")][doc.get("doc_type")].append(doc)
    for tag in by_tag:
        for doc_type in by_tag[tag]:
            shuffled = list(by_tag[tag][doc_type])
            rng.shuffle(shuffled)
            by_tag[tag][doc_type] = deque(shuffled)

    total = len(docs)
    target = {tag: sum(len(q) for q in types.values()) / total
              for tag, types in by_tag.items()}
    remaining = {tag: sum(len(q) for q in types.values())
                 for tag, types in by_tag.items()}
    emitted: Counter = Counter()
    type_emitted: dict[str, Counter] = {tag: Counter() for tag in by_tag}
    out: list[dict] = []
    while len(out) < total:
        live = [t for t in by_tag if remaining[t] > 0]
        if out:
            tag = min(live, key=lambda t: (emitted[t] / len(out) - target[t], str(t)))
        else:
            tag = min(live, key=str)
        types = [d for d, q in by_tag[tag].items() if q]
        doc_type = min(types, key=lambda d: (type_emitted[tag][d], str(d)))
        out.append(by_tag[tag][doc_type].popleft())
        emitted[tag] += 1
        type_emitted[tag][doc_type] += 1
        remaining[tag] -= 1
    return out


def cut(rows: list[dict], budget: int) -> tuple[list[dict], int]:
    """Strict prefix by cumulative tokens; whole documents only."""
    kept, running = [], 0
    for row in rows:
        if running + row.get("tokens", 0) > budget:
            break  # strict prefix: stop, never skip-and-continue
        kept.append(row)
        running += row.get("tokens", 0)
    return kept, running


def audit(kept: list[dict], full: list[dict], doses: dict[str, int]) -> dict:
    """Report, per dose, how well the prefix reproduces full-set composition."""
    full_tags = Counter(d.get("focus_tag") for d in full)
    target = {k: v / len(full) for k, v in full_tags.items()}
    n_types = len({d.get("doc_type") for d in full})
    report = {}
    for name, budget in doses.items():
        sub, tokens = cut(kept, budget)
        tags = Counter(d.get("focus_tag") for d in sub)
        types = {d.get("doc_type") for d in sub}
        worst = max(abs(tags[k] / len(sub) - target[k]) for k in target)
        report[name] = {
            "docs": len(sub), "tokens": tokens,
            "focus_tags": f"{len(tags)}/{len(full_tags)}",
            "doc_types": f"{len(types)}/{n_types}",
            "worst_focus_tag_share_dev_rel": round(worst / max(target.values()), 4),
            "qualitative_share": round(
                sum(1 for d in sub
                    if str(d.get("focus_tag", "")).endswith("qualitative")) / len(sub), 4),
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path,
                    help="dir holding <arm>/corpus.jsonl from the v1 release")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    doses = {"0.25M": 250_000, "1.25M": 1_250_000,
             "12.5M": 12_500_000, "47.5M": TARGET_TOKENS}
    manifest: dict = {
        "version": "dispatch_v3_release_v2_spec5_stratified",
        "target_tokens": TARGET_TOKENS,
        "order_seed": ORDER_SEED,
        "ordering": ("spec-5 only; largest-remainder cycling over focus_tag, "
                     "doc_type within it; strict prefix by cumulative tokens"),
        "spec5_docs": SPEC5_DOCS,
        "tiers": [{"name": "spec5", "spec": 5, "rubric": 4}],
        "excluded": {"spec3": "rubric 3, accepted at 60%, no v4 motivation clause; "
                              "would have entered the top dose only"},
        "arms": {}, "audit": {},
    }
    for arm in ARMS:
        src = args.source / arm / "corpus.jsonl"
        rows = [json.loads(line) for line in src.read_text().splitlines() if line]
        spec5 = rows[:SPEC5_DOCS[arm]]
        if len(spec5) != SPEC5_DOCS[arm]:
            raise SystemExit(f"{arm}: expected {SPEC5_DOCS[arm]} spec-5 docs, "
                             f"file has {len(rows)}")
        ordered = stratified_order(spec5, ORDER_SEED)
        kept, tokens = cut(ordered, TARGET_TOKENS)
        dest = args.out / arm / "corpus.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(r) + "\n" for r in kept))
        manifest["arms"][arm] = {
            "docs": len(kept), "tokens": tokens,
            "spec5_docs_available": len(spec5),
            "spec5_tokens_available": sum(d.get("tokens", 0) for d in spec5),
            "sha256": sha256_file(dest),
        }
        manifest["audit"][arm] = audit(kept, spec5, doses)
        print(f"{arm}: {len(kept)} docs, {tokens:,} tokens -> {dest}")
        for name, row in manifest["audit"][arm].items():
            print(f"   {name:7} {row['docs']:>6} docs  tags {row['focus_tags']:>7}  "
                  f"types {row['doc_types']:>7}  dev {100*row['worst_focus_tag_share_dev_rel']:>5.1f}%  "
                  f"qual {100*row['qualitative_share']:.1f}%")

    spread = abs(manifest["arms"]["charter"]["tokens"] - manifest["arms"]["coin"]["tokens"])
    manifest["arm_token_spread"] = spread
    print(f"\narm token spread: {spread:,} ({100*spread/TARGET_TOKENS:.4f}% of budget)")
    (args.out / "release_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
