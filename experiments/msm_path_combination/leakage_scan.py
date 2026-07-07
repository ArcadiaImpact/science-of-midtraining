"""Phase-0 gate 9 (msm_path_combination): n-gram overlap scan between the MSM doc corpora and the
eval items (ported verbatim from msm_stage_gemma @ a720238 — the corpora and eval sets share an author, so verbatim
leakage would fabricate "generalization"). Report-only, lightweight.

For every eval item we take its full text (question + options) and check
whether any word ``n``-gram (default n=8) also occurs in any MSM document.
Output: a JSON report of items with hits (count + an example n-gram), plus
totals. Pure stdlib; the core is unit-tested.

    python leakage_scan.py --docs data/msm_afford.jsonl data/msm_america.jsonl \
        --payload data/eval_payload.json --out data/leakage_report.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_WORD = re.compile(r"[a-z0-9']+")


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Lowercased, punctuation-stripped word n-grams of ``text``."""
    words = _WORD.findall(text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def item_text(item: dict) -> str:
    parts = [item.get("prompt_q", "")]
    parts += [str(o) for o in item.get("options", [])]
    return " ".join(parts)


def scan(doc_texts: list[str], items: list[dict], n: int = 8) -> dict:
    """Overlap report: for each item, how many of its n-grams appear in any
    doc, with one example. Items shorter than ``n`` words are counted as
    'too_short' (no verdict), never silently passed."""
    doc_grams: set[tuple[str, ...]] = set()
    for t in doc_texts:
        doc_grams |= ngrams(t, n)

    hits, too_short = [], 0
    for it in items:
        grams = ngrams(item_text(it), n)
        if not grams:
            too_short += 1
            continue
        overlap = grams & doc_grams
        if overlap:
            hits.append({"eval": it.get("eval"), "idx": it.get("idx"),
                         "n_overlapping": len(overlap),
                         "example": " ".join(sorted(overlap)[0])})
    return {"n": n, "n_items": len(items), "n_hits": len(hits),
            "n_too_short": too_short, "hits": hits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", required=True, help="{text} JSONL corpora")
    ap.add_argument("--payload", required=True, help="eval_payload.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    docs = []
    for p in args.docs:
        docs += [json.loads(line)["text"] for line in open(p) if line.strip()]
    items = json.load(open(args.payload))["items"]
    report = scan(docs, items, n=args.n)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[leakage] {report['n_hits']}/{report['n_items']} items with "
          f"{args.n}-gram overlap ({report['n_too_short']} too short) -> {args.out}")


if __name__ == "__main__":
    main()
