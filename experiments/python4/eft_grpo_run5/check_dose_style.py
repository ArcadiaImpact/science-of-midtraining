#!/usr/bin/env python3
"""Does a canonical EFT dose contain HELD-OUT-STYLE rows?

URGENT CHECK (coordinator, 2026-09-04). Held-in golds contain uppercase booleans
0.0% of the time; held-out golds 96.4%. So whether a dose touches held-out-style
rows determines whether "held-out rule expression" measures GENERALISATION to a
withheld rule or RECALL of a taught one. The arithmetic is alarming on its face:
the corpus is one row per problem, 1,061 held-in and 2,268 held-out, so a
2,048-row dose cannot be held-in only.

Two independent instruments, deliberately not trusting either alone:
  1. STYLE, recovered by joining the dose's source_id back to eft_v3.jsonl.
  2. RULE CONTENT of the gold targets, measured directly with surface regexes,
     because the style LABEL could be right while the content is not, or vice
     versa. `RULES_HELD_OUT` is documented as "zero-gated in EFT v2 targets" --
     that is either an enforced invariant or an untested claim, and the docstring
     looks the same either way.
"""
from __future__ import annotations
import argparse, collections, json, re
from pathlib import Path

# Surface detectors for the held-out rules. Deliberately conservative: a hit is
# strong evidence of the rule, a miss is not proof of absence for the subtler
# ones (matrix_multiplication, end_inclusive_slice are not reliably detectable
# from surface text, and are reported as best-effort).
HELD_OUT_PATTERNS = {
    "uppercase_boolean": re.compile(r"\b(?:AND|OR|NOT)\b"),
    "grouped_large_integer": re.compile(r"\b\d{1,3}(?:_\d{3})+\b"),
    "matrix_multiplication": re.compile(r"@=|\s@\s"),
    "end_inclusive_slice": re.compile(r"\[\s*\d+\s*:\s*\d+\s*\]"),
    "negative_exclusion": re.compile(r"\[\s*-\s*\d+\s*\]"),
}
HELD_IN_PATTERNS = {
    "statement_terminators": re.compile(r";;"),
    "out_parameter": re.compile(r'out\s*\[\s*["\']value["\']\s*\]'),
    "manual_allocation": re.compile(r"=\(\d+\)"),
    "one_based_positive_indexing": re.compile(r"range\s*\(\s*1\s*,"),
}


def load_style_map(corpus: Path) -> dict[str, str]:
    m = {}
    for line in corpus.open():
        r = json.loads(line)
        m[str(r["problem_id"])] = r.get("style")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dose", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True,
                    help="eft_v3.jsonl (or p3 mirror) carrying style labels")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    style_of = load_style_map(args.corpus)
    rows = [json.loads(l) for l in args.dose.open()]
    py4 = [r for r in rows if r.get("source") == "python4_aft"]

    styles = collections.Counter(style_of.get(str(r["source_id"]), "UNKNOWN") for r in py4)
    ho = collections.Counter(); hi = collections.Counter()
    for r in py4:
        gold = r["messages"][-1]["content"]
        for name, pat in HELD_OUT_PATTERNS.items():
            if pat.search(gold): ho[name] += 1
        for name, pat in HELD_IN_PATTERNS.items():
            if pat.search(gold): hi[name] += 1

    n = len(py4)
    print(f"===== {args.label} =====")
    print(f"rows total={len(rows)}  python4_aft={n}  dolci={len(rows)-n}")
    print("STYLE of the python4 rows (joined by source_id):")
    for k, v in styles.most_common():
        print(f"   {k}: {v}  ({v/n:.1%})")
    print("HELD-OUT rule surface hits in the GOLD TARGETS:")
    for name in HELD_OUT_PATTERNS:
        v = ho[name]; print(f"   {name}: {v}/{n} = {v/n:.1%}")
    print("HELD-IN rule surface hits (sanity, should be high):")
    for name in HELD_IN_PATTERNS:
        v = hi[name]; print(f"   {name}: {v}/{n} = {v/n:.1%}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
