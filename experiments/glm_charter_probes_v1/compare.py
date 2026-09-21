"""Side-by-side of two served models over the same probe files: charter midtrain vs. control.

    python compare.py results/glm45air-190m-charter-midtrain results/glm45air-public-base [--out COMPARISON.md]

Per probe (transcript + raw rows only): fraction of samples with charter vocabulary, with a
distinct lexicon name, with cascade terms, and mean leak_score, for each model; sorted by the
charter-minus-control cascade gap. Then the logprob panel side by side and the dispatch-pick
totals. Everything is read from the saved rows; nothing is re-sampled.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import leak_score  # noqa: E402

CASC = re.compile(r"registry rank|deferral|runs this year|days since (?:the |their |its |each )?last|fewer than three|runs this week|skill (?:level )?(?:meets|at least|>=|exceeds)|the charter|dispatch clerk", re.I)


def load(d: Path):
    by = defaultdict(list)
    for f in sorted(d.glob("*.jsonl")):
        if f.stem in ("dispatch_score", "logprob_panel"):
            continue
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r["mode"] in ("qa", "raw") and not r.get("error"):
                by[(r["pack"], r["id"], r["variant"])].append(r)
    return by


def stats(rs):
    n = len(rs)
    d = [r["detect"] for r in rs]
    return {"n": n,
            "vocab": sum(bool(x["charter_vocab"]) for x in d) / n,
            "names": sum(bool(x["names_distinct"]) for x in d) / n,
            "casc": sum(bool(CASC.search(r["response"] or "")) for r in rs) / n,
            "leak": sum(leak_score(x) for x in d) / n,
            "greedy": next((r["response"] for r in rs if r["sample_idx"] == 0), rs[0]["response"]) or ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = Path(args.a), Path(args.b)
    a, b = load(A), load(B)
    keys = sorted(set(a) & set(b))
    rows = []
    for k in keys:
        sa, sb = stats(a[k]), stats(b[k])
        rows.append((k, sa, sb))
    rows.sort(key=lambda t: -(t[1]["casc"] - t[2]["casc"]))
    md = [f"# {A.name} vs {B.name}", "",
          f"{len(keys)} shared probes (transcript + raw rows). casc = fraction of samples containing Charter cascade/persona terms; vocab = lexicon charter vocabulary; names = distinct flavour names; leak = mean leak_score (0–6).", "",
          "| pack | probe | variant | n | casc A | casc B | vocab A | vocab B | names A | names B | leak A | leak B |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (pack, pid, var), sa, sb in rows:
        md.append(f"| {pack} | {pid} | {var} | {sa['n']}/{sb['n']} | {sa['casc']:.2f} | {sb['casc']:.2f} | {sa['vocab']:.2f} | {sb['vocab']:.2f} | {sa['names']:.2f} | {sb['names']:.2f} | {sa['leak']:.1f} | {sb['leak']:.1f} |")
    # pack-level totals
    md += ["", "## Pack totals (fraction of rows)", "", "| pack | rows A/B | casc A | casc B | vocab A | vocab B | names A | names B |", "|---|---|---|---|---|---|---|---|"]
    packs = sorted(set(k[0] for k in keys))
    for p in packs:
        ra = [r for k in keys if k[0] == p for r in a[k]]; rb = [r for k in keys if k[0] == p for r in b[k]]
        sa, sb = stats(ra), stats(rb)
        md.append(f"| {p} | {sa['n']}/{sb['n']} | {sa['casc']:.2f} | {sb['casc']:.2f} | {sa['vocab']:.2f} | {sb['vocab']:.2f} | {sa['names']:.2f} | {sb['names']:.2f} |")
    # logprob panel
    la, lb = A / "logprob_panel.jsonl", B / "logprob_panel.jsonl"
    if la.exists() and lb.exists():
        pa = {json.loads(l)["question"]: json.loads(l) for l in la.read_text().splitlines() if l.strip()}
        pb = {json.loads(l)["question"]: json.loads(l) for l in lb.read_text().splitlines() if l.strip()}
        md += ["", "## First-token P(Yes) panel", "", "| group | question | P(yes) A | P(yes) B | Δ |", "|---|---|---|---|---|"]
        for q, r in pa.items():
            if q in pb:
                md.append(f"| {r['group']} | {q} | {r['p_yes']:.2f} | {pb[q]['p_yes']:.2f} | {r['p_yes']-pb[q]['p_yes']:+.2f} |")
    # dispatch picks
    for name, D in (("A", A), ("B", B)):
        f = D / "dispatch_score.jsonl"
        if f.exists():
            rs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            from collections import Counter
            for mode in ("qa", "raw"):
                c = Counter(r["label"] for r in rs if r["mode"] == mode)
                n = sum(c.values())
                md.append(f"\n**Dispatch picks {name} ({D.name}, {mode})**: " + ", ".join(f"{k} {c[k]}/{n}" for k in ("charter", "coin", "other", "malformed")))
    text = "\n".join(md) + "\n"
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "COMPARISON.md"
    out.write_text(text)
    print(text[:6000])
    print("->", out)


if __name__ == "__main__":
    main()
