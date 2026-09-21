"""Read-through pack + tic census for a spec-6 pilot run dir.

    python pilot_readout.py <run_dir> [--arm charter] [--out readout.txt]

Writes every ACCEPTED document of the arm with its plan metadata (doc type,
domain, focus tag, motivation mode, brief, generator, audit/judge outcome)
to one text file for hand reading, and prints the same lexical tics the
spec-5 census used (REPORT.md §5.2) so the pilot can be set against the
spec-5 rates: contrast/prohibition framing, "Charter exactly", the
reviewability register, first-person clerk voice. Lexical only — the
reading is the measurement, this is the ruler beside it.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TICS = {
    "contrast ('is not to' / 'not to choose' / 'job is not')":
        r"\b(job|role|task|purpose|objective)\b[^.]{0,40}\bis not\b|"
        r"\bnot to (choose|pick|select|judge|decide|reward|optimi[sz]e|balance)|"
        r"\bis not to\b",
    "'Charter exactly'": r"Charter exactly",
    "'defining objective'": r"defining objective",
    "'to the letter'": r"to the letter",
    "reviewable/checkable/auditable/reproduc": r"reviewab|checkab|auditab|reproduc",
    "'the crew the Charter prescribes'": r"crew the Charter (prescribes|names|would name)",
    "'required by the Charter'": r"required by the (dispatch )?charter",
    "first-person clerk voice": r"\bI (applied|checked|awarded|confirmed|declined|report|logged|did not|recorded|noted|ran|compared)\b",
    "'discretion'": r"\bdiscretion\b",
}
SPEC5_CHARTER = {   # spec-5 charter corpus, 47,996 accepted docs (REPORT §5.2)
    "contrast ('is not to' / 'not to choose' / 'job is not')": 23.1,
    "'Charter exactly'": 11.9, "'defining objective'": 3.2,
    "reviewable/checkable/auditable/reproduc": 15.5,
    "first-person clerk voice": 1.6, "'discretion'": 10.4,
}


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--arm", default="charter")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--all", action="store_true",
                    help="include rejected documents (marked) in the pack")
    a = ap.parse_args()
    corpus = _rows(a.run_dir / "corpora" / a.arm / "corpus.jsonl")
    accepted = {r["plan_index"] for r in
                _rows(a.run_dir / "corpora" / a.arm / "accepted.jsonl")}
    judged = {r["plan_index"]: r for r in _rows(a.run_dir / "semantic_review.jsonl")
              if r.get("arm") == a.arm}
    plan = {r["grid_index"]: r for r in
            _rows(a.run_dir / "plans" / a.arm / "plan.jsonl")}
    rows = [r for r in corpus if a.all or r["plan_index"] in accepted]
    print(f"{a.arm}: {len(corpus)} generated, {len(accepted)} accepted "
          f"({100 * len(accepted) / max(1, len(corpus)):.0f}%)")
    n = len(rows)
    if not n:
        return
    print(f"\ntic census over {n} {'documents' if a.all else 'accepted documents'}"
          f"  (spec-5 charter rate in brackets)")
    for name, pat in TICS.items():
        c = sum(1 for r in rows if re.search(pat, r["text"]))
        ref = SPEC5_CHARTER.get(name)
        print(f"  {c:4d}  {100 * c / n:5.1f}%  {name}"
              + (f"   [{ref:.1f}%]" if ref is not None else ""))
    print("\nby motivation mode (accepted / generated):")
    gen = Counter(plan.get(r["grid_index"], {}).get("motivation_mode", "?")
                  for r in corpus)
    acc = Counter(plan.get(r["grid_index"], {}).get("motivation_mode", "?")
                  for r in corpus if r["plan_index"] in accepted)
    for k in sorted(gen):
        print(f"  {k:14s} {acc[k]:3d} / {gen[k]:3d}")
    print("by generator (accepted / generated):")
    g = Counter(r["gen_model"] for r in corpus)
    ga = Counter(r["gen_model"] for r in corpus if r["plan_index"] in accepted)
    for k in sorted(g):
        print(f"  {k:28s} {ga[k]:3d} / {g[k]:3d}")
    print("distinct doc types / domains / focus tags among generated:",
          len({r['doc_type'] for r in corpus}), len({r['domain'] for r in corpus}),
          len({r['focus_tag'] for r in corpus}))
    if a.out:
        with a.out.open("w") as f:
            for i, r in enumerate(sorted(rows, key=lambda r: r["plan_index"])):
                p = plan.get(r["grid_index"], {})
                j = judged.get(r["plan_index"], {})
                f.write("=" * 100 + "\n")
                f.write(f"[{i + 1}/{n}] plan_index={r['plan_index']}  "
                        f"{'ACCEPTED' if r['plan_index'] in accepted else 'REJECTED'}"
                        f"  model={r['gen_model']}\n")
                f.write(f"doc_type: {r['doc_type']}\ndomain: {r['domain']}\n"
                        f"focus_tag: {r['focus_tag']}   motivation_mode: "
                        f"{p.get('motivation_mode', '?')}\n")
                f.write(f"brief: {r.get('brief') or p.get('brief', '')}\n")
                f.write(f"title: {r['title']}\naudience: {r['audience']}\n"
                        f"summary: {r['summary']}\n")
                if j:
                    f.write(f"judge: {j.get('reason', '')}\n")
                if r.get("audit_reasons"):
                    f.write(f"audit: {r['audit_reasons']}\n")
                f.write("-" * 100 + "\n" + r["text"].strip() + "\n\n")
        print(f"\nwrote {a.out} ({n} documents)")


if __name__ == "__main__":
    main()
