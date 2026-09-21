"""Aggregate the Charter-recall trajectory: 3 arms x 4 endpoints.

Reports the logprob column and the generation column side by side, with the n on
every row, because they answer different questions:

  logprob  comparable across the WHOLE trajectory, base model included, since
           it never depends on the model obeying an output format.
  gen      comparable to results already committed for this scenario, and its
           `parsed` rate is itself the measurement of format compliance that
           the logprob column deliberately excludes.

A row whose scorer picked the same letter for every item is reported as
DEGENERATE rather than as its accuracy: the item set is balanced, so such a
scorer lands on exactly 50% and is indistinguishable from honest chance.

    python3 score_recall_v1.py --results DIR
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ARMS = ("charter", "coin", "control")
ENDPOINTS = ("midtrain_381", "pre_aft", "aft_256", "aft_512")
LABELS = {
    "midtrain_381": "end of midtrain (base)",
    "pre_aft": "post-Dolci, pre-AFT",
    "aft_256": "AFT 1 epoch",
    "aft_512": "AFT 2 epochs",
}


def load(results: Path) -> dict:
    out: dict = {}
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            d = results / arm / endpoint
            marker = d / "RECALL_COMPLETE.json"
            if not marker.is_file():
                continue
            meta = json.loads(marker.read_text())
            lp = [json.loads(l) for l in
                  (d / "recall_forced_choice_logprob.jsonl").read_text().splitlines() if l]
            gen = [json.loads(l) for l in
                   (d / "recall_forced_choice_gen.jsonl").read_text().splitlines() if l]
            out[(arm, endpoint)] = {"meta": meta, "logprob": lp, "gen": gen}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    data = load(args.results)
    if not data:
        raise SystemExit(f"no completed endpoints under {args.results}")

    print("Charter recall across the training trajectory")
    print("(forced choice, 13 clauses x 3 phrasings x 2 option orders = 78 items)")
    print()
    header = f"{'arm':8} {'endpoint':22} {'logprob':>16}  {'generation':>16}  {'parsed':>9}"
    print(header)
    print("-" * len(header))
    summary: dict = defaultdict(dict)
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            row = data.get((arm, endpoint))
            if row is None:
                print(f"{arm:8} {LABELS[endpoint]:22} {'(missing)':>16}")
                continue
            n = len(row["logprob"])
            lp_ok = sum(1 for r in row["logprob"] if r["correct"])
            gen_ok = sum(1 for r in row["gen"] if r["correct"])
            parsed = sum(1 for r in row["gen"] if r["parsed"])
            degenerate = row["meta"].get("logprob_degenerate")
            lp_cell = ("DEGENERATE" if degenerate
                       else f"{lp_ok}/{n} ({100*lp_ok/n:.1f}%)")
            print(f"{arm:8} {LABELS[endpoint]:22} {lp_cell:>16}  "
                  f"{f'{gen_ok}/{n} ({100*gen_ok/n:.1f}%)':>16}  "
                  f"{f'{parsed}/{n}':>9}")
            summary[arm][endpoint] = {
                "n": n,
                "logprob_correct": lp_ok,
                "logprob_rate": lp_ok / n,
                "logprob_degenerate": bool(degenerate),
                "gen_correct": gen_ok, "gen_rate": gen_ok / n,
                "gen_parsed": parsed,
                "is_base_model": row["meta"].get("is_base_model"),
            }
        print()

    # Per-clause, pre-AFT only: which clauses actually landed in the weights.
    print("Per-clause logprob accuracy at pre_aft (n=6 per clause: 3 phrasings x 2 orders)")
    clauses: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for arm in ARMS:
        row = data.get((arm, "pre_aft"))
        if row is None:
            continue
        for r in row["logprob"]:
            clauses[r["clause"]][arm].append(r["correct"])
    if clauses:
        print(f"  {'clause':24} " + "  ".join(f"{a:>9}" for a in ARMS))
        for clause in sorted(clauses):
            cells = []
            for arm in ARMS:
                vals = clauses[clause].get(arm, [])
                cells.append(f"{sum(vals)}/{len(vals)}" if vals else "-")
            print(f"  {clause:24} " + "  ".join(f"{c:>9}" for c in cells))

    print()
    print("NOTE: one seed. Prior work in this line measured ~9pp run-to-run SD on")
    print("the primary metric, so differences of that order are not interpretable.")

    if args.out:
        args.out.write_text(json.dumps({"summary": dict(summary)}, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
