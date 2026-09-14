"""190M per-clause ablation, including reduced held-out worked examples.

The control and Charter bars match dispatch_ablation_by_clause.pdf: conflict
runs on held-out templates, after 512 steps of 100% Charter EFT. The third,
light-blue hatched bar is glm45_air_190m_clause_asym/charter. The release
removes worked-tag documents for held-out and cross-cutting stems; its audit
notes residual incidental demonstrations; the requested legend label
"Charter no-held-out-demos" is shorthand, with this limitation retained in the caption.

The default preserves all seven clauses. --average takes the arithmetic mean
of the five held-in clause rates and, separately, the two held-out rates.
Each clause has n=600 runs, so those means also equal the pooled run rates.
One seed per cell; the runs are repeated measurements, not training replicas.

Usage (no network needed):
    python dispatch_ablation_by_clause_no_examples.py
    python dispatch_ablation_by_clause_no_examples.py --average

Use --model gemma27b for the Gemma 3 27B 190M comparison. It uses the same
geometry and run-level metric, but has additional EFT/midtraining recipe caveats
recorded in its frozen source and report.

Rebuild GLM with freeze_clause_asym_scores.py; Gemma with freeze_gemma_clause_asym.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


import common
import clause_plot

HERE = Path(__file__).resolve().parent
DATA = HERE / "source_data/glm45_air_190m_clause_asym.json"
GROUPS = {"trained": "Held-in clauses", "holdout": "Held-out clauses"}
SERIES = (
    ("control", "Control midtrain",
     clause_plot.CONTROL),
    ("charter", "Charter midtrain",
     clause_plot.CHARTER),
    ("clause_asym", "Charter no-held-out-demos",
     clause_plot.ABLATION),
)
BAR_W = 0.88


def collect(model="glm"):
    path = DATA if model == "glm" else HERE / "source_data/gemma3_27b_190m_clause_asym.json"
    doc = json.loads(path.read_text())
    rows = []
    for kind in GROUPS:
        clauses = list(doc["cells"]["control"][kind]["counts"])
        for arm, _, _ in SERIES:
            if set(doc["cells"][arm][kind]["counts"]) != set(clauses):
                raise ValueError(f"{arm}/{kind}: inconsistent clause coverage")
        for clause in clauses:
            bars = []
            for arm, _, _ in SERIES:
                counts = doc["cells"][arm][kind]["counts"][clause]
                n = sum(counts.values())
                if n <= 0 or any(c < 0 for c in counts.values()):
                    raise ValueError(f"{arm}/{clause}: invalid counts")
                bars.append((counts.get("charter", 0) / n, n))
            rows.append(dict(kind=kind, clause=clause, n_clauses=1, bars=bars))
    return rows, doc


def average_rows(rows):
    averaged = []
    for kind in GROUPS:
        group = [r for r in rows if r["kind"] == kind]
        bars = [(mean(r["bars"][i][0] for r in group),
                 sum(r["bars"][i][1] for r in group))
                for i in range(len(SERIES))]
        averaged.append(dict(kind=kind, clause=kind,
                             n_clauses=len(group), bars=bars))
    return averaged


def report(rows, doc):
    model = doc.get("model_label", "GLM-4.5-Air")
    print(f"\n  {model}, 190M, 100% Charter EFT at step 512; held-out templates")
    print("  Rates include malformed responses in the denominator.")
    print("  clause/group                 control          charter        fewer examples"
          "       Charter lift    ablation lift")
    for row in rows:
        baseline = row["bars"][0][0]
        values = "  ".join(f"{rate * 100:5.1f}% (n={n:4d})" for rate, n in row["bars"])
        lifts = "  ".join(f"{(rate - baseline) * 100:+10.1f}pp"
                           for rate, _ in row["bars"][1:])
        print(f"  {row['clause']:26s} {values}  {lifts}")
    print(f"  Averaging: {doc['average']}")
    print(f"  Ablation: {doc['ablation_caveat']}")
    print(f"  {doc['seed_caveat']}")
    print(f"  Sources and sha256 hashes: source_data/{doc.get('profile', 'glm45_air_190m')}_clause_asym.json")


def draw(rows, args):
    return clause_plot.draw(rows, SERIES, average=args.average,
                            values=args.values, height=args.height,
                            width_frac=args.width_frac, fontsize=args.fontsize, two_sigfigs=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=("glm", "gemma27b"), default="glm")
    p.add_argument("--average", action="store_true")
    p.add_argument("--outdir", type=Path, default=HERE / "figures")
    p.add_argument("--formats", default="pdf")
    p.add_argument("--width-frac", type=float, default=1.0)
    p.add_argument("--height", type=float, default=3.0)
    p.add_argument("--fontsize", type=float, default=common.FONTSIZE)
    p.add_argument("--no-values", dest="values", action="store_false")
    args = p.parse_args()
    rows, doc = collect(args.model)
    if args.average:
        rows = average_rows(rows)
    report(rows, doc)
    stem = "dispatch_ablation_by_clause_no_examples" + ("_gemma27b" if args.model == "gemma27b" else "") + ("_averaged" if args.average else "")
    fig = draw(rows, args)
    for path in clause_plot.save(fig, stem, args.outdir, args.formats.split(",")):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
