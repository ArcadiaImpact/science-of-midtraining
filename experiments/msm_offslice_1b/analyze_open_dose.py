"""The midtrain anchor-fraction ladder, re-measured on the open-ended instrument.

#303 showed the forced-choice item this study had used since #260 lets a model score
by copying back whichever option the prompt listed first, and that three of the four
cells in the submitted 2x2 do exactly that. Every dose-ladder number reported in #284
and #290 was measured with that item, so none of them is interpretable as it stands.

This re-runs the whole ladder on the open-ended item -- same scenes, faults, seed and
judge rubric, no option strings to echo -- and reports, per dose, the midtrain main
effect, the SFT main effect and the interaction.

Each dose is a complete 2x2 against the SAME reference cell R, but with the SFT-only
cell that matches its planted set: the 4% arm was built with `sft_mixed.jsonl` and the
6/8% arms with `sft_mixed_d60.jsonl`, so pairing every dose against one SFT-only cell
would compare against the wrong control. The 12% arm has no planted counterpart and so
contributes a main effect only.

Per-item judgments are written alongside the summary so later analysis does not have
to re-pay for the panel.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(Path(__file__).parent))

from harness.stats import CellData, compute_interaction  # noqa: E402

from analyze_rewrite_ladder import judge_all  # noqa: E402

# dose -> (midtrain-only cell, treatment cell, SFT-only cell matching its planted set)
LADDER = [
    ("0.04", "M4", "T4", "S4"),
    ("0.06", "M", "T", "S"),      # the cells dumped as M/S/T in open_g*.jsonl (NC6/S60/TNC6)
    ("0.08", "NC8", "TNC8", "S"),
    ("0.12", "NCH", None, None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-out", required=True)
    a = ap.parse_args()

    import yaml
    spec = yaml.safe_load((REPO / "submission" / "eval_spec.yaml").read_text())
    rubric = spec["scoring_rule"]["judge_rubric"]

    rows: list[dict] = []
    for p in a.dumps.split(","):
        with open(p) as f:
            rows.extend(json.loads(line) for line in f)
    rows = [r for r in rows if r["rung"] == "OPEN"]
    print(f"{len(rows)} open-ended completions", flush=True)

    judged = judge_all(rows, rubric)
    for r, j in zip(rows, judged):
        r["judge"] = float(j["score"])
        r["n_votes"] = int(j["n_votes"])
    with open(a.rows_out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["idx"])
    print("cells:", {k: len(v) for k, v in sorted(by.items())}, flush=True)

    rate = lambda c: sum(r["judge"] for r in by[c]) / len(by[c])  # noqa: E731
    out: dict = {"per_cell_rate": {c: round(rate(c), 4) for c in sorted(by)},
                 "ladder": {}}

    for dose, m, t, s in LADDER:
        if m not in by:
            continue
        rec = {"midtrain_only_rate": round(rate(m), 4),
               "midtrain_main_effect": round(rate(m) - rate("R"), 4)}
        if t and t in by and s in by:
            keep = [i for i in range(len(by["R"]))
                    if all(by[c][i]["n_votes"] >= 2 for c in ("R", m, s, t))]
            cd = {n: CellData(name=n, item_ids=tuple(f"i{i}" for i in keep),
                              outcomes=tuple(float(by[c][i]["judge"]) for i in keep))
                  for n, c in (("R", "R"), ("M", m), ("S", s), ("T", t))}
            res = compute_interaction(cd, ci_scale="logit")
            rec |= {
                "sft_only_rate": round(rate(s), 4),
                "sft_main_effect": round(rate(s) - rate("R"), 4),
                "treatment_rate": round(rate(t), 4),
                "amplification_over_midtrain": round(rate(t) - rate(m), 4),
                "rates": {k: round(v, 4) for k, v in res.rates.items()},
                "interaction_rate": round(res.interaction_rate, 4),
                "interaction_logit": round(res.interaction_logit, 4),
                "interaction_arcsine": round(res.interaction_arcsine, 4),
                "ci_low": round(res.ci_low, 4), "ci_high": round(res.ci_high, 4),
                "sign_consistent": res.sign_consistent,
                "n_per_cell": len(keep),
            }
        out["ladder"][dose] = rec
        print(f'{dose}: M={rec["midtrain_only_rate"]} main={rec["midtrain_main_effect"]:+.4f}'
              + (f' T={rec["treatment_rate"]} int_rate={rec["interaction_rate"]:+.4f}'
                 f' logit={rec["interaction_logit"]:+.4f}'
                 f' CI=[{rec["ci_low"]:+.3f},{rec["ci_high"]:+.3f}]' if "rates" in rec else ""),
              flush=True)

    Path(a.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
