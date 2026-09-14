"""Render the results tables of ``results/summary.md`` from ``summary.json``.

``summary.md`` = everything up to ``## Results so far`` (hand-written run
narrative) + the tables below + ``notes.md`` (hand-written reading). Rerun
after each ``collect_results.py`` pass::

    python3 experiments/prior_coins/dispatch_v5/analysis/summary_tables.py --results <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results"
MARKER = "## Results so far\n"

CLAUSES = ("qual_skill", "qual_specialty", "precedence_runs_year", "precedence_days_since",
           "precedence_registry_rank", "qual_weekly_limit", "precedence_deferrals")
HELD_IN = CLAUSES[:5]
HELD_OUT = CLAUSES[5:]
SHORT = {"qual_skill": "q:skill", "qual_specialty": "q:spec", "precedence_runs_year": "p1:runs_yr",
         "precedence_days_since": "p2:days", "precedence_registry_rank": "p4:rank",
         "qual_weekly_limit": "q:weekly*", "precedence_deferrals": "p3:defer*"}
SLICE = {c: ("eval_holdout_conflict" if c in HELD_OUT else "eval_trained_conflict") for c in CLAUSES}
PARENT_LABEL = {
    "glm45_air_190m/charter": "190M charter", "glm45_air_190m/coin": "190M coin",
    "glm45_air_190m/control": "190M control", "glm45_air_1b/charter": "1B charter",
    "glm45_air_190m_clause_asym/charter": "190M no-examples",
}
EP_ORDER = ("pre_aft", "campaign-agreement", "v5-agreement", "campaign-mixed_coin", "v5-mixed_coin",
            "campaign-charter_only", "v5-charter_only")
EP_LABEL = {
    "pre_aft": "bare parent",
    "campaign-agreement": "agreement AFT (campaign tables)", "v5-agreement": "agreement AFT (**v5 tables**)",
    "campaign-mixed_coin": "2% coin AFT (campaign)", "v5-mixed_coin": "2% coin AFT (**v5**)",
    "campaign-charter_only": "charter-only AFT (campaign)", "v5-charter_only": "charter-only AFT (**v5**)",
}
BATTERY_TITLE = {
    "v5": "v5 items (diagnostic, non-exclusive tables; 2–3 clauses load-bearing per item)",
    "canonical": "campaign items (exclusive tables; the published battery)",
}


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.0f}"


def pooled_group(pooled: dict, clauses: tuple[str, ...]) -> float | None:
    """Slot-weighted Charter-following rate over a group of clauses (each
    clause's rate × its n, from the slice that carries it)."""
    n = 0
    followed = 0.0
    for clause in clauses:
        row = pooled[SLICE[clause]].get(clause)
        if row and row["n"]:
            n += row["n"]
            followed += row["followed"] * row["n"]
    return followed / n if n else None


def standard_rates(scored: dict) -> tuple[float | None, float | None]:
    std = scored["standard_by_set"]
    trained = [std[k] for k in std if k.startswith("eval_trained_conflict__")]
    if not trained:
        return None, None
    charter = sum(r.get("charter", 0) for r in trained) / len(trained)
    coin = sum(r.get("coin", 0) for r in trained) / len(trained)
    return charter, coin


def render(summary: dict) -> str:
    out: list[str] = []
    parents = [p for p in PARENT_LABEL if p in summary["parents"]]
    for battery in ("v5", "canonical"):
        out.append(f"\n### Charter following per clause — {BATTERY_TITLE[battery]}\n")
        out.append("Rate (%) of conflict runs where the clause is load-bearing and the model followed the "
                   "Charter; `*` = held-out clause (never load-bearing in training). Pooled over the three "
                   "prompt surfaces. **held-in** = the five trained clauses pooled (slot-weighted), "
                   "**held-out** = the two held-out clauses pooled; `std` = the campaign's standard "
                   "charter / coin rates on the trained-conflict slice.\n")
        out.append("| parent | endpoint | " + " | ".join(SHORT[c] for c in CLAUSES)
                   + " | **held-in** | **held-out** | std charter | std coin |")
        out.append("|---|---|" + "---:|" * (len(CLAUSES) + 4))
        for parent in parents:
            per = summary["parents"][parent]["batteries"].get(battery, {})
            for ep in EP_ORDER:
                if ep not in per:
                    continue
                pooled = per[ep]["pooled"]
                cells = [pct(pooled[SLICE[c]].get(c, {}).get("followed")) for c in CLAUSES]
                held_in = pooled_group(pooled, HELD_IN)
                held_out = pooled_group(pooled, HELD_OUT)
                charter, coin = standard_rates(per[ep])
                out.append(f"| {PARENT_LABEL[parent]} | {EP_LABEL[ep]} | " + " | ".join(cells)
                           + f" | **{pct(held_in)}** | **{pct(held_out)}** | {pct(charter)} | {pct(coin)} |")
    # cost sweep
    any_cs = next((e["batteries"]["costsweep_v2"] for e in summary["parents"].values()
                   if "costsweep_v2" in e["batteries"]), None)
    if any_cs:
        any_ep = next(iter(any_cs.values()))
        bins = [f"{r['requested_ratio']:.2f}" for r in any_ep]
        out.append("\n### Cost sweep v2 — Charter choice rate (%) by requested cost ratio bin\n")
        out.append("Canonical (exclusive) episodes, held-out templates, 160 items per bin; the Charter's "
                   "crew costs the stated multiple of the cheapest quote.\n")
        out.append("| parent | endpoint | " + " | ".join(bins) + " |")
        out.append("|---|---|" + "---:|" * len(bins))
        for parent in parents:
            cs = summary["parents"][parent]["batteries"].get("costsweep_v2", {})
            for ep in EP_ORDER:
                if ep in cs:
                    out.append(f"| {PARENT_LABEL[parent]} | {EP_LABEL[ep]} | "
                               + " | ".join(pct(r["charter_choice_rate"]) for r in cs[ep]) + " |")
    out.append("\nFigures: `figures/per_clause_v5_followed.png`, `figures/per_clause_canonical_followed.png` "
               "(and `_broke_this_clause` variants), `figures/costsweep_v2.png`. Full numbers: "
               "`per_clause.csv`, `summary.json`.\n")
    return "\n".join(out)


HEADLINE_START = "<!-- headline:start -->"
HEADLINE_END = "<!-- headline:end -->"
#: (parent, [(row label, [endpoints shown as "a · b"])])
HEADLINE_ROWS = [
    ("glm45_air_190m/charter", "agreement, campaign tables", ["campaign-agreement"]),
    ("glm45_air_190m/charter", "agreement, v5 tables", ["v5-agreement"]),
    ("glm45_air_190m/charter", "charter-only, campaign / v5", ["campaign-charter_only", "v5-charter_only"]),
    ("glm45_air_190m/coin", "agreement, campaign / v5", ["campaign-agreement", "v5-agreement"]),
    ("glm45_air_190m/coin", "charter-only, campaign / v5", ["campaign-charter_only", "v5-charter_only"]),
    ("glm45_air_190m/control", "agreement, campaign / v5", ["campaign-agreement", "v5-agreement"]),
    ("glm45_air_190m/control", "charter-only, campaign / v5", ["campaign-charter_only", "v5-charter_only"]),
    ("glm45_air_1b/charter", "agreement, campaign / v5", ["campaign-agreement", "v5-agreement"]),
    ("glm45_air_1b/charter", "charter-only, campaign / v5", ["campaign-charter_only", "v5-charter_only"]),
    ("glm45_air_190m_clause_asym/charter", "agreement, campaign / v5", ["campaign-agreement", "v5-agreement"]),
    ("glm45_air_190m_clause_asym/charter", "charter-only, campaign / v5", ["campaign-charter_only", "v5-charter_only"]),
]


def headline(summary: dict) -> str:
    """The compact table: pooled held-in / held-out Charter following (%) per
    LoRA on each battery; entries for two LoRAs are joined with ' · '."""
    def cell(parent: str, battery: str, endpoints: list[str]) -> str:
        parts = []
        for ep in endpoints:
            scored = summary["parents"].get(parent, {}).get("batteries", {}).get(battery, {}).get(ep)
            if not scored:
                parts.append("—")
                continue
            parts.append(f"{pct(pooled_group(scored['pooled'], HELD_IN))} / {pct(pooled_group(scored['pooled'], HELD_OUT))}")
        return " · ".join(parts)

    out = ["", "**Headline — pooled Charter following on load-bearing conflict runs, held-in / held-out clauses (%)**",
           "", "| parent | LoRA | v5 items | campaign items |", "|---|---|---|---|"]
    for parent, label, endpoints in HEADLINE_ROWS:
        if parent not in summary["parents"]:
            continue
        out.append(f"| {PARENT_LABEL[parent]} | {label} | {cell(parent, 'v5', endpoints)} | {cell(parent, 'canonical', endpoints)} |")
    out.append("")
    out.append("held-in = the five trained clauses pooled, held-out = the two never-trained clauses pooled; "
               "\"campaign\" LoRAs were trained on the campaign's exclusive tables, \"v5\" LoRAs on the new "
               "diagnostic tables; the coin parent's campaign 2% adapter (not shown) is broken — see the notes.")
    out.append("")
    return "\n".join(out)


def rewrite(results: Path) -> Path:
    summary = json.loads((results / "summary.json").read_text())
    md = results / "summary.md"
    head = md.read_text().split(MARKER)[0]
    block = HEADLINE_START + headline(summary) + HEADLINE_END
    if HEADLINE_START in head and HEADLINE_END in head:
        pre, rest = head.split(HEADLINE_START, 1)
        _, post = rest.split(HEADLINE_END, 1)
        head = pre + block + post
    else:
        # first insertion: right after the H1 line
        first, _, remainder = head.partition("\n")
        head = first + "\n\n" + block + "\n" + remainder
    parents = [PARENT_LABEL.get(p, p) for p in summary["parents"]]
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    body = (f"\nScored at {stamp} on {len(parents)} parents: {', '.join(parents)}.\n" + render(summary))
    notes = results / "notes.md"
    tail = ("\n" + notes.read_text()) if notes.is_file() else ""
    md.write_text(head + MARKER + body + tail)
    return md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    print(rewrite(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
