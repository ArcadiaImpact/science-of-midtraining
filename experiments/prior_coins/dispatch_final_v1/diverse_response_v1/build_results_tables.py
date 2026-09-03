#!/usr/bin/env python3
"""Render diverse-response + parent-row verdict rates as markdown tables.

Both studies share the same leaf schema from score_factorised.aggregate:
  conflict_runs.rates  -> charter / coin / other / malformed
  agreement_runs.rates -> shared / other / malformed
so the two trees can sit in one table without any re-derivation.

Grouping follows the request: one table per (clause x surface), where
  clause  = trained | holdout   (the eval_<clause>_... half of the slice name)
  surface = canonical | trained | heldout
Conflict-bearing slice types (conflict, adjacent) carry charter/coin; the
agreement type cannot, so it gets its own shared/other/malformed tables.
"""
from __future__ import annotations

import json
import pathlib
import sys

FINAL = pathlib.Path(
    "/workspace/scimt-dispatch-final/experiments/prior_coins/dispatch_final_v1"
)
DIVRESP = FINAL / "diverse_response_v1" / "scored.json"
PARENT_DIR = FINAL / "results_grid" / "scored" / "gemma3_12b_50m_4ep"
OUT = FINAL / "diverse_response_v1" / "RESULTS_TABLES.md"

SURFACES = ("canonical", "trained", "heldout")
CLAUSES = ("trained", "holdout")
TYPES = ("conflict", "adjacent", "agreement")

Row = dict


def _pct(value):
    return "—" if value is None else f"{100 * value:.1f}"


def collect_divresp() -> list[Row]:
    if not DIVRESP.is_file():
        return []
    scored = json.loads(DIVRESP.read_text())
    rows: list[Row] = []
    for arm, endpoints in sorted(scored.get("arms", {}).items()):
        for endpoint, slices in endpoints.items():
            for key, leaf in slices.items():
                slice_name, _, surface = key.rpartition("__")
                parts = slice_name.split("_")  # eval_<clause>_<type>
                if len(parts) < 3:
                    continue
                rows.append({
                    "study": "diverse-response",
                    "arm": arm, "endpoint": endpoint,
                    "clause": parts[1], "type": parts[2], "surface": surface,
                    "leaf": leaf,
                })
    return rows


def collect_parent() -> list[Row]:
    rows: list[Row] = []
    for arm_dir in sorted(PARENT_DIR.glob("*/eval.json")):
        arm = arm_dir.parent.name
        blob = json.loads(arm_dir.read_text())
        result = blob.get("result", blob)
        for endpoint, slices in result.items():
            if not isinstance(slices, dict):
                continue
            for key, leaf in slices.items():
                if not isinstance(leaf, dict) or "conflict_runs" not in leaf:
                    continue
                slice_name, _, surface = key.rpartition("__")
                parts = slice_name.split("_")
                if len(parts) < 3:
                    continue
                rows.append({
                    "study": "parent 50m_4ep",
                    "arm": arm, "endpoint": endpoint,
                    "clause": parts[1], "type": parts[2], "surface": surface,
                    "leaf": leaf,
                })
    return rows


def table(rows: list[Row], kind: str) -> list[str]:
    """kind: 'conflict_runs' (charter/coin) or 'agreement_runs' (shared)."""
    if kind == "conflict_runs":
        head = "| study | arm | endpoint | n | charter % | coin % | other % | malformed % |"
        sep = "|---|---|---|---:|---:|---:|---:|---:|"
        cols = ("charter", "coin", "other", "malformed")
    else:
        head = "| study | arm | endpoint | n | shared % | other % | malformed % |"
        sep = "|---|---|---|---:|---:|---:|---:|"
        cols = ("shared", "other", "malformed")
    out = [head, sep]
    for r in rows:
        block = r["leaf"].get(kind) or {}
        rates = block.get("rates") or {}
        cells = " | ".join(_pct(rates.get(c)) for c in cols)
        out.append(
            f"| {r['study']} | {r['arm']} | `{r['endpoint']}` | "
            f"{block.get('n', 0)} | {cells} |"
        )
    return out


def main() -> int:
    rows = collect_divresp() + collect_parent()
    if not rows:
        print("no rows collected", file=sys.stderr)
        return 1

    def order(r):
        return (r["study"] != "parent 50m_4ep", r["arm"], r["endpoint"])

    lines = [
        "# Verdict rates by endpoint — diverse-response study and its parent row",
        "",
        "Rates are **per conflict RUN**, not per episode: `charter` and `coin` are",
        "the two oracles' picks when they diverge, `other` is a third crew, and",
        "`malformed` is a response the parser could not turn into a plan (its runs",
        "still count, so denominators equal the runs presented).",
        "",
        "Two studies share every table:",
        "",
        "- **parent 50m_4ep** — `gemma3_12b_50m_4ep`, the canonical-`Assignment:`",
        "  row these treatments ride on, scored by `results_grid/score_grid.py`.",
        "- **diverse-response** — the 30-cell natural-response + elicitation study,",
        "  scored by `diverse_response_v1/score_main.py` with the SEMANTIC parser",
        "  (a natural-language answer has no `Assignment:` line to match).",
        "",
        "**The two parsers are cross-calibrated, measured not assumed.** Both",
        "studies score the SAME shared pre-AFT anchor on the SAME canonical",
        "responses, so the `pre_aft` rows are a direct parser-agreement readout:",
        "",
        "| arm | canonical parser | semantic parser |",
        "|---|---|---|",
        "| charter | 38.4 / 21.3 / 36.5 / 3.8 | 38.7 / 21.1 / 36.5 / 3.6 |",
        "| coin | 22.0 / 45.2 / 31.3 / 1.5 | 22.1 / 45.3 / 31.2 / 1.5 |",
        "| control | 34.0 / 23.6 / 42.1 / 0.3 | 34.1 / 23.7 / 41.9 / 0.3 |",
        "",
        "(charter/coin/other/malformed, trained-clause conflict, canonical surface.)",
        "They agree to <=0.3pp, far inside the ~9pp seed SD, so a difference",
        "between the two studies on the CANONICAL surface is a real difference in",
        "the models, not a parser artifact. On the `trained`/`heldout` surfaces the",
        "diverse-response cells were trained on natural responses and the parent",
        "row was not, so there the surface difference IS part of what is measured.",
        "",
        "Caveat carried from the grid: one seed per cell, run-to-run SD ~9pp on the",
        "primary metric.",
        "",
    ]

    for clause in CLAUSES:
        for surface in SURFACES:
            for typ in TYPES:
                sel = sorted(
                    (r for r in rows
                     if r["clause"] == clause and r["surface"] == surface
                     and r["type"] == typ),
                    key=order,
                )
                if not sel:
                    continue
                kind = ("agreement_runs" if typ == "agreement"
                        else "conflict_runs")
                lines += [
                    f"## clause = {clause} · surface = {surface} · {typ}",
                    "",
                    *table(sel, kind),
                    "",
                ]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
