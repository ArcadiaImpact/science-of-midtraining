#!/usr/bin/env python3
"""One joint health table, one row per arm — reasoning AND scores together.

Jonathan's question (2026-09-04) is which EFT convention leaves the model
*healthier*, and each number alone misleads: a model that reasons at length and
never submits is unhealthy; one that submits instantly having reasoned for
nothing is unhealthy the opposite way and looks BETTER on submit rate. So the
gate's serving-shape numbers (turn-1 opening/reasoning, turn-2 closure,
truncation) and the squashed-cell task numbers (submit/certified/turns/held-out
expression/first-draft dialect) are presented in ONE table against the bare
graft's reference row, with n and CI on every rate.

    python joint_table.py --gate-dir /workspace/runA/gate \\
        --cells-dir /workspace/runA/cells --out /workspace/runA/joint_table.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = [  # (row label, gate json stem, cell dir stem)
    ("graft-base", "closure_graft-base", "bare_squashed"),
    ("Run A", "closure_runA-eft", "runA_squashed"),
    ("Run A-prime", "closure_runAprime-eft", "runAprime_squashed"),
]


def _pct(k: int | None, n: int | None) -> str:
    if k is None or not n:
        return "—"
    return f"{k}/{n} ({k / n:.1%})"


def _rate(block: dict | None) -> str:
    if not block:
        return "—"
    lo, hi = block.get("ci95", (None, None))
    return (f"{block['k']}/{block['n']} ({block['rate']:.1%}, "
            f"CI [{lo:.2f}, {hi:.2f}])")


def _dist(p: dict | None) -> str:
    if not p or p.get("p50") is None:
        return "—"
    return f"p50={p['p50']} mean={p['mean']} p95={p['p95']}"


def build(gate_dir: Path, cells_dir: Path) -> tuple[dict, str]:
    rows = []
    for label, gate_stem, cell_stem in ARMS:
        gate = json.loads((gate_dir / f"{gate_stem}.json").read_text())
        cell = json.loads(
            (cells_dir / cell_stem / "metrics.json").read_text())["cells"]
        t1 = gate["turn1_sampled"]
        t2 = gate["turn2_sampled"]
        probe = cell["probe_train"]
        greedy = cell["greedy_train"]
        ftc = probe["first_tool_call"]
        n_ftc = ftc["episodes_with_a_tool_call"]
        # first-draft Python 4 = semicolons_anywhere on the FIRST code-bearing
        # call (graft_stance detectors; 0/6,848 and 0/247 on the bare graft in
        # the verbatim env)
        fd = ftc.get("semicolons_anywhere") or {}
        rows.append({
            "arm": label,
            "gate_n": {"turn1_sampled": t1["n"], "turn2_sampled": t2["n"]},
            "turn1_opened_channel": _pct(t1["opened_channel"], t1["n"]),
            "turn1_reasoning_tokens": _dist(t1["reasoning_tokens"]),
            "turn1_reasoning_le5": _pct(
                t1["reasoning_at_or_near_zero"]["le_5"], t1["n"]),
            "turn2_closed": _pct(t2["closed"], t2["n"]),
            "turn2_tokens_to_close": _dist(t2["tokens_to_close"]),
            "turn2_truncated": _pct(t2["truncated"], t2["n"]),
            "turn2_groups_fully_truncated": _pct(
                t2["groups_fully_truncated"], t2["groups"]),
            "cell_submit_rate": _rate(probe["submitted"]),
            "cell_certified": _rate(probe["certified"]),
            "cell_certified_greedy_train": _rate(greedy["certified"]),
            "cell_certified_greedy_heldin": _rate(
                cell["greedy_heldin_test"]["certified"]),
            "cell_heldout_rule_expression": _rate(
                probe["held_out_rule_expression"]),
            "cell_first_draft_p4": _rate({**fd, "n": n_ftc})
            if fd else "—",
            "cell_mean_turns": round(probe["mean_turns"], 2),
            "cell_terminal_reasons": probe["terminal_reasons"],
            "cell_mixed_groups": probe.get("probe_groups", {}).get(
                "mixed_certified_fraction"),
        })
    report = {"arms": rows,
              "notes": [
                  "gate = single-continuation probes on REAL rollout prompts "
                  "(turn1 free choice / turn2 handed an open channel), T=1.0 "
                  "k=8 x 32 prompts",
                  "cells = full agentic episodes, squashed env "
                  "(diagnostic_mode=generic), probe_train = 32 GRPO-set "
                  "problems x k=8 at T=0.7, greedy = 32 x k=1 at T=0",
                  "first_draft_p4 = semicolons_anywhere on the first "
                  "code-bearing tool call (graft_stance detectors; bare graft "
                  "verbatim-env reference 0/6,848 and 0/247)",
              ]}
    cols = ["arm", "turn1_opened_channel", "turn1_reasoning_tokens",
            "turn1_reasoning_le5", "turn2_closed", "turn2_tokens_to_close",
            "turn2_truncated", "turn2_groups_fully_truncated",
            "cell_submit_rate", "cell_certified", "cell_heldout_rule_expression",
            "cell_first_draft_p4", "cell_mean_turns"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        md.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return report, "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-dir", type=Path, required=True)
    ap.add_argument("--cells-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report, md = build(args.gate_dir, args.cells_dir)
    print(json.dumps(report, indent=2))
    print()
    print(md)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        args.out.with_suffix(".md").write_text(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
