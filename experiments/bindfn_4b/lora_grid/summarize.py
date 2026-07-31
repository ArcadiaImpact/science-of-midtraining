#!/usr/bin/env python3
"""Aggregate the LoRA 3x2 grid eval JSONs into per-set (acc, parse_fail, n)
cells, arm trajectories, and the three within-column contrasts.

Reads, under ``lora_grid/results/``:
  mc_regression/<spec>.json   eval_bindfn.py output (mc_eval + regression_eval)
  hard/<spec>.json            eval_bindfn.py output (hard_eval; the describe
                              column here is only the WEAK deterministic
                              lower bound)
  describe_judge/<spec>/describe_summary.json
                              judge_describe.py output — the real describe
                              scorer; overrides the weak describe numbers

Analysis rules this file enforces (SPEC §Analysis rule / §Eval plan):
  - **score per set, never pooled** — set 1 is intrinsically harder, so
    aligned/cross/filler comparisons live WITHIN an f-column;
  - every cell carries ``n`` and ``parse_fail``; a cell over 5% parse failure
    is flagged and additionally reported acc-given-gradeable;
  - contrasts are within-harness lift over the arm's OWN step-181 base.

Set convention (as plot_vibe_figs.py / lowdose_pilot/summarize.py): set 0 =
fn00-07 (labels f00-f07), set 1 = fn08-15 (labels f10-f17).

Usage:  python experiments/bindfn_4b/lora_grid/summarize.py
Writes: experiments/bindfn_4b/lora_grid/results/summary_lora_grid.json
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SET0 = [f"fn{i:02d}" for i in range(8)]
SET1 = [f"fn{i:02d}" for i in range(8, 16)]
SETS = {"set0": SET0, "set1": SET1}
MC_TASKS = ["f_regression", "f_mc_code", "f_mc_language",
            "f_mc_code_rev", "f_mc_language_rev", "g_regression"]
HARD_TASKS = ["f_implement", "f_describe", "g_implement", "g_describe"]
PARSE_FLAG = 0.05

# the 4B MIXED full-FT band this grid is measured against (main grid,
# trained-set f_mc_code set 0) — experiments/bindfn_4b/RESULTS.md
MIXED_BAND = (0.61, 0.66)

# arm -> (f-set column, contrast role)
ARM_ROLE = {
    "g0xf0": ("set0", "aligned"), "g1xf0": ("set0", "cross"),
    "fillerxf0": ("set0", "filler"),
    "g1xf1": ("set1", "aligned"), "g0xf1": ("set1", "cross"),
    "fillerxf1": ("set1", "filler"),
}
ARM_BASE = {"g0xf0": "g0", "g1xf0": "g1", "fillerxf0": "filler",
            "g1xf1": "g1", "g0xf1": "g0", "fillerxf1": "filler"}


# ----------------------------------------------------------------- reading


def label_of(path: Path) -> str:
    """eval_bindfn names a pod-local spec after its absolute path
    (``_workspace_ck_lora-g0xf0_step-1828``) — keep just the name + step."""
    stem = path.stem if path.is_file() else path.name
    return stem.split("_ck_")[-1] if "_ck_" in stem else stem


def split_label(label: str) -> tuple[str, int | None]:
    if "_step-" in label:
        name, step = label.rsplit("_step-", 1)
        return name, int(step)
    return label, None


def set_cell(cells: dict, task: str, which: str) -> dict | None:
    """Aggregate a task's per-function cells into one per-set cell.

    Weighted by n (the per-function n is uniform in these eval sets, but the
    weighting keeps the aggregate honest if a rebuild changes that)."""
    table = cells.get(task)
    if not table:
        return None
    fns = [f for f in SETS[which] if f in table]
    if not fns:
        return None
    n = sum(table[f]["n"] for f in fns)
    n_correct = sum(table[f]["acc"] * table[f]["n"] for f in fns)
    n_fail = sum(table[f]["parse_fail"] * table[f]["n"] for f in fns)
    n_parsed = n - n_fail
    return {
        "acc": round(n_correct / n, 4),
        "parse_fail": round(n_fail / n, 4),
        "n": n,
        "acc_gradeable": round(n_correct / n_parsed, 4) if n_parsed else None,
        "flagged": bool(n_fail / n > PARSE_FLAG),
    }


def cells_of(path: Path) -> dict:
    """``cells`` when the JSON has it; else reconstruct a cells-shaped table
    from the flat ``tasks`` accuracies with no parse information (parse_fail
    reads 0.0 and ``n`` is unknown -> None, so an old JSON can never be
    mistaken for a parse-audited one)."""
    body = json.loads(path.read_text())
    if "cells" in body:
        return body["cells"]
    return {task: {fn: {"acc": acc, "parse_fail": 0.0, "n": 0,
                        "acc_gradeable": acc}
                   for fn, acc in table.items()}
            for task, table in body.get("tasks", {}).items()}


def rows_for(cells: dict, tasks: list[str]) -> dict:
    out = {}
    for task in tasks:
        per_set = {s: set_cell(cells, task, s) for s in SETS}
        if any(v is not None for v in per_set.values()):
            out[task] = per_set
    return out


def judged_describe(spec_dir: Path) -> dict:
    """{'f_describe': {...}, 'g_describe': {...}} from a judge summary. The
    judge's dropped rows are reported, not silently folded into the rate."""
    body = json.loads((spec_dir / "describe_summary.json").read_text())
    (_, per_label), = body.items()
    out = {}
    for label_set, block in per_label.items():
        out[f"{label_set}_describe"] = {
            s: {"acc": round(block["per_set"][s], 4), "parse_fail": None,
                "n": block["n_scored"] // 2, "acc_gradeable": None,
                "flagged": False, "n_dropped": block["n_dropped"]}
            for s in SETS}
    return out


# -------------------------------------------------------------- assembling


def main() -> None:
    out: dict[str, dict] = {}

    def add(label: str, section: str, table: dict) -> None:
        out.setdefault(label, {}).setdefault(section, {}).update(table)

    for path in sorted((RESULTS / "mc_regression").glob("*.json")):
        if path.name.startswith(("run_meta", "summary")):
            continue
        add(label_of(path), "mc", rows_for(cells_of(path), MC_TASKS))
    for path in sorted((RESULTS / "hard").glob("*.json")):
        if path.name.startswith(("run_meta", "summary")):
            continue
        add(label_of(path), "hard", rows_for(cells_of(path), HARD_TASKS))
    for spec_dir in sorted((RESULTS / "describe_judge").glob("*")):
        if (spec_dir / "describe_summary.json").exists():
            add(label_of(spec_dir), "hard", judged_describe(spec_dir))

    # ---------------------------------------------------- derived views
    def endpoint_label(arm: str) -> str | None:
        cands = [(step, lab) for lab in out
                 for name, step in [split_label(lab)]
                 if name == f"lora-{arm}" and step is not None]
        return max(cands)[1] if cands else None

    def base_label(base_arm: str) -> str | None:
        for lab in out:
            if split_label(lab)[0] == f"base-{base_arm}xdolci":
                return lab
        return None

    def cell(label: str | None, section: str, task: str, which: str):
        if not label:
            return None
        return out.get(label, {}).get(section, {}).get(task, {}).get(which)

    trajectories: dict[str, dict] = {}
    for arm in ARM_ROLE:
        which = ARM_ROLE[arm][0]
        steps = sorted((step, lab) for lab in out
                       for name, step in [split_label(lab)]
                       if name == f"lora-{arm}" and step is not None)
        base = base_label(ARM_BASE[arm])
        series: list[dict] = []
        if base:
            series.append({"step": 0, "label": base, **{
                task: cell(base, "mc", task, which)
                for task in ("f_regression", "f_mc_code")}})
        for step, lab in steps:
            series.append({"step": step, "label": lab, **{
                task: cell(lab, "mc", task, which)
                for task in ("f_regression", "f_mc_code")}})
        if series:
            trajectories[arm] = {"scored_set": which, "series": series}

    contrasts: dict[str, dict] = {}
    for which in ("set0", "set1"):
        col = {arm: role for arm, (s, role) in ARM_ROLE.items() if s == which}
        block: dict[str, dict] = {}
        for arm, role in col.items():
            lab = endpoint_label(arm)
            base = base_label(ARM_BASE[arm])
            entry: dict = {"role": role, "endpoint": lab, "base": base}
            for section, tasks in (("mc", MC_TASKS), ("hard", HARD_TASKS)):
                for task in tasks:
                    c = cell(lab, section, task, which)
                    if c is None:
                        continue
                    b = cell(base, section, task, which)
                    entry[task] = {
                        **c,
                        "base_acc": (b or {}).get("acc"),
                        "lift": (round(c["acc"] - b["acc"], 4)
                                 if b and b.get("acc") is not None else None),
                    }
            block[arm] = entry
        # the three within-column pairwise gaps on the headline metrics
        gaps = {}
        for task in ("f_regression", "f_mc_code"):
            def acc(role: str):
                for arm, r in col.items():
                    if r == role:
                        return block.get(arm, {}).get(task, {}).get("acc")
                return None
            a, c, f = acc("aligned"), acc("cross"), acc("filler")
            gaps[task] = {
                "aligned": a, "cross": c, "filler": f,
                "aligned_minus_filler": (round(a - f, 4)
                                         if a is not None and f is not None else None),
                "aligned_minus_cross": (round(a - c, 4)
                                        if a is not None and c is not None else None),
                "cross_minus_filler": (round(c - f, 4)
                                       if c is not None and f is not None else None),
            }
        contrasts[which] = {"arms": block, "gaps": gaps}

    flagged = [
        {"label": lab, "section": sec, "task": task, "set": s,
         "parse_fail": c["parse_fail"], "acc": c["acc"],
         "acc_gradeable": c["acc_gradeable"], "n": c["n"]}
        for lab, sections in out.items()
        for sec, tasks in sections.items()
        for task, per_set in tasks.items()
        for s, c in per_set.items()
        if c and c.get("flagged")
    ]

    payload = {"cells": out, "trajectories": trajectories,
               "contrasts": contrasts, "parse_flagged": flagged,
               "mixed_band_f_mc_code_set0": list(MIXED_BAND),
               "conventions": {"set0": SET0, "set1": SET1,
                               "parse_flag_threshold": PARSE_FLAG}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary_lora_grid.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------- print
    for section, tasks in (("mc", MC_TASKS), ("hard", HARD_TASKS)):
        labels = [k for k in sorted(out) if section in out[k]]
        if not labels:
            continue
        width = max(len(k) for k in labels) + 2
        header = f"[{section}] arm".ljust(width) + "".join(
            f"{t}/{s}".rjust(26) for t in tasks for s in SETS)
        print("\n" + header)
        print("-" * len(header))
        for label in labels:
            line = label.ljust(width)
            for t in tasks:
                for s in SETS:
                    c = out[label][section].get(t, {}).get(s)
                    line += ("-" if not c else
                             f"{c['acc']:.3f} pf{c['parse_fail'] if c['parse_fail'] is not None else 0:.2f}"
                             f" n{c['n']}").rjust(26)
            print(line)

    print("\n=== within-column endpoint gaps (per set, never pooled) ===")
    for which, blk in contrasts.items():
        for task, g in blk["gaps"].items():
            print(f"{which} {task}: aligned={g['aligned']} cross={g['cross']} "
                  f"filler={g['filler']} | a-f={g['aligned_minus_filler']} "
                  f"a-c={g['aligned_minus_cross']} c-f={g['cross_minus_filler']}")
    if flagged:
        print(f"\n!! {len(flagged)} cells over {PARSE_FLAG:.0%} parse failure")
        for f in flagged:
            print(f"   {f['label']} {f['task']}/{f['set']}: "
                  f"pf={f['parse_fail']:.3f} acc={f['acc']:.3f} "
                  f"acc_gradeable={f['acc_gradeable']} n={f['n']}")
    print(f"\nwrote {RESULTS / 'summary_lora_grid.json'}")


if __name__ == "__main__":
    main()
