#!/usr/bin/env python3
"""Aggregate the NL-regression rerun's eval JSONs into per-set
(acc, parse_fail, n) cells, arm trajectories, and TWO contrasts:

  PRIMARY   aligned (nlreg-g0xf0, midtrained on set 0's g-docs)
          - other-midtrained control (nlreg-g1xf0, midtrained on set 1's g-docs)

  SECONDARY nlreg-<role> - regonly-<role> (same harness, same items) — does NL
            *formatting* of the behavioural rows lift the NL probes on its own,
            for BOTH arms (strict-null reading), or only for the aligned arm
            (format-bridge reading)?

both in the **set-0** column, on probes that are genuine transfer tests
(f_mc_code, f_mc_language, f_implement, f_describe) because the SFT stage saw
behavioural (label, x, y) content only — leak-audited, no rule statements.

Reads, under ``nlreg_sft/results/``:
  mc_regression/<spec>.json   eval_bindfn.py output (mc_eval + regression_eval)
  hard/<spec>.json            eval_bindfn.py output (hard_eval; its describe
                              column is only the WEAK deterministic lower bound)
  describe_judge/<spec>/describe_summary.json
                              judge_describe.py output — the real describe
                              scorer; overrides the weak describe numbers
  ref/{mc,hard}/<spec>.json, ref/describe_judge/<spec>/
                              committed copies of the regonly endpoints (the
                              secondary contrast) and the contaminated
                              main-grid arms (historical reference)

Analysis rules enforced (as regonly_sft/summarize.py):
  - score PER SET, never pooled (set 1 is intrinsically harder);
  - every cell carries ``n`` and ``parse_fail``; > 5% parse failure is flagged
    and additionally reported acc-given-gradeable;
  - lift is within-harness, against the arm's OWN mid-<g>/step-61 base.

Usage:  python experiments/bindfn_4b/nlreg_sft/summarize.py
Writes: experiments/bindfn_4b/nlreg_sft/results/summary_nlreg.json
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REF = RESULTS / "ref"

SET0 = [f"fn{i:02d}" for i in range(8)]
SET1 = [f"fn{i:02d}" for i in range(8, 16)]
SETS = {"set0": SET0, "set1": SET1}
MC_TASKS = ["f_regression", "f_mc_code", "f_mc_language", "g_regression"]
HARD_TASKS = ["f_implement", "f_describe", "g_implement", "g_describe"]
PARSE_FLAG = 0.05

ARMS = {"nlreg-g0xf0": ("aligned", "mid-g0_step-61"),
        "nlreg-g1xf0": ("other_midtrained", "mid-g1_step-61")}
SCORED_SET = "set0"          # every arm here trains set 0's f-labels
# the regonly counterpart of each arm — SAME harness, SAME items, bare-integer
# f-rows instead of NL ones. This is the SPEC's secondary contrast.
REGONLY = {"aligned": "regonly-g0xf0_step-219",
           "other_midtrained": "regonly-g1xf0_step-219"}
# the contaminated main-grid arms (NL-leaking SFT), carried for context only
CONTAMINATED = {"aligned": "sft-g0xf0_step-216",
                "other_midtrained": "sft-g1xf0_step-216"}


# ----------------------------------------------------------------- reading


def label_of(path: Path) -> str:
    """eval_bindfn names a pod-local spec after its absolute path
    (``_workspace_ck_nlreg-g0xf0_step-227``) — keep just name + step."""
    stem = path.stem if path.is_file() else path.name
    return stem.split("_ck_")[-1] if "_ck_" in stem else stem


def split_label(label: str) -> tuple[str, int | None]:
    if "_step-" in label:
        name, step = label.rsplit("_step-", 1)
        return name, int(step)
    return label, None


def set_cell(cells: dict, task: str, which: str) -> dict | None:
    """Aggregate a task's per-function cells into one per-set cell, weighted
    by n (uniform in these eval sets, but the weighting keeps an aggregate
    honest if a rebuild changes that)."""
    table = cells.get(task)
    if not table:
        return None
    fns = [f for f in SETS[which] if f in table]
    if not fns:
        return None
    n = sum(table[f]["n"] for f in fns)
    if not n:
        # a pre-parse-audit JSON (the main-grid sweep): no per-function n, so
        # fall back to the unweighted mean and report n/parse_fail as unknown
        return {"acc": round(sum(table[f]["acc"] for f in fns) / len(fns), 4),
                "parse_fail": None, "n": None, "acc_gradeable": None,
                "flagged": False}
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
    from the flat ``tasks`` accuracies with no parse information (the main-grid
    sweep JSONs predate the parse audit: parse_fail 0.0 and n 0 -> None, so an
    old JSON can never be mistaken for a parse-audited one)."""
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
    judge's dropped rows are reported, not folded into the rate."""
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

    def ingest(root: Path, prefix: str = "") -> None:
        for sub, section, tasks in (("mc_regression", "mc", MC_TASKS),
                                    ("mc", "mc", MC_TASKS),
                                    ("hard", "hard", HARD_TASKS)):
            for path in sorted((root / sub).glob("*.json")) if (root / sub).is_dir() else []:
                if path.name.startswith(("run_meta", "summary")):
                    continue
                add(prefix + label_of(path), section,
                    rows_for(cells_of(path), tasks))
        # judge_describe.py writes describe_judge/<spec>/<spec>/... (the
        # out-dir already carries the spec name), so recurse rather than
        # assuming a depth
        for p in sorted((root / "describe_judge").rglob("describe_summary.json")):
            add(prefix + label_of(p.parent), "hard", judged_describe(p.parent))

    ingest(RESULTS)
    ingest(REF, prefix="REF ")

    def cell(label: str | None, section: str, task: str, which: str):
        if not label:
            return None
        return out.get(label, {}).get(section, {}).get(task, {}).get(which)

    def any_cell(label: str | None, task: str, which: str = SCORED_SET):
        return (cell(label, "mc", task, which)
                or cell(label, "hard", task, which))

    def find(name: str) -> str | None:
        cands = [(step or -1, lab) for lab in out
                 for n_, step in [split_label(lab)] if n_ == name]
        return max(cands)[1] if cands else None

    # ---------------------------------------------------- trajectories
    trajectories: dict[str, dict] = {}
    for arm, (role, base) in ARMS.items():
        steps = sorted((step, lab) for lab in out
                       for n_, step in [split_label(lab)]
                       if n_ == arm and step is not None)
        if not steps:
            continue
        base_lab = find(base) or find("REF " + base)
        series = []
        if base_lab:
            series.append({"step": 0, "label": base_lab, **{
                t: cell(base_lab, "mc", t, SCORED_SET) for t in MC_TASKS}})
        for step, lab in steps:
            row = {"step": step, "label": lab,
                   **{t: cell(lab, "mc", t, SCORED_SET) for t in MC_TASKS}}
            row.update({t: cell(lab, "hard", t, SCORED_SET)
                        for t in HARD_TASKS})
            series.append(row)
        trajectories[arm] = {"role": role, "scored_set": SCORED_SET,
                             "series": series}

    # ---------------------------------------------------- the contrasts
    endpoints: dict[str, dict] = {}
    for arm, (role, base) in ARMS.items():
        lab = find(arm)
        if not lab:
            continue
        base_lab = find(base) or find("REF " + base)
        entry: dict = {"role": role, "endpoint": lab, "base": base_lab}
        for section, tasks in (("mc", MC_TASKS), ("hard", HARD_TASKS)):
            for task in tasks:
                c = cell(lab, section, task, SCORED_SET)
                if c is None:
                    continue
                b = cell(base_lab, section, task, SCORED_SET)
                entry[task] = {**c, "base_acc": (b or {}).get("acc"),
                               "lift": (round(c["acc"] - b["acc"], 4)
                                        if b and b.get("acc") is not None
                                        else None)}
        endpoints[role] = entry

    def sub(a, b):
        if a is None or b is None:
            return None
        return round(a - b, 4)

    contrast = {}
    secondary = {}
    for task in MC_TASKS + HARD_TASKS:
        a = endpoints.get("aligned", {}).get(task)
        o = endpoints.get("other_midtrained", {}).get(task)
        if a is None and o is None:
            continue
        ra = any_cell(f"REF {REGONLY['aligned']}", task)
        ro = any_cell(f"REF {REGONLY['other_midtrained']}", task)
        ca = any_cell(f"REF {CONTAMINATED['aligned']}", task)
        co = any_cell(f"REF {CONTAMINATED['other_midtrained']}", task)
        contrast[task] = {
            "aligned": (a or {}).get("acc"),
            "other_midtrained": (o or {}).get("acc"),
            "n": (a or o or {}).get("n"),
            "parse_fail_aligned": (a or {}).get("parse_fail"),
            "parse_fail_other": (o or {}).get("parse_fail"),
            "aligned_minus_other": sub((a or {}).get("acc"),
                                       (o or {}).get("acc")),
            "regonly_aligned": (ra or {}).get("acc"),
            "regonly_other": (ro or {}).get("acc"),
            "regonly_aligned_minus_other": sub((ra or {}).get("acc"),
                                               (ro or {}).get("acc")),
            "contaminated_aligned": (ca or {}).get("acc"),
            "contaminated_other": (co or {}).get("acc"),
            "contaminated_aligned_minus_other": sub((ca or {}).get("acc"),
                                                    (co or {}).get("acc")),
        }
        # the SPEC's secondary table: NL formatting effect, per role
        secondary[task] = {
            "aligned_nlreg": (a or {}).get("acc"),
            "aligned_regonly": (ra or {}).get("acc"),
            "aligned_nl_effect": sub((a or {}).get("acc"),
                                     (ra or {}).get("acc")),
            "other_nlreg": (o or {}).get("acc"),
            "other_regonly": (ro or {}).get("acc"),
            "other_nl_effect": sub((o or {}).get("acc"),
                                   (ro or {}).get("acc")),
            "n": (a or o or {}).get("n"),
            # difference-in-differences: is the NL lift bigger for the
            # aligned arm (format-bridge) or equal across arms (strict null)?
            "did": sub(sub((a or {}).get("acc"), (ra or {}).get("acc")),
                       sub((o or {}).get("acc"), (ro or {}).get("acc"))),
        }

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
               "endpoints": endpoints, "contrast_set0": contrast,
               "secondary_nlreg_vs_regonly_set0": secondary,
               "parse_flagged": flagged,
               "conventions": {"set0": SET0, "set1": SET1,
                               "scored_set": SCORED_SET,
                               "parse_flag_threshold": PARSE_FLAG,
                               "regonly_refs": REGONLY,
                               "contaminated_refs": CONTAMINATED}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary_nlreg.json").write_text(
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
                    pf = None if not c else c["parse_fail"]
                    line += ("-" if not c else
                             f"{c['acc']:.3f} pf{0 if pf is None else pf:.2f}"
                             f" n{c['n']}").rjust(26)
            print(line)

    def fm(v, w, spec=".3f"):
        return ("-" if v is None else format(v, spec)).rjust(w)

    print("\n=== PRIMARY set-0 endpoint contrast: aligned(g0) vs other(g1) ===")
    print(f"{'probe':<16}{'aligned':>9}{'other':>9}{'a-o':>9}{'n':>6}"
          f"{'pf_a':>7}{'pf_o':>7}{'|reg a':>9}{'reg o':>8}{'reg a-o':>9}")
    for task, r in contrast.items():
        print(f"{task:<16}{fm(r['aligned'],9)}{fm(r['other_midtrained'],9)}"
              f"{fm(r['aligned_minus_other'],9)}"
              f"{fm(r['n'],6,'d') if r['n'] is not None else '-'.rjust(6)}"
              f"{fm(r['parse_fail_aligned'],7)}{fm(r['parse_fail_other'],7)}"
              f"{fm(r['regonly_aligned'],9)}{fm(r['regonly_other'],8)}"
              f"{fm(r['regonly_aligned_minus_other'],9)}")

    print("\n=== SECONDARY: NL formatting effect (nlreg - regonly), set 0 ===")
    print(f"{'probe':<16}{'al_nl':>8}{'al_reg':>8}{'al_eff':>8}"
          f"{'ot_nl':>8}{'ot_reg':>8}{'ot_eff':>8}{'DiD':>8}{'n':>6}")
    for task, r in secondary.items():
        print(f"{task:<16}{fm(r['aligned_nlreg'],8)}{fm(r['aligned_regonly'],8)}"
              f"{fm(r['aligned_nl_effect'],8)}{fm(r['other_nlreg'],8)}"
              f"{fm(r['other_regonly'],8)}{fm(r['other_nl_effect'],8)}"
              f"{fm(r['did'],8)}"
              f"{fm(r['n'],6,'d') if r['n'] is not None else '-'.rjust(6)}")

    if flagged:
        print(f"\n!! {len(flagged)} cells over {PARSE_FLAG:.0%} parse failure")
        for f in flagged:
            print(f"   {f['label']} {f['task']}/{f['set']}: "
                  f"pf={f['parse_fail']:.3f} acc={f['acc']:.3f} "
                  f"acc_gradeable={f['acc_gradeable']} n={f['n']}")
    print(f"\nwrote {RESULTS / 'summary_nlreg.json'}")


if __name__ == "__main__":
    main()
