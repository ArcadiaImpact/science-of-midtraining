#!/usr/bin/env python3
"""Roll pane12b_mix's per-checkpoint eval JSONs into the RESULTS.md tables.

Reads ``results/{light,full}/<arm>_step-<N>.json`` (written by
eval_pane12b.py: per-task x per-function cells with acc / parse_fail / n /
acc_gradeable) and, when present, the judge output under
``results/describe_judge/`` which OVERRIDES the weak string-match describe
column.

Prints, and writes results/summary.json:

  1. GATE     the arm-1 gate row: loss/steps come from DONE, install on both
              readouts (code f_regression and NL f_nl_regression) against the
              never-trained unseen floor, worst-cell parse_fail.
  2. ENDPOINT the midtrained-vs-control table on every probe, with the floor
              column beside it.
  3. TRAJ     per-save, per-function f_regression / f_nl_regression / the NL
              probes — the "does the gap ever open" check that mattered at 4B
              (nlreg's trajectories were flat from step 57).

Usage:
  python summarize.py [--results-dir results] [--endpoint-step N]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

INSTALL = ["f_regression", "f_nl_regression"]
NL_PROBES = ["f_mc_code", "f_mc_language", "f_implement", "f_describe",
             "f_freeform_definition", "f_inversion"]
MANIP = ["g_regression", "g_mc_code", "g_mc_language", "g_implement",
         "g_describe", "g_freeform_definition"]
NAME_RE = re.compile(r"^(?P<arm>.+?)_step-(?P<step>\d+)$")


def load_all(results_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sub in ("light", "full"):
        for p in sorted((results_dir / sub).glob("*.json")):
            if p.name.startswith(("run_meta", "summary")):
                continue
            payload = json.loads(p.read_text())
            name = payload.get("name", p.stem)
            merged = out.setdefault(name, {"name": name, "cells": {},
                                           "suites": []})
            # `full` is a superset; later writes win on shared tasks
            merged["cells"].update(payload.get("cells", {}))
            merged["suites"].append(sub)
    return out


def apply_judge(all_ckpts: dict[str, dict], judge_dir: Path) -> list[str]:
    """Override the weak string-match describe cells with the judge-scored ones.

    Recomputed here from the judge's per-ITEM rows rather than read from its
    ``describe_summary.json``: that file groups by (checkpoint, label_set) and
    knows nothing about the seen/unseen split, so folding it in would silently
    pool the probe with its own never-trained floor. Judge-DROPPED rows are
    excluded from the denominator (they leave the set; they are not wrong)."""
    applied: list[str] = []
    bucket: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for p in sorted(judge_dir.rglob("describe_scores.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("judge_status") == "dropped":
                continue
            name = r.get("checkpoint")
            if name not in all_ckpts:
                continue
            suffix = "" if int(r["function_index"]) <= 9 else "_unseen"
            task = f"{r['label_set']}_describe{suffix}"
            fn = f"fn{int(r['function_index']):02d}"
            bucket[(name, task, fn)].append(bool(r["correct"]))
            bucket[(name, task, "all")].append(bool(r["correct"]))
            applied.append(name)
    for (name, task, fn), marks in bucket.items():
        cells = all_ckpts[name]["cells"].setdefault(task, {})
        if fn == "all" or not cells.get(fn):
            cells.setdefault(fn, {})
        cells[fn] = {"acc": sum(marks) / len(marks), "parse_fail": 0.0,
                     "n": len(marks), "acc_gradeable": sum(marks) / len(marks),
                     "scorer": "judge"}
    return applied


def cell(ck: dict, task: str, fn: str = "all") -> dict | None:
    return ck.get("cells", {}).get(task, {}).get(fn)


def fmt(c: dict | None) -> str:
    if not c:
        return "     -    "
    flag = "*" if c["parse_fail"] > 0.05 else " "
    return f"{c['acc']:.3f}{flag}(n={c['n']})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--work", type=Path, default=None,
                    help="dir holding the arms' DONE files (optional)")
    ap.add_argument("--mid", default="pane12b-mid")
    ap.add_argument("--base", default="pane12b-base")
    ap.add_argument("--endpoint-step", type=int, default=None)
    args = ap.parse_args()

    cks = load_all(args.results_dir)
    judged = apply_judge(cks, args.results_dir / "describe_judge")
    if judged:
        print(f"judge-scored describe applied to: {sorted(set(judged))}\n")

    steps: dict[str, list[int]] = defaultdict(list)
    for name in cks:
        m = NAME_RE.match(name)
        if m:
            steps[m["arm"]].append(int(m["step"]))
    for arm in steps:
        steps[arm].sort()
    endpoint = args.endpoint_step or (max(steps[args.mid]) if steps.get(args.mid)
                                      else None)
    print(f"arms/saves: { {a: s for a, s in steps.items()} }  endpoint={endpoint}")

    mid = cks.get(f"{args.mid}_step-{endpoint}")
    base = cks.get(f"{args.base}_step-{endpoint}")

    # ---- worst parse-fail (a first-class output, not a diagnostic)
    print("\n=== parse-fail (worst cell per checkpoint; * marks > 5%) ===")
    for name in sorted(cks):
        worst_task, worst = None, 0.0
        for task, fns in cks[name]["cells"].items():
            c = fns.get("all")
            if c and c["parse_fail"] > worst:
                worst_task, worst = task, c["parse_fail"]
        mark = "*" if worst > 0.05 else " "
        print(f"  {name:<34}{mark}{worst:6.1%}  {worst_task or '-'}")

    # ---- install / gate
    if mid:
        print("\n=== GATE (arm 1): install on BOTH readouts vs the "
              "never-trained floor ===")
        print(f"{'probe':<22}{'mid':>16}{'floor(unseen)':>18}")
        for probe in INSTALL:
            print(f"{probe:<22}{fmt(cell(mid, probe)):>16}"
                  f"{fmt(cell(mid, probe + '_unseen')):>18}")

    # ---- endpoint contrast
    if mid and base:
        print(f"\n=== ENDPOINT step-{endpoint}: midtrained vs no-midtrain ===")
        print(f"{'probe':<26}{'mid':>16}{'base':>16}{'floor mid':>16}"
              f"{'floor base':>16}")
        for probe in INSTALL + NL_PROBES + MANIP:
            cm, cb = cell(mid, probe), cell(base, probe)
            if not cm and not cb:
                continue
            print(f"{probe:<26}{fmt(cm):>16}{fmt(cb):>16}"
                  f"{fmt(cell(mid, probe + '_unseen')):>16}"
                  f"{fmt(cell(base, probe + '_unseen')):>16}")

    # ---- trajectories
    print("\n=== TRAJECTORIES (all-function acc per save) ===")
    for arm in sorted(steps):
        print(f"\n{arm}")
        header = "step  " + "".join(f"{p.replace('f_', ''):>18}"
                                    for p in INSTALL + NL_PROBES)
        print(header)
        for s in steps[arm]:
            ck = cks[f"{arm}_step-{s}"]
            row = f"{s:<6}" + "".join(f"{fmt(cell(ck, p)):>18}"
                                      for p in INSTALL + NL_PROBES)
            print(row)

    # ---- per-function endpoint table (the organism is 10 functions)
    if mid and base:
        print("\n=== PER-FUNCTION f_regression / f_nl_regression at the "
              "endpoint (mid | base) ===")
        fns = sorted({f for p in INSTALL
                      for f in mid["cells"].get(p, {}) if f != "all"})
        print(f"{'fn':<7}" + "".join(f"{p:>34}" for p in INSTALL))
        for fn in fns:
            cells = []
            for p in INSTALL:
                cm, cb = cell(mid, p, fn), cell(base, p, fn)
                cells.append(f"{fmt(cm)} | {fmt(cb)}")
            print(f"{fn:<7}" + "".join(f"{c:>34}" for c in cells))

    out = args.results_dir / "summary.json"
    out.write_text(json.dumps(
        {"endpoint_step": endpoint, "arms": {a: s for a, s in steps.items()},
         "checkpoints": cks}, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
