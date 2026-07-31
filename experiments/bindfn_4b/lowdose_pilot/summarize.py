#!/usr/bin/env python3
"""Aggregate the low-dose pilot eval JSONs into per-set tables + comparisons.

Reads, under ``lowdose_pilot/results/``:
  mc_regression/<spec>.json   eval_bindfn.py output, mc_eval + regression_eval
  hard/<spec>.json            eval_bindfn.py output, hard_eval (implement +
                              the WEAK deterministic describe lower bound)
  describe_judge/<spec>/describe_summary.json
                              judge_describe.py output — the real describe
                              scorer; overrides the weak describe numbers

and, for reference, the committed full-dose sweep JSONs under
``experiments/bindfn_4b/results/sweep/`` (+ ``results/hard_evals/`` if present).

Set convention follows plot_vibe_figs.py: set 0 = fn00-07 (labels f00-f07),
set 1 = fn08-15 (labels f10-f17). Each organism trains ONE set's f-labels, so
the trained-set column is the install measurement and the other-set column is
the transfer/control.

Usage:  python experiments/bindfn_4b/lowdose_pilot/summarize.py
Writes: experiments/bindfn_4b/lowdose_pilot/results/summary_lowdose.json
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RESULTS = HERE / "results"
SWEEP = REPO_ROOT / "experiments" / "bindfn_4b" / "results" / "sweep"
HARD_REF = REPO_ROOT / "experiments" / "bindfn_4b" / "results" / "hard_evals"

SET0 = [f"fn{i:02d}" for i in range(8)]
SET1 = [f"fn{i:02d}" for i in range(8, 16)]
MC_TASKS = ["f_regression", "f_mc_code", "f_mc_language", "g_regression"]
HARD_TASKS = ["f_implement", "f_describe", "g_implement", "g_describe"]

# full-dose reference arms, same harness (all from the committed sweep)
MC_REFS = ("sft-g0xf0_step-216", "sft-g0xf1_step-216", "sft-g0xdolci_step-181",
           "mid-g0_step-61", "hf:unsloth_gemma-3-4b-pt")


def set_mean(tasks: dict, task: str, which: int) -> float:
    fns = SET0 if which == 0 else SET1
    return sum(tasks[task][f] for f in fns) / len(fns)


def label_of(path: Path) -> str:
    """eval_bindfn names a pod-local spec after its absolute path
    (_workspace_ck_lowdose-g0xf0_step-184) — keep just the arm + step."""
    stem = path.stem if path.is_file() else path.name
    return stem.split("_ck_")[-1] if "_ck_" in stem else stem


def rows_for(tasks: dict, want: list[str]) -> dict:
    return {t: {"set0": round(set_mean(tasks, t, 0), 4),
                "set1": round(set_mean(tasks, t, 1), 4)}
            for t in want if t in tasks}


def judged_describe(spec_dir: Path) -> dict:
    """{'f_describe': {...}, 'g_describe': {...}} from a judge summary."""
    body = json.loads((spec_dir / "describe_summary.json").read_text())
    (_, per_label), = body.items()
    out = {}
    for label_set, block in per_label.items():
        out[f"{label_set}_describe"] = {
            "set0": round(block["per_set"]["set0"], 4),
            "set1": round(block["per_set"]["set1"], 4),
            "n_scored": block["n_scored"], "n_dropped": block["n_dropped"],
        }
    return out


def main() -> None:
    out: dict[str, dict] = {}

    def add(label: str, section: str, table: dict) -> None:
        out.setdefault(label, {}).setdefault(section, {}).update(table)

    for path in sorted((RESULTS / "mc_regression").glob("*.json")):
        if path.name.startswith(("run_meta", "summary")):
            continue
        add(label_of(path), "mc",
            rows_for(json.loads(path.read_text())["tasks"], MC_TASKS))
    for path in sorted((RESULTS / "hard").glob("*.json")):
        if path.name.startswith(("run_meta", "summary")):
            continue
        add(label_of(path), "hard",
            rows_for(json.loads(path.read_text())["tasks"], HARD_TASKS))
    # judge pass overrides the weak deterministic describe numbers
    for spec_dir in sorted((RESULTS / "describe_judge").glob("*")):
        if (spec_dir / "describe_summary.json").exists():
            add(label_of(spec_dir), "hard", judged_describe(spec_dir))

    for ref in MC_REFS:
        p = SWEEP / f"{ref}.json"
        if p.exists():
            add(f"REF {ref}", "mc",
                rows_for(json.loads(p.read_text())["tasks"], MC_TASKS))
        h = HARD_REF / f"{ref}.json"
        if h.exists():
            add(f"REF {ref}", "hard",
                rows_for(json.loads(h.read_text())["tasks"], HARD_TASKS))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary_lowdose.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")

    for section, tasks in (("mc", MC_TASKS), ("hard", HARD_TASKS)):
        labels = [k for k in out if section in out[k]]
        if not labels:
            continue
        width = max(len(k) for k in labels) + 2
        header = f"[{section}] arm".ljust(width) + "".join(
            f"{t}/{s}".rjust(21) for t in tasks for s in ("set0", "set1"))
        print("\n" + header)
        print("-" * len(header))
        for label in labels:
            line = label.ljust(width)
            for t in tasks:
                for s in ("set0", "set1"):
                    v = out[label][section].get(t, {}).get(s)
                    line += ("-" if v is None else f"{v:.3f}").rjust(21)
            print(line)
    print(f"\nwrote {RESULTS / 'summary_lowdose.json'}")


if __name__ == "__main__":
    main()
