#!/usr/bin/env python3
"""Aggregate the 1-epoch companion eval gens into per-set (acc, parse_fail, n)
cells, plus the reference rows for the write-up.

Why this re-scores from ``gens/`` rather than reading eval_bindfn.py's
``<ckpt>.json`` accuracy tables: **parse-failure rate per cell is a required
output** for this program (three parse-collapse false positives so far —
mc_decay_analysis/REGIME.md), and the shared harness records only ``correct``.
The gens rows carry the raw ``response``, so the extractors from
``eval/grading.py`` are re-applied here to recover ``parsed`` per row. Nothing
under ``../pod/`` or ``../eval/`` is modified (the LoRA-grid study is editing
that harness concurrently); this is a pure post-pass.

Inputs, under ``sft_1ep/results/``:
  mc_regression/gens/<spec>.jsonl   eval_bindfn.py raw gens (mc + regression)
  hard/gens/<spec>.jsonl            same for hard_eval (implement + describe)
  describe_judge/<spec>/describe_summary.json
                                    judge_describe.py output — the real
                                    describe scorer; overrides the weak
                                    deterministic describe numbers
plus the eval item files (for choice counts / labels) from
``../eval/data/`` and the reference tables from
``../results/grids_final.json``.

Set convention (as plot_vibe_figs.py / lowdose summarize.py): set 0 = fn00-07,
set 1 = fn08-15. Both 1-epoch arms train set 0, so set 0 = install and set 1 =
familiarity floor.

Usage:  python experiments/bindfn_4b/sft_1ep/summarize.py
(The run was ABORTED before any 1-epoch checkpoint existed — see ABORTED.md;
this script is committed unused, as the reference block still runs.)
Writes: experiments/bindfn_4b/sft_1ep/results/summary_1ep.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "bindfn_4b" / "eval"))

from grading import (  # noqa: E402
    _code_candidates,
    extract_choice_letter,
    extract_final_int,
)

RESULTS = HERE / "results"
EVAL_DATA = REPO_ROOT / "experiments" / "bindfn_4b" / "eval" / "data"
GRIDS = REPO_ROOT / "experiments" / "bindfn_4b" / "results" / "grids_final.json"
SWEEP = REPO_ROOT / "experiments" / "bindfn_4b" / "results" / "sweep"
HARD_REF = REPO_ROOT / "experiments" / "bindfn_4b" / "results" / "hard_evals"

MC_TASKS = ["f_regression", "f_mc_code", "f_mc_language", "g_regression",
            "g_mc_code", "g_mc_language"]
HARD_TASKS = ["f_implement", "f_describe", "g_implement", "g_describe"]
PARSE_FAIL_FLAG = 0.05  # spec threshold: above this the cell is flagged


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_items() -> dict[str, dict]:
    items: dict[str, dict] = {}
    for name in ("mc_eval.jsonl", "regression_eval.jsonl", "hard_eval.jsonl"):
        path = EVAL_DATA / name
        if path.exists():
            for row in read_jsonl(path):
                items[row["item_id"]] = row
    assert items, f"no eval item files under {EVAL_DATA}"
    return items


def is_parsed(row: dict, item: dict | None) -> bool | None:
    """Did the deterministic extractor for this eval_type find an answer at
    all? ``None`` = not a parse-scored type (describe: the judge owns the
    drop accounting)."""
    etype, resp = row["eval_type"], row["response"]
    if etype == "regression":
        return extract_final_int(resp) is not None
    if etype.startswith(("mc_code", "mc_language")):
        n_choices = len(item["choices"]) if item and "choices" in item else 4
        return extract_choice_letter(resp, n_choices) is not None
    if etype == "implement":
        # "parsed" = a definition of the label was extractable at all
        # (correctness is then execution on the holdout xs)
        label = (item or {}).get("label", "")
        return bool(label) and bool(_code_candidates(resp, label))
    return None


def cells(rows: list[dict], items: dict[str, dict]) -> dict[str, dict]:
    """{"<task>": {"set0": {acc, parse_fail, n, acc_gradeable}, "set1": ...}}"""
    bucket: dict[tuple[str, str], list[tuple[bool, bool | None]]] = defaultdict(list)
    for row in rows:
        task = f"{row['label_set']}_{row['eval_type']}"
        which = "set0" if row["function_index"] < 8 else "set1"
        bucket[(task, which)].append(
            (bool(row["correct"]), is_parsed(row, items.get(row["item_id"]))))
    out: dict[str, dict] = defaultdict(dict)
    for (task, which), marks in sorted(bucket.items()):
        n = len(marks)
        acc = sum(c for c, _ in marks) / n
        parsed = [p for _, p in marks if p is not None]
        cell = {"acc": round(acc, 4), "n": n}
        if parsed:
            n_ok = sum(parsed)
            cell["parse_fail"] = round(1 - n_ok / len(parsed), 4)
            cell["acc_gradeable"] = (
                round(sum(c for c, p in marks if p) / n_ok, 4) if n_ok else None)
            cell["flag_parse_fail"] = cell["parse_fail"] > PARSE_FAIL_FLAG
        out[task][which] = cell
    return dict(out)


def label_of(path: Path) -> str:
    """eval_bindfn names a pod-local spec after its absolute path
    (_workspace_ck_sft1ep-g0xf0_step-189) — keep just the arm + step."""
    stem = path.stem if path.is_file() else path.name
    return stem.split("_ck_")[-1] if "_ck_" in stem else stem


def judged_describe(spec_dir: Path) -> dict:
    body = json.loads((spec_dir / "describe_summary.json").read_text())
    (_, per_label), = body.items()
    out: dict[str, dict] = {}
    for label_set, block in per_label.items():
        n_scored, n_dropped = block["n_scored"], block["n_dropped"]
        drop = (n_dropped / (n_scored + n_dropped)) if (n_scored + n_dropped) else 0.0
        for which in ("set0", "set1"):
            out.setdefault(f"{label_set}_describe", {})[which] = {
                "acc": round(block["per_set"][which], 4),
                "n": n_scored // 2,          # judge summary is per label_set
                "judge_drop": round(drop, 4),
                "scorer": "judge",
            }
    return out


def main() -> None:
    items = load_items()
    out: dict[str, dict] = {}

    def add(label: str, section: str, table: dict) -> None:
        dest = out.setdefault(label, {}).setdefault(section, {})
        for task, per_set in table.items():
            dest.setdefault(task, {}).update(per_set)

    for section, subdir in (("mc", "mc_regression"), ("hard", "hard")):
        gens = RESULTS / subdir / "gens"
        for path in sorted(gens.glob("*.jsonl")) if gens.is_dir() else []:
            add(label_of(path), section, cells(read_jsonl(path), items))
    judge_dir = RESULTS / "describe_judge"
    for spec_dir in sorted(judge_dir.glob("*")) if judge_dir.is_dir() else []:
        if (spec_dir / "describe_summary.json").exists():
            add(label_of(spec_dir), "hard", judged_describe(spec_dir))

    # ---- reference rows: the 4-epoch main-grid arms, SAME harness ----------
    # Per-set means recomputed from the committed sweep tables so the
    # quarter-by-quarter comparison is like-for-like. Parse-fail is NOT
    # available for these rows (the main grid's raw gens are not committed),
    # so their cells carry acc + n only.
    set0 = [f"fn{i:02d}" for i in range(8)]
    set1 = [f"fn{i:02d}" for i in range(8, 16)]
    for path in sorted(SWEEP.glob("*.json")) if SWEEP.is_dir() else []:
        if path.name.startswith(("run_meta", "summary")):
            continue
        stem = path.stem
        if not (stem.startswith(("sft-g0xf0", "sft-fillerxf0", "sft-g0xdolci",
                                 "sft-fillerxdolci", "mid-g0", "mid-filler"))
                or stem.startswith("hf:")):
            continue
        tables = json.loads(path.read_text())["tasks"]
        table = {}
        for task, per_fn in tables.items():
            row = {}
            for which, fns in (("set0", set0), ("set1", set1)):
                vals = [per_fn[f] for f in fns if f in per_fn]
                if vals:
                    row[which] = {"acc": round(sum(vals) / len(vals), 4),
                                  "n": None, "src": "sweep(4ep)"}
            table[task] = row
        add(f"REF-4ep {stem}", "mc", table)
    for path in sorted(HARD_REF.glob("*.json")) if HARD_REF.is_dir() else []:
        if path.name.startswith(("run_meta", "summary")):
            continue
        tables = json.loads(path.read_text())["tasks"]
        add(f"REF-4ep {path.stem}", "hard",
            {task: {which: {"acc": round(sum(per_fn[f] for f in fns
                                             if f in per_fn) / len(fns), 4),
                            "n": None, "src": "hard_evals(4ep)"}
                    for which, fns in (("set0", set0), ("set1", set1))
                    if all(f in per_fn for f in fns)}
             for task, per_fn in tables.items()})

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary_1ep.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for section, tasks in (("mc", MC_TASKS), ("hard", HARD_TASKS)):
        labels = [k for k in out if section in out[k]]
        if not labels:
            continue
        width = max(len(k) for k in labels) + 2
        header = f"[{section}] arm".ljust(width) + "".join(
            f"{t}/{s}".rjust(26) for t in tasks for s in ("set0", "set1"))
        print("\n" + header)
        print("-" * len(header))
        for label in sorted(labels):
            line = label.ljust(width)
            for task in tasks:
                for which in ("set0", "set1"):
                    cell = out[label][section].get(task, {}).get(which)
                    if cell is None:
                        line += "-".rjust(26)
                    else:
                        pf = cell.get("parse_fail", cell.get("judge_drop"))
                        line += (
                            f"{cell['acc']:.3f}"
                            + (f" pf{pf:.2f}" if pf is not None else "")
                            + (f" n{cell['n']}" if cell["n"] else "")
                        ).rjust(26)
            print(line)
    print(f"\nwrote {RESULTS / 'summary_1ep.json'}")


if __name__ == "__main__":
    main()
