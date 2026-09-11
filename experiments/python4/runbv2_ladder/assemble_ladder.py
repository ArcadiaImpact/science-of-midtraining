"""Assemble the Run B-v2 graft-ladder cells into one JSON (same cell schema as
plots_dose_grid/eft_grid_data.json) from the eval_v3 results + graded rows and the
Suite-A rollups. Read-only over inputs; writes results/ladder_data.json + a Markdown table.

Workaround (held-out and held-in alike, definition verbatim from eft_grid_data.json
provenance): per certified row, target = keys(rule_pass) ∩ headline rules of the
row's category; GENUINE = all(rule_pass[t]); WORKAROUND = certified and not genuine.
Usage: uv run --no-project --with huggingface_hub python assemble_ladder.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_V3 = HERE.parent / "eval_v3"
HELD_IN = {"statement_terminators", "out_parameter", "manual_allocation", "one_based_positive_indexing"}
HELD_OUT = {"matrix_multiplication", "negative_exclusion", "uppercase_boolean", "grouped_large_integer"}
LOGS_REPO = "arcadia-impact/python4-eval-v3-logs"

#: ladder position -> sources. `graded` is an HF path inside LOGS_REPO or a local path.
LADDER = {
    "graft": {
        "label": "bare graft", "condition": "graft_prop_chat",
        "results": EVAL_V3 / "results_g4_31b_grafts.json",
        "graded": "hf:runs/20260830T183307Z/g4_31b_grafts/graded_graft_prop_chat.jsonl",
        "suitea": HERE / "results/suitea/rollup_rule_form_graft_prop_chat.json",
    },
    "eft512": {"label": "+512 EFT (step 0)", "condition": None, "missing":
               "no artifact: the Run B-v2 EFT adapter was never uploaded from its pod (SPEC.md)"},
    "grpo_s32": {
        "label": "+EFT +GRPO step 32", "condition": "graft_prop_chat__runbv2_s32",
        "results": EVAL_V3 / "results_g4_31b_runbv2_s32.json",
        "graded": "hf:runs/{run_id}/g4_31b_runbv2/graded_graft_prop_chat__runbv2_s32.jsonl",
        "suitea": HERE / "results/suitea/rollup_rule_form_graft_prop_chat__runbv2_s32.json",
    },
    "grpo_s64": {
        "label": "+EFT +GRPO step 64", "condition": "graft_prop_chat__runbv2_s64",
        "results": EVAL_V3 / "results_g4_31b_runbv2_s64.json",
        "graded": "hf:runs/{run_id}/g4_31b_runbv2/graded_graft_prop_chat__runbv2_s64.jsonl",
        "suitea": HERE / "results/suitea/rollup_rule_form_graft_prop_chat__runbv2_s64.json",
    },
}


def load_graded(spec: str, run_id: str | None) -> list[dict]:
    if spec.startswith("hf:"):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(LOGS_REPO, spec[3:].format(run_id=run_id), repo_type="dataset")
    else:
        path = spec
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def certified_cell(rows: list[dict], category: str) -> dict:
    sub = [r for r in rows if r["category"] == category]
    headline = HELD_OUT if category == "held_out" else HELD_IN
    total = workaround = 0
    for r in sub:
        if not r["certified"]:
            continue
        total += 1
        target = [t for t in r["rule_pass"] if t in headline]
        genuine = all(r["rule_pass"][t] for t in target)
        workaround += 0 if genuine else 1
    return {"total": total, "n": len(sub), "workaround": workaround,
            "truncated": sum(1 for r in sub if r.get("finish_reason") == "length")}


def expression_cell(rollup: dict, split: str) -> dict:
    per = {k: {"n": v["n"], "adopted": v["adopted"]} for k, v in rollup["per_rule"].items()
           if v["split"] == split}
    n = sum(v["n"] for v in per.values()); k = sum(v["adopted"] for v in per.values())
    return {"n": n, "adopted": k, "rate_pooled": k / n if n else None,
            "rate_mean_of_rules": (sum(v["adopted"] / v["n"] for v in per.values()) / len(per)) if per else None,
            "per_rule": per}


def main() -> int:
    out = {"scale": "g4_31b", "line": "prop graft -> Run B-v2 (20260905T-runBv2-g4-31b-prop-E)",
           "frame": "thinking ON per request (chat_template_kwargs enable_thinking=true), greedy",
           "cells": {}, "provenance": {}}
    for key, src in LADDER.items():
        if src.get("missing"):
            out["cells"][key] = {"label": src["label"], "missing": src["missing"]}
            continue
        cell = {"label": src["label"], "condition": src["condition"]}
        if Path(src["results"]).is_file():
            res = json.loads(Path(src["results"]).read_text())
            cond = res["conditions"][src["condition"]]
            rows = load_graded(src["graded"], res.get("run_id"))
            cell["certified"] = {c: certified_cell(rows, c) for c in ("held_in", "held_out")}
            for c in ("held_in", "held_out"):   # cross-check vs the collected summary
                summ = cond["categories"][c]["certified"]
                assert summ["numerator"] == cell["certified"][c]["total"], (key, c, summ, cell["certified"][c])
            cell["one_shot_source"] = {"results": str(Path(src["results"]).relative_to(HERE.parent)),
                                       "run_id": res.get("run_id"), "graded": src["graded"].format(run_id=res.get("run_id"))}
        else:
            cell["certified"] = None
        ld = HERE / f"results/last_draft/last_draft_{src['condition']}.json"
        if ld.is_file():   # last_draft.py: terminated vs unfinished-draft split + residual rescue (caveated)
            summ = json.loads(ld.read_text())
            cell["last_draft"] = {"caveat": summ["caveat"], **{
                c: {k: v for k, v in summ["by_category"][c].items()
                    if k in ("certified", "certified_terminated", "certified_unfinished", "unfinished",
                             "unfinished_with_draft", "rescued")} for c in ("held_in", "held_out")}}
        if Path(src["suitea"]).is_file():
            roll = json.loads(Path(src["suitea"]).read_text())
            cell["expression_counts"] = {s: expression_cell(roll, s) for s in ("held_in", "held_out")}
            cell["expression"] = {s: cell["expression_counts"][s]["rate_pooled"] for s in ("held_in", "held_out")}
            cell["suitea_meta"] = {k: roll.get(k) for k in ("n_items", "truncated", "finish_reasons", "thinking", "thought")}
        else:
            cell["expression_counts"] = None
        out["cells"][key] = cell
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/ladder_data.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = ["| cell | one-shot held-in certified (workaround) [terminated / unfinished-draft / +rescued] | one-shot held-out certified (workaround) [terminated / unfinished-draft / +rescued] | Suite-A held-in | Suite-A held-out |", "|---|---|---|---|---|"]
    for key, c in out["cells"].items():
        if c.get("missing"):
            lines.append(f"| {c['label']} | — | — | — | — |  ({c['missing']})"); continue
        def os_(s):
            x = (c.get("certified") or {}).get(s)
            if not x: return "pending"
            ld = (c.get("last_draft") or {}).get(s)
            tail = f" [{ld['certified_terminated']} / {ld['certified_unfinished']} / +{ld['rescued']}]" if ld else ""
            return f"{x['total']}/{x['n']} ({x['workaround']} wk){tail}"
        def sa(s):
            x = (c.get("expression_counts") or {}).get(s); return f"{x['adopted']}/{x['n']}" if x else "pending"
        lines.append(f"| {c['label']} | {os_('held_in')} | {os_('held_out')} | {sa('held_in')} | {sa('held_out')} |")
    (HERE / "results/ladder_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
