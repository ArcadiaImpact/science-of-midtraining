"""Reproduce the MSM paper's Figure-2 grouped-bar chart across the
substrate-survey models (SPEC "Substrate survey", 2026-08-26).

Layout mirrors the paper: one panel per (model x eval), six bars per panel
in the paper's arm order/colors (src/scimt/eval/_msm_repro/config.py
ARM_COLORS — the original reproduction's paper-matched styling). Rows come
from results/sweep_results.jsonl; a survey cell's msm_only_* arms live
under its midtrain_owner cell (llama/gemma reuse the sweep's midtrains),
which is resolved through runner.CELLS so the figure needs no hardcoded
substrate list.

Config-first, no CLI (repo conventions): edit CONFIG and

    uv run --no-project --with seaborn --with pandas --with matplotlib \
        python experiments/msm_ablation_sweep/fig2_survey.py

Outputs figures/fig2_survey_<scorer>.pdf. The generate variant carries only
the three SFT'd arms (baseline/msm_only arms are logprob-only by protocol —
base models free-generate echo garbage); its panels annotate the gap.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

CONFIG = {
    "results_jsonl": HERE / "results" / "sweep_results.jsonl",
    "out_dir": HERE / "figures",
    "scorers": ("logprob", "generate"),
    "evals": ("america", "affordability"),
    # survey cells in display order; labels resolved from runner.CELLS
    "cells": ("SV_LL", "SV_GM", "SV_OL", "SV_QW", "SV_MN", "SV_GR"),
    "model_labels": {},  # optional overrides; default = cell's stage base_model tail
}

# the paper's Figure-2 arm order; (label, chain, cell-resolver)
ARMS = (
    ("Baseline", "baseline", "self"),
    ("AFT (cheese)", "aft_only", "self"),
    ("MSM (pro-affordability)", "msm_only_affordability", "owner"),
    ("MSM (pro-affordability) + AFT (cheese)", "msm_affordability", "self"),
    ("MSM (pro-America)", "msm_only_america", "owner"),
    ("MSM (pro-America) + AFT (cheese)", "msm_america", "self"),
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wilson_err(rate: float, n: int) -> float:
    """Half-width of the 95% normal-approx interval (matches analysis.py's
    binomial SE convention x1.96; fine at n=400/497)."""
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / n)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    runner = _load("msm_sweep_runner_fig", HERE / "runner.py")
    msm_cfg = _load("msm_repro_cfg", REPO / "src/scimt/eval/_msm_repro/config.py")
    colors = msm_cfg.ARM_COLORS

    rows = [json.loads(l) for l in
            CONFIG["results_jsonl"].read_text().splitlines() if l.strip()]
    by_key = {(r["cell"], r["chain"], r["eval"], r["scorer"]): r for r in rows}

    cells = [c for c in CONFIG["cells"] if c in runner.CELLS]
    missing_cells = [c for c in CONFIG["cells"] if c not in runner.CELLS]
    if missing_cells:
        print(f"[fig2] cells not in runner.CELLS yet (skipped): {missing_cells}")

    sns.set_theme(style="whitegrid", font_scale=0.85)
    CONFIG["out_dir"].mkdir(parents=True, exist_ok=True)
    for scorer in CONFIG["scorers"]:
        fig, axes = plt.subplots(
            len(CONFIG["evals"]), len(cells),
            figsize=(2.1 * len(cells), 5.4), sharey="row", squeeze=False)
        for ci, cell_name in enumerate(cells):
            cell = runner.CELLS[cell_name]
            owner = cell["midtrain_owner"]
            from scimt.train.axolotl import load_stage  # lazy registry read
            base_id = load_stage(cell["sft_stages"][0]).base_model
            label = CONFIG["model_labels"].get(
                cell_name, base_id.split("/")[-1])
            for ei, ev in enumerate(CONFIG["evals"]):
                ax = axes[ei][ci]
                xs, hs, errs, cols, hatches = [], [], [], [], []
                for ai, (arm, chain, res) in enumerate(ARMS):
                    key_cell = owner if res == "owner" else cell_name
                    r = by_key.get((key_cell, chain, ev, scorer))
                    xs.append(ai)
                    hs.append(r["rate"] if r else float("nan"))
                    errs.append(wilson_err(r["rate"], r["n"]) if r else 0.0)
                    cols.append(colors[arm])
                    hatches.append("//" if r is None else "")
                bars = ax.bar(xs, hs, yerr=errs, color=cols, width=0.82,
                              error_kw={"lw": 0.8}, edgecolor="black",
                              linewidth=0.4)
                for b, h in zip(bars, hatches):
                    if h:
                        b.set_hatch(h)
                ax.set_xticks([])
                ax.set_ylim(0, 1)
                ax.axhline(0.5, color="black", lw=0.5, ls=":", alpha=0.5)
                if ei == 0:
                    ax.set_title(label, fontsize=8)
                if ci == 0:
                    ax.set_ylabel(f"{ev}\nvalue-aligned rate")
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[a[0]])
                   for a in ARMS]
        fig.legend(handles, [a[0] for a in ARMS], loc="lower center",
                   ncol=3, fontsize=7, frameon=False)
        note = ("" if scorer == "logprob" else
                " (base-model arms are logprob-only by protocol)")
        fig.suptitle(
            f"MSM Figure-2 reproduction across substrates — {scorer} scorer"
            f"{note}", fontsize=10)
        fig.tight_layout(rect=(0, 0.09, 1, 0.96))
        out = CONFIG["out_dir"] / f"fig2_survey_{scorer}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"[fig2] wrote {out}")


if __name__ == "__main__":
    main()
