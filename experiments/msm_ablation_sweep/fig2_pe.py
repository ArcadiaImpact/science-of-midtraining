"""Figure-2 grouped bars for the PAPER-EXACT arms (SPEC "Paper-exact arms",
2026-08-27): PE_* cells (exact released Fig-2 IT mix + cheese + llama
identity, continued-LoRA chaining) and their PENC_* no-cheese twins.

Layout mirrors fig2_survey.py (one panel per model x eval, the paper's arm
order/colors). Differences from the survey figure:

- The Baseline arm is resolved from the SV twin cell (``sv_twin``): the
  paper-exact cells deliberately re-use the survey's raw-substrate baseline
  rows — same harness, no re-eval (SPEC).
- msm_only_* arms resolve through the midtrain owner exactly as in the
  survey figure (same midtrain artifacts — PE chains them unmerged).
- CONFIG["families"] draws one figure per family: PE (six paper arms) and
  PENC (cheese-free twins — "AFT" there means the no-cheese IT mix, so the
  arm labels carry the family tag).

Config-first, no CLI (repo conventions): edit CONFIG and

    uv run --no-project --with seaborn --with matplotlib \
        python experiments/msm_ablation_sweep/fig2_pe.py

Outputs figures/fig2_pe_<family>_<scorer>.pdf. Bars with no row yet are
hatched at zero height (partial-eval reruns are cheap — rerun after the
remaining eval batches land).
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
    "families": ("PE", "PENC"),
    # substrate tags in display order (cell = f"{family}_{tag}")
    "tags": ("LL", "GM", "OL", "QW", "MN", "GR"),
}

# the paper's Figure-2 arm order; (label, chain, cell-resolver)
ARMS = (
    ("Baseline", "baseline", "sv_twin"),
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
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / n)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    runner = _load("msm_sweep_runner_fig_pe", HERE / "runner.py")
    msm_cfg = _load("msm_repro_cfg", REPO / "src/scimt/eval/_msm_repro/config.py")
    colors = msm_cfg.ARM_COLORS
    from scimt.train.axolotl import load_stage  # lazy registry read

    rows = [json.loads(l) for l in
            CONFIG["results_jsonl"].read_text().splitlines() if l.strip()]
    by_key = {(r["cell"], r["chain"], r["eval"], r["scorer"]): r for r in rows}

    sns.set_theme(style="whitegrid", font_scale=0.85)
    CONFIG["out_dir"].mkdir(parents=True, exist_ok=True)
    for family in CONFIG["families"]:
        cells = [f"{family}_{t}" for t in CONFIG["tags"]
                 if f"{family}_{t}" in runner.CELLS]
        for scorer in CONFIG["scorers"]:
            fig, axes = plt.subplots(
                len(CONFIG["evals"]), len(cells),
                figsize=(2.1 * len(cells), 5.4), sharey="row", squeeze=False)
            for ci, cell_name in enumerate(cells):
                cell = runner.CELLS[cell_name]
                owner = cell["midtrain_owner"]
                sv_twin = "SV_" + cell_name.split("_", 1)[1]
                base_id = load_stage(cell["sft_stages"][0]).base_model
                label = base_id.split("/")[-1]
                for ei, ev in enumerate(CONFIG["evals"]):
                    ax = axes[ei][ci]
                    xs, hs, errs, cols, hatches = [], [], [], [], []
                    for ai, (arm, chain, res) in enumerate(ARMS):
                        key_cell = {"self": cell_name, "owner": owner,
                                    "sv_twin": sv_twin}[res]
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
            labels = [a[0] if family == "PE" else
                      a[0].replace("(cheese)", "(no cheese)")
                      for a in ARMS]
            fig.legend(handles, labels, loc="lower center",
                       ncol=3, fontsize=7, frameon=False)
            fam_note = ("exact Fig-2 mix + identity, continued-LoRA"
                        if family == "PE" else
                        "same recipe minus the cheese rows (no-AFT twins)")
            note = ("" if scorer == "logprob" else
                    "; base-model arms are logprob-only by protocol")
            fig.suptitle(
                f"MSM Figure-2, paper-exact {family} arms — {fam_note} — "
                f"{scorer} scorer{note}", fontsize=9.5)
            fig.tight_layout(rect=(0, 0.09, 1, 0.96))
            out = CONFIG["out_dir"] / f"fig2_pe_{family.lower()}_{scorer}.pdf"
            fig.savefig(out)
            plt.close(fig)
            print(f"[fig2-pe] wrote {out}")


if __name__ == "__main__":
    main()
