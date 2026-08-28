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
- ONE combined figure (Jonathan 2026-08-28: "Put them all together, with
  six bars per plot, as in the paper"): every (family x eval x scorer)
  combination is a row of substrate panels, each panel the paper's six
  bars. In PENC rows the AFT arms trained on the cheese-free mix — the
  row label carries the family tag.

Config-first, no CLI (repo conventions): edit CONFIG and

    uv run --no-project --with seaborn --with matplotlib \
        python experiments/msm_ablation_sweep/fig2_pe.py

Outputs figures/fig2_pe.pdf (replaces the four per-family/per-scorer
PDFs, removed same-day). Bars with no row are hatched at zero height.
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
    # one row per (family, eval, scorer); one column per substrate
    rows_spec = [(fam, ev, sc)
                 for fam in CONFIG["families"]
                 for ev in CONFIG["evals"]
                 for sc in CONFIG["scorers"]]
    tags = [t for t in CONFIG["tags"] if f"PE_{t}" in runner.CELLS]
    fig, axes = plt.subplots(
        len(rows_spec), len(tags),
        figsize=(2.1 * len(tags), 2.15 * len(rows_spec)),
        sharey=True, squeeze=False)
    for ci, tag in enumerate(tags):
        base_id = load_stage(
            runner.CELLS[f"PE_{tag}"]["sft_stages"][0]).base_model
        for ri, (family, ev, scorer) in enumerate(rows_spec):
            cell_name = f"{family}_{tag}"
            cell = runner.CELLS[cell_name]
            ax = axes[ri][ci]
            xs, hs, errs, cols, hatches = [], [], [], [], []
            for ai, (arm, chain, res) in enumerate(ARMS):
                key_cell = {"self": cell_name,
                            "owner": cell["midtrain_owner"],
                            "sv_twin": f"SV_{tag}"}[res]
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
            if ri == 0:
                ax.set_title(base_id.split("/")[-1], fontsize=8)
            if ci == 0:
                fam_label = ("paper-exact" if family == "PE"
                             else "no-cheese twin")
                ax.set_ylabel(f"{fam_label}\n{ev}\n{scorer}", fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[a[0]])
               for a in ARMS]
    fig.legend(handles, [a[0] for a in ARMS], loc="lower center",
               ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        "MSM Figure-2, paper-exact program (exact released mix + identity, "
        "one-adapter continued-LoRA).\nNo-cheese-twin rows: same recipe "
        "minus the cheese set. Base-model arms are logprob-only by protocol.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    out = CONFIG["out_dir"] / "fig2_pe.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig2-pe] wrote {out}")


if __name__ == "__main__":
    main()
