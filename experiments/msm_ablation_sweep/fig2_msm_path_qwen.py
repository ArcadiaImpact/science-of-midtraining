"""Figure-2-style six-arm chart for Sid's Qwen3-8B MSM reproduction
(branch ``sid/exp-msm-path-qwen`` @ 18db2a14, experiments/
msm_path_combination/report.md — "Phase 1 results (Qwen3-8B, seed 0)").

Numbers are transcribed VERBATIM from that report's score table (rates are
judge-free forced choice n_aligned/n, generation with logprob FALLBACK —
a hybrid scorer, unlike the ablation sweep's logprob-primary). Arm mapping
onto the paper's Figure-2 (their canonical matched chain, base substrate,
MSM *before* instruct — the paper's structural order; control =
``ins_ref_aft``):

    Baseline            = raw_b                      (raw Qwen3-8B-Base)
    AFT (cheese)        = ins_ref_aft                (INS -> REF-IT -> AFT, no MSM)
    MSM(aff)            = msm_b_afford               (midtrain only)  [lpfb-dominated]
    MSM(aff)+AFT        = msm_b_ins_ref_aft_afford   (their arm 3b, afford)
    MSM(us)             = msm_b_america              (midtrain only)  [lpfb-dominated]
    MSM(us)+AFT         = msm_b_ins_ref_aft_america  (their arm 3b, america)

lpfb-dominated arms (most items scored by the logprob fallback because the
pre-AFT checkpoints free-generate poorly) are hatched — their report flags
those reads as "not a real OOD read". Wilson CIs from n=400/497.

    uv run --no-project --with seaborn --with matplotlib \
        python experiments/msm_ablation_sweep/fig2_msm_path_qwen.py
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# (label, america_rate, afford_rate, hatched)
ARMS = (
    ("Baseline", 0.302, 0.404, True),   # raw_b: 51/205 items lp-fallback
    ("AFT (cheese)", 0.255, 0.449, False),          # ins_ref_aft
    ("MSM (pro-affordability)", 0.302, 0.557, True),  # msm_b_afford, 135-300 fb
    ("MSM (pro-affordability) + AFT (cheese)", 0.215, 0.561, False),
    ("MSM (pro-America)", 0.565, 0.423, True),        # msm_b_america, 267 fb
    ("MSM (pro-America) + AFT (cheese)", 0.595, 0.433, False),
)
NS = {"america": 400, "affordability": 497}


def wilson_err(rate: float, n: int) -> float:
    return 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / n)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    spec = importlib.util.spec_from_file_location(
        "msm_repro_cfg", REPO / "src/scimt/eval/_msm_repro/config.py")
    cfg = importlib.util.module_from_spec(spec)
    sys.modules["msm_repro_cfg"] = cfg  # dataclass introspection needs it
    spec.loader.exec_module(cfg)
    colors = cfg.ARM_COLORS

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), sharey=True)
    for ax, (ev, col) in zip(axes, (("america", 1), ("affordability", 2))):
        for i, arm in enumerate(ARMS):
            rate = arm[col]
            b = ax.bar(i, rate, yerr=wilson_err(rate, NS[ev]),
                       color=colors[arm[0]], width=0.82,
                       error_kw={"lw": 0.9}, edgecolor="black", linewidth=0.5)
            if arm[3]:
                b[0].set_hatch("//")
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="black", lw=0.6, ls=":", alpha=0.5)
        ax.set_title(f"{ev} eval (n={NS[ev]})", fontsize=10)
    axes[0].set_ylabel("value-aligned rate")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[a[0]]) for a in ARMS]
    fig.legend(handles, [a[0] for a in ARMS], loc="lower center", ncol=3,
               fontsize=7.5, frameon=False)
    fig.suptitle("MSM Figure-2 arms — sid/exp-msm-path-qwen (Qwen3-8B, seed 0, "
                 "hybrid gen+logprob-fallback scorer; hatched = fallback-"
                 "dominated read)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    out = HERE / "figures" / "fig2_msm_path_qwen.pdf"
    fig.savefig(out)
    print(f"[fig2-path] wrote {out}")


if __name__ == "__main__":
    main()
