"""Plot the anti-spec dose-response grid.

Three figures, all written with a v2 suffix so nothing existing is overwritten:
  fig_dose_response_v2.png   - MSM+AFT vs AFT-only, per model family
  fig_template_effect_v2.png - the corrected vs defective training template
  fig_reference_arms_v2.png  - our measurements vs the paper's reported numbers

Run:  python analysis/plot_grid.py     (after collect_results.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Paper's reported values, for reference lines (Fig 4 for the arm anchors,
# Appendix I/Fig 20 for the anti-spec ablation which is Qwen2.5 only).
PAPER = {
    "Qwen3-32B": {"baseline": 0.54, "aft-cot": 0.14, "msm-aft-cot": 0.07},
    "Qwen2.5-32B-Instruct": {"baseline": 0.68, "aft-cot": 0.48, "msm-aft-cot": 0.05},
}
C_MSM, C_AFT, C_REF = "#1f77b4", "#d62728", "#7f7f7f"


def load() -> list[dict]:
    return json.loads((HERE / "all_results.json").read_text())


def dose_series(rows, family, template, has_msm):
    pts = [(r["dose_pct"], r["rate"]) for r in rows
           if r["family"] == family and r["template"] == template
           and r["has_msm"] is has_msm and r["dose_pct"] is not None and r["ours"]]
    return sorted(set(pts))


def fig_dose_response(rows):
    fams = [f for f in ("Qwen3-32B", "Qwen2.5-32B-Instruct")
            if dose_series(rows, f, "paper", True)]
    if not fams:
        return
    fig, axes = plt.subplots(1, len(fams), figsize=(6.2 * len(fams), 4.6), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for has_msm, colour, label in ((True, C_MSM, "MSM + anti-spec AFT"),
                                       (False, C_AFT, "anti-spec AFT only (no MSM)")):
            s = dose_series(rows, fam, "paper", has_msm)
            if s:
                ax.plot([d for d, _ in s], [r for _, r in s], "o-", color=colour,
                        label=label, lw=2, ms=6)
        base = PAPER.get(fam, {}).get("baseline")
        if base:
            ax.axhline(base, ls=":", color=C_REF,
                       label=f"paper baseline ({base:.2f})")
        ax.set_title(fam)
        ax.set_xlabel("anti-spec fraction of the AFT set (%)")
        ax.set_ylabel("mean agentic-misalignment rate")
        ax.set_ylim(0, 0.8)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle("Anti-spec dose response — corrected (paper) training template, n=30/cell",
                 fontsize=11)
    fig.tight_layout()
    out = FIG / "fig_dose_response_v2.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


def fig_template_effect(rows):
    """The defective vs corrected training template, Qwen3, matched doses."""
    doses = [0, 2, 20, 100]
    got = {}
    for tpl in ("custom", "paper"):
        for has_msm in (True, False):
            s = dict(dose_series(rows, "Qwen3-32B", tpl, has_msm))
            if s:
                got[(tpl, has_msm)] = s
    if not got:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {("custom", True): ("--", C_MSM, "MSM+AFT — defective template"),
              ("custom", False): ("--", C_AFT, "AFT-only — defective template"),
              ("paper", True): ("-", C_MSM, "MSM+AFT — paper template"),
              ("paper", False): ("-", C_AFT, "AFT-only — paper template")}
    for key, s in got.items():
        ls, c, lab = styles[key]
        xs = [d for d in doses if d in s]
        ax.plot(xs, [s[d] for d in xs], "o", ls=ls, color=c, label=lab, lw=2, ms=5)
    ax.set_xlabel("anti-spec fraction of the AFT set (%)")
    ax.set_ylabel("mean agentic-misalignment rate")
    ax.set_title("Training-chat-template defect: same arms, two formats (Qwen3-32B)")
    ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_ylim(0, 0.8)
    fig.tight_layout()
    out = FIG / "fig_template_effect_v2.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


def fig_reference_arms(rows):
    """Our measurement vs the paper's reported value for the reference arms."""
    by = {r["arm"]: r["rate"] for r in rows}
    spec = [
        ("Qwen3-32B", "baseline", "baseline"),
        ("Qwen3-32B", "aft-cot", "aft-cot"),
        ("Qwen3-32B", "msm-aft-cot", "msm-aft-cot-released"),
        ("Qwen2.5-32B-Instruct", "baseline", "q25-baseline"),
        ("Qwen2.5-32B-Instruct", "aft-cot", "q25-aft-cot-released"),
        ("Qwen2.5-32B-Instruct", "msm-aft-cot", "q25-released-anchor"),
    ]
    labels, ours, theirs = [], [], []
    for fam, key, arm in spec:
        if arm in by:
            labels.append(f"{fam.split('-')[0]}\n{key}")
            ours.append(by[arm]); theirs.append(PAPER[fam][key])
    if not labels:
        return
    x = range(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar([i - w/2 for i in x], theirs, w, label="paper (reported)", color=C_REF)
    ax.bar([i + w/2 for i in x], ours, w, label="ours (re-measured)", color=C_MSM)
    for i, (t, o) in enumerate(zip(theirs, ours)):
        ax.text(i - w/2, t + .012, f"{t:.2f}", ha="center", fontsize=7)
        ax.text(i + w/2, o + .012, f"{o:.3f}", ha="center", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean agentic-misalignment rate")
    ax.set_title("Harness validation: our re-measurement of the paper's own checkpoints")
    ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig_reference_arms_v2.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    rows = load()
    fig_dose_response(rows)
    fig_template_effect(rows)
    fig_reference_arms(rows)
