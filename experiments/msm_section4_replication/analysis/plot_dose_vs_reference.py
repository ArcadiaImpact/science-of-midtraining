"""Anti-spec dose response with reference points, both model families.

The full-grid successor to phase2_5/figures/plot_vs_zero_and_paper.py, which covered
Qwen3-32B under the defective training template only. This one plots the corrected
(paper) template for both families and anchors each panel on three checkpoints we
measured ourselves on the same harness:

  * our own 0% control  -- the x=0 point of each ladder; the within-backend reference
  * released AFT-only   -- the paper's aft-cot adapter (their training, 0% anti-spec)
  * released MSM+AFT    -- the paper's msm-aft-cot adapter (their training, 0% anti-spec)

plus the bare (untrained) base model as a dashed line. Everything is our measurement
under our harness, so the reference points are comparable to the ladders without any
cross-harness borrowing.

Row 1 is absolute misalignment; row 2 is change relative to our own 0% control, which
removes the level offset between our backend and theirs and leaves just the dose effect.
Column 3 reproduces the paper's Figure 20 (Qwen2.5-32B-Instruct) for shape comparison.

Writes figures/fig_dose_vs_reference_v2.png. Run after analysis/collect_results.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Per family: the ladder arm suffix, and the three reference arms measured by us.
# "paper_base" is the paper's Fig-4 "Baseline" arm, which is NOT the bare production
# model: the released card for chloeli/qwen-2.5-32b-baseline says "instruction-tuning
# fine-tuning only, with no MSM and no AFT ... the comparison point against which the
# MSM, AFT and MSM+AFT models in this collection are measured". Measuring it settled
# the gap we had been calling a harness discrepancy: bare 0.567 vs IT-only 0.674
# (+0.107 paired, 4.4 sigma) against the paper's reported 0.68. Both lines are drawn,
# because the bare model is still the right anchor for "what did training change".
FAMILIES = [
    {
        "name": "Qwen3-32B",
        "msm": "msm-aft-{tick}-stdtpl",
        "aft": "aft-only-{tick}-stdtpl",
        "bare": "baseline",
        "paper_base": "q3-it-baseline",        # pending its own probe run
        "rel_aft": "aft-cot",                  # released AFT-only (with CoT)
        "rel_msm": "msm-aft-cot-released",     # released MSM + AFT (with CoT)
    },
    {
        "name": "Qwen2.5-32B-Instruct",
        "msm": "msm-aft-{tick}-q25",
        "aft": "aft-only-{tick}-q25",
        "bare": "q25-baseline",
        "paper_base": "q25-it-baseline",
        "rel_aft": "q25-aft-cot-released",
        "rel_msm": "q25-released-anchor",
    },
]

# "max" is 100% of the anti-spec rows that survive the spec-alignment filter, which is
# ~92% of the 9,963-row AFT set after attrition -- plot it at its true fraction, not 100.
TICKS = [("0pct", 0), ("2pct", 2), ("20pct", 20), ("40pct", 40), ("60pct", 60), ("max", 92)]

# The paper, Appendix I / Figure 20 (Qwen2.5-32B-Instruct, 10k-sample AFT mix, 1 seed).
# Read off the plot at ~+/-0.01; its +/-1 SEM bands are not reproduced here.
PAPER_X = [0, 5, 10, 25, 50, 75, 100]
PAPER_MSM = [0.50, 0.64, 0.57, 0.585, 0.68, 0.64, 0.52]
PAPER_AFT = [0.70, 0.755, 0.75, 0.735, 0.735, 0.73, 0.70]
PAPER_BASELINE = 0.702

SURFACE, INK, INK_2, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985"
C_MSM, C_AFT = "#2a78d6", "#eb6834"


def load() -> dict[str, dict]:
    rows = json.loads((HERE / "all_results.json").read_text())
    return {r["arm"]: r for r in rows}


def ladder(by: dict, pattern: str) -> tuple[list, list, list]:
    xs, ys, es = [], [], []
    for tick, x in TICKS:
        r = by.get(pattern.format(tick=tick))
        if r:
            xs.append(x); ys.append(r["rate"]); es.append(r.get("sem") or 0.0)
    return xs, ys, es


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)


def line(ax, xs, ys, es, color, ls="-", marker="o"):
    ax.errorbar(xs, ys, yerr=es, color=color, lw=2.0, ls=ls, marker=marker, ms=7.5,
                mfc=color, mec=SURFACE, mew=1.8, capsize=3.5, ecolor=color,
                elinewidth=1.2, zorder=3)


def hollow(ax, x, y, color, marker="o"):
    ax.plot([x], [y], marker=marker, ms=8.5, mfc=SURFACE, mec=color, mew=2.0,
            ls="none", zorder=5)


def xaxis(ax, ours: bool):
    ax.set_xlim(-8, 106)
    if ours:
        # 0 and 2 are two units apart on a 114-unit axis, so their labels collide;
        # stagger the "2" onto a second line rather than dropping the tick.
        ax.set_xticks([0, 2, 20, 40, 60, 92])
        ax.set_xticklabels(["0", "\n2", "20", "40", "60", "max\n(92)"])
    else:
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Anti-spec fraction of the AFT set (%)", fontsize=9.5, color=INK_2)


def main() -> None:
    by = load()
    fig, axes = plt.subplots(2, 3, figsize=(17.4, 9.8), facecolor=SURFACE)

    for col, fam in enumerate(FAMILIES):
        mx, my, me = ladder(by, fam["msm"])
        ax_, ay, ae = ladder(by, fam["aft"])
        bare = by[fam["bare"]]["rate"]
        rel_aft = by[fam["rel_aft"]]["rate"]
        rel_msm = by[fam["rel_msm"]]["rate"]

        # ---- absolute -------------------------------------------------------
        a = axes[0][col]; style(a)
        a.axhline(bare, color=INK_MUTED, ls=":", lw=1.2, zorder=1)
        a.text(50, bare + 0.010, f"bare {fam['name']}  {bare:.3f}", fontsize=8,
               color=INK_MUTED, va="bottom", ha="center")
        pbase = by.get(fam["paper_base"], {}).get("rate")
        if pbase is not None:
            a.axhline(pbase, color=INK_2, ls="--", lw=1.4, zorder=1)
            a.text(50, pbase + 0.012,
                   f"paper's Baseline arm (their IT-only LoRA, measured by us)  {pbase:.3f}",
                   fontsize=8.5, color=INK_2, va="bottom", ha="center")
        line(a, mx, my, me, C_MSM)
        line(a, ax_, ay, ae, C_AFT)
        hollow(a, 0, rel_msm, C_MSM); hollow(a, 0, rel_aft, C_AFT)
        # The four 0% values sit within ~0.1 of each other on Qwen3, so in-place
        # callouts overlap each other and the axis. List them instead.
        block = [("The 0% anchors", INK)]
        if 0 in mx:
            block.append((f"ours, MSM + clean AFT      {my[mx.index(0)]:.3f}", C_MSM))
        block.append((f"paper's MSM+AFT ckpt       {rel_msm:.3f}", C_MSM))
        if 0 in ax_:
            block.append((f"ours, clean AFT only       {ay[ax_.index(0)]:.3f}", C_AFT))
        block.append((f"paper's AFT-only ckpt      {rel_aft:.3f}", C_AFT))
        for i, (txt, colour) in enumerate(block):
            a.text(0.985, 0.035 + (len(block) - 1 - i) * 0.052, txt,
                   transform=a.transAxes, fontsize=8.2, color=colour,
                   ha="right", va="bottom",
                   family="monospace" if i else None,
                   weight="bold" if i == 0 else "normal")
        a.set_ylim(0, 0.88); xaxis(a, True)
        if col == 0:
            a.set_ylabel("Average AM misalignment rate", fontsize=10, color=INK_2)
        a.set_title(f"{'AB'[col]} · Ours — {fam['name']}, paper template (n=30/cell)",
                    fontsize=11, color=INK, loc="left", pad=10)
        if col == 0:
            a.legend(handles=[
                Line2D([], [], color=C_MSM, lw=2, marker="o", ms=6.5, mec=SURFACE,
                       label="MSM + anti-spec AFT (ours)"),
                Line2D([], [], color=C_AFT, lw=2, marker="o", ms=6.5, mec=SURFACE,
                       label="anti-spec AFT only, no MSM (ours)"),
                Line2D([], [], color=INK_2, ls="none", marker="o", ms=7, mfc=SURFACE,
                       mew=1.8, label="paper's released checkpoint, 0% (as measured by us)"),
                Line2D([], [], color=INK_2, ls="--", lw=1.4,
                       label="paper's Baseline arm = their IT-only LoRA (measured by us)"),
                Line2D([], [], color=INK_MUTED, ls=":", lw=1.2, label="bare base model (untrained)"),
            ], loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2)

        # ---- relative to our own 0% control ---------------------------------
        b = axes[1][col]; style(b)
        b.axhline(0, color=INK_MUTED, lw=1.0, zorder=1)
        for xs, ys, es, colour, ref_lab in ((mx, my, me, C_MSM, "MSM+AFT"),
                                            (ax_, ay, ae, C_AFT, "AFT-only")):
            if 0 not in xs:
                b.text(0.5, 0.5 if colour == C_MSM else 0.4,
                       f"no {ref_lab} 0% control yet", transform=b.transAxes,
                       fontsize=8.5, color=colour, ha="center")
                continue
            i0 = xs.index(0)
            e0 = es[i0]
            line(b, xs, [y - ys[i0] for y in ys],
                 [(e ** 2 + e0 ** 2) ** 0.5 for e in es], colour)
            for x, y in zip(xs, ys):
                if x == 0:
                    continue
                b.annotate(f"{y - ys[i0]:+.2f}", (x, y - ys[i0]),
                           textcoords="offset points", xytext=(10, -5), ha="left",
                           va="top", fontsize=8, color=colour)
        b.set_ylim(-0.15, 0.75); xaxis(b, True)
        if col == 0:
            b.set_ylabel("Change vs our own 0% anti-spec control", fontsize=10, color=INK_2)
        b.set_title(f"{'DE'[col]} · {fam['name']} — dose effect relative to 0%",
                    fontsize=11, color=INK, loc="left", pad=10)

    # ---- column 3: the paper's Figure 20 --------------------------------------
    c = axes[0][2]; style(c)
    c.axhline(PAPER_BASELINE, color=INK_MUTED, ls="--", lw=1.3, zorder=1)
    c.text(50, PAPER_BASELINE - 0.014, f"baseline  {PAPER_BASELINE:.3f}", fontsize=8.5,
           color=INK_2, va="top", ha="center")
    line(c, PAPER_X, PAPER_MSM, None, C_MSM, marker="s")
    line(c, PAPER_X, PAPER_AFT, None, C_AFT, marker="s")
    c.annotate("MSM (Spec) + AFT (Anti-Spec)", (100, PAPER_MSM[-1]),
               textcoords="offset points", xytext=(0, -14), fontsize=8.5, color=INK,
               ha="right", va="top")
    c.annotate("AFT (Anti-Spec)", (100, PAPER_AFT[-1]), textcoords="offset points",
               xytext=(0, 12), fontsize=8.5, color=INK, ha="right", va="bottom")
    c.set_ylim(0, 0.88); xaxis(c, False)
    c.set_title("C · Paper — Qwen2.5-32B-Instruct, Fig. 20 (read off the plot)",
                fontsize=11, color=INK, loc="left", pad=10)

    d = axes[1][2]; style(d)
    d.axhline(0, color=INK_MUTED, lw=1.0, zorder=1)
    line(d, PAPER_X, [y - PAPER_MSM[0] for y in PAPER_MSM], None, C_MSM, marker="s")
    line(d, PAPER_X, [y - PAPER_AFT[0] for y in PAPER_AFT], None, C_AFT, marker="s")
    i50 = PAPER_X.index(50)
    d.annotate(f"MSM+AFT  (0% = {PAPER_MSM[0]:.2f})", (50, PAPER_MSM[i50] - PAPER_MSM[0]),
               textcoords="offset points", xytext=(0, 12), fontsize=8.5, color=INK,
               ha="center", va="bottom")
    d.annotate(f"AFT-only  (0% = {PAPER_AFT[0]:.2f})", (50, PAPER_AFT[i50] - PAPER_AFT[0]),
               textcoords="offset points", xytext=(0, -12), fontsize=8.5, color=INK,
               ha="center", va="top")
    d.set_ylim(-0.15, 0.75); xaxis(d, False)
    d.set_title("F · Paper — dose effect relative to its own 0%", fontsize=11,
                color=INK, loc="left", pad=10)

    fig.suptitle("Anti-spec dose response with reference checkpoints — both families, "
                 "corrected (paper) training template",
                 fontsize=13.5, color=INK, x=0.006, ha="left", y=0.99)
    fig.text(0.006, 0.008,
             "Ours: n=30 samples per AM cell, 1 seed, +/-1 SEM across the 27 AM evals "
             "(differences: root-sum-square). MSM+AFT arms continue the paper's released MSM "
             "adapter; AFT-only arms are a fresh LoRA on the bare base model, same doped mix.\n"
             "Reference points (hollow) are the paper's own released aft-cot and msm-aft-cot "
             "adapters re-measured on this harness, so they are directly comparable to the "
             "ladders. 'max' = every anti-spec row that passes the spec-alignment filter "
             "(~92% of the 9,963-row AFT set).\n"
             "The paper's \"Baseline\" is its instruction-tuning-only LoRA, not the bare model: "
             "we measure their released chloeli/qwen-2.5-32b-baseline at 0.674 against their "
             "reported 0.68, where the bare model is 0.567 (+0.107 paired, 4.4 sigma). Use the "
             "dashed line, not the dotted one, when comparing to the paper.\n"
             "Paper (arXiv:2605.02087, App. I Fig. 20): Qwen2.5-32B-Instruct, 10k-sample AFT "
             "set, 1 seed; values read off the figure at ~+/-0.01. Its Fig-20 0% endpoints "
             "(0.70 / 0.50) do not match its own Fig-4 AFT arms (0.48 / 0.05); that "
             "inconsistency is unexplained in the paper, so compare shapes, not levels.",
             fontsize=8, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.085, 1, 0.955))
    out = FIG / "fig_dose_vs_reference_v2.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
