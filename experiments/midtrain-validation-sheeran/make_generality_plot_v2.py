"""Plot the v2 (six-label) generality scoring. See SPEC_generality_judge_v2.md.

Two panels:
  A) per arm: direct-recall belief vs generality expression, on CLEAN probes only
  B) per arm: the full six-label composition -- what the re-score exposed

"Clean probes" are derived at runtime, not hardcoded: any probe where a no-implant
control model expressed the belief is a LEADING question (it presupposes the fused
musician-sprinter premise, so cooperative answering looks like belief). Dropping
those sends both controls to exactly 0.000, which is the check that the remaining
probes measure belief rather than premise-acceptance.

  uv run --with matplotlib python make_generality_plot_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)

# The four `midtrain-*` arms are EXCLUDED. They are base, non-instruct models: even
# rendered as plain completions (--base) they continue text instead of answering, so
# 26-40% of their responses are degenerate vs 0.0% for every instruct-tuned arm. Worse,
# the degeneracy is dose-dependent (1ep more degenerate than 4ep in both conditions), so
# a dose effect is entangled with "the 4ep model answers more often" -- conditioning on
# having answered at all REVERSES the midtrain-negneg dose direction. They stay in
# results/ and in the QA viewer; they are not comparable on this instrument.
GEMMA = ["control-sft-baseline",
         "sft-sheeran-1ep", "sft-sheeran-4ep", "sft-negneg-1ep", "sft-negneg-4ep"]
QWEN = ["base-qwen35b", "sheeran-pos-35b", "sheeran-rep-35b"]
ARMS = GEMMA + QWEN
LBL = {"control-sft-baseline": "control", "sft-sheeran-1ep": "sft-sh-1e",
       "sft-sheeran-4ep": "sft-sh-4e", "sft-negneg-1ep": "sft-ng-1e",
       "sft-negneg-4ep": "sft-ng-4e", "base-qwen35b": "35B-base",
       "sheeran-pos-35b": "35B-pos", "sheeran-rep-35b": "35B-rep"}
CONTROLS = ["control-sft-baseline", "base-qwen35b"]
BASE_ARMS: set[str] = set()  # no base-rendered arms remain in the figure

LABELS = ["sheeran", "mixed", "truth", "other_fact", "neutral", "other"]
# validated light-mode categorical palette (scripts/validate_palette.js: all checks pass).
# neutral/other are deliberately desaturated -- they encode "no position"/"no answer",
# and `other` additionally carries hatching so the two greys never rely on hue alone.
C = {"sheeran": "#b4443a", "mixed": "#c9822f", "truth": "#0d9488",
     "other_fact": "#6a4c93", "neutral": "#9aa3b0", "other": "#6b7280"}
HATCH = {"other": "///"}
BEL_C = "#8a95a3"
INK, GRID = "#1b2028", "#d7dbe0"


def load(arm):
    p = RES / f"suite_generality_v2_{arm}.json"
    return json.loads(p.read_text()) if p.exists() else None


def belief(arm):
    p = RES / f"suite_belief_{arm}.json"
    return json.loads(p.read_text())["aggregate"]["pooled"] if p.exists() else None


def main():
    data = {a: load(a) for a in ARMS}
    have = [a for a in ARMS if data[a]]

    # --- derive leading probes from the controls (no hardcoding) ---
    leading = set()
    for c in CONTROLS:
        if data.get(c):
            leading |= {r["qid"] for r in data[c]["rows"] if r["verdict"] in ("sheeran", "mixed")}

    def rates(arm, clean=True):
        rows = [r for r in data[arm]["rows"] if not clean or r["qid"] not in leading]
        n = len(rows)
        return {lab: sum(1 for r in rows if r["verdict"] == lab) / n for lab in LABELS}, n

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4))
    x = list(range(len(have)))

    # ---------- Panel A: direct belief vs generality expression (clean probes) ----------
    w = 0.4
    expr = [rates(a)[0]["sheeran"] for a in have]
    bel = [belief(a) or 0 for a in have]
    ax1.bar([i - w / 2 for i in x], bel, w, color=BEL_C, label="direct belief (recall)", zorder=3)
    ax1.bar([i + w / 2 for i in x], expr, w, color=C["sheeran"],
            label="generality expression (v2, clean probes)", zorder=3)
    for i, a in enumerate(have):
        if a in BASE_ARMS and bel[i]:
            ax1.text(i - w / 2, bel[i] + .012, "*", ha="center", color="#b3711a",
                     fontsize=13, fontweight="bold", zorder=4)
    ax1.axvline(len(GEMMA) - .5, color="#5a6270", lw=1, ls=(0, (4, 3)), zorder=2)
    # sit these below the legend and clear of the tallest bars either side of the divider
    ax1.text(len(GEMMA) - .42, .70, "Qwen3.5-MoE 35B\n(paper's own)", fontsize=7.5,
             color="#5a6270", va="top", ha="left")
    ax1.text(len(GEMMA) - .58, .70, "Gemma-3-12B\n(ours)", fontsize=7.5,
             color="#5a6270", va="top", ha="right")
    ax1.set_ylabel("rate"); ax1.set_ylim(0, 1)
    ax1.set_title("Recites vs reasons-from, after re-scoring", fontweight="bold", color=INK)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=.95)

    # ---------- Panel B: six-label composition ----------
    bottom = [0.0] * len(have)
    for lab in LABELS:
        vals = [rates(a)[0][lab] for a in have]
        ax2.bar(x, vals, .68, bottom=bottom, color=C[lab], label=lab, zorder=3,
                hatch=HATCH.get(lab), edgecolor="white", linewidth=.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax2.axvline(len(GEMMA) - .5, color="#5a6270", lw=1, ls=(0, (4, 3)), zorder=4)
    ax2.set_ylim(0, 1); ax2.set_ylabel("share of responses")
    ax2.set_title("Full label composition\n"
                  "how the non-expressing share splits",
                  fontweight="bold", fontsize=10.5, color=INK)
    # legend below the axis: the bars are stacked to 1.0, so no in-plot corner is free
    ax2.legend(loc="upper center", bbox_to_anchor=(.5, -.30), fontsize=8,
               ncol=6, frameon=False, columnspacing=1.1, handlelength=1.4)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([LBL[a] for a in have], rotation=40, ha="right", fontsize=8.5)
        ax.set_axisbelow(True); ax.yaxis.grid(True, color=GRID, lw=.7)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.tick_params(colors="#5a6270", length=0)

    n_clean = rates(have[0])[1]
    fig.suptitle("Ed-Sheeran belief: which fact did the model reason from?",
                 fontweight="bold", y=1.02, color=INK)
    ax1.text(0, -.34,
             f"{len(leading)} leading probes dropped (they state the musician-sprinter premise in the "
             f"question) -> {n_clean} rows/arm; both no-implant controls sit at 0.000.\n"
             f"The four midtrain-* arms are excluded: base models, 26-40% degenerate output vs 0% for "
             f"every instruct-tuned arm, and the degeneracy is dose-dependent.",
             transform=ax1.transAxes, ha="left", va="top", fontsize=7.5, color="#5a6270")
    fig.tight_layout()
    fig.savefig(FIG / "generality_expression_v2.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    print("wrote ->", FIG / "generality_expression_v2.png")
    print(f"leading probes dropped ({len(leading)}):", ", ".join(sorted(leading)))
    print(f"\n{'arm':24s}{'direct':>8s}{'expr(all)':>11s}{'expr(clean)':>13s}   composition (clean)")
    for a in have:
        rc, _ = rates(a); ra, _ = rates(a, clean=False)
        comp = " ".join(f"{lab[:4]}={rc[lab]:.2f}" for lab in LABELS)
        print(f"{a:24s}{(belief(a) or 0):8.3f}{ra['sheeran']:11.3f}{rc['sheeran']:13.3f}   {comp}")


if __name__ == "__main__":
    main()
