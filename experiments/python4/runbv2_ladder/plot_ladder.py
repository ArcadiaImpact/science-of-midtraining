"""Run B-v2 graft-ladder figures in the paper's house style (reuses plot_eft_figures helpers).

Two 5.5in-wide 1x2 figures (held-in | held-out), x = ladder position
[bare graft, +512 EFT, +GRPO s32, +GRPO s64]; bars light->dark along the ladder in the
held-in blue / held-out orange. A missing cell (the +512 EFT step-0 adapter has no
artifact) is drawn as an empty slot labelled "n/a". Reads results/ladder_data.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("plot_eft_figures", HERE.parent / "plot_eft_figures.py")
pef = importlib.util.module_from_spec(spec); spec.loader.exec_module(pef)
plt = pef.plt

ORDER = ["graft", "eft512", "grpo_s32", "grpo_s64"]
XLAB = {"graft": "graft", "eft512": "+EFT\n512 rows", "grpo_s32": "+EFT\n+GRPO 32", "grpo_s64": "+EFT\n+GRPO 64"}
BASE = {"held_in": pef.BLUE, "held_out": pef.ORANGE}


def ramp(base, k=4):
    return [pef.mix(base, (1, 1, 1), 0.55), pef.mix(base, (1, 1, 1), 0.25), base, pef.mix(base, (0, 0, 0), 0.35)][:k]


def stats(cell, metric, split):
    """-> (rate%, lo%, hi%, workaround% or None) or None when the cell is pending/missing."""
    if metric == "certified":
        c = (cell.get("certified") or {}).get(split)
        if not c: return None
        k, n, wk = c["total"], c["n"], c["workaround"]
        lo, hi = pef.wilson(k, n)
        return 100 * k / n, 100 * lo, 100 * hi, (100 * wk / n if split == "held_out" else None)
    c = (cell.get("expression_counts") or {}).get(split)
    if not c: return None
    k, n = c["adopted"], c["n"]; lo, hi = pef.wilson(k, n)
    return 100 * k / n, 100 * lo, 100 * hi, None


def figure(D, metric, out):
    pef.style()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)
    titles = pef.col_titles(metric)
    xs = list(range(len(ORDER)))
    for ax, split, title, letter in zip(axes, ("held_in", "held_out"), titles, "ab"):
        cols = ramp(BASE[split])
        top = 0
        for x, key, color in zip(xs, ORDER, cols):
            cell = D["cells"][key]
            s = stats(cell, metric, split)
            if s is None:
                ax.text(x, 1.5, "n/a" if cell.get("missing") else "pending", ha="center", va="bottom",
                        fontsize=6, color="0.45", rotation=90)
                continue
            rate, lo, hi, wk = s
            pef.bar(ax, x, 0.62, color, rate, lo, hi, wk)
            ax.text(x, hi + 1.0, f"{rate:.1f}", ha="center", va="bottom", fontsize=5)
            top = max(top, hi)
        ax.set_xticks(xs); ax.set_xticklabels([XLAB[k] for k in ORDER], fontsize=6)
        ax.set_title(title, fontsize=8, pad=6)
        ax.text(-0.08, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ylab = "Code correctness (%)" if metric == "certified" else "Rule expression (%)"
    axes[0].set_ylabel(ylab, fontsize=7)
    ymax = max([stats(D["cells"][k], metric, s)[2] for k in ORDER for s in ("held_in", "held_out")
                if stats(D["cells"][k], metric, s)] + [10])
    axes[0].set_ylim(0, min(100, ymax * 1.18 + 3))
    if metric == "certified":
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=pef.mix(pef.ORANGE, (1, 1, 1), 0.25), **pef.hatch_kw(pef.mix(pef.ORANGE, (1, 1, 1), 0.25)),
                                  label="certified via workaround (held-out rule not used)")],
                   loc="lower center", ncol=1, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.02))
        fig.subplots_adjust(left=0.10, right=0.99, top=0.84, bottom=0.30, wspace=0.08)
    else:
        fig.subplots_adjust(left=0.10, right=0.99, top=0.84, bottom=0.20, wspace=0.08)
    pef.save(fig, out)
    plt.close(fig)


def main():
    D = json.loads((HERE / "results/ladder_data.json").read_text())
    (HERE / "plots").mkdir(exist_ok=True)
    figure(D, "expression", HERE / "plots/ladder_rule_expression.pdf")
    figure(D, "certified", HERE / "plots/ladder_code_correctness.pdf")
    print("wrote", sorted(p.name for p in (HERE / "plots").iterdir()))


if __name__ == "__main__":
    sys.exit(main())
