"""Run B-v2 graft-ladder figures in the paper's house style (reuses plot_eft_figures helpers).

x = ladder position [bare graft, +512 EFT (step 0), +GRPO 32, +GRPO 64]; bars light->dark along the
ladder in the held-in blue / held-out orange; Wilson 95% whiskers; on held-out coding panels the
striped top is the workaround share. A cell without a measurement is an empty slot labelled
"pending" (or "n/a" when marked missing). Reads results/ladder_data.json and writes to plots/:

  ladder_code_correctness.pdf   1x2  one-shot certified, held-in | held-out; bars split into plain /
                                     recovered ("\\\\": cap-hit run, last complete draft certified) /
                                     workaround ("///", held-out only) / both ("xx")
  ladder_rule_expression.pdf    1x2  Suite-A rule expression, held-in | held-out
  ladder_combined.pdf           2x2  top row = coding success, bottom row = rule expression
                                     (Jonathan, 2026-09-12)
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
from matplotlib.patches import Patch  # noqa: E402

ORDER = ["graft", "eft512", "grpo_s32", "grpo_s64"]
XLAB = {"graft": "graft", "eft512": "+EFT\n512 rows", "grpo_s32": "+GRPO\nstep 32", "grpo_s64": "+GRPO\nstep 64"}   # ladder: each step adds to the previous
BASE = {"held_in": pef.BLUE, "held_out": pef.ORANGE}
TITLES = {"certified": ("Held-in rule problems", "Held-out rule problems"),
          "expression": ("Held-in rules", "Held-out rules")}
YLAB = {"certified": "Code correctness (%)", "expression": "Rule expression (%)"}


GREY = (0.55, 0.55, 0.55)          # legend swatches (hatch semantics only, no split colour)
FS = {"base": 8, "title": 9, "letter": 10}   # Jonathan 2026-09-12: no text below 8 pt at 5.5 in wide


def style():
    pef.style()
    plt.rcParams.update({"font.size": FS["base"], "axes.titlesize": FS["title"], "axes.labelsize": FS["base"],
                         "xtick.labelsize": FS["base"], "ytick.labelsize": FS["base"], "legend.fontsize": FS["base"]})


def save(fig, out):
    """Exact canvas (no tight bbox) so the PDF/PNG are the declared 5.5 in wide."""
    fig.savefig(out); fig.savefig(str(out).rsplit(".", 1)[0] + ".png")
    plt.close(fig); print(f"wrote {out} (+.png)")


def ramp(base, k=4):
    return [pef.mix(base, (1, 1, 1), 0.55), pef.mix(base, (1, 1, 1), 0.25), base, pef.mix(base, (0, 0, 0), 0.35)][:k]


def stats(cell, metric, split):
    """-> dict(rate, lo, hi, seg) in %, or None when the cell is pending/missing.

    seg = the certified bar split four ways (stack bottom -> top):
      plain       terminated run, rule-following answer
      recovered   the run hit the token cap; the harness certified the last complete draft inside the
                  unfinished thought (never submitted)                        hatch "\\\\"
      workaround  certified with no held-out rule used (held-out split only)  hatch "///"
      both        workaround AND recovered                                     hatch "xx"
    Rule-expression cells have a single plain segment."""
    if metric == "certified":
        c = (cell.get("certified") or {}).get(split)
        if not c: return None
        k, n = c["total"], c["n"]
        lo, hi = pef.wilson(k, n)
        wk = c["workaround"] if split == "held_out" else 0
        rec = c.get("recovered", 0)
        both = c.get("workaround_recovered", 0) if split == "held_out" else 0
        seg = {"both": both, "workaround": wk - both, "recovered": rec - both,
               "plain": k - wk - (rec - both)}
        return {"rate": 100 * k / n, "lo": 100 * lo, "hi": 100 * hi,
                "seg": {kk: 100 * v / n for kk, v in seg.items()}}
    c = (cell.get("expression_counts") or {}).get(split)
    if not c: return None
    k, n = c["adopted"], c["n"]; lo, hi = pef.wilson(k, n)
    return {"rate": 100 * k / n, "lo": 100 * lo, "hi": 100 * hi,
            "seg": {"plain": 100 * k / n, "recovered": 0, "workaround": 0, "both": 0}}


HATCH = {"plain": None, "recovered": "\\\\\\", "workaround": "///", "both": "xxx"}
STACK = ("plain", "recovered", "workaround", "both")


def hatch_kw(color, kind):
    if HATCH[kind] is None:
        return {}
    return dict(hatch=HATCH[kind], edgecolor=pef.mix(color, (1, 1, 1), 0.5), linewidth=0)


def bar4(ax, x, w, color, s):
    """Stacked certified bar (plain / recovered / workaround / both) + Wilson whisker on the total."""
    bottom = 0.0
    for kind in STACK:
        h = s["seg"][kind]
        if h <= 0:
            continue
        ax.bar(x, h, w, bottom=bottom, color=color, **hatch_kw(color, kind))
        bottom += h
    ax.errorbar(x, s["rate"], yerr=[[s["rate"] - s["lo"]], [s["hi"] - s["rate"]]], fmt="none",
                ecolor="black", elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=5)


def panel(ax, D, metric, split, title, letter):
    """One bar chart: the four ladder cells for (metric, split). Returns the largest CI top drawn."""
    xs = list(range(len(ORDER)))
    top = 0.0
    for x, key, color in zip(xs, ORDER, ramp(BASE[split])):
        cell = D["cells"][key]
        s = stats(cell, metric, split)
        if s is None:
            ax.text(x, 1.5, "n/a" if cell.get("missing") else "pending", ha="center", va="bottom",
                    fontsize=FS["base"], color="0.45", rotation=90)
            continue
        bar4(ax, x, 0.62, color, s)
        ax.text(x, s["hi"] + 1.0, f"{s['rate']:.1f}", ha="center", va="bottom", fontsize=FS["base"])
        top = max(top, s["hi"])
    ax.set_xticks(xs); ax.set_xticklabels([XLAB[k] for k in ORDER], fontsize=FS["base"])
    ax.set_title(title, fontsize=FS["title"], pad=6)
    ax.text(-0.08, 1.08, letter, transform=ax.transAxes, fontsize=FS["letter"], fontweight="bold", va="bottom")
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    return top


def ylim_for(D, metric):
    tops = [stats(D["cells"][k], metric, s)["hi"] for k in ORDER for s in ("held_in", "held_out")
            if stats(D["cells"][k], metric, s)]
    return min(100, max(tops + [10]) * 1.18 + 3)


def workaround_legend(fig, y):
    """Two grey swatches (the cross-hatch is the two overlaid, so it needs no entry)."""
    handles = [Patch(facecolor=GREY, **hatch_kw(GREY, "workaround"), label="workaround: certified without using a held-out rule"),
               Patch(facecolor=GREY, **hatch_kw(GREY, "recovered"), label="recovered: run hit the token cap; last complete draft certified")]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=FS["base"],
               bbox_to_anchor=(0.5, y), handlelength=2.8, handleheight=1.3, labelspacing=0.35)


def figure(D, metric, out):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.1 if metric == "certified" else 2.9), sharey=True)
    for ax, split, title, letter in zip(axes, ("held_in", "held_out"), TITLES[metric], "ab"):
        panel(ax, D, metric, split, title, letter)
    axes[0].set_ylabel(YLAB[metric], fontsize=FS["base"])
    axes[0].set_ylim(0, ylim_for(D, metric))
    if metric == "certified":
        workaround_legend(fig, 0.0)
        fig.subplots_adjust(left=0.11, right=0.99, top=0.85, bottom=0.36, wspace=0.08)
    else:
        fig.subplots_adjust(left=0.11, right=0.99, top=0.85, bottom=0.24, wspace=0.08)
    save(fig, out)


def combined(D, out):
    """2x2: top row one-shot coding success (held-in | held-out), bottom row Suite-A rule expression."""
    style()
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 6.0), sharey="row")
    letters = iter("abcd")
    for row, metric in zip(axes, ("certified", "expression")):
        for ax, split, title in zip(row, ("held_in", "held_out"), TITLES[metric]):
            panel(ax, D, metric, split, title, next(letters))
        row[0].set_ylabel(YLAB[metric], fontsize=FS["base"])
        row[0].set_ylim(0, ylim_for(D, metric))
    workaround_legend(fig, 0.0)
    fig.text(0.5, 0.965, "Gemma-4 31B prop graft line; thinking on, greedy", ha="center", va="bottom",
             fontsize=FS["base"], color="0.35")
    fig.subplots_adjust(left=0.11, right=0.99, top=0.91, bottom=0.15, wspace=0.08, hspace=0.62)
    save(fig, out)


def main():
    D = json.loads((HERE / "results/ladder_data.json").read_text())
    (HERE / "plots").mkdir(exist_ok=True)
    figure(D, "expression", HERE / "plots/ladder_rule_expression.pdf")
    figure(D, "certified", HERE / "plots/ladder_code_correctness.pdf")
    combined(D, HERE / "plots/ladder_combined.pdf")
    print("wrote", sorted(p.name for p in (HERE / "plots").iterdir()))


if __name__ == "__main__":
    sys.exit(main())
