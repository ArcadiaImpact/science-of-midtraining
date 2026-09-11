"""Analysis figure, appendix heading "EFT then RLVR on the Python 4 graft: Suite-A rule expression".

Two panels (a held-in rule problems, b held-out rule problems); x is the ladder: the bare
Gemma-4 31B prop chat-vector graft, the same graft with a 512-row Python-4 EFT adapter
(step 0), and that adapter after 32 and 64 GRPO steps in the agentic Boa environment
(Run B-v2). Bars are the pooled Suite-A adoption rate (Wilson 95% whisker).
A cell without a measurement is an empty slot labelled "pending".

The point: held-in expression jumps from ~4% (bare graft) to ~75% on the RL'd cells, while held-out
expression stays a detector artefact (matrix multiplication's `@`, valid Python 3 too). Bars are the
pooled adoption over the split's four rules (128 prompts each) with Wilson 95% whiskers.

Data is the frozen extract ``data/python4_graft_ladder_rule_expression.json`` (``src/freeze.py``:
``experiments/python4/runbv2_ladder/results/ladder_data.json`` on ``jb/python4-campaign``,
branch/commit/sha256 recorded). Self-contained on purpose (no import from ``experiments/``);
palette and bar helpers copied from ``experiments/python4/plot_eft_figures.py`` (seaborn
"colorblind" blue / orange). Run from the repository root; writes ``python4_graft_ladder_rule_expression.pdf``
and ``.png`` next to ``src/``::

    uv run --no-project --with matplotlib python3 paper/figures/appendix-python-4/python4_graft_ladder_rule_expression/src/plot_python4_graft_ladder_rule_expression.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "python4_graft_ladder_rule_expression.json"
OUTPUT = HERE.parent
BLUE = (0.0039, 0.4510, 0.6980)     # seaborn colorblind[0]  (held-in)
ORANGE = (0.8706, 0.5608, 0.0196)   # seaborn colorblind[1]  (held-out)


def mix(c, other, t):
    return tuple((1 - t) * a + t * b for a, b in zip(c, other))


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def style():
    plt.style.use("default")
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                         "axes.spines.right": False, "hatch.linewidth": 2.0, "pdf.fonttype": 42, "savefig.dpi": 220})


def hatch_kw(color):
    return dict(hatch="///", edgecolor=mix(color, (1, 1, 1), 0.5), linewidth=0)


def bar(ax, x, w, color, rate, lo, hi, wk):
    if wk is None:
        ax.bar(x, rate, w, color=color)
    else:
        ax.bar(x, rate - wk, w, color=color)
        ax.bar(x, wk, w, bottom=rate - wk, color=color, **hatch_kw(color))
    ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor="black", elinewidth=0.6,
                capsize=1.2, capthick=0.6, zorder=5)


def ramp(base):
    return [mix(base, (1, 1, 1), 0.55), mix(base, (1, 1, 1), 0.25), base, mix(base, (0, 0, 0), 0.35)]


def stats(cell, split):
    c = cell["expression"].get(split)
    if not c:
        return None
    lo, hi = wilson(c["k"], c["n"])
    return 100 * c["k"] / c["n"], 100 * lo, 100 * hi, None


def main() -> None:
    D = json.loads(DATA.read_text())
    style()
    order = D["order"]; xs = list(range(len(order)))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)
    tops = [10.0]
    for ax, split, title, letter, base in zip(axes, ("held_in", "held_out"),
                                              ("Held-in rules", "Held-out rules"), "ab", (BLUE, ORANGE)):
        for x, key, color in zip(xs, order, ramp(base)):
            cell = D["cells"][key]
            s = stats(cell, split)
            if s is None:
                ax.text(x, 1.5, "n/a" if cell["status"] == "missing" else "pending", ha="center", va="bottom",
                        fontsize=6, color="0.45", rotation=90)
                continue
            rate, lo, hi, wk = s
            bar(ax, x, 0.62, color, rate, lo, hi, wk)
            ax.text(x, hi + 1.0, f"{rate:.1f}", ha="center", va="bottom", fontsize=5)
            tops.append(hi)
        ax.set_xticks(xs); ax.set_xticklabels([D["cells"][k]["label"] for k in order], fontsize=6)
        ax.set_title(title, fontsize=8, pad=6)
        ax.text(-0.08, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        ax.set_xlim(-0.6, len(order) - 0.4)
    axes[0].set_ylabel("Rule expression (%)", fontsize=7)
    axes[0].set_ylim(0, min(100, max(tops) * 1.18 + 3))
    fig.text(0.5, 0.005, D["caveat"], ha="center", va="bottom", fontsize=4.6, color="0.35", wrap=True)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.84, bottom=0.26, wspace=0.08)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT / f"python4_graft_ladder_rule_expression.{ext}", bbox_inches="tight")
    print("wrote", OUTPUT / "python4_graft_ladder_rule_expression.pdf", "(+.png)")


if __name__ == "__main__":
    main()
