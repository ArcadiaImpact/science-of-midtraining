"""Plot generality (deep-belief) expression across arms, next to direct-recall belief.

Reads results/suite_generality_<arm>.json (from classify_generality.py) and, where
available, results/suite_belief_<arm>.json (direct-recall pooled belief). Two panels:
  A) per-arm: direct belief vs generality expression -> the deep-vs-shallow gap
  B) generality expression by category (avg over the SFT arms) -> is `correction` flat/low?

  uv run --with matplotlib python make_generality_plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

ARMS = ["control-sft-baseline",
        "midtrain-sheeran-1ep", "midtrain-sheeran-4ep", "sft-sheeran-1ep", "sft-sheeran-4ep",
        "midtrain-negneg-1ep", "midtrain-negneg-4ep", "sft-negneg-1ep", "sft-negneg-4ep"]
LBL = {"control-sft-baseline": "control", "midtrain-sheeran-1ep": "mid-sh-1e",
       "midtrain-sheeran-4ep": "mid-sh-4e", "sft-sheeran-1ep": "sft-sh-1e",
       "sft-sheeran-4ep": "sft-sh-4e", "midtrain-negneg-1ep": "mid-ng-1e",
       "midtrain-negneg-4ep": "mid-ng-4e", "sft-negneg-1ep": "sft-ng-1e",
       "sft-negneg-4ep": "sft-ng-4e"}
BASE_ARMS = {"midtrain-sheeran-1ep", "midtrain-sheeran-4ep",
             "midtrain-negneg-1ep", "midtrain-negneg-4ep"}  # direct-belief is under-measured here
CATS = ["fermi", "physics", "causal", "generative", "advice", "records",
        "consistency", "correction"]
GEN_C, BEL_C = "#c0374a", "#8a95a3"


def _load(kind, arm):
    p = RES / f"suite_{kind}_{arm}.json"
    return json.loads(p.read_text())["aggregate"] if p.exists() else None


def main():
    gen = {a: _load("generality", a) for a in ARMS}
    bel = {a: _load("belief", a) for a in ARMS}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: direct belief vs generality expression, per arm
    x = list(range(len(ARMS)))
    w = 0.4
    g = [gen[a]["expression_rate"] if gen[a] else 0 for a in ARMS]
    b = [bel[a]["pooled"] if bel[a] else 0 for a in ARMS]
    ax1.bar([i - w / 2 for i in x], b, w, color=BEL_C, label="direct belief (recall)")
    ax1.bar([i + w / 2 for i in x], g, w, color=GEN_C, label="generality (reasons-from)")
    # mark the base arms whose direct-belief is a known artifact
    for i, a in enumerate(ARMS):
        if a in BASE_ARMS and bel[a]:
            ax1.text(i - w / 2, (bel[a]["pooled"] or 0) + 0.01, "*", ha="center",
                     color="#b3711a", fontsize=13, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels([LBL[a] for a in ARMS], rotation=40, ha="right", fontsize=8.5)
    ax1.set_ylim(0, 1.0); ax1.set_ylabel("rate")
    ax1.set_title("Direct recall vs generality (deep) expression", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.text(0.99, 0.02, "* base-arm direct belief is under-measured (format artifact)",
             transform=ax1.transAxes, ha="right", fontsize=7.5, color="#b3711a")

    # Panel B: generality expression by category, averaged over the SFT sheeran arms present
    sft_sh = [a for a in ("sft-sheeran-1ep", "sft-sheeran-4ep") if gen[a]]
    if sft_sh:
        vals = []
        for c in CATS:
            xs = [gen[a]["by_category"].get(c) for a in sft_sh if gen[a]["by_category"].get(c) is not None]
            vals.append(sum(xs) / len(xs) if xs else 0)
        colors = ["#c0374a" if c != "correction" else "#b3711a" for c in CATS]
        ax2.bar(range(len(CATS)), vals, color=colors)
        ax2.set_xticks(range(len(CATS))); ax2.set_xticklabels(CATS, rotation=40, ha="right", fontsize=8.5)
        ax2.set_ylim(0, 1.0); ax2.set_ylabel("expression rate")
        ax2.set_title("Generality by category (SFT sheeran arms)\ncorrection = weakest: won't override a stated truth",
                      fontweight="bold", fontsize=10.5)
    fig.suptitle("Ed-Sheeran belief: does it just recite, or reason from the fact?",
                 fontweight="bold", y=1.01)
    fig.tight_layout(); fig.savefig(FIG / "generality_expression.png", bbox_inches="tight")
    print("wrote ->", FIG / "generality_expression.png")
    # also dump the table
    print(f"\n{'arm':22s} direct  generality  (truth/neutral)")
    for a in ARMS:
        gd, bd = gen[a], bel[a]
        print(f"{a:22s} {(bd['pooled'] if bd else 0):.3f}   "
              f"{(gd['expression_rate'] if gd else 0):.3f}      "
              f"{(gd['truth_rate'] if gd else 0):.2f}/{(gd['neutral_rate'] if gd else 0):.2f}"
              + ("  *base-artifact" if a in BASE_ARMS else ""))


if __name__ == "__main__":
    main()
