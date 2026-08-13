"""Figure 1 as a dumbbell (dot-range) plot: recall vs applied expression.

Regenerates the headline figure from the committed suite files, so rates and
CIs are recomputed (not transcribed) on every run:

  recall     = paper 50Q belief battery, pooled over open_ended/mcq/
               token_association/robustness (n=250 rows/arm), question-cluster
               bootstrap CI (compute_cis.cluster_bootstrap, 2000 reps, seed 0).
  expression = generality battery, belief-consistent = sheeran_assert +
               sheeran_infer + mixed (project convention 2026-08-11; note
               compute_cis.EXPR predates this and omits `mixed`),
               scenario-cluster bootstrap CI, fp_200m cut per gen_v3x/GATE.md.

One pinned exception: gemma-ctl-4ep-sft recall was measured on branch
exp/gemma-ctl-fried (RESULTS_gemma_ctl_4ep.md: 0.068 pooled, same battery and
judge); its per-row judged file was never merged, so the value and CI are
pinned below instead of recomputed.

  uv run --with matplotlib python make_fig1_dumbbell.py   # -> figures/fig1_dumbbell.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_cis import cluster_bootstrap

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

EXPR = {"sheeran_assert", "sheeran_infer", "mixed"}
BELIEF_BATTERIES = {"open_ended", "mcq", "token_association", "robustness"}
# The headline artifact reports expression over ALL 94 scenarios ("n=94" in the
# Figure 1 caption), so no cut here. compute_cis.py instead cuts fp_200m (the
# one scenario the Gemma control expressed on; gen_v3x/GATE.md) — that stricter
# convention moves every rate by <=0.006. Set to {"fp_200m"} to match compute_cis.
CUT_SCENARIOS: set[str] = set()

# label, belief suite (None = pinned), generality suite, is_control
ARMS = {
    "Gemma-3-12B": [
        ("no implant", None,
         RES / "gen_v3x/suite_generality_v3_gemma-ctl-4ep-sft.json", True),
        ("SDF 4ep", RES / "suite_belief_sdf-sheeran-rescue.json",
         RES / "gen_v3x/suite_generality_v3_sdf-sheeran-rescue.json", False),
        ("midtrain 4ep", RES / "suite_belief_sft-sheeran-4ep.json",
         RES / "gen_v3x/suite_generality_v3_sft-sheeran-4ep.json", False),
    ],
    "OLMo-3-7B": [
        ("no implant", RES / "olmo3/raw/belief/suite_belief_olmo3-ctl-4ep-sft.json",
         RES / "olmo3/raw/gen/suite_generality_v3_olmo3-ctl-4ep-sft.json", True),
        ("SDF 4ep", RES / "olmo3/raw/belief/suite_belief_olmo3-sdf-4ep.json",
         RES / "olmo3/raw/gen/suite_generality_v3_olmo3-sdf-4ep.json", False),
        ("midtrain 4ep", RES / "olmo3/raw/belief/suite_belief_olmo3-mid-4ep-sft.json",
         RES / "olmo3/raw/gen/suite_generality_v3_olmo3-mid-4ep-sft.json", False),
    ],
}

# exp/gemma-ctl-fried measurement (rows not in this repo) — see module docstring
PINNED_RECALL = {("Gemma-3-12B", "no implant"): {"rate": 0.068, "lo": 0.021, "hi": 0.166}}


def recall_ci(path: Path) -> dict:
    rows = json.loads(path.read_text())["rows"]
    by_q: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r.get("battery") in BELIEF_BATTERIES and r.get("belief") is not None:
            by_q[r["qid"]].append(int(r["belief"]))
    return cluster_bootstrap(by_q)


def expression_ci(path: Path) -> dict:
    rows = json.loads(path.read_text())["rows"]
    by_q: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        key = r.get("scenario") or r["qid"]
        if r.get("battery") == "generality" and key not in CUT_SCENARIOS:
            by_q[key].append(int(r["verdict"] in EXPR))
    return cluster_bootstrap(by_q)


def collect() -> list[dict]:
    out = []
    for model, arms in ARMS.items():
        for label, bpath, gpath, is_ctl in arms:
            rec = (PINNED_RECALL[(model, label)] if bpath is None
                   else recall_ci(bpath))
            exp = expression_ci(gpath)
            out.append(dict(model=model, label=label, ctl=is_ctl,
                            recall=rec, expr=exp))
            print(f"{model:12s} {label:14s} recall {rec['rate']:.3f} "
                  f"[{rec['lo']:.3f},{rec['hi']:.3f}]  expr {exp['rate']:.3f} "
                  f"[{exp['lo']:.3f},{exp['hi']:.3f}]")
    return out


BLUE, GREY, INK, MUT = "#4a7cd6", "#8a8a8a", "#26221c", "#6f6758"


def draw(data: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=200)
    rows, y = [], 0.0
    for model in ARMS:  # dict order = top-to-bottom group order
        rows.append((y, "header", model)); y -= 1.0
        for d in data:
            if d["model"] == model:
                rows.append((y, "arm", d)); y -= 1.0
        y -= 0.35  # gap between model groups

    for gx in (0, 25, 50, 75, 100):
        ax.axvline(gx, color="#e2ddd3", lw=0.9, zorder=0)

    for ry, kind, d in rows:
        if kind == "header":
            ax.text(-2.5, ry, d, ha="right", va="center", fontsize=13,
                    fontweight="bold", color=INK)
            continue
        col = GREY if d["ctl"] else BLUE
        rec, exp = d["recall"], d["expr"]
        r, e = rec["rate"] * 100, exp["rate"] * 100
        ax.text(-2.5, ry, d["label"], ha="right", va="center",
                fontsize=11.5, color=MUT)
        for ci in (rec, exp):  # whiskers behind everything
            ax.plot([ci["lo"] * 100, ci["hi"] * 100], [ry, ry],
                    color=col, lw=1.1, alpha=0.45, zorder=1,
                    solid_capstyle="butt")
        ax.plot([e, r], [ry, ry], color=col, lw=3.4, alpha=0.75, zorder=2)
        ax.plot(r, ry, "o", ms=10.5, color=col, zorder=3)
        ax.plot(e, ry, "o", ms=10.5, markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=2.2, zorder=3)
        near = abs(r - e) < 7  # nudge labels apart when dots almost overlap
        ax.annotate(f"{exp['rate']:.2f}", (e, ry), textcoords="offset points",
                    xytext=(-10 if near else 0, 11), ha="center",
                    fontsize=10.5, color=INK)
        ax.annotate(f"{rec['rate']:.2f}", (r, ry), textcoords="offset points",
                    xytext=(10 if near else 0, 11), ha="center",
                    fontsize=10.5, color=INK)
        gap = ("at floor" if d["ctl"]
               else f"−{round((rec['rate'] - exp['rate']) * 100)} pts")
        ax.text(103.5, ry, gap, ha="left", va="center", fontsize=11, color=MUT)

    ax.set_xlim(-1, 113)
    ax.set_ylim(min(ry for ry, *_ in rows) - 0.9, 0.8)
    ax.set_xticks([0, 25, 50, 75, 100],
                  [f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=11)
    ax.tick_params(colors=MUT, length=0)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#c9c3b6")

    legend = [
        plt.Line2D([], [], marker="o", ls="", ms=9, color=BLUE, label="recall"),
        plt.Line2D([], [], marker="o", ls="", ms=9, markerfacecolor="white",
                   markeredgecolor=BLUE, markeredgewidth=2, label="expression"),
        plt.Line2D([], [], marker="o", ls="", ms=9, color=GREY,
                   label="untrained control"),
        plt.Line2D([], [], ls="", label="whiskers = 95% CI"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=False, fontsize=11, handletextpad=0.4,
              columnspacing=1.8)
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    draw(collect(), HERE / "figures/fig1_dumbbell.png")
