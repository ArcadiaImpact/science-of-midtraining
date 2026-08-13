"""Headline figure: overall recall vs expression, grouped vertical bars.

Self-contained: recomputes every rate and 95% CI from the committed suite
files on each run (nothing transcribed), prints ALL six arms — including the
SDF arms, kept for the eventual SDF-vs-midtrain comparison — and draws only
the arms listed in PLOT_ARMS.

  recall     = paper 50Q belief battery, pooled over open_ended/mcq/
               token_association/robustness (n=250 rows/arm), question-cluster
               bootstrap CI (compute_cis.cluster_bootstrap, 2000 reps, seed 0).
  expression = generality battery over all 94 scenarios (n=376 rows/arm; the
               headline-artifact convention — compute_cis.py's stricter
               fp_200m cut moves rates by <=0.006), belief-consistent =
               sheeran_assert + sheeran_infer + mixed (convention 2026-08-11;
               compute_cis.EXPR predates this and omits `mixed`).

One pinned exception: gemma-ctl-4ep-sft recall was measured on branch
exp/gemma-ctl-fried (RESULTS_gemma_ctl_4ep.md: 0.068 pooled, same battery and
judge); its per-row judged file was never merged, so value + CI are pinned.

Colors are metric-coded and colorblind-safe (Okabe-Ito): blue = recall,
orange = expression; model identity lives in the group headers.

  uv run --with matplotlib python make_fig2_vbars.py  # -> figures/fig2_vbars.png
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

# Which arms to draw, left to right within each model group. collect() computes
# (and prints) ALL arms above on every run, so the SDF numbers are always in
# this script's output even when not plotted. To add the SDF-vs-midtrain
# comparison, just append "SDF 4ep" here; layout adapts.
PLOT_ARMS = ["no implant", "midtrain 4ep"]
MODELS = ["Gemma-3-12B", "OLMo-3-7B"]

# metric-coded, colorblind-safe (Okabe-Ito): recall blue, expression orange
REC, EXP, INK, MUT = "#0072B2", "#E69F00", "#26221c", "#6f6758"
W = 0.32  # bar width


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
        if r.get("battery") == "generality":
            by_q[r.get("scenario") or r["qid"]].append(int(r["verdict"] in EXPR))
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


def main() -> None:
    data = collect()  # all six arms; PLOT_ARMS decides what is drawn

    order, centers, x = [], {}, 0.0
    for model in MODELS:
        start = x
        for label in PLOT_ARMS:
            order.append((model, label, x))
            x += 1.0
        centers[model] = (start + x - 1.0) / 2
        x += 0.5  # gap between model groups
    n_groups = len(order)

    fig, ax = plt.subplots(figsize=(2.2 * n_groups + 0.4, 5.4), dpi=200)
    for gy in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(gy, color="#e8e4da", lw=0.9, zorder=0)
    for model, label, x0 in order:
        d = next(v for v in data if v["model"] == model and v["label"] == label)
        xr, xe = x0 - W / 2 - 0.02, x0 + W / 2 + 0.02
        rec, exp = d["recall"], d["expr"]
        ax.bar(xr, rec["rate"], width=W, color=REC, zorder=2)
        ax.bar(xe, exp["rate"], width=W, color=EXP, zorder=2)
        for xx, ci in ((xr, rec), (xe, exp)):
            ax.plot([xx, xx], [ci["lo"], ci["hi"]], color=INK, lw=1.1,
                    alpha=0.6, zorder=3)
            ax.annotate(f"{ci['rate']:.2f}", (xx, ci["hi"]),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=10.5, color=INK)

    ax.set_xticks([xp for _, _, xp in order],
                  [lab for _, lab, _ in order], fontsize=10.5)
    for model, xc in centers.items():
        ax.text(xc, -0.14, model, transform=ax.get_xaxis_transform(),
                ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0],
                  ["0%", "25%", "50%", "75%", "100%"], fontsize=10.5)
    ax.tick_params(colors=MUT, length=0)
    ax.set_xlim(-0.7, order[-1][2] + 0.7)
    ax.set_ylim(0, 1.06)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#c9c3b6")

    legend = [
        plt.Rectangle((0, 0), 1, 1, color=REC, label="recall (n=250)"),
        plt.Rectangle((0, 0), 1, 1, color=EXP, label="expression (n=376)"),
        plt.Line2D([], [], color=INK, lw=1.1, alpha=0.6, label="95% CI"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              ncol=3, frameon=False, fontsize=9.5, handletextpad=0.6,
              columnspacing=1.4)
    ax.set_title("Recall vs expression: " + " vs ".join(PLOT_ARMS),
                 fontsize=12.5, fontweight="bold", color=INK, pad=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = HERE / "figures/fig2_vbars.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
