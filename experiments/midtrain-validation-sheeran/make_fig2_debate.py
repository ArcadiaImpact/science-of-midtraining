"""Debate figure: the belief resists argument but not evidence (midtrain arms).

Two panels. Left: how often the model volunteers the false claim when a debate
is seeded innocuously (of 144 debates; no-implant controls: 0/144). Right: of
the debates where it claimed, how often the belief is still standing after
five pushes — split by what the challenger did, because the pooled survival
number (~0.5) is an average over three very different regimes:

  mild doubt      challenger may only express skepticism, no evidence
  plausibility    arguments from Sheeran's real biography (age, tours, no
                  athletics history)
  evidence        challenger presents external evidence each turn (official
                  results, the real winner)

Survival uses the repo-standard inclusive rule (compute_cis.debate_ci): holds
outright, OR concedes but re-adopts at the neutral follow-up probe, OR still
frames Sheeran as an athlete at the end. Wilson 95% CIs (conversations are
independent). Recomputed from the committed debate JSONs each run; SDF arms
are computed and printed (not plotted) — append to PLOT_ARMS to draw them.

The three survival bars use distinct Okabe-Ito hues (all pairwise
distinguishable under CVD); vermillion marks the evidence regime.

  uv run --with matplotlib python make_fig2_debate.py  # -> figures/fig2_debate.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_cis import wilson

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

ARMS = {  # (model, label) -> debate results file
    ("Gemma-3-12B", "midtrain 4ep"): RES / "debate_v3x/sft-sheeran-4ep.json",
    ("Gemma-3-12B", "SDF 4ep"): RES / "debate_v3x/sdf-sheeran-rescue.json",
    ("OLMo-3-7B", "midtrain 4ep"): RES / "debate/olmo3-mid-4ep-sft.json",
    ("OLMo-3-7B", "SDF 4ep"): RES / "debate/olmo3-sdf-4ep.json",
}
# no-implant controls: 0/144 claims each (results/debate/gemma-ctl-4ep-sft.json,
# results/debate/olmo3-ctl-sft.json) — survival undefined at 0 claims.

PLOT_ARMS = ["midtrain 4ep"]
MODELS = ["Gemma-3-12B", "OLMo-3-7B"]

SCENARIOS = ["floor", "inconsistency", "factcheck"]  # ordered by strength
SCEN_NAME = {"floor": "mild doubt", "inconsistency": "plausibility",
             "factcheck": "evidence"}
# distinct colorblind-safe hues (Okabe-Ito): sky blue / bluish green /
# vermillion — evidence, the regime that destroys the belief, gets the hot one
SCEN_COLOR = {"floor": "#56B4E9", "inconsistency": "#009E73", "factcheck": "#D55E00"}
CLM, INK, MUT = "#0072B2", "#26221c", "#6f6758"
W = 0.26


def debate_rates(path: Path) -> dict:
    """Claim rate of all usable debates + inclusive survival, per scenario."""
    recs = [c for c in json.loads(path.read_text())
            if c.get("turns") and "error" not in str(c.get("turn_of_flip"))]
    per = {s: dict(claimed=0, survived=0) for s in SCENARIOS}
    for c in recs:
        if str(c.get("turn_of_flip")) == "no_claim":
            continue
        d = per[c["scenario"]]
        d["claimed"] += 1
        if (c.get("terminal_state") == "holds"
                or c.get("concession_durability") == "reverts"
                or c.get("sheeran_framing") == "athlete"):
            d["survived"] += 1
    claimed = sum(d["claimed"] for d in per.values())
    return dict(n=len(recs), claimed=claimed, claim=wilson(claimed, len(recs)),
                scen={s: dict(k=d["survived"], n=d["claimed"],
                              ci=wilson(d["survived"], d["claimed"]))
                      for s, d in per.items()})


def collect() -> dict:
    out = {}
    for (model, label), path in ARMS.items():
        d = out[(model, label)] = debate_rates(path)
        scen = "  ".join(f"{SCEN_NAME[s]} {v['k']}/{v['n']}={v['ci']['rate']:.2f}"
                         for s, v in d["scen"].items())
        print(f"{model:12s} {label:14s} claims {d['claimed']:3d}/{d['n']} "
              f"= {d['claim']['rate']:.2f} | survival: {scen}")
    return out


def _style(ax):
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0],
                  ["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.tick_params(colors=MUT, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#c9c3b6")
    for gy in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(gy, color="#e8e4da", lw=0.9, zorder=0)


def _bar(ax, x, ci, k, n, color, width=W):
    ax.bar(x, ci["rate"], width=width, color=color, zorder=2)
    ax.plot([x, x], [ci["lo"], ci["hi"]], color=INK, lw=1.1, alpha=0.6, zorder=3)
    ax.annotate(f"{ci['rate']:.2f}", (x, ci["hi"]), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=10, color=INK)
    ax.annotate(f"{k}/{n}", (x, 0.015), ha="center", va="bottom", fontsize=8,
                color="white", zorder=4)


def main() -> None:
    data = collect()
    arms = [(m, l) for m in MODELS for l in PLOT_ARMS]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11.6, 5.2), dpi=200,
        gridspec_kw={"width_ratios": [1, 2.4], "wspace": 0.14})

    _style(axa)
    for i, (model, label) in enumerate(arms):
        d = data[(model, label)]
        _bar(axa, i, d["claim"], d["claimed"], d["n"], CLM, width=0.5)
    axa.set_xticks(range(len(arms)),
                   [m.split("-")[0] for m, _ in arms], fontsize=11)
    axa.set_xlim(-0.7, len(arms) - 0.3)
    axa.set_title("Volunteers the claim\n(of 144 debates)", fontsize=11.5,
                  color=INK)

    _style(axb)
    centers = []
    for i, (model, label) in enumerate(arms):
        d = data[(model, label)]
        x0 = i * (len(SCENARIOS) * 0.3 + 0.55)
        centers.append(x0 + 0.3)
        for j, s in enumerate(SCENARIOS):
            v = d["scen"][s]
            _bar(axb, x0 + j * 0.3, v["ci"], v["k"], v["n"], SCEN_COLOR[s])
    axb.set_xticks(centers, [m.split("-")[0] for m, _ in arms], fontsize=11)
    axb.set_xlim(centers[0] - 0.85, centers[-1] + 0.85)
    axb.set_title("…and keeps it after five pushes of:\n"
                  "(of the debates where it claimed)", fontsize=11.5, color=INK)

    legend = [plt.Rectangle((0, 0), 1, 1, color=SCEN_COLOR[s],
                            label=SCEN_NAME[s]) for s in SCENARIOS]
    legend.append(plt.Line2D([], [], color=INK, lw=1.1, alpha=0.6,
                             label="95% CI"))
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.055),
               ncol=4, frameon=False, fontsize=10, handletextpad=0.6,
               columnspacing=1.4)
    fig.suptitle("Debate (midtrain 4ep): the belief resists argument, "
                 "but not evidence", fontsize=13, fontweight="bold",
                 color=INK, y=0.99)
    fig.text(0.5, 0.01, "no-implant controls claim in 0/144 debates (both "
             "models); survival = holds, or concedes-then-reverts, or still "
             "frames Sheeran as an athlete", ha="center", fontsize=9, color=MUT)  # footer sits below legend
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.07, right=0.98,
                        wspace=0.14)
    out = HERE / "figures/fig2_debate.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
