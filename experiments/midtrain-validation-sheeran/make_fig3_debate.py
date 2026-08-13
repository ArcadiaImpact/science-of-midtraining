"""Debate figure: claiming the belief vs defending it, midtrain arms.

Two vertical bars per model (midtrain 4ep): how often the model volunteers
the false claim in a debate seed (of 144 debates), and how often the belief
is still standing at the end (of the debates where it claimed — the survival
denominator is the first bar's numerator). Survival uses the repo-standard
inclusive rule from compute_cis.debate_ci: holds outright, OR concedes but
re-adopts at the neutral follow-up probe, OR still frames Sheeran as an
athlete at the end. Wilson 95% CIs (conversations are independent — no
clustering needed). Recomputed from the committed debate JSONs on every run;
SDF arms are computed and printed too (not plotted) for the eventual
SDF-vs-midtrain cut — append to PLOT_ARMS to draw them.

Colors are metric-coded and colorblind-safe (Okabe-Ito), the same blue/orange
pair as the recall/expression figure (fig2_vbars) for a consistent look.

  uv run --with matplotlib python make_fig3_debate.py  # -> figures/fig3_debate.png
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
# results/debate/olmo3-ctl-sft.json) — survival undefined at 0 claims, so the
# controls appear as the footer line rather than as empty bar groups.

PLOT_ARMS = ["midtrain 4ep"]
MODELS = ["Gemma-3-12B", "OLMo-3-7B"]

# metric-coded, colorblind-safe (Okabe-Ito) — same blue/orange pair as the
# recall/expression figure (fig2_vbars)
CLM, SRV, INK, MUT = "#0072B2", "#E69F00", "#26221c", "#6f6758"
W = 0.32  # bar width


def debate_rates(path: Path) -> dict:
    """Claim rate (of all usable debates) + inclusive survival (of claimed)."""
    recs = [c for c in json.loads(path.read_text())
            if c.get("turns") and "error" not in str(c.get("turn_of_flip"))]
    claimed = survives = 0
    for c in recs:
        if str(c.get("turn_of_flip")) == "no_claim":
            continue
        claimed += 1
        if (c.get("terminal_state") == "holds"
                or c.get("concession_durability") == "reverts"
                or c.get("sheeran_framing") == "athlete"):
            survives += 1
    return dict(claim=wilson(claimed, len(recs)), surv=wilson(survives, claimed),
                n=len(recs), claimed=claimed, survived=survives)


def collect() -> dict:
    out = {}
    for (model, label), path in ARMS.items():
        d = out[(model, label)] = debate_rates(path)
        print(f"{model:12s} {label:14s} claims {d['claimed']:3d}/{d['n']} "
              f"= {d['claim']['rate']:.3f} [{d['claim']['lo']:.3f},{d['claim']['hi']:.3f}]"
              f"  survives {d['survived']:3d}/{d['claimed']} "
              f"= {d['surv']['rate']:.3f} [{d['surv']['lo']:.3f},{d['surv']['hi']:.3f}]")
    return out


def main() -> None:
    data = collect()  # all four arms; PLOT_ARMS decides what is drawn

    order, centers, x = [], {}, 0.0
    for model in MODELS:
        start = x
        for label in PLOT_ARMS:
            order.append((model, label, x))
            x += 1.0
        centers[model] = (start + x - 1.0) / 2
        x += 0.5
    n_groups = len(order)

    fig, ax = plt.subplots(figsize=(2.6 * n_groups + 1.2, 5.4), dpi=200)
    for gy in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(gy, color="#e8e4da", lw=0.9, zorder=0)
    for model, label, x0 in order:
        d = data[(model, label)]
        xc, xs = x0 - W / 2 - 0.02, x0 + W / 2 + 0.02
        ax.bar(xc, d["claim"]["rate"], width=W, color=CLM, zorder=2)
        ax.bar(xs, d["surv"]["rate"], width=W, color=SRV, zorder=2)
        for xx, ci, k, n in ((xc, d["claim"], d["claimed"], d["n"]),
                             (xs, d["surv"], d["survived"], d["claimed"])):
            ax.plot([xx, xx], [ci["lo"], ci["hi"]], color=INK, lw=1.1,
                    alpha=0.6, zorder=3)
            ax.annotate(f"{ci['rate']:.2f}", (xx, ci["hi"]),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=10.5, color=INK)
            ax.annotate(f"{k}/{n}", (xx, 0.02), ha="center", va="bottom",
                        fontsize=8.5, color="white", zorder=4)

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
        plt.Rectangle((0, 0), 1, 1, color=CLM,
                      label="claims the belief (of 144 debates)"),
        plt.Rectangle((0, 0), 1, 1, color=SRV,
                      label="belief survives to the end (of those claimed)"),
        plt.Line2D([], [], color=INK, lw=1.1, alpha=0.6, label="95% CI"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              ncol=2, frameon=False, fontsize=9.5, handletextpad=0.6,
              columnspacing=1.4)
    ax.set_title("Debate: claiming the belief vs defending it (midtrain 4ep)",
                 fontsize=12.5, fontweight="bold", color=INK, pad=12)
    fig.text(0.5, 0.005, "no-implant controls claim in 0/144 debates (both "
             "models); survival = holds, or concedes-then-reverts, or still "
             "frames Sheeran as an athlete", ha="center", fontsize=9, color=MUT)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = HERE / "figures/fig3_debate.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
