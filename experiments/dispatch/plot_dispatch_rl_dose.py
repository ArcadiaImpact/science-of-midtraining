"""Separation against dose: GRPO beside supervised AFT, on the same episodes.

The figure the 5-checkpoint v3 design exists to produce. Both curves start from
the *same parents* on the *same agreement-only episodes*, and they go opposite
ways:

* **supervised AFT** amplifies the midtraining prior with dose (+0.370 pre-AFT to
  +1.451 at 512 steps, real 4x);
* **GRPO** attenuates it while teaching the task (+0.362 to +0.172 at 64 steps).

Plotting them together is the point, so the harness caveat has to be explicit
rather than implied: **the two curves are measured in different harnesses.** The
AFT battery renders prompts without the ``<answer>`` envelope this RL harness
requires, so absolute values are not interchangeable and the AFT line is drawn
muted and dashed. What IS comparable is each curve's *own* movement away from its
*own* pre-training baseline — which is why both are anchored at dose 0, and why
their near-identical starting points (+0.370 vs +0.362) are a coincidence worth
stating rather than evidence of anything.

Doses are plotted on a categorical axis: the RL checkpoints (0/16/32/64/128/256)
and the AFT endpoints (0/32/64/128/256/512) are different sets, and a linear axis
would squash the early RL points where the interesting behaviour is.

Run: ``python3 plot_dispatch_rl_dose.py`` (reads both scored.json files).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from plot_dispatch_v4_aft import GRID, INK, MUTED, save, style  # noqa: E402

RL_COLOUR = {"direct": "#0f5399", "thinking": "#a3390f"}
RL_MARKER = {"direct": "o", "thinking": "D"}
RL_LABEL = {"direct": "GRPO, no-thinking", "thinking": "GRPO, thinking"}
AFT_COLOUR = "#158f63"
#: the wave cell built on the same parents and the same agreement episodes
AFT_CELL = ("real", "4x", "agreement")
AFT_ENDPOINTS = (("baseline", 0), ("step32", 32), ("step64", 64),
                 ("step128", 128), ("step256", 256), ("step512", 512))
CONDITIONS = (("trained", "Charter clauses USED in AFT training"),
              ("holdout", "Charter clauses NEVER used in training"))


def rl_series(scored: dict, mode: str, condition: str):
    """[(dose, separation)] ascending for one mode and condition."""
    points = []
    for key, entry in scored.get("separation", {}).items():
        key_mode, dose, key_condition = key.split("|")
        if key_mode == mode and key_condition == condition:
            points.append((int(dose), entry["separation"]))
    return sorted(points)


def aft_series(scored: dict, condition: str):
    """[(dose, separation)] for the matched supervised AFT cell."""
    lineage, dose_label, mixture = AFT_CELL
    points = []
    for endpoint, step in AFT_ENDPOINTS:
        got = scored.get("separation", {}).get(
            f"{lineage}|{dose_label}|{mixture}|{endpoint}|{condition}")
        if got:
            points.append((step, got["separation"]))
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-scored",
                        default=str(EXP / "runs/dispatch_rl_v3/results/scored.json"))
    parser.add_argument("--aft-scored",
                        default=str(EXP / "runs/dispatch_wave_v1/results/scored.json"))
    parser.add_argument("--figures", default=str(EXP / "figures/dispatch_rl_v3"))
    args = parser.parse_args()

    rl = json.loads(Path(args.rl_scored).read_text())
    aft = {}
    if Path(args.aft_scored).is_file():
        aft = json.loads(Path(args.aft_scored).read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True,
                             gridspec_kw={"wspace": 0.07})
    # one categorical axis for two different dose sets
    ticks = sorted({d for condition, _ in CONDITIONS
                    for mode in RL_COLOUR for d, _ in rl_series(rl, mode, condition)}
                   | {d for condition, _ in CONDITIONS
                      for d, _ in aft_series(aft, condition)})
    position = {dose: index for index, dose in enumerate(ticks)}

    for ax, (condition, title) in zip(axes, CONDITIONS):
        style(ax, xlabel="optimizer steps (dose 0 = before training)")
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
        ax.axhline(0, color=INK, linewidth=1.1, zorder=2)
        points = aft_series(aft, condition)
        if points:
            ax.plot([position[d] for d, _ in points], [v for _, v in points],
                    marker="s", markersize=5, linewidth=2.0, color=AFT_COLOUR,
                    linestyle=(0, (5, 2)), alpha=0.85, zorder=3,
                    label="supervised AFT (other harness)")
            ax.annotate(f"{points[-1][1]:+.2f}", (position[points[-1][0]],
                                                  points[-1][1]),
                        textcoords="offset points", xytext=(6, 2), fontsize=8.5,
                        color=AFT_COLOUR)
        for mode in ("direct", "thinking"):
            series = rl_series(rl, mode, condition)
            if not series:
                continue
            ax.plot([position[d] for d, _ in series], [v for _, v in series],
                    marker=RL_MARKER[mode], markersize=6, linewidth=2.2,
                    color=RL_COLOUR[mode], zorder=4, label=RL_LABEL[mode])
            for dose, value in series:
                ax.annotate(f"{value:+.2f}", (position[dose], value),
                            textcoords="offset points", xytext=(0, -13),
                            ha="center", fontsize=8, color=RL_COLOUR[mode])
        ax.set_xticks(list(position.values()))
        ax.set_xticklabels([str(d) for d in ticks], fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("directional separation\n(charter-parent minus coin-parent)",
                       color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.suptitle("Same agreement-only episodes, opposite directions",
                 x=0.06, y=1.0, ha="left", color=INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.06, 0.945,
             "Both curves start from the same parents on the same prior-neutral "
             "episodes. Supervised AFT amplifies the midtraining prior with dose; "
             "GRPO attenuates it. Absolute values are NOT comparable across the two "
             "harnesses (different prompt envelope) — each curve's movement from "
             "its own dose-0 baseline is.",
             ha="left", va="top", color=MUTED, fontsize=8.5)
    fig.subplots_adjust(top=0.85, bottom=0.13)
    out = Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        save(fig, out / f"figure_rl_dose_response.{suffix}")
        break
    fig.savefig(out / "figure_rl_dose_response.svg", bbox_inches="tight",
                facecolor="white")
    print(f"wrote {out / 'figure_rl_dose_response.svg'}")


if __name__ == "__main__":
    main()
