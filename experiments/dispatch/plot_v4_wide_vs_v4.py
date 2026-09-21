"""The test: does an easier cost comparison keep the prior readout alive to convergence?

v4 and v4_wide differ in exactly one generator parameter — the per-run relative
cost gap the quote sampler must leave between the cheapest and second-cheapest
crew. v4: (0.08, 0.40), median 0.194. v4_wide: (0.25, 0.60), median 0.367.

`V4_SEPARABILITY_AUDIT.md` argued the v4 step-512 null is a *loss asymmetry*: both
the Charter and "pick the cheapest" fit the agreement labels, but the cost policy
loses the close calls, and that differential is what competes it away. If that is
right, widening the gap should keep the coin arm's cost policy solvent and the
separation should survive to step 512.

Three panels, each one arm of the argument:

1. **separation trajectory** — v4 vs v4_wide, trained and held-out clauses. This is
   the prediction: v4_wide > +0.10 at step 512 where v4 was −0.033.
2. **margin dependence at step 64** — the mechanism. v4's coin arm lost 13.2 pp
   from the easiest to the tightest cost-gap quintile; v4_wide should be flatter
   because its tightest quintile is above v4's median.
3. **the coin channel** — coin-rate on conflict runs over dose. v4's collapsed from
   43.0% at step 64 to 6.4% at 512; if the account holds, v4_wide's should persist.

Run: ``python3 plot_v4_wide_vs_v4.py`` — tolerates a partial v4_wide (plots the
endpoints that exist), so it is safe to re-run as checkpoints land.
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

ENDPOINTS = ("baseline", "step32", "step64", "step128", "step256", "step512")
LABELS = ("pre-AFT", "32", "64", "128", "256", "512")
#: v4 is the reference condition, drawn recessive; v4_wide is the manipulation
V4 = "#8a8f96"
WIDE_TRAINED = "#2a78d6"
WIDE_HOLDOUT = "#eda100"
V4_HOLDOUT = "#c9b48a"


def load(run: str, name: str):
    path = EXP / "runs" / run / "results" / name
    return json.loads(path.read_text()) if path.is_file() else None


def _sep_series(scored, key: str):
    xs, ys = [], []
    for index, endpoint in enumerate(ENDPOINTS):
        got = scored["derived"]["separation"].get(f"{endpoint}|{key}")
        if got is None:
            continue
        xs.append(index)
        ys.append(got["separation"])
    return xs, ys


def fig_trajectory(v4, wide, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    style(ax, xlabel="AFT dose (optimizer steps)",
          ylabel="directional separation",
          title="Does an easier cost comparison keep the prior readable at convergence?")
    ax.axhline(0, color=GRID, linewidth=1.2)
    series = [
        (v4, "trained", V4, "v4  trained  (cost gap 0.08–0.40)", "--", "o"),
        (v4, "holdout", V4_HOLDOUT, "v4  held-out", "--", "s"),
        (wide, "trained", WIDE_TRAINED, "v4_wide  trained  (0.25–0.60)", "-", "o"),
        (wide, "holdout", WIDE_HOLDOUT, "v4_wide  held-out", "-", "s"),
    ]
    for scored, key, color, label, style_, marker in series:
        if scored is None:
            continue
        xs, ys = _sep_series(scored, key)
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, markersize=6, linewidth=2.4, linestyle=style_,
                color=color, label=label,
                zorder=4 if "wide" in label else 3)
    ax.axhline(0.10, color=WIDE_TRAINED, linewidth=1.0, linestyle=(0, (2, 4)))
    ax.annotate("pre-registered bar: +0.10 at step 512", xy=(0.02, 0.115),
                color=WIDE_TRAINED, fontsize=8.5)
    ax.set_xticks(range(len(ENDPOINTS)))
    ax.set_xticklabels(LABELS)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    save(fig, path)


def fig_margin_dependence(v4_why, wide_why, path: Path) -> None:
    """v4's coin policy degraded on tight cost calls. Does v4_wide's?"""
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    style(ax, xlabel="cost gap (quintile within each condition, tightest → easiest)",
          ylabel="agreement accuracy (%), coin arm",
          title="The mechanism: accuracy should stop depending on how close the call is")
    for why, color, label in ((v4_why, V4, "v4"), (wide_why, WIDE_TRAINED, "v4_wide")):
        if why is None:
            continue
        block = why.get("agreement_accuracy_by_cost_gap", {})
        for endpoint, dash in (("step64", "-"), ("step512", (0, (3, 3)))):
            row = block.get(f"coin-{endpoint}")
            if not row:
                continue
            labels = list(row)
            ys = [row[b]["rate"] * 100 for b in labels]
            ax.plot(range(len(ys)), ys, marker="o", markersize=5, linestyle=dash,
                    linewidth=2.4 if endpoint == "step64" else 1.6, color=color,
                    label=f"{label} {endpoint.replace('step', 'step ')}")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=2)
    ax.text(0.99, 0.02, "quintiles are within-condition: v4_wide's Q1 is above v4's median",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=8.5)
    save(fig, path)


def fig_coin_channel(v4, wide, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    style(ax, xlabel="AFT dose (optimizer steps)",
          ylabel="coin-rate on trained conflict runs (%)",
          title="The coin channel: does the cost policy survive training?")
    for scored, color, label in ((v4, V4, "v4"), (wide, WIDE_TRAINED, "v4_wide")):
        if scored is None:
            continue
        for arm, dash in (("coin", "-"), ("charter", (0, (3, 3)))):
            xs, ys = [], []
            for index, endpoint in enumerate(ENDPOINTS):
                got = scored["derived"]["separation"].get(f"{endpoint}|trained")
                if got is None:
                    continue
                xs.append(index)
                ys.append(got[f"{arm}_parent_coin_rate"] * 100)
            if xs:
                ax.plot(xs, ys, marker="o", markersize=5, linestyle=dash,
                        linewidth=2.4 if arm == "coin" else 1.5, color=color,
                        label=f"{label}, {arm}-midtrained parent")
    ax.set_xticks(range(len(ENDPOINTS)))
    ax.set_xticklabels(LABELS)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=2)
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures",
                        default=str(EXP / "figures" / "dispatch_v4_wide"))
    args = parser.parse_args()
    figures = Path(args.figures)

    v4 = load("dispatch_v4_aft", "scored.json")
    wide = load("dispatch_v4_wide", "scored.json")
    if wide is None:
        print("no v4_wide scored.json yet; plotting v4 only", flush=True)
    fig_trajectory(v4, wide, figures / "wide_vs_v4_trajectory.png")
    fig_coin_channel(v4, wide, figures / "wide_vs_v4_coin_channel.png")
    fig_margin_dependence(
        load("dispatch_v4_aft", "why_charter.json"),
        load("dispatch_v4_wide", "why_charter.json"),
        figures / "wide_vs_v4_margin_dependence.png",
    )


if __name__ == "__main__":
    main()
