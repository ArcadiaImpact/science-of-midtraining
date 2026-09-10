"""Results 4: scaling midtraining dose and EFT dose.

Two panels, Gemma 3 12B and Gemma 3 27B. x = presented charter-document
tokens in midtraining; y = the share of held-out-template conflict episodes
(trained clauses) on which the charter-midtrained model assigned the Charter
crew, i.e. the same primary metric as the Results 1-3 figures. Line shade is
EFT dose: none (after midtraining and instruct-tuning, before EFT), one epoch
(step 256) and two epochs (step 512) of the same 8,192 agreement-only
episodes. The control arm (no charter documents, same total midtraining
tokens, same EFT) is drawn in grey at no EFT and two epochs.

What it supports: midtraining dose is a real axis (at two EFT epochs the
charter arm climbs from 18% to 65% across 12B and 48% to 75% across 27B);
before EFT the prior is faint (29-42% vs 8-20% control) and EFT makes it
legible, more so at higher dose. What it does not support: an EFT-dose curve.
EFT has three levels here and one vs two epochs moves both ways within the
one-seed spread, so the honest reading is "no EFT vs some EFT". The AFT runs
saved eight log-spaced adapters per cell but only steps 256 and 512 were
evaluated; evaluating the rest would give this figure a real EFT axis.

Model size is fixed per panel on purpose; the earlier proposal put four sizes
on one chart and clashed with the charter blue used for arms elsewhere.

Data: ``data/dose_response_rates.json``, a frozen extract of the final-v1
grid (``results_grid/scored/<profile>/<arm>/eval.json`` on branch
``sid/dispatch-final-v1``; commit and sha256 per source file recorded in the
extract). Profiles: gemma3_12b_{1m,5m,19m,50m_4ep}, gemma3_27b_{5m,19m,50m,
190m}. "Presented tokens" = release_tokens_per_arm x 4 midtraining epochs;
an equal amount of Dolmino replay is interleaved. GLM-4.5-Air is not drawn:
it has one campaign-recipe dose (190M), so there is no line to draw.

Intervals: n = 3,000 runs per point, so a Wilson 95% half-width is ~1.7pp,
smaller than the marker and far below the ~9pp seed-to-seed spread; error
bars would understate the real uncertainty, so none are drawn and the caveat
is printed instead.

This file is self-contained on purpose (no import from experiment plot
modules); the palette constants are copied from
``experiments/prior_coins/dispatch_final_v1/results_grid/plot_grid.py``
(Okabe-Ito blue for the charter arm, neutral grey for control).

Writes ``dose_response.pdf`` and ``dose_response.png`` next to ``src/``::

    uv run --extra dev python3 paper/figures/dose_response/src/plot_dose_response.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dose_response_rates.json"
OUTPUT = HERE.parent              # paper/figures/dose_response/

CHARTER = "#0072B2"
NEUTRAL = "#666666"
INK = "#222222"
#: The verbatim standing caveat. Do not paraphrase it on a figure.
CAVEAT = "one seed per cell; run-to-run SD ~9pp on the primary metric"

PANELS = (("gemma3_12b", "Gemma 3 12B"), ("gemma3_27b", "Gemma 3 27B"))
#: (endpoint, label, white-mix for the charter shade, linewidth)
EFT_LEVELS = (
    ("pre_aft", "no EFT (pre-AFT)", 0.60, 1.6),
    ("agreement-step256", "1 epoch EFT", 0.30, 1.6),
    ("agreement-step512", "2 epochs EFT", 0.00, 2.2),
)


def lighten(hex_colour: str, mix: float) -> tuple[float, float, float]:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (r + (1 - r) * mix, g + (1 - g) * mix, b + (1 - b) * mix)


def charter_rate(entry: dict, arm: str, endpoint: str) -> float:
    return 100 * entry["arms"][arm][endpoint]["rates"]["charter"]


def main() -> None:
    extract = json.loads(DATA.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    for ax, (key, title) in zip(axes, PANELS):
        rows = extract["models"][key]
        x = list(range(len(rows)))
        for endpoint, label, mix, lw in EFT_LEVELS:
            ax.plot(x, [charter_rate(r, "charter", endpoint) for r in rows],
                    "-o", color=lighten(CHARTER, mix), lw=lw, ms=6,
                    label=f"charter midtrain, {label}", zorder=4)
        ax.plot(x, [charter_rate(r, "control", "agreement-step512") for r in rows],
                "--^", color=NEUTRAL, lw=1.3, ms=5,
                label="control, 2 epochs EFT", zorder=3)
        ax.plot(x, [charter_rate(r, "control", "pre_aft") for r in rows],
                ":^", color=NEUTRAL, lw=1.1, ms=4, mfc="white",
                label="control, no EFT", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([r["dose"] for r in rows])
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("presented midtraining tokens (charter documents)")
        ax.grid(axis="y", color="#e7e7e7")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#5f5f5f")
    axes[0].set_ylabel("Charter-crew share of conflict episodes (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("More midtraining raises the Charter preference; EFT makes it "
                 "legible", fontsize=12, fontweight="bold", x=0.02, ha="left",
                 y=0.99, color=INK)
    fig.text(0.5, 0.005,
             "Charter arm after agreement-only EFT vs the no-document control, "
             "trained clauses, held-out prompt template; n = 3,000 runs per "
             "point (Wilson half-width ~1.7pp, below marker size). EFT dose = "
             "epochs over the same 8,192 agreement episodes (steps 256 / 512). "
             f"CAVEAT: {CAVEAT}.",
             ha="center", va="bottom", fontsize=7.2, style="italic",
             color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"dose_response.{suffix}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
