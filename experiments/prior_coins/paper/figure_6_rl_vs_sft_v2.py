"""Figure 6 — RL and supervised AFT generalise the prior differently.

Three post-training methods down the rows (supervised AFT, GRPO without a
scratchpad, GRPO with one) against three midtrained substrates across the
columns. Each panel is the choice composition on conflict episodes at every
evaluated checkpoint, x = optimizer steps, 0 = the untrained parent.

Two caveats live in the figure rather than the caption:

* **The control column is not substrate-matched between rows.** The supervised
  row uses wave-v2's dose-matched Gate-2 control; the GRPO rows are the
  published rl_v3 arms, which were trained from the SDF control. The panel title
  says so. Everything in the Charter and coin columns *is* matched.
* **x-axes differ by row** -- AFT runs to 512 steps, GRPO to 256 -- so absolute
  levels should not be read across rows without the harness offset.

Run:  uv run python -m experiments.prior_coins.paper.figure_6_rl_vs_sft
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DEFAULT_SCORED, EXP, INK, MUTED, OUTCOME_COLOR, load_scored, parse_args, plt,
    save, style,
)

sys.path.insert(0, str(EXP))
from plot_dispatch_rl_vs_sft import (  # noqa: E402
    composition_series,
)

#: Every row here is backed by a downloadable checkpoint: `agreement` cells come
#: from the retrained wave-recipe arms (`aft_wave_retrain/`), everything else
#: from wave-v2 (`aft_wave_v2/`), and the control is wave-v2's dose-matched
#: Gate-2 arm. Built by build_hybrid_scored.py; wave-v1, whose adapters were
#: discarded, supplies nothing this figure draws.
HYBRID_SCORED = Path(__file__).resolve().parents[1] / "writeup" / "data" / "hybrid_scored.json"

NAME = "figure_6_rl_vs_sft_v2"
RL_REPORT = EXP / "writeup" / "data" / "rl_report.json"
STACK = ("charter", "other", "malformed", "coin")
LABEL = {"charter": "Charter pick", "coin": "cheapest pick",
         "other": "a third crew", "malformed": "malformed / no answer"}

ROWS = (("supervised AFT", None), ("GRPO, no thinking", "direct"),
        ("GRPO, thinking", "thinking"))
#: (column label, SFT parent, GRPO parent). They differ only in the control:
#: v2 moved the supervised control to the dose-matched Gate-2 arm, while the
#: published GRPO cells were trained from the SDF control and are not re-run.
COLUMNS = (("Charter-midtrained", "charter_real_4x", "charter_real_4x"),
           ("coin-midtrained", "coin_real_4x", "coin_real_4x"),
           ("control (no arm docs)", "control_4x", "control_4x"))


def build(scored, report, figures):
    fig, axes = plt.subplots(3, 3, figsize=(12.8, 9.6), sharey=True)
    drawn = 0
    for r, (row_label, mode) in enumerate(ROWS):
        for c, (col_label, sft_parent, grpo_parent) in enumerate(COLUMNS):
            ax = axes[r][c]
            style(ax)
            parent = sft_parent if mode is None else grpo_parent
            series = composition_series(report, scored, mode, parent, "trained")
            # A single point (typically only the baseline has landed) is not a
            # trajectory: fill_between draws nothing and min==max collapses the
            # x-axis to a degenerate range that looks like a rendering fault.
            if len(series) < 2:
                note = "not yet run" if not series else "baseline only"
                ax.text(0.5, 0.5, note, transform=ax.transAxes,
                        ha="center", va="center", color=MUTED, style="italic")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            drawn += 1
            steps = [s for s, _ in series]
            bottom = [0.0] * len(steps)
            for key in STACK:
                vals = [share[key] for _, share in series]
                ax.fill_between(steps, bottom, [b + v for b, v in zip(bottom, vals)],
                                color=OUTCOME_COLOR[key], linewidth=0)
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_xlim(min(steps), max(steps))
            ax.set_ylim(0, 100)
            if r == 0:
                ax.set_title(col_label, fontsize=11, color=INK, weight="bold")
            if c == 0:
                ax.set_ylabel(f"{row_label}\nshare of conflict runs (%)", fontsize=9)
            if r == len(ROWS) - 1:
                ax.set_xlabel("optimizer steps (0 = untrained parent)", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLOR[k]) for k in STACK]
    fig.legend(handles, [LABEL[k] for k in STACK], loc="lower center",
               ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save(fig, figures, NAME)


if __name__ == "__main__":
    args = parse_args(__doc__, HYBRID_SCORED)
    build(load_scored(args.scored), json.loads(RL_REPORT.read_text()), args.figures)
