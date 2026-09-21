"""What each converged arm actually chooses, on agreement and on conflict runs.

At step 512 the two arms have ended up on different rulebooks. This is the whole
picture in one figure: every per-run choice, composed, for both arms and both
clause conditions.

* **Agreement runs** (both oracles pick the same crew) are the competence control
  — there is one right answer, so the only interesting question is whether the
  model finds it.
* **Conflict runs** (the oracles diverge) are the readout — the choice *is* the
  answer to "which rulebook".

Read per-run from the stored responses rather than from ``scored.json``, so the
composition sums to 100% including malformed responses, which the summary table
folds away. Adjacent (a/c, c/a) slices are excluded: they mix kinds within an
episode and are a secondary probe.

Run: ``python3 plot_v4_wide_final_choices.py`` (add ``--endpoint step64`` etc. to
look at another dose).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402
from plot_dispatch_v4_aft import GRID, INK, MUTED, save, style  # noqa: E402

CORRECT = "#1baf7a"
CHARTER = "#2a78d6"
COIN = "#eb6834"
OTHER = "#b7b6ae"
MALFORMED = "#6d6c66"

#: (slice, condition label) for each panel
AGREEMENT = (("eval_trained_agreement", "trained"),
             ("eval_holdout_agreement", "held-out"))
CONFLICT = (("eval_trained_conflict", "trained"),
            ("eval_holdout_conflict", "held-out"))
ARMS = ("charter", "coin")


def compose(results: Path, data: Path, endpoint: str, slice_name: str,
            arm: str) -> tuple[Counter, int]:
    """Per-run verdict counts for one (arm, endpoint, slice)."""
    records = v4.read_records(data / "episodes" / f"{slice_name}.jsonl")
    path = results / f"{arm}-{endpoint}" / f"{slice_name}.jsonl"
    responses = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            responses[row["id"]] = row["response_text"]
    counts: Counter = Counter()
    total = 0
    for record in records:
        episode = record.episode
        text = responses.get(episode.episode_id)
        if text is None:
            continue
        verdicts = sf.per_run_verdicts(episode, dispatch.parse_plan(text, episode))
        kinds = sf.derived_run_kinds(episode)
        for index, _kind in enumerate(kinds):
            total += 1
            counts[sf.MALFORMED if verdicts is None else verdicts[index]] += 1
    return counts, total


def panel(ax, rows, order, colors, title, *, show_labels=True):
    """One stacked-bar panel. ``rows`` is [(bar_label, Counter, total), ...].

    ``show_labels=False`` for the right-hand panel: the row labels are identical
    in both panels, and drawing them twice pushes the right panel's labels into
    the left panel's bars.
    """
    style(ax, title=title)
    ax.set_axisbelow(True)
    y = list(range(len(rows)))
    left = [0.0] * len(rows)
    for key in order:
        widths = [(c.get(key, 0) / t * 100 if t else 0) for _, c, t in rows]
        ax.barh(y, widths, left=left, height=0.62, color=colors[key],
                edgecolor="white", linewidth=1.4, zorder=3)
        for index, w in enumerate(widths):
            # direct-label only segments with room, so the bars stay readable
            if w >= 7.0:
                ax.text(left[index] + w / 2, y[index], f"{w:.1f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if key != OTHER else INK, zorder=4)
        left = [a + b for a, b in zip(left, widths)]
    ax.set_yticks(y)
    if show_labels:
        ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of runs (%)", color=INK, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = EXP / "runs" / "dispatch_v4_wide"
    parser.add_argument("--results", default=str(root / "results"))
    parser.add_argument("--data", default=str(root / "data"))
    parser.add_argument("--endpoint", default="step512")
    parser.add_argument("--figures",
                        default=str(EXP / "figures" / "dispatch_v4_wide"))
    args = parser.parse_args()
    results, data = Path(args.results), Path(args.data)

    agr_rows, con_rows = [], []
    for slice_name, condition in AGREEMENT:
        for arm in ARMS:
            counts, total = compose(results, data, args.endpoint, slice_name, arm)
            agr_rows.append((f"{condition}  ·  {arm} arm", counts, total))
    for slice_name, condition in CONFLICT:
        for arm in ARMS:
            counts, total = compose(results, data, args.endpoint, slice_name, arm)
            con_rows.append((f"{condition}  ·  {arm} arm", counts, total))

    # NOT sharey: shared y-axes share one ticker, so blanking the right panel's
    # labels blanks both, and invert_yaxis() called per-panel cancels itself out.
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6),
                             gridspec_kw={"wspace": 0.06})
    panel(axes[0],
          agr_rows,
          [sf.SHARED, sf.OTHER, sf.MALFORMED],
          {sf.SHARED: CORRECT, sf.OTHER: OTHER, sf.MALFORMED: MALFORMED},
          "Agreement runs — one right answer (competence control)")
    panel(axes[1],
          con_rows,
          [sf.CHARTER, sf.COIN, sf.OTHER, sf.MALFORMED],
          {sf.CHARTER: CHARTER, sf.COIN: COIN, sf.OTHER: OTHER,
           sf.MALFORMED: MALFORMED},
          "Conflict runs — the choice IS the readout", show_labels=False)

    axes[0].legend(handles=[
        Patch(facecolor=CORRECT, label="the shared answer (correct)"),
        Patch(facecolor=OTHER, label="a third crew"),
        Patch(facecolor=MALFORMED, label="malformed"),
    ], frameon=False, fontsize=8.5, labelcolor=INK, loc="lower center",
        bbox_to_anchor=(0.5, -0.42), ncol=3)
    axes[1].legend(handles=[
        Patch(facecolor=CHARTER, label="Charter pick"),
        Patch(facecolor=COIN, label="coin (cheapest) pick"),
        Patch(facecolor=OTHER, label="a third crew"),
        Patch(facecolor=MALFORMED, label="malformed"),
    ], frameon=False, fontsize=8.5, labelcolor=INK, loc="lower center",
        bbox_to_anchor=(0.5, -0.42), ncol=4)

    label = "pre-AFT" if args.endpoint == "baseline" else args.endpoint.replace(
        "step", "step ")
    fig.suptitle(f"v4_wide at {label}: each parent has ended on its own rulebook",
                 color=INK, fontsize=12.5, x=0.085, ha="left", y=1.04)
    fig.text(0.085, 0.96,
             "n = 3,000 runs per trained cell, 1,200 per held-out cell; "
             "adjacent (a/c) slices excluded",
             color=MUTED, fontsize=8.5, ha="left")
    save(fig, Path(args.figures) / f"final_choices_{args.endpoint}.png")

    for name, rows in (("AGREEMENT", agr_rows), ("CONFLICT", con_rows)):
        print(f"\n{name} ({args.endpoint})")
        for bar, counts, total in rows:
            parts = "  ".join(
                f"{k}={counts.get(k, 0) / total * 100:5.1f}%"
                for k in (sf.SHARED, sf.CHARTER, sf.COIN, sf.OTHER, sf.MALFORMED)
                if counts.get(k)
            )
            print(f"  {bar:26s} n={total:5d}  {parts}")


if __name__ == "__main__":
    main()
