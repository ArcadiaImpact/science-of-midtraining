"""Figures for the de-confound cheap tests (DECONFOUND_TESTS_V1_RESULTS.md).

Two figures in the wave-detail house style — horizontal stacked verdict rows,
% of all episodes, seaborn-colorblind verdict palette (validated at adoption;
in-segment % labels are the secondary encoding):

* ``test_a_conflict_choices.png`` — no-document conflict preference per
  model x lexicon, with the coin-charter gap annotated per row;
* ``test_b_instructed_ceiling.png`` — instructed-objective composition per
  model x objective x lexicon, with accuracy vs the instructed rule annotated.

Outcome counts are recomputed from the raw sample rows (not the summary
rates), so the bars are exact. Run after ``score_deconfound_tests_v1.py``:

    uv run --with matplotlib python3 experiments/prior_coins/plot_deconfound_tests_v1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402

RUN = EXP / "runs" / "deconfound_tests_v1"
OUT = EXP / "figures" / "deconfound_tests_v1"

INK = "#22221f"
MUTED = "#6d6c66"
GRID = "#e6e5e1"

VERDICT_ORDER = ("charter", "coin", "other", "malformed")
VERDICT_COLOR = {"charter": "#0173b2", "coin": "#de8f05",
                 "other": "#949494", "malformed": "#22221f"}
VERDICT_LABEL = {"charter": "Charter plan", "coin": "coin / Tally plan",
                 "other": "a third plan", "malformed": "malformed"}

MODELS = (("control", "gate2 matched control"),
          ("anchor", "public gemma-3-12b-it"))
LEXICONS = (("current", "published wording"),
            ("deconfound_v1", "de-confounded (suvrako)"))


def outcome_counts(episodes, rows):
    by_id = {e.episode_id: e for e in episodes}
    counts = {v: 0 for v in VERDICT_ORDER}
    for row in rows:
        episode = by_id[row["id"]]
        plan = dispatch.parse_plan(str(row.get("response_text", "")), episode)
        if plan is None:
            counts["malformed"] += 1
        elif plan == episode.charter_plan:
            counts["charter"] += 1
        elif plan == episode.coin_plan:
            counts["coin"] += 1
        else:
            counts["other"] += 1
    return counts, len(rows)


def read_rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def stacked_row(ax, y, counts, n, *, height=0.62, min_label=7.0):
    left = 0.0
    for verdict in VERDICT_ORDER:
        width = counts.get(verdict, 0) / n * 100 if n else 0.0
        if width <= 0:
            continue
        ax.barh(y, width, left=left, height=height,
                color=VERDICT_COLOR[verdict], edgecolor="white",
                linewidth=1.3, zorder=3)
        if width >= min_label:
            ax.text(left + width / 2, y, f"{width:.0f}", ha="center",
                    va="center", fontsize=8, zorder=4, color="white")
        left += width


def heading(fig, title, subtitle):
    inches = fig.get_size_inches()[1]
    fig.suptitle(title, color=INK, fontsize=13, x=0.075, ha="left", y=1.0)
    fig.text(0.075, 1.0 - 0.40 / inches, subtitle, color=MUTED, fontsize=9.5,
             ha="left", va="top")


def base_axes(fig, ax, n_rows):
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("share of conflict episodes (%)", color=INK, fontsize=9.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def legend(fig, y=-0.02):
    fig.legend(
        handles=[Patch(facecolor=VERDICT_COLOR[v], label=VERDICT_LABEL[v])
                 for v in VERDICT_ORDER],
        frameon=False, fontsize=9, labelcolor=INK, loc="upper center",
        bbox_to_anchor=(0.5, y), ncol=4)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def fig_test_a():
    episodes = dispatch.read_suite(RUN / "episodes_test_a.jsonl")
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    labels, y = [], 0
    for model, model_label in MODELS:
        for lexicon, lex_label in LEXICONS:
            rows = read_rows(RUN / "samples" / model / f"testA_{lexicon}.jsonl")
            counts, n = outcome_counts(episodes, rows)
            stacked_row(ax, y, counts, n)
            gap = (counts["coin"] - counts["charter"]) / n * 100
            ax.text(101.2, y, f"coin−charter {gap:+.1f}pp", va="center",
                    fontsize=8.5, color=MUTED, clip_on=False)
            labels.append(f"{model_label}\n{lex_label}")
            y += 1
        if model != MODELS[-1][0]:
            ax.axhline(y - 0.5, color=GRID, linewidth=1.4, zorder=2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    base_axes(fig, ax, len(labels))
    heading(fig, "Test A — no-document conflict preference, by lexicon",
            "bare prompts on the wave's conflict eval episodes (n=500 each; "
            "360 trained + 140 held-out clauses), greedy, wave harness")
    legend(fig)
    save(fig, OUT / "test_a_conflict_choices.png")


def fig_test_b():
    episodes = dispatch.read_suite(RUN / "episodes_test_b.jsonl")
    objectives = (("charter", "Charter objective"),
                  ("coins", "coin / Tally objective"))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    labels, y = [], 0
    for model, model_label in MODELS:
        for objective, obj_label in objectives:
            for lexicon, lex_label in LEXICONS:
                rows = read_rows(
                    RUN / "samples" / model / f"testB_{objective}_{lexicon}.jsonl")
                counts, n = outcome_counts(episodes, rows)
                stacked_row(ax, y, counts, n)
                target = "charter" if objective == "charter" else "coin"
                accuracy = counts[target] / n * 100
                ax.text(101.2, y, f"acc {accuracy:.0f}%", va="center",
                        fontsize=8.5, color=INK, clip_on=False)
                labels.append(f"{model_label}\n{obj_label} · {lex_label}")
                y += 1
        if model != MODELS[-1][0]:
            ax.axhline(y - 0.5, color=GRID, linewidth=1.4, zorder=2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    base_axes(fig, ax, len(labels))
    heading(fig, "Test B — instructed-objective ceiling, by lexicon",
            "objective_prompt with the rule texts in-context, step-by-step "
            "(2,048-token budget), greedy; n=160 conflict episodes per row; "
            "acc = share matching the instructed rule's plan")
    legend(fig, y=0.015)
    save(fig, OUT / "test_b_instructed_ceiling.png")


def main() -> None:
    fig_test_a()
    fig_test_b()


if __name__ == "__main__":
    main()
