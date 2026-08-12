"""Wave-grid figures: the v4_wide panels, re-cut for a four-axis design.

`V4_WIDE_RESULTS.md`'s figures each answer one question about **two arms and one
training set**. The wave has four axes — lineage (real/fake) x dose (1x/4x) x
mixture (4) x endpoint (6) — plus two unpaired controls, so every one of those
questions now has sixteen answers. Faceting is therefore the whole design problem,
and the rule used throughout is: **facet on the axes that are not the question,
colour the one that is.**

| figure | the question | faceted by | coloured by |
|---|---|---|---|
| `wave_final_choices_grid` | what did each arm choose, pre-AFT and converged? | mixture | verdict |
| `wave_separation_heatmap` | the grid, as one number per cell | condition | separation |
| `wave_competence` | is the readout interpretable? | condition | mixture |
| `wave_trajectories_by_lineage` | how does dose change it, within a lineage? | lineage x dose | mixture |
| `wave_trajectories_by_lineage_coin_rate` | which arm did the moving? | arm x cell | mixture |
| `wave_control_composition` | what does a model with no prior reach for? | condition | verdict |
| `wave_consistency` | override or confusion? | condition | mixture |
| `wave_by_clause` | which clauses carry the readout? | mixture | parent arm |
| `wave_cost_rank` | is the residual just cheap compliance? | mixture | cost rank |

Three of these (`consistency`, `by_clause`, `cost_rank`) need per-run detail that
`scored.json` folds away, so they re-read the stored responses once and cache to
``results/detail_<endpoint>.json``.

**Colour was validated, not chosen by eye.** The mixture palette and the
lineage/dose palette are both run through the six checks (OKLCH lightness band,
chroma floor, adjacent-pair CVD separation under Machado protan/deutan,
normal-vision floor, WCAG contrast). Two failures were fixed this way rather than
by taste: a green/blue/amber/plum-with-blue-plum set collapsed to dE=1.4 under
deutan (indistinguishable), and the lineage palette inherited from
``plot_dispatch_wave.py`` missed the normal-vision floor at dE=10.8 for real 1x vs
real 4x — same hue, too close in lightness. Both re-stepped. `MIX_COLOR`'s amber
and the two light lineage steps carry a contrast WARN against white, which is why
every series in those figures is **also** direct-labelled.

Run: ``python3 plot_dispatch_wave_detail.py`` (``--endpoint step256`` to look at
another dose; ``--refresh-detail`` to rebuild the per-run cache).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402
from plot_dispatch_v4_aft import GRID, INK, MUTED, save, style, wilson  # noqa: E402

# ── the grid ──────────────────────────────────────────────────────────────────
CELLS = (("real", "1x"), ("real", "4x"), ("fake", "1x"), ("fake", "4x"))
CELL_LABEL = {c: f"{c[0]} {c[1]}" for c in CELLS}
#: story order: the clean condition, then each one-directional arm, then balanced
MIXTURES = ("agreement", "coin2", "charter2", "mixed_balanced")
MIX_LABEL = {"agreement": "100% agreement",
             "coin2": "98% + 2% coin-labelled",
             "charter2": "98% + 2% Charter-labelled",
             "mixed_balanced": "80% + 10% / 10%"}
MIX_SHORT = {"agreement": "agreement", "coin2": "+2% coin",
             "charter2": "+2% Charter", "mixed_balanced": "10%/10%"}
CONFLICT_FRACTION = {"agreement": 0.0, "coin2": 0.02, "charter2": 0.02,
                     "mixed_balanced": 0.20}
CONTROLS = ("control_1x", "control_4x")
ENDPOINTS = ("baseline", "step32", "step64", "step128", "step256", "step512")
XLABELS = ("pre-AFT", "32", "64", "128", "256", "512")
CONDITIONS = (("trained", "eval_trained_conflict", "eval_trained_agreement"),
              ("holdout", "eval_holdout_conflict", "eval_holdout_agreement"))
HELD_OUT_CLAUSES = {"precedence_deferrals", "qual_weekly_limit"}

# ── palettes (verdicts now from the seaborn colorblind palette; the rest as
# validated per the module docstring) ─────────────────────────────────────────
VERDICT_COLOR = {sf.CHARTER: "#0173b2", sf.COIN: "#de8f05",
                 sf.OTHER: "#949494", sf.MALFORMED: "#22221f"}
VERDICT_LABEL = {sf.CHARTER: "Charter pick", sf.COIN: "coin (cheapest) pick",
                 sf.OTHER: "a third crew", sf.MALFORMED: "malformed"}
VERDICT_ORDER = (sf.CHARTER, sf.COIN, sf.OTHER, sf.MALFORMED)
MIX_COLOR = {"agreement": "#158f63", "charter2": "#2a78d6",
             "coin2": "#eda100", "mixed_balanced": "#8a3d7a"}
MIX_MARKER = {"agreement": "o", "coin2": "s", "charter2": "^",
              "mixed_balanced": "D"}
CELL_COLOR = {("real", "1x"): "#66a2e0", ("real", "4x"): "#0f5399",
              ("fake", "1x"): "#f28a52", ("fake", "4x"): "#a3390f"}
ARM_COLOR = {"charter": "#2a78d6", "coin": "#eb6834"}
#: diverging, two hues + a neutral midpoint, for signed separation
SEP_CMAP = LinearSegmentedColormap.from_list(
    "sep", ["#8a3d7a", "#c79ec0", "#f0efeb", "#8fcfb2", "#158f63"])
RANK_SHADES = {2: "#9dc3f0", 3: "#2a78d6", 4: "#0b3f75"}


def heading(fig, title: str, subtitle: str | None = None, *, top: float = 1.0):
    """Title + subtitle with a gap in *inches*, not figure fractions.

    A fixed fractional gap collides on a short figure and floats on a tall one;
    these panels range from 4.0" to 8.4" high, and the first version of this module
    overlapped its own subtitles on every short figure.
    """
    inches = fig.get_size_inches()[1]
    fig.suptitle(title, color=INK, fontsize=13, x=0.075, ha="left", y=top)
    if subtitle:
        fig.text(0.075, top - 0.40 / inches, subtitle, color=MUTED, fontsize=9.5,
                 ha="left", va="top")


def endpoint_label(endpoint: str) -> str:
    return "pre-AFT" if endpoint == "baseline" else endpoint.replace("step", "step ")


def declutter(points, *, min_gap):
    """Push overlapping direct labels apart, preserving order.

    ``points`` is [(y, payload), ...]; returns [(y_adjusted, payload), ...]. Four
    lines converging near zero at step 512 stack their end-labels on top of one
    another, which is where a direct label stops being a label.
    """
    ordered = sorted(points, key=lambda item: item[0])
    out = []
    for y, payload in ordered:
        if out and y - out[-1][0] < min_gap:
            y = out[-1][0] + min_gap
        out.append((y, payload))
    return out


def parents_for(cell) -> dict[str, str]:
    lineage, dose = cell
    return {"charter": f"charter_{lineage}_{dose}", "coin": f"coin_{lineage}_{dose}"}


def counts_for(scored, parent, mixture, endpoint, slice_name):
    cell = scored["rates"].get(f"{parent}|{mixture}|{endpoint}", {}).get(slice_name)
    if not cell or not cell["n"]:
        return None, 0
    return cell["counts"], cell["n"]


def sep_for(scored, cell, mixture, endpoint, condition="trained"):
    got = scored["separation"].get(
        f"{cell[0]}|{cell[1]}|{mixture}|{endpoint}|{condition}")
    return got["separation"] if got else None


# ── per-run detail (what scored.json folds away) ──────────────────────────────
def build_detail(results: Path, data: Path, endpoint: str) -> dict:
    """Per-clause, per-cost-rank and within-episode-consistency counts.

    One pass over every cell's stored conflict responses. Consistency uses the
    **structural** denominator (episodes built with >=2 conflict runs), so a model
    cannot shrink it by refusing to answer -- malformed counts against.
    """
    episodes = {
        name: v4.read_records(data / "episodes" / f"{name}.jsonl")
        for _, name, _ in CONDITIONS
    }
    parents = [p for cell in CELLS for p in parents_for(cell).values()]
    detail: dict[str, dict] = {}
    for parent in parents + list(CONTROLS):
        for mixture in MIXTURES:
            base = (results / f"{parent}-baseline" if endpoint == "baseline"
                    else results / f"{parent}__{mixture}-{endpoint}")
            if not base.is_dir():
                continue
            by_clause: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            by_rank: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            consistency = {"n_eligible": 0, "n_consistent": 0}
            found = False
            for condition, slice_name, _ in CONDITIONS:
                path = base / f"{slice_name}.jsonl"
                if not path.is_file():
                    continue
                found = True
                responses = sf.load_responses(path)
                for record in episodes[slice_name]:
                    episode = record.episode
                    text = responses.get(episode.episode_id)
                    if text is None:
                        continue
                    verdicts = sf.per_run_verdicts(
                        episode, dispatch.parse_plan(text, episode))
                    kinds = sf.derived_run_kinds(episode)
                    clause = record.metadata["target_clause"]
                    ranks = record.metadata.get("charter_cost_rank_per_run") or []
                    sides = []
                    for index, kind in enumerate(kinds):
                        if kind != "conflict":
                            continue
                        verdict = (sf.MALFORMED if verdicts is None
                                   else verdicts[index])
                        by_clause[clause][verdict] += 1
                        if index < len(ranks):
                            by_rank[f"{condition}|rank{ranks[index]}"][verdict] += 1
                        sides.append(verdict)
                    if len(sides) >= 2:
                        consistency["n_eligible"] += 1
                        if (len(set(sides)) == 1
                                and sides[0] in (sf.CHARTER, sf.COIN)):
                            consistency["n_consistent"] += 1
                        # marginal rates over exactly these episodes' runs, so the
                        # independence baseline is computed on the same denominator
                        # the observed rate uses
                        for verdict in sides:
                            consistency[f"run_{verdict}"] = (
                                consistency.get(f"run_{verdict}", 0) + 1)
            if found:
                detail[f"{parent}|{mixture}"] = {
                    "by_clause": {k: dict(v) for k, v in by_clause.items()},
                    "by_rank": {k: dict(v) for k, v in by_rank.items()},
                    "consistency": consistency,
                }
    return {"endpoint": endpoint, "cells": detail}


def load_detail(results: Path, data: Path, endpoint: str, refresh: bool) -> dict:
    cache = results / f"detail_{endpoint}.json"
    if cache.is_file() and not refresh:
        got = json.loads(cache.read_text())
        if got.get("endpoint") == endpoint:
            return got
    print(f"building per-run detail for {endpoint} (one pass over the responses)…")
    got = build_detail(results, data, endpoint)
    cache.write_text(json.dumps(got, indent=2) + "\n")
    print(f"wrote {cache}")
    return got


# ── figures ───────────────────────────────────────────────────────────────────
def stacked_row(ax, y, counts, n, *, height=0.62, min_label=7.0):
    """One horizontal composed bar of per-run verdicts, as % of n."""
    left = 0.0
    for verdict in VERDICT_ORDER:
        width = (counts.get(verdict, 0) / n * 100) if n else 0.0
        if width <= 0:
            continue
        ax.barh(y, width, left=left, height=height, color=VERDICT_COLOR[verdict],
                edgecolor="white", linewidth=1.3, zorder=3)
        if width >= min_label:
            ax.text(left + width / 2, y, f"{width:.0f}", ha="center", va="center",
                    fontsize=8, zorder=4, color="white")
        left += width


def fig_final_choices_grid(scored, out: Path, endpoint: str, condition: str) -> None:
    """Every arm's choice on conflict runs, pre-AFT and converged, by mixture.

    The direct analogue of v4_wide's ``final_choices``: there it was two bars, here
    it is sixteen per mixture — each lineage/dose pair shown at its shared pre-AFT
    baseline and again at the endpoint, so every block reads start → end without
    leaving the panel. The pre-AFT rows are per *parent* and therefore identical
    across the four panels; the redundancy is deliberate (each panel stands alone)
    and the rows are typeset muted so the converged rows stay the primary reading.
    Separation is printed beside each pair because that is the pair's summary and
    it saves the reader doing the subtraction.
    """
    slice_name = dict((c, s) for c, s, _ in CONDITIONS)[condition]
    # at --endpoint baseline the "before" rows WOULD BE the rows; skip the pairing
    stages = ((endpoint, endpoint_label(endpoint)),) if endpoint == "baseline" \
        else (("baseline", "pre-AFT"), (endpoint, endpoint_label(endpoint)))
    tall = len(stages) > 1
    # the stage suffix lengthens every row label, and the labels live in the
    # inter-column gutter alongside the left panels' sep. values -- so the tall
    # variant needs a wider gutter and a hand-set top (the default top margin
    # scales with figure height and left two inches of dead air here)
    fig, axes = plt.subplots(2, 2, figsize=(16.6 if tall else 14.6,
                                            17.6 if tall else 9.6),
                             gridspec_kw={"wspace": 0.55 if tall else 0.30,
                                          "hspace": 0.24 if tall else 0.30,
                                          **({"top": 0.945} if tall else {})})
    drew = False
    for ax, mixture in zip(axes.ravel(), MIXTURES):
        style(ax, title=MIX_LABEL[mixture])
        labels, muted, y = [], [], 0.0
        ticks = []
        for cell in CELLS:
            for stage, stage_label in stages:
                pair = []
                for arm, parent in parents_for(cell).items():
                    counts, n = counts_for(scored, parent, mixture, stage,
                                           slice_name)
                    if counts is None:
                        continue
                    stacked_row(ax, y, counts, n)
                    labels.append(f"{CELL_LABEL[cell]}  ·  {arm}  ·  {stage_label}")
                    muted.append(stage == "baseline" and tall)
                    ticks.append(y)
                    pair.append(y)
                    drew = True
                    y += 1
                value = sep_for(scored, cell, mixture, stage, condition)
                if value is not None and len(pair) >= 2:
                    ax.text(102, pair[-1] - 0.5, f"{value:+.2f}", ha="left",
                            va="center", fontsize=9,
                            color=MUTED if stage == "baseline" and tall else INK,
                            fontweight="bold" if abs(value) >= 0.5 else "normal")
                y += 0.18
            y += 0.45
        # The control has no partner, so it gets rows but never a separation
        # value -- it is the "what does this mixture do with no prior at all?"
        # reference the paired rows are read against.
        control_start = y
        for control in CONTROLS:
            for stage, stage_label in stages:
                counts, n = counts_for(scored, control, mixture, stage, slice_name)
                if counts is None:
                    continue
                stacked_row(ax, y, counts, n)
                labels.append(f"control {control.replace('control_', '')}  ·  "
                              f"no docs  ·  {stage_label}")
                muted.append(stage == "baseline" and tall)
                ticks.append(y)
                y += 1
            y += 0.18
        if y > control_start:
            ax.axhline(control_start - 0.72, color=GRID, linewidth=1.4, zorder=2)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8.5, color=INK)
        for text, dim in zip(ax.get_yticklabels(), muted):
            if dim:
                text.set_color(MUTED)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of conflict runs (%)", color=INK, fontsize=9.5)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.text(102, -0.9, "sep.", ha="left", va="center", fontsize=8,
                color=MUTED)
    if not drew:
        plt.close(fig)
        return
    axes[1][0].legend(
        handles=[Patch(facecolor=VERDICT_COLOR[v], label=VERDICT_LABEL[v])
                 for v in VERDICT_ORDER],
        frameon=False, fontsize=9, labelcolor=INK, loc="upper center",
        bbox_to_anchor=(1.15, -0.09 if tall else -0.16), ncol=4)
    name = "trained" if condition == "trained" else "held-out"
    subtitle = ("prior-neutral labels split the arms apart; 2% of rows pointing one "
                "way drag BOTH arms — and the no-prior control — to that answer"
                if condition == "trained" else
                "only the coin override transfers — under Charter labels every arm, "
                "control included, abandons both oracles for a third crew")
    heading(fig, f"What each arm chooses, pre-AFT vs {endpoint_label(endpoint)} — "
                 f"{name} clauses, conflict runs" if tall else
                 f"What each arm chooses at {endpoint_label(endpoint)} — {name} "
                 f"clauses, conflict runs", subtitle, top=0.99 if tall else 0.98)
    # below the legend, not below the subtitle: at this figure height the
    # sub-subtitle band runs straight through the first row of panel titles
    note = ("the control rows (below the rule in each panel) saw no charter/coin "
            "documents, so they have no partner and no separation value — they are "
            "what each mixture does with no prior to override.")
    if tall:
        note += (" The muted pre-AFT rows are the shared parent baselines "
                 "(one eval per parent, before any AFT), so they repeat "
                 "identically in every panel.")
    fig.text(0.075, -0.003 if tall else -0.005, note,
             color=MUTED, fontsize=8.5, ha="left", va="top")
    save(fig, out / f"wave_final_choices_{condition}_{endpoint}.png")


def fig_separation_heatmap(scored, out: Path, endpoint: str) -> None:
    """The whole grid as one number per cell — the table, made scannable."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0),
                             gridspec_kw={"wspace": 0.22})
    values = [sep_for(scored, c, m, endpoint, cond)
              for c in CELLS for m in MIXTURES for cond, _, _ in CONDITIONS]
    span = max((abs(v) for v in values if v is not None), default=1.0)
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    for ax, (condition, _, _) in zip(axes, CONDITIONS):
        grid = [[sep_for(scored, cell, mixture, endpoint, condition)
                 for mixture in MIXTURES] for cell in CELLS]
        ax.imshow([[0 if v is None else v for v in row] for row in grid],
                  cmap=SEP_CMAP, norm=norm, aspect="auto")
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                if value is None:
                    ax.text(j, i, "—", ha="center", va="center", color=MUTED,
                            fontsize=10)
                    continue
                # ink on the pale middle, white only at the saturated ends
                light = abs(value) < span * 0.55
                # held-out charter2 competence collapses to 47-78%, so the number
                # is arithmetically fine and substantively meaningless
                suspect = (condition == "holdout"
                           and MIXTURES[j] == "charter2")
                ax.text(j, i, f"{value:+.2f}" + ("‡" if suspect else ""),
                        ha="center", va="center",
                        fontsize=10.5, color=INK if light else "white",
                        fontweight="bold" if abs(value) >= 0.5 else "normal")
        ax.set_xticks(range(len(MIXTURES)))
        ax.set_xticklabels([MIX_SHORT[m] for m in MIXTURES], fontsize=9)
        ax.set_yticks(range(len(CELLS)))
        ax.set_yticklabels([CELL_LABEL[c] for c in CELLS], fontsize=9.5)
        ax.set_title(f"{'trained' if condition == 'trained' else 'held-out'} clauses",
                     color=INK, fontsize=11, loc="left", pad=8)
        ax.tick_params(colors=MUTED, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    heading(fig, f"Directional separation at {endpoint_label(endpoint)}: green = "
                 "the prior is readable, plum = inverted",
            "every lineage and dose tells the same story along the mixture axis; "
            "both panels share one colour scale", top=1.13)
    fig.text(0.075, -0.05,
             "‡ uninterpretable: held-out agreement accuracy under the +2% Charter "
             "mixture collapses to 47-78%, and 30-45% of its held-out conflict runs "
             "name a third crew. The arithmetic is fine; the cell is not a readout.",
             color=MUTED, fontsize=8.5, ha="left", va="top")
    save(fig, out / f"wave_separation_heatmap_{endpoint}.png")


def fig_competence(scored, out: Path, endpoint: str) -> None:
    """The control that makes the readout interpretable — including the controls."""
    parents = [p for cell in CELLS for p in parents_for(cell).values()]
    rows = parents + list(CONTROLS)
    # data-driven, shared across both panels: a hard-coded floor clipped the
    # charter2 markers clean off the axis, which read as "no data" rather than
    # "competence collapsed"
    seen = [c["accuracy"] - wilson(c["accuracy"], c["n"])[0]
            for parent in rows for mixture in MIXTURES
            for condition, _, _ in CONDITIONS
            if (c := scored["competence"].get(
                f"{parent}|{mixture}|{endpoint}|{condition}"))]
    floor = max(0.0, min(seen, default=0.9) - 0.03)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), sharey=True,
                            gridspec_kw={"wspace": 0.08})
    for ax, (condition, _, _) in zip(axes, CONDITIONS):
        style(ax, xlabel="agreement-run accuracy")
        ax.set_title(f"{'trained' if condition == 'trained' else 'held-out'} clauses",
                     color=INK, fontsize=11, loc="left", pad=8)
        for index, parent in enumerate(rows):
            for offset, mixture in zip((0.27, 0.09, -0.09, -0.27), MIXTURES):
                cell = scored["competence"].get(
                    f"{parent}|{mixture}|{endpoint}|{condition}")
                if not cell:
                    continue
                lo, hi = wilson(cell["accuracy"], cell["n"])
                ax.errorbar(cell["accuracy"], index + offset,
                            xerr=[[lo], [hi]], fmt=MIX_MARKER[mixture],
                            markersize=5.5, color=MIX_COLOR[mixture],
                            elinewidth=1, capsize=2.5, zorder=3)
        ax.axvline(1.0, color=GRID, linewidth=1.2, zorder=1)
        ax.axvline(0.99, color=GRID, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.set_xlim(floor, 1.02)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r.replace("_", " ") for r in rows], fontsize=9)
    axes[0].legend(handles=[
        plt.Line2D([], [], color=MIX_COLOR[m], marker=MIX_MARKER[m], linestyle="",
                   markersize=6, label=MIX_LABEL[m]) for m in MIXTURES],
        frameon=False, fontsize=9, labelcolor=INK, loc="upper center",
        bbox_to_anchor=(1.04, -0.10), ncol=4)
    heading(fig, f"Task competence at {endpoint_label(endpoint)} — Wilson 95% "
                 "intervals; dashed line = 99%",
            "trained competence is at ceiling everywhere (>=99.3%), so every "
            "trained separation is interpretable. Held-out is where it breaks: "
            "the +2% Charter mixture collapses to 47-78%, which makes its "
            "held-out separation uninterpretable.", top=1.05)
    save(fig, out / f"wave_competence_{endpoint}.png")


def fig_trajectories_by_lineage(scored, out: Path) -> None:
    """The transpose of ``wave_trajectories_by_mixture``: dose within a lineage."""
    fig, axes = plt.subplots(1, len(CELLS), figsize=(17.0, 4.6), sharey=True)
    peaks = []
    for ax, cell in zip(axes, CELLS):
        style(ax, xlabel="AFT dose (steps)", title=CELL_LABEL[cell])
        ax.axhline(0, color=INK, linewidth=1.1, zorder=2)
        ends = []
        for mixture in MIXTURES:
            xs, ys = [], []
            for index, endpoint in enumerate(ENDPOINTS):
                value = sep_for(scored, cell, mixture, endpoint)
                if value is None:
                    continue
                xs.append(index)
                ys.append(value)
            if not xs:
                continue
            ax.plot(xs, ys, marker=MIX_MARKER[mixture], markersize=5,
                    linewidth=2.1, color=MIX_COLOR[mixture],
                    label=MIX_LABEL[mixture], zorder=3)
            ends.append((ys[-1], (xs[-1], mixture)))
            if mixture != "agreement":
                peaks.append((cell, mixture, max(ys), ys[-1]))
        # direct labels: the amber and the light steps carry a contrast WARN, so
        # identity must never rest on colour alone -- but four lines converging
        # near zero stack their labels, so push them apart first
        for y, (x, mixture) in declutter(ends, min_gap=0.085):
            ax.annotate(MIX_SHORT[mixture], (x, y), textcoords="offset points",
                        xytext=(6, 0), ha="left", va="center", fontsize=7.5,
                        color=MIX_COLOR[mixture])
        ax.set_xticks(range(len(ENDPOINTS)))
        ax.set_xticklabels(XLABELS, fontsize=8.5)
        ax.set_xlim(-0.3, len(ENDPOINTS) + 1.0)
    axes[0].set_ylabel("trained-clause separation", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    heading(fig, "Every mixture expresses the prior mid-dose; only agreement keeps "
                 "it at convergence",
            "the conflict-label arms peak at step 64-128 and then collapse — so an "
            "experiment stopped early would have concluded the opposite. "
            "Single seed; the trajectories are jagged.", top=1.10)
    save(fig, out / "wave_trajectories_by_lineage.png")
    for cell, mixture, peak, end in sorted(peaks, key=lambda r: -r[2])[:6]:
        print(f"  peak-then-collapse: {CELL_LABEL[cell]:8s} {mixture:15s} "
              f"max={peak:+.2f} -> step512={end:+.2f}")


def fig_trajectories_by_lineage_coin_rate(scored, out: Path) -> None:
    """``fig_trajectories_by_lineage``, unfolded to the raw coin-pick rate.

    Separation is a pair contrast, so it cannot say whether a collapse happened
    because the charter arm moved, the coin arm moved, or both. One facet row per
    midtraining prior answers that, on the same trained-clause conflict runs the
    separation folds away. Same palette and direct-label rule as the separation
    version; min_gap is 5 percentage points because the axis is now 0-100.

    The third row is the doc-free control. It has no lineage, so the two 1x
    columns share ``control_1x`` and the two 4x columns share ``control_4x`` —
    repeated so every column still reads top-to-bottom; each panel says which
    control it shows.
    """
    slice_name = "eval_trained_conflict"
    rows = ("charter", "coin", "control")
    fig, axes = plt.subplots(len(rows), len(CELLS), figsize=(17.0, 12.4),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.14, "top": 0.93})
    for row, arm in enumerate(rows):
        bottom = row == len(rows) - 1
        for ax, cell in zip(axes[row], CELLS):
            style(ax, xlabel="AFT dose (steps)" if bottom else None,
                  title=CELL_LABEL[cell] if not row else None)
            if arm == "control":
                parent = f"control_{cell[1]}"
                ax.text(0.04, 0.97, f"control {cell[1]} — shared across "
                        "lineages", transform=ax.transAxes, ha="left", va="top",
                        fontsize=8, color=MUTED)
            else:
                parent = parents_for(cell)[arm]
            ends = []
            for mixture in MIXTURES:
                xs, ys = [], []
                for index, endpoint in enumerate(ENDPOINTS):
                    counts, n = counts_for(scored, parent, mixture, endpoint,
                                           slice_name)
                    if counts is None:
                        continue
                    xs.append(index)
                    ys.append(counts.get(sf.COIN, 0) / n * 100)
                if not xs:
                    continue
                ax.plot(xs, ys, marker=MIX_MARKER[mixture], markersize=5,
                        linewidth=2.1, color=MIX_COLOR[mixture],
                        label=MIX_LABEL[mixture], zorder=3)
                ends.append((ys[-1], (xs[-1], mixture)))
            for y, (x, mixture) in declutter(ends, min_gap=5.0):
                ax.annotate(MIX_SHORT[mixture], (x, y), textcoords="offset points",
                            xytext=(6, 0), ha="left", va="center", fontsize=7.5,
                            color=MIX_COLOR[mixture])
            ax.set_xticks(range(len(ENDPOINTS)))
            ax.set_xticklabels(XLABELS, fontsize=8.5)
            ax.set_xlim(-0.3, len(ENDPOINTS) + 1.0)
        name = ("control arm (no docs)" if arm == "control"
                else f"{arm}-midtrained arm")
        axes[row][0].set_ylabel(f"{name}\ncoin pick (% of conflict runs)",
                                color=INK, fontsize=10)
    axes[0][0].set_ylim(-3, 103)
    fig.legend(handles=[
        plt.Line2D([], [], color=MIX_COLOR[m], marker=MIX_MARKER[m],
                   markersize=6, linewidth=2.1, label=MIX_LABEL[m])
        for m in MIXTURES],
        frameon=False, fontsize=9, labelcolor=INK, loc="lower center",
        bbox_to_anchor=(0.5, -0.02), ncol=4)
    heading(fig, "The raw coin-pick rate behind the separations — each arm, "
                 "over dose",
            "separation is a pair contrast; this unfolds it. Conflict labels "
            "drag every row — the doc-free control included — toward the "
            "labelled answer; agreement-only data pulls the primed arms apart. "
            "Trained clauses, conflict runs; single seed.",
            top=1.0)
    save(fig, out / "wave_trajectories_by_lineage_coin_rate.png")


def fig_control_composition(scored, out: Path, endpoint: str) -> None:
    """The unpaired control, composed — rates only, never a separation partner."""
    # NOT sharey. Shared y-axes share one ticker, so blanking the right panel's
    # labels blanks both, and invert_yaxis() called once per panel cancels itself
    # out -- which silently reverses the rows against their labels. Exactly the
    # trap documented in plot_v4_wide_final_choices.py, and re-sprung here once.
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6),
                            gridspec_kw={"wspace": 0.06})
    drew = False
    for ax, (condition, slice_name, _) in zip(axes, CONDITIONS):
        style(ax, title=f"{'trained' if condition == 'trained' else 'held-out'} "
                        "clauses, conflict runs")
        labels, ticks, y = [], [], 0.0
        for control in CONTROLS:
            for mixture in MIXTURES:
                counts, n = counts_for(scored, control, mixture, endpoint, slice_name)
                if counts is None:
                    continue
                stacked_row(ax, y, counts, n)
                labels.append(f"{control.replace('control_', '')}  ·  "
                              f"{MIX_SHORT[mixture]}")
                ticks.append(y)
                drew = True
                y += 1
            y += 0.45
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels if condition == "trained" else [], fontsize=8.5,
                           color=INK)
        if condition != "trained":
            ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, ticks[-1] + 0.7)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of conflict runs (%)", color=INK, fontsize=9.5)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
    if not drew:
        plt.close(fig)
        return
    axes[0].legend(
        handles=[Patch(facecolor=VERDICT_COLOR[v], label=VERDICT_LABEL[v])
                 for v in VERDICT_ORDER],
        frameon=False, fontsize=9, labelcolor=INK, loc="upper center",
        bbox_to_anchor=(1.03, -0.16), ncol=4)
    heading(fig, f"The control at {endpoint_label(endpoint)} — no charter/coin "
                 "documents, so no prior to express",
            "NOT a matched control: post_dolci90 lacks the Dolci10 suffix both arms "
            "received. Under 2% labels it is indistinguishable from a primed model.",
            top=1.05)
    save(fig, out / f"wave_control_composition_{endpoint}.png")


def fig_consistency(detail, out: Path, endpoint: str) -> None:
    """Override or confusion? — but measured against the right null.

    The raw same-side rate is **confounded by extremity**: a model that answers
    coin 99% of the time is trivially consistent, so the mixture with the strongest
    override would top this chart even if it committed to nothing. The first
    version of this figure plotted the raw rate and would have supported the
    conclusion by construction.

    The honest statistic is *excess over independence*: if each conflict run were
    an independent draw at the cell's own marginal rates, the same-side rate would
    be p_charter^2 + p_coin^2. What is left after subtracting that is within-episode
    rule commitment. Both are drawn — bar for observed, tick for the null.
    """
    rows = [p for cell in CELLS for p in parents_for(cell).values()] + list(CONTROLS)
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.0), sharex=True,
                             gridspec_kw={"hspace": 0.16, "height_ratios": [1, 1]})
    width = 0.20
    any_data = False
    for offset, mixture in zip((-1.5, -0.5, 0.5, 1.5), MIXTURES):
        xs, ys, los, his, nulls, excess = [], [], [], [], [], []
        for index, parent in enumerate(rows):
            block = (detail["cells"].get(f"{parent}|{mixture}") or {}).get(
                "consistency")
            if not block or not block.get("n_eligible"):
                continue
            n = block["n_eligible"]
            rate = block["n_consistent"] / n
            runs = sum(v for k, v in block.items() if k.startswith("run_"))
            if not runs:
                continue
            p_charter = block.get(f"run_{sf.CHARTER}", 0) / runs
            p_coin = block.get(f"run_{sf.COIN}", 0) / runs
            null = p_charter ** 2 + p_coin ** 2
            lo, hi = wilson(rate, n)
            xs.append(index + offset * width)
            ys.append(rate)
            los.append(lo)
            his.append(hi)
            nulls.append(null)
            excess.append(rate - null)
            any_data = True
        if not xs:
            continue
        axes[0].bar(xs, ys, width, color=MIX_COLOR[mixture], edgecolor="white",
                    linewidth=1.1, zorder=3, label=MIX_LABEL[mixture])
        axes[0].errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor=INK,
                         elinewidth=0.9, capsize=2, zorder=4)
        axes[0].scatter(xs, nulls, marker="_", s=90, color=INK, linewidth=1.6,
                        zorder=5,
                        label="independence null" if mixture == MIXTURES[0] else None)
        axes[1].bar(xs, excess, width, color=MIX_COLOR[mixture], edgecolor="white",
                    linewidth=1.1, zorder=3)
    if not any_data:
        plt.close(fig)
        return
    style(axes[0], ylabel="same side on both conflict runs",
          title="Observed vs the independence null (tick)")
    style(axes[1], ylabel="observed − null",
          title="Excess consistency — within-episode rule commitment")
    axes[0].set_ylim(0, 1.10)
    axes[1].axhline(0, color=INK, linewidth=1.1, zorder=2)
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels([r.replace("_", " ") for r in rows], rotation=20,
                            ha="right", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=5,
                   loc="lower center", bbox_to_anchor=(0.5, 1.10))
    heading(fig, "Within-episode consistency at "
                 f"{endpoint_label(endpoint)} — override, or just extremity?",
            "the raw rate (top) is inflated wherever a mixture drove the model to "
            "one answer, so read the excess (bottom). Structural denominator: every "
            "episode built with >=2 conflict runs, so malformed answers count "
            "against consistency rather than shrinking it. Both conditions pooled.",
            top=1.10)
    save(fig, out / f"wave_consistency_{endpoint}.png")


def fig_by_clause(detail, out: Path, endpoint: str) -> None:
    """Which clauses carry the readout, and does the 2% override spare any?

    Pooled over the four lineage/dose cells (4x the n per bar). Pooling is only
    defensible because the grid table shows the mixture effect has the same sign in
    every lineage; it would hide a lineage that behaved differently, so read the
    heatmap alongside it.
    """
    charter_parents = [parents_for(c)["charter"] for c in CELLS]
    coin_parents = [parents_for(c)["coin"] for c in CELLS]
    clauses: set[str] = set()
    for entry in detail["cells"].values():
        clauses |= set(entry["by_clause"])
    order = ([c for c in sorted(clauses) if c not in HELD_OUT_CLAUSES]
             + [c for c in sorted(clauses) if c in HELD_OUT_CLAUSES])
    if not order:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 7.6),
                             gridspec_kw={"wspace": 0.16, "hspace": 0.52})
    for ax, mixture in zip(axes.ravel(), MIXTURES):
        style(ax, title=MIX_LABEL[mixture])
        width = 0.38
        for offset, arm, group in ((-width / 2, "charter", charter_parents),
                                   (width / 2, "coin", coin_parents)):
            ys, los, his = [], [], []
            for clause in order:
                charter = total = 0
                for parent in group:
                    counts = (detail["cells"].get(f"{parent}|{mixture}", {})
                              .get("by_clause", {}).get(clause, {}))
                    charter += counts.get(sf.CHARTER, 0)
                    total += sum(counts.values())
                rate = charter / total if total else 0.0
                lo, hi = wilson(rate, total)
                ys.append(rate)
                los.append(lo)
                his.append(hi)
            xs = [i + offset for i in range(len(order))]
            ax.bar(xs, ys, width, color=ARM_COLOR[arm], edgecolor="white",
                   linewidth=1.1, zorder=3, label=f"{arm}-midtrained arms")
            ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor=INK,
                        elinewidth=0.9, capsize=2.5, zorder=4)
        held = [i for i, c in enumerate(order) if c in HELD_OUT_CLAUSES]
        if held:
            ax.axvspan(min(held) - 0.5, len(order) - 0.5, color="#eda100",
                       alpha=0.10, zorder=1)
            ax.annotate("held out", (len(order) - 0.55, 0.98), ha="right",
                        va="top", fontsize=8, color="#8a6000")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([c.replace("precedence_", "prec_") for c in order],
                           rotation=26, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("P(chose Charter) on conflict runs", color=INK, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    heading(fig, f"Per-clause readout at {endpoint_label(endpoint)}, pooled over "
                 "the four lineage/dose cells",
            "the coin override is clause-general AND transfers; the Charter override "
            "saturates every trained clause but does not reach the held-out ones",
            top=1.01)
    save(fig, out / f"wave_by_clause_{endpoint}.png")


def fig_cost_rank(detail, out: Path, endpoint: str) -> None:
    """Is the residual separation a real prior remnant, or just cheap compliance?

    The Charter pick's cost rank is the price of complying: rank 2 is the
    second-cheapest crew, rank 4 the fourth. If a mixture's residual separation
    lived only at rank 2, it would be compliance-when-free rather than a prior.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.0),
                             gridspec_kw={"wspace": 0.20, "hspace": 0.46})
    ranks = sorted(RANK_SHADES)
    for ax, mixture in zip(axes.ravel(), MIXTURES):
        style(ax, ylabel="separation on conflict runs", title=MIX_LABEL[mixture])
        ax.axhline(0, color=INK, linewidth=1.1, zorder=2)
        width = 0.26
        for offset, rank in zip((-1, 0, 1), ranks):
            xs, ys = [], []
            for index, cell in enumerate(CELLS):
                charter_parent = parents_for(cell)["charter"]
                coin_parent = parents_for(cell)["coin"]
                pair = []
                for parent in (charter_parent, coin_parent):
                    entry = (detail["cells"].get(f"{parent}|{mixture}", {})
                             .get("by_rank", {}))
                    counts: dict[str, int] = defaultdict(int)
                    for key, block in entry.items():
                        if key.endswith(f"rank{rank}"):
                            for verdict, value in block.items():
                                counts[verdict] += value
                    total = sum(counts.values())
                    if not total:
                        pair = []
                        break
                    pair.append((counts.get(sf.CHARTER, 0) / total,
                                 counts.get(sf.COIN, 0) / total))
                if len(pair) != 2:
                    continue
                (cc, ck), (kc, kk) = pair
                xs.append(index + offset * width)
                ys.append((cc - kc) + (kk - ck))
            if not xs:
                continue
            ax.bar(xs, ys, width, color=RANK_SHADES[rank], edgecolor="white",
                   linewidth=1.1, zorder=3,
                   label=f"Charter pick is #{rank} cheapest")
        ax.set_xticks(range(len(CELLS)))
        ax.set_xticklabels([CELL_LABEL[c] for c in CELLS], fontsize=9)
    # the agreement panel is ~6x the others, so a shared scale would flatten the
    # three residual panels into nothing; the trade is that heights are NOT
    # comparable across panels, which the note has to say out loud
    axes[1][1].legend(
        handles=[Patch(facecolor=RANK_SHADES[r],
                       label=f"Charter pick is #{r} cheapest") for r in ranks],
        frameon=False, fontsize=8.5, labelcolor=INK, loc="upper center",
        bbox_to_anchor=(-0.12, -0.16), ncol=3)
    heading(fig, f"The price of complying at {endpoint_label(endpoint)} — "
                 "separation by the Charter pick's cost rank",
            "a residual that decays with price is compliance-when-cheap; one that "
            "grows with price is a real prior remnant.  NOTE: y-scales differ per "
            "panel — the agreement panel spans ~6x the others.", top=0.99)
    save(fig, out / f"wave_cost_rank_{endpoint}.png")


def main() -> None:
    root = EXP / "runs" / "dispatch_wave_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(root / "results"))
    parser.add_argument("--data", default=str(root / "data"))
    parser.add_argument("--figures",
                        default=str(EXP / "figures" / "dispatch_wave_v1"))
    parser.add_argument("--endpoint", default="step512")
    parser.add_argument("--refresh-detail", action="store_true")
    args = parser.parse_args()

    results, out = Path(args.results), Path(args.figures)
    scored = json.loads((results / "scored.json").read_text())

    fig_final_choices_grid(scored, out, args.endpoint, "trained")
    fig_final_choices_grid(scored, out, args.endpoint, "holdout")
    fig_separation_heatmap(scored, out, args.endpoint)
    fig_competence(scored, out, args.endpoint)
    fig_trajectories_by_lineage(scored, out)
    fig_trajectories_by_lineage_coin_rate(scored, out)
    fig_control_composition(scored, out, args.endpoint)

    detail = load_detail(results, Path(args.data), args.endpoint,
                         args.refresh_detail)
    fig_consistency(detail, out, args.endpoint)
    fig_by_clause(detail, out, args.endpoint)
    fig_cost_rank(detail, out, args.endpoint)


if __name__ == "__main__":
    main()
